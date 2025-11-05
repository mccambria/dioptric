import json, os, glob, math
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import io


# ---------------------- utils ----------------------
def _read_hyperfine_table(
    path: str,
    distance_cutoff: float | None = None,
    NV_AXIS=(1.0, 1.0, 1.0),
    UNITS_A_IS_MHZ: bool = False,
) -> Dict[int, dict]:
    """
    Reads nv-2.txt-like table with columns:
      index distance x y z Axx Ayy Azz Axy Axz Ayz
    and returns a dict:
      {site_id: {"r","A_par_kHz","B_perp_kHz","x","y","z"}}

    NV_AXIS: NV direction (unit vector). If your file is already in NV frame, use (0,0,1).
    UNITS_A_IS_MHZ: set True if A components are in MHz (will convert to kHz).
    """
    file_path = Path(path)
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Find the first data row (like your code): first line starting with an integer ID
    def _is_int_start(s: str) -> bool:
        s = s.lstrip()
        if not s:
            return False
        tok = s.split()[0]
        try:
            int(tok)
            return True
        except Exception:
            return False

    try:
        data_start = next(i for i, line in enumerate(lines) if _is_int_start(line))
    except StopIteration:
        raise RuntimeError(f"Could not find data start in hyperfine table: {path}")
    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        skiprows=data_start,
        header=None,
        names=[
            "index",
            "distance",
            "x",
            "y",
            "z",
            "Axx",
            "Ayy",
            "Azz",
            "Axy",
            "Axz",
            "Ayz",
        ],
        engine="python",
    )

    if distance_cutoff is not None:
        df = df[df["distance"] < float(distance_cutoff)].copy()

    if df.empty:
        raise RuntimeError(f"Hyperfine table {path} produced no rows (after cutoff?).")

    # NV axis (unit)
    n = np.asarray(NV_AXIS, dtype=float)
    n /= np.linalg.norm(n) if np.linalg.norm(n) > 0 else 1.0

    # Convert A tensor to A_par and B_perp
    # A = [[Axx, Axy, Axz],
    #      [Axy, Ayy, Ayz],
    #      [Axz, Ayz, Azz]]
    scale = 1000.0 if UNITS_A_IS_MHZ else 1.0  # convert MHz → kHz if needed

    Axx = df["Axx"].to_numpy(float) * scale
    Ayy = df["Ayy"].to_numpy(float) * scale
    Azz = df["Azz"].to_numpy(float) * scale
    Axy = df["Axy"].to_numpy(float) * scale
    Axz = df["Axz"].to_numpy(float) * scale
    Ayz = df["Ayz"].to_numpy(float) * scale

    # Build A @ n efficiently for all rows
    # For each row, a = A n = [Axx nx + Axy ny + Axz nz, Axy nx + Ayy ny + Ayz nz, Axz nx + Ayz ny + Azz nz]
    nx, ny, nz = n
    ax = Axx * nx + Axy * ny + Axz * nz
    ay = Axy * nx + Ayy * ny + Ayz * nz
    az = Axz * nx + Ayz * ny + Azz * nz

    # A_par = n · (A n)
    A_par = ax * nx + ay * ny + az * nz  # kHz

    # Perp vector = (A n) - A_par * n
    px = ax - A_par * nx
    py = ay - A_par * ny
    pz = az - A_par * nz
    B_perp = np.sqrt(px * px + py * py + pz * pz)  # kHz

    # Pack output dict
    ids = df["index"].astype(int).to_numpy()
    x = df["x"].to_numpy(float)
    y = df["y"].to_numpy(float)
    z = df["z"].to_numpy(float)
    r = df["distance"].to_numpy(float)

    feats = {}
    for i in range(len(df)):
        sid = int(ids[i])
        feats[sid] = {
            "r": float(r[i]),
            "A_par_kHz": float(A_par[i]),
            "B_perp_kHz": float(B_perp[i]),
            "x": float(x[i]),
            "y": float(y[i]),
            "z": float(z[i]),
        }
    return feats


def _stack_feats(
    site_ids: List[int], feat_dict: Dict[int, Dict[str, float]], keys: List[str]
) -> np.ndarray:
    out = np.zeros((len(site_ids), len(keys)), dtype=np.float32)
    for i, sid in enumerate(site_ids):
        d = feat_dict.get(sid)
        if d is None:
            # unseen id → zeros (rare)
            continue
        for j, k in enumerate(keys):
            out[i, j] = float(d.get(k, 0.0))
    return out


# ---------------------- Dataset ----------------------


class ShardedEchoDataset(Dataset):
    """
    Reads shards produced by make_spin_echo_dataset.py:
      shard_XXXX.npz → arrays: traces[T], taus_us[T]
      shard_XXXX.json → list of metas (per-trace) including site_ids (positives)

    For each trace, we construct a candidate list:
      - positives = meta["site_ids"]
      - negatives = random sample from hyperfine table excluding positives
      - features = subset of ["r","A_par_kHz","B_perp_kHz","x","y","z"]

    Returns a dict per item:
      {
        "trace": FloatTensor[T],
        "cand_feats": FloatTensor[M, F],
        "cand_labels": FloatTensor[M],
        "trace_id": str,  # "shard_0001/1234"
      }
    """

    def __init__(
        self,
        shards_dir: str,
        hyperfine_path: str,
        B_vec_G: Tuple[
            float, float, float
        ] = None,  # not used here but kept for API continuity
        feature_keys: List[str] = ("r", "A_par_kHz", "B_perp_kHz", "x", "y", "z"),
        negatives_per_trace: int = 16,
        rng_seed: int = 123,
        drop_empty: bool = True,
    ):
        self.root = Path(shards_dir)
        self.feature_keys = list(feature_keys)
        self.neg_per = int(negatives_per_trace)
        self.drop_empty = bool(drop_empty)

        # discover shards
        self.npz_paths = sorted(self.root.glob("shard_*.npz"))
        self.json_paths = [self.root / (p.stem + ".json") for p in self.npz_paths]
        if not self.npz_paths:
            raise FileNotFoundError(f"No shard_*.npz in {self.root}")

        # load json metas and build index
        self._index = []  # list of (shard_i, local_idx)
        self._metas = []
        self._npz = []  # lazy handles
        total = 0
        for i, jp in enumerate(self.json_paths):
            with open(jp, "r", encoding="utf-8") as f:
                meta = json.load(f)
            metas = meta["metas"]
            if self.drop_empty:
                loc = [(i, j) for j, m in enumerate(metas) if m.get("site_ids")]
            else:
                loc = [(i, j) for j in range(len(metas))]
            self._index.extend(loc)
            self._metas.append(metas)
            self._npz.append(None)  # filled on first access
            total += len(loc)
        if total == 0:
            raise RuntimeError("Dataset empty after applying drop_empty filter")

        # hyperfine features
        # self._hf = _read_hyperfine_table(hyperfine_path)
        self._hf = _read_hyperfine_table(
            hyperfine_path,
            distance_cutoff=8.0,  # or your cutoff (e.g., 8.0)
            NV_AXIS=(1.0, 1.0, 1.0),  # pick your NV orientation (unit vector)
            UNITS_A_IS_MHZ=True,  # set True if Axx etc. are in MHz
        )

        self._hf_ids = np.array(sorted(self._hf.keys()), dtype=np.int64)

        # RNG for negatives
        self._rng = np.random.default_rng(rng_seed)

    def __len__(self) -> int:
        return len(self._index)

    # inside ShardedEchoDataset

    # at top

    # inside ShardedEchoDataset
    def _load_npz(self, shard_idx: int):
        """Windows-safe: read file fully, then np.load from an in-memory buffer."""
        path = self.npz_paths[shard_idx]
        path = str(path) if isinstance(path, Path) else path

        # Read into memory to avoid zipfile/handle weirdness on Windows + workers + network drives
        with open(path, "rb") as f:
            buf = f.read()

        with np.load(io.BytesIO(buf), allow_pickle=False, mmap_mode=None) as z:
            arrays = {k: np.array(z[k], copy=True) for k in z.files}

        # Normalize dtypes/contiguity (optional but nice)
        if "traces" in arrays:
            arrays["traces"] = np.ascontiguousarray(arrays["traces"], dtype=np.float32)

        if "cand_feats" in arrays:
            arrays["cand_feats"] = np.ascontiguousarray(
                arrays["cand_feats"], dtype=np.float32
            )
        if "cand_labels" in arrays:
            arrays["cand_labels"] = np.ascontiguousarray(
                arrays["cand_labels"], dtype=np.float32
            )
        if "mask" in arrays:
            arrays["mask"] = np.ascontiguousarray(arrays["mask"], dtype=np.bool_)
        if "cand_batch_idx" in arrays:
            arrays["cand_batch_idx"] = np.ascontiguousarray(
                arrays["cand_batch_idx"], dtype=np.int64
            )

        return arrays

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        si, li = self._index[idx]
        npz = self._load_npz(si)
        traces = npz["traces"]
        trace = torch.from_numpy(np.array(traces[li], dtype=np.float32))  # [T]

        meta = self._metas[si][li]
        pos_ids = list(map(int, meta.get("site_ids", [])))

        # negatives
        neg_ids = []
        if self.neg_per > 0:
            pos_set = set(pos_ids)
            # fast uniform sample without replacement
            # sample a little extra in case of accidental overlap
            k = min(self.neg_per + len(pos_ids) + 8, len(self._hf_ids))
            cand = self._rng.choice(self._hf_ids, size=k, replace=False).tolist()
            for sid in cand:
                if sid not in pos_set:
                    neg_ids.append(int(sid))
                    if len(neg_ids) >= self.neg_per:
                        break

        # build features + labels
        all_ids = pos_ids + neg_ids
        labels = np.array([1.0] * len(pos_ids) + [0.0] * len(neg_ids), dtype=np.float32)
        feats = _stack_feats(all_ids, self._hf, self.feature_keys)  # [M, F]

        ex = {
            "trace": trace,  # [T]
            "cand_feats": torch.from_numpy(feats),  # [M, F]
            "cand_labels": torch.from_numpy(labels),  # [M]
            "trace_id": f"{self.npz_paths[si].stem}/{li}",
        }
        return ex


# ---------------------- collate ----------------------
def packed_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    B = len(batch)
    traces = torch.stack([b["trace"] for b in batch], dim=0)  # [B, T]

    feats, labels, idx_map, ids = [], [], [], []
    for b, s in enumerate(batch):
        M = s["cand_feats"].shape[0]
        if M == 0:
            continue
        feats.append(s["cand_feats"])
        labels.append(s["cand_labels"])
        idx_map.append(torch.full((M,), b, dtype=torch.long))
        ids.append(s["trace_id"])

    # infer feature dim F if we have at least one non-empty item
    if feats:
        cand_feats = torch.cat(feats, dim=0)
        F = cand_feats.shape[1]
    else:
        # try to read F from any example; else 0
        F = next(
            (s["cand_feats"].shape[1] for s in batch if s["cand_feats"].ndim == 2), 0
        )
        cand_feats = torch.zeros((0, F), dtype=torch.float32)

    cand_labels = (
        torch.cat(labels, dim=0) if labels else torch.zeros((0,), dtype=torch.float32)
    )
    cand_idx_to_trace = (
        torch.cat(idx_map, dim=0) if idx_map else torch.zeros((0,), dtype=torch.long)
    )

    return {
        "traces": traces,  # [B, T]
        "cand_feats": cand_feats,  # [sumM, F]
        "cand_labels": cand_labels,  # [sumM]
        "cand_idx_to_trace": cand_idx_to_trace,  # [sumM]
        "cand_batch_idx": cand_idx_to_trace,  # alias for model
        "trace_ids": ids if ids else [s["trace_id"] for s in batch],
    }
