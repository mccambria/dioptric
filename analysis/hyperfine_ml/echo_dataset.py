# echo_dataset.py
import json, time
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import io, os, zipfile

# ---------- physics constants / axes ----------
GAMMA_13C_MHz_per_G = 0.0010705  # 13C gyromagnetic ratio in MHz/G
NV_AXIS_UNIT = np.asarray((1.0, 1.0, 1.0), float)
NV_AXIS_UNIT /= (
    np.linalg.norm(NV_AXIS_UNIT) if np.linalg.norm(NV_AXIS_UNIT) > 0 else 1.0
)


# ---------------------- hyperfine utils ----------------------
def _read_hyperfine_table(
    path: str,
    distance_cutoff: float | None = None,
    NV_AXIS=(1.0, 1.0, 1.0),
    UNITS_A_IS_MHZ: bool = False,
) -> Dict[int, dict]:
    """
    Reads nv-2.txt-like table with columns:
      index distance x y z Axx Ayy Azz Axy Axz Ayz
    Returns a dict:
      {site_id: {"r","A_par_kHz","B_perp_kHz","x","y","z"}}
    """
    file_path = Path(path)
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # find first data row (first line starting with an int)
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

    # Convert A tensor to A_par (kHz) and B_perp (kHz)
    scale = 1000.0 if UNITS_A_IS_MHZ else 1.0  # MHz→kHz if needed
    Axx = df["Axx"].to_numpy(float) * scale
    Ayy = df["Ayy"].to_numpy(float) * scale
    Azz = df["Azz"].to_numpy(float) * scale
    Axy = df["Axy"].to_numpy(float) * scale
    Axz = df["Axz"].to_numpy(float) * scale
    Ayz = df["Ayz"].to_numpy(float) * scale

    nx, ny, nz = n
    ax = Axx * nx + Axy * ny + Axz * nz
    ay = Axy * nx + Ayy * ny + Ayz * nz
    az = Axz * nx + Ayz * ny + Azz * nz
    A_par = ax * nx + ay * ny + az * nz  # kHz

    px = ax - A_par * nx
    py = ay - A_par * ny
    pz = az - A_par * nz
    B_perp = np.sqrt(px * px + py * py + pz * pz)  # kHz

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


def _build_one_feat_row(
    site: dict,
    want: List[str],
    B_vec: np.ndarray | None,
    B_mag: float,
    B_par: float,
    B_perp: float,
) -> np.ndarray:
    """
    site: {"r","A_par_kHz","B_perp_kHz","x","y","z"}
    want: ordered feature_keys
    Returns: np.float32[ len(want) ], computing B-derived + frequency hints if requested.
    """
    out = np.zeros((len(want),), dtype=np.float32)

    # base site values
    r = float(site.get("r", 0.0))
    A_k = float(site.get("A_par_kHz", 0.0))
    Bk = float(site.get("B_perp_kHz", 0.0))
    x, y, z = (
        float(site.get("x", 0.0)),
        float(site.get("y", 0.0)),
        float(site.get("z", 0.0)),
    )

    # geometry
    rvec = np.asarray([x, y, z], float)
    rnorm = np.linalg.norm(rvec)
    rhat = (rvec / rnorm) if rnorm > 0 else np.zeros(3, float)

    # frequency cues (MHz)
    f0 = GAMMA_13C_MHz_per_G * B_mag
    half_A_MHz = (A_k / 1000.0) * 0.5
    f1p = abs(f0 + half_A_MHz)
    f1m = abs(f0 - half_A_MHz)

    # angle cue
    cosBrhat = 0.0
    if B_vec is not None and rnorm > 0 and B_mag > 0:
        cosBrhat = float(np.dot(B_vec / B_mag, rhat))

    name_to_val = {
        "r": r,
        "A_par_kHz": A_k,
        "B_perp_kHz": Bk,
        "x": x,
        "y": y,
        "z": z,
        "B_mag_G": B_mag,
        "B_par_G": B_par,
        "B_perp_G": B_perp,
        "cosBrhat": cosBrhat,
        "f0_MHz": f0,
        "f1p_MHz": f1p,
        "f1m_MHz": f1m,
    }
    for j, k in enumerate(want):
        out[j] = float(name_to_val.get(k, 0.0))
    return out.astype(np.float32, copy=False)


# ---------------------- Dataset ----------------------
class ShardedEchoDataset(Dataset):
    """
    Shards produced by make_spin_echo_dataset.py:
      shard_XXXX.npz → arrays: traces[T], taus_us[T]
      shard_XXXX.json → list of metas (per-trace) including site_ids (positives)

    For each trace, we build candidates:
      - positives = meta["site_ids"]
      - negatives = random sample from hyperfine table excluding positives
      - features = subset of ["r","A_par_kHz","B_perp_kHz","x","y","z",
                              "B_mag_G","B_par_G","B_perp_G","cosBrhat",
                              "f0_MHz","f1p_MHz","f1m_MHz"]

    Returns per item:
      {
        "trace": FloatTensor[T],
        "cand_feats": FloatTensor[M, F],
        "cand_labels": FloatTensor[M],
        "trace_id": str,
      }
    """

    def __init__(
        self,
        shards_dir: str,
        hyperfine_path: str,
        B_vec_G: Tuple[float, float, float] | None = None,
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
        self._metas: List[List[dict]] = []
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
            total += len(loc)
        if total == 0:
            raise RuntimeError("Dataset empty after applying drop_empty filter")

        # After building lists:
        # After building lists:
        bad = []
        for npz_p, json_p in zip(self.npz_paths, self.json_paths):
            try:
                with np.load(npz_p, allow_pickle=False) as z:
                    if "traces" not in z.files:
                        bad.append((npz_p.name, "missing 'traces'"))
                        continue
                    n_traces = int(z["traces"].shape[0])
                with open(json_p, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                n_meta = len(meta.get("metas", []))
                if n_traces != n_meta:
                    bad.append((npz_p.name, f"traces={n_traces} metas={n_meta}"))
            except (zipfile.BadZipFile, OSError, ValueError) as e:
                bad.append((npz_p.name, f"{type(e).__name__}: {e}"))

        if bad:
            msg = "Shard integrity check failed for:\n" + "\n".join(
                f"  {n}: {why}" for n, why in bad
            )
            raise RuntimeError(msg)
        # hyperfine features (A∥, B⊥, r, x,y,z)
        self._hf = _read_hyperfine_table(
            hyperfine_path,
            distance_cutoff=8.0,
            NV_AXIS=NV_AXIS_UNIT,
            UNITS_A_IS_MHZ=True,
        )
        self._hf_ids = np.array(sorted(self._hf.keys()), dtype=np.int64)

        # B-field bookkeeping
        self.B = None
        if B_vec_G is not None:
            self.B = np.asarray(B_vec_G, dtype=float)
            self.B_mag = float(np.linalg.norm(self.B))
            self.B_par = float(np.dot(self.B, NV_AXIS_UNIT))
            self.B_perp = float(np.sqrt(max(self.B_mag**2 - self.B_par**2, 0.0)))
        else:
            self.B_mag = self.B_par = self.B_perp = 0.0

        # RNG for negatives
        self._rng = np.random.default_rng(rng_seed)

        # Precompute feature normalization (z-score over all sites for used keys)
        all_rows = []
        for sid, site in self._hf.items():
            all_rows.append(
                _build_one_feat_row(
                    site, self.feature_keys, self.B, self.B_mag, self.B_par, self.B_perp
                )
            )
        A = (
            np.stack(all_rows, axis=0)
            if all_rows
            else np.zeros((1, len(self.feature_keys)), np.float32)
        )
        self.mu = A.mean(axis=0).astype(np.float32)
        self.sigma = (A.std(axis=0) + 1e-6).astype(np.float32)

    def __len__(self) -> int:
        return len(self._index)

    # def _load_npz(self, shard_idx: int) -> Dict[str, np.ndarray]:
    #     """
    #     Robust shard loader: direct np.load with copy-out, small retry.
    #     """
    #     path = str(self.npz_paths[shard_idx])
    #     last_err = None
    #     for attempt in range(3):
    #         try:
    #             with np.load(path, allow_pickle=False, mmap_mode=None) as z:
    #                 arrays = {k: np.array(z[k], copy=True) for k in z.files}
    #             # normalize dtypes/contiguity
    #             if "traces" in arrays:
    #                 arrays["traces"] = np.ascontiguousarray(
    #                     arrays["traces"], dtype=np.float32
    #                 )
    #             if "taus_us" in arrays:
    #                 arrays["taus_us"] = np.ascontiguousarray(
    #                     arrays["taus_us"], dtype=np.float32
    #                 )
    #             return arrays
    #         except OSError as e:
    #             last_err = e
    #             time.sleep(0.05 * (attempt + 1))
    #     raise last_err

    def _load_npz(self, shard_idx: int) -> Dict[str, np.ndarray]:
        path = self.npz_paths[shard_idx]
        last_err = None
        for attempt in range(3):
            try:
                # skip obviously incomplete files
                if path.stat().st_size < 4096:
                    raise zipfile.BadZipFile("too small to be a valid npz")
                # read atomically into RAM
                with open(path, "rb") as f:
                    blob = f.read()
                with np.load(io.BytesIO(blob), allow_pickle=False) as z:
                    arrays = {k: np.array(z[k], copy=True) for k in z.files}
                # normalize dtypes/contiguity
                arrays["traces"] = np.ascontiguousarray(
                    arrays["traces"], dtype=np.float32
                )
                arrays["taus_us"] = np.ascontiguousarray(
                    arrays["taus_us"], dtype=np.float32
                )
                return arrays
            except (zipfile.BadZipFile, OSError, ValueError) as e:
                last_err = e
                time.sleep(0.05 * (attempt + 1))
        raise zipfile.BadZipFile(
            f"Failed to read npz {path.name}: {last_err}"
        ) from last_err

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

        feat_rows = []
        for sid in all_ids:
            site = self._hf.get(int(sid))
            if site is None:
                feat_rows.append(np.zeros((len(self.feature_keys),), np.float32))
            else:
                feat_rows.append(
                    _build_one_feat_row(
                        site,
                        self.feature_keys,
                        self.B,
                        self.B_mag,
                        self.B_par,
                        self.B_perp,
                    )
                )
        feats = (
            np.stack(feat_rows, axis=0)
            if feat_rows
            else np.zeros((0, len(self.feature_keys)), np.float32)
        )

        # z-score
        feats = (feats - self.mu[None, :]) / self.sigma[None, :]

        return {
            "trace": trace,  # [T]
            "cand_feats": torch.from_numpy(feats),  # [M, F]
            "cand_labels": torch.from_numpy(labels),  # [M]
            "trace_id": f"{self.npz_paths[si].stem}/{li}",
        }


# ---------------------- collate ----------------------
def packed_collate(batch):
    B = len(batch)
    traces = torch.stack([b["trace"] for b in batch], dim=0)  # [B, T]

    feats, labels, idx_map, ids = [], [], [], []
    F = None
    for b, s in enumerate(batch):
        M = s["cand_feats"].shape[0]
        if F is None:
            F = int(s["cand_feats"].shape[1]) if s["cand_feats"].ndim == 2 else 0
        if M > 0:
            feats.append(s["cand_feats"])
            labels.append(s["cand_labels"])
            idx_map.append(torch.full((M,), b, dtype=torch.long))
        ids.append(s["trace_id"])

    cand_feats = (
        torch.cat(feats, dim=0) if feats else torch.zeros((0, F), dtype=torch.float32)
    )
    cand_labels = (
        torch.cat(labels, dim=0) if labels else torch.zeros((0,), dtype=torch.float32)
    )
    cand_idx_to_trace = (
        torch.cat(idx_map, dim=0) if idx_map else torch.zeros((0,), dtype=torch.long)
    )

    return {
        "traces": traces,
        "cand_feats": cand_feats,
        "cand_labels": cand_labels,
        "cand_idx_to_trace": cand_idx_to_trace,
        "cand_batch_idx": cand_idx_to_trace,
        "trace_ids": ids,
    }
