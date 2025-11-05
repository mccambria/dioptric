# train_echo.py
# Windows-safe trainer for EchoMatcher with AMP, class-balance weighting,
# and automatic DataLoader fallback if multi-worker zip reads misbehave.

import os
from pathlib import Path
import multiprocessing as mp
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from echo_dataset import ShardedEchoDataset, packed_collate
from echo_model import EchoMatcher

# -------------------- CONFIG --------------------
SHARDS_DIR = r"G:\nvdata\pc_Purcell\branch_master\make_spin_echo_dataset\2025_11\2025_11_04-00_56_54-dataset_spin_echo"
HYPERFINE = r"analysis\nv_hyperfine_coupling\nv-2.txt"
FEATURES = ["r", "A_par_kHz", "B_perp_kHz", "x", "y", "z"]

BATCH_SIZE = 64  # try 96/128 if VRAM allows
NEG_PER = 16
LR = 2e-4
WEIGHT_DEC = 1e-4
GRAD_ACCUM = 2  # larger effective batch without more VRAM
LOG_EVERY = 50
SAVE_EVERY = 1000
CKPT_PATH = str(Path(SHARDS_DIR) / "echo_chkpt" / "echo_chkpt.pt")

# ---- NOTE ----
# For multi-worker stability on Windows, make sure echo_dataset.py copies arrays
# out of np.load(...) and closes the file immediately (no lingering ZipExtFile).
# See the _load_npz_fully(...) pattern we discussed.


# -------------------- UTILS --------------------
def ensure_parent_dir(p: str):
    Path(p).parent.mkdir(parents=True, exist_ok=True)


def memgb():
    if not torch.cuda.is_available():
        return "CUDA not available"
    free, total = torch.cuda.mem_get_info()
    used = (total - free) / 1024**3
    return f"{used:.2f}/{total/1024**3:.2f} GB used"


def make_loader(ds, batch_size, pin_memory=True):
    """
    Try num_workers=2 → 1 → 0 automatically.
    """
    for nw in (2, 1, 0):
        try:
            dl = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=True,
                num_workers=nw,
                collate_fn=packed_collate,
                pin_memory=pin_memory and torch.cuda.is_available(),
                persistent_workers=False,  # safer on Windows
                prefetch_factor=2 if nw > 0 else None,
                drop_last=False,
            )
            # Probe one batch to trigger worker path early
            _ = next(iter(dl))
            print(f"[dataloader] ok with num_workers={nw}")
            return dl
        except Exception as e:
            print(f"[dataloader] num_workers={nw} failed → {e}")
            continue
    raise RuntimeError("All DataLoader worker settings failed (2→1→0).")


def maybe_load_ckpt(model, opt, scaler):
    if not CKPT_PATH or not Path(CKPT_PATH).exists():
        return 0
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)
    opt.load_state_dict(ckpt["opt"])
    try:
        scaler.load_state_dict(ckpt["scaler"])
    except Exception:
        pass
    step0 = int(ckpt.get("step", 0))
    print(f"[ckpt] resumed from {CKPT_PATH} @ step {step0}")
    return step0


# -------------------- MAIN --------------------
def main():
    # Windows multiprocess start method
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    # CUDA/TF32 niceties
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[init] device={device}")

    # Dataset
    ds = ShardedEchoDataset(
        shards_dir=SHARDS_DIR,
        hyperfine_path=HYPERFINE,
        feature_keys=FEATURES,
        negatives_per_trace=NEG_PER,
        rng_seed=20251102,
        drop_empty=True,
    )

    # DataLoader with fallback logic
    dl = make_loader(ds, BATCH_SIZE, pin_memory=True)

    # Infer dims from the probed batch already pulled by make_loader()
    # (re-get the first batch for clarity)
    first = next(iter(dl))
    T = first["traces"].shape[1]
    F = first["cand_feats"].shape[1]

    # Model
    model = EchoMatcher(trace_len=T, feat_dim=F, d_latent=128).to(device)
    try:
        model = torch.compile(model)  # optional, fine to disable if it errors
    except Exception as e:
        print("[warn] torch.compile disabled:", e)

    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DEC)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # Resume if checkpoint exists
    ensure_parent_dir(CKPT_PATH)
    global_step = maybe_load_ckpt(model, opt, scaler)

    model.train()
    opt.zero_grad(set_to_none=True)

    # Training loop
    for step, batch in enumerate(dl, start=1 + global_step):
        # H2D
        batch = {
            k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()
        }
        y = batch["cand_labels"]

        # Per-batch class balance
        n_pos = int((y > 0.5).sum().item())
        n_neg = int((y <= 0.5).sum().item())
        if n_pos == 0 or n_neg == 0:
            continue
        pos_weight = torch.tensor([n_neg / max(1, n_pos)], device=device)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(batch)  # [sumM]
            loss = (
                torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, y, pos_weight=pos_weight
                )
                / GRAD_ACCUM
            )

        scaler.scale(loss).backward()

        if (step % GRAD_ACCUM) == 0:
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

        # First-batch info
        if step == 1 + global_step:
            B = int(batch["traces"].shape[0])
            sumM = int(batch["cand_feats"].shape[0])
            Mavg = sumM / max(1, B)
            print(
                f"[batch] B={B}, T={T}, sumM={sumM}, M≈{Mavg:.1f}, pos={n_pos}, neg={n_neg}"
            )
            print("[mem ]", memgb())

        # Logs
        if step % LOG_EVERY == 0:
            with torch.no_grad():
                p = torch.sigmoid(logits)
                pos_m = p[y > 0.5].mean().item() if (y > 0.5).any() else float("nan")
                neg_m = p[y <= 0.5].mean().item() if (y <= 0.5).any() else float("nan")
                g2 = 0.0
                for _, param in model.named_parameters():
                    if param.grad is not None:
                        g = param.grad.detach().float()
                        g2 += (g * g).sum().item()
            print(
                f"step {step:6d} | loss {(loss.item()*GRAD_ACCUM):.4f} | P+ {pos_m:.3f}  P- {neg_m:.3f} | grad^2 {g2:.3e}"
            )

        # Checkpoint
        if SAVE_EVERY and (step % SAVE_EVERY == 0):
            torch.save(
                {
                    "step": step,
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "scaler": scaler.state_dict(),
                    "config": {
                        "T": T,
                        "F": F,
                        "BATCH_SIZE": BATCH_SIZE,
                        "NEG_PER": NEG_PER,
                        "LR": LR,
                        "WEIGHT_DEC": WEIGHT_DEC,
                        "GRAD_ACCUM": GRAD_ACCUM,
                        "features": FEATURES,
                        "shards_dir": SHARDS_DIR,
                        "hyperfine": HYPERFINE,
                    },
                },
                CKPT_PATH,
            )
            print(f"[ckpt] saved → {CKPT_PATH}")


if __name__ == "__main__":
    main()
