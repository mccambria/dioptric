# train_echo.py
# Windows-safe trainer for EchoMatcher with warmup, class-balance weighting,
# and DataLoader fallback.

import os

os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")

from pathlib import Path
import multiprocessing as mp
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import math
from echo_dataset import ShardedEchoDataset, packed_collate
from echo_model import EchoMatcher
import importlib.util
import numpy as np
import random

from echo_utils import (
    ensure_parent_dir,
    memgb,
    MetricsLog,
    plot_training_curves,
    plot_pr_roc,
    plot_score_hist,
    plot_example_trace_with_topk,
    eval_once,
    eval_collect,
)


def _has_triton() -> bool:
    return importlib.util.find_spec("triton") is not None


def _should_compile() -> bool:
    # Avoid torch.compile on Windows (NT) or when Triton is missing
    if os.name == "nt":
        return False
    if not _has_triton():
        return False
    return True


# -------------------- CONFIG --------------------
# Measured lab field (Gauss)
B_VEC_G = (-46.18287122, -17.44411563, -5.57779074)
# Features now include B-derived + frequency hints (dataset will z-score them)
FEATURES = [
    "r",
    "A_par_kHz",
    "B_perp_kHz",
    "x",
    "y",
    "z",
    "B_mag_G",
    "B_par_G",
    "B_perp_G",
    "cosBrhat",
    "f0_MHz",
    "f1p_MHz",
    "f1m_MHz",
]
BATCH_SIZE = 128
NEG_PER = 8
LR = 5e-4
WEIGHT_DEC = 1e-4
GRAD_ACCUM = 1
LOG_EVERY = 50
SAVE_EVERY = 100
VAL_EVERY = 500
VAL_PLOTS_EVERY = 1000  # heavy plots less often than metrics
D_LATENT = 256

# Paths
SHARDS_DIR = r"G:\nvdata\pc_slmmachine\branch_master\make_spin_echo_dataset\2025_11"
DATA_DIR = str(Path(SHARDS_DIR) / "2025_11_04-23_21_07-dataset_spin_echo")
HYPERFINE = r"analysis\nv_hyperfine_coupling\nv-2.txt"
CKPT_PATH = str(
    Path(SHARDS_DIR) / "echo_chkpt" / f"echo_chkpt_lat{D_LATENT}_neg{NEG_PER}.pt"
)
PLOTS_DIR = Path(CKPT_PATH).with_name("plots")
LOG_CSV = PLOTS_DIR / "train_log.csv"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
log = MetricsLog(LOG_CSV)


# -------------------- UTILS --------------------
def _seed_worker(worker_id):
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def make_loader(ds, batch_size, pin_memory=True):
    """
    Try num_workers=6 → 4 → 2 → 1 → 0 automatically.
    """
    for nw in (6, 4, 2, 1, 0):
        try:
            dl = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=True,
                num_workers=nw,
                collate_fn=packed_collate,
                pin_memory=pin_memory and torch.cuda.is_available(),
                persistent_workers=False,  # safer on Windows
                prefetch_factor=4 if nw > 0 else None,
                drop_last=False,
                worker_init_fn=_seed_worker,  # reproducible shuffling
            )
            _ = next(iter(dl))  # probe one batch
            print(f"[dataloader] ok with num_workers={nw}")
            return dl
        except Exception as e:
            print(f"[dataloader] num_workers={nw} failed → {e}")
            continue
    raise RuntimeError("All DataLoader worker settings failed (6→4→2→1→0).")


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


def set_lr(step, opt, TOTAL_STEPS=20000):
    warm = 1000
    if step < warm:
        scale = step / max(1, warm)
    else:
        t = (step - warm) / max(1, TOTAL_STEPS - warm)
        scale = 0.5 * (1 + math.cos(math.pi * t))
    for g in opt.param_groups:
        g["lr"] = LR * scale


def focal_bce_with_logits(
    logits,
    targets,
    alpha: float | None = None,  # e.g. 0.25; set None if using pos_weight
    gamma: float = 2.0,  # try 1.0 first for stability, then 2.0
    pos_weight: torch.Tensor | None = None,  # use either alpha or pos_weight
    reduction: str = "mean",
    eps: float = 1e-6,
):
    # base BCE per-sample (stable, logits-based)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none", pos_weight=pos_weight
    )
    # p_t = prob of the true class
    p = torch.sigmoid(logits)
    p_t = (p * targets) + (1 - p) * (1 - targets)
    p_t = p_t.clamp(min=eps, max=1 - eps)

    # focal modulation
    mod = (1.0 - p_t) ** gamma
    loss = mod * bce

    # alpha balance (only if not also using pos_weight)
    if alpha is not None:
        w = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        loss = w * loss

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss


def sample_val_batch(val_dl, device):
    # lightweight: just grab one mini-batch from the val loader
    val_iter = iter(val_dl)
    try:
        batch = next(val_iter)
    except StopIteration:
        return None
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    return batch


# -------------------- MAIN --------------------
def main():
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[init] device={device}")

    # Dataset
    ds = ShardedEchoDataset(
        shards_dir=DATA_DIR,
        hyperfine_path=HYPERFINE,
        B_vec_G=B_VEC_G,
        feature_keys=FEATURES,
        negatives_per_trace=NEG_PER,
        rng_seed=20251102,
        drop_empty=True,
    )

    # after you build 'ds'
    idxs = list(range(len(ds)))
    val_frac = 0.1
    split = int(len(ds) * (1.0 - val_frac))
    train_idx, val_idx = idxs[:split], idxs[split:]
    train_ds = torch.utils.data.Subset(ds, train_idx)
    val_ds = torch.utils.data.Subset(ds, val_idx)
    # DataLoader
    train_dl = make_loader(train_ds, BATCH_SIZE, pin_memory=True)
    val_dl = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        collate_fn=packed_collate,
        pin_memory=True,
    )

    # Infer dims
    first = next(iter(train_dl))
    T = first["traces"].shape[1]
    F = first["cand_feats"].shape[1]

    # Model
    model = EchoMatcher(trace_len=T, feat_dim=F, d_latent=D_LATENT).to(device)
    if _should_compile():
        try:
            model = torch.compile(model, mode="default")
        except Exception as e:
            print("[warn] torch.compile disabled (setup-time):", e)
    else:
        print("[info] torch.compile disabled (Windows or no Triton)")

    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DEC)

    # (Stabilize first runs in FP32; AMP can be added later)
    scaler = None

    ensure_parent_dir(CKPT_PATH)
    global_step = maybe_load_ckpt(model, opt, scaler)

    model.train()
    opt.zero_grad(set_to_none=True)

    # warmup = 0  # 0 => disable warmup
    # If you use set_lr with cosine schedule:
    # TOTAL_STEPS = 20000  # or steps_per_epoch * NUM_EPOCHS
    best_ap = -1.0
    EPOCHS = 8
    steps_per_epoch = math.ceil(len(train_ds) / BATCH_SIZE)
    TOTAL_STEPS = EPOCHS * steps_per_epoch  # used by set_lr

    global_step = maybe_load_ckpt(model, opt, scaler)
    # Training loop

    for epoch in range(EPOCHS):
        # for step, batch in enumerate(train_dl, start=1 + global_step):
        for step_in_epoch, batch in enumerate(train_dl, start=1):
            step = global_step + (epoch * steps_per_epoch) + step_in_epoch
            set_lr(step, opt, TOTAL_STEPS=TOTAL_STEPS)
            # sanitize and H2D
            for k in ("traces", "cand_feats", "cand_labels"):
                if torch.is_tensor(batch[k]):
                    batch[k] = torch.nan_to_num(
                        batch[k], nan=0.0, posinf=1e6, neginf=-1e6
                    )
            batch = {
                k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                for k, v in batch.items()
            }
            y = batch["cand_labels"]

            # class balance
            n_pos = int((y > 0.5).sum().item())
            n_neg = int((y <= 0.5).sum().item())
            if n_pos == 0 or n_neg == 0:
                continue
            pos_weight = torch.tensor([n_neg / max(1, n_pos)], device=device)

            # if warmup > 0:
            #     set_lr(step)  # warmup

            # forward
            logits = model(batch)

            loss = (
                focal_bce_with_logits(
                    logits,
                    y,
                    alpha=None,  # disable alpha if using pos_weight
                    gamma=1.0,  # start at 1.0; move to 2.0 if stable
                    pos_weight=pos_weight,
                )
                / GRAD_ACCUM
            )

            if not torch.isfinite(loss):
                print(f"[warn] non-finite loss at step {step}; skipping batch")
                opt.zero_grad(set_to_none=True)
                continue

            # backward
            loss.backward()

            # grad^2 (for logging)
            g2 = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    g = p.grad.detach().float()
                    g2 += (g * g).sum().item()

            # clip & step
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if (step % GRAD_ACCUM) == 0:
                opt.step()
                opt.zero_grad(set_to_none=True)

            # first batch info
            if step == 1 + global_step:
                B = int(batch["traces"].shape[0])
                sumM = int(batch["cand_feats"].shape[0])
                Mavg = sumM / max(1, B)
                print(
                    f"[batch] B={B}, T={T}, sumM={sumM}, M≈{Mavg:.1f}, pos={n_pos}, neg={n_neg}"
                )
                print("[mem ]", memgb())

            # logs
            if step % LOG_EVERY == 0:
                with torch.no_grad():
                    p = torch.sigmoid(logits)
                    pos_m = (
                        p[y > 0.5].mean().item() if (y > 0.5).any() else float("nan")
                    )
                    neg_m = (
                        p[y <= 0.5].mean().item() if (y <= 0.5).any() else float("nan")
                    )
                print(
                    f"step {step:6d} | loss {(loss.item()*GRAD_ACCUM):.4f} | "
                    f"P+ {pos_m:.3f}  P- {neg_m:.3f} | grad^2 {g2:.3e}"
                )

            # quick scalar logs every LOG_EVERY
            if step % LOG_EVERY == 0:
                log.add(step, loss=float(loss.item() * GRAD_ACCUM))
                plot_training_curves(LOG_CSV, PLOTS_DIR)  # make sure it closes figures

            # metrics + checkpointing every VAL_EVERY
            if step % VAL_EVERY == 0:
                ap, r1, r3, r5 = eval_once(model, val_dl, device)
                print(
                    f"[val] step {step} | AP={ap:.3f} | R@1={r1:.3f} R@3={r3:.3f} R@5={r5:.3f}"
                )
                log.add(step, ap=ap, r1=r1, r3=r3, r5=r5)
                plot_training_curves(LOG_CSV, PLOTS_DIR)

                if ap > best_ap:
                    best_ap = ap
                    torch.save(
                        {"step": step, "model": model.state_dict()},
                        Path(CKPT_PATH).with_name("best.pt"),
                    )
                    print(f"[ckpt] ↑ new best AP {best_ap:.3f} saved to best.pt")
                else:
                    print(f"[ckpt] best AP so far {best_ap:.3f} (no save)")

                # heavy diagnostics (PR/ROC, hist, example trace) less often
                if step % VAL_PLOTS_EVERY == 0:
                    with torch.no_grad():
                        y_true, y_score = eval_collect(model, val_dl, device)
                    tag = f"step{step}"
                    plot_pr_roc(y_true, y_score, PLOTS_DIR, tag)
                    plot_score_hist(y_true, y_score, PLOTS_DIR, tag)

                    # IMPORTANT: use a validation batch (not the training 'batch')
                    vb = sample_val_batch(val_dl, device)
                    if vb is not None:
                        with torch.no_grad():
                            logits_vb = model(vb)
                        plot_example_trace_with_topk(vb, logits_vb, PLOTS_DIR, tag, k=5)

            # checkpoint
            if SAVE_EVERY and (step % SAVE_EVERY == 0):
                torch.save(
                    {
                        "step": step,
                        "model": model.state_dict(),
                        "opt": opt.state_dict(),
                        "scaler": None,
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
                            "B_vec_G": B_VEC_G,
                        },
                    },
                    CKPT_PATH,
                )
                print(f"[ckpt] saved → {CKPT_PATH}")


if __name__ == "__main__":
    main()
