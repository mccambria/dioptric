# echo_utils.py
# echo_utils.py (top of file)
import os

os.environ.setdefault("MPLBACKEND", "Agg")  # safe for headless runs

from pathlib import Path
import csv
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

# Optional deps used by plotting/eval helpers:
try:
    import pandas as pd
except Exception:
    pd = None  # plot_training_curves will skip if pd is None

try:
    from sklearn.metrics import (
        precision_recall_curve,
        average_precision_score,
        roc_curve,
        auc,
    )
except Exception:
    precision_recall_curve = average_precision_score = roc_curve = auc = None


def _safe_read_csv(path):
    if pd is None:
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None

# ---------- FS / CUDA ----------
def ensure_parent_dir(p: str):
    Path(p).parent.mkdir(parents=True, exist_ok=True)


def memgb():
    if not torch.cuda.is_available():
        return "CUDA not available"
    free, total = torch.cuda.mem_get_info()
    used = (total - free) / 1024**3
    return f"{used:.2f}/{total/1024**3:.2f} GB used"


# ---------- CSV logger ----------
class MetricsLog:
    def __init__(self, csv_path):
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.csv_path, "w", newline="") as f:
                csv.writer(f).writerow(["step", "loss", "ap", "r1", "r3", "r5"])

    def add(self, step, loss=None, ap=None, r1=None, r3=None, r5=None):
        row = [
            int(step),
            "" if loss is None else float(loss),
            "" if ap is None else float(ap),
            "" if r1 is None else float(r1),
            "" if r3 is None else float(r3),
            "" if r5 is None else float(r5),
        ]
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow(row)


# ---------- plotting ----------
def _safe_read_csv(path):
    import pandas as pd

    try:
        return pd.read_csv(path)
    except Exception:
        return None


def plot_training_curves(csv_path, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = _safe_read_csv(csv_path)
    if df is None or len(df) == 0:
        return
    if "loss" in df.columns:
        plt.figure(figsize=(6, 4))
        df.dropna(subset=["loss"]).plot(x="step", y="loss", legend=False)
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.title("Training loss")
        plt.tight_layout()
        plt.savefig(out / "loss_curve.png", dpi=160)
        plt.close()
    for col, title in [
        ("ap", "AP"),
        ("r1", "Recall@1"),
        ("r3", "Recall@3"),
        ("r5", "Recall@5"),
    ]:
        if col in df.columns and df[col].notna().any():
            plt.figure(figsize=(6, 4))
            df.dropna(subset=[col]).plot(x="step", y=col, legend=False)
            plt.xlabel("step")
            plt.ylabel(col)
            plt.title(title)
            plt.tight_layout()
            plt.savefig(out / f"{col}_curve.png", dpi=160)
            plt.close()


def plot_pr_roc(y_true, y_score, out_dir, tag):
    from sklearn.metrics import (
        precision_recall_curve,
        average_precision_score,
        roc_curve,
        auc,
    )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(y_true, np.float32)
    y_score = np.asarray(y_score, np.float32)
    if y_true.size == 0:
        return
    P, R, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(5, 4))
    plt.plot(R, P)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR {tag} (AP={ap:.3f})")
    plt.tight_layout()
    plt.savefig(out / f"pr_{tag}.png", dpi=160)
    plt.close()
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], "--", lw=0.8)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"ROC {tag} (AUC={roc_auc:.3f})")
    plt.tight_layout()
    plt.savefig(out / f"roc_{tag}.png", dpi=160)
    plt.close()


def plot_score_hist(y_true, y_score, out_dir, tag):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(y_true, np.float32)
    y_score = np.asarray(y_score, np.float32)
    pos = y_score[y_true > 0.5]
    neg = y_score[y_true <= 0.5]
    plt.figure(figsize=(6, 4))
    plt.hist(neg, bins=40, alpha=0.6, label="neg")
    plt.hist(pos, bins=40, alpha=0.6, label="pos")
    plt.xlabel("score")
    plt.ylabel("count")
    plt.legend()
    plt.title(f"Scores {tag}")
    plt.tight_layout()
    plt.savefig(out / f"scores_{tag}.png", dpi=160)
    plt.close()


def plot_example_trace_with_topk(batch, logits, out_dir, tag, k=5):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        scores = torch.sigmoid(logits).detach().cpu().numpy()
        labels = batch["cand_labels"].detach().cpu().numpy()
        idx_map = batch["cand_idx_to_trace"].detach().cpu().numpy()
        traces = batch["traces"].detach().cpu().numpy()
    tid = 0
    m = idx_map == tid
    s = scores[m]
    l = labels[m]
    topk = np.argsort(-s)[:k]
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4))
    plt.plot(traces[tid])
    info = ", ".join([f"{s[i]:.2f}" for i in topk])
    plt.title(f"Trace 0 | top-{k} scores: {info} | pos@topk={int(l[topk].sum())}/{k}")
    plt.xlabel("τ index")
    plt.ylabel("signal")
    plt.tight_layout()
    plt.savefig(out / f"trace_topk_{tag}.png", dpi=160)
    plt.close()


# ---------- eval helpers ----------
@torch.no_grad()
def eval_once(model, dl, device):
    from sklearn.metrics import average_precision_score

    model.eval()
    y_all, p_all = [], []
    r1 = r3 = r5 = 0
    N = 0
    for batch in dl:
        batch = {
            k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
        }
        y = batch["cand_labels"]
        p = torch.sigmoid(model(batch))
        y_all.append(y.cpu())
        p_all.append(p.cpu())
        idx = batch["cand_batch_idx"]
        B = int(batch["traces"].shape[0])
        for b in range(B):
            sel = idx == b
            if sel.sum() == 0:
                continue
            yb, pb = y[sel], p[sel]
            if (yb > 0.5).any():
                N += 1
                top = torch.topk(pb, k=min(5, pb.numel())).indices
                ytop = yb[top]
                r1 += int((ytop[:1] > 0.5).any().item())
                r3 += int((ytop[:3] > 0.5).any().item())
                r5 += int((ytop[:5] > 0.5).any().item())
    ap = average_precision_score(torch.cat(y_all).numpy(), torch.cat(p_all).numpy())
    model.train()
    return ap, r1 / max(1, N), r3 / max(1, N), r5 / max(1, N)


@torch.no_grad()
def eval_collect(model, dl, device):
    model.eval()
    ys, ps = [], []
    for batch in dl:
        batch = {
            k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
        }
        logits = model(batch)
        ys.append(batch["cand_labels"].detach().cpu().numpy())
        ps.append(torch.sigmoid(logits).detach().cpu().numpy())
    model.train()
    import numpy as np

    return np.concatenate(ys) if ys else np.array([]), (
        np.concatenate(ps) if ps else np.array([])
    )
