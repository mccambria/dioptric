# train_echo.py
import torch, torch.optim as optim
from torch.utils.data import DataLoader
from ml.echo_dataset import ShardedEchoDataset, packed_collate
from ml.echo_model import EchoMatcher

SHARDS_DIR = "dataset_spin_echo"
HYPERFINE = "analysis/nv_hyperfine_coupling/nv-2.txt"
B_VEC_G   = [-6.18037755, -18.54113264, -43.26264283]  # your lab field (Gauss)
FEATURES  = ["r","A_par_kHz","B_perp_kHz","x","y","z"]

ds = ShardedEchoDataset(
    shards_dir=SHARDS_DIR,
    hyperfine_path=HYPERFINE,
    B_vec_G=B_VEC_G,
    feature_keys=FEATURES,
    negatives_per_trace=16,
    rng_seed=20251102,
    drop_empty=True,
)

dl = DataLoader(ds, batch_size=32, num_workers=4, collate_fn=packed_collate, pin_memory=True)

# infer T and F from first batch
first = next(iter(dl))
T = first["traces"].shape[1]
F = first["cand_feats"].shape[1]
model = EchoMatcher(trace_len=T, feat_dim=F, d_latent=128).cuda()

opt = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
bce = torch.nn.BCEWithLogitsLoss()

for step, batch in enumerate(dl, start=1):
    batch = {k:(v.cuda(non_blocking=True) if torch.is_tensor(v) else v) for k,v in batch.items()}
    logits = model(batch)                   # [sumM]
    loss = bce(logits, batch["cand_labels"])
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

    if step % 50 == 0:
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            # quick sanity: mean prob on positives vs negatives
            pos = probs[batch["cand_labels"] > 0.5].mean().item() if (batch["cand_labels"] > 0.5).any() else float('nan')
            neg = probs[batch["cand_labels"] < 0.5].mean().item() if (batch["cand_labels"] < 0.5).any() else float('nan')
        print(f"step {step:6d} | loss {loss.item():.4f} | P+ {pos:.3f}  P- {neg:.3f}")
