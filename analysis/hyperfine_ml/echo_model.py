# ml/echo_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class TraceEncoder(nn.Module):
    def __init__(self, in_len: int, d_latent: int = 128):
        super().__init__()
        # simple 1D conv stack → global pool
        self.conv1 = nn.Conv1d(1, 32, kernel_size=9, padding=4)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=9, padding=4)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=9, padding=4)
        self.proj  = nn.Linear(128, d_latent)

    def forward(self, x):  # x: [B, T]
        x = x.unsqueeze(1)                 # [B,1,T]
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = F.gelu(self.conv3(x))
        x = x.mean(dim=-1)                 # [B, 128]
        z = F.normalize(self.proj(x), dim=-1)  # [B, d]
        return z

class SiteEncoder(nn.Module):
    def __init__(self, feat_dim: int, d_latent: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, 128), nn.GELU(),
            nn.Linear(128, d_latent),
        )
    def forward(self, feats):              # feats: [sumM, F]
        return F.normalize(self.mlp(feats), dim=-1)

class EchoMatcher(nn.Module):
    def __init__(self, trace_len: int, feat_dim: int, d_latent: int = 128):
        super().__init__()
        self.trace_enc = TraceEncoder(trace_len, d_latent)
        self.site_enc  = SiteEncoder(feat_dim, d_latent)

    def forward(self, batch):
        z_trace = self.trace_enc(batch["traces"])        # [B,d]
        z_site  = self.site_enc(batch["cand_feats"])     # [sumM,d]
        # map site embeddings to owning trace via cand_batch_idx
        z_owner = z_trace[batch["cand_batch_idx"]]       # [sumM,d]
        logits = (z_owner * z_site).sum(dim=-1)          # [sumM] dot product
        return logits
