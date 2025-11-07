from __future__ import annotations
import torch, torch.nn as nn

from .tcn import CausalTCN

class Generator(nn.Module):
    def __init__(self, cond_dim: int, noise_dim: int, n_country: int, emb: int = 8, lstm_units: int = 128, heads: int = 8, drop: float = 0.25, quantiles=(0.1,0.5,0.9)):
        super().__init__()
        self.q = quantiles
        self.ce = nn.Embedding(n_country, emb)
        self.pn = nn.Linear(noise_dim, cond_dim)
        self.tcn = CausalTCN(cond_dim*2 + emb, hid=lstm_units)
        self.lstm = nn.LSTM(lstm_units, lstm_units, 1, batch_first=True)
        self.mha = nn.MultiheadAttention(lstm_units, heads, batch_first=True, dropout=drop)
        self.ln = nn.LayerNorm(lstm_units)
        self.drop = nn.Dropout(drop)
        self.mu = nn.Linear(lstm_units, 1)
        self.ls = nn.Linear(lstm_units, 1)
        self.qh = nn.ModuleList([nn.Linear(lstm_units, 1) for _ in self.q])
        self.last_attn = None

    def forward(self, cond, noise, cid):
        B, L, D = cond.shape
        emb = self.ce(cid).unsqueeze(1).repeat(1, L, 1)
        z = self.pn(noise)
        h = self.tcn(torch.cat([cond, z, emb], -1))
        h, _ = self.lstm(h)
        L = h.size(1)
        mask = torch.triu(torch.ones(L, L, device=h.device, dtype=torch.bool), diagonal=1)
        att, w = self.mha(h, h, h, attn_mask=mask, need_weights=True)
        self.last_attn = w.detach()
        h = self.drop(self.ln(att))
        mu = self.mu(h)
        ls = torch.clamp(self.ls(h), -5.0, 3.0)
        qs = [q(h) for q in self.qh]
        return mu, ls, qs
