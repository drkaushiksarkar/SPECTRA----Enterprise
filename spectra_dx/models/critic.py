from __future__ import annotations
import torch.nn as nn, torch.nn.functional as F

class Critic(nn.Module):
    def __init__(self, cond_dim: int, n_country: int, emb: int = 8, lstm: int = 128, drop: float = 0.25):
        super().__init__()
        self.ce = nn.Embedding(n_country, emb)
        self.lstm = nn.LSTM(cond_dim + 1 + emb, lstm, 1, batch_first=True)
        self.fc = nn.Linear(lstm, 64)
        self.drop = nn.Dropout(drop)
        self.out = nn.Linear(64, 1)

    def forward(self, cond, ts, c):
        B, L, D = cond.shape
        e = self.ce(c).unsqueeze(1).repeat(1, L, 1)
        h,_ = self.lstm(torch.cat([cond, ts, e], -1))
        f = F.gelu(self.fc(h[:, -1, :]))
        return self.out(self.drop(f)).squeeze(1), f
