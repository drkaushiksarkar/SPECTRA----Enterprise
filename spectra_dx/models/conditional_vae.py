from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F

class ConditionalVAE(nn.Module):
    def __init__(self, inp: int, hid: int, lat: int, n_country: int, n_year: int, emb: int = 10, drop: float = 0.25, beta: float = 2.0, contractive: float = 1e-3):
        super().__init__()
        self.beta = beta
        self.contractive = contractive
        self.ce = nn.Embedding(n_country, emb)
        self.ye = nn.Embedding(n_year, emb)
        self.p = nn.Linear(inp, hid // 2)
        self.f = nn.Linear(hid // 2 + 2*emb, hid)
        self.mu = nn.Linear(hid, lat)
        self.lv = nn.Linear(hid, lat)
        self.fd = nn.Linear(lat + 2*emb, hid)
        self.out = nn.Linear(hid, inp)
        self.l1 = nn.LayerNorm(hid)
        self.l2 = nn.LayerNorm(hid)
        self.drop = nn.Dropout(drop)

    def encode(self, x, c, y):
        h = torch.relu(self.p(x))
        h = torch.cat([h, self.ce(c), self.ye(y)], 1)
        h = self.drop(torch.relu(self.l1(self.f(h))))
        mu = self.mu(h); lv = self.lv(h)
        return mu, lv

    def reparam(self, m, l):
        std = (0.5*l).exp()
        return m + torch.randn_like(std)*std

    def decode(self, z, c, y):
        h = torch.cat([z, self.ce(c), self.ye(y)], 1)
        h = self.drop(torch.relu(self.l2(self.fd(h))))
        return self.out(h)

    def forward(self, x, c, y):
        mu, lv = self.encode(x, c, y)
        z = self.reparam(mu, lv)
        return self.decode(z, c, y), mu, lv

    def loss(self, xh, x, mu, lv, x_in, c, y):
        recon = F.mse_loss(xh, x)
        kl = -0.5*torch.mean(1 + lv - mu.pow(2) - lv.exp())
        x_in.requires_grad_(True)
        mu_c, _ = self.encode(x_in, c, y)
        g = torch.autograd.grad(mu_c.sum(), x_in, create_graph=True)[0]
        contractive = (g.pow(2).sum(1).mean())
        return recon + self.beta*kl + self.contractive*contractive
