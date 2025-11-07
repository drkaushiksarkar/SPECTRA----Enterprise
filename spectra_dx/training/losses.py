from __future__ import annotations
import math, torch, torch.nn.functional as F

def pinball(p, t, q):
    e = t - p
    return torch.mean(torch.maximum(q*e, (q-1)*e))

def tv_l1(x):
    d = x[:,1:] - x[:,:-1]
    return d.abs().mean(), x.abs().mean()

def gradient_penalty(critic, yr, yf, cond, cc, lam=10.0, device="cpu"):
    B = yr.size(0)
    eps = torch.rand(B,1,1, device=device)
    xi = (eps*yr + (1-eps)*yf).requires_grad_(True)
    s,_ = critic(cond, xi, cc)
    g = torch.autograd.grad(s, xi, grad_outputs=torch.ones_like(s), create_graph=True, retain_graph=True, only_inputs=True)[0]
    return ((g.view(B,-1).norm(2,1) - 1)**2).mean() * lam

def generator_objective(G, D, X, Y, C, z, w, pc_idx, device="cpu", ablate=None, quantiles=(0.1,0.5,0.9)):
    mu, ls, qs = G(X, z, C)
    sig = (ls.exp()).clamp(1e-3, 50.0)
    if ablate and ablate.get("heteroscedastic", True):
        nll = 0.5*(((Y - mu)/sig)**2 + 2*ls + math.log(2*math.pi)).mean()
    else:
        nll = F.l1_loss(mu, Y)

    ql = 0.0
    if ablate and ablate.get("quantiles", True):
        for i, q in enumerate(quantiles):
            ql += pinball(qs[i], Y, q)
        ql /= len(quantiles)

    # Smoothness
    tv, l1 = tv_l1(X[..., pc_idx])
    fm = 0.0; adv_term = 0.0
    if ablate and ablate.get("adversarial", True):
        with torch.no_grad():
            _, fr = D(X, Y, C)
        s_fake, ff = D(X, mu, C)
        fm = F.l1_loss(ff, fr); adv_term = -s_fake.mean()

    return w["nll"]*nll + w["q"]*ql + w["tv"]*tv + w["l1"]*l1 + (w["fm"]*fm if ablate.get("adversarial", True) else 0.0) + (w["adv"]*adv_term if ablate.get("adversarial", True) else 0.0)
