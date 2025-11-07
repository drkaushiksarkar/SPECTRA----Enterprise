from __future__ import annotations
import os, json, math, yaml, time
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader, TensorDataset
from rich import print
from spectra_dx.utils.logging import get_logger
from spectra_dx.utils.repro import set_all_seeds
from spectra_dx.dataio.data import load_climate, load_dengue, sequences_by_country
from spectra_dx.dataio.features import add_calendar, latent_moments
from spectra_dx.models.conditional_vae import ConditionalVAE
from spectra_dx.models.generator import Generator
from spectra_dx.models.critic import Critic
from spectra_dx.training.losses import generator_objective, gradient_penalty

log = get_logger("train-global")

def train(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    set_all_seeds(cfg.get("seed", 42))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Data
    climate_csv = cfg["data"]["climate_csv"]
    dengue_csv = cfg["data"]["dengue_csv"]
    out_dir = cfg["data"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    spatial_cols = cfg["spatial_features"]
    dfc, Xc, scX = load_climate(climate_csv, spatial_cols)
    dfe = load_dengue(dengue_csv, cfg["data"]["country_col"], cfg["data"]["target_col"])

    df = dfc.merge(dfe, on=["Year","Month"], how="inner", suffixes=("_clim","")).copy()
    df["adm_0_name"] = dfc["country"]  # ensure country column
    df = add_calendar(df)

    # Prepare targets (log1p incidence, country-wise standardization)
    target_col = cfg["data"]["target_col"]
    y = np.log1p(np.clip(df[target_col].values.reshape(-1,1), 0, None)).astype(np.float32)
    Ys = np.zeros_like(y, np.float32)
    country_le = {c:i for i, c in enumerate(sorted(df["adm_0_name"].unique()))}
    df["country_id"] = df["adm_0_name"].map(country_le).astype(np.int64)
    scalers = {}
    for cid, g in df.groupby("country_id"):
        idx = g.index.values
        mu, sigma = float(y[idx].mean()), float(y[idx].std() + 1e-6)
        scalers[int(cid)] = (mu, sigma)
        Ys[idx] = (y[idx] - mu) / sigma

    seq_len = cfg["sequence"]["length"]
    Xseq, Yseq, Cseq = sequences_by_country(df, Xc, Ys, seq_len, id_col="country_id")

    # Split naive tail for val/test
    val_h = cfg["splits"]["val_h_months"]; test_h = cfg["splits"]["test_h_months"]
    n = len(Xseq); n_val = max(1, int(n*0.1)); n_test = max(1, int(n*0.1))
    X_train, Y_train, C_train = Xseq[:n-n_val-n_test], Yseq[:n-n_val-n_test], Cseq[:n-n_val-n_test]
    X_val, Y_val, C_val = Xseq[n-n_val-n_test:n-n_test], Yseq[n-n_val-n_test:n-n_test], Cseq[n-n_val-n_test:n-n_test]
    X_test, Y_test, C_test = Xseq[n-n_test:], Yseq[n-n_test:], Cseq[n-n_test:]

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(Y_train), torch.tensor(C_train))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(Y_val), torch.tensor(C_val))
    test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(Y_test), torch.tensor(C_test))

    bs = cfg["train"]["batch_size"]
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=bs, shuffle=False)

    # Models
    n_country = len(country_le)
    n_year = int(df["Year"].nunique())
    inp = Xc.shape[1]
    lat = cfg["model"]["latent_dim"]
    hid = cfg["model"]["hidden_mult"] * lat

    vae = ConditionalVAE(inp=inp, hid=hid, lat=lat, n_country=n_country, n_year=n_year,
                         emb=10, drop=cfg["model"]["dropout"], beta=cfg["model"]["beta"], contractive=cfg["model"]["contractive"]).to(device)

    cond_dim = inp + 1 + 2  # X + prev_y + month sin/cos (simple condition)
    G = Generator(cond_dim=cond_dim, noise_dim=cfg["model"]["noise_dim"], n_country=n_country,
                  emb=8, lstm_units=cfg["model"]["lstm_units"], heads=cfg["model"]["heads"], drop=cfg["model"]["dropout"]).to(device)
    D = Critic(cond_dim=cond_dim, n_country=n_country, emb=8, lstm=cfg["model"]["lstm_units"], drop=cfg["model"]["dropout"]).to(device)

    # Optims
    opt_vae = torch.optim.AdamW(vae.parameters(), lr=cfg["train"]["lr_vae"], weight_decay=cfg["train"]["weight_decay"])
    opt_G = torch.optim.AdamW(G.parameters(), lr=cfg["train"]["lr_g"], weight_decay=cfg["train"]["weight_decay"])
    opt_D = torch.optim.AdamW(D.parameters(), lr=cfg["train"]["lr_d"], weight_decay=cfg["train"]["weight_decay"])

    # Training (skeleton; VAE pretrain skipped for brevity; focus on G/D loop matching the paper)
    weights = {"nll": 1.0, "q": 0.8, "tv": 0.1, "l1": 0.1, "fm": 1.0, "adv": 0.5}
    ablate = {
        "adversarial": cfg["ablations"]["adversarial"],
        "heteroscedastic": cfg["ablations"]["heteroscedastic"],
        "quantiles": cfg["ablations"]["quantiles"],
        "spectral": cfg["ablations"]["spectral"],
    }

    def run_epoch(loader, train=True):
        G.train(train); D.train(train)
        tot = 0.0
        for Xb, Yb, Cb in loader:
            Xb, Yb, Cb = Xb.to(device), Yb.to(device), Cb.to(device)
            # Teacher forcing: simple previous y inclusion (standardized); prepend 0
            prev = torch.cat([torch.zeros_like(Yb[:, :1, :]), Yb[:, :-1, :]], dim=1)
            # cond: X + prev + (month_sin, month_cos) from df (simplified: zeros placeholder)
            cal = torch.zeros((Xb.size(0), Xb.size(1), 2), device=device)
            cond = torch.cat([Xb, prev, cal], dim=-1)

            # Update D
            opt_D.zero_grad(set_to_none=True)
            noise = torch.randn(Xb.size(0), Xb.size(1), cfg["model"]["noise_dim"], device=device)
            with torch.no_grad():
                mu, _, _ = G(cond, noise, Cb)
            sr, _ = D(cond, Yb, Cb)
            sf, _ = D(cond, mu, Cb)
            d_loss = (sf.mean() - sr.mean())
            d_loss = d_loss + gradient_penalty(D, Yb, mu, cond, Cb, lam=10.0, device=device)
            if train:
                d_loss.backward()
                opt_D.step()

            # Update G
            opt_G.zero_grad(set_to_none=True)
            noise = torch.randn(Xb.size(0), Xb.size(1), cfg["model"]["noise_dim"], device=device)
            g_loss = generator_objective(G, D, cond, Yb, Cb, noise, weights, pc_idx=0, device=device, ablate=ablate, quantiles=tuple(cfg["quantiles"]))
            if train:
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(G.parameters(), cfg["train"]["clip_norm"])
                opt_G.step()
            tot += float(g_loss.detach().cpu().item())
        return tot/len(loader)

    best = math.inf; patience = cfg["train"]["patience"]; bad = 0
    for epoch in range(cfg["train"]["epochs_gan"]):
        tr = run_epoch(train_dl, train=True)
        va = run_epoch(val_dl, train=False)
        log.info(f"epoch {epoch:03d} | train={tr:.4f} val={va:.4f}")
        if va < best:
            best = va; bad = 0
            torch.save({"G": G.state_dict(), "D": D.state_dict(), "cfg": cfg}, os.path.join(out_dir, "global_best.pt"))
        else:
            bad += 1
            if bad >= patience:
                break

    # Test inference (mean only as placeholder)
    chk = torch.load(os.path.join(out_dir, "global_best.pt"), map_location=device)
    G.load_state_dict(chk["G"])
    G.eval()
    preds = []; trues = []
    with torch.no_grad():
        for Xb, Yb, Cb in test_dl:
            Xb, Yb, Cb = Xb.to(device), Yb.to(device), Cb.to(device)
            prev = torch.cat([torch.zeros_like(Yb[:, :1, :]), Yb[:, :-1, :]], dim=1)
            cal = torch.zeros((Xb.size(0), Xb.size(1), 2), device=device)
            cond = torch.cat([Xb, prev, cal], dim=-1)
            noise = torch.randn(Xb.size(0), Xb.size(1), cfg["model"]["noise_dim"], device=device)
            mu, _, _ = G(cond, noise, Cb)
            preds.append(mu.cpu().numpy()); trues.append(Yb.cpu().numpy())
    preds = np.concatenate(preds, 0).reshape(-1)
    trues = np.concatenate(trues, 0).reshape(-1)

    # Save quick report
    from sklearn.metrics import mean_squared_error, r2_score
    smape = float(100*np.mean(2*np.abs(preds-trues)/(np.abs(trues)+np.abs(preds)+1e-8)))
    rmse = float(np.sqrt(mean_squared_error(trues, preds)))
    r2 = float(r2_score(trues, preds)))
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump({"SMAPE": smape, "RMSE": rmse, "R2": r2}, f, indent=2)
    log.info(f"Test SMAPE={smape:.2f} RMSE={rmse:.3f} R2={r2:.3f} (standardized scale)")
