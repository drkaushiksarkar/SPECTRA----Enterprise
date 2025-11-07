from __future__ import annotations
import numpy as np, pandas as pd

def add_calendar(df: pd.DataFrame):
    df["month_sin"] = np.sin(2*np.pi*df["Month"]/12.).astype(np.float32)
    df["month_cos"] = np.cos(2*np.pi*df["Month"]/12.).astype(np.float32)
    return df

def latent_moments(df: pd.DataFrame, lat_dim: int, with_std: bool = True):
    # Placeholder aggregator for per-cell latents (if present)
    mus, sgs = [], []
    for _ in range(len(df)):
        mus.append(np.zeros(lat_dim, np.float32))
        sgs.append(np.zeros(lat_dim, np.float32))
    Zm = np.vstack(mus); Zs = np.vstack(sgs)
    cols = []
    for i in range(lat_dim):
        df[f"pc_mean_{i+1}"] = Zm[:, i].astype(np.float32); cols.append(f"pc_mean_{i+1}")
        if with_std:
            df[f"pc_std_{i+1}"] = Zs[:, i].astype(np.float32); cols.append(f"pc_std_{i+1}")
    return df, cols + ["month_sin","month_cos"]
