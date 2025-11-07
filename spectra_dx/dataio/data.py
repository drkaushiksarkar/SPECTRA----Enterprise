from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_climate(path: str, spatial_cols: list[str]):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df["Year"] = df["date"].dt.year
    df["Month"] = df["date"].dt.month
    df["country_code"] = pd.Categorical(df["country"]).codes
    df["year_code"] = pd.Categorical(df["Year"]).codes
    X = df[spatial_cols].values.astype(np.float32)
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X).astype(np.float32)
    return df, Xs, sc

def load_dengue(path: str, country_col: str, target_col: str):
    df = pd.read_csv(path)
    df = df.sort_values([country_col, "Year", "Month"])
    if target_col not in df:
        # Create placeholder target if missing (will be overwritten by user data)
        df[target_col] = np.nan
    return df

def sequences_by_country(df: pd.DataFrame, X: np.ndarray, y: np.ndarray, seq_len: int, id_col: str):
    ym = (df["Year"].astype(int) * 12 + df["Month"].astype(int)).values
    C = df[id_col].values
    SX, SY, SC = [], [], []
    for cid in np.unique(C):
        idx = np.where(C == cid)[0]
        idx = idx[np.argsort(ym[idx])]
        months = ym[idx]
        for i in range(len(idx) - seq_len + 1):
            sl = idx[i:i+seq_len]
            if np.all(np.diff(months[i:i+seq_len]) == 1):
                SX.append(X[sl]); SY.append(y[sl]); SC.append(cid)
    return np.asarray(SX, np.float32), np.asarray(SY, np.float32), np.asarray(SC, np.int64)
