from __future__ import annotations
import numpy as np

def smape(y, p):
    y = np.asarray(y).ravel(); p = np.asarray(p).ravel()
    return 100*np.mean(2*np.abs(p-y)/(np.abs(y)+np.abs(p)+1e-8))

def coverage(y, lo, hi):
    y = np.asarray(y).ravel()
    return float(np.mean((y >= lo) & (y <= hi)))
