from __future__ import annotations
import os, json, yaml
from rich import print

def evaluate(config_path: str):
    # Placeholder: load artifacts and produce per-country tables, plots.
    cfg = yaml.safe_load(open(config_path))
    out = cfg["data"]["out_dir"]
    rep = os.path.join(out, "report.json")
    if os.path.exists(rep):
        print(open(rep).read())
    else:
        print("{'warning': 'no report found; run training first'}")
