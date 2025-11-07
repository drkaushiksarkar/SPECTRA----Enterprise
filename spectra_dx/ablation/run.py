from __future__ import annotations
import yaml
from rich import print

def run_ablation(config_path: str):
    # Placeholder: iterate configuration grid and trigger training with switches.
    cfg = yaml.safe_load(open(config_path))
    print({'ablation': cfg.get('ablations', {})})
