from __future__ import annotations
import argparse
from spectra_dx.training.train_global import train as train_global_impl
from spectra_dx.training.finetune import finetune_country as finetune_impl
from spectra_dx.evaluation.eval import evaluate as eval_impl
from spectra_dx.ablation.run import run_ablation as ablate_impl

def train_global():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    a = p.parse_args()
    train_global_impl(a.config)

def finetune_country():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--country", required=True)
    a = p.parse_args()
    finetune_impl(a.config, a.country)

def evaluate():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    a = p.parse_args()
    eval_impl(a.config)

def ablate():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    a = p.parse_args()
    ablate_impl(a.config)

def visualize():
    print("Visualization entrypoint placeholder: produces attention maps, diagnostics, SHAP-like plots.")
