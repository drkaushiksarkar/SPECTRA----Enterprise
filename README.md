# SPECTRA-Dx (Enterprise)

A production-grade, modular implementation of **SPECTRA** — the *Spatiotemporal Probabilistic Encoder‑fed Climate‑aware Transfer‑learning‑ready Risk‑forecasting Adversarial* architecture — built for enterprise deployment.

> **Goal**: Learn the forward predictive distribution of climate‑sensitive infectious disease signals using meteorological reanalysis and surveillance histories, support multi‑country modeling, and enable diagnostics‑guided transfer learning for country specialization.

---

## Highlights

- **Climate encoder**: conditional contractive β‑VAE to compress high‑dimensional, collinear meteorology.
- **Forecaster**: TCN + LSTM + Multi‑Head Self‑Attention.
- **Adversarial training**: WGAN‑GP critic with feature‑matching; heteroscedastic NLL + quantile + TV + L1 + adversarial losses.
- **Transfer‑learning‑ready**: global model + per‑country fine‑tuning.
- **Ablations**: configurable switches for adversarial loss, spectral loss, PLS features, etc.
- **Evaluation**: SMAPE, RMSE, R², 90% coverage; per‑country tables + plots.
- **Visualizations**: attention weights, lag attributions, SHAP‑style drivers (stubs), diagnostics.
- **Reproducible**: deterministic seeds, config‑driven, CLI entrypoints, Docker.
- **Placeholders**: CSV path placeholders are required; replace with your own file paths (see `conf/`).

## Repository Layout

```
spectra-dx-enterprise/
├─ spectra_dx/                # Library code
│  ├─ configs/                # YAML configs (global, ablation, finetune)
│  ├─ dataio/                 # Data loading & featurization
│  ├─ models/                 # β‑VAE encoder, Generator, Critic
│  ├─ training/               # Train loops (global), finetune (country)
│  ├─ evaluation/             # Metrics, reports, visualizations
│  ├─ ablation/               # Ablation switches & runner
│  ├─ utils/                  # Logging, seeds, misc
│  └─ cli.py                  # Console entrypoints
├─ scripts/                   # Example run scripts
├─ tests/                     # Unit tests (smoke)
├─ docker/                    # Containerization
└─ README.md
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Quickstart

1. **Configure** data paths (placeholders) in `spectra_dx/configs/global.yaml`:
   ```yaml
   data:
     climate_csv: "<CLIMATE_CSV_PATH>"
     dengue_csv: "<DENGUE_CSV_PATH>"
     out_dir: "artifacts"
   ```
2. **Train global model**:
   ```bash
   spectra-train-global --config spectra_dx/configs/global.yaml
   ```
3. **Evaluate**:
   ```bash
   spectra-eval --config spectra_dx/configs/global.yaml
   ```
4. **Fine‑tune for a country**:
   ```bash
   spectra-finetune-country --config spectra_dx/configs/finetune_bangladesh.yaml --country "Bangladesh"
   ```
5. **Run ablations**:
   ```bash
   spectra-ablate --config spectra_dx/configs/ablation.yaml
   ```

## Data & Features (Summary)

- Meteorology from ERA5 monthly means (single‑level) over country grids, flattened to tabular after GRIB decode.
- Epidemiology preprocessed to monthly dengue incidence (cases per 100,000), log‑scaled and standardized per country.
- Conditional embeddings: country, year; calendar features: month sin/cos; optional PLS on latent moments.

## Model (Summary)

- **Encoder**: conditional contractive β‑VAE; outputs meteorological latent.
- **Generator**: Causal TCN → LSTM → MHSA → μ/σ² quantiles (0.1/0.5/0.9).
- **Critic**: sequence discriminator (WGAN‑GP) with feature‑matching head.
- **Losses**: heteroscedastic NLL, pinball/quantile, total‑variation + L1 (smoothness), adversarial; optional spectral loss.
- **Optimization**: TTUR, gradient clipping, early stopping; teacher‑forcing anneal.

## Transfer Learning

- Start from global encoder+generator weights, fine‑tune on target country with diagnostics‑aware schedule.
- Keep data standardization country‑specific for targets.

## Reproducibility

- Fixed seeds, versioned configs, artifact logging to `artifacts/`.

## Docker

```bash
docker build -t spectra-dx -f docker/Dockerfile .
docker run --rm -v $PWD:/work -w /work spectra-dx spectra-train-global --config spectra_dx/configs/global.yaml
```

## Disclaimer

This repo provides research software for modeling and forecasting. It is **not** a medical device and should be used with expert oversight.
