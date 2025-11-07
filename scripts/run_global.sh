#!/usr/bin/env bash
set -euo pipefail
spectra-train-global --config spectra_dx/configs/global.yaml
spectra-eval --config spectra_dx/configs/global.yaml
