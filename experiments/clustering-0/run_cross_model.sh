#!/bin/bash
# QDG Cross-Model Pipeline
#
# Run this AFTER both models have completed run_full_pipeline.sh:
#   bash run_full_pipeline.sh pythia-19m
#   bash run_full_pipeline.sh pythia-125m
#   bash run_cross_model.sh
#
# Generates the cross-model taxonomy mirror and ALL publication figures.
# Figures live here (not in run_full_pipeline.sh) because every figure needs
# both models; generating them per-model would fail on the first model.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
export QDG_ENV=${QDG_ENV:?QDG_ENV is not set. Export QDG_ENV=ucloud or QDG_ENV=local before running.}
export QDG_MODEL=${QDG_MODEL:-pythia-19m}

echo "QDG Cross-Model Pipeline (started $(date))"

echo ""
echo "Step 1/3: Token taxonomy mirror (results-mirror/taxonomy/) - $(date)"
python -u pipeline/10_taxonomy_mirror.py

echo ""
echo "Step 2/3: Publication figures - overview, fingerprint, taxonomy, methods - $(date)"
python -u figures/fig_01_02_overview.py
python -u figures/fig_03_fingerprint.py
python -u figures/fig_04_taxonomy.py
python -u figures/fig_05_10_methods.py
python -u figures/fig_12_13_taxonomy_detail.py

echo ""
echo "Step 3/3: Bootstrap stability figure (fig_11) - $(date)"
python -u figures/fig_11_stability.py

echo ""
echo "Cross-model pipeline complete (finished $(date))"
python -c "
import os, sys
sys.path.insert(0, '.')
os.environ['QDG_MODEL'] = 'pythia-19m'
from config import PATHS
print(f'  Figures: {PATHS[\"repo_dir\"]}/figures/')
"
