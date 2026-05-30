#!/bin/bash
# QDG Master Script: Full end-to-end reproduction from scratch.
#
# Runs the complete pipeline for both Pythia models and produces all
# publication figures. Expects a GPU with ≥24 GB VRAM and ≥32 GB RAM.
#
# Usage:
#   export QDG_ENV=local       # or QDG_ENV=ucloud
#   bash run_all.sh
#
# Total wall-time: ~6-8 days on a single V100 (dominated by the similarity
# matrix computation: ~4 h for 19M and ~35 h for 125M).
#
# See REPRODUCE.md for hardware requirements and expected results.

set -e

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "========================================"
echo "QDG Full Reproduction (started $(date))"
echo "  Repo: $REPO_DIR"
echo "  QDG_ENV: ${QDG_ENV:-ucloud}"
echo "========================================"

# Step 0: Data setup (~1.5 h, CPU + GPU)
echo ""
echo "SETUP: Downloading Pile and computing token indices - $(date)"
cd "$REPO_DIR"

echo "  [0a] Downloading Pile test set..."
# QDG_NDOCS lets a smoke run download fewer docs (e.g. QDG_NDOCS=20000); unset = full 200000
python setup/download_pile.py ${QDG_NDOCS:+--n-docs $QDG_NDOCS}

echo "  [0b] Computing per-token losses (GPU)..."
python setup/compute_losses.py

echo "  [0c] Selecting zero-loss token indices..."
python setup/select_tokens.py

# Per-model pipeline
cd "$REPO_DIR/experiments/clustering-0"

echo ""
echo "========================================"
echo "MODEL: pythia-19m (started $(date))"
echo "  (~3-4 days)"
echo "========================================"
bash run_full_pipeline.sh pythia-19m

echo ""
echo "========================================"
echo "MODEL: pythia-125m (started $(date))"
echo "  (~3-4 days)"
echo "========================================"
bash run_full_pipeline.sh pythia-125m

# Cross-model figures
echo ""
echo "========================================"
echo "CROSS-MODEL: Stability figures - $(date)"
echo "========================================"
bash run_cross_model.sh

echo ""
echo "========================================"
echo "ALL DONE (finished $(date))"
echo "  Figures: $REPO_DIR/figures/"
echo "  See REPRODUCE.md for expected results."
echo "========================================"
