# How to Reproduce: QDG Pipeline with SSC-Lasso Clustering

This guide reproduces the full thesis experiment from scratch. The pipeline implements:

1. The original Michaud et al. (2023) QDG replication on Pythia-19M and Pythia-125M,
2. A systematic clustering comparison (Spectral, SSC-Lasso, SSC-OMP, Hierarchical),
3. Bootstrap stability analysis of the recommended SSC-Lasso configuration,
4. The quanta-fingerprint AI-vs-human classifier.

## Hardware Requirements

- **GPU**: NVIDIA GPU with ≥24 GB VRAM (V100 32 GB tested; A100 works faster).
  The similarity matrix computation (`pipeline/01_similarity_matrix.py`) is the
  bottleneck: ~4 h on V100 for Pythia-19M and ~35 h for Pythia-125M.
- **CPU/RAM**: ≥16 GB RAM for clustering. 32 GB recommended.
- **Disk**: ≥50 GB free space for similarity matrices, model weights, and results.

The heavy GPU steps (01, 05) can be run on any CUDA-capable machine.
All other steps are CPU-only and can run on a laptop.

## Environment Setup

### 1. Clone the repository

```bash
git clone https://github.com/besttestfest/quanta-subspace-clustering.git
cd quanta-subspace-clustering
```

### 2. Set your base directory

The pipeline expects data, models, and results under a single `BASE_DIR`.
Edit `experiments/clustering-0/config.py` to add your path, or use the
built-in `local` environment which maps to `~/Master`:

```bash
export QDG_ENV=local          # BASE_DIR = ~/Master  (default for local machines)
```

### 3. Install dependencies

```bash
pip install -r requirements.txt --break-system-packages

# For a specific CUDA build of PyTorch (optional - pip auto-selects if omitted):
# pip install torch==2.2.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
```

Verify paths resolve correctly:
```bash
cd experiments/clustering-0
python config.py
```

---

## Quick Start (Automated)

```bash
cd experiments/clustering-0

# Run full pipeline for each model (~3-4 days per model on a single V100):
bash run_full_pipeline.sh pythia-19m
bash run_full_pipeline.sh pythia-125m

# Cross-model figures - run once after BOTH models complete (~5 min):
bash run_cross_model.sh
```

---

## Step-by-Step Reproduction

### Step 0: Download data (~30 min, CPU/network)

Run from the **repo root** (`quanta-subspace-clustering/`):

```bash
python setup/download_pile.py
python setup/compute_losses.py
python setup/select_tokens.py
```

Output: `$BASE_DIR/data/the_pile_test_canonical_200k/`
First document should start with: "Roman Catholic Diocese of Tambacounda".
Token selection output: `$BASE_DIR/quanta-subspace-clustering/tmp/zero_and_induction_idxs.pkl`
Expected: ~8.95M non-induction zeros, 10,000 tokens selected (every 50th).

---

All remaining steps are run from `experiments/clustering-0/`.

### Step 1: Gradient similarity matrix (~4 h for 19M / ~35 h for 125M, GPU)

```bash
export QDG_MODEL=pythia-19m       # or pythia-125m
python -u pipeline/01_similarity_matrix.py
```

Output: `results/clustering-0/full_more.pt` (~800 MB, 10k × 10k cosine similarity matrix)

Verification:
```python
import torch
idxs, C, C_abs = torch.load('results/clustering-0/full_more.pt',
                              map_location='cpu', weights_only=False)
assert C.shape == (10000, 10000)
assert abs(C.diagonal().mean().item() - 1.0) < 1e-5
```

### Step 2: Clustering - all 4 methods (~4-6 h, CPU)

```bash
python -u pipeline/02_clustering.py
```

Runs Spectral, SSC-Lasso, SSC-OMP, and Hierarchical clustering in sequence.
All use `random_state=42` for reproducibility.

Outputs under `results/clustering-0/`:
- `clusters_full_more.pkl` (Spectral)
- `clusters_ssc_full_more.pkl` (SSC-Lasso)
- `clusters_ssc_full_more_omp.pkl` (SSC-OMP)
- `clusters_hierarchical.pkl` (Hierarchical)

### Step 3: Envelope and power-law analysis (~30 min, CPU)

```bash
python -u pipeline/03_envelope_analysis.py
```

Computes apples-to-apples envelope slopes for all methods using the
per-method k-sweep described in Appendix D of the thesis.

### Step 4: Quanta fingerprint (~30-60 min per method, GPU)

```bash
python -u pipeline/05_fingerprint.py --cluster-method spectral
python -u pipeline/05_fingerprint.py --cluster-method ssc-lasso
python -u pipeline/05_fingerprint.py --cluster-method hierarchical
```

Runs the AI-vs-human logistic regression classifier on the three paradigms
used in the thesis tables (Spectral, SSC-Lasso, Hierarchical). SSC-OMP is
excluded as it does not appear in any thesis result table, saving ~22 h.
AI documents are generated once by the first invocation and shared across
all subsequent methods (`shared_ai_docs.pkl`).

### Step 5: Taxonomy and feature selection (~10 min, CPU)

```bash
python -u pipeline/04_taxonomy_categories.py
```

Outputs: `$RESULTS_DIR/taxonomy_analysis.json` and `$RESULTS_DIR/l1_selection_results.json`.

### Step 6: Token taxonomy (~5 min, CPU)

```bash
python -u pipeline/09_quanta_taxonomy.py
```

Assigns each cluster to a linguistic category (CONTENT, PROPER_NOUNS, SYNTACTIC, etc.) for all three paradigms. Run `pipeline/10_taxonomy_mirror.py` once both models are complete (see Step 14).

### Step 7: Bootstrap stability (50 iterations, ~25 h, CPU)

SSC-Lasso (main method):

```bash
BOOT_N_JOBS=20 BOOT_N_ITER=50 python -u pipeline/06_bootstrap_stability.py
```

Output: `results/clustering-0/bootstrap_stability_ssc_lasso.json`
Results are checkpointed after every iteration - safe to interrupt and resume.

Spectral and Hierarchical (supplementary, ~4-8 h CPU each):

```bash
python -u pipeline/06b_bootstrap_spectral_hierarchical.py --method spectral
python -u pipeline/06b_bootstrap_spectral_hierarchical.py --method hierarchical
```

Output: `results/clustering-0/bootstrap_stability_{spectral,hierarchical}.json`

### Step 8: Fingerprint robustness (~2-3 h, CPU)

Run once per paradigm:

```bash
python -u pipeline/07_fingerprint_robustness.py \
    --fp-subdir quanta_fingerprint_ssc-lasso --paradigm SSC-Lasso
python -u pipeline/07_fingerprint_robustness.py \
    --fp-subdir quanta_fingerprint --paradigm Spectral
python -u pipeline/07_fingerprint_robustness.py \
    --fp-subdir quanta_fingerprint_hierarchical --paradigm Hierarchical
```

Recomputes Table 5 and Appendix C numbers across all paradigms.

### Step 9: Bag-of-words baseline (~10 min, CPU)

```bash
python -u pipeline/08_bow_baseline.py
```

### Step 10: Taxonomy mirror + all publication figures (~20 min, CPU)

Every publication figure needs **both** models, so they are generated **once, after both models are complete**, by `run_cross_model.sh`:

```bash
bash run_cross_model.sh
```

This runs `pipeline/10_taxonomy_mirror.py` (writes `results-mirror/taxonomy/` for both models) and then all figure scripts:
`fig_01_02_overview.py` (Figs 1-2), `fig_03_fingerprint.py` (Fig 3), `fig_04_taxonomy.py` (Fig 4), `fig_05_10_methods.py` (Figs 5-10), `fig_12_13_taxonomy_detail.py` (Figs 12-13), and `fig_11_stability.py` (Fig 11). Figures are written under `figures/contribution-{1,2,3}/`.

---

## Directory Structure

After both models complete, the expected layout is:

```
$BASE_DIR/
  quanta-subspace-clustering/              # This repo
    figures/
      contribution-1/              # Figures 1-2
      contribution-2/
        appendix/                  # Figures 5-11
      contribution-3/              # Figures 3-4
        appendix/                  # Figures 12-13
    results-mirror/                # Tracked reference results (git-committed)
      bootstrap_stability/         # NMI/ARI JSON per model × method
      fingerprint_robustness/      # MCC JSON per model × paradigm
      bow_baseline/                # BoW baseline JSON per model
  data/
    the_pile_test_canonical_200k/  # Tokenized Pile test set
  results/
    clustering-0/                  # Pythia-19M results
      full_more.pt                 # 10k × 10k similarity matrix (~800 MB)
      clusters_full_more.pkl
      clusters_ssc_full_more.pkl
      clusters_ssc_full_more_omp.pkl
      clusters_hierarchical.pkl
      bootstrap_stability_ssc_lasso.json
      quanta_taxonomy_spectral_k400.pkl
      quanta_taxonomy_ssc_lasso_k400.pkl
      quanta_taxonomy_hierarchical_k400.pkl
      figures/
        quanta_fingerprint/        # Spectral fingerprint results
        quanta_fingerprint_ssc-lasso/
        quanta_fingerprint_hierarchical/
    clustering-0-pythia-125m/      # Pythia-125M results (same structure)
    crossmodel_alignment/
      results.json                 # Cross-model ARI/NMI per method
  models/                          # Cached Pythia weights (auto-downloaded)
```

---

## Expected Results

### Envelope slopes (paper target -1.237)

**Pythia-19M**

| Method       |  Slope | \|Δ\| |
|:-------------|-------:|------:|
| SSC-Lasso    | -1.245 | 0.008 |
| Hierarchical | -1.215 | 0.022 |
| Spectral     | -1.194 | 0.043 |
| SSC-OMP      | -1.449 | 0.212 |

**Pythia-125M**

| Method       |  Slope | \|Δ\| |
|:-------------|-------:|------:|
| Spectral     | -1.241 | 0.004 |
| Hierarchical | -1.249 | 0.012 |
| SSC-OMP      | -1.303 | 0.066 |
| SSC-Lasso    | -1.329 | 0.092 |

### Bootstrap stability (50 iterations, 80% subsampling)

**SSC-Lasso** (within-model token subsampling, k=400):

| Model       | NMI (mean ± std) | ARI (mean ± std) |
|:------------|:----------------:|:----------------:|
| Pythia-19M  | 0.731 ± 0.020    | 0.381 ± 0.094    |
| Pythia-125M | 0.685 ± 0.014    | 0.329 ± 0.042    |

**Spectral** (low ARI reflects eigenvector rotation under perturbation, not structural instability):

| Model       | NMI (mean ± std) | ARI (mean ± std) |
|:------------|:----------------:|:----------------:|
| Pythia-19M  | 0.687 ± 0.005    | 0.134 ± 0.009    |
| Pythia-125M | 0.670 ± 0.006    | 0.124 ± 0.007    |

**Hierarchical** (most stable within-model):

| Model       | NMI (mean ± std) | ARI (mean ± std) |
|:------------|:----------------:|:----------------:|
| Pythia-19M  | 0.772 ± 0.002    | 0.440 ± 0.022    |
| Pythia-125M | 0.752 ± 0.002    | 0.357 ± 0.012    |

### Cross-model cluster agreement (19M vs 125M, k=400)

| Method       |   ARI |   NMI |
|:-------------|------:|------:|
| Spectral     | 0.058 | 0.547 |
| Hierarchical | 0.045 | 0.483 |
| SSC-Lasso    | 0.017 | 0.313 |
| SSC-OMP      | 0.007 | 0.333 |

### Quanta-fingerprint classifier robustness (all methods, k=400, shared AI docs)

All three methods use documents from `shared_ai_docs.pkl`. With the default
`N_AI_DOCS=600`, approximately 498 documents have zero-loss tokens and are used
for classification. MCC values are from `pipeline/07_fingerprint_robustness.py`
(5 seeds × 5-fold CV) and are archived in `results-mirror/fingerprint_robustness/`.

| Model       | Clustering   | MCC (full, 5-seed CV)  |
|:------------|:-------------|:----------------------:|
| Pythia-19M  | SSC-Lasso    | +0.884 ± 0.002         |
| Pythia-19M  | Spectral     | +0.777 ± 0.005         |
| Pythia-19M  | Hierarchical | +0.675 ± 0.006         |
| Pythia-125M | SSC-Lasso    | +0.960 ± 0.001         |
| Pythia-125M | Spectral     | +0.860 ± 0.001         |
| Pythia-125M | Hierarchical | +0.740 ± 0.005         |

---

## Pipeline Timing (Single V100 32 GB, per model)

| Step | Description | Time | Hardware |
|:-----|:------------|-----:|:--------:|
| setup | Download Pile, compute losses, select tokens | ~1.5 h | CPU+GPU |
| 01 | Gradient similarity matrix | ~4 h (19M) / ~35 h (125M) | GPU |
| 02 | Clustering (4 methods) | ~4-6 h | CPU |
| 03-05 | Envelope, taxonomy, fingerprint | ~2-4 h | CPU+GPU |
| 06 | Token taxonomy | ~5 min | CPU |
| 07-09 | Bootstrap, robustness, BoW baseline | ~25 h | CPU |
| 10-13 | Figures | ~20 min | CPU |
| **Total per model** | | **~3-4 days** | |

For both models + cross-model: **~6-8 days** total.

---

## Software Versions

```
Python       3.12
PyTorch      ≥2.0  (2.2.0+cu121 tested)
transformers 4.38.2
scikit-learn 1.4.0
numpy        1.26.4
scipy        1.12.0
datasets     2.18.0
matplotlib   3.9.0
tqdm         4.66.0
joblib       1.3.0
```

---

## Known Differences from the Original Paper Code

1. **Clustering methods**: Evaluates 4 paradigms (Spectral, SSC-Lasso, SSC-OMP, Hierarchical) vs. spectral-only in the original.
2. **Dataset source**: HuggingFace `EleutherAI/pile_val_test` instead of `test.jsonl.zst` (original source expired).
3. **Block length**: `block_len=250` (same as original paper). Larger models (e.g. Pythia-125M) have much larger per-token gradients and may run out of GPU memory in step 01; lower the block size with `export QDG_BLOCK_LEN=25` (no code edit needed).
4. **sklearn ≥1.8**: L1 C-grid auto-tuned for newer liblinear tolerance defaults.
5. **SpectralClustering seed**: `random_state=42`; cluster labels differ between runs but envelope slopes reproduce within ±0.005.
6. **Envelope values in Table 1**: The thesis reports apples-to-apples envelope slopes from an extended (d,α) grid search run separately on UCloud. The values from `run_full_pipeline.sh` (step 3) will be close but may differ slightly as the pipeline uses a reduced grid. The exact thesis values (e.g. SSC-Lasso |Δ|=0.008 on Pythia-19M) required the full L-grid sweep.
7. **Fingerprint MCC variability**: AI text is generated at temperature=1.0 and saved to `shared_ai_docs.pkl` so all methods use identical documents. Results are reproducible across runs of the same pipeline, but will differ from the thesis values (which used a separately generated AI corpus). Expect MCC within ±0.05-0.10 of thesis Table 2 values.
8. **Baseline evaluation**: `pipeline/08_bow_baseline.py` classifies the document selection recorded by `pipeline/05_fingerprint.py` (`classification_docs.json`) and fits TF-IDF inside each training fold; `pipeline/04_taxonomy_categories.py` reports cross-validated L1 selection scores. Fresh runs of these scripts therefore give slightly more conservative numbers than the archived `results-mirror/` values, which were produced with an earlier evaluation protocol.

---

## Troubleshooting

**CUDA out of memory during step 01**
Lower the gradient block size: `export QDG_BLOCK_LEN=25` (default 250), then re-run. This is the common fix for Pythia-125M, whose gradients are much larger than Pythia-19M's. Alternatively use a GPU with more VRAM.

**`ModuleNotFoundError: No module named 'config'`**
Run from `experiments/clustering-0/`: `cd experiments/clustering-0`

**Step 05 crashes with "k=400 not found in pkl"**
Verify step 02 completed. Check: `ls -lh results/clustering-0/clusters_*.pkl`

**Cross-model alignment says "no data"**
By design - `run_cross_model.sh` requires both models to have completed step 03.

**Bootstrap (step 7) is slow**
Set `BOOT_N_JOBS` to the number of available CPU cores. Progress is checkpointed
after every iteration - safe to kill and resume at any time.
