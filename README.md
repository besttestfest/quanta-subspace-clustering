# Subspace Clustering and the Representational Utility of Quanta Fingerprints

Anonymous code release for peer review.

Code for the paper of the same name. A recent theory of neural scaling proposes that a language model's capabilities decompose into discrete chunks known as *quanta*, discoverable by clustering per-token gradients. Existing work identifies quanta with a single clustering algorithm and judges them only by whether their size distribution follows the predicted power law. We argue that a clustering method should be evaluated on two criteria: its **envelope fit** (how well it recovers the predicted size distribution) and its **representational utility** (how useful the resulting clusters are as document-level features).

This repository contains the code for all three contributions of the paper:

1. A reproduction of the quanta-discovery findings of Michaud et al. (2023) on Pythia-19m and Pythia-125m, establishing a reproducible baseline with an envelope methodology.
2. A controlled comparison of four clustering paradigms (Spectral, SSC-Lasso, SSC-OMP, Hierarchical) on envelope fit, and of three of them (Spectral, SSC-Lasso, Hierarchical) on representational utility in a downstream task.
3. The central result: a better power-law fit does not entail higher representational utility. SSC-Lasso fingerprints outperform every evaluated paradigm on AI-versus-human text classification (MCC 0.884 on Pythia-19m, 0.960 on Pythia-125m), even though spectral clustering achieves the better envelope fit on the larger model.

Based on: [The Quantization Model of Neural Scaling](https://arxiv.org/abs/2303.13506) (Michaud et al., 2023).
Forked from: [ejmichaud/quantization-model](https://github.com/ejmichaud/quantization-model).

## Where the results are (no need to run anything)

All paper results are already committed to this repository, so you can inspect them directly:

- **Figures** (exact figures from the paper): [`figures/`](figures/)
- **All table numbers**, with the source file for each: [`CANONICAL_RESULTS.md`](CANONICAL_RESULTS.md)
- **Archived result data** (JSON, per analysis): [`results-mirror/`](results-mirror/)

These committed artifacts match the paper one-to-one. The pipeline below regenerates them, but results are reproducible only **within a documented tolerance, not bit-for-bit**: the pipeline uses random seeds and freshly generated AI text on each run. Expect envelope slopes within ±0.005 and MCC within ±0.05-0.10 of the reported values (see [REPRODUCE.md -> Known Differences](REPRODUCE.md)). This is normal for a stochastic ML pipeline and does not indicate an error.

## Reproducing from scratch (optional, ~6-8 days on one GPU)

You only need this to regenerate the results above; it is not required to view them.
Requires a Linux machine with GPU (tested on a Tesla V100-SXM2-32GB node).

```bash
# clone this repository, then:
cd quanta-subspace-clustering
pip install -r requirements.txt

# Full end-to-end reproduction (both models + all figures):
export QDG_ENV=local   # or ucloud, sets BASE_DIR (see REPRODUCE.md)
bash run_all.sh
```

`run_all.sh` downloads The Pile test set, computes per-token losses, selects the 10,000 tokens used for clustering, then runs the full pipeline for both Pythia-19m and Pythia-125m and generates all paper figures.

Total runtime is roughly 6-8 days on a single V100 (most of it is the similarity-matrix computation in `pipeline/01_similarity_matrix.py`). See [REPRODUCE.md](REPRODUCE.md) for detailed step-by-step instructions and expected results.

## Repository structure

```
setup/                         Data preparation scripts
experiments/clustering-0/
  pipeline/                    Core pipeline (01-10), run in order
  figures/                     Figure-generation scripts (fig_01-fig_12)
  run_full_pipeline.sh         Runs pipeline/ steps end-to-end for one model
  run_cross_model.sh           Cross-model alignment + stability figures (run after both models)
  config.py                    Paths and model configuration
  plot_style.py                Shared matplotlib style
figures/                       Output figures (contribution-1/, -2/, -3/)
run_all.sh                     Master script: full end-to-end reproduction
```

### Pipeline steps (`pipeline/`)

| Script | What it does |
|---|---|
| `01_similarity_matrix.py` | Gradient similarity matrix computation (GPU, ~4 h for 19M / ~35 h for 125M) |
| `02_clustering.py` | Spectral, SSC-Lasso, SSC-OMP, and Hierarchical clustering |
| `03_envelope_analysis.py` | Envelope slopes and power-law validation (apples-to-apples comparison) |
| `04_taxonomy_categories.py` | Quanta taxonomy and L1 feature selection |
| `05_fingerprint.py` | AI-vs-human quanta fingerprint classifier (GPU, ~30-60 min per method) |
| `06_bootstrap_stability.py` | Bootstrap stability analysis (50 iterations) |
| `07_fingerprint_robustness.py` | Fingerprint paradigm-robustness across clustering methods |
| `08_bow_baseline.py` | Bag-of-words baseline classifier |
| `09_quanta_taxonomy.py` | Token taxonomy: assigns each cluster to a linguistic category |
| `10_taxonomy_mirror.py` | Writes taxonomy summaries to `results-mirror/taxonomy/` |

### Figure scripts (`figures/`)

| Script | Figures produced |
|---|---|
| `fig_01_02_overview.py` | Rank-frequency distribution and cross-model structure |
| `fig_03_fingerprint.py` | Fingerprint analysis (divergence, similarity, logistic regression, active quanta) |
| `fig_04_taxonomy.py` | Quanta taxonomy and category analysis |
| `fig_05_10_methods.py` | Method comparison and envelope fits (overlays, hyperparameter heatmaps, per-method fits) |
| `fig_11_stability.py` | Bootstrap stability (NMI/ARI across iterations) |
| `fig_12_13_taxonomy_detail.py` | Taxonomy detail (category distribution and paper-quanta recovery) |

## Key results

### Envelope slopes (apples-to-apples, paper target -1.237)

For each method we sweep its hyperparameters, then for every rank keep the max cluster size across the sweep, and fit a power law on ranks 100-1000 in log-log space. This is the same envelope construction used in the original paper, applied identically to every method.

**Pythia-19m (top methods)**

| Method | Slope | \|Δ\| |
|---|---:|---:|
| **SSC-Lasso** | **-1.245** | **0.008** |
| Hierarchical (complete) | -1.215 | 0.022 |
| Spectral (baseline) | -1.194 | 0.043 |
| SSC-OMP | -1.449 | 0.212 |

**Pythia-125m (top methods)**

| Method | Slope | \|Δ\| |
|---|---:|---:|
| **Spectral (baseline)** | **-1.241** | **0.004** |
| Hierarchical (complete) | -1.249 | 0.012 |
| SSC-OMP | -1.303 | 0.066 |
| SSC-Lasso | -1.329 | 0.092 |

Key findings: SSC-Lasso achieves the best envelope fit on Pythia-19m (|Δ|=0.008); Spectral retains a slight edge on Pythia-125m (|Δ|=0.004). Hierarchical is consistently competitive on both models.

### Quanta fingerprints: AI-vs-human text classification

All three methods use the same `shared_ai_docs.pkl` corpus, making results directly comparable. MCC values below are from `pipeline/07_fingerprint_robustness.py` (5 seeds × 5-fold CV), archived in `results-mirror/fingerprint_robustness/`, and match the paper Table 2 values. Re-running the pipeline from scratch regenerates the AI corpus and may shift MCC by ±0.05-0.10 (see REPRODUCE.md §Known Differences).

| Model | Clustering | MCC (5-seed CV) |
|---|---|---:|
| Pythia-19m | Spectral | +0.777 |
| Pythia-19m | **SSC-Lasso** | **+0.884** |
| Pythia-19m | Hierarchical | +0.675 |
| Pythia-125m | Spectral | +0.860 |
| Pythia-125m | **SSC-Lasso** | **+0.960** |
| Pythia-125m | Hierarchical | +0.740 |

SSC-Lasso fingerprints yield substantially stronger AI/Human discrimination than spectral on both models (~+0.10 MCC), despite spectral winning the envelope fit on Pythia-125m. Envelope fit and downstream utility are dissociable.

### L1 feature selection: signal is distributed, not concentrated

| Model | All 400 (L2) | L1-sparse (best C) | Selected features |
|---|---:|---:|---:|
| Pythia-19m | 0.891 / +0.622 | 0.830 / +0.497 | 5 (C=0.178) |
| Pythia-125m | 0.889 / +0.641 | 0.790 / +0.445 | 13 (C=0.316) |

The signal is distributed across many quanta; L1-sparse models underperform L2. L1 is used as an interpretability tool for category enrichment, not as the main classifier.

## Original paper

Based on [ejmichaud/quantization-model](https://github.com/ejmichaud/quantization-model). See that repository for the original Michaud et al. code and instructions.
