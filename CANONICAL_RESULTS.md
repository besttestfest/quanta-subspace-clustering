# Canonical Results Reference

Numerical results reported in the thesis, with source file references.

---

## 1. Envelope Slopes (Contribution 1 & 2)

Source: `results-mirror/unified_comparison.json`

| Method | Pythia-19M slope | \|Δ\| | Pythia-125M slope | \|Δ\| |
|--------|-----------------|-------|-------------------|-------|
| SSC-Lasso | -1.2451 | 0.008 | -1.3287 | 0.092 |
| Spectral | -1.1944 | 0.043 | -1.2412 | 0.004 |
| Hierarchical | -1.2150 | 0.022 | -1.2486 | 0.012 |
| SSC-OMP | -1.4487 | 0.212 | -1.3034 | 0.066 |

Paper target: β* = -1.237

---

## 2. Fingerprint Classification (Contribution 3, Table 2)

Balanced accuracy from `pipeline/05_fingerprint.py` (single-seed 5-fold CV).
MCC from `pipeline/07_fingerprint_robustness.py` (5 seeds × 5 folds); source files in `results-mirror/fingerprint_robustness/`.

### SSC-Lasso
| Model | Bal. acc | MCC |
|-------|----------|-----|
| Pythia-19M | **0.945** | **+0.884** |
| Pythia-125M | **0.984** | **+0.960** |

### Spectral
| Model | Bal. acc | MCC |
|-------|----------|-----|
| Pythia-19M | 0.901 | +0.777 |
| Pythia-125M | 0.938 | +0.860 |

### Hierarchical complete
| Model | Bal. acc | MCC |
|-------|----------|-----|
| Pythia-19M | 0.830 | +0.675 |
| Pythia-125M | 0.873 | +0.740 |

### SSC-Lasso improvement over Spectral
- Pythia-19M: +0.884 - 0.777 = **+0.107 MCC**
- Pythia-125M: +0.960 - 0.860 = **+0.100 MCC**

---

## 3. Bootstrap Stability (Contribution 2, Figure 11)

Source: `results-mirror/bootstrap_stability/` (n=50 iterations per method/model)

| Method | Model | NMI mean±std | NMI median | ARI mean±std | ARI median |
|--------|-------|-------------|-----------|-------------|-----------|
| Hierarchical | Pythia-19M  | 0.772±0.002 | 0.772 | 0.440±0.022 | 0.438 |
| Hierarchical | Pythia-125M | 0.752±0.002 | 0.752 | 0.357±0.012 | 0.357 |
| SSC-Lasso | Pythia-19M  | 0.731±0.020 | 0.732 | 0.381±0.094 | 0.367 |
| SSC-Lasso | Pythia-125M | 0.685±0.014 | 0.687 | 0.329±0.042 | 0.330 |
| Spectral | Pythia-19M  | 0.687±0.005 | 0.688 | 0.134±0.009 | 0.132 |
| Spectral | Pythia-125M | 0.670±0.006 | 0.670 | 0.124±0.007 | 0.125 |

ARI ranges cited in thesis:
- Hierarchical: 0.36-0.44
- SSC-Lasso: 0.33-0.38
- Spectral: 0.12-0.13

---

## 4. Taxonomy (Contribution 3, Figures 12-13)

Source: `results-mirror/taxonomy/pythia-19m_ssc_lasso.json`

Cross-model Pearson r = **0.997** (token-share correlation, SSC-Lasso k=400)

Top categories at k=400 (token share):

| Category | Pythia-19M | Pythia-125M |
|----------|-----------|------------|
| CONTENT | 77.3% | 88.9% |
| PROPER_NOUNS | 6.7% | 0.6% |
| SYNTACTIC | 6.6% | 2.8% |
| FORMATTING | 4.7% | 4.8% |
| FUNCTION_WORDS | 2.4% | 0.8% |
| NUMERIC | 2.4% | 2.2% |

---

## 5. Cross-Model Alignment

Source: `results-mirror/crossmodel_alignment/results.json`

| Method | Cross-model ARI (19m vs 125m, k=400) |
|--------|--------------------------------------|
| Spectral | 0.058 |
| Hierarchical | 0.045 |
| SSC-Lasso | 0.017 |
| SSC-OMP | 0.007 |

Spectral is 3.5× more cross-model stable than SSC-Lasso (0.058 / 0.017).

---

## 6. Figure Files

All figures committed to git under `figures/`:

| Figure | Path |
|--------|------|
| Fig 1 | `contribution-1/fig_01_rank_frequency.pdf` |
| Fig 2 | `contribution-1/fig_02_similarity_structure.pdf` |
| Fig 3 | `contribution-3/fig_03_fingerprint.pdf` |
| Fig 4 | `contribution-3/fig_04_taxonomy_comparison.pdf` |
| Fig 5 | `contribution-2/appendix/fig_05_method_overlay_19m.pdf` |
| Fig 6 | `contribution-2/appendix/fig_06_method_overlay_125m.pdf` |
| Fig 7 | `contribution-2/appendix/fig_07_ssc_evaluation_19m.pdf` |
| Fig 8 | `contribution-2/appendix/fig_08_method_comparison_19m.pdf` |
| Fig 9 | `contribution-2/appendix/fig_09_method_comparison_125m.pdf` |
| Fig 10 | `contribution-2/appendix/fig_10_ssc_evaluation_125m.pdf` |
| Fig 11 | `contribution-2/appendix/fig_11_bootstrap_stability.pdf` |
| Fig 12 | `contribution-3/appendix/fig_12_taxonomy_and_recovery.pdf` |
| Fig 13 | `contribution-3/appendix/fig_13_taxonomy_bootstrap_ci.pdf` |
