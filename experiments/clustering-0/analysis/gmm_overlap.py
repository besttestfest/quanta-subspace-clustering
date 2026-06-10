"""GMM soft-clustering overlap diagnostics.

Tests whether quanta overlap by fitting a Gaussian Mixture Model (k=400) on
the PCA-reduced gradient similarity matrix and inspecting the per-token
posterior over clusters. If tokens concentrate their probability mass on a
single component, hard clustering (as used in the main pipeline) is
empirically justified and the Quantization Model's discreteness assumption
holds for this data.

Reported diagnostics per token:
  - top-1 posterior probability (and the >0.90 "exclusive" fraction)
  - effective number of quanta = exp(entropy of the posterior)
  - top-1/top-2 probability ratio
  - hard-partition (argmax) envelope slope, for reference

Usage (from experiments/clustering-0):
  QDG_MODEL=pythia-19m  python -u analysis/gmm_overlap.py
  QDG_MODEL=pythia-125m python -u analysis/gmm_overlap.py

Output:
  <results_dir>/clusters_gmm_overlap.pkl
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import time
from collections import Counter

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from config import PATHS, MODEL_NAME

K          = int(os.environ.get("GMM_K", 400))
PCA_DIM    = int(os.environ.get("GMM_PCA_DIM", 50))
COV_TYPE   = os.environ.get("GMM_COV", "diag")
MAX_ITER   = int(os.environ.get("GMM_MAX_ITER", 200))
SEED       = int(os.environ.get("GMM_SEED", 42))

OUT_PATH = os.path.join(PATHS["results_dir"], "clusters_gmm_overlap.pkl")

print(f"Loading similarity matrix from: {PATHS['similarity_matrix']}")
_data = torch.load(PATHS["similarity_matrix"], map_location="cpu", weights_only=False)
C_raw = np.array(_data[1], dtype=np.float32)
del _data
# Angular affinity, same transform as the clustering pipeline
C = 1.0 - np.arccos(np.clip(C_raw, -1.0, 1.0)) / np.pi
del C_raw
print(f"C: {C.shape}, range [{C.min():.3f}, {C.max():.3f}]")

print(f"\nPCA to d={PCA_DIM}")
pca = PCA(n_components=PCA_DIM, random_state=SEED)
Y = pca.fit_transform(C)
print(f"  Explained variance: {pca.explained_variance_ratio_.sum()*100:.1f}%")

print(f"\nGMM: k={K}, cov={COV_TYPE}, max_iter={MAX_ITER}")
print("(this typically takes 2-15 min)")
t0 = time.time()
gmm = GaussianMixture(
    n_components=K,
    covariance_type=COV_TYPE,
    max_iter=MAX_ITER,
    random_state=SEED,
)
gmm.fit(Y)
W = gmm.predict_proba(Y).astype(np.float32)
elapsed = time.time() - t0
print(f"  Done in {elapsed:.0f}s ({elapsed/60:.1f} min)")
print(f"  Converged: {gmm.converged_}, iterations: {gmm.n_iter_}")
print(f"  Log-likelihood: {gmm.score(Y) * len(Y):.1f}")

print("\n=== Overlap diagnostics ===")

top1 = W.max(axis=1)
top2 = np.sort(W, axis=1)[:, -2]
with np.errstate(divide="ignore", invalid="ignore"):
    ent = -np.where(W > 0, W * np.log(W), 0.0).sum(axis=1)
eff_k = np.exp(ent)

pcts = [5, 25, 50, 75, 95]
print("\nTop-1 probability per token:")
print(f"  mean = {top1.mean():.3f}, median = {np.median(top1):.3f}")
print(f"  percentiles 5/25/50/75/95: {np.round(np.percentile(top1, pcts), 3)}")

print("\nEffective number of quanta per token (exp entropy):")
print(f"  mean = {eff_k.mean():.2f}, median = {np.median(eff_k):.2f}")
print(f"  percentiles 5/25/50/75/95: {np.round(np.percentile(eff_k, pcts), 2)}")

n = len(top1)
cats = [
    ("Exclusively in 1 quantum (>90%)",                 (top1 > 0.90).sum()),
    ("Clearly in 1 quantum (50-90%)",                   ((top1 > 0.50) & (top1 <= 0.90)).sum()),
    ("Mostly 1 quantum, some uncertainty (30-50%)",     ((top1 > 0.30) & (top1 <= 0.50)).sum()),
    ("Overlap in 2-3 quanta (15-30%)",                  ((top1 > 0.15) & (top1 <= 0.30)).sum()),
    ("Spread across many (<15%)",                       (top1 <= 0.15).sum()),
]
print("\nDiscrete overlap categories (by top-1 probability):")
for name, cnt in cats:
    print(f"  {name}: {cnt:>6d} ({100*cnt/n:.1f}%)")

ratio = top1 / np.maximum(top2, 1e-30)
print("\nTop-1 / Top-2 ratio:")
print(f"  median = {np.median(ratio):.2f} (>>1 = clean assignment, ~1 = ambiguous)")
print(f"  fraction with ratio < 2:  {100*(ratio < 2).mean():.1f}% (ambiguous)")
print(f"  fraction with ratio > 10: {100*(ratio > 10).mean():.1f}% (clean)")

hard_labels = W.argmax(axis=1)
n_active = len(np.unique(hard_labels))
sizes = np.array(sorted(Counter(hard_labels.tolist()).values(), reverse=True))
ranks = np.arange(1, len(sizes) + 1)
mask = (ranks >= 10) & (ranks <= 1000) & (sizes > 0)
slope = None
if mask.sum() >= 3:
    slope, _ = np.polyfit(np.log10(ranks[mask].astype(float)),
                          np.log10(sizes[mask].astype(float)), 1)
print("\nHard partition (argmax of W):")
print(f"  Active clusters: {n_active}/{K}")
print(f"  Sizes: max={sizes.max()}, min={sizes.min()}, median={int(np.median(sizes))}")
if slope is not None:
    print(f"  Envelope slope (rank 10-1000): {slope:.4f} (paper -1.237, |Δ|={abs(slope + 1.237):.4f})")

out = {
    "method": "GMM (soft membership)",
    "params": {"k": K, "pca_dim": PCA_DIM, "covariance_type": COV_TYPE,
               "max_iter": MAX_ITER, "seed": SEED},
    "W": W,
    "hard_labels": hard_labels.astype(np.int32),
    "n_active": int(n_active),
    "envelope_slope": float(slope) if slope is not None else None,
    "diagnostics": {
        "top1_mean": float(top1.mean()),
        "top1_median": float(np.median(top1)),
        "top1_percentiles": {p: float(np.percentile(top1, p)) for p in pcts},
        "effective_k_mean": float(eff_k.mean()),
        "effective_k_median": float(np.median(eff_k)),
        "ratio_top1_top2_median": float(np.median(ratio)),
        "frac_ambiguous": float((ratio < 2).mean()),
        "frac_clean": float((ratio > 10).mean()),
        "n_exclusive_gt90": int(cats[0][1]),
        "n_clear_50_90": int(cats[1][1]),
        "n_mostly_30_50": int(cats[2][1]),
        "n_overlap_15_30": int(cats[3][1]),
        "n_spread_lt15": int(cats[4][1]),
    },
    "n_iter": int(gmm.n_iter_),
    "converged": bool(gmm.converged_),
    "log_likelihood": float(gmm.score(Y) * len(Y)),
    "compute_time_seconds": round(elapsed, 1),
}
with open(OUT_PATH, "wb") as f:
    pickle.dump(out, f)
print(f"\nSaved: {OUT_PATH}")
