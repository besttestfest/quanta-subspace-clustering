"""Multi-axis robustness analysis of the AI-vs-human fingerprint classifier.

Recomputes Table 5 / Appendix C numbers for the paper. Reads the fingerprint
matrix saved by pipeline/05_fingerprint.py and reports five metrics per
(paradigm, model) combination:

  1. Full MCC      - L2 logreg on all k cluster-usage features (5 seeds × 5-fold CV)
  2. Surface MCC   - L2 logreg on 3 document-level summary features only:
                       (Shannon entropy, # active quanta, peak probability)
  3. Δ              - Full - Surface (signal attributable to per-cluster identity)
  4. Max 1-q MCC   - best single-cluster classifier (one feature only)
  5. Permutation   - label-shuffle baseline (should be ≈ 0)
  6. Seed Δ        - std of per-seed mean MCC across 5 random seeds

Each row is flagged:
  - "distributed"        Δ > 0.05 and best single-q < 0.95 × full
  - "surface-dominated"  Δ ≤ 0.05
  - "one-cluster-led"    best single-q ≥ 0.95 × full

Usage:
  cd experiments/clustering-0

  # Use the canonical hyphen run (recommended for paper):
  QDG_MODEL=pythia-19m  python -u pipeline/07_fingerprint_robustness.py \
      --fp-subdir quanta_fingerprint_ssc-lasso       --paradigm SSC-Lasso
  QDG_MODEL=pythia-19m  python -u pipeline/07_fingerprint_robustness.py \
      --fp-subdir quanta_fingerprint                 --paradigm Spectral
  QDG_MODEL=pythia-19m  python -u pipeline/07_fingerprint_robustness.py \
      --fp-subdir quanta_fingerprint_hierarchical --paradigm Hierarchical
  # Repeat with QDG_MODEL=pythia-125m

Outputs:
  results/<model>/figures/<fp-subdir>/robustness.json     (per run)
  results-mirror/fingerprint_robustness/<model>_<paradigm>.json (final summary)

The script REQUIRES fingerprint_matrix.npz to exist in the chosen subdir.
Run pipeline/05_fingerprint.py first if it does not.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import StratifiedKFold

from config import PATHS, MODEL_NAME

parser = argparse.ArgumentParser()
parser.add_argument("--fp-subdir", required=True,
                    help="Sub-directory under <results_dir>/figures/ that "
                         "contains fingerprint_matrix.npz "
                         "(e.g. 'quanta_fingerprint_ssc-lasso').")
parser.add_argument("--paradigm", required=True,
                    help="Human-readable paradigm name for the report "
                         "(SSC-Lasso, Spectral, Hierarchical).")
parser.add_argument("--n-seeds", type=int, default=5,
                    help="Number of random seeds for cross-validation.")
parser.add_argument("--n-folds", type=int, default=5,
                    help="Stratified K-fold cross-validation splits.")
parser.add_argument("--surface-dominated-threshold", type=float, default=0.05,
                    help="Flag as surface-dominated if Δ ≤ this value.")
parser.add_argument("--one-cluster-led-threshold", type=float, default=0.95,
                    help="Flag as one-cluster-led if max 1-q MCC ≥ "
                         "(this fraction) × full MCC.")
args = parser.parse_args()


fp_dir   = os.path.join(PATHS["results_dir"], "figures", args.fp_subdir)
fp_path  = os.path.join(fp_dir, "fingerprint_matrix.npz")
if not os.path.exists(fp_path):
    sys.exit(f"ERROR: {fp_path} not found.\n"
             f"Run pipeline/05_fingerprint.py first with the matching cluster_method.")

print(f"Fingerprint robustness: {MODEL_NAME} / {args.paradigm}")
print(f"  matrix: {fp_path}")

d = np.load(fp_path)
X        = d["X"]
y        = d["y"].astype(np.int64)
n_human  = int(d["n_human"])
n_ai     = int(d["n_ai"])
n_clust  = int(d["n_clusters"])
n_docs, k = X.shape
assert k == n_clust, f"X has {k} cluster cols but n_clusters = {n_clust}"

print(f"  shape: {n_docs} docs × {n_clust} clusters  "
      f"(human={n_human}, ai={n_ai})")
print(f"  evaluation: {args.n_seeds} seeds × {args.n_folds}-fold "
      f"= {args.n_seeds * args.n_folds} evaluations\n")


def cv_mcc(X_in, y_in, n_seeds, n_folds, C=1.0):
    """L2 logistic regression, n_seeds × n_folds CV. Returns mean MCC over all
    folds across all seeds, plus per-seed mean MCCs (for std)."""
    seed_means = []
    for seed in range(n_seeds):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        fold_mccs = []
        for tr, te in skf.split(X_in, y_in):
            clf = LogisticRegression(
                penalty="l2", C=C, solver="lbfgs",
                max_iter=2000, class_weight="balanced",
            )
            clf.fit(X_in[tr], y_in[tr])
            pred = clf.predict(X_in[te])
            fold_mccs.append(matthews_corrcoef(y_in[te], pred))
        seed_means.append(np.mean(fold_mccs))
    seed_means = np.array(seed_means)
    return float(seed_means.mean()), float(seed_means.std()), seed_means.tolist()


def surface_features(X_fp):
    """Three document-level summary statistics: Shannon entropy of fingerprint
    distribution, number of active quanta, peak probability."""
    eps = 1e-12
    # Renormalise rows to probabilities (X may already be normalised)
    P = X_fp / (X_fp.sum(axis=1, keepdims=True) + eps)
    ent  = -(P * np.log(P + eps)).sum(axis=1)         # Shannon entropy
    nact = (X_fp > 0).sum(axis=1).astype(np.float64)  # active quanta count
    peak = P.max(axis=1)                               # peak probability
    return np.column_stack([ent, nact, peak])


print("[1/5] Full MCC (all features)")
full_mcc, full_std, full_per_seed = cv_mcc(X, y, args.n_seeds, args.n_folds)
print(f"      MCC = {full_mcc:+.4f} ± {full_std:.4f}  (per-seed: "
      f"{[f'{m:+.4f}' for m in full_per_seed]})")


print("\n[2/5] Surface MCC (3 features: entropy, active-quanta, peak)")
X_surf = surface_features(X)
surf_mcc, surf_std, _ = cv_mcc(X_surf, y, args.n_seeds, args.n_folds)
print(f"      MCC = {surf_mcc:+.4f} ± {surf_std:.4f}")


delta = full_mcc - surf_mcc
print(f"\n[3/5] Δ (Full - Surface) = {delta:+.4f}")


print(f"\n[4/5] Max single-quantum MCC (best of {n_clust} one-feature classifiers)")
best_mcc = -np.inf
best_q   = -1
for q in range(n_clust):
    Xq = X[:, q:q+1]
    if Xq.sum() == 0:
        continue  # inactive quantum on all docs
    m, _, _ = cv_mcc(Xq, y, n_seeds=1, n_folds=args.n_folds)
    if m > best_mcc:
        best_mcc, best_q = m, q
print(f"      max 1-q MCC = {best_mcc:+.4f}  (cluster q{best_q})")


print("\n[5/5] Permutation baseline (label shuffle)")
perm_rng = np.random.RandomState(0)
y_perm = perm_rng.permutation(y)
perm_mcc, _, _ = cv_mcc(X, y_perm, args.n_seeds, args.n_folds)
print(f"      MCC = {perm_mcc:+.4f}  (should be ≈ 0)")


mega_results = None
if best_mcc >= args.one_cluster_led_threshold * full_mcc:
    print(f"\n[+] Mega-cluster detected (q{best_q} alone reaches "
          f"{best_mcc/full_mcc:.0%} of full MCC). Running ablation.")
    keep = [c for c in range(n_clust) if c != best_q]
    X_no = X[:, keep]
    # Renormalise so each fingerprint sums to 1 after dropping the mega-cluster,
    # preventing residual mass differences from leaking cluster-1 signal
    row_sums = X_no.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0  # guard against all-zero rows
    X_no = X_no / row_sums
    no_mcc, no_std, _ = cv_mcc(X_no, y, args.n_seeds, args.n_folds)
    print(f"      Full MCC (all {n_clust})              = {full_mcc:+.4f}")
    print(f"      MCC of q{best_q} alone                  = {best_mcc:+.4f}")
    print(f"      MCC of {n_clust-1} features without q{best_q} (renorm) = {no_mcc:+.4f}")
    mega_results = {
        "best_q": int(best_q),
        "best_q_mcc": float(best_mcc),
        "full_mcc": float(full_mcc),
        "ablation_mcc_without_best_q": float(no_mcc),
        "ablation_std": float(no_std),
    }


if delta <= args.surface_dominated_threshold:
    flag = "surface-dominated"
elif best_mcc >= args.one_cluster_led_threshold * full_mcc:
    flag = "one-cluster-led"
else:
    flag = "distributed"

print(f"\nFlag: {flag}")


out = {
    "model":              MODEL_NAME,
    "paradigm":           args.paradigm,
    "fp_subdir":          args.fp_subdir,
    "n_docs":             int(n_docs),
    "n_human":            int(n_human),
    "n_ai":               int(n_ai),
    "n_clusters":         int(n_clust),
    "n_seeds":            int(args.n_seeds),
    "n_folds":            int(args.n_folds),
    "full_mcc_mean":      float(full_mcc),
    "full_mcc_std":       float(full_std),
    "full_mcc_per_seed":  full_per_seed,
    "surface_mcc_mean":   float(surf_mcc),
    "surface_mcc_std":    float(surf_std),
    "delta":              float(delta),
    "max_1q_mcc":         float(best_mcc),
    "max_1q_cluster":     int(best_q),
    "permutation_mcc":    float(perm_mcc),
    "flag":               flag,
    "mega_cluster":       mega_results,
}

local_out = os.path.join(fp_dir, "robustness.json")
with open(local_out, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved local: {local_out}")

mirror_dir = os.path.join(PATHS["repo_dir"], "results-mirror",
                          "fingerprint_robustness")
os.makedirs(mirror_dir, exist_ok=True)
mirror_name = f"{MODEL_NAME}_{args.paradigm.replace('-', '_').lower()}.json"
mirror_out  = os.path.join(mirror_dir, mirror_name)
with open(mirror_out, "w") as f:
    json.dump(out, f, indent=2)
print(f"Saved mirror: {mirror_out}")
