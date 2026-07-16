"""Bootstrap stability for the recommended SSC-Lasso configuration.

For each iteration:
  1. Sub-sample 80% of tokens without replacement.
  2. Re-run the full SSC-Lasso pipeline on the sub-sample
     (Lasso self-expression + ∞-norm normalisation + W = |C|+|C|ᵀ
      + SpectralClustering at k=400).
  3. Compare to the canonical k=400 labels from clusters_ssc_full_more.pkl
     (sliced to the same sub-sampled tokens) via ARI and NMI.
  4. Compute envelope-slope of the bootstrap clustering.
  5. Append to checkpoint.json - saved AFTER every iteration so the run
     can be killed at any time and resumed without losing work.

Recommended SSC-Lasso config per model:
  pythia-19m:  d=500, α=0.0001  (envelope |Δ|=0.008 - overall winner)
  pythia-125m: d=200, α=0.01    (envelope |Δ|=0.092 - SSC winner)

Usage:
  cd experiments/clustering-0
  QDG_MODEL=pythia-19m  BOOT_N_JOBS=20 BOOT_N_ITER=50  python -u pipeline/06_bootstrap_stability.py
  QDG_MODEL=pythia-125m BOOT_N_JOBS=20 BOOT_N_ITER=50  python -u pipeline/06_bootstrap_stability.py

Outputs:
  results/<model>/bootstrap_stability_ssc_lasso.json
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import pickle
import signal
from collections import Counter

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from joblib import Parallel, delayed

from config import PATHS, MODEL_NAME

DEFAULT_CONFIGS = {
    "pythia-19m":  {"d": 500, "alpha": 0.0001},
    "pythia-125m": {"d": 200, "alpha": 0.01},
}

if MODEL_NAME not in DEFAULT_CONFIGS:
    sys.exit(f"Unknown model {MODEL_NAME!r} - expected one of {list(DEFAULT_CONFIGS)}")

# Allow BOOT_D / BOOT_ALPHA env vars to override the canonical config
CFG_D     = int(os.environ.get("BOOT_D",     DEFAULT_CONFIGS[MODEL_NAME]["d"]))
CFG_ALPHA = float(os.environ.get("BOOT_ALPHA", DEFAULT_CONFIGS[MODEL_NAME]["alpha"]))
K_TARGET  = 400  # canonical k from Michaud et al.
SUBSAMPLE_FRAC = float(os.environ.get("BOOT_SUBSAMPLE_FRAC", "0.8"))
N_ITER         = int(os.environ.get("BOOT_N_ITER", "50"))
N_JOBS         = int(os.environ.get("BOOT_N_JOBS", "20"))
SEED_BASE      = 42
PAPER_SLOPE    = -1.237
ITER_TIMEOUT_S = int(os.environ.get("BOOT_ITER_TIMEOUT_S", "900"))   # 15 min
SPECTRAL_NINIT = int(os.environ.get("BOOT_SPECTRAL_NINIT", "5"))


class IterTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise IterTimeout(f"iteration exceeded {ITER_TIMEOUT_S}s")

# If BOOT_D or BOOT_ALPHA is set explicitly, the output file is config-suffixed
# so multiple test runs do not clobber each other; otherwise use the original
# filename for back-compat with the canonical recommended config per model
if "BOOT_D" in os.environ or "BOOT_ALPHA" in os.environ:
    OUT_PATH = os.path.join(
        PATHS["results_dir"],
        f"bootstrap_stability_ssc_lasso_d{CFG_D}_a{CFG_ALPHA}.json",
    )
else:
    OUT_PATH = os.path.join(PATHS["results_dir"], "bootstrap_stability_ssc_lasso.json")


def envelope_slope(labels, lo=100, hi=1000):
    """Single-clustering envelope slope. Same convention as pipeline/03_envelope_analysis.py."""
    counts = sorted(Counter(np.asarray(labels).flatten().tolist()).values(),
                    reverse=True)
    ranks = np.arange(1, len(counts) + 1)
    sizes = np.array(counts)
    mask = (ranks >= lo) & (ranks <= hi) & (sizes > 0)
    if mask.sum() < 3:
        # Fall back to the largest available rank window
        mask = sizes > 0
        if mask.sum() < 3:
            return None
    slope, _ = np.polyfit(np.log10(ranks[mask].astype(float)),
                          np.log10(sizes[mask].astype(float)), 1)
    return float(slope)


def fit_lasso_one(i, Y_sub, alpha):
    """Self-expression Lasso for one point against all others in the sub-sample."""
    n = Y_sub.shape[0]
    others_idx = np.concatenate([np.arange(i), np.arange(i + 1, n)])
    X = Y_sub[others_idx].T   # (d, n-1)
    y = Y_sub[i]               # (d,)
    lasso = Lasso(alpha=alpha, max_iter=5000, fit_intercept=False, tol=1e-4)
    lasso.fit(X, y)
    return i, others_idx, lasso.coef_.astype(np.float32)


def ssc_lasso_one_iteration(Y_sub, alpha, k):
    """One full SSC-Lasso pipeline on a sub-sample. Returns labels (length n)."""
    n = Y_sub.shape[0]
    if N_JOBS > 1:
        results = Parallel(n_jobs=N_JOBS, backend="loky", verbose=0)(
            delayed(fit_lasso_one)(i, Y_sub, alpha) for i in range(n))
    else:
        results = [fit_lasso_one(i, Y_sub, alpha) for i in range(n)]

    coef = np.zeros((n, n), dtype=np.float32)
    for i, others_idx, c in results:
        coef[i, others_idx] = c

    # Algorithm 1 step 2: row-wise ∞-norm normalisation
    for i in range(n):
        c_inf = np.max(np.abs(coef[i, :]))
        if c_inf > 1e-10:
            coef[i, :] /= c_inf

    # Algorithm 1 step 3: W = |C| + |C|ᵀ
    W = np.abs(coef) + np.abs(coef).T

    # Algorithm 1 step 4: spectral clustering on W
    # n_init lowered from 30 -> 5 (k-means re-initialisation is not the main cost
    # on disconnected affinity graphs; eigendecomposition is); eigen_tol relaxed
    # so ARPACK does not chase degenerate small eigenvalues forever
    labels = SpectralClustering(
        n_clusters=k, affinity="precomputed",
        n_init=SPECTRAL_NINIT, random_state=0,
        eigen_tol=1e-3,
    ).fit_predict(W)
    return labels.astype(np.int32)


print(f"Bootstrap stability: {MODEL_NAME} (SSC-Lasso d={CFG_D}, α={CFG_ALPHA})")
print(f"  iterations target: {N_ITER}, subsample: {int(SUBSAMPLE_FRAC*100)}%, "
      f"n_jobs: {N_JOBS}")

t0 = time.time()
print(f"\nLoading similarity matrix from: {PATHS['similarity_matrix']}")
idxs, C_raw, _ = torch.load(PATHS["similarity_matrix"], weights_only=False)
C_raw = C_raw.cpu().numpy().astype(np.float64)
N = C_raw.shape[0]
print(f"  shape: {C_raw.shape}, time: {time.time()-t0:.1f}s")

t0 = time.time()
C_angular = 1 - np.arccos(np.clip(C_raw, -1.0, 1.0)) / np.pi
np.fill_diagonal(C_angular, 1.0)
print(f"  angular transform: {time.time()-t0:.1f}s")

t0 = time.time()
print(f"\nFitting PCA(n_components={CFG_D}) once on the full angular matrix")
Y_full = PCA(n_components=CFG_D).fit_transform(C_angular)
print(f"  PCA done: shape={Y_full.shape}, time: {time.time()-t0:.1f}s")
del C_raw, C_angular

print(f"\nLoading reference SSC-Lasso labels at k={K_TARGET}")
ref_path = os.path.join(PATHS["results_dir"], "clusters_ssc_full_more.pkl")
with open(ref_path, "rb") as f:
    ssc_pkl = pickle.load(f)
key = (CFG_D, CFG_ALPHA)
if key not in ssc_pkl:
    sys.exit(f"Config {key} not found in {ref_path}. Available: "
             f"{sorted([k for k in ssc_pkl if isinstance(k, tuple)])}")
ref_labels = np.asarray(ssc_pkl[key]["clusters"][K_TARGET], dtype=np.int32)
print(f"  reference labels: shape={ref_labels.shape}, "
      f"n_active={len(np.unique(ref_labels))}")
del ssc_pkl


results = {
    "model": MODEL_NAME,
    "config": {"d": CFG_D, "alpha": CFG_ALPHA, "k": K_TARGET},
    "subsample_frac": SUBSAMPLE_FRAC,
    "n_iter_target": N_ITER,
    "iterations": [],
}

if os.path.exists(OUT_PATH):
    try:
        with open(OUT_PATH) as f:
            existing = json.load(f)
        if (existing.get("config") == results["config"]
            and existing.get("model") == MODEL_NAME):
            results = existing
            results["n_iter_target"] = N_ITER  # allow extending the target
            print(f"\nResuming from checkpoint: {len(results['iterations'])} "
                  f"iterations already done.")
        else:
            print(f"\n[WARNING] checkpoint config mismatch - starting fresh.")
    except Exception as e:
        print(f"\n[WARNING] could not load checkpoint: {e} - starting fresh.")

done_iters = {it["iter"] for it in results["iterations"]}
remaining = [i for i in range(N_ITER) if i not in done_iters]
print(f"\nIterations to run: {len(remaining)} ({remaining[:5]}{'...' if len(remaining)>5 else ''})")


n_sub = int(SUBSAMPLE_FRAC * N)
print(f"\nSub-sample size: {n_sub} of {N} tokens\n")

for iter_idx in remaining:
    t_iter = time.time()
    seed = SEED_BASE + iter_idx
    rng = np.random.RandomState(seed)
    sub = np.sort(rng.choice(N, size=n_sub, replace=False))

    Y_sub = Y_full[sub]
    print(f"[iter {iter_idx+1}/{N_ITER}] seed={seed}  n_sub={len(sub)}  "
          f"running SSC-Lasso (timeout={ITER_TIMEOUT_S}s) ...", flush=True)

    # Per-iter timeout via SIGALRM - if a single sub-sample produces a
    # pathological affinity graph (disconnected -> eigendecomposition hangs)
    # we abort, log it, and move on instead of stalling for hours
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(ITER_TIMEOUT_S)
    try:
        boot_labels = ssc_lasso_one_iteration(Y_sub, CFG_ALPHA, K_TARGET)
    except IterTimeout as e:
        signal.alarm(0)
        print(f"  [TIMEOUT] {e}", flush=True)
        results["iterations"].append({
            "iter": iter_idx, "seed": int(seed), "n_sub": int(n_sub),
            "error": f"IterTimeout: {e}",
            "elapsed_s": round(time.time() - t_iter, 1),
        })
        tmp = OUT_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(results, f, indent=2)
        os.replace(tmp, OUT_PATH)
        continue
    except Exception as e:
        signal.alarm(0)
        print(f"  [FAIL] {type(e).__name__}: {e}", flush=True)
        results["iterations"].append({
            "iter": iter_idx, "seed": int(seed), "n_sub": int(n_sub),
            "error": f"{type(e).__name__}: {e}",
            "elapsed_s": round(time.time() - t_iter, 1),
        })
        tmp = OUT_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(results, f, indent=2)
        os.replace(tmp, OUT_PATH)
        continue
    finally:
        signal.alarm(0)

    # Compare to reference (sliced to the same sub-sampled tokens)
    ref_sub = ref_labels[sub]
    ari = float(adjusted_rand_score(ref_sub, boot_labels))
    nmi = float(normalized_mutual_info_score(ref_sub, boot_labels))
    n_active = int(len(np.unique(boot_labels)))
    # lo=10/hi=400 used here because bootstrap subsamples ~80% of tokens,
    # giving fewer clusters than the full run; the canonical window (lo=100,
    # hi=1000) would often exceed the available rank range. The bootstrap
    # slope is therefore NOT directly comparable to the paper target (-1.237)
    # and is stored for diagnostic purposes only - NMI/ARI are the primary
    # bootstrap metrics reported in the paper
    slope = envelope_slope(boot_labels, lo=10, hi=min(400, n_sub - 1))

    elapsed = time.time() - t_iter
    rec = {
        "iter": iter_idx,
        "seed": int(seed),
        "n_sub": int(n_sub),
        "ari": ari,
        "nmi": nmi,
        "n_active": n_active,
        "envelope_slope": slope,
        "elapsed_s": round(elapsed, 1),
    }
    results["iterations"].append(rec)

    # save atomically via tmp + rename after every iteration
    tmp = OUT_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2)
    os.replace(tmp, OUT_PATH)

    completed = [it for it in results["iterations"] if "ari" in it]
    aris = np.array([it["ari"] for it in completed])
    nmis = np.array([it["nmi"] for it in completed])
    print(f"  ARI={ari:.4f}  NMI={nmi:.4f}  active={n_active}/400  "
          f"slope={slope:+.3f}  ({elapsed:.0f}s)  "
          f"running mean: ARI={aris.mean():.4f}±{aris.std():.4f}  "
          f"NMI={nmis.mean():.4f}±{nmis.std():.4f}", flush=True)


completed = [it for it in results["iterations"] if "ari" in it]
if completed:
    aris  = np.array([it["ari"]  for it in completed])
    nmis  = np.array([it["nmi"]  for it in completed])
    slopes = np.array([it["envelope_slope"] for it in completed
                       if it["envelope_slope"] is not None])
    summary = {
        "n_completed":   len(completed),
        "ari_mean":      float(aris.mean()),
        "ari_std":       float(aris.std()),
        "ari_min":       float(aris.min()),
        "ari_max":       float(aris.max()),
        "nmi_mean":      float(nmis.mean()),
        "nmi_std":       float(nmis.std()),
        "slope_mean":    float(slopes.mean()) if len(slopes) else None,
        "slope_std":     float(slopes.std())  if len(slopes) else None,
        "paper_slope":   PAPER_SLOPE,
    }
    results["summary"] = summary

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    # Also write the tracked results-mirror copy so figures/fig_11_stability.py
    # can read it directly (matches the pipeline/06b output convention).
    # Skipped when BOOT_D/BOOT_ALPHA override the canonical config, so test
    # runs never overwrite the canonical mirror file.
    if "BOOT_D" in os.environ or "BOOT_ALPHA" in os.environ:
        print("  Non-canonical config (BOOT_D/BOOT_ALPHA set) - mirror write skipped.")
    else:
        mirror_dir = os.path.join(PATHS["repo_dir"], "results-mirror", "bootstrap_stability")
        os.makedirs(mirror_dir, exist_ok=True)
        mirror_path = os.path.join(mirror_dir, f"{MODEL_NAME}_ssc_lasso.json")
        with open(mirror_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Mirror: {mirror_path}")

    print(f"\nSummary - {len(completed)} completed iterations")
    print(f"  ARI : {summary['ari_mean']:.4f} ± {summary['ari_std']:.4f}  "
          f"(min={summary['ari_min']:.4f}, max={summary['ari_max']:.4f})")
    print(f"  NMI : {summary['nmi_mean']:.4f} ± {summary['nmi_std']:.4f}")
    if summary["slope_mean"] is not None:
        print(f"  slope: {summary['slope_mean']:+.4f} ± {summary['slope_std']:.4f}  "
              f"(paper = {PAPER_SLOPE})")
    print(f"\nSaved: {OUT_PATH}")
else:
    print("\nNo iterations completed.")
