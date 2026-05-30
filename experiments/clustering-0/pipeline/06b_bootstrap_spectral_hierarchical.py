"""Bootstrap stability for Spectral and Hierarchical clustering.

For each iteration:
  1. Sub-sample 80% of tokens without replacement.
  2. Re-run clustering on the angular-distance submatrix at k=400.
  3. Compare to canonical k=400 labels via ARI and NMI.
  4. Append to checkpoint.json after every iteration.

Usage (from experiments/clustering-0):
  QDG_MODEL=pythia-19m  python -u pipeline/06b_bootstrap_spectral_hierarchical.py --method spectral
  QDG_MODEL=pythia-19m  python -u pipeline/06b_bootstrap_spectral_hierarchical.py --method hierarchical
  QDG_MODEL=pythia-125m python -u pipeline/06b_bootstrap_spectral_hierarchical.py --method spectral
  QDG_MODEL=pythia-125m python -u pipeline/06b_bootstrap_spectral_hierarchical.py --method hierarchical

Outputs:
  results/<model>/bootstrap_stability_<method>.json
  results-mirror/bootstrap_stability/<model>_<method>.json
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import pickle
import time

import numpy as np
import torch
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from tqdm import tqdm

from config import PATHS, MODEL_NAME


parser = argparse.ArgumentParser()
parser.add_argument("--method", required=True, choices=["spectral", "hierarchical"],
                    help="Which clustering method to bootstrap.")
parser.add_argument("--n-iter",       type=int,   default=int(os.environ.get("BOOT_N_ITER", "50")))
parser.add_argument("--subsample",    type=float, default=float(os.environ.get("BOOT_SUBSAMPLE_FRAC", "0.8")))
parser.add_argument("--k",            type=int,   default=400)
parser.add_argument("--seed",         type=int,   default=42)
parser.add_argument("--spectral-ninit", type=int, default=int(os.environ.get("BOOT_SPECTRAL_NINIT", "5")))
args = parser.parse_args()

METHOD        = args.method
N_ITER        = args.n_iter
SUBSAMPLE     = args.subsample
K             = args.k
SEED_BASE     = args.seed
SPECTRAL_NINIT = args.spectral_ninit

OUT_LOCAL  = os.path.join(PATHS["results_dir"], f"bootstrap_stability_{METHOD}.json")
MIRROR_DIR = os.path.join(os.path.dirname(os.path.dirname(PATHS["results_dir"])),
                          "quantization-model", "results-mirror", "bootstrap_stability")
os.makedirs(MIRROR_DIR, exist_ok=True)
OUT_MIRROR = os.path.join(MIRROR_DIR, f"{MODEL_NAME}_{METHOD}.json")

print(f"Bootstrap stability - method={METHOD}, model={MODEL_NAME}")
print(f"  n_iter={N_ITER}, subsample={SUBSAMPLE}, k={K}")
print(f"  Output: {OUT_LOCAL}")

print(f"\nLoading similarity matrix: {PATHS['similarity_matrix']}")
_data = torch.load(PATHS["similarity_matrix"], map_location="cpu", weights_only=False)
# .pt files are (idxs, C_tensor, extra) - tensors have .numpy()
idxs = _data[0]
C_raw = _data[1]
C = np.array(C_raw, dtype=np.float32)
n_tokens = C.shape[0]
print(f"  Matrix shape: {C.shape}")

# Angular distance matrix (same transform as original pipeline)
ANG = np.arccos(np.clip(C, -1.0, 1.0)) / np.pi   # in [0, 1]
del C_raw, C, _data

if METHOD == "spectral":
    with open(PATHS["clusters_output"], "rb") as f:
        sp_pkl = pickle.load(f)
    ref_labels = np.asarray(sp_pkl[K][0], dtype=np.int32)
    del sp_pkl
    print(f"  Spectral reference labels: k={K}, unique={len(np.unique(ref_labels))}")
else:
    with open(PATHS["clusters_hierarchical_output"], "rb") as f:
        hier_pkl = pickle.load(f)
    ref_labels = np.asarray(hier_pkl["labels"][K], dtype=np.int32)
    del hier_pkl
    print(f"  Hierarchical reference labels: k={K}, unique={len(np.unique(ref_labels))}")

results = []
start_iter = 0
if os.path.exists(OUT_LOCAL):
    try:
        with open(OUT_LOCAL) as f:
            existing = json.load(f)
        results = existing.get("iterations", [])
        start_iter = len(results)
        print(f"  Resuming from iteration {start_iter}")
    except Exception as e:
        print(f"  [WARNING] could not load checkpoint: {e} - starting fresh.")

summary = None  # guard for N_ITER=0 or pre-completed run
n_sub = int(n_tokens * SUBSAMPLE)
rng = np.random.default_rng(SEED_BASE)

for it in tqdm(range(start_iter, N_ITER), desc=f"Bootstrap {METHOD}"):
    t0 = time.time()

    # Subsample
    idx = np.sort(rng.choice(n_tokens, size=n_sub, replace=False))
    sub_ang = ANG[np.ix_(idx, idx)]

    # Cluster
    if METHOD == "spectral":
        affinity = 1.0 - sub_ang          # affinity in [0, 1]
        sc = SpectralClustering(
            n_clusters=K,
            affinity="precomputed",
            n_init=SPECTRAL_NINIT,
            random_state=SEED_BASE + it,
            n_jobs=-1,
        )
        boot_labels = sc.fit_predict(affinity)
    else:
        # Hierarchical complete-linkage on condensed distance matrix
        condensed = squareform(sub_ang, checks=False)
        Z = linkage(condensed, method="complete")
        boot_labels = fcluster(Z, t=K, criterion="maxclust") - 1  # 0-indexed

    # Compare against canonical labels on the subsampled subset
    ref_sub = ref_labels[idx]
    ari = float(adjusted_rand_score(ref_sub, boot_labels))
    nmi = float(normalized_mutual_info_score(ref_sub, boot_labels, average_method="arithmetic"))

    elapsed = time.time() - t0
    results.append({"iter": it, "ari": ari, "nmi": nmi, "elapsed_s": round(elapsed, 1)})

    # Save checkpoint after every iteration
    summary = {
        "model":      MODEL_NAME,
        "method":     METHOD,
        "n_iter_done": len(results),
        "n_iter_total": N_ITER,
        "subsample_frac": SUBSAMPLE,
        "k": K,
        "ari_mean":  float(np.mean([r["ari"] for r in results])),
        "ari_std":   float(np.std([r["ari"]  for r in results])),
        "ari_median":float(np.median([r["ari"] for r in results])),
        "nmi_mean":  float(np.mean([r["nmi"] for r in results])),
        "nmi_std":   float(np.std([r["nmi"]  for r in results])),
        "nmi_median":float(np.median([r["nmi"] for r in results])),
        "iterations": results,
    }
    with open(OUT_LOCAL, "w") as f:
        json.dump(summary, f, indent=2)

    if (it + 1) % 5 == 0 or it == N_ITER - 1:
        print(f"  iter {it+1:3d}/{N_ITER}  ARI={ari:.3f}  NMI={nmi:.3f}  "
              f"(mean ARI={summary['ari_mean']:.3f} ± {summary['ari_std']:.3f}, "
              f"NMI={summary['nmi_mean']:.3f} ± {summary['nmi_std']:.3f})  [{elapsed:.1f}s]")

if start_iter >= N_ITER and os.path.exists(OUT_LOCAL):
    # All iterations already done in a previous run - load the checkpoint.
    with open(OUT_LOCAL) as f:
        summary = json.load(f)

if summary is None:
    print("\nNo iterations completed (N_ITER=0?). Skipping mirror write.")
else:
    with open(OUT_MIRROR, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDone. Final results:")
    print(f"  ARI: {summary['ari_mean']:.3f} ± {summary['ari_std']:.3f}  (median {summary['ari_median']:.3f})")
    print(f"  NMI: {summary['nmi_mean']:.3f} ± {summary['nmi_std']:.3f}  (median {summary['nmi_median']:.3f})")
    print(f"  Saved: {OUT_LOCAL}")
    print(f"  Mirror: {OUT_MIRROR}")
