"""Spectral, SSC-Lasso, SSC-OMP, and Hierarchical clustering.

Loads similarity matrix, applies PCA, runs multiple clustering methods.

Usage:
  python pipeline/02_clustering.py                        # All methods
  python pipeline/02_clustering.py --skip-spectral        # SSC + Hierarchical only
  python pipeline/02_clustering.py --skip-ssc-lasso       # Spectral + SSC-OMP + Hierarchical
  python pipeline/02_clustering.py --skip-ssc-omp         # Spectral + SSC-Lasso + Hierarchical
  python pipeline/02_clustering.py --skip-hierarchical    # Spectral + SSC only
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import defaultdict
import argparse
import pickle
import time

import numpy as np
from tqdm.auto import tqdm
import torch
import sklearn.cluster
from sklearn.linear_model import Lasso
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from collections import Counter
from joblib import Parallel, delayed

from config import PATHS

# default 1 = sequential; set SSC_N_JOBS>1 to parallelize the Lasso loop
SSC_N_JOBS = int(os.environ.get("SSC_N_JOBS", "1"))

parser = argparse.ArgumentParser(description="QDG Clustering pipeline")
parser.add_argument("--skip-spectral", action="store_true",
                    help="Skip spectral clustering")
parser.add_argument("--skip-ssc-lasso", action="store_true",
                    help="Skip SSC-Lasso")
parser.add_argument("--skip-ssc-omp", action="store_true",
                    help="Skip SSC-OMP")
parser.add_argument("--skip-hierarchical", action="store_true",
                    help="Skip Hierarchical clustering")
parser.add_argument("--no-arccos", action="store_true",
                    help="Skip angular transformation (PCA on raw cosine similarity). "
                         "For L2-normalized gradients this is mathematically equivalent "
                         "to PCA on raw gradients up to scaling.")
parser.add_argument("--similarity-matrix", default=None,
                    help="Override PATHS['similarity_matrix']")
parser.add_argument("--output-dir", default=None,
                    help="Override PATHS['results_dir'] for all output pkl files")
parser.add_argument("--max-k", type=int, default=None,
                    help="Cap all cluster count lists at this value (for smoke testing)")
args = parser.parse_args()

sim_matrix_path = args.similarity_matrix or PATHS["similarity_matrix"]
out_dir = args.output_dir or PATHS["results_dir"]
os.makedirs(out_dir, exist_ok=True)

print(f"Loading similarity matrix from: {sim_matrix_path}")
idxs, C, C_abs = torch.load(sim_matrix_path, weights_only=False)
C = C.cpu().numpy()
C_abs = C_abs.cpu().numpy()

if args.no_arccos:
    print("[--no-arccos] Skipping angular transformation; using raw cosine similarity")
    np.fill_diagonal(C, 1.0)
    # Spectral clustering with affinity='precomputed' requires a non-negative
    # similarity matrix; raw cosine can be in [-1, 1]. Shift+scale into [0, 1]
    # so the spectral baseline does not crash under --no-arccos
    C = (C + 1.0) / 2.0
    np.fill_diagonal(C, 1.0)
    D_angular = 1 - C
    np.fill_diagonal(D_angular, 0)
else:
    C = 1 - np.arccos(np.clip(C, -1.0, 1.0)) / np.pi
    D_angular = 1 - C
    np.fill_diagonal(D_angular, 0)

N = C.shape[0]
print(f"Matrix shape: {C.shape}")
print(f"Similarity range: [{C.min():.3f}, {C.max():.3f}]")

# Configuration
CLUSTER_COUNTS = [10, 20, 30, 40, 50, 60, 70, 80, 90,
                  100, 125, 150, 175, 200, 225, 250, 275, 300, 350, 400,
                  500, 600, 700, 800, 900, 1000, 1100, 1200, 1400, 1500]

# For hierarchical: reduced set of k values
K_VALUES_HIERARCHICAL = [10, 20, 50, 100, 200, 400, 600, 800, 1000, 1500]

# Cap k values for smoke testing (--max-k flag)
if args.max_k is not None:
    CLUSTER_COUNTS        = [k for k in CLUSTER_COUNTS        if k <= args.max_k]
    K_VALUES_HIERARCHICAL = [k for k in K_VALUES_HIERARCHICAL if k <= args.max_k]
    print(f"[--max-k={args.max_k}] Cluster counts capped: "
          f"Spectral={CLUSTER_COUNTS}, Hierarchical={K_VALUES_HIERARCHICAL}")

# SSC-Lasso regularization parameters
SSC_ALPHAS = [0.0001, 0.0005, 0.001, 0.005, 0.01]

# SSC-OMP: number of non-zero coefficients per point
SSC_OMP_N_NONZERO = [5, 10, 20, 50, 100]

# PCA dimensions for SSC
SSC_PCA_DIMS = [50, 100, 200, 500]

# L-grid for coord-descent robustness around historical optimum (d=500, α=0.0001)
SSC_LGRID_PAIRS = [
    (500, 0.0001), (500, 0.0005), (500, 0.001), (500, 0.005), (500, 0.01),
    (200, 0.0001), (100, 0.0001), (50, 0.0001),
]
USE_LGRID = os.environ.get("SSC_LGRID", "0") == "1"
if USE_LGRID:
    print(f"[SSC_LGRID=1] Using L-grid: {len(SSC_LGRID_PAIRS)} explicit (d, α) pairs")

# Cluster counts for SSC evaluation
SSC_CLUSTER_COUNTS = [10, 20, 50, 100, 200, 400]

PAPER_SLOPE = -1.237
RESULTS_DIR = out_dir

# When running --no-arccos ablation, write results to separate files
# so we can compare against the angular-transformed pipeline
if args.no_arccos:
    OUT_SPECTRAL = os.path.join(out_dir, os.path.basename(PATHS["clusters_output"].replace(".pkl", "_no_arccos.pkl")))
    OUT_SSC      = os.path.join(out_dir, os.path.basename(PATHS["clusters_ssc_output"].replace(".pkl", "_no_arccos.pkl")))
    OUT_HIER     = os.path.join(out_dir, os.path.basename(PATHS["clusters_hierarchical_output"].replace(".pkl", "_no_arccos.pkl")))
else:
    OUT_SPECTRAL = os.path.join(out_dir, os.path.basename(PATHS["clusters_output"]))
    OUT_SSC      = os.path.join(out_dir, os.path.basename(PATHS["clusters_ssc_output"]))
    OUT_HIER     = os.path.join(out_dir, os.path.basename(PATHS["clusters_hierarchical_output"]))


def compute_envelope(labels_dict):
    """Compute rank-frequency envelope across k values."""
    all_sizes = {}
    for k, labels in sorted(labels_dict.items()):
        if labels is None:
            continue
        counts = sorted(Counter(labels).values(), reverse=True)
        for rank, size in enumerate(counts, 1):
            if rank not in all_sizes or size > all_sizes[rank]:
                all_sizes[rank] = size
    ranks = np.array(sorted(all_sizes.keys()))
    sizes = np.array([all_sizes[r] for r in ranks])
    return ranks, sizes


def fit_slope(ranks, sizes, rank_min=100, rank_max=1000):
    """Log-log linear fit on envelope (Michaud et al. method)."""
    mask = (ranks >= rank_min) & (ranks <= rank_max) & (sizes > 0)
    if mask.sum() < 3:
        return np.nan
    log_r = np.log10(ranks[mask].astype(float))
    log_s = np.log10(sizes[mask].astype(float))
    slope, intercept = np.polyfit(log_r, log_s, 1)
    return slope


def compute_silhouette(labels, D, name=""):
    """Compute silhouette score, handling edge cases."""
    n_clusters = len(set(labels))
    if n_clusters < 2 or n_clusters >= len(labels):
        return np.nan
    try:
        return silhouette_score(D, labels, metric="precomputed")
    except Exception:
        return np.nan


# PCA dimensionality reduction (following paper)
if not (args.skip_ssc_lasso and args.skip_ssc_omp):
    print("\nPCA dimensionality reduction of C_angular")
    print("(Elhamifar & Vidal 2013, Table 2: applying PCA to data points)")

    max_d = max(SSC_PCA_DIMS)
    print(f"Computing PCA with up to d={max_d} components...")
    pca_full = PCA(n_components=max_d, random_state=42)
    Y_pca_full = pca_full.fit_transform(C)

    explained = pca_full.explained_variance_ratio_
    cumvar = np.cumsum(explained)
    print("Cumulative variance:")
    for d in SSC_PCA_DIMS:
        print(f"  d={d:4d}: {cumvar[d-1]*100:.1f}%")
    print(f"Top 10 eigenvalues: {explained[:10].round(4)}")


def ssc_build_and_cluster(coef_matrix, method_name, cluster_counts, start_time):
    """
    Algorithm 1 steps 2-4: normalize coefficients, build affinity graph, cluster.
    """
    N = coef_matrix.shape[0]

    # Step 2: Normalize c_i <- c_i / ||c_i||_inf
    for i in range(N):
        c_inf = np.max(np.abs(coef_matrix[i, :]))
        if c_inf > 1e-10:
            coef_matrix[i, :] /= c_inf

    # Step 3: W = |C| + |C|^T
    W = np.abs(coef_matrix) + np.abs(coef_matrix).T

    elapsed = time.time() - start_time
    n_edges = int(np.sum(W > 1e-6))
    nnz_per_row = np.mean(np.sum(np.abs(coef_matrix) > 1e-6, axis=1))

    print(f"Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Edges in W: {n_edges} ({n_edges / N / N * 100:.2f}%)")
    print(f"Mean nonzeros per row: {nnz_per_row:.1f}")

    if n_edges == 0:
        print("Warning: No edges in graph!")

    # Step 4: Spectral clustering on W
    cluster_results = dict()
    for n_clusters in tqdm(cluster_counts, desc=f"{method_name} clusters"):
        try:
            labels = sklearn.cluster.SpectralClustering(
                n_clusters=n_clusters, affinity='precomputed', n_init=30, random_state=42
            ).fit_predict(W)
            cluster_results[n_clusters] = labels.tolist()
        except Exception as e:
            print(f"Error at k={n_clusters}: {e}")
            cluster_results[n_clusters] = None

    return {
        "clusters": cluster_results,
        "n_edges": n_edges,
        "nnz_per_row": round(nnz_per_row, 1),
        "compute_time_seconds": round(elapsed, 1),
    }


# 1. Spectral clustering (baseline)
if not args.skip_spectral:
    print("\nSpectral clustering (original QDG baseline)")

    results_spectral = dict()

    for n_clusters in tqdm(CLUSTER_COUNTS, desc="Spectral"):
        clusters_labels = sklearn.cluster.SpectralClustering(
            n_clusters=n_clusters, affinity='precomputed', n_init=30, random_state=42
        ).fit_predict(C)
        clusters_labels_abs = sklearn.cluster.SpectralClustering(
            n_clusters=n_clusters, affinity='precomputed', n_init=30, random_state=42
        ).fit_predict(C_abs)
        results_spectral[n_clusters] = (clusters_labels.tolist(), clusters_labels_abs.tolist())

        with open(OUT_SPECTRAL, "wb") as f:
            pickle.dump(results_spectral, f)

    print(f"Spectral results saved: {OUT_SPECTRAL}")
else:
    print("[SKIP] Spectral clustering (--skip-spectral)")


# 2. SSC-Lasso (Algorithm 1 with PCA and L1 minimization)
if not args.skip_ssc_lasso:
    print("\nSSC-Lasso (Algorithm 1 + PCA, Elhamifar & Vidal 2013)")

    results_ssc = dict()
    if os.path.exists(OUT_SSC):
        try:
            with open(OUT_SSC, "rb") as f:
                results_ssc = pickle.load(f)
            print(f"Loaded {len(results_ssc)} existing (d,α) pair(s) from "
                  f"{OUT_SSC}")
        except Exception as e:
            print(f"(Could not load existing pkl: {e} - starting fresh)")
            results_ssc = dict()

    # Build the iteration order
    if USE_LGRID:
        pairs_iter = list(SSC_LGRID_PAIRS)
        print(f"Iterating {len(pairs_iter)} L-grid pairs")
    else:
        pairs_iter = [(d, a) for d in SSC_PCA_DIMS for a in SSC_ALPHAS]
        print(f"Iterating full grid: {len(pairs_iter)} pairs "
              f"({len(SSC_PCA_DIMS)} d × {len(SSC_ALPHAS)} α)")

    # Group by d to avoid recomputing Y_d slices
    pairs_by_d = defaultdict(list)
    for (d, a) in pairs_iter:
        pairs_by_d[d].append(a)

    for d in sorted(pairs_by_d.keys()):
        print(f"\nPCA d={d} (explained variance: {cumvar[d-1]*100:.1f}%)")
        print(f"Lasso problem: ({d}, {N-1}) per point")

        Y_d = Y_pca_full[:, :d]

        for alpha in pairs_by_d[d]:
            if (d, alpha) in results_ssc:
                print(f"SSC-Lasso d={d}, alpha={alpha} - already computed, skipping")
                continue
            print(f"SSC-Lasso d={d}, alpha={alpha} "
                  f"(n_jobs={SSC_N_JOBS})")
            start_time = time.time()

            coef_matrix = np.zeros((N, N), dtype=np.float32)

            def _fit_one(ii):
                others_idx = np.concatenate(
                    [np.arange(ii), np.arange(ii + 1, N)])
                X = Y_d[others_idx].T
                y = Y_d[ii]
                lasso = Lasso(
                    alpha=alpha,
                    max_iter=5000,
                    fit_intercept=False,
                    tol=1e-4,
                )
                lasso.fit(X, y)
                return ii, others_idx, lasso.coef_.astype(np.float32)

            if SSC_N_JOBS > 1:
                results_inner = Parallel(
                    n_jobs=SSC_N_JOBS, backend="loky", verbose=0,
                )(delayed(_fit_one)(ii) for ii in tqdm(
                    range(N), desc=f"d={d} alpha={alpha}", leave=False))
            else:
                results_inner = [_fit_one(ii) for ii in tqdm(
                    range(N), desc=f"d={d} alpha={alpha}", leave=False)]

            for ii, others_idx, coefs in results_inner:
                coef_matrix[ii, others_idx] = coefs

            # Steps 2-4 of Algorithm 1
            result = ssc_build_and_cluster(
                coef_matrix, f"SSC-Lasso d={d} alpha={alpha}",
                SSC_CLUSTER_COUNTS, start_time
            )
            result["pca_dim"] = d
            result["alpha"] = alpha

            results_ssc[(d, alpha)] = result
            with open(OUT_SSC, "wb") as f:
                pickle.dump(results_ssc, f)

    print(f"SSC-Lasso results saved: {OUT_SSC} "
          f"({len(results_ssc)} total pairs)")
else:
    print("[SKIP] SSC-Lasso (--skip-ssc-lasso)")


# 3. SSC-OMP (Algorithm 1 with PCA and Orthogonal Matching Pursuit)
if not args.skip_ssc_omp:
    print("\nSSC-OMP (Algorithm 1 + PCA, Elhamifar & Vidal 2013)")

    omp_output = OUT_SSC.replace(".pkl", "_omp.pkl")
    results_omp = dict()

    for d in SSC_PCA_DIMS:
        print(f"\nPCA d={d}")

        Y_d = Y_pca_full[:, :d]

        for K in SSC_OMP_N_NONZERO:
            print(f"SSC-OMP d={d}, K={K}")
            start_time = time.time()

            coef_matrix = np.zeros((N, N), dtype=np.float32)

            for ii in tqdm(range(N), desc=f"d={d} K={K}", leave=False):
                others_idx = np.concatenate([np.arange(ii), np.arange(ii + 1, N)])
                X = Y_d[others_idx].T
                y = Y_d[ii]

                omp = OrthogonalMatchingPursuit(
                    n_nonzero_coefs=K,
                    fit_intercept=False,
                )
                omp.fit(X, y)

                coef_matrix[ii, others_idx] = omp.coef_.astype(np.float32)

            result = ssc_build_and_cluster(
                coef_matrix, f"SSC-OMP d={d} K={K}",
                SSC_CLUSTER_COUNTS, start_time
            )
            result["pca_dim"] = d
            result["K"] = K
            results_omp[(d, K)] = result

            with open(omp_output, "wb") as f:
                pickle.dump(results_omp, f)

    print(f"SSC-OMP results saved: {omp_output}")
else:
    print("[SKIP] SSC-OMP (--skip-ssc-omp)")


# 4. Hierarchical clustering (complete-linkage only)
if not args.skip_hierarchical:
    print("\nHierarchical clustering (complete-linkage)")

    print("  Computing condensed distance matrix...")
    D_condensed = squareform(D_angular, checks=False)

    print(f"\n  Linkage: complete")
    t0 = time.time()
    Z = linkage(D_condensed, method="complete")
    elapsed = time.time() - t0
    print(f"    Linkage computed in {elapsed:.1f}s")

    labels_dict = {}
    silhouettes = {}
    for k in K_VALUES_HIERARCHICAL:
        labels = fcluster(Z, t=k, criterion="maxclust") - 1  # fcluster is 1-based
        actual_k = len(set(labels))
        sil = compute_silhouette(labels, D_angular)
        labels_dict[k] = labels
        silhouettes[k] = sil
        print(f"    k={k}: silhouette={sil:.4f}, actual_clusters={actual_k}")

    env_r, env_s = compute_envelope(labels_dict)
    slope = fit_slope(env_r, env_s)
    delta = abs(slope - PAPER_SLOPE)
    print(f"    Envelope slope: {slope:.3f} (paper: {PAPER_SLOPE}, delta={delta:.3f})")

    hierarchical_results = {
        "method": "Hierarchical (complete)",
        "labels": labels_dict,
        "silhouettes": silhouettes,
        "slope": slope,
        "delta": delta,
        "time": elapsed,
    }
    with open(OUT_HIER, "wb") as f:
        pickle.dump(hierarchical_results, f)
    print(f"Hierarchical results saved: {OUT_HIER}")
else:
    print("[SKIP] Hierarchical clustering (--skip-hierarchical)")


# Summary
print("\nDone - Results saved to:")
if not args.skip_spectral:
    print(f"  Spectral: {OUT_SPECTRAL}")
if not args.skip_ssc_lasso:
    print(f"  SSC-Lasso: {OUT_SSC}")
if not args.skip_ssc_omp:
    omp_output = OUT_SSC.replace(".pkl", "_omp.pkl")
    print(f"  SSC-OMP: {omp_output}")
if not args.skip_hierarchical:
    print(f"  Hierarchical: {OUT_HIER}")
