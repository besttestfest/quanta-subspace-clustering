"""Envelope and power-law analysis for clustering methods.

Evaluates envelope slopes (rank-frequency) and fits power-law distributions.

Usage:
  python pipeline/03_envelope_analysis.py                # All analyses
  python pipeline/03_envelope_analysis.py --skip-baseline     # Skip Spectral
  python pipeline/03_envelope_analysis.py --skip-ssc-compare  # Skip SSC comparison
  python pipeline/03_envelope_analysis.py --skip-envelope     # Skip envelope computation
  python pipeline/03_envelope_analysis.py --skip-powerlaw     # Skip power-law fitting
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import argparse
import json
from collections import Counter

import numpy as np
import torch

from config import PATHS

parser = argparse.ArgumentParser(description="Envelope and power-law analysis")
parser.add_argument("--skip-baseline", action="store_true",
                    help="Skip Spectral baseline analysis")
parser.add_argument("--skip-ssc-compare", action="store_true",
                    help="Skip SSC comparison")
parser.add_argument("--skip-envelope", action="store_true",
                    help="Skip envelope slope computation")
parser.add_argument("--skip-powerlaw", action="store_true",
                    help="Skip power-law fitting")
args = parser.parse_args()

RESULTS_DIR = PATHS["results_dir"]

PAPER_SLOPE = -1.237


def compute_envelope(labels_dict):
    """Compute rank-frequency envelope across k values."""
    all_sizes = {}
    for k, labels in sorted(labels_dict.items()):
        if labels is None:
            continue
        labels = np.asarray(labels)
        counts = sorted(Counter(labels).values(), reverse=True)
        for rank, size in enumerate(counts, 1):
            if rank not in all_sizes or size > all_sizes[rank]:
                all_sizes[rank] = size
    ranks = np.array(sorted(all_sizes.keys()))
    sizes = np.array([all_sizes[r] for r in ranks])
    return ranks, sizes


def fit_slope(ranks, sizes, rank_min=100, rank_max=1000):
    """Log-log linear fit on envelope."""
    mask = (ranks >= rank_min) & (ranks <= rank_max) & (sizes > 0)
    if mask.sum() < 3:
        return np.nan
    log_r = np.log10(ranks[mask].astype(float))
    log_s = np.log10(sizes[mask].astype(float))
    slope, intercept = np.polyfit(log_r, log_s, 1)
    return slope


def fit_powerlaw(sizes):
    """Fit power-law distribution to cluster sizes via MLE."""
    sizes = sizes[sizes > 0]
    if len(sizes) < 3:
        return None
    alpha = 1 + len(sizes) / np.sum(np.log(sizes / (sizes.min() - 0.5)))
    return alpha


def powerlaw_bic(sizes, alpha):
    """BIC for power-law fit."""
    sizes = sizes[sizes > 0]
    n = len(sizes)
    xmin = sizes.min() - 0.5  # discrete correction, matching fit_powerlaw
    logL = (n * np.log(alpha - 1)
            + n * (alpha - 1) * np.log(xmin)
            - alpha * np.sum(np.log(sizes)))
    bic = -2 * logL + 1 * np.log(n)
    return bic


def exponential_bic(sizes):
    """BIC for exponential fit."""
    sizes = sizes[sizes > 0]
    n = len(sizes)
    lam = 1 / sizes.mean()
    logL = n * (np.log(lam) - lam * sizes.mean())
    bic = -2 * logL + 1 * np.log(n)
    return bic


print("Loading similarity matrix...")
idxs, C_raw, C_abs_raw = torch.load(PATHS["similarity_matrix"], weights_only=False)
C = C_raw.cpu().numpy().astype(np.float64)
C_abs = C_abs_raw.cpu().numpy().astype(np.float64)
C_angular = 1 - np.arccos(np.clip(C, -1.0, 1.0)) / np.pi
D_angular = 1 - C_angular
np.fill_diagonal(D_angular, 0)

if not args.skip_baseline:
    print("\nSpectral baseline analysis")

    if os.path.exists(PATHS["clusters_output"]):
        with open(PATHS["clusters_output"], "rb") as f:
            results_spectral = pickle.load(f)
        print(f"Loaded Spectral: {len(results_spectral)} cluster counts")
        
        labels_dict = {k: v[0] for k, v in results_spectral.items()}
        ranks, sizes = compute_envelope(labels_dict)
        slope = fit_slope(ranks, sizes)
        delta = abs(slope - PAPER_SLOPE)
        
        print(f"Envelope: {len(ranks)} ranks, slope={slope:.4f}, delta={delta:.4f}")
        
        output_path = os.path.join(RESULTS_DIR, "envelope_spectral.json")
        with open(output_path, "w") as f:
            json.dump({
                "method": "Spectral",
                "slope": float(slope),
                "paper_slope": float(PAPER_SLOPE),
                "delta": float(delta),
                "n_ranks": int(len(ranks)),
            }, f, indent=2)
        print(f"Saved: {output_path}")
    else:
        print(f"Spectral results not found at {PATHS['clusters_output']}")

if not args.skip_ssc_compare:
    print("\nSSC comparison (Lasso + OMP)")

    ssc_methods = {}

    if os.path.exists(PATHS["clusters_ssc_output"]):
        with open(PATHS["clusters_ssc_output"], "rb") as f:
            results_ssc_lasso = pickle.load(f)
        
        for (d, alpha), result in results_ssc_lasso.items():
            if "clusters" in result:
                labels_dict = result["clusters"]
                ranks, sizes = compute_envelope(labels_dict)
                slope = fit_slope(ranks, sizes)
                
                ssc_methods[(d, alpha, "Lasso")] = {
                    "slope": slope,
                    "delta": abs(slope - PAPER_SLOPE),
                }

    omp_path = PATHS["clusters_ssc_output"].replace(".pkl", "_omp.pkl")
    if os.path.exists(omp_path):
        with open(omp_path, "rb") as f:
            results_ssc_omp = pickle.load(f)
        
        for (d, K), result in results_ssc_omp.items():
            if "clusters" in result:
                labels_dict = result["clusters"]
                ranks, sizes = compute_envelope(labels_dict)
                slope = fit_slope(ranks, sizes)
                
                ssc_methods[(d, K, "OMP")] = {
                    "slope": slope,
                    "delta": abs(slope - PAPER_SLOPE),
                }

    print(f"SSC methods found: {len(ssc_methods)}")
    for (param1, param2, method), metrics in sorted(ssc_methods.items(),
                                                     key=lambda x: x[1]["delta"]):
        print(f"  {method:8s} {param1:5} {param2:8}: slope={metrics['slope']:.4f}, delta={metrics['delta']:.4f}")

    output_path = os.path.join(RESULTS_DIR, "ssc_comparison.json")
    # JSON doesn't support tuple keys; flatten to "Method_d=X_param2=Y" form
    ssc_methods_serializable = {}
    for (param1, param2, method), metrics in ssc_methods.items():
        param2_label = "alpha" if method == "Lasso" else "K"
        key = f"{method}_d={param1}_{param2_label}={param2}"
        ssc_methods_serializable[key] = {
            **metrics,
            "method": method,
            "d": param1,
            param2_label: param2,
        }
    with open(output_path, "w") as f:
        json.dump(ssc_methods_serializable, f, indent=2, default=str)
    print(f"Saved: {output_path}")

if not args.skip_envelope:
    print("\nEnvelope slope analysis")

    envelope_results = {}

    # Spectral
    if os.path.exists(PATHS["clusters_output"]):
        with open(PATHS["clusters_output"], "rb") as f:
            results_spectral = pickle.load(f)
        labels_dict = {k: v[0] for k, v in results_spectral.items()}
        ranks, sizes = compute_envelope(labels_dict)
        slope = fit_slope(ranks, sizes)
        envelope_results["Spectral"] = {
            "slope": float(slope),
            "delta": float(abs(slope - PAPER_SLOPE)),
        }

    # SSC-Lasso - pick the (d, alpha) pair with the smallest |slope - PAPER_SLOPE|
    if os.path.exists(PATHS["clusters_ssc_output"]):
        with open(PATHS["clusters_ssc_output"], "rb") as f:
            results_ssc = pickle.load(f)
        if results_ssc:
            best_key, best_slope, best_delta = None, np.nan, np.inf
            for key, entry in results_ssc.items():
                if not isinstance(entry, dict) or "clusters" not in entry:
                    continue
                r, s = compute_envelope(entry["clusters"])
                sl = fit_slope(r, s)
                d = abs(sl - PAPER_SLOPE) if not np.isnan(sl) else np.inf
                if d < best_delta:
                    best_delta, best_slope, best_key = d, sl, key
            if best_key is not None:
                envelope_results["SSC-Lasso"] = {
                    "slope": float(best_slope),
                    "delta": float(best_delta),
                    "config": str(best_key),
                }

    # SSC-OMP - same: pick the (d, K) pair with the smallest |slope - PAPER_SLOPE|
    omp_path = PATHS["clusters_ssc_output"].replace(".pkl", "_omp.pkl")
    if os.path.exists(omp_path):
        with open(omp_path, "rb") as f:
            results_omp = pickle.load(f)
        if results_omp:
            best_key, best_slope, best_delta = None, np.nan, np.inf
            for key, entry in results_omp.items():
                if not isinstance(entry, dict) or "clusters" not in entry:
                    continue
                r, s = compute_envelope(entry["clusters"])
                sl = fit_slope(r, s)
                d = abs(sl - PAPER_SLOPE) if not np.isnan(sl) else np.inf
                if d < best_delta:
                    best_delta, best_slope, best_key = d, sl, key
            if best_key is not None:
                envelope_results["SSC-OMP"] = {
                    "slope": float(best_slope),
                    "delta": float(best_delta),
                    "config": str(best_key),
                }

    # Hierarchical
    hier_path = PATHS["clusters_hierarchical_output"]
    if os.path.exists(hier_path):
        with open(hier_path, "rb") as f:
            hier_results = pickle.load(f)
        envelope_results["Hierarchical"] = {
            "slope": float(hier_results["slope"]),
            "delta": float(hier_results["delta"]),
        }

    print(f"Envelope slopes (4 methods):")
    for method, data in sorted(envelope_results.items(),
                                key=lambda x: x[1]["delta"]):
        print(f"  {method:20s}: {data['slope']:8.4f} (delta={data['delta']:.4f})")

    output_path = os.path.join(RESULTS_DIR, "envelope_slopes.json")
    with open(output_path, "w") as f:
        json.dump(envelope_results, f, indent=2)
    print(f"Saved: {output_path}")

if not args.skip_powerlaw:
    print("\nPower-law validation")

    powerlaw_results = {}

    methods_to_test = [
        ("Spectral", PATHS["clusters_output"], None),
        ("SSC-Lasso", PATHS["clusters_ssc_output"], "Lasso"),
        ("SSC-OMP", PATHS["clusters_ssc_output"].replace(".pkl", "_omp.pkl"), "OMP"),
        ("Hierarchical", os.path.join(RESULTS_DIR, "clusters_hierarchical.pkl"), "Hier"),
    ]

    for method_name, path, variant in methods_to_test:
        if not os.path.exists(path):
            continue

        print(f"\n{method_name}:")

        with open(path, "rb") as f:
            data = pickle.load(f)

        if method_name == "Spectral":
            labels_dict = {k: v[0] for k, v in data.items()}
        elif method_name == "Hierarchical":
            labels_dict = data["labels"]
        else:
            # SSC: use first config
            first_key = sorted(data.keys())[0]
            labels_dict = data[first_key]["clusters"]

        all_sizes = []
        for k, labels in labels_dict.items():
            if labels is None:
                continue
            labels = np.asarray(labels)
            sizes = sorted(Counter(labels).values(), reverse=True)
            all_sizes.extend(sizes)

        all_sizes = np.array(all_sizes)

        # Fit power-law (alpha_pl can be None if the size distribution is
        # too short or degenerate - guard the print and JSON output below)
        alpha_pl = fit_powerlaw(all_sizes)
        bic_pl = powerlaw_bic(all_sizes, alpha_pl) if alpha_pl else np.nan
        bic_exp = exponential_bic(all_sizes)
        delta_bic = bic_exp - bic_pl if not np.isnan(bic_pl) else np.nan

        if alpha_pl is None:
            print(f"  Power-law fit failed (degenerate size distribution)")
            print(f"  Exponential BIC={bic_exp:.1f}")
        else:
            print(f"  Power-law α={alpha_pl:.3f}, BIC={bic_pl:.1f}")
            print(f"  Exponential BIC={bic_exp:.1f}, ΔBIC={delta_bic:.1f}")

        powerlaw_results[method_name] = {
            "alpha": float(alpha_pl) if alpha_pl else None,
            "bic_powerlaw": float(bic_pl) if not np.isnan(bic_pl) else None,
            "bic_exponential": float(bic_exp),
            "delta_bic": float(delta_bic) if not np.isnan(delta_bic) else None,
        }

    output_path = os.path.join(RESULTS_DIR, "powerlaw_validation.json")
    with open(output_path, "w") as f:
        json.dump(powerlaw_results, f, indent=2)
    print(f"\nSaved: {output_path}")

print("\nDone - Results saved to:", RESULTS_DIR)
