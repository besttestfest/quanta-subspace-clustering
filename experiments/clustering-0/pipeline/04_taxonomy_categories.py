"""Taxonomy and feature selection analysis for clusters.

Analyzes token category distribution and L1 feature selection.

Usage:
  python pipeline/04_taxonomy_categories.py                  # All analyses
  python pipeline/04_taxonomy_categories.py --skip-taxonomy  # Skip taxonomy
  python pipeline/04_taxonomy_categories.py --skip-selection # Skip feature selection
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
import pickle
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef

from config import PATHS

parser = argparse.ArgumentParser(description="Taxonomy and feature selection analysis")
parser.add_argument("--skip-taxonomy", action="store_true",
                    help="Skip taxonomy analysis")
parser.add_argument("--skip-selection", action="store_true",
                    help="Skip L1 feature selection")
args = parser.parse_args()

RESULTS_DIR = PATHS["results_dir"]
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_token_categories():
    """Load per-cluster token categories from quanta_fingerprint output.

    Returns a flat dict mapping str(cluster_id) -> category_name, e.g.
    {"0": "SYNTACTIC", "1": "NUMERIC", ...}.  This is the format expected
    by the taxonomy loop: categories.get(str(label), "UNKNOWN").
    """
    # SSC-Lasso fingerprint directories use a hyphen in newer runs and
    # an underscore in legacy runs; check both spellings
    candidates = [
        os.path.join(RESULTS_DIR, "figures", "quanta_fingerprint", "per_cluster_categories.json"),
        os.path.join(RESULTS_DIR, "figures", "quanta_fingerprint_ssc-lasso", "per_cluster_categories.json"),
        os.path.join(RESULTS_DIR, "figures", "quanta_fingerprint_ssc_lasso", "per_cluster_categories.json"),
        os.path.join(PATHS["base_dir"], "token_categories.json"),
    ]
    for path in candidates:
        if not os.path.exists(path):
            continue
        print(f"  Loading categories from: {path}")
        with open(path) as f:
            raw = json.load(f)
        # New format (05_fingerprint.py output):
        #   {"model": ..., "paradigm": ..., "per_cluster": [{"cluster": 0,
        #    "category": "SYNTACTIC", ...}, ...], "per_category": {...}}
        # Flatten to {str(cluster_id): category_name}.
        if isinstance(raw, dict) and "per_cluster" in raw:
            return {str(item["cluster"]): item["category"]
                    for item in raw["per_cluster"]
                    if "cluster" in item and "category" in item}
        # Legacy format: already a flat {str_id: category} mapping.
        if isinstance(raw, dict):
            return raw
    return {}


if not args.skip_taxonomy:
    print("\nToken taxonomy analysis")

    categories = load_token_categories()
    if not categories:
        print("Warning: No token categories loaded. Skipping taxonomy analysis.")
    else:
        print(f"Loaded {len(categories)} token category mappings")

        # Aggregate category distribution per clustering method
        taxonomy_results = {}

        # Spectral
        if os.path.exists(PATHS["clusters_output"]):
            with open(PATHS["clusters_output"], "rb") as f:
                clusters = pickle.load(f)

            max_k = max(clusters.keys())
            labels = np.array(clusters[max_k][0])

            category_counts = defaultdict(int)
            for label in labels:
                cat = categories.get(str(label), "UNKNOWN")
                category_counts[cat] += 1

            taxonomy_results["Spectral"] = dict(category_counts)
            print(f"Spectral (k={max_k}): {len(category_counts)} categories")

        # SSC-Lasso - pick the (d, α) with most active clusters at k=400
        if os.path.exists(PATHS["clusters_ssc_output"]):
            with open(PATHS["clusters_ssc_output"], "rb") as f:
                clusters_ssc = pickle.load(f)

            best_pair, best_n_active = None, -1
            for pair, entry in clusters_ssc.items():
                if not isinstance(entry, dict) or "clusters" not in entry:
                    continue
                lbl = entry["clusters"].get(400)
                if lbl is None:
                    continue
                n_active = len(np.unique(np.asarray(lbl)))
                if n_active > best_n_active:
                    best_n_active = n_active
                    best_pair = pair

            if best_pair is not None:
                result = clusters_ssc[best_pair]
                max_k = max(result["clusters"].keys())
                labels = np.array(result["clusters"][max_k])

                category_counts = defaultdict(int)
                for label in labels:
                    cat = categories.get(str(label), "UNKNOWN")
                    category_counts[cat] += 1

                taxonomy_results["SSC-Lasso"] = dict(category_counts)
                print(f"SSC-Lasso: {len(category_counts)} categories")

        # SSC-OMP - same selection rule as SSC-Lasso above
        omp_path = PATHS["clusters_ssc_output"].replace(".pkl", "_omp.pkl")
        if os.path.exists(omp_path):
            with open(omp_path, "rb") as f:
                clusters_omp = pickle.load(f)

            best_pair, best_n_active = None, -1
            for pair, entry in clusters_omp.items():
                if not isinstance(entry, dict) or "clusters" not in entry:
                    continue
                lbl = entry["clusters"].get(400)
                if lbl is None:
                    continue
                n_active = len(np.unique(np.asarray(lbl)))
                if n_active > best_n_active:
                    best_n_active = n_active
                    best_pair = pair

            if best_pair is not None:
                result = clusters_omp[best_pair]
                max_k = max(result["clusters"].keys())
                labels = np.array(result["clusters"][max_k])

                category_counts = defaultdict(int)
                for label in labels:
                    cat = categories.get(str(label), "UNKNOWN")
                    category_counts[cat] += 1

                taxonomy_results["SSC-OMP"] = dict(category_counts)
                print(f"SSC-OMP: {len(category_counts)} categories")

        # Hierarchical
        if os.path.exists(PATHS["clusters_hierarchical_output"]):
            with open(PATHS["clusters_hierarchical_output"], "rb") as f:
                clusters_hier = pickle.load(f)

            if isinstance(clusters_hier, dict) and "labels" in clusters_hier:
                labels_dict = clusters_hier["labels"]
                # Use largest k value available
                max_k = max(labels_dict.keys()) if labels_dict else None
                if max_k is not None:
                    labels = np.array(labels_dict[max_k])

                    category_counts = defaultdict(int)
                    for label in labels:
                        cat = categories.get(str(label), "UNKNOWN")
                        category_counts[cat] += 1

                    taxonomy_results["Hierarchical"] = dict(category_counts)
                    print(f"Hierarchical (k={max_k}): {len(category_counts)} categories")

        output_path = os.path.join(RESULTS_DIR, "taxonomy_analysis.json")
        with open(output_path, "w") as f:
            json.dump(taxonomy_results, f, indent=2)
        print(f"Saved: {output_path}")

if not args.skip_selection:
    print("\nL1 feature selection")

    # Load fingerprint matrix produced by 05_fingerprint.py.
    # Prefer SSC-Lasso fingerprint; fall back to spectral baseline.
    candidates = [
        os.path.join(RESULTS_DIR, "figures", "quanta_fingerprint_ssc-lasso", "fingerprint_matrix.npz"),
        os.path.join(RESULTS_DIR, "figures", "quanta_fingerprint_ssc_lasso", "fingerprint_matrix.npz"),
        os.path.join(RESULTS_DIR, "figures", "quanta_fingerprint", "fingerprint_matrix.npz"),
        os.path.join(RESULTS_DIR, "features.npz"),
    ]
    features_path = next((p for p in candidates if os.path.exists(p)), None)
    if features_path is None:
        print("Warning: fingerprint_matrix.npz not found. Skipping feature selection.")
    else:
        print(f"  Loading features from: {features_path}")
        features = np.load(features_path)
        # Standard fingerprint format uses "X"/"y"; fall back to common variants
        X = features["X"] if "X" in features.files else features[features.files[0]]
        y = features["y"] if "y" in features.files else features[features.files[1]]

        print(f"Features shape: {X.shape}, targets: {y.shape}")

        selection_results = {}

        for C_val in [0.1, 0.178, 0.316, 1.0]:
            try:
                lr = LogisticRegression(
                    penalty="l1",
                    solver="liblinear",
                    C=C_val,
                    max_iter=1000,
                    random_state=42
                )
                lr.fit(X, y)

                y_pred = lr.predict(X)
                bal_acc = balanced_accuracy_score(y, y_pred)
                mcc = matthews_corrcoef(y, y_pred)

                n_selected = np.sum(np.abs(lr.coef_[0]) > 1e-6)

                selection_results[f"C_{C_val}"] = {
                    "balanced_accuracy": float(bal_acc),
                    "mcc": float(mcc),
                    "n_selected": int(n_selected),
                    "n_features": int(X.shape[1]),
                }
                print(f"  C={C_val:6.3f}: bal_acc={bal_acc:.3f}, MCC={mcc:.3f}, "
                      f"selected={n_selected}/{X.shape[1]}")

            except Exception as e:
                print(f"  C={C_val}: Error - {e}")

        output_path = os.path.join(RESULTS_DIR, "l1_selection_results.json")
        with open(output_path, "w") as f:
            json.dump(selection_results, f, indent=2)
        print(f"Saved: {output_path}")

print("\nDone - Results saved to:", RESULTS_DIR)
