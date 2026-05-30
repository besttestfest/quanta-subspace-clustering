"""Generate results-mirror/taxonomy/ JSON summary files from quanta_taxonomy pkl files.

Reads quanta_taxonomy_{method}_k400.pkl for both models and writes canonical
summary JSONs to results-mirror/taxonomy/.  Run this AFTER pipeline/09_quanta_taxonomy.py
has been run for both models so all .pkl files are up-to-date.

Usage (from experiments/clustering-0/):
    python pipeline/10_taxonomy_mirror.py

Produces (relative to repo root):
    results-mirror/taxonomy/pythia-19m_{spectral,ssc_lasso,hierarchical}.json
    results-mirror/taxonomy/pythia-125m_{spectral,ssc_lasso,hierarchical}.json
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pickle
from collections import Counter

import numpy as np
from scipy.stats import pearsonr

from config import PATHS

REPO_DIR   = PATHS["repo_dir"]
BASE_DIR   = PATHS["base_dir"]
MIRROR_DIR = os.path.join(REPO_DIR, "results-mirror", "taxonomy")
os.makedirs(MIRROR_DIR, exist_ok=True)

# Results live under BASE_DIR/results/, not inside the repo.
# pythia-19m  -> clustering-0
# pythia-125m -> clustering-0-pythia-125m
RESULTS_19M  = os.path.join(BASE_DIR, "results", "clustering-0")
RESULTS_125M = os.path.join(BASE_DIR, "results", "clustering-0-pythia-125m")

for label, path in [("19m", RESULTS_19M), ("125m", RESULTS_125M)]:
    if not os.path.isdir(path):
        sys.exit(f"Results directory for {label} not found: {path}")

print(f"19m results:  {RESULTS_19M}")
print(f"125m results: {RESULTS_125M}")

METHODS = {
    "spectral":    "spectral",
    "ssc_lasso":   "ssc_lasso",
    "hierarchical": "hierarchical",
}

CONTENT_BEARING_CATS = {"CONTENT", "PROPER_NOUNS"}


def load_taxonomy(results_dir, method):
    pkl_path = os.path.join(results_dir, f"quanta_taxonomy_{method}_k400.pkl")
    if not os.path.exists(pkl_path):
        print(f"  MISSING: {pkl_path}")
        return None
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def summarise(taxonomy):
    ca = taxonomy["cluster_analysis"]
    total_clusters = len(ca)
    total_tokens   = sum(v["size"] for v in ca.values())

    cat_counts = Counter(v["category"] for v in ca.values())
    cat_sizes  = {}
    for v in ca.values():
        cat = v["category"]
        cat_sizes[cat] = cat_sizes.get(cat, 0) + v["size"]

    cat_cluster_pct = {c: round(n / total_clusters * 100, 2)
                       for c, n in cat_counts.items()}
    cat_token_pct   = {c: round(s / total_tokens * 100, 2)
                       for c, s in cat_sizes.items()}

    content_bearing = sum(cat_counts.get(c, 0) for c in CONTENT_BEARING_CATS)
    content_bearing_pct = round(content_bearing / total_clusters * 100, 1)

    return {
        "total_clusters":     total_clusters,
        "total_tokens":       total_tokens,
        "category_counts":    dict(cat_counts),
        "category_sizes":     cat_sizes,
        "category_cluster_pct": cat_cluster_pct,
        "category_token_pct": cat_token_pct,
        "content_bearing_pct": content_bearing_pct,
        "repetitive_pct":     0.0,
    }


def cross_model_pearson(summary_19m, summary_125m):
    all_cats = sorted(
        set(summary_19m["category_token_pct"]) | set(summary_125m["category_token_pct"])
    )
    v19  = np.array([summary_19m["category_token_pct"].get(c, 0.0)  for c in all_cats])
    v125 = np.array([summary_125m["category_token_pct"].get(c, 0.0) for c in all_cats])

    all_cats_cl = sorted(
        set(summary_19m["category_cluster_pct"]) | set(summary_125m["category_cluster_pct"])
    )
    c19  = np.array([summary_19m["category_cluster_pct"].get(c, 0.0)  for c in all_cats_cl])
    c125 = np.array([summary_125m["category_cluster_pct"].get(c, 0.0) for c in all_cats_cl])

    r_tok, _  = pearsonr(v19, v125)
    r_clust, _ = pearsonr(c19, c125)
    return round(float(r_clust), 4), round(float(r_tok), 4)


for method_key, method_label in METHODS.items():
    print(f"\n[{method_key}]")

    t19m  = load_taxonomy(RESULTS_19M,  method_key)
    t125m = load_taxonomy(RESULTS_125M, method_key)

    if t19m is None or t125m is None:
        print(f"  Skipping {method_key} - missing pkl file(s).")
        continue

    s19m  = summarise(t19m)
    s125m = summarise(t125m)

    r_cluster, r_token = cross_model_pearson(s19m, s125m)
    print(f"  cross_model r (cluster): {r_cluster:.4f}   r (token): {r_token:.4f}")

    for model_name, summary, taxonomy in [
        ("pythia-19m",  s19m,  t19m),
        ("pythia-125m", s125m, t125m),
    ]:
        out = {
            "model":  model_name,
            "method": method_label,
            "k": 400,
            "total_clusters": summary["total_clusters"],
            "total_tokens":   summary["total_tokens"],
            "category_counts":     summary["category_counts"],
            "category_sizes":      summary["category_sizes"],
            "category_cluster_pct": summary["category_cluster_pct"],
            "category_token_pct":  summary["category_token_pct"],
            "content_bearing_pct": summary["content_bearing_pct"],
            "repetitive_pct":      summary["repetitive_pct"],
            "cross_model_pearson_r_cluster_share": r_cluster,
            "cross_model_pearson_r_token_share":   r_token,
            "cross_model_pearson_r": r_token,
        }
        fname = f"{model_name}_{method_key}.json"
        out_path = os.path.join(MIRROR_DIR, fname)
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  Wrote: {out_path}")

print("\nDone. Verify and commit results-mirror/taxonomy/.")
