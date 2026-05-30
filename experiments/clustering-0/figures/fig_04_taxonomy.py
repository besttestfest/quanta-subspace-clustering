"""Taxonomy comparison figure: Spectral vs SSC-Lasso vs Hierarchical cluster categories.

Generates a side-by-side grouped bar chart comparing the token-category
distribution of Spectral, SSC-Lasso, and Hierarchical clusters for both
Pythia-19M and Pythia-125M.  Illustrates that paradigm choice shapes what
a quantum *is*, not just how well the envelope fits.

Output:
    figures/contribution-3/fig_04_taxonomy_comparison.pdf/.png

Usage (UCloud):
    cd experiments/clustering-0
    python figures/fig_04_taxonomy.py
"""

import os
import sys
import pickle
import warnings
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PATHS
from plot_style import apply_style, savefig_both, METHOD_COLORS, normalize_category

BASE_DIR = PATHS["base_dir"]
REPO_DIR = PATHS["repo_dir"]

RESULTS_19M  = os.path.join(BASE_DIR, "results", "clustering-0")
RESULTS_125M = os.path.join(BASE_DIR, "results", "clustering-0-pythia-125m")
OUTPUT_DIR   = os.path.join(REPO_DIR, "figures", "contribution-3")
os.makedirs(OUTPUT_DIR, exist_ok=True)

apply_style()

METHODS = ("Spectral", "SSC-Lasso", "Hierarchical")
MODELS  = ("Pythia-19M", "Pythia-125M")

# Load taxonomy files
def load_taxonomy(rdir, method):
    key  = method.lower().replace("-", "_")
    fname = f"quanta_taxonomy_{key}_k400.pkl"
    path  = os.path.join(rdir, fname)
    if not os.path.exists(path):
        print(f"  MISSING: {path}")
        sys.exit(1)
    with open(path, "rb") as f:
        return pickle.load(f)

def category_pcts(taxonomy):
    """Return {normalised_category: pct} from taxonomy dict."""
    raw = taxonomy.get("category_sizes", {})
    counts = Counter()
    for cat, n in raw.items():
        counts[normalize_category(cat)] += n
    total = sum(counts.values())
    return {cat: 100 * n / total for cat, n in counts.items()} if total else {}

print("Loading taxonomies...")
RESULTS = {"Pythia-19M": RESULTS_19M, "Pythia-125M": RESULTS_125M}
tax = {method: {model: load_taxonomy(RESULTS[model], method)
                for model in MODELS}
       for method in METHODS}

# Compute percentages
pcts = {method: {model: category_pcts(tax[method][model])
                 for model in MODELS}
        for method in METHODS}

# Union of all categories, sorted by SSC-Lasso Pythia-19M descending
all_cats_raw = set()
for m in pcts.values():
    for p in m.values():
        all_cats_raw.update(p.keys())

ref = pcts["SSC-Lasso"]["Pythia-19M"]
all_cats = sorted(all_cats_raw, key=lambda c: -ref.get(c, 0))

# Plot
fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=False)

n_methods = len(METHODS)
total_width = 0.7
width = total_width / n_methods

for ax, model in zip(axes, MODELS):
    x = np.arange(len(all_cats))

    for i, method in enumerate(METHODS):
        vals   = [pcts[method][model].get(c, 0) for c in all_cats]
        offset = -total_width / 2 + (i + 0.5) * width
        ax.bar(x + offset, vals, width,
               label=method,
               color=METHOD_COLORS[method],
               alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(all_cats, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Percentage of Tokens (%)")
    ax.set_title(f"{model}", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 100)

plt.suptitle(
    "Cluster taxonomy: Spectral vs. SSC-Lasso vs. Hierarchical ($k = 400$)",
    fontsize=12, y=1.01,
)
plt.tight_layout()

out_path = os.path.join(OUTPUT_DIR, "fig_04_taxonomy_comparison.png")
savefig_both(out_path)
plt.close()
print(f"Saved: {out_path}")
print("Done.")
