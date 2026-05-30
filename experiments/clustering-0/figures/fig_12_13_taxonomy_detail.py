"""Publication figures: Taxonomy detail and bootstrap CIs (fig_12_13_taxonomy_detail.py).

Generates two figures used in the thesis appendix:
  fig_12_taxonomy_and_recovery      - token category distribution + paper quanta recovery
  fig_13_taxonomy_bootstrap_ci      - bootstrap 95% CIs for top-6 token categories

Figures are written to {REPO_DIR}/figures/contribution-3/appendix/.

Prerequisites (run first):
  pipeline/09_quanta_taxonomy.py  for BOTH models

Usage:
  cd experiments/clustering-0
  python -u figures/fig_12_13_taxonomy_detail.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import warnings
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

warnings.filterwarnings("ignore")

from config import PATHS
from plot_style import (
    apply_style, savefig_both, MODEL_COLORS, normalize_category,
)

BASE_DIR = PATHS["base_dir"]
REPO_DIR = PATHS["repo_dir"]

RESULTS_19M = os.path.join(BASE_DIR, "results", "clustering-0")
RESULTS_125M = os.path.join(BASE_DIR, "results", "clustering-0-pythia-125m")

OUTPUT_DIR     = os.path.join(REPO_DIR, "figures", "contribution-3")
OUTPUT_DIR_APP = os.path.join(REPO_DIR, "figures", "contribution-3", "appendix")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR_APP, exist_ok=True)

apply_style()


def normalize_categories(cat_dict):
    out = Counter()
    for cat, val in cat_dict.items():
        out[normalize_category(cat)] += val
    return dict(out)


# Symbols that are unambiguously code-syntax (not natural-language punctuation)
# We deliberately exclude general punctuation like '.', ',', '!', '?', ':' which
# appear constantly in prose
_CODE_SYMBOLS = {
    ";", "{", "}", "(", ")", "[", "]",
    "</", "/>", "==", "!=", "=>", "->", "&&", "||", "::",
    "++", "--", "+=", "-=", "*=", "/=",
}
_CODE_KEYWORDS = {
    "def", "var", "let", "const", "import", "from", "class", "function",
    "return", "elif", "while", "switch",
    "true", "false", "null", "None", "True", "False",
}


def _is_code_token(t):
    s = (t[0] if isinstance(t, (list, tuple)) else str(t)).strip()
    return bool(s) and (s in _CODE_SYMBOLS or s in _CODE_KEYWORDS)


def count_paper_quanta(cluster_analysis):
    """Count clusters dedicated to the three paper-quanta examples
    (numerical, newline, code). For code we use a strict token-based
    heuristic because the taxonomy has no dedicated CODE label."""
    n_numerical = n_newline = n_code = 0
    for info in cluster_analysis.values():
        cat = info.get("category", "")
        top_tokens = info.get("top_tokens", []) or []
        token_strs = [t[0] if isinstance(t, (list, tuple)) else str(t)
                      for t in top_tokens]

        if cat in ("NUMERIC", "NUMERISK"):
            n_numerical += 1
        elif cat in ("FORMATTING", "FORMATERING"):
            if any("\\n" in t or "\n" in t for t in token_strs):
                n_newline += 1

        # Code detection: at least 2 of the top-5 tokens must be unambiguous
        # code symbols/keywords for a cluster to count as code-like
        if top_tokens:
            n_code_hits = sum(_is_code_token(t) for t in top_tokens[:5])
            if n_code_hits >= 2:
                n_code += 1
    return n_numerical, n_newline, n_code


print("Loading taxonomy data (SSC-Lasso)...")
taxonomy = {}
for model_name, rdir in [("Pythia-19m", RESULTS_19M), ("Pythia-125m", RESULTS_125M)]:
    # Prefer SSC-Lasso taxonomy; fall back to Spectral for backwards compat
    path_ssc = os.path.join(rdir, "quanta_taxonomy_ssc_lasso_k400.pkl")
    path_spectral = os.path.join(rdir, "quanta_taxonomy_spectral_k400.pkl")
    if os.path.exists(path_ssc):
        path = path_ssc
        print(f"  {model_name}: using SSC-Lasso taxonomy")
    elif os.path.exists(path_spectral):
        path = path_spectral
        print(f"  {model_name}: SSC-Lasso taxonomy not found, falling back to Spectral")
    else:
        print(f"  MISSING: {path_ssc}")
        sys.exit(1)
    with open(path, "rb") as f:
        taxonomy[model_name] = pickle.load(f)
    print(f"  {model_name}: loaded "
          f"({len(taxonomy[model_name]['cluster_analysis'])} clusters)")

# Figure 12 - Quanta taxonomy & paper quanta recovery (appendix)

print("\nGenerating Figure 12: Quanta Taxonomy and Recovery")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
sizes_19 = normalize_categories(taxonomy["Pythia-19m"]["category_sizes"])
sizes_125 = normalize_categories(taxonomy["Pythia-125m"]["category_sizes"])
all_cats = sorted(
    set(list(sizes_19.keys()) + list(sizes_125.keys())),
    key=lambda c: -(sizes_19.get(c, 0) + sizes_125.get(c, 0))
)
total_19 = sum(sizes_19.values())
total_125 = sum(sizes_125.values())
x = np.arange(len(all_cats))
width = 0.35
pcts_19 = [100 * sizes_19.get(c, 0) / total_19 for c in all_cats]
pcts_125 = [100 * sizes_125.get(c, 0) / total_125 for c in all_cats]
ax.bar(x - width / 2, pcts_19, width, label="Pythia-19m",
       color=MODEL_COLORS["Pythia-19m"], alpha=0.85)
ax.bar(x + width / 2, pcts_125, width, label="Pythia-125m",
       color=MODEL_COLORS["Pythia-125m"], alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(all_cats, rotation=35, ha="right")
ax.set_ylabel("Percentage of Tokens (%)")
ax.legend(fontsize=9)
ax.set_title("(a) Token category distribution ($k = 400$)")

ax = axes[1]
quanta_labels = ["Numerical\nSequences", "Newline\nPrediction", "Code\nPatterns"]
x_q = np.arange(len(quanta_labels))
width_q = 0.35
counts_19m = count_paper_quanta(taxonomy["Pythia-19m"]["cluster_analysis"])
counts_125m = count_paper_quanta(taxonomy["Pythia-125m"]["cluster_analysis"])
bars1 = ax.bar(x_q - width_q / 2, counts_19m, width_q, label="Pythia-19m",
               color=MODEL_COLORS["Pythia-19m"], alpha=0.85)
bars2 = ax.bar(x_q + width_q / 2, counts_125m, width_q, label="Pythia-125m",
               color=MODEL_COLORS["Pythia-125m"], alpha=0.85)
for bars in [bars1, bars2]:
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2., h + 0.15,
                    str(int(h)), ha="center", va="bottom", fontsize=10,
                    fontweight="bold")
ax.set_xticks(x_q)
ax.set_xticklabels(quanta_labels, fontsize=10)
ax.set_ylabel("Number of Dedicated Clusters")
ax.legend(fontsize=9)
ax.set_ylim(0, max(max(counts_19m), max(counts_125m)) + 3)
ax.set_title("(b) Paper-quanta recovery")

plt.suptitle("Quanta taxonomy and paper-quanta recovery",
             fontsize=13, y=1.00)
plt.tight_layout()
fig12_path = os.path.join(OUTPUT_DIR_APP, "fig_12_taxonomy_and_recovery.png")
savefig_both(fig12_path)
plt.close()
print(f"  Saved: {fig12_path}")


# Figure 13 - Bootstrap 95% CI for category percentages
#
# Note: panel (a) "Spectral cluster size distribution" was removed - it duplicated
# fig_01_rank_frequency. This is now a single-panel figure showing the
# uncertainty around fig_12's category percentages

print("Generating Figure 13: Bootstrap 95% CI for category proportions")
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

n_boot = 5000
rng = np.random.RandomState(42)
top_cats_raw = sorted(
    normalize_categories(taxonomy["Pythia-19m"]["category_sizes"]).items(),
    key=lambda x: -x[1]
)[:6]
top_cats = [c for c, _ in top_cats_raw]
for model_name, color in [("Pythia-19m", MODEL_COLORS["Pythia-19m"]),
                          ("Pythia-125m", MODEL_COLORS["Pythia-125m"])]:
    d = taxonomy[model_name]
    cluster_info = d["cluster_analysis"]
    cluster_cats, cluster_sizes = [], []
    for cid, info in cluster_info.items():
        raw_cat = info.get("category", "UNKNOWN")
        cat = normalize_category(raw_cat)
        cluster_cats.append(cat)
        cluster_sizes.append(info.get("size", 0))
    cluster_cats = np.array(cluster_cats)
    cluster_sizes = np.array(cluster_sizes, dtype=float)
    boot_pcts = {c: [] for c in top_cats}
    for _ in range(n_boot):
        idxs = rng.choice(len(cluster_cats), size=len(cluster_cats), replace=True)
        b_cats = cluster_cats[idxs]
        b_sizes = cluster_sizes[idxs]
        b_total = b_sizes.sum()
        for cat in top_cats:
            mask = b_cats == cat
            pct = 100 * b_sizes[mask].sum() / b_total if b_total > 0 else 0
            boot_pcts[cat].append(pct)
    offset = 0.15 if model_name == "Pythia-19m" else -0.15
    for i, cat in enumerate(top_cats):
        vals = np.array(boot_pcts[cat])
        ci_low, ci_high = np.percentile(vals, [2.5, 97.5])
        mean_val = vals.mean()
        ax.errorbar(mean_val, i + offset,
                    xerr=[[mean_val - ci_low], [ci_high - mean_val]],
                    fmt="o", color=color, capsize=4, markersize=6, capthick=1.5)
ax.set_yticks(np.arange(len(top_cats)))
ax.set_yticklabels(top_cats)
ax.set_xlabel("Percentage of Tokens (%)")
ax.invert_yaxis()
legend_elements = [
    Line2D([0], [0], marker="o", color=MODEL_COLORS["Pythia-19m"],
           label="Pythia-19m", markersize=6, linestyle="None"),
    Line2D([0], [0], marker="o", color=MODEL_COLORS["Pythia-125m"],
           label="Pythia-125m", markersize=6, linestyle="None"),
]
ax.legend(handles=legend_elements, fontsize=9)
sizes_19_norm = normalize_categories(taxonomy["Pythia-19m"]["category_sizes"])
sizes_125_norm = normalize_categories(taxonomy["Pythia-125m"]["category_sizes"])
pcts_a = [100 * sizes_19_norm.get(c, 0) / total_19 for c in top_cats]
pcts_b = [100 * sizes_125_norm.get(c, 0) / total_125 for c in top_cats]
r, p = stats.pearsonr(pcts_a, pcts_b)
ax.set_title(f"Bootstrap 95% CI for top-6 categories\n"
             f"($n = {n_boot}$ resamples, cross-model $r = {r:.3f}$)")

plt.tight_layout()
fig13_path = os.path.join(OUTPUT_DIR_APP, "fig_13_taxonomy_bootstrap_ci.png")
savefig_both(fig13_path)
plt.close()
print(f"  Saved: {fig13_path}")

print(f"\nPaper quanta recovery:")
print(f"  Pythia-19m:  numerical={counts_19m[0]}, newline={counts_19m[1]}, code={counts_19m[2]}")
print(f"  Pythia-125m: numerical={counts_125m[0]}, newline={counts_125m[1]}, code={counts_125m[2]}")
print(f"\nCross-model taxonomy correlation (k=400):")
print(f"  r = {r:.3f}, p = {p:.2e}")
print(f"\nFigures saved to: {OUTPUT_DIR_APP}")
