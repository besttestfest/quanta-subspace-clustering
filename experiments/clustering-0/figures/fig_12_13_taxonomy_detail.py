"""Publication figure: Taxonomy detail (fig_12_13_taxonomy_detail.py).

Generates the taxonomy appendix figure:
  fig_12_taxonomy_and_recovery      - token category distribution + paper quanta recovery

The figure is written to {REPO_DIR}/figures/contribution-3/appendix/.

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

# Quanta taxonomy & paper quanta recovery (appendix figure)

print("\nGenerating taxonomy figure: Quanta Taxonomy and Recovery")
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

print(f"\nPaper quanta recovery:")
print(f"  Pythia-19m:  numerical={counts_19m[0]}, newline={counts_19m[1]}, code={counts_19m[2]}")
print(f"  Pythia-125m: numerical={counts_125m[0]}, newline={counts_125m[1]}, code={counts_125m[2]}")
print(f"\nFigure saved to: {OUTPUT_DIR_APP}")
