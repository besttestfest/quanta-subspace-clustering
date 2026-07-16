"""Publication figures: Bootstrap stability.

Generates one figure used in the paper appendix:
  fig_11_bootstrap_stability  - NMI/ARI distributions across 50 bootstrap
                             iterations for both Pythia-19M and 125M

Figure is written to {REPO_DIR}/figures/contribution-2/appendix/.

Data source:
  results-mirror/bootstrap_stability/pythia-{19m,125m}_ssc_lasso.json
  (canonical n=50 iteration files committed to the repo)

Usage:
  cd experiments/clustering-0
  python -u figures/fig_11_stability.py
"""

import os
import sys
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PATHS
from plot_style import apply_style, savefig_both, MODEL_COLORS

apply_style()

REPO_DIR = PATHS["repo_dir"]
OUTPUT_DIR = os.path.join(REPO_DIR, "figures", "contribution-2", "appendix")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Canonical bootstrap data committed to the repo (results-mirror).
# These are the verified n=50 iteration files for both models.
BOOT_19M  = os.path.join(REPO_DIR, "results-mirror", "bootstrap_stability",
                         "pythia-19m_ssc_lasso.json")
BOOT_125M = os.path.join(REPO_DIR, "results-mirror", "bootstrap_stability",
                         "pythia-125m_ssc_lasso.json")
def load_json(path, label):
    if not os.path.exists(path):
        print(f"  WARNING: {label} not found at {path}")
        return None
    with open(path) as f:
        return json.load(f)


print("Generating fig_11_bootstrap_stability ...")

boot_19m  = load_json(BOOT_19M,  "bootstrap 19m")
boot_125m = load_json(BOOT_125M, "bootstrap 125m")

COLORS = MODEL_COLORS  # shared palette: "Pythia-19m"/"Pythia-125m" -> hex
MODELS = [("Pythia-19m", boot_19m), ("Pythia-125m", boot_125m)]

fig, (ax_nmi, ax_ari) = plt.subplots(1, 2, figsize=(9, 5))

for ax, metric, ylabel, panel_label in [
    (ax_nmi, "nmi", "NMI", "(a) NMI - 50 bootstrap iterations"),
    (ax_ari, "ari", "ARI", "(b) ARI - 50 bootstrap iterations"),
]:
    for pos, (name, boot) in enumerate(MODELS, 1):
        if boot is None:
            continue
        vals = [it[metric] for it in boot["iterations"] if metric in it]
        color = COLORS[name]

        vp = ax.violinplot([vals], positions=[pos], widths=0.6,
                           showmeans=False, showmedians=False, showextrema=False)
        for body in vp["bodies"]:
            body.set_facecolor(color)
            body.set_alpha(0.45)
            body.set_edgecolor(color)

        m, s = np.mean(vals), np.std(vals)
        ax.plot([pos], [m], "o", color=color, markersize=6, zorder=4)
        ax.plot([pos, pos], [m - s, m + s], "-", color=color, lw=2, zorder=3)
        ax.plot([pos - 0.12, pos + 0.12], [m, m], "-", color=color, lw=2.5, zorder=3)

        rng = np.random.default_rng(42)
        jitter = rng.uniform(-0.10, 0.10, len(vals))
        ax.scatter(pos + jitter, vals, color=color, alpha=0.35, s=10, zorder=2)

        ax.text(pos + 0.35, m,
                f"{m:.3f}±{s:.3f}", ha="left", va="center", fontsize=8.5,
                color=color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                          edgecolor=color, linewidth=0.8, alpha=1.0))

    ax.set_xticks(range(1, len(MODELS) + 1))
    ax.set_xticklabels([n for n, _ in MODELS], fontsize=9)  # "Pythia-19m", "Pythia-125m"
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(panel_label, fontsize=10, pad=6)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.set_xlim(0.4, len(MODELS) + 0.6)

# Footnote - configs are read from the data files so the caption always
# matches what was actually run.
def _cfg_str(boot):
    cfg = (boot or {}).get("config", {})
    if cfg:
        return f"d={cfg.get('d')}, α={cfg.get('alpha')}, k={cfg.get('k', 400)}"
    return "k=400"

cfg_19m, cfg_125m = _cfg_str(boot_19m), _cfg_str(boot_125m)
if cfg_19m == cfg_125m:
    cfg_line = f"SSC-Lasso ({cfg_19m})"
else:
    cfg_line = f"SSC-Lasso (19M: {cfg_19m}; 125M: {cfg_125m})"

fig.text(0.5, -0.04,
         f"{cfg_line}, 80% token subsample per iteration.\n"
         "Each dot is one bootstrap run; crosshair = mean ± 1 std.",
         ha="center", va="top", fontsize=8, color="#444")

plt.suptitle("SSC-Lasso bootstrap stability (50 iterations)",
             fontsize=12, y=1.02)
plt.tight_layout()

out_path = os.path.join(OUTPUT_DIR, "fig_11_bootstrap_stability.png")
savefig_both(out_path)
plt.close()
print(f"  Saved: {out_path}")

print(f"\nOutputs in: {OUTPUT_DIR}")
