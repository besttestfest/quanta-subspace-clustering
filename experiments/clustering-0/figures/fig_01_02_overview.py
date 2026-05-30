"""Publication figures: Rank-frequency and similarity structure (fig_01_02_overview.py).

Generates two figures used in the thesis main body:
  fig_01_rank_frequency      - rank-frequency envelope (Pythia-19M and Pythia-125M)
  fig_02_similarity_structure - gradient-similarity matrix and cross-model structure

Figures are written to {REPO_DIR}/figures/contribution-1/.

Run once (uses both models automatically):
  cd experiments/clustering-0
  python -u figures/fig_01_02_overview.py
"""
import os
import sys
import pickle
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PATHS
from plot_style import (
    apply_style, savefig_both, MODEL_COLORS, PAPER_SLOPE, EXPECTED_SLOPE,
)

BASE_DIR = PATHS["base_dir"]
REPO_DIR = PATHS["repo_dir"]

RESULTS_19M = os.path.join(BASE_DIR, "results", "clustering-0")
RESULTS_125M = os.path.join(BASE_DIR, "results", "clustering-0-pythia-125m")

OUTPUT_DIR = os.path.join(REPO_DIR, "figures", "contribution-1")
os.makedirs(OUTPUT_DIR, exist_ok=True)

apply_style()


def compute_envelope(cluster_dict):
    """Compute rank-frequency envelope across all k values."""
    all_sizes = {}
    for k, labels_or_tuple in sorted(cluster_dict.items()):
        if isinstance(labels_or_tuple, tuple):
            labels = np.array(labels_or_tuple[0])
        else:
            labels = np.array(labels_or_tuple)
        counts = sorted(Counter(labels).values(), reverse=True)
        for rank, size in enumerate(counts, 1):
            if rank not in all_sizes or size > all_sizes[rank]:
                all_sizes[rank] = size
    ranks = sorted(all_sizes.keys())
    sizes = [all_sizes[r] for r in ranks]
    return np.array(ranks), np.array(sizes)


def fit_envelope_slope(ranks, sizes, rank_min=100, rank_max=1000):
    """Log-log linear fit on envelope (ranks 100-1000, as in paper)."""
    mask = (ranks >= rank_min) & (ranks <= rank_max) & (sizes > 0)
    if mask.sum() < 3:
        return None, None, None
    log_r = np.log10(ranks[mask].astype(float))
    log_s = np.log10(sizes[mask].astype(float))
    slope, intercept = np.polyfit(log_r, log_s, 1)
    return slope, intercept, 10**intercept


print("Loading spectral clustering results...")
spectral = {}
for model, rdir in [("Pythia-19m", RESULTS_19M), ("Pythia-125m", RESULTS_125M)]:
    path = os.path.join(rdir, "clusters_full_more.pkl")
    if not os.path.exists(path):
        print(f"  MISSING: {path}")
        print(f"  Run the pipeline for {model} first.")
        sys.exit(1)
    with open(path, "rb") as f:
        spectral[model] = pickle.load(f)
    print(f"  {model}: {len(spectral[model])} k-values")

sim_path_19m = os.path.join(RESULTS_19M, "full_more.pt")
idxs, C_raw, C_abs_raw = torch.load(sim_path_19m, map_location="cpu", weights_only=False)
C = C_raw.numpy().astype(np.float64)
C_angular = 1 - np.arccos(np.clip(C, -1.0, 1.0)) / np.pi


# Figure 1: Rank-Frequency Replication

print("Generating Figure 1: Rank-Frequency Replication")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, (model, rdir) in zip(axes, [("Pythia-19m", RESULTS_19M), ("Pythia-125m", RESULTS_125M)]):
    results = spectral[model]
    ks = sorted(results.keys())
    norm = Normalize(vmin=min(ks), vmax=max(ks))
    cmap = plt.cm.viridis

    all_sizes = {}
    for k in ks:
        if isinstance(results[k], tuple):
            labels = np.array(results[k][0])
        else:
            labels = np.array(results[k])
        counts = sorted(Counter(labels).values(), reverse=True)
        ranks = np.arange(1, len(counts) + 1)
        ax.plot(ranks, counts, "-", color=cmap(norm(k)), alpha=0.6, linewidth=0.8)
        for rank, size in zip(ranks, counts):
            if rank not in all_sizes or size > all_sizes[rank]:
                all_sizes[rank] = size

    env_ranks = np.array(sorted(all_sizes.keys()))
    env_sizes = np.array([all_sizes[r] for r in env_ranks])
    ax.plot(env_ranks, env_sizes, "-", color="gray", linewidth=1.0, alpha=0.4, label="Envelope")

    slope, intercept, anchor = fit_envelope_slope(env_ranks, env_sizes)

    if slope is not None:
        r_line = np.logspace(0, 3.2, 100)

        # Reference lines: paper and expected drawn in black (consistent with
        # fig_02b, fig_03, fig_03b); measured line in a distinct neutral
        # colour that does not collide with the model-comparison palette
        s_ours = anchor * r_line**slope
        ax.plot(r_line, s_ours, "-", color="#cc0000", linewidth=2.5, alpha=0.85,
                label=f"Measured: $\\hat{{\\beta}} = {slope:.3f}$")

        s_paper = anchor * r_line**PAPER_SLOPE
        ax.plot(r_line, s_paper, "--", color="black", linewidth=2.0, alpha=0.85,
                label=f"Michaud et al.: $\\beta^* = {PAPER_SLOPE}$")

        s_expected = anchor * r_line**EXPECTED_SLOPE
        ax.plot(r_line, s_expected, ":", color="black", linewidth=1.5, alpha=0.5,
                label=f"Expected: $\\beta^* = {EXPECTED_SLOPE}$")

        delta = abs(slope - PAPER_SLOPE)
        textstr = (f"$\\hat{{\\beta}} = {slope:.3f}$\n"
                   f"$\\beta^* = {PAPER_SLOPE}$\n"
                   f"$|\\Delta| = {delta:.3f}$")
        props = dict(boxstyle="round,pad=0.35", facecolor="white",
                     alpha=0.9, edgecolor="lightgray")
        ax.text(0.97, 0.97, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment="top", horizontalalignment="right",
                bbox=props)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Number of clusters ($k$)", shrink=0.8)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Cluster rank")
    ax.set_ylabel("Cluster size")
    ax.set_xlim(1, 2000)
    ax.set_ylim(0.8, 15000)
    ax.legend(loc="lower left", fontsize=8)

    panel = "(a)" if model == "Pythia-19m" else "(b)"
    ax.set_title(f"{panel} {model}")

plt.suptitle("Rank-frequency distribution of gradient clusters",
             fontsize=13, y=1.00)
plt.tight_layout()

fig1_path = os.path.join(OUTPUT_DIR, "fig_01_rank_frequency.png")
savefig_both(fig1_path)
plt.close()
print(f"  Saved: {fig1_path}")


# Figure 2: Gradient Similarity Structure

print("Generating Figure 2: Gradient Similarity Structure")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
labels_400 = np.array(spectral["Pythia-19m"][400][0]) if isinstance(spectral["Pythia-19m"][400], tuple) else np.array(spectral["Pythia-19m"][400])
sort_idx = np.argsort(labels_400)
subset = sort_idx[:500]
C_sub = C_angular[np.ix_(subset, subset)]

im = ax.imshow(C_sub, cmap="magma", aspect="equal",
               vmin=C_angular.min(), vmax=1.0)
cbar = plt.colorbar(im, ax=ax, label="Angular similarity", shrink=0.8)
ax.set_xlabel("Token index (reordered by cluster)")
ax.set_ylabel("Token index (reordered by cluster)")
ax.set_title("(a) Gradient similarity matrix (Pythia-19m, $k = 400$)")

ax = axes[1]
for model, color in MODEL_COLORS.items():
    env_ranks, env_sizes = compute_envelope(spectral[model])
    slope, intercept, anchor = fit_envelope_slope(env_ranks, env_sizes)
    ax.plot(env_ranks, env_sizes, "-", color=color, linewidth=1.8, alpha=0.9,
            label=f"{model} (slope = {slope:.3f})")

r_line = np.logspace(0, 3.2, 100)
env_r_19m, env_s_19m = compute_envelope(spectral["Pythia-19m"])
_, _, anchor_19m = fit_envelope_slope(env_r_19m, env_s_19m)
s_paper = anchor_19m * r_line**PAPER_SLOPE
ax.plot(r_line, s_paper, "k:", linewidth=2.0, alpha=0.6,
        label=f"Michaud et al. ($\\beta^* = {PAPER_SLOPE}$)")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Cluster rank")
ax.set_ylabel("Cluster size (envelope)")
ax.set_xlim(1, 2000)
ax.set_ylim(0.8, 15000)
ax.legend(loc="lower left", fontsize=9)
ax.set_title("(b) Cross-model envelope comparison")

plt.suptitle("Gradient cluster structure",
             fontsize=13, y=1.00)
plt.tight_layout()

fig2_path = os.path.join(OUTPUT_DIR, "fig_02_similarity_structure.png")
savefig_both(fig2_path)
plt.close()
print(f"  Saved: {fig2_path}")

print(f"\nFigures saved to: {OUTPUT_DIR}")
