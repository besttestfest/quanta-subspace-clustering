"""Publication figures: Methods comparison (fig_05_10_methods.py).

Generates six figures used in the thesis appendix:
  fig_05_method_overlay_19m      - envelope overlay + |Δ| bar chart (Pythia-19M)
  fig_06_method_overlay_125m     - envelope overlay + |Δ| bar chart (Pythia-125M)
  fig_07_ssc_evaluation_19m      - SSC-Lasso hyperparameter heatmap (Pythia-19M)
  fig_08_method_comparison_19m   - per-method rank-frequency 2x2 grid (Pythia-19M)
  fig_09_method_comparison_125m  - per-method rank-frequency 2x2 grid (Pythia-125M)
  fig_10_ssc_evaluation_125m     - SSC-Lasso hyperparameter heatmap (Pythia-125M)

Figures are written to {REPO_DIR}/figures/contribution-2/appendix/.

Run once per model (set QDG_MODEL env var):
  cd experiments/clustering-0
  python -u figures/fig_05_10_methods.py                        # Pythia-19M
  QDG_MODEL=pythia-125m python -u figures/fig_05_10_methods.py  # Pythia-125M
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

from config import PATHS, MODEL_NAME
from plot_style import apply_style, savefig_both, METHOD_COLORS, PAPER_SLOPE

BASE_DIR = PATHS["base_dir"]
REPO_DIR = PATHS["repo_dir"]
RESULTS_DIR = PATHS["results_dir"]

OUTPUT_DIR     = os.path.join(REPO_DIR, "figures", "contribution-2")
OUTPUT_DIR_APP = os.path.join(REPO_DIR, "figures", "contribution-2", "appendix")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR_APP, exist_ok=True)

MODEL_LABEL  = MODEL_NAME.replace("pythia-", "Pythia-")
MODEL_SUFFIX = MODEL_NAME.replace("pythia-", "")
IS_125M      = MODEL_NAME == "pythia-125m"

METHODS_KEEP = ["Spectral", "SSC-Lasso", "SSC-OMP", "Hierarchical"]

apply_style()


def load_pickle(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def labels_to_rank_freq(labels, drop_noise=True):
    labels = np.asarray(labels)
    if drop_noise:
        labels = labels[labels != -1]
    if len(labels) == 0:
        return np.array([]), np.array([])
    counts = sorted(Counter(labels).values(), reverse=True)
    ranks = np.arange(1, len(counts) + 1)
    return ranks, np.array(counts)


def compute_envelope_slope(all_rank_freq, fit_range=(100, 1000)):
    envelope = {}
    for ranks, sizes in all_rank_freq:
        for r, s in zip(ranks, sizes):
            if r not in envelope or s > envelope[r]:
                envelope[r] = s
    if not envelope:
        return None
    env_ranks = np.array(sorted(envelope.keys()))
    env_sizes = np.array([envelope[r] for r in env_ranks])
    mask = (env_ranks >= fit_range[0]) & (env_ranks <= fit_range[1]) & (env_sizes > 0)
    if mask.sum() < 3:
        return None
    log_r = np.log10(env_ranks[mask].astype(float))
    log_s = np.log10(env_sizes[mask].astype(float))
    slope, _ = np.polyfit(log_r, log_s, 1)
    return float(slope)


def envelope_arrays(all_rank_freq):
    envelope = {}
    for ranks, sizes in all_rank_freq:
        for r, s in zip(ranks, sizes):
            if r not in envelope or s > envelope[r]:
                envelope[r] = s
    if not envelope:
        return np.array([]), np.array([])
    env_ranks = np.array(sorted(envelope.keys()))
    env_sizes = np.array([envelope[r] for r in env_ranks])
    return env_ranks, env_sizes


def fit_intercept(env_ranks, env_sizes, fit_range=(100, 1000)):
    mask = (env_ranks >= fit_range[0]) & (env_ranks <= fit_range[1]) & (env_sizes > 0)
    if mask.sum() < 3:
        return None, None
    log_r = np.log10(env_ranks[mask].astype(float))
    log_s = np.log10(env_sizes[mask].astype(float))
    slope, intercept = np.polyfit(log_r, log_s, 1)
    return float(slope), float(intercept)


def load_spectral(res_dir):
    pkl = load_pickle(os.path.join(res_dir, "clusters_full_more.pkl"))
    if pkl is None:
        return None
    rfs = []
    primary = None
    for k, v in pkl.items():
        labels = np.array(v[0] if isinstance(v, tuple) else v)
        rf = labels_to_rank_freq(labels, drop_noise=False)
        if len(rf[0]) > 0:
            rfs.append(rf)
        if k == 400:
            primary = labels
    slope = compute_envelope_slope(rfs)
    return {"rfs": rfs, "slope": slope,
            "config": "envelope over 30 k-values (k=400 highlighted)",
            "labels": primary}


def load_ssc(res_dir, kind="lasso"):
    fname = ("clusters_ssc_full_more.pkl" if kind == "lasso"
             else "clusters_ssc_full_more_omp.pkl")
    pkl = load_pickle(os.path.join(res_dir, fname))
    if pkl is None:
        return None

    per_pair = {}
    for key, entry in pkl.items():
        if not (isinstance(entry, dict) and "clusters" in entry):
            continue
        clusters_dict = entry["clusters"]
        if not isinstance(clusters_dict, dict):
            continue
        for k, labels in clusters_dict.items():
            labels = np.array(labels)
            if labels.ndim == 0:
                continue
            if len(np.unique(labels)) < 5:
                continue
            per_pair.setdefault(key, []).append((k, labels))

    if not per_pair:
        return None

    per_pair_env = {}
    for pair, items in per_pair.items():
        rfs = [labels_to_rank_freq(lbl, drop_noise=False) for _, lbl in items]
        s = compute_envelope_slope(rfs)
        if s is not None:
            per_pair_env[pair] = {"slope": s, "rfs": rfs, "items": items,
                                  "delta": abs(s - PAPER_SLOPE)}

    if not per_pair_env:
        return None

    best_pair = min(per_pair_env.keys(), key=lambda p: per_pair_env[p]["delta"])
    best = per_pair_env[best_pair]
    primary = None
    for k, lbl in best["items"]:
        if k == 400:
            primary = lbl
            break
    if primary is None:
        best_n = 0
        for k, lbl in best["items"]:
            n = len(np.unique(lbl))
            if n > best_n:
                best_n = n
                primary = lbl

    if kind == "lasso":
        cfg = f"best (d,α)={best_pair} (k-sweep={len(best['rfs'])})"
    else:
        cfg = f"best (d,K)={best_pair} (k-sweep={len(best['rfs'])})"

    return {"rfs": best["rfs"], "slope": best["slope"],
            "config": cfg, "labels": primary}


def load_hierarchical(res_dir):
    # Primary path: alternative clustering results (full k-sweep)
    pkl = load_pickle(os.path.join(
        res_dir, "clusters_alternative", "alternative_clustering_results.pkl"))
    # Fallback: canonical hierarchical output from pipeline/02_clustering.py
    if pkl is None:
        fallback = os.path.join(res_dir, "clusters_hierarchical.pkl")
        pkl_raw = load_pickle(fallback)
        if pkl_raw is None:
            return None
        # Wrap in the expected format: {"hierarchical_complete": pkl_raw}
        pkl = {"hierarchical_complete": pkl_raw}
    if pkl is None:
        return None
    per_link = {}
    for hk in [k for k in pkl.keys()
               if isinstance(k, str) and k.startswith("hierarchical_")]:
        entry = pkl[hk]
        if not isinstance(entry, dict):
            continue
        labels_dict = entry.get("labels") or {}
        rfs, k400 = [], None
        for k, lbl in labels_dict.items():
            if lbl is None:
                continue
            lbl = np.asarray(lbl)
            if lbl.ndim == 0:
                continue
            if len(np.unique(lbl)) < 5:
                continue
            rfs.append(labels_to_rank_freq(lbl, drop_noise=False))
            if k == 400:
                k400 = lbl
        if not rfs:
            continue
        s = compute_envelope_slope(rfs)
        if s is None:
            continue
        link = hk.replace("hierarchical_", "")
        per_link[link] = {"slope": s, "rfs": rfs, "labels_k400": k400,
                          "delta": abs(s - PAPER_SLOPE)}
    if not per_link:
        return None
    best_name = min(per_link.keys(), key=lambda n: per_link[n]["delta"])
    best = per_link[best_name]
    return {"rfs": best["rfs"], "slope": best["slope"],
            "config": f"linkage={best_name} (k-sweep={len(best['rfs'])})",
            "labels": best["labels_k400"]}


METHOD_LOADERS = {
    "Spectral":     load_spectral,
    "SSC-Lasso":    lambda r: load_ssc(r, kind="lasso"),
    "SSC-OMP":      lambda r: load_ssc(r, kind="omp"),
    "Hierarchical": load_hierarchical,
}

print(f"=== fig_05_10_methods.py comparison (4 methods) - {MODEL_LABEL} ===")
print(f"Results dir: {RESULTS_DIR}")
print()

method_data = {}
for name in METHODS_KEEP:
    print(f"  loading {name}...", end=" ", flush=True)
    try:
        d = METHOD_LOADERS[name](RESULTS_DIR)
    except Exception as exc:
        print(f"ERROR: {exc}")
        d = None
    if d is None:
        print("missing - skipping")
        continue
    s = d["slope"]
    if s is None:
        print(f"envelope=None [{d['config']}]")
    else:
        print(f"slope={s:+.4f} |Δ|={abs(s - PAPER_SLOPE):.3f} [{d['config']}]")
    method_data[name] = d

if "Spectral" not in method_data:
    sys.exit("FATAL: spectral baseline missing")
print()


def _delta_key(name):
    s = method_data[name]["slope"]
    return float("inf") if s is None else abs(s - PAPER_SLOPE)

# Fixed order across both models for visual consistency
ordered = [n for n in METHODS_KEEP if n in method_data]

# For the |Δ| bar chart we sort by delta; for the per-method
# panel grid we keep the fixed order so the same method appears
# in the same cell on both 19m and 125m figures
ordered_by_delta = sorted(ordered, key=_delta_key)


# Figures 5/6 - envelope overlay + |Δ| bar chart

print("Generating Figures 5/6 (method_overlay) ...")

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

ax = axes[0]
for name in ordered_by_delta:
    d = method_data[name]
    if d["slope"] is None:
        continue
    env_r, env_s = envelope_arrays(d["rfs"])
    if len(env_r) == 0:
        continue
    color = METHOD_COLORS.get(name, "gray")
    ax.plot(env_r, env_s, "-", color=color, linewidth=1.8, alpha=0.9,
            label=f"{name}  $\\hat{{\\beta}} = {d['slope']:+.3f}$")

spec_r, spec_s = envelope_arrays(method_data["Spectral"]["rfs"])
_, spec_b = fit_intercept(spec_r, spec_s)
if spec_b is not None:
    anchor = 10 ** spec_b
    r_line = np.logspace(0, 3.2, 100)
    ax.plot(r_line, anchor * r_line ** PAPER_SLOPE, "k--", linewidth=2.0,
            alpha=0.7, label=f"Paper target $\\beta^* = {PAPER_SLOPE}$")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(1, 2000)
ax.set_ylim(0.8, 15000)
ax.set_xlabel("Cluster rank")
ax.set_ylabel("Cluster size (envelope)")
ax.legend(loc="lower left", fontsize=9, framealpha=0.85)
ax.set_title("(a) Envelope overlay")

ax = axes[1]
finite = [n for n in ordered_by_delta if method_data[n]["slope"] is not None]
deltas = [abs(method_data[n]["slope"] - PAPER_SLOPE) for n in finite]
colors = [METHOD_COLORS.get(n, "gray") for n in finite]

y_pos = np.arange(len(finite))
ax.barh(y_pos, deltas, color=colors, edgecolor="black", linewidth=0.6)
ax.set_yticks(y_pos)
ax.set_yticklabels(finite)
ax.invert_yaxis()
ax.axvline(0.05, color="#2e8b57", linestyle=":", linewidth=1.0, alpha=0.7,
           label=r"$|\Delta| = 0.05$")
ax.axvline(0.21, color="#cd853f", linestyle=":", linewidth=1.0, alpha=0.7,
           label=r"$|\Delta| = 0.21$")
ax.set_xlabel(r"$|\hat{\beta} - \beta^*|$" +
              f"  (paper target $\\beta^* = {PAPER_SLOPE}$)")
ax.legend(loc="lower right", fontsize=8)

degenerate = [n for n in ordered_by_delta if method_data[n]["slope"] is None]
if degenerate:
    ax.text(0.99, 0.02,
            "Envelope = N/A: " + ", ".join(degenerate),
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            color="#666", style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      alpha=0.85, edgecolor="lightgray"))

ax.set_title("(b) Slope deviation from paper")

plt.suptitle(f"Envelope comparison across methods ({MODEL_LABEL})",
             fontsize=13, y=1.00)
plt.tight_layout()
# fig_05 (19m) and fig_06 (125m) in contribution-2/appendix/
_fig56_name = "fig_06" if IS_125M else "fig_05"
fig56_path = os.path.join(OUTPUT_DIR_APP, f"{_fig56_name}_method_overlay_{MODEL_SUFFIX}.png")
savefig_both(fig56_path)
plt.close()
print(f"  Saved: {fig56_path}")
print()


# Figure 7/10 - SSC-Lasso hyperparameter heatmap

print("Generating Figure 7/10 (ssc_evaluation) ...")

ssc_lasso_pkl = load_pickle(os.path.join(RESULTS_DIR, "clusters_ssc_full_more.pkl"))

lasso_slopes = {}
best_lasso_key = None
best_lasso_diff = float("inf")
if ssc_lasso_pkl is not None:
    for pair, entry in ssc_lasso_pkl.items():
        if not (isinstance(entry, dict) and "clusters" in entry):
            continue
        rfs = []
        for k, lbl in entry["clusters"].items():
            lbl = np.asarray(lbl)
            if lbl.ndim == 0:
                continue
            if len(np.unique(lbl)) < 5:
                continue
            rfs.append(labels_to_rank_freq(lbl, drop_noise=False))
        if not rfs:
            continue
        s = compute_envelope_slope(rfs)
        if s is not None:
            lasso_slopes[pair] = s
            d = abs(s - PAPER_SLOPE)
            if d < best_lasso_diff:
                best_lasso_diff = d
                best_lasso_key = pair

if best_lasso_key is not None:
    print(f"  best SSC-Lasso (d, α) = {best_lasso_key}  "
          f"slope={lasso_slopes[best_lasso_key]:+.3f}  |Δ|={best_lasso_diff:.3f}")

fig, ax = plt.subplots(1, 1, figsize=(8, 6))

if not lasso_slopes:
    ax.text(0.5, 0.5, "SSC-Lasso pkl missing",
            transform=ax.transAxes, ha="center", va="center", fontsize=11,
            color="#888", style="italic")
    ax.axis("off")
else:
    d_vals = sorted(set(p[0] for p in lasso_slopes.keys()))
    a_vals = sorted(set(p[1] for p in lasso_slopes.keys()))
    M = np.full((len(d_vals), len(a_vals)), np.nan)
    for i, d in enumerate(d_vals):
        for j, a in enumerate(a_vals):
            if (d, a) in lasso_slopes:
                v = lasso_slopes[(d, a)]
                M[i, j] = np.nan if abs(v) < 0.01 else v
    # Diverging colormap centered on PAPER_SLOPE (-1.237):
    #   white = exactly at target, red = steeper, blue = shallower
    M_masked = np.ma.masked_invalid(M)
    cmap = plt.get_cmap("RdBu").copy()   # blue=shallow, red=steep, white=target
    cmap.set_bad("#f0f0f0")
    finite_vals = M[~np.isnan(M)]
    max_dev = np.max(np.abs(finite_vals - PAPER_SLOPE)) if len(finite_vals) else 0.6
    im = ax.imshow(M_masked, aspect="auto", cmap=cmap,
                   vmin=PAPER_SLOPE - max_dev, vmax=PAPER_SLOPE + max_dev)
    ax.set_xticks(range(len(a_vals)))
    ax.set_xticklabels([f"{a}" for a in a_vals], rotation=45, ha="right")
    ax.set_yticks(range(len(d_vals)))
    ax.set_yticklabels([f"d={d}" for d in d_vals])
    ax.set_xlabel(r"Lasso regularization $\alpha$")
    ax.set_ylabel("PCA dimensions")
    for i in range(len(d_vals)):
        for j in range(len(a_vals)):
            v = M[i, j]
            if np.isnan(v):
                continue
            is_best = (best_lasso_key is not None and
                       d_vals[i] == best_lasso_key[0] and
                       a_vals[j] == best_lasso_key[1])
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                    fontweight="bold" if is_best else "normal",
                    color="black")
            if is_best:
                rect = plt.Rectangle((j - 0.48, i - 0.48), 0.96, 0.96,
                                      linewidth=2.5, edgecolor="black",
                                      facecolor="none")
                ax.add_patch(rect)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(r"Envelope slope $\hat\beta$  (white = paper target $-1.237$)")
    if best_lasso_key:
        sub = (f"best $(d, \\alpha) = ({best_lasso_key[0]}, "
               f"{best_lasso_key[1]})$, slope $= {lasso_slopes[best_lasso_key]:+.3f}$")
    else:
        sub = ""
    ax.set_title(f"Envelope slope by (d, $\\alpha$)"
                 + (f"\n{sub}" if sub else ""))

plt.suptitle(f"SSC-Lasso evaluation ({MODEL_LABEL})",
             fontsize=13, y=1.00)
plt.tight_layout()
# fig_07 (19m) and fig_10 (125m) both in contribution-2/appendix/
if IS_125M:
    fig710_path = os.path.join(OUTPUT_DIR_APP, f"fig_10_ssc_evaluation_{MODEL_SUFFIX}.png")
else:
    fig710_path = os.path.join(OUTPUT_DIR_APP, f"fig_07_ssc_evaluation_{MODEL_SUFFIX}.png")
# bbox_inches='tight' so the below-axes legend is not cropped
savefig_both(fig710_path, bbox_inches="tight")
plt.close()
print(f"  Saved: {fig710_path}")
print()

# Figures 8/9 - 2x2 grid of best-config rank-frequency per method (appendix)

print("Generating Figures 8/9 (method_comparison) ...")

_n = len(ordered)
_ncols = 2
_nrows = (_n + 1) // 2
fig, _axes_grid = plt.subplots(_nrows, _ncols, figsize=(10, 5 * _nrows))
axes = _axes_grid.flatten() if _n > 1 else [_axes_grid]
for _i in range(_n, _nrows * _ncols):
    axes[_i].set_visible(False)

for idx, name in enumerate(ordered):
    ax = axes[idx]
    d = method_data[name]
    color = METHOD_COLORS.get(name, "gray")

    for ranks, sizes in d["rfs"]:
        if len(ranks) == 0:
            continue
        ax.plot(ranks, sizes, "-", color=color, alpha=0.25, linewidth=0.7)

    env_r, env_s = envelope_arrays(d["rfs"])
    if len(env_r) > 0:
        ax.plot(env_r, env_s, "-", color=color, linewidth=1.6, alpha=0.95,
                label="Envelope")

    slope = d["slope"]
    intercept = None
    if slope is not None:
        _, intercept = fit_intercept(env_r, env_s)
    if slope is not None and intercept is not None:
        anchor = 10 ** intercept
        r_line = np.logspace(0, 3.2, 80)
        ax.plot(r_line, anchor * r_line ** slope, "-", color="#cc0000",
                linewidth=2.0, alpha=0.9,
                label=f"Measured: $\\hat{{\\beta}} = {slope:+.3f}$")
        ax.plot(r_line, anchor * r_line ** PAPER_SLOPE, "k--",
                linewidth=1.4, alpha=0.7,
                label=f"Paper: $\\beta^* = {PAPER_SLOPE}$")
    elif slope is None:
        ax.text(0.5, 0.5, "envelope\ndegenerated",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=11, color="#888", style="italic")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, 2000)
    ax.set_ylim(0.8, 15000)
    ax.set_xlabel("Cluster rank")
    ax.set_ylabel("Cluster size")
    if slope is None:
        title = f"{name}\nenvelope = N/A"
    else:
        title = (f"{name}  ($\\hat{{\\beta}} = {slope:+.3f}$, "
                 f"$|\\Delta|={abs(slope - PAPER_SLOPE):.3f}$)")
    ax.set_title(title, color=color)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.85)

plt.suptitle(f"Rank-frequency by clustering method ({MODEL_LABEL})",
             fontsize=13, y=1.00)
plt.tight_layout()
_fig89_name = "fig_09" if IS_125M else "fig_08"
fig89_path = os.path.join(OUTPUT_DIR_APP, f"{_fig89_name}_method_comparison_{MODEL_SUFFIX}.png")
savefig_both(fig89_path)
plt.close()
print(f"  Saved: {fig89_path}")
print()


print("=== summary ===")
print(f"{'method':<14} {'slope':>9}  {'|Δ|':>7}   config")
for name in ordered:
    d = method_data[name]
    if d["slope"] is None:
        print(f"{name:<14} {'    N/A':>9}  {'   N/A':>7}   {d['config']}")
    else:
        print(f"{name:<14} {d['slope']:+9.4f}  {abs(d['slope'] - PAPER_SLOPE):7.4f}   {d['config']}")
print(f"{'PAPER':<14} {PAPER_SLOPE:+9.4f}  {0.0:7.4f}   target")
print()
print(f"Outputs in: {OUTPUT_DIR_APP}")
