"""Publication figures: Quanta fingerprint analysis (fig_03_fingerprint.py).

Generates one figure used in the thesis main body:
  fig_03_fingerprint - KL/JS divergence, cosine/Pearson similarity,
                       logistic regression balanced accuracy, and active
                       quanta count (Human vs AI, Pythia-19M and Pythia-125M)

Figure is written to {REPO_DIR}/figures/contribution-3/.

Prerequisites (run first):
  pipeline/05_fingerprint.py  for BOTH models (SSC-Lasso paradigm)

Run once per model (set QDG_MODEL env var):
  cd experiments/clustering-0
  python -u figures/fig_03_fingerprint.py                        # Pythia-19M
  QDG_MODEL=pythia-125m python -u figures/fig_03_fingerprint.py  # Pythia-125M
"""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PATHS
from plot_style import apply_style, savefig_both, MODEL_COLORS, HUMAN_AI_COLORS

BASE_DIR = PATHS["base_dir"]
REPO_DIR = PATHS["repo_dir"]

RESULTS_19M = os.path.join(BASE_DIR, "results", "clustering-0")
RESULTS_125M = os.path.join(BASE_DIR, "results", "clustering-0-pythia-125m")

OUTPUT_DIR = os.path.join(REPO_DIR, "figures", "contribution-3")
os.makedirs(OUTPUT_DIR, exist_ok=True)

apply_style()


# Prefer the SSC-Lasso fingerprint results (the recommended paradigm in this
# thesis) over the spectral baseline. Fall back to spectral if SSC-Lasso is
# unavailable so the script remains runnable in either configuration
FP_SUBDIR_PREFERENCE = [
    "quanta_fingerprint_ssc-lasso",   # 600/6000 sample run
    "quanta_fingerprint_ssc_lasso",   # earlier 200/2000 run
    "quanta_fingerprint",             # spectral baseline
]


def _resolve_fingerprint_dir(rdir):
    for sub in FP_SUBDIR_PREFERENCE:
        path = os.path.join(rdir, "figures", sub, "results.json")
        if os.path.exists(path):
            return os.path.dirname(path), sub
    return None, None


print("Loading fingerprint results...")
fp_results = {}
fp_data = {}
fp_paradigm = None  # tracked for the figure subtitle

for model_name, rdir in [("Pythia-19m", RESULTS_19M), ("Pythia-125m", RESULTS_125M)]:
    fp_dir, sub = _resolve_fingerprint_dir(rdir)
    if fp_dir is None:
        print(f"  MISSING: no fingerprint results found under {rdir}/figures/")
        sys.exit(1)
    results_path = os.path.join(fp_dir, "results.json")
    with open(results_path, "r") as f:
        fp_results[model_name] = json.load(f)
    if fp_paradigm is None:
        fp_paradigm = sub
    metric_name = fp_results[model_name].get("logreg_metric", "balanced_accuracy")
    mcc_str = ""
    if fp_results[model_name].get("logreg_mcc") is not None:
        mcc_str = f", MCC={fp_results[model_name]['logreg_mcc']:+.3f}"
    print(f"  {model_name} [{sub}]: {metric_name}={fp_results[model_name]['logreg_accuracy']:.3f}{mcc_str}")

    vec_path = os.path.join(fp_dir, "fingerprint_vectors.npz")
    if os.path.exists(vec_path):
        vecs = np.load(vec_path)
        fp_data[model_name] = {
            "human_avg_fp": vecs["human_avg_fp"],
            "ai_avg_fp": vecs["ai_avg_fp"],
        }
        print(f"  {model_name}: loaded fingerprint vectors ({len(vecs['human_avg_fp'])} clusters)")
    else:
        fp_data[model_name] = {"human_avg_fp": np.array([]), "ai_avg_fp": np.array([])}

PARADIGM_LABEL = {
    "quanta_fingerprint_ssc-lasso": "SSC-Lasso",
    "quanta_fingerprint_ssc_lasso": "SSC-Lasso",
    "quanta_fingerprint":           "Spectral",
}.get(fp_paradigm, fp_paradigm)


# Figure 3: Quanta Fingerprint Analysis (2x2 grid)
#
# Panels:
#   (a) Distribution divergence (KL, JS) between Human and AI fingerprints
#   (b) Fingerprint similarity (cosine, Pearson) between Human and AI
#   (c) LogReg balanced accuracy: classifying Human vs AI from quanta usage
#   (d) Active quanta count: Human vs AI

print("\nGenerating Figure 3: Quanta Fingerprint Analysis (2x2)")

models = list(MODEL_COLORS.keys())
fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# (a) Divergence
ax = axes[0, 0]
metrics = ["KL(H||AI)", "KL(AI||H)", "JS div"]
x = np.arange(len(metrics))
width = 0.35
vals_19m = [fp_results["Pythia-19m"]["kl_human_ai"],
            fp_results["Pythia-19m"]["kl_ai_human"],
            fp_results["Pythia-19m"]["js_divergence"]]
vals_125m = [fp_results["Pythia-125m"]["kl_human_ai"],
             fp_results["Pythia-125m"]["kl_ai_human"],
             fp_results["Pythia-125m"]["js_divergence"]]
bars1 = ax.bar(x - width/2, vals_19m, width, label="Pythia-19m",
               color=MODEL_COLORS["Pythia-19m"], alpha=0.85)
bars2 = ax.bar(x + width/2, vals_125m, width, label="Pythia-125m",
               color=MODEL_COLORS["Pythia-125m"], alpha=0.85)
for bars in [bars1, bars2]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.05,
                f"{h:.2f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylabel("Divergence")
ax.legend(fontsize=9)
ax.set_title("(a) Distribution divergence: Human vs AI")

# (b) Similarity
ax = axes[0, 1]
metrics_b = ["Cosine\nsimilarity", "Pearson\ncorrelation"]
x_b = np.arange(len(metrics_b))
vals_19m_b = [fp_results["Pythia-19m"]["cosine_similarity"],
              fp_results["Pythia-19m"]["correlation"]]
vals_125m_b = [fp_results["Pythia-125m"]["cosine_similarity"],
               fp_results["Pythia-125m"]["correlation"]]
bars1 = ax.bar(x_b - width/2, vals_19m_b, width, label="Pythia-19m",
               color=MODEL_COLORS["Pythia-19m"], alpha=0.85)
bars2 = ax.bar(x_b + width/2, vals_125m_b, width, label="Pythia-125m",
               color=MODEL_COLORS["Pythia-125m"], alpha=0.85)
for bars in [bars1, bars2]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.01,
                f"{h:.3f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")
ax.set_xticks(x_b)
ax.set_xticklabels(metrics_b)
ax.set_ylabel("Score")
ax.set_ylim(0, 0.7)
ax.legend(fontsize=9)
ax.set_title("(b) Fingerprint similarity: Human vs AI")

# (c) LogReg balanced accuracy + MCC inline (one line per metric per bar)
ax = axes[1, 0]
accuracies = [fp_results[m]["logreg_accuracy"] for m in models]
stds = [fp_results[m]["logreg_std"] for m in models]
mccs = [fp_results[m].get("logreg_mcc") for m in models]
colors = [MODEL_COLORS[m] for m in models]
bars = ax.bar(models, [100 * a for a in accuracies],
              yerr=[100 * s for s in stds],
              color=colors, alpha=0.85, capsize=8,
              edgecolor="white", linewidth=0.5)
ax.axhline(y=50, color="#aa0000", linestyle="--", alpha=0.5, linewidth=1.2,
           label="Chance level")
for bar, acc, mcc in zip(bars, accuracies, mccs):
    label = f"{100*acc:.1f}%"
    if mcc is not None:
        label += f"\nMCC = {mcc:+.3f}"
    ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 2.5,
            label, ha="center", va="bottom", fontsize=10)
ax.set_ylabel("Balanced accuracy (%)")
ax.set_ylim(0, 118)  # headroom for two-line value labels
ax.legend(fontsize=9, loc="upper left")
ax.set_title("(c) Logistic regression on quanta fingerprints")

# (d) Active quanta - uses HUMAN_AI_COLORS (not MODEL_COLORS) so the
# encoding does not collide with the model-vs-model encoding in (a-c)
ax = axes[1, 1]
x = np.arange(len(models))
human_active = [fp_results[m]["human_active_quanta"] for m in models]
ai_active = [fp_results[m]["ai_active_quanta"] for m in models]
bars1 = ax.bar(x - width/2, human_active, width, label="Human",
               color=HUMAN_AI_COLORS["Human"], alpha=0.85)
bars2 = ax.bar(x + width/2, ai_active, width, label="AI-generated",
               color=HUMAN_AI_COLORS["AI"], alpha=0.85)
for bars in [bars1, bars2]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., h + 3,
                str(int(h)), ha="center", va="bottom",
                fontsize=11, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylabel("Active quanta (out of 400)")
ax.set_ylim(0, 450)
ax.legend(fontsize=9)
ax.set_title("(d) Active quanta: Human vs AI")

plt.suptitle(f"Quanta fingerprint analysis ({PARADIGM_LABEL} clustering, $k = 400$)",
             fontsize=13, y=1.00)
plt.tight_layout()
fig3_path = os.path.join(OUTPUT_DIR, "fig_03_fingerprint.png")
savefig_both(fig3_path)
plt.close()
print(f"  Saved: {fig3_path}")


print(f"\nFingerprint results summary:")
for model_name in models:
    r = fp_results[model_name]
    print(f"  {model_name}:")
    print(f"    Balanced acc: {100*r['logreg_accuracy']:.1f}% ± {100*r['logreg_std']:.1f}%")
    if r.get("logreg_mcc") is not None:
        mcc_std = r.get("logreg_mcc_std")
        std_str = f" ± {mcc_std:.3f}" if mcc_std is not None else ""
        print(f"    MCC:          {r['logreg_mcc']:+.3f}{std_str}")
    print(f"    Active quanta: human={r['human_active_quanta']}, AI={r['ai_active_quanta']}")
    print(f"    JS divergence: {r['js_divergence']:.3f}")
    print(f"    KL(H||AI): {r['kl_human_ai']:.3f}")

print(f"\nFigures saved to: {OUTPUT_DIR}")
