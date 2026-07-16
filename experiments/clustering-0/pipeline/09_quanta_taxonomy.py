"""Per-cluster token taxonomy for Spectral, SSC-Lasso, and Hierarchical clusters.

Builds quanta_taxonomy_{method}_k400.pkl for each of the three clustering
methods used in the paper. These files are required by:
    figures/fig_04_taxonomy.py
    figures/fig_12_13_taxonomy_detail.py

Output (written to RESULTS_DIR for the active model):
    quanta_taxonomy_spectral_k400.pkl
    quanta_taxonomy_ssc_lasso_k400.pkl
    quanta_taxonomy_hierarchical_k400.pkl

Usage:
    cd experiments/clustering-0
    QDG_MODEL=pythia-19m  python pipeline/09_quanta_taxonomy.py
    QDG_MODEL=pythia-125m python pipeline/09_quanta_taxonomy.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import warnings
from collections import Counter, defaultdict

warnings.filterwarnings("ignore")

import numpy as np
import pyarrow as pa
from transformers import AutoTokenizer

from config import PATHS, MODEL_NAME

K = 400
RESULTS_DIR = PATHS["results_dir"]

# Canonical SSC-Lasso (d, alpha) per model - must match pipeline/06_bootstrap_stability.py
SSC_BEST = {
    "pythia-19m":  (500, 0.0001),
    "pythia-125m": (200, 0.01),
}


def load_pile_arrow(pile_path):
    arrow_files = sorted(f for f in os.listdir(pile_path) if f.endswith(".arrow"))
    if not arrow_files:
        raise FileNotFoundError(f"No .arrow files found in {pile_path}")
    tables = []
    for fname in arrow_files:
        with open(os.path.join(pile_path, fname), "rb") as src:
            reader = pa.ipc.open_stream(src)
            tables.append(reader.read_all())
    table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
    return {
        "input_ids": table["input_ids"].to_pylist(),
        "preds_len": table["preds_len"].to_pylist(),
        "_len": len(table),
    }


def classify_token(tok_str):
    t = tok_str.strip()
    if not t or "\n" in tok_str:
        return "FORMATTING"
    if all(c in "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~" for c in t):
        return "SYNTACTIC"
    if any(c.isdigit() for c in t):
        return "NUMERIC"
    if t.lower() in {"the","a","an","and","or","but","in","on","at","to","for",
                     "of","with","by","from","as","is","was","are","were","be",
                     "been","have","has","had","do","does","did","will","would",
                     "could","should","may","might","shall","not","no","nor",
                     "it","its","this","that","these","those","he","she","they",
                     "we","i","you","him","her","them","us","my","your","our",
                     "their","his","her"}:
        return "FUNCTION_WORDS"
    if t[0].isupper() and len(t) > 1:
        return "PROPER_NOUNS"
    if any(c in t for c in ["_","->","=>","==","!=","//","/*","*/","()","[]","{}"]):
        return "CODE"
    return "CONTENT"


def build_taxonomy(labels, token_strings, method, extra_meta=None):
    cluster_to_indices = defaultdict(list)
    for i, label in enumerate(labels):
        cluster_to_indices[int(label)].append(i)

    cluster_analysis = {}
    category_sizes = defaultdict(int)
    for cluster_id, member_idxs in cluster_to_indices.items():
        toks = [token_strings[i] for i in member_idxs]
        token_counts = Counter(toks)
        type_counts  = Counter(classify_token(t) for t in toks)
        dom_category = type_counts.most_common(1)[0][0]
        cluster_analysis[str(cluster_id)] = {
            "category":   dom_category,
            "top_tokens": [(tok, cnt) for tok, cnt in token_counts.most_common(20)],
            "size":       len(member_idxs),
        }
        category_sizes[dom_category] += len(member_idxs)

    result = {
        "cluster_analysis": cluster_analysis,
        "category_sizes":   dict(category_sizes),
        "model":            MODEL_NAME,
        "method":           method,
        "k":                K,
    }
    if extra_meta:
        result.update(extra_meta)
    return result


print(f"Quanta taxonomy: {MODEL_NAME}")

print("Loading Pile dataset...")
dataset = load_pile_arrow(PATHS["pile_canonical"])
starting_idxs = np.array([0] + list(np.cumsum(dataset["preds_len"])))

print("Loading zero-loss token indices...")
with open(PATHS["zero_induction_idxs"], "rb") as f:
    non_induction_zeros, _, _ = pickle.load(f)
idxs = non_induction_zeros[::50][:10_000]

step = 143000
print(f"Loading tokenizer ({MODEL_NAME} step {step})...")
tokenizer = AutoTokenizer.from_pretrained(
    f"EleutherAI/{MODEL_NAME}",
    revision=f"step{step}",
    cache_dir=f"{PATHS['pythia_cache']}/{MODEL_NAME}/step{step}",
)

print("Decoding tokens...")
token_strings = []
for flat_idx in idxs:
    doc = int(np.searchsorted(starting_idxs, flat_idx, side="right") - 1)
    pos = flat_idx - starting_idxs[doc]
    ids = dataset["input_ids"][doc]
    if isinstance(ids[0], list):
        ids = ids[0]
    tok_id = ids[pos] if pos < len(ids) else ids[-1]
    token_strings.append(tokenizer.decode([tok_id]))
print(f"  {len(token_strings)} tokens decoded")


print("\n[1/3] Spectral")
with open(PATHS["clusters_output"], "rb") as f:
    clusters_sp = pickle.load(f)
if K not in clusters_sp:
    sys.exit(f"  k={K} not found. Available: {sorted(clusters_sp.keys())}")
labels_sp, _ = clusters_sp[K]   # format: (labels_array, labels_Cabs)
labels_sp = np.array(labels_sp)
print(f"  {len(np.unique(labels_sp))} clusters at k={K}")
taxonomy_sp = build_taxonomy(labels_sp, token_strings, "spectral")
out = os.path.join(RESULTS_DIR, "quanta_taxonomy_spectral_k400.pkl")
with open(out, "wb") as f:
    pickle.dump(taxonomy_sp, f)
print(f"  Saved: {out}")


print("\n[2/3] SSC-Lasso")
if MODEL_NAME not in SSC_BEST:
    sys.exit(f"  No canonical SSC config for {MODEL_NAME!r}. Add to SSC_BEST.")
d_opt, alpha_opt = SSC_BEST[MODEL_NAME]
with open(PATHS["clusters_ssc_output"], "rb") as f:
    clusters_ssc = pickle.load(f)
key = (d_opt, alpha_opt)
if key not in clusters_ssc:
    sys.exit(f"  Config {key} not found. Available: {[k for k in clusters_ssc if isinstance(k, tuple)][:5]}")
entry = clusters_ssc[key]
if not isinstance(entry, dict) or "clusters" not in entry:
    sys.exit(f"  Unexpected SSC entry format at {key}.")
if K not in entry["clusters"]:
    sys.exit(f"  k={K} not in SSC-Lasso entry. Available: {sorted(entry['clusters'].keys())}")
labels_ssc = np.array(entry["clusters"][K])
print(f"  {len(np.unique(labels_ssc))} clusters at k={K}  (d={d_opt}, α={alpha_opt})")
taxonomy_ssc = build_taxonomy(labels_ssc, token_strings, "ssc-lasso",
                               extra_meta={"d": d_opt, "alpha": alpha_opt})
out = os.path.join(RESULTS_DIR, "quanta_taxonomy_ssc_lasso_k400.pkl")
with open(out, "wb") as f:
    pickle.dump(taxonomy_ssc, f)
print(f"  Saved: {out}")


print("\n[3/3] Hierarchical")
with open(PATHS["clusters_hierarchical_output"], "rb") as f:
    clusters_hier = pickle.load(f)
labels_dict = clusters_hier.get("labels", {})
if K not in labels_dict:
    sys.exit(f"  k={K} not found in hierarchical labels. Available: {sorted(labels_dict.keys())}")
labels_hier = np.array(labels_dict[K])
print(f"  {len(np.unique(labels_hier))} clusters at k={K}")
taxonomy_hier = build_taxonomy(labels_hier, token_strings, "hierarchical")
out = os.path.join(RESULTS_DIR, "quanta_taxonomy_hierarchical_k400.pkl")
with open(out, "wb") as f:
    pickle.dump(taxonomy_hier, f)
print(f"  Saved: {out}")

print("\nDone.")
