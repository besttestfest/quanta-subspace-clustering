"""AI-vs-human text classification using cluster usage fingerprints.

Trains LogisticRegression classifier on cluster distributions per document.

Usage:
  python pipeline/05_fingerprint.py                       # Spectral baseline
  python pipeline/05_fingerprint.py --cluster-method ssc-lasso  # SSC-Lasso
  python pipeline/05_fingerprint.py --all-methods         # Generate for all 3 (spectral, ssc-lasso, hierarchical)
  python pipeline/05_fingerprint.py --load-cached         # Skip generation, load npz

Environment:
  QDG_MODEL: pythia-19m (default) or pythia-125m
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import json
import warnings
from collections import defaultdict
import argparse

# Suppress sklearn ≥1.8 deprecation warnings (hundreds of CV iterations)
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

import numpy as np
import torch
from tqdm.auto import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

import pyarrow as pa


class PileArrowDataset:
    """Drop-in replacement for datasets.load_from_disk - reads .arrow files
    directly via PyArrow, avoiding datasets library version incompatibilities.

    Supports:
        dataset["preds_len"]       -> list of ints
        len(dataset)               -> int
        dataset[idx]               -> {"input_ids": [[tok, ...]], "preds_len": int}
    """

    def __init__(self, pile_path):
        arrow_files = sorted(
            f for f in os.listdir(pile_path) if f.endswith(".arrow")
        )
        if not arrow_files:
            raise FileNotFoundError(f"No .arrow files found in {pile_path}")
        tables = []
        for fname in arrow_files:
            with open(os.path.join(pile_path, fname), "rb") as src:
                reader = pa.ipc.open_stream(src)
                tables.append(reader.read_all())
        table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
        self._input_ids = table["input_ids"].to_pylist()
        self._preds_len = table["preds_len"].to_pylist()

    def __len__(self):
        return len(self._preds_len)

    def __getitem__(self, key):
        if isinstance(key, str):
            if key == "preds_len":
                return self._preds_len
            if key == "input_ids":
                return self._input_ids
            raise KeyError(key)
        # Integer index - return row compatible with original datasets API
        ids = self._input_ids[int(key)]
        # Normalise to list-of-list (datasets API wraps sequences this way)
        if not ids or not isinstance(ids[0], list):
            ids = [ids]
        return {"input_ids": ids, "preds_len": self._preds_len[int(key)]}
from transformers import AutoTokenizer, GPTNeoXForCausalLM
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    matthews_corrcoef,
    make_scorer,
)

from config import PATHS, MODEL_NAME

K = 400                    # Number of clusters (same as taxonomy analysis)
N_AI_DOCS = int(os.environ.get("N_AI_DOCS", 600))      # 600 matches the paper run; N_AI_DOCS=200 for the older run
MAX_DOC_LEN = 256          # Max tokens per generated doc
TEMPERATURE = 1.0          # Sampling temperature
LOSS_THRESHOLD = 0.1       # Zero-loss threshold (same as paper)
N_AI_TOKENS = int(os.environ.get("N_AI_TOKENS", 6000)) # 6000 matches the paper run; N_AI_TOKENS=2000 for the older run
BATCH_GRAD = 50            # Gradient computation batch size
MIN_TOKENS_PER_DOC = 3     # Min tokens per doc for classification
PROMPT_LEN = 10            # Human prompt tokens at the start of each AI doc

# Best (d, alpha) for SSC-Lasso per model - winners of the apples-to-apples
# envelope sweep produced by pipeline/03_envelope_analysis.py
SSC_BEST = {
    "pythia-19m":  (500, 0.0001),   # slope = -1.245, |Δ| = 0.008
    "pythia-125m": (200, 0.01),     # slope = -1.329, |Δ| = 0.092
}

# Best (d, K=n_nonzero_coefs) for SSC-OMP per model
SSC_OMP_BEST = {
    "pythia-19m":  (100, 5),        # slope = -1.449, |Δ| = 0.212
    "pythia-125m": (200, 20),       # slope = -1.303, |Δ| = 0.066
}

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description="Quanta fingerprint and AI detection analysis")
parser.add_argument("--cluster-method", default="spectral",
                    choices=["spectral", "ssc-lasso", "ssc-omp", "hierarchical"],
                    help="Clustering method to use")
parser.add_argument("--all-methods", action="store_true",
                    help="Generate for all 3 paper methods sequentially (spectral, ssc-lasso, hierarchical)")
parser.add_argument("--load-cached", action="store_true",
                    help="Skip generation, load existing fingerprint_matrix.npz")
parser.add_argument("--skip-fingerprint", action="store_true",
                    help="Skip entire fingerprint pipeline (no output produced)")
args = parser.parse_args()

# If --all-methods, run sequentially for the 3 paper methods.
# SSC-OMP is excluded: it does not appear in any paper result table (saves ~22 h).
# Use --cluster-method ssc-omp explicitly if you need it.
if args.all_methods:
    methods_to_run = ["spectral", "ssc-lasso", "hierarchical"]
else:
    methods_to_run = [args.cluster_method]

RESULTS_DIR = PATHS["results_dir"]
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_cluster_labels(method):
    """Load cluster labels at k=400 for the specified method.

    Returns:
        (labels, method_description)
        labels: list/array of length 10000 with cluster ids in [0, K)
        method_description: human-readable string
    """
    if method == "spectral":
        with open(PATHS["clusters_output"], "rb") as f:
            cd = pickle.load(f)
        if K not in cd:
            avail = sorted([k for k in cd.keys() if isinstance(k, int)])
            raise ValueError(f"k={K} not found in spectral pkl. Available: {avail}")
        lc, _ = cd[K]  # tuple (labels_C, labels_Cabs)
        return lc, f"Spectral (k={K}, angular)"

    elif method == "ssc-lasso":
        with open(PATHS["clusters_ssc_output"], "rb") as f:
            cd = pickle.load(f)
        cfg = SSC_BEST.get(MODEL_NAME)
        if cfg is None:
            raise ValueError(f"No SSC_BEST config for model {MODEL_NAME!r}")
        if cfg not in cd:
            raise ValueError(f"(d, alpha)={cfg} not in SSC pkl. Available: {sorted(cd.keys())}")
        entry = cd[cfg]
        if not isinstance(entry, dict) or "clusters" not in entry:
            raise ValueError(f"Unexpected SSC entry shape at {cfg}")
        cl_dict = entry["clusters"]
        if K not in cl_dict:
            raise ValueError(f"k={K} not in SSC clusters[{cfg}]. Available: {sorted(cl_dict.keys())}")
        return cl_dict[K], f"SSC-Lasso (d={cfg[0]}, α={cfg[1]}, k={K})"

    elif method == "ssc-omp":
        with open(PATHS["clusters_ssc_omp_output"], "rb") as f:
            cd = pickle.load(f)
        cfg = SSC_OMP_BEST.get(MODEL_NAME)
        if cfg is None:
            raise ValueError(f"No SSC_OMP_BEST config for model {MODEL_NAME!r}")
        if cfg not in cd:
            raise ValueError(f"(d, K)={cfg} not in SSC-OMP pkl. Available: {sorted(cd.keys())}")
        entry = cd[cfg]
        if not isinstance(entry, dict) or "clusters" not in entry:
            raise ValueError(f"Unexpected SSC-OMP entry shape at {cfg}")
        cl_dict = entry["clusters"]
        if K not in cl_dict:
            raise ValueError(f"k={K} not in SSC-OMP clusters[{cfg}]. Available: {sorted(cl_dict.keys())}")
        return cl_dict[K], f"SSC-OMP (d={cfg[0]}, K={cfg[1]}, k={K})"

    elif method == "hierarchical":
        path = PATHS["clusters_hierarchical_output"]
        if not os.path.exists(path):
            raise FileNotFoundError(f"clusters_hierarchical.pkl not found at: {path}")
        with open(path, "rb") as f:
            cd = pickle.load(f)
        labels_dict = cd.get("labels", {})
        if K not in labels_dict:
            avail = sorted(labels_dict.keys())
            raise ValueError(f"k={K} not in hierarchical pkl. Available: {avail}")
        return list(labels_dict[K]), f"Hierarchical (complete linkage, k={K})"

    else:
        raise ValueError(f"Unknown cluster method: {method}")


def run_fingerprint_analysis(method_name):
    """Run full fingerprint analysis for a given clustering method."""

    print(f"\nFingerprint analysis: {method_name} ({MODEL_NAME}, k={K}, {N_AI_DOCS} AI docs)")

    # Output directories (backward compat: 'quanta_fingerprint' for spectral)
    if method_name == "spectral":
        figure_dir = f"{RESULTS_DIR}/figures/quanta_fingerprint"
    else:
        figure_dir = f"{RESULTS_DIR}/figures/quanta_fingerprint_{method_name}"
    os.makedirs(figure_dir, exist_ok=True)

    print("Step 1: Loading existing data...")

    # Load dataset
    dataset = PileArrowDataset(PATHS["pile_canonical"])
    starting_indexes = np.array([0] + list(np.cumsum(dataset["preds_len"])))

    def loss_idx_to_dataset_idx(idx):
        sample_index = np.searchsorted(starting_indexes, idx, side="right") - 1
        pred_in_sample_index = idx - starting_indexes[sample_index]
        return int(sample_index), int(pred_in_sample_index)

    # Load token indices
    with open(PATHS["zero_induction_idxs"], "rb") as f:
        non_induction_zeros, zero_idxs, induction_idxs = pickle.load(f)
    # Load cluster labels first, then match the token window to however many
    # tokens were actually clustered (10000 at full scale; fewer when
    # pipeline/01 was run with --n-tokens for smoke testing).
    labels, label_desc = load_cluster_labels(method_name)
    labels = list(labels)
    print(f"  Loaded {label_desc}: {len(labels)} tokens")
    idxs = non_induction_zeros[::50][:len(labels)]
    n_unique = len(set(labels))
    if n_unique != K:
        print(f"  WARNING: {n_unique} unique cluster ids, expected {K}")

    # Load similarity matrix (needed to assign new tokens to clusters)
    print("  Loading similarity matrix...")
    sim_data = torch.load(PATHS["similarity_matrix"], map_location='cpu', weights_only=False)
    existing_idxs, C_sim, C_abs_sim = sim_data
    print(f"  Similarity matrix: {C_sim.shape}")

    print("\nStep 2: Building human quanta fingerprints...")

    token_to_doc = {}
    doc_to_tokens = defaultdict(list)
    for i, flat_idx in enumerate(idxs):
        doc_idx, pos = loss_idx_to_dataset_idx(flat_idx)
        token_to_doc[i] = doc_idx
        doc_to_tokens[doc_idx].append(i)

    n_human_docs = len(doc_to_tokens)
    print(f"  {len(idxs):,} tokens span {n_human_docs} documents")
    tokens_per_doc = [len(v) for v in doc_to_tokens.values()]
    print(f"  Tokens per doc: min={min(tokens_per_doc)}, max={max(tokens_per_doc)}, mean={np.mean(tokens_per_doc):.1f}")

    # Build fingerprint: for each doc, count how many tokens fall in each cluster
    human_fingerprints = {}
    for doc_idx, token_indices in doc_to_tokens.items():
        fp = np.zeros(K)
        for ti in token_indices:
            fp[labels[ti]] += 1
        if fp.sum() > 0:
            fp = fp / fp.sum()
        human_fingerprints[doc_idx] = fp

    human_fp_matrix = np.array(list(human_fingerprints.values()))
    human_avg_fp = human_fp_matrix.mean(axis=0)
    print(f"  Human avg fingerprint: {(human_avg_fp > 0).sum()} active quanta out of {K}")

    print(f"\nStep 3: Loading model and generating AI text...")

    # Shared AI docs: generate once and reuse across all methods so that
    # fingerprint comparisons are apples-to-apples (same AI text for all).
    shared_ai_path = os.path.join(RESULTS_DIR, "shared_ai_docs.pkl")

    if os.path.exists(shared_ai_path):
        print(f"  Loading shared AI docs from: {shared_ai_path}")
        with open(shared_ai_path, "rb") as f:
            ai_docs = [torch.tensor(d) for d in pickle.load(f)["docs"]]
        print(f"  Loaded {len(ai_docs)} documents")
        total_ai_tokens = sum(len(d) for d in ai_docs)
        print(f"  Total AI tokens: {total_ai_tokens}")
        step = 143000
        try:
            model = GPTNeoXForCausalLM.from_pretrained(
                f"EleutherAI/{MODEL_NAME}",
                revision=f"step{step}",
                cache_dir=f"{PATHS['pythia_cache']}/{MODEL_NAME}/step{step}",
            ).to(device)
        except AttributeError:
            model = GPTNeoXForCausalLM.from_pretrained(
                f"EleutherAI/{MODEL_NAME}",
                revision=f"step{step}",
                cache_dir=f"{PATHS['pythia_cache']}/{MODEL_NAME}/step{step}",
                use_safetensors=False,
            ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(
            f"EleutherAI/{MODEL_NAME}",
            revision=f"step{step}",
            cache_dir=f"{PATHS['pythia_cache']}/{MODEL_NAME}/step{step}",
        )
        model.eval()
    else:
        step = 143000
        try:
            model = GPTNeoXForCausalLM.from_pretrained(
                f"EleutherAI/{MODEL_NAME}",
                revision=f"step{step}",
                cache_dir=f"{PATHS['pythia_cache']}/{MODEL_NAME}/step{step}",
            ).to(device)
        except AttributeError:
            print("  Safetensors error, trying PyTorch format...")
            model = GPTNeoXForCausalLM.from_pretrained(
                f"EleutherAI/{MODEL_NAME}",
                revision=f"step{step}",
                cache_dir=f"{PATHS['pythia_cache']}/{MODEL_NAME}/step{step}",
                use_safetensors=False,
            ).to(device)

        tokenizer = AutoTokenizer.from_pretrained(
            f"EleutherAI/{MODEL_NAME}",
            revision=f"step{step}",
            cache_dir=f"{PATHS['pythia_cache']}/{MODEL_NAME}/step{step}",
        )
        model.eval()

        print(f"  Generating {N_AI_DOCS} AI documents (temp={TEMPERATURE})...")
        np.random.seed(42)  # Fix seed for reproducible prompt selection
        torch.manual_seed(42)  # Fix seed for reproducible sampling
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        ai_docs = []
        for doc_i in tqdm(range(N_AI_DOCS), desc="Generating"):
            prompt_doc = dataset[np.random.randint(0, len(dataset))]
            prompt_ids = prompt_doc["input_ids"][0][:PROMPT_LEN]
            input_ids = torch.tensor([prompt_ids], device=device)

            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    max_new_tokens=MAX_DOC_LEN,
                    temperature=TEMPERATURE,
                    do_sample=True,
                    top_k=0,  # Pure sampling
                    pad_token_id=tokenizer.eos_token_id,
                )
            ai_docs.append(output[0].cpu())

        print(f"  Generated {len(ai_docs)} documents")
        total_ai_tokens = sum(len(d) for d in ai_docs)
        print(f"  Total AI tokens: {total_ai_tokens}")

        # Save for reuse by subsequent methods in --all-methods run
        with open(shared_ai_path, "wb") as f:
            pickle.dump({"docs": [d.tolist() for d in ai_docs],
                         "n_docs": len(ai_docs),
                         "model": MODEL_NAME}, f)
        print(f"  Saved shared AI docs: {shared_ai_path}")

    print(f"\nStep 4: Finding zero-loss tokens in AI text...")

    ai_zero_loss_tokens = []

    for doc_i, doc_ids in enumerate(tqdm(ai_docs, desc="Finding zero-loss")):
        input_ids = doc_ids.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[0]

        # Only consider generated tokens (skip targets inside the human prompt)
        for t in range(PROMPT_LEN - 1, len(doc_ids) - 1):
            log_probs = torch.log_softmax(logits[t], dim=-1)
            target = doc_ids[t + 1]
            loss = -log_probs[target].item()

            if loss < LOSS_THRESHOLD:
                ai_zero_loss_tokens.append({
                    'doc_idx': doc_i,
                    'token_pos': t + 1,
                    'input_ids': doc_ids[:t+2].tolist(),
                    'loss': loss,
                })

    print(f"  Found {len(ai_zero_loss_tokens)} zero-loss tokens in AI text")
    print(f"  Zero-loss rate: {100*len(ai_zero_loss_tokens)/total_ai_tokens:.1f}%")

    if len(ai_zero_loss_tokens) > N_AI_TOKENS:
        np.random.seed(42)
        indices = np.random.choice(len(ai_zero_loss_tokens), N_AI_TOKENS, replace=False)
        ai_zero_loss_tokens = [ai_zero_loss_tokens[i] for i in sorted(indices)]
        print(f"  Sampled {N_AI_TOKENS} tokens for gradient computation")

    print(f"\nStep 5: Computing gradients for {len(ai_zero_loss_tokens)} AI tokens...")

    param_names = [n for n, p in model.named_parameters()]
    # Exclude layernorm and embeddings from gradient computation
    highsignal_names = [name for name in param_names if
                        ('layernorm' not in name) and ('embed' not in name)]
    len_g = sum(model.state_dict()[name].numel() for name in highsignal_names)
    print(f"  Gradient dimension: {len_g:,}")

    def get_flattened_gradient(model, param_subset):
        grads = []
        for name, p in model.named_parameters():
            if name in param_subset:
                grads.append(p.grad)
        return torch.cat([g.flatten() for g in grads])

    unique_labels = np.unique(np.asarray(labels).flatten())
    active_clusters = sorted(unique_labels.tolist())
    n_active = len(active_clusters)
    print(f"  Active clusters: {n_active} out of {K}")

    # Select representative tokens per cluster
    cluster_to_rep_idx = {}
    print("  Selecting representative tokens per active cluster...")
    labels_np = np.asarray(labels).flatten()
    representatives = []
    for i, c in enumerate(active_clusters):
        cluster_indices = np.where(labels_np == c)[0]
        cluster_to_rep_idx[c] = i
        if len(cluster_indices) == 0:
            representatives.append(0)
        elif len(cluster_indices) == 1:
            representatives.append(cluster_indices[0])
        else:
            sub_sim = C_sim[cluster_indices][:, cluster_indices]
            if hasattr(sub_sim, 'numpy'):
                sub_sim = sub_sim.numpy()
            avg_sim = np.nanmean(sub_sim, axis=1)
            if len(avg_sim) == 0 or np.all(np.isnan(avg_sim)):
                representatives.append(cluster_indices[0])
            else:
                best = cluster_indices[np.nanargmax(avg_sim)]
                representatives.append(best)

    rep_indices = np.array(representatives)
    N_REPS = len(rep_indices)
    print(f"  Selected {N_REPS} representative tokens (one per active cluster)")

    del C_sim, C_abs_sim, sim_data
    import gc; gc.collect()

    # Compute gradients for representative tokens
    print(f"  Computing gradients for {N_REPS} representative tokens (float16)...")
    rep_gradients = torch.zeros((N_REPS, len_g), dtype=torch.float16)

    for batch_start in tqdm(range(0, N_REPS, BATCH_GRAD), desc="Rep gradients"):
        batch_end = min(batch_start + BATCH_GRAD, N_REPS)
        for j in range(batch_start, batch_end):
            rep_token_idx = idxs[rep_indices[j]]
            sample_index, pred_in_sample_index = loss_idx_to_dataset_idx(rep_token_idx)
            sample = dataset[sample_index]
            input_ids = torch.tensor(sample["input_ids"], device=device)
            target_idx = pred_in_sample_index + 1

            model.zero_grad()
            logits = model(input_ids).logits
            logp = torch.log_softmax(logits[0, target_idx - 1, :], dim=-1)
            target_token = input_ids[0, target_idx]
            loss = -logp[target_token]
            loss.backward()

            g = get_flattened_gradient(model, highsignal_names)
            rep_gradients[j] = g.cpu().half()
            del g, logits, logp, loss
            torch.cuda.empty_cache()

    print(f"  Normalizing representative gradients (row-by-row)...")
    for j in range(N_REPS):
        row = rep_gradients[j].float()
        norm = row.norm()
        if norm > 0:
            rep_gradients[j] = (row / norm).half()
        del row
    print(f"  Representative gradients: {rep_gradients.shape}, dtype={rep_gradients.dtype}")
    print(f"  Memory: {rep_gradients.nelement() * 2 / 1e9:.1f} GB")

    # Compute gradients for AI tokens and assign to clusters
    MATMUL_CHUNK = 50
    print(f"\n  Computing gradients for AI tokens and assigning to clusters...")
    print(f"  (matmul in chunks of {MATMUL_CHUNK}, {N_REPS} active reps)")
    ai_cluster_assignments = []
    all_sims_best = []

    for i, token_info in enumerate(tqdm(ai_zero_loss_tokens, desc="AI gradients")):
        input_ids = torch.tensor([token_info['input_ids']], device=device)
        target_idx = token_info['token_pos']

        model.zero_grad()
        logits = model(input_ids).logits
        logp = torch.log_softmax(logits[0, target_idx - 1, :], dim=-1)
        target_token = input_ids[0, target_idx]
        loss = -logp[target_token]
        loss.backward()

        g = get_flattened_gradient(model, highsignal_names)
        g_cpu = g.cpu().float()
        g_norm = g_cpu / (g_cpu.norm() + 1e-8)
        g_norm = g_norm.unsqueeze(0)

        sims = torch.zeros(N_REPS)
        for chunk_start in range(0, N_REPS, MATMUL_CHUNK):
            chunk_end = min(chunk_start + MATMUL_CHUNK, N_REPS)
            chunk = rep_gradients[chunk_start:chunk_end].float()
            sims[chunk_start:chunk_end] = torch.matmul(g_norm, chunk.T).squeeze(0)
            del chunk

        best_rep_idx = sims.argmax().item()
        best_sim = sims[best_rep_idx].item()
        assigned_cluster = active_clusters[best_rep_idx]

        ai_cluster_assignments.append({
            'doc_idx': token_info['doc_idx'],
            'cluster': assigned_cluster,
            'similarity': best_sim,
        })
        all_sims_best.append(best_sim)

        del g, g_cpu, g_norm, logits, logp, loss, sims
        torch.cuda.empty_cache()

    all_sims_best = np.array(all_sims_best)
    print(f"  Assigned {len(ai_cluster_assignments)} AI tokens to clusters")
    print(f"  Assignment quality (cosine sim to nearest rep):")
    print(f"    Mean: {all_sims_best.mean():.4f}, Std: {all_sims_best.std():.4f}")
    print(f"    Min: {all_sims_best.min():.4f}, Max: {all_sims_best.max():.4f}")
    print(f"    <0.1: {(all_sims_best < 0.1).sum()}/{len(all_sims_best)} ({100*(all_sims_best < 0.1).mean():.1f}%)")
    print(f"    >0.3: {(all_sims_best > 0.3).sum()}/{len(all_sims_best)} ({100*(all_sims_best > 0.3).mean():.1f}%)")

    print(f"\nStep 6: Building AI quanta fingerprints...")

    ai_doc_tokens = defaultdict(list)
    for assignment in ai_cluster_assignments:
        ai_doc_tokens[assignment['doc_idx']].append(assignment['cluster'])

    ai_fingerprints = {}
    for doc_idx, cluster_list in ai_doc_tokens.items():
        fp = np.zeros(K)
        for c in cluster_list:
            fp[c] += 1
        if fp.sum() > 0:
            fp = fp / fp.sum()
        ai_fingerprints[doc_idx] = fp

    n_ai_docs_with_tokens = len(ai_fingerprints)
    ai_fp_matrix = np.array(list(ai_fingerprints.values()))
    ai_avg_fp = ai_fp_matrix.mean(axis=0)

    print(f"  AI docs with zero-loss tokens: {n_ai_docs_with_tokens}/{N_AI_DOCS}")
    print(f"  AI avg fingerprint: {(ai_avg_fp > 0).sum()} active quanta out of {K}")

    print(f"\nStep 7: Comparing quanta distributions...")

    # A) Global fingerprint comparison
    print(f"\n  A) Global average fingerprint comparison:")
    print(f"     Human: {(human_avg_fp > 0).sum()} active quanta")
    print(f"     AI:    {(ai_avg_fp > 0).sum()} active quanta")

    eps = 1e-10
    human_smooth = human_avg_fp + eps
    human_smooth /= human_smooth.sum()
    ai_smooth = ai_avg_fp + eps
    ai_smooth /= ai_smooth.sum()

    kl_human_ai = stats.entropy(human_smooth, ai_smooth)
    kl_ai_human = stats.entropy(ai_smooth, human_smooth)
    js_divergence = 0.5 * stats.entropy(human_smooth, 0.5*(human_smooth+ai_smooth)) + \
                    0.5 * stats.entropy(ai_smooth, 0.5*(human_smooth+ai_smooth))

    print(f"     KL(human||AI) = {kl_human_ai:.4f}")
    print(f"     KL(AI||human) = {kl_ai_human:.4f}")
    print(f"     JS divergence = {js_divergence:.4f}")

    # B) Chi-squared test
    human_total_counts = np.zeros(K)
    for doc_idx, token_indices in doc_to_tokens.items():
        for ti in token_indices:
            human_total_counts[labels[ti]] += 1

    ai_total_counts = np.zeros(K)
    for assignment in ai_cluster_assignments:
        ai_total_counts[assignment['cluster']] += 1

    active_mask = (human_total_counts > 0) | (ai_total_counts > 0)
    chi2 = p_value = None
    if active_mask.sum() > 1:
        contingency = np.vstack([human_total_counts[active_mask],
                                 ai_total_counts[active_mask]])
        chi2, p_value, chi2_dof, _ = stats.chi2_contingency(contingency)
        print(f"\n  B) Chi-squared test of homogeneity (raw counts, dof={chi2_dof}):")
        print(f"     chi2 = {chi2:.2f}, p = {p_value:.6f}")
        print(f"     {'SIGNIFICANT' if p_value < 0.05 else 'NOT significant'} at p<0.05")

    # C) Cosine similarity
    cos_sim = np.dot(human_avg_fp, ai_avg_fp) / (np.linalg.norm(human_avg_fp) * np.linalg.norm(ai_avg_fp) + eps)
    print(f"\n  C) Cosine similarity of avg fingerprints: {cos_sim:.4f}")

    # D) Per-document classification
    print(f"\n  D) Per-document classification via fingerprint:")

    human_docs_kept = [doc for doc in human_fingerprints
                       if len(doc_to_tokens[doc]) >= MIN_TOKENS_PER_DOC]
    ai_docs_kept = [doc for doc in ai_fingerprints
                    if len(ai_doc_tokens[doc]) >= MIN_TOKENS_PER_DOC]
    human_fps_filtered = [human_fingerprints[doc] for doc in human_docs_kept]
    ai_fps_filtered = [ai_fingerprints[doc] for doc in ai_docs_kept]

    # Record the exact document selection so downstream baselines
    # (pipeline/08_bow_baseline.py) classify the identical corpus.
    doc_sel_path = os.path.join(RESULTS_DIR, "classification_docs.json")
    with open(doc_sel_path, "w") as f:
        json.dump({
            "model": MODEL_NAME,
            "min_tokens_per_doc": MIN_TOKENS_PER_DOC,
            "human_doc_idxs": [int(d) for d in human_docs_kept],
            "ai_doc_idxs": [int(d) for d in ai_docs_kept],
        }, f, indent=2)
    print(f"     Saved document selection: {doc_sel_path}")

    scores = None
    mcc_scores = None
    if len(human_fps_filtered) >= 10 and len(ai_fps_filtered) >= 10:
        X = np.array(human_fps_filtered + ai_fps_filtered)
        y = np.array([0]*len(human_fps_filtered) + [1]*len(ai_fps_filtered))

        # Save fingerprint matrix for reproducibility
        np.savez_compressed(
            f"{figure_dir}/fingerprint_matrix.npz",
            X=X.astype(np.float32),
            y=y.astype(np.int64),
            n_human=len(human_fps_filtered),
            n_ai=len(ai_fps_filtered),
            n_clusters=int(X.shape[1]),
        )

        clf = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
        cv_folds = min(5, min(len(human_fps_filtered), len(ai_fps_filtered)))
        scores = cross_val_score(clf, X, y, cv=cv_folds, scoring='balanced_accuracy')
        mcc_scores = cross_val_score(clf, X, y, cv=cv_folds, scoring=make_scorer(matthews_corrcoef))

        print(f"     Human docs (≥{MIN_TOKENS_PER_DOC} tokens): {len(human_fps_filtered)}")
        print(f"     AI docs (≥{MIN_TOKENS_PER_DOC} tokens): {len(ai_fps_filtered)}")
        print(f"     LogReg {cv_folds}-fold CV balanced accuracy: {100*scores.mean():.1f}% ± {100*scores.std():.1f}%")
        print(f"     LogReg {cv_folds}-fold CV MCC:               {mcc_scores.mean():+.3f} ± {mcc_scores.std():.3f}")
        print(f"     {'ABOVE CHANCE' if scores.mean() > 0.6 else 'NEAR CHANCE (~50%)'}")
    else:
        print(f"     Not enough docs with ≥{MIN_TOKENS_PER_DOC} tokens")
        print(f"     Human: {len(human_fps_filtered)}, AI: {len(ai_fps_filtered)}")

    # E) Top differentiating quanta
    print(f"\n  E) Top differentiating quanta (biggest human vs AI difference):")
    diffs = human_avg_fp - ai_avg_fp
    sorted_quanta = np.argsort(np.abs(diffs))[::-1]
    print(f"     {'Cluster':>8} {'Human%':>8} {'AI%':>8} {'Diff':>8}")
    for q in sorted_quanta[:15]:
        print(f"     {q:>8d} {100*human_avg_fp[q]:>7.2f}% {100*ai_avg_fp[q]:>7.2f}% {100*diffs[q]:>+7.2f}%")

    if not args.skip_fingerprint:
        print(f"\nStep 8: Generating figures...")

        plt.rcParams.update({
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "figure.facecolor": "white",
        })

        # Figure 1: Average quanta fingerprints
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        sort_idx = np.argsort(human_avg_fp)[::-1]

        axes[0].bar(range(K), human_avg_fp[sort_idx], color='#1f77b4', alpha=0.7, width=1.0)
        axes[0].set_ylabel('Probability')
        axes[0].set_title(f'Human Quanta Fingerprint (The Pile, {n_human_docs} docs)')
        axes[0].set_xlim(-1, K+1)

        axes[1].bar(range(K), ai_avg_fp[sort_idx], color='#d62728', alpha=0.7, width=1.0)
        axes[1].set_ylabel('Probability')
        axes[1].set_xlabel(f'Cluster Index (sorted by human frequency, k={K})')
        axes[1].set_title(f'AI Quanta Fingerprint ({MODEL_NAME}, {n_ai_docs_with_tokens} docs)')

        fig.suptitle(f'Quanta Fingerprint Comparison - {MODEL_NAME}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{figure_dir}/fig1_fingerprint_comparison.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{figure_dir}/fig1_fingerprint_comparison.pdf", bbox_inches='tight')
        plt.close()
        print(f"  Saved fig1_fingerprint_comparison")

        # Figure 2: Scatter plot
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(100*human_avg_fp, 100*ai_avg_fp, alpha=0.5, s=20, color='#2c3e50')

        max_val = max(100*human_avg_fp.max(), 100*ai_avg_fp.max())
        ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, linewidth=1, label='Perfect agreement')

        corr = np.corrcoef(human_avg_fp, ai_avg_fp)[0, 1]
        ax.set_xlabel('Human usage (%)')
        ax.set_ylabel('AI usage (%)')
        ax.set_title(f'Per-Cluster Usage: Human vs AI (r={corr:.3f})')
        ax.legend()

        plt.tight_layout()
        plt.savefig(f"{figure_dir}/fig2_usage_scatter.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{figure_dir}/fig2_usage_scatter.pdf", bbox_inches='tight')
        plt.close()
        print(f"  Saved fig2_usage_scatter")

        # Figure 3: Difference plot
        fig, ax = plt.subplots(figsize=(14, 5))
        colors = ['#1f77b4' if d > 0 else '#d62728' for d in diffs[sort_idx]]
        ax.bar(range(K), 100*diffs[sort_idx], color=colors, alpha=0.7, width=1.0)
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.set_xlabel(f'Cluster Index (sorted by human frequency)')
        ax.set_ylabel('Difference (Human - AI) in %')
        ax.set_title(f'Quanta Usage Difference - {MODEL_NAME} (blue=more human, red=more AI)')
        ax.set_xlim(-1, K+1)

        plt.tight_layout()
        plt.savefig(f"{figure_dir}/fig3_usage_difference.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{figure_dir}/fig3_usage_difference.pdf", bbox_inches='tight')
        plt.close()
        print(f"  Saved fig3_usage_difference")

        # Figure 4: Summary statistics
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        metrics = ['KL(H||AI)', 'KL(AI||H)', 'JS div']
        values = [kl_human_ai, kl_ai_human, js_divergence]
        axes[0].bar(metrics, values, color=['#3498db', '#3498db', '#e74c3c'], alpha=0.8)
        axes[0].set_ylabel('Divergence')
        axes[0].set_title('Distribution Divergence')
        for i, v in enumerate(values):
            axes[0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=10)

        if scores is not None:
            bar_labels = ['Cosine sim', 'LogReg acc']
            bar_vals = [cos_sim, scores.mean()]
            bar_colors = ['#2ecc71', '#e67e22']
            axes[1].bar(bar_labels, bar_vals, color=bar_colors, alpha=0.8)
            axes[1].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Chance level')
            axes[1].set_ylabel('Score')
            axes[1].set_title('Similarity & Classification')
            axes[1].set_ylim(0, 1.1)
            axes[1].legend()
            for i, v in enumerate(bar_vals):
                axes[1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)

        bar_labels = ['Human', 'AI']
        bar_vals = [(human_avg_fp > 0).sum(), (ai_avg_fp > 0).sum()]
        axes[2].bar(bar_labels, bar_vals, color=['#1f77b4', '#d62728'], alpha=0.8)
        axes[2].set_ylabel(f'Active Quanta (out of {K})')
        axes[2].set_title('Number of Active Quanta')
        for i, v in enumerate(bar_vals):
            axes[2].text(i, v + 2, str(v), ha='center', fontsize=10)

        fig.suptitle(f'Quanta Fingerprint Summary - {MODEL_NAME}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{figure_dir}/fig4_summary.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{figure_dir}/fig4_summary.pdf", bbox_inches='tight')
        plt.close()
        print(f"  Saved fig4_summary")

        # Save results JSON
        results = {
            'model': MODEL_NAME,
            'cluster_method': method_name,
            'cluster_method_desc': label_desc,
            'k': K,
            'n_active_clusters': n_active,
            'n_human_docs': n_human_docs,
            'n_ai_docs': n_ai_docs_with_tokens,
            'n_ai_tokens_analyzed': len(ai_cluster_assignments),
            'human_active_quanta': int((human_avg_fp > 0).sum()),
            'ai_active_quanta': int((ai_avg_fp > 0).sum()),
            'assignment_sim_mean': float(all_sims_best.mean()),
            'assignment_sim_std': float(all_sims_best.std()),
            'assignment_sim_min': float(all_sims_best.min()),
            'assignment_sim_max': float(all_sims_best.max()),
            'assignment_low_quality_pct': float((all_sims_best < 0.1).mean()),
            'kl_human_ai': float(kl_human_ai),
            'kl_ai_human': float(kl_ai_human),
            'js_divergence': float(js_divergence),
            'cosine_similarity': float(cos_sim),
            'chi2': float(chi2) if chi2 is not None else None,
            'chi2_p_value': float(p_value) if p_value is not None else None,
            'logreg_accuracy': float(scores.mean()) if scores is not None else None,
            'logreg_std': float(scores.std()) if scores is not None else None,
            'logreg_metric': 'balanced_accuracy',
            'logreg_mcc': float(mcc_scores.mean()) if mcc_scores is not None else None,
            'logreg_mcc_std': float(mcc_scores.std()) if mcc_scores is not None else None,
            'correlation': float(corr),
        }

        with open(f"{figure_dir}/results.json", "w") as f:
            json.dump(results, f, indent=2)

        np.savez_compressed(
            f"{figure_dir}/fingerprint_vectors.npz",
            human_avg_fp=human_avg_fp,
            ai_avg_fp=ai_avg_fp,
        )
        print(f"  Saved fingerprint_vectors.npz")

        # Save generated AI texts for reproducibility
        with open(f"{figure_dir}/ai_generated_texts.pkl", "wb") as f:
            pickle.dump({
                "texts": [d.tolist() for d in ai_docs],
                "n_ai_docs_generated": N_AI_DOCS,
                "n_ai_docs_with_tokens": n_ai_docs_with_tokens,
                "n_ai_zero_loss_tokens": len(ai_zero_loss_tokens),
                "temperature": TEMPERATURE,
                "max_doc_len": MAX_DOC_LEN,
                "random_seed": 42,
            }, f)
        print(f"  Saved ai_generated_texts.pkl ({len(ai_docs)} docs)")

        print(f"\nFingerprint analysis complete: {figure_dir}/")
        print(f"  JS divergence: {js_divergence:.4f}")
        print(f"  Cosine similarity: {cos_sim:.4f}")
        print(f"  Fingerprint correlation: {corr:.3f}")
        if scores is not None:
            print(f"  LogReg balanced accuracy: {100*scores.mean():.1f}% ± {100*scores.std():.1f}%")
            if mcc_scores is not None:
                print(f"  LogReg MCC:               {mcc_scores.mean():+.3f} ± {mcc_scores.std():.3f}")
        print()


if __name__ == "__main__":
    for method in methods_to_run:
        if not args.skip_fingerprint:
            run_fingerprint_analysis(method)

    print(f"\nFingerprint analysis complete for all requested methods")
