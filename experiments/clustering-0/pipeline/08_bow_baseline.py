"""Bag-of-Words (TF-IDF) baseline for AI-vs-human classification.

Fair comparison against the quanta fingerprint classifier (pipeline/05_fingerprint.py):
uses the SAME documents (784 human, ~497 AI; thesis Table 6 / Appendix D), the same
5-seed x 5-fold stratified CV protocol, and the same L2 logistic-regression setup as
pipeline/07_fingerprint_robustness.py.

Documents:
  - Both classes are taken from <results_dir>/classification_docs.json, the exact
    document selection recorded by pipeline/05_fingerprint.py, so counts and
    membership match the fingerprint experiment one-to-one.
  - Fallback (classification_docs.json absent): human docs are those spanned by the
    10,000 zero-loss tokens with >= MIN_TOKENS_PER_DOC of them; AI docs are the
    shared corpus (shared_ai_docs.pkl) filtered by raw length, generated here only
    if the shared file does not exist yet.
TF-IDF vocabulary and IDF weights are fitted inside each training fold only.

Usage (UCloud, GPU node - run pipeline/05_fingerprint.py first so shared_ai_docs.pkl exists):
    cd experiments/clustering-0
    QDG_MODEL=pythia-19m  python pipeline/08_bow_baseline.py
    QDG_MODEL=pythia-125m python pipeline/08_bow_baseline.py

Output:
    results-mirror/bow_baseline/<model>_bow.json
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pickle
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")

import numpy as np
import pyarrow as pa
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, GPTNeoXForCausalLM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from config import PATHS, MODEL_NAME


def load_pile_arrow(pile_path):
    """Read Pile Arrow files directly without the datasets library.

    Returns a dict with keys 'input_ids' (list of lists) and 'preds_len'
    (list of ints), mimicking the subset of the datasets API used here.
    """
    arrow_files = sorted(
        f for f in os.listdir(pile_path) if f.endswith(".arrow")
    )
    if not arrow_files:
        raise FileNotFoundError(f"No .arrow files found in {pile_path}")

    tables = []
    for fname in arrow_files:
        fpath = os.path.join(pile_path, fname)
        with open(fpath, "rb") as src:
            reader = pa.ipc.open_stream(src)
            tables.append(reader.read_all())

    table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]

    # Convert to plain Python lists once so indexing is O(1)
    input_ids = table["input_ids"].to_pylist()
    preds_len = table["preds_len"].to_pylist()
    return {"input_ids": input_ids, "preds_len": preds_len, "_len": len(preds_len)}

N_AI_DOCS        = 600        # generate this many; select those with ≥ MIN_TOKENS
MAX_DOC_LEN      = 256
TEMPERATURE      = 1.0
LOSS_THRESHOLD   = 0.1
MIN_TOKENS_PER_DOC = 3
N_FEATURES       = 10_000     # TF-IDF vocabulary size
N_SEEDS          = 5
N_FOLDS          = 5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"TF-IDF BoW baseline: {MODEL_NAME}")
print(f"  device: {device}")

print("\nLoading Pile dataset and token indices...")
dataset       = load_pile_arrow(PATHS["pile_canonical"])
starting_idxs = np.array([0] + list(np.cumsum(dataset["preds_len"])))

with open(PATHS["zero_induction_idxs"], "rb") as f:
    non_induction_zeros, _, _ = pickle.load(f)

idxs = non_induction_zeros[::50][:10_000]

def flat_to_doc(flat_idx):
    doc  = int(np.searchsorted(starting_idxs, flat_idx, side="right") - 1)
    pos  = flat_idx - starting_idxs[doc]
    return doc, int(pos)

print("Reconstructing human documents from zero-loss token indices...")
doc_to_tokens = defaultdict(list)
for i, flat_idx in enumerate(idxs):
    doc_idx, _ = flat_to_doc(flat_idx)
    doc_to_tokens[doc_idx].append(i)

# Use the exact document selection recorded by pipeline/05_fingerprint.py
# when available, so both classifiers see the identical corpus.
doc_sel = None
doc_sel_path = os.path.join(PATHS["results_dir"], "classification_docs.json")
if os.path.exists(doc_sel_path):
    with open(doc_sel_path) as f:
        doc_sel = json.load(f)
    print(f"  Loaded document selection: {doc_sel_path}")

if doc_sel is not None:
    human_doc_idxs = [int(d) for d in doc_sel["human_doc_idxs"]]
else:
    human_doc_idxs = [d for d, toks in doc_to_tokens.items()
                      if len(toks) >= MIN_TOKENS_PER_DOC]
print(f"  Human docs with ≥{MIN_TOKENS_PER_DOC} zero-loss tokens: {len(human_doc_idxs)}")

step = 143000
print(f"\nLoading {MODEL_NAME} tokenizer (step {step})...")
tokenizer = AutoTokenizer.from_pretrained(
    f"EleutherAI/{MODEL_NAME}",
    revision=f"step{step}",
    cache_dir=f"{PATHS['pythia_cache']}/{MODEL_NAME}/step{step}",
)

# Reuse the SAME AI corpus as the fingerprint experiment (pipeline/05_fingerprint.py)
# so the BoW baseline is computed on identical documents (thesis Table 6 / Appendix D).
# Generate the corpus here only as a fallback if the shared file does not exist yet.
shared_ai_path = os.path.join(PATHS["results_dir"], "shared_ai_docs.pkl")
if os.path.exists(shared_ai_path):
    print(f"  Loading shared AI docs from: {shared_ai_path}")
    with open(shared_ai_path, "rb") as f:
        ai_doc_ids = [list(d) for d in pickle.load(f)["docs"]]
    print(f"  Loaded {len(ai_doc_ids)} shared AI documents")
else:
    print(f"  shared_ai_docs.pkl not found - generating {N_AI_DOCS} AI documents "
          f"(seed=42, temp={TEMPERATURE})...")
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
    model.eval()
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    ai_doc_ids = []
    for _ in tqdm(range(N_AI_DOCS), desc="Generating"):
        rand_doc_idx = int(np.random.randint(0, dataset["_len"]))
        prompt_ids_raw = dataset["input_ids"][rand_doc_idx]
        # input_ids may be stored as [[...]] (nested list) or [...]
        prompt_ids = (prompt_ids_raw[0] if isinstance(prompt_ids_raw[0], list)
                      else prompt_ids_raw)[:10]
        input_ids   = torch.tensor([prompt_ids], device=device)
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=MAX_DOC_LEN,
                temperature=TEMPERATURE,
                do_sample=True,
                top_k=0,
                pad_token_id=tokenizer.eos_token_id,
            )
        ai_doc_ids.append(output[0].cpu().tolist())
    print(f"  Generated {len(ai_doc_ids)} AI documents")

# Qualifying AI docs: prefer the selection recorded by the fingerprint
# experiment; otherwise fall back to the minimum-length criterion.
if doc_sel is not None:
    ai_qualifying = [ai_doc_ids[int(i)] for i in doc_sel["ai_doc_idxs"]]
    print(f"  AI docs (from classification_docs.json): {len(ai_qualifying)}")
else:
    ai_qualifying = [ids for ids in ai_doc_ids if len(ids) >= MIN_TOKENS_PER_DOC + 10]
    print(f"  AI docs qualifying (length ≥ {MIN_TOKENS_PER_DOC+10} tokens): {len(ai_qualifying)}")

print("\nDecoding documents to text...")
human_texts = []
for doc_idx in tqdm(human_doc_idxs, desc="Human"):
    ids = dataset["input_ids"][doc_idx]
    # input_ids may be stored as [[...]] (nested list) or [...]
    if isinstance(ids[0], list):
        ids = ids[0]
    human_texts.append(tokenizer.decode(ids, skip_special_tokens=True))

ai_texts = []
for ids in tqdm(ai_qualifying, desc="AI"):
    ai_texts.append(tokenizer.decode(ids, skip_special_tokens=True))

n_human = len(human_texts)
n_ai    = len(ai_texts)
n_total = n_human + n_ai
print(f"  Final dataset: {n_human} human + {n_ai} AI = {n_total} documents")

print(f"\nTF-IDF (max_features={N_FEATURES}) + L2 LogReg, fitted per training fold...")
texts = np.array(human_texts + ai_texts, dtype=object)
y     = np.array([0] * n_human + [1] * n_ai)


def make_clf(C=1.0):
    """TF-IDF + L2 LogReg pipeline; vocabulary/IDF are fitted on training folds only."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=N_FEATURES,
            sublinear_tf=True,
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"\b\w+\b",
            ngram_range=(1, 1),
        )),
        ("clf", LogisticRegression(
            penalty="l2", C=C, solver="lbfgs",
            max_iter=2000, class_weight="balanced",
        )),
    ])


def cv_metrics(texts_in, y_in, n_seeds, n_folds, C=1.0):
    """Same CV protocol as pipeline/07_fingerprint_robustness.py.
    Returns mean/std (and per-seed) for both MCC and balanced accuracy."""
    mcc_seed_means, bal_seed_means = [], []
    for seed in range(n_seeds):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        fold_mccs, fold_bals = [], []
        for tr, te in skf.split(texts_in, y_in):
            clf = make_clf(C)
            clf.fit(list(texts_in[tr]), y_in[tr])
            pred = clf.predict(list(texts_in[te]))
            fold_mccs.append(matthews_corrcoef(y_in[te], pred))
            fold_bals.append(balanced_accuracy_score(y_in[te], pred))
        mcc_seed_means.append(float(np.mean(fold_mccs)))
        bal_seed_means.append(float(np.mean(fold_bals)))
    return (float(np.mean(mcc_seed_means)), float(np.std(mcc_seed_means)), mcc_seed_means,
            float(np.mean(bal_seed_means)), float(np.std(bal_seed_means)), bal_seed_means)

print(f"\nRunning {N_SEEDS} seeds × {N_FOLDS}-fold CV...")
mcc_mean, mcc_std, mcc_per_seed, bal_mean, bal_std, bal_per_seed = cv_metrics(texts, y, N_SEEDS, N_FOLDS)

print(f"\nTF-IDF BoW n={n_total} ({n_human} human + {n_ai} AI)")
print(f"  MCC      = {mcc_mean:+.4f} ± {mcc_std:.4f}")
print(f"  Bal. acc = {bal_mean:.4f} ± {bal_std:.4f}")
print(f"  Per-seed MCC: {[f'{m:+.4f}' for m in mcc_per_seed]}")

out = {
    "model":         MODEL_NAME,
    "n_human":       n_human,
    "n_ai":          n_ai,
    "n_total":       n_total,
    "n_features":    N_FEATURES,
    "n_seeds":       N_SEEDS,
    "n_folds":       N_FOLDS,
    "mcc_mean":      mcc_mean,
    "mcc_std":       mcc_std,
    "mcc_per_seed":  mcc_per_seed,
    "bal_acc_mean":  bal_mean,
    "bal_acc_std":   bal_std,
    "bal_acc_per_seed": bal_per_seed,
}

mirror_dir = os.path.join(PATHS["repo_dir"], "results-mirror", "bow_baseline")
os.makedirs(mirror_dir, exist_ok=True)
out_path = os.path.join(mirror_dir, f"{MODEL_NAME}_bow.json")
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {out_path}")
