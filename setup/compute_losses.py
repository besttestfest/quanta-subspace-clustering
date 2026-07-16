"""
Compute per-token cross-entropy losses for Pythia models on The Pile.

Evaluates each model on 200k documents and saves losses as a numpy array.
Only pythia-19m is needed for token selection, but pythia-125m is also
evaluated as it is the second model used in this paper.

Adapted from original/experiments/pythia-2/eval.py (Michaud et al.)
with paths updated for this project.

Requires GPU. Runtime: ~1h per model on V100.

Output: {RESULTS_DIR}/losses_pythia-2.npy
"""

import argparse
import os
import sys

import numpy as np
import torch
from transformers import GPTNeoXForCausalLM, AutoTokenizer
import pyarrow as pa

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                "..", "experiments", "clustering-0"))
from config import PATHS

MODEL_NAMES = [
    "pythia-19m",
    "pythia-125m",
]

STEP = 143000

parser = argparse.ArgumentParser(description="Compute per-token losses for Pythia models")
parser.add_argument("--input-dir", default=PATHS["pile_canonical"],
                    help="Where to read the pile dataset from")
parser.add_argument("--output-path",
                    default=os.path.join(PATHS["results_dir"], "losses_pythia-2.npy"),
                    help="Where to save losses")
parser.add_argument("--n-docs", type=int, default=None,
                    help="Limit to first N documents (for smoke testing)")
parser.add_argument("--force", action="store_true",
                    help="Overwrite existing output")
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device.type == "cpu":
    print("WARNING: No GPU detected. This will be very slow.")

if os.path.exists(args.output_path) and not args.force:
    print(f"Losses already exist at {args.output_path}, skipping.")
    sys.exit(0)


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
        "text": table["text"].to_pylist(),
        "preds_len": table["preds_len"].to_pylist(),
        "_len": len(table),
    }


# Load tokenized Pile
print(f"Loading dataset from {args.input_dir}...")
dataset = load_pile_arrow(args.input_dir)

tokenizer = AutoTokenizer.from_pretrained(
    f"EleutherAI/{MODEL_NAMES[0]}",
    revision=f"step{STEP}",
    cache_dir=PATHS["pythia_cache"],
)

n_docs = min(args.n_docs, dataset["_len"]) if args.n_docs is not None else dataset["_len"]

all_losses = []

for model_name in MODEL_NAMES:
    print(f"\nEvaluating {model_name}...")
    model = GPTNeoXForCausalLM.from_pretrained(
        f"EleutherAI/{model_name}",
        revision=f"step{STEP}",
        cache_dir=os.path.join(PATHS["pythia_cache"], model_name,
                               f"step{STEP}"),
    ).to(device)

    model_losses = []
    for i in range(n_docs):
        text = dataset["text"][i] if dataset["text"][i] else ""
        if text:
            tokens = tokenizer(text, return_tensors="pt",
                               max_length=1024, truncation=True).to(device)
            with torch.no_grad():
                logits = model(**tokens).logits
            targets = tokens.input_ids
            losses = torch.nn.functional.cross_entropy(
                logits[0, :-1, :], targets[0, 1:], reduction="none")
            model_losses.append(losses.cpu().tolist())
        else:
            model_losses.append([])
        if i % 5000 == 0:
            print(f"  {i}/{n_docs}")

    # Flatten to 1D array (one loss per prediction)
    flat = []
    for doc_losses in model_losses:
        flat.extend(doc_losses)
    all_losses.append(flat)
    del model
    torch.cuda.empty_cache()

# Stack into (n_tokens, n_models) array. Models that have fewer total
# tokens are padded with NaN (not 0.0) so downstream filters that select
# zero-loss tokens (loss < 0.1) do not mistakenly include the padding.
max_len = max(len(l) for l in all_losses)
curves = np.full((max_len, len(MODEL_NAMES)), np.nan)
for j, flat in enumerate(all_losses):
    curves[:len(flat), j] = flat

os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
np.save(args.output_path, curves)
print(f"\nSaved losses to {args.output_path}")
print(f"Shape: {curves.shape} (tokens x models)")
