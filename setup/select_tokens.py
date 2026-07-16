"""
Select 10,000 zero-loss non-induction tokens for QDG clustering.

Finds tokens where pythia-19m achieves < 0.1 nats cross-entropy,
filters out tokens predictable by trigram induction, and samples
every 50th remaining token.

Adapted from original/scripts/zero_and_induction_idxs.py (Michaud et al.)
with paths updated for this project.

Input:  losses_pythia-2.npy (from compute_losses.py)
        the_pile_test_canonical_200k/ (from download_pile.py)
Output: tmp/zero_and_induction_idxs.pkl
"""

import argparse
import os
import sys
import pickle
from collections import defaultdict

import numpy as np
import datasets
from tqdm.auto import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                "..", "experiments", "clustering-0"))
from config import PATHS

parser = argparse.ArgumentParser(description="Select zero-loss non-induction tokens")
parser.add_argument("--force", action="store_true",
                    help="Overwrite existing output")
args = parser.parse_args()

output_path = PATHS["zero_induction_idxs"]
if os.path.exists(output_path) and not args.force:
    print(f"Token indices already exist at {output_path}, skipping.")
    sys.exit(0)

# Load dataset
print(f"Loading dataset from {PATHS['pile_canonical']}...")
dataset = datasets.load_from_disk(PATHS["pile_canonical"])

# Load losses (column 0 = pythia-19m)
losses_path = os.path.join(PATHS["results_dir"], "losses_pythia-2.npy")
print(f"Loading losses from {losses_path}...")
curves = np.load(losses_path)

# Tokens with near-zero loss on pythia-19m
zero_idxs = (curves[:, 0] < 0.1).nonzero()[0]
print(f"Tokens with loss < 0.1: {len(zero_idxs)}")

# Find induction tokens (predictable by trigram repetition)
print("Scanning for induction tokens...")
induction_idxs = []
i = 0
for doc_idx in tqdm(range(len(dataset))):
    tokens = dataset[doc_idx]["input_ids"][0]
    if len(tokens) > 1:
        i += 1  # skip first token (no prediction)
        document_trigrams = defaultdict(int)
        for j in range(2, len(tokens)):
            trigram = tuple(tokens[j-2:j+1])
            if trigram in document_trigrams:
                induction_idxs.append(i)
            document_trigrams[trigram] += 1
            i += 1
    # Note: documents with 0 or 1 tokens have preds_len=0 and contribute
    # nothing to the global flat index (both are skipped by len > 1).

# Filter: zero-loss tokens that are NOT induction
non_induction_zeros = sorted(set(zero_idxs) - set(induction_idxs))
print(f"Non-induction zero-loss tokens: {len(non_induction_zeros)}")
print(f"Sampled every 50th: {len(non_induction_zeros[::50])}")

os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "wb") as f:
    pickle.dump((non_induction_zeros, zero_idxs, induction_idxs), f)

print(f"Saved to {output_path}")
