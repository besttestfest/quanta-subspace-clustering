"""
Download and tokenize The Pile test set (200k documents).

Saves a HuggingFace Dataset to disk with tokenized fields needed
by the rest of the pipeline.

Output: {BASE_DIR}/data/the_pile_test_canonical_200k/
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                "..", "experiments", "clustering-0"))
from config import PATHS

from datasets import load_dataset
from transformers import AutoTokenizer

STEP = 143000
TOKENIZER_MODEL = "EleutherAI/pythia-19m"

parser = argparse.ArgumentParser(description="Download and tokenize The Pile test set")
parser.add_argument("--output-dir", default=PATHS["pile_canonical"],
                    help="Where to save the dataset")
parser.add_argument("--n-docs", type=int, default=200000,
                    help="Number of documents to download")
parser.add_argument("--force", action="store_true",
                    help="Overwrite if exists")
args = parser.parse_args()

if os.path.exists(args.output_dir) and not args.force:
    print(f"Dataset already exists at {args.output_dir}, skipping.")
    sys.exit(0)

print(f"Downloading The Pile test set ({args.n_docs} documents)...")
ds = load_dataset("EleutherAI/pile_val_test", split=f"test[:{args.n_docs}]")

print("Tokenizing with Pythia tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_MODEL, revision=f"step{STEP}")

def tokenize(sample):
    tokens = tokenizer(sample["text"], return_tensors="pt",
                       max_length=1024, truncation=True)["input_ids"]
    return {"input_ids": tokens}

ds = ds.map(tokenize)
ds = ds.map(lambda s: {"split_by_token": tokenizer.batch_decode(s["input_ids"][0])})
ds = ds.map(lambda s: {"tokens_len": len(s["input_ids"][0])})
ds = ds.map(lambda s: {"preds_len": max(s["tokens_len"] - 1, 0)})

os.makedirs(os.path.dirname(os.path.abspath(args.output_dir)), exist_ok=True)
ds.save_to_disk(args.output_dir)
print(f"Saved to {args.output_dir}")

# Verify
first_token = ds[0]["split_by_token"][0]
print(f"First token of first document: '{first_token}'")
print("Expected: 'Roman' (from 'Roman Catholic Diocese of Tambacounda')")
