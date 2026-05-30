import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoTokenizer, GPTNeoXForCausalLM

import pyarrow as pa
from config import PATHS

parser = argparse.ArgumentParser(
    description="Compute or merge similarity matrices"
)
parser.add_argument(
    "--merge-parts",
    action="store_true",
    help="Merge partial similarity matrices from multi-GPU computation"
)
parser.add_argument(
    "--n-gpus",
    type=int,
    default=4,
    help="Number of GPUs (for --merge-parts)"
)
parser.add_argument(
    "--n-tokens",
    type=int,
    default=None,
    help="Use first N tokens instead of 10000 (for smoke testing)"
)
parser.add_argument(
    "--output-path",
    default=None,
    help="Override PATHS['similarity_matrix'] for output"
)
args = parser.parse_args()

if args.merge_parts:
    print(f"Merging {args.n_gpus} partial similarity matrices...")
    base_path = PATHS["similarity_matrix"]
    
    C_total = None
    C_abs_total = None
    idxs = None
    
    for g in range(args.n_gpus):
        part_path = base_path.replace(".pt", f"_gpu{g}.pt")
        print(f"  Loading {part_path}")
        idxs_g, C_g, C_abs_g = torch.load(part_path, weights_only=False)
        
        if C_total is None:
            idxs = idxs_g
            C_total = C_g
            C_abs_total = C_abs_g
        else:
            C_total += C_g
            C_abs_total += C_abs_g
    
    print(f"Saving merged result to {base_path}")
    torch.save((idxs, C_total, C_abs_total), base_path)
    print("Done.")
    exit(0)


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
    result = {
        "text": table["text"].to_pylist(),
        "preds_len": table["preds_len"].to_pylist(),
        "_len": len(table),
    }
    if "split_by_token" in table.schema.names:
        result["split_by_token"] = table["split_by_token"].to_pylist()
    return result


dataset = load_pile_arrow(PATHS["pile_canonical"])
starting_indexes = np.array([0] + list(np.cumsum(dataset["preds_len"])))


def _row(idx):
    row = {"text": dataset["text"][idx], "preds_len": dataset["preds_len"][idx]}
    if "split_by_token" in dataset:
        row["split_by_token"] = dataset["split_by_token"][idx]
    return row


def loss_idx_to_dataset_idx(idx):
    """Map loss index to sample index and position within sample.

    Args:
        idx: Index in range [0, 10658635].

    Returns:
        Tuple of (sample_index in [0, 20000], pred_in_sample_index in [0, 1023]).
    """
    sample_index = np.searchsorted(starting_indexes, idx, side="right") - 1
    pred_in_sample_index = idx - starting_indexes[sample_index]
    return int(sample_index), int(pred_in_sample_index)


def get_context(idx):
    """Get dataset sample and predicted token index.

    Args:
        idx: Index in range [0, 10658635].

    Returns:
        Tuple of (sample, token_idx in [1, 1024]).
    """
    sample_index, pred_index = loss_idx_to_dataset_idx(idx)
    return _row(sample_index), pred_index + 1



device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'
model_name = PATHS["model_name"]
step = 143000

model = GPTNeoXForCausalLM.from_pretrained(
    f"EleutherAI/{model_name}",
    revision=f"step{step}",
    cache_dir=f"{PATHS['pythia_cache']}/{model_name}/step{step}",
).to(device)

tokenizer = AutoTokenizer.from_pretrained(
    f"EleutherAI/{model_name}",
    revision=f"step{step}",
    cache_dir=f"{PATHS['pythia_cache']}/{model_name}/step{step}",
)

with open(PATHS["zero_induction_idxs"], "rb") as f:
    non_induction_zeros, zero_idxs, induction_idxs = pickle.load(f)

print(f"zero_idxs: {len(zero_idxs)}")
print(f"induction_idxs: {len(induction_idxs)}")
print(f"non_induction_zeros: {len(non_induction_zeros)}")


def get_flattened_gradient(model, param_subset):
    """Flatten and concatenate gradients for specified parameters."""
    grads = []
    for name, p in model.named_parameters():
        if name in param_subset:
            grads.append(p.grad)
    return torch.cat([g.flatten() for g in grads])


# Exclude layernorm and embeddings from gradient computation
param_names = [n for n, p in model.named_parameters()]
highsignal_names = [
    name for name in param_names
    if ('layernorm' not in name) and ('embed' not in name)
]

len_g = sum(model.state_dict()[name].numel() for name in highsignal_names)

n_tokens = args.n_tokens if args.n_tokens is not None else 10000
# stride 50: spreads samples across the dataset instead of clustering them
idxs = non_induction_zeros[::50][:n_tokens]
S = len(idxs)
print(f"Number of samples: {S}")

block_len = int(os.environ.get("QDG_BLOCK_LEN", 250))  # lower (e.g. QDG_BLOCK_LEN=25) if step 01 runs out of GPU memory on larger models
blocks = [idxs[i:min(len(idxs), i + block_len)] for i in range(0, len(idxs), block_len)]

C = torch.zeros((S, S), device="cpu")
C_abs = torch.zeros((S, S), device="cpu")

iouter = 0
for i_index, iblock in enumerate(tqdm(blocks)):
    Gi = torch.zeros((len(iblock), len_g), device=device)

    for i, idx in enumerate(iblock):
        model.zero_grad()
        document, l = get_context(idx)
        prompt = document['text']
        tokens = tokenizer(prompt, return_tensors='pt', max_length=1024, truncation=True).to(device)
        logits = model(**tokens).logits
        targets = tokens.input_ids
        ls = F.cross_entropy(logits[0, :-1, :], targets[0, 1:], reduction='none')
        ls_l = ls[l - 1]  # l is 1-based, ls is 0-based
        ls_l.backward()
        g = get_flattened_gradient(model, highsignal_names)
        Gi[i] = g

    Gi = F.normalize(Gi, p=2, dim=1)

    j_index = i_index
    jouter = iouter  # diagonal starts at the same offset as the outer block

    for jblock in tqdm(blocks[j_index:], leave=False):
        Gj = torch.zeros((len(jblock), len_g), device=device)

        for j, idx in enumerate(jblock):
            model.zero_grad()
            document, l = get_context(idx)
            prompt = document['text']
            tokens = tokenizer(prompt, return_tensors='pt', max_length=1024, truncation=True).to(device)
            logits = model(**tokens).logits
            targets = tokens.input_ids
            ls = F.cross_entropy(logits[0, :-1, :], targets[0, 1:], reduction='none')
            ls_l = ls[l - 1]
            ls_l.backward()
            g = get_flattened_gradient(model, highsignal_names)
            Gj[j] = g

        Gj = F.normalize(Gj, p=2, dim=1)

        Cij = torch.matmul(Gi, Gj.T)
        C[iouter:iouter + len(iblock), jouter:jouter + len(jblock)] = Cij.cpu()
        C[jouter:jouter + len(jblock), iouter:iouter + len(iblock)] = Cij.T.cpu()

        # element-wise absolute similarity; Gi is already unit-normalized
        Cij_abs = torch.matmul(Gi.abs(), Gj.T.abs())
        C_abs[iouter:iouter + len(iblock), jouter:jouter + len(jblock)] = Cij_abs.cpu()
        C_abs[jouter:jouter + len(jblock), iouter:iouter + len(iblock)] = Cij_abs.T.cpu()

        jouter += len(jblock)

    iouter += len(iblock)

output_path = args.output_path if args.output_path is not None else PATHS["similarity_matrix"]
if args.output_path is not None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
torch.save((idxs, C, C_abs), output_path)
