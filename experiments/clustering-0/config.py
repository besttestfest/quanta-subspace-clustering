"""
Configuration for the QDG clustering pipeline.

Usage:
    from config import PATHS
    QDG_MODEL=pythia-125m python pipeline/01_similarity_matrix.py
"""

import os

ENV = os.environ.get("QDG_ENV", "ucloud")
MODEL_NAME = os.environ.get("QDG_MODEL", "pythia-19m")

if ENV == "ucloud":
    BASE_DIR = "/work/Master"
elif ENV == "local":
    BASE_DIR = os.path.expanduser("~/Master")
else:
    raise ValueError(
        f"Unknown QDG_ENV={ENV!r}. Set QDG_ENV=ucloud or QDG_ENV=local."
    )

# Optional override - useful for testing from scratch without touching existing results:
#   export QDG_BASE_DIR=/work/fresh-test
if os.environ.get("QDG_BASE_DIR"):
    BASE_DIR = os.environ["QDG_BASE_DIR"]

# Repository root. Defaults to <BASE_DIR>/quantization-model for existing
# setups; falls back to the checkout containing this file (so the pipeline
# works regardless of the clone's directory name). Override with QDG_REPO_DIR.
_CHECKOUT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO_DIR = os.environ.get("QDG_REPO_DIR", f"{BASE_DIR}/quantization-model")
if not os.path.isdir(REPO_DIR):
    REPO_DIR = _CHECKOUT_DIR
PILE_CANONICAL = f"{BASE_DIR}/data/the_pile_test_canonical_200k"
PYTHIA_CACHE = f"{BASE_DIR}/models"

if MODEL_NAME == "pythia-19m":
    RESULTS_DIR = f"{BASE_DIR}/results/clustering-0"
else:
    RESULTS_DIR = f"{BASE_DIR}/results/clustering-0-{MODEL_NAME}"

PATHS = {
    "base_dir":          BASE_DIR,
    "repo_dir":          REPO_DIR,
    "results_dir":       RESULTS_DIR,
    "pile_canonical":    PILE_CANONICAL,
    "pythia_cache":      PYTHIA_CACHE,
    "model_name":        MODEL_NAME,
    "similarity_matrix": f"{RESULTS_DIR}/full_more.pt",
    "clusters_output":   f"{RESULTS_DIR}/clusters_full_more.pkl",
    "clusters_ssc_output": f"{RESULTS_DIR}/clusters_ssc_full_more.pkl",
    "clusters_ssc_omp_output": f"{RESULTS_DIR}/clusters_ssc_full_more_omp.pkl",
    "clusters_hierarchical_output": f"{RESULTS_DIR}/clusters_hierarchical.pkl",
    "zero_induction_idxs": f"{REPO_DIR}/tmp/zero_and_induction_idxs.pkl",
}

os.makedirs(RESULTS_DIR, exist_ok=True)

if __name__ == "__main__":
    print(f"Environment: {ENV}, Model: {MODEL_NAME}")
    for key, path in PATHS.items():
        status = "OK" if os.path.exists(path) else "MISSING"
        print(f"  [{status}] {key}: {path}")
