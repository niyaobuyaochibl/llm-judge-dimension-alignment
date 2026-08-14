"""
Configuration for the LLM-as-a-Judge Dimension Analysis experiments.
"""

from pathlib import Path

PROJECT_ROOT = Path("/root/autodl-tmp/llm_diversity_eval")
DIVERSITY_ROOT = Path("/root/autodl-tmp/diversity_experiment")
LLM_JUDGE_ROOT = Path("/root/autodl-tmp/llm_judge_recsys")
RESULTS_DIR = PROJECT_ROOT / "results"

MODEL_PATH_7B = "/root/autodl-tmp/models/Qwen/Qwen2.5-7B-Instruct"
MODEL_PATH_14B = "/root/autodl-tmp/models/Qwen2.5-14B-Instruct"

SEED = 42
N_USERS = 50

DATASETS = ["mind", "yelp", "amazon_books", "amazon_cds"]

DS_DIR_MAP = {
    "mind": "mind",
    "yelp": "yelp",
    "amazon_books": "amazon_books",
    "amazon_cds": "amazon_cds",
}

# Key model pairs per dataset: (model_a, model_b)
# Selected to cover: accuracy-wins, diversity-wins, and conflict scenarios
EXPERIMENT_PAIRS = {
    "mind": [
        ("gbaf", "fixed"),            # GBAF more accurate, Fixed more diverse
        ("gbaf", "gradnorm_only"),    # Similar accuracy, gradnorm much more diverse
        ("lightgcn_only", "fixed"),   # LightGCN more accurate but extremely low diversity
        ("gbaf", "pcgrad"),           # GBAF more accurate, PCGrad much more diverse
    ],
    "yelp": [
        ("gbaf", "fixed"),            # GBAF much more accurate, Fixed more diverse
        ("gbaf", "concat_mlp"),       # Similar accuracy, concat_mlp more diverse
        ("lightgcn_only", "fixed"),   # LightGCN more accurate, Fixed much more diverse
        ("concat_mlp", "gradnorm_only"),  # Similar accuracy, different diversity
    ],
    "amazon_books": [
        ("gbaf", "fixed"),            # GBAF more accurate, Fixed more diverse
        ("gradnorm_only", "concat_mlp"),  # Similar accuracy, concat more diverse
        ("lightgcn_only", "fixed"),   # LightGCN more accurate, Fixed much more diverse
        ("gbaf", "concat_mlp"),       # GBAF more accurate, concat more diverse
    ],
    "amazon_cds": [
        ("gbaf", "fixed"),            # GBAF much more accurate, Fixed more diverse
        ("gbaf", "concat_mlp"),       # GBAF more accurate, concat more diverse
        ("gradnorm_only", "concat_mlp"),  # Similar accuracy, different diversity
        ("lightgcn_only", "fixed"),   # LightGCN more accurate, Fixed much more diverse
    ],
}
