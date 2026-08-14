"""
Load recommendation lists, item metadata, and popularity groups
from the existing diversity experiment infrastructure.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np

DIVERSITY_ROOT = Path("/root/autodl-tmp/diversity_experiment")
DATASETS_DIR = DIVERSITY_ROOT / "datasets"
RESULTS_DIR = DIVERSITY_ROOT / "diversity_results"


@dataclass
class DatasetInfo:
    name: str
    n_items: int
    n_users: int
    item_texts: Dict[int, str]
    popularity_groups: Optional[Dict[str, List[int]]]
    items_df: Optional[object]  # pandas DataFrame


def load_item_texts(dataset: str) -> Dict[int, str]:
    path = DATASETS_DIR / dataset / "item_texts.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def load_popularity_groups(dataset: str) -> Optional[Dict[str, List[int]]]:
    path = DATASETS_DIR / dataset / "item_popularity_groups.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        groups = pickle.load(f)
    result = {}
    for key in ["cold_items", "warm_items", "hot_items"]:
        if key in groups:
            items = groups[key]
            result[key] = list(items) if isinstance(items, set) else items
    return result


def load_topk_lists(dataset: str, model: str, seed: int = 42) -> Dict[int, List[int]]:
    path = RESULTS_DIR / dataset / model / f"topk_lists_seed{seed}.json"
    with open(path) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def load_diversity_metrics(dataset: str, model: str, seed: int = 42) -> dict:
    path = RESULTS_DIR / dataset / model / f"diversity_seed{seed}.json"
    with open(path) as f:
        return json.load(f)


def load_dataset_info(dataset: str) -> DatasetInfo:
    stats_path = DATASETS_DIR / dataset / "stats.json"
    n_items_path = DATASETS_DIR / dataset / "n_items.pkl"
    n_users_path = DATASETS_DIR / dataset / "n_users.pkl"

    with open(n_items_path, "rb") as f:
        n_items = pickle.load(f)
    with open(n_users_path, "rb") as f:
        n_users = pickle.load(f)

    item_texts = load_item_texts(dataset)
    pop_groups = load_popularity_groups(dataset)

    items_df = None
    items_path = DATASETS_DIR / dataset / "items.pkl"
    if items_path.exists():
        import pandas as pd
        items_df = pd.read_pickle(items_path)

    return DatasetInfo(
        name=dataset,
        n_items=n_items,
        n_users=n_users,
        item_texts=item_texts,
        popularity_groups=pop_groups,
        items_df=items_df,
    )


def get_available_experiments() -> Dict[str, List[str]]:
    """Return {dataset: [model1, model2, ...]} for all available topk_lists."""
    result = {}
    for ds_dir in sorted(RESULTS_DIR.iterdir()):
        if not ds_dir.is_dir():
            continue
        models = []
        for model_dir in sorted(ds_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            if list(model_dir.glob("topk_lists_*.json")):
                models.append(model_dir.name)
        if models:
            result[ds_dir.name] = models
    return result


def format_item_for_prompt(item_id: int, item_texts: Dict[int, str],
                           position: Optional[int] = None) -> str:
    text = item_texts.get(item_id, f"Item #{item_id}")
    if position is not None:
        return f"{position}. {text}"
    return text


def format_recommendation_list(item_ids: List[int],
                                item_texts: Dict[int, str],
                                numbered: bool = True) -> str:
    lines = []
    for i, item_id in enumerate(item_ids):
        text = item_texts.get(item_id, f"Item #{item_id}")
        if numbered:
            lines.append(f"{i+1}. {text}")
        else:
            lines.append(f"- {text}")
    return "\n".join(lines)


def build_user_profile_summary(user_id: int, dataset: str,
                                item_texts: Dict[int, str],
                                max_items: int = 10) -> str:
    """Build a text summary of user's training history."""
    import pandas as pd
    train_path = DATASETS_DIR / dataset / "train.pkl"
    train_data = pd.read_pickle(train_path)
    user_rows = train_data[train_data["user_idx"] == user_id]
    interacted = user_rows["item_idx"].tolist()[:max_items]
    if not interacted:
        return "This user has no recorded interaction history."
    items_str = "\n".join(
        f"- {item_texts.get(iid, f'Item #{iid}')}" for iid in interacted
    )
    return f"This user has previously interacted with:\n{items_str}"


if __name__ == "__main__":
    experiments = get_available_experiments()
    print("Available experiments:")
    for ds, models in experiments.items():
        print(f"  {ds}: {models}")

    info = load_dataset_info("ml-1m")
    print(f"\nML-1M: {info.n_users} users, {info.n_items} items, "
          f"{len(info.item_texts)} texts")
    if info.popularity_groups:
        for g, items in info.popularity_groups.items():
            print(f"  {g}: {len(items)} items")
