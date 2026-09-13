"""
Diversity metrics for recommendation evaluation.
All metrics operate on per-user top-K recommendation lists.
"""

import numpy as np
from collections import Counter
from typing import Dict, List, Optional


def coverage_at_k(topk_lists: Dict[int, List[int]], n_items: int) -> float:
    """Catalog coverage: fraction of items appearing in any user's top-K."""
    recommended = set()
    for items in topk_lists.values():
        recommended.update(items)
    return len(recommended) / n_items if n_items > 0 else 0.0


def gini_at_k(topk_lists: Dict[int, List[int]], n_items: int) -> float:
    """Gini coefficient of item recommendation frequency (0=equal, 1=concentrated)."""
    freq = np.zeros(n_items, dtype=np.float64)
    for items in topk_lists.values():
        for item in items:
            if 0 <= item < n_items:
                freq[item] += 1
    freq_sorted = np.sort(freq)
    n = len(freq_sorted)
    if n == 0 or freq_sorted.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * freq_sorted) - (n + 1) * np.sum(freq_sorted)) / (n * np.sum(freq_sorted)))


def entropy_at_k(topk_lists: Dict[int, List[int]], n_items: int) -> float:
    """Normalized Shannon entropy of item frequency distribution."""
    freq = np.zeros(n_items, dtype=np.float64)
    for items in topk_lists.values():
        for item in items:
            if 0 <= item < n_items:
                freq[item] += 1
    total = freq.sum()
    if total == 0:
        return 0.0
    prob = freq[freq > 0] / total
    raw_entropy = -np.sum(prob * np.log2(prob))
    max_entropy = np.log2(n_items) if n_items > 1 else 1.0
    return float(raw_entropy / max_entropy)


def ils_at_k(
    topk_lists: Dict[int, List[int]],
    embeddings: np.ndarray,
    use_text: bool = True,
) -> float:
    """
    Intra-List Similarity: mean pairwise cosine similarity within top-K lists.
    Lower = more diverse. embeddings shape: (n_items, dim).
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    normed = embeddings / norms

    user_ils_values = []
    for items in topk_lists.values():
        items = [i for i in items if 0 <= i < len(normed)]
        if len(items) < 2:
            continue
        vecs = normed[items]
        sim_matrix = vecs @ vecs.T
        k = len(items)
        upper_sum = (sim_matrix.sum() - np.trace(sim_matrix)) / 2
        n_pairs = k * (k - 1) / 2
        user_ils_values.append(float(upper_sum / n_pairs))
    return float(np.mean(user_ils_values)) if user_ils_values else 0.0


def novelty_at_k(
    topk_lists: Dict[int, List[int]],
    item_popularity: np.ndarray,
    n_users: int,
) -> float:
    """
    Average novelty: mean(-log2(pop(i)/n_users)) over all recommended items.
    Higher = more novel (less popular items recommended).
    """
    values = []
    for items in topk_lists.values():
        for item in items:
            if 0 <= item < len(item_popularity):
                pop = item_popularity[item]
                prob = max(pop, 1.0) / max(n_users, 1)
                values.append(-np.log2(prob))
    return float(np.mean(values)) if values else 0.0


def tail_coverage_at_k(
    topk_lists: Dict[int, List[int]],
    item_popularity: np.ndarray,
    tail_percentile: float = 80.0,
) -> float:
    """
    Coverage restricted to long-tail items (bottom tail_percentile% by popularity).
    """
    threshold = np.percentile(item_popularity[item_popularity > 0], 100 - tail_percentile)
    tail_items = set(np.where(item_popularity <= threshold)[0])
    if not tail_items:
        return 0.0
    recommended_tail = set()
    for items in topk_lists.values():
        for item in items:
            if item in tail_items:
                recommended_tail.add(item)
    return len(recommended_tail) / len(tail_items)


def aggregate_diversity(topk_lists: Dict[int, List[int]]) -> int:
    """Total number of unique items recommended across all users."""
    recommended = set()
    for items in topk_lists.values():
        recommended.update(items)
    return len(recommended)


def compute_item_popularity_array(train_data, n_items: int) -> np.ndarray:
    """Compute item interaction counts from training data."""
    pop = np.zeros(n_items, dtype=np.float64)
    if hasattr(train_data, "values"):
        pairs = train_data[["user_idx", "item_idx"]].values
    else:
        pairs = train_data
    for _, item_id in pairs:
        item_id = int(item_id)
        if 0 <= item_id < n_items:
            pop[item_id] += 1
    return pop


def compute_all_diversity_metrics(
    topk_lists: Dict[int, List[int]],
    n_items: int,
    n_users: int,
    item_popularity: np.ndarray,
    text_embeddings: Optional[np.ndarray] = None,
    cf_embeddings: Optional[np.ndarray] = None,
    k: int = 10,
) -> dict:
    """Compute all diversity metrics in one call."""
    results = {
        "coverage@k": coverage_at_k(topk_lists, n_items),
        "gini@k": gini_at_k(topk_lists, n_items),
        "entropy@k": entropy_at_k(topk_lists, n_items),
        "novelty@k": novelty_at_k(topk_lists, item_popularity, n_users),
        "tail_coverage@k": tail_coverage_at_k(topk_lists, item_popularity),
        "aggregate_diversity": aggregate_diversity(topk_lists),
        "k": k,
    }
    if text_embeddings is not None:
        results["ils_text@k"] = ils_at_k(topk_lists, text_embeddings)
    if cf_embeddings is not None:
        results["ils_cf@k"] = ils_at_k(topk_lists, cf_embeddings)
    return results
