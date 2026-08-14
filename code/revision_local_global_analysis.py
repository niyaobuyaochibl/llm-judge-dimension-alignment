"""
Revision analyses for the TKDE major revision.

This script separates prompt-visible/list-level signals from global
system-level metrics. It is designed to address the construct-validity
comments about using Coverage as the main diversity target when the LLM
judge only observes one user's recommendation lists.

Outputs are written to:
  /root/autodl-tmp/llm_diversity_eval/revision_results/
"""

import json
import math
import pickle
import sys
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path("/root/autodl-tmp/llm_judge_recsys/code")))

from experiment_config import DATASETS, RESULTS_DIR, DIVERSITY_ROOT  # noqa: E402
from data_loader import load_diversity_metrics, load_item_texts, load_topk_lists  # noqa: E402


OUT_DIR = Path("/root/autodl-tmp/llm_diversity_eval/revision_results")
DATASETS_DIR = DIVERSITY_ROOT / "datasets"


PROMPTS = ["balanced", "relevance", "diversity", "novelty"]

LOCAL_METRICS = [
    "local_text_distance",
    "local_category_count",
    "local_category_entropy",
    "description_length",
    "history_similarity",
    "average_popularity",
]

GLOBAL_METRICS = [
    "ndcg@10",
    "coverage@k",
    "novelty@k",
    "entropy@k",
    "ils_text@k",
]


def load_primary_results():
    """Load Qwen-7B result files, excluding aggregate and validation files."""
    rows = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        if path.name.startswith("all_results"):
            continue
        if "_14b" in path.name or "_mistral" in path.name or "_deepseek" in path.name:
            continue
        with open(path) as f:
            rows.append(json.load(f))
    return rows


def text_word_count(text):
    return len(str(text).split())


def entropy(values):
    values = [v for v in values if v]
    if not values:
        return np.nan
    counts = np.array(list(Counter(values).values()), dtype=float)
    probs = counts / counts.sum()
    return float(-(probs * np.log2(probs)).sum())


def parse_categories(dataset, text):
    """Extract prompt-visible category/topic labels from item text when present."""
    parts = [p.strip().lower() for p in str(text).split("|")]
    if dataset == "mind":
        # MIND texts are usually: title | abstract | category | subcategory.
        return [p for p in parts[2:] if p]
    if dataset == "yelp":
        # Yelp texts are usually: business name | comma-separated categories | city.
        if len(parts) >= 2:
            return [p.strip() for p in parts[1].split(",") if p.strip()]
        return []
    # Amazon product texts often do not expose clean categories in this processed
    # version. Use trailing pipe-delimited fields only when they exist.
    if len(parts) >= 3:
        return [p for p in parts[2:] if p]
    return []


class DatasetLocalAnalyzer:
    def __init__(self, dataset):
        self.dataset = dataset
        self.item_texts = load_item_texts(dataset)
        self.item_ids = sorted(self.item_texts)
        self.texts = [str(self.item_texts[i]) for i in self.item_ids]
        self.id_to_row = {item_id: idx for idx, item_id in enumerate(self.item_ids)}
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=50000,
            min_df=1,
            token_pattern=r"(?u)\b\w\w+\b",
        )
        self.tfidf = self.vectorizer.fit_transform(self.texts)
        self.train_history = self._load_train_history()
        self.item_popularity = self._load_item_popularity()

    def _load_train_history(self):
        path = DATASETS_DIR / self.dataset / "train.pkl"
        train = pd.read_pickle(path)
        grouped = train.groupby("user_idx")["item_idx"].apply(list)
        return {int(k): [int(x) for x in v] for k, v in grouped.items()}

    def _load_item_popularity(self):
        path = DATASETS_DIR / self.dataset / "train.pkl"
        train = pd.read_pickle(path)
        counts = train.groupby("item_idx").size()
        return {int(k): int(v) for k, v in counts.items()}

    def _rows_for_items(self, item_ids):
        return [self.id_to_row[i] for i in item_ids if i in self.id_to_row]

    def mean_pairwise_distance(self, item_ids):
        rows = self._rows_for_items(item_ids)
        if len(rows) < 2:
            return np.nan
        sim = cosine_similarity(self.tfidf[rows])
        upper = sim[np.triu_indices_from(sim, k=1)]
        return float(1.0 - np.mean(upper))

    def history_similarity(self, user_id, item_ids, max_history=10):
        list_rows = self._rows_for_items(item_ids)
        hist_items = self.train_history.get(int(user_id), [])[:max_history]
        hist_rows = self._rows_for_items(hist_items)
        if not list_rows or not hist_rows:
            return np.nan
        sim = cosine_similarity(self.tfidf[list_rows], self.tfidf[hist_rows])
        return float(np.mean(sim))

    def average_popularity(self, item_ids):
        counts = [self.item_popularity.get(int(i), 0) for i in item_ids]
        if not counts:
            return np.nan
        return float(np.mean(np.log1p(counts)))

    def list_metrics(self, user_id, item_ids):
        item_ids = [int(i) for i in item_ids[:10]]
        texts = [self.item_texts.get(i, "") for i in item_ids]
        cats = []
        for text in texts:
            cats.extend(parse_categories(self.dataset, text))

        return {
            "local_text_distance": self.mean_pairwise_distance(item_ids),
            "local_category_count": float(len(set(cats))) if cats else np.nan,
            "local_category_entropy": entropy(cats),
            "description_length": float(np.mean([text_word_count(t) for t in texts])) if texts else np.nan,
            "history_similarity": self.history_similarity(user_id, item_ids),
            "average_popularity": self.average_popularity(item_ids),
        }

    def dataset_richness(self):
        lengths = np.array([text_word_count(t) for t in self.texts], dtype=float)
        category_counts = []
        category_pool = []
        for text in self.texts:
            cats = parse_categories(self.dataset, text)
            category_counts.append(len(set(cats)))
            category_pool.extend(cats)
        category_counts = np.array(category_counts, dtype=float)
        return {
            "dataset": self.dataset,
            "n_items": len(self.item_ids),
            "mean_item_words": float(np.mean(lengths)),
            "median_item_words": float(np.median(lengths)),
            "p90_item_words": float(np.percentile(lengths, 90)),
            "items_with_categories_pct": float(np.mean(category_counts > 0) * 100.0),
            "mean_categories_per_item": float(np.mean(category_counts)),
            "unique_category_tokens": int(len(set(category_pool))),
            "category_entropy": entropy(category_pool),
        }


@lru_cache(maxsize=None)
def get_analyzer(dataset):
    return DatasetLocalAnalyzer(dataset)


@lru_cache(maxsize=None)
def get_topk(dataset, model):
    return load_topk_lists(dataset, model, 42)


@lru_cache(maxsize=None)
def get_global_metrics(dataset, model):
    return load_diversity_metrics(dataset, model, 42)


@lru_cache(maxsize=None)
def get_list_metrics(dataset, model, user_id):
    lists = get_topk(dataset, model)
    item_ids = lists[int(user_id)]
    return get_analyzer(dataset).list_metrics(int(user_id), tuple(item_ids))


def winner_from_values(model_a, val_a, model_b, val_b, higher_is_better=True):
    if pd.isna(val_a) or pd.isna(val_b):
        return "NA"
    if math.isclose(float(val_a), float(val_b), rel_tol=1e-12, abs_tol=1e-12):
        return "TIE"
    if higher_is_better:
        return model_a if val_a > val_b else model_b
    return model_a if val_a < val_b else model_b


def llm_pair_winner(result):
    if result["prefer_a"] > result["prefer_b"]:
        return result["model_a"]
    if result["prefer_b"] > result["prefer_a"]:
        return result["model_b"]
    return "TIE"


def is_conflict(result):
    return result["ndcg_winner"] != result["cov_winner"]


def pair_metric_means(result, metric):
    dataset = result["dataset"]
    model_a = result["model_a"]
    model_b = result["model_b"]
    user_ids = [int(t["user_id"]) for t in result["per_trial"]]

    vals_a = np.array([get_list_metrics(dataset, model_a, uid)[metric] for uid in user_ids], dtype=float)
    vals_b = np.array([get_list_metrics(dataset, model_b, uid)[metric] for uid in user_ids], dtype=float)
    mean_a = float(np.nanmean(vals_a)) if np.any(~np.isnan(vals_a)) else np.nan
    mean_b = float(np.nanmean(vals_b)) if np.any(~np.isnan(vals_b)) else np.nan
    return mean_a, mean_b


def build_pair_rows(results):
    rows = []
    for r in results:
        row = {
            "dataset": r["dataset"],
            "model_a": r["model_a"],
            "model_b": r["model_b"],
            "dimension": r["dimension"],
            "is_conflict": is_conflict(r),
            "llm_winner": llm_pair_winner(r),
            "ndcg_winner": r["ndcg_winner"],
            "coverage_winner": r["cov_winner"],
            "prefer_a": r["prefer_a"],
            "prefer_b": r["prefer_b"],
            "tie": r["tie"],
            "consistency_rate": r["consistency_rate"],
        }
        row["align_ndcg"] = int(row["llm_winner"] == row["ndcg_winner"]) if row["llm_winner"] != "TIE" else np.nan
        row["align_coverage"] = int(row["llm_winner"] == row["coverage_winner"]) if row["llm_winner"] != "TIE" else np.nan

        gt_a = get_global_metrics(r["dataset"], r["model_a"])
        gt_b = get_global_metrics(r["dataset"], r["model_b"])
        for metric in GLOBAL_METRICS:
            higher = metric != "ils_text@k"
            winner = winner_from_values(
                r["model_a"], gt_a.get(metric, np.nan),
                r["model_b"], gt_b.get(metric, np.nan),
                higher_is_better=higher,
            )
            safe_metric = metric.replace("@", "_at_").replace("/", "_")
            row[f"{safe_metric}_a"] = gt_a.get(metric, np.nan)
            row[f"{safe_metric}_b"] = gt_b.get(metric, np.nan)
            row[f"{safe_metric}_winner"] = winner
            row[f"align_{safe_metric}"] = int(row["llm_winner"] == winner) if winner not in {"NA", "TIE"} else np.nan

        for metric in LOCAL_METRICS:
            val_a, val_b = pair_metric_means(r, metric)
            winner = winner_from_values(r["model_a"], val_a, r["model_b"], val_b)
            row[f"{metric}_a"] = val_a
            row[f"{metric}_b"] = val_b
            row[f"{metric}_winner"] = winner
            row[f"align_{metric}"] = int(row["llm_winner"] == winner) if winner not in {"NA", "TIE"} else np.nan

        rows.append(row)
    return pd.DataFrame(rows)


def build_user_rows(results):
    rows = []
    for r in results:
        dataset = r["dataset"]
        model_a = r["model_a"]
        model_b = r["model_b"]
        for trial in r["per_trial"]:
            pref = trial.get("effective_preference")
            if pref not in {model_a, model_b}:
                continue
            uid = int(trial["user_id"])
            row = {
                "dataset": dataset,
                "model_a": model_a,
                "model_b": model_b,
                "dimension": r["dimension"],
                "user_id": uid,
                "effective_preference": pref,
                "ndcg_winner": r["ndcg_winner"],
                "coverage_winner": r["cov_winner"],
                "align_ndcg": int(pref == r["ndcg_winner"]),
                "align_coverage": int(pref == r["cov_winner"]),
                "is_conflict": is_conflict(r),
            }
            for metric in LOCAL_METRICS:
                val_a = get_list_metrics(dataset, model_a, uid)[metric]
                val_b = get_list_metrics(dataset, model_b, uid)[metric]
                winner = winner_from_values(model_a, val_a, model_b, val_b)
                row[f"{metric}_a"] = val_a
                row[f"{metric}_b"] = val_b
                row[f"{metric}_winner"] = winner
                row[f"align_{metric}"] = int(pref == winner) if winner not in {"NA", "TIE"} else np.nan
                row[f"delta_{metric}_chosen_minus_other"] = (
                    val_a - val_b if pref == model_a else val_b - val_a
                )
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_alignment(df, level):
    records = []
    metrics = ["ndcg", "coverage"] + LOCAL_METRICS
    for dimension, group in df.groupby("dimension"):
        for metric in metrics:
            col = f"align_{metric}"
            if col not in group:
                continue
            vals = group[col].dropna()
            if vals.empty:
                continue
            records.append({
                "level": level,
                "dimension": dimension,
                "metric": metric,
                "n": int(vals.shape[0]),
                "alignment_pct": float(vals.mean() * 100.0),
            })
    return pd.DataFrame(records)


def gap_stratified_summary(pair_df):
    rows = []
    conflict = pair_df[pair_df["is_conflict"]].copy()
    if conflict.empty:
        return pd.DataFrame()

    conflict["abs_delta_ndcg"] = (
        conflict["ndcg_at_10_a"] - conflict["ndcg_at_10_b"]
    ).abs()
    conflict["abs_delta_coverage"] = (
        conflict["coverage_at_k_a"] - conflict["coverage_at_k_b"]
    ).abs()

    ndcg_med = conflict["abs_delta_ndcg"].median()
    cov_med = conflict["abs_delta_coverage"].median()
    conflict["ndcg_gap_group"] = np.where(conflict["abs_delta_ndcg"] <= ndcg_med, "small", "large")
    conflict["coverage_gap_group"] = np.where(conflict["abs_delta_coverage"] <= cov_med, "small", "large")
    conflict["small_ndcg_large_cov"] = (
        (conflict["ndcg_gap_group"] == "small") & (conflict["coverage_gap_group"] == "large")
    )

    for dimension, dfg in conflict.groupby("dimension"):
        for group_col in ["ndcg_gap_group", "coverage_gap_group", "small_ndcg_large_cov"]:
            for group_value, sub in dfg.groupby(group_col):
                vals = sub["align_ndcg_at_10"].dropna()
                if vals.empty:
                    continue
                rows.append({
                    "dimension": dimension,
                    "grouping": group_col,
                    "group": str(group_value),
                    "n_pairs": int(vals.shape[0]),
                    "ndcg_alignment_pct": float(vals.mean() * 100.0),
                    "mean_abs_delta_ndcg": float(sub["abs_delta_ndcg"].mean()),
                    "mean_abs_delta_coverage": float(sub["abs_delta_coverage"].mean()),
                })
    return pd.DataFrame(rows)


def domain_richness_summary(results):
    rows = []
    for dataset in DATASETS:
        analyzer = get_analyzer(dataset)
        rich = analyzer.dataset_richness()
        ds_results = [r for r in results if r["dataset"] == dataset and is_conflict(r)]
        for dimension in PROMPTS:
            dim_results = [r for r in ds_results if r["dimension"] == dimension]
            if not dim_results:
                continue
            winners = [llm_pair_winner(r) for r in dim_results]
            ndcg_align = [
                int(w == r["ndcg_winner"])
                for w, r in zip(winners, dim_results)
                if w != "TIE"
            ]
            row = dict(rich)
            row["dimension"] = dimension
            row["n_conflict_pairs"] = len(ndcg_align)
            row["ndcg_alignment_pct"] = float(np.mean(ndcg_align) * 100.0) if ndcg_align else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def representative_cases(results, user_df):
    """Select concrete cases for the appendix."""
    cases = []
    for r in results:
        if not is_conflict(r):
            continue
        dataset = r["dataset"]
        model_a = r["model_a"]
        model_b = r["model_b"]
        for trial in r["per_trial"]:
            pref = trial.get("effective_preference")
            if pref not in {model_a, model_b}:
                continue
            uid = int(trial["user_id"])
            local_a = get_list_metrics(dataset, model_a, uid)
            local_b = get_list_metrics(dataset, model_b, uid)
            cases.append({
                "dataset": dataset,
                "dimension": r["dimension"],
                "model_a": model_a,
                "model_b": model_b,
                "user_id": uid,
                "effective_preference": pref,
                "ndcg_winner": r["ndcg_winner"],
                "coverage_winner": r["cov_winner"],
                "fwd_verdict": trial.get("fwd_verdict"),
                "rev_verdict": trial.get("rev_verdict"),
                "consistent": trial.get("consistent"),
                "a_text_distance": local_a["local_text_distance"],
                "b_text_distance": local_b["local_text_distance"],
                "a_category_count": local_a["local_category_count"],
                "b_category_count": local_b["local_category_count"],
                "a_desc_len": local_a["description_length"],
                "b_desc_len": local_b["description_length"],
                "a_history_similarity": local_a["history_similarity"],
                "b_history_similarity": local_b["history_similarity"],
                "a_average_popularity": local_a["average_popularity"],
                "b_average_popularity": local_b["average_popularity"],
            })
    df = pd.DataFrame(cases)
    selected = []

    # Diversity prompt failure: LLM chooses NDCG winner while Coverage winner differs.
    fail = df[
        (df["dimension"] == "diversity")
        & (df["effective_preference"] == df["ndcg_winner"])
        & (df["effective_preference"] != df["coverage_winner"])
    ]
    if not fail.empty:
        row = fail.iloc[0].copy()
        row["case_type"] = "diversity_prompt_failure"
        selected.append(row)

    # Novelty success: LLM chooses Coverage winner instead of NDCG winner.
    success = df[
        (df["dimension"] == "novelty")
        & (df["effective_preference"] == df["coverage_winner"])
        & (df["effective_preference"] != df["ndcg_winner"])
    ]
    if not success.empty:
        row = success.iloc[0].copy()
        row["case_type"] = "novelty_or_beyond_accuracy_success"
        selected.append(row)

    # Position flip: inconsistent trial from original result files.
    for r in results:
        for trial in r["per_trial"]:
            if not trial.get("consistent"):
                selected.append(pd.Series({
                    "case_type": "order_swap_flip",
                    "dataset": r["dataset"],
                    "dimension": r["dimension"],
                    "model_a": r["model_a"],
                    "model_b": r["model_b"],
                    "user_id": int(trial["user_id"]),
                    "effective_preference": trial.get("effective_preference"),
                    "ndcg_winner": r["ndcg_winner"],
                    "coverage_winner": r["cov_winner"],
                    "fwd_verdict": trial.get("fwd_verdict"),
                    "rev_verdict": trial.get("rev_verdict"),
                    "consistent": trial.get("consistent"),
                }))
                return pd.DataFrame(selected)
    return pd.DataFrame(selected)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = load_primary_results()
    print(f"Loaded {len(results)} primary Qwen-7B result files.")

    pair_df = build_pair_rows(results)
    user_df = build_user_rows(results)

    pair_df.to_csv(OUT_DIR / "pair_level_local_global_metrics.csv", index=False)
    user_df.to_csv(OUT_DIR / "user_level_local_global_metrics.csv", index=False)

    pair_alignment = summarize_alignment(pair_df, "pair")
    user_alignment = summarize_alignment(user_df[user_df["is_conflict"]], "user")
    alignment = pd.concat([pair_alignment, user_alignment], ignore_index=True)
    alignment.to_csv(OUT_DIR / "local_global_alignment_summary.csv", index=False)

    gap_df = gap_stratified_summary(pair_df)
    gap_df.to_csv(OUT_DIR / "gap_stratified_alignment.csv", index=False)

    richness_df = domain_richness_summary(results)
    richness_df.to_csv(OUT_DIR / "domain_richness_alignment.csv", index=False)

    case_df = representative_cases(results, user_df)
    case_df.to_csv(OUT_DIR / "representative_case_index.csv", index=False)

    print("\nAlignment summary:")
    print(alignment.pivot_table(
        index=["level", "dimension"],
        columns="metric",
        values="alignment_pct",
        aggfunc="first",
    ).round(1).to_string())

    print("\nGap-stratified summary:")
    print(gap_df.round(4).to_string(index=False))

    print("\nDomain richness summary:")
    print(richness_df.round(3).to_string(index=False))

    print(f"\nWrote revision analysis files to {OUT_DIR}")


if __name__ == "__main__":
    main()
