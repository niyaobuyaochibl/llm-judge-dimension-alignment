"""Analyses added for the TKDE R2 minor revision.

This script addresses three focused requests from the second-round review:

1. report unrounded aggregate NDCG/Recall values and paired confidence
   intervals for every selected model pair;
2. test pair-level conclusions after excluding NDCG near ties, defined before
   looking at judge outcomes as pairs whose paired 95% CI contains zero; and
3. distinguish alignment with a system-average metric winner from alignment
   with the metric winner for the exact user whose two lists were judged.

It also records eligible denominators, metric ties, and missing values for the
prompt-visible local-metric analysis, and characterizes the six-pair subset
used by the expensive diagnostic/API/human checks.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


PROJECT_ROOT = Path("/root/autodl-tmp/llm_diversity_eval")
DIVERSITY_ROOT = Path("/root/autodl-tmp/diversity_experiment")
RESULTS_DIR = PROJECT_ROOT / "results"
REVISION_DIR = PROJECT_ROOT / "revision_results"
DATASETS_DIR = DIVERSITY_ROOT / "datasets"
DIVERSITY_RESULTS_DIR = DIVERSITY_ROOT / "diversity_results"

PROMPTS = ["balanced", "relevance", "diversity", "novelty"]
TOL = 1e-12

SIX_PAIR_SUBSET = {
    ("mind", "gbaf", "fixed"),
    ("mind", "lightgcn_only", "fixed"),
    ("yelp", "gbaf", "concat_mlp"),
    ("yelp", "lightgcn_only", "fixed"),
    ("amazon_cds", "gbaf", "fixed"),
    ("amazon_cds", "lightgcn_only", "fixed"),
}

LOCAL_METRICS = [
    "local_text_distance",
    "local_category_count",
    "local_category_entropy",
    "description_length",
    "history_similarity",
    "average_popularity",
]


def load_primary_results() -> list[dict]:
    rows = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        name = path.name
        if name.startswith("all_results"):
            continue
        if any(tag in name for tag in ("_14b", "_mistral", "_deepseek")):
            continue
        with path.open(encoding="utf-8") as handle:
            rows.append(json.load(handle))
    return rows


@lru_cache(maxsize=None)
def load_test_truth(dataset: str) -> dict[int, set[int]]:
    test = pd.read_pickle(DATASETS_DIR / dataset / "test.pkl")
    grouped = test.groupby("user_idx")["item_idx"].apply(list)
    return {
        int(user_id): {int(item_id) for item_id in items}
        for user_id, items in grouped.items()
    }


@lru_cache(maxsize=None)
def load_topk(dataset: str, model: str) -> dict[int, list[int]]:
    path = DIVERSITY_RESULTS_DIR / dataset / model / "topk_lists_seed42.json"
    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    return {int(user_id): [int(x) for x in items[:10]] for user_id, items in raw.items()}


def user_accuracy(topk: list[int], truth: set[int]) -> tuple[float, float]:
    hit_count = sum(item in truth for item in topk)
    recall = hit_count / max(len(truth), 1)
    dcg = sum(
        1.0 / np.log2(rank + 2)
        for rank, item in enumerate(topk)
        if item in truth
    )
    ideal_hits = min(len(truth), len(topk))
    idcg = sum(1.0 / np.log2(rank + 2) for rank in range(ideal_hits))
    ndcg = dcg / idcg if idcg > 0 else 0.0
    return float(ndcg), float(recall)


@lru_cache(maxsize=None)
def model_user_accuracy(dataset: str, model: str) -> dict[int, tuple[float, float]]:
    truth = load_test_truth(dataset)
    lists = load_topk(dataset, model)
    common_users = sorted(set(truth) & set(lists))
    return {
        user_id: user_accuracy(lists[user_id], truth[user_id])
        for user_id in common_users
    }


def paired_mean_ci(diff: np.ndarray, confidence: float = 0.95) -> tuple[float, float, float]:
    """Paired t interval over user-level differences."""
    diff = np.asarray(diff, dtype=float)
    n = diff.size
    mean = float(np.mean(diff))
    if n < 2:
        return mean, math.nan, math.nan
    se = float(np.std(diff, ddof=1) / np.sqrt(n))
    critical = float(student_t.ppf(0.5 + confidence / 2.0, n - 1))
    return mean, mean - critical * se, mean + critical * se


def pair_key(result: dict) -> tuple[str, str, str]:
    return result["dataset"], result["model_a"], result["model_b"]


def is_conflict(result: dict) -> bool:
    return result["ndcg_winner"] != result["cov_winner"]


def llm_pair_winner(result: dict) -> str:
    if result["prefer_a"] > result["prefer_b"]:
        return result["model_a"]
    if result["prefer_b"] > result["prefer_a"]:
        return result["model_b"]
    return "TIE"


def selected_pair_results(results: list[dict]) -> dict[tuple[str, str, str], dict]:
    """Return one record per pair; aggregate metrics do not depend on prompt."""
    selected = {}
    for result in results:
        selected.setdefault(pair_key(result), result)
    return selected


def build_paired_accuracy_table(results: list[dict]) -> pd.DataFrame:
    rows = []
    for (dataset, model_a, model_b), result in sorted(selected_pair_results(results).items()):
        acc_a = model_user_accuracy(dataset, model_a)
        acc_b = model_user_accuracy(dataset, model_b)
        users = sorted(set(acc_a) & set(acc_b))
        a = np.asarray([acc_a[user_id] for user_id in users], dtype=float)
        b = np.asarray([acc_b[user_id] for user_id in users], dtype=float)

        ndcg_delta, ndcg_low, ndcg_high = paired_mean_ci(a[:, 0] - b[:, 0])
        recall_delta, recall_low, recall_high = paired_mean_ci(a[:, 1] - b[:, 1])
        rows.append(
            {
                "dataset": dataset,
                "model_a": model_a,
                "model_b": model_b,
                "n_paired_users": len(users),
                "ndcg_a": float(np.mean(a[:, 0])),
                "ndcg_b": float(np.mean(b[:, 0])),
                "delta_ndcg_a_minus_b": ndcg_delta,
                "delta_ndcg_ci95_low": ndcg_low,
                "delta_ndcg_ci95_high": ndcg_high,
                "ndcg_near_tie_ci_includes_zero": bool(ndcg_low <= 0.0 <= ndcg_high),
                "recall_a": float(np.mean(a[:, 1])),
                "recall_b": float(np.mean(b[:, 1])),
                "delta_recall_a_minus_b": recall_delta,
                "delta_recall_ci95_low": recall_low,
                "delta_recall_ci95_high": recall_high,
                "recall_near_tie_ci_includes_zero": bool(recall_low <= 0.0 <= recall_high),
                "ndcg_winner": result["ndcg_winner"],
                "coverage_winner": result["cov_winner"],
                "is_conflict": is_conflict(result),
                "in_six_pair_subset": (dataset, model_a, model_b) in SIX_PAIR_SUBSET,
            }
        )
    return pd.DataFrame(rows)


def metric_winner(model_a: str, value_a: float, model_b: str, value_b: float) -> str:
    if not np.isfinite(value_a) or not np.isfinite(value_b):
        return "NA"
    if math.isclose(float(value_a), float(value_b), rel_tol=TOL, abs_tol=TOL):
        return "TIE"
    return model_a if value_a > value_b else model_b


def build_user_accuracy_alignment(results: list[dict]) -> pd.DataFrame:
    rows = []
    for result in results:
        if not is_conflict(result):
            continue
        dataset, model_a, model_b = pair_key(result)
        acc_a = model_user_accuracy(dataset, model_a)
        acc_b = model_user_accuracy(dataset, model_b)
        for trial in result["per_trial"]:
            preference = trial.get("effective_preference")
            if preference not in {model_a, model_b}:
                continue
            user_id = int(trial["user_id"])
            ndcg_a, recall_a = acc_a[user_id]
            ndcg_b, recall_b = acc_b[user_id]
            ndcg_winner_user = metric_winner(model_a, ndcg_a, model_b, ndcg_b)
            recall_winner_user = metric_winner(model_a, recall_a, model_b, recall_b)
            rows.append(
                {
                    "dataset": dataset,
                    "model_a": model_a,
                    "model_b": model_b,
                    "dimension": result["dimension"],
                    "user_id": user_id,
                    "effective_preference": preference,
                    "system_average_ndcg_winner": result["ndcg_winner"],
                    "align_system_average_ndcg": int(preference == result["ndcg_winner"]),
                    "user_ndcg_a": ndcg_a,
                    "user_ndcg_b": ndcg_b,
                    "user_ndcg_winner": ndcg_winner_user,
                    "align_user_ndcg": (
                        int(preference == ndcg_winner_user)
                        if ndcg_winner_user not in {"TIE", "NA"}
                        else np.nan
                    ),
                    "user_recall_a": recall_a,
                    "user_recall_b": recall_b,
                    "user_recall_winner": recall_winner_user,
                    "align_user_recall": (
                        int(preference == recall_winner_user)
                        if recall_winner_user not in {"TIE", "NA"}
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def summarize_user_accuracy_alignment(user_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dimension in PROMPTS:
        group = user_df[user_df["dimension"] == dimension]
        total = int(len(group))
        record = {"dimension": dimension, "consistent_verdicts": total}
        for name, winner_col, align_col in (
            ("system_average_ndcg", "system_average_ndcg_winner", "align_system_average_ndcg"),
            ("user_ndcg", "user_ndcg_winner", "align_user_ndcg"),
            ("user_recall", "user_recall_winner", "align_user_recall"),
        ):
            eligible = group[align_col].notna()
            n = int(eligible.sum())
            aligned = int(group.loc[eligible, align_col].sum())
            ties = int((group[winner_col] == "TIE").sum())
            missing = int((group[winner_col] == "NA").sum())
            record[f"{name}_aligned"] = aligned
            record[f"{name}_eligible_n"] = n
            record[f"{name}_alignment_pct"] = 100.0 * aligned / n if n else math.nan
            record[f"{name}_metric_ties"] = ties
            record[f"{name}_missing"] = missing
        rows.append(record)
    return pd.DataFrame(rows)


def build_near_tie_robustness(results: list[dict], pair_accuracy: pd.DataFrame) -> pd.DataFrame:
    near_tie = {
        (row.dataset, row.model_a, row.model_b): bool(row.ndcg_near_tie_ci_includes_zero)
        for row in pair_accuracy.itertuples()
    }
    rows = []
    for dimension in PROMPTS:
        prompt_results = [r for r in results if r["dimension"] == dimension and is_conflict(r)]
        for subset_label, subset in (
            ("all_conflict_pairs", prompt_results),
            (
                "exclude_ndcg_near_ties",
                [r for r in prompt_results if not near_tie[pair_key(r)]],
            ),
        ):
            pair_n = len(subset)
            pair_ndcg = sum(llm_pair_winner(r) == r["ndcg_winner"] for r in subset)
            pair_coverage = sum(llm_pair_winner(r) == r["cov_winner"] for r in subset)
            pair_ties = sum(llm_pair_winner(r) == "TIE" for r in subset)
            consistent = [
                (trial["effective_preference"], r["ndcg_winner"])
                for r in subset
                for trial in r["per_trial"]
                if trial.get("effective_preference") in {r["model_a"], r["model_b"]}
            ]
            user_n = len(consistent)
            user_ndcg = sum(preference == winner for preference, winner in consistent)
            rows.append(
                {
                    "dimension": dimension,
                    "subset": subset_label,
                    "pair_n": pair_n,
                    "pair_ndcg_aligned": pair_ndcg,
                    "pair_coverage_aligned": pair_coverage,
                    "pair_ties": pair_ties,
                    "pair_ndcg_alignment_pct": 100.0 * pair_ndcg / pair_n if pair_n else math.nan,
                    "system_average_user_alignment_n": user_n,
                    "system_average_user_ndcg_aligned": user_ndcg,
                    "system_average_user_ndcg_alignment_pct": (
                        100.0 * user_ndcg / user_n if user_n else math.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_local_metric_denominators() -> pd.DataFrame:
    path = REVISION_DIR / "user_level_local_global_metrics.csv"
    data = pd.read_csv(path, keep_default_na=False)
    data = data[data["is_conflict"].astype(str).str.lower() == "true"].copy()
    rows = []
    metric_specs = [
        ("system_average_ndcg", "ndcg_winner", "align_ndcg"),
        ("system_average_coverage", "coverage_winner", "align_coverage"),
    ] + [(metric, f"{metric}_winner", f"align_{metric}") for metric in LOCAL_METRICS]

    for dimension in PROMPTS:
        group = data[data["dimension"] == dimension]
        total = int(len(group))
        for metric, winner_col, align_col in metric_specs:
            winner = group[winner_col].astype(str)
            align_numeric = pd.to_numeric(group[align_col], errors="coerce")
            eligible = align_numeric.notna()
            n = int(eligible.sum())
            aligned = int(align_numeric[eligible].sum())
            rows.append(
                {
                    "dimension": dimension,
                    "metric": metric,
                    "consistent_verdicts": total,
                    "eligible_n": n,
                    "aligned": aligned,
                    "alignment_pct": 100.0 * aligned / n if n else math.nan,
                    "metric_ties_excluded": int((winner == "TIE").sum()),
                    "missing_metric_excluded": int(winner.isin(["", "NA", "nan", "NaN"]).sum()),
                }
            )
    return pd.DataFrame(rows)


def build_six_pair_characterization(
    results: list[dict], pair_accuracy: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    conflict = pair_accuracy[pair_accuracy["is_conflict"]].copy()
    conflict["abs_delta_ndcg"] = conflict["delta_ndcg_a_minus_b"].abs()

    # Coverage gaps come from the primary result JSON records.
    coverage_gaps = {
        pair_key(r): abs(float(r["gt_cov_a"]) - float(r["gt_cov_b"]))
        for r in results
    }
    conflict["abs_delta_coverage"] = [
        coverage_gaps[(row.dataset, row.model_a, row.model_b)]
        for row in conflict.itertuples()
    ]

    summary_rows = []
    for label, group in (
        ("all_15_conflict_pairs", conflict),
        ("six_pair_validation_subset", conflict[conflict["in_six_pair_subset"]]),
    ):
        summary_rows.append(
            {
                "subset": label,
                "n_pairs": int(len(group)),
                "n_datasets": int(group["dataset"].nunique()),
                "datasets": ";".join(sorted(group["dataset"].unique())),
                "min_abs_delta_ndcg": float(group["abs_delta_ndcg"].min()),
                "median_abs_delta_ndcg": float(group["abs_delta_ndcg"].median()),
                "max_abs_delta_ndcg": float(group["abs_delta_ndcg"].max()),
                "min_abs_delta_coverage": float(group["abs_delta_coverage"].min()),
                "median_abs_delta_coverage": float(group["abs_delta_coverage"].median()),
                "max_abs_delta_coverage": float(group["abs_delta_coverage"].max()),
                "ndcg_near_tie_pairs": int(group["ndcg_near_tie_ci_includes_zero"].sum()),
            }
        )

    alignment_rows = []
    for dimension in PROMPTS:
        dim_results = [r for r in results if r["dimension"] == dimension and is_conflict(r)]
        for label, group in (
            ("all_15_conflict_pairs", dim_results),
            ("six_pair_validation_subset", [r for r in dim_results if pair_key(r) in SIX_PAIR_SUBSET]),
        ):
            n = len(group)
            ndcg = sum(llm_pair_winner(r) == r["ndcg_winner"] for r in group)
            coverage = sum(llm_pair_winner(r) == r["cov_winner"] for r in group)
            ties = sum(llm_pair_winner(r) == "TIE" for r in group)
            alignment_rows.append(
                {
                    "dimension": dimension,
                    "subset": label,
                    "n_pairs": n,
                    "ndcg_aligned": ndcg,
                    "coverage_aligned": coverage,
                    "ties": ties,
                    "ndcg_alignment_pct": 100.0 * ndcg / n if n else math.nan,
                }
            )

    details = conflict[
        [
            "dataset",
            "model_a",
            "model_b",
            "abs_delta_ndcg",
            "abs_delta_coverage",
            "ndcg_near_tie_ci_includes_zero",
            "in_six_pair_subset",
        ]
    ].sort_values(["in_six_pair_subset", "dataset", "model_a", "model_b"], ascending=[False, True, True, True])
    details.to_csv(REVISION_DIR / "six_pair_subset_pair_details.csv", index=False)
    return pd.DataFrame(summary_rows), pd.DataFrame(alignment_rows)


def main() -> None:
    REVISION_DIR.mkdir(parents=True, exist_ok=True)
    results = load_primary_results()
    print(f"Loaded {len(results)} primary Qwen-7B result records.")

    pair_accuracy = build_paired_accuracy_table(results)
    pair_accuracy.to_csv(REVISION_DIR / "paired_accuracy_differences.csv", index=False)

    user_accuracy_alignment = build_user_accuracy_alignment(results)
    user_accuracy_alignment.to_csv(REVISION_DIR / "user_level_accuracy_metrics.csv", index=False)
    user_summary = summarize_user_accuracy_alignment(user_accuracy_alignment)
    user_summary.to_csv(REVISION_DIR / "user_accuracy_alignment_summary.csv", index=False)

    robustness = build_near_tie_robustness(results, pair_accuracy)
    robustness.to_csv(REVISION_DIR / "near_tie_robustness.csv", index=False)

    local_denominators = build_local_metric_denominators()
    local_denominators.to_csv(REVISION_DIR / "local_metric_denominators.csv", index=False)

    subset_summary, subset_alignment = build_six_pair_characterization(results, pair_accuracy)
    subset_summary.to_csv(REVISION_DIR / "six_pair_subset_summary.csv", index=False)
    subset_alignment.to_csv(REVISION_DIR / "six_pair_subset_alignment.csv", index=False)

    with pd.option_context("display.max_columns", None, "display.width", 240):
        print("\nPaired aggregate accuracy differences")
        print(pair_accuracy.to_string(index=False))
        print("\nSystem-average vs exact-user accuracy alignment")
        print(user_summary.to_string(index=False))
        print("\nNear-tie robustness")
        print(robustness.to_string(index=False))
        print("\nLocal metric denominators")
        print(local_denominators.to_string(index=False))
        print("\nSix-pair subset characterization")
        print(subset_summary.to_string(index=False))
        print(subset_alignment.to_string(index=False))


if __name__ == "__main__":
    main()
