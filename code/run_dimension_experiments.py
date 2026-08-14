"""
Core experiment: LLM-as-a-Judge Dimension Analysis.
For each model pair, run LLM evaluation with standard + dimension-specific prompts,
then correlate with ground truth accuracy and diversity metrics.
"""

import sys
import json
import time
import random
import traceback
from pathlib import Path
from collections import Counter
from dataclasses import dataclass, asdict

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path("/root/autodl-tmp/llm_judge_recsys/code")))

from experiment_config import *
from dimension_prompts import get_dimension_prompt
from judge_pipeline import LLMJudge, JudgeConfig
from data_loader import (
    load_topk_lists, load_item_texts, load_diversity_metrics,
    build_user_profile_summary, format_recommendation_list,
)


@dataclass
class PairExperimentResult:
    dataset: str
    model_a: str
    model_b: str
    dimension: str
    n_trials: int
    prefer_a: int
    prefer_b: int
    tie: int
    consistency_rate: float
    per_trial: list


def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def run_pairwise_dimension_eval(
    judge: LLMJudge,
    dataset: str,
    model_a: str,
    model_b: str,
    dimension: str,
    n_users: int = 50,
    seed: int = 42,
):
    """Run bidirectional pairwise evaluation for a specific dimension."""
    ds_dir = DS_DIR_MAP[dataset]

    lists_a = load_topk_lists(dataset, model_a, seed)
    lists_b = load_topk_lists(dataset, model_b, seed)
    item_texts = load_item_texts(dataset)

    common_users = sorted(set(lists_a.keys()) & set(lists_b.keys()))
    rng = random.Random(seed)
    rng.shuffle(common_users)
    sample_users = common_users[:n_users]

    trials = []
    for uid in sample_users:
        items_a = lists_a[uid]
        items_b = lists_b[uid]

        profile = build_user_profile_summary(uid, dataset, item_texts)
        text_a = format_recommendation_list(items_a, item_texts)
        text_b = format_recommendation_list(items_b, item_texts)

        # Forward: A=model_a, B=model_b
        sys_fwd, usr_fwd = get_dimension_prompt(dimension, profile, text_a, text_b)
        res_fwd = judge.judge(sys_fwd, usr_fwd)

        # Reverse: A=model_b, B=model_a
        sys_rev, usr_rev = get_dimension_prompt(dimension, profile, text_b, text_a)
        res_rev = judge.judge(sys_rev, usr_rev)

        fwd_verdict = res_fwd.verdict
        rev_verdict = res_rev.verdict

        # Check consistency
        flip_map = {"A": "B", "B": "A", "TIE": "TIE"}
        consistent = fwd_verdict == flip_map.get(rev_verdict, "?")

        # Determine effective preference (in terms of model_a/model_b)
        if consistent:
            if fwd_verdict == "A":
                effective = model_a
            elif fwd_verdict == "B":
                effective = model_b
            else:
                effective = "TIE"
        else:
            effective = "INCONSISTENT"

        trials.append({
            "user_id": uid,
            "fwd_verdict": fwd_verdict,
            "rev_verdict": rev_verdict,
            "consistent": consistent,
            "effective_preference": effective,
        })

    n_consistent = sum(1 for t in trials if t["consistent"])
    consistency_rate = n_consistent / len(trials) if trials else 0

    prefer_a = sum(1 for t in trials if t["effective_preference"] == model_a)
    prefer_b = sum(1 for t in trials if t["effective_preference"] == model_b)
    tie = sum(1 for t in trials if t["effective_preference"] == "TIE")

    return PairExperimentResult(
        dataset=dataset,
        model_a=model_a,
        model_b=model_b,
        dimension=dimension,
        n_trials=len(trials),
        prefer_a=prefer_a,
        prefer_b=prefer_b,
        tie=tie,
        consistency_rate=consistency_rate,
        per_trial=trials,
    )


def load_ground_truth_metrics(dataset, model, seed=42):
    """Load precomputed diversity/accuracy metrics for a model."""
    try:
        return load_diversity_metrics(dataset, model, seed)
    except Exception:
        return {}


def run_all_experiments(judge, dimensions=None, datasets=None, n_users=50):
    """Run full experiment matrix."""
    if dimensions is None:
        dimensions = ["balanced", "relevance", "diversity", "novelty"]
    if datasets is None:
        datasets = DATASETS

    all_results = []
    for ds in datasets:
        pairs = EXPERIMENT_PAIRS.get(ds, [])
        for model_a, model_b in pairs:
            gt_a = load_ground_truth_metrics(ds, model_a)
            gt_b = load_ground_truth_metrics(ds, model_b)

            if not gt_a or not gt_b:
                log(f"  Skipping {ds}/{model_a} vs {model_b}: missing metrics")
                continue

            for dim in dimensions:
                log(f"  [{ds}] {model_a} vs {model_b} ({dim}) ...")
                try:
                    result = run_pairwise_dimension_eval(
                        judge, ds, model_a, model_b, dim,
                        n_users=n_users, seed=SEED,
                    )

                    ndcg_a = gt_a.get("ndcg@10", 0)
                    ndcg_b = gt_b.get("ndcg@10", 0)
                    cov_a = gt_a.get("coverage@k", 0)
                    cov_b = gt_b.get("coverage@k", 0)
                    nov_a = gt_a.get("novelty@k", 0)
                    nov_b = gt_b.get("novelty@k", 0)

                    record = {
                        **asdict(result),
                        "gt_ndcg_a": ndcg_a, "gt_ndcg_b": ndcg_b,
                        "gt_cov_a": cov_a, "gt_cov_b": cov_b,
                        "gt_nov_a": nov_a, "gt_nov_b": nov_b,
                        "ndcg_winner": model_a if ndcg_a > ndcg_b else model_b,
                        "cov_winner": model_a if cov_a > cov_b else model_b,
                        "nov_winner": model_a if nov_a > nov_b else model_b,
                    }
                    all_results.append(record)

                    out_path = RESULTS_DIR / f"{ds}_{model_a}_vs_{model_b}_{dim}.json"
                    with open(out_path, "w") as f:
                        json.dump(record, f, indent=2, default=str)

                    log(f"    Consistency: {result.consistency_rate:.3f}, "
                        f"Prefer {model_a}: {result.prefer_a}, "
                        f"Prefer {model_b}: {result.prefer_b}, "
                        f"TIE: {result.tie}")
                except Exception as e:
                    log(f"    ERROR: {e}")
                    traceback.print_exc()

    return all_results


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    log(f"Loading LLM judge: {MODEL_PATH_7B}")
    config = JudgeConfig(
        model_name=MODEL_PATH_7B,
        torch_dtype="float16",
        max_new_tokens=32,
        do_sample=False,
    )
    judge = LLMJudge(config)
    start = time.time()

    # Phase 1: Run "balanced" prompt on ALL pairs across all datasets
    log("=" * 70)
    log("PHASE 1: Balanced (standard) evaluation across all pairs")
    log("=" * 70)
    phase1_results = run_all_experiments(
        judge, dimensions=["balanced"], n_users=N_USERS,
    )

    # Phase 2: Run dimension-specific prompts on ALL pairs
    log("=" * 70)
    log("PHASE 2: Dimension-specific evaluation")
    log("=" * 70)
    phase2_results = run_all_experiments(
        judge, dimensions=["relevance", "diversity", "novelty"], n_users=N_USERS,
    )

    all_results = phase1_results + phase2_results

    # Save combined results
    with open(RESULTS_DIR / "all_results_7b.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    elapsed = time.time() - start
    log(f"\nAll experiments complete! Total: {elapsed/60:.1f} min")

    # Print summary
    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)
    for r in all_results:
        dim = r["dimension"]
        ds = r["dataset"]
        ma, mb = r["model_a"], r["model_b"]
        pa, pb = r["prefer_a"], r["prefer_b"]
        cons = r["consistency_rate"]
        ndcg_w = r["ndcg_winner"]
        cov_w = r["cov_winner"]
        llm_w = ma if pa > pb else (mb if pb > pa else "TIE")
        agree_ndcg = "Y" if llm_w == ndcg_w else ("~" if llm_w == "TIE" else "N")
        agree_cov = "Y" if llm_w == cov_w else ("~" if llm_w == "TIE" else "N")
        log(f"  [{dim:10s}] {ds:13s} {ma:15s} vs {mb:15s} | "
            f"LLM={llm_w:15s} NDCG={agree_ndcg} COV={agree_cov} cons={cons:.2f}")

    judge.cleanup()


if __name__ == "__main__":
    main()
