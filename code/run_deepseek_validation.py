"""
Cross-family validation with DeepSeek-V3 (671B MoE) via API.
Tests whether Relevance Dominance persists in a top-tier large model
from a third model family (beyond Qwen and Mistral).
"""

import os
import sys
import json
import time
import random
import re
from pathlib import Path
from dataclasses import dataclass, asdict
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path("/root/autodl-tmp/llm_judge_recsys/code")))

from experiment_config import *
from dimension_prompts import get_dimension_prompt
from data_loader import (
    load_topk_lists, load_item_texts, load_diversity_metrics,
    build_user_profile_summary, format_recommendation_list,
)

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

VALIDATION_PAIRS = {
    "mind": [("gbaf", "fixed"), ("lightgcn_only", "fixed")],
    "yelp": [("gbaf", "concat_mlp"), ("lightgcn_only", "fixed")],
    "amazon_cds": [("gbaf", "fixed"), ("lightgcn_only", "fixed")],
}

VALIDATION_DIMS = ["balanced", "diversity", "novelty"]


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


def call_deepseek(client, system_msg, user_msg, max_retries=3):
    """Call DeepSeek API and extract verdict."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=16,
                temperature=0,
            )
            text = response.choices[0].message.content.strip().upper()
            match = re.search(r'\b(A|B|TIE)\b', text)
            if match:
                return match.group(1)
            if "TIE" in text:
                return "TIE"
            if text.startswith("A"):
                return "A"
            if text.startswith("B"):
                return "B"
            log(f"    Warning: unparseable response '{text}', treating as TIE")
            return "TIE"
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                log(f"    API error: {e}, retrying in {wait}s...")
                time.sleep(wait)
            else:
                log(f"    API error after {max_retries} retries: {e}")
                raise
    return "TIE"


def run_pairwise_eval_deepseek(
    client,
    dataset: str,
    model_a: str,
    model_b: str,
    dimension: str,
    n_users: int = 50,
    seed: int = 42,
):
    """Run bidirectional pairwise evaluation via DeepSeek API."""
    lists_a = load_topk_lists(dataset, model_a, seed)
    lists_b = load_topk_lists(dataset, model_b, seed)
    item_texts = load_item_texts(dataset)

    common_users = sorted(set(lists_a.keys()) & set(lists_b.keys()))
    rng = random.Random(seed)
    rng.shuffle(common_users)
    sample_users = common_users[:n_users]

    trials = []
    for i, uid in enumerate(sample_users):
        items_a = lists_a[uid]
        items_b = lists_b[uid]

        profile = build_user_profile_summary(uid, dataset, item_texts)
        text_a = format_recommendation_list(items_a, item_texts)
        text_b = format_recommendation_list(items_b, item_texts)

        sys_fwd, usr_fwd = get_dimension_prompt(dimension, profile, text_a, text_b)
        fwd_verdict = call_deepseek(client, sys_fwd, usr_fwd)

        sys_rev, usr_rev = get_dimension_prompt(dimension, profile, text_b, text_a)
        rev_verdict = call_deepseek(client, sys_rev, usr_rev)

        flip_map = {"A": "B", "B": "A", "TIE": "TIE"}
        consistent = fwd_verdict == flip_map.get(rev_verdict, "?")

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

        if (i + 1) % 10 == 0:
            log(f"      User {i+1}/{n_users} done")

    consist_count = sum(1 for t in trials if t["consistent"])
    pref_a = sum(1 for t in trials if t["effective_preference"] == model_a)
    pref_b = sum(1 for t in trials if t["effective_preference"] == model_b)
    tie = sum(1 for t in trials if t["effective_preference"] == "TIE")

    return PairExperimentResult(
        dataset=dataset,
        model_a=model_a,
        model_b=model_b,
        dimension=dimension,
        n_trials=n_users,
        prefer_a=pref_a,
        prefer_b=pref_b,
        tie=tie,
        consistency_rate=consist_count / n_users,
        per_trial=trials,
    )


def load_ground_truth_metrics(dataset, model):
    metrics = load_diversity_metrics(dataset, model, seed=SEED)
    return metrics


def main():
    out_dir = RESULTS_DIR / "deepseek"
    out_dir.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("DEEPSEEK_API_KEY is not set.")
    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)

    log("Testing DeepSeek API connection...")
    try:
        test = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": "Reply with OK"}],
            max_tokens=8,
        )
        log(f"API OK: {test.choices[0].message.content.strip()}")
    except Exception as e:
        log(f"API connection failed: {e}")
        return

    start = time.time()
    all_results = []
    total_api_calls = 0

    for ds, pairs in VALIDATION_PAIRS.items():
        for model_a, model_b in pairs:
            gt_a = load_ground_truth_metrics(ds, model_a)
            gt_b = load_ground_truth_metrics(ds, model_b)

            for dim in VALIDATION_DIMS:
                log(f"  [{ds}] {model_a} vs {model_b} ({dim}) ...")
                try:
                    result = run_pairwise_eval_deepseek(
                        client, ds, model_a, model_b, dim,
                        n_users=N_USERS, seed=SEED,
                    )
                    total_api_calls += N_USERS * 2

                    record = {
                        **asdict(result),
                        "gt_ndcg_a": gt_a.get("ndcg@10", 0),
                        "gt_ndcg_b": gt_b.get("ndcg@10", 0),
                        "gt_cov_a": gt_a.get("coverage@k", 0),
                        "gt_cov_b": gt_b.get("coverage@k", 0),
                        "gt_nov_a": gt_a.get("novelty@k", 0),
                        "gt_nov_b": gt_b.get("novelty@k", 0),
                        "ndcg_winner": model_a if gt_a.get("ndcg@10", 0) > gt_b.get("ndcg@10", 0) else model_b,
                        "cov_winner": model_a if gt_a.get("coverage@k", 0) > gt_b.get("coverage@k", 0) else model_b,
                    }
                    all_results.append(record)

                    fname = out_dir / f"{ds}_{model_a}_vs_{model_b}_{dim}_deepseek.json"
                    with open(fname, "w") as f:
                        json.dump(record, f, indent=2, default=str)

                    llm_w = model_a if result.prefer_a > result.prefer_b else (
                        model_b if result.prefer_b > result.prefer_a else "TIE")
                    ndcg_agree = "Y" if llm_w == record["ndcg_winner"] else "N"
                    cov_agree = "Y" if llm_w == record["cov_winner"] else "N"
                    log(f"    Cons={result.consistency_rate:.2f} Pref_A={result.prefer_a} "
                        f"Pref_B={result.prefer_b} LLM={llm_w} NDCG={ndcg_agree} COV={cov_agree}")
                except Exception as e:
                    log(f"    ERROR: {e}")
                    import traceback; traceback.print_exc()

    with open(out_dir / "all_results_deepseek.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    elapsed = time.time() - start
    log(f"\n{'='*60}")
    log(f"DeepSeek-V3 validation complete!")
    log(f"Total API calls: {total_api_calls}")
    log(f"Total time: {elapsed/60:.1f} min")

    ndcg_aligned = 0
    total_conflict = 0
    for r in all_results:
        if r["ndcg_winner"] != r["cov_winner"]:
            total_conflict += 1
            llm_w = r["model_a"] if r["prefer_a"] > r["prefer_b"] else (
                r["model_b"] if r["prefer_b"] > r["prefer_a"] else "TIE")
            if llm_w == r["ndcg_winner"]:
                ndcg_aligned += 1

    for dim in VALIDATION_DIMS:
        dim_results = [r for r in all_results if r["dimension"] == dim]
        dim_conflict = [r for r in dim_results if r["ndcg_winner"] != r["cov_winner"]]
        dim_ndcg = 0
        for r in dim_conflict:
            llm_w = r["model_a"] if r["prefer_a"] > r["prefer_b"] else (
                r["model_b"] if r["prefer_b"] > r["prefer_a"] else "TIE")
            if llm_w == r["ndcg_winner"]:
                dim_ndcg += 1
        n = len(dim_conflict)
        pct = dim_ndcg / n * 100 if n > 0 else 0
        avg_cons = sum(r["consistency_rate"] for r in dim_results) / len(dim_results) if dim_results else 0
        log(f"  {dim:12s}: NDCG alignment = {dim_ndcg}/{n} ({pct:.0f}%), "
            f"avg consistency = {avg_cons:.3f}")

    log(f"{'='*60}")


if __name__ == "__main__":
    main()
