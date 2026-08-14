"""
Pointwise local-diversity scoring experiment for the TKDE revision.

Instead of asking the LLM to choose between List A and List B directly, this
protocol scores each displayed list independently on local list-level diversity
(1--5) and then compares scores. This tests a stronger structured protocol while
avoiding pairwise position bias.
"""

import argparse
import json
import random
import re
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path("/root/autodl-tmp/llm_judge_recsys/code")))

from data_loader import (  # noqa: E402
    build_user_profile_summary,
    format_recommendation_list,
    load_diversity_metrics,
    load_item_texts,
    load_topk_lists,
)
from experiment_config import DATASETS, EXPERIMENT_PAIRS, MODEL_PATH_7B, RESULTS_DIR, SEED  # noqa: E402
from judge_pipeline import JudgeConfig, LLMJudge  # noqa: E402


VALIDATION_PAIRS = {
    "mind": [("gbaf", "fixed"), ("lightgcn_only", "fixed")],
    "yelp": [("gbaf", "concat_mlp"), ("lightgcn_only", "fixed")],
    "amazon_cds": [("gbaf", "fixed"), ("lightgcn_only", "fixed")],
}

SYSTEM_PROMPT = (
    "You are an expert recommender-system evaluator. Score LOCAL LIST-LEVEL "
    "DIVERSITY only. Diversity means the displayed list covers a broad range "
    "of categories, genres, topics, item types, or styles and avoids repeated "
    "or highly similar items. Ignore relevance, popularity, description length, "
    "overall quality, and system-level catalog coverage."
)

USER_TEMPLATE = """## User Profile
{user_profile}

## Recommendation List
{rec_list}

Score the LOCAL LIST-LEVEL DIVERSITY of this single list on a 1-5 scale:
1 = almost all items are highly similar or repetitive
2 = low variety
3 = moderate variety
4 = high variety
5 = very broad variety across categories/topics/styles

Silently identify the main category/topic/style of each item before scoring. Respond with exactly one line: SCORE: <1-5>

Your score:"""


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_score(raw):
    match = re.search(r"SCORE\s*:\s*([1-5])", raw, re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.search(r"\b([1-5])\b", raw)
    if match:
        return int(match.group(1))
    return None


def metric_winner(dataset, model_a, model_b, key, higher=True):
    ma = load_diversity_metrics(dataset, model_a, seed=SEED)
    mb = load_diversity_metrics(dataset, model_b, seed=SEED)
    va = ma.get(key, 0)
    vb = mb.get(key, 0)
    if va == vb:
        return "TIE"
    if higher:
        return model_a if va > vb else model_b
    return model_a if va < vb else model_b


def score_one(judge, user_profile, rec_list):
    user_prompt = USER_TEMPLATE.format(user_profile=user_profile, rec_list=rec_list)
    res = judge.judge(SYSTEM_PROMPT, user_prompt)
    return parse_score(res.raw_output), res.raw_output, res.prompt_tokens, res.completion_tokens


def run_pair(judge, dataset, model_a, model_b, n_users, seed):
    lists_a = load_topk_lists(dataset, model_a, seed)
    lists_b = load_topk_lists(dataset, model_b, seed)
    item_texts = load_item_texts(dataset)
    common_users = sorted(set(lists_a) & set(lists_b))
    rng = random.Random(seed)
    rng.shuffle(common_users)
    sample_users = common_users[:n_users]

    trials = []
    for idx, uid in enumerate(sample_users, start=1):
        profile = build_user_profile_summary(uid, dataset, item_texts)
        text_a = format_recommendation_list(lists_a[uid], item_texts)
        text_b = format_recommendation_list(lists_b[uid], item_texts)

        score_a, raw_a, pt_a, ct_a = score_one(judge, profile, text_a)
        score_b, raw_b, pt_b, ct_b = score_one(judge, profile, text_b)

        if score_a is None or score_b is None:
            pref = "UNPARSEABLE"
        elif score_a > score_b:
            pref = model_a
        elif score_b > score_a:
            pref = model_b
        else:
            pref = "TIE"

        trials.append({
            "user_id": int(uid),
            "score_a": score_a,
            "score_b": score_b,
            "effective_preference": pref,
            "raw_a": raw_a,
            "raw_b": raw_b,
            "prompt_tokens_a": pt_a,
            "prompt_tokens_b": pt_b,
            "completion_tokens_a": ct_a,
            "completion_tokens_b": ct_b,
        })
        if idx % 10 == 0:
            log(f"      {idx}/{n_users} users complete")

    pref_a = sum(t["effective_preference"] == model_a for t in trials)
    pref_b = sum(t["effective_preference"] == model_b for t in trials)
    ties = sum(t["effective_preference"] == "TIE" for t in trials)
    unparseable = sum(t["effective_preference"] == "UNPARSEABLE" for t in trials)
    return {
        "dataset": dataset,
        "model_a": model_a,
        "model_b": model_b,
        "protocol": "pointwise_local_diversity_score",
        "n_trials": len(trials),
        "prefer_a": pref_a,
        "prefer_b": pref_b,
        "tie": ties,
        "unparseable": unparseable,
        "per_trial": trials,
        "gt_ndcg_a": load_diversity_metrics(dataset, model_a, seed=SEED).get("ndcg@10", 0),
        "gt_ndcg_b": load_diversity_metrics(dataset, model_b, seed=SEED).get("ndcg@10", 0),
        "gt_cov_a": load_diversity_metrics(dataset, model_a, seed=SEED).get("coverage@k", 0),
        "gt_cov_b": load_diversity_metrics(dataset, model_b, seed=SEED).get("coverage@k", 0),
        "ndcg_winner": metric_winner(dataset, model_a, model_b, "ndcg@10"),
        "cov_winner": metric_winner(dataset, model_a, model_b, "coverage@k"),
        "entropy_winner": metric_winner(dataset, model_a, model_b, "entropy@k"),
        "ils_text_winner": metric_winner(dataset, model_a, model_b, "ils_text@k", higher=False),
    }


def llm_winner(record):
    if record["prefer_a"] > record["prefer_b"]:
        return record["model_a"]
    if record["prefer_b"] > record["prefer_a"]:
        return record["model_b"]
    return "TIE"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-users", type=int, default=50)
    parser.add_argument("--all-pairs", action="store_true")
    parser.add_argument("--out-subdir", default="revision_pointwise")
    parser.add_argument("--model-path", default=MODEL_PATH_7B)
    return parser.parse_args()


def main():
    args = parse_args()
    pairs = EXPERIMENT_PAIRS if args.all_pairs else VALIDATION_PAIRS
    out_dir = RESULTS_DIR / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"Loading local judge: {args.model_path}")
    judge = LLMJudge(JudgeConfig(
        model_name=args.model_path,
        torch_dtype="float16",
        max_new_tokens=24,
        do_sample=False,
    ))

    records = []
    started = time.time()
    for dataset in DATASETS:
        if dataset not in pairs:
            continue
        for model_a, model_b in pairs[dataset]:
            log(f"[{dataset}] {model_a} vs {model_b} (pointwise diversity scoring)")
            try:
                record = run_pair(judge, dataset, model_a, model_b, args.n_users, SEED)
                records.append(record)
                out_path = out_dir / f"{dataset}_{model_a}_vs_{model_b}_pointwise_local_diversity_score.json"
                with open(out_path, "w") as f:
                    json.dump(record, f, indent=2, default=str)
                w = llm_winner(record)
                log(
                    f"    LLM={w} PrefA={record['prefer_a']} PrefB={record['prefer_b']} "
                    f"Tie={record['tie']} Unparseable={record['unparseable']} "
                    f"NDCG={'Y' if w == record['ndcg_winner'] else 'N'} "
                    f"COV={'Y' if w == record['cov_winner'] else 'N'} "
                    f"ILS={'Y' if w == record['ils_text_winner'] else 'N'}"
                )
            except Exception as exc:
                log(f"    ERROR: {exc}")
                traceback.print_exc()

    with open(out_dir / "all_results_pointwise_local_diversity_score.json", "w") as f:
        json.dump(records, f, indent=2, default=str)

    conflict = [r for r in records if r["ndcg_winner"] != r["cov_winner"]]
    if conflict:
        ndcg = sum(llm_winner(r) == r["ndcg_winner"] for r in conflict)
        cov = sum(llm_winner(r) == r["cov_winner"] for r in conflict)
        ils = sum(llm_winner(r) == r["ils_text_winner"] for r in conflict)
        log(
            f"Summary pairs={len(conflict)} NDCG={ndcg/len(conflict)*100:.1f}% "
            f"Coverage={cov/len(conflict)*100:.1f}% ILS={ils/len(conflict)*100:.1f}%"
        )
    log(f"Pointwise scoring complete in {(time.time()-started)/60:.1f} min")
    judge.cleanup()


if __name__ == "__main__":
    main()
