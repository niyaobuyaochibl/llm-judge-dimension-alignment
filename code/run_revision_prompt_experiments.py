"""
Run stronger prompting and global-statistics prompt experiments for the TKDE
major revision.

Default run: Qwen2.5-7B on the 6-pair validation subset used for cross-model
validation. Use --all-pairs for all 16 original pairs.
"""

import argparse
import json
import random
import sys
import time
import traceback
from dataclasses import asdict, dataclass
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
from experiment_config import (  # noqa: E402
    DATASETS,
    EXPERIMENT_PAIRS,
    MODEL_PATH_7B,
    RESULTS_DIR,
    SEED,
)
from judge_pipeline import JudgeConfig, LLMJudge  # noqa: E402
from revision_prompt_protocols import ALL_REVISION_PROTOCOLS, get_revision_prompt  # noqa: E402


VALIDATION_PAIRS = {
    "mind": [("gbaf", "fixed"), ("lightgcn_only", "fixed")],
    "yelp": [("gbaf", "concat_mlp"), ("lightgcn_only", "fixed")],
    "amazon_cds": [("gbaf", "fixed"), ("lightgcn_only", "fixed")],
}


@dataclass
class PairExperimentResult:
    dataset: str
    model_a: str
    model_b: str
    protocol: str
    n_trials: int
    prefer_a: int
    prefer_b: int
    tie: int
    inconsistent: int
    consistency_rate: float
    per_trial: list


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_metrics(dataset, model):
    return load_diversity_metrics(dataset, model, seed=SEED)


def winner(gt_a, gt_b, key, model_a, model_b, higher=True):
    va = gt_a.get(key, 0)
    vb = gt_b.get(key, 0)
    if va == vb:
        return "TIE"
    if higher:
        return model_a if va > vb else model_b
    return model_a if va < vb else model_b


def run_pair_protocol(judge, dataset, model_a, model_b, protocol, n_users, seed):
    lists_a = load_topk_lists(dataset, model_a, seed)
    lists_b = load_topk_lists(dataset, model_b, seed)
    item_texts = load_item_texts(dataset)
    metrics_a = load_metrics(dataset, model_a)
    metrics_b = load_metrics(dataset, model_b)

    common_users = sorted(set(lists_a) & set(lists_b))
    rng = random.Random(seed)
    rng.shuffle(common_users)
    sample_users = common_users[:n_users]

    trials = []
    for idx, uid in enumerate(sample_users, start=1):
        profile = build_user_profile_summary(uid, dataset, item_texts)
        text_a = format_recommendation_list(lists_a[uid], item_texts)
        text_b = format_recommendation_list(lists_b[uid], item_texts)

        sys_fwd, usr_fwd = get_revision_prompt(
            protocol, profile, text_a, text_b, metrics_a=metrics_a, metrics_b=metrics_b
        )
        res_fwd = judge.judge(sys_fwd, usr_fwd)

        sys_rev, usr_rev = get_revision_prompt(
            protocol, profile, text_b, text_a, metrics_a=metrics_b, metrics_b=metrics_a
        )
        res_rev = judge.judge(sys_rev, usr_rev)

        flip_map = {"A": "B", "B": "A", "TIE": "TIE"}
        fwd = res_fwd.verdict
        rev = res_rev.verdict
        consistent = fwd == flip_map.get(rev, "?")

        if consistent:
            if fwd == "A":
                effective = model_a
            elif fwd == "B":
                effective = model_b
            elif fwd == "TIE":
                effective = "TIE"
            else:
                effective = "UNPARSEABLE"
        else:
            effective = "INCONSISTENT"

        trials.append({
            "user_id": int(uid),
            "fwd_verdict": fwd,
            "rev_verdict": rev,
            "consistent": consistent,
            "effective_preference": effective,
            "fwd_raw": res_fwd.raw_output,
            "rev_raw": res_rev.raw_output,
            "fwd_prompt_tokens": res_fwd.prompt_tokens,
            "rev_prompt_tokens": res_rev.prompt_tokens,
            "fwd_completion_tokens": res_fwd.completion_tokens,
            "rev_completion_tokens": res_rev.completion_tokens,
        })

        if idx % 10 == 0:
            log(f"      {idx}/{n_users} users complete")

    prefer_a = sum(t["effective_preference"] == model_a for t in trials)
    prefer_b = sum(t["effective_preference"] == model_b for t in trials)
    tie = sum(t["effective_preference"] == "TIE" for t in trials)
    inconsistent = sum(t["effective_preference"] == "INCONSISTENT" for t in trials)
    consistency_rate = sum(t["consistent"] for t in trials) / len(trials) if trials else 0

    return PairExperimentResult(
        dataset=dataset,
        model_a=model_a,
        model_b=model_b,
        protocol=protocol,
        n_trials=len(trials),
        prefer_a=prefer_a,
        prefer_b=prefer_b,
        tie=tie,
        inconsistent=inconsistent,
        consistency_rate=consistency_rate,
        per_trial=trials,
    )


def result_record(result):
    r = asdict(result)
    gt_a = load_metrics(result.dataset, result.model_a)
    gt_b = load_metrics(result.dataset, result.model_b)
    r.update({
        "gt_ndcg_a": gt_a.get("ndcg@10", 0),
        "gt_ndcg_b": gt_b.get("ndcg@10", 0),
        "gt_cov_a": gt_a.get("coverage@k", 0),
        "gt_cov_b": gt_b.get("coverage@k", 0),
        "gt_entropy_a": gt_a.get("entropy@k", 0),
        "gt_entropy_b": gt_b.get("entropy@k", 0),
        "gt_nov_a": gt_a.get("novelty@k", 0),
        "gt_nov_b": gt_b.get("novelty@k", 0),
        "gt_tail_cov_a": gt_a.get("tail_coverage@k", 0),
        "gt_tail_cov_b": gt_b.get("tail_coverage@k", 0),
        "ndcg_winner": winner(gt_a, gt_b, "ndcg@10", result.model_a, result.model_b),
        "cov_winner": winner(gt_a, gt_b, "coverage@k", result.model_a, result.model_b),
        "entropy_winner": winner(gt_a, gt_b, "entropy@k", result.model_a, result.model_b),
        "nov_winner": winner(gt_a, gt_b, "novelty@k", result.model_a, result.model_b),
        "tail_cov_winner": winner(gt_a, gt_b, "tail_coverage@k", result.model_a, result.model_b),
    })
    return r


def llm_winner(record):
    if record["prefer_a"] > record["prefer_b"]:
        return record["model_a"]
    if record["prefer_b"] > record["prefer_a"]:
        return record["model_b"]
    return "TIE"


def print_summary(records):
    log("\nSummary")
    by_protocol = {}
    for r in records:
        by_protocol.setdefault(r["protocol"], []).append(r)
    for protocol, rows in by_protocol.items():
        conflict = [r for r in rows if r["ndcg_winner"] != r["cov_winner"]]
        if not conflict:
            continue
        ndcg = sum(llm_winner(r) == r["ndcg_winner"] for r in conflict)
        cov = sum(llm_winner(r) == r["cov_winner"] for r in conflict)
        ent = sum(llm_winner(r) == r["entropy_winner"] for r in conflict)
        avg_cons = sum(r["consistency_rate"] for r in conflict) / len(conflict)
        log(
            f"  {protocol:24s} pairs={len(conflict):2d} "
            f"NDCG={ndcg/len(conflict)*100:5.1f}% "
            f"Coverage={cov/len(conflict)*100:5.1f}% "
            f"Entropy={ent/len(conflict)*100:5.1f}% "
            f"Cons={avg_cons:.3f}"
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-users", type=int, default=50)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--all-pairs", action="store_true")
    parser.add_argument("--protocols", nargs="+", default=ALL_REVISION_PROTOCOLS)
    parser.add_argument("--out-subdir", default="revision_prompts")
    parser.add_argument("--model-path", default=MODEL_PATH_7B)
    return parser.parse_args()


def main():
    args = parse_args()
    pairs = EXPERIMENT_PAIRS if args.all_pairs else VALIDATION_PAIRS
    out_dir = RESULTS_DIR / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"Loading local judge: {args.model_path}")
    config = JudgeConfig(
        model_name=args.model_path,
        torch_dtype="float16",
        max_new_tokens=96,
        do_sample=False,
    )
    judge = LLMJudge(config)

    records = []
    started = time.time()
    for dataset in DATASETS:
        if dataset not in pairs:
            continue
        for model_a, model_b in pairs[dataset]:
            for protocol in args.protocols:
                log(f"[{dataset}] {model_a} vs {model_b} ({protocol})")
                try:
                    result = run_pair_protocol(
                        judge, dataset, model_a, model_b, protocol,
                        n_users=args.n_users, seed=args.seed,
                    )
                    record = result_record(result)
                    records.append(record)
                    out_path = out_dir / f"{dataset}_{model_a}_vs_{model_b}_{protocol}.json"
                    with open(out_path, "w") as f:
                        json.dump(record, f, indent=2, default=str)
                    w = llm_winner(record)
                    log(
                        f"    LLM={w} PrefA={record['prefer_a']} PrefB={record['prefer_b']} "
                        f"Tie={record['tie']} Incon={record['inconsistent']} "
                        f"Cons={record['consistency_rate']:.2f} "
                        f"NDCG={'Y' if w == record['ndcg_winner'] else 'N'} "
                        f"COV={'Y' if w == record['cov_winner'] else 'N'}"
                    )
                except Exception as exc:
                    log(f"    ERROR: {exc}")
                    traceback.print_exc()

    with open(out_dir / "all_results_revision_prompts.json", "w") as f:
        json.dump(records, f, indent=2, default=str)

    print_summary(records)
    elapsed = (time.time() - started) / 60
    log(f"Revision prompt experiments complete in {elapsed:.1f} min")
    judge.cleanup()


if __name__ == "__main__":
    main()
