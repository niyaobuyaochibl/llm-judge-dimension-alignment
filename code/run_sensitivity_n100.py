"""100-user sensitivity experiment for the TKDE revision.

Runs Qwen-7B on six representative conflict pairs and three prompt types,
writing outputs to results/sensitivity_n100 without touching main results.
"""

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path('/root/autodl-tmp/llm_judge_recsys/code')))

from experiment_config import MODEL_PATH_7B, PROJECT_ROOT, SEED
from judge_pipeline import LLMJudge, JudgeConfig
from run_dimension_experiments import run_pairwise_dimension_eval, load_ground_truth_metrics

PAIRS = [
    ("mind", "gbaf", "fixed"),
    ("mind", "lightgcn_only", "fixed"),
    ("yelp", "gbaf", "concat_mlp"),
    ("yelp", "lightgcn_only", "fixed"),
    ("amazon_cds", "gbaf", "fixed"),
    ("amazon_cds", "lightgcn_only", "fixed"),
]
DIMENSIONS = ["balanced", "diversity", "novelty"]
N_USERS = 100
OUT_DIR = PROJECT_ROOT / "results" / "sensitivity_n100"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def metric_winners(ds, a, b):
    gt_a = load_ground_truth_metrics(ds, a)
    gt_b = load_ground_truth_metrics(ds, b)
    ndcg_a, ndcg_b = gt_a.get("ndcg@10", 0), gt_b.get("ndcg@10", 0)
    cov_a, cov_b = gt_a.get("coverage@k", 0), gt_b.get("coverage@k", 0)
    return {
        "gt_ndcg_a": ndcg_a,
        "gt_ndcg_b": ndcg_b,
        "gt_cov_a": cov_a,
        "gt_cov_b": cov_b,
        "ndcg_winner": a if ndcg_a > ndcg_b else b,
        "cov_winner": a if cov_a > cov_b else b,
    }


def llm_winner(record):
    if record["prefer_a"] > record["prefer_b"]:
        return record["model_a"]
    if record["prefer_b"] > record["prefer_a"]:
        return record["model_b"]
    return "TIE"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config = JudgeConfig(
        model_name=MODEL_PATH_7B,
        torch_dtype="float16",
        max_new_tokens=32,
        do_sample=False,
    )
    log(f"Loading judge: {MODEL_PATH_7B}")
    judge = LLMJudge(config)
    all_records = []
    start = time.time()
    try:
        for ds, a, b in PAIRS:
            for dim in DIMENSIONS:
                log(f"[{ds}] {a} vs {b} ({dim}, n={N_USERS})")
                result = run_pairwise_dimension_eval(
                    judge, ds, a, b, dim, n_users=N_USERS, seed=SEED
                )
                record = {**asdict(result), **metric_winners(ds, a, b)}
                record["n_users_requested"] = N_USERS
                all_records.append(record)
                out = OUT_DIR / f"{ds}_{a}_vs_{b}_{dim}_n100.json"
                out.write_text(json.dumps(record, indent=2, default=str))
                w = llm_winner(record)
                log(
                    f"    LLM={w} PrefA={record['prefer_a']} PrefB={record['prefer_b']} "
                    f"Tie={record['tie']} Cons={record['consistency_rate']:.3f} "
                    f"NDCG={'Y' if w == record['ndcg_winner'] else 'N'} "
                    f"COV={'Y' if w == record['cov_winner'] else 'N'}"
                )
        (OUT_DIR / "all_results_sensitivity_n100.json").write_text(
            json.dumps(all_records, indent=2, default=str)
        )
        by_dim = {d: [] for d in DIMENSIONS}
        for r in all_records:
            by_dim[r["dimension"]].append(r)
        log("Summary")
        for dim, rows in by_dim.items():
            ndcg = sum(llm_winner(r) == r["ndcg_winner"] for r in rows)
            cov = sum(llm_winner(r) == r["cov_winner"] for r in rows)
            cons = sum(r["consistency_rate"] for r in rows) / len(rows)
            log(f"  {dim:9s} pairs={len(rows)} NDCG={ndcg/len(rows)*100:.1f}% COV={cov/len(rows)*100:.1f}% Cons={cons:.3f}")
    finally:
        judge.cleanup()
    log(f"Done in {(time.time() - start)/60:.1f} min")


if __name__ == "__main__":
    main()
