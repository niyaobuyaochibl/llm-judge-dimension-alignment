"""
14B validation: Run balanced + diversity + novelty experiments with Qwen-14B
on a subset of key pairs to validate findings.
"""

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path("/root/autodl-tmp/llm_judge_recsys/code")))

from experiment_config import *
from run_dimension_experiments import (
    run_pairwise_dimension_eval, load_ground_truth_metrics, log,
)
from judge_pipeline import LLMJudge, JudgeConfig
from dataclasses import asdict


VALIDATION_PAIRS = {
    "mind": [("gbaf", "fixed"), ("lightgcn_only", "fixed")],
    "yelp": [("gbaf", "concat_mlp"), ("lightgcn_only", "fixed")],
    "amazon_cds": [("gbaf", "fixed"), ("lightgcn_only", "fixed")],
}

VALIDATION_DIMS = ["balanced", "diversity", "novelty"]


def main():
    out_dir = RESULTS_DIR / "14b"
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"Loading Qwen-14B (4-bit): {MODEL_PATH_14B}")
    config = JudgeConfig(
        model_name=MODEL_PATH_14B,
        torch_dtype="float16",
        max_new_tokens=32,
        do_sample=False,
        load_in_4bit=True,
    )
    judge = LLMJudge(config)
    start = time.time()

    all_results = []
    for ds, pairs in VALIDATION_PAIRS.items():
        for model_a, model_b in pairs:
            gt_a = load_ground_truth_metrics(ds, model_a)
            gt_b = load_ground_truth_metrics(ds, model_b)

            for dim in VALIDATION_DIMS:
                log(f"  [{ds}] {model_a} vs {model_b} ({dim}) ...")
                try:
                    result = run_pairwise_dimension_eval(
                        judge, ds, model_a, model_b, dim, n_users=N_USERS, seed=SEED,
                    )
                    record = {
                        **asdict(result),
                        "gt_ndcg_a": gt_a.get("ndcg@10", 0),
                        "gt_ndcg_b": gt_b.get("ndcg@10", 0),
                        "gt_cov_a": gt_a.get("coverage@k", 0),
                        "gt_cov_b": gt_b.get("coverage@k", 0),
                        "ndcg_winner": model_a if gt_a.get("ndcg@10",0) > gt_b.get("ndcg@10",0) else model_b,
                        "cov_winner": model_a if gt_a.get("coverage@k",0) > gt_b.get("coverage@k",0) else model_b,
                    }
                    all_results.append(record)
                    fname = out_dir / f"{ds}_{model_a}_vs_{model_b}_{dim}_14b.json"
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

    with open(out_dir / "all_results_14b.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    elapsed = time.time() - start
    log(f"\n14B validation complete! {elapsed/60:.1f} min")
    judge.cleanup()


if __name__ == "__main__":
    main()
