"""
Statistical analysis of DeepSeek-V3 validation results.
Computes user-level binomial tests, pair-level alignment,
and comparison with Qwen-7B/14B/Mistral results.
"""

import sys
import json
import numpy as np
from pathlib import Path
from scipy import stats
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiment_config import RESULTS_DIR

DEEPSEEK_DIR = RESULTS_DIR / "deepseek"


def load_all_results(result_dir, suffix=""):
    """Load individual result files from a directory."""
    results = []
    for f in sorted(result_dir.glob("*.json")):
        if f.name.startswith("all_results"):
            continue
        with open(f) as fh:
            results.append(json.load(fh))
    return results


def analyze_pair_level(results):
    """Pair-level analysis: which model does the LLM pick at pair level."""
    print("=" * 70)
    print("PAIR-LEVEL ANALYSIS (DeepSeek-V3)")
    print("=" * 70)

    for dim in ["balanced", "diversity", "novelty"]:
        dim_results = [r for r in results if r["dimension"] == dim]
        conflict_results = [r for r in dim_results
                           if r.get("ndcg_winner") != r.get("cov_winner")]

        print(f"\n--- {dim.upper()} prompt ---")
        ndcg_aligned = 0
        for r in conflict_results:
            llm_w = r["model_a"] if r["prefer_a"] > r["prefer_b"] else (
                r["model_b"] if r["prefer_b"] > r["prefer_a"] else "TIE")
            is_ndcg = llm_w == r["ndcg_winner"]
            if is_ndcg:
                ndcg_aligned += 1
            print(f"  {r['dataset']:12s} {r['model_a']:15s} vs {r['model_b']:10s}: "
                  f"A={r['prefer_a']:2d} B={r['prefer_b']:2d} cons={r['consistency_rate']:.2f} "
                  f"LLM={llm_w:15s} NDCG={'Y' if is_ndcg else 'N'}")

        n = len(conflict_results)
        pct = ndcg_aligned / n * 100 if n > 0 else 0
        if n > 0:
            p_binom = stats.binomtest(ndcg_aligned, n, 0.5, alternative='greater').pvalue
            print(f"  NDCG alignment: {ndcg_aligned}/{n} ({pct:.1f}%), "
                  f"binomial p={p_binom:.4f}")
        print(f"  Avg consistency: "
              f"{np.mean([r['consistency_rate'] for r in dim_results]):.3f}")


def analyze_user_level(results):
    """User-level analysis: per-trial NDCG preference rates."""
    print("\n" + "=" * 70)
    print("USER-LEVEL ANALYSIS (DeepSeek-V3)")
    print("=" * 70)

    for dim in ["balanced", "diversity", "novelty"]:
        dim_results = [r for r in results if r["dimension"] == dim]
        conflict_results = [r for r in dim_results
                           if r.get("ndcg_winner") != r.get("cov_winner")]

        prefers_ndcg = 0
        prefers_cov = 0
        total_valid = 0

        for r in conflict_results:
            ndcg_w = r["ndcg_winner"]
            cov_w = r["cov_winner"]
            for trial in r["per_trial"]:
                ep = trial["effective_preference"]
                if ep == "INCONSISTENT" or ep == "TIE":
                    continue
                if ep == ndcg_w:
                    prefers_ndcg += 1
                    total_valid += 1
                elif ep == cov_w:
                    prefers_cov += 1
                    total_valid += 1

        if total_valid > 0:
            rate = prefers_ndcg / total_valid
            p_binom = stats.binomtest(prefers_ndcg, total_valid, 0.5,
                                       alternative='greater').pvalue
            print(f"\n  {dim:12s}: NDCG pref = {prefers_ndcg}/{total_valid} "
                  f"({rate*100:.1f}%), p={p_binom:.6f}")
        else:
            print(f"\n  {dim:12s}: no valid trials")


def compare_with_other_models():
    """Compare DeepSeek results with Qwen-7B, Qwen-14B, Mistral-7B."""
    print("\n" + "=" * 70)
    print("CROSS-MODEL COMPARISON")
    print("=" * 70)

    model_dirs = {
        "Qwen-7B": RESULTS_DIR,
        "Qwen-14B": RESULTS_DIR / "14b",
        "Mistral-7B": RESULTS_DIR / "mistral",
        "DeepSeek-V3": RESULTS_DIR / "deepseek",
    }

    for model_name, mdir in model_dirs.items():
        if not mdir.exists():
            continue

        results = load_all_results(mdir)
        if not results:
            continue

        print(f"\n--- {model_name} ---")
        for dim in ["balanced", "diversity", "novelty"]:
            dim_results = [r for r in results if r["dimension"] == dim]
            conflict_results = [r for r in dim_results
                               if r.get("ndcg_winner") != r.get("cov_winner")]

            # Pair-level
            ndcg_pairs = 0
            for r in conflict_results:
                llm_w = r["model_a"] if r["prefer_a"] > r["prefer_b"] else (
                    r["model_b"] if r["prefer_b"] > r["prefer_a"] else "TIE")
                if llm_w == r["ndcg_winner"]:
                    ndcg_pairs += 1
            n_pairs = len(conflict_results)
            pair_pct = ndcg_pairs / n_pairs * 100 if n_pairs > 0 else 0

            # User-level
            prefers_ndcg = 0
            total_valid = 0
            for r in conflict_results:
                ndcg_w = r["ndcg_winner"]
                cov_w = r["cov_winner"]
                for trial in r["per_trial"]:
                    ep = trial["effective_preference"]
                    if ep in ("INCONSISTENT", "TIE"):
                        continue
                    if ep == ndcg_w:
                        prefers_ndcg += 1
                        total_valid += 1
                    elif ep == cov_w:
                        total_valid += 1

            user_pct = prefers_ndcg / total_valid * 100 if total_valid > 0 else 0
            p_val = stats.binomtest(prefers_ndcg, total_valid, 0.5,
                                     alternative='greater').pvalue if total_valid > 0 else 1.0

            avg_cons = np.mean([r['consistency_rate'] for r in dim_results]) if dim_results else 0

            print(f"  {dim:12s}: pair={ndcg_pairs}/{n_pairs} ({pair_pct:5.1f}%), "
                  f"user={prefers_ndcg}/{total_valid} ({user_pct:5.1f}%), "
                  f"p={p_val:.6f}, cons={avg_cons:.3f}")


def detailed_table():
    """Print a detailed table for all DeepSeek experiments."""
    print("\n" + "=" * 70)
    print("DETAILED RESULTS TABLE (for paper)")
    print("=" * 70)

    results = load_all_results(DEEPSEEK_DIR)

    print(f"\n{'Dataset':12s} {'Model A':15s} {'Model B':10s} {'Prompt':10s} "
          f"{'Pref A':>6s} {'Pref B':>6s} {'Tie':>4s} {'Cons':>5s} "
          f"{'NDCG':>5s} {'COV':>5s}")
    print("-" * 95)

    for r in results:
        llm_w = r["model_a"] if r["prefer_a"] > r["prefer_b"] else (
            r["model_b"] if r["prefer_b"] > r["prefer_a"] else "TIE")
        ndcg_ok = "Y" if llm_w == r.get("ndcg_winner") else "N"
        cov_ok = "Y" if llm_w == r.get("cov_winner") else "N"
        incon = r["n_trials"] - r["prefer_a"] - r["prefer_b"] - r["tie"]

        print(f"{r['dataset']:12s} {r['model_a']:15s} {r['model_b']:10s} {r['dimension']:10s} "
              f"{r['prefer_a']:6d} {r['prefer_b']:6d} {r['tie']:4d} {r['consistency_rate']:5.2f} "
              f"{ndcg_ok:>5s} {cov_ok:>5s}")


def main():
    results = load_all_results(DEEPSEEK_DIR)
    print(f"Loaded {len(results)} DeepSeek-V3 experiment results\n")

    detailed_table()
    analyze_pair_level(results)
    analyze_user_level(results)
    compare_with_other_models()


if __name__ == "__main__":
    main()
