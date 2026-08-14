"""
Supplementary statistical analysis for the paper:
1. Binomial tests for alignment rates
2. Multi-metric alignment (Coverage, Novelty, Entropy, ILS)
3. Per-user correlation analysis
4. Preference margin analysis
5. Dataset statistics table
"""

import sys
import json
import pickle
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path("/root/autodl-tmp/llm_judge_recsys/code")))

from experiment_config import *
from data_loader import load_diversity_metrics

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_all_results():
    results = []
    for f in sorted(RESULTS_DIR.glob("*.json")):
        if f.name.startswith("all_results"):
            continue
        if "14b" in f.name:
            continue
        with open(f) as fh:
            results.append(json.load(fh))
    return results


def classify_pair(r):
    return "agree" if r["ndcg_winner"] == r["cov_winner"] else "conflict"


def llm_winner(r):
    if r["prefer_a"] > r["prefer_b"]:
        return r["model_a"]
    elif r["prefer_b"] > r["prefer_a"]:
        return r["model_b"]
    return "TIE"


# ===== 1. STATISTICAL TESTS =====
def statistical_tests():
    results = load_all_results()
    by_dim = defaultdict(list)
    for r in results:
        by_dim[r["dimension"]].append(r)

    print("=" * 70)
    print("1. STATISTICAL SIGNIFICANCE TESTS")
    print("=" * 70)

    for dim in ["balanced", "relevance", "diversity", "novelty"]:
        conflict = [r for r in by_dim[dim] if classify_pair(r) == "conflict"]
        n = len(conflict)
        ndcg_aligned = sum(1 for r in conflict if llm_winner(r) == r["ndcg_winner"])

        binom_result = stats.binomtest(ndcg_aligned, n, 0.5)
        binom_p = binom_result.pvalue
        ci = binom_result.proportion_ci(confidence_level=0.95)
        ci_lo, ci_hi = ci.low, ci.high

        print(f"  {dim:12s}: {ndcg_aligned}/{n} NDCG-aligned = {ndcg_aligned/n*100:.1f}%")
        print(f"     Binomial p = {binom_p:.4f} (H0: random=50%)")
        print(f"     95% CI: [{ci_lo*100:.1f}%, {ci_hi*100:.1f}%]")
        print()


# ===== 2. MULTI-METRIC ALIGNMENT =====
def multi_metric_alignment():
    results = load_all_results()
    by_dim = defaultdict(list)
    for r in results:
        by_dim[r["dimension"]].append(r)

    metrics = ["ndcg@10", "coverage@k", "novelty@k", "entropy@k", "ils_text@k"]
    metric_names = ["NDCG", "Coverage", "Novelty", "Entropy", "ILS-text (inv)"]

    print("=" * 70)
    print("2. MULTI-METRIC ALIGNMENT ANALYSIS")
    print("=" * 70)

    alignment_matrix = {}
    for dim in ["balanced", "relevance", "diversity", "novelty"]:
        alignment_matrix[dim] = {}
        for met, met_name in zip(metrics, metric_names):
            count_aligned = 0
            count_total = 0
            for r in by_dim[dim]:
                gt_a = load_diversity_metrics(r["dataset"], r["model_a"])
                gt_b = load_diversity_metrics(r["dataset"], r["model_b"])

                val_a = gt_a.get(met, 0)
                val_b = gt_b.get(met, 0)

                if met == "ils_text@k":
                    met_winner = r["model_a"] if val_a < val_b else r["model_b"]
                else:
                    met_winner = r["model_a"] if val_a > val_b else r["model_b"]

                w = llm_winner(r)
                if w != "TIE":
                    count_total += 1
                    if w == met_winner:
                        count_aligned += 1

            pct = count_aligned / count_total * 100 if count_total else 0
            alignment_matrix[dim][met_name] = pct

        print(f"  [{dim}]")
        for met_name in metric_names:
            print(f"    {met_name:12s}: {alignment_matrix[dim][met_name]:.1f}%")
        print()

    # Generate figure
    fig, ax = plt.subplots(figsize=(12, 6.4))
    dims = ["balanced", "relevance", "diversity", "novelty"]
    dim_labels = ["Balanced", "Relevance", "Diversity", "Novelty"]
    x = np.arange(len(metric_names))
    width = 0.18
    # Prompt palette deliberately avoids the metric colors (NDCG=blue,
    # Coverage=orange) used elsewhere, so color never doubles as both a
    # prompt and a metric encoding across figures.
    colors = ["#607D8B", "#4CAF50", "#009688", "#9C27B0"]

    for i, (dim, color) in enumerate(zip(dims, colors)):
        vals = [alignment_matrix[dim][mn] for mn in metric_names]
        ax.bar(x + i * width, vals, width, label=dim_labels[i], color=color, alpha=0.8)

    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(metric_names, fontsize=15)
    ax.set_ylabel("Alignment Rate (%)", fontsize=14)
    fig.suptitle("LLM Judge Alignment with Multiple Metrics by Prompt Type",
                 fontsize=15, y=0.995)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4,
               fontsize=12, frameon=False, bbox_to_anchor=(0.5, 0.93))
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylim(0, 100)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    fig_path = PROJECT_ROOT / "paper" / "figures" / "multi_metric_alignment.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fig_path}")

    return alignment_matrix


# ===== 3. PER-USER CORRELATION =====
def per_user_correlation():
    results = load_all_results()
    by_dim = defaultdict(list)
    for r in results:
        by_dim[r["dimension"]].append(r)

    print("=" * 70)
    print("3. PER-USER PREFERENCE MARGIN ANALYSIS")
    print("=" * 70)

    for dim in ["balanced", "diversity", "novelty"]:
        margins = []
        for r in by_dim[dim]:
            total = r["prefer_a"] + r["prefer_b"] + r["tie"]
            if total == 0:
                continue
            margin = abs(r["prefer_a"] - r["prefer_b"]) / total
            margins.append(margin)

        avg_margin = np.mean(margins) if margins else 0
        print(f"  {dim:12s}: avg preference margin = {avg_margin:.3f} "
              f"(min={min(margins):.3f}, max={max(margins):.3f})")

    # Preference strength vs metric difference
    print(f"\n  Correlation between NDCG difference and preference strength:")
    for dim in ["balanced", "diversity", "novelty"]:
        ndcg_diffs = []
        pref_strengths = []
        for r in by_dim[dim]:
            ndcg_diff = abs(r["gt_ndcg_a"] - r["gt_ndcg_b"])
            total = r["prefer_a"] + r["prefer_b"] + r["tie"]
            if total == 0:
                continue
            pref_str = (r["prefer_a"] - r["prefer_b"]) / total
            if r["ndcg_winner"] == r["model_b"]:
                pref_str = -pref_str
            ndcg_diffs.append(ndcg_diff)
            pref_strengths.append(pref_str)

        if len(ndcg_diffs) > 2:
            corr, p_val = stats.spearmanr(ndcg_diffs, pref_strengths)
            print(f"    {dim:12s}: Spearman r={corr:.3f}, p={p_val:.4f}")


# ===== 4. DATASET STATISTICS =====
def dataset_statistics():
    print("=" * 70)
    print("4. DATASET STATISTICS")
    print("=" * 70)

    ds_dirs = {"mind": "mind", "yelp": "yelp",
               "amazon_books": "amazon_books", "amazon_cds": "amazon_cds"}

    for ds_name, ds_dir in ds_dirs.items():
        stats_path = Path(f"/root/autodl-tmp/diversity_experiment/datasets/{ds_dir}/stats.json")
        if stats_path.exists():
            with open(stats_path) as f:
                s = json.load(f)
            print(f"  {ds_name}: users={s.get('n_users','?')}, items={s.get('n_items','?')}, "
                  f"interactions={s.get('n_train','?')}")
        else:
            print(f"  {ds_name}: stats.json not found at {stats_path}")

    print()
    # Per dataset, model metrics summary
    print("  Model diversity profile summary:")
    for ds_name, ds_dir in ds_dirs.items():
        metrics = load_diversity_metrics(ds_name, "gbaf")
        if metrics:
            ndcg = metrics.get("ndcg@10", 0)
            cov = metrics.get("coverage@k", 0)
            print(f"    {ds_name}/gbaf: NDCG={ndcg:.4f}, Cov={cov:.4f}")


# ===== 5. EFFECT SIZE ANALYSIS =====
def effect_size_analysis():
    results = load_all_results()
    by_dim = defaultdict(list)
    for r in results:
        by_dim[r["dimension"]].append(r)

    print("=" * 70)
    print("5. EFFECT SIZE: PREFERENCE DISTRIBUTION")
    print("=" * 70)

    # Standardized abbreviations consistent with the main paper / tables.
    ABBREV = {
        "gbaf": "gbaf", "fixed": "fixed", "gradnorm_only": "grdn",
        "lightgcn_only": "lgcn", "pcgrad": "pcgrd", "concat_mlp": "concat",
    }
    DS_SHORT = {"mind": "MIND", "yelp": "Yelp",
                "amazon_books": "AmzBk", "amazon_cds": "AmzCD"}

    def fmt_label(r):
        ds = DS_SHORT.get(r["dataset"], r["dataset"][:4])
        a = ABBREV.get(r["model_a"], r["model_a"])
        b = ABBREV.get(r["model_b"], r["model_b"])
        return f"{ds}\n{a}-{b}"

    fig, axes = plt.subplots(2, 2, figsize=(15, 12.5))
    for idx, dim in enumerate(["balanced", "relevance", "diversity", "novelty"]):
        ax = axes[idx // 2][idx % 2]
        conflict = [r for r in by_dim[dim] if classify_pair(r) == "conflict"]

        ndcg_fracs = []
        labels = []
        for r in conflict:
            total = r["prefer_a"] + r["prefer_b"] + r["tie"]
            ndcg_w = r["ndcg_winner"]
            if ndcg_w == r["model_a"]:
                ndcg_frac = r["prefer_a"] / total if total else 0
            else:
                ndcg_frac = r["prefer_b"] / total if total else 0
            ndcg_fracs.append(ndcg_frac * 100)
            labels.append(fmt_label(r))

        colors = ["#2196F3" if f > 50 else "#FF9800" for f in ndcg_fracs]
        bars = ax.bar(range(len(ndcg_fracs)), ndcg_fracs, color=colors, alpha=0.85)
        ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5)
        ax.set_ylim(0, 100)
        ax.set_ylabel("NDCG-winner preference %", fontsize=15)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=12, rotation=90)
        ax.tick_params(axis="y", labelsize=13)

        avg = np.mean(ndcg_fracs)
        # Put the mean in the title rather than over the bars.  The red line
        # remains the graphical encoding, while no text occupies data space.
        ax.set_title(f"{dim.capitalize()} Prompt (mean={avg:.0f}%)",
                     fontsize=17)
        ax.axhline(y=avg, color="red", linestyle="-", alpha=0.35)

        print(f"  {dim}: per-pair NDCG-winner preference %: {[f'{v:.0f}' for v in ndcg_fracs]}")
        print(f"    Mean={avg:.1f}%, Median={np.median(ndcg_fracs):.1f}%")

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#2196F3", alpha=0.85,
              label="NDCG winner preferred (>50%)"),
        Patch(facecolor="#FF9800", alpha=0.85,
              label="Diversity (Coverage) winner preferred (\u226450%)"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=2,
               fontsize=15, frameon=False, bbox_to_anchor=(0.5, 1.0))
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig_path = PROJECT_ROOT / "paper" / "figures" / "preference_distribution.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {fig_path}")


def main():
    statistical_tests()
    print()
    alignment_matrix = multi_metric_alignment()
    print()
    per_user_correlation()
    print()
    dataset_statistics()
    print()
    effect_size_analysis()
    print("\n\nAll supplementary analyses complete.")


if __name__ == "__main__":
    main()
