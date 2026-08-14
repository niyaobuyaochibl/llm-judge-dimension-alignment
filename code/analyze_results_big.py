"""
Analyze dimension experiment results and generate publication-ready figures and tables.
"""

import sys
import json
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiment_config import *

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_all_results():
    """Load all experiment result JSONs."""
    results = []
    for f in sorted(RESULTS_DIR.glob("*.json")):
        if f.name == "all_results_7b.json":
            continue
        with open(f) as fh:
            results.append(json.load(fh))
    return results


def classify_pair(r):
    """Determine if NDCG and Coverage winners agree or conflict."""
    ndcg_w = r["ndcg_winner"]
    cov_w = r["cov_winner"]
    return "agree" if ndcg_w == cov_w else "conflict"


def llm_winner(r):
    """Determine which model the LLM prefers."""
    if r["prefer_a"] > r["prefer_b"]:
        return r["model_a"]
    elif r["prefer_b"] > r["prefer_a"]:
        return r["model_b"]
    return "TIE"


def llm_aligns_with(r):
    """Determine which metric the LLM aligns with."""
    w = llm_winner(r)
    if w == "TIE":
        return "tie"
    if w == r["ndcg_winner"] and w == r["cov_winner"]:
        return "both"
    if w == r["ndcg_winner"]:
        return "ndcg"
    if w == r["cov_winner"]:
        return "coverage"
    return "neither"


def main():
    results = load_all_results()
    print(f"Loaded {len(results)} experiment results\n")

    # Group by dimension
    by_dim = defaultdict(list)
    for r in results:
        by_dim[r["dimension"]].append(r)

    # === TABLE 1: Overall alignment rates ===
    print("=" * 80)
    print("TABLE: LLM Alignment with Accuracy vs Diversity Metrics")
    print("=" * 80)
    print(f"{'Dimension':<12} {'#Pairs':>6} {'#Conflict':>9} {'NDCG%':>7} {'COV%':>7} {'Both%':>7} {'TIE%':>7} {'Avg Cons':>9}")
    print("-" * 80)

    alignment_summary = {}
    for dim in ["balanced", "relevance", "diversity", "novelty"]:
        dim_results = by_dim[dim]
        n_total = len(dim_results)

        conflict_results = [r for r in dim_results if classify_pair(r) == "conflict"]
        n_conflict = len(conflict_results)

        alignments = [llm_aligns_with(r) for r in conflict_results]
        n_ndcg = alignments.count("ndcg")
        n_cov = alignments.count("coverage")
        n_tie = alignments.count("tie")

        avg_cons = np.mean([r["consistency_rate"] for r in dim_results])

        pct_ndcg = n_ndcg / n_conflict * 100 if n_conflict else 0
        pct_cov = n_cov / n_conflict * 100 if n_conflict else 0
        pct_tie = n_tie / n_conflict * 100 if n_conflict else 0

        print(f"{dim:<12} {n_total:>6} {n_conflict:>9} {pct_ndcg:>6.1f}% {pct_cov:>6.1f}% "
              f"{'--':>7} {pct_tie:>6.1f}% {avg_cons:>8.3f}")

        alignment_summary[dim] = {
            "n_conflict": n_conflict, "pct_ndcg": pct_ndcg,
            "pct_cov": pct_cov, "avg_cons": avg_cons,
        }

    # === TABLE 2: Per-dataset breakdown ===
    print(f"\n{'=' * 80}")
    print("TABLE: Per-Dataset Alignment (Conflict Pairs Only)")
    print("=" * 80)
    print(f"{'Dataset':<14} {'Dimension':<12} {'Pair':<35} {'LLM Winner':<16} {'NDCG':>5} {'COV':>5} {'Cons':>5}")
    print("-" * 100)

    for ds in DATASETS:
        for dim in ["balanced", "relevance", "diversity", "novelty"]:
            for r in by_dim[dim]:
                if r["dataset"] != ds:
                    continue
                pair_type = classify_pair(r)
                if pair_type == "agree":
                    continue
                w = llm_winner(r)
                align = llm_aligns_with(r)
                ndcg_mark = "Y" if align == "ndcg" else "N"
                cov_mark = "Y" if align == "coverage" else "N"
                pair_str = f"{r['model_a']} vs {r['model_b']}"
                print(f"{ds:<14} {dim:<12} {pair_str:<35} {w:<16} {ndcg_mark:>5} {cov_mark:>5} {r['consistency_rate']:>5.2f}")
        print()

    # === TABLE 3: Dimension-specific prompt effectiveness ===
    print(f"\n{'=' * 80}")
    print("TABLE: Dimension-Specific Prompt Effectiveness")
    print("(Does the relevance prompt improve NDCG alignment?")
    print(" Does the diversity prompt improve Coverage alignment?)")
    print("=" * 80)

    for ds in DATASETS:
        print(f"\n[{ds}]")
        for r_bal in by_dim["balanced"]:
            if r_bal["dataset"] != ds or classify_pair(r_bal) == "agree":
                continue
            pair_str = f"{r_bal['model_a']} vs {r_bal['model_b']}"
            bal_align = llm_aligns_with(r_bal)

            for dim in ["relevance", "diversity", "novelty"]:
                for r_dim in by_dim[dim]:
                    if (r_dim["dataset"] == ds and
                        r_dim["model_a"] == r_bal["model_a"] and
                        r_dim["model_b"] == r_bal["model_b"]):
                        dim_align = llm_aligns_with(r_dim)
                        shifted = "SHIFTED" if dim_align != bal_align else "same"
                        print(f"  {pair_str:<35} {dim:<12} bal={bal_align:<8} dim={dim_align:<8} {shifted}")

    # === FIGURE 1: Alignment bar chart ===
    # This overview is typeset across both IEEE columns.  A compact canvas
    # removes unused vertical space while preserving print-size labels.
    fig, axes = plt.subplots(1, 2, figsize=(14, 3.15))

    dims = ["balanced", "relevance", "diversity", "novelty"]
    dim_labels = ["Balanced\n(Standard)", "Relevance\nOnly", "Diversity\nOnly", "Novelty\nOnly"]
    ndcg_pcts = [alignment_summary[d]["pct_ndcg"] for d in dims]
    cov_pcts = [alignment_summary[d]["pct_cov"] for d in dims]
    cons_vals = [alignment_summary[d]["avg_cons"] for d in dims]

    x = np.arange(len(dims))
    width = 0.35

    ax1 = axes[0]
    bars1 = ax1.bar(x - width/2, ndcg_pcts, width, label="Aligns with NDCG", color="#2196F3")
    bars2 = ax1.bar(x + width/2, cov_pcts, width,
                    label="Aligns with Coverage", color="#FF9800",
                    hatch="//", edgecolor="#C96F00", linewidth=0.8)
    ax1.set_ylabel("Alignment Rate (%)", fontsize=17)
    ax1.set_title("(a) Alignment on Conflict Pairs", fontsize=17)
    ax1.set_xticks(x)
    ax1.set_xticklabels(dim_labels, fontsize=16)
    ax1.tick_params(axis="y", labelsize=15)
    # Direct series labels use reserved headroom and cannot conceal a bar.
    ax1.text(0.37, 0.975, "NDCG", transform=ax1.transAxes,
             ha="center", va="top", fontsize=14, fontweight="bold",
             color="#177BC1")
    ax1.text(0.63, 0.975, "Coverage", transform=ax1.transAxes,
             ha="center", va="top", fontsize=14, fontweight="bold",
             color="#C96F00")
    ax1.set_ylim(0, 105)
    ax1.axhline(y=50, color="gray", linestyle="--", alpha=0.5)
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f"{bar.get_height():.0f}%", ha="center", va="bottom", fontsize=15)
    for bar in bars2:
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f"{bar.get_height():.0f}%", ha="center", va="bottom", fontsize=15)

    ax2 = axes[1]
    # Prompt palette (avoids metric blue/orange used in panel (a)).
    prompt_colors = ["#607D8B", "#4CAF50", "#009688", "#9C27B0"]
    bars3 = ax2.bar(x, cons_vals, 0.5, color=prompt_colors)
    ax2.set_ylabel("Bidirectional Consistency", fontsize=17)
    ax2.set_title("(b) Average Consistency by Prompt Type", fontsize=17)
    ax2.set_xticks(x)
    ax2.set_xticklabels(dim_labels, fontsize=16)
    ax2.tick_params(axis="y", labelsize=15)
    # All observed values lie below 0.5; ending at 0.6 avoids devoting half
    # of the panel to empty space while retaining a zero baseline.
    ax2.set_ylim(0, 0.6)
    for bar in bars3:
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=15)

    plt.tight_layout()
    fig_path = PROJECT_ROOT / "paper" / "figures_bigfont_scratch" / "alignment_overview.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved figure: {fig_path}")

    # === FIGURE 2: Per-dataset heatmap ===
    DS_DISPLAY = {"mind": "MIND", "yelp": "Yelp",
                  "amazon_books": "Amazon\nBooks", "amazon_cds": "Amazon\nCDs"}
    fig2, axes2 = plt.subplots(1, 4, figsize=(13, 4.42), sharey=True)
    for idx, dim in enumerate(dims):
        ax = axes2[idx]
        ds_labels = DATASETS
        ndcg_by_ds = []
        cov_by_ds = []
        for ds in DATASETS:
            conflict_rs = [r for r in by_dim[dim]
                          if r["dataset"] == ds and classify_pair(r) == "conflict"]
            if not conflict_rs:
                ndcg_by_ds.append(0)
                cov_by_ds.append(0)
                continue
            aligns = [llm_aligns_with(r) for r in conflict_rs]
            ndcg_by_ds.append(aligns.count("ndcg") / len(conflict_rs) * 100)
            cov_by_ds.append(aligns.count("coverage") / len(conflict_rs) * 100)

        y = np.arange(len(DATASETS))
        ax.barh(y - 0.2, ndcg_by_ds, 0.4, label="NDCG", color="#2196F3", alpha=0.8)
        ax.barh(y + 0.2, cov_by_ds, 0.4, label="Coverage", color="#FF9800", alpha=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels([DS_DISPLAY.get(d, d.replace("_", "\n")) for d in DATASETS], fontsize=14)
        ax.set_xlabel("Alignment %", fontsize=14)
        ax.set_title(dim_labels[idx], fontsize=14)
        ax.tick_params(axis="x", labelsize=13)
        ax.set_xlim(0, 105)
        ax.axvline(x=50, color="gray", linestyle="--", alpha=0.5)
    # Keep the legend completely outside every plotting area so it cannot
    # conceal a horizontal bar (the earlier lower-right legend covered the
    # MIND NDCG bar in panel 1).
    handles, labels = axes2[0].get_legend_handles_labels()
    fig2.suptitle("LLM Judge Alignment by Dataset and Prompt Type",
                  fontsize=16, y=0.995)
    fig2.legend(handles, labels, loc="upper center", ncol=2,
                fontsize=13, frameon=False, bbox_to_anchor=(0.5, 0.945))
    fig2.tight_layout(rect=[0, 0, 1, 0.88])
    fig2_path = PROJECT_ROOT / "paper" / "figures_bigfont_scratch" / "alignment_by_dataset.png"
    plt.savefig(fig2_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {fig2_path}")

    # === FIGURE 3: Prompt shift analysis ===
    shifts = {"relevance": {"to_ndcg": 0, "to_cov": 0, "same": 0},
              "diversity": {"to_ndcg": 0, "to_cov": 0, "same": 0},
              "novelty": {"to_ndcg": 0, "to_cov": 0, "same": 0}}

    for ds in DATASETS:
        for r_bal in by_dim["balanced"]:
            if r_bal["dataset"] != ds or classify_pair(r_bal) == "agree":
                continue
            bal_align = llm_aligns_with(r_bal)
            for dim in ["relevance", "diversity", "novelty"]:
                for r_dim in by_dim[dim]:
                    if (r_dim["dataset"] == ds and
                        r_dim["model_a"] == r_bal["model_a"] and
                        r_dim["model_b"] == r_bal["model_b"]):
                        dim_align = llm_aligns_with(r_dim)
                        if dim_align == bal_align:
                            shifts[dim]["same"] += 1
                        elif dim_align == "coverage":
                            shifts[dim]["to_cov"] += 1
                        elif dim_align == "ndcg":
                            shifts[dim]["to_ndcg"] += 1

    print(f"\n{'=' * 60}")
    print("Prompt Shift Analysis (from balanced baseline)")
    print("=" * 60)
    for dim, s in shifts.items():
        total = sum(s.values())
        print(f"  {dim}: same={s['same']}/{total}, to_cov={s['to_cov']}/{total}, to_ndcg={s['to_ndcg']}/{total}")

    plt.close("all")
    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
