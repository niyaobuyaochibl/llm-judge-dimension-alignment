"""
Aggregate diversity evaluation results and generate tables/figures for the
Applied Intelligence paper: "The Impact of Text Fusion on Recommendation Diversity".

Usage:
    python analyze_diversity.py --results_dir diversity_results --output_dir paper_figures
"""

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[WARN] matplotlib not available, skipping figure generation")


DATASETS = ["ml1m", "amazonff", "yelp", "mind", "amazon_books", "amazon_cds"]
CORE_VARIANTS = ["lightgcn_only", "fixed", "gbaf", "gradnorm_only", "concat_mlp", "pcgrad"]
ACCURACY_METRICS = ["recall@10", "ndcg@10"]
DIVERSITY_METRICS = ["coverage@k", "gini@k", "entropy@k", "ils_text@k", "novelty@k", "tail_coverage@k"]

MODE_LABELS = {
    "ml1m": "Borderline",
    "amazonff": "A: Beneficial",
    "yelp": "B: Collapse",
    "mind": "C: No-Value",
    "amazon_books": "B: Collapse",
    "amazon_cds": "B: Collapse",
}

VARIANT_DISPLAY = {
    "lightgcn_only": "CF-only",
    "fixed": "Fixed (λ=0.5)",
    "gbaf": "GBAF",
    "gradnorm_only": "GradNorm",
    "concat_mlp": "Concat-MLP",
    "pcgrad": "PCGrad",
}


def load_results(results_dir):
    """Load all diversity JSON result files."""
    records = []
    for dataset in DATASETS:
        for variant in CORE_VARIANTS:
            d = Path(results_dir) / dataset / variant
            if not d.exists():
                continue
            for f in sorted(d.glob("diversity_seed*.json")):
                with open(f) as fh:
                    rec = json.load(fh)
                rec["dataset"] = dataset
                rec["variant"] = variant
                rec["mode"] = MODE_LABELS.get(dataset, "?")
                records.append(rec)
    return pd.DataFrame(records)


def build_main_table(df, output_dir):
    """Table 2: Full results (mean ± std across seeds)."""
    rows = []
    for dataset in DATASETS:
        for variant in CORE_VARIANTS:
            sub = df[(df["dataset"] == dataset) & (df["variant"] == variant)]
            if sub.empty:
                continue
            row = {"Dataset": dataset, "Mode": MODE_LABELS.get(dataset, "?"),
                   "Method": VARIANT_DISPLAY.get(variant, variant)}
            for m in ACCURACY_METRICS + DIVERSITY_METRICS:
                if m in sub.columns:
                    mean_v = sub[m].mean()
                    std_v = sub[m].std()
                    row[m] = f"{mean_v:.4f}±{std_v:.4f}"
                    row[f"{m}_mean"] = mean_v
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(Path(output_dir) / "main_results_table.csv", index=False)
    print(f"[Saved] main_results_table.csv ({len(table)} rows)")

    latex_path = Path(output_dir) / "main_results_table.tex"
    cols = ["Dataset", "Mode", "Method"] + ACCURACY_METRICS + DIVERSITY_METRICS
    avail = [c for c in cols if c in table.columns]
    table[avail].to_latex(latex_path, index=False, escape=False)
    print(f"[Saved] main_results_table.tex")
    return table


def build_delta_table(df, output_dir):
    """Table 3: % change vs CF-only baseline for each metric."""
    rows = []
    for dataset in DATASETS:
        cf_sub = df[(df["dataset"] == dataset) & (df["variant"] == "lightgcn_only")]
        if cf_sub.empty:
            continue
        baselines = {}
        for m in ACCURACY_METRICS + DIVERSITY_METRICS:
            if m in cf_sub.columns:
                baselines[m] = cf_sub[m].mean()

        for variant in CORE_VARIANTS:
            if variant == "lightgcn_only":
                continue
            sub = df[(df["dataset"] == dataset) & (df["variant"] == variant)]
            if sub.empty:
                continue
            row = {"Dataset": dataset, "Mode": MODE_LABELS.get(dataset, "?"),
                   "Method": VARIANT_DISPLAY.get(variant, variant)}
            for m in ACCURACY_METRICS + DIVERSITY_METRICS:
                if m in sub.columns and m in baselines and baselines[m] != 0:
                    delta = (sub[m].mean() - baselines[m]) / abs(baselines[m]) * 100
                    row[f"Δ{m}(%)"] = f"{delta:+.1f}"
                    row[f"Δ{m}_val"] = delta
            rows.append(row)
    delta_table = pd.DataFrame(rows)
    delta_table.to_csv(Path(output_dir) / "delta_table.csv", index=False)
    print(f"[Saved] delta_table.csv ({len(delta_table)} rows)")
    return delta_table


def plot_pareto_scatter(df, output_dir):
    """Figure 1: Accuracy vs Diversity scatter (one subplot per dataset)."""
    if not HAS_MATPLOTLIB:
        return
    datasets_to_plot = [d for d in DATASETS if d in df["dataset"].unique()]
    n = len(datasets_to_plot)
    if n == 0:
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    colors = {"lightgcn_only": "#757575", "fixed": "#E53935", "gbaf": "#43A047",
              "gradnorm_only": "#1E88E5", "concat_mlp": "#FB8C00", "pcgrad": "#8E24AA"}
    markers = {"lightgcn_only": "s", "fixed": "v", "gbaf": "D",
               "gradnorm_only": "^", "concat_mlp": "o", "pcgrad": "X"}

    for idx, dataset in enumerate(datasets_to_plot):
        ax = axes[idx]
        sub = df[df["dataset"] == dataset]
        for variant in CORE_VARIANTS:
            vsub = sub[sub["variant"] == variant]
            if vsub.empty or "recall@10" not in vsub.columns or "coverage@k" not in vsub.columns:
                continue
            ax.scatter(
                vsub["recall@10"].values, vsub["coverage@k"].values,
                c=colors.get(variant, "gray"), marker=markers.get(variant, "o"),
                s=80, label=VARIANT_DISPLAY.get(variant, variant), alpha=0.8, edgecolors="white", linewidths=0.5,
            )
        ax.set_title(f"{dataset} ({MODE_LABELS.get(dataset, '')})", fontsize=11)
        ax.set_xlabel("Recall@10")
        ax.set_ylabel("Coverage@10")
        ax.grid(True, alpha=0.3)

    for idx in range(len(datasets_to_plot), len(axes)):
        axes[idx].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(CORE_VARIANTS), fontsize=9, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(Path(output_dir) / "pareto_scatter.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(Path(output_dir) / "pareto_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[Saved] pareto_scatter.pdf/png")


def plot_mode_comparison_bars(df, output_dir):
    """Figure 2: Grouped bar chart comparing diversity metrics across modes."""
    if not HAS_MATPLOTLIB:
        return
    mode_datasets = {"A: Beneficial": ["amazonff"], "B: Collapse": ["yelp", "amazon_books", "amazon_cds"],
                     "C: No-Value": ["mind"]}
    metrics_to_plot = ["coverage@k", "gini@k", "novelty@k", "tail_coverage@k"]
    variants_to_show = ["lightgcn_only", "fixed", "gbaf", "concat_mlp"]

    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(16, 4.5))
    colors = {"lightgcn_only": "#757575", "fixed": "#E53935", "gbaf": "#43A047", "concat_mlp": "#FB8C00"}
    mode_order = ["A: Beneficial", "B: Collapse", "C: No-Value"]

    for ax_idx, metric in enumerate(metrics_to_plot):
        ax = axes[ax_idx]
        x = np.arange(len(mode_order))
        width = 0.18
        for v_idx, variant in enumerate(variants_to_show):
            vals = []
            for mode in mode_order:
                dslist = mode_datasets[mode]
                sub = df[(df["dataset"].isin(dslist)) & (df["variant"] == variant)]
                vals.append(sub[metric].mean() if not sub.empty and metric in sub.columns else 0)
            ax.bar(x + v_idx * width, vals, width, label=VARIANT_DISPLAY.get(variant, variant),
                   color=colors.get(variant, "gray"), edgecolor="white", linewidth=0.5)
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(mode_order, fontsize=9)
        ax.set_title(metric.replace("@k", "@10"), fontsize=11)
        ax.grid(axis="y", alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.05))
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(Path(output_dir) / "mode_comparison_bars.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(Path(output_dir) / "mode_comparison_bars.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[Saved] mode_comparison_bars.pdf/png")


def plot_recovery_asymmetry(df, output_dir):
    """Figure 3: Accuracy recovery vs diversity recovery for GradNorm on collapse datasets."""
    if not HAS_MATPLOTLIB:
        return
    collapse_datasets = ["yelp", "amazon_books", "amazon_cds"]
    repair_variants = ["gbaf", "gradnorm_only", "concat_mlp"]
    acc_metric = "recall@10"
    div_metrics = ["coverage@k", "novelty@k", "tail_coverage@k"]

    rows = []
    for dataset in collapse_datasets:
        cf_sub = df[(df["dataset"] == dataset) & (df["variant"] == "lightgcn_only")]
        fixed_sub = df[(df["dataset"] == dataset) & (df["variant"] == "fixed")]
        if cf_sub.empty or fixed_sub.empty:
            continue
        for variant in repair_variants:
            vsub = df[(df["dataset"] == dataset) & (df["variant"] == variant)]
            if vsub.empty:
                continue
            cf_acc = cf_sub[acc_metric].mean()
            fixed_acc = fixed_sub[acc_metric].mean()
            repair_acc = vsub[acc_metric].mean()
            if cf_acc != fixed_acc:
                acc_recovery = (repair_acc - fixed_acc) / (cf_acc - fixed_acc) * 100
            else:
                acc_recovery = 0
            for dm in div_metrics:
                if dm not in cf_sub.columns:
                    continue
                cf_div = cf_sub[dm].mean()
                fixed_div = fixed_sub[dm].mean()
                repair_div = vsub[dm].mean()
                if cf_div != fixed_div:
                    div_recovery = (repair_div - fixed_div) / (cf_div - fixed_div) * 100
                else:
                    div_recovery = 0
                rows.append({
                    "dataset": dataset, "variant": VARIANT_DISPLAY.get(variant, variant),
                    "acc_recovery": acc_recovery, "div_metric": dm, "div_recovery": div_recovery,
                })

    if not rows:
        return
    rec_df = pd.DataFrame(rows)
    rec_df.to_csv(Path(output_dir) / "recovery_asymmetry.csv", index=False)
    print(f"[Saved] recovery_asymmetry.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="diversity_results")
    parser.add_argument("--output_dir", type=str, default="paper_figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = load_results(args.results_dir)
    if df.empty:
        print("[ERROR] No results found. Run evaluate_diversity.py first.")
        return

    print(f"[Loaded] {len(df)} result records from {df['dataset'].nunique()} datasets")

    build_main_table(df, args.output_dir)
    build_delta_table(df, args.output_dir)
    plot_pareto_scatter(df, args.output_dir)
    plot_mode_comparison_bars(df, args.output_dir)
    plot_recovery_asymmetry(df, args.output_dir)

    print(f"\nAll outputs saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
