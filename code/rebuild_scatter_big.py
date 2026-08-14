"""Rebuild the supplemental scatter figure without in-axes legends."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ORDER = ["balanced", "relevance", "diversity", "novelty"]
LABELS = {
    "balanced": "Balanced",
    "relevance": "Relevance",
    "diversity": "Diversity",
    "novelty": "Novelty",
}
COLORS = {
    "balanced": "#42A5F5",
    "relevance": "#81C784",
    "diversity": "#FFA726",
    "novelty": "#AB47BC",
}
MARKERS = {
    "balanced": "o",
    "relevance": "s",
    "diversity": "^",
    "novelty": "D",
}


def build(input_csv: Path, output_png: Path) -> None:
    data = pd.read_csv(input_csv)
    data = data[data["is_conflict"].astype(str).str.lower().eq("true")].copy()
    total = data["prefer_a"] + data["prefer_b"] + data["tie"]
    data["preference_margin"] = (
        (data["prefer_a"] - data["prefer_b"]).abs() / total
    )
    data["coverage_gap"] = (
        data["coverage_at_k_a"] - data["coverage_at_k_b"]
    ).abs()
    data["ndcg_preference_pct"] = data.apply(
        lambda row: 100.0
        * (row["prefer_a"] if row["ndcg_winner"] == row["model_a"]
           else row["prefer_b"])
        / (row["prefer_a"] + row["prefer_b"] + row["tie"]),
        axis=1,
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.1))
    for dimension in ORDER:
        subset = data[data["dimension"] == dimension]
        style = dict(
            s=70,
            marker=MARKERS[dimension],
            color=COLORS[dimension],
            edgecolor="#555555",
            linewidth=0.7,
            alpha=0.9,
            label=LABELS[dimension],
        )
        axes[0].scatter(
            subset["consistency_rate"], subset["preference_margin"], **style
        )
        axes[1].scatter(
            subset["coverage_gap"], subset["ndcg_preference_pct"], **style
        )

    axes[0].set_title("(a) Consistency vs. Preference Margin\n(per conflict pair)",
                      fontsize=15)
    axes[0].set_xlabel("Bidirectional Consistency", fontsize=14)
    axes[0].set_ylabel("Preference Margin", fontsize=14)
    axes[0].set_xlim(-0.02, 0.85)
    axes[0].set_ylim(-0.02, 1.05)

    axes[1].set_title("(b) Coverage Gap vs. NDCG Preference\n(threshold effect)",
                      fontsize=15)
    axes[1].set_xlabel(r"$|\Delta\mathrm{Coverage}|$ between models", fontsize=14)
    axes[1].set_ylabel("NDCG-winner preference (%)", fontsize=14)
    axes[1].axhline(50, color="gray", linestyle="--", alpha=0.5)
    axes[1].set_xlim(0.04, 0.90)
    axes[1].set_ylim(5, 104)

    for ax in axes:
        ax.tick_params(labelsize=14)
        ax.grid(False)

    # A single external legend guarantees that no point is hidden.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4,
               fontsize=14, frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    workspace = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=workspace / "revision_results" / "pair_level_local_global_metrics.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=workspace / "package" / "figures" / "scatter_margin_consistency.png",
    )
    args = parser.parse_args()
    build(args.input, args.output)


if __name__ == "__main__":
    main()
