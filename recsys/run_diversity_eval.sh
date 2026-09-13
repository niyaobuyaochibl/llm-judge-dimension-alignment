#!/bin/bash
# Batch diversity evaluation for Applied Intelligence paper.
# Evaluates core model variants across all datasets.
#
# Usage:
#   bash run_diversity_eval.sh              # Run all
#   bash run_diversity_eval.sh yelp         # Run one dataset
#   bash run_diversity_eval.sh yelp gbaf    # Run one dataset+variant

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_BASE="/root/autodl-tmp/extended_fusion/results"
OUTPUT_BASE="${SCRIPT_DIR}/diversity_results"
CONFIGS_DIR="${SCRIPT_DIR}/configs/extended"
SEEDS=(42 123 999 2024 2025)

FILTER_DATASET="${1:-all}"
FILTER_VARIANT="${2:-all}"

# ─── Dataset → (config_prefix, checkpoint_dir_pattern) mappings ───
# Each entry: "dataset_key|config_file|checkpoint_dir"
EVAL_JOBS=(
    # ML-1M
    "ml1m|lightgcn_only|movielens_lightgcn_only.yaml|movielens_lightgcn_only"
    "ml1m|fixed|movielens_fixed_baseline.yaml|ml1m/fixed"
    "ml1m|gbaf|movielens_gbaf.yaml|movielens_gbaf"
    "ml1m|concat_mlp|movielens_concat_mlp.yaml|concat_mlp/ml1m"
    # Amazon Fine Food
    "amazonff|lightgcn_only|amazon_lightgcn_only.yaml|amazon_lightgcn_only"
    "amazonff|fixed|amazon_fixed_baseline.yaml|amazonff/fixed"
    "amazonff|gbaf|amazon_gbaf.yaml|amazon_gbaf"
    "amazonff|gradnorm_only|amazon_gradnorm_only.yaml|amazon_gradnorm_only"
    "amazonff|concat_mlp|amazon_concat_mlp.yaml|concat_mlp/amazon_ff"
    "amazonff|pcgrad|amazon_pcgrad.yaml|amazon_pcgrad"
    # Yelp
    "yelp|lightgcn_only|yelp_lightgcn_only.yaml|yelp_lightgcn_only"
    "yelp|fixed|yelp_fixed_baseline.yaml|yelp/fixed"
    "yelp|gbaf|yelp_gbaf.yaml|yelp/gbaf"
    "yelp|gradnorm_only|yelp_gradnorm_only.yaml|yelp_gradnorm_only"
    "yelp|concat_mlp|yelp_concat_mlp.yaml|concat_mlp/yelp"
    "yelp|pcgrad|yelp_pcgrad.yaml|yelp_pcgrad"
    # MIND
    "mind|lightgcn_only|mind_lightgcn_only.yaml|mind_lightgcn_only"
    "mind|fixed|mind_fixed_baseline.yaml|mind/fixed"
    "mind|gbaf|mind_gbaf.yaml|mind/gbaf"
    "mind|gradnorm_only|mind_gradnorm_only.yaml|mind_gradnorm_only"
    "mind|concat_mlp|mind_concat_mlp.yaml|concat_mlp/mind"
    "mind|pcgrad|mind_pcgrad.yaml|mind_pcgrad"
    # Amazon Books
    "amazon_books|lightgcn_only|amazon_books_lightgcn_only.yaml|amazon_books_lightgcn_only"
    "amazon_books|fixed|amazon_books_fixed_baseline.yaml|amazon_books_fixed"
    "amazon_books|gbaf|amazon_books_gbaf.yaml|amazon_books_gbaf"
    "amazon_books|gradnorm_only|amazon_books_gradnorm_only.yaml|amazon_books_gradnorm_only"
    "amazon_books|concat_mlp|amazon_books_concat_mlp.yaml|concat_mlp/amazon_books"
    # Amazon CDs
    "amazon_cds|lightgcn_only|amazon_cds_lightgcn_only.yaml|amazon_cds_lightgcn_only"
    "amazon_cds|fixed|amazon_cds_fixed_baseline.yaml|amazon_cds_fixed"
    "amazon_cds|gbaf|amazon_cds_gbaf.yaml|amazon_cds_gbaf"
    "amazon_cds|gradnorm_only|amazon_cds_gradnorm_only.yaml|amazon_cds_gradnorm_only"
    "amazon_cds|concat_mlp|amazon_cds_concat_mlp.yaml|concat_mlp/amazon_cds"
)

total=0
success=0
skipped=0
failed=0

for job in "${EVAL_JOBS[@]}"; do
    IFS='|' read -r dataset variant config_name ckpt_dir <<< "$job"

    if [[ "$FILTER_DATASET" != "all" && "$dataset" != "$FILTER_DATASET" ]]; then
        continue
    fi
    if [[ "$FILTER_VARIANT" != "all" && "$variant" != "$FILTER_VARIANT" ]]; then
        continue
    fi

    config_path="${CONFIGS_DIR}/${config_name}"
    if [[ ! -f "$config_path" ]]; then
        echo "[WARN] Config not found: $config_path — skipping $dataset/$variant"
        ((skipped++))
        continue
    fi

    for seed in "${SEEDS[@]}"; do
        ckpt_path="${RESULTS_BASE}/${ckpt_dir}/best_model_seed${seed}.pt"
        out_dir="${OUTPUT_BASE}/${dataset}/${variant}"

        if [[ ! -f "$ckpt_path" ]]; then
            echo "[SKIP] Checkpoint missing: $ckpt_path"
            ((skipped++))
            continue
        fi

        out_file="${out_dir}/diversity_seed${seed}.json"
        if [[ -f "$out_file" ]]; then
            echo "[SKIP] Already done: $out_file"
            ((skipped++))
            continue
        fi

        ((total++))
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "[RUN] $dataset / $variant / seed=$seed"
        echo "  Config: $config_path"
        echo "  Ckpt:   $ckpt_path"
        echo "  Output: $out_dir"

        python "${SCRIPT_DIR}/evaluate_diversity.py" \
            --config "$config_path" \
            --checkpoint "$ckpt_path" \
            --seed "$seed" \
            --output_dir "$out_dir" \
            --save_lists \
            2>&1 | tail -10

        if [[ $? -eq 0 ]]; then
            ((success++))
            echo "[OK] $dataset / $variant / seed=$seed"
        else
            ((failed++))
            echo "[FAIL] $dataset / $variant / seed=$seed"
        fi
    done
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Diversity Evaluation Summary:"
echo "  Total runs: $total"
echo "  Success:    $success"
echo "  Failed:     $failed"
echo "  Skipped:    $skipped"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
