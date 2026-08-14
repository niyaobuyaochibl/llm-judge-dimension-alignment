"""Create upload-ready ChatGPT validation batches for the TKDE revision.

This script does not call any external API.  It packages the same local-list
pairwise prompts used in the paper into small Markdown/JSONL batches that can
be uploaded to ChatGPT manually.  Returned A/B/TIE verdicts can be parsed with
parse_gpt_upload_results.py.
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path('/root/autodl-tmp/llm_judge_recsys/code')))

from experiment_config import PROJECT_ROOT, SEED
from dimension_prompts import get_dimension_prompt
from run_sensitivity_n100 import PAIRS, metric_winners
from data_loader import (
    load_topk_lists,
    load_item_texts,
    build_user_profile_summary,
    format_recommendation_list,
)

DEFAULT_DIMENSIONS = ["balanced", "diversity", "novelty"]


def make_case_id(dataset, model_a, model_b, dimension, user_id, order):
    return f"{dataset}__{model_a}_vs_{model_b}__{dimension}__u{user_id}__{order}"


def build_cases(n_users, dimensions):
    cases = []
    metadata = []
    for dataset, model_a, model_b in PAIRS:
        lists_a = load_topk_lists(dataset, model_a, SEED)
        lists_b = load_topk_lists(dataset, model_b, SEED)
        item_texts = load_item_texts(dataset)
        common_users = sorted(set(lists_a) & set(lists_b))
        rng = random.Random(SEED)
        rng.shuffle(common_users)
        sample_users = common_users[:n_users]
        winners = metric_winners(dataset, model_a, model_b)

        for dimension in dimensions:
            for user_id in sample_users:
                profile = build_user_profile_summary(user_id, dataset, item_texts)
                text_a = format_recommendation_list(lists_a[user_id], item_texts)
                text_b = format_recommendation_list(lists_b[user_id], item_texts)
                for order, list_a_model, list_b_model, list_a_text, list_b_text in [
                    ("fwd", model_a, model_b, text_a, text_b),
                    ("rev", model_b, model_a, text_b, text_a),
                ]:
                    system_prompt, user_prompt = get_dimension_prompt(
                        dimension, profile, list_a_text, list_b_text
                    )
                    case_id = make_case_id(dataset, model_a, model_b, dimension, user_id, order)
                    cases.append({
                        "case_id": case_id,
                        "system_prompt": system_prompt,
                        "user_prompt": user_prompt,
                    })
                    metadata.append({
                        "case_id": case_id,
                        "base_id": make_case_id(dataset, model_a, model_b, dimension, user_id, "pair"),
                        "dataset": dataset,
                        "model_a": model_a,
                        "model_b": model_b,
                        "dimension": dimension,
                        "user_id": user_id,
                        "order": order,
                        "list_a_model": list_a_model,
                        "list_b_model": list_b_model,
                        "ndcg_winner": winners["ndcg_winner"],
                        "cov_winner": winners["cov_winner"],
                        "gt_ndcg_a": winners["gt_ndcg_a"],
                        "gt_ndcg_b": winners["gt_ndcg_b"],
                        "gt_cov_a": winners["gt_cov_a"],
                        "gt_cov_b": winners["gt_cov_b"],
                    })
    return cases, metadata


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def render_batch_md(batch_id, rows):
    lines = [
        f"# GPT Validation Batch {batch_id:03d}",
        "",
        "You are simulating an LLM-as-a-judge evaluation for a recommender-system study.",
        "For each case below, apply the SYSTEM_PROMPT and USER_PROMPT exactly as written.",
        "Treat cases independently. Return only JSONL records with this schema:",
        "",
        '```json',
        '{"case_id":"...","verdict":"A|B|TIE"}',
        '```',
        "",
        "Do not include explanations, Markdown tables, or extra text. If the evidence is genuinely ambiguous, use TIE.",
        "",
    ]
    for i, row in enumerate(rows, start=1):
        lines.extend([
            f"## Case {i}: {row['case_id']}",
            "",
            "SYSTEM_PROMPT:",
            "```text",
            row["system_prompt"],
            "```",
            "",
            "USER_PROMPT:",
            "```text",
            row["user_prompt"],
            "```",
            "",
        ])
    return "\n".join(lines).rstrip() + "\n"


def write_readme(out_dir, n_users, batch_size, n_cases, n_batches):
    readme = f"""# ChatGPT Upload Package for GPT Judge Validation

This package contains local-list pairwise evaluation cases for a manual closed-source GPT validation. It does not contain ground-truth winners in the upload batches; ground-truth metadata is stored separately for aggregation.

## What This Validates

- This is a proprietary GPT judge robustness check, not a human baseline.
- Cases use the same six representative conflict pairs as the revision diagnostics and sensitivity checks.
- Prompts include balanced, diversity, and novelty conditions.
- Each user case is evaluated in both forward and reversed A/B order to measure positional consistency.

## Package Size

- Users per pair/prompt: {n_users}
- Total cases requiring GPT verdicts: {n_cases}
- Batch size: {batch_size}
- Number of batches: {n_batches}

## Recommended ChatGPT Procedure

1. Start a fresh ChatGPT conversation and choose the exact model you want to report, e.g., GPT-4o or GPT-5 if available.
2. Upload one file from `batches_md/`, starting with `batch_001.md`.
3. Paste this instruction:

```text
Please complete the uploaded validation batch. For every case, apply the SYSTEM_PROMPT and USER_PROMPT exactly as written. Return only JSONL lines with this schema: {{"case_id":"...","verdict":"A|B|TIE"}}. Do not add explanations or Markdown tables.
```

4. Save the returned JSONL as `completed/batch_001.jsonl`.
5. Repeat for all batch files.
6. Run `python code/parse_gpt_upload_results.py --package-dir {out_dir}` from the project root.

## Files

- `cases_all.jsonl`: all upload cases in machine-readable form.
- `metadata.csv`: mapping from case IDs to datasets, model pairs, order, and metric winners. Do not upload this if you want GPT to stay blind to ground truth.
- `batches_md/`: human-friendly upload batches.
- `batches_jsonl/`: machine-readable batches with the same cases.
- `completed/`: put ChatGPT-returned JSONL files here.
- `gpt_output_schema.json`: expected output schema.

## Reporting Note

When writing the paper, report the exact ChatGPT model name and date of execution. Because this is manual file-upload validation, describe it as a closed-source GPT judge check, not as an API-controlled deterministic experiment.
"""
    (out_dir / "README_upload_to_chatgpt.md").write_text(readme, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-users", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--dimensions", nargs="+", default=DEFAULT_DIMENSIONS)
    parser.add_argument(
        "--out-dir",
        default=str(PROJECT_ROOT / "gpt_upload_package"),
        help="Output directory for upload package.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    batches_md = out_dir / "batches_md"
    batches_jsonl = out_dir / "batches_jsonl"
    completed = out_dir / "completed"
    for path in [out_dir, batches_md, batches_jsonl, completed]:
        path.mkdir(parents=True, exist_ok=True)

    cases, metadata = build_cases(args.n_users, args.dimensions)
    write_jsonl(out_dir / "cases_all.jsonl", cases)
    write_csv(out_dir / "metadata.csv", metadata)
    (out_dir / "gpt_output_schema.json").write_text(
        json.dumps({"case_id": "string", "verdict": "A|B|TIE"}, indent=2),
        encoding="utf-8",
    )

    n_batches = 0
    for start in range(0, len(cases), args.batch_size):
        n_batches += 1
        batch = cases[start:start + args.batch_size]
        write_jsonl(batches_jsonl / f"batch_{n_batches:03d}.jsonl", batch)
        (batches_md / f"batch_{n_batches:03d}.md").write_text(
            render_batch_md(n_batches, batch), encoding="utf-8"
        )

    write_readme(out_dir, args.n_users, args.batch_size, len(cases), n_batches)
    print(f"Wrote {len(cases)} cases in {n_batches} batches to {out_dir}")


if __name__ == "__main__":
    main()
