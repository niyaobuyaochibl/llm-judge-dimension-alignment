"""Create a small human-annotation package for the TKDE revision.

The package reuses the same six representative conflict pairs used for the
revision diagnostics.  By default it creates a lightweight diversity-focused
human baseline:

* local-only cases: one randomized A/B order per user-level diversity case;
* global-stat cases: one randomized A/B order per system pair, with Coverage@10
  shown as the primary diversity criterion.

No ground-truth winners are written into annotator-facing files.  Metadata for
aggregation is stored separately in metadata.csv.
"""

import argparse
import csv
import html
import json
import random
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_PACKAGE = PROJECT_ROOT / "gpt_upload_package"
DEFAULT_OUT_DIR = PROJECT_ROOT / "human_annotation_package"
VALID_VERDICTS = ["A", "B", "TIE"]


def read_jsonl(path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames=None):
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def local_title(meta):
    dim = meta["dimension"].replace("_", " ").title()
    dataset = meta["dataset"].replace("_", " ").title()
    return f"{dim} judgment | {dataset} | user {meta['user_id']}"


def local_instruction(dimension):
    if dimension == "diversity":
        return (
            "Judge visible local list-level diversity only. Choose the list that "
            "covers a broader range of categories, genres, topics, item types, "
            "or styles among the 10 displayed items. Ignore relevance, accuracy, "
            "and system-level catalog coverage."
        )
    if dimension == "novelty":
        return (
            "Judge visible novelty only. Choose the list that appears to contain "
            "less familiar, less mainstream, or more exploratory recommendations "
            "for this user. Ignore relevance unless novelty is tied."
        )
    if dimension == "balanced":
        return (
            "Judge overall recommendation quality for this user, considering "
            "relevance, visible diversity, and overall usefulness."
        )
    return "Choose the better list under the task instruction."


def global_prompt(meta, list_a_model, list_b_model):
    values = {
        meta["model_a"]: {
            "ndcg": float(meta["gt_ndcg_a"]),
            "coverage": float(meta["gt_cov_a"]),
        },
        meta["model_b"]: {
            "ndcg": float(meta["gt_ndcg_b"]),
            "coverage": float(meta["gt_cov_b"]),
        },
    }
    a = values[list_a_model]
    b = values[list_b_model]
    return f"""You are judging SYSTEM-LEVEL DISTRIBUTIONAL DIVERSITY for two recommender systems.

Coverage@10 is the primary criterion. Higher Coverage@10 means the system exposes users to a broader portion of the item catalog. NDCG@10 is shown only for context and should not override Coverage@10 for this task.

## Aggregate System Statistics

System that produced List A:
- Coverage@10: {a['coverage']:.4f}
- NDCG@10: {a['ndcg']:.4f}

System that produced List B:
- Coverage@10: {b['coverage']:.4f}
- NDCG@10: {b['ndcg']:.4f}

## Instructions
Choose the system with better system-level diversity using Coverage@10 as the primary criterion.

Respond with A, B, or TIE."""


def choose_one_order(base_rows, rng):
    rows = sorted(base_rows, key=lambda r: r["case_id"])
    if len(rows) == 1:
        return rows[0]
    return rng.choice(rows)


def build_package(source_package, out_dir, annotators, dimensions, seed):
    cases = {r["case_id"]: r for r in read_jsonl(source_package / "cases_all.jsonl")}
    metadata_rows = read_csv(source_package / "metadata.csv")
    metadata = {r["case_id"]: r for r in metadata_rows}

    by_base = defaultdict(list)
    for row in metadata_rows:
        if row["dimension"] in dimensions:
            by_base[row["base_id"]].append(row)

    pair_representatives = {}
    for row in metadata_rows:
        if row["dimension"] != "diversity" or row["order"] != "fwd":
            continue
        key = (row["dataset"], row["model_a"], row["model_b"])
        pair_representatives.setdefault(key, row)

    out_dir.mkdir(parents=True, exist_ok=True)
    for sub in ["forms_html", "forms_md", "answer_sheets", "completed", "cases_jsonl"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    package_metadata = []
    package_cases = []

    for annotator_idx in range(1, annotators + 1):
        annotator_id = f"annotator_{annotator_idx:02d}"
        rng = random.Random(seed + annotator_idx * 997)

        local_cases = []
        for base_id in sorted(by_base):
            chosen_meta = choose_one_order(by_base[base_id], rng)
            source_case = cases[chosen_meta["case_id"]]
            case = {
                "annotator_id": annotator_id,
                "case_id": chosen_meta["case_id"],
                "base_id": chosen_meta["base_id"],
                "condition": "local_only",
                "dimension": chosen_meta["dimension"],
                "title": local_title(chosen_meta),
                "criterion": local_instruction(chosen_meta["dimension"]),
                "prompt": source_case["user_prompt"],
            }
            local_cases.append(case)

        rng.shuffle(local_cases)

        global_cases = []
        for (dataset, model_a, model_b), meta in sorted(pair_representatives.items()):
            order = rng.choice(["fwd", "rev"])
            if order == "fwd":
                list_a_model, list_b_model = model_a, model_b
            else:
                list_a_model, list_b_model = model_b, model_a
            case_id = f"{dataset}__{model_a}_vs_{model_b}__global_diversity__{annotator_id}__{order}"
            base_id = f"{dataset}__{model_a}_vs_{model_b}__global_diversity__pair"
            case = {
                "annotator_id": annotator_id,
                "case_id": case_id,
                "base_id": base_id,
                "condition": "global_stats",
                "dimension": "global_diversity",
                "title": f"Global diversity judgment | {dataset.replace('_', ' ').title()}",
                "criterion": (
                    "Use the aggregate system statistics. Coverage@10 is the "
                    "primary system-level diversity criterion; higher is better."
                ),
                "prompt": global_prompt(meta, list_a_model, list_b_model),
            }
            global_cases.append(case)

            package_metadata.append({
                "case_id": case_id,
                "base_id": base_id,
                "condition": "global_stats",
                "dataset": dataset,
                "model_a": model_a,
                "model_b": model_b,
                "dimension": "global_diversity",
                "user_id": "",
                "order": order,
                "list_a_model": list_a_model,
                "list_b_model": list_b_model,
                "ndcg_winner": meta["ndcg_winner"],
                "cov_winner": meta["cov_winner"],
                "gt_ndcg_a": meta["gt_ndcg_a"],
                "gt_ndcg_b": meta["gt_ndcg_b"],
                "gt_cov_a": meta["gt_cov_a"],
                "gt_cov_b": meta["gt_cov_b"],
            })

        annotator_cases = local_cases + global_cases
        package_cases.extend(annotator_cases)
        write_jsonl(out_dir / "cases_jsonl" / f"{annotator_id}.jsonl", annotator_cases)
        write_csv(
            out_dir / "answer_sheets" / f"{annotator_id}.csv",
            [
                {
                    "annotator_id": annotator_id,
                    "case_id": c["case_id"],
                    "verdict": "",
                    "confidence_1_to_5": "",
                    "notes_optional": "",
                }
                for c in annotator_cases
            ],
            fieldnames=[
                "annotator_id",
                "case_id",
                "verdict",
                "confidence_1_to_5",
                "notes_optional",
            ],
        )
        (out_dir / "forms_md" / f"{annotator_id}.md").write_text(
            render_markdown_form(annotator_id, annotator_cases), encoding="utf-8"
        )
        (out_dir / "forms_html" / f"{annotator_id}.html").write_text(
            render_html_form(annotator_id, annotator_cases), encoding="utf-8"
        )

        for case in local_cases:
            meta = metadata[case["case_id"]]
            package_metadata.append({
                "case_id": case["case_id"],
                "base_id": case["base_id"],
                "condition": "local_only",
                "dataset": meta["dataset"],
                "model_a": meta["model_a"],
                "model_b": meta["model_b"],
                "dimension": meta["dimension"],
                "user_id": meta["user_id"],
                "order": meta["order"],
                "list_a_model": meta["list_a_model"],
                "list_b_model": meta["list_b_model"],
                "ndcg_winner": meta["ndcg_winner"],
                "cov_winner": meta["cov_winner"],
                "gt_ndcg_a": meta["gt_ndcg_a"],
                "gt_ndcg_b": meta["gt_ndcg_b"],
                "gt_cov_a": meta["gt_cov_a"],
                "gt_cov_b": meta["gt_cov_b"],
            })

    write_jsonl(out_dir / "cases_all.jsonl", package_cases)
    write_csv(out_dir / "metadata.csv", package_metadata)
    (out_dir / "README_human_annotation.md").write_text(
        render_readme(out_dir, annotators, dimensions, len(package_cases)),
        encoding="utf-8",
    )

    return len(package_cases), len(package_metadata)


def render_markdown_form(annotator_id, cases):
    lines = [
        f"# Human Annotation Form: {annotator_id}",
        "",
        "Please complete each case independently. Do not look at metadata.csv.",
        "For every case, choose exactly one verdict: A, B, or TIE.",
        "",
        "Use TIE only when the two lists/systems are genuinely indistinguishable under the stated criterion.",
        "",
    ]
    for idx, case in enumerate(cases, start=1):
        lines.extend([
            f"## Case {idx:03d}: {case['case_id']}",
            "",
            f"Condition: {case['condition']}",
            f"Criterion: {case['criterion']}",
            "",
            "```text",
            case["prompt"],
            "```",
            "",
            "Answer:",
            "- Verdict: ",
            "- Confidence 1-5: ",
            "- Notes optional: ",
            "",
        ])
    return "\n".join(lines).rstrip() + "\n"


def render_html_form(annotator_id, cases):
    data = json.dumps(cases, ensure_ascii=False).replace("</", "<\\/")
    case_cards = []
    for idx, case in enumerate(cases, start=1):
        cid = html.escape(case["case_id"])
        title = html.escape(case["title"])
        condition = html.escape(case["condition"])
        criterion = html.escape(case["criterion"])
        prompt = html.escape(case["prompt"])
        radios = "\n".join(
            f'<label><input type="radio" name="{cid}" value="{v}"> {v}</label>'
            for v in VALID_VERDICTS
        )
        case_cards.append(f"""
<section class="case-card" id="case-{idx}">
  <div class="case-head">
    <div>
      <div class="case-number">Case {idx:03d}</div>
      <h2>{title}</h2>
    </div>
    <code>{cid}</code>
  </div>
  <p><strong>Condition:</strong> {condition}</p>
  <p><strong>Criterion:</strong> {criterion}</p>
  <pre>{prompt}</pre>
  <div class="answer-row" data-case-id="{cid}">
    <span>Verdict</span>
    {radios}
    <label>Confidence
      <select class="confidence">
        <option value=""></option>
        <option value="1">1</option>
        <option value="2">2</option>
        <option value="3">3</option>
        <option value="4">4</option>
        <option value="5">5</option>
      </select>
    </label>
    <input class="notes" type="text" placeholder="optional note">
  </div>
</section>""")

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Human Annotation Form {html.escape(annotator_id)}</title>
<style>
body {{
  margin: 0;
  font-family: Arial, Helvetica, sans-serif;
  background: #f6f7f9;
  color: #20242a;
}}
header {{
  position: sticky;
  top: 0;
  z-index: 10;
  background: #ffffff;
  border-bottom: 1px solid #d9dde3;
  padding: 14px 22px;
}}
main {{
  max-width: 980px;
  margin: 0 auto;
  padding: 20px;
}}
h1 {{
  font-size: 20px;
  margin: 0 0 8px;
}}
h2 {{
  font-size: 17px;
  margin: 2px 0 0;
}}
.toolbar {{
  display: flex;
  gap: 12px;
  align-items: center;
  flex-wrap: wrap;
}}
button {{
  border: 1px solid #1d4ed8;
  background: #1d4ed8;
  color: white;
  border-radius: 6px;
  padding: 8px 12px;
  cursor: pointer;
}}
.case-card {{
  background: #ffffff;
  border: 1px solid #d9dde3;
  border-radius: 8px;
  margin: 16px 0;
  padding: 16px;
}}
.case-head {{
  display: flex;
  justify-content: space-between;
  gap: 16px;
  align-items: start;
}}
.case-number {{
  font-size: 12px;
  color: #667085;
  text-transform: uppercase;
}}
code {{
  font-size: 11px;
  color: #475467;
  word-break: break-all;
}}
pre {{
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 6px;
  padding: 12px;
  max-height: 620px;
  overflow: auto;
  line-height: 1.45;
}}
.answer-row {{
  display: flex;
  align-items: center;
  gap: 14px;
  flex-wrap: wrap;
  border-top: 1px solid #e5e7eb;
  padding-top: 12px;
}}
.notes {{
  min-width: 260px;
  flex: 1;
  padding: 6px 8px;
}}
select {{
  margin-left: 6px;
}}
.status {{
  color: #475467;
  font-size: 14px;
}}
</style>
</head>
<body>
<header>
  <h1>Human Annotation Form: {html.escape(annotator_id)}</h1>
  <div class="toolbar">
    <button type="button" onclick="downloadJsonl()">Download JSONL</button>
    <button type="button" onclick="downloadCsv()">Download CSV</button>
    <span class="status" id="progress"></span>
  </div>
</header>
<main>
  <p>Complete each case independently. Do not open metadata.csv while annotating. Choose A, B, or TIE under the stated criterion.</p>
  {''.join(case_cards)}
</main>
<script id="cases-data" type="application/json">{data}</script>
<script>
const annotatorId = {json.dumps(annotator_id)};
const cases = JSON.parse(document.getElementById('cases-data').textContent);

function collectRows() {{
  return cases.map(c => {{
    const checked = document.querySelector(`input[name="${{CSS.escape(c.case_id)}}"]:checked`);
    const row = document.querySelector(`[data-case-id="${{CSS.escape(c.case_id)}}"]`);
    return {{
      annotator_id: annotatorId,
      case_id: c.case_id,
      verdict: checked ? checked.value : "",
      confidence_1_to_5: row.querySelector('.confidence').value,
      notes_optional: row.querySelector('.notes').value
    }};
  }});
}}

function updateProgress() {{
  const rows = collectRows();
  const done = rows.filter(r => r.verdict).length;
  document.getElementById('progress').textContent = `${{done}} / ${{rows.length}} completed`;
}}

function download(filename, text, mime) {{
  const blob = new Blob([text], {{type: mime}});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}}

function downloadJsonl() {{
  const rows = collectRows().filter(r => r.verdict);
  const text = rows.map(r => JSON.stringify(r)).join("\\n") + "\\n";
  download(`${{annotatorId}}.jsonl`, text, 'application/jsonl');
}}

function csvEscape(v) {{
  const s = String(v ?? "");
  return /[",\\n]/.test(s) ? `"${{s.replaceAll('"', '""')}}"` : s;
}}

function downloadCsv() {{
  const rows = collectRows();
  const header = ['annotator_id', 'case_id', 'verdict', 'confidence_1_to_5', 'notes_optional'];
  const text = [header.join(',')].concat(rows.map(r => header.map(h => csvEscape(r[h])).join(','))).join("\\n") + "\\n";
  download(`${{annotatorId}}.csv`, text, 'text/csv');
}}

document.addEventListener('change', updateProgress);
updateProgress();
</script>
</body>
</html>
"""


def render_readme(out_dir, annotators, dimensions, n_cases):
    dim_text = ", ".join(dimensions)
    return f"""# Human Annotation Package

This package supports a small human sanity check for the TKDE revision.

## Design

- Annotators: {annotators}
- Local-only dimensions: {dim_text}
- Local-only workload per annotator: 60 cases per dimension
- Global-stat workload per annotator: 6 cases
- Total annotator-facing cases: {n_cases}

The recommended revision use is diversity-focused: local-only annotators judge visible list-level diversity from the displayed top-10 lists, while the global-stat condition checks whether humans can follow system-level Coverage@10 when it is explicitly supplied.

## Files

- `forms_html/annotator_XX.html`: browser-based form with JSONL/CSV download buttons.
- `forms_md/annotator_XX.md`: Markdown backup form.
- `answer_sheets/annotator_XX.csv`: empty answer sheet if annotators prefer editing CSV.
- `completed/`: put completed `.jsonl` or `.csv` files here.
- `metadata.csv`: hidden aggregation metadata. Do not show this to annotators.
- `cases_all.jsonl`: annotator-facing cases in machine-readable form.

## Procedure

1. Give each annotator exactly one HTML file from `forms_html/`.
2. Ask them not to open `metadata.csv`.
3. They should choose A, B, or TIE for every case and optionally provide confidence.
4. Save their downloaded `.jsonl` or `.csv` files into `completed/`.
5. Run:

```bash
python code/parse_human_annotation_results.py --package-dir {out_dir}
```

## Ethics Note

This package collects only recommendation-list preferences and optional notes. It does not need annotator names. Check local institutional requirements before recruiting external participants or collecting any personal information.
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-package", default=str(DEFAULT_SOURCE_PACKAGE))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--annotators", type=int, default=3)
    parser.add_argument("--dimensions", nargs="+", default=["diversity"])
    parser.add_argument("--seed", type=int, default=20260627)
    return parser.parse_args()


def main():
    args = parse_args()
    n_cases, n_meta = build_package(
        Path(args.source_package),
        Path(args.out_dir),
        annotators=args.annotators,
        dimensions=args.dimensions,
        seed=args.seed,
    )
    print(f"Wrote {n_cases} annotator-facing cases to {args.out_dir}")
    print(f"Wrote {n_meta} metadata rows")


if __name__ == "__main__":
    main()
