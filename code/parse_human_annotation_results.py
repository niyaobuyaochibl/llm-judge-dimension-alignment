"""Parse completed human annotation files for the TKDE revision."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


VALID = {"A", "B", "TIE"}


def read_metadata(path):
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {r["case_id"]: r for r in rows}


def iter_jsonl(path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def iter_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        yield from csv.DictReader(f)


def read_completed(completed_dir):
    rows = []
    for path in sorted(completed_dir.glob("*.jsonl")):
        rows.extend(iter_jsonl(path))
    for path in sorted(completed_dir.glob("*.csv")):
        rows.extend(iter_csv(path))

    by_annotator_case = {}
    for row in rows:
        annotator_id = str(row.get("annotator_id", "")).strip()
        case_id = str(row.get("case_id", "")).strip()
        verdict = str(row.get("verdict", "")).strip().upper()
        if not annotator_id or not case_id or verdict not in VALID:
            continue
        key = (annotator_id, case_id)
        clean = {
            "annotator_id": annotator_id,
            "case_id": case_id,
            "verdict": verdict,
            "confidence_1_to_5": str(row.get("confidence_1_to_5", "")).strip(),
            "notes_optional": str(row.get("notes_optional", "")).strip(),
        }
        if key in by_annotator_case and by_annotator_case[key]["verdict"] != verdict:
            raise ValueError(f"Conflicting verdict for {annotator_id} / {case_id}")
        by_annotator_case[key] = clean
    return list(by_annotator_case.values())


def effective_model(meta, verdict):
    if verdict == "TIE":
        return "TIE"
    if verdict == "A":
        return meta["list_a_model"]
    if verdict == "B":
        return meta["list_b_model"]
    return "INVALID"


def majority_from_counts(counts, model_a, model_b):
    options = [model_a, model_b, "TIE"]
    max_count = max(counts.get(o, 0) for o in options)
    winners = [o for o in options if counts.get(o, 0) == max_count]
    if len(winners) == 1:
        return winners[0]
    return "TIE"


def pairwise_agreement(labels):
    if len(labels) < 2:
        return ""
    total = 0
    agree = 0
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            total += 1
            agree += labels[i] == labels[j]
    return agree / total if total else ""


def fleiss_kappa(case_label_counts, categories):
    rows = []
    for counts in case_label_counts:
        n = sum(counts.get(c, 0) for c in categories)
        if n < 2:
            continue
        rows.append((counts, n))
    if not rows:
        return ""

    p_i = []
    total_labels = 0
    cat_totals = Counter()
    for counts, n in rows:
        total_labels += n
        cat_totals.update({c: counts.get(c, 0) for c in categories})
        p_i.append((sum(counts.get(c, 0) ** 2 for c in categories) - n) / (n * (n - 1)))
    p_bar = sum(p_i) / len(p_i)
    p_e = sum((cat_totals[c] / total_labels) ** 2 for c in categories)
    if p_e == 1:
        return 1.0
    return (p_bar - p_e) / (1 - p_e)


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def pct(num, den):
    return 0.0 if den == 0 else 100.0 * num / den


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-dir", required=True)
    args = parser.parse_args()

    package_dir = Path(args.package_dir)
    metadata = read_metadata(package_dir / "metadata.csv")
    completed = read_completed(package_dir / "completed")
    if not completed:
        raise SystemExit(f"No completed human annotation files found in {package_dir / 'completed'}")

    annotation_rows = []
    unknown = []
    for row in completed:
        case_id = row["case_id"]
        if case_id not in metadata:
            unknown.append(case_id)
            continue
        meta = metadata[case_id]
        eff = effective_model(meta, row["verdict"])
        annotation_rows.append({
            "annotator_id": row["annotator_id"],
            "case_id": case_id,
            "base_id": meta["base_id"],
            "condition": meta["condition"],
            "dataset": meta["dataset"],
            "model_a": meta["model_a"],
            "model_b": meta["model_b"],
            "dimension": meta["dimension"],
            "user_id": meta["user_id"],
            "order": meta["order"],
            "verdict": row["verdict"],
            "effective_preference": eff,
            "ndcg_winner": meta["ndcg_winner"],
            "cov_winner": meta["cov_winner"],
            "aligns_ndcg": eff == meta["ndcg_winner"],
            "aligns_cov": eff == meta["cov_winner"],
            "confidence_1_to_5": row["confidence_1_to_5"],
            "notes_optional": row["notes_optional"],
        })

    if unknown:
        print(f"Warning: ignored {len(set(unknown))} unknown case IDs")

    by_base = defaultdict(list)
    for row in annotation_rows:
        by_base[row["base_id"]].append(row)

    case_rows = []
    for base_id, rows in sorted(by_base.items()):
        first = rows[0]
        counts = Counter(r["effective_preference"] for r in rows)
        majority = majority_from_counts(counts, first["model_a"], first["model_b"])
        labels = [r["effective_preference"] for r in rows]
        case_rows.append({
            "base_id": base_id,
            "condition": first["condition"],
            "dataset": first["dataset"],
            "model_a": first["model_a"],
            "model_b": first["model_b"],
            "dimension": first["dimension"],
            "user_id": first["user_id"],
            "n_annotations": len(rows),
            "prefer_model_a": counts.get(first["model_a"], 0),
            "prefer_model_b": counts.get(first["model_b"], 0),
            "tie": counts.get("TIE", 0),
            "majority_preference": majority,
            "ndcg_winner": first["ndcg_winner"],
            "cov_winner": first["cov_winner"],
            "aligns_ndcg": majority == first["ndcg_winner"],
            "aligns_cov": majority == first["cov_winner"],
            "pairwise_agreement": pairwise_agreement(labels),
        })

    by_pair = defaultdict(list)
    for row in annotation_rows:
        key = (
            row["condition"],
            row["dataset"],
            row["model_a"],
            row["model_b"],
            row["dimension"],
        )
        by_pair[key].append(row)

    pair_rows = []
    for key, rows in sorted(by_pair.items()):
        condition, dataset, model_a, model_b, dimension = key
        counts = Counter(r["effective_preference"] for r in rows)
        majority = majority_from_counts(counts, model_a, model_b)
        base_counts = []
        for case in [c for c in case_rows if c["condition"] == condition and c["dataset"] == dataset and c["model_a"] == model_a and c["model_b"] == model_b and c["dimension"] == dimension]:
            base_counts.append({
                model_a: int(case["prefer_model_a"]),
                model_b: int(case["prefer_model_b"]),
                "TIE": int(case["tie"]),
            })
        pair_rows.append({
            "condition": condition,
            "dataset": dataset,
            "model_a": model_a,
            "model_b": model_b,
            "dimension": dimension,
            "n_annotations": len(rows),
            "prefer_model_a": counts.get(model_a, 0),
            "prefer_model_b": counts.get(model_b, 0),
            "tie": counts.get("TIE", 0),
            "human_winner": majority,
            "ndcg_winner": rows[0]["ndcg_winner"],
            "cov_winner": rows[0]["cov_winner"],
            "aligns_ndcg": majority == rows[0]["ndcg_winner"],
            "aligns_cov": majority == rows[0]["cov_winner"],
            "mean_case_pairwise_agreement": mean_numeric(
                c["pairwise_agreement"] for c in case_rows
                if c["condition"] == condition
                and c["dataset"] == dataset
                and c["model_a"] == model_a
                and c["model_b"] == model_b
                and c["dimension"] == dimension
            ),
            "fleiss_kappa": fleiss_kappa(base_counts, [model_a, model_b, "TIE"]),
        })

    summary_rows = []
    for (condition, dimension), rows in sorted(group_by(pair_rows, ["condition", "dimension"]).items()):
        ndcg = sum(r["aligns_ndcg"] for r in rows)
        cov = sum(r["aligns_cov"] for r in rows)
        ties = sum(r["human_winner"] == "TIE" for r in rows)
        summary_rows.append({
            "condition": condition,
            "dimension": dimension,
            "pairs": len(rows),
            "ndcg_pairs": ndcg,
            "coverage_pairs": cov,
            "tie_pairs": ties,
            "ndcg_pair_pct": pct(ndcg, len(rows)),
            "coverage_pair_pct": pct(cov, len(rows)),
            "mean_pairwise_agreement": mean_numeric(r["mean_case_pairwise_agreement"] for r in rows),
            "mean_fleiss_kappa": mean_numeric(r["fleiss_kappa"] for r in rows),
        })

    annotator_rows = []
    for annotator_id, rows in sorted(group_by(annotation_rows, ["annotator_id"]).items()):
        annotator_rows.append({
            "annotator_id": annotator_id,
            "n_annotations": len(rows),
            "local_only": sum(r["condition"] == "local_only" for r in rows),
            "global_stats": sum(r["condition"] == "global_stats" for r in rows),
            "tie_rate": pct(sum(r["effective_preference"] == "TIE" for r in rows), len(rows)),
            "coverage_alignment_rate": pct(sum(r["aligns_cov"] for r in rows), len(rows)),
            "ndcg_alignment_rate": pct(sum(r["aligns_ndcg"] for r in rows), len(rows)),
        })

    write_csv(package_dir / "human_annotation_per_annotation.csv", annotation_rows)
    write_csv(package_dir / "human_annotation_per_case.csv", case_rows)
    write_csv(package_dir / "human_annotation_pair_summary.csv", pair_rows)
    write_csv(package_dir / "human_annotation_dimension_summary.csv", summary_rows)
    write_csv(package_dir / "human_annotation_annotator_summary.csv", annotator_rows)

    print(f"Read {len(annotation_rows)} valid annotations")
    print("\nDimension summary")
    for row in summary_rows:
        print(
            f"  {row['condition']:12s} {row['dimension']:16s} pairs={row['pairs']} "
            f"NDCG={row['ndcg_pairs']}/{row['pairs']} "
            f"Coverage={row['coverage_pairs']}/{row['pairs']} "
            f"Tie={row['tie_pairs']}/{row['pairs']} "
            f"Agree={format_float(row['mean_pairwise_agreement'])} "
            f"Kappa={format_float(row['mean_fleiss_kappa'])}"
        )


def group_by(rows, fields):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[f] for f in fields)].append(row)
    if len(fields) == 1:
        return {k[0]: v for k, v in grouped.items()}
    return dict(grouped)


def mean_numeric(values):
    vals = [float(v) for v in values if v != "" and v is not None]
    if not vals:
        return ""
    return sum(vals) / len(vals)


def format_float(value):
    if value == "" or value is None:
        return "NA"
    return f"{float(value):.3f}"


if __name__ == "__main__":
    main()
