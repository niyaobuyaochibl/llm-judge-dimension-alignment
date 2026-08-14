"""Parse manually returned ChatGPT validation verdicts.

Expected completed files live in <package-dir>/completed/*.jsonl and contain
one JSON object per line:
    {"case_id": "...", "verdict": "A"}
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

VALID = {"A", "B", "TIE"}


def read_metadata(path):
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {r["case_id"]: r for r in rows}


def iter_json_objects(text):
    text = text.strip()
    if not text:
        return
    if text.startswith("["):
        for obj in json.loads(text):
            yield obj
        return
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("```"):
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def read_completed(completed_dir):
    verdicts = {}
    files = sorted(completed_dir.glob("*.jsonl")) + sorted(completed_dir.glob("*.json"))
    for path in files:
        for obj in iter_json_objects(path.read_text(encoding="utf-8")):
            cid = str(obj.get("case_id", "")).strip()
            verdict = str(obj.get("verdict", "")).strip().upper()
            if verdict not in VALID:
                continue
            if cid in verdicts and verdicts[cid] != verdict:
                raise ValueError(f"Conflicting verdict for {cid}: {verdicts[cid]} vs {verdict}")
            verdicts[cid] = verdict
    return verdicts, files


def effective_model(meta, verdict):
    if verdict == "TIE":
        return "TIE"
    if verdict == "A":
        return meta["list_a_model"]
    if verdict == "B":
        return meta["list_b_model"]
    return "INVALID"


def winner_from_counts(row):
    if row["prefer_a"] > row["prefer_b"]:
        return row["model_a"]
    if row["prefer_b"] > row["prefer_a"]:
        return row["model_b"]
    return "TIE"


def pct(num, den):
    return 0.0 if den == 0 else 100.0 * num / den


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-dir", required=True)
    args = parser.parse_args()

    package_dir = Path(args.package_dir)
    metadata = read_metadata(package_dir / "metadata.csv")
    verdicts, files = read_completed(package_dir / "completed")
    if not verdicts:
        raise SystemExit(f"No completed verdicts found in {package_dir / 'completed'}")

    missing = sorted(set(metadata) - set(verdicts))
    extra = sorted(set(verdicts) - set(metadata))
    if extra:
        print(f"Warning: {len(extra)} verdicts have unknown case_id and will be ignored")
    print(f"Read {len(verdicts)} verdicts from {len(files)} files")
    print(f"Missing verdicts: {len(missing)} / {len(metadata)}")

    by_base = defaultdict(dict)
    for case_id, meta in metadata.items():
        if case_id not in verdicts:
            continue
        by_base[meta["base_id"]][meta["order"]] = (meta, verdicts[case_id])

    trials = []
    for base_id, orders in sorted(by_base.items()):
        if "fwd" not in orders or "rev" not in orders:
            continue
        fwd_meta, fwd_verdict = orders["fwd"]
        rev_meta, rev_verdict = orders["rev"]
        fwd_eff = effective_model(fwd_meta, fwd_verdict)
        rev_eff = effective_model(rev_meta, rev_verdict)
        consistent = fwd_eff == rev_eff
        effective = fwd_eff if consistent else "INCONSISTENT"
        trials.append({
            "base_id": base_id,
            "dataset": fwd_meta["dataset"],
            "model_a": fwd_meta["model_a"],
            "model_b": fwd_meta["model_b"],
            "dimension": fwd_meta["dimension"],
            "user_id": fwd_meta["user_id"],
            "fwd_verdict": fwd_verdict,
            "rev_verdict": rev_verdict,
            "consistent": consistent,
            "effective_preference": effective,
            "ndcg_winner": fwd_meta["ndcg_winner"],
            "cov_winner": fwd_meta["cov_winner"],
        })

    grouped = defaultdict(list)
    for t in trials:
        grouped[(t["dataset"], t["model_a"], t["model_b"], t["dimension"])].append(t)

    pair_rows = []
    for (dataset, model_a, model_b, dimension), rows in sorted(grouped.items()):
        prefer_a = sum(1 for r in rows if r["effective_preference"] == model_a)
        prefer_b = sum(1 for r in rows if r["effective_preference"] == model_b)
        tie = sum(1 for r in rows if r["effective_preference"] == "TIE")
        inconsistent = sum(1 for r in rows if r["effective_preference"] == "INCONSISTENT")
        consistent_n = len(rows) - inconsistent
        row = {
            "dataset": dataset,
            "model_a": model_a,
            "model_b": model_b,
            "dimension": dimension,
            "n_trials": len(rows),
            "prefer_a": prefer_a,
            "prefer_b": prefer_b,
            "tie": tie,
            "inconsistent": inconsistent,
            "consistency_rate": consistent_n / len(rows) if rows else 0,
            "ndcg_winner": rows[0]["ndcg_winner"],
            "cov_winner": rows[0]["cov_winner"],
        }
        row["llm_winner"] = winner_from_counts(row)
        row["aligns_ndcg"] = row["llm_winner"] == row["ndcg_winner"]
        row["aligns_cov"] = row["llm_winner"] == row["cov_winner"]
        pair_rows.append(row)

    out_trials = package_dir / "gpt_validation_per_trial.csv"
    if trials:
        with out_trials.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(trials[0].keys()))
            writer.writeheader()
            writer.writerows(trials)

    out_pairs = package_dir / "gpt_validation_pair_summary.csv"
    if pair_rows:
        with out_pairs.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(pair_rows[0].keys()))
            writer.writeheader()
            writer.writerows(pair_rows)

    summary_rows = []
    for dim in sorted({r["dimension"] for r in pair_rows}):
        rows = [r for r in pair_rows if r["dimension"] == dim]
        ndcg = sum(r["aligns_ndcg"] for r in rows)
        cov = sum(r["aligns_cov"] for r in rows)
        cons = sum(r["consistency_rate"] for r in rows) / len(rows) if rows else 0
        summary_rows.append({
            "dimension": dim,
            "pairs": len(rows),
            "ndcg_pairs": ndcg,
            "coverage_pairs": cov,
            "ndcg_pair_pct": pct(ndcg, len(rows)),
            "coverage_pair_pct": pct(cov, len(rows)),
            "mean_consistency": cons,
        })

    out_summary = package_dir / "gpt_validation_dimension_summary.csv"
    if summary_rows:
        with out_summary.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    print("\nDimension summary")
    for r in summary_rows:
        print(
            f"  {r['dimension']:9s} pairs={r['pairs']} "
            f"NDCG={r['ndcg_pair_pct']:.1f}% Coverage={r['coverage_pair_pct']:.1f}% "
            f"Cons={r['mean_consistency']:.3f}"
        )
    print(f"\nWrote {out_trials}")
    print(f"Wrote {out_pairs}")
    print(f"Wrote {out_summary}")


if __name__ == "__main__":
    main()
