"""Aggregate OpenRouter validation verdicts using the GPT upload metadata."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

VALID = {'A', 'B', 'TIE'}


def read_metadata(path):
    with path.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    return {r['case_id']: r for r in rows}


def read_verdicts(path):
    verdicts = {}
    with path.open(encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            cid = obj.get('case_id')
            verdict = str(obj.get('verdict', '')).upper()
            if cid and verdict in VALID:
                verdicts[cid] = verdict
    return verdicts


def effective_model(meta, verdict):
    if verdict == 'TIE':
        return 'TIE'
    return meta['list_a_model'] if verdict == 'A' else meta['list_b_model']


def winner_from_counts(row):
    if row['prefer_a'] > row['prefer_b']:
        return row['model_a']
    if row['prefer_b'] > row['prefer_a']:
        return row['model_b']
    return 'TIE'


def pct(num, den):
    return 0.0 if den == 0 else 100.0 * num / den


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--package-dir', default='/root/autodl-tmp/llm_diversity_eval/gpt_upload_package')
    parser.add_argument('--verdicts', required=True, help='completed_<model>.jsonl from run_openrouter_validation.py')
    parser.add_argument('--out-prefix', default=None)
    args = parser.parse_args()

    package_dir = Path(args.package_dir)
    verdict_path = Path(args.verdicts)
    metadata = read_metadata(package_dir / 'metadata.csv')
    verdicts = read_verdicts(verdict_path)
    missing = sorted(set(metadata) - set(verdicts))
    print(f'Read {len(verdicts)} verdicts from {verdict_path}')
    print(f'Missing verdicts: {len(missing)} / {len(metadata)}')

    by_base = defaultdict(dict)
    for case_id, meta in metadata.items():
        if case_id not in verdicts:
            continue
        by_base[meta['base_id']][meta['order']] = (meta, verdicts[case_id])

    trials = []
    for base_id, orders in sorted(by_base.items()):
        if 'fwd' not in orders or 'rev' not in orders:
            continue
        fwd_meta, fwd_verdict = orders['fwd']
        rev_meta, rev_verdict = orders['rev']
        fwd_eff = effective_model(fwd_meta, fwd_verdict)
        rev_eff = effective_model(rev_meta, rev_verdict)
        consistent = fwd_eff == rev_eff
        effective = fwd_eff if consistent else 'INCONSISTENT'
        trials.append({
            'base_id': base_id,
            'dataset': fwd_meta['dataset'],
            'model_a': fwd_meta['model_a'],
            'model_b': fwd_meta['model_b'],
            'dimension': fwd_meta['dimension'],
            'user_id': fwd_meta['user_id'],
            'fwd_verdict': fwd_verdict,
            'rev_verdict': rev_verdict,
            'consistent': consistent,
            'effective_preference': effective,
            'ndcg_winner': fwd_meta['ndcg_winner'],
            'cov_winner': fwd_meta['cov_winner'],
        })

    grouped = defaultdict(list)
    for t in trials:
        grouped[(t['dataset'], t['model_a'], t['model_b'], t['dimension'])].append(t)

    pair_rows = []
    for (dataset, model_a, model_b, dimension), rows in sorted(grouped.items()):
        prefer_a = sum(1 for r in rows if r['effective_preference'] == model_a)
        prefer_b = sum(1 for r in rows if r['effective_preference'] == model_b)
        tie = sum(1 for r in rows if r['effective_preference'] == 'TIE')
        inconsistent = sum(1 for r in rows if r['effective_preference'] == 'INCONSISTENT')
        row = {
            'dataset': dataset,
            'model_a': model_a,
            'model_b': model_b,
            'dimension': dimension,
            'n_trials': len(rows),
            'prefer_a': prefer_a,
            'prefer_b': prefer_b,
            'tie': tie,
            'inconsistent': inconsistent,
            'consistency_rate': (len(rows) - inconsistent) / len(rows) if rows else 0,
            'ndcg_winner': rows[0]['ndcg_winner'],
            'cov_winner': rows[0]['cov_winner'],
        }
        row['llm_winner'] = winner_from_counts(row)
        row['aligns_ndcg'] = row['llm_winner'] == row['ndcg_winner']
        row['aligns_cov'] = row['llm_winner'] == row['cov_winner']
        pair_rows.append(row)

    summary_rows = []
    for dim in sorted({r['dimension'] for r in pair_rows}):
        rows = [r for r in pair_rows if r['dimension'] == dim]
        nd = sum(r['aligns_ndcg'] for r in rows)
        cov = sum(r['aligns_cov'] for r in rows)
        tie_pairs = sum(r['llm_winner'] == 'TIE' for r in rows)
        cons = sum(float(r['consistency_rate']) for r in rows) / len(rows) if rows else 0
        votes = sum(int(r['prefer_a']) + int(r['prefer_b']) for r in rows)
        nd_votes = cov_votes = 0
        for r in rows:
            if r['model_a'] == r['ndcg_winner']:
                nd_votes += int(r['prefer_a'])
            elif r['model_b'] == r['ndcg_winner']:
                nd_votes += int(r['prefer_b'])
            if r['model_a'] == r['cov_winner']:
                cov_votes += int(r['prefer_a'])
            elif r['model_b'] == r['cov_winner']:
                cov_votes += int(r['prefer_b'])
        summary_rows.append({
            'dimension': dim,
            'pairs': len(rows),
            'ndcg_pairs': nd,
            'coverage_pairs': cov,
            'tie_pairs': tie_pairs,
            'ndcg_pair_pct': pct(nd, len(rows)),
            'coverage_pair_pct': pct(cov, len(rows)),
            'mean_consistency': cons,
            'non_tie_votes': votes,
            'ndcg_votes': nd_votes,
            'coverage_votes': cov_votes,
            'ndcg_vote_pct': pct(nd_votes, votes),
            'coverage_vote_pct': pct(cov_votes, votes),
        })

    prefix = args.out_prefix
    if prefix is None:
        prefix = verdict_path.with_suffix('').name
    out_dir = verdict_path.parent
    write_csv(out_dir / f'{prefix}_per_trial.csv', trials)
    write_csv(out_dir / f'{prefix}_pair_summary.csv', pair_rows)
    write_csv(out_dir / f'{prefix}_dimension_summary.csv', summary_rows)

    print('\nDimension summary')
    for r in summary_rows:
        print(
            f"  {r['dimension']:9s} pairs={r['pairs']} "
            f"NDCG={r['ndcg_pair_pct']:.1f}% Coverage={r['coverage_pair_pct']:.1f}% "
            f"TiePairs={r['tie_pairs']} Cons={r['mean_consistency']:.3f} "
            f"NDCGVotes={r['ndcg_votes']}/{r['non_tie_votes']} ({r['ndcg_vote_pct']:.1f}%)"
        )


if __name__ == '__main__':
    main()
