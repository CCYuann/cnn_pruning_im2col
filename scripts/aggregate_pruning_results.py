#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def read_rows(root: Path):
    rows = []
    for path in sorted(root.rglob('*results.csv')):
        if path.name.startswith('all_') or path.name.startswith('summary_'):
            continue
        with path.open(newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                row = dict(row)
                row['source_file'] = str(path)
                rows.append(row)
    return rows


def f(row, key):
    value = row.get(key, '')
    return None if value == '' else float(value)


def main():
    parser = argparse.ArgumentParser(description='Aggregate pruning experiment CSV files.')
    parser.add_argument('root', type=Path, help='Run folder to scan recursively.')
    parser.add_argument('--out', type=Path, default=None, help='Combined CSV path.')
    args = parser.parse_args()

    rows = read_rows(args.root)
    if not rows:
        raise SystemExit(f'No result rows found under {args.root}')

    out = args.out or args.root / 'all_results_combined.csv'
    fieldnames = list(rows[0].keys())
    if 'source_file' not in fieldnames:
        fieldnames.append('source_file')
    with out.open('w', newline='', encoding='utf-8') as fobj:
        writer = csv.DictWriter(fobj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    by_key = {}
    for row in rows:
        method = row.get('method', '')
        if method not in {'Cross-filter l1_sum (global_cluster)', 'Cross-filter max_abs (global_cluster)'}:
            continue
        key = (row.get('dataset', ''), row.get('model', ''), row.get('source_file', ''))
        by_key.setdefault(key, {})[method] = row

    diffs = []
    for key, pair in by_key.items():
        if len(pair) != 2:
            continue
        l1 = f(pair['Cross-filter l1_sum (global_cluster)'], 'after_extra_or_ft_acc')
        mx = f(pair['Cross-filter max_abs (global_cluster)'], 'after_extra_or_ft_acc')
        speed = f(pair['Cross-filter max_abs (global_cluster)'], 'estimated_speedup')
        if l1 is None or mx is None:
            continue
        diffs.append((key, (mx - l1) * 100, speed))

    print(f'Wrote {out}')
    if diffs:
        mean = sum(item[1] for item in diffs) / len(diffs)
        print(f'max_abs - l1_sum mean: {mean:+.2f} percentage points over {len(diffs)} model/file pairs')
        for (dataset, model, source), diff, speed in diffs:
            print(f'{dataset:12s} {model:18s} diff={diff:+.2f} pp speedup={speed:.2f}x source={source}')


if __name__ == '__main__':
    main()
