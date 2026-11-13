import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

IGNORE = {'seed', 'name_keys', 'data_root', 'data_dir', 'num_workers'}

def main():
    p = argparse.ArgumentParser(description='')
    p.add_argument('result_dir')
    p.add_argument('--csv', default=None, help='also write the table to this CSV file')
    p.add_argument('--ddof', type=int, default=0, help='0 for Tables 2 and 3, 1 for Table 1')
    args = p.parse_args()

    runs = [json.loads(f.read_text()) for f in sorted(Path(args.result_dir).glob('*.json'))]
    if not runs:
        raise SystemExit(f'no result files in {args.result_dir}')
    varying = sorted(k for k in set().union(*(r['config'] for r in runs)) - IGNORE
                     if len({json.dumps(r['config'].get(k)) for r in runs}) > 1)
    groups = defaultdict(list)
    for r in runs:
        groups[tuple(json.dumps(r['config'].get(k)) for k in varying)].append(r['metrics'])

    metric_keys = list(dict.fromkeys(
        k for r in runs for k, v in r['metrics'].items()
        if isinstance(v, (int, float)) and not isinstance(v, bool) and k not in ('minutes', 'n_params')))
    header = varying + ['n'] + metric_keys
    rows = []
    for key, ms in groups.items():
        row = [json.loads(v) for v in key] + [len(ms)]
        for k in metric_keys:
            vals = np.array([m[k] for m in ms if m.get(k) is not None], dtype=float)
            if len(vals) == 0:
                row.append('')
            elif len(vals) == 1:
                row.append(f'{vals[0]:.4g}')
            else:
                row.append(f'{vals.mean():.4g} ± {vals.std(ddof=args.ddof):.2g}')
        rows.append(row)

    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
    for line in [header] + rows:
        print('  '.join(str(x).ljust(w) for x, w in zip(line, widths)))
    if args.csv:
        import csv
        with open(args.csv, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerows([header] + rows)

if __name__ == '__main__':
    main()
