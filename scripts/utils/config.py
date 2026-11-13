import argparse
import copy
import itertools

import yaml

def _yaml_value(text):
    value = yaml.safe_load(text)

    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value

def _fix_floats(obj):
    if isinstance(obj, dict):
        return {k: _fix_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_fix_floats(v) for v in obj]
    if isinstance(obj, str):
        try:
            return float(obj) if any(c in obj for c in 'eE.') and obj.lower() not in ('nan', 'inf') else obj
        except ValueError:
            return obj
    return obj

def parse_overrides(items):
    out = {}
    for item in items:
        if '=' not in item:
            raise ValueError(f'--set expects key=value, got {item!r}')
        key, value = item.split('=', 1)
        out[key.strip()] = _fix_floats(_yaml_value(value))
    return out

def load_config(path, overrides=()):
    with open(path) as f:
        cfg = _fix_floats(yaml.safe_load(f) or {})
    cfg.setdefault('base', {})
    cfg['overrides'] = parse_overrides(overrides)
    return cfg

def expand_runs(cfg):
    base = cfg.get('base', {})
    grid = cfg.get('grid') or {}
    runs = cfg.get('runs') or [{}]
    keys = list(grid)
    combos = list(itertools.product(*(grid[k] for k in keys))) if keys else [()]
    out = []
    for run in runs:
        for combo in combos:
            c = copy.deepcopy(base)
            c.update(dict(zip(keys, combo)))
            c.update(run)
            c.update(cfg.get('overrides', {}))
            if c not in out:
                out.append(c)
    return out

def run_name(cfg, keys, overrides=None):

    short = {'use_global_norm': 'gn', 'warmup_steps': 'warmup', 'batch_size': 'bs', 'weight_decay': 'wd'}
    neutral = {'data_root', 'data_dir', 'num_workers', 'name_keys', 'lr_from', 'compile', 'log_interval'}
    keys = list(keys) + sorted(k for k in (overrides or {}) if k not in keys and k not in neutral)
    parts = []
    for k in keys:
        if k not in cfg:
            continue
        v, tag = cfg[k], short.get(k, k)
        if isinstance(v, bool):
            parts.append(tag if v else f'no_{tag}')
        elif isinstance(v, str):
            parts.append(v)
        elif isinstance(v, float):
            parts.append(f'{tag}{v:g}')
        elif isinstance(v, (list, tuple)):
            parts.append(tag + '-'.join(f'{x:g}' if isinstance(x, float) else str(x) for x in v))
        else:
            parts.append(f'{tag}{v}')
    return '_'.join(parts).replace('/', '-').replace('<', '').replace('>', '').replace(' ', '-')

def parse_args(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--config', required=True, help='YAML file in scripts/configs/')
    p.add_argument('--set', action='append', default=[], metavar='KEY=VALUE',
                   help='override a base setting (repeatable)')
    p.add_argument('--out', default='results', help='output root (default: results/)')
    p.add_argument('--device', default=None, help='e.g. cuda:1 (default: $RLO_DEVICE or cuda)')
    p.add_argument('--only', type=int, default=None,
                   help='run only the run with this index (for job arrays)')
    p.add_argument('--dry-run', action='store_true', help='list the runs and exit')
    p.add_argument('--force', action='store_true', help='rerun runs whose result file exists')
    return p.parse_args()
