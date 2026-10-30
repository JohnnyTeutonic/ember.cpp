"""Fail-closed receipt gate for E4's ten synthetic families (no GPU launch)."""
import argparse
import json
import math
from pathlib import Path
GPU = ('mixtral', 'qwen2moe', 'qwen3moe', 'deepseek2', 'deepseek2_qlora', 'gemma2', 'phi3_longrope', 'phi3_swa')
FALLBACK = ('deepseek2_dense_yarn', 'deepseek3')
FAMILIES = GPU + FALLBACK

def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def check(root):
    root = Path(root)
    errors = []
    for family in FAMILIES:
        reports = []
        for mode in ('default', 'det1', 'det2'):
            path = root / f'{family}_{mode}.json'
            try:
                r = json.loads(path.read_text())
                require(r['cases'] == 40 and len(r['rows']) == 40, 'incomplete cases')
                require([row['case'] for row in r['rows']] == list(range(40)), 'case inventory')
                require(r['cpu_greedy_vs_hf'] == '40/40', 'CPU greedy/HF')
                require(all((row['cpu_greedy_equal_hf'] is True for row in r['rows'])), 'CPU case/HF')
                expected_gpu = family in GPU
                require(r['device_supported'] is expected_gpu, 'unexpected device support')
                require(r['on_gpu'] is expected_gpu, 'unexpected execution backend')
                metrics = ['cpu_worst_first_rel_hf']
                if expected_gpu:
                    require(r['dev_greedy_vs_hf'] == r['dev_greedy_vs_cpu'] == '40/40', 'device greedy')
                    require(r['dev_step_top1_cpu'] == '640/640', 'device top1')
                    require(r['rows_dumped'] == 640 and r['vocab'] > 0, 'dump shape')
                    require(all((row['dev_greedy_equal_hf'] is True and row['dev_greedy_equal_cpu'] is True and (row['dev_step_top1_cpu'] == 16) for row in r['rows'])), 'device cases')
                    metrics += ['dev_worst_first_rel_hf', 'dev_worst_first_rel_cpu', 'dev_worst_step_rel_cpu']
                    dump = root / f'{family}_{mode}.bin'
                    require(dump.stat().st_size == 640 * r['vocab'] * 4, 'truncated dump')
                for metric in metrics:
                    require(math.isfinite(r[metric]) and 0 <= r[metric] <= 0.0001, metric)
                reports.append(r)
            except (OSError, ValueError, KeyError, TypeError, AssertionError) as exc:
                errors.append(f'{path.name}: {exc}')
        if len(reports) == 3:
            if reports[1] != reports[2]:
                errors.append(f'{family}: deterministic reports differ')
            if family in GPU:
                a, b = (root / f'{family}_det{k}.bin' for k in (1, 2))
                if a.exists() and b.exists() and (a.read_bytes() != b.read_bytes()):
                    errors.append(f'{family}: deterministic row bits differ')
    try:
        counts = json.loads((root / 'suite_v1.6.1_cuda_build.json').read_text())['counts']
        require(sum(counts.values()) == 598, 'incomplete suite')
        require(counts.get('FAIL') == counts.get('XFAIL') == counts.get('XPASS') == 0, 'suite failures')
        require(counts.get('PASS', 0) >= 554, 'suite pass regression')
    except (OSError, ValueError, KeyError, TypeError, AssertionError) as exc:
        errors.append(f'suite: {exc}')
    return {'pass': not errors, 'errors': errors, 'families': list(FAMILIES), 'device_families': list(GPU), 'CPU_fallback_families': list(FALLBACK)}
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root')
    args = parser.parse_args()
    result = check(args.root)
    (Path(args.root) / 'gate_summary.json').write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps(result))
    raise SystemExit(0 if result['pass'] else 1)
