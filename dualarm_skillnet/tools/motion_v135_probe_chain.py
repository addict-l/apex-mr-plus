"""Repeat three original contexts after the mandatory normal-motion gate."""
import argparse
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import overnight_batch as old
from tools.motion_v135_control import BASE, audit, record, version
from tools.overnight_regression import health


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt', default='m3_probes_a1')
    parser.add_argument('--gate-job', default='m3_parallel_gate_a1')
    args = parser.parse_args()
    output = old.confined(BASE/'diagnostics'/args.attempt)
    gate = old.confined(BASE/'jobs'/args.gate_job/'result.json')
    while not gate.is_file():
        if (BASE/'global_stop.json').exists():
            return 2
        time.sleep(2)
    if old.read_json(gate).get('status') != 'passed':
        old.atomic_json(output/'state.json', {'status': 'blocked', 'reason': 'normal-motion gate failed', 'gate': str(gate)})
        return 2
    cases = [('table', 'sim20/06/simulation/gen_f2_s20261115_014', 1),
             ('branch', 'sim20/07/simulation/gen_f2_s20261115_012', 1),
             ('interarm', 'sim20/08/simulation/gen_f3_s20261115_021', 45)]
    pinned = version()
    old.atomic_json(output/'code_version.json', pinned)
    startup_streak = 0
    for name, relative, index in cases:
        folder = output/name
        committed = folder/'committed.json'
        if committed.exists():
            continue
        with old.execution_lock(ROOT/'results/sim_v1_3/overnight_execution.lock'):
            if version() != pinned:
                raise RuntimeError('diagnostic code changed; retain attempt and create another')
            if (BASE/'global_stop.json').exists() or not audit()['passed']:
                return 2
            machine = health()
            if not machine['passed']:
                old.atomic_json(BASE/'global_stop.json', {'reason': 'probe environment/cleanup', 'health': machine})
                return 2
            folder.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, '-X', 'utf8', '-u', 'tools/motion_v135_replay.py',
                '--source', 'results/sim_v1_3/v134/experiments/c_wall_v134/runtime_fix2/'+relative,
                '--mode', 'action', '--record-index', str(index), '--use-backend',
                '--diagnostic-seconds', '180', '--out', str(folder)]
            old.atomic_json(folder/'command.json', {'command': command, 'at': old.utc_now(),
                'change_from_baseline': 'complete-action endpoint backtracking + revalidated point IK hints + scoped8/24 strategies',
                'formal_run': False, 'diagnostic_budget': 180, 'baseline_diagnostic_budget': 120})
            with (folder/'supervisor.log').open('w', encoding='utf-8') as stream:
                child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
                old.atomic_json(output/'state.json', {'status': 'running', 'case': name, 'pid': child.pid, 'at': old.utc_now()})
                code = child.wait()
            results = list(folder.glob('*/result.json'))
            actual = old.read_json(results[-1]) if results else {}
            machine = health()
            startup_failure = actual.get('status') == 'error' and actual.get('phase') in {'simulator_start', 'simulator_connect'}
            startup_streak = startup_streak+1 if startup_failure else 0
            result = {'status': actual.get('status', 'timeout' if code == 124 else 'missing'),
                'passed': actual.get('passed', False), 'returncode': code, 'cleanup': machine,
                'evidence': str(results[-1]) if results else None, 'at': old.utc_now(), 'startup_streak': startup_streak}
            old.atomic_json(committed, result)
            record(args.attempt+'_'+name, result['status'], evidence=str(committed))
            print(__import__('json').dumps({'case': name, **result}, ensure_ascii=False), flush=True)
            if startup_streak >= 3 or not machine['passed']:
                old.atomic_json(BASE/'global_stop.json', {'reason': 'probe startup3 or failed cleanup', 'detail': result})
                return 2
    old.atomic_json(output/'state.json', {'status': 'finished', 'at': old.utc_now()})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
