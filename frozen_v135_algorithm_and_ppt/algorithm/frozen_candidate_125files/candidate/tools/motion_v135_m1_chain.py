"""Serial, resumable diagnostic replays of immutable v134 contexts."""
from pathlib import Path
import json
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import overnight_batch as old
from tools.motion_v135_control import BASE, audit, record
from tools.overnight_regression import health


def main():
    while not (BASE/'jobs/m1_tpg02_a1/result.json').exists():
        if (BASE/'global_stop.json').exists():
            return 2
        time.sleep(2)
    cases = [
        ('table', 'sim20/06/simulation/gen_f2_s20261115_014', ['--mode', 'action', '--record-index', '1']),
        ('branch', 'sim20/07/simulation/gen_f2_s20261115_012', ['--mode', 'action', '--record-index', '1']),
        ('interarm', 'sim20/08/simulation/gen_f3_s20261115_021', ['--mode', 'action', '--record-index', '45']),
        ('home', 'walls3/037/simulation/gen_f3_s20261115_037', ['--mode', 'home', '--home-arm', 'B'])]
    for name, relative, extra in cases:
        output = BASE/'m1'/name
        result_path = output/'committed.json'
        if result_path.exists():
            continue
        with old.execution_lock(ROOT/'results/sim_v1_3/overnight_execution.lock'):
            if (BASE/'global_stop.json').exists() or not audit()['passed']:
                return 2
            environment = health()
            if not environment['passed']:
                old.atomic_json(BASE/'global_stop.json', {'reason': 'M1 replay preflight', 'health': environment})
                return 2
            output.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, '-X', 'utf8', '-u', 'tools/motion_v135_replay.py',
                '--source', 'results/sim_v1_3/v134/experiments/c_wall_v134/runtime_fix2/'+relative,
                '--diagnostic-seconds', '120', '--out', str(output), *extra]
            old.atomic_json(output/'command.json', {'command': command, 'at': old.utc_now()})
            with (output/'supervisor.log').open('w', encoding='utf-8') as stream:
                child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                         stdin=subprocess.DEVNULL)
                old.atomic_json(output/'state.json', {'status': 'running', 'pid': child.pid, 'at': old.utc_now()})
                code = child.wait()
            environment = health()
            result = {'status': 'completed' if code == 0 else ('timeout' if code == 124 else 'failed'),
                      'returncode': code, 'at': old.utc_now(), 'cleanup': environment,
                      'artifacts': [str(p) for p in output.glob('*/result.json')]}
            old.atomic_json(result_path, result)
            record('M1_'+name, result['status'], evidence=str(result_path))
            print(json.dumps({'case': name, **result}, ensure_ascii=False), flush=True)
            if not environment['passed']:
                old.atomic_json(BASE/'global_stop.json', {'reason': 'M1 replay cleanup', 'health': environment})
                return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
