"""Dispatch one owned diagnostic/gate with every artifact in this round."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import overnight_batch as old
from tools.motion_v135_control import BASE


def run(*, output, task=None, gate=None, seconds=1200.):
    output = Path(output).resolve()
    if not output.is_relative_to(BASE):
        raise ValueError('单任务证据必须位于motion_v135独立命名空间')
    if (task is None) == (gate is None):
        raise ValueError('只指定task或gate之一')
    if gate is not None and gate not in old.LIVE_TESTS:
        raise ValueError('只允许原五项真实门控')
    from simbridge import runner
    original = runner.Simulator
    class NamespaceSimulator(original):
        def __init__(self, ignored_root, *args, **kwargs):
            super().__init__(output/'instance', *args, **kwargs)
    runner.Simulator = NamespaceSimulator
    try:
        job = {'output': str(output), 'timeout_seconds': seconds}
        if gate is not None:
            result = old.gate_job({**job, 'test_name': gate})
        else:
            result = old.sim_job({**job, 'task': str(old.confined(task))})
    finally:
        runner.Simulator = original
    old.atomic_json(output/'worker_result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--task')
    choice.add_argument('--gate', choices=old.LIVE_TESTS)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=1200.)
    args = parser.parse_args()
    result = run(output=args.out, task=args.task, gate=args.gate, seconds=args.seconds)
    print(__import__('json').dumps(result, ensure_ascii=False), flush=True)
    return 0 if result.get('passed') else 1


if __name__ == '__main__':
    raise SystemExit(main())
