"""Use the repository's detached WMI launcher with local receipts."""
from pathlib import Path
import json
import sys

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE.parents[4]))

from tools import motion_v135_detached as detached

detached.BASE = HERE / 'overnight'


if sys.argv[1] == '--worker':
    raise SystemExit(detached.worker(Path(sys.argv[2])))

run_id, launch_id = sys.argv[1:]
spawn_wmi = detached._spawn_wmi


def spawn_local_worker(command, timeout):
    command[command.index('--worker') - 1] = str(Path(__file__).resolve())
    return spawn_wmi(command, timeout)


detached._spawn_wmi = spawn_local_worker
receipt = detached.launch(launch_id, detached.BASE / 'launcher' / f'{launch_id}.log',
                          [HERE / 'showcase_overnight.py', '--run-id', run_id], via='wmi')
print(json.dumps(receipt, ensure_ascii=False))
