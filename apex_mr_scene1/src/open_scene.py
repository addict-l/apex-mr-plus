"""Open a project-owned CoppeliaSim GUI and keep its stdin alive."""
from pathlib import Path
import argparse
import json
import time
import traceback

from sim_connection import Simulator

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', default='APEX_MR_Plus.ttt')
    parser.add_argument('--preset', type=int, default=0, choices=[0, 1, 2])
    args = parser.parse_args()
    scene = ROOT / args.scene
    if not scene.is_file():
        raise FileNotFoundError(scene)
    owner = Simulator(ROOT, port=23041, timeout=40)
    owner.start(headless=False)
    client, sim = owner.connect()
    sim.loadScene(str(scene))
    try:
        script = sim.getObject('/APEX_MR_Plus/SceneControls')
    except Exception:
        script = None
    if script is not None:
        for attempt in range(30):
            try:
                sim.callScriptFunction('initialize', script)
                break
            except Exception:
                if attempt == 29:
                    raise
                time.sleep(.1)
        sim.callScriptFunction('setPreset', script, args.preset)
    (ROOT / 'logs' / 'gui_session.json').write_text(json.dumps({
        'pid': owner.process.pid, 'port': owner.port, 'scene': str(scene),
        'preset': args.preset if script is not None else None,
        'stage': 'environment_preview' if script is not None else 'normal_dynamic_assembly'
    }, indent=2), encoding='utf-8')
    print('READY: ' + str(scene), flush=True)
    try:
        while owner.process.poll() is None:
            time.sleep(.5)
    finally:
        if owner.client is not None:
            owner._disconnect(owner.client)
        if owner._log_file:
            owner._log_file.close()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        (ROOT / 'logs').mkdir(exist_ok=True)
        (ROOT / 'logs' / 'gui_error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise
