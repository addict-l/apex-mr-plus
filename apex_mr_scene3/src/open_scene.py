"""Open the current main scene or explicit static preview without changing it."""
import argparse
from pathlib import Path
import time
from sim_connection import Simulator

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    choice=parser.add_mutually_exclusive_group()
    choice.add_argument('--scene',type=Path,help='Explicit scene file; relative paths resolve against scene3')
    choice.add_argument('--preview',action='store_true',help='Open the separate legacy five-state design preview')
    args=parser.parse_args()
    scene = args.scene or ROOT / 'apex_mr_scene3.ttt'
    if args.preview:
        scene=ROOT/'scenes/00_Scene3_Design_Preview.ttt'
        if not scene.is_file():scene=ROOT/'backups/design_preview/apex_mr_scene3.ttt'
    elif not scene.is_absolute():scene=ROOT/scene
    if not scene.is_file():
        raise FileNotFoundError('Scene missing: '+str(scene)+'. Use --preview for design states, or run the APEX pipeline to create an audited main scene.')
    owner = Simulator(ROOT, port=23054, timeout=40)
    try:
        owner.start(headless=False)
        _, sim = owner.connect()
        sim.loadScene(str(scene))
        config=sim.readCustomTableData(sim.getObject('/APEX_MR_Scene3'),'Scene3APEX')
        if isinstance(config,dict) and config.get('program',{}).get('nodes'):
            print('Scene 3 APEX planned scene opened. Start simulation to execute its embedded TPG.',flush=True)
        else:
            print('Scene 3 design preview opened. The five buttons switch static design states.',flush=True)
        while owner.process.poll() is None:
            time.sleep(.5)
    finally:
        owner.close()

if __name__ == '__main__':
    main()
