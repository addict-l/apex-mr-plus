"""Fixture for the final selected grasp/support mode and footprint margin."""
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from sim_connection import Simulator
from normal.probe_support_load import run_case

def main():
    scene=ROOT/'scenes'/'10_Normal_Assembly.ttt';digest=hashlib.sha256(scene.read_bytes()).hexdigest()
    out=ROOT/'results'/'normal'
    allocation=json.loads((out/'allocation.json').read_text(encoding='utf-8'))
    problem=json.loads((out/'allocation_problem.json').read_text(encoding='utf-8'))
    choice=next(c for c in allocation['choices'] if c['part']=='cap')
    candidate=problem['candidates'][choice['candidate_index']]
    support=candidate['support_candidates'][choice['support_candidate_index']]
    if candidate['arm']!='B' or support['arm']!='A':raise RuntimeError('Fixture currently expects cap installer B/supporter A')
    data={'scope':'final allocated mode support footprint fixture; not full assembly',
          'scene_sha256':digest,'shared_scene_saved':False,'choice':choice,
          'support_candidate':support,'cases':[]}
    with Simulator(ROOT,port=23045,timeout=40) as owner:
        client,sim=owner.connect();ik=client.require('simIK')
        for label,dy in [('chosen_yaw_original_y',0.),('chosen_yaw_y_plus_2mm',.002),('chosen_yaw_y_plus_5mm',.005)]:
            result=run_case(sim,scene,candidate,support,'supported',ik=ik,
                            beam_z_offset=-.0015,tcp_z_offset=-.0015,tcp_y_offset=dy,label=label)
            data['cases'].append(result)
            print(json.dumps({'case':label,'obb':result['initial_pad_beam_obb'],
                              'summary':result['summary'],'tcp_correction':result['tcp_correction']},ensure_ascii=False,indent=2),flush=True)
    data['shared_scene_unchanged']=digest==hashlib.sha256(scene.read_bytes()).hexdigest()
    (out/'chosen_support_probe.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':main()
