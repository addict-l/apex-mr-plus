"""Replay staged support/insertion/retreat on a measured assembly prefix.

This injects only the initialized ownership/constraint state into an in-memory
copy of the real central controller. Its contact compensation, motor targets,
force tare, reset, insertion load, and overload monitoring remain unchanged.
The shared TTT/controller/reference files are never saved or modified.
"""
from __future__ import annotations
from copy import deepcopy
import hashlib,json,math
from pathlib import Path
import sys,time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from sim_connection import Simulator
from normal.probe_actual_support_chain import initialize
from normal.probe_support_load import stop
from normal.program import compress_collinear

def node_from_task(task):
    path=compress_collinear(task['q_path']);segments=[]
    vmax=min(.6,float(task['metadata'].get('max_velocity_rad_s',.6)))
    amax=min(1.2,float(task['metadata'].get('max_acceleration_rad_s2',1.2)))
    for q0,q1 in zip(path,path[1:]):
        d=max(abs(a-b) for a,b in zip(q0,q1))
        if d<1e-9:continue
        duration=max(1.875*d/vmax,math.sqrt(10/math.sqrt(3)*d/amax),.08)
        segments.append({'q0':q0,'q1':q1,'duration':duration})
    duration=max(.08,sum(s['duration'] for s in segments))+float(task['metadata'].get('dwell_s',.2))
    return {'id':'Fixture_'+task['id'],'task_id':task['id'],'arm':task['arm'],'kind':task['kind'],'part':task['part'],
            'node_type':'skill','q_samples':deepcopy(task['q_path']),'q_end':list(task['q_path'][-1]),
            'motion_segments':segments,'duration':duration,'metadata':deepcopy(task['metadata']),
            'payload':deepcopy(task.get('payload'))}

def main():
    out=ROOT/'results'/'normal';scene=ROOT/'scenes'/'10_Normal_Assembly.ttt'
    # The static fixture freezes the exact failed-run source before any root
    # execution.json replacement, preserving a reproducible assembly prefix.
    prefix=json.loads((out/'actual_support_chain_probe.json').read_text(encoding='utf-8'))
    row=prefix['prefix_row'];choice=prefix['choice'];support=prefix['support_candidate']
    problem=json.loads((out/'allocation_problem.json').read_text(encoding='utf-8'))
    candidate=problem['candidates'][choice['candidate_index']]
    stage_path=out/'contact_skill_staging'/'staged_skills.json';stage_bytes=stage_path.read_bytes();staged=json.loads(stage_bytes)
    wanted=['cap_A_support_start_047','cap_B_place_048','cap_A_support_end_051']
    tasks=[next(t for t in staged['tasks'] if t['id']==name) for name in wanted]
    nodes=[node_from_task(t) for t in tasks]
    edges=[{'source':a['id'],'target':b['id'],'type':'task'} for a,b in zip(nodes,nodes[1:])]
    program={'nodes':nodes,'edges':edges}
    controller_path=ROOT/'normal'/'controller.lua';source=controller_path.read_text(encoding='utf-8')
    marker=" for _,edge in ipairs(program.edges) do pred[edge.target][#pred[edge.target]+1]=edge.source end"
    seed_code="""
 local seed=sim.readCustomTableData(cfg.root,'ActualSupportSequenceSeed')
 for name,h in pairs(seed.snap_sensors) do snapSensors[name]=h end
 held.B='cap';graspSensors.cap=seed.grasp_sensor
"""
    if source.count(marker)!=1:raise RuntimeError('Controller initialization marker changed; inspect injection')
    injected=source.replace(marker,marker+seed_code)
    scene_sha=hashlib.sha256(scene.read_bytes()).hexdigest()
    data={'scope':'Initialized measured six-part prefix and actual held cap; 3-skill dynamics fixture, not full assembly',
          'prefix_time_s':row['time'],'prefix_execution_sha256':prefix['prefix_execution_sha256'],
          'scene_sha256':scene_sha,'shared_scene_saved':False,'controller_sha256':hashlib.sha256(source.encode()).hexdigest(),
          'staged_source_sha256':hashlib.sha256(stage_bytes).hexdigest(),'chosen_mode':choice,
          'program':program,'controller_modification':'seed existing dynamic Snap constraints and cap ownership only'}
    with Simulator(ROOT,port=23045,timeout=40) as owner:
        client,sim=owner.connect();ik=client.require('simIK')
        cfg,old_script,targets,dq,correction,snaps=initialize(sim,ik,scene,row,candidate,support,False)
        sim.removeObjects([old_script])
        grasp=sim.getObject('/APEX_MR_Plus/B_UR5/ActualFixtureHeldCap',{'noError':True})
        if grasp<0:
            handles=sim.getObjectsInTree(cfg['root'],sim.object_forcesensor_type,0)
            grasp=next(h for h in handles if sim.getObjectAlias(h)=='ActualFixtureHeldCap')
        cfg['program']=program;sim.writeCustomTableData(cfg['root'],'NormalAssembly',cfg)
        sim.writeCustomTableData(cfg['root'],'ActualSupportSequenceSeed',{'snap_sensors':snaps,'grasp_sensor':grasp})
        script=sim.createScript(sim.scripttype_simulation,injected,0,'lua')
        sim.setObjectAlias(script,'ActualPrefixSequenceController');sim.setObjectParent(script,cfg['root'],True)
        sim.setStepping(True);sim.startSimulation();last_print=-5;wall=time.monotonic()
        while True:
            sim.step();state=sim.callScriptFunction('getState',script)
            if state['time']-last_print>=2 or state.get('failure') or state.get('done'):
                print(json.dumps(state,ensure_ascii=False),flush=True);last_print=state['time']
            if state.get('failure') or state.get('done'):break
            if state['time']>45 or time.monotonic()-wall>240:raise TimeoutError('Sequence fixture timed out')
        execution=sim.callScriptFunction('getResult',script)
        data['execution']=execution;data['fixture_success']=bool(execution.get('done') and not execution.get('failure'))
        summary={}
        for name in ['L1','R1','L2','R2','L3','beam','cap']:
            values=[r['snap'][name]['torque_norm_Nm'] for r in execution['trace'] if name in r.get('snap',{}) and r['snap'][name].get('valid')]
            summary[name]={'peak_Nm':max(values) if values else None}
        data['snap_trace_summary']=summary
        data['peak_support_normal_N']=execution['peak_support_normal_N'];data['wall_time_s']=time.monotonic()-wall
        stop(sim)
    data['shared_scene_unchanged']=scene_sha==hashlib.sha256(scene.read_bytes()).hexdigest()
    path=out/'actual_support_sequence_probe.json';path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'output':str(path),'success':data['fixture_success'],'failure':execution.get('failure'),
                      'snaps':summary,'peak_support_normal_N':execution['peak_support_normal_N']},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
