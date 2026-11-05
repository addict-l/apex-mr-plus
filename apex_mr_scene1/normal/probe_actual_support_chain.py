"""Isolated actual-prefix full-chain support fixture; never saves a scene.

Six already assembled bodies are initialized at a failed full run's measured
precontact poses. This tests mechanics and contact skills, not autonomous
execution of the preceding assembly.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from sim_connection import Simulator
from normal.scene import resolve
from normal.probe_support_load import LUA,make_bond,stop,obb_overlap

LUA_CHAIN=LUA.replace("return {time_s=sim.getSimulationTime(),load_on=active,", "local snap={} for name,h in pairs(cfg.snap_sensors) do snap[name]=worldZ(h) end\n return {snap=snap,time_s=sim.getSimulationTime(),load_on=active,")

def compensated_q(sim,ik,arm,q,dz):
    for joint,value in zip(arm['joints'],q):sim.setJointPosition(joint,value)
    before=sim.getObjectPose(arm['tip'],-1);desired=list(before);desired[2]+=dz
    sim.setObjectPose(arm['target'],desired,-1)
    env=ik.createEnvironment();group=ik.createGroup(env)
    ik.setGroupCalculation(env,group,ik.method_damped_least_squares,.02,100)
    try:
        ik.addElementFromScene(env,group,arm['model'],arm['tip'],arm['target'],ik.constraint_pose)
        result=ik.handleGroup(env,group,{'syncWorlds':True,'allowError':False})
        after=sim.getObjectPose(arm['tip'],-1)
        values=[sim.getJointPosition(j) for j in arm['joints']]
        error=math.dist(after[:3],desired[:3])
        if error>.0001:raise RuntimeError('Fixture correction IK residual '+str(error))
        return values,{'dz_m':dz,'before':before,'desired':desired,'achieved':after,'error_m':error,'result':result,
                       'maxdq_rad':max(abs(a-b) for a,b in zip(values,q))}
    finally:ik.eraseEnvironment(env)

def initialize(sim,ik,scene,row,candidate,support,contact=True):
    sim.loadScene(str(scene));cfg=resolve(sim);parts={p['id']:p for p in cfg['parts']}
    base=sim.getObject('/APEX_MR_Plus/CastleBaseplate');snaps={}
    for name in ['L1','R1','L2','R2','L3','beam']:
        body=cfg['blocks'][name];p=parts[name];pose=row['parts'][name]
        sim.setObjectPose(body,pose,-1)
        parent=base
        if name=='beam':parent=cfg['blocks']['L3']
        elif p.get('dependencies'):parent=cfg['blocks'][p['dependencies'][0]]
        anchor=list(pose[:3]);anchor[2]-=p['size'][2]/2
        if name=='beam':anchor=list(row['parts']['L3'][:3]);anchor[2]+=.04
        snaps[name]=make_bond(sim,'ActualFixtureSnap_'+name,parent,body,anchor)
        sim.setObjectQuaternion(snaps[name],pose[3:],-1)
        # Sensor frame changes must not change the measured body pose.
        sim.setObjectPose(body,pose,-1)
    a=cfg['arms']['A'];b=cfg['arms']['B']
    targets={'A':list(support['q_contact'] if contact else support['q_precontact']),'B':list(row['qB'])}
    for side,arm in cfg['arms'].items():
        for joint,value in zip(arm['joints'],targets[side]):
            sim.setJointPosition(joint,value);sim.setJointTargetPosition(joint,value)
    cap=cfg['blocks']['cap'];sim.setObjectPose(cap,row['parts']['cap'],-1)
    grasp=make_bond(sim,'ActualFixtureHeldCap',b['palm'],cap,row['parts']['cap'][:3])
    # Match the controller's actual underside plane at the contact endpoint.
    contact_q=list(support['q_contact'])
    for j,v in zip(a['joints'],contact_q):sim.setJointPosition(j,v)
    tcp=sim.getObjectPosition(a['tip'],-1);m=sim.getObjectMatrix(cfg['blocks']['beam'],-1)
    half=parts['beam']['size'][2]/2
    bx,by,bz=m[3]-m[2]*half,m[7]-m[6]*half,m[11]-m[10]*half
    actual_bottom=bz-(m[2]*(tcp[0]-bx)+m[6]*(tcp[1]-by))/m[10]
    dz=actual_bottom-(parts['beam']['goal'][2]-half)
    corrected,correction=compensated_q(sim,ik,a,contact_q,dz)
    dq=[x-y for x,y in zip(corrected,contact_q)]
    targets['A']=corrected if contact else list(support['q_precontact'])
    for j,v in zip(a['joints'],targets['A']):sim.setJointPosition(j,v);sim.setJointTargetPosition(j,v)
    pad=a['support_pad'];sim.setObjectInt32Param(pad,sim.shapeintparam_respondable,1)
    fixture={'beam':cfg['blocks']['beam'],'pad':pad,'pad_sensor':a['support_sensor'],'tool_sensor':a['sensor'],
             'tipA':a['tip'],'tipB':b['tip'],'jointsA':a['joints'],'jointsB':b['joints'],
             'beam_snap':snaps['beam'],'snap_sensors':snaps,'load_point':parts['cap']['goal'],'force_N':8.,'targets':targets}
    sim.writeCustomTableData(cfg['root'],'SupportLoadProbe',fixture)
    script=sim.createScript(sim.scripttype_simulation,LUA_CHAIN,0,'lua')
    sim.setObjectAlias(script,'ActualPrefixSupportFixture');sim.setObjectParent(script,cfg['root'],True)
    return cfg,script,targets,dq,correction,snaps

def summarize(samples,dt):
    result={}
    for stage in dict.fromkeys(r['stage'] for r in samples):
        rows=[r for r in samples if r['stage']==stage];tail=rows[-max(1,round(.1/dt)):]
        item={'snaps':{}}
        for name in ['L1','R1','L2','R2','L3','beam']:
            values=[r['snap'][name]['torque_norm_Nm'] for r in rows if r['snap'][name]['valid']]
            count=longest=0
            for r in rows:
                v=r['snap'][name];count=count+1 if v['valid'] and v['torque_norm_Nm']>1.2 else 0;longest=max(count,longest)
            item['snaps'][name]={'peak_Nm':max(values) if values else None,
                'tail_mean_Nm':statistics.fmean(r['snap'][name]['torque_norm_Nm'] for r in tail if r['snap'][name]['valid']),
                'longest_over_1_2_Nm_s':longest*dt}
        item['peak_abs_pad_world_z_N']=max(abs(r['pad'].get('force_world_z_N',0)) for r in rows)
        item['tail_pad_world_z_N']=statistics.fmean(r['pad']['force_world_z_N'] for r in tail if r['pad']['valid'])
        item['peak_joint_error_rad']=max(max(r['max_joint_error_rad'].values()) for r in rows)
        result[stage]=item
    result['all_snap_peaks_below_1_2_Nm']=all(v['peak_Nm'] is not None and v['peak_Nm']<1.2 for stage in result.values() if isinstance(stage,dict) for v in stage['snaps'].values())
    return result

def run_static(sim,ik,scene,row,candidate,support):
    cfg,script,targets,dq,correction,snaps=initialize(sim,ik,scene,row,candidate,support,True)
    initial={name:sim.getObjectPose(h,-1) for name,h in cfg['blocks'].items()}
    overlap=obb_overlap(sim,cfg['arms']['A']['support_pad'],cfg['blocks']['beam'])
    dt=sim.getFloatParam(sim.floatparam_simulation_time_step);samples=[]
    sim.setStepping(True);sim.startSimulation()
    for stage,duration in [('settle',.5),('loaded',.5),('unloaded',.5)]:
        sim.callScriptFunction('setLoad',script,stage=='loaded')
        for _ in range(round(duration/dt)):
            sim.step();sample=sim.callScriptFunction('sampleFixture',script);sample['stage']=stage;samples.append(sample)
    stop(sim)
    return {'case':'actual_prefix_static_contact_8N','contact_correction':correction,'dq':dq,'targets':targets,
            'initial_poses':initial,'initial_obb':overlap,'dt_s':dt,'summary':summarize(samples,dt),'samples':samples}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=23045)
    parser.add_argument('--time',type=float,default=113.3);args=parser.parse_args()
    out=ROOT/'results'/'normal';scene=ROOT/'scenes'/'10_Normal_Assembly.ttt'
    execution_path=out/'execution.json';execution_bytes=execution_path.read_bytes();execution=json.loads(execution_bytes)
    row=max((r for r in execution['trace'] if r['time']<=args.time),key=lambda r:r['time'])
    allocation=json.loads((out/'allocation.json').read_text(encoding='utf-8'))
    problem=json.loads((out/'allocation_problem.json').read_text(encoding='utf-8'))
    choice=next(c for c in allocation['choices'] if c['part']=='cap')
    candidate=problem['candidates'][choice['candidate_index']];support=candidate['support_candidates'][choice['support_candidate_index']]
    scene_digest=hashlib.sha256(scene.read_bytes()).hexdigest()
    data={'scope':'Measured six-part prefix initialized as full dynamic Snap chain; fixture, not full assembly',
          'shared_scene_saved':False,'scene_sha256':scene_digest,'prefix_execution_sha256':hashlib.sha256(execution_bytes).hexdigest(),
          'prefix_failure':execution.get('failure'),'prefix_time_s':row['time'],'prefix_row':row,'choice':choice,
          'support_candidate':support,'snap_breaking_enabled':False,'force_N':8.,'cases':[]}
    with Simulator(ROOT,port=args.port,timeout=40) as owner:
        client,sim=owner.connect();ik=client.require('simIK')
        case=run_static(sim,ik,scene,row,candidate,support);data['cases'].append(case)
        print(json.dumps({'case':case['case'],'correction':case['contact_correction'],'summary':case['summary']},ensure_ascii=False,indent=2),flush=True)
    data['shared_scene_unchanged']=scene_digest==hashlib.sha256(scene.read_bytes()).hexdigest()
    path=out/'actual_support_chain_probe.json';path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    print(str(path),flush=True)

if __name__=='__main__':main()
