"""Dynamic support-load fixture; not an end-to-end assembly execution.

The scene is loaded into an isolated simulator, fixture parts are initialized
at their goals while stopped, and ideal rigid force-sensor bonds are added.
No shared TTT is saved. A precontact control distinguishes shoe self-weight
from actual contact load in the supported case.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from copy import deepcopy
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from sim_connection import Simulator
from normal.scene import resolve


LUA='''sim=require 'sim'
local cfg,beam,active=false,nil,false
local function worldZ(sensor)
 local status,f,t=sim.readForceSensor(sensor)
 if status&1==0 then return {valid=false,status=status} end
 local m=sim.getObjectMatrix(sensor,sim.handle_world)
 return {valid=true,status=status,force_local_N=f,torque_local_Nm=t,
         force_world_z_N=m[9]*f[1]+m[10]*f[2]+m[11]*f[3],
         force_norm_N=math.sqrt(f[1]*f[1]+f[2]*f[2]+f[3]*f[3]),
         torque_norm_Nm=math.sqrt(t[1]*t[1]+t[2]*t[2]+t[3]*t[3])}
end
function sysCall_init()
 cfg=sim.readCustomTableData(sim.getObject('/APEX_MR_Plus'),'SupportLoadProbe')
 beam=cfg.beam;active=false
end
function setLoad(enabled) active=enabled end
function sysCall_dyn(data)
 if not data.afterStep and active then
  local p=sim.getObjectPosition(beam,sim.handle_world)
  local rx=cfg.load_point[1]-p[1];local ry=cfg.load_point[2]-p[2]
  local f=cfg.force_N
  -- The external-wrench buffer persists until sim.handleDynamics completes.
  -- Without this reset flag, repeated substep calls accumulate 8,16,... N.
  sim.addForceAndTorque(beam+sim.handleflag_resetforcetorque,{0,0,-f},{-ry*f,rx*f,0})
 end
end
function sampleFixture()
 local qA,qB={},{}
 for i,j in ipairs(cfg.jointsA) do qA[i]=sim.getJointPosition(j) end
 for i,j in ipairs(cfg.jointsB) do qB[i]=sim.getJointPosition(j) end
 local errors={}
 for _,side in ipairs({'A','B'}) do
  local q=side=='A' and qA or qB; local target=cfg.targets[side]
  local worst=0
  for i,v in ipairs(q) do worst=math.max(worst,math.abs(math.atan2(math.sin(v-target[i]),math.cos(v-target[i])))) end
  errors[side]=worst
 end
 return {time_s=sim.getSimulationTime(),load_on=active,
         pad=worldZ(cfg.pad_sensor),tool=worldZ(cfg.tool_sensor),
         beam_snap=worldZ(cfg.beam_snap),beam_pose=sim.getObjectPose(beam,sim.handle_world),
         tcp_A=sim.getObjectPose(cfg.tipA,sim.handle_world),tcp_B=sim.getObjectPose(cfg.tipB,sim.handle_world),
         max_joint_error_rad=errors,
         pad_dynamic=sim.isDynamicallyEnabled(cfg.pad),
         pad_sensor_dynamic=sim.isDynamicallyEnabled(cfg.pad_sensor),
         beam_snap_dynamic=sim.isDynamicallyEnabled(cfg.beam_snap)}
end'''


def stop(sim):
    sim.stopSimulation()
    for _ in range(30):
        if sim.getSimulationState()==sim.simulation_stopped:return
        sim.step()
    raise RuntimeError('Fixture simulator did not stop')


def make_bond(sim,name,parent,body,anchor):
    sensor=sim.createForceSensor(0,[0,1,10,0,0],[.006,500.,100.,0.,0.])
    sim.setObjectAlias(sensor,name)
    sim.setObjectParent(sensor,parent,True)
    sim.setObjectPose(sensor,list(anchor)+[0.,0.,0.,1.],-1)
    sim.setObjectInt32Param(sensor,sim.objintparam_visibility_layer,0)
    sim.setObjectParent(body,sensor,True)
    sim.resetDynamicObject(body)
    return sensor


def candidate_from_disk():
    problem=ROOT/'results'/'normal'/'allocation_problem.json'
    candidates=ROOT/'results'/'normal'/'candidate_qualification.json'
    if problem.exists():
        source=problem;items=json.loads(source.read_text(encoding='utf-8'))['candidates']
    else:
        source=candidates;items=json.loads(source.read_text(encoding='utf-8'))
    for index,candidate in enumerate(items):
        if candidate.get('part')=='cap' and candidate.get('arm')=='B' and candidate.get('feasible'):
            for support_index,support in enumerate(candidate.get('support_candidates',[])):
                if support.get('arm')=='A' and support.get('feasible') and support.get('q_contact') and support.get('q_precontact'):
                    return source,index,support_index,candidate,support
    raise RuntimeError('No cap installer B / supporter A qualified endpoint candidate found')


def summarize(samples,initial_beam,dt):
    answer={}
    for stage in ['settle','loaded']:
        rows=[row for row in samples if row['stage']==stage]
        pad=[row['pad'] for row in rows if row['pad']['valid']]
        snaps=[row['beam_snap'] for row in rows if row['beam_snap']['valid']]
        if not pad or not snaps:raise RuntimeError('No valid force samples in '+stage)
        tail=rows[-max(1,round(.10/dt)):]
        answer[stage]={
          'valid_pad_samples':len(pad),'valid_snap_samples':len(snaps),
          'peak_abs_pad_world_z_N':max(abs(v['force_world_z_N']) for v in pad),
          'peak_pad_force_norm_N':max(v['force_norm_N'] for v in pad),
          'tail_mean_pad_world_z_N':statistics.fmean(row['pad']['force_world_z_N'] for row in tail if row['pad']['valid']),
          'peak_beam_snap_torque_Nm':max(v['torque_norm_Nm'] for v in snaps),
          'tail_mean_beam_snap_torque_Nm':statistics.fmean(row['beam_snap']['torque_norm_Nm'] for row in tail if row['beam_snap']['valid']),
          'peak_beam_translation_drift_m':max(math.dist(row['beam_pose'][:3],initial_beam[:3]) for row in rows),
          'peak_joint_tracking_error_rad':max(max(row['max_joint_error_rad'].values()) for row in rows),
        }
        for limit,key in [(1.2,'1_2'),(1.38,'1_38')]:
            consecutive=0;longest=0;over_count=0
            for row in rows:
                over=row['beam_snap']['valid'] and row['beam_snap']['torque_norm_Nm']>limit
                consecutive=consecutive+1 if over else 0
                longest=max(longest,consecutive);over_count+=int(over)
            answer[stage]['torque_over_'+key+'_Nm_samples']=over_count
            answer[stage]['longest_torque_over_'+key+'_Nm_s']=longest*dt
    answer['all_dynamic']=all(row['pad_dynamic'] and row['pad_sensor_dynamic'] and row['beam_snap_dynamic'] for row in samples)
    answer['pad_preload_exceeds_30_N']=answer['settle']['peak_abs_pad_world_z_N']>30.
    answer['beam_torque_exceeds_1_2_Nm']=answer['loaded']['peak_beam_snap_torque_Nm']>1.2
    return answer


def obb_overlap(sim,first,second):
    """Exact cuboid OBB separating-axis check, including rotated footprints."""
    boxes=[]
    for h in [first,second]:
        low=np.array([sim.getObjectFloatParam(h,getattr(sim,'objfloatparam_objbbox_min_'+axis)) for axis in 'xyz'])
        high=np.array([sim.getObjectFloatParam(h,getattr(sim,'objfloatparam_objbbox_max_'+axis)) for axis in 'xyz'])
        matrix=np.array(sim.getObjectMatrix(h,-1)).reshape(3,4)
        rotation=matrix[:,:3];center=matrix[:,3]+rotation@((low+high)/2)
        boxes.append((center,rotation,(high-low)/2))
    (ca,ra,ha),(cb,rb,hb)=boxes
    axes=[('pad_'+str(i),ra[:,i]) for i in range(3)]+[('beam_'+str(i),rb[:,i]) for i in range(3)]
    axes.extend(('cross_'+str(i)+'_'+str(j),np.cross(ra[:,i],rb[:,j])) for i in range(3) for j in range(3))
    checks=[]
    for name,axis in axes:
        length=np.linalg.norm(axis)
        if length<1e-10:continue
        axis=axis/length
        radius_a=float(np.sum(ha*np.abs(ra.T@axis)))
        radius_b=float(np.sum(hb*np.abs(rb.T@axis)))
        separation=abs(float(np.dot(cb-ca,axis)))
        checks.append({'axis':name,'world_direction':axis.tolist(),'overlap_margin_m':radius_a+radius_b-separation})
    return {'intersect':all(c['overlap_margin_m']>=-1e-9 for c in checks),
            'minimum_overlap_margin_m':min(c['overlap_margin_m'] for c in checks),
            'separating_axes':[c for c in checks if c['overlap_margin_m']<-1e-9],
            'pad_center_world':ca.tolist(),'beam_center_world':cb.tolist(),
            'pad_axes_world':ra.tolist(),'beam_axes_world':rb.tolist()}


def run_case(sim,scene,candidate,support,case,settle_s=.5,loaded_s=.5,
             *,ik=None,beam_z_offset=0.,tcp_z_offset=0.,tcp_y_offset=0.,label=None):
    sim.loadScene(str(scene));cfg=resolve(sim)
    parts={p['id']:p for p in cfg['parts']}
    fixtures={}
    baseplate=sim.getObject('/APEX_MR_Plus/CastleBaseplate')
    for name in ['L1','R1','L2','R2','L3','beam']:
        p=parts[name];body=cfg['blocks'][name]
        position=list(p['goal'])
        if name=='beam':position[2]+=beam_z_offset
        sim.setObjectPose(body,position+[0,0,0,1],-1)
        parent=baseplate
        if name=='beam':parent=cfg['blocks']['L3']
        elif p.get('dependencies'):parent=cfg['blocks'][p['dependencies'][0]]
        anchor=list(p['goal']);anchor[2]-=p['size'][2]/2
        if name=='beam':
            anchor=list(parts['L3']['goal']);anchor[2]+=parts['L3']['size'][2]/2
        fixtures[name]=make_bond(sim,'ProbeSnap_'+name,parent,body,anchor)
    targets={'A':list(support['q_contact'] if case=='supported' else support['q_precontact']),
             'B':list(candidate['endpoint_evidence'].get('goal_approach',cfg['arms']['B']['home_q']))}
    for side,arm in cfg['arms'].items():
        for joint,value in zip(arm['joints'],targets[side]):
            sim.setJointPosition(joint,value);sim.setJointTargetPosition(joint,value)
    correction=None
    if tcp_z_offset or tcp_y_offset:
        if ik is None:raise ValueError('A simIK client is required for TCP compensation')
        arm=cfg['arms']['A'];before=sim.getObjectPose(arm['tip'],-1)
        desired=list(before);desired[2]+=tcp_z_offset;desired[1]+=tcp_y_offset
        sim.setObjectPose(arm['target'],desired,-1)
        env=ik.createEnvironment();group=ik.createGroup(env)
        ik.setGroupCalculation(env,group,ik.method_damped_least_squares,.05,150)
        try:
            ik.addElementFromScene(env,group,arm['model'],arm['tip'],arm['target'],ik.constraint_pose)
            result=ik.handleGroup(env,group,{'syncWorlds':True,'allowError':False})
            after=sim.getObjectPose(arm['tip'],-1)
            error=math.dist(after[:3],desired[:3])
            q=[sim.getJointPosition(j) for j in arm['joints']]
            correction={'requested_tcp_z_offset_m':tcp_z_offset,'requested_tcp_y_offset_m':tcp_y_offset,'before_tcp_pose':before,
                        'target_tcp_pose':desired,'achieved_tcp_pose':after,'position_error_m':error,
                        'max_joint_change_rad':max(abs(a-b) for a,b in zip(q,targets['A'])),
                        'ik_result':result}
            if error>1e-4:raise RuntimeError('TCP offset IK residual exceeds 0.1 mm')
            targets['A']=q
            for joint,value in zip(arm['joints'],q):sim.setJointTargetPosition(joint,value)
        finally:ik.eraseEnvironment(env)
    a=cfg['arms']['A'];pad=a['support_pad']
    sim.setObjectInt32Param(pad,sim.shapeintparam_respondable,1)
    sim.setObjectInt32Param(pad,sim.objintparam_visibility_layer,1)
    fixture={'beam':cfg['blocks']['beam'],'pad':pad,'pad_sensor':a['support_sensor'],'tool_sensor':a['sensor'],
             'tipA':a['tip'],'tipB':cfg['arms']['B']['tip'],'jointsA':a['joints'],'jointsB':cfg['arms']['B']['joints'],
             'beam_snap':fixtures['beam'],'load_point':parts['cap']['goal'],'force_N':8.,'targets':targets}
    sim.writeCustomTableData(cfg['root'],'SupportLoadProbe',fixture)
    script=sim.createScript(sim.scripttype_simulation,LUA,0,'lua')
    sim.setObjectAlias(script,'SupportLoadFixture');sim.setObjectParent(script,cfg['root'],True)
    initial_beam=sim.getObjectPose(cfg['blocks']['beam'],-1)
    initial_overlap=obb_overlap(sim,pad,cfg['blocks']['beam'])
    dt=sim.getFloatParam(sim.floatparam_simulation_time_step)
    counts={'settle':round(settle_s/dt),'loaded':round(loaded_s/dt)}
    samples=[]
    sim.setStepping(True);sim.startSimulation()
    for stage,count in counts.items():
        if stage=='loaded':sim.callScriptFunction('setLoad',script,True)
        for _ in range(count):
            sim.step()
            row=sim.callScriptFunction('sampleFixture',script);row['stage']=stage;samples.append(row)
    stop(sim)
    return {'fixture':label or case,'support_position':support['tcp_position'],'support_precontact_position':support['precontact_position'],
            'beam_initial_z_offset_m':beam_z_offset,'tcp_z_offset_m':tcp_z_offset,'tcp_y_offset_m':tcp_y_offset,
            'tcp_correction':correction,'initial_pad_beam_obb':initial_overlap,
            'support_targets':targets,'initial_beam_pose':initial_beam,'dt_s':dt,'samples':samples,
            'summary':summarize(samples,initial_beam,dt)}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=23045)
    parser.add_argument('--case',choices=['both','supported','precontact'],default='both')
    parser.add_argument('--height-offset-test',action='store_true')
    args=parser.parse_args()
    scene=ROOT/'scenes'/'10_Normal_Assembly.ttt'
    digest=hashlib.sha256(scene.read_bytes()).hexdigest()
    source,index,support_index,candidate,support=candidate_from_disk()
    if args.height_offset_test:
        previous=json.loads((ROOT/'results'/'normal'/'support_load_probe.json').read_text(encoding='utf-8'))
        old_supported=next(c for c in previous['cases'] if c['fixture']=='supported')
        old_precontact=next(c for c in previous['cases'] if c['fixture']=='precontact')
        support=deepcopy(support);candidate=deepcopy(candidate)
        support['q_contact']=old_supported['support_targets']['A']
        support['q_precontact']=old_precontact['support_targets']['A']
        support['tcp_position']=old_supported['support_position']
        candidate['endpoint_evidence']['goal_approach']=old_supported['support_targets']['B']
    data={'scope':'Initialized support-load fixture, not complete assembly or planner execution',
          'scene':str(scene),'scene_sha256':digest,'shared_scene_saved':False,
          'candidate_source':str(source),'candidate_index':index,'support_candidate_index':support_index,
          'applied_force_N':8.,'load_point':'cap.goal','snap_breaking_enabled':False,
          'force_application':'reset cumulative external force/torque before every dynamic-substep application',
          'unsupported_static_expected_beam_torque_Nm':.22*9.81*.20+8.*.33,
          'cases':[]}
    with Simulator(ROOT,port=args.port,timeout=40) as owner:
        client,sim=owner.connect()
        if args.height_offset_test:
            ik=client.require('simIK')
            data['profile']='beam measured-height offset sensitivity and bounded TCP correction'
            for label,beam_dz,tcp_dz in [('nominal_supported',0.,0.),('beam_down_nominal_tcp',-.0015,0.),
                                          ('beam_down_compensated_tcp',-.0015,-.0015)]:
                result=run_case(sim,scene,candidate,support,'supported',ik=ik,
                                beam_z_offset=beam_dz,tcp_z_offset=tcp_dz,label=label)
                data['cases'].append(result)
                print(json.dumps({'case':label,**result['summary'],'tcp_correction':result['tcp_correction']},ensure_ascii=False,indent=2),flush=True)
        else:
            for case in (['precontact','supported'] if args.case=='both' else [args.case]):
                result=run_case(sim,scene,candidate,support,case);data['cases'].append(result)
                print(json.dumps({'case':case,**result['summary']},ensure_ascii=False,indent=2),flush=True)
    data['shared_scene_unchanged']=digest==hashlib.sha256(scene.read_bytes()).hexdigest()
    if len(data['cases'])==2:
        control,loaded=data['cases']
        tare=control['summary']['settle']['tail_mean_pad_world_z_N']
        data['supported_contact_estimate']={
            'tare_world_z_N_from_precontact':tare,
            'settle_tail_net_normal_magnitude_N':abs(loaded['summary']['settle']['tail_mean_pad_world_z_N']-tare),
            'loaded_tail_net_normal_magnitude_N':abs(loaded['summary']['loaded']['tail_mean_pad_world_z_N']-tare),
            'loaded_peak_torque_reduction_Nm':control['summary']['loaded']['peak_beam_snap_torque_Nm']-loaded['summary']['loaded']['peak_beam_snap_torque_Nm']}
    filename='support_height_offset_probe.json' if args.height_offset_test else 'support_load_probe.json'
    output=ROOT/'results'/'normal'/filename
    output.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':main()
