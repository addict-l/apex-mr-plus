"""Isolate runtime respondable toggling from support endpoint geometry."""
from pathlib import Path
import hashlib,json,math,statistics,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from sim_connection import Simulator
from normal.scene import resolve
from normal.probe_support_load import make_bond,LUA,stop

def world_z_extent(sim,h):
    low=[sim.getObjectFloatParam(h,getattr(sim,'objfloatparam_objbbox_min_'+axis)) for axis in 'xyz']
    high=[sim.getObjectFloatParam(h,getattr(sim,'objfloatparam_objbbox_max_'+axis)) for axis in 'xyz']
    m=sim.getObjectMatrix(h,-1)
    center=m[11]+sum(m[8+i]*(low[i]+high[i])/2 for i in range(3))
    half=sum(abs(m[8+i])*(high[i]-low[i])/2 for i in range(3))
    return [center-half,center+half]

def main():
    scene=ROOT/'scenes'/'10_Normal_Assembly.ttt';digest=hashlib.sha256(scene.read_bytes()).hexdigest()
    old=json.loads((ROOT/'results'/'normal'/'support_load_probe.json').read_text(encoding='utf-8'))
    support=next(c for c in old['cases'] if c['fixture']=='supported')
    qA=support['support_targets']['A'];qB=support['support_targets']['B']
    data={'scope':'runtime respondable toggle fixture; no assembly execution',
          'scene_sha256':digest,'beam_z_offset_m':-.0015,'tcp_z_offset_m':-.0015,
          'shared_scene_saved':False,'phases':[]}
    with Simulator(ROOT,port=23045,timeout=40) as owner:
        client,sim=owner.connect();sim.loadScene(str(scene));cfg=resolve(sim);parts={p['id']:p for p in cfg['parts']}
        base=sim.getObject('/APEX_MR_Plus/CastleBaseplate');snaps={}
        for name in ['L1','R1','L2','R2','L3','beam']:
            p=parts[name];h=cfg['blocks'][name];xyz=list(p['goal'])
            if name=='beam':xyz[2]-=.0015
            sim.setObjectPose(h,xyz+[0,0,0,1],-1)
            parent=base if not p['dependencies'] else cfg['blocks'][p['dependencies'][0]]
            anchor=list(p['goal']);anchor[2]-=p['size'][2]/2
            if name=='beam':
                parent=cfg['blocks']['L3'];anchor=list(parts['L3']['goal']);anchor[2]+=.04
            snaps[name]=make_bond(sim,'ToggleProbeSnap_'+name,parent,h,anchor)
        targets={'A':list(qA),'B':list(qB)}
        for side,a in cfg['arms'].items():
            for j,q in zip(a['joints'],targets[side]):sim.setJointPosition(j,q);sim.setJointTargetPosition(j,q)
        a=cfg['arms']['A'];pose=sim.getObjectPose(a['tip'],-1);desired=list(pose);desired[2]-=.0015
        sim.setObjectPose(a['target'],desired,-1);ik=client.require('simIK');env=ik.createEnvironment();group=ik.createGroup(env)
        ik.setGroupCalculation(env,group,ik.method_damped_least_squares,.05,150)
        ik.addElementFromScene(env,group,a['model'],a['tip'],a['target'],ik.constraint_pose)
        ik.handleGroup(env,group,{'syncWorlds':True,'allowError':False});ik.eraseEnvironment(env)
        targets['A']=[sim.getJointPosition(j) for j in a['joints']]
        for j,q in zip(a['joints'],targets['A']):sim.setJointTargetPosition(j,q)
        pad=a['support_pad'];sim.setObjectInt32Param(pad,sim.shapeintparam_respondable,0)
        fixture={'beam':cfg['blocks']['beam'],'pad':pad,'pad_sensor':a['support_sensor'],'tool_sensor':a['sensor'],
          'tipA':a['tip'],'tipB':cfg['arms']['B']['tip'],'jointsA':a['joints'],'jointsB':cfg['arms']['B']['joints'],
          'beam_snap':snaps['beam'],'load_point':parts['cap']['goal'],'force_N':8.,'targets':targets}
        sim.writeCustomTableData(cfg['root'],'SupportLoadProbe',fixture)
        script=sim.createScript(sim.scripttype_simulation,LUA,0,'lua');sim.setObjectAlias(script,'SupportToggleProbe');sim.setObjectParent(script,cfg['root'],True)
        data['initial_fk']={'tcp_A':sim.getObjectPose(a['tip'],-1),'desired_tcp_A':desired,
             'pad_z_extent':world_z_extent(sim,pad),'beam_z_extent':world_z_extent(sim,cfg['blocks']['beam'])}
        sim.setStepping(True);sim.startSimulation()
        for phase,count in [('respondable_off',25),('flag_on_without_reset',20),('flag_on_after_reset',25),('after_reset_loaded_8N',25)]:
            if phase=='flag_on_without_reset':sim.setObjectInt32Param(pad,sim.shapeintparam_respondable,1)
            if phase=='flag_on_after_reset':sim.resetDynamicObject(pad)
            if phase=='after_reset_loaded_8N':sim.callScriptFunction('setLoad',script,True)
            rows=[]
            for _ in range(count):
                sim.step();row=sim.callScriptFunction('sampleFixture',script)
                row['pad_z_extent']=world_z_extent(sim,pad);row['beam_z_extent']=world_z_extent(sim,cfg['blocks']['beam'])
                row['z_penetration_bound_m']=row['pad_z_extent'][1]-row['beam_z_extent'][0]
                rows.append(row)
            valid=[r for r in rows if r['pad']['valid'] and r['beam_snap']['valid']]
            tail=valid[-5:]
            summary={'valid_samples':len(valid),
              'pad_z_force_peak_N':max(abs(r['pad']['force_world_z_N']) for r in valid),
              'pad_z_force_tail_mean_N':statistics.fmean(r['pad']['force_world_z_N'] for r in tail),
              'beam_snap_torque_peak_Nm':max(r['beam_snap']['torque_norm_Nm'] for r in valid),
              'beam_snap_torque_tail_mean_Nm':statistics.fmean(r['beam_snap']['torque_norm_Nm'] for r in tail),
              'tail_penetration_bound_m':statistics.fmean(r['z_penetration_bound_m'] for r in tail),
              'tail_tcp_A':tail[-1]['tcp_A']}
            data['phases'].append({'phase':phase,'summary':summary,'samples':rows})
            print(json.dumps({'phase':phase,**summary},ensure_ascii=False,indent=2),flush=True)
        stop(sim)
    tare=data['phases'][0]['summary']['pad_z_force_tail_mean_N']
    for phase in data['phases']:
        phase['summary']['net_contact_force_tail_magnitude_N']=abs(phase['summary']['pad_z_force_tail_mean_N']-tare)
    data['shared_scene_unchanged']=digest==hashlib.sha256(scene.read_bytes()).hexdigest()
    (ROOT/'results'/'normal'/'support_respondable_toggle_probe.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':main()
