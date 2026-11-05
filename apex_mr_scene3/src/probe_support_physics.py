"""Paired physical counterfactual: the same bridge with/without A-pad contact.

The saved source is loaded fresh for each case and is never overwritten.
Both arms retain their commanded initial joint states. Only the A support
pad's physical collision response is changed for the no-support case.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sim_connection import Simulator
from apex.scene import resolve


def run_case(client,sim,scene,enabled,duration):
    sim.loadScene(str(scene))
    cfg=resolve(sim)
    pad=cfg['arms']['A']['support_pad']
    sensor=cfg['arms']['A']['support_sensor']
    sim.setObjectInt32Param(pad,sim.shapeintparam_respondable,int(enabled))
    initial=sim.getObjectPose(cfg['blocks']['Bridge'],sim.handle_world)
    pad_weight=sim.getShapeMass(pad)*9.81
    client.setStepping(True)
    sim.startSimulation()
    rows=[]
    try:
        beginning=sim.getSimulationTime()
        for step in range(2000):
            client.step()
            stamp=sim.getSimulationTime()
            status,force,torque=sim.readForceSensor(sensor)
            matrix=sim.getObjectMatrix(sensor,sim.handle_world)
            fz=sum(matrix[8+i]*force[i] for i in range(3)) if status & 1 else None
            bridge=sim.getObjectPose(cfg['blocks']['Bridge'],sim.handle_world)
            rows.append({'time_s':stamp-beginning,'A_pad_force_status':status,
                         'A_pad_force_local_N':force,'A_pad_torque_local_Nm':torque,
                         'A_pad_force_world_z_N':fz,
                         'A_pad_load_above_selfweight_N':max(0.,abs(fz)-pad_weight) if fz is not None else None,
                         'hinge_angle_rad':sim.getJointPosition(cfg['bridge_hinge']),
                         'hinge_torque_Nm':sim.getJointForce(cfg['bridge_hinge']),
                         'bridge_pose':bridge,
                         'bridge_position_error_m':math.sqrt(sum((bridge[i]-initial[i])**2 for i in range(3))),
                         'bridge_vertical_drop_m':initial[2]-bridge[2]})
            if stamp-beginning>=duration: break
        else:
            raise RuntimeError('Probe exceeded its maximum stepping count')
        dynamic={'Bridge':sim.isDynamicallyEnabled(cfg['blocks']['Bridge']),
                 'BridgeHinge':sim.isDynamicallyEnabled(cfg['bridge_hinge']),
                 'A_PadForceSensor':sim.isDynamicallyEnabled(sensor),
                 'A_SupportShoe':sim.isDynamicallyEnabled(pad)}
        for side in ('A','B'):
            for i,joint in enumerate(cfg['arms'][side]['joints'],1):
                dynamic[f'{side}_joint_{i}']=sim.isDynamicallyEnabled(joint)
        tail=[row for row in rows if row['time_s']>=rows[-1]['time_s']-.3]
        loads=[row['A_pad_load_above_selfweight_N'] for row in tail if row['A_pad_load_above_selfweight_N'] is not None]
        return {'A_pad_respondable':enabled,'initial_bridge_pose':initial,
                'pad_selfweight_N':pad_weight,'samples':rows,'dynamic_enabled':dynamic,
                'final_bridge_drop_m':rows[-1]['bridge_vertical_drop_m'],
                'final_bridge_position_error_m':rows[-1]['bridge_position_error_m'],
                'mean_final_A_load_N':sum(loads)/len(loads) if loads else None,
                'all_final_force_samples_valid':len(loads)==len(tail)}
    finally:
        sim.stopSimulation()
        # Stop completes asynchronously; bounded stepping avoids mutating or
        # replacing a still-running scene for the next counterfactual.
        for _ in range(100):
            if sim.getSimulationState()==sim.simulation_stopped: break
            client.step()
        else:
            raise RuntimeError('Simulator did not stop after the physical probe')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--port',type=int,default=23063)
    parser.add_argument('--duration',type=float,default=1.6)
    args=parser.parse_args()
    if not math.isfinite(args.duration) or args.duration<.5:
        parser.error('--duration must be finite and at least 0.5 seconds')
    scene=ROOT/'scenes/10_Scene3_Dynamic.ttt'
    output=ROOT/'results/apex/initial_support_physics.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    report={'kind':'paired_initial_support_dynamics_probe','scene':str(scene),
            'scene_sha256':hashlib.sha256(scene.read_bytes()).hexdigest(),
            'source_scene_modified':False,'success':False,
            'scope':'Initial hold and no-support counterfactual only; not complete assembly execution.',
            'method':'Fresh scene per case; commanded initial robot poses unchanged; A pad response enabled/disabled.',
            'cases':{}}
    try:
        with Simulator(ROOT,port=args.port,timeout=40) as owner:
            client,sim=owner.connect()
            for enabled,name in [(False,'without_A_support'),(True,'with_A_support')]:
                report['cases'][name]=run_case(client,sim,scene,enabled,args.duration)
                case=report['cases'][name]
                print(json.dumps({'case':name,'bridge_drop_m':case['final_bridge_drop_m'],
                                  'A_load_N':case['mean_final_A_load_N']}),flush=True)
        held=report['cases']['with_A_support'];free=report['cases']['without_A_support']
        report['checks']={
            'held_bridge_position_error_below_3mm':held['final_bridge_position_error_m']<.003,
            'held_A_load_above_1N':held['mean_final_A_load_N'] is not None and held['mean_final_A_load_N']>1.,
            'unheld_bridge_drops_over_20mm':free['final_bridge_drop_m']>.020,
            'unheld_A_load_below_0p1N':free['mean_final_A_load_N'] is not None and free['mean_final_A_load_N']<.1,
            'valid_final_force_samples':all(case['all_final_force_samples_valid'] for case in [held,free]),
            'all_required_bodies_joints_dynamic':all(all(case['dynamic_enabled'].values()) for case in [held,free])}
        report['success']=all(report['checks'].values())
    except Exception:
        report['error']=traceback.format_exc()
        raise
    finally:
        output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'output':str(output),'success':report['success'],'checks':report['checks']}),flush=True)
    if not report['success']: raise SystemExit(1)


if __name__=='__main__': main()
