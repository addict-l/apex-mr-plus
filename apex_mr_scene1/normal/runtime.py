"""Embed and execute a central TPG dispatcher on real rigid-body dynamics."""
from __future__ import annotations
import json,math,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1]


def install_controller(sim,cfg,tpg,scene_name='scenes/11_Normal_APEX_MR_Planned.ttt'):
    from .labels import apply_normal_labels
    root=cfg['root'];cfg['program']=tpg
    apply_normal_labels(sim,root)
    sim.writeCustomTableData(root,'NormalAssembly',cfg)
    script=sim.createScript(sim.scripttype_simulation,(Path(__file__).with_name('controller.lua')).read_text(encoding='utf-8'),0,'lua')
    sim.setObjectAlias(script,'CentralTPGScheduler');sim.setObjectParent(script,root,True)
    sim.saveScene(str(ROOT/scene_name))
    return script


def capture(sim,camera,path,caption=None):
    sim.handleVisionSensor(camera);raw,res=sim.getVisionSensorImg(camera)
    rgb=np.frombuffer(raw,np.uint8).reshape(res[1],res[0],3)[::-1].copy()
    im=Image.fromarray(rgb)
    if caption:
        draw=ImageDraw.Draw(im);font=ImageFont.truetype(r'C:\Windows\Fonts\msyh.ttc',23)
        draw.rectangle([0,0,im.width,46],fill=(20,34,52));draw.text((16,8),caption,font=font,fill='white')
    im.save(path);return np.asarray(im)


def execute(sim,client,cfg,outdir,timeout_sim_s=1800,timeout_wall_s=1200,video=True):
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    scheduler=sim.getObject('/APEX_MR_Plus/CentralTPGScheduler')
    camera=sim.getObject('/APEX_MR_Plus/OverviewRGB')
    writer=None
    if video:
        import cv2
        writer=cv2.VideoWriter(str(outdir/'normal_execution.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),8,(1600,1100))
    initial_image=capture(sim,camera,outdir/'initial_dynamic_scene.png','正常模式：ILP → RRT-Connect → TPG → 动力学执行')
    wall=time.monotonic();last_status=-10;frame_number=0;support_frame=False
    sim.setStepping(False);sim.startSimulation()
    try:
        while True:
            time.sleep(.18)
            state=sim.callScriptFunction('getState',scheduler)
            t=state['time']
            if t-last_status>=5 or state.get('failure') or state.get('done'):
                print(json.dumps(state,ensure_ascii=False),flush=True);last_status=t
            if not support_frame and state.get('peak_support_normal_N',0)>.4 and 'place' in str(state.get('active_B','')):
                capture(sim,camera,outdir/'supporting_insertion.png',
                        f"实际动力学：A 臂支撑，B 臂安装 | t={t:.2f}s")
                support_frame=True
            if writer is not None:
                import cv2
                frame=capture(sim,camera,outdir/'latest_execution.png',f"正常动力学执行 | t={t:.2f}s | 节点 {state['completed']}/{state['total']}")
                writer.write(cv2.cvtColor(frame,cv2.COLOR_RGB2BGR));frame_number+=1
            if state.get('done') or state.get('failure'):break
            if t>timeout_sim_s or time.monotonic()-wall>timeout_wall_s:raise TimeoutError('Normal dynamic execution budget exceeded')
        result=sim.callScriptFunction('getResult',scheduler)
        result['wall_time_s']=time.monotonic()-wall;result['video_frames']=frame_number
        errors={};angles={}
        for p in cfg['parts']:
            pose=result['part_poses'][p['id']]
            errors[p['id']]=float(np.linalg.norm(np.array(pose[:3])-p['goal']))
            angles[p['id']]=2*math.acos(min(1.,abs(float(pose[6]))))
        result['goal_position_errors_m']=errors
        result['goal_orientation_errors_rad']=angles
        support_needed=any((p['mass']*9.81+p.get('insertion_force',0))*p.get('lever_arm',0)>p.get('connection_moment_limit',1.2) for p in cfg['parts'])
        result['success']=bool(result.get('done') and not result.get('failure') and max(errors.values())<.009 and max(angles.values())<.04
                               and len(result.get('snap_dynamic',{}))==len(cfg['parts'])
                               and all(result['snap_dynamic'].values()) and (not support_needed or result.get('peak_support_normal_N',0)>.4))
        capture(sim,camera,outdir/('completed_dynamic_scene.png' if result['success'] else 'failed_dynamic_scene.png'),
                '正常装配完成（实际动力学）' if result['success'] else '动力学执行诊断：'+str(result.get('failure')))
        (outdir/'execution.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        return result
    finally:
        if writer is not None:writer.release()
        sim.stopSimulation()
