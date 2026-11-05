"""Reproducible CoppeliaSim design preview. No simulated execution is claimed."""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation
from sim_connection import Simulator
from label_mesh import make_label

ROOT = Path(__file__).resolve().parents[1]
ORANGE, BLUE = [1., .35, .07], [.035, .40, .85]
CREAM, GREEN, GOLD = [.85, .73, .52], [.12, .56, .37], [1., .71, .08]

def look_at(eye, target):
    z = np.array(target, dtype=float) - eye
    z /= np.linalg.norm(z)
    x = np.cross([0., 0., 1.], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return list(eye) + Rotation.from_matrix(np.column_stack([x,y,z])).as_quat().tolist()

def pose(xyz, yaw=0):
    return list(xyz) + Rotation.from_euler('z',yaw).as_quat().tolist()

def text_asset(name, lines, dark=False):
    im = Image.new('RGB',(1500,260),'#112c39' if dark else '#eff4f3')
    d = ImageDraw.Draw(im)
    font = 'C:/Windows/Fonts/msyh.ttc'
    for i, line in enumerate(lines):
        d.text((45,25+i*105),line,font=ImageFont.truetype(font,65 if i==0 else 43),
               fill='#f4f7f5' if dark else '#174b52')
    path=ROOT/'assets'/f'{name}.png'
    im.save(path)
    return path

def build(sim, client, install):
    sim.closeScene()
    root=sim.createDummy(.001)
    sim.setObjectAlias(root,'APEX_MR_Scene3')
    sim.setObjectInt32Param(root,sim.objintparam_visibility_layer,0)
    sim.setArrayParam(sim.arrayparam_background_color1,[.90,.94,.96])
    sim.setArrayParam(sim.arrayparam_background_color2,[.97,.98,.99])
    sim.setArrayParam(sim.arrayparam_ambient_light,[.45,.45,.45])
    solids=[]

    def group(name, xyz=(0,0,0), parent=root):
        h=sim.createDummy(.002)
        sim.setObjectAlias(h,name)
        sim.setObjectParent(h,parent,True)
        sim.setObjectPosition(h,-1,list(xyz))
        sim.setObjectInt32Param(h,sim.objintparam_visibility_layer,0)
        return h

    def box(name,size,xyz,color,parent=root,collision=True,obstacle=False,kind=None):
        h=sim.createPrimitiveShape(kind or sim.primitiveshape_cuboid,list(size),3)
        sim.setObjectAlias(h,name)
        sim.setObjectParent(h,parent,True)
        sim.setObjectPosition(h,-1,list(xyz))
        sim.setShapeColor(h,None,sim.colorcomponent_ambient_diffuse,color)
        sim.setObjectInt32Param(h,sim.shapeintparam_static,1)
        sim.setObjectInt32Param(h,sim.shapeintparam_respondable,0)
        prop=sim.objectspecialproperty_renderable
        if collision: prop |= sim.objectspecialproperty_collidable | sim.objectspecialproperty_measurable
        sim.setObjectSpecialProperty(h,prop)
        if obstacle: solids.append(h)
        return h

    def label(name,lines,xyz,size,vertical=False,dark=False):
        return make_label(sim,root,name,text_asset(name,lines,dark),xyz,size,vertical)

    def brick(name,xyz,size,color):
        p=group(name,xyz)
        box(name+'_Body',size,xyz,color,p)
        nx=max(2,int(size[0]/.028)); ny=max(2,int(size[1]/.028))
        for ix in range(nx):
            for iy in range(ny):
                q=[xyz[0]+(ix-(nx-1)/2)*size[0]/nx,xyz[1]+(iy-(ny-1)/2)*size[1]/ny,xyz[2]+size[2]/2+.002]
                box(f'{name}_Stud_{ix}_{iy}',[.012,.012,.004],q,color,p,False,kind=sim.primitiveshape_cylinder)
        return p

    box('Workbench',[1.90,1.27,.065],[0,0,.7275],[.70,.77,.80],obstacle=True)
    box('StudioBackdrop',[8.,.03,5.],[0,1.4,2.5],[.80,.86,.89],collision=False)
    for x in [-.80,.80]:
        for y in [-.49,.49]: box('Leg',[.065,.065,.67],[x,y,.36],[.19,.24,.29])
    box('Baseplate',[.70,.72,.012],[0,.12,.766],[.25,.33,.38])
    # Left permanent tower and unfinished right tower.
    for n in range(3):
        p=brick('LeftTower_'+str(n+1),[-.22,.20,.812+n*.08],[.12,.14,.08],CREAM)
        solids.append(sim.getObject(':/LeftTower_'+str(n+1)+'_Body',{'proxy':p}))
    for n in range(2):
        p=brick('RightTower_'+str(n+1),[.24,.20,.812+n*.08],[.12,.14,.08],CREAM)
        solids.append(sim.getObject(':/RightTower_'+str(n+1)+'_Body',{'proxy':p}))
    brick('Bridge',[.04,.20,1.032],[.52,.16,.04],GREEN)
    # This hinge is a visible marker. Its load limit belongs to the task model.
    box('ProvisionalHinge',[.025,.19,.026],[-.22,.20,1.025],[.82,.17,.15],collision=False)
    for x in [-.035,.155]:
        box('PocketSideWall',[.012,.33,.18],[x,.265,.862],[.45,.54,.59],obstacle=True)
    box('PocketRearWall',[.20,.012,.18],[.06,.43,.862],[.45,.54,.59],obstacle=True)

    sources={'K1':[-.53,-.36,.806], 'K2':[-.38,-.36,.7785], 'Gate':[.48,-.35,.851]}
    goals={'K1':[.24,.20,.972], 'K2':[.24,.20,1.0685], 'Gate':[.06,.105,.857]}
    brick('K1',sources['K1'],[.12,.14,.08],CREAM)
    brick('K2',sources['K2'],[.10,.18,.025],[.83,.20,.15])
    brick('Gate',sources['Gate'],[.174,.012,.17],[.25,.48,.64])
    for side,x,col in [('A',-.48,ORANGE),('B',.48,BLUE)]:
        box(side+'_SupplyMat',[.36,.25,.006],[x,-.36,.763],col,collision=False)
        box(side+'_Mount',[.23,.24,.022],[(-.70 if side=='A' else .70),.13,.771],[.20,.25,.30])
    box('SharedBuffer',[.18,.17,.008],[0,-.36,.764],[.23,.62,.63])
    label('BufferLabel',['共享交接区','容量 1 件'],[0,-.50,.764],[.25,.043])
    label('ALabel',['A 专属供料','K1 塔顶 / K2 锁桥砖'],[-.48,-.535,.764],[.38,.066])
    label('BLabel',['B 安装与支撑','临时工具 / 收口挡板'],[.48,-.535,.764],[.38,.066])

    park=[.48,.43,.760]
    jack=group('TemporarySupport',park)
    # All geometry starts in tool-local coordinates (then translated as a group).
    def tool_position(offset): return [p+d for p,d in zip(park,offset)]
    box('BallastFoot',[.14,.28,.012],tool_position([0,0,.006]),GOLD,jack)
    box('JackStem',[.045,.045,.20],tool_position([0,0,.112]),GOLD,jack)
    box('RetractingPiston',[.025,.025,.022],tool_position([0,0,.223]),[.58,.63,.66],jack)
    box('SupportShoe',[.10,.22,.012],tool_position([0,.10,.234]),GOLD,jack)
    box('Handle',[.10,.018,.018],tool_position([0,-.02,.148]),[.20,.25,.28],jack)
    label('ToolLabel',['可拆 C 形支撑','配重底座 / 卸载后降低 8 mm'],[.44,-.225,.764],[.40,.069])
    outer=group('OuterGuide')
    for dx in [-.079,.079]: box('OuterLine',[.006,.30,.002],[.06+dx,-.05,.774],[.10,.85,.64],outer,False)
    inner=group('InnerGuide')
    for dx in [-.079,.079]: box('InnerLine',[.006,.30,.002],[.06+dx,.26,.774],[1.,.18,.12],inner,False)
    # Arrow is below the jack footprint and serves only as a withdrawal cue.
    for y in [-.21,-.24,-.27]: box('ExtractionDash',[.016,.014,.002],[.06,y,.774],[.15,.78,.62],outer,False)
    box('Backboard',[1.18,.025,.23],[0,.59,1.38],[.06,.16,.21],collision=False)
    label('Title',['APEX-MR  /  SCENE 03','临时支撑 → 释放机械臂 → 永久连接 → 支撑回收'],[0,.575,1.38],[1.14,.197],True,True)

    ik=client.require('simIK')
    arms={}; robot_cols={}
    for side,x,yaw,col in [('A',-.70,0,ORANGE),('B',.70,math.pi,BLUE)]:
        model=sim.loadModel(str(install/'models/robots/non-mobile/UR5.ttm'))
        sim.setObjectAlias(model,side+'_UR5')
        scripts=sim.getObjectsInTree(model,sim.object_script_type,0)
        if scripts: sim.removeObjects(scripts)
        sim.setObjectParent(model,root,True)
        sim.setObjectPose(model,pose([x,.13,.783],yaw),-1)
        joints=sim.getObjectsInTree(model,sim.object_joint_type,0)
        for i,j in enumerate(joints,1):
            sim.setObjectAlias(j,f'{side}_joint_{i}')
            sim.setJointMode(j,sim.jointmode_kinematic,0)
            sim.setJointInterval(j,True,[-math.pi,2*math.pi])
        for h in sim.getObjectsInTree(model,sim.object_shape_type,0):
            sim.setObjectInt32Param(h,sim.shapeintparam_static,1)
            sim.setObjectInt32Param(h,sim.shapeintparam_respondable,0)
            _,c=sim.getShapeColor(h,None,sim.colorcomponent_ambient_diffuse)
            if c and max(c)>.35: sim.setShapeColor(h,None,sim.colorcomponent_ambient_diffuse,col)
        connection=sim.getObject(':/connection',{'proxy':model})
        palm=box(side+'_Palm',[.05,.084,.023],[0,0,0],[.16,.20,.25],connection)
        sim.setObjectPose(palm,[0,0,.041,0,0,0,1],connection)
        for direction in [-1,1]:
            finger=box(side+'_Finger_'+str(direction),[.018,.009,.060],[0,0,0],[.54,.59,.65],connection)
            sim.setObjectPose(finger,[0,direction*.032,.082,0,0,0,1],connection)
        tip=group(side+'_TCP',parent=connection)
        sim.setObjectPose(tip,[0,0,.112,0,0,0,1],connection)
        target=group(side+'_IK_Target')
        env=ik.createEnvironment(); g=ik.createGroup(env)
        ik.setGroupCalculation(env,g,ik.method_damped_least_squares,.05,150)
        _,mapping,_=ik.addElementFromScene(env,g,model,tip,target,ik.constraint_pose)
        arms[side]=dict(model=model,joints=joints,tip=tip,target=target,env=env,group=g,mapping=mapping,connection=connection)
        c=sim.createCollection(0); sim.addItemToCollection(c,sim.handle_tree,model,0); robot_cols[side]=c

    obstacles=sim.createCollection(0)
    for h in solids: sim.addItemToCollection(obstacles,sim.handle_single,h,0)
    validator=sim.createScript(sim.scripttype_customization,"""sim=require 'sim'
function sysCall_init() end
function firstClear(configs,joints,robot,obstacles)
 for i,q in ipairs(configs) do
  for n,j in ipairs(joints) do sim.setJointPosition(j,q[n]) end
  if sim.checkCollision(robot,obstacles)==0 then return q,i end
 end
 return {},0
end""",0,'lua')
    sim.initScript(validator)
    poses={}; errors={}
    def solve(key,side,xyz,quat,seed=None):
        a=arms[side]; seed=seed or [0,-1.15,1.7,-2.1,-1.57,0]
        sim.setObjectPose(a['target'],list(xyz)+list(quat),-1)
        for j,q in zip(a['joints'],seed): sim.setJointPosition(j,q)
        ik.handleGroup(a['env'],a['group'],{'syncWorlds':True,'allowError':False})
        err=np.linalg.norm(np.array(sim.getObjectPosition(a['tip'],-1))-xyz)
        if err>.001 or sim.checkCollision(robot_cols[side],obstacles)[0]:
            configs=ik.findConfigs(a['env'],a['group'],[a['mapping'][j] for j in a['joints']],{'maxTime':2.,'findMultiple':True,'findAlt':False})
            configs=sorted(configs,key=lambda q:sum((v-s)**2 for v,s in zip(q,seed)))
            q,index=sim.callScriptFunction('firstClear',validator,configs,a['joints'],robot_cols[side],obstacles)
            if not q: raise RuntimeError(f'No obstacle-clear endpoint for {key}: {len(configs)} candidates')
        actual=sim.getObjectPose(a['tip'],-1)
        err=float(np.linalg.norm(np.array(actual[:3])-xyz))
        angle=float((Rotation.from_quat(actual[3:]).inv()*Rotation.from_quat(quat)).magnitude())
        if err>.001 or angle>.02: raise RuntimeError(f'IK endpoint failed: {key}: {err}, {angle}')
        poses[key]=[sim.getJointPosition(j) for j in a['joints']]
        errors[key]={'position_m':err,'orientation_rad':angle,'table_towers_pocket_collision':False}
        print(json.dumps({'endpoint':key,**errors[key]}),flush=True)
    solve('A_home','A',[-.36,-.05,1.18],[1,0,0,0])
    solve('B_home','B',[.36,-.05,1.18],[0,1,0,0])
    solve('A_hold','A',[.06,.14,1.006],Rotation.from_euler('x',-math.pi/2).as_quat(),poses['A_home'])
    solve('A_supply','A',[-.53,-.36,.916],[1,0,0,0],poses['A_home'])
    solve('B_install','B',[.24,.20,1.16],[0,1,0,0],poses['B_home'])
    # Pad is a contact marker at the support endpoint, not a force sensor.
    pad=box('A_SupportPad',[.075,.075,.012],[0,0,0],[.18,.25,.30],arms['A']['connection'],False)
    sim.setObjectPose(pad,[0,0,.112]+Rotation.from_euler('x',math.pi/2).as_quat().tolist(),arms['A']['connection'])
    for a in arms.values(): ik.eraseEnvironment(a['env'])
    sim.removeObjects([validator]); sim.destroyCollection(obstacles)

    overview=look_at([1.40,-2.95,2.55],[0,.02,1.01])
    detail=look_at([.95,-1.45,1.85],[.04,.16,.98])
    cameras={}
    for name,cp in [('OverviewRGB',overview),('DetailRGB',detail)]:
        h=sim.createVisionSensor(1+2+4+128,[1600,1100,0,0],[.025,10,math.radians(48),.01,0,0,0,.91,.94,.97,0,0])
        sim.setObjectAlias(h,name); sim.setObjectParent(h,root,True); sim.setObjectPose(h,cp,-1)
        sim.setObjectInt32Param(h,sim.objintparam_visibility_layer,0); cameras[name]=h
    sim.setObjectPose(sim.getObject('/DefaultCamera'),overview,-1)
    sim.adjustView(0,sim.getObject('/DefaultCamera'),0)

    def state(a,b,support,installed,gate,description):
        obj={'TemporarySupport':support}
        for name in ['K1','K2','Gate']: obj[name]=pose(goals[name] if (gate if name=='Gate' else installed) else sources[name])
        return dict(A=a,B=b,objects=obj,description=description)
    states=[
        state('A_hold','B_home',pose(park),False,False,'初始：A 扶持未闭合悬梁。K1 / K2 在 A 专属供料区。B 可放置临时支撑。'),
        state('A_supply','B_home',pose([.06,-.05,.772]),False,False,'承载确认后的目标状态：外侧支撑接替 A；A 可取料。此快照不表示已测得支撑反力。'),
        state('A_home','B_install',pose([.06,-.05,.772]),True,False,'K1 / K2 已就位。需验证永久连接，再卸载支撑并降低鞋面 8 mm。'),
        state('A_home','B_home',pose(park),True,True,'目标：永久结构完成，支撑已取回工具位，双臂空闲。外侧支撑也允许先封板再抽出。'),
        state('A_home','B_home',pose([.06,.26,.772],math.pi),True,True,'反例：先封板导致内侧支撑被困。合法替代方案是验证永久连接后，先拆支撑再封板。')
    ]
    cfg={'version':1,'stage':'static_design_preview','poses':poses,'states':states,'preset':0,
         'overview_pose':overview,'detail_pose':detail,'ik_endpoints':errors,
         'motion_planner_connected':False,'dynamics_validated':False}
    sim.writeCustomTableData(root,'Scene3',cfg)
    script=sim.createScript(sim.scripttype_customization,(ROOT/'src/scene_ui.lua').read_text(encoding='utf-8'),0,'lua')
    sim.setObjectAlias(script,'SceneControls'); sim.setObjectParent(script,root,True); sim.initScript(script)
    sim.callScriptFunction('initialize',script)
    names=['01_A_Holding','02_Temporary_Support','03_Permanent_Connection','04_Support_Returned','05_Trapped_Inner_Support']
    endpoint_pairs=[]
    for i,name in enumerate(names):
        sim.callScriptFunction('setPreset',script,i)
        collision=bool(sim.checkCollision(robot_cols['A'],robot_cols['B'])[0])
        endpoint_pairs.append({'preset':i,'interarm_collision':collision})
        if collision: raise RuntimeError(f'Inter-arm endpoint collision in preset {i}')
        sim.saveScene(str(ROOT/'scenes'/f'{name}.ttt'))
        for cam in (['OverviewRGB','DetailRGB'] if i in [1,4] else ['OverviewRGB']):
            sim.handleVisionSensor(cameras[cam]); raw,res=sim.getVisionSensorImg(cameras[cam])
            im=Image.fromarray(np.frombuffer(raw,np.uint8).reshape(res[1],res[0],3)[::-1].copy())
            im.save(ROOT/'previews'/f'{name}{"_detail" if cam=="DetailRGB" else ""}.png')
    for c in robot_cols.values(): sim.destroyCollection(c)
    sim.callScriptFunction('setPreset',script,0)
    # A preview build must never overwrite an audited dynamic main delivery.
    preview_scene = ROOT/'scenes'/'00_Scene3_Design_Preview.ttt'
    sim.saveScene(str(preview_scene))
    cfg.update({'endpoint_pairs':endpoint_pairs,'scene':str(preview_scene.relative_to(ROOT)),'snapshots':names,
                'collision_check_scope':'Endpoint checks against table, tower bodies and pocket walls; inter-arm pairs. No continuous paths, self-collision or full tool/part collision certification.',
                'sources':sources,'goals':goals})
    (ROOT/'config/scene_manifest.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf-8')
    print('SAVED DESIGN PREVIEW '+str(preview_scene),flush=True)

if __name__=='__main__':
    import argparse
    argparse.ArgumentParser(description='Build legacy design preview into scenes/00_Scene3_Design_Preview.ttt; never overwrite the main scene.').parse_args()
    for folder in ['assets','config','previews','scenes','results']: (ROOT/folder).mkdir(exist_ok=True)
    with Simulator(ROOT,port=23053,timeout=40) as owner:
        client,sim=owner.connect()
        build(sim,client,owner.install)
