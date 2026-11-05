"""Dynamic Scene 3 with a hinged bridge and a transportable instrumented jack."""
from pathlib import Path
import json,math,sys
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from sim_connection import Simulator

def mass_inertia(sim,h,size,mass):
    x,y,z=size
    sim.setShapeMass(h,mass)
    sim.setShapeInertia(h,[mass*(y*y+z*z)/12,0,0,0,mass*(x*x+z*z)/12,0,0,0,mass*(x*x+y*y)/12],[1,0,0,0,0,1,0,0,0,0,1,0])

def resolve(sim):
    root=sim.getObject('/APEX_MR_Scene3')
    cfg=sim.readCustomTableData(root,'Scene3APEX'); cfg['root']=root
    runtime_file=ROOT/'config/apex_runtime.json'
    if runtime_file.exists():cfg.setdefault('runtime',{}).update(json.loads(runtime_file.read_text(encoding='utf-8')))
    lookup={sim.getObjectAlias(h):h for h in sim.getObjectsInTree(root,sim.handle_all,0)}
    cfg['blocks']={p['id']:lookup[p['id']] for p in cfg['parts']}
    cfg['obstacles']=[lookup[n] for n in cfg['obstacle_aliases']]
    cfg['table']=lookup['Workbench'];cfg['camera']=lookup['OverviewRGB']
    cfg['articulations']={'SupportLift':lookup['SupportLift']}
    cfg['support_sensor']=lookup['SupportLoadSensor'];cfg['support_pad']=lookup['SupportContactPad']
    cfg['baseplate']=lookup['Baseplate'];cfg['right_tower']=lookup['RightTower_2_Body']
    cfg['bridge_hinge']=lookup['BridgeHinge']
    for side,a in cfg['arms'].items():
        a['model']=lookup[side+'_UR5'];a['joints']=[lookup[f'{side}_joint_{i}'] for i in range(1,7)]
        for key,suffix in [('tip','TCP'),('target','IK_Target'),('connection','connection'),('palm','GripperPalm'),('sensor','ToolForceSensor'),('support_sensor','PadForceSensor'),('support_pad','SupportShoe')]:
            if key=='connection': a[key]=sim.getObject(':/connection',{'proxy':a['model']})
            else:a[key]=lookup[side+'_'+suffix]
        a['shapes']=sim.getObjectsInTree(a['model'],sim.object_shape_type,0)
        a['fingers']=[h for h in a['shapes'] if sim.getObjectAlias(h).startswith(side+'_Finger')]
    return cfg

def build(sim,client,install):
    sim.loadScene(str(ROOT/'backups/design_preview/apex_mr_scene3.ttt'))
    root=sim.getObject('/APEX_MR_Scene3')
    preview=sim.readCustomTableData(root,'Scene3')
    def find(alias):
        for h in sim.getObjectsInTree(root,sim.handle_all,0):
            if sim.getObjectAlias(h)==alias:return h
        raise KeyError(alias)
    def remove(alias):
        h=find(alias);sim.removeObjects(sim.getObjectsInTree(h,sim.handle_all,0))
    for name in ['SceneControls','A_UR5','B_UR5','A_IK_Target','B_IK_Target','OuterGuide','InnerGuide']:remove(name)
    # Place the hinge at the tower's right edge, leaving no bearing surface
    # ahead of the pivot that could accidentally prop the cantilever up.
    for layer in range(1,4):
        h=find('LeftTower_'+str(layer));p=sim.getObjectPosition(h,-1);p[0]=-.28
        sim.setObjectPosition(h,-1,p)
    def shape(name,size,pos,color,static=True,parent=root,respondable=True):
        h=sim.createPrimitiveShape(sim.primitiveshape_cuboid,list(size),3)
        sim.setObjectAlias(h,name);sim.setObjectParent(h,parent,True);sim.setObjectPosition(h,-1,list(pos))
        sim.setShapeColor(h,None,sim.colorcomponent_ambient_diffuse,color)
        sim.setObjectInt32Param(h,sim.shapeintparam_static,int(static))
        sim.setObjectInt32Param(h,sim.shapeintparam_respondable,int(respondable))
        sim.setObjectSpecialProperty(h,sim.objectspecialproperty_renderable|sim.objectspecialproperty_collidable|sim.objectspecialproperty_measurable)
        return h
    # Make all workcell bearing surfaces physically respondable. Decorative studs stay visual.
    obstacles=[]
    for h in sim.getObjectsInTree(root,sim.object_shape_type,0):
        name=sim.getObjectAlias(h)
        if name in ['Workbench','Baseplate','SharedBuffer','A_SupplyMat','B_SupplyMat'] or name.startswith('Pocket') or name.startswith(('LeftTower_','RightTower_')) and name.endswith('_Body'):
            sim.setObjectInt32Param(h,sim.shapeintparam_respondable,1)
            sim.setObjectSpecialProperty(h,sim.objectspecialproperty_renderable|sim.objectspecialproperty_collidable|sim.objectspecialproperty_measurable)
            if name!='Workbench':obstacles.append(h)
    # Unique aliases for repeated walls are necessary for portable scene resolution.
    for n,h in enumerate([h for h in obstacles if sim.getObjectAlias(h)=='PocketSideWall']):sim.setObjectAlias(h,'PocketSideWall_'+str(n))
    specs=[('K1',[.12,.14,.076],.18,[.24,.20,.970]),('K2',[.10,.18,.08],.14,[.24,.20,1.096]),
           ('Gate',[.174,.012,.17],.12,[.06,.105,.857]),('Bridge',[.52,.16,.04],.45,[.04,.20,1.032])]
    parts=[];blocks={}
    for name,size,mass,goal in specs:
        group=find(name);body=find(name+'_Body')
        children=sim.getObjectsInTree(group,sim.handle_all,0)
        sim.setObjectParent(body,root,True)
        if name=='K1':
            # A 76mm rigid body plus 4mm visual studs leaves insertion
            # clearance below the supported beam. An 80mm solid cube jammed
            # against the actual sagging beam and produced 269N contact.
            sim.scaleObject(body,1,1,.076/.08,0)
            p=sim.getObjectPosition(body,-1);p[2]=.804;sim.setObjectPosition(body,-1,p)
        if name=='K2':
            sim.scaleObject(body,1,1,.08/.025,0)
            p=sim.getObjectPosition(body,-1);p[2]=.806;sim.setObjectPosition(body,-1,p)
        for h in children:
            if h not in [group,body]:
                sim.setObjectParent(h,body,True);sim.setObjectInt32Param(h,sim.shapeintparam_respondable,0)
                sim.setObjectSpecialProperty(h,sim.objectspecialproperty_renderable)
                if name=='K2':
                    p=sim.getObjectPosition(h,body);p[2]=.042;sim.setObjectPosition(h,body,p)
                if name=='K1':
                    p=sim.getObjectPosition(h,body);p[2]=.040;sim.setObjectPosition(h,body,p)
        sim.removeObjects([group]);sim.setObjectAlias(body,name)
        sim.setObjectInt32Param(body,sim.shapeintparam_static,0);sim.setObjectInt32Param(body,sim.shapeintparam_respondable,1)
        mass_inertia(sim,body,size,mass);blocks[name]=body
        source=sim.getObjectPosition(body,-1)
        parts.append(dict(id=name,size=size,mass=mass,source=source,goal=goal,dependencies=[],insertion_force=8. if name=='K2' else 0.))
    # A real passive revolute hinge: the bridge cannot remain cantilevered without support.
    hinge=sim.createJoint(sim.joint_revolute_subtype,sim.jointmode_dynamic,0)
    sim.setObjectAlias(hinge,'BridgeHinge');sim.setObjectParent(hinge,find('LeftTower_3_Body'),True)
    sim.setObjectPose(hinge,[-.22,.20,1.012]+Rotation.from_euler('x',math.pi/2).as_quat().tolist(),-1)
    sim.setJointInterval(hinge,False,[-.7,1.4])
    sim.setObjectInt32Param(hinge,sim.jointintparam_dynctrlmode,sim.jointdynctrl_velocity)
    sim.setObjectInt32Param(hinge,sim.jointintparam_velocity_lock,0)
    sim.setJointTargetVelocity(hinge,0.);sim.setJointTargetForce(hinge,.35)
    sim.setObjectInt32Param(hinge,sim.objintparam_visibility_layer,0)
    sim.setObjectParent(blocks['Bridge'],hinge,True)
    # Transportable compound ballast/column/handle; independently driven/load-sensed shoe.
    oldtool=find('TemporarySupport');foot=find('BallastFoot');stem=find('JackStem');handle=find('Handle')
    for h,size,mass in [(foot,[.14,.28,.012],2.),(stem,[.045,.045,.20],.22),(handle,[.10,.018,.018],.03)]:
        mass_inertia(sim,h,size,mass);sim.setObjectInt32Param(h,sim.shapeintparam_static,0)
        sim.setObjectInt32Param(h,sim.shapeintparam_respondable,1);sim.setObjectParent(h,root,True)
    tool=sim.groupShapes([foot,stem,handle])
    sim.setObjectAlias(tool,'TemporarySupportBase')
    # Keep the root frame at the old design foot-bottom reference.
    sim.relocateShapeFrame(tool,[.48,.43,.760,0,0,0,1])
    piston=find('RetractingPiston');pad=find('SupportShoe')
    sim.setObjectParent(piston,root,True);sim.setObjectParent(pad,root,True)
    sim.removeObjects([oldtool]);sim.setObjectAlias(tool,'TemporarySupport');blocks['TemporarySupport']=tool
    lift=sim.createJoint(sim.joint_prismatic_subtype,sim.jointmode_dynamic,0)
    sim.setObjectAlias(lift,'SupportLift');sim.setObjectParent(lift,tool,True)
    sim.setObjectPose(lift,[.48,.43,.975,0,0,0,1],-1)
    sim.setJointInterval(lift,False,[0,.012]);sim.setJointPosition(lift,0.)
    sim.setObjectInt32Param(lift,sim.jointintparam_dynctrlmode,sim.jointdynctrl_position)
    sim.setJointTargetPosition(lift,0.);sim.setJointTargetForce(lift,80.)
    sim.setObjectInt32Param(lift,sim.objintparam_visibility_layer,0)
    sim.setObjectParent(piston,lift,True);sim.setObjectPosition(piston,-1,[.48,.43,.975])
    # Internal carrier overlaps the stem and shoe by design. Its collision
    # response creates a false unloaded preload; the external foot and shoe
    # supply physical contact while this carrier retains mass and dynamics.
    sim.setObjectInt32Param(piston,sim.shapeintparam_static,0);sim.setObjectInt32Param(piston,sim.shapeintparam_respondable,0)
    mass_inertia(sim,piston,[.025,.025,.022],.03)
    fs=sim.createForceSensor(0,[0,1,5,0,0],[.008,1000,100,0,0])
    sim.setObjectAlias(fs,'SupportLoadSensor');sim.setObjectParent(fs,piston,True)
    sim.setObjectPose(fs,[.48,.53,.986,0,0,0,1],-1);sim.setObjectInt32Param(fs,sim.objintparam_visibility_layer,0)
    sim.setObjectParent(pad,fs,True);sim.setObjectAlias(pad,'SupportContactPad')
    sim.setObjectPosition(pad,-1,[.48,.53,.986])
    sim.setObjectInt32Param(pad,sim.shapeintparam_static,0);sim.setObjectInt32Param(pad,sim.shapeintparam_respondable,1)
    mass_inertia(sim,pad,[.10,.22,.012],.08)
    parts.append(dict(id='TemporarySupport',size=[.14,.35,.24],mass=2.36,source=[.48,.43,.760],goal=[.48,.43,.760],dependencies=[]))
    homes={side+'_home':preview['poses'][side+'_home'] for side in ['A','B']}
    qualified=json.loads((ROOT/'config/dynamic_homes.json').read_text(encoding='utf-8'))
    homes['A_home']=qualified['A_hold'];homes['B_home']=qualified['B_free']
    # BUILD_ARMS is generated below from the proven dynamic UR5 baseline.
    arms = build_arms(sim,install,root,shape,homes)
    # A's supporting pad is horizontal for the forward-facing initial TCP.
    asensor=arms['A']['support_sensor'];apad=arms['A']['support_pad']
    sim.setObjectPose(asensor,[0,-.006,.071]+Rotation.from_euler('x',math.pi/2).as_quat().tolist(),arms['A']['palm'])
    sim.setObjectPose(apad,[0,0,0,0,0,0,1],asensor)
    sim.setObjectInt32Param(apad,sim.shapeintparam_respondable,1);sim.setObjectInt32Param(apad,sim.objintparam_visibility_layer,1)
    # Lower A TCP enough for fingers to stay below the beam; only the shoe contacts it.
    ik=client.require('simIK');env=ik.createEnvironment();g=ik.createGroup(env)
    ik.setGroupCalculation(env,g,ik.method_damped_least_squares,.02,200)
    a=arms['A'];ik.addElementFromScene(env,g,a['model'],a['tip'],a['target'],ik.constraint_pose)
    sim.setObjectPose(a['target'],[-.08,.14,1.002]+Rotation.from_euler('x',-math.pi/2).as_quat().tolist(),-1)
    ik.handleGroup(env,g,{'syncWorlds':True,'allowError':False});ik.eraseEnvironment(env)
    for side,a in arms.items():
        a['home_q']=[sim.getJointPosition(j) for j in a['joints']]
        for j,q in zip(a['joints'],a['home_q']):sim.setJointTargetPosition(j,q)
    sim.setInt32Param(sim.intparam_dynamic_engine,sim.physics_bullet)
    sim.setFloatProperty(sim.handle_scene,'dynamicsStepSize',.002);sim.setIntProperty(sim.handle_scene,'bullet.iterations',150)
    sim.setVector3Property(sim.handle_scene,'gravity',[0,0,-9.81]);sim.setFloatParam(sim.floatparam_simulation_time_step,.02)
    sim.setBoolParam(sim.boolparam_realtime_simulation,False)
    for h in [find('Workbench'),*obstacles,*blocks.values(),pad,apad]:
        sim.setFloatProperty(h,'bullet.friction',.8);sim.setFloatProperty(h,'bullet.frictionOld',.8)
        sim.setFloatProperty(h,'bullet.restitution',0.);sim.setFloatProperty(h,'bullet.linearDamping',.04);sim.setFloatProperty(h,'bullet.angularDamping',.04)
    cfg=dict(root=root,root_path='/APEX_MR_Scene3',parts=parts,blocks=blocks,arms=arms,obstacles=obstacles,
             obstacle_aliases=[sim.getObjectAlias(h) for h in obstacles],table=find('Workbench'),camera=find('OverviewRGB'),
             initial={p['id']:sim.getObjectPose(blocks[p['id']],-1) for p in parts},goal={p['id']:p['goal'] for p in parts},sizes={p['id']:p['size'] for p in parts},
             articulations={'SupportLift':lift},initial_articulations={'SupportLift':0.},support_sensor=fs,support_pad=pad,
             baseplate=find('Baseplate'),right_tower=find('RightTower_2_Body'),bridge_hinge=hinge,
             support_surface_handles=[h for h in obstacles if not sim.getObjectAlias(h).startswith('Pocket')],
             initial_support_contacts={'A':[['A_SupportShoe','Bridge']]},dt=.02,program={},
             runtime={'force_min_N':1.,'loaded_dwell_s':.3,'unload_force_max_N':.8,'pose_tolerance_m':.009},
             physics={'engine':'Bullet','initial_bridge':'passive pitch hinge with 0.35Nm friction brake and actual A-pad contact',
                      'grasp':'ideal dynamic force-sensor constraints','jack':'compound ballast, dynamic prismatic lift and load sensor',
                      'connections':'ideal rigid snap at measured pose; no LEGO deformation'})
    cfg['free_q']={'A':qualified['A_free'],'B':qualified['B_free']}
    runtime_file=ROOT/'config/apex_runtime.json'
    if runtime_file.exists():cfg['runtime'].update(json.loads(runtime_file.read_text(encoding='utf-8')))
    sim.writeCustomTableData(root,'Scene3APEX',cfg)
    sim.saveScene(str(ROOT/'scenes/10_Scene3_Dynamic.ttt'))
    report=[]
    for h in sim.getObjectsInTree(root,sim.object_shape_type,0):
        if sim.getObjectInt32Param(h,sim.shapeintparam_static)==0:
            inertia,frame=sim.getShapeInertia(h);report.append({'alias':sim.getObjectAlias(h),'mass_kg':sim.getShapeMass(h),'inertia':inertia,'frame':frame})
    (ROOT/'results/apex/rigid_bodies.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (ROOT/'results/apex/problem.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    print(json.dumps({'scene':'10_Scene3_Dynamic.ttt','dynamic_bodies':len(report)}),flush=True)
    return cfg

def build_arms(sim,install,root,shape,homes):
    arms={}
    for side,x,yaw,color in [('A',-.70,0,[1.,.34,.055]),('B',.70,math.pi,[.025,.40,.86])]:
        model=sim.loadModel(str(install/'models/robots/non-mobile/UR5.ttm'))
        sim.setObjectAlias(model,side+'_UR5');sim.setObjectParent(model,root,True)
        sim.setObjectInt32Param(model,sim.shapeintparam_static,1)
        for s in sim.getObjectsInTree(model,sim.object_script_type,0):sim.removeObjects([s])
        sim.setObjectPose(model,[x,.13,.783]+Rotation.from_euler('z',yaw).as_quat().tolist(),-1)
        joints=sim.getObjectsInTree(model,sim.object_joint_type,0)
        for i,j in enumerate(joints,1):
            sim.setObjectAlias(j,f'{side}_joint_{i}')
            sim.setJointMode(j,sim.jointmode_dynamic,0)
            sim.setJointInterval(j,True,[-math.pi,2*math.pi])
            sim.setObjectInt32Param(j,sim.jointintparam_dynctrlmode,sim.jointdynctrl_position)
            sim.setJointPosition(j,homes[side+'_home'][i-1])
            sim.setJointTargetPosition(j,homes[side+'_home'][i-1])
            sim.setJointTargetForce(j,200. if i<4 else 50.)
        for h in sim.getObjectsInTree(model,sim.object_shape_type,0):
            if sim.getObjectInt32Param(h,sim.objintparam_visibility_layer)&1:
                sim.setShapeColor(h,None,sim.colorcomponent_ambient_diffuse,color)
        connection=sim.getObject(':/connection',{'proxy':model})
        sensor=sim.createForceSensor(0,[0,1,5,0,0],[.012,1000,100,0,0])
        sim.setObjectAlias(sensor,side+'_ToolForceSensor');sim.setObjectParent(sensor,connection,False)
        sim.setObjectPose(sensor,[0,0,.024,0,0,0,1],connection)
        # The stock connection is itself a force sensor; avoid sensor->sensor.
        dynamic_parent=sim.getObjectParent(connection)
        while dynamic_parent!=-1:
            if sim.getObjectType(dynamic_parent)==sim.object_shape_type and sim.getObjectInt32Param(dynamic_parent,sim.shapeintparam_static)==0:
                break
            dynamic_parent=sim.getObjectParent(dynamic_parent)
        if dynamic_parent==-1: raise RuntimeError('No dynamic wrist link above native connection')
        sim.setObjectParent(sensor,dynamic_parent,True)
        sim.setObjectInt32Param(sensor,sim.objintparam_visibility_layer,0)
        palm=shape(side+'_GripperPalm',[.18,.032,.020],[0,0,0],[.17,.21,.27],False,sensor,False)
        sim.setObjectPose(palm,[0,0,.017,0,0,0,1],sensor)
        mass_inertia(sim,palm,[.18,.032,.02],.12)
        for d in [-1,1]:
            finger=shape(side+'_Finger_'+('neg' if d<0 else 'pos'),[.010,.016,.055],[0,0,0],[.5,.55,.60],True,palm,False)
            sim.setObjectPose(finger,[d*.075,0,.042,0,0,0,1],palm)
        tip=sim.createDummy(.004);sim.setObjectAlias(tip,side+'_TCP');sim.setObjectParent(tip,palm,False)
        sim.setObjectPose(tip,[0,0,.071,0,0,0,1],palm)
        sim.setObjectInt32Param(tip,sim.objintparam_visibility_layer,0)
        # Dynamic pad with its own force sensor: contact load reaches the arm.
        pad_sensor=sim.createForceSensor(0,[0,1,5,0,0],[.008,500,50,0,0])
        sim.setObjectAlias(pad_sensor,side+'_PadForceSensor');sim.setObjectParent(pad_sensor,palm,False)
        sim.setObjectPose(pad_sensor,[0,0,.067,0,0,0,1],palm)
        sim.setObjectInt32Param(pad_sensor,sim.objintparam_visibility_layer,0)
        pad=shape(side+'_SupportShoe',[.05,.06,.008],[0,0,0],[.22,.26,.30],False,pad_sensor,False)
        sim.setObjectPose(pad,[0,0,0,0,0,0,1],pad_sensor)
        mass_inertia(sim,pad,[.05,.06,.008],.03)
        sim.setObjectInt32Param(pad,sim.objintparam_visibility_layer,0)
        target=sim.createDummy(.004);sim.setObjectAlias(target,side+'_IK_Target');sim.setObjectParent(target,root,True)
        sim.setObjectInt32Param(target,sim.objintparam_visibility_layer,0)
        shapes=sim.getObjectsInTree(model,sim.object_shape_type,0)
        arms[side]={'model':model,'joints':joints,'tip':tip,'connection':connection,'sensor':sensor,'support_sensor':pad_sensor,'support_pad':pad,'palm':palm,'target':target,
                    'fingers':[h for h in shapes if sim.getObjectAlias(h).startswith(side+'_Finger')], 'shapes':shapes,
                    'home_q':homes[side+'_home'],'quaternion':[1,0,0,0] if side=='A' else [0,1,0,0]}
    return arms


if __name__=='__main__':
    with Simulator(ROOT,port=23060,timeout=40) as owner:
        client,sim=owner.connect();build(sim,client,owner.install)
