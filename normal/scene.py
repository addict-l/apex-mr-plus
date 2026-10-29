"""Dynamic UR5 workcell, nominal inertias, and an explicit assembly problem."""
from __future__ import annotations
import json,math,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from sim_connection import Simulator
from label_mesh import make_label

TABLE_Z=.76
PARTS=[
 {'id':'L1','source':[-.56,-.40,.817],'goal':[-.225,.22,.813],'size':[.13,.13,.08],'mass':.16,'dependencies':[]},
 {'id':'R1','source':[.40,-.40,.817],'goal':[.225,.22,.813],'size':[.13,.13,.08],'mass':.16,'dependencies':[]},
 {'id':'L2','source':[-.39,-.40,.817],'goal':[-.225,.22,.895],'size':[.13,.13,.08],'mass':.16,'dependencies':['L1']},
 {'id':'R2','source':[.57,-.40,.817],'goal':[.225,.22,.895],'size':[.13,.13,.08],'mass':.16,'dependencies':['R1']},
 {'id':'L3','source':[-.56,-.23,.817],'goal':[-.225,.22,.977],'size':[.13,.13,.08],'mass':.16,'dependencies':['L2']},
 {'id':'beam','source':[0,-.35,.812],'goal':[-.025,.22,1.035],'size':[.416,.078,.032],'mass':.22,'dependencies':['L3','R2'],
  'insertion_force':0.,'lever_arm':.20,'connection_moment_limit':1.2},
 {'id':'cap','source':[.45,-.23,.796],'goal':[.105,.22,1.069],'size':[.052,.052,.032],'mass':.06,'dependencies':['beam'],
  'insertion_force':8.,'lever_arm':.33,'connection_moment_limit':1.2},
]
COLORS={'L1':[.82,.68,.43],'L2':[.89,.76,.52],'L3':[.95,.84,.63],
        'R1':[.82,.68,.43],'R2':[.95,.84,.63],'beam':[.08,.60,.28],'cap':[.86,.08,.10]}
for _part in PARTS:
    _part['source'][2] = .807 if _part['id']=='beam' else (.787 if _part['id']=='cap' else .811)


def mass_inertia(sim,h,size,mass):
    x,y,z=size
    inertia=[mass*(y*y+z*z)/12,0,0,0,mass*(x*x+z*z)/12,0,0,0,mass*(x*x+y*y)/12]
    sim.setShapeMass(h,mass)
    sim.setShapeInertia(h,inertia,[1,0,0,0,0,1,0,0,0,0,1,0])


def resolve(sim):
    root=sim.getObject('/APEX_MR_Plus')
    cfg=sim.readCustomTableData(root,'NormalAssembly')
    cfg['root']=root
    cfg['table']=sim.getObject('/APEX_MR_Plus/Workbench')
    cfg['camera']=sim.getObject('/APEX_MR_Plus/OverviewRGB')
    cfg['blocks']={p['id']:sim.getObject('/APEX_MR_Plus/'+p['id']) for p in cfg['parts']}
    cfg['obstacles']=[sim.getObject('/APEX_MR_Plus/'+name) for name in ['CastleBaseplate','A_SupplyTray','B_SupplyTray']]
    for side,a in cfg['arms'].items():
        a['model']=sim.getObject('/APEX_MR_Plus/'+side+'_UR5')
        a['joints']=[sim.getObject(':/'+side+'_joint_'+str(i),{'proxy':a['model']}) for i in range(1,7)]
        a['tip']=sim.getObject(':/'+side+'_TCP',{'proxy':a['model']})
        a['connection']=sim.getObject(':/connection',{'proxy':a['model']})
        a['target']=sim.getObject('/APEX_MR_Plus/'+side+'_IK_Target')
        a['shapes']=sim.getObjectsInTree(a['model'],sim.object_shape_type,0)
        a['fingers']=[h for h in a['shapes'] if sim.getObjectAlias(h).startswith(side+'_Finger')]
        a['palm']=sim.getObject(':/'+side+'_GripperPalm',{'proxy':a['model']})
        a['sensor']=sim.getObject(':/'+side+'_ToolForceSensor',{'proxy':a['model']})
        a['support_sensor']=sim.getObject(':/'+side+'_PadForceSensor',{'proxy':a['model']})
        a['support_pad']=sim.getObject(':/'+side+'_SupportShoe',{'proxy':a['model']})
    return cfg


def build(sim,client,install):
    sim.loadScene(str(ROOT/'backups'/'environment_preview'/'APEX_MR_Plus.ttt'))
    root=sim.getObject('/APEX_MR_Plus')
    old=sim.readCustomTableData(root,'APEX_MR_Plus')
    def remove_tree(handle):
        sim.removeObjects(sim.getObjectsInTree(handle,sim.handle_all,0))
    remove=['SceneControls','CastleStructure','BridgeModule','RedConnector','ConnectorGoalOutline',
            'ConnectorInstallTarget','TemporaryPier','FaultIndicator','A_UR5','B_UR5','A_IK_Target','B_IK_Target']
    for name in remove:
        try: remove_tree(sim.getObject('/APEX_MR_Plus/'+name))
        except Exception: pass
    # Material bodies are all new dynamic parts; remove decorative supply bricks.
    for h in sim.getObjectsInTree(root,sim.handle_all,0):
        try:
            name=sim.getObjectAlias(h)
            if name.startswith(('A_SupplyBrick_','B_SupplyBrick_')) and '_Body' not in name and '_Stud' not in name:
                remove_tree(h)
        except Exception: pass
    def shape(name,size,pos,color,static=True,parent=root,respondable=True):
        h=sim.createPrimitiveShape(sim.primitiveshape_cuboid,list(size),3)
        sim.setObjectAlias(h,name); sim.setObjectParent(h,parent,True); sim.setObjectPosition(h,-1,list(pos))
        sim.setShapeColor(h,None,sim.colorcomponent_ambient_diffuse,color)
        sim.setObjectInt32Param(h,sim.shapeintparam_static,int(static))
        sim.setObjectInt32Param(h,sim.shapeintparam_respondable,int(respondable))
        prop=sim.objectspecialproperty_renderable
        if respondable: prop|=sim.objectspecialproperty_collidable|sim.objectspecialproperty_measurable
        sim.setObjectSpecialProperty(h,prop)
        return h
    baseplate=shape('CastleBaseplate',[.67,.31,.012],[0,.22,.766],[.24,.32,.40])
    # Wider trays for 130 mm demonstration blocks; no source overlaps.
    for side,x in [('A',-.475),('B',.475)]:
        tray=sim.getObject('/APEX_MR_Plus/'+side+'_SupplyTray')
        remove_tree(tray)
        shape(side+'_SupplyTray',[.37,.34,.010],[x,-.33,.765],[.23,.28,.34])
    blocks={}
    parts=json.loads(json.dumps(PARTS))
    for p in parts:
        p.setdefault('insertion_force',0.);p.setdefault('lever_arm',0.);p.setdefault('connection_moment_limit',1.2)
        p['grasps']=[{'id':'top_x','tcp_offset':[0,0,p['size'][2]/2-.010],
                      'quaternion_by_arm':{'A':[1,0,0,0],'B':[0,1,0,0]}},
                     {'id':'top_y','tcp_offset':[0,0,p['size'][2]/2-.010],
                      'quaternion_by_arm':{'A':[math.sqrt(.5),math.sqrt(.5),0,0],
                                           'B':[-math.sqrt(.5),math.sqrt(.5),0,0]}}]
        h=shape(p['id'],p['size'],p['source'],COLORS[p['id']],False)
        blocks[p['id']]=h;mass_inertia(sim,h,p['size'],p['mass'])
        nx=6 if p['id']=='beam' else (3 if p['size'][0]>.10 else 2)
        ny=2 if p['id'] in ['beam','cap'] else 3
        for i in range(nx):
            for j in range(ny):
                stud=sim.createPrimitiveShape(sim.primitiveshape_cylinder,[.014,.014,.004],3)
                sim.setObjectAlias(stud,p['id']+f'_VisualStud_{i}_{j}')
                sim.setObjectParent(stud,h,False)
                sim.setObjectPosition(stud,h,[(i-(nx-1)/2)*p['size'][0]/nx,(j-(ny-1)/2)*p['size'][1]/ny,p['size'][2]/2+.002])
                sim.setObjectInt32Param(stud,sim.shapeintparam_static,1)
                sim.setObjectInt32Param(stud,sim.shapeintparam_respondable,0)
                sim.setObjectSpecialProperty(stud,sim.objectspecialproperty_renderable)
                sim.setShapeColor(stud,None,sim.colorcomponent_ambient_diffuse,COLORS[p['id']])
    arms={}
    for side,x,yaw,color in [('A',-.70,0,[1.,.34,.055]),('B',.70,math.pi,[.025,.40,.86])]:
        model=sim.loadModel(str(install/'models/robots/non-mobile/UR5.ttm'))
        sim.setObjectAlias(model,side+'_UR5');sim.setObjectParent(model,root,True)
        sim.setObjectInt32Param(model,sim.shapeintparam_static,1)
        for s in sim.getObjectsInTree(model,sim.object_script_type,0):sim.removeObjects([s])
        sim.setObjectPose(model,[x,.13,.783]+Rotation.from_euler('z',yaw).as_quat().tolist(),-1)
        joints=sim.getObjectsInTree(model,sim.object_joint_type,0)
        homes=old['poses']
        proposal=ROOT/'results'/'normal'/'motion_home_proposal.json'
        if proposal.exists():
            candidate=json.loads(proposal.read_text(encoding='utf-8'))
            proposed=candidate.get('home_q',candidate.get('homes',candidate))
            if side in proposed and isinstance(proposed[side],list): homes={**homes,side+'_home':proposed[side]}
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
        palm=shape(side+'_GripperPalm',[.032,.18,.020],[0,0,0],[.17,.21,.27],False,sensor,False)
        sim.setObjectPose(palm,[0,0,.017,0,0,0,1],sensor)
        mass_inertia(sim,palm,[.032,.18,.02],.12)
        for d in [-1,1]:
            finger=shape(side+'_Finger_'+('neg' if d<0 else 'pos'),[.016,.010,.055],[0,0,0],[.5,.55,.60],True,palm,False)
            sim.setObjectPose(finger,[0,d*.075,.042,0,0,0,1],palm)
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
    sim.setInt32Param(sim.intparam_dynamic_engine,sim.physics_bullet)
    sim.setFloatProperty(sim.handle_scene,'dynamicsStepSize',.002)
    sim.setIntProperty(sim.handle_scene,'bullet.iterations',150)
    sim.setVector3Property(sim.handle_scene,'gravity',[0,0,-9.81])
    sim.setFloatParam(sim.floatparam_simulation_time_step,.02)
    sim.setBoolParam(sim.boolparam_realtime_simulation,False)
    for h in [sim.getObject('/APEX_MR_Plus/Workbench'),baseplate,*blocks.values()]:
        sim.setFloatProperty(h,'bullet.friction',.8);sim.setFloatProperty(h,'bullet.frictionOld',.8)
        sim.setFloatProperty(h,'bullet.restitution',0.)
        sim.setFloatProperty(h,'bullet.linearDamping',.04);sim.setFloatProperty(h,'bullet.angularDamping',.04)
    cfg={'root':root,'table':sim.getObject('/APEX_MR_Plus/Workbench'),'arms':arms,'parts':parts,'blocks':blocks,
         'initial':{p['id']:p['source'] for p in parts},'goal':{p['id']:p['goal'] for p in parts},
         'sizes':{p['id']:p['size'] for p in parts},'obstacles':[baseplate]+[sim.getObject('/APEX_MR_Plus/'+s+'_SupplyTray') for s in ['A','B']], 'table_z':TABLE_Z,'dt':.02,
         'camera':sim.getObject('/APEX_MR_Plus/OverviewRGB'),'program':{},
         'physics':{'engine':'Bullet 2.83','dt':.02,'substep':.002,'gravity':[0,0,-9.81],
                    'motor':'native UR5 dynamic position servo; no runtime setJointPosition',
                    'grasp':'dynamic rigid force-sensor constraint, not frictional finger simulation',
                    'connection':'idealized rigid snap constraint after measured placement; not LEGO deformation',
                    'studs':'visual only; cuboid contact geometry','assembly_sequence_input':[p['id'] for p in parts]}}
    sim.writeCustomTableData(root,'NormalAssembly',cfg)
    sim.saveScene(str(ROOT/'scenes'/'10_Normal_Assembly.ttt'))
    report=[]
    for h in sim.getObjectsInTree(root,sim.object_shape_type,0):
        if sim.getObjectInt32Param(h,sim.shapeintparam_static)==0:
            inertia,transform=sim.getShapeInertia(h)
            report.append({'path':sim.getObjectAlias(h,1),'mass_kg':sim.getShapeMass(h),'inertia_kg_m2':inertia,'inertia_frame':transform})
    (ROOT/'results'/'normal'/'rigid_bodies.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (ROOT/'results'/'normal'/'problem.json').write_text(json.dumps({'arms':['A','B'],'parts':parts,'physics':cfg['physics']},indent=2),encoding='utf-8')
    print(json.dumps({'scene':'10_Normal_Assembly.ttt','dynamic_bodies':len(report),'parts':len(parts)}),flush=True)
    return cfg

if __name__=='__main__':
    with Simulator(ROOT,port=23043,timeout=40) as owner:
        client,sim=owner.connect();build(sim,client,owner.install)
