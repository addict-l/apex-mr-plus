"""Small, archive-derived dynamic robot and scene construction helpers.

This module deliberately omits all archive paths, scene loading, and output code.
"""
import math
from PIL import Image
from scipy.spatial.transform import Rotation

def mass_inertia(sim,h,size,mass):
    x,y,z=size
    sim.setShapeMass(h,mass)
    sim.setShapeInertia(h,[mass*(y*y+z*z)/12,0,0,0,mass*(x*x+z*z)/12,0,0,0,mass*(x*x+y*y)/12],[1,0,0,0,0,1,0,0,0,0,1,0])

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

def make_label(sim, root, name, png, pos, size, vertical=False):
    im=Image.open(png).convert('RGB')
    w,h=size
    vertices=[-w/2,-h/2,0, w/2,-h/2,0, w/2,h/2,0, -w/2,h/2,0]
    indices=[0,1,2,0,2,3]
    uv=[0,0,1,0,1,1, 0,0,1,1,0,1]
    handle=sim.createShape(8|64,0,vertices,indices,None,uv,im.tobytes(),list(im.size))
    sim.setObjectAlias(handle,name)
    sim.setObjectParent(handle,root,True)
    sim.setObjectPosition(handle,-1,list(pos))
    sim.setObjectOrientation(handle,-1,[math.pi/2 if vertical else 0,0,0])
    sim.setShapeColor(handle,None,sim.colorcomponent_ambient_diffuse,[1,1,1])
    sim.setObjectInt32Param(handle,sim.shapeintparam_static,1)
    sim.setObjectInt32Param(handle,sim.shapeintparam_respondable,0)
    sim.setObjectSpecialProperty(handle,sim.objectspecialproperty_renderable)
    return handle
