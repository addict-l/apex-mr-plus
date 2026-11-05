from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from sim_connection import Simulator
with Simulator(ROOT,port=23043,timeout=30) as owner:
    client,sim=owner.connect()
    model=sim.loadModel(str(owner.install/'models/robots/non-mobile/UR5.ttm'))
    data=[]
    for h in sim.getObjectsInTree(model,sim.object_shape_type,0):
        data.append({'alias':sim.getObjectAlias(h),'static':sim.getObjectInt32Param(h,sim.shapeintparam_static),
                     'respondable':sim.getObjectInt32Param(h,sim.shapeintparam_respondable),
                     'mass':sim.getShapeMass(h),'layer':sim.getObjectInt32Param(h,sim.objintparam_visibility_layer)})
    joints=[]
    for h in sim.getObjectsInTree(model,sim.object_joint_type,0):
        joints.append({'name':sim.getObjectAlias(h),'mode':sim.getJointMode(h),'force':sim.getJointTargetForce(h),
                       'ctrl':sim.getObjectInt32Param(h,sim.jointintparam_dynctrlmode)})
    print(json.dumps({'shapes':data,'joints':joints},indent=2),flush=True)
