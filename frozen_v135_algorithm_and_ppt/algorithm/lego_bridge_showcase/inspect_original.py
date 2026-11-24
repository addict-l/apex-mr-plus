"""Read-only measurements from the original asset copies."""
from bridge_runtime import HERE, atomic_json, scene_session


def scene_records(sim):
    objects = sim.getObjectsInTree(sim.handle_scene, sim.handle_all, 0)
    result = []
    types = {sim.sceneobject_joint: "joint", sim.sceneobject_shape: "shape",
             sim.sceneobject_script: "script", sim.sceneobject_dummy: "dummy"}
    for handle in objects:
        typ = sim.getObjectType(handle)
        if typ not in types:
            continue
        alias = sim.getObjectAlias(handle)
        parent = sim.getObjectParent(handle)
        row = {"id":handle, "alias":alias, "path":sim.getObjectAlias(handle, 1), "type":types[typ],
               "parent":parent, "parent_alias":sim.getObjectAlias(parent) if parent >= 0 else None,
               "pose":sim.getObjectPose(handle, sim.handle_world)}
        if typ == sim.sceneobject_joint:
            row["jpos"] = sim.getJointPosition(handle)
            row["jinterval"] = sim.getJointInterval(handle)
            row["jmode"] = sim.getJointMode(handle)
            row["jkind"] = sim.getJointType(handle)
        if typ == sim.sceneobject_shape:
            row["special"] = sim.getObjectSpecialProperty(handle)
            row["static"] = sim.getObjectInt32Param(handle, sim.shapeintparam_static)
            row["respondable"] = sim.getObjectInt32Param(handle, sim.shapeintparam_respondable)
            row["layer"] = sim.getObjectInt32Param(handle, sim.objintparam_visibility_layer)
            row["bb"] = sim.getShapeBB(handle)
        result.append(row)
    return result


with scene_session("original_inspection") as (sim, client):
    for name in ("01_-_Initial", "02_-_Completed_bridge"):
        sim.loadScene(str(HERE/"assets"/(name+".ttt")))
        records = scene_records(sim)
        atomic_json(HERE/"native"/(name+"_inventory.json"), records)
        print(name, len(records), flush=True)
