from bridge_runtime import HERE, atomic_json, scene_session

rows = []
with scene_session("script_probe") as (sim, client):
    sim.loadScene(str(HERE/"assets/01_-_Initial.ttt"))
    for side in ("A","B"):
        h=sim.getObject(f"/UR5_{side}/Script")
        row={"side":side,"handle":h,"legacy":sim.getScriptInt32Param(h,sim.scriptintparam_enabled)}
        for value in (False, True):
            sim.setBoolProperty(h,"scriptDisabled",value)
            row[str(value)]={"scriptDisabled":sim.getBoolProperty(h,"scriptDisabled"),
                             "legacy":sim.getScriptInt32Param(h,sim.scriptintparam_enabled)}
        rows.append(row)
        print(row, flush=True)
    atomic_json(HERE/"native/script_enable_probe.json", rows)
