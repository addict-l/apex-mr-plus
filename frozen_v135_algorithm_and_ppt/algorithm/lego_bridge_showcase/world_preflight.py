"""Check saved manifest identity, stopped physics and native clearance."""
from bridge_runtime import HERE, atomic_json, scene_session
from bridge_geometry import BridgeWorld

with scene_session("world_preflight") as (sim, client):
    world=BridgeWorld(sim,client)
    got=world.check()
    rows={}
    for side, chain in world.chains.items():
        q=world.q(side)
        sols=world.ik(side,world.tcp(side),q)
        rows[side]={"fk_error":chain.error,"ik_solutions":len(sols),
                    "open_interval":sim.getJointInterval(world.arms[side]["jaw_opening"]["handle"]),
                    "home_q":q.tolist()}
    evidence={"initial_geometry":got,"shape_scope":world.geometry_scope,
              "registration":world.query_registration,"calibration":rows,
              "scene_stopped":sim.getSimulationState()==sim.simulation_stopped}
    atomic_json(HERE/"native"/"world_preflight_receipt.json",evidence)
    print(got, rows, flush=True)
