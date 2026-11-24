"""Local seated-neighbor sweeps and a diagnostic replay of the blocked handoff."""
from pathlib import Path
import json
import math

import numpy as np

from bridge_runtime import HERE, atomic_json, capture, scene_session
from bridge_geometry import BridgeWorld
from lego_grasp_model import matrix, pier_taker_grasp, pose_error, translate
from p0_2_handover import (
    CASES, OPEN_M, Rejected, RunTrace, closed_width, contact, continuous_move,
    final_bottom, h, interarm, jaw_sweep, pad_handles, part_sizes, place_supports,
    require_gate, reversible, set_width,
)


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def snapshot_collision(world):
    source = load_json(HERE/"p0_2_receipt.json")["directions"][0]
    reference = source["handoff_endpoint_trials"][0]
    settings = reference["first_rejection"]
    case = CASES[0]
    with reversible(world):
        place_supports(world,case)
        target = np.asarray(reference["handover_bottom_m"])
        part = case["part"]
        world.set_bottom(part,matrix(target))
        dims = part_sizes(world,part)
        for role,axis in (("giver",1),("taker",0)):
            side=case[role]
            world.setq(side,settings["qs_rad"][side])
            set_width(world,side,closed_width(dims[axis]))
        measurement=interarm(world)
        if measurement["ok"]:
            raise ValueError("The archived interarm collision did not reproduce")
        picture=capture(world.sim,HERE/"p0_previews/hand_mid_orthogonal_collision.png",
                        target+[.17,-.27,.16],target+[0.,0.,.03],fov=32)
        return {"role_assignment":"A_to_B", "scope":"rejected orthogonal endpoint; no attachment",
                "archived_pair":settings["interarm"]["pair"],"native":measurement,"preview":picture,
                "q_rad":settings["qs_rad"],"settings":reference}


def local_trial(world, case, direction, height, route, events):
    part, side = case["part"], case["taker"]
    goal = final_bottom(world,part)
    above = translate(goal,[0,0,.047])
    target = pier_taker_grasp(above,direction,height)
    world.set_bottom(part,above)
    dims=part_sizes(world,part)
    set_width(world,side,closed_width(dims[0]))
    native=False
    for q in world.ik(side,target,max_solutions=3,seed_count=12):
        world.setq(side,q)
        fit, allow=contact(world,side,part,dims,0)
        check=world.check(allow=allow)
        if check["ok"]:
            native=True
            break
    require_gate(native,"local staged ±X grasp has no validated starting configuration")
    trace=RunTrace(world,route,events)
    route["wall_fit"]=fit
    binding = np.linalg.inv(world.tcp(side)) @ world.bottom(part)
    with trace.stage("staged_positive_x_side_grasp"):
        route["attachment_drift_m"]=world.attach(part,side)
        trace.sample(allow,part_binding=(part,side,binding))
    with trace.stage("vertical_neighbor_seating"):
        continuous_move(world,trace,side,pier_taker_grasp(goal,direction,height),
                        allowed=allow,binding=(part,side,binding))
        residual=pose_error(goal,world.bottom(part))
        require_gate(residual[0]<=1e-5 and residual[1]<=2e-4,"staged neighbor seat mismatch",residual=residual)
        route["seat_error"]=residual
        route["release_drift_m"]=world.attach(part,None)
        trace.sample(allow)
    with trace.stage("open_at_neighbor"):
        jaw_sweep(world,trace,side,OPEN_M,preallowed=allow)
    with trace.stage("vertical_open_retreat"):
        continuous_move(world,trace,side,translate(world.tcp(side),[0.,0.,.047]))
    route["passed"]=True


def main():
    data={"scope":"local ±X placement and clearance only; staged held state, no handoff-motion credit",
          "passed":False,"directions":[]}
    try:
        with scene_session("p0_2_neighbor_local") as (sim,client):
            world=BridgeWorld(sim,client,batch_joints=True)
            prior=load_json(HERE/"native/p0_2_neighbor_local_attempt1_receipt.json")
            data["capture"]=prior["capture"]
            trace_path=HERE/"p0_2_neighbor_local_samples.jsonl"
            data["trace_path"]=str(trace_path)
            with trace_path.open("w",encoding="utf-8",buffering=1) as events:
                for case in CASES:
                    out={"id":case["id"],"part":case["part"],"side":case["taker"],
                         "slot":case["slot"],"trials":[]}
                    data["directions"].append(out)
                    print(case["id"],"local neighbor sweep",flush=True)
                    for attempt, (direction,height) in enumerate(((1,.0087),(-1,.0087),(1,.011)),1):
                        route={"id":case["id"],"attempt":attempt,"grip_direction":direction,
                               "height_m":height,"passed":False,"stages":[]}
                        out["trials"].append(route)
                        with reversible(world):
                            place_supports(world,case)
                            try:
                                local_trial(world,case,direction,height,route,events)
                            except Rejected as err:
                                route.update(reason=err.reason,evidence=err.evidence)
                        if route["passed"]:
                            break
                    out["passed"]=any(row["passed"] for row in out["trials"])
                    atomic_json(HERE/"p0_2_neighbor_local_receipt.json",data)
        data["passed"]=all(row["passed"] for row in data["directions"])
    except Exception as err:
        data["error"]=f"{type(err).__name__}: {err}"
        raise
    finally:
        atomic_json(HERE/"p0_2_neighbor_local_receipt.json",data)
    print("local neighbor sweeps:", [(r["id"],r["passed"]) for r in data["directions"]],flush=True)


if __name__=="__main__":
    main()
