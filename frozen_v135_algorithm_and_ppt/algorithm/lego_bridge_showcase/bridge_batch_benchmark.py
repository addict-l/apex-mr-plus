
"""Replay frozen v1 states through scalar and batched native mesh queries."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import json
import time

import numpy as np

from bridge_runtime import HERE, atomic_json, scene_session
from bridge_geometry import BridgeWorld
from bridge_motion import connect_free, edge_samples
from p0_2_handover import CASES,HANDOVER_CENTERS_M,closed_width
from lego_grasp_model import jaw_q

DECK="Bridge_deck_12x4"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def at_bottom(world,state,name,into):
    m=np.eye(4)
    m[:3,3]=into
    state["parts"][name]={"owner":None,"pose":(m@world.offsets[name]).tolist()}


def mate(world,side,name):
    return [[row["handle"],world.parts[name]["handle"]]
            for row in world.arms[side]["shape_roles"]["Grip_pad"]]


def state_list(world):
    base=world.state()
    base.pop("follower")
    out=[]
    archived=read(HERE/"p0_1_receipt.json")
    deck=archived["attempts"][archived["chosen_attempt"]]
    side=deck["side"]
    phases={row["phase"]:row for row in deck["phases"]}
    mtx=np.asarray(phases["close_on_short_edge"]["last"]["actual_tcp"])
    binding=(np.linalg.inv(mtx)@np.asarray(base["parts"][DECK]["pose"])).tolist()
    for phase,row in phases.items():
        for label in ("closest","last"):
            saved=row[label]
            fixture=deepcopy(base)
            fixture["q"][side]=saved["joint_rad"]
            fixture["jaw"][side]=saved["jaw_m"]
            if saved["held"]:
                fixture["parts"][DECK]={"owner":side,"binding":binding}
            if phase in {"approach_completed_piers","seat_deck_on_four_piers",
                         "open_at_seated_deck","open_vertical_retreat"}:
                for name,position in deck["completed_fixture_bottom_m"].items():
                    at_bottom(world,fixture,name,position)
                if not saved["held"]:
                    at_bottom(world,fixture,DECK,world.parts[DECK]["target_bottom_m"])
            if saved.get("intended_pad_contact") or saved["held"]:
                fixture["allow"]=mate(world,side,DECK)
            out.append(("deck:"+phase+":"+label,fixture,True))

    trace=[json.loads(line) for line in (HERE/"p0_2_neighbor_local_samples.jsonl")
           .read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    for case in CASES:
        selected=[row for row in trace if row["case"]==case["id"] and row["attempt"]==1]
        part,side=case["part"],case["taker"]
        fixture=deepcopy(base)
        for name in (*case["lower"],case["neighbor"]):
            at_bottom(world,fixture,name,world.parts[name]["target_bottom_m"])
        initial=deepcopy(fixture)
        relative=np.asarray(world.parts[part]["target_bottom_m"])+[0.,0.,.047]
        at_bottom(world,initial,part,relative)
        initial["q"]=selected[0]["qs_rad"]
        size=np.asarray(world.parts[part]["bounds_max_m"])-world.parts[part]["bounds_min_m"]
        jaw=jaw_q(closed_width(size[0]))
        initial["jaw"][side]=jaw
        applied=world.apply_state(initial)
        if not applied["ok"]:
            raise RuntimeError(applied)
        binding=(np.linalg.inv(world.tcp(side))@np.asarray(initial["parts"][part]["pose"])).tolist()
        listed=[]
        for row in selected:
            state=deepcopy(fixture)
            state["q"]=row["qs_rad"]
            command=row.get("command") or {}
            if "jaw_q_m" in command:
                jaw=command["jaw_q_m"]
            state["jaw"][side]=jaw
            if row.get("binding_error_m") is not None:
                state["parts"][part]={"owner":side,"binding":binding}
                state["allow"]=mate(world,side,part)
            else:
                at_bottom(world,state,part,world.parts[part]["target_bottom_m"])
                if row["phase"] in {"vertical_neighbor_seating","open_at_neighbor"}:
                    state["allow"]=mate(world,side,part)
            listed.append((f'neighbor:{case["id"]}:{row["phase"]}:{row["sample"]}',state,
                           row["clearance"]["ok"]))
        indexes=np.linspace(0,len(listed)-1,37,dtype=int)
        out.extend(listed[index] for index in indexes)

    bad=deepcopy(base)
    bad["q"][deck["side"]]=archived["negative"]["joint_rad"]
    bad["jaw"][deck["side"]]=jaw_q(archived["negative"]["negative_grip_m"])
    out.append(("negative:narrow_deck",bad,False))
    case=CASES[0]
    p2=read(HERE/"p0_2_receipt.json")
    wrong=next(row for row in p2["negative_cases"] if row["id"]=="wrong_y_grasp_at_neighbor")
    bad=deepcopy(base)
    for name in (*case["lower"],case["neighbor"]):
        at_bottom(world,bad,name,world.parts[name]["target_bottom_m"])
    bad["q"]=wrong["qs_rad"]
    dims=np.asarray(world.parts[case["part"]]["bounds_max_m"])-world.parts[case["part"]]["bounds_min_m"]
    bad["jaw"][case["taker"]]=jaw_q(float(dims[1]))
    out.append(("negative:adjacent_wrong_axis",bad,False))
    world.apply_state(base)
    return base,out


def identity(row):
    return row["ok"],tuple(row.get("pair_handles") or ()),tuple(
        tuple(entry) for entry in row.get("allowed_collisions") or ())


def benchmark(world,base,entries):
    count=len(entries)
    if not 64 <= count <= 128:
        raise ValueError(f"Expected 64–128 archived states, got {count}")
    scalar=[]
    started=time.monotonic()
    try:
        for label,state,expected in entries:
            applied=world.apply_state(state)
            if not applied["ok"]:
                raise RuntimeError(f"Scalar apply rejected {label}: {applied}")
            reply=world.check(state["allow"])
            scalar.append(reply)
    finally:
        world.apply_state(base)
    scalar_time=time.monotonic()-started
    before=world.state()
    started=time.monotonic()
    result=world.batch_check([item[1] for item in entries],stop_at_failure=False,
                             heartbeat=lambda row:print("archived batch",row,flush=True))
    batch_time=time.monotonic()-started
    after=world.state()
    restoration=(before["q"]==after["q"] and before["jaw"]==after["jaw"]
              and all(before["parts"][part]["parent"]==row["parent"]
                      and np.max(np.abs(np.asarray(before["parts"][part]["pose"])-row["pose"]))<=1e-8
                      for part,row in after["parts"].items()))
    equal=[]
    for label_index,(label,_,expected) in enumerate(entries):
        old=scalar[label_index]
        new=result["rows"][label_index] if label_index<len(result["rows"]) else {}
        clearance=old["minimum_distance_m"]
        passed=bool(new and identity(old)==identity(new)
                    and abs(clearance-new["minimum_distance_m"])<=1e-8
                    and old["ok"] is expected)
        equal.append({"label":label,"expected":expected,"matches":passed,
                      "scalar":old,"batch":new})
    passed=all(row["matches"] for row in equal) and restoration
    return {"passed":passed,"sample_count":count,"matched":sum(row["matches"] for row in equal),
            "categories":dict(Counter(row[0].split(":")[0] for row in entries)),
            "first_negative_batch_sample":result.get("first_failure"),
            "scalar_seconds":scalar_time,"batch_seconds":batch_time,
            "samples_per_second":{"legacy":count/scalar_time,"batch":count/batch_time},
            "throughput_speedup":scalar_time/batch_time,"restored":restoration and result.get("restored"),
            "batch_error":result.get("error"),"timing":result["timing"],"rows":equal}


def native_rrt(world, base, entries):
    # Mine a fixed, small set of actual archived joint endpoints. Search
    # validated connections only when the native straight segment collides.
    relevant=[state["q"]["A"] for tag,state,ok in entries
              if ok and tag in {"deck:open_jaws_at_park:last","deck:raise_clear_of_stock:last",
                                "deck:open_approach:last","deck:open_descend_to_deck:last"}]
    states=[base["q"]["A"],*relevant]
    for tag,state,ok in entries:
        if ok and tag.startswith("neighbor:B_to_A:vertical_open_retreat") and len(states)<8:
            states.append(state["q"]["A"])
    fi=deepcopy(base)
    fi["jaw"]["A"]=.017
    fi["measure_distance"]=False
    probes=[]
    began=time.monotonic()
    for i,a in enumerate(states):
        fi["q"]["A"]=a
        for j,b in enumerate(states[i+1:],i+1):
            if time.monotonic()-began>150.:
                break
            rows=[{**fi,"q":{**fi["q"],"A":list(qa)}} for qa in [*edge_samples(a,b)]]
            direct=world.batch_check(rows,deadline=time.monotonic()+10.)
            if direct["ok"]:
                continue
            index=direct.get("first_failure")
            if index is None or direct["rows"][index]["kind"]!="native_collision":
                continue
            status=connect_free(world,"A",fi,[b],timeout_s=22.,seed=721+i*len(states)+j)
            probes.append({"start":a,"goal":b,"blocked_direct":direct["rows"][index],
                           "failed_direct_q":rows[index]["q"]["A"],"bounded_search":status})
            if status.get("ok") and status["method"]=="RRTConnect":
                return {"passed":True,"attempts":probes,"wall_seconds":time.monotonic()-began}
    return {"passed":False,"attempts":probes,"wall_seconds":time.monotonic()-began,
            "reason":"no archived blocked direct pair certified by bounded RRT"}


if __name__=="__main__":
    folder=HERE/"native"
    with scene_session("bridge_batch_benchmark") as (sim,client):
        world=BridgeWorld(sim,client,batch_joints=True)
        initial,entries=state_list(world)
        result=benchmark(world,initial,entries)
        atomic_json(folder/"batch_benchmark_receipt.json",result)
        print("batch exact matches",result["matched"],"/",result["sample_count"],flush=True)
        if not result["passed"]:
            raise SystemExit(1)
        rrt=native_rrt(world,initial,entries)
        atomic_json(folder/"rrt_benchmark_receipt.json",rrt)
        print("native RRT proof",rrt["passed"],flush=True)
        if not rrt["passed"]:
            raise SystemExit(1)
