"""Native v1 front gate for the overnight Mode B handoff."""
from contextlib import contextmanager
from copy import deepcopy
from hashlib import sha256
from itertools import product
from pathlib import Path
import json
import time
import tomllib
import traceback

import numpy as np

from bridge_geometry import BridgeWorld
from bridge_handover import HAND_MID, HANDOVER_REVISION, execute_handoff, verify_handoff_withdrawal
from bridge_motion import MAX_JOINT_SAMPLE_RAD, connect_free, continuous_ik, near_waypoints
from bridge_runtime import HERE, atomic_json, capture, scene_session
from lego_grasp_model import linear_targets, matrix, pier_taker_grasp, pose_error, translate
from p0_2_handover import (
    CASES, OPEN_M, ORIENTATION_LIMIT_RAD, POSITION_LIMIT_M, Rejected, RunTrace,
    closed_width, contact, final_bottom, force_probe, h, interarm, jaw_sweep, pad_handles,
    part_sizes, place_supports, require_gate, reversible, set_width, wrong_neighbor,
)
from p0_2_height_taker import giver_pose, taker_pose

OUTPUT = HERE / "mode_b"
RECEIPT = OUTPUT / "handover_gate.json"
TRACE = OUTPUT / "handover_gate_samples.jsonl"
SOURCE = HERE / "p0_2_height_receipt.json"
SCENE = HERE / "assets/lego_bridge_kinematic_v1.ttt"
MANIFEST = HERE / "assets/scene_manifest.json"
V1_SHA256 = "f79e0e2f407b54dbfc7af4ea92bcc2588562c311c9d17636aca9ba6ab8d56372"
MAX_STEP = .001
SUPERVISOR_SECONDS = 1800.
WORK_SECONDS = 1760.
DEADLINE = None


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def remaining():
    seconds = DEADLINE - time.monotonic()
    require_gate(seconds > 0., "Mode B gate work deadline reached", timeout=True)
    return seconds


class GateWorld(BridgeWorld):
    """Keep the original native collision predicates, with distances opt-in."""

    def check(self, allow=(), requested_distance_m=.025, *, measure=False, filtered_clearance=False, **kwargs):
        if not measure:
            remaining()
        return super().check(allow, requested_distance_m, measure=measure,
                             filtered_clearance=filtered_clearance, **kwargs)

    def batch_check(self, states, *, measure=False, deadline=None, **kwargs):
        return super().batch_check(states, measure=measure, deadline=deadline or DEADLINE, **kwargs)


class GateTrace(RunTrace):
    def __init__(self, world, row, events, report):
        super().__init__(world, row, events)
        self.report = report

    @contextmanager
    def stage(self, label):
        remaining()
        print(f'{self.output["id"]} {label}', flush=True)
        try:
            with super().stage(label) as evidence:
                evidence["minimum_distance_m"] = "not_measured"
                yield evidence
        finally:
            atomic_json(RECEIPT, self.report)

    def emit(self, item):
        self.events.write(json.dumps({"case": self.output["id"], "attempt": self.output["attempt"],
                     "phase": self.current["phase"], "sample": self.current["samples"], **item},
                     ensure_ascii=False, allow_nan=False) + "\n")
        self.current["samples"] += 1

    def sample(self, allowed=(), *, command=None, part_binding=None):
        reading = self.world.check(allow=allowed)
        binding_errors = None
        if part_binding:
            name, owner, offset = part_binding
            binding_errors = pose_error(self.world.tcp(owner) @ offset, self.world.bottom(name))
        q = {side: self.world.q(side).tolist() for side in ("A", "B")}
        self.emit({"qs_rad": q, "command": command, "clearance": reading,
                   "binding_error": binding_errors})
        require_gate(reading["ok"], "native geometry rejected the swept state",
                     clearance=reading, qs_rad=q, allow=[list(x) for x in allowed])
        if binding_errors:
            require_gate(binding_errors[0] <= POSITION_LIMIT_M
                         and binding_errors[1] <= ORIENTATION_LIMIT_RAD,
                         "rigid payload binding changed during interpolation", error=binding_errors)
        return reading

    def key_clearance(self, allowed=()):
        remaining()
        reading = self.world.check(allow=allowed, measure=True, filtered_clearance=True)
        self.emit({"kind": "verified_keypoint_clearance", "clearance": reading})
        require_gate(reading.get("ok"), "native collision at verified keypoint",
                     clearance=reading, allow=[list(x) for x in allowed])
        require_gate(isinstance(reading.get("minimum_distance_m"), (int, float))
                     and reading["minimum_distance_m"] > 0.,
                     "No positive external native clearance at verified keypoint", clearance=reading)
        self.current.update(key_clearance=reading, minimum_distance_m=reading["minimum_distance_m"],
                            pair=reading.get("pair"))


def as_pose(native_tcp):
    data = np.eye(4)
    data[:3] = np.asarray(native_tcp, dtype=float).reshape(3, 4)
    return data


def state_for(world, side, q=None, target=None, allowed=()):
    state = world.state()
    state["allow"] = [list(pair) for pair in allowed]
    state["measure_distance"] = False
    if q is not None:
        state["q"][side] = list(q)
    if target is not None:
        state["tcp_targets"] = {side: np.asarray(target).tolist()}
    return state


def near_targets(start, goal):
    # Keep the existing tool-corner interpolation bound and add position
    # margin for independent native FK readback at consecutive waypoints.
    previous = start
    for target in near_waypoints(start, goal):
        yield from linear_targets(previous, target, step_m=.00088)
        previous = target


def solved_samples(solution, start_pose, side):
    """Audit the actual simulator TCP displacement, never only the commands."""
    rows = solution.get("rows", [])
    path = solution.get("q_path", [])
    previous = start_pose
    actual = []
    for index, row in enumerate(rows):
        if not row.get("ok", False):
            break
        measured = as_pose(row["tcp"][side])
        step = pose_error(previous, measured)[0]
        require_gate(step <= MAX_STEP, "measured native TCP step exceeds 1 mm",
                     step_m=step, sample=index, stable_q_rad=path[index+1])
        actual.append((measured, path[index+1], row, step))
        previous = measured
    return actual


def move(world, trace, side, target, *, allowed=(), binding=None):
    remaining()
    state = state_for(world, side, allowed=allowed)
    origin = world.tcp(side)
    result = continuous_ik(world, side, state, list(near_targets(origin, target)), deadline=DEADLINE)
    path = result.get("q_path", [])
    rows = result.get("rows", [])
    trace.current.setdefault("motions", []).append({"kind":"continuous_ik", "samples":len(rows),
              "q_path":path, "timing":result.get("timing"), "reason":result.get("reason"),
              "first_failure":result.get("first_failure"), "error":result.get("error")})
    actual = solved_samples(result, origin, side)
    for _,q,row,step in actual:
        trace.emit({"side":side, "q_rad":q, "kind":"native_batch_waypoint",
                    "tcp_step_m":step, "native":row})
    if not result.get("ok"):
        index = result.get("first_failure")
        native = rows[index] if isinstance(index,int) and index < len(rows) else None
        evidence = {"native":native, "failed_sample":index, "requested_tcp":result.get("requested_tcp"),
                    "side":side, "initial_state":state, "q_path":path,
                    "native_complete":False, "reason":result.get("reason") or result.get("error")}
        if native is not None and index+1 < len(path):
            evidence["failing_q_rad"] = path[index+1]
        raise Rejected("continuous batch failed", evidence)
    require_gate(bool(actual), "no native Cartesian waypoint checked", side=side)
    world.setq(side, actual[-1][1])
    trace.sample(allowed, command={"moving_arm":side,"kind":"committed_dense_endpoint"},
                 part_binding=binding)


def high_connect(world, trace, side, goal_qs, target, *, allowed=(), binding=None, secs=42., via_qs=()):
    state = state_for(world, side, allowed=allowed)
    limits = min(secs, remaining())
    result = connect_free(world, side, state, goal_qs, timeout_s=limits,
                          seed=20261122, max_iterations=1800, via_qs=via_qs)
    trace.current.setdefault("rrt_connections", []).append(result)
    failure = result.get("first_failure") or {}
    require_gate(result.get("ok"), "bounded free-space connector failed",
                 initial_state=state, side=side, native=failure,
                 failing_q_rad=failure.get("q"), rrt=result)
    for q in result["q_path"]:
        trace.emit({"side":side,"q_rad":q,"kind":"native_free_connector",
                    "native_scope":f"complete active geometry and <={MAX_JOINT_SAMPLE_RAD:g} rad samples"})
    world.setq(side, result["q_path"][-1])
    pe,ae = pose_error(target,world.tcp(side))
    require_gate(pe <= POSITION_LIMIT_M and ae <= ORIENTATION_LIMIT_RAD,
                 "free-space connector endpoint native FK mismatch", error=[pe,ae], side=side)
    trace.sample(allowed, command={"moving_arm":side,"kind":"committed_free_endpoint"},
                 part_binding=binding)


def load_park_via(side):
    path = HERE / "outboard_park_a1/parking_calibration.json"
    if not path.is_file():
        return ()
    data = load(path)
    park = data.get("park_q") or {}
    if data.get("passed") is True and side in park:
        return (park[side],)
    return ()


def connect_pose(world, trace, side, target, *, seeds=(), allowed=(), binding=None,
                 prefer_seed_branch=False, via_qs=()):
    if prefer_seed_branch and seeds:
        preferred = world.ik(side, target, seed=seeds[0], seed_count=1, max_solutions=1)
        branch = preferred[0] if preferred and float(np.max(np.abs(preferred[0]-seeds[0]))) <= .15 else None
        if branch is not None:
            try:
                high_connect(world, trace, side, [branch], target, allowed=allowed,
                             binding=binding, via_qs=via_qs)
                return
            except Rejected:
                pass
    goals=[]
    for seed in [*seeds,world.q(side)]:
        remaining()
        options = world.ik(side,target,seed=seed,seed_count=8 if not goals else 1,max_solutions=4)
        for q in options:
            if not any(float(np.max(np.abs(q-other))) < .001 for other in goals):
                goals.append(q)
    require_gate(bool(goals), "no IK endpoint for free-space connector", side=side, target=target.tolist())
    high_connect(world, trace, side, goals, target, allowed=allowed, binding=binding, via_qs=via_qs)


def stock_approach(world, trace, side, grasp):
    with trace.stage("giver_open_at_stock"):
        jaw_sweep(world,trace,side,OPEN_M)
    original = world.state()
    startup_q = world.q(side)
    # Reverse a conservative near-contact column from the exact stock grasp.
    # A different wrist/elbow solution can avoid the observed mount strip.
    branches=world.ik(side,grasp,seed=startup_q,seed_count=38,max_solutions=10)
    target=translate(grasp,[0.,0.,.049])
    tested=[]
    last_stop=None
    with trace.stage("giver_stock_column_precheck") as proof:
        proof["scope"]="reversible probe, no traveled credit"
        proof["tested"]=tested
        for q in branches:
            remaining()
            state=state_for(world,side,q=q,target=grasp)
            start=world.batch_check([state],deadline=DEADLINE)
            row={"stock_q_rad":q.tolist(),"passed":False,"start":start}
            tested.append(row)
            if not start["ok"]:
                native=(start.get("rows") or [{}])[0]
                if native.get("kind")=="native_collision":
                    last_stop={"native":native,"initial_state":state,
                               "failing_q_rad":q.tolist(),"side":side}
                continue
            state.pop("tcp_targets",None)
            sample=as_pose(start["rows"][0]["tcp"][side])
            result=continuous_ik(world,side,state,list(near_targets(grasp,target)),deadline=DEADLINE)
            row["column"]=result
            actual=solved_samples(result,sample,side)
            if not result.get("ok"):
                index=result.get("first_failure")
                if isinstance(index,int) and index<len(result.get("rows",[])):
                    native=result["rows"][index]
                    if native.get("kind")=="native_collision":
                        last_stop={"native":native,"initial_state":state,
                                   "failing_q_rad":result["q_path"][index+1],"side":side}
                continue
            require_gate(bool(actual),"empty native stock column",side=side)
            # The exact reverse path is symmetric for the same fixed scene and
            # jaws. A free connector only receives the witnessed hover state.
            row.update(passed=True,hover_q_rad=actual[-1][1])
            row["stock_path"]=[q.tolist(),*[item[1] for item in actual]]
        require_gate(any(row["passed"] for row in tested),
                     "no tested stock IK branch has a clear vertical approach", **(last_stop or {}))

    valid=[row for row in tested if row["passed"]]
    with trace.stage("giver_free_stock_connector"):
        high_connect(world,trace,side,[row["hover_q_rad"] for row in valid],target,secs=65.)
    q=world.q(side)
    selected=min(valid,key=lambda row:float(np.max(np.abs(q-np.array(row["hover_q_rad"])))))
    require_gate(np.max(np.abs(q-selected["hover_q_rad"])) <= 1e-7,
                 "stock connector did not end on certified approach branch",side=side)
    with trace.stage("giver_prechecked_vertical_stock_descent"):
        # Recheck the exact returned sequence in the traveled ordering after
        # the free connector. No assumed or substituted samples enter credit.
        q_path=list(reversed(selected["stock_path"]))
        origin=world.tcp(side)
        states=[state_for(world,side)]
        states=[{**states[0],"q":{**states[0]["q"],side:point}} for point in q_path[1:]]
        result=world.batch_check(states,deadline=DEADLINE)
        rows=result.get("rows",[])
        result["q_path"]=q_path
        if not result["ok"]:
            index=result.get("first_failure")
            evidence={"initial_state":original,"side":side,"native":None}
            if index is not None and index<len(rows):
                evidence.update(failing_q_rad=q_path[index+1],native=rows[index])
            raise Rejected("native reverse-stock replay rejected",evidence)
        for _,q,native,step in solved_samples(result,origin,side):
            trace.emit({"side":side,"q_rad":q,"kind":"native_reverse_stock_replay",
                        "tcp_step_m":step,"native":native})
        world.setq(side,q_path[-1])
        trace.sample()


def run_chain(world, case, candidate, dims, trace):
    giver,taker,part=(case[key] for key in ("giver","taker","part"))
    bottom=world.bottom(part)
    mid=matrix(HAND_MID)
    goal=final_bottom(world,part)
    hold_seed=candidate["qs_rad"]
    gp=lambda p:giver_pose(p,candidate["tilt_deg"],case["incline"],
                           candidate["giver_height_m"],candidate["giver_outboard_m"])
    tp=lambda p:taker_pose(p,candidate["taker_tilt_deg"],candidate["taker_offset_m"],
                           candidate["taker_height_m"])
    giver_pads=[(pad,h(world.parts[part])) for pad in pad_handles(world,giver)]
    taker_pads=[(pad,h(world.parts[part])) for pad in pad_handles(world,taker)]
    with trace.stage("taker_initial_opening"):
        jaw_sweep(world,trace,taker,OPEN_M)
    stock_approach(world,trace,giver,gp(bottom))
    with trace.stage("giver_close_and_attach") as stage:
        stage["wall_fit"]=jaw_sweep(world,trace,giver,closed_width(dims[1]),
                                    at_final=lambda:contact(world,giver,part,dims,1))
        stage["attachment_position_error_m"]=float(world.attach(part,giver))
        require_gate(stage["attachment_position_error_m"]<=POSITION_LIMIT_M,"grasp moved brick")
        giver_binding=np.linalg.inv(world.tcp(giver))@world.bottom(part)
    with trace.stage("giver_vertical_lift"):
        move(world,trace,giver,translate(gp(bottom),[0.,0.,.032]),allowed=giver_pads,
             binding=(part,giver,giver_binding))
    taker_binding=None

    def giver_arrive(target):
        with trace.stage("giver_travel_to_hand_mid"):
            connect_pose(world,trace,giver,target,seeds=[hold_seed[giver]],allowed=giver_pads,
                         binding=(part,giver,giver_binding),prefer_seed_branch=True,
                         via_qs=load_park_via(giver))

    def taker_arrive(hover, target):
        with trace.stage("taker_open_approach"):
            via_qs=load_park_via(taker)
            reverse=state_for(world,taker,q=hold_seed[taker],allowed=giver_pads)
            hover_q=None
            try:
                inverse=continuous_ik(world,taker,reverse,list(near_targets(target,hover)),deadline=DEADLINE)
                if inverse.get("ok") and inverse.get("q_path"):
                    hover_q=inverse["q_path"][-1]
            except Rejected:
                hover_q=None
            if hover_q is not None:
                try:
                    high_connect(world,trace,taker,[hover_q],hover,allowed=giver_pads,via_qs=via_qs)
                except Rejected:
                    connect_pose(world,trace,taker,hover,seeds=[hold_seed[taker]],
                                 allowed=giver_pads,via_qs=via_qs)
            else:
                connect_pose(world,trace,taker,hover,seeds=[hold_seed[taker]],
                             allowed=giver_pads,via_qs=via_qs)
            move(world,trace,taker,target,allowed=giver_pads)

    def transfer():
        nonlocal taker_binding
        with trace.stage("taker_close_and_atomic_transfer") as stage:
            stage["taker_wall_fit"]=jaw_sweep(world,trace,taker,closed_width(dims[0]),preallowed=giver_pads,
                                             at_final=lambda:contact(world,taker,part,dims,0))
            stage["giver_wall_fit"],_=contact(world,giver,part,dims,1)
            before=world.bottom(part)
            stage["api_position_drift_m"]=float(world.attach(part,taker))
            pe,ae=pose_error(before,world.bottom(part))
            stage.update(world_position_drift_m=pe,world_orientation_drift_rad=ae)
            require_gate(pe<=POSITION_LIMIT_M and ae<=ORIENTATION_LIMIT_RAD
                         and stage["api_position_drift_m"]<=POSITION_LIMIT_M,"handover moved brick")
            taker_binding=np.linalg.inv(world.tcp(taker))@world.bottom(part)
            trace.sample([*giver_pads,*taker_pads],part_binding=(part,taker,taker_binding))
            stage["interarm"]=interarm(world)
            require_gate(stage["interarm"]["ok"],"verified transfer has no positive interarm clearance",
                         interarm=stage["interarm"])
            trace.key_clearance([*giver_pads,*taker_pads])

    def giver_open():
        with trace.stage("giver_open_after_transfer"):
            jaw_sweep(world,trace,giver,OPEN_M,preallowed=taker_pads)

    def giver_retreat(target, label):
        with trace.stage(label):
            move(world,trace,giver,target,allowed=taker_pads,binding=(part,taker,taker_binding))

    def verify_withdrawal():
        with trace.stage("giver_withdrawal_clearance") as stage:
            stage["interarm"]=verify_handoff_withdrawal(world)
            trace.sample(taker_pads,part_binding=(part,taker,taker_binding))
            return stage["interarm"]

    held_target=mid@np.linalg.inv(giver_binding)
    execute_handoff(mid,held_target,tp(mid),giver_arrive=giver_arrive,taker_arrive=taker_arrive,
                    transfer=transfer,giver_open=giver_open,giver_retreat=giver_retreat,
                    verify_withdrawal=verify_withdrawal,actual_bottom=lambda:world.bottom(part))
    with trace.stage("taker_carries_to_neighbor"):
        connect_pose(world,trace,taker,translate(tp(goal),[0.,0.,.032]),allowed=taker_pads,
                     binding=(part,taker,taker_binding))
    with trace.stage("taker_vertical_seat") as stage:
        move(world,trace,taker,tp(goal),allowed=taker_pads,binding=(part,taker,taker_binding))
        pe,ae=pose_error(goal,world.bottom(part))
        stage.update(target_position_error_m=pe,target_orientation_error_rad=ae)
        require_gate(pe<=POSITION_LIMIT_M and ae<=ORIENTATION_LIMIT_RAD,"neighbor seat pose mismatch",
                     position_error_m=pe,angular_error_rad=ae)
        stage["release_drift_m"]=float(world.attach(part,None))
        require_gate(stage["release_drift_m"]<=POSITION_LIMIT_M,"neighbor release moved brick")
        trace.key_clearance(taker_pads)
    with trace.stage("taker_opens_and_retreats"):
        jaw_sweep(world,trace,taker,OPEN_M)
        move(world,trace,taker,translate(tp(goal),[0.,0.,.035]))
    return True


def lifecycle_verified(value):
    return (value.get("stopped") is True and value.get("visible_windows")==[]
            and value.get("cleanup",{}).get("passed") is True)


def endpoint_candidate(historical, home_q=None):
    previous_passes=[row["candidate"] for row in historical.get("route_attempts",[]) if row.get("passed")]
    for row in previous_passes:
        if tuple(row["handover_bottom_m"])==HAND_MID:
            return row
    matches=[row for row in historical.get("arm_positives",[])
             if tuple(row["handover_bottom_m"])==HAND_MID]
    if not matches:
        return None
    if home_q is not None:
        matches.sort(key=lambda row: max(
            float(np.max(np.abs(np.asarray(row["qs_rad"][side])-np.asarray(home_q[side]))))
            for side in ("A","B")))
    return matches[0]


def wrong_handover_at_hand_mid(world, case, dims):
    with reversible(world):
        bottom=matrix(HAND_MID)
        world.set_bottom(case["part"],bottom)
        bad=pier_taker_grasp(bottom)
        for side in ("A","B"):
            set_width(world,side,dims[0])
            force_probe(world,side,bad)
        pair=interarm(world)
        rejected=world.check()
        contact_pairs=[]
        for a,b in product(pad_handles(world,"A"),pad_handles(world,"B")):
            if world.sim.checkCollision(a,b)[0]:
                contact_pairs.append([world.sim.getObjectAlias(a),world.sim.getObjectAlias(b)])
        return {"id":"wrong_handover","passed":not rejected["ok"] and not pair["ok"] and bool(contact_pairs),
                "native":rejected,"interarm":pair,"colliding_pad_pairs":contact_pairs,
                "hand_mid_bottom_m":HAND_MID,"qs_rad":{side:world.q(side).tolist() for side in ("A","B")}}


def checked_task():
    pin=load(HERE/"showcase_input_lock.json")["inputs"]["mode_b"]
    task=Path(pin["file"])
    require_gate(task.resolve()==(OUTPUT/"task.toml").resolve() and digest(task)==pin["sha256"],
                 "Mode B gate task differs from the locked task input")
    hand_off=pin["handoff"]
    contents=tomllib.loads(task.read_text(encoding="utf-8"))
    location=next(loc for loc in contents["location"] if loc["id"]=="hand_mid")
    require_gate(hand_off["revision"]==HANDOVER_REVISION and tuple(hand_off["bottom_m"])==HAND_MID
                 and location["kind"]=="handover" and tuple(location["pos"])==HAND_MID,
                 "Mode B gate task differs from the native hand_mid contract")
    return pin["sha256"]


def snapshot(world, row, exc):
    """Reproduce only this gate's recorded stop, or retain its native metadata."""
    evidence=exc.evidence
    state=evidence.get("initial_state")
    rejected_q=evidence.get("failing_q_rad")
    side=evidence.get("side")
    phase=row["stages"][-1]["phase"] if row["stages"] else "precheck"
    result={"scope":"diagnostic only","phase":phase,"reason":exc.reason}
    name=f'diagnostic_scene_{row["id"]}'
    if time.monotonic() > DEADLINE:
        return {**result,"capture_skipped":"work deadline"}
    if (evidence.get("native") or {}).get("kind")=="native_collision" and side in ("A","B") and state and rejected_q:
        pose=deepcopy(state)
        pose.pop("tcp_targets",None)
        pose.pop("expected_tcp",None)
        pose["q"][side]=rejected_q
        with reversible(world):
            applied=world.apply_state(pose)
            if applied.get("ok") is not True:
                return {**result,"reconstruction_error":applied}
            reading=world.check(allow=pose.get("allow",[]))
            if reading["ok"]:
                return {**result,"reconstruction_error":"native collision did not reproduce", "native":reading}
            expected=evidence["native"].get("pair_handles")
            if expected and sorted(expected)!=sorted(reading.get("pair_handles") or []):
                return {**result,"reconstruction_error":"native collision pair differs","native":reading}
            center=world.tcp(side)[:3,3]
            label=f'诊断现场（非成功） | {row["id"]} | {phase} | '+" / ".join(reading.get("pair") or [])
            png=capture(world.sim,OUTPUT/(name+".png"),center+[.24,-.39,.26],center,fov=42)
            caption=OUTPUT/(name+"_caption.txt")
            caption.write_text(label+"\n",encoding="utf-8")
            return {**result,"file":png,"caption":label,"caption_file":str(caption),
                    "source":"current rejected native sample", "native":reading, "state":pose}
    current=evidence.get("clearance")
    if current and current.get("kind")=="native_collision":
        allowance=evidence.get("allow",[])
        reading=world.check(allow=allowance)
        if not reading["ok"] and reading.get("pair_handles")==current.get("pair_handles"):
            centers=[world.tcp(arm)[:3,3] for arm in ("A","B")]
            center=np.mean(centers, axis=0)
            label=f'诊断现场（非成功） | {row["id"]} | {phase} | '+" / ".join(reading.get("pair") or [])
            png=capture(world.sim,OUTPUT/(name+".png"),center+[.24,-.39,.26],center,fov=42)
            caption=OUTPUT/(name+"_caption.txt")
            caption.write_text(label+"\n",encoding="utf-8")
            return {**result,"file":png,"caption":label,"caption_file":str(caption),
                    "source":"current rejected scalar sample","native":reading,"state":world.state()}
    return {**result,"capture_skipped":"no exact rejected native collision state"}


def main():
    global DEADLINE
    began=time.monotonic()
    work_end=began+WORK_SECONDS
    DEADLINE=work_end
    OUTPUT.mkdir(exist_ok=True)
    data={"passed":False,"directions":[],"negative_cases":[],"source":str(SOURCE),
          "trace":str(TRACE),"scene_sha256":V1_SHA256,"work_limit_seconds":WORK_SECONDS,
          "supervisor_hard_limit_seconds":SUPERVISOR_SECONDS,
          "handover_revision":HANDOVER_REVISION,"hand_mid_bottom_m":HAND_MID,
          "clearance_policy":"external at atomic transfer and neighbor seating, interarm before carry",
          "lifecycle":str(HERE/"native/mode_b_handover_gate_lifecycle.json")}
    atomic_json(RECEIPT,data)
    try:
        assert digest(SCENE)==load(MANIFEST)["scene_sha256"]==V1_SHA256
        data.update(source_sha256=digest(SOURCE),manifest_sha256=digest(MANIFEST),task_sha256=checked_task())
        previous=load(SOURCE)
        assert lifecycle_verified(load(HERE/"native/p0_2_height_lifecycle.json"))
        original={row["id"]:row for row in previous["directions"]}
        with scene_session("mode_b_handover_gate") as (sim,client):
            world=GateWorld(sim,client,batch_joints=True)
            home_q=world.state()["q"]
            case=CASES[0]
            with reversible(world):
                place_supports(world,case)
                dims=part_sizes(world,case["part"])
                for name,probe in (("wrong_handover",wrong_handover_at_hand_mid),
                                   ("wrong_y_grasp_at_neighbor",wrong_neighbor)):
                    try:
                        outcome=probe(world,case,dims)
                    except Rejected as error:
                        outcome={"id":name,"passed":False,"reason":error.reason,"evidence":error.evidence}
                    data["negative_cases"].append(outcome)
                    atomic_json(RECEIPT,data)
            with TRACE.open("w",encoding="utf-8",buffering=1) as events:
                for index,case in enumerate(CASES):
                    # Keep time for a live second direction even if the first times out.
                    DEADLINE=time.monotonic()+max(0.,work_end-time.monotonic())/(len(CASES)-index)
                    row={"id":case["id"],"part":case["part"],"slot":case["slot"],
                         "attempt":1,"passed":False,"stages":[],
                         "work_deadline_elapsed_s":max(0.,DEADLINE-began)}
                    data["directions"].append(row)
                    candidate=endpoint_candidate(original.get(case["id"],{}), home_q=home_q)
                    if candidate is None:
                        row["reason"]="No archived v1 native endpoint at the locked hand_mid"
                        atomic_json(RECEIPT,data)
                        continue
                    row["candidate"]=candidate
                    with reversible(world):
                        place_supports(world,case)
                        dims=part_sizes(world,case["part"])
                        try:
                            row["passed"]=run_chain(world,case,candidate,dims,GateTrace(world,row,events,data))
                        except (Rejected, ValueError) as error:
                            exc=error if isinstance(error,Rejected) else Rejected(str(error))
                            row.update(reason=exc.reason,evidence=exc.evidence)
                            print(f'{case["id"]} failed: {exc.reason}',flush=True)
                            atomic_json(RECEIPT,data)
                            try:
                                row["diagnostic"]=snapshot(world,row,exc)
                            except Exception as error:
                                row["diagnostic"]={"scope":"diagnostic only","capture_error":str(error)}
                            if "diagnostic" not in data or ("file" not in data["diagnostic"] and "file" in row["diagnostic"]):
                                data["diagnostic"]=row["diagnostic"]
                    atomic_json(RECEIPT,data)
        data["passed"]=(len(data["directions"])==2 and all(row["passed"] for row in data["directions"])
                        and len(data["negative_cases"])==2 and all(row.get("passed") for row in data["negative_cases"]))
    except Exception as exc:
        data["error"]=f"{type(exc).__name__}: {exc}"
        data["traceback"]=traceback.format_exc()
    finally:
        atomic_json(RECEIPT,data)
    print(f'Mode B gate passed={data["passed"]}',flush=True)
    return 0 if data["passed"] else 1


if __name__=="__main__":
    raise SystemExit(main())
