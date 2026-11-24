"""Stopped-scene native calibration and two isolated Mode A motion gates."""
from argparse import ArgumentParser
from copy import deepcopy
from itertools import product
import os
from pathlib import Path
from time import monotonic

import numpy as np

from bridge_geometry import BridgeWorld, CalibratedChain, _handle, _matrix
from bridge_motion import MAX_JOINT_SAMPLE_RAD, connect_free
from bridge_runtime import HERE, atomic_json, scene_session, transform


SIDES = ("A", "B")
OUTPUT = HERE / "outboard_park_a1"
NAMES = {"calibrate": "parking_calibration.json", "a007_a008": "precheck_a007_a008.json",
         "deck": "precheck_deck.json"}
ARCHIVED_PLAN_SHA = "737a6190e57797f47eed919d2d85b324b3424e74cfee7123a866682f4ac0242d"
ARCHIVED_REFERENCE_SHA = "ac4fee69cc48051ac8502fca857de16a110ba3a0a7e09ebbce739030dbad6644"

PROJECTION = r"""
local sim = require('sim')
local function lateral_range(shape)
    local pose = sim.getObjectMatrix(shape[1], sim.handle_world)
    local center, extent, rotation = shape[2], shape[3], shape[4]
    local x = pose[4] + pose[1]*center[1] + pose[2]*center[2] + pose[3]*center[3]
    local rx=2e-8
    for i=1,3 do
        local axis = pose[1]*rotation[i] + pose[2]*rotation[i+3] + pose[3]*rotation[i+6]
        rx=rx+math.abs(axis)*extent[i]
    end
    return x-rx, x+rx
end

function projectionScan(joints, shapes, candidates, plate, outward)
    local before, answer = {}, {}
    for i,h in ipairs(joints) do before[i]=sim.getJointPosition(h) end
    local plateLo, plateHi = lateral_range(plate)
    local ok, failure = xpcall(function()
        for i,q in ipairs(candidates) do
            for j,h in ipairs(joints) do sim.setJointPosition(h,q[j]) end
            local limit, boxes = math.huge, {}
            for _,shape in ipairs(shapes) do
                local lo, hi = lateral_range(shape)
                local gap = outward>0 and lo-plateHi or plateLo-hi
                limit=math.min(limit,gap)
                if #candidates==1 then boxes[#boxes+1]={shape[1],lo,hi,gap} end
            end
            answer[i]={gap_m=limit, individual=boxes}
        end
    end, debug.traceback)
    for i,h in ipairs(joints) do sim.setJointPosition(h,before[i]) end
    if not ok then error(failure) end
    return {rows=answer, plate_lo_m=plateLo, plate_hi_m=plateHi}
end
"""


def bbox_record(world, handle):
    widths, local_pose = world.sim.getShapeBB(handle)
    pose = _matrix(local_pose)
    return [handle, pose[:3, 3].tolist(), (np.asarray(widths) * .5).tolist(), pose[:3, :3].ravel().tolist()]


class ProjectionProbe:
    def __init__(self, world):
        self.world = world
        self.sim = world.sim
        board = _handle(world.manifest["board"])
        self.plate = bbox_record(world, board)
        self.plate_center_x = float((transform(self.sim, board) @ np.r_[self.plate[1], 1.])[0])
        self.shapes, self.outward = {}, {}
        for side in SIDES:
            arm = world.arms[side]
            self.outward[side] = 1 if transform(self.sim, _handle(arm["root"]))[0, 3] > self.plate_center_x else -1
            self.shapes[side] = [bbox_record(world, h) for h in map(_handle, arm["shapes"])
                                 if world.handle_roles[h]["joint_group"] > 0]
        if self.outward["A"] == self.outward["B"] or not all(self.shapes.values()):
            raise RuntimeError("v1 mounting directions or moving geometry cannot be resolved")
        self.script = self.sim.createScript(self.sim.scripttype_customization, PROJECTION, 0, "lua")
        self.sim.setObjectAlias(self.script, "LEGO_Bridge_OutboardProjection")
        self.sim.initScript(self.script)

    def scan(self, side, qs):
        joints = [_handle(row) for row in self.world.arms[side]["joints"]]
        result = self.sim.callScriptFunction("projectionScan", self.script, joints,
            self.shapes[side], qs, self.plate, self.outward[side])
        for row in result["rows"]:
            for box in row.get("individual", ()):
                box.append(self.world._names[box[0]])
        return result

    def close(self):
        self.sim.removeObjects([self.script])


def candidates(chain, home):
    offsets = (0., *[sign * step for step in range(10, 181, 10) for sign in (-1, 1)])
    forms = ((0., 0.), (-15., 0.), (15., 0.), (-30., 15.), (30., -15.),
             (-45., 20.), (45., -20.))
    answer = []
    for swivel, (shoulder, elbow) in product(offsets, forms):
        q = np.asarray(home) + np.deg2rad([swivel, shoulder, elbow, 0., 0., 0.])
        if np.all(q >= chain.lower) and np.all(q <= chain.upper):
            answer.append(q.tolist())
    return answer


def native_static(world, state, *, distance=False):
    result = world.batch_check([state], stop_at_failure=False, measure=distance,
                               filtered_clearance=distance, all_collisions=True)
    if not result["restored"] or result.get("error") or len(result["rows"]) != 1:
        raise RuntimeError(f"Incomplete native static evidence: {result}")
    return result["rows"][0]


def path_proof(world, side, start, target, seed):
    result = connect_free(world, side, start, target, timeout_s=18., seed=seed)
    path = np.asarray(result["q_path"])
    result["endpoints_and_spacing_passed"] = bool(
        result["ok"] and np.max(np.abs(path[0]-start["q"][side])) <= 1e-7
        and np.max(np.abs(path[-1]-target)) <= 1e-7
        and (len(path) < 2 or np.max(np.abs(np.diff(path, axis=0))) <= MAX_JOINT_SAMPLE_RAD+1e-8))
    return result


def maximum_open_state(world, parked):
    opened = deepcopy(parked)
    opened["jaw"] = {side: world.arms[side]["jaw_interval_m"][1] for side in SIDES}
    # The commanded opening also changes the expected follower readback.
    opened["follower"] = dict(opened["jaw"])
    return opened


def calibrate(world, output):
    started = monotonic()
    initial = world.state()
    probe = ProjectionProbe(world)
    trials = []
    report = {"passed": False, "scene_sha256": world.manifest["scene_sha256"],
              "home_q": initial["q"], "candidate_trials": trials,
              "tested_shape_pairs": world.query_registration["pair_count"]}
    atomic_json(output, report)
    try:
        report["home_static"] = native_static(world, initial)
        if not report["home_static"]["ok"]:
            return report
        options = {}
        for side in SIDES:
            width = world.arms[side]["jaw_interval_m"][1]
            world.jaw(side, width)
            try:
                qs = candidates(world.chains[side], initial["q"][side])
                projection = probe.scan(side, qs)
            finally:
                world.jaw(side, initial["jaw"][side])
            outside = [(q, row["gap_m"]) for q, row in zip(qs, projection["rows"], strict=True)
                       if row["gap_m"] > 0]
            options[side] = sorted(outside, key=lambda row: (
                0 if row[1] >= 0.12 else 1,
                -row[1] if row[1] < 0.12 else 0.,
                np.max(np.abs(np.asarray(row[0])-initial["q"][side]))))
            report.setdefault("projection_screen", {})[side] = {
                "generated": len(qs), "strict_outboard": len(outside),
                "evaluated_jaw_m": width, "moving_shape_count": len(probe.shapes[side]),
                "outward_x_sign": probe.outward[side],
                "plate_x_m": [projection["plate_lo_m"], projection["plate_hi_m"]]}
        if not all(options.values()):
            report["reason"] = "no candidate puts every complete moving-link/tool bound outboard"
            return report
        validated = {}
        for side, rows in options.items():
            endpoints = []
            for q, gap in rows[:14]:
                snap = deepcopy(initial)
                snap["q"][side] = q
                witness = native_static(world, snap)
                endpoints.append({"q": q, "gap_m": gap, "native": witness})
            validated[side] = [row for row in endpoints if row["native"]["ok"]]
            report.setdefault("native_single_endpoints", {})[side] = endpoints
        for choice in product(*(validated[side] for side in SIDES)):
            trial_index = len(trials)
            if trial_index >= 12 or monotonic()-started >= 210.:
                break
            parked = deepcopy(initial)
            parked["q"] = {side: row["q"] for side, row in zip(SIDES, choice)}
            opened = maximum_open_state(world, parked)
            pair = native_static(world, parked)
            pair_open = native_static(world, opened)
            trial = {"park_q": parked["q"], "both_parked": pair, "both_parked_max_open": pair_open,
                     "transition_paths": {}}
            trials.append(trial)
            if not pair["ok"] or not pair_open["ok"]:
                continue
            for side in SIDES:
                # Hold the other arm at its verified outboard pose throughout
                # both actual edges. Also prove the first departure from both home.
                other_home = deepcopy(initial)
                moving_home = deepcopy(parked)
                moving_home["q"][side] = initial["q"][side]
                connections = (("home_to_park_peer_home", other_home, parked["q"][side]),
                               ("home_to_park_peer_outboard", moving_home, parked["q"][side]),
                               ("park_to_home_peer_outboard", parked, initial["q"][side]))
                paths = {}
                trial["transition_paths"][side] = paths
                for index, (label, state, target) in enumerate(connections):
                    if monotonic()-started >= 222.:
                        break
                    paths[label] = path_proof(world, side, state, target, 721+trial_index+20*index)
                    print("Outboard v1 transition", trial_index, side, label, paths[label]["ok"], flush=True)
                    atomic_json(output, {**report, "wall_seconds": monotonic()-started})
                    if not paths[label]["endpoints_and_spacing_passed"]:
                        break
                if len(paths) != len(connections) or not all(path["endpoints_and_spacing_passed"] for path in paths.values()):
                    break
            atomic_json(output, {**report, "wall_seconds": monotonic()-started})
            if (set(trial["transition_paths"]) != set(SIDES)
                    or not all(len(paths)==3 and all(path["endpoints_and_spacing_passed"] for path in paths.values())
                               for paths in trial["transition_paths"].values())):
                continue
            evidence = {}
            for side in SIDES:
                world.jaw(side, opened["jaw"][side])
                try:
                    result = probe.scan(side, [parked["q"][side]])
                finally:
                    world.jaw(side, initial["jaw"][side])
                row = result["rows"][0]
                evidence[side] = {"minimum_outboard_gap_m": row["gap_m"], "individual_shape_x_bounds": row["individual"],
                                  "outward_x_sign": probe.outward[side]}
            report.update(passed=all(row["minimum_outboard_gap_m"] > 0 for row in evidence.values()),
                          park_q=parked["q"], geometry_evidence=evidence, transition_paths=trial["transition_paths"],
                          both_parked=pair, both_parked_max_open=pair_open)
            return report
        report["reason"] = "no candidate pair passed both static geometry and the recorded return/departure paths"
        return report
    finally:
        probe.close()
        restored = world.state()
        report["restored"] = same_physical_state(initial, restored)
        report["restored_measurements"] = {
            "maximum_joint_error_rad": max(float(np.max(np.abs(np.asarray(initial["q"][side])-restored["q"][side])))
                                               for side in SIDES),
            "maximum_jaw_error_m": max(abs(initial["jaw"][side]-restored["jaw"][side]) for side in SIDES),
            "maximum_part_matrix_delta": max(float(np.max(np.abs(np.asarray(row["pose"])-restored["parts"][name]["pose"])))
                                                 for name, row in initial["parts"].items()),
        }
        report["passed"] = report["passed"] and report["restored"]
        report["wall_seconds"] = monotonic()-started


def prior_plan(world):
    from lego_showcase_runner import digest, parse_actions, read

    prior = Path(os.environ["BRIDGE_PRIOR_MODE_A"]).resolve()
    if not prior.is_relative_to(HERE) or not {"mode_a", "previous_attempt"} <= set(prior.parts):
        raise RuntimeError(f"Expected the isolated archived Mode A source, got {prior}")
    plan_path, ref_path = prior / "plan.json", prior / "reference.json"
    reference, plan = read(ref_path), read(plan_path)
    plan_sha, reference_sha = digest(plan_path), digest(ref_path)
    if (plan_sha != ARCHIVED_PLAN_SHA or reference_sha != ARCHIVED_REFERENCE_SHA
            or reference["plan_sha256"] != plan_sha or reference["scene_sha256"] != world.manifest["scene_sha256"]):
        raise RuntimeError("Archived plan, reference and v1 scene fingerprints disagree")
    actions = parse_actions(plan)
    committed = reference["actions_committed"]
    if len(actions) != 18 or len(committed) != 6 or [row["action"] for row in committed] != actions[:6]:
        raise RuntimeError("Archived Mode A is not the witnessed committed a006 prefix")
    prior_state = committed[-1]["native_state"]
    commanded = deepcopy(reference["current"])
    # Deposits encode an implicit home parent in the command. Verify every
    # explicit claim against the separately captured native a006 snapshot.
    implicit_parents = []
    for name, item in commanded["parts"].items():
        if "parent" not in item:
            item["parent"] = prior_state["parts"][name]["parent"]
            implicit_parents.append(name)
    if not same_physical_state(prior_state, commanded):
        raise RuntimeError("Archived current state is different from committed a006")
    evidence = {"plan": str(plan_path), "reference": str(ref_path), "plan_sha256": plan_sha,
                "reference_sha256": reference_sha, "scene_sha256": reference["scene_sha256"],
                "physical_source": "actions_committed[a006].native_state",
                "command_implicit_parents_checked": sorted(implicit_parents),
                "committed_actions": [row["action"]["id"] for row in committed]}
    return actions, {**reference, "current": prior_state}, evidence


def same_physical_state(expected, actual):
    for field in ("q", "jaw", "follower"):
        if field not in expected or field not in actual:
            continue
        for side in SIDES:
            if np.max(np.abs(np.asarray(expected[field][side])-actual[field][side])) > 1e-8:
                return False
    for name, row in expected["parts"].items():
        observed = actual["parts"][name]
        if row.get("owner") != observed.get("owner") or row.get("parent") != observed.get("parent"):
            return False
        for field in ("pose", "binding"):
            if field in row and np.max(np.abs(np.asarray(row[field])-observed[field])) > 1e-8:
                return False
    return True


def load_calibration(world, directory):
    from lego_showcase_runner import read

    data = read(directory / NAMES["calibrate"])
    if not data.get("passed") or data["scene_sha256"] != world.manifest["scene_sha256"]:
        raise RuntimeError("Outboard calibration gate did not pass on this v1 scene")
    return data


def run_action(trace, action):
    from lego_showcase_runner import NativeBlocker, apply

    trace.action_id = action["id"]
    start, before = len(trace.tasks), deepcopy(trace.state)
    try:
        getattr(trace, action["kind"])(action)
    except NativeBlocker as error:
        # Preserve the first newly tried primitive and contact evidence, while
        # leaving the simulator at the verified action entry for later gates.
        partial = trace.tasks[start:]
        del trace.tasks[start:]
        trace.state = before
        apply(trace.world, before)
        return {"passed": False, "action": action, "failure": str(error), "detail": error.detail,
                "replayable_prefix": partial, "rollback_restored": True}
    native_state = trace.world.state()
    added = trace.tasks[start:]
    trace.committed.append({"action": action, "trajectory_ids": [task["id"] for task in added],
                            "native_state": native_state})
    return {"passed": True, "action": action, "new_primitives": [task["metadata"]["primitive"] for task in added],
            "native_state": native_state}


def a007_a008(world, output):
    from lego_showcase_runner import MotionTrace, apply

    started = monotonic()
    actions, reference, prior = prior_plan(world)
    calibration = load_calibration(world, output.parent)
    folder = output.parent / "a007_a008_native"
    folder.mkdir(exist_ok=True)
    trace = MotionTrace(world, "mode_a", folder, actions, prior["plan_sha256"], parking_calibration=calibration)
    trace.state = deepcopy(reference["current"])
    trace.tasks = deepcopy(reference["tasks"])
    trace.committed = deepcopy(reference["actions_committed"])
    trace.variants = deepcopy(reference.get("grasp_variants", {}))
    prefix_count = len(trace.tasks)
    apply(world, trace.state)
    if not same_physical_state(trace.state, world.state()):
        raise RuntimeError("Native geometry does not reproduce exact a006 scene state")
    before = native_static(world, trace.state)
    report = {"passed": False, "source": prior, "a006_restored": True, "a006_static": before,
              "actions": [], "compiler_reference": str(trace.trace)}
    if not before["ok"]:
        return report
    for action in actions[6:8]:
        print("Archived a006 motion precheck:", action["id"], action["kind"], action["arm"], flush=True)
        row = run_action(trace, action)
        report["actions"].append(row)
        report["wall_seconds"] = monotonic()-started
        trace.receipt("planning" if row["passed"] else "blocked", failure=None if row["passed"] else row)
        atomic_json(output, report)
        if not row["passed"]:
            report["first_blocker"] = row
            return report
    old_part = actions[6]["part"]
    expected = np.asarray(world.parts[old_part]["target_bottom_m"])
    position_error = float(np.linalg.norm(world.bottom(old_part)[:3, 3]-expected))
    primitives = trace.tasks[prefix_count:]
    shared = [task for task in primitives if task["metadata"]["primitive"] == "shared_ingress_idle_park"]
    joint_side = "B" if actions[6]["arm"] == "A" else "A"
    matching = [task for task in shared if task["arm"] == joint_side
                and np.max(np.abs(np.asarray(task["q_path"][-1])-calibration["park_q"][joint_side])) <= 1e-7]
    joins = [task for task in primitives if task["metadata"]["primitive"] == "approach_pick_free_join"]
    post_place = [task for task in primitives if task["metadata"]["primitive"] == "post_place_park"
                  and np.max(np.abs(np.asarray(task["q_path"][-1])-calibration["park_q"][task["arm"]])) <= 1e-7]
    ending = native_static(world, world.state())
    automatic = len(matching) == 1 and bool(joins) and matching[0]["reference_order"] < joins[0]["reference_order"]
    report.update(first_blocker=None, automatic_outboard_before_pick=automatic, single_post_place_outboard=len(post_place)==1,
                  part_position_error_m=position_error, final_static=ending,
                  outboard_candidates=[task["id"] for task in matching],
                  wall_seconds=monotonic()-started,
                  passed=bool(automatic and len(post_place)==1 and position_error <= 1e-5 and ending["ok"]))
    trace.receipt("complete" if report["passed"] else "blocked")
    return report


def deck_fixture(world, part, calibration):
    from lego_showcase_runner import apply, matrix

    initial = world.state()
    requested = deepcopy(initial)
    requested["q"] = deepcopy(calibration["park_q"])
    for name in set(world.parts)-{part}:
        target = matrix(world.parts[name]["target_bottom_m"]) @ world.offsets[name]
        requested["parts"][name]["pose"] = target.tolist()
    apply(world, requested)
    observed = world.state()
    if not same_physical_state(requested, observed):
        raise RuntimeError("Real scene did not restore parked arms, deck stock and the eight completed pier meshes")
    evidence = {"fixture_only": True, "actual_original_pier_meshes": {}, "native_static": native_static(world, observed)}
    for name in sorted(set(world.parts)-{part}):
        distance, orientation = CalibratedChain.pose_error(world.bottom(name), matrix(world.parts[name]["target_bottom_m"]))
        evidence["actual_original_pier_meshes"][name] = {"position_error_m": distance, "rotation_error_rad": orientation}
    if not evidence["native_static"]["ok"] or any(row["position_error_m"] > 1e-5 or row["rotation_error_rad"] > 2e-4
                                                      for row in evidence["actual_original_pier_meshes"].values()):
        raise RuntimeError(f"Precheck deck fixture does not match v1 original geometry: {evidence}")
    return initial, observed, evidence


def deck(world, output):
    from lego_showcase_runner import DECK, MotionTrace, NativeBlocker, apply, matrix, validate_support

    started = monotonic()
    actions, _, source = prior_plan(world)
    calibration = load_calibration(world, output.parent)
    actions = deepcopy(actions[-2:])
    if actions[0]["kind"] != "pick" or actions[1]["kind"] != "place" or any(row["part"] != DECK for row in actions):
        raise RuntimeError("Final two archived symbolic actions are not the deck pick and place")
    action_ids = {action["id"] for action in actions}
    for action in actions:
        action["dependencies"] = [needs for needs in action["dependencies"] if needs in action_ids]
    initial, fixture, evidence = deck_fixture(world, DECK, calibration)
    report = {"passed": False, "source": source, "scene_sha256": world.manifest["scene_sha256"],
              "physical_fixture": evidence, "trials": [],
              "selection_rule": "first direction that passes entire same-binding native chain, support and fixed-home return"}
    for direction in (1, -1):
        folder = output.parent / ("deck_dir_pos" if direction > 0 else "deck_dir_neg")
        folder.mkdir(exist_ok=True)
        apply(world, initial)
        trace = MotionTrace(world, "mode_a", folder, actions, source["plan_sha256"],
                            parking_calibration=calibration, deck_direction=direction, fixture_context=True)
        apply(world, fixture)
        trace.state = deepcopy(fixture)
        trial = {"direction": direction, "passed": False, "actions": [], "reference": str(trace.trace)}
        report["trials"].append(trial)
        begun = monotonic()
        print("Native deck pick/place orientation:", direction, flush=True)
        atomic_json(output, {**report, "wall_seconds": monotonic()-started})
        try:
            for action in actions:
                row = run_action(trace, action)
                trial["actions"].append(row)
                trace.receipt("planning" if row["passed"] else "blocked", failure=None if row["passed"] else row)
                atomic_json(output, {**report, "wall_seconds": monotonic()-started})
                if not row["passed"]:
                    trial["first_blocker"] = row
                    break
            if len(trial["actions"]) != 2 or not all(row["passed"] for row in trial["actions"]):
                continue
            pick, place = trial["actions"]
            assigned = actions[0]["arm"]
            held = pick["native_state"]["parts"][DECK]
            trial["selected_binding"] = held.get("binding")
            trial["selected_direction"] = trace.variants.get(DECK)
            transfers = [task["metadata"]["effect"] for task in trace.tasks
                         if task["metadata"].get("effect") and task["part"] == DECK]
            binding_ok = (len(transfers)==2 and held.get("owner")==assigned
                          and np.max(np.abs(np.asarray(transfers[0]["after"]["binding"])-trial["selected_binding"])) <= 1e-8
                          and np.max(np.abs(np.asarray(transfers[1]["before"]["binding"])-trial["selected_binding"])) <= 1e-8
                          and place["native_state"]["parts"][DECK]["owner"] is None)
            trial["constant_binding_passed"] = binding_ok
            distance, orientation = CalibratedChain.pose_error(world.bottom(DECK), matrix(world.parts[DECK]["target_bottom_m"]))
            trial["deck_seat_position_error_m"] = distance
            trial["deck_seat_orientation_error_rad"] = orientation
            trial["supports"] = validate_support(world, DECK)
            trace.action_id = None
            for side in SIDES:
                if np.max(np.abs(np.asarray(trace.state["q"][side])-calibration["home_q"][side])) > 1e-7:
                    trace.park(side, "home", "precheck_deck_return_home")
            trial["home_error_rad"] = {side:float(np.max(np.abs(np.asarray(world.q(side))-calibration["home_q"][side])))
                                       for side in SIDES}
            trial["final_static"] = native_static(world, world.state())
            trial["passed"] = bool(binding_ok and trial["selected_direction"] == direction and distance <= 1e-5
                                   and orientation <= 2e-4 and max(trial["home_error_rad"].values()) <= 1e-7
                                   and trial["final_static"]["ok"])
            if not trial["passed"]:
                trial["first_blocker"] = {"failure": "final binding, seating, fixed home or collision validation rejected"}
            trace.receipt("complete" if trial["passed"] else "blocked")
        except NativeBlocker as error:
            trial["first_blocker"] = {"failure": str(error), "detail": error.detail}
            trace.receipt("blocked", failure=trial["first_blocker"])
        finally:
            trial["wall_seconds"] = monotonic()-begun
            atomic_json(output, {**report, "wall_seconds": monotonic()-started})
        if trial["passed"]:
            break
    successful = [row for row in report["trials"] if row["passed"]]
    if successful:
        chosen = successful[0]
        report.update(passed=True, selected_direction=chosen["direction"], selected_binding=chosen["selected_binding"],
                      selected_reference=chosen["reference"], selected_arm=actions[0]["arm"])
    else:
        report["first_blocker"] = next((row["first_blocker"] for row in report["trials"] if row.get("first_blocker")), None)
    apply(world, initial)
    report["restored"] = same_physical_state(initial, world.state())
    report["passed"] = bool(report["passed"] and report["restored"])
    report["wall_seconds"] = monotonic()-started
    return report


def main():
    parser = ArgumentParser()
    parser.add_argument("stage", choices=NAMES)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    options = parser.parse_args()
    directory = options.output_dir.resolve()
    if not directory.is_relative_to(HERE):
        parser.error("diagnostic output must be inside the LEGO showcase")
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / NAMES[options.stage]
    handler = {"calibrate": calibrate, "a007_a008": a007_a008, "deck": deck}[options.stage]
    try:
        with scene_session("outboard_"+options.stage) as (sim, client):
            report = handler(BridgeWorld(sim, client, batch_joints=True), output)
    except Exception as error:
        from lego_showcase_runner import read

        report = read(output) if output.exists() else {}
        report.update(passed=False, stage=options.stage, environment_or_fixture_exception=repr(error))
        atomic_json(output, report)
        raise
    atomic_json(output, report)
    print(options.stage, "passed=" + str(report["passed"]), output, flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
