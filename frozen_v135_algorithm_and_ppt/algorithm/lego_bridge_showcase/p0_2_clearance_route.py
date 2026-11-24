"""Bounded high-stock approach for saved native height-separated handoff targets."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import traceback

import numpy as np

from bridge_geometry import BridgeWorld
from bridge_runtime import HERE, atomic_json, capture, scene_session
from lego_grasp_model import linear_targets, matrix, pose_error, translate
from p0_2_handover import (
    CASES, OPEN_M, ORIENTATION_LIMIT_RAD, POSITION_LIMIT_M, MAX_JOINT_STEP_RAD,
    Rejected, RunTrace, closed_width, contact, continuous_move, final_bottom, h, interarm,
    jaw_sweep, pad_handles, part_sizes, place_supports, require_gate, reversible,
)
from p0_2_height_taker import giver_pose, taker_pose

HEIGHT = .245
MAX_JOIN_SAMPLES = 2300
TRACE = HERE/"p0_2_clearance_route_samples.jsonl"
RECEIPT = HERE/"p0_2_clearance_route_receipt.json"
SOURCE = HERE/"p0_2_height_receipt.json"


def fixed_pose(world, side, expected):
    actual = world.tcp(side)
    pe, re = pose_error(expected, actual)
    require_gate(pe <= POSITION_LIMIT_M and re <= ORIENTATION_LIMIT_RAD,
                 "native stock waypoint did not match its requested pose",
                 arm=side, position_error_m=pe, orientation_error_rad=re)
    return actual


def stock_column(world, side, grasp, evidence):
    """Probe pickup branches backward, in a disposable stopped fixture."""
    start = world.q(side)
    high = grasp.copy()
    high[2, 3] = HEIGHT
    targets = list(linear_targets(grasp, high, step_m=.00090))
    solutions = world.ik(side, grasp, seed=start, seed_count=36, max_solutions=8)
    evidence.update(endpoint_ik_count=len(solutions), tested=[])
    for rank, q in enumerate(solutions):
        row = {"rank": rank, "pickup_q_rad": q.tolist(), "samples": 0, "passed": False}
        evidence["tested"].append(row)
        with reversible(world):
            try:
                world.setq(side, q)
                measured = fixed_pose(world, side, grasp)
                hit = world.check()
                require_gate(hit["ok"], "stock endpoint failed native geometry", clearance=hit)
                column = [(measured.copy(), q.copy())]
                for target in targets:
                    qs = world.ik(side, target, seed=q, seed_count=1, max_solutions=1)
                    require_gate(bool(qs), "stock column failed continuous IK",
                                 target_xyz_m=target[:3, 3].tolist(), q_rad=q.tolist())
                    follow = qs[0]
                    require_gate(np.max(np.abs(follow-q)) <= MAX_JOINT_STEP_RAD,
                                 "stock column would switch branches",
                                 target_xyz_m=target[:3, 3].tolist(), q_rad=q.tolist())
                    world.setq(side, follow)
                    actual = fixed_pose(world, side, target)
                    delta = float(np.linalg.norm(measured[:3, 3]-actual[:3, 3]))
                    require_gate(delta <= .001, "measured stock column step exceeds 1 mm",
                                 tcp_step_m=delta)
                    hit = world.check()
                    row["samples"] += 1
                    require_gate(hit["ok"], "stock column reached native collision",
                                 target_xyz_m=target[:3, 3].tolist(), q_rad=follow.tolist(),
                                 clearance=hit)
                    column.append((actual.copy(), follow.copy()))
                    measured, q = actual, follow
                row.update(passed=True, high_q_rad=q.tolist(), high_target_m=high[:3, 3].tolist())
                return column
            except Rejected as error:
                row.update(reason=error.reason, evidence=error.evidence)
    raise Rejected("no tested stock branch connected from exact grasp up to high hover", evidence)


def joint_waypoints(chain, start, target):
    """Generate bounded joint-linear subdivisions before a native check."""
    start, target = np.asarray(start, dtype=float), np.asarray(target, dtype=float)
    raw = [(start, target, chain.fk(start)[0], chain.fk(target)[0])]
    result = []
    while raw:
        q0, q1, p0, p1 = raw.pop()
        disp, angle = pose_error(p0, p1)
        if disp <= .00086 and angle <= np.deg2rad(.45) and np.max(np.abs(q1-q0)) <= .025:
            result.append(q1)
            if len(result) + len(raw) > MAX_JOIN_SAMPLES:
                raise Rejected("high joint connector exceeds sample cap",
                               {"cap": MAX_JOIN_SAMPLES})
            continue
        mid = (q0+q1)/2
        at_mid = chain.fk(mid)[0]
        raw.append((mid, q1, at_mid, p1))
        raw.append((q0, mid, p0, at_mid))
    return result


def checked_join(world, trace, side, target_q, target_pose):
    """Actual checked joint interpolation, never a branch teleport."""
    at = world.tcp(side)
    waypoints = joint_waypoints(world.chains[side], world.q(side), target_q)
    for q in waypoints:
        world.setq(side, q)
        reached = world.tcp(side)
        delta = float(np.linalg.norm(reached[:3, 3]-at[:3, 3]))
        require_gate(delta <= .001, "high join measured TCP step exceeds 1 mm", tcp_step_m=delta)
        trace.sample(command={"moving_arm": side, "goal_xyz_m": reached[:3, 3].tolist(),
                              "tcp_step_m": delta, "interpolation": "q"})
        at = reached
    fixed_pose(world, side, target_pose)


def open_approach(world, trace, side, grasp):
    with trace.stage("giver_open") as row:
        jaw_sweep(world, trace, side, OPEN_M)
    row = {"phase": "giver_stock_branch_precheck", "scope": "reversible branch fixture, not travelled"}
    trace.output["stages"].append(row)
    try:
        column = stock_column(world, side, grasp, row)
        row["passed"] = True
    except Rejected as error:
        row.update(passed=False, reason=error.reason)
        raise

    high_pose, hover_q = column[-1]
    entry_q = world.q(side).copy()
    start = world.tcp(side)
    lifted = start.copy()
    lifted[2, 3] = HEIGHT

    with trace.stage("giver_high_vertical_lift"):
        continuous_move(world, trace, side, lifted)

    # Both connector modes are separately logged. A failed probe retains
    # its own native collision row and is excluded from the accepted route.
    at_lift = world.q(side).copy()
    along = lifted.copy()
    along[:2, 3] = high_pose[:2, 3]
    for mode in ("high_cartesian_then_branch", "high_direct_joint_branch"):
        try:
            with trace.stage(mode) as join:
                join["initial_q_rad"] = entry_q.tolist()
                join["high_branch_q_rad"] = hover_q.tolist()
                if mode.startswith("high_cartesian"):
                    continuous_move(world, trace, side, along)
                    continuous_move(world, trace, side, high_pose)
                checked_join(world, trace, side, hover_q, high_pose)
                join["selected_connector"] = True
            break
        except Rejected:
            join["disposable_rejected_connector"] = True
            if mode == "high_direct_joint_branch":
                raise
            # The rejected connector is rolled back outside the accepted
            # transcript; the second test restarts at the checked high lift.
            world.setq(side, at_lift)
    with trace.stage("giver_prechecked_vertical_stock_descent") as descent:
        before = world.tcp(side)
        for expected, q in reversed(column[:-1]):
            world.setq(side, q)
            actual = fixed_pose(world, side, expected)
            step = float(np.linalg.norm(before[:3, 3]-actual[:3, 3]))
            require_gate(step <= .001, "replayed native descent exceeds 1 mm", tcp_step_m=step)
            trace.sample(command={"moving_arm": side, "goal_xyz_m": expected[:3, 3].tolist(),
                                  "tcp_step_m": step, "interpolation": "validated stock column"})
            before = actual
        descent["stock_end_q_rad"] = world.q(side).tolist()


def positive_chain(world, case, candidate, dims, trace, allow_preview):
    giver, taker, part = (case[key] for key in ("giver", "taker", "part"))
    bottom0 = world.bottom(part)
    hand_middle = matrix(candidate["handover_bottom_m"])
    goal = final_bottom(world, part)
    tilt = candidate["tilt_deg"]
    gp = lambda bottom: giver_pose(bottom, tilt, case["incline"], candidate["giver_height_m"], candidate["giver_outboard_m"])
    tp = lambda bottom: taker_pose(bottom, candidate["taker_tilt_deg"], candidate["taker_offset_m"], candidate["taker_height_m"])
    giver_pad_allow = [(pad, h(world.parts[part])) for pad in pad_handles(world, giver)]
    taker_pad_allow = [(pad, h(world.parts[part])) for pad in pad_handles(world, taker)]
    with trace.stage("taker_initial_opening"):
        jaw_sweep(world, trace, taker, OPEN_M)
    open_approach(world, trace, giver, gp(bottom0))
    with trace.stage("giver_close_and_attach") as stage:
        stage["wall_fit"] = jaw_sweep(world, trace, giver, closed_width(dims[1]),
                                      at_final=lambda: contact(world, giver, part, dims, 1))
        stage["attachment_position_error_m"] = float(world.attach(part, giver))
        require_gate(stage["attachment_position_error_m"] <= POSITION_LIMIT_M,
                "first grip changed the world-frame brick pose")
        g_binding = np.linalg.inv(world.tcp(giver)) @ world.bottom(part)
    with trace.stage("giver_vertical_lift"):
        continuous_move(world, trace, giver, translate(gp(bottom0), [0., 0., .032]),
                        allowed=giver_pad_allow, binding=(part, giver, g_binding))
    with trace.stage("giver_travel_to_hand_mid"):
        continuous_move(world, trace, giver, gp(hand_middle), allowed=giver_pad_allow,
                        binding=(part, giver, g_binding))
    with trace.stage("taker_open_approach"):
        jaw_sweep(world, trace, taker, OPEN_M, preallowed=giver_pad_allow)
        # Approach downward using the same world-X closing faces as final seat.
        approach = translate(tp(hand_middle), [0., 0., .025])
        continuous_move(world, trace, taker, approach, allowed=giver_pad_allow)
        continuous_move(world, trace, taker, tp(hand_middle), allowed=giver_pad_allow)
    with trace.stage("taker_close_and_atomic_transfer") as stage:
        stage["taker_wall_fit"] = jaw_sweep(world, trace, taker, closed_width(dims[0]), preallowed=giver_pad_allow,
            at_final=lambda: contact(world, taker, part, dims, 0))
        stage["giver_wall_fit"], _ = contact(world, giver, part, dims, 1)
        stage["interarm"] = interarm(world)
        require_gate(stage["interarm"]["ok"], "orthogonal handover lacked positive interarm clearance",
                interarm=stage["interarm"])
        before = world.bottom(part)
        stage["api_position_drift_m"] = float(world.attach(part, taker))
        position, orientation = pose_error(before, world.bottom(part))
        stage["world_position_drift_m"], stage["world_orientation_drift_rad"] = position, orientation
        require_gate(position <= POSITION_LIMIT_M and orientation <= ORIENTATION_LIMIT_RAD
                and stage["api_position_drift_m"] <= POSITION_LIMIT_M, "handover changed brick world pose")
        t_binding = np.linalg.inv(world.tcp(taker)) @ world.bottom(part)
        trace.sample([*giver_pad_allow, *taker_pad_allow], part_binding=(part, taker, t_binding))
        if allow_preview:
            center = before[:3, 3]
            try:
                stage["preview"] = capture(world.sim, HERE/"p0_previews"/f"hand_mid_clearance_{case['id']}.png",
                    center + np.array([.22, -.31, .19]), center + np.array([0., 0., .025]), fov=37)
            except (RuntimeError, ValueError) as error:
                stage["preview_error"] = str(error)
    with trace.stage("giver_open_and_retreat"):
        jaw_sweep(world, trace, giver, OPEN_M, preallowed=taker_pad_allow)
        safe = world.tcp(giver)
        retreat = translate(safe, -safe[:3, 2]*.039)
        continuous_move(world, trace, giver, retreat, allowed=taker_pad_allow)
        exterior = translate(retreat, [-case["incline"]*.06, 0., .018])
        continuous_move(world, trace, giver, exterior, allowed=taker_pad_allow)
    with trace.stage("taker_carries_to_neighbor"):
        approach = translate(tp(goal), [0., 0., .032])
        continuous_move(world, trace, taker, approach, allowed=taker_pad_allow,
                        binding=(part, taker, t_binding))
    with trace.stage("taker_vertical_seat") as stage:
        continuous_move(world, trace, taker, tp(goal), allowed=taker_pad_allow,
                        binding=(part, taker, t_binding))
        delta, orient = pose_error(goal, world.bottom(part))
        stage["target_position_error_m"], stage["target_orientation_error_rad"] = delta, orient
        require_gate(delta <= POSITION_LIMIT_M and orient <= ORIENTATION_LIMIT_RAD,
                "brick did not reach the independently extracted target pose",
                position_error_m=delta, orientation_error_rad=orient)
        stage["release_drift_m"] = float(world.attach(part, None))
        require_gate(stage["release_drift_m"] <= POSITION_LIMIT_M, "release changed the seated world pose")
    with trace.stage("taker_opens_and_retreats"):
        jaw_sweep(world, trace, taker, OPEN_M)
        continuous_move(world, trace, taker, translate(tp(goal), [0., 0., .035]))
    return True


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction", choices=["A_to_B", "B_to_A"])
    args = parser.parse_args()
    prior = json.loads(SOURCE.read_text(encoding="utf-8-sig"))
    candidates = {row["id"]: row for row in prior["directions"]}
    lifecycle = json.loads((HERE/"native/p0_2_height_lifecycle.json").read_text(encoding="utf-8-sig"))
    if not lifecycle.get("cleanup", {}).get("passed"):
        raise RuntimeError("Height screen must release its original simulator before route validation")
    data = {"phase": "P0-2 checked high stock route", "passed": False, "directions": [],
            "source_receipt": str(SOURCE), "source_receipt_sha256": digest(SOURCE),
            "scene_sha256": digest(HERE/"assets/lego_bridge_kinematic_v1.ttt"),
            "scene_manifest_sha256": digest(HERE/"assets/scene_manifest.json"),
            "trace_path": str(TRACE), "lifecycle": str(HERE/"native/p0_2_clearance_route_lifecycle.json")}
    atomic_json(RECEIPT, data)
    try:
        with scene_session("p0_2_clearance_route") as (sim, client):
            world = BridgeWorld(sim, client)
            with TRACE.open("w", encoding="utf-8", buffering=1) as events:
                for case in CASES:
                    if args.direction and case["id"] != args.direction:
                        continue
                    earlier = candidates.get(case["id"], {})
                    result = {"id": case["id"], "part": case["part"], "slot": case["slot"],
                              "attempt": 1, "passed": False, "stages": []}
                    data["directions"].append(result)
                    if earlier.get("passed"):
                        result.update(passed=True, reuse_height_complete_route=True)
                        continue
                    positive = earlier.get("arm_positives") or []
                    if not positive:
                        result["reason"] = "prior height-separated run had no full native endpoint for this direction"
                        continue
                    candidate = positive[0]
                    result["candidate"] = candidate
                    print(case["id"] + " validating alternate stock clearance on saved native endpoint", flush=True)
                    with reversible(world):
                        place_supports(world, case)
                        dims = part_sizes(world, case["part"])
                        try:
                            result["passed"] = positive_chain(world, case, candidate, dims,
                                RunTrace(world, result, events), allow_preview=True)
                        except Rejected as error:
                            result.update(reason=error.reason, evidence=error.evidence)
                            print(case["id"] + " alternate route blocked: " + error.reason, flush=True)
                    atomic_json(RECEIPT, data)
            data["passed"] = bool(data["directions"]) and all(row["passed"] for row in data["directions"])
    except Exception as error:
        data.update(error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise
    finally:
        atomic_json(RECEIPT, data)
    return 0 if data["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(run())
