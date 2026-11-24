"""Isolated rerun of Phase 0 handoff with X-axis taker inclination."""
from contextlib import contextmanager
from itertools import product
from pathlib import Path
import math
import traceback

import numpy as np

from bridge_geometry import BridgeWorld
from bridge_runtime import HERE, atomic_json, capture, scene_session, set_transform, transform
from lego_grasp_model import matrix, pier_giver_grasp, pose_error, translate
from p0_2_handover import (
    CASES, HANDOVER_CENTERS_M, OPEN_M, ORIENTATION_LIMIT_RAD, POSITION_LIMIT_M,
    TILT_ANGLES_DEG, Rejected, RunTrace, closed_width, collect, contact,
    continuous_move, endpoints, final_bottom, h, interarm, jaw_sweep, pad_handles,
    part_sizes, place_supports, require_gate, reversible, set_width,
)

RECEIPT = HERE/"p0_2_tilted_receipt.json"
TAKER_TILTS_DEG = (35., 45., 55., 65., 70.)
TAKER_OFFSETS_M = (.003, .006, .008)
TAKER_HEIGHT_M = .0087


def taker_pose(bottom, inclination_deg, pad_offset_m):
    """Preserve the opposed X faces while leaning away from the Y neighbor."""
    angle = math.radians(inclination_deg)
    z = np.array([0., -math.sin(angle), -math.cos(angle)])
    x = np.array([1., 0., 0.])
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack((x, np.cross(z, x), z))
    pose[:3, 3] = [0., pad_offset_m, TAKER_HEIGHT_M]
    return matrix(bottom) @ pose


@contextmanager
def tool_fixture(world):
    """Measure actual tool meshes, restoring their mounts before arm tests."""
    sim = world.sim
    mounts, tcp_to_tool, shapes = {}, {}, {}
    for side, arm in world.arms.items():
        tool = h(arm["tool"])
        mounts[side] = transform(sim, tool)
        tcp_to_tool[side] = np.linalg.inv(world.tcp(side)) @ mounts[side]
        shapes[side] = collect(sim, arm["tool_shapes"])

    def position(side, tcp):
        set_transform(sim, h(world.arms[side]["tool"]), tcp @ tcp_to_tool[side])
        errors = pose_error(tcp, world.tcp(side))
        require_gate(errors[0] <= POSITION_LIMIT_M and errors[1] <= ORIENTATION_LIMIT_RAD,
                     "static tool fixture did not reproduce the exact target", arm=side, error=errors)

    def check_pair():
        hit, pair = sim.checkCollision(shapes["A"], shapes["B"])
        if hit:
            return {"ok": False, "collision_pair": [sim.getObjectAlias(item) for item in pair]}
        distance, values, pair = sim.checkDistance(shapes["A"], shapes["B"], .100)
        return {"ok": True, "distance_m": float(values[6]) if distance else .100,
                "pair": [sim.getObjectAlias(item) for item in pair] if distance else None,
                "lower_bound": not bool(distance)}

    try:
        yield position, check_pair
    finally:
        for side, pose in mounts.items():
            set_transform(sim, h(world.arms[side]["tool"]), pose)
        for group in shapes.values():
            sim.destroyCollection(group)
        for side, original in mounts.items():
            errors = pose_error(original, transform(sim, h(world.arms[side]["tool"])))
            require_gate(errors[0] <= 1e-8 and errors[1] <= 2e-6,
                         "static tool measurement changed the original mounts", arm=side, error=errors)


def screen_tools(world, case, dims):
    """Screen relative tool poses once, before solving any assembled-arm IK."""
    sim = world.sim
    part, giver, taker, neighbor = (case[key] for key in ("part", "giver", "taker", "neighbor"))
    mid = matrix(HANDOVER_CENTERS_M[0])
    part_handle = h(world.parts[part])
    target = final_bottom(world, part)
    accepted, tested = [], []
    with reversible(world), tool_fixture(world) as (position, check_pair):
        world.set_bottom(part, mid)
        set_width(world, giver, closed_width(dims[1]))
        set_width(world, taker, closed_width(dims[0]))
        for g_tilt, t_tilt, offset in product(TILT_ANGLES_DEG, TAKER_TILTS_DEG, TAKER_OFFSETS_M):
            row = {"tilt_deg": g_tilt, "taker_tilt_deg": t_tilt, "taker_offset_m": offset}
            tested.append(row)
            position(giver, pier_giver_grasp(mid, g_tilt, case["incline"]))
            position(taker, taker_pose(mid, t_tilt, offset))
            pair = check_pair()
            row["tool_pair"] = pair
            if not pair["ok"] or pair["distance_m"] <= 0.:
                continue
            try:
                contact(world, giver, part, dims, 1)
                contact(world, taker, part, dims, 0)
            except Rejected as error:
                row["wall_failure"] = error.evidence
                continue
            # Native seat endpoint against the real adjacent brick. The
            # fixture does not claim a robot path or a continuous clearance.
            world.set_bottom(part, target)
            position(taker, taker_pose(target, t_tilt, offset))
            hits = []
            for ref in world.arms[taker]["tool_shapes"]:
                handle = h(ref)
                for block in (h(world.parts[neighbor]), h(world.parts[case["lower"][1]])):
                    sign, raw = sim.checkCollision(handle, block)
                    if sign:
                        hits.append([sim.getObjectAlias(item) for item in raw])
            row["seat_primitive_collisions"] = hits
            world.set_bottom(part, mid)
            if hits:
                continue
            row["score_m"] = pair["distance_m"]
            accepted.append(row)
    return sorted(accepted, key=lambda row: -row["score_m"]), tested


def screen_arms(world, case, dims, options):
    giver, taker, part = (case[key] for key in ("giver", "taker", "part"))
    results, tested = [], []
    with reversible(world):
        initial = {side: world.q(side) for side in ("A", "B")}
        set_width(world, giver, closed_width(dims[1]))
        set_width(world, taker, closed_width(dims[0]))
        for params, xyz in product(options[:7], HANDOVER_CENTERS_M[:3]):
            row = {**params, "handover_bottom_m": list(xyz), "pairs_tested": 0}
            tested.append(row)
            mid = matrix(xyz)
            world.set_bottom(part, mid)
            for side, q in initial.items():
                world.setq(side, q)
            gp = pier_giver_grasp(mid, row["tilt_deg"], case["incline"])
            tp = taker_pose(mid, row["taker_tilt_deg"], row["taker_offset_m"])
            for gq, tq in product(endpoints(world, giver, gp), endpoints(world, taker, tp)):
                row["pairs_tested"] += 1
                world.setq(giver, gq)
                world.setq(taker, tq)
                try:
                    g_fit, g_allow = contact(world, giver, part, dims, 1)
                    t_fit, t_allow = contact(world, taker, part, dims, 0)
                except Rejected as error:
                    row.setdefault("first_rejection", {"reason": error.reason, "evidence": error.evidence})
                    continue
                clearance = world.check(allow=[*g_allow, *t_allow])
                across = interarm(world)
                if not clearance["ok"] or not across["ok"]:
                    row.setdefault("first_rejection", {"clearance": clearance, "interarm": across})
                    continue
                results.append({**row, "clearance": clearance, "interarm": across,
                                "wall_fit": {giver: g_fit, taker: t_fit},
                                "qs_rad": {giver: gq.tolist(), taker: tq.tolist()}})
                break
    return sorted(results, key=lambda row: -row["interarm"]["minimum_distance_m"]), tested


def positive_chain(world, case, candidate, dims, trace, allow_preview):
    giver, taker, part = (case[key] for key in ("giver", "taker", "part"))
    bottom0 = world.bottom(part)
    hand_middle = matrix(candidate["handover_bottom_m"])
    goal = final_bottom(world, part)
    tilt = candidate["tilt_deg"]
    gp = lambda bottom: pier_giver_grasp(bottom, tilt, case["incline"])
    tp = lambda bottom: taker_pose(bottom, candidate["taker_tilt_deg"], candidate["taker_offset_m"])
    giver_pad_allow = [(pad, h(world.parts[part])) for pad in pad_handles(world, giver)]
    taker_pad_allow = [(pad, h(world.parts[part])) for pad in pad_handles(world, taker)]
    with trace.stage("taker_initial_opening"):
        jaw_sweep(world, trace, taker, OPEN_M)
    with trace.stage("giver_open_and_approach_stock"):
        jaw_sweep(world, trace, giver, OPEN_M)
        current = world.tcp(giver)
        lift = translate(current, [0., 0., max(0., .145-current[2, 3])])
        continuous_move(world, trace, giver, lift)
        target = translate(gp(bottom0), [0., 0., .026])
        travel = target.copy()
        travel[2, 3] = max(lift[2, 3], target[2, 3])
        continuous_move(world, trace, giver, travel)
        continuous_move(world, trace, giver, target)
    with trace.stage("giver_vertical_stock_descent"):
        continuous_move(world, trace, giver, gp(bottom0))
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
                stage["preview"] = capture(world.sim, HERE/"p0_previews"/f"hand_mid_tilted_{case['id']}.png",
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


def run():
    data = {"phase": "P0-2 tilted taker", "passed": False, "directions": [], "trace_path":
            str(HERE/"p0_2_tilted_samples.jsonl"), "lifecycle": str(HERE/"native/p0_2_tilted_lifecycle.json")}
    atomic_json(RECEIPT, data)
    try:
        with scene_session("p0_2_tilted") as (sim, client):
            world = BridgeWorld(sim, client)
            with Path(data["trace_path"]).open("w", encoding="utf-8", buffering=1) as events:
                for case in CASES:
                    row = {"id": case["id"], "part": case["part"], "slot": case["slot"],
                           "passed": False, "route_attempts": []}
                    data["directions"].append(row)
                    print(f"{case['id']} tilted-taker tool geometry screen", flush=True)
                    try:
                        with reversible(world):
                            place_supports(world, case)
                            dims = part_sizes(world, case["part"])
                            tools, trials = screen_tools(world, case, dims)
                            row["tool_trials"], row["tool_positives"] = trials, tools
                            atomic_json(RECEIPT, data)
                            require_gate(bool(tools), "no tested relative tool configuration meets exact static geometry")
                            arm_positives, arm_trials = screen_arms(world, case, dims, tools)
                            row["arm_trials"], row["arm_positives"] = arm_trials, arm_positives
                            atomic_json(RECEIPT, data)
                            require_gate(bool(arm_positives), "no tested full-arm handoff endpoint met native geometry")
                            static = arm_positives[0]
                            with reversible(world):
                                xyz = np.asarray(static["handover_bottom_m"])
                                world.set_bottom(case["part"], matrix(xyz))
                                for arm, q in static["qs_rad"].items():
                                    world.setq(arm, q)
                                set_width(world, case["giver"], closed_width(dims[1]))
                                set_width(world, case["taker"], closed_width(dims[0]))
                                try:
                                    row["static_endpoint_preview"] = capture(world.sim,
                                        HERE/"p0_previews"/f"hand_mid_tilted_{case['id']}_endpoint_only.png",
                                        xyz+[.22,-.31,.19], xyz+[0.,0.,.025], fov=37)
                                except (RuntimeError, ValueError) as error:
                                    row["preview_error"] = str(error)
                            for iteration, candidate in enumerate(arm_positives[:3], 1):
                                route = {"id": case["id"], "attempt": iteration, "candidate": candidate,
                                         "passed": False, "stages": []}
                                row["route_attempts"].append(route)
                                with reversible(world):
                                    try:
                                        route["passed"] = positive_chain(world, case, candidate, dims,
                                            RunTrace(world, route, events), allow_preview=True)
                                    except Rejected as error:
                                        route.update(reason=error.reason, evidence=error.evidence)
                                if route["passed"]:
                                    row["passed"] = True
                                    break
                            row["stages"] = row["route_attempts"][-1]["stages"]
                            if not row["passed"]:
                                row["reason"] = row["route_attempts"][-1]["reason"]
                    except Rejected as error:
                        row.update(reason=error.reason, evidence=error.evidence)
                    print(f"{case['id']} tilted path passed={row['passed']} {row.get('reason','')}", flush=True)
                    atomic_json(RECEIPT, data)
            data["passed"] = all(row["passed"] for row in data["directions"]) and len(data["directions"]) == 2
    except Exception as error:
        data.update(error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise
    finally:
        atomic_json(RECEIPT, data)
    return 0 if data["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(run())
