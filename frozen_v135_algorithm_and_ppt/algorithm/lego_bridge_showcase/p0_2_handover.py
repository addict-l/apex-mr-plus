import math
"""Phase 0 bilateral crossing and adjacent-seat geometry proofs."""
from contextlib import contextmanager
from itertools import product
from pathlib import Path
import traceback

import numpy as np

from bridge_runtime import HERE, atomic_json, capture, scene_session
from lego_grasp_model import (
    PIER_DIMS_M, bad_y_pier_grasp, jaw_q, linear_targets, matrix,
    pad_wall_receipt, pier_giver_grasp, pier_taker_grasp, pose_error, translate,
)


POSITION_LIMIT_M = 1e-5
ORIENTATION_LIMIT_RAD = 2e-4
MAX_JOINT_STEP_RAD = .25
OPEN_M = .021
TILT_ANGLES_DEG = (55., 50., 60., 65., 45.)
HANDOVER_CENTERS_M = ((0., .102, .176), (0., .124, .198), (0., .135, .238),
                      (0., .072, .207), (0., .155, .178))
RECEIPT = HERE / "p0_2_receipt.json"
CASES = (
    {"id": "A_to_B", "giver": "A", "taker": "B", "part": "Orange_02", "neighbor": "Orange_01",
     "lower": ("Blue_01", "Blue_02"), "slot": "R2_b", "incline": 1},
    {"id": "B_to_A", "giver": "B", "taker": "A", "part": "Green_02", "neighbor": "Green_01",
     "lower": ("Red_01", "Red_02"), "slot": "L2_b", "incline": -1},
)


class Rejected(RuntimeError):
    def __init__(self, reason, evidence=None):
        super().__init__(reason)
        self.reason = reason
        self.evidence = evidence or {}


def require_gate(condition, reason, **receipt):
    if not condition:
        raise Rejected(reason, receipt)


def h(ref):
    return int(ref["handle"] if isinstance(ref, dict) else ref)


def pad_handles(world, side):
    return [h(row) for row in world.arms[side]["shape_roles"]["Grip_pad"]]


def final_bottom(world, part):
    return matrix(world.parts[part]["target_bottom_m"])


def part_sizes(world, part, dimensions=PIER_DIMS_M):
    mesh = world.sim.getShapeMesh(h(world.parts[part]))
    vertex = np.asarray(mesh[0], dtype=float).reshape(-1, 3)
    handle = h(world.parts[part])
    observed = np.ptp(vertex @ matrix(world.sim.getObjectMatrix(handle, -1))[:3, :3].T, axis=0)
    require_gate(np.all(np.abs(observed[:2]-dimensions[:2]) <= .0001),
            "brick width disagrees with the registered nominal piece", observed=observed.tolist())
    return float(observed[0]), float(observed[1]), float(dimensions[2])


def contact(world, side, part, dims, axis):
    pads = pad_handles(world, side)
    proof = pad_wall_receipt(world.sim, pads, world.bottom(part), dims, axis)
    require_gate(proof["ok"], "both pads must cover their intended brick faces without crossing them",
            arm=side, part=part, wall_fit=proof)
    return proof, [(pad, h(world.parts[part])) for pad in pads]


@contextmanager
def reversible(world):
    sim = world.sim
    bras = {side: (world.q(side).copy(), sim.getJointPosition(h(row["jaw_opening"])))
            for side, row in world.arms.items()}
    bricks = {name: (sim.getObjectParent(h(row)), world.bottom(name).copy())
              for name, row in world.parts.items()}
    try:
        yield
    finally:
        for name, (parent, _) in bricks.items():
            sim.setObjectParent(h(world.parts[name]), parent, True)
        for side, (q, opening) in bras.items():
            world.setq(side, q)
            world.jaw(side, opening)
        for name, (_, bottom) in bricks.items():
            world.set_bottom(name, bottom)


class RunTrace:
    def __init__(self, world, output, events):
        self.world = world
        self.output = output
        self.events = events
        self.current = None
        self.path = None

    @contextmanager
    def stage(self, label):
        self.current = {"phase": label, "passed": False, "samples": 0,
                        "minimum_distance_m": None, "pair": None}
        self.output["stages"].append(self.current)
        try:
            yield self.current
            self.current["passed"] = True
        except Rejected as error:
            self.current.update(reason=error.reason, evidence=error.evidence)
            raise
        finally:
            self.current = None
            self.events.flush()

    def sample(self, allowed=(), *, command=None, part_binding=None):
        reading = self.world.check(allow=allowed)
        read_distance = reading.get("minimum_distance_m")
        qs = {side: self.world.q(side).tolist() for side in ("A", "B")}
        details = {"case": self.output["id"], "attempt": self.output["attempt"], "phase": self.current["phase"],
                   "sample": self.current["samples"], "qs_rad": qs,
                   "command": command, "clearance": reading}
        pos_delta = ang_delta = None
        if part_binding:
            name, owner, binding = part_binding
            pos_delta, ang_delta = pose_error(self.world.tcp(owner) @ binding, self.world.bottom(name))
            details.update(binding_error_m=pos_delta, binding_angular_error_rad=ang_delta)
        import json
        self.events.write(json.dumps(details, ensure_ascii=False, allow_nan=False) + "\n")
        self.current["samples"] += 1
        old_min = self.current["minimum_distance_m"]
        if read_distance is not None and (old_min is None or read_distance < old_min):
            self.current.update(minimum_distance_m=read_distance, pair=reading.get("pair"),
                                pair_handles=reading.get("pair_handles"), qs_at_minimum_rad=qs)
        require_gate(reading["ok"], "native geometry rejected the swept state", clearance=reading, qs_rad=qs)
        if part_binding:
            require_gate(pos_delta <= POSITION_LIMIT_M and ang_delta <= ORIENTATION_LIMIT_RAD,
                    "rigid payload binding changed during interpolation",
                    position_error_m=pos_delta, orientation_error_rad=ang_delta)
        return reading


def place_supports(world, case):
    for part in (*case["lower"], case["neighbor"]):
        world.set_bottom(part, final_bottom(world, part))


def closed_width(mesh_width):
    # Reserve 2 um air-side at each pad for the FK numerical residue,
    # inside the specified measured contact band and 0.1 mm aperture error.
    return mesh_width + .000004


def set_width(world, side, width):
    world.jaw(side, jaw_q(width))


def endpoints(world, side, goal):
    return world.ik(side, goal, seed=world.q(side), max_solutions=3, seed_count=9)


def continuous_move(world, trace, side, target, *, allowed=(), binding=None):
    measured = world.tcp(side)
    for goal in linear_targets(measured, target, step_m=.00095):
        seeds = world.q(side)
        solutions = world.ik(side, goal, seed=seeds, seed_count=1, max_solutions=1)
        require_gate(bool(solutions), "no continuous IK for the next 1 mm waypoint",
                arm=side, xyz_m=goal[:3, 3].tolist(), previous_q_rad=seeds.tolist())
        q = solutions[0]
        multiple = float(np.max(np.abs(q-seeds)))
        require_gate(multiple <= MAX_JOINT_STEP_RAD, "next waypoint requires a joint branch switch",
                arm=side, delta_rad=multiple, previous_q_rad=seeds.tolist(), q_rad=q.tolist())
        world.setq(side, q)
        actual = world.tcp(side)
        error_m, error_rad = pose_error(goal, actual)
        require_gate(error_m <= POSITION_LIMIT_M and error_rad <= ORIENTATION_LIMIT_RAD,
                "native TCP failed waypoint precision", arm=side,
                position_error_m=error_m, orientation_error_rad=error_rad)
        actual_step = float(np.linalg.norm(actual[:3, 3]-measured[:3, 3]))
        # Use a slightly finer commanded grid if observed errors exceed the 1 mm rule.
        require_gate(actual_step <= .001, "measured TCP step exceeded one millimeter",
                arm=side, distance_m=actual_step)
        trace.sample(allowed, command={"moving_arm": side, "goal_xyz_m": goal[:3, 3].tolist(),
                                       "tcp_step_m": actual_step}, part_binding=binding)
        measured = actual


def jaw_sweep(world, trace, side, goal_width, *, preallowed=(), at_final=None, binding=None):
    q_start = float(world.sim.getJointPosition(h(world.arms[side]["jaw_opening"])))
    end = jaw_q(goal_width)
    count = max(1, math.ceil(abs(end-q_start)/.00025))
    proof = None
    for sample in range(1, count+1):
        q = q_start+(end-q_start)*sample/count
        world.jaw(side, q)
        allowed = preallowed
        if at_final and sample == count:
            proof, more = at_final()
            allowed = (*preallowed, *more)
        trace.sample(allowed, command={"moving_jaw": side, "jaw_q_m": q}, part_binding=binding)
    return proof


def collect(sim, shapes):
    collection = sim.createCollection(1)
    for role in shapes:
        handle = h(role)
        if sim.getObjectSpecialProperty(handle) & sim.objectspecialproperty_collidable:
            sim.addItemToCollection(collection, sim.handle_single, handle, 0)
    return collection


def interarm(world):
    sim = world.sim
    first, second = (collect(sim, world.arms[side]["shapes"]) for side in ("A", "B"))
    try:
        hit = sim.checkCollision(first, second)
        clearance = sim.checkDistance(first, second, .100)
        gap = float(clearance[1][6]) if clearance[0] else .100
        pair = [sim.getObjectAlias(x) for x in clearance[2]] if clearance[0] else None
        return {"ok": not hit[0] and gap > 0., "minimum_distance_m": gap,
                "exact": bool(clearance[0]), "pair": pair}
    finally:
        sim.destroyCollection(first)
        sim.destroyCollection(second)


def handover_fixture(world, case, dims):
    """Try setpoints as disposable probes, retaining the best real clearance."""
    giver, taker, part = (case[key] for key in ("giver", "taker", "part"))
    accepted, tried = [], []
    with reversible(world):
        base_q = {side: world.q(side).copy() for side in ("A", "B")}
        set_width(world, giver, closed_width(dims[1]))
        set_width(world, taker, closed_width(dims[0]))
        for tilt, xyz, direction in product(TILT_ANGLES_DEG, HANDOVER_CENTERS_M, (1, -1)):
            for side, q in base_q.items():
                world.setq(side, q)
            bottom = matrix(xyz)
            world.set_bottom(part, bottom)
            gp = pier_giver_grasp(bottom, tilt, case["incline"])
            tp = pier_taker_grasp(bottom, direction)
            gqs, tqs = endpoints(world, giver, gp), endpoints(world, taker, tp)
            result = {"tilt_deg": tilt, "handover_bottom_m": list(xyz),
                      "taker_direction": direction, "giver_ik": len(gqs), "taker_ik": len(tqs),
                      "pairs_tested": 0, "passed": False}
            tried.append(result)
            for gq, tq in product(gqs, tqs):
                result["pairs_tested"] += 1
                world.setq(giver, gq)
                world.setq(taker, tq)
                try:
                    ga, ga_allow = contact(world, giver, part, dims, 1)
                    ta, ta_allow = contact(world, taker, part, dims, 0)
                except Rejected as error:
                    result.setdefault("first_rejection", {"reason": error.reason, "evidence": error.evidence})
                    continue
                paired = interarm(world)
                check = world.check(allow=[*ga_allow, *ta_allow])
                if not paired["ok"] or not check["ok"]:
                    result.setdefault("first_rejection", {"reason": "native collision",
                                                            "clearance": check, "interarm": paired,
                                                            "qs_rad": {giver: gq.tolist(), taker: tq.tolist()}})
                    continue
                result["passed"] = True
                metric = paired["minimum_distance_m"]
                best = {**result, "qs_rad": {giver: gq.tolist(), taker: tq.tolist()},
                        "interarm": paired, "clearance": check, "wall_fit": {giver: ga, taker: ta},
                        "score_m": metric}
                accepted.append(best)
            if not gqs or not tqs:
                result["first_rejection"] = {"reason": "no endpoint IK"}
    accepted.sort(key=lambda row: -row["score_m"])
    return accepted, tried


def positive_chain(world, case, candidate, dims, trace, allow_preview):
    giver, taker, part = (case[key] for key in ("giver", "taker", "part"))
    bottom0 = world.bottom(part)
    hand_middle = matrix(candidate["handover_bottom_m"])
    goal = final_bottom(world, part)
    tilt, direction = candidate["tilt_deg"], candidate["taker_direction"]
    gp = lambda bottom: pier_giver_grasp(bottom, tilt, case["incline"])
    tp = lambda bottom: pier_taker_grasp(bottom, direction)
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
                stage["preview"] = capture(world.sim, HERE/"p0_previews/hand_mid_orthogonal.png",
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


def force_probe(world, side, tcp):
    solutions = endpoints(world, side, tcp)
    if not solutions:
        raise Rejected("negative target was not reached by IK", {"arm": side})
    world.setq(side, solutions[0])


def wrong_handover(world, case, dims):
    sim = world.sim
    with reversible(world):
        bottom = matrix(HANDOVER_CENTERS_M[0])
        world.set_bottom(case["part"], bottom)
        bad = pier_taker_grasp(bottom)
        for side in ("A", "B"):
            set_width(world, side, dims[0])
            force_probe(world, side, bad)
        pair = interarm(world)
        rejected = world.check()
        contact_pairs = []
        for a, b in product(pad_handles(world, "A"), pad_handles(world, "B")):
            if sim.checkCollision(a, b)[0]:
                contact_pairs.append([sim.getObjectAlias(a), sim.getObjectAlias(b)])
        return {"id": "wrong_handover", "passed": not rejected["ok"] and not pair["ok"]
                and bool(contact_pairs), "native": rejected, "interarm": pair,
                "colliding_pad_pairs": contact_pairs,
                "qs_rad": {side: world.q(side).tolist() for side in ("A", "B")}}


def wrong_neighbor(world, case, dims):
    sim = world.sim
    taker, name = case["taker"], case["neighbor"]
    with reversible(world):
        bottom = final_bottom(world, case["part"])
        bad = bad_y_pier_grasp(bottom)
        set_width(world, taker, dims[1])
        force_probe(world, taker, bad)
        rejected = world.check()
        probes = []
        for pad in pad_handles(world, taker):
            hit, pair = sim.checkCollision(pad, h(world.parts[name]))
            if hit:
                probes.append([sim.getObjectAlias(item) for item in pair])
        return {"id": "wrong_y_grasp_at_neighbor", "passed": not rejected["ok"] and bool(probes),
                "native": rejected, "colliding_pad_neighbor_pairs": probes,
                "qs_rad": {side: world.q(side).tolist() for side in ("A", "B")}}


def run():
    from bridge_geometry import BridgeWorld

    data = {"phase": "P0-2", "passed": False, "directions": [], "negative_cases": [],
            "interpolation_tcp_step_m": .001,
            "evidence_scope": "native collision and distance at each actual sampled kinematic state"}
    atomic_json(RECEIPT, data)
    try:
        with scene_session("p0_2") as (sim, client):
            world = BridgeWorld(sim, client, batch_joints=True)
            trace_path = HERE/"p0_2_samples.jsonl"
            data["trace_path"] = str(trace_path)
            with trace_path.open("w", encoding="utf-8", buffering=1) as events:
                for case in CASES:
                    report = {"id": case["id"], "part": case["part"], "slot": case["slot"],
                              "passed": False, "stages": []}
                    data["directions"].append(report)
                    with reversible(world):
                        place_supports(world, case)
                        dims = part_sizes(world, case["part"])
                        report["brick_size_m"] = list(dims)
                        neighbor = world.bottom(case["neighbor"])
                        target = final_bottom(world, case["part"])
                        report["adjacent_seam_m"] = abs(float(target[1, 3]-neighbor[1, 3]))-dims[1]
                        print(case["id"] + " probing exact orthogonal handover endpoints", flush=True)
                        try:
                            require_gate(0. < report["adjacent_seam_m"] <= .0002+1e-8,
                                    "original neighboring faces do not reproduce the nominal seam",
                                    measured_seam_m=report["adjacent_seam_m"])
                            accepted, trials = handover_fixture(world, case, dims)
                            report["handoff_endpoint_trials"] = trials
                            report["handoff_endpoint_candidates"] = accepted
                            require_gate(bool(accepted), "no orthogonal handover candidate passed native endpoint geometry")
                            if not (HERE/"p0_previews/hand_mid_endpoint_candidate.png").is_file():
                                with reversible(world):
                                    top = accepted[0]
                                    x = np.asarray(top["handover_bottom_m"])
                                    world.set_bottom(case["part"], matrix(x))
                                    for side, q in top["qs_rad"].items():
                                        world.setq(side, q)
                                    set_width(world, case["giver"], closed_width(dims[1]))
                                    set_width(world, case["taker"], closed_width(dims[0]))
                                    try:
                                        report["endpoint_only_preview"] = capture(world.sim,
                                            HERE/"p0_previews/hand_mid_endpoint_candidate.png",
                                            x + np.array([.22, -.31, .19]), x + np.array([0., 0., .025]), fov=37)
                                    except (RuntimeError, ValueError) as error:
                                        report["endpoint_preview_error"] = str(error)
                            report["route_attempts"] = []
                            distinct = set()
                            for selected in accepted:
                                identity = (selected["tilt_deg"], tuple(selected["handover_bottom_m"]),
                                            selected["taker_direction"])
                                if identity in distinct:
                                    continue
                                distinct.add(identity)
                                iteration = len(distinct)
                                report["selected_candidate"] = selected
                                route = {"id": case["id"], "attempt": iteration, "candidate": selected,
                                         "passed": False, "stages": []}
                                report["route_attempts"].append(route)
                                with reversible(world):
                                    print(case["id"] + f" executing distinct continuous route {iteration}", flush=True)
                                    try:
                                        route["passed"] = positive_chain(world, case, selected, dims,
                                            RunTrace(world, route, events),
                                            allow_preview=not (HERE/"p0_previews/hand_mid_orthogonal.png").is_file())
                                    except Rejected as error:
                                        route.update(reason=error.reason, evidence=error.evidence)
                                        print(case["id"] + " route blocked: " + error.reason, flush=True)
                                report.update(stages=route["stages"], passed=route["passed"])
                                if route["passed"]:
                                    break
                                report.update(reason=route["reason"], evidence=route["evidence"])
                                if iteration >= 3:
                                    break
                            if report["passed"]:
                                report.pop("reason", None)
                                report.pop("evidence", None)
                        except Rejected as error:
                            report.update(reason=error.reason, evidence=error.evidence)
                            print(case["id"] + " blocked: " + error.reason, flush=True)
                    atomic_json(RECEIPT, data)
                with reversible(world):
                    example = CASES[0]
                    place_supports(world, example)
                    dims = part_sizes(world, example["part"])
                    for function, label in ((wrong_handover, "wrong_handover"),
                                             (wrong_neighbor, "wrong_y_grasp_at_neighbor")):
                        try:
                            data["negative_cases"].append(function(world, example, dims))
                        except Rejected as error:
                            data["negative_cases"].append({"id": label, "passed": False,
                                "reason": error.reason, "evidence": error.evidence})
                        atomic_json(RECEIPT, data)
            data["handoff_preview"] = str(HERE/"p0_previews/hand_mid_orthogonal.png")
            data["handoff_preview_created"] = Path(data["handoff_preview"]).is_file()
            data["passed"] = (data["handoff_preview_created"] and len(data["directions"]) == len(CASES)
                              and all(case["passed"] for case in data["directions"])
                              and len(data["negative_cases"]) == 2
                              and all(case["passed"] for case in data["negative_cases"]))
    except Exception as error:
        data.update(error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise
    finally:
        data["lifecycle"] = str(HERE/"native/p0_2_lifecycle.json")
        atomic_json(RECEIPT, data)
    return 0 if data["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(run())

