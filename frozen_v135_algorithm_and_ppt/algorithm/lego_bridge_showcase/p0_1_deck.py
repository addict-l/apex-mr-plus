"""Stopped-scene, native-geometry Phase 0 deck preflight."""
from contextlib import contextmanager
from pathlib import Path
import json
import math
import traceback

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from bridge_runtime import HERE, capture, scene_session
from bridge_geometry import BridgeWorld
from lego_grasp_model import deck_grasp, jaw_q

PART = "Bridge_deck_12x4"
DECK_DIMS = (.09584, .03184, .0032)
OPEN = .040
HOLD = .0318
CLOSED = .006
WIDTH_TOL = .0001
P_TOL = .00005
R_TOL = .0002
DISTANCE_LIMIT = .025
SAMPLE_STEP = .00095


class GateBlocked(RuntimeError):
    def __init__(self, detail):
        super().__init__(str(detail))
        self.detail = detail


def require_gate(condition, **detail):
    if not condition:
        raise GateBlocked(detail)


def handle(value):
    return value["handle"] if isinstance(value, dict) else value


def matrix(values):
    m = np.eye(4)
    m[:3] = np.asarray(values).reshape(3, 4)
    return m


def move_z(transform, z):
    target = transform.copy()
    target[2, 3] += z
    return target


def xyz_close(a, b, tolerance):
    return float(np.linalg.norm(np.asarray(a)-np.asarray(b))) <= tolerance


def rigid_mismatch(actual, wanted):
    dp = float(np.linalg.norm(actual[:3, 3] - wanted[:3, 3]))
    da = float(Rotation.from_matrix(wanted[:3, :3].T @ actual[:3, :3]).magnitude())
    return dp, da


def mesh_points(sim, item, relative_to=-1):
    vertices, *_ = sim.getShapeMesh(handle(item))
    transform = matrix(sim.getObjectMatrix(handle(item), relative_to))
    return np.asarray(vertices).reshape(-1, 3) @ transform[:3, :3].T + transform[:3, 3]


def measure_gap(w, side):
    row = w.arms[side]
    pads = row["shape_roles"]["Grip_pad"]
    require_gate(len(pads) == 2, issue="Grip_pad count", side=side, count=len(pads))
    xx = sorted((mesh_points(w.sim, p, handle(row["tool"]))[:, 0] for p in pads),
                key=lambda value: float(np.mean(value)))
    return float(np.min(xx[1]) - np.max(xx[0]))


def allowed_seat_validation(w, data):
    regions = data["support_regions"]
    require_gate(len(regions) == 4, issue="four independent support regions", count=len(regions))
    slots = {row["slot"] for row in regions}
    keys = {row["part"] for row in regions}
    require_gate(len(slots) == len(keys) == 4, issue="support region identity duplicated")
    positions = []
    deck = data["parts"][PART]["target_bottom_m"]
    ranges = []
    for region in regions:
        midpoint = np.asarray(region["center_m"])
        require_gate(len(midpoint) == 3 and abs(midpoint[2]-deck[2]) <= 1e-5,
                issue="support plane does not match completed deck", slot=region["slot"])
        require_gate(np.all(np.abs(midpoint[:2]-np.asarray(deck[:2])) < np.asarray(DECK_DIMS[:2])/2),
                issue="support center lies outside bridge deck", slot=region["slot"])
        bounds = np.asarray(region["bounds_xy_m"], dtype=float)
        require_gate(bounds.size == 4, issue="invalid 2D support bounds", slot=region["slot"])
        ranges.append(bounds)
        positions.append(midpoint)
    require_gate(all(not xyz_close(a, b, 1e-5) for i, a in enumerate(positions) for b in positions[i+1:]),
            issue="support centers coincide")
    for index, a in enumerate(ranges):
        for b in ranges[index+1:]:
            require_gate(not np.array_equal(a, b), issue="support rectangles coincide")
    for name, part in data["parts"].items():
        require_gate(xyz_close(w.bottom(name)[:3, 3], part["initial_bottom_m"], 1e-5),
                issue="derived initial scene bottom disagrees with manifest", part=name,
                actual_m=w.bottom(name)[:3, 3].tolist(), initial_m=part["initial_bottom_m"])
    require_gate(len(data["parts"]) == 9, issue="unexpected LEGO count")
    return {"initial_bottom_m": {k: w.bottom(k)[:3, 3].tolist() for k in data["parts"]},
            "completed_reference_bottom_m": {k: v["target_bottom_m"] for k, v in data["parts"].items()},
            "completed_support_regions": regions}


@contextmanager
def preserved(w):
    joints = {side: w.q(side).copy() for side in w.arms}
    jaws = {side: w.sim.getJointPosition(handle(row["jaw_opening"])) for side, row in w.arms.items()}
    objects = {name: (w.sim.getObjectParent(handle(row)), w.bottom(name).copy())
               for name, row in w.parts.items()}
    try:
        yield
    finally:
        for name, (parent, transform) in objects.items():
            w.sim.setObjectParent(handle(w.parts[name]), parent, True)
            w.set_bottom(name, transform)
        for side in w.arms:
            w.setq(side, joints[side])
            w.jaw(side, jaws[side])


def distance(w, allowed):
    response = w.check(allow=allowed, requested_distance_m=DISTANCE_LIMIT)
    return response


def opening_gate(w):
    output = {}
    with preserved(w):
        for side in w.arms:
            row = w.arms[side]
            original = float(w.sim.getJointPosition(handle(row["jaw_opening"])))
            values = []
            for width in (CLOSED, .0158, HOLD, OPEN):
                q = jaw_q(width)
                w.jaw(side, q)
                measured = measure_gap(w, side)
                error = abs(measured-width)
                a, b = [w.sim.getJointPosition(handle(row[n])) for n in ("jaw_opening", "jaw_follower")]
                require_gate(error <= WIDTH_TOL and abs(a-q) <= WIDTH_TOL/2 and abs(b-q) <= WIDTH_TOL/2,
                        issue="opening precision", side=side, nominal_m=width, measured_m=measured,
                        expected_joint_m=q, measured_joints_m=[a,b])
                values.append({"shape_mesh_opening_m": measured, "nominal_m": width,
                               "error_m": error, "joint_positions_m": [a, b]})
            coverage = []
            for q in np.linspace(0., jaw_q(OPEN), 35):
                w.jaw(side, float(q))
                check = distance(w, ())
                coverage.append({"q_m": float(q), "native": check})
            output[side] = {"four_openings": values, "travel_samples": coverage,
                            "original_park_pose_full_scene_clear": all(item["native"]["ok"] for item in coverage),
                            "no_self_collision_on_travel": row["jaw_evidence"]["passed"]}
            w.jaw(side, original)
    return output


class DeckSweep:
    def __init__(self, w, side, phases):
        self.w = w
        self.side = side
        vertices = mesh_points(w.sim, w.parts[PART])
        bottom = w.bottom(PART)
        local = (vertices - bottom[:3,3]) @ bottom[:3,:3]
        self.contact_width_m = float(np.ptp(local[:,1]))
        require_gate(abs(self.contact_width_m-HOLD) <= WIDTH_TOL, issue="physical deck width", measured_m=self.contact_width_m)
        self.phases = phases
        self.q = w.q(side).copy()
        self.previous_tip = w.tcp(side).copy()
        self.jaw = float(w.sim.getJointPosition(handle(w.arms[side]["jaw_opening"])))
        self.pad_contacts = [(handle(p), handle(w.parts[PART]))
                             for p in w.arms[side]["shape_roles"]["Grip_pad"]]

    def _verify(self, title, index, target, jaw, held, grip_contact):
        self.w.jaw(self.side, jaw)
        try:
            at = rigid_mismatch(self.w.tcp(self.side), target)
            solutions = ([self.q] if at[0] <= 1e-6 and at[1] <= 2e-6 else
                         self.w.ik(self.side, target, seed=self.q, max_solutions=1, seed_count=1))
        except Exception as e:
            require_gate(False, issue="IK query error", stage=title, sample=index, error=str(e),
                    q_previous=self.q.tolist(), target_tcp=target.tolist())
        require_gate(bool(solutions), issue="no continuous IK solution", stage=title, sample=index,
                q_previous=self.q.tolist(), target_tcp=target.tolist())
        q = np.asarray(solutions[0], dtype=float)
        continuity = float(np.max(np.abs(q-self.q)))
        require_gate(continuity <= .25, issue="IK branch jump", stage=title, sample=index,
                previous_joint_rad=self.q.tolist(), joint_rad=q.tolist(),
                max_joint_change_rad=continuity, target_tcp=target.tolist())
        self.w.setq(self.side, q)
        actual_tcp = self.w.tcp(self.side)
        dp, da = rigid_mismatch(actual_tcp, target)
        measured_step = float(np.linalg.norm(actual_tcp[:3, 3]-self.previous_tip[:3, 3]))
        require_gate(measured_step <= .001, issue="native TCP integration step",
                stage=title, sample=index, measured_step_m=measured_step)
        require_gate(dp <= P_TOL and da <= R_TOL, issue="precise tip FK", stage=title, sample=index,
                position_error_m=dp, orientation_error_rad=da, joint_rad=q.tolist())
        allow = ()
        if grip_contact:
            guard = self.wall_guard(title)
            allow = self.pad_contacts
        else:
            guard = None
        check = distance(self.w, allow)
        state = {"sample": index, "joint_rad": q.tolist(), "jaw_m": jaw, "tcp_target": target.tolist(),
                 "tcp_position_error_m": dp, "tcp_orientation_error_rad": da,
                 "actual_tcp": actual_tcp.tolist(), "measured_position_step_m": measured_step,
                 "held": held, "geometry": check}
        if guard is not None:
            state["intended_pad_contact"] = guard
        require_gate(check["ok"], issue="native mesh collision", stage=title, **state)
        self.previous_tip = actual_tcp
        self.q = q
        return state

    def wall_guard(self, title):
        jaw = float(self.w.sim.getJointPosition(handle(self.w.arms[self.side]["jaw_opening"])))
        require_gate(abs(jaw-jaw_q(self.contact_width_m)) <= 1e-6, issue="pad contact mentioned outside closed aperture",
                stage=title, jaw_m=jaw)
        pad_refs = self.w.arms[self.side]["shape_roles"]["Grip_pad"]
        pads = [handle(ref) for ref in pad_refs]
        bottom = np.asarray(self.w.bottom(PART))
        # Prove both inner edge planes only touch the two designated side walls.
        # Retain the tighter mesh transform tolerance than the opening test.
        detail = []
        for p in pads:
            v = mesh_points(self.w.sim, p)
            local = (v - bottom[:3, 3]) @ bottom[:3, :3]
            sign = -1 if float(np.mean(local[:, 1])) < 0 else 1
            inner = float(np.min(sign*local[:, 1]))
            overlap_z = min(DECK_DIMS[2], float(np.max(local[:, 2]))) - max(0., float(np.min(local[:, 2])))
            reach_x = min(DECK_DIMS[0]/2, float(np.max(local[:, 0]))) - max(-DECK_DIMS[0]/2, float(np.min(local[:, 0])))
            require_gate(-1e-6 <= inner-self.contact_width_m/2 <= 1e-5 and overlap_z > .0001 and reach_x > 0.,
                    issue="designated pad/side contact cannot be established", stage=title,
                    pad=self.w.sim.getObjectAlias(p, 1), half_width_m=self.contact_width_m/2,
                    inner_face_m=inner, wall_overlap_m=overlap_z, axial_overlap_m=reach_x)
            detail.append({"handle": p, "contact_side_sign": sign, "projected_side_plane_m": inner,
                           "overlap_height_m": overlap_z, "minimum_tip_height_above_bottom_m": float(np.min(local[:,2]))})
        require_gate({v["contact_side_sign"] for v in detail} == {-1,1}, issue="pads not opposed", stage=title)
        return detail

    def sweep(self, title, target, jaw=None, *, held=False, grip=False, contact_at_end=False):
        initial = self.w.tcp(self.side)
        jaw0 = self.jaw
        jaw = jaw0 if jaw is None else float(jaw)
        distance_m = float(np.linalg.norm(initial[:3, 3]-target[:3, 3]))
        starts = np.stack((initial[:3,:3],target[:3,:3]))
        slerp = Slerp([0.,1.], Rotation.from_matrix(starts))
        angle = float(Rotation.from_matrix(starts[0].T @ starts[1]).magnitude())
        n = max(1, math.ceil(distance_m / SAMPLE_STEP), math.ceil(angle/.01), math.ceil(abs(jaw-jaw0)/.00025))
        row = {"phase": title, "status": "running", "sample_count": 0, "planned_samples": n,
               "max_position_increment_m": distance_m/n, "max_opening_axis_increment_m": abs(jaw-jaw0)/n,
               "minimum_clearance_m": None, "closest": None, "max_measured_position_increment_m": 0.}
        self.phases.append(row)
        print(f"P0-1 {self.side} {title}: {n} samples", flush=True)
        for i in range(1, n+1):
            t = i/n
            desired = np.eye(4)
            desired[:3,:3] = slerp(t).as_matrix()
            desired[:3,3] = initial[:3,3] + t * (target[:3,3]-initial[:3,3])
            w = jaw0 + t * (jaw - jaw0)
            try:
                state = self._verify(title, i, desired, w, held, grip or (contact_at_end and i==n))
            except GateBlocked as exc:
                row.update(status="blocked", first_failure=exc.detail)
                raise
            measured = state["geometry"].get("minimum_distance_m")
            if row["closest"] is None or (measured is not None and row["minimum_clearance_m"] is None
                                          or measured is not None and measured < row["minimum_clearance_m"]):
                row.update(minimum_clearance_m=measured, closest=state)
            row["max_measured_position_increment_m"] = max(row["max_measured_position_increment_m"], state["measured_position_step_m"])
            row["sample_count"] += 1
        self.jaw = jaw
        row.update(last=state, status="passed", clearance_bound_m=DISTANCE_LIMIT)


def setup_piers(w, data):
    # This fixture is staged after lifting the deck so no lower pier is
    # installed through the deck's starting footprint.
    saved = w.bottom(PART).copy()
    placed = {}
    for name, row in sorted(data["parts"].items(), key=lambda item: item[1]["target_bottom_m"][2]):
        if name == PART:
            continue
        target = w.bottom(name).copy()
        target[:3,3] = row["target_bottom_m"]
        w.set_bottom(name, target)
        require_gate(xyz_close(w.bottom(name)[:3,3], row["target_bottom_m"], 1e-5),
                issue="placed pier differs from completed reference", part=name)
        placed[name] = w.bottom(name)[:3,3].tolist()
    require_gate(rigid_mismatch(w.bottom(PART), saved)[0] <= 1e-5, issue="fixture staging moved held deck")
    return placed


def run_trial(world, side, height, flipped, result):
    w = world
    phases = result["phases"]
    sweep = DeckSweep(w, side, phases)
    deck0 = w.bottom(PART).copy()
    destination = deck0.copy()
    destination[:3,3] = w.manifest["parts"][PART]["target_bottom_m"]
    # A higher grip catch clears the neighbouring rows of physical baseplate
    # studs. The other tested height is listed in the receipt.
    grasp = np.asarray(deck_grasp(deck0, height_m=height), dtype=float)
    grip_at_slot = grasp.copy()
    grip_at_slot[:3,3] += destination[:3,3]-deck0[:3,3]
    if flipped:
        flip = Rotation.from_euler("z", math.pi).as_matrix()
        grasp[:3,:3] = grasp[:3,:3] @ flip
        grip_at_slot[:3,:3] = grip_at_slot[:3,:3] @ flip
    parked = sweep.w.tcp(side)
    sweep.sweep("open_jaws_at_park", parked, jaw_q(OPEN))
    raised = parked.copy()
    raised[2,3] = max(parked[2,3], grasp[2,3]+.11)
    sweep.sweep("raise_clear_of_stock", raised)
    elevated_pick = move_z(grasp, .11)
    sweep.sweep("open_approach", elevated_pick)
    sweep.sweep("open_descend_to_deck", grasp)
    pads = w.arms[side]["shape_roles"]["Grip_pad"]
    fingertips = w.arms[side]["shape_roles"]["Finger"]
    tip_clearance = min(float(np.min(mesh_points(w.sim,p)[:,2])) for p in [*pads,*fingertips])-deck0[2,3]
    require_gate(tip_clearance >= .0004, issue="deck fingertip clearance below 0.4 mm", measured_m=tip_clearance)
    sweep.sweep("close_on_short_edge", grasp, jaw_q(sweep.contact_width_m), contact_at_end=True)
    result["deck_grasp_opening_m"] = sweep.contact_width_m
    result["pickup_side_contact"] = sweep.wall_guard("pickup_bind")
    result["fingertip_bottom_above_baseplate_m"] = tip_clearance
    drift = w.attach(PART, side)
    require_gate(drift <= 1e-5, issue="rigid attachment changed deck world pose", position_drift_m=drift)
    result["attach_world_drift_m"] = drift
    sweep.sweep("held_deck_vertical_lift", elevated_pick, held=True, grip=True)
    preview_dir = HERE / "p0_previews"
    preview_dir.mkdir(exist_ok=True)
    direction = "reversed" if flipped else "forward"
    picture = preview_dir / f"deck_grasp_closeup_{side}_{direction}_{height*1e6:.0f}um.png"
    focus = w.bottom(PART)[:3,3] + [0.,0.,.014]
    viewer = focus + np.array([.132,-.168,.085])
    try:
        result["preview"] = capture(w.sim,picture,viewer.tolist(),focus.tolist(),fov=38)
        result["preview_path"] = str(picture)
    except Exception as exc:
        result["preview_error"] = str(exc)
    result["completed_fixture_bottom_m"] = setup_piers(w, w.manifest)
    fixture = distance(w, sweep.pad_contacts)
    require_gate(fixture["ok"], issue="preload completed piers interferes with held deck", native=fixture)
    result["fixture_geometry"] = fixture
    if np.linalg.norm(elevated_pick[:3,3] - move_z(grip_at_slot,.07)[:3,3]) > SAMPLE_STEP:
        sweep.sweep("approach_completed_piers", move_z(grip_at_slot,.07), held=True, grip=True)
    sweep.sweep("seat_deck_on_four_piers", grip_at_slot, held=True, grip=True)
    dp, da = rigid_mismatch(w.bottom(PART), destination)
    require_gate(dp <= 1e-5 and da <= R_TOL, issue="seated deck differs from completed reference",
            position_error_m=dp, angle_error_rad=da, actual=w.bottom(PART).tolist())
    result["seat_pose_error"] = {"position_m": dp, "orientation_rad": da}
    # Inspect actual contact at each nominal support rather than just counting
    # support rectangles. The lifted challenge distinguishes intended seating
    # from a penetrated stud/cavity interface.
    support_rows = []
    part_handle = handle(w.parts[PART])
    before = w.bottom(PART).copy()
    try:
        w.set_bottom(PART, move_z(before,.00002))
        for support in w.manifest["support_regions"]:
            lower = handle(w.parts[support["part"]])
            collision, _ = w.sim.checkCollision(part_handle,lower)
            answer = w.sim.checkDistance(part_handle,lower,.001)
            clearance = float(answer[1][6]) if answer[0] > 0 else None
            require_gate(collision == 0 and clearance is not None and abs(clearance-.00002) <= 1e-5,
                    issue="four-contact support verification", part=support["part"], collision=collision,
                    lifted_separation_m=clearance)
            support_rows.append({"slot": support["slot"], "part": support["part"],
                                 "lifted_separation_m": clearance})
    finally:
        w.set_bottom(PART,before)
    result["seating_contacts"] = support_rows
    drift = w.attach(PART, None)
    require_gate(drift <= 1e-5, issue="deck release changed its seated position", position_drift_m=drift)
    result["release_world_drift_m"] = drift
    sweep.sweep("open_at_seated_deck", grip_at_slot, jaw_q(OPEN))
    sweep.sweep("open_vertical_retreat", move_z(grip_at_slot,.065))
    measured = measure_gap(w, side)
    require_gate(abs(measured-OPEN) <= WIDTH_TOL, issue="final clearance/jaw aperture", measured_m=measured)
    require_gate(xyz_close(w.bottom(PART)[:3,3], destination[:3,3], 1e-5),
            issue="deck moved while opening the jaws")
    result["status"] = "passed"


def artifact(path, record):
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False, default=str)+"\n",
                       encoding="utf-8")
    pending.replace(path)

def negative_probe(w, side, height):
    """Construct a deliberately narrow grip, while preserving pivots and bricks."""
    with preserved(w):
        target = deck_grasp(w.bottom(PART), height_m=height)
        solutions = w.ik(side, target, seed=w.q(side), max_solutions=8, seed_count=24)
        candidates = []
        for q in solutions:
            w.setq(side, q)
            dp, da = rigid_mismatch(w.tcp(side), target)
            if dp > P_TOL or da > R_TOL:
                continue
            w.jaw(side, jaw_q(OPEN))
            baseline = distance(w, ())
            w.jaw(side, jaw_q(.0288))
            negative = distance(w, ())
            candidates.append({"joint_rad": np.asarray(q).tolist(), "open_reference": baseline,
                               "deliberately_narrow_grip": negative})
            pair = set(negative.get("pair_handles") or [])
            roles = w.arms[side]["shape_roles"]
            pads = {handle(p) for p in [*roles["Grip_pad"], *roles["Finger"]]}
            deck = handle(w.parts[PART])
            if baseline["ok"] and not negative["ok"] and deck in pair and pair & pads:
                return {"status": "correctly_rejected", "nominal_grip_m": HOLD,
                        "negative_grip_m": .0288, "joint_rad": np.asarray(q).tolist(),
                        "opened_reference": baseline, "native": negative}
        return {"status": "not_verified", "negative_grip_m": .0288,
                "candidates": candidates, "reason": "coupled reference/collision pair not reproduced"}


def main():
    path = HERE / "p0_1_receipt.json"
    result = {"gate": "P0-1", "status": "running", "attempts": [], "negative": None}
    try:
        with scene_session("p0_1") as (sim, client):
            w = BridgeWorld(sim, client)
            require_gate(sim.getSimulationState() == sim.simulation_stopped,
                    issue="geometry gate must run with dynamics stopped")
            result["coords"] = allowed_seat_validation(w, w.manifest)
            result["opening_checks"] = opening_gate(w)
            artifact(path, result)
            if any(not row["no_self_collision_on_travel"] for row in result["opening_checks"].values()):
                result["status"] = "blocked"
                result["blocking_issue"] = "native jaw rail or moving-tool self collision during full travel"
            else:
                # Both short-edge orientations are physical rotation poses.
                # Use an above-stud target first, keeping the full clearance
                # and contact checks for each attempt.
                for side, flipped, h in (("A",False,.0023), ("B",False,.0023),
                                         ("A",True,.0023), ("B",True,.0023),
                                         ("A",False,.0005), ("B",False,.0005)):
                    trial = {"side": side, "jaw_axis_reversed": flipped, "tcp_height_m": h,
                             "phases": [], "status": "running"}
                    result["attempts"].append(trial)
                    with preserved(w):
                        try:
                            run_trial(w,side,h,flipped,trial)
                        except GateBlocked as e:
                            trial["status"] = "blocked"
                            trial["first_failure"] = e.detail
                    artifact(path, result)
                    if trial["status"] == "passed":
                        result["chosen_attempt"] = len(result["attempts"]) - 1
                        break
                chosen = next((row for row in result["attempts"] if row["status"] == "passed"), None)
                for side in ([chosen["side"]] if chosen else ["A", "B"]):
                    result["negative"] = negative_probe(w, side, .0023)
                    if result["negative"]["status"] == "correctly_rejected":
                        break
                result["status"] = ("passed" if chosen and chosen.get("preview_path")
                                    and result["negative"]["status"] == "correctly_rejected" else "blocked")
                if result["status"] == "blocked":
                    result["blocking_issue"] = ("no continuous deck pick/place swept path passed"
                        if not chosen else "required negative probe or preview is missing native evidence")
    except Exception as e:
        result.update(status="blocked", blocking_issue=str(e),
                      exception=e.detail if isinstance(e, GateBlocked) else repr(e),
                      traceback=traceback.format_exc())
    artifact(path, result)
    print(f"P0-1 {result['status']}: {path}", flush=True)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
