"""Rebuild the two grippers in a copy and verify the original LEGO meshes."""
from hashlib import sha256
from itertools import combinations, product
import json
from pathlib import Path

import numpy as np

from bridge_runtime import HERE, atomic_json, scene_session


ASSETS = HERE / "assets"
INPUT = ASSETS / "01_-_Initial.ttt"
COMPLETED = ASSETS / "02_-_Completed_bridge.ttt"
OUTPUT = ASSETS / "lego_bridge_kinematic_v1.ttt"
BOARD_TOP = .002
PITCH = .008
BODY_HEIGHT = .016
SEAT_EPSILON = 1e-5
PARTS = {f"{family}_{i:02}" for family in ("Red", "Orange", "Blue", "Green")
         for i in (1, 2)} | {"Bridge_deck_12x4"}
SLOTS = {"Green_01": "L2_f", "Green_02": "L2_b",
         "Orange_01": "R2_f", "Orange_02": "R2_b"}
REWORK = {"Housing", "Identification_band", "Rail_cover", "Slide", "Finger",
          "Grip_pad", "Press_stem", "Stud_press_pad"}
FRONT = .0787


def file_sha(path):
    with path.open("rb") as stream:
        return sha256(stream.read()).hexdigest()


def ref(sim, handle):
    return {"handle": handle, "path": sim.getObjectAlias(handle, 2)}


def matrix(sim, handle, relative=-1):
    value = np.eye(4)
    value[:3] = np.asarray(sim.getObjectMatrix(handle, relative), dtype=float).reshape(3, 4)
    return value


def apply(matrix_, points):
    return points @ matrix_[:3, :3].T + matrix_[:3, 3]


def descendants(sim, root, kind=None):
    return sim.getObjectsInTree(root, sim.handle_all if kind is None else kind, 0)


def named(sim, root, alias):
    matches = [h for h in descendants(sim, root) if sim.getObjectAlias(h) == alias]
    if len(matches) != 1:
        raise ValueError(f"{alias!r} requires a unique match, found {len(matches)}")
    return matches[0]


def mesh(sim, handle):
    value = sim.getShapeMesh(handle)
    return np.asarray(value[0], dtype=float).reshape(-1, 3), np.asarray(value[1], dtype=np.int64)


def bounds(points):
    return points.min(axis=0), points.max(axis=0)


def mesh_hash(vertices, indices):
    return sha256(vertices.astype("<f8").tobytes() + indices.astype("<i8").tobytes()).hexdigest()


def brick_state(sim):
    kit = sim.getObject("/LEGO_Workcell/Bridge_kit")
    rows = {}
    for h in descendants(sim, kit, sim.sceneobject_shape):
        name = sim.getObjectAlias(h)
        vertices, faces = mesh(sim, h)
        world = matrix(sim, h)
        lower, upper = bounds(apply(world, vertices))
        bottom = (lower + upper) / 2
        bottom[2] = lower[2]
        offset = world[:3, 3] - bottom
        rows[name] = {**ref(sim, h), "bottom_m": bottom.tolist(),
                      "mesh_pose": sim.getObjectPose(h, sim.handle_world),
                      "rel": offset.tolist(), "bounds_min_m": lower.tolist(),
                      "bounds_max_m": upper.tolist(),
                      "mesh_vertex_count": len(vertices), "mesh_triangle_count": len(faces)//3,
                      "mesh_sha256": mesh_hash(vertices, faces)}
    assert set(rows) == PARTS, sorted(rows)
    return rows


def support_regions(rows):
    deck = rows["Bridge_deck_12x4"]
    first, last = map(np.asarray, (deck["bounds_min_m"], deck["bounds_max_m"]))
    distinct = []
    for part, slot in SLOTS.items():
        row = rows[part]
        lower, upper = map(np.asarray, (row["bounds_min_m"], row["bounds_max_m"]))
        left, right = np.maximum(first[:2], lower[:2]), np.minimum(last[:2], upper[:2])
        body_top = row["bottom_m"][2] + BODY_HEIGHT
        assert np.all(right > left) and abs(deck["bottom_m"][2] - body_top) <= SEAT_EPSILON
        distinct.append({"slot": slot, "part": part,
                         "center_m": [*((left + right) / 2).tolist(), body_top],
                         "bounds_xy_m": [left.tolist(), right.tolist()],
                         "area_m2": float(np.prod(right - left))})
    assert len(distinct) == 4
    for one, two in combinations(distinct, 2):
        first, second = np.asarray(one["bounds_xy_m"]), np.asarray(two["bounds_xy_m"])
        assert np.any(np.minimum(first[1], second[1]) <= np.maximum(first[0], second[0]))
    return distinct


def seat_probe(sim, completed, regions):
    # At a supported interface exact surface contact can report a collision.
    # Separate the deck temporarily and restore its exact original world pose.
    deck = completed["Bridge_deck_12x4"]["handle"]
    pose = sim.getObjectPose(deck, sim.handle_world)
    reference_lift = 2e-5
    lifted = pose.copy()
    lifted[2] += reference_lift
    proof = []
    try:
        sim.setObjectPose(deck, lifted, sim.handle_world)
        for region in regions:
            support = completed[region["part"]]["handle"]
            collision = bool(sim.checkCollision(deck, support)[0])
            hit, data, pair = sim.checkDistance(deck, support, .001)
            distance = float(data[6]) if hit else None
            proof.append({"slot": region["slot"], "part": region["part"],
                          "collision": collision, "lift_m": reference_lift, "distance_m": distance,
                          "pair": pair, "passed": bool(hit and not collision
                                                      and abs(distance-reference_lift) <= SEAT_EPSILON)})
    finally:
        sim.setObjectPose(deck, pose, sim.handle_world)
    return proof


def transformed_shape(role, points):
    points = points.copy()
    low, high = bounds(points)
    center = (low + high)/2
    if role in {"Housing", "Identification_band", "Rail_cover"}:
        width, depth = {"Housing": (.056, .029),
                        "Identification_band": (.0568, .0298),
                        "Rail_cover": (.052, .029)}[role]
        points[:, 0] = (points[:, 0] - center[0]) * (width/(high[0]-low[0])) + center[0]
        points[:, 1] = (points[:, 1] - center[1]) * (depth/(high[1]-low[1])) + center[1]
        if role == "Rail_cover":
            # A full-width guide face at z=62 mm retains bearing overlap
            # with each slide at q=17 mm. Preserve the rounded sections.
            at_front = np.abs(points[:, 2] - high[2]) < 1e-8
            current = np.ptp(points[at_front, 0])
            points[at_front, 0] = (points[at_front, 0] - center[0]) * (.052/current) + center[0]
    elif role == "Slide":
        # Increase the guide bearing length along the permitted motion axis,
        # retaining the original contact face, carriage and rounded profile.
        points[:, 0] = (points[:, 0] - center[0]) * (.008/(high[0]-low[0])) + center[0]
    elif role == "Finger":
        points[:, 0] = (points[:, 0] - center[0]) * (.0045/(high[0]-low[0])) + center[0]
        points[:, 1] = (points[:, 1] - center[1]) * (.0065/(high[1]-low[1])) + center[1]
    elif role == "Grip_pad":
        points[:, 1] = (points[:, 1] - center[1]) * (.006/(high[1]-low[1])) + center[1]
    elif role == "Stud_press_pad":
        points[:, 2] += FRONT - high[2]
    elif role == "Press_stem":
        top = FRONT - .002
        points[:, 2] = low[2] + (points[:, 2] - low[2]) * (top-low[2])/(high[2]-low[2])
    return points


def replace_shape(sim, handle, tool):
    role = sim.getObjectAlias(handle)
    parent = sim.getObjectParent(handle)
    layer = sim.getObjectInt32Param(handle, sim.objintparam_visibility_layer)
    props = sim.getObjectSpecialProperty(handle)
    rgb = {}
    for component in (sim.colorcomponent_ambient_diffuse, sim.colorcomponent_specular):
        result = sim.getShapeColor(handle, None, component)
        if result[0]:
            rgb[component] = result[1]
    vertices, indices = mesh(sim, handle)
    original = apply(matrix(sim, handle, tool), vertices)
    wanted = transformed_shape(role, original)
    updated = sim.createMeshShape(0, .559, wanted.flatten().tolist(), indices.tolist())
    aligned = matrix(sim, updated)
    # The mesh vertices were supplied in tool-frame axes; the creator's
    # returned object frame is stored as a transform relative to that tool.
    sim.setObjectParent(updated, parent, True)
    sim.setObjectMatrix(updated, aligned[:3].flatten().tolist(), tool)
    sim.setObjectInt32Param(updated, sim.objintparam_visibility_layer, layer)
    sim.setObjectInt32Param(updated, sim.shapeintparam_static, 1)
    sim.setObjectInt32Param(updated, sim.shapeintparam_respondable, 0)
    sim.setObjectSpecialProperty(updated, props)
    for component, color in rgb.items():
        sim.setShapeColor(updated, None, component, color)
    sim.removeObjects([handle])
    sim.setObjectAlias(updated, role)
    mesh_v, mesh_f = mesh(sim, updated)
    measured = apply(matrix(sim, updated, tool), mesh_v)
    assert len(mesh_f) == len(indices), (role, len(mesh_f), len(indices))
    expected_ends, actual_ends = (*bounds(wanted),), (*bounds(measured),)
    assert np.max(np.abs(np.array(expected_ends) - np.array(actual_ends))) < 1e-7, role
    return updated


def geometry_extents(sim, handle, relative):
    return bounds(apply(matrix(sim, handle, relative), mesh(sim, handle)[0]))


def collision_hit(sim, a, b):
    result, pair = sim.checkCollision(a, b)
    return bool(result), pair


def jaw_q(sim, opening, follower, q, direct):
    sim.setJointPosition(opening, q)
    if direct:
        sim.setJointPosition(follower, q)
    actual = np.array([sim.getJointPosition(opening), sim.getJointPosition(follower)])
    if np.max(np.abs(actual-q)) > 1e-8:
        raise RuntimeError(f"Jaw synchronism at {q}: {actual}")


def jaw_test(sim, tool, groups, opening, follower, direct):
    pads = groups["Grip_pad"]
    left = [groups[key][0] for key in ("Slide", "Finger", "Grip_pad")]
    right = [groups[key][1] for key in ("Slide", "Finger", "Grip_pad")]
    stationary = [h for h in descendants(sim, tool, sim.sceneobject_shape)
                  if h not in left and h not in right]
    cover = groups["Rail_cover"][0]
    coverage = []
    aperture = []
    failed = []
    rail_contacts = []
    tests = list(np.linspace(0., .017, 35)) + [.0049, .0129]
    for q in sorted(set(tests)):
        jaw_q(sim, opening, follower, float(q), direct)
        for a, b in [*product(left, right), *product(left+right, stationary)]:
            collides, pair = collision_hit(sim, a, b)
            if not collides:
                continue
            if a in groups["Slide"] and b == cover:
                bottom, top = geometry_extents(sim, a, tool)
                fixed_lo, fixed_hi = geometry_extents(sim, b, tool)
                # The only designed moving/stationary contact is the guide
                # plane. Assert that their volumes do not overlap on Z.
                if abs(bottom[2]-fixed_hi[2]) <= 1e-7:
                    rail_contacts.append({"q_m": q, "pair": pair})
                    continue
            failed.append({"q_m": q, "pair": pair, "shapes": [ref(sim, a), ref(sim, b)]})
        left_end = geometry_extents(sim, pads[0], tool)[1][0]
        right_start = geometry_extents(sim, pads[1], tool)[0][0]
        gap = float(right_start - left_end)
        if any(abs(q-key) < 1e-8 for key in (0., .0049, .0129, .017)):
            expected = float(.006 + 2*q)
            aperture.append({"q_m": q, "expected_m": expected,
                             "measured_m": gap, "error_m": abs(gap-expected),
                             "passed": abs(gap-expected) <= .0001})
        for side, glide in enumerate(groups["Slide"]):
            slide_low, slide_hi = geometry_extents(sim, glide, tool)
            cover_low, cover_hi = geometry_extents(sim, cover, tool)
            overlap = np.minimum(slide_hi, cover_hi) - np.maximum(slide_low, cover_low)
            coverage.append({"q_m": q, "jaw": "left" if side == 0 else "right",
                             "width_m": float(overlap[0]), "depth_m": float(overlap[1]),
                             "plane_mismatch_m": float(slide_low[2]-cover_hi[2])})
    bearing = min(coverage, key=lambda row: row["width_m"])
    return {"openings": aperture, "configuration_count": len(set(tests)),
            "unexpected_collisions": failed, "rail_surface_contacts": rail_contacts,
            "min_bearing_overlap": bearing, "bearing_samples": coverage,
            "passed": len(aperture) == 4 and all(row["passed"] for row in aperture) and not failed
                      and all(row["width_m"] > 0 and row["depth_m"] > 0
                              and abs(row["plane_mismatch_m"]) <= 1e-7 for row in coverage)}


def link_group(sim, handle, joints, base):
    current = handle
    while current not in joints and current != base:
        current = sim.getObjectParent(current)
        if current < 0:
            raise ValueError(f"shape {handle} escaped from base {base}")
    return 0 if current == base else joints.index(current) + 1


def arm_build(sim, side, source):
    base = sim.getObject(f"/UR5_{side}")
    tool = named(sim, base, "Brick_microgripper_" + side)
    tcp = named(sim, tool, "TCP_between_fingertips")
    opening = named(sim, tool, "Jaw_opening")
    follower = named(sim, tool, "Jaw_follower")
    joints = [h for h in descendants(sim, base, sim.sceneobject_joint)
              if sim.getJointType(h) == sim.joint_revolute_subtype]
    assert len(joints) == 6
    prior_limits = [sim.getJointInterval(h) for h in joints]
    q = [sim.getJointPosition(h) for h in joints]
    for h in joints:
        sim.setJointMode(h, sim.jointmode_kinematic)
    for h, value in zip(joints, q):
        sim.setJointPosition(h, value)
    assert prior_limits == [sim.getJointInterval(h) for h in joints]
    assert all(sim.getJointMode(h)[0] == sim.jointmode_kinematic for h in joints)
    scripts = []
    for h in descendants(sim, base, sim.sceneobject_script):
        enabled_before = not sim.getBoolProperty(h, "scriptDisabled")
        sim.setBoolProperty(h, "scriptDisabled", True)
        assert sim.getBoolProperty(h, "scriptDisabled") is True
        scripts.append({**ref(sim, h), "enabled_before": enabled_before, "enabled": False})
    assert len(scripts) == 1 and scripts[0]["path"].endswith("/Script")

    tcp_frame = sim.getObjectPose(tcp, tool)
    q_jaw = sim.getJointPosition(opening)
    for jaw in (opening, follower):
        assert sim.getJointType(jaw) == sim.joint_prismatic_subtype
        assert sim.getJointInterval(jaw)[0] is False
        sim.setJointInterval(jaw, False, [0., .017])
    sim.setJointPosition(opening, 0.)
    direct = abs(sim.getJointPosition(follower)) > 1e-8
    if direct:
        sim.setJointMode(follower, sim.jointmode_kinematic)
    jaw_q(sim, opening, follower, 0., direct)
    for h in descendants(sim, tool, sim.sceneobject_shape):
        if sim.getObjectAlias(h) in REWORK:
            replace_shape(sim, h, tool)
    roles = {}
    tool_shapes = descendants(sim, tool, sim.sceneobject_shape)
    for h in tool_shapes:
        roles.setdefault(sim.getObjectAlias(h), []).append(h)
    for handles in roles.values():
        handles.sort(key=lambda h: (matrix(sim, h, tool)[0, 3], h))
    proof = jaw_test(sim, tool, roles, opening, follower, direct)
    jaw_q(sim, opening, follower, 0., direct)
    sizes = {}
    for role in sorted(REWORK):
        sizes[role] = []
        for h in roles[role]:
            low, high = geometry_extents(sim, h, tool)
            sizes[role].append({"size_m": (high-low).tolist(),
                                "min_m": low.tolist(), "max_m": high.tolist()})
    press_clearance = float(tcp_frame[2] - sizes["Stud_press_pad"][0]["max_m"][2])
    pad_bottom_error = max(abs(tcp_frame[2] - row["max_m"][2]) for row in sizes["Grip_pad"])
    assert press_clearance >= .0097 and pad_bottom_error <= 1e-7
    proof.update(press_tcp_clearance_m=press_clearance, sizes=sizes,
                 pad_bottom_tcp_error_m=pad_bottom_error)
    jaw_q(sim, opening, follower, q_jaw, direct)
    arm_all = [h for h in descendants(sim, base, sim.sceneobject_shape) if h not in tool_shapes]
    special_mask = sim.objectspecialproperty_collidable | sim.objectspecialproperty_measurable
    arm_shapes = [h for h in arm_all if sim.getObjectSpecialProperty(h) & special_mask]
    records = [{"group": link_group(sim, h, joints, base), **ref(sim, h),
                "collision_mesh": h in arm_shapes,
                "special": sim.getObjectSpecialProperty(h)} for h in arm_all]
    mount = sim.getObject("/Bench_mount_" + side)
    return {"root": ref(sim, base), "tool": ref(sim, tool), "tcp": ref(sim, tcp),
            "flange": ref(sim, named(sim, base, "connection")), "mount": ref(sim, mount),
            "tcp_tool_pose": tcp_frame, "initial_q": q, "initial_jaw_q_m": q_jaw,
            "joints": [ref(sim, h) for h in joints], "joint_intervals": prior_limits,
            "jaw_opening": ref(sim, opening), "jaw_follower": ref(sim, follower),
            "jaw_driver_mode": "direct_both" if direct else "opening_with_dependent_follower",
            "jaw_interval_m": [0., .017], "jaw_evidence": proof,
            "shapes": [ref(sim, h) for h in arm_shapes + tool_shapes],
            "arm_shapes": [ref(sim, h) for h in arm_shapes],
            "tool_shapes": [ref(sim, h) for h in tool_shapes], "all_arm_meshes": records,
            "arm_shape_groups": {str(row["handle"]): row["group"] for row in records},
            "shape_roles": {key: [ref(sim, h) for h in value] for key, value in roles.items()},
            "disabled_scripts": scripts, "source_part_catalog": source}


def refresh_refs(sim, value, old_to_reopened):
    """Recover handles from full paths after Coppelia compacts object IDs."""
    if isinstance(value, dict):
        if set(("handle", "path")) <= value.keys():
            initial_handle = value["handle"]
            actual = sim.getObject(value["path"])
            assert sim.getObjectAlias(actual, 2) == value["path"]
            assert old_to_reopened.setdefault(initial_handle, actual) == actual
            value["handle"] = actual
        for child in value.values():
            refresh_refs(sim, child, old_to_reopened)
    elif isinstance(value, list):
        for child in value:
            refresh_refs(sim, child, old_to_reopened)


def build(sim):
    input_hash = {p.name: file_sha(p) for p in (INPUT, COMPLETED)}
    sim.loadScene(str(COMPLETED))
    completed = brick_state(sim)
    regions = support_regions(completed)
    support_proof = seat_probe(sim, completed, regions)
    sim.loadScene(str(INPUT))
    start = brick_state(sim)
    for name, row in start.items():
        assert completed[name]["mesh_sha256"] == row["mesh_sha256"]
        assert abs(row["bottom_m"][2] - BOARD_TOP) <= SEAT_EPSILON
        assert abs(completed[name]["bottom_m"][2] - ({"Bridge_deck_12x4": .034}.get(
            name, .018 if name in SLOTS else .002))) <= SEAT_EPSILON

    sim.setBoolParam(sim.boolparam_dynamics_handling_enabled, False)
    assert sim.getBoolParam(sim.boolparam_dynamics_handling_enabled) is False
    arms = {side: arm_build(sim, side, "8 mm pitch, 16 mm custom pier bricks") for side in ("A", "B")}
    assert sim.getSimulationState() == sim.simulation_stopped
    board = ref(sim, sim.getObject("/Main_plate_48x32"))
    all_arm_meshes = {row["handle"] for arm in arms.values() for row in
                      [*arm["all_arm_meshes"], *arm["tool_shapes"]]}
    parts = {name: {**ref(sim, row["handle"]),
                    "rel": row["rel"], "mesh_origin_from_design_bottom": row["rel"],
                    "design_bottom_from_mesh": [-x for x in row["rel"]],
                    "initial_bottom_m": row["bottom_m"], "target_bottom_m": completed[name]["bottom_m"],
                    "initial_mesh_pose": row["mesh_pose"], "target_mesh_pose": completed[name]["mesh_pose"],
                    "mesh_sha256": row["mesh_sha256"], "bounds_min_m": row["bounds_min_m"],
                    "bounds_max_m": row["bounds_max_m"], "mesh_triangle_count": row["mesh_triangle_count"]}
             for name, row in start.items()}
    part_handles = {row["handle"] for row in parts.values()}
    workcell = sim.getObject("/LEGO_Workcell")
    environmental = [ref(sim, h) for h in descendants(sim, workcell, sim.sceneobject_shape)
                     if h not in all_arm_meshes and h not in part_handles
                     and sim.getObjectSpecialProperty(h) & sim.objectspecialproperty_collidable]
    assert board in environmental
    assert not OUTPUT.exists(), f"Inspect or move the prior candidate before rebuilding: {OUTPUT}"
    sim.saveScene(str(OUTPUT))
    sim.loadScene(str(OUTPUT))
    reopened = brick_state(sim)
    for name, row in reopened.items():
        assert row["path"] == parts[name]["path"]
        assert row["mesh_sha256"] == parts[name]["mesh_sha256"]
        assert np.max(np.abs(np.array(row["bottom_m"]) - np.array(parts[name]["initial_bottom_m"]))) <= SEAT_EPSILON
    assert sim.getBoolParam(sim.boolparam_dynamics_handling_enabled) is False
    old_to_reopened = {}
    refresh_refs(sim, [arms, parts, board, environmental], old_to_reopened)
    mechanical_contacts = []
    for side, arm in arms.items():
        arm["arm_shape_groups"] = {str(row["handle"]): row["group"] for row in arm["all_arm_meshes"]}
        for row in arm["disabled_scripts"]:
            assert sim.getBoolProperty(row["handle"], "scriptDisabled") is True
        for joint, interval, value in zip(arm["joints"], arm["joint_intervals"], arm["initial_q"]):
            assert sim.getJointMode(joint["handle"])[0] == sim.jointmode_kinematic
            assert sim.getJointInterval(joint["handle"]) == interval
            assert abs(sim.getJointPosition(joint["handle"])-value) <= 1e-8
        proof = arm["jaw_evidence"]
        for result in [*proof["rail_surface_contacts"], *proof["unexpected_collisions"]]:
            result["pair_before_reopen"] = result["pair"]
            result["pair"] = [old_to_reopened.get(handle, handle) for handle in result["pair"]]
        if proof["passed"]:
            cover = arm["shape_roles"]["Rail_cover"][0]
            for slide in arm["shape_roles"]["Slide"]:
                mechanical_contacts.append({"side": side, "pair": [slide, cover],
                    "reason": "verified coplanar manufactured slide-guide mating face",
                    "verified_joint_range_m": [0., .017]})
    assert {p.name: file_sha(p) for p in (INPUT, COMPLETED)} == input_hash

    receipt = {"asset_kind": "8 mm pitch, 16 mm custom pier bricks",
               "pure_kinematics": True, "simulation_stopped": True, "dynamics_enabled": False,
               "board_top_m": BOARD_TOP, "pitch_m": PITCH, "pier_body_height_m": BODY_HEIGHT,
               "scene": str(OUTPUT), "scene_sha256": file_sha(OUTPUT), "input_sha256": input_hash,
               "workcell": ref(sim, sim.getObject("/LEGO_Workcell")), "board": board, "arms": arms, "parts": parts,
               "environment_shapes": environmental, "mechanical_contacts": mechanical_contacts,
               "resolved_object_id_changes": sum(old != new for old, new in old_to_reopened.items()),
               "support_regions": regions, "support_precheck": support_proof,
               "build_gate_passed": all(arm["jaw_evidence"]["passed"] for arm in arms.values())
                                    and all(row["passed"] for row in support_proof)}
    atomic_json(ASSETS / "scene_manifest.json", receipt)
    print(json.dumps({"build_gate_passed": receipt["build_gate_passed"],
                      "jaw": {arm: row["jaw_evidence"]["passed"] for arm, row in arms.items()},
                      "support": support_proof, "sha256": receipt["scene_sha256"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    with scene_session("build") as (sim, client):
        build(sim)
