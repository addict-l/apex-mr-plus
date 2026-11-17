"""Conservative TCP reach bounds measured from a verified live model chain.

For a fixed-base chain containing only revolute joints, each distance from
root to first joint, between adjacent joint origins, and from last joint to
TCP is invariant under those rotations. Their sum bounds root-to-TCP distance
by the triangle inequality. Joint sampling or a UR5 dimension table is never
used as an infeasibility proof. Unsupported or incomplete models are left
uncertified, so ordinary finite IK exhaustion retains its UNKNOWN meaning.
"""
from __future__ import annotations

from copy import deepcopy
import math


METHOD = "measured_serial_revolute_chain_triangle_bound_v1"
MARGIN = .01


def _integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _vector(value, count):
    if not isinstance(value, (list, tuple)) or len(value) != count:
        raise ValueError(f"需要 {count} 维实数向量")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item) for item in value):
        raise ValueError("测量含非有限数或非实数")
    return [float(item) for item in value]


def _pose(value):
    result = _vector(value, 7)
    if abs(math.fsum(item*item for item in result[3:])-1.) > 1e-5:
        raise ValueError("测得姿态四元数未归一化")
    return result


def measure_reach_certificate(sim, arm, *, max_ancestors=256):
    """Read the actual TCP ancestry and measure a provable outer sphere.

    The scene is not modified. Exactly the configured six joint handles must
    occur, in order, on the TCP-to-model path and in the model's joint tree;
    every one must report the simulator's revolute type. Missing APIs, unknown
    topology, an extra joint, a moving snapshot, or invalid measurements return
    ``certified=False`` rather than a guessed radius.
    """
    certificate = {"schema_version": 1, "method": METHOD, "certified": False,
        "radius": None, "origin": None, "margin": MARGIN,
        "model": None, "tip": None,
        "joint_handles": [], "actual_topology": [], "segments": []}
    try:
        if not isinstance(arm, dict) or not isinstance(arm.get("joints"), (list, tuple)):
            raise ValueError("臂配置未提供可核实的关节列表")
        model, tip, expected = arm["model"], arm["tip"], list(arm["joints"])
        certificate.update(model=model, tip=tip, joint_handles=expected)
        if (not _integer(model) or not _integer(tip) or model == tip
                or len(expected) != 6 or len(set(expected)) != 6
                or any(not _integer(joint) for joint in expected)):
            raise ValueError("模型、TCP 或六个关节句柄不完整")
        joint_object_type = sim.object_joint_type
        root_type = sim.getObjectType(model)
        if root_type != sim.object_shape_type or sim.getObjectInt32Param(model, sim.shapeintparam_static) != 1:
            raise ValueError("模型根不是已核实的固定基座形状")
        certificate["root_fixed"] = {"object_type": root_type, "shape_object_type": sim.object_shape_type,
                                     "static_parameter": sim.shapeintparam_static, "value": 1}
        # Current CoppeliaSim names it joint_revolute; older versions expose
        # the subtype alias. Either must come from the actual simulator API.
        try:
            revolute_type = sim.joint_revolute
        except AttributeError:
            revolute_type = sim.joint_revolute_subtype
        if not _integer(joint_object_type) or not _integer(revolute_type):
            raise ValueError("仿真器未提供可核实的关节类型常量")
        tree_joints = list(sim.getObjectsInTree(model, joint_object_type, 0))
        if len(tree_joints) != 6 or set(tree_joints) != set(expected):
            raise ValueError("模型关节树不是指定的六关节集合")
        before_q = _vector([sim.getJointPosition(joint) for joint in expected], 6)
        root_before = _pose(sim.getObjectPose(model, -1))
        chain, seen, cursor = [], set(), tip
        for _ in range(max_ancestors):
            if not _integer(cursor) or cursor < 0 or cursor in seen:
                raise ValueError("TCP 祖先链断开或存在环")
            seen.add(cursor)
            object_type = sim.getObjectType(cursor)
            if not _integer(object_type):
                raise ValueError("祖先对象类型无法核实")
            parent = sim.getObjectParent(cursor)
            if not _integer(parent):
                raise ValueError("祖先父句柄无法核实")
            entry = {"handle": cursor, "parent": parent, "alias": str(sim.getObjectAlias(cursor)),
                     "object_type": object_type, "is_joint": object_type == joint_object_type}
            if entry["is_joint"]:
                entry["joint_type"] = sim.getJointType(cursor)
                if entry["joint_type"] != revolute_type:
                    raise ValueError("TCP 祖先链含非转动关节；不能使用固定长度球界")
                entry["joint_kind"] = "revolute"
            chain.append(entry)
            certificate["actual_topology"] = list(reversed(deepcopy(chain)))
            if cursor == model:
                break
            cursor = parent
        else:
            raise ValueError("TCP 祖先链超过核实上限")
        chain.reverse()
        if chain[0]["handle"] != model or chain[0]["is_joint"]:
            raise ValueError("祖先链没有终止于固定的模型根")
        base_ancestry, cursor = [], chain[0]["parent"]
        for _ in range(max_ancestors):
            if cursor == -1:
                break
            if not _integer(cursor) or cursor < 0 or cursor in seen:
                raise ValueError("固定基座的世界祖先链断开或存在环")
            seen.add(cursor)
            kind, parent = sim.getObjectType(cursor), sim.getObjectParent(cursor)
            if not _integer(kind) or not _integer(parent) or kind == joint_object_type:
                raise ValueError("模型根之上存在关节或无法核实的祖先，不能证明固定基座")
            base_ancestry.append({"handle": cursor, "parent": parent,
                "alias": str(sim.getObjectAlias(cursor)), "object_type": kind, "is_joint": False})
            cursor = parent
        else:
            raise ValueError("固定基座的世界祖先链超过核实上限")
        certificate["base_ancestry"] = base_ancestry
        actual_joints = [entry["handle"] for entry in chain if entry["is_joint"]]
        if actual_joints != expected:
            raise ValueError("TCP 祖先链与指定六关节的拓扑顺序不一致")
        points = [model, *expected, tip]
        aliases = {entry["handle"]: entry["alias"] for entry in chain}
        poses = {handle: _pose(sim.getObjectPose(handle, -1)) for handle in points}
        after_q = _vector([sim.getJointPosition(joint) for joint in expected], 6)
        root_after = _pose(sim.getObjectPose(model, -1))
        if any(abs(left-right) > 1e-10 for left, right in zip(before_q, after_q)):
            raise ValueError("测量期间关节状态发生变化")
        if any(abs(left-right) > 1e-10 for left, right in zip(root_before, root_after)):
            raise ValueError("测量期间模型根位姿发生变化")
        for start, end in zip(points, points[1:]):
            displacement = [right-left for left, right in zip(poses[start][:3], poses[end][:3])]
            certificate["segments"].append({"from_handle": start, "to_handle": end,
                "from_alias": aliases[start], "to_alias": aliases[end],
                "vector_world": displacement, "length": math.hypot(*displacement)})
        certificate.update(certified=True, origin=poses[model][:3],
            radius=math.fsum(segment["length"] for segment in certificate["segments"]),
            actual_topology=chain, object_type_joint=joint_object_type, joint_type_revolute=revolute_type,
            measurement={"root_pose": root_after, "joint_positions": after_q,
                "point_handles": points, "point_world_positions": [poses[handle][:3] for handle in points],
                "joint_tree_handles": tree_joints, "units": "m"},
            proof="只含六个转动关节的固定基座串行链；相邻关节原点距离和末端偏移长度之和给出 TCP 外球上界",
            scope="当前实际模型的运动学 TCP 位置必要条件；球内不证明 IK、姿态、避碰或动力学可行")
        if verified_sphere(arm, certificate) is None:
            raise ValueError("测量证书的自检未通过")
        return certificate
    except Exception as exc:
        certificate.update(certified=False, radius=None, origin=None,
                           reason=f"{type(exc).__name__}: {exc}")
        return certificate


def verified_sphere(arm, certificate=None):
    """Check complete measurement evidence before trusting a stored bound."""
    if not isinstance(arm, dict):
        return None
    certificate = arm.get("reach_certificate") if certificate is None else certificate
    try:
        if not isinstance(certificate, dict) or certificate.get("certified") is not True or certificate.get("method") != METHOD:
            return None
        model, tip, joints = arm["model"], arm["tip"], list(arm["joints"])
        if (certificate["model"] != model or certificate["tip"] != tip
                or certificate["joint_handles"] != joints or len(joints) != 6 or len(set(joints)) != 6):
            return None
        chain = certificate["actual_topology"]
        handles = [entry["handle"] for entry in chain]
        if (not handles or handles[0] != model or handles[-1] != tip or len(set(handles)) != len(handles)
                or any(entry["parent"] != previous for previous, entry in zip(handles, chain[1:]))):
            return None
        fixed = certificate["root_fixed"]
        if fixed["value"] != 1 or fixed["object_type"] != fixed["shape_object_type"] or fixed["object_type"] != chain[0]["object_type"]:
            return None
        base = certificate["base_ancestry"]
        if (chain[0]["parent"] != (base[0]["handle"] if base else -1)
                or (base and base[-1]["parent"] != -1)
                or any(left["parent"] != right["handle"] for left, right in zip(base, base[1:]))
                or any(entry["is_joint"] or entry["object_type"] == certificate["object_type_joint"] for entry in base)
                or len({entry["handle"] for entry in base}) != len(base)
                or set(handles) & {entry["handle"] for entry in base}):
            return None
        actual_joints = [entry for entry in chain if entry["is_joint"]]
        if ([entry["handle"] for entry in actual_joints] != joints
                or any(entry.get("joint_kind") != "revolute" or entry.get("joint_type") != certificate["joint_type_revolute"]
                       for entry in actual_joints)
                or any(entry["is_joint"] != (entry["object_type"] == certificate["object_type_joint"]) for entry in chain)):
            return None
        measurement = certificate["measurement"]
        points = [model, *joints, tip]
        if measurement["point_handles"] != points or set(measurement["joint_tree_handles"]) != set(joints) or len(measurement["joint_tree_handles"]) != 6:
            return None
        positions = [_vector(point, 3) for point in measurement["point_world_positions"]]
        if len(positions) != 8 or measurement["units"] != "m":
            return None
        origin = _vector(certificate["origin"], 3)
        root_pose = _pose(measurement["root_pose"])
        _vector(measurement["joint_positions"], 6)
        if math.dist(origin, positions[0]) > 1e-10 or math.dist(origin, root_pose[:3]) > 1e-10:
            return None
        if "base_pose" in arm and math.dist(origin, _pose(arm["base_pose"])[:3]) > 1e-8:
            return None
        segments = certificate["segments"]
        if len(segments) != 7:
            return None
        lengths = []
        for index, segment in enumerate(segments):
            vector = _vector(segment["vector_world"], 3)
            length = segment["length"]
            if (segment["from_handle"] != points[index] or segment["to_handle"] != points[index+1]
                    or not isinstance(length, (int, float)) or isinstance(length, bool)
                    or not math.isfinite(length) or length < 0
                    or abs(math.hypot(*vector)-length) > 1e-10
                    or math.dist(vector, [right-left for left, right in zip(positions[index], positions[index+1])]) > 1e-10):
                return None
            lengths.append(length)
        radius, margin = certificate["radius"], certificate["margin"]
        if (not isinstance(radius, (int, float)) or isinstance(radius, bool) or not math.isfinite(radius)
                or radius <= 0 or abs(math.fsum(lengths)-radius) > 1e-10
                or not isinstance(margin, (int, float)) or isinstance(margin, bool)
                or not math.isfinite(margin) or margin < MARGIN):
            return None
        return {"origin": origin, "radius": float(radius), "margin": float(margin)}
    except (KeyError, TypeError, ValueError, IndexError, OverflowError):
        return None


def outer_sphere_rejection(arm, target_pose):
    """Return proof only for a target strictly outside a verified sphere."""
    sphere = verified_sphere(arm)
    if sphere is None or target_pose is None:
        return None
    try:
        target = _pose(target_pose)[:3]
    except ValueError:
        return None
    distance = math.dist(target, sphere["origin"])
    bound = sphere["radius"] + sphere["margin"]
    if distance <= bound:
        return None
    return {"ok": False, "status": "INVALID", "proven_infeasible": True,
        "problem": {"reason": "outside_certified_serial_chain_sphere",
                    "distance": distance, "radius": sphere["radius"], "margin": sphere["margin"],
                    "outer_bound": bound, "target_world": target, "origin_world": sphere["origin"]},
        "reach_certificate": deepcopy(arm["reach_certificate"])}
