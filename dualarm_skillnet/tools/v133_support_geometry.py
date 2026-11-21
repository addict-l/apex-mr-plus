"""Measure the assembled support tool without starting a simulator."""
from __future__ import annotations

import argparse
import ast
from itertools import product
import json
import math
from pathlib import Path
import sys

from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simbridge.frames import Frames
from simbridge.layout import BASE_POSITIONS
from simbridge.workcell import configure_support_tools


def original_gripper_geometry():
    """Read actual unchanged box sizes/poses from build_arms' source AST."""
    tree = ast.parse((ROOT/"apexmr/scene_helpers.py").read_text(encoding="utf-8"))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_arms")
    sizes, poses = {}, {}
    for node in ast.walk(function):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in {"palm", "finger"} and isinstance(node.value, ast.Call):
                sizes[node.targets[0].id] = ast.literal_eval(node.value.args[1])
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "setObjectPose" and isinstance(node.args[0], ast.Name)
                and node.args[0].id in {"finger", "tip"}):
            poses[node.args[0].id] = node.args[1]
    tip = ast.literal_eval(poses["tip"])
    finger = poses["finger"]
    lateral = finger.elts[0]
    if not (isinstance(lateral, ast.BinOp) and isinstance(lateral.op, ast.Mult)
            and isinstance(lateral.left, ast.Name) and lateral.left.id == "d"):
        raise ValueError("未识别的原始手指位置表达式，不能猜测几何")
    spread = ast.literal_eval(lateral.right)
    centers = [[side*spread, *[ast.literal_eval(n) for n in finger.elts[1:3]]] for side in (-1, 1)]
    return {"palm_size": sizes["palm"], "finger_size": sizes["finger"],
            "finger_centres_in_palm": centers, "tcp_in_palm": tip[:3]}


def corners(center, size, rotation):
    return [(rotation.apply([sign[i]*size[i]/2 for i in range(3)])+center).tolist()
            for sign in product((-1, 1), repeat=3)]


class Recorder:
    def __init__(self):
        self.poses = {}

    def setObjectPose(self, handle, pose, parent):
        self.poses[handle] = (pose, parent)


def measure():
    geometry, sim = original_gripper_geometry(), Recorder()
    arms = {side: {"palm": side+"palm", "support_sensor": side+"sensor", "support_pad": side+"shoe"}
            for side in ("A", "B")}
    configure_support_tools(sim, arms)
    bases = {side: {"base_pose": [*BASE_POSITIONS[side], 0, 0, 0, 1]} for side in arms}
    top = Frames(ROOT/"tasks/s1_parallel_place.toml")
    support = Frames(ROOT/"tasks/sim/s3a_cantilever_sim.toml", arm_config=bases)
    point = support.support_point("beam")
    beam = support.slot_pose("beam")
    beam_size = support.catalog[support.slots["beam"]["type"]].size
    measurements = {}
    for side, arm in arms.items():
        sensor, parent = sim.poses[arm["support_sensor"]]
        pad, pad_parent = sim.poses[arm["support_pad"]]
        sensor_rotation, pad_rotation = Rotation.from_quat(sensor[3:]), Rotation.from_quat(pad[3:])
        size = arm["support_pad_size"]
        row = {"sensor_in_palm": sensor, "shoe_in_sensor": pad, "shoe_size_m": size,
               "sensor_parent_is_palm": parent == arm["palm"], "shoe_parent_is_sensor": pad_parent == arm["support_sensor"]}
        tcp = top.grasp_pose("red", "stock_l1", side)
        rotation = Rotation.from_quat(tcp[3:])
        palm = [tcp[i]-rotation.apply(geometry["tcp_in_palm"])[i] for i in range(3)]
        center = rotation.apply(sensor[:3])+palm
        body = top.block_pose("red", "stock_l1")
        block_top = body[2]+top.catalog[top.blocks["red"]["type"]].size[2]/2
        shoe = corners(center, size, rotation*sensor_rotation*pad_rotation)
        row["top_grasp"] = {"shoe_min_z_m": min(p[2] for p in shoe), "tcp_z_m": tcp[2],
            "brick_top_z_m": block_top, "shoe_brick_clearance_m": min(p[2] for p in shoe)-block_top,
            "shoe_tcp_clearance_m": min(p[2] for p in shoe)-tcp[2]}
        tcp = support.support_pose("beam", side)
        rotation = Rotation.from_quat(tcp[3:])
        direction = rotation.apply([0, 0, 1])
        palm = [tcp[i]-rotation.apply(geometry["tcp_in_palm"])[i] for i in range(3)]
        center = rotation.apply(sensor[:3])+palm
        shoe = corners(center, size, rotation*sensor_rotation*pad_rotation)
        point_local = rotation.inv().apply([point[i]-palm[i] for i in range(3)])
        tips, all_fingers = [], []
        finger_size = geometry["finger_size"]
        for finger_center in geometry["finger_centres_in_palm"]:
            front_z = finger_center[2]+finger_size[2]/2
            face_center = rotation.apply([finger_center[0], finger_center[1], front_z])+palm
            nearest = [min(max(point_local[i], finger_center[i]-finger_size[i]/2),
                           finger_center[i]+finger_size[i]/2) for i in (0, 1)]+[front_z]
            nearest_world = rotation.apply(nearest)+palm
            face = [(rotation.apply([finger_center[0]+a*finger_size[0]/2,
                                      finger_center[1]+b*finger_size[1]/2, front_z])+palm).tolist()
                    for a, b in product((-1, 1), repeat=2)]
            all_fingers.extend(corners(rotation.apply(finger_center)+palm, finger_size, rotation))
            width_inside = lambda values: all(abs(p[1]-beam[1]) <= beam_size[1]/2+1e-12 for p in values)
            tips.append({"face_center_world_m": face_center.tolist(), "face_corners_world_m": face,
                "face_center_distance_to_support_m": math.dist(face_center, point),
                "nearest_tip_face_distance_to_support_m": math.dist(nearest_world, point),
                "nearest_tip_face_point_world_m": nearest_world.tolist(),
                "farthest_tip_face_corner_distance_m": max(math.dist(p, point) for p in face),
                "centre_within_80mm": math.dist(face_center, point) <= .08,
                "nearest_face_within_80mm": math.dist(nearest_world, point) <= .08,
                "centre_in_beam_width": width_inside([face_center]),
                "whole_face_in_beam_width": width_inside(face)})
        row["support"] = {"tcp_world": tcp, "support_point_world_m": point,
            "tcp_axial_compensation_m": sum((tcp[i]-point[i])*direction[i] for i in range(3)),
            "shoe_top_to_beam_bottom_m": max(p[2] for p in shoe)-point[2],
            "shoe_top_xy_offset_from_support_m": math.dist(center[:2], point[:2]),
            "finger_upper_edge_clearance_m": point[2]-max(p[2] for p in all_fingers),
            "beam_width_m": beam_size[1], "tips": tips}
        measurements[side] = row
    fixed = all(row["top_grasp"]["shoe_brick_clearance_m"] >= .005
                and abs(row["sensor_in_palm"][2]-.035) < 1e-12
                and abs(row["support"]["tcp_axial_compensation_m"]-.036) < 1e-12
                and abs(row["support"]["shoe_top_to_beam_bottom_m"]) < 1e-10
                and row["support"]["shoe_top_xy_offset_from_support_m"] < 1e-10
                and row["support"]["finger_upper_edge_clearance_m"] >= .008
                for row in measurements.values())
    strict_centre = all(tip["centre_within_80mm"] for row in measurements.values() for tip in row["support"]["tips"])
    return {"geometry_source": "未改的apexmr/scene_helpers.py原始实体尺寸/TCP/手指位姿AST，实际configure_support_tools装配和Frames位姿",
        "original_gripper": geometry, "arms": measurements,
        "fixed_shift_clearance_and_contact_passed": fixed,
        "strict_tip_centre_80mm_passed": strict_centre,
        "nearest_tip_face_80mm_passed": all(t["nearest_face_within_80mm"] for r in measurements.values() for t in r["support"]["tips"]),
        "tip_centres_in_beam_width": all(t["centre_in_beam_width"] for r in measurements.values() for t in r["support"]["tips"]),
        "whole_tip_faces_in_beam_width": all(t["whole_face_in_beam_width"] for r in measurements.values() for t in r["support"]["tips"]),
        "exact_geometry_contract_passed": fixed and strict_centre,
        "measurement_policy": "指尖面是手指局部+z端矩形面；分别公开面中心、到实体面的最短距离、最远角及世界y投影。最近实体面通过不等于中心通过；所有阈值保持原值。",
        "scope": "纯几何，不代表真实IK、碰撞证书或动力学通过"}


def main(argv=None):
    parser = argparse.ArgumentParser(description="仅测量D托板几何，不启动仿真器")
    parser.add_argument("--out", type=Path, default=ROOT/"results/sim_v1_3/v133/support_geometry.json")
    args = parser.parse_args(argv)
    output = args.out.resolve()
    if not output.is_relative_to((ROOT/"results/sim_v1_3/v133").resolve()):
        parser.error("几何报告只能写入新的v133目录")
    report = measure()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in {"arms", "original_gripper"}},ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
