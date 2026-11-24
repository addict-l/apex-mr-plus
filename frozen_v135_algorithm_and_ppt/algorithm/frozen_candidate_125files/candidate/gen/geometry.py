"""Generator checks and distance evidence, without simulation or search."""
from copy import deepcopy
from itertools import combinations
import math

from skillnet.core.logic import parse_condition
from simbridge.frames import Frames
from simbridge.layout import BASE_POSITIONS, check_layout


VALID_RADIUS = .72
POSSIBLE_RADIUS = .82
DISTANCE_EPS = 1e-12


def reach_status(arm, position):
    """Classify the task reference point's horizontal distance to its base."""
    distance = math.dist(BASE_POSITIONS[arm][:2], position[:2])
    if distance <= VALID_RADIUS + DISTANCE_EPS:
        return "VALID"
    if distance <= POSSIBLE_RADIUS + DISTANCE_EPS:
        return "UNKNOWN"
    return "INVALID"


def complete_reach_table(task):
    """Replace all template reach entries with exhaustive distance evidence."""
    table = {"default": "UNKNOWN", **{
        status.lower(): {arm: [] for arm in task["arms"]}
        for status in ("VALID", "UNKNOWN", "INVALID")}}
    for row in (*task.get("location", []), *task.get("slot", [])):
        for arm in task["arms"]:
            table[reach_status(arm, row["pos"]).lower()][arm].append(row["id"])
    task.setdefault("capabilities", {})["reach"] = table
    return table


def distance_prior_copy(task):
    """Explicit offline fallback; no input mutation and no IK claim."""
    result = deepcopy(task)
    complete_reach_table(result)
    result["capability_source"] = "distance_prior"
    return result


def candidate_pool_report(pools):
    """All candidates, including unused ones, are legal physical pad sites."""
    task = {"name": "candidate_pool", "location": [], "block": [], "slot": []}
    for arm, points in pools.items():
        if len(points) < 6:
            raise ValueError(f"臂 {arm} 的合法料位候选少于 6 个")
        task["location"].extend({"id": f"candidate_{arm}_{i}", "kind": "stock",
                                 "owner": arm, "pos": list(point)}
                                for i, point in enumerate(points))
    report = check_layout(task)
    if not report["valid"]:
        raise ValueError("料位候选池不合法：" + str(report["violations"]))
    return {"counts": {arm: len(points) for arm, points in pools.items()},
            "minimum_spacing_m": report["metrics"]["minimum_reference_spacing_m"],
            "stock_base_ranges": report["metrics"]["stock_base_ranges"]}


def check_generation_layout(task, family, parameters):
    """Extend workcell geometry with ground-goal spacing and static COM.

    This projected mass check does not certify dynamics, friction or real IK.
    It tests the beam and its installed anchors after temporary support is gone.
    """
    report = deepcopy(check_layout(task))
    frames = Frames(task)
    extra = {"minimum_target_spacing_m": None, "assembled_com": {}}

    def reject(code, objects, detail):
        report["violations"].append({"code": code, "objects": list(objects), "detail": detail})

    targets = {}
    if family in {"F1", "F2"}:
        for text in task["goal"]:
            condition = parse_condition(text)
            if not condition.negated and condition.fact.predicate == "at":
                _, place = condition.fact.args
                if frames.locations[place]["kind"] == "board":
                    targets[place] = frames.references[place][:2]
    elif family == "F3":
        # Stacked layers share XY intentionally. Only distinct columns count.
        targets = {name: frames.references[name][:2] for name, row in frames.slots.items()
                   if not row.get("below")}
    distances = []
    for first, second in combinations(targets, 2):
        distance = math.dist(targets[first], targets[second])
        distances.append(distance)
        if distance < .20 - DISTANCE_EPS:
            reject("target_spacing", [first, second], f"不同地面目标/塔列间距 {distance:.6f} m，小于 0.20 m")
    extra["minimum_target_spacing_m"] = min(distances, default=None)

    for name, row in frames.slots.items():
        if not row.get("overhang"):
            continue
        parts = [name, *row.get("anchors", [])]
        masses = {part: frames.catalog[frames.slots[part]["type"]].mass for part in parts}
        total = sum(masses.values())
        center = [sum(masses[part]*frames.slot_pose(part)[axis] for part in parts)/total
                  for axis in (0, 1)]
        lower = row.get("below", [])
        if not lower:
            reject("com_without_permanent_pillar", [name], "合成重心必须有永久柱脚印承托")
            continue
        # These generators use one aligned column. The union bounding box is
        # retained as evidence; disjoint support polygons are not generated.
        low = [min(frames.slot_pose(part)[axis]
                   - frames.catalog[frames.slots[part]["type"]].size[axis]/2 for part in lower)
               for axis in (0, 1)]
        high = [max(frames.slot_pose(part)[axis]
                    + frames.catalog[frames.slots[part]["type"]].size[axis]/2 for part in lower)
                for axis in (0, 1)]
        margin = min(*(center[axis]-low[axis] for axis in (0, 1)),
                     *(high[axis]-center[axis] for axis in (0, 1)))
        extra["assembled_com"][name] = {"parts": parts, "masses_kg": masses,
            "total_mass_kg": total, "xy_m": center, "pillar_footprint_min_xy_m": low,
            "pillar_footprint_max_xy_m": high, "minimum_margin_m": margin,
            "inside_permanent_pillar": margin >= -DISTANCE_EPS,
            "scope": "梁与配重的静态投影重心；不代表动力学稳定性验收"}
        if margin < -DISTANCE_EPS:
            reject("assembled_com_outside_pillar", parts+lower, "移除临时承托后，梁与配重的合成重心落在柱脚印之外")
    pools = parameters.get("stock_candidates")
    if pools is not None:
        extra["stock_candidates"] = candidate_pool_report(pools)
    report["metrics"]["generator_checks"] = extra
    report["valid"] = not report["violations"]
    return report
