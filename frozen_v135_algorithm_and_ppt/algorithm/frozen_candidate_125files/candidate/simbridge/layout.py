"""Pure geometric checks for generated workcells; no simulator or search."""
from itertools import combinations
import math

from scipy.spatial.transform import Rotation

from skillnet.core.logic import parse_condition
from .frames import BOARD_TOP, TABLE_TOP, Frames


PAD_SIZE = (.18, .18, .012)
TABLE_SIZE = (1.90, 1.27, .065)
TABLE_CENTER = (0., 0., TABLE_TOP-TABLE_SIZE[2]/2)
BOARD_SIZE = (.70, .72, .012)
BOARD_CENTER = (0., .12, BOARD_TOP-BOARD_SIZE[2]/2)
BASE_POSITIONS = {"A": (-.70, .13, .783), "B": (.70, .13, .783)}
MOUNT_SIZE = (.23, .24, .022)
MIN_REFERENCE_SPACING = .20
STOCK_REACH_RANGE = (.30, .65)
EPS = 1e-9


def bounds(center, size, quaternion=(0., 0., 0., 1.)):
    """World AABB; all current fixtures have axis-aligned body frames."""
    matrix = Rotation.from_quat(quaternion).as_matrix()
    half = [sum(abs(matrix[i, j])*size[j]/2 for j in range(3)) for i in range(3)]
    return {"min": [center[i]-half[i] for i in range(3)],
            "max": [center[i]+half[i] for i in range(3)]}


def overlaps(first, second, *, axes=(0, 1, 2), tolerance=EPS):
    """Strict positive-volume overlap; touching bearing faces are allowed."""
    return all(min(first["max"][axis], second["max"][axis])
               - max(first["min"][axis], second["min"][axis]) > tolerance for axis in axes)


def contains(outer, inner, *, axes=(0, 1)):
    return all(outer["min"][axis]-EPS <= inner["min"][axis]
               and inner["max"][axis] <= outer["max"][axis]+EPS for axis in axes)


def check_layout(task, catalog=None):
    """Check one task path/raw dict/Problem without any IK or planning budget.

    Stock/buffer pads are bearing geometry. Handover markers are nonphysical
    visual labels, matching workcell.build; they still obey point spacing.
    The assembled snapshot includes temporary slots, so their coexistence
    with pillars and installed blocks cannot conceal a physical intersection.
    """
    frames = Frames(task, catalog)
    violations, stock_ranges, overhangs = [], {}, {}

    def reject(code, objects, detail):
        violations.append({"code": code, "objects": list(objects), "detail": detail})

    board = bounds(BOARD_CENTER, BOARD_SIZE)
    table = bounds(TABLE_CENTER, TABLE_SIZE)
    static = {"Workbench": table, "Baseplate": board}
    for side, position in BASE_POSITIONS.items():
        static[side+"_Mount"] = bounds([position[0], position[1], .771], MOUNT_SIZE)

    references = {name: row for name, row in frames.locations.items()
                  if row["kind"] in {"stock", "buffer", "handover"}}
    minimum_spacing = None
    for first, second in combinations(references, 2):
        distance = math.dist(frames.references[first][:2], frames.references[second][:2])
        minimum_spacing = distance if minimum_spacing is None else min(minimum_spacing, distance)
        if distance < MIN_REFERENCE_SPACING-EPS:
            reject("reference_spacing", [first, second], f"水平间距 {distance:.6f} m，小于 0.20 m")

    capability_reach = frames.task.get("capabilities", {}).get("reach", {}).get("valid", {})
    for name, row in references.items():
        reference = frames.references[name]
        if not all(math.isfinite(value) for value in reference):
            reject("nonfinite_reference", [name], "位置坐标必须有限")
            continue
        if row["kind"] == "stock":
            owner = row.get("owner")
            exclusive = [side for side in BASE_POSITIONS if name in capability_reach.get(side, [])]
            owner = owner or (exclusive[0] if len(exclusive) == 1 else
                              min(BASE_POSITIONS, key=lambda side: math.dist(reference[:2], BASE_POSITIONS[side][:2])))
            if owner not in BASE_POSITIONS:
                reject("stock_owner", [name], f"未知所属臂 {owner}")
            else:
                distance = math.dist(reference[:2], BASE_POSITIONS[owner][:2])
                stock_ranges[name] = {"owner": owner, "horizontal_distance_m": distance}
                if not STOCK_REACH_RANGE[0]-EPS <= distance <= STOCK_REACH_RANGE[1]+EPS:
                    reject("stock_base_range", [name, owner], f"料位到所属基座 {distance:.6f} m，不在 0.30–0.65 m")
        if row["kind"] in {"stock", "buffer"}:
            pad = bounds([reference[0], reference[1], reference[2]-PAD_SIZE[2]/2], PAD_SIZE)
            static["Pad_"+name] = pad
            if overlaps(board, pad, axes=(0, 1)):
                reject("pad_board_footprint", [name, "Baseplate"], "承托垫的水平投影必须完全位于底板之外")
            if not contains(table, pad):
                reject("pad_outside_table", [name, "Workbench"], "承托垫必须完全位于台面内")

    initial = dict(static)
    for name, row in frames.blocks.items():
        pose = frames.block_pose(name)
        body = bounds(pose[:3], frames.catalog[row["type"]].size, pose[3:])
        initial["Block_"+name] = body
        if not contains(table, body):
            reject("initial_body_outside_table", [name], "初始积木的水平投影超出台面")
    for first, second in combinations(initial, 2):
        if overlaps(initial[first], initial[second]):
            reject("initial_overlap", [first, second], "初始刚体存在正体积重叠")

    target = dict(static)
    slots = {}
    for name, row in frames.slots.items():
        pose = frames.slot_pose(name)
        shape = bounds(pose[:3], frames.catalog[row["type"]].size, pose[3:])
        slots[name] = shape
        target["Slot_"+name] = shape
        if not contains(board, shape):
            reject("slot_outside_board", [name], "槽位实体的水平投影超出底板")
    for name, row in frames.slots.items():
        lower = row.get("below", [])
        for below in lower:
            if not overlaps(slots[name], slots[below], axes=(0, 1)):
                reject("missing_bearing_overlap", [name, below], "上下层没有水平承托交叠")
            if abs(slots[name]["min"][2]-slots[below]["max"][2]) > EPS:
                reject("uneven_bearing", [name, below], "上下层承托面高度不一致")
        if not row.get("overhang"):
            continue
        if not lower:
            reject("overhang_without_pillar", [name], "悬臂必须有下层柱体")
            continue
        beam = slots[name]
        axis = max((0, 1), key=lambda i: beam["max"][i]-beam["min"][i])
        low = min(slots[part]["min"][axis] for part in lower)
        high = max(slots[part]["max"][axis] for part in lower)
        negative, positive = low-beam["min"][axis], beam["max"][axis]-high
        direction = -1 if negative >= positive else 1
        length = max(negative, positive)
        column_center = (low+high)/2
        beam_center = (beam["min"][axis]+beam["max"][axis])/2
        if length < .18-EPS or abs(column_center-beam_center) <= EPS:
            reject("insufficient_overhang", [name], f"柱体须偏离梁中心且悬出不少于 0.18 m，当前 {length:.6f} m")
        if beam["min"][2] < BOARD_TOP+.152-EPS:
            reject("low_beam", [name], "梁底面必须至少高于底板 0.152 m")
        for anchor in row.get("anchors", []):
            shape = slots[anchor]
            anchor_center = (shape["min"][axis]+shape["max"][axis])/2
            if direction*(anchor_center-column_center) >= -EPS or not contains(beam, shape):
                reject("counterweight_geometry", [name, anchor], "配重须位于柱的另一侧且完全落在梁上")
        temporary = [part for part, definition in frames.slots.items()
                     if definition.get("platform_for") == name]
        referenced = {parent for part in temporary for parent in frames.slots[part].get("below", [])}
        tops = [part for part in temporary if part not in referenced]
        for part in tops:
            shape = slots[part]
            center = (shape["min"][axis]+shape["max"][axis])/2
            if direction*(center-column_center) <= EPS or not contains(beam, shape):
                reject("platform_under_wrong_end", [name, part], "临时台必须位于悬出端正下方并落在梁投影内")
            if abs(shape["max"][2]-beam["min"][2]) > EPS:
                reject("platform_height", [name, part], "临时台顶面必须与柱顶/梁底等高")
        overhangs[name] = {"axis": axis, "free_direction": direction,
                          "length_m": length, "beam_bottom_m": beam["min"][2],
                          "pillar_center": column_center, "platform_top_slots": tops}

    conditions = (task.goal if hasattr(task, "goal") else
                  [parse_condition(text) for text in frames.task.get("goal", [])])
    for condition in conditions:
        if condition.negated or condition.fact.predicate != "at":
            continue
        block, place = condition.fact.args
        pose = frames.block_pose(block, place)
        shape = bounds(pose[:3], frames.catalog[frames.blocks[block]["type"]].size, pose[3:])
        target["Goal_"+block] = shape
        if frames.locations.get(place, {}).get("kind") == "board" and not contains(board, shape):
            reject("goal_outside_board", [block, place], "目标积木的水平投影超出底板")
    for first, second in combinations(target, 2):
        if overlaps(target[first], target[second]):
            reject("target_overlap", [first, second], "装配中共存的实体存在正体积重叠")

    return {"task": frames.task.get("name"), "valid": not violations,
            "violations": violations,
            "metrics": {"minimum_reference_spacing_m": minimum_spacing,
                        "stock_base_ranges": stock_ranges, "overhangs": overhangs,
                        "initial_shape_count": len(initial), "target_shape_count": len(target)},
            "slot_poses": {name: frames.slot_pose(name) for name in frames.slots},
            "visual_handover_markers": [name for name, row in references.items() if row["kind"] == "handover"],
            "scope": "纯几何：正体积 AABB 相交、接触面、承托垫、臂座、边界和悬臂布局；不代表 IK、动力学或稳定性验收"}


def assert_layout(task, catalog=None):
    report = check_layout(task, catalog)
    if not report["valid"]:
        raise ValueError("工作站布局不合法：" + "; ".join(
            item["code"]+" "+", ".join(item["objects"]) for item in report["violations"]))
    return report
