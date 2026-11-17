"""Block scene loading, derived conditions, and structural invariants."""

from itertools import combinations
import math
import tomllib

from skillnet.capability.oracle import CapabilityOracle
from skillnet.core.library import load_library
from skillnet.core.logic import Condition, Fact, is_identifier, parse_condition
from skillnet.core.problem import Problem, Proposal


DERIVED_BASIS = {
    "slot_ready": frozenset({"filled"}),
    "slot_clear": frozenset(),
    "has_support": frozenset({"supporting", "filled"}),
    "insert_supported": frozenset({"supporting", "filled"}),
}


class BlocksHooks:
    derived_predicates = frozenset(DERIVED_BASIS)
    derived_basis = DERIVED_BASIS
    resource_predicates = frozenset({"free", "arm_at"})

    def __init__(self, arms, locations, blocks, slots, positions, speeds):
        self.arms = tuple(arms)
        self.locations = locations
        self.blocks = blocks
        self.slots = slots
        self.positions = positions
        self.speeds = speeds
        self.platforms = {
            slot: tuple(other for other, row in slots.items() if row.get("platform_for") == slot)
            for slot in slots
        }
        self.above = {
            slot: tuple(other for other, row in slots.items() if slot in row.get("below", ()))
            for slot in slots
        }

    def evaluate(self, pred, args, state):
        if len(args) != 1 or args[0] not in self.slots:
            return False
        slot = args[0]
        row = self.slots[slot]
        if pred == "slot_ready":
            return all(Fact("filled", (lower,)) in state for lower in row.get("below", ()))
        if pred == "slot_clear":
            return not any(Fact("filled", (upper,)) in state for upper in self.above[slot])
        if pred == "has_support":
            if any(Fact("supporting", (arm, slot)) in state for arm in self.arms):
                return True
            platform = self.platforms[slot]
            return bool(platform) and all(Fact("filled", (part,)) in state for part in platform)
        if pred == "insert_supported":
            return all(self.evaluate("has_support", (target,), state)
                       for target in row.get("needs_support_during", ()))
        return False

    def invariant(self, state):
        for slot, row in self.slots.items():
            if not row.get("overhang", False) or Fact("filled", (slot,)) not in state:
                continue
            anchored = all(Fact("filled", (anchor,)) in state for anchor in row["anchors"])
            if not anchored and not self.evaluate("has_support", (slot,), state):
                return f"槽位 {slot} 不稳定：锚定槽位尚未全部装好，且没有臂或完整临时支撑台提供支撑"
        return None

    def expand(self, fact):
        if fact.predicate not in self.derived_predicates:
            return [fact]
        if len(fact.args) != 1 or fact.args[0] not in self.slots:
            return []
        slot = fact.args[0]
        if fact.predicate == "slot_ready":
            return [Fact("filled", (lower,)) for lower in self.slots[slot].get("below", ())]
        if fact.predicate == "slot_clear":
            return []
        if fact.predicate == "has_support":
            return ([Fact("supporting", (arm, slot)) for arm in self.arms]
                    + [Fact("filled", (part,)) for part in self.platforms[slot]])
        return [base for target in self.slots[slot].get("needs_support_during", ())
                for base in self.expand(Fact("has_support", (target,)))]

    def justify(self, fact, state):
        if fact.predicate not in self.derived_predicates:
            return [fact] if fact in state else []
        if not self.evaluate(fact.predicate, fact.args, state):
            return []
        slot = fact.args[0]
        if fact.predicate == "slot_ready":
            return self.expand(fact)
        if fact.predicate == "slot_clear":
            return []
        if fact.predicate == "has_support":
            for arm in self.arms:
                supporting = Fact("supporting", (arm, slot))
                if supporting in state:
                    return [supporting]
            return [Fact("filled", (part,)) for part in self.platforms[slot]]
        return [base for target in self.slots[slot].get("needs_support_during", ())
                for base in self.justify(Fact("has_support", (target,)), state)]

    def invariant_needs(self, action):
        result = []
        for fact in action.add:
            if fact.predicate != "filled" or len(fact.args) != 1:
                continue
            slot = fact.args[0]
            row = self.slots.get(slot, {})
            if row.get("overhang", False):
                result.append(Fact("has_support", (slot,)))
                result.extend(Fact("filled", (anchor,)) for anchor in row["anchors"])
        return result

    def optimistic_time(self, fact, times):
        """Earliest derived truth in the delete-relaxed temporal problem."""
        if fact.predicate not in self.derived_predicates:
            return times.get(fact, math.inf)
        if len(fact.args) != 1 or fact.args[0] not in self.slots:
            return math.inf
        slot = fact.args[0]
        row = self.slots[slot]
        if fact.predicate == "slot_clear":
            return 0.0
        if fact.predicate == "slot_ready":
            return max((times.get(Fact("filled", (lower,)), math.inf)
                        for lower in row.get("below", ())), default=0.0)
        if fact.predicate == "has_support":
            alternatives = [times.get(Fact("supporting", (arm, slot)), math.inf) for arm in self.arms]
            platform = self.platforms[slot]
            if platform:
                alternatives.append(max(times.get(Fact("filled", (part,)), math.inf) for part in platform))
            return min(alternatives, default=math.inf)
        return max((self.optimistic_time(Fact("has_support", (target,)), times)
                    for target in row.get("needs_support_during", ())), default=0.0)

    def optimistic_effect_time(self, fact, earliest, times, action=None):
        """An overhang cannot be filled earlier than its first stable time."""
        if fact.predicate != "filled" or len(fact.args) != 1:
            return earliest
        slot = fact.args[0]
        row = self.slots.get(slot, {})
        if not row.get("overhang", False):
            return earliest
        anchors = max((times.get(Fact("filled", (anchor,)), math.inf)
                       for anchor in row["anchors"]), default=0.0)
        excluded = set(action.occupies) if action is not None and action.name in {"insert", "co_insert"} else set()
        alternatives = [times.get(Fact("supporting", (arm, slot)), math.inf)
                        for arm in self.arms if arm not in excluded]
        platform = self.platforms[slot]
        if platform:
            alternatives.append(max(times.get(Fact("filled", (part,)), math.inf) for part in platform))
        support = min(alternatives, default=math.inf)
        return max(earliest, min(anchors, support))

    def position(self, place):
        return self.positions[place]

    def reference_actions(self, actions, initial, goal):
        """Restrict only a feasible-reference search, never the final UCS."""
        positive = {getattr(c, "fact", c) for c in goal if not getattr(c, "negated", False)}
        protected = {f for f in positive if f.predicate not in self.resource_predicates}
        placements = {f for f in initial | positive if f.predicate == "at"}
        result = set()
        for action in actions:
            if protected.intersection(action.delete):
                continue
            if any(f.predicate == "at" and f not in placements for f in action.add):
                continue
            if any(c.fact.predicate == "at" and not c.negated and c.fact not in placements
                   for c in action.pre):
                continue
            result.add(action.key)
        return frozenset(result)

    def speed(self, arm):
        return self.speeds[arm]

    def scene_summary(self):
        lines = []
        for block, row in self.blocks.items():
            placement = f"位于 {row['at']}" if "at" in row else f"已装入 {row['installed_in']}"
            lines.append(f"积木 {block}，类型 {row['type']}，{placement}")
        for place, row in self.locations.items():
            lines.append(f"位置 {place}，类别 {row['kind']}，坐标 {self.positions[place]}")
        for slot, row in self.slots.items():
            below = ", ".join(row.get("below", ())) or "底板"
            detail = f"槽位 {slot}，类型 {row['type']}，下接 {below}，坐标 {self.positions[slot]}"
            if row.get("overhang"):
                detail += f"，悬空，锚定槽位 {', '.join(row['anchors'])}"
            if row.get("platform_for"):
                detail += f"，临时支撑 {row['platform_for']}"
            if row.get("support_point"):
                detail += "，允许臂托举"
            if row.get("needs_support_during"):
                detail += f"，插装时需支撑 {', '.join(row['needs_support_during'])}"
            lines.append(detail)
        return lines


def _vector(value, label):
    if (not isinstance(value, (list, tuple)) or len(value) != 3
            or any(isinstance(item, bool) or not isinstance(item, (int, float))
                   or not math.isfinite(item) for item in value)):
        raise ValueError(f"{label} 必须是三个有限数值组成的坐标")
    return tuple(float(item) for item in value)


def _rows(data, key, seen):
    rows = data.get(key, [])
    if not isinstance(rows, list):
        raise ValueError(f"{key} 必须是表列表")
    result = {}
    for row in rows:
        if not isinstance(row, dict) or not is_identifier(row.get("id")):
            raise ValueError(f"{key} 的 id 必须是标识符")
        identifier = row["id"]
        if identifier in seen:
            raise ValueError(f"id 全局重复：{identifier}")
        seen.add(identifier)
        result[identifier] = dict(row)
    return result


def _references(row, key, slots):
    values = row.get(key, [])
    if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
        raise ValueError(f"槽位 {row['id']} 的 {key} 必须是槽位名称列表")
    for value in values:
        if value not in slots:
            raise ValueError(f"槽位 {row['id']} 的 {key} 引用了不存在的槽位 {value}")
    return values


def problem_from_data(data, library=None):
    if not isinstance(data, dict):
        raise ValueError("任务必须是 TOML 表")
    instruction = data.get("instruction", "")
    if not isinstance(instruction, str):
        raise ValueError("任务 instruction 必须是字符串")
    instruction = instruction.strip()
    name = data.get("name")
    if not is_identifier(name):
        raise ValueError("任务 name 必须是标识符")
    arms = data.get("arms")
    if not isinstance(arms, list) or not arms or any(not is_identifier(arm) for arm in arms):
        raise ValueError("arms 必须是非空臂名称列表")
    if len(set(arms)) != len(arms):
        raise ValueError("arms 存在重复的臂名称")
    seen = set()
    arm_rows = _rows(data, "arm", seen)
    locations = _rows(data, "location", seen)
    blocks = _rows(data, "block", seen)
    slots = _rows(data, "slot", seen)
    if set(arms) != set(arm_rows):
        raise ValueError("arms 与 [[arm]] 的 id 必须完全一致")
    positions = {}
    speeds = {}
    initial = set()
    static = set()
    for arm in arms:
        home = f"home_{arm}"
        if home in seen:
            raise ValueError(f"id 全局重复：{home} 与自动生成的初始位置冲突")
        seen.add(home)
        positions[home] = _vector(arm_rows[arm].get("home"), f"臂 {arm} 的 home")
        speed = arm_rows[arm].get("speed")
        if isinstance(speed, bool) or not isinstance(speed, (int, float)) or not math.isfinite(speed) or speed <= 0:
            raise ValueError(f"臂 {arm} 的 speed 必须是大于 0 的有限数值")
        speeds[arm] = float(speed)
        initial.update((Fact("free", (arm,)), Fact("arm_at", (arm, home))))
    for place, row in locations.items():
        if row.get("kind") not in {"stock", "buffer", "board", "handover"}:
            raise ValueError(f"位置 {place} 的 kind 必须是 stock、buffer、board 或 handover")
        positions[place] = _vector(row.get("pos"), f"位置 {place} 的 pos")
        predicate = "handover_point" if row["kind"] == "handover" else "placeable"
        static.add(Fact(predicate, (place,)))
    for slot, row in slots.items():
        positions[slot] = _vector(row.get("pos"), f"槽位 {slot} 的 pos")
        if not isinstance(row.get("type"), str) or not row["type"]:
            raise ValueError(f"槽位 {slot} 缺少有效的 type")
        for flag in ("overhang", "support_point", "temporary"):
            if not isinstance(row.get(flag, False), bool):
                raise ValueError(f"槽位 {slot} 的 {flag} 必须是布尔值")
            if row.get(flag, False):
                static.add(Fact(flag, (slot,)))
        for key, predicate, reverse in (("below", "below", True), ("anchors", "anchor", False),
                                        ("needs_support_during", "needs_support_during", False)):
            for target in _references(row, key, slots):
                args = (target, slot) if reverse else (slot, target)
                static.add(Fact(predicate, args))
        if row.get("overhang", False) and not row.get("anchors"):
            raise ValueError(f"槽位 {slot} 的 overhang=true 时 anchors 不能为空")
        platform_for = row.get("platform_for", "")
        if not isinstance(platform_for, str):
            raise ValueError(f"槽位 {slot} 的 platform_for 必须是字符串")
        if platform_for:
            if not row.get("temporary", False):
                raise ValueError(f"槽位 {slot} 的 platform_for 只能用于临时槽位")
            if platform_for not in slots:
                raise ValueError(f"槽位 {slot} 的 platform_for 引用了不存在的槽位 {platform_for}")
            static.add(Fact("platform_part", (slot, platform_for)))
    visiting = set()
    visited = set()

    def visit(slot):
        if slot in visiting:
            raise ValueError(f"below 关系成环，涉及槽位 {slot}")
        if slot in visited:
            return
        visiting.add(slot)
        for lower in slots[slot].get("below", ()):
            visit(lower)
        visiting.remove(slot)
        visited.add(slot)

    for slot in slots:
        visit(slot)
    occupied_locations = set()
    occupied_slots = set()
    for block, row in blocks.items():
        if not isinstance(row.get("type"), str) or not row["type"]:
            raise ValueError(f"积木 {block} 缺少有效的 type")
        if ("at" in row) == ("installed_in" in row):
            raise ValueError(f"积木 {block} 必须在 at 与 installed_in 中二选一")
        if "at" in row:
            place = row["at"]
            if not isinstance(place, str) or place not in locations:
                raise ValueError(f"积木 {block} 引用了不存在的位置 {place}")
            if place in occupied_locations:
                raise ValueError(f"位置 {place} 重复放置积木")
            occupied_locations.add(place)
            initial.add(Fact("at", (block, place)))
        else:
            slot = row["installed_in"]
            if not isinstance(slot, str) or slot not in slots:
                raise ValueError(f"积木 {block} 引用了不存在的槽位 {slot}")
            if row["type"] != slots[slot]["type"]:
                raise ValueError(f"积木 {block} 与 installed_in 槽位 {slot} 的类型不匹配")
            if slot in occupied_slots:
                raise ValueError(f"槽位 {slot} 重复安装积木")
            occupied_slots.add(slot)
            initial.update((Fact("installed", (block, slot)), Fact("filled", (slot,))))
        if not isinstance(row.get("two_arms", False), bool):
            raise ValueError(f"积木 {block} 的 two_arms 必须是布尔值")
        if row.get("two_arms", False):
            static.add(Fact("two_arm_part", (block,)))
        for slot, slot_row in slots.items():
            if row["type"] == slot_row["type"]:
                static.add(Fact("fits", (block, slot)))
    for place, row in locations.items():
        if row["kind"] != "handover" and place not in occupied_locations:
            initial.add(Fact("empty", (place,)))
    static.update(Fact("arm_pair", pair) for pair in combinations(arms, 2))
    fill_all = data.get("fill_all_slots", True)
    if not isinstance(fill_all, bool):
        raise ValueError("fill_all_slots 必须是布尔值")
    goal = [Condition(Fact("free", (arm,))) for arm in arms]
    for slot, row in slots.items():
        if row.get("temporary", False):
            goal.append(Condition(Fact("filled", (slot,)), True))
        elif fill_all:
            goal.append(Condition(Fact("filled", (slot,))))
    extra_goal = data.get("goal", [])
    if not isinstance(extra_goal, list):
        raise ValueError("goal 必须是条件字符串列表")
    explicit_goal = []
    for text in extra_goal:
        condition = parse_condition(text)
        if any(arg not in seen for arg in condition.fact.args):
            raise ValueError(f"目标 {text!r} 引用了不存在的对象")
        if condition not in explicit_goal:
            explicit_goal.append(condition)
        if condition not in goal:
            goal.append(condition)
    hooks = BlocksHooks(arms, locations, blocks, slots, positions, speeds)
    violation = hooks.invariant(initial)
    if violation:
        raise ValueError(f"初始状态错误：{violation}")
    if library is None:
        library = load_library(derived_basis=DERIVED_BASIS)
    overlap = library.dynamic_predicates & hooks.derived_predicates
    if overlap:
        raise ValueError(f"谓词不能同时是动态和推导条件：{', '.join(sorted(overlap))}")
    proposal = None
    if "proposal" in data:
        raw = data["proposal"]
        if not isinstance(raw, dict):
            raise ValueError("proposal 必须是表")
        for field in ("steps", "goal"):
            values = raw.get(field, [])
            if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
                raise ValueError(f"proposal.{field} 必须是字符串列表")
        if not isinstance(raw.get("instruction", ""), str):
            raise ValueError("proposal.instruction 必须是字符串")
        proposal = Proposal(raw.get("instruction", ""), raw.get("goal", list(extra_goal)),
                            raw.get("steps", []), "任务文件")
    return Problem(name, data.get("description", ""),
                   {"arm": tuple(arms), "loc": tuple(locations), "block": tuple(blocks), "slot": tuple(slots)},
                   frozenset(static), frozenset(initial), tuple(goal), library, hooks,
                   CapabilityOracle(data.get("capabilities", {})), proposal,
                   instruction=instruction, explicit_goal=tuple(explicit_goal))


def load_problem(path, skills="skills/blocks_v1.toml"):
    try:
        with open(path, "rb") as source:
            data = tomllib.load(source)
    except tomllib.TOMLDecodeError as error:
        raise ValueError(f"任务 TOML 格式错误：{error}") from error
    library = load_library(skills, derived_basis=DERIVED_BASIS)
    return problem_from_data(data, library)
