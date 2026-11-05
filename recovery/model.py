"""Explicit holding/support state and input checks for the problem-1 prototype.

Capabilities are declared three-valued facts, not IK or path certificates.
Snapshots must be taken at a quiescent skill boundary. Continuous support is
part of State, so finishing acquire_support does not make that arm available.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import math
import re
from typing import Any, Mapping


ARMS = ("A", "B")
STATUSES = {"VALID", "INVALID", "UNKNOWN"}
OPERATIONS = {"pick", "place", "install", "support", "handover", "work"}


def number(value: Any, name: str, minimum: float = 0.) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return value


def _names(values, label):
    if not isinstance(values, list) or any(not isinstance(v, str) or not v for v in values):
        raise ValueError(f"{label} must be a list of nonempty strings")
    if len(set(values)) != len(values):
        raise ValueError(f"{label} contains duplicates")
    return values


@dataclass(frozen=True)
class State:
    locations: tuple[tuple[str, str], ...]
    supports: tuple[tuple[str, str], ...]
    completed: frozenset[str]

    @classmethod
    def from_dict(cls, data: Mapping) -> State:
        if data.get("running"):
            raise ValueError("snapshot must be quiescent: running skills cannot be discarded")
        return cls(tuple(sorted(data["locations"].items())),
                   tuple(sorted(data.get("supports", {}).items())),
                   frozenset(_names(data.get("completed", []), "completed")))

    def to_dict(self) -> dict:
        return {"locations": dict(self.locations), "supports": dict(self.supports),
                "completed": sorted(self.completed)}

    def held(self, arm: str) -> str | None:
        return next((p for p, location in self.locations if location == f"held:{arm}"), None)

    def free(self, arm: str) -> bool:
        return not self.held(arm) and arm not in dict(self.supports)


@dataclass(frozen=True)
class Action:
    kind: str
    arm: str
    target: str
    duration: float
    part: str | None = None
    destination: str | None = None
    receiver: str | None = None

    @property
    def actors(self) -> tuple[str, ...]:
        return (self.arm, self.receiver) if self.receiver else (self.arm,)

    def to_dict(self) -> dict:
        return {**asdict(self), "resources": list(self.actors)}

    @classmethod
    def from_dict(cls, value: Mapping) -> Action:
        return cls(**{key: value.get(key) for key in cls.__dataclass_fields__})


class Problem:
    def __init__(self, data: Mapping):
        self.data = deepcopy(dict(data))
        self.name = str(data.get("name", "custom"))
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.name):
            raise ValueError("scenario name must be a simple identifier, not a file path")
        self.parts = {p["id"]: deepcopy(p) for p in data.get("parts", [])}
        if len(self.parts) != len(data.get("parts", [])):
            raise ValueError("duplicate part id")
        for pid, part in self.parts.items():
            if not isinstance(pid, str) or not pid:
                raise ValueError("part ids must be nonempty strings")
            number(part["mass"], f"{pid}.mass")
        self.tasks = {t["id"]: deepcopy(t) for t in data["tasks"]}
        if len(self.tasks) != len(data["tasks"]):
            raise ValueError("duplicate task id")
        self.slots = _names(data.get("slots", []), "slots")
        self.buffers = _names(data.get("buffers", []), "buffers")
        if not set(self.buffers) <= set(self.slots):
            raise ValueError("buffers must be declared slots")
        self.initial = State.from_dict(data["snapshot"])
        self.capabilities = deepcopy(data["capabilities"])
        if set(self.capabilities) != set(ARMS):
            raise ValueError("capabilities must declare A and B")
        for arm, operations in self.capabilities.items():
            if not set(operations) <= OPERATIONS:
                raise ValueError(f"unknown capability operation for {arm}")
            for operation, facts in operations.items():
                if not isinstance(facts, Mapping) or any(v not in STATUSES for v in facts.values()):
                    raise ValueError(f"{arm}.{operation} must map targets to VALID/INVALID/UNKNOWN")
        installed_parts = []
        for tid, task in self.tasks.items():
            if not isinstance(tid, str) or not tid or task.get("kind") not in {"install", "work"}:
                raise ValueError("tasks need nonempty id and kind install/work")
            if task.get("original_arm") not in ARMS:
                raise ValueError(f"{tid}: original_arm must be A or B")
            if number(task["duration"], f"{tid}.duration") <= 0:
                raise ValueError("action durations must be positive")
            deps = _names(task.get("dependencies", []), f"{tid}.dependencies")
            if tid in deps or not set(deps) <= self.tasks.keys() | self.initial.completed:
                raise ValueError(f"{tid}: unknown or self dependency")
            if task["kind"] == "install":
                if task.get("part") not in self.parts:
                    raise ValueError(f"{tid}: install references unknown part")
                installed_parts.append(task["part"])
                number(task.get("insertion_force_N", 0.), f"{tid}.insertion_force_N")
                for key in ("load_x_m", "load_z_m"):
                    number(task.get(key, 0.), f"{tid}.{key}", -math.inf)
            elif task.get("independent_of_assembly") is not True:
                raise ValueError("work tasks must explicitly be independent_of_assembly")
        if len(set(installed_parts)) != len(installed_parts):
            raise ValueError("this finite model supports one install goal per part")
        pending = set(self.tasks)
        known = set(self.initial.completed)
        while pending:
            ready = {t for t in pending if set(self.tasks[t].get("dependencies", [])) <= known}
            if not ready:
                raise ValueError("task dependency cycle")
            known.update(ready)
            pending -= ready
        self.physics = deepcopy(data["physics"])
        self._check_physics()
        self.durations = {"pick": 1., "place": 1., "handover": 1.5,
                          "acquire_support": 1., "release_support": .5,
                          **deepcopy(data.get("durations", {}))}
        for key, value in self.durations.items():
            if number(value, f"durations.{key}") <= 0:
                raise ValueError("action durations must be positive")
        self.validate_state(self.initial)

    def _check_physics(self):
        physics = self.physics
        number(physics["gravity_m_s2"], "gravity_m_s2")
        for load in physics.get("loads", []):
            for key in ("fx_N", "fz_N", "x_m", "z_m", "moment_Nm"):
                number(load.get(key, 0.), f"load.{key}", -math.inf)
        for key in ("force_x_limit_N", "force_z_limit_N", "moment_limit_Nm"):
            number(physics["connection"][key], f"connection.{key}")
        reinforcement = physics["connection"].get("reinforced_by")
        if reinforcement:
            if reinforcement not in self.tasks or self.tasks[reinforcement]["kind"] != "install":
                raise ValueError("reinforced_by must reference an install goal")
            number(physics["connection"]["reinforced_moment_limit_Nm"], "reinforced moment limit")
        for cid, contact in physics.get("contacts", {}).items():
            if contact["arm"] not in ARMS:
                raise ValueError(f"{cid}: invalid support arm")
            for key in ("x_m", "z_m"):
                number(contact.get(key, 0.), f"{cid}.{key}", -math.inf)
            for key in ("normal_limit_N", "friction"):
                number(contact[key], f"{cid}.{key}")

    def validate_state(self, state: State):
        locations, supports = dict(state.locations), dict(state.supports)
        if set(locations) != set(self.parts):
            raise ValueError("snapshot must locate every declared part exactly once")
        occupied_slots = []
        for pid, loc in locations.items():
            if loc not in self.slots and loc not in {"installed", "held:A", "held:B"}:
                raise ValueError(f"{pid}: unknown location {loc}")
            if loc in self.slots:
                occupied_slots.append(loc)
        if len(set(occupied_slots)) != len(occupied_slots):
            raise ValueError("two parts cannot occupy the same finite slot")
        for arm in ARMS:
            if sum(loc == f"held:{arm}" for loc in locations.values()) > 1:
                raise ValueError("one arm cannot hold two parts")
            if arm in supports:
                contact = self.physics.get("contacts", {}).get(supports[arm])
                if not contact or contact["arm"] != arm or state.held(arm):
                    raise ValueError("support must bind its own empty arm to a declared contact")
        if not set(supports) <= set(ARMS):
            raise ValueError("unknown supporting arm")
        for tid, task in self.tasks.items():
            if tid in state.completed and not set(task.get("dependencies", [])) <= state.completed:
                raise ValueError("completed task is missing an actual prerequisite")
            if task["kind"] == "install":
                if (tid in state.completed) != (locations[task["part"]] == "installed"):
                    raise ValueError("installed part and completed install record disagree")

    def status(self, arm: str, operation: str, target: str) -> str:
        # Missing declarations are deliberately UNKNOWN, never an implicit pass.
        return self.capabilities[arm].get(operation, {}).get(target, "UNKNOWN")

    def finished(self, state: State) -> bool:
        return (self.tasks.keys() <= state.completed and not state.supports
                and all(state.held(arm) is None for arm in ARMS))


def transition(problem: Problem, state: State, action: Action) -> State:
    """Apply the completed skill effect, after its preconditions were checked."""
    locations, supports, completed = dict(state.locations), dict(state.supports), set(state.completed)
    if action.kind == "pick":
        locations[action.part] = f"held:{action.arm}"
    elif action.kind == "place":
        locations[action.part] = action.destination
    elif action.kind == "handover":
        locations[action.part] = f"held:{action.receiver}"
    elif action.kind == "acquire_support":
        supports[action.arm] = action.target
    elif action.kind == "release_support":
        del supports[action.arm]
    elif action.kind in {"install", "work"}:
        completed.add(action.target)
        if action.kind == "install":
            locations[action.part] = "installed"
    else:
        raise ValueError(f"unknown action kind {action.kind}")
    result = State(tuple(sorted(locations.items())), tuple(sorted(supports.items())), frozenset(completed))
    problem.validate_state(result)
    return result


def apply_event(problem: Problem, event: Mapping | None) -> tuple[Problem, list[str]]:
    """Apply declared capability/physics edits without changing physical history."""
    data = deepcopy(problem.data)
    changed = []
    for update in (event or {}).get("updates", []):
        path = update["fact"].split("/")
        if path[0] not in {"capabilities", "physics"} or len(path) < 2:
            raise ValueError("events may update capability/physics facts only")
        cursor = data
        for key in path[:-1]:
            if key not in cursor or not isinstance(cursor[key], dict):
                raise ValueError(f"unknown event fact: {update['fact']}")
            cursor = cursor[key]
        if path[-1] not in cursor:
            raise ValueError(f"unknown event fact: {update['fact']}")
        if cursor[path[-1]] != update["value"]:
            cursor[path[-1]] = deepcopy(update["value"])
            changed.append(update["fact"])
    return Problem(data), changed
