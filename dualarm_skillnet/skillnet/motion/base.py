"""Motion backend contract and deterministic offline test backends."""

from dataclasses import dataclass, field
from typing import Protocol

from skillnet.capability.oracle import Status
from skillnet.core.logic import Fact, parse_fact
from skillnet.core.skill import Action


@dataclass
class MotionResult:
    status: Status
    detail: str
    segments: list[str] = field(default_factory=list)


class MotionBackend(Protocol):
    def validate(self, action: Action, state: frozenset[Fact]) -> MotionResult: ...
    def qualify(self, fact: Fact) -> Status | None: ...


class NullBackend:
    def validate(self, action: Action, state: frozenset[Fact]) -> MotionResult:
        return MotionResult(Status.VALID, "占位后端：未做几何检查",
                            [str(primitive) for primitive in action.primitives])

    def qualify(self, fact: Fact) -> Status | None:
        return None


class ScriptedBackend(NullBackend):
    def __init__(self, fail_once=None, qualify=None):
        self._pending_failures = set()
        for text in fail_once or ():
            fact = parse_fact(text)
            self._pending_failures.add((fact.predicate, fact.args))
        self._qualifications = {}
        for text, value in (qualify or {}).items():
            fact = parse_fact(text)
            try:
                status = Status(value)
            except (ValueError, TypeError) as error:
                raise ValueError(f"运动后端能力 {fact} 的结论必须是 VALID、INVALID 或 UNKNOWN") from error
            self._qualifications[fact] = status

    def validate(self, action: Action, state: frozenset[Fact]) -> MotionResult:
        if action.key in self._pending_failures:
            self._pending_failures.remove(action.key)
            return MotionResult(Status.INVALID, f"脚本后端：动作 {action} 首次验证失败", [])
        return super().validate(action, state)

    def qualify(self, fact: Fact) -> Status | None:
        return self._qualifications.get(fact)
