"""Three-valued capability evidence, independent of planning state."""

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from skillnet.core.logic import Fact


class Status(str, Enum):
    VALID = "VALID"
    INVALID = "INVALID"
    UNKNOWN = "UNKNOWN"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class Evidence:
    fact: Fact
    status: Status
    source: str


def _status(value: str | Status, context: str) -> Status:
    try:
        return Status(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{context}：能力结论必须是 VALID、INVALID 或 UNKNOWN，收到 {value!r}") from exc


def _arguments(entries: object, context: str) -> Iterable[tuple[str, ...]]:
    if isinstance(entries, dict):
        for first, seconds in entries.items():
            if not isinstance(first, str) or not isinstance(seconds, list):
                raise ValueError(f"{context}：映射格式必须为字符串到参数列表")
            for second in seconds:
                if not isinstance(second, str) or not first or not second:
                    raise ValueError(f"{context}：能力参数必须是非空字符串")
                yield (first, second)
        return
    if isinstance(entries, list):
        for row in entries:
            if not isinstance(row, list) or not row or not all(isinstance(arg, str) and arg for arg in row):
                raise ValueError(f"{context}：列表格式必须是非空字符串参数列表的列表")
            yield tuple(row)
        return
    raise ValueError(f"{context}：能力条目必须使用参数映射或参数列表的列表")


class CapabilityOracle:
    """Use explicit evidence before a predicate's declared default."""

    def __init__(self, tables: dict | None = None):
        self.defaults: dict[str, Status] = {}
        self.entries: dict[Fact, Evidence] = {}
        if tables is None:
            tables = {}
        if not isinstance(tables, dict):
            raise ValueError("capabilities：能力表必须是字典")
        for predicate, table in tables.items():
            context = f"能力表 {predicate}"
            if not isinstance(predicate, str) or not predicate.isidentifier():
                raise ValueError(f"{context}：能力名称必须是标识符")
            if not isinstance(table, dict):
                raise ValueError(f"{context}：谓词定义必须是表")
            unknown_keys = set(table) - {"default", "valid", "invalid", "unknown"}
            if unknown_keys:
                raise ValueError(f"{context}：未知条目 {', '.join(sorted(unknown_keys))}")
            self.defaults[predicate] = _status(table.get("default", "UNKNOWN"), f"{context}.default")
            for status in Status:
                key = status.value.lower()
                for args in _arguments(table.get(key, []), f"{context}.{key}"):
                    fact = Fact(predicate, args)
                    previous = self.entries.get(fact)
                    if previous is not None and previous.status != status:
                        raise ValueError(f"{context}：{fact} 同时声明为 {previous.status.value} 和 {status.value}")
                    self.entries[fact] = Evidence(fact, status, f"任务能力表 {predicate}.{key}")

    def check(self, fact: Fact) -> Evidence:
        explicit = self.entries.get(fact)
        if explicit is not None:
            return explicit
        if fact.predicate in self.defaults:
            return Evidence(fact, self.defaults[fact.predicate], f"任务能力表 {fact.predicate}.default")
        return Evidence(fact, Status.UNKNOWN, "未声明的能力谓词")

    def update(self, fact: Fact, status: str | Status, source: str) -> None:
        self.entries[fact] = Evidence(fact, _status(status, str(fact)), source)

    def evaluate(self, facts: Iterable[Fact]) -> tuple[Status, tuple[Evidence, ...]]:
        evidence = tuple(self.check(fact) for fact in facts)
        statuses = {item.status for item in evidence}
        if Status.INVALID in statuses:
            return Status.INVALID, evidence
        if Status.UNKNOWN in statuses:
            return Status.UNKNOWN, evidence
        return Status.VALID, evidence
