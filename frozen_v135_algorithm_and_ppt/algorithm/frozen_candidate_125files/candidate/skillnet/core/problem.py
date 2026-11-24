"""Input objects shared by domains, decomposers and the planner."""
from dataclasses import dataclass, field
from .logic import Fact, Condition


@dataclass
class Proposal:
    instruction: str = ""
    goal: list[str] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)
    source: str = "任务文件"


@dataclass
class Problem:
    name: str
    description: str
    objects: dict[str, tuple[str, ...]]
    static: frozenset[Fact]
    initial: frozenset[Fact]
    goal: tuple[Condition, ...]
    library: object
    hooks: object
    oracle: object
    proposal: Proposal | None = None
    instruction: str = ""
    explicit_goal: tuple[Condition, ...] = ()
