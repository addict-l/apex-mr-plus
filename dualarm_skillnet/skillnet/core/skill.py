"""Parameterized skills and grounded actions."""

from dataclasses import dataclass

from .logic import Condition, Fact

ActionKey = tuple[str, tuple[str, ...]]


@dataclass(frozen=True)
class Skill:
    name: str
    title: str
    stage: str
    description: str
    params: dict[str, str]
    pre: tuple[Condition, ...]
    capability: tuple[Fact, ...]
    add: tuple[Fact, ...]
    delete: tuple[Fact, ...]
    occupies: tuple[str, ...]
    move_to: dict[str, str]
    op_time: float
    structural: bool
    primitives: tuple[Fact, ...]

    @property
    def signature(self) -> str:
        return f"{self.name}({', '.join(self.params)})"

    def ground(self, args: tuple[str, ...]) -> "Action":
        args = tuple(args)
        if len(args) != len(self.params):
            raise ValueError(f"技能 {self.name} 参数个数错误，需要 {len(self.params)} 个")
        bindings = {f"?{name}": value for name, value in zip(self.params, args)}
        return Action(
            self, args,
            tuple(item.substitute(bindings) for item in self.pre),
            tuple(item.substitute(bindings) for item in self.capability),
            tuple(item.substitute(bindings) for item in self.add),
            tuple(item.substitute(bindings) for item in self.delete),
            tuple(bindings[item] for item in self.occupies),
            {bindings[arm]: bindings[place] for arm, place in self.move_to.items()},
            tuple(item.substitute(bindings) for item in self.primitives),
        )


@dataclass(frozen=True, eq=False)
class Action:
    skill: Skill
    args: tuple[str, ...]
    pre: tuple[Condition, ...]
    capability: tuple[Fact, ...]
    add: tuple[Fact, ...]
    delete: tuple[Fact, ...]
    occupies: tuple[str, ...]
    move_to: dict[str, str]
    primitives: tuple[Fact, ...]

    @property
    def name(self) -> str:
        return self.skill.name

    @property
    def key(self) -> ActionKey:
        return self.name, self.args

    @property
    def text(self) -> str:
        return str(Fact(self.name, self.args))

    @property
    def structural(self) -> bool:
        return self.skill.structural

    @property
    def op_time(self) -> float:
        return self.skill.op_time

    @property
    def title(self) -> str:
        return self.skill.title

    def __str__(self) -> str:
        return self.text

    def __hash__(self) -> int:
        return hash(self.key)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Action) and self.key == other.key
