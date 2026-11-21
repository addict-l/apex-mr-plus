"""Immutable symbolic facts and conditions."""

from dataclasses import dataclass


def is_identifier(value: str) -> bool:
    return isinstance(value, str) and value.isidentifier()


@dataclass(frozen=True, order=True)
class Fact:
    predicate: str
    args: tuple[str, ...] = ()

    def __str__(self) -> str:
        return f"{self.predicate}({', '.join(self.args)})"

    def substitute(self, bindings: dict[str, str]) -> "Fact":
        return Fact(self.predicate, tuple(bindings.get(arg, arg) for arg in self.args))


@dataclass(frozen=True)
class Condition:
    fact: Fact
    negated: bool = False

    def __str__(self) -> str:
        return ("not " if self.negated else "") + str(self.fact)

    def substitute(self, bindings: dict[str, str]) -> "Condition":
        return Condition(self.fact.substitute(bindings), self.negated)


def parse_fact(text: str) -> Fact:
    if not isinstance(text, str):
        raise ValueError(f"事实必须是字符串，收到 {text!r}")
    value = text.strip()
    if "(" not in value or not value.endswith(")"):
        raise ValueError(f"事实格式错误：{text!r}，应为 predicate(arg, ...)")
    name, arguments = value[:-1].split("(", 1)
    name = name.strip()
    if not is_identifier(name):
        raise ValueError(f"事实谓词不是有效标识符：{text!r}")
    args = tuple(part.strip() for part in arguments.split(",")) if arguments.strip() else ()
    for arg in args:
        token = arg[1:] if arg.startswith("?") else arg
        if not is_identifier(token):
            raise ValueError(f"事实参数不是有效标识符：{text!r} 中的 {arg!r}")
    return Fact(name, args)


def parse_condition(text: str) -> Condition:
    if not isinstance(text, str):
        raise ValueError(f"条件必须是字符串，收到 {text!r}")
    value = text.strip()
    negative = value.startswith("not ") or value.startswith("not\t")
    return Condition(parse_fact(value[3:].strip() if negative else value), negative)
