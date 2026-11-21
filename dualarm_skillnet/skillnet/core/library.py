"""Validated TOML skill libraries and automatically derived networks."""

from dataclasses import dataclass
import math
import tomllib

from .logic import is_identifier, parse_condition
from .skill import Skill


@dataclass(frozen=True, order=True)
class NetworkEdge:
    source: str
    target: str
    label: str


@dataclass
class Library:
    skills: tuple[Skill, ...]
    version: str = ""
    derived_basis: dict[str, frozenset[str]] | None = None

    @property
    def by_name(self) -> dict[str, Skill]:
        return {skill.name: skill for skill in self.skills}

    @property
    def dynamic_predicates(self) -> frozenset[str]:
        return frozenset(
            {"arm_at"} | {fact.predicate for skill in self.skills for fact in skill.add + skill.delete}
        )

    def network(self, derived_basis=None) -> tuple[NetworkEdge, ...]:
        basis = derived_basis if derived_basis is not None else (self.derived_basis or {})
        edges = set()
        for source in self.skills:
            provided = {fact.predicate for fact in source.add}
            for target in self.skills:
                for condition in target.pre:
                    if condition.negated:
                        continue
                    predicate = condition.fact.predicate
                    if predicate in provided:
                        edges.add(NetworkEdge(source.name, target.name, predicate))
                    for dependency in sorted(basis.get(predicate, frozenset()) & provided):
                        label = f"{predicate}（由 {dependency} 推出）"
                        edges.add(NetworkEdge(source.name, target.name, label))
        return tuple(sorted(edges))

    def to_mermaid(self, derived_basis=None) -> str:
        lines = ["flowchart TD"]
        for skill in self.skills:
            title = skill.title.replace('"', "&quot;")
            lines.append(f'  {skill.name}["{title} {skill.signature}"]')
        for edge in self.network(derived_basis):
            lines.append(f'  {edge.source} -->|"{edge.label}"| {edge.target}')
        return "\n".join(lines) + "\n"


def _load_skill(data: dict) -> Skill:
    name = data.get("name", "<未命名>")

    def fail(entry: str, detail: str):
        raise ValueError(f"技能 {name} 的 {entry} 错误：{detail}")

    if not is_identifier(name):
        fail("name", "名称必须是标识符")
    params = data.get("params")
    if not isinstance(params, dict) or not params:
        fail("params", "必须是非空参数表")
    for param, kind in params.items():
        if not is_identifier(param) or kind not in {"arm", "block", "loc", "slot"}:
            fail(f"params.{param}", f"参数名或类型不合法：{kind!r}")

    def conditions(field, allow_not=False):
        raw = data.get(field, [])
        if not isinstance(raw, list):
            fail(field, "必须是字符串列表")
        result = []
        for index, entry in enumerate(raw, 1):
            label = f"{field}[{index}] {entry!r}"
            try:
                condition = parse_condition(entry)
            except ValueError as error:
                fail(label, str(error))
            if condition.negated and not allow_not:
                fail(label, "not 只能出现在 pre")
            for argument in condition.fact.args:
                if argument.startswith("?") and argument[1:] not in params:
                    fail(label, f"未声明变量 {argument}")
            result.append(condition if allow_not else condition.fact)
        return tuple(result)

    pre = conditions("pre", True)
    capability = conditions("capability")
    add = conditions("add")
    delete = conditions("delete")
    primitives = conditions("primitives")
    occupies = data.get("occupies")
    if not isinstance(occupies, list) or not occupies:
        fail("occupies", "至少需要一个 arm 类型参数")

    def parameter_kind(value):
        if not isinstance(value, str) or not value.startswith("?"):
            return None
        return params.get(value[1:])

    for arm in occupies:
        if parameter_kind(arm) != "arm":
            fail(f"occupies {arm!r}", "必须引用 arm 类型参数")
    if len(set(occupies)) != len(occupies):
        fail("occupies", "参与臂参数重复")
    move_to = data.get("move_to", {})
    if not isinstance(move_to, dict):
        fail("move_to", "必须是参数映射表")
    for arm, place in move_to.items():
        if parameter_kind(arm) != "arm":
            fail(f"move_to {arm!r}", "键必须引用 arm 类型参数")
        if arm not in occupies:
            fail(f"move_to {arm!r}", "移动的臂必须在 occupies 中")
        if parameter_kind(place) not in {"loc", "slot"}:
            fail(f"move_to {arm!r}={place!r}", "目标必须引用 loc 或 slot 类型参数")
    op_time = data.get("op_time")
    if isinstance(op_time, bool) or not isinstance(op_time, (int, float)) or not math.isfinite(op_time) or op_time <= 0:
        fail("op_time", "必须是大于 0 的有限数值")
    structural = data.get("structural", False)
    if not isinstance(structural, bool):
        fail("structural", "必须是布尔值")
    for field in ("title", "description", "stage"):
        if field in data and not isinstance(data[field], str):
            fail(field, "必须是字符串")
    return Skill(
        name, data.get("title", name), data.get("stage", ""), data.get("description", ""),
        dict(params), pre, capability, add, delete, tuple(occupies), dict(move_to),
        float(op_time), structural, primitives,
    )


def library_from_data(data: dict, derived_basis=None) -> Library:
    raw_skills = data.get("skill")
    if not isinstance(raw_skills, list) or not raw_skills:
        raise ValueError("技能库必须包含非空 [[skill]] 列表")
    skills = []
    seen = set()
    for entry in raw_skills:
        if not isinstance(entry, dict):
            raise ValueError("技能定义必须是表")
        skill = _load_skill(entry)
        if skill.name in seen:
            raise ValueError(f"技能 {skill.name} 的 name 错误：名称重复")
        seen.add(skill.name)
        skills.append(skill)
    library = Library(tuple(skills), data.get("version", ""), derived_basis)
    overlap = library.dynamic_predicates & set(derived_basis or {})
    if overlap:
        raise ValueError(f"谓词不能同时是动态条件与推导条件：{', '.join(sorted(overlap))}")
    return library


def load_library(path="skills/blocks_v1.toml", derived_basis=None) -> Library:
    try:
        with open(path, "rb") as source:
            data = tomllib.load(source)
    except tomllib.TOMLDecodeError as error:
        raise ValueError(f"技能库 TOML 格式错误：{error}") from error
    return library_from_data(data, derived_basis)
