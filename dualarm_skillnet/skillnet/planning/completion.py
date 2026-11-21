"""Validate a proposed sequence and repair missing preconditions and goals."""

from dataclasses import dataclass, field
from math import isfinite
from time import monotonic

from skillnet.capability.oracle import Status
from skillnet.core.logic import Fact
from skillnet.core.skill import Action
from .search import search


@dataclass(frozen=True)
class CompletionEntry:
    kind: str
    action_text: str
    reason: str
    proposal_index: int | None = None

    @property
    def action(self):
        return self.action_text


@dataclass
class CompletionReport:
    entries: list[CompletionEntry] = field(default_factory=list)
    plan: list[Action] = field(default_factory=list)
    goal_reached: bool = False
    unresolved: list[str] = field(default_factory=list)
    feedback_for_model: list[str] = field(default_factory=list)
    abandoned: bool = False
    abandon_reason: str | None = None
    elapsed: float = 0.0
    processed_steps: int = 0
    deleted_steps: int = 0


class _AbandonRepair(Exception):
    pass


def _replay(engine, actions, state, check=lambda: None):
    """Check a whole repair before exposing any part as accepted."""
    for action in actions:
        check()
        failed = engine.failed(action, state)
        if failed:
            raise ValueError(f"顺序重放 {action} 时前置条件不成立：{', '.join(map(str, failed))}")
        status, evidence = engine.oracle.evaluate(action.capability)
        if status != Status.VALID:
            failures = ", ".join(f"{item.fact} = {item.status.value}"
                                 for item in evidence if item.status != Status.VALID)
            raise ValueError(f"顺序重放 {action} 时能力检查失败：{failures}")
        state = engine.execute(action, state)
    return state


def complete(engine, steps: list[str], goal=None, *, time_limit=20.0) -> CompletionReport:
    began = monotonic()
    if isinstance(time_limit, bool) or not isinstance(time_limit, (int, float)) or not isfinite(time_limit) or time_limit < 0:
        raise ValueError("补全时间预算必须是非负有限秒数")
    budget = min(20.0, time_limit)
    deadline = began + budget
    report = CompletionReport()
    targets = engine.goal if goal is None else tuple(goal)
    state = engine.initial
    deleted = set()
    added = set()
    block_names = frozenset(engine.problem.objects["block"])

    def abandon(reason):
        report.abandoned = True
        report.abandon_reason = reason
        raise _AbandonRepair

    def check_time():
        if monotonic() >= deadline:
            abandon(f"补全过程达到时间上限：总预算最多 20 秒，本轮可用 {budget:.3f} 秒")

    def record(kind, text, reason, index=None):
        report.entries.append(CompletionEntry(kind, text, reason, index))
        if index is not None:
            report.deleted_steps += int(kind == "删除")
            if report.processed_steps >= 4 and report.deleted_steps * 2 > report.processed_steps:
                abandon(f"方案质量过低：已处理 {report.processed_steps} 步，删除 {report.deleted_steps} 步，超过一半")

    def accept(action, kind, reason, index=None):
        nonlocal state, deleted, added
        check_time()
        state = engine.execute(action, state)
        deleted = (deleted | set(action.delete)) - set(action.add)
        added = (added - set(action.delete)) | set(action.add)
        report.plan.append(action)
        record(kind, str(action), reason, index)

    def repair(conditions, reason, protected=()):
        check_time()
        sub_limit = min(5.0, max(0.0, deadline - monotonic()))
        result = search(engine, start=state, goal=conditions, protected=protected,
                        max_expansions=50_000, time_limit=sub_limit)
        check_time()
        if result.status == "budget":
            abandon(f"单次补全子搜索达到上限：最多 50000 个状态或 {sub_limit:.3f} 秒；"
                    f"已扩展 {result.expanded} 个状态，用时 {result.elapsed:.3f} 秒")
        if result.status != "solved":
            unknown = ", ".join(map(str, sorted(result.unknown)))
            detail = f"补全搜索结论 {result.status}"
            if unknown:
                detail += f"，未验证能力：{unknown}"
            return detail
        actions = [step.action for step in sorted(result.steps, key=lambda item: item.start)]
        try:
            final = _replay(engine, actions, state, check_time)
            if not engine.satisfied(conditions, final):
                return "补全子计划顺序重放后仍未满足所需条件"
        except ValueError as error:
            return str(error)
        for action in actions:
            accept(action, "补出", reason)
        return None

    def process_steps():
        nonlocal state
        for index, text in enumerate(steps, 1):
            check_time()
            report.processed_steps += 1
            try:
                action = engine.parse_action(text)
            except (ValueError, TypeError) as error:
                record("删除", str(text), str(error), index)
                continue
            status, evidence = engine.oracle.evaluate(action.capability)
            if status != Status.VALID:
                failures = "；".join(f"{item.fact} = {item.status.value}（{item.source}）"
                                    for item in evidence if item.status != Status.VALID)
                record("删除", text, f"能力检查不通过：{failures}", index)
                continue
            failed = engine.failed(action, state)
            obsolete = []
            for condition in failed:
                if condition.negated:
                    if condition.fact in added:
                        obsolete.append(f"{condition.fact} 已由前面的步骤添加")
                elif condition.fact in deleted and block_names.intersection(condition.fact.args):
                    obsolete.append(f"{condition.fact} 已由前面的步骤删除且未再添加")
            if obsolete:
                record("多余", text, "；".join(obsolete), index)
                continue
            if failed:
                protected = [condition.fact for condition in action.pre
                             if not condition.negated
                             and condition.fact.predicate in engine.dynamic_predicates
                             and engine.holds(condition, state)]
                reason = f"为满足 {action} 的前置条件：{', '.join(map(str, failed))}"
                error = repair(failed, reason, protected)
                if error:
                    record("删除", text, f"{reason}；{error}", index)
                    continue
            failed = engine.failed(action, state)
            if failed:
                record("删除", text, f"补全后前置条件仍不满足：{', '.join(map(str, failed))}", index)
                continue
            try:
                engine.execute(action, state)
            except ValueError as error:
                record("删除", text, f"稳定规则不通过：{error}", index)
                continue
            accept(action, "保留", "前置条件、能力检查及稳定规则均通过", index)

        check_time()
        if not engine.satisfied(targets, state):
            missing = [condition for condition in targets if not engine.holds(condition, state)]
            reason = "补齐任务目标：" + ", ".join(map(str, missing))
            error = repair(targets, reason)
            if error:
                report.feedback_for_model.append(f"{reason}；{error}")

    try:
        process_steps()
        check_time()
    except _AbandonRepair:
        pass
    report.unresolved = [str(condition) for condition in targets if not engine.holds(condition, state)]
    report.goal_reached = not report.unresolved
    if report.unresolved and not report.abandoned:
        for entry in report.entries:
            if entry.kind == "删除":
                prefix = f"第 {entry.proposal_index} 步 " if entry.proposal_index is not None else ""
                report.feedback_for_model.append(f"{prefix}{entry.action_text}：{entry.reason}")
        report.feedback_for_model.append("仍未解决的目标：" + ", ".join(report.unresolved))
    else:
        report.feedback_for_model.clear()
    report.elapsed = monotonic() - began
    return report
