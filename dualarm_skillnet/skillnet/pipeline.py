"""Orchestrate proposal repair, search, qualification and motion feedback."""
from collections import Counter
from copy import copy, deepcopy
from dataclasses import dataclass, field
from time import monotonic

from skillnet.capability.oracle import Status
from skillnet.core.engine import Engine
from skillnet.core.logic import Fact
from skillnet.decompose.base import FixedDecomposer
from skillnet.motion.base import NullBackend
from skillnet.planning.completion import CompletionReport, complete
from skillnet.planning.relevance import relevant_actions
from skillnet.planning.search import search
from skillnet.planning.taskgraph import build_task_graph


@dataclass
class PipelineResult:
    problem: object
    proposal: object
    completion: object
    schedule: object
    graph: object
    motion: list
    feedback: list[str]
    status: str
    model_calls: list[dict] = field(default_factory=list)
    model_goal_comparison: dict | None = None
    model_error: str | None = None
    completion_elapsed: float = 0.0


def _motion_reach_facts(action, failed_arm=None):
    facts = tuple(fact for fact in action.capability
                  if fact.predicate == "reach" and len(fact.args) == 2)
    if not facts and action.name == "support_off":
        bindings = dict(zip(action.skill.params, action.args))
        facts = (Fact("reach", (bindings["arm"], bindings["slot"])),)
    specific = tuple(fact for fact in facts if fact.args[0] == failed_arm)
    return specific or facts


def run(problem, *, decomposer=None, backend=None, prune=True, max_rounds=None,
        max_model_rounds=2):
    backend = backend if backend is not None else NullBackend()
    if max_rounds is None:
        max_rounds = getattr(backend, "feedback_round_limit", 6)
    if max_rounds < 1 or max_model_rounds < 0:
        raise ValueError("搜索反馈轮数必须至少为 1，模型反馈轮数不能为负")
    decomposer = decomposer if decomposer is not None else FixedDecomposer()
    # Measured qualification keeps its existing public persistence contract;
    # motion-level avoidance generalizations belong to this run only.
    input_oracle = problem.oracle
    problem = copy(problem)
    problem.oracle = deepcopy(input_oracle)
    engine = Engine(problem)
    feedback = []
    model_error = None

    def request_proposal(messages=None):
        nonlocal model_error
        try:
            return (decomposer.decompose(problem) if messages is None
                    else decomposer.decompose(problem, messages))
        except (ValueError, OSError) as error:
            model_error = str(error)
            feedback.append(f"模型调用失败：{model_error}；退回不使用模型，直接规划任务目标。")
            return None

    proposal = request_proposal()
    last_proposal = proposal
    completion = None
    completion_elapsed = 0.0
    model_rounds = 0
    while proposal is not None:
        remaining = max(0.0, 20.0 - completion_elapsed)
        if remaining <= 0:
            completion = CompletionReport(abandoned=True,
                abandon_reason="补全过程达到 20 秒总上限", unresolved=list(map(str, engine.goal)))
        else:
            repair_began = monotonic()
            completion = complete(engine, proposal.steps, time_limit=remaining)
            spent = max(completion.elapsed, monotonic() - repair_began)
            completion_elapsed += spent
            if completion_elapsed >= 20.0 and not completion.abandoned:
                completion.abandoned = True
                completion.abandon_reason = "补全过程达到 20 秒总上限"
                completion.feedback_for_model.clear()
        if completion.abandoned:
            feedback.append(f"放弃修补：{completion.abandon_reason}；不使用 hint，转入独立搜索。")
            break
        if not completion.unresolved or model_rounds >= max_model_rounds:
            break
        messages = completion.feedback_for_model or ["仍未解决的目标：" + ", ".join(completion.unresolved)]
        model_rounds += 1
        feedback.append(f"向小模型反馈（第 {model_rounds} 次）：" + "；".join(messages))
        proposal = request_proposal(messages)
        if proposal is None:
            if model_error is not None:
                completion = None
            break
        last_proposal = proposal
    hint = ([action.key for action in completion.plan]
            if completion is not None and not completion.abandoned else None)
    allowed = relevant_actions(engine)[0] if prune else None
    cuts = set()
    uncertain_cuts = set()
    action_cut_counts = Counter()
    unknown_seen = set()
    motion = []
    schedule = None
    round_record = None

    def record_round():
        nonlocal round_record
        if round_record is not None:
            recorder = getattr(backend, "record_feedback_round", None)
            if callable(recorder):
                recorder(deepcopy(round_record))
            round_record = None

    def finish(status, graph=None):
        from skillnet.decompose.goals import compare_model_goals

        if round_record is not None:
            round_record["terminal_status"] = status
        record_round()
        schedule.unknown = frozenset(f for f in unknown_seen if engine.oracle.check(f).status == Status.UNKNOWN)
        calls = deepcopy(getattr(decomposer, "calls", []))
        return PipelineResult(problem, proposal, completion, schedule, graph, motion, feedback, status,
                              calls, compare_model_goals(problem, last_proposal), model_error,
                              completion_elapsed)

    for round_index in range(max_rounds):
        round_record = {"round": round_index+1, "round_limit": max_rounds,
                        "cuts_before": len(cuts), "motion": [], "qualifications": []}
        schedule = search(engine, hint=hint, allowed=allowed, cuts=cuts)
        round_record["search_status"] = schedule.status
        unknown_seen.update(schedule.unknown)
        if prune and schedule.status != "solved":
            feedback.append(f"相关动作剪枝搜索结论为 {schedule.status}，回退到完整搜索。")
            schedule = search(engine, hint=hint, cuts=cuts)
            round_record["full_search_status"] = schedule.status
            unknown_seen.update(schedule.unknown)
        updated = False
        for fact in sorted(unknown_seen):
            if engine.oracle.check(fact).status != Status.UNKNOWN:
                continue
            qualified = backend.qualify(fact)
            if qualified is None or Status(qualified) == Status.UNKNOWN:
                continue
            engine.oracle.update(fact, qualified, "运动后端按需验证")
            input_oracle.update(fact, qualified, "运动后端按需验证")
            round_record["qualifications"].append({"fact": str(fact), "status": Status(qualified).value})
            feedback.append(f"运动后端按需验证：{fact} → {Status(qualified).value}。")
            updated = True
        if updated:
            if round_index + 1 == max_rounds:
                feedback.append("已达到搜索反馈轮数上限，能力更新后尚未重新搜索。")
                return finish("budget")
            record_round()
            continue
        if schedule.status != "solved":
            if schedule.status == "no_plan" and uncertain_cuts:
                feedback.append("规避运动未知动作后的受限搜索已穷尽；仍有未证伪的路线，整体结论保留 unknown。")
                round_record["conditional_no_plan"] = "unknown_motion_cuts"
                return finish("unknown")
            return finish(schedule.status)

        motion = []
        failed = False
        for step in sorted(schedule.steps, key=lambda item: item.start):
            validation = backend.validate(step.action, step.state)
            info = deepcopy(getattr(backend, "validation_context", {}))
            attempt = {"action": step.action.text, "state": sorted(map(str, step.state)),
                       "initial_status": validation.status.value, "initial_detail": validation.detail,
                       "initial_context": info}
            round_record["motion"].append(attempt)
            if getattr(backend, "fatal_error", None):
                motion.append(validation)
                feedback.append("运动后端远程接口失联；停止升级和后续重搜："+str(backend.fatal_error))
                return finish("unknown")
            if validation.status == Status.UNKNOWN:
                escalation = getattr(backend, "validate_escalated", None)
                if callable(escalation):
                    validation = escalation(step.action, step.state)
                    info = deepcopy(getattr(backend, "validation_context", info))
                    attempt.update(escalated=True, escalation_context=info)
                else:
                    attempt.update(escalated=False, escalation_unavailable=True)
                if getattr(backend, "fatal_error", None):
                    motion.append(validation)
                    feedback.append("运动后端远程接口失联；停止升级和后续重搜："+str(backend.fatal_error))
                    return finish("unknown")
            attempt.update(status=validation.status.value, detail=validation.detail)
            motion.append(validation)
            if validation.status in {Status.INVALID, Status.UNKNOWN}:
                cut = (step.state, step.action.key)
                if cut not in cuts:
                    cuts.add(cut)
                    action_cut_counts[step.action.key] += 1
                if validation.status == Status.UNKNOWN:
                    uncertain_cuts.add(cut)
                    feedback.append(f"运动验证未知：{step.action}，开始时间 {step.start:.3f} 秒；"
                                    f"原因：{validation.detail}；已规避该状态下的此动作并重新搜索。")
                else:
                    feedback.append(f"运动失败：{step.action}，开始时间 {step.start:.3f} 秒；"
                                    f"原因：{validation.detail}。已禁止该状态下的此动作并重新搜索。")
                count = action_cut_counts[step.action.key]
                attempt.update(cut=True, action_cut_count=count, downgraded_capabilities=[])
                if count >= 3:
                    for fact in _motion_reach_facts(step.action, info.get("failed_arm")):
                        if engine.oracle.check(fact).source.startswith("运动级三次规避"):
                            continue
                        engine.oracle.update(fact, Status.INVALID, "运动级三次规避："+step.action.text)
                        attempt["downgraded_capabilities"].append(str(fact))
                        feedback.append(f"运动级三次规避：{step.action} 已累计 {count} 次；"
                                        f"本次运行 {fact} → INVALID。")
                feedback.append(f"搜索反馈第 {round_index+1}/{max_rounds} 轮："
                                f"{step.action}，分类 {info.get('classification', 'backend_'+validation.status.value.lower())}，"
                                f"累计规避 {count} 次。")
                failed = True
                break
        if not failed:
            return finish("solved", build_task_graph(engine, schedule.steps))
        if round_index + 1 == max_rounds:
            feedback.append("已达到搜索反馈轮数上限，当前时间表仍未通过全部运动验证。")
            return finish("budget")
        record_round()
    raise RuntimeError("搜索反馈循环未返回结论")
