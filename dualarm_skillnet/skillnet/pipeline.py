"""Orchestrate proposal repair, search, qualification and motion feedback."""
from collections import Counter
from copy import copy, deepcopy
from dataclasses import dataclass, field
import hashlib
import math
from time import monotonic

from skillnet.capability.oracle import Status
from skillnet.core.engine import Engine
from skillnet.core.logic import Fact
from skillnet.decompose.base import FixedDecomposer
from skillnet.motion.base import NullBackend
from skillnet.planning.completion import CompletionReport, complete
from skillnet.planning.relevance import relevant_actions
from skillnet.planning.search import SearchResult, search
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
        max_model_rounds=2, deadline=None, post_first_solution_limit: float | None = None):
    early_stop_enabled = post_first_solution_limit is not None
    search_policy = {"post_first_solution_limit": post_first_solution_limit} if early_stop_enabled else {}
    backend = backend if backend is not None else NullBackend()
    if max_rounds is None:
        max_rounds = getattr(backend, "feedback_round_limit", 6)
    if max_rounds < 1 or max_model_rounds < 0:
        raise ValueError("搜索反馈轮数必须至少为 1，模型反馈轮数不能为负")
    inherited = getattr(getattr(backend, "compiler", None), "deadline", None)
    if inherited is not None:
        deadline = inherited if deadline is None else min(deadline, inherited)
    if deadline is not None and (isinstance(deadline, bool) or not math.isfinite(deadline)):
        raise ValueError("任务截止时间必须为有限单调时钟值")
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
        if deadline is not None:
            remaining = max(0., min(remaining, deadline-monotonic()))
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
    cut_records = {}
    reopened_versions = set()
    unknown_seen = set()
    motion = []
    schedule = None
    schedule_pruned = False
    round_record = None
    unrestricted_lower_bound = None
    oracle_epoch = 0

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
            if cuts or (early_stop_enabled and schedule_pruned):
                # A restricted graph certificate applies only to the current
                # action set and cuts. Carry forward an unrestricted lower
                # bound only while the same capability model is in force.
                raw = {"lower_bound": schedule.lower_bound,
                       "suboptimality_bound": schedule.suboptimality_bound}
                lower = unrestricted_lower_bound
                if (lower is not None and status == "solved"
                        and lower > schedule.makespan+1e-8):
                    lower = None
                schedule.lower_bound = lower
                schedule.suboptimality_bound = (
                    max(1., schedule.makespan/lower)
                    if status == "solved" and lower is not None and lower > 0 else None)
                round_record["certificate_scope"] = {
                    "scope": "original symbolic cost model, not executed trajectory duration",
                    "oracle_epoch": oracle_epoch, "cuts": len(cuts),
                    "temporary_cuts": len(uncertain_cuts), "restricted_search": raw,
                    "unrestricted_lower_bound": lower,
                    "suboptimality_bound": schedule.suboptimality_bound}
                if early_stop_enabled:
                    round_record["certificate_scope"]["pruned"] = schedule_pruned
                    feedback.append("当前动作范围的证书单独保留；最终界仅使用同一能力模型下未剪枝、未规避的符号搜索下界。")
                else:
                    feedback.append("运动规避后的受限图证书单独保留；最终界仅使用同一能力模型下未加规避的符号搜索下界。")
        record_round()
        # This is the final motion-aware status. Raw symbolic conclusions are
        # retained in feedback-round evidence, never serialized as a conflicting
        # final search status (e.g. conditional no_plan versus UNKNOWN).
        schedule.status = status
        schedule.unknown = frozenset(f for f in unknown_seen if engine.oracle.check(f).status == Status.UNKNOWN)
        calls = deepcopy(getattr(decomposer, "calls", []))
        return PipelineResult(problem, proposal, completion, schedule, graph, motion, feedback, status,
                              calls, compare_model_goals(problem, last_proposal), model_error,
                              completion_elapsed)

    def bounded_search(**kwargs):
        nonlocal unrestricted_lower_bound, schedule_pruned
        schedule_pruned = kwargs.get("allowed") is not None
        callback = getattr(backend, "progress_callback", None)
        search_began = monotonic()
        if callable(callback):
            callback({"query": "symbolic_search", "phase": "before",
                      "round": round_index+1, "cuts": len(cuts),
                      "temporary_cuts": len(uncertain_cuts), "pruned": schedule_pruned})
        remaining = 60. if deadline is None else min(60., deadline-monotonic())
        if remaining <= 0:
            result = SearchResult("budget", [], 0., 0, 0, {}, frozenset(), 0.,
                early_stop_enabled=early_stop_enabled, post_first_solution_limit=post_first_solution_limit)
        else:
            result = search(engine, hint=hint, cuts=cuts, time_limit=remaining, **search_policy, **kwargs)
        if callable(callback):
            callback({"query": "symbolic_search", "phase": "after", "round": round_index+1,
                      "cuts": len(cuts), "temporary_cuts": len(uncertain_cuts),
                      "status": result.status, "expanded": result.expanded,
                      "generated": result.generated, "elapsed_seconds": monotonic()-search_began,
                      "lower_bound": result.lower_bound, "suboptimality_bound": result.suboptimality_bound,
                      "pruned": schedule_pruned, "oracle_epoch": oracle_epoch,
                      "certificate_scope": "restricted" if cuts or schedule_pruned else "unrestricted",
                      "early_stop_enabled": result.early_stop_enabled,
                      "early_stop_triggered": result.early_stop_triggered,
                      "early_stop_reason": result.early_stop_reason,
                      "first_solution_seconds": result.first_solution_seconds,
                      "post_first_solution_limit": result.post_first_solution_limit})
        lower = result.lower_bound
        if (not cuts and not schedule_pruned
                and isinstance(lower, (int, float)) and not isinstance(lower, bool)
                and math.isfinite(lower) and lower >= 0):
            unrestricted_lower_bound = (lower if unrestricted_lower_bound is None
                                        else max(lower, unrestricted_lower_bound))
        return result

    def reopen_candidates():
        """Only an evidenced, previously unseen candidate version reopens a cut."""
        reconsider = getattr(backend, "reconsider_cuts", None)
        if not callable(reconsider) or not cut_records:
            return False
        changed = False
        for update in reconsider(deepcopy(list(cut_records.values()))) or ():
            ident, version = update.get("cut_id"), update.get("new_candidate_version")
            old = next((row for row in cut_records.values() if row["cut_id"] == ident), None)
            if (old is None or not old["temporary"] or not version
                    or version == old.get("candidate_version")
                    or (ident, version) in reopened_versions or not update.get("reason")):
                continue
            cut = next(key for key, row in cut_records.items() if row is old)
            if cut not in cuts:
                continue
            cuts.remove(cut)
            uncertain_cuts.discard(cut)
            reopened_versions.add((ident, version))
            old.update(candidate_version=version, reopened_reason=update["reason"])
            round_record.setdefault("reopened_cuts", []).append(deepcopy(update))
            feedback.append("新的相关运动候选允许重新考虑临时规避："+str(update["reason"]))
            changed = True
        return changed

    def avoid_failure(step, validation, info, attempt):
        """One scoped cut path for action and final-reference failures."""
        cut = (step.state, step.action.key)
        repeated_cut = cut in cuts
        if not repeated_cut:
            cuts.add(cut)
            action_cut_counts[step.action.key] += 1
        # A finite geometric candidate is a local avoidance, even if
        # its particular endpoint is proven colliding. It is never a
        # proof that the arm cannot reach the symbolic location.
        temporary = info.get("failure_scope") != "global"
        if temporary or validation.status == Status.UNKNOWN:
            uncertain_cuts.add(cut)
        ident = hashlib.sha256(repr((tuple(sorted(map(str, step.state))), step.action.key)).encode()).hexdigest()
        cut_record = {"cut_id": ident, "action_key": step.action.key,
            "action": step.action.text, "state": sorted(map(str, step.state)),
            "context_id": info.get("context_id"), "candidate_id": info.get("candidate_id"),
            "candidate_version": info.get("candidate_version"), "temporary": temporary,
            "source": info.get("classification", "backend_"+validation.status.value.lower()),
            "scope": info.get("failure_scope", "state/action candidate"),
            "round": round_index+1}
        cut_records[cut] = cut_record
        attempt["cut_evidence"] = deepcopy(cut_record)
        if validation.status == Status.UNKNOWN:
            feedback.append(f"运动验证未知：{step.action}，开始时间 {step.start:.3f} 秒；"
                            f"原因：{validation.detail}；已规避该状态下的此动作并重新搜索。")
        else:
            feedback.append(f"运动失败：{step.action}，开始时间 {step.start:.3f} 秒；"
                            f"原因：{validation.detail}。已临时规避该状态下的此动作并重新搜索。")
        count = action_cut_counts[step.action.key]
        attempt.update(cut=True, action_cut_count=count, downgraded_capabilities=[])
        feedback.append(f"搜索反馈第 {round_index+1}/{max_rounds} 轮："
                        f"{step.action}，分类 {info.get('classification', 'backend_'+validation.status.value.lower())}，"
                        f"累计规避 {count} 次。")
        if repeated_cut and info.get("deduplicated") and info.get("retryable") is False:
            feedback.append("相同失败候选与临时规避均无变化，无法取得新进展；保存证据并结束为 unknown。")
            round_record["no_progress"] = deepcopy(cut_record)
            return True
        return False

    for round_index in range(max_rounds):
        round_record = {"round": round_index+1, "round_limit": max_rounds,
                        "cuts_before": len(cuts), "motion": [], "qualifications": []}
        schedule = bounded_search(allowed=allowed)
        round_record["search_status"] = schedule.status
        unknown_seen.update(schedule.unknown)
        if prune and schedule.status != "solved":
            feedback.append(f"相关动作剪枝搜索结论为 {schedule.status}，回退到完整搜索。")
            schedule = bounded_search()
            round_record["full_search_status"] = schedule.status
            unknown_seen.update(schedule.unknown)
        updated = False
        for fact in sorted(unknown_seen):
            if engine.oracle.check(fact).status != Status.UNKNOWN:
                continue
            if deadline is not None and monotonic() >= deadline:
                feedback.append("任务共享墙钟预算已耗尽，停止资格查询与反馈搜索。")
                return finish("budget")
            qualified = backend.qualify(fact)
            if qualified is None or Status(qualified) == Status.UNKNOWN:
                continue
            engine.oracle.update(fact, qualified, "运动后端按需验证")
            input_oracle.update(fact, qualified, "运动后端按需验证")
            round_record["qualifications"].append({"fact": str(fact), "status": Status(qualified).value})
            feedback.append(f"运动后端按需验证：{fact} → {Status(qualified).value}。")
            updated = True
        if updated:
            # New capabilities can introduce cheaper routes. Do not carry a
            # bound from the preceding capability model across that change.
            oracle_epoch += 1
            unrestricted_lower_bound = None
            if round_index + 1 == max_rounds:
                feedback.append("已达到搜索反馈轮数上限，能力更新后尚未重新搜索。")
                return finish("budget")
            record_round()
            continue
        if schedule.status != "solved":
            if schedule.status in {"no_plan", "unknown"} and reopen_candidates():
                if round_index+1 == max_rounds:
                    return finish("budget")
                record_round()
                continue
            if schedule.status == "no_plan" and uncertain_cuts:
                feedback.append("临时规避运动候选后的受限搜索已穷尽；仍有未证伪的路线，整体结论保留 unknown。")
                round_record["conditional_no_plan"] = "unknown_motion_cuts"
                return finish("unknown")
            return finish(schedule.status)

        motion = []
        failed = False
        begin_reference = getattr(backend, "begin_reference", None)
        if callable(begin_reference):
            begin_reference(round_index+1, deadline=deadline)
        for step in sorted(schedule.steps, key=lambda item: item.start):
            if deadline is not None and monotonic() >= deadline:
                feedback.append("任务共享墙钟预算已耗尽，停止动作验证与重搜。")
                return finish("budget")
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
                if avoid_failure(step, validation, info, attempt):
                    return finish("unknown")
                failed = True
                break
        if not failed:
            graph = build_task_graph(engine, schedule.steps)
            finalize = getattr(backend, "finalize_reference", None)
            if not callable(finalize):
                return finish("solved", graph)
            if deadline is not None and monotonic() >= deadline:
                feedback.append("完整参考轨迹验证前任务共享墙钟预算已耗尽。")
                return finish("budget")
            validation = finalize(schedule, graph)
            info = deepcopy(getattr(backend, "validation_context", {}))
            attempt = {"phase": "final_reference", "status": validation.status.value,
                       "detail": validation.detail, "context": info}
            round_record["final_reference"] = attempt
            motion.append(validation)
            if validation.status == Status.VALID:
                return finish("solved", graph)
            segment = info.get("failed_segment", "final_reference")
            feedback.append(f"完整参考轨迹验证未通过：{segment}；{validation.detail}")
            if getattr(backend, "fatal_error", None):
                feedback.append("运动后端远程接口失联，停止完整参考轨迹反馈。")
                return finish("unknown")
            if deadline is not None and monotonic() >= deadline:
                return finish("budget")
            source = info.get("source_step")
            ordered = sorted(schedule.steps, key=lambda item: item.start)
            if (isinstance(source, int) and not isinstance(source, bool)
                    and 1 <= source <= len(ordered)
                    and info.get("failure_scope") != "terminal"):
                if avoid_failure(ordered[source-1], validation, info, attempt):
                    return finish("unknown")
                failed = True
            else:
                # A blocked fixed final home constrains the entire reference.
                # Inventing a reach fact or a source action would overstate its
                # evidence. Retain the full terminal failure for later work.
                round_record["terminal_reference_failure"] = deepcopy(info)
                return finish("unknown")
        if round_index + 1 == max_rounds:
            feedback.append("已达到搜索反馈轮数上限，当前时间表仍未通过全部运动验证。")
            return finish("budget")
        record_round()
    raise RuntimeError("搜索反馈循环未返回结论")
