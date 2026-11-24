"""Chinese reports and portable schedule artifacts."""

from dataclasses import asdict
from collections import defaultdict
import json
from pathlib import Path

from skillnet.core.engine import Engine
from skillnet.capability.oracle import Status
from skillnet.planning.relevance import relevant_actions
from skillnet.planning.taskgraph import TaskGraph


STATUS_TITLES = {
    "solved": "找到方案",
    "no_plan": "确定无解",
    "unknown": "能力尚未验证",
    "budget": "超出搜索预算",
}


def _relevance(result):
    engine = Engine(result.problem)
    _, reasons = relevant_actions(engine)
    entries = []
    for key, facts in sorted(reasons.items()):
        action = engine.actions[key]
        status, evidence = engine.oracle.evaluate(action.capability)
        entries.append({
            "action": str(action), "skill": action.name, "title": action.title,
            "status": status.value, "reasons": [str(fact) for fact in facts],
            "evidence": [{"fact": str(item.fact), "status": item.status.value,
                          "source": item.source} for item in evidence],
        })
    return entries


def plan_to_dict(result) -> dict:
    schedule = result.schedule
    completion = result.completion
    return {
        "task": result.problem.name,
        "description": result.problem.description,
        "instruction": getattr(result.problem, "instruction", ""),
        "status": result.status,
        "makespan": schedule.makespan,
        "suboptimality_bound": getattr(schedule, "suboptimality_bound", None),
        "proposal": asdict(result.proposal) if result.proposal is not None else None,
        "used_proposal": result.proposal is not None,
        "model_calls": list(getattr(result, "model_calls", [])),
        "model_goal_comparison": getattr(result, "model_goal_comparison", None),
        "model_error": getattr(result, "model_error", None),
        "completion_elapsed": getattr(result, "completion_elapsed", getattr(completion, "elapsed", 0.0)),
        "completion": None if completion is None else {
            "entries": [asdict(entry) for entry in completion.entries],
            "plan": [str(action) for action in completion.plan],
            "goal_reached": completion.goal_reached,
            "unresolved": completion.unresolved,
            "feedback_for_model": completion.feedback_for_model,
            "abandoned": bool(getattr(completion, "abandoned", False)),
            "abandon_reason": getattr(completion, "abandon_reason", None),
            "elapsed": getattr(completion, "elapsed", 0.0),
            "processed_steps": getattr(completion, "processed_steps", 0),
            "deleted_steps": getattr(completion, "deleted_steps", 0),
        },
        "schedule": [{
            "action": str(step.action), "skill": step.action.name,
            "title": step.action.skill.title, "arms": list(step.action.occupies),
            "start": step.start, "end": step.end,
            "state_before": [str(fact) for fact in sorted(step.state)],
        } for step in sorted(schedule.steps, key=lambda step: step.start)],
        "search": {
            "status": schedule.status, "expanded": schedule.expanded,
            "generated": schedule.generated, "rejected": schedule.rejected,
            "elapsed": schedule.elapsed,
            "suboptimality_bound": getattr(schedule, "suboptimality_bound", None),
            "lower_bound": getattr(schedule, "lower_bound", None),
            "weight": getattr(schedule, "weight", 1.),
            "stages": list(getattr(schedule, "stages", ())),
        },
        "unknown": [str(fact) for fact in sorted(schedule.unknown)],
        "motion": [{"status": str(item.status), "detail": item.detail,
                    "segments": [str(segment) for segment in item.segments]}
                   for item in result.motion],
        "feedback": list(result.feedback),
        "relevance": _relevance(result),
    }


def render_report(result) -> str:
    problem = result.problem
    schedule = result.schedule
    lines = [f"任务：{problem.name} —— {problem.description}", "", "① 初始状态："]
    lines.extend(f"- {fact}" for fact in sorted(problem.initial))
    lines.append("目标：" + "、".join(map(str, problem.goal)))
    lines.extend(["", "② 小模型方案（来源）："])
    if result.proposal is None:
        lines.append("未提供小模型方案。")
    else:
        lines.append(result.proposal.source)
        if result.proposal.instruction:
            lines.append("任务指令：" + result.proposal.instruction)
        lines.extend(f"{index}. {text}" for index, text in enumerate(result.proposal.steps, 1))
        if not result.proposal.steps:
            lines.append("方案步骤为空。")
    calls = getattr(result, "model_calls", [])
    if calls:
        lines.append(f"模型请求：{len(calls)} 次，合计耗时 {sum(call.get('elapsed', 0.0) for call in calls):.3f} 秒；完整调用记录见 plan.json。")
    comparison = getattr(result, "model_goal_comparison", None)
    if comparison is not None:
        lines.append("模型目标对比（仅对照，任务文件目标不变）：")
        recall = comparison.get("goal_recall")
        precision = comparison.get("goal_precision")
        invalid = comparison.get("goal_invalid")
        lines.append("完整目标召回率：" + (f"{recall:.3f}" if recall is not None else "未计算")
                     + "；精确率：" + (f"{precision:.3f}" if precision is not None else "未计算")
                     + "；无效条数：" + (str(invalid) if invalid is not None else "未计算"))
        lines.append("显式目标覆盖：" + ("完整" if comparison.get("covers_explicit") else "不完整"))
        for key, label in (("covered", "已覆盖"), ("missing", "缺少"), ("extra", "多余"), ("errors", "错误")):
            lines.append(label + "：" + ("；".join(map(str, comparison.get(key, []))) or "无"))
    model_error = getattr(result, "model_error", None)
    if model_error:
        lines.append("模型接入反馈：" + str(model_error))
    lines.extend(["", "③ 校验与补全："])
    if result.completion is None:
        lines.append("未提供方案，跳过校验与补全。")
    else:
        elapsed = getattr(result, "completion_elapsed", getattr(result.completion, "elapsed", 0.0))
        lines.append(f"补全耗时：{elapsed:.3f} 秒（累计各反馈轮次，不含模型推理）。")
        if getattr(result.completion, "abandoned", False):
            lines.append("已放弃修补：" + str(result.completion.abandon_reason)
                         + "；后续独立搜索不使用该方案作为参考，以下保留放弃前的检查记录。")
        lines.extend(f"- {entry.kind} {entry.action_text} —— {entry.reason}"
                     for entry in result.completion.entries)
        if not result.completion.entries:
            lines.append("没有需要处理的方案步骤。")
        if result.completion.unresolved:
            lines.append("未解决的目标：" + "、".join(result.completion.unresolved))
        else:
            lines.append("补全后的顺序方案已达成任务目标。")
    title = STATUS_TITLES.get(result.status, result.status)
    lines.extend(["", f"④ 搜索结果：{result.status}（{title}），完成时间 {schedule.makespan:.3f} s，"
                  f"扩展 {schedule.expanded} 个状态，用时 {schedule.elapsed:.3f} s"])
    lines.extend(f"   t={step.start:.3f}–{step.end:.3f}  {','.join(step.action.occupies)}  "
                 f"{step.action.skill.title}  {step.action}"
                 for step in sorted(schedule.steps, key=lambda step: step.start))
    lines.extend(["", "⑤ 运动验证与反馈："])
    if result.motion:
        for index, validation in enumerate(result.motion, 1):
            lines.append(f"- 第 {index} 步 {validation.status}：{validation.detail}")
            if validation.segments:
                lines.append("  原语：" + " → ".join(map(str, validation.segments)))
    else:
        lines.append("尚无运动验证结果。")
    lines.extend(f"- {feedback}" for feedback in result.feedback)
    if not result.feedback:
        lines.append("没有反馈记录。")
    graph = result.graph
    lines.extend(["", f"⑥ 任务顺序图：{len(graph.nodes) if graph else 0} 个节点、"
                  f"{len(graph.edges) if graph else 0} 条依赖"])
    if graph is not None:
        for node in graph.nodes:
            purpose = "；".join(node.purposes) or "完成该步骤的状态转换"
            lines.append(f"- 第 {node.id} 步 {node.action}：{purpose}")
    groups = defaultdict(list)
    for entry in _relevance(result):
        if entry["status"] == Status.VALID.value:
            groups[entry["skill"]].append(entry)
    if groups:
        lines.extend(["", "相关动作的用途："])
        for skill in problem.library.skills:
            entries = groups.get(skill.name, ())
            if not entries:
                continue
            purposes = sorted({reason for entry in entries for reason in entry["reasons"]})
            summary = "、".join(purposes[:8])
            if len(purposes) > 8:
                summary += f"等 {len(purposes)} 项事实"
            lines.append(f"- {skill.title}（{skill.name}）：{len(entries)} 个能力有效动作；为了 {summary}")
        lines.append("完整相关动作列表及能力证据见 plan.json。")
    if schedule.unknown:
        lines.extend(["", "未验证的能力："])
        lines.extend(f"- {fact} = UNKNOWN" for fact in sorted(schedule.unknown))
    return "\n".join(lines) + "\n"


def write_outputs(result, out="out") -> Path:
    directory = Path(out) / result.problem.name
    directory.mkdir(parents=True, exist_ok=True)
    graph = result.graph if result.graph is not None else TaskGraph([], [])
    (directory / "report.md").write_text(render_report(result), encoding="utf-8")
    (directory / "plan.json").write_text(json.dumps(plan_to_dict(result), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (directory / "taskgraph.mmd").write_text(graph.to_mermaid(), encoding="utf-8")
    (directory / "taskgraph.json").write_text(json.dumps(graph.to_json(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return directory
