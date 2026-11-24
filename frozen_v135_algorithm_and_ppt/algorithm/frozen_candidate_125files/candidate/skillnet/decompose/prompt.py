"""Chinese offline model prompts, JSON parsing, and repair feedback."""

import json

from skillnet.capability.oracle import Status
from skillnet.core.logic import Fact, parse_condition, parse_fact
from skillnet.core.problem import Proposal


def _capability_lines(problem):
    lines = ["能力摘要"]
    places = problem.objects["loc"] + problem.objects["slot"]
    for arm in problem.objects["arm"]:
        reach = {status: [] for status in Status}
        for place in places:
            evidence = problem.oracle.check(Fact("reach", (arm, place)))
            reach[evidence.status].append(place)
        for status, label in ((Status.VALID, "能到达"), (Status.INVALID, "不能到达"),
                              (Status.UNKNOWN, "可达性未验证")):
            lines.append(f"{arm} {label}：{', '.join(reach[status]) or '无'}")
        liftable = {status: [] for status in Status}
        for block in problem.objects["block"]:
            evidence = problem.oracle.check(Fact("liftable", (arm, block)))
            liftable[evidence.status].append(block)
        for status, label in ((Status.VALID, "可单臂提起"), (Status.INVALID, "不可单臂提起"),
                              (Status.UNKNOWN, "提起能力未验证")):
            lines.append(f"{arm} {label}：{', '.join(liftable[status]) or '无'}")
    two_arm = [block for block in problem.objects["block"]
               if Fact("two_arm_part", (block,)) in problem.static]
    lines.append(f"需要两只臂拿的积木：{', '.join(two_arm) or '无'}")
    return lines


def build_prompt(problem, mode="with_capabilities") -> str:
    if mode not in {"minimal", "with_capabilities", "nl_only"}:
        raise ValueError("提示词模式必须是 minimal、with_capabilities 或 nl_only")
    instruction = problem.instruction or problem.description
    lines = ["请为双臂积木任务生成可执行的技能步骤。",
             f"任务 {problem.name}",
             f"任务指令：{instruction}"]
    if mode != "nl_only":
        lines.append("任务目标：" + ", ".join(map(str, problem.goal)))
    lines.extend(["可用机械臂：" + ", ".join(problem.objects["arm"]), "场景信息"])
    lines.extend(problem.hooks.scene_summary())
    lines.append("技能清单")
    for skill in problem.library.skills:
        lines.append(f"{skill.signature} — {skill.title}，{skill.description}")
        lines.append("参数类型：" + ", ".join(f"{name}={kind}" for name, kind in skill.params.items()))
        lines.append("前置条件：" + (", ".join(map(str, skill.pre)) or "无"))
        lines.append("开始时删除：" + (", ".join(map(str, skill.delete)) or "无"))
        lines.append("结束时添加：" + (", ".join(map(str, skill.add)) or "无"))
    if mode in {"with_capabilities", "nl_only"}:
        lines.extend(_capability_lines(problem))
    lines.extend([
        "只能使用上述技能库中的技能和场景中存在的对象，参数顺序必须与技能签名一致。",
        "同类型的两个参数必须取不同对象。遵守前置条件、上下装配关系和支撑要求。",
        "请只输出一个 JSON 对象，包含 goal 和 steps 两个字符串数组。",
    ])
    if mode == "nl_only":
        lines.extend([
            "请根据任务指令自行理解要达成的状态，在 goal 中写出已代入场景对象的目标条件。",
            "steps 按执行顺序填写已代入对象的动作字符串。不要输出解释或 Markdown 代码围栏。",
            '{"goal": [], "steps": []}',
            "上面仅示意 JSON 对象结构；两个数组的实际内容都需要你根据本任务自行填写。",
        ])
    else:
        lines.extend([
            "goal 填写任务目标，steps 按执行顺序填写已代入对象的动作字符串。不要输出解释或 Markdown 代码围栏。",
            '{"goal": ["at(red, board_l1)"], "steps": ["pick(B, red, stock_r1)"]}',
            "上面的对象名称仅示意 JSON 格式，实际输出必须使用本任务中的对象和目标。",
        ])
    return "\n".join(lines) + "\n"


def parse_model_output(text: str) -> Proposal:
    if not isinstance(text, str):
        raise ValueError("模型输出必须是文本")
    start = text.find("{")
    if start < 0:
        raise ValueError("模型输出中没有 JSON 对象")
    try:
        value, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError as error:
        raise ValueError(f"模型输出中的第一个 JSON 对象格式错误，位置 {error.pos}") from error
    if not isinstance(value, dict):
        raise ValueError("模型输出必须包含 JSON 对象")
    for field in ("goal", "steps"):
        if field not in value:
            raise ValueError(f"模型输出缺少必需字段 {field}")
        entries = value[field]
        if not isinstance(entries, list) or any(not isinstance(item, str) for item in entries):
            raise ValueError(f"模型输出的 {field} 必须是字符串数组")
        for index, entry in enumerate(entries, 1):
            try:
                (parse_condition if field == "goal" else parse_fact)(entry)
            except ValueError as error:
                raise ValueError(f"模型输出的 {field} 第 {index} 项格式错误：{error}") from error
    return Proposal("", list(value["goal"]), list(value["steps"]), "模型输出")


def build_feedback(report) -> list[str]:
    if not report.unresolved:
        return []
    messages = list(report.feedback_for_model)
    for entry in report.entries:
        if entry.kind != "删除":
            continue
        if any(entry.action_text in message and entry.reason in message for message in messages):
            continue
        index = entry.proposal_index
        prefix = f"第 {index} 步 " if index is not None else ""
        messages.append(f"{prefix}{entry.action_text}：{entry.reason}")
    for unresolved in report.unresolved:
        if not any(unresolved in message for message in messages):
            messages.append(f"仍未解决的目标：{unresolved}")
    return messages
