"""Reproducible local-model comparisons, with a checkpoint after each run."""

from collections import Counter
import csv
from dataclasses import dataclass
import json
from pathlib import Path
from time import monotonic

from skillnet.domains.blocks import load_problem
from skillnet.pipeline import run
from skillnet.report import write_outputs


DEFAULT_MODEL = "qwen2.5:7b-instruct"
DEFAULT_HOST = "http://127.0.0.1:11434"
SETTINGS = ("none", "minimal", "with_capabilities", "nl_only")
PREVIOUS_CSV = Path("results/llm_v1_2/experiment.csv")
FIELDS = (
    "task", "setting", "model", "json_valid", "model_success", "parse_retries", "model_calls",
    "model_steps", "kept", "added", "deleted", "redundant",
    "capability_invalid_deletions", "goal_covers_explicit", "goal_covered",
    "goal_missing", "goal_extra", "goal_errors", "goal_recall", "goal_precision", "goal_invalid",
    "completion_goal_reached", "repair_abandoned", "repair_abandon_reason", "completion_elapsed",
    "status", "makespan", "expanded", "search_elapsed", "inference_elapsed",
    "total_elapsed", "matches_none", "timing_limit", "timing_within_budget", "model_error",
)
_JSON_FIELDS = frozenset({"goal_covered", "goal_missing", "goal_extra", "goal_errors"})


@dataclass
class ExperimentResult:
    rows: list[dict]
    success: bool
    directory: Path


def task_paths(tasks=None) -> list[Path]:
    """Select base tasks by name or explicit project-local TOML path."""
    root = Path.cwd().resolve()
    if tasks is None:
        selected = sorted(path for path in Path("tasks").glob("*.toml")
                          if not path.stem.endswith("_with_proposal"))
    else:
        selected = []
        for value in tasks:
            path = Path(value)
            if path.suffix != ".toml":
                path = Path("tasks") / (str(value) + ".toml")
            selected.append(path)
    if not selected:
        raise ValueError("没有可用于实验的任务")
    result = []
    seen = set()
    names = set()
    for path in selected:
        resolved = path.resolve()
        if not resolved.is_relative_to(root) or not resolved.is_file():
            raise ValueError(f"实验任务必须是项目目录内存在的 TOML 文件：{path}")
        if resolved.stem.endswith("_with_proposal"):
            raise ValueError(f"对比实验不包含带固定 proposal 的任务：{path}")
        if resolved in seen:
            continue
        if resolved.stem in names:
            raise ValueError(f"实验任务文件名重复：{path.stem}")
        seen.add(resolved)
        names.add(resolved.stem)
        result.append(resolved)
    return result


def _row(result, setting, model, total_elapsed, baseline):
    completion = result.completion
    entries = completion.entries if completion is not None else []
    counts = Counter(entry.kind for entry in entries)
    calls = list(getattr(result, "model_calls", []))
    comparison = getattr(result, "model_goal_comparison", None) or {}
    proposal = result.proposal
    schedule = result.schedule
    matches = (result.status == "solved" and baseline[0] == "solved"
               and abs(schedule.makespan - baseline[1]) <= 1e-6)
    inference_elapsed = sum(call.get("elapsed", 0.0) for call in calls)
    timing_limit = baseline[2] + inference_elapsed + 30.0
    return {
        "task": result.problem.name,
        "setting": setting,
        "model": model if setting != "none" else "",
        "json_valid": bool(calls[-1].get("json_valid")) if calls else None,
        "model_success": (None if setting == "none" else bool(
            proposal is not None and calls and calls[-1].get("json_valid")
            and getattr(result, "model_error", None) is None)),
        "parse_retries": sum(int(call.get("retry", 0) > 0) for call in calls),
        "model_calls": len(calls),
        "model_steps": len(proposal.steps) if proposal is not None else 0,
        "kept": counts["保留"], "added": counts["补出"],
        "deleted": counts["删除"], "redundant": counts["多余"],
        "capability_invalid_deletions": sum(
            entry.kind == "删除" and "能力检查不通过" in entry.reason for entry in entries),
        "goal_covers_explicit": comparison.get("covers_explicit"),
        "goal_covered": comparison.get("covered", []),
        "goal_missing": comparison.get("missing", []),
        "goal_extra": comparison.get("extra", []),
        "goal_errors": comparison.get("errors", []),
        "goal_recall": comparison.get("goal_recall"),
        "goal_precision": comparison.get("goal_precision"),
        "goal_invalid": comparison.get("goal_invalid"),
        "completion_goal_reached": completion.goal_reached if completion is not None else None,
        "repair_abandoned": bool(getattr(completion, "abandoned", False)),
        "repair_abandon_reason": getattr(completion, "abandon_reason", None),
        "completion_elapsed": getattr(result, "completion_elapsed", getattr(completion, "elapsed", 0.0)),
        "status": result.status, "makespan": schedule.makespan,
        "expanded": schedule.expanded, "search_elapsed": schedule.elapsed,
        "inference_elapsed": inference_elapsed,
        "total_elapsed": total_elapsed, "matches_none": matches,
        "timing_limit": timing_limit, "timing_within_budget": total_elapsed <= timing_limit,
        "model_error": getattr(result, "model_error", None),
    }


def _save_calls(directory, calls):
    if not calls:
        return
    directory.mkdir(parents=True, exist_ok=True)
    for index, record in enumerate(calls, 1):
        prefix = directory / f"{index:03d}"
        prefix.with_name(prefix.name + "_prompt.txt").write_text(record.get("prompt", ""), encoding="utf-8")
        prefix.with_name(prefix.name + "_response.txt").write_text(record.get("raw_output", ""), encoding="utf-8")
        prefix.with_name(prefix.name + "_record.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _previous_measurements():
    if not PREVIOUS_CSV.is_file():
        return {}
    with PREVIOUS_CSV.open(encoding="utf-8-sig", newline="") as source:
        previous = {}
        for row in csv.DictReader(source):
            try:
                previous[(row["task"], row["setting"])] = float(row["total_elapsed"])
            except (KeyError, TypeError, ValueError):
                continue
    return previous


def _cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def _summary(rows, expected_rows, model):
    complete = len(rows) == expected_rows
    consistent = bool(rows) and all(row["matches_none"] for row in rows)
    model_rows = [row for row in rows if row["setting"] != "none"]
    valid_models = sum(row["model_success"] is True for row in model_rows)
    expected_models = expected_rows // len(SETTINGS) * (len(SETTINGS) - 1)
    models_passed = len(model_rows) == expected_models and valid_models == expected_models
    timely = bool(rows) and all(row["timing_within_budget"] for row in rows)
    passed = complete and consistent and models_passed and timely
    lines = ["# 本地小模型对比实验", "", f"模型：`{model}`。",
             "四种设置：none（直接搜索）、minimal（含形式化目标）、with_capabilities（另含能力摘要）、nl_only（只从自然语言推断目标，仍提供场景、技能和能力）。",
             f"已记录 {len(rows)}/{expected_rows} 组。验收：{'通过' if passed else '未通过（尚未完成、规划不一致、模型方案无效或耗时超限）'}。",
             f"规划一致性：{'通过' if consistent else '未通过'}（已记录运行）；有效模型方案 {valid_models}/{expected_models}。",
             f"耗时约束：{'通过' if timely else '未通过'}，每组完整链路不得超过该任务 none 耗时 + 本组模型推理耗时 + 30 秒。", "",
             "| 任务 | 设置 | JSON | 补全达成 | 放弃修补 | 保留/补出/删除/多余 | 完成时间（秒） | 扩展状态 | 模型耗时（秒） | 补全耗时（秒） | 链路耗时（秒） | 与 none 一致 |",
             "|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|"]
    for row in rows:
        json_status = "不适用" if row["json_valid"] is None else ("合法" if row["json_valid"] else "失败")
        repaired = ("不适用" if row["completion_goal_reached"] is None
                    else "是" if row["completion_goal_reached"] else "否")
        counts = "/".join(str(row[key]) for key in ("kept", "added", "deleted", "redundant"))
        abandoned = "是" if row["repair_abandoned"] else "否"
        lines.append(f"| {row['task']} | {row['setting']} | {json_status} | {repaired} | {abandoned} | {counts} | "
                     f"{row['makespan']:.6f} | {row['expanded']} | {row['inference_elapsed']:.3f} | "
                     f"{row['completion_elapsed']:.3f} | "
                     f"{row['total_elapsed']:.3f} | "
                     f"{'是' if row['matches_none'] else '否'}（{row['status']}） |")
    lines.extend(["", "## 简短结论", ""])
    if passed:
        lines.append("所有任务四种设置均为 solved，完成时间与直接搜索一致，误差不超过 1e-6，且完整链路耗时约束全部满足。模型方案仅用于校验补全和搜索参考，任务文件目标保持不变。")
    elif not complete:
        lines.append("实验尚未跑完，以上只汇总已完成的运行，不对未记录设置作结论。")
    elif consistent and not models_passed:
        lines.append(f"四种设置的规划一致性通过，但有效模型方案仅 {valid_models}/{expected_models}；模型失败后的规划回退不能使总体验收通过。")
    elif not timely:
        lines.append("存在完整链路耗时超限的运行，不能仅凭 solved 或完成时间一致通过验收。")
    else:
        lines.append("存在未解决任务或完成时间偏差，尚未满足最优性验收，必须核查对应计划与调用记录。")
    for setting in SETTINGS[1:]:
        group = [row for row in rows if row["setting"] == setting]
        if not group:
            continue
        lines.append(f"- {setting}：最终 JSON 合法 {sum(row['json_valid'] is True for row in group)}/{len(group)}，"
                     f"有效模型方案 {sum(row['model_success'] is True for row in group)}/{len(group)}，"
                     f"解析重试 {sum(row['parse_retries'] for row in group)} 次，"
                     f"补全达成 {sum(row['completion_goal_reached'] is True for row in group)}/{len(group)}，"
                     f"放弃修补 {sum(row['repair_abandoned'] for row in group)} 组，"
                     f"能力检查不通过导致删除 {sum(row['capability_invalid_deletions'] for row in group)} 步，"
                     f"模型请求合计 {sum(row['inference_elapsed'] for row in group):.3f} 秒，"
                     f"完整链路合计 {sum(row['total_elapsed'] for row in group):.3f} 秒。")
    if any(row['completion_goal_reached'] is False for row in model_rows):
        lines.append("存在可解析提案但补全未达目标的设置，最终由独立搜索求解；JSON 合法或最终 solved 不代表模型步骤已经可执行。")
    failures = [row for row in rows if row["model_error"]]
    if failures:
        lines.append(f"模型调用失败后回退直接搜索的运行有 {len(failures)} 组，不能把回退后的 solved 当作模型成功生成方案。")

    lines.extend(["", "## nl_only 目标理解", "",
                  "以下召回率和精确率与完整任务目标（显式目标及默认目标）比较，不使用模型目标作为规划目标。",
                  "| 任务 | 召回率 | 精确率 | 无效条数 | 缺失目标 | 多余目标 |",
                  "|---|---:|---:|---:|---|---|"])
    natural_rows = [row for row in rows if row["setting"] == "nl_only"]
    for row in natural_rows:
        recall = "未取得" if row["goal_recall"] is None else f"{row['goal_recall']:.3f}"
        precision = "未取得" if row["goal_precision"] is None else f"{row['goal_precision']:.3f}"
        invalid = "未取得" if row["goal_invalid"] is None else str(row["goal_invalid"])
        lines.append(f"| {row['task']} | {recall} | {precision} | {invalid} | "
                     f"{_cell('、'.join(row['goal_missing']) or '无')} | "
                     f"{_cell('、'.join(row['goal_extra']) or '无')} |")
    if not natural_rows:
        lines.append("尚无已完成的 nl_only 设置。")

    lines.extend(["", "## 放弃修补统计", ""])
    abandoned_rows = [row for row in rows if row["repair_abandoned"]]
    lines.append(f"已记录设置中有 {len(abandoned_rows)} 组放弃修补；放弃后独立搜索不使用该方案作为参考。")
    if abandoned_rows:
        lines.extend(["", "| 任务 | 设置 | 原因 | 补全耗时（秒） |", "|---|---|---|---:|"])
        for row in abandoned_rows:
            lines.append(f"| {row['task']} | {row['setting']} | {_cell(row['repair_abandon_reason'])} | {row['completion_elapsed']:.3f} |")

    lines.extend(["", "## v1.2 耗时对比", "",
                  "旧结果只读；表中均为完整链路墙钟时间，不与计划完成时间混淆。",
                  "| 任务 | 设置 | v1.2（秒） | v1.2.1（秒） | 新减旧（秒） |",
                  "|---|---|---:|---:|---:|"])
    previous = _previous_measurements()
    for row in rows:
        old = previous.get((row["task"], row["setting"]))
        if old is None:
            lines.append(f"| {row['task']} | {row['setting']} | 无旧设置 | {row['total_elapsed']:.3f} | 不适用 |")
        else:
            lines.append(f"| {row['task']} | {row['setting']} | {old:.3f} | {row['total_elapsed']:.3f} | {row['total_elapsed']-old:.3f} |")
    if not previous:
        lines.append("未找到可读取的 v1.2 对照数据。")

    exceeded = [row for row in rows if not row["timing_within_budget"]]
    if exceeded:
        lines.extend(["", "## 耗时超限", ""])
        lines.extend(f"- {row['task']} / {row['setting']}：{row['total_elapsed']:.3f} 秒，允许上限 {row['timing_limit']:.3f} 秒。"
                     for row in exceeded)
    lines.extend(["", "## 指标与复查", "",
                  "experiment.csv 为逐任务、逐设置的完整指标。调用记录在 <设置>/<任务>/calls/，包括每次提示词、原始输出、原始 API 响应、请求参数与耗时；同目录的 plan.json 保存最终时间表、补全说明和模型目标对比。",
                  "JSON 列描述最后一次输出是否为合法 JSON；是否满足提案结构另见调用记录的 proposal_valid，所有失败尝试也保存在记录中。parse_retries 为解析重试请求数，不包含未解决目标触发的新一轮模型方案。",
                  "model_success 对 none 为空；模型设置只有取得可解析提案、实际调用记录、最后输出合法 JSON 且没有模型错误时才为真，空步骤数组仍可作为有效提案交给规则层补全。总体验收同时要求规划一致性与全部模型设置实际取得有效提案。",
                  "能力删除数包含 INVALID 或 UNKNOWN 能力检查不通过的删除；四类步骤数来自最终一轮完整补全报告。none 没有模型或补全，因此模型 JSON／补全达成指标为空。",
                  "search_elapsed 是最终调度搜索耗时（含启发值构建与可选参考搜索），不含补全子搜索；total_elapsed 是包含模型、补全和调度的完整链路墙钟时间。模型耗时是所有请求墙钟时间之和，不并入计划完成时间。小模型目标仅做对照，不替换任务目标。",
                  "goal_recall、goal_precision、goal_invalid 及缺失／多余列表按完整任务目标统计；goal_covers_explicit 仅为保留的显式目标兼容指标。比较口径与逐条错误见 plan.json。",
                  "completion_elapsed 累计各反馈轮次的补全时间，不包括模型推理；单次修补子搜索最多 50000 状态或 5 秒，整个补全最多 20 秒。放弃原因见 repair_abandon_reason，部分步骤仅保留供复查。",
                  "可按 experiment --tasks 选择任务复查；每完成一组就更新 CSV 与本汇总。", ""])
    return "\n".join(lines)


def _checkpoint(directory, rows, expected_rows, model):
    destination = directory / "experiment.csv"
    with destination.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if key in _JSON_FIELDS else value
                             for key, value in row.items()})
    (directory / "summary.md").write_text(_summary(rows, expected_rows, model), encoding="utf-8")


def run_experiment(*, model=DEFAULT_MODEL, out="results/llm_v1_2_1", tasks=None,
                   host=DEFAULT_HOST, decomposer_factory=None):
    paths = task_paths(tasks)
    directory = Path(out).resolve()
    if not directory.is_relative_to(Path.cwd().resolve()):
        raise ValueError("实验结果只能写入项目目录内")
    old_directory = (Path.cwd() / "results/llm_v1_2").resolve()
    if directory == old_directory or old_directory in directory.parents:
        raise ValueError("不能覆盖 v1.2 旧实验结果，请使用 results/llm_v1_2_1 或其他新目录")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("实验模型名称不能为空")
    if decomposer_factory is None:
        from skillnet.decompose.ollama import OllamaDecomposer
        decomposer_factory = OllamaDecomposer
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    expected_rows = len(paths) * len(SETTINGS)
    _checkpoint(directory, rows, expected_rows, model)
    for path in paths:
        baseline = None
        for setting in SETTINGS:
            began = monotonic()
            problem = load_problem(path)
            problem.proposal = None
            decomposer = None if setting == "none" else decomposer_factory(model=model, host=host, mode=setting)
            result = run(problem, decomposer=decomposer)
            total_elapsed = monotonic() - began
            if baseline is None:
                baseline = (result.status, result.schedule.makespan, total_elapsed)
            row = _row(result, setting, model, total_elapsed, baseline)
            rows.append(row)
            run_directory = write_outputs(result, directory / setting)
            _save_calls(run_directory / "calls", getattr(result, "model_calls", []))
            _checkpoint(directory, rows, expected_rows, model)
            print(f"实验 {len(rows)}/{expected_rows}：{problem.name} / {setting}，{result.status}，"
                  f"完成时间 {result.schedule.makespan:.6f} 秒，模型耗时 {row['inference_elapsed']:.3f} 秒",
                  flush=True)
    success = (len(rows) == expected_rows and all(row["matches_none"] for row in rows)
               and all(row["model_success"] for row in rows if row["setting"] != "none")
               and all(row["timing_within_budget"] for row in rows))
    return ExperimentResult(rows, success, directory)
