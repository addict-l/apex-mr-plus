"""Command-line entry points for offline dual-arm planning."""

import argparse
from pathlib import Path
import sys

from skillnet.core.library import load_library
from skillnet.domains.blocks import DERIVED_BASIS, load_problem
from skillnet.report import render_report, write_outputs


class ChineseArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        kwargs["add_help"] = False
        super().__init__(*args, **kwargs)
        self.add_argument("-h", "--help", action="help", help="显示帮助并退出")

    def format_usage(self):
        return super().format_usage().replace("usage:", "用法：", 1)

    def format_help(self):
        return (super().format_help().replace("usage:", "用法：", 1)
                .replace("positional arguments:", "位置参数：").replace("options:", "选项："))

    def error(self, message):
        translations = {
            "the following arguments are required:": "缺少必需参数：",
            "unrecognized arguments:": "无法识别的参数：",
            "invalid choice:": "无效选项：",
            "choose from": "请选择",
            "argument ": "参数 ",
            "expected one argument": "需要一个参数值",
        }
        for source, target in translations.items():
            message = message.replace(source, target)
        self.print_usage(sys.stderr)
        self.exit(2, f"错误：{message}\n")


def _parser():
    parser = ChineseArgumentParser(prog="skillnet", description="双臂积木技能网络规划器")
    commands = parser.add_subparsers(dest="command", title="命令", required=True)
    network = commands.add_parser("network", help="查看并导出技能网络")
    network.add_argument("--skills", default="skills/blocks_v1.toml", metavar="PATH", help="技能库 TOML 路径")
    network.add_argument("--out", default="out", metavar="DIR", help="输出目录，默认 out")
    run = commands.add_parser("run", help="规划一个任务并输出中文报告")
    run.add_argument("task", metavar="TASK", help="任务 TOML 路径")
    run.add_argument("--skills", default="skills/blocks_v1.toml", metavar="PATH", help="技能库 TOML 路径")
    run.add_argument("--out", default="out", metavar="DIR", help="输出目录，默认 out")
    run.add_argument("--no-proposal", action="store_true", help="跳过任务文件中的小模型方案")
    run.add_argument("--no-prune", action="store_true", help="直接使用完整动作集合搜索")
    run.add_argument("--model", metavar="NAME", help="使用本机 Ollama 模型生成方案")
    run.add_argument("--mode", choices=("minimal", "with_capabilities", "nl_only"), default="with_capabilities", help="模型提示词模式")
    run.add_argument("--host", default="http://127.0.0.1:11434", metavar="URL", help="本机 Ollama 地址")
    prompt = commands.add_parser("prompt", help="生成小模型提示词")
    prompt.add_argument("task", metavar="TASK", help="任务 TOML 路径")
    prompt.add_argument("--mode", choices=("minimal", "with_capabilities", "nl_only"), default="with_capabilities", help="提示词模式，默认包含能力摘要")
    all_tasks = commands.add_parser("all", help="依次规划全部示例任务")
    all_tasks.add_argument("--out", default="out", metavar="DIR", help="输出目录，默认 out")
    experiment = commands.add_parser("experiment", help="运行本地模型四种设置的对比实验")
    experiment.add_argument("--model", default="qwen2.5:7b-instruct", metavar="NAME", help="已下载的本机模型名称")
    experiment.add_argument("--out", default="results/llm_v1_2_1", metavar="DIR", help="实验结果目录")
    experiment.add_argument("--tasks", nargs="+", metavar="TASK", help="任务名或项目内任务 TOML 路径，默认九个基础任务")
    experiment.add_argument("--host", default="http://127.0.0.1:11434", metavar="URL", help="本机 Ollama 地址")
    return parser


def _network(args):
    library = load_library(args.skills, derived_basis=DERIVED_BASIS)
    print("技能清单：")
    for skill in library.skills:
        print(f"  {skill.signature}  {skill.title}")
    print("技能网络连线：")
    for edge in library.network():
        print(f"  {edge.source} → {edge.target}：{edge.label}")
    directory = Path(args.out)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "skill_network.mmd"
    destination.write_text(library.to_mermaid(), encoding="utf-8")
    print(f"技能网络已写入 {destination}")
    return 0


def _run_task(task, out, skills="skills/blocks_v1.toml", no_proposal=False, prune=True,
              model=None, mode="with_capabilities", host="http://127.0.0.1:11434"):
    from skillnet.pipeline import run

    problem = load_problem(task, skills=skills)
    if no_proposal:
        problem.proposal = None
    decomposer = None
    if model is not None:
        from skillnet.decompose.ollama import OllamaDecomposer
        problem.proposal = None
        decomposer = OllamaDecomposer(model=model, mode=mode, host=host)
    result = run(problem, prune=prune, decomposer=decomposer)
    print(render_report(result), end="")
    destination = write_outputs(result, out)
    print(f"输出已写入 {destination}\n")
    return result


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = _parser().parse_args(argv)
    try:
        if args.command == "network":
            return _network(args)
        if args.command == "prompt":
            from skillnet.decompose.prompt import build_prompt

            print(build_prompt(load_problem(args.task), args.mode))
            return 0
        if args.command == "run":
            if args.model is not None and args.no_proposal:
                raise ValueError("--model 与 --no-proposal 不能同时使用")
            result = _run_task(args.task, args.out, args.skills, args.no_proposal, not args.no_prune,
                               args.model, args.mode, args.host)
            return 0 if result.status == "solved" else 1
        if args.command == "experiment":
            from skillnet.experiment import run_experiment
            result = run_experiment(model=args.model, out=args.out, tasks=args.tasks, host=args.host)
            print(f"实验结果已写入 {result.directory}")
            print("实验验收：" + ("通过" if result.success else "未通过"))
            return 0 if result.success else 1
        paths = sorted(Path("tasks").glob("*.toml"))
        if not paths:
            raise ValueError("tasks 目录中没有任务文件")
        results = [_run_task(path, args.out) for path in paths]
        print("任务汇总：")
        print("任务 | 结论 | 是否使用 proposal | 完成时间（秒） | 步骤数 | 扩展状态数 | 耗时（秒）")
        for result in results:
            schedule = result.schedule
            used_proposal = "是" if result.proposal is not None else "否"
            print(f"{result.problem.name} | {result.status} | {used_proposal} | {schedule.makespan:.3f} | "
                  f"{len(schedule.steps)} | {schedule.expanded} | {schedule.elapsed:.3f}")
        return 0 if all(result.status == "solved" for result in results) else 1
    except (ValueError, OSError) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
