"""Chinese CLI for isolated, real dual-arm simulator runs."""
import json

from skillnet.__main__ import ChineseArgumentParser
from .runner import DEFAULT_OUT, PORT, batch, run_task


def main(argv=None):
    parser = ChineseArgumentParser(prog="simbridge", description="双臂积木真实运动与动力学仿真")
    commands = parser.add_subparsers(dest="command", required=True, title="命令")
    single = commands.add_parser("run", help="运行并验收一个任务")
    single.add_argument("task", metavar="TASK", help="任务 TOML 路径")
    single.add_argument("--port", type=int, default=PORT, help="独立仿真端口，固定为 23210")
    single.add_argument("--headless", action="store_true", default=True, help="以无界面方式启动（默认开启）")
    single.add_argument("--video", action="store_true", help="请求可选录像；默认关闭")
    single.add_argument("--out", default=str(DEFAULT_OUT), metavar="DIR", help="项目内的输出根目录")
    single.add_argument("--timeout", type=float, default=1200., help="每任务墙钟超时秒数，默认 1200")
    multiple = commands.add_parser("batch", help="按清单单线运行任务")
    multiple.add_argument("--manifest", required=True, metavar="TOML", help="含 [[run]] 的任务清单")
    from .qualifier import DEFAULT_OUT as QUALIFY_OUT
    qualify = commands.add_parser("qualify", help="真实IK验证所有臂/位置组合并保存新能力缓存")
    qualify.add_argument("--tasks", nargs="+", required=True, metavar="TASK", help="任务TOML路径列表")
    qualify.add_argument("--out", default=str(QUALIFY_OUT), metavar="DIR", help="v134资格输出根目录")
    qualify.add_argument("--prior-qualification", nargs="+", metavar="QUALIFIED", help="与任务逐项对应的旧IK缓存；只升级原UNKNOWN")
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            result = run_task(args.task, port=args.port, headless=args.headless,
                              video=args.video, out=args.out, timeout_wall_s=args.timeout)
        elif args.command == "batch":
            result = batch(args.manifest)
        else:
            from .qualifier import qualify_tasks
            options = {} if args.prior_qualification is None else {"prior_qualification": args.prior_qualification}
            result = qualify_tasks(args.tasks, out=args.out, **options)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if args.command == "qualify":
            return 0 if result["status"] == "completed" else 1
        return 0 if result["passed"] else 1
    except (ValueError, OSError) as exc:
        parser.exit(2, f"错误：{exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
