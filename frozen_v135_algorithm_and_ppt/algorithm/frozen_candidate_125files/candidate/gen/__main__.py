"""CLI for task generation and geometry/rule smoke checks only."""
from skillnet.__main__ import ChineseArgumentParser

from .generator import DEFAULT_OUT, generate_samples


def main(argv=None):
    parser = ChineseArgumentParser(prog="gen", description="生成积木任务并进行纯几何与规则重放")
    parser.add_argument("--seed", type=int, default=20261115, help="固定随机种子")
    parser.add_argument("--count", type=int, default=2, help="每族样例数，离线冒烟默认 2")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="项目内输出目录")
    parser.add_argument("--distance-prior", action="store_true", help="显式写入离线距离先验；不是IK结果")
    args = parser.parse_args(argv)
    try:
        report = generate_samples(args.seed, args.count, args.out, distance_prior=args.distance_prior)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"错误：{exc}\n")
    print(f"已生成 {report['tasks_generated']} 个任务，五族各 {report['per_family']} 个。")
    print("纯几何及规则重放通过；没有运行独立规划搜索、IK、动力学或带墙钟预算的实验。")
    print(f"其中 {report['conditional_witnesses']} 个见证包含 UNKNOWN 能力，仍待资格确认，不能计作能力全部通过。")
    print(f"证据：{args.out}/smoke_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
