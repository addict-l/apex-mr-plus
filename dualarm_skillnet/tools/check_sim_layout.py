"""Check task geometry without starting a simulator or planner."""
import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simbridge.layout import check_layout


def main(argv=None):
    parser = argparse.ArgumentParser(description="纯几何检查仿真任务布局，不启动仿真或规划搜索")
    parser.add_argument("tasks", nargs="+", type=Path, help="任务 TOML 文件")
    parser.add_argument("--out", type=Path, help="项目内的 JSON 结果路径")
    args = parser.parse_args(argv)
    reports = [check_layout(path) for path in args.tasks]
    result = {"valid": all(report["valid"] for report in reports), "reports": reports}
    rendered = json.dumps(result, ensure_ascii=False, indent=2)+"\n"
    if args.out:
        destination = args.out.resolve()
        if not destination.is_relative_to(ROOT):
            parser.error("检查结果只能写入项目根目录内")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
