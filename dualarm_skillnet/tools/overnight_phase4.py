"""Prepare and run the authorized fourth step in one persistent process."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import overnight_batch as batch
from tools.overnight_prepare import DEFAULT_MANIFEST, prepare_plan, rebuild_precheck_comparison


def main():
    result = 2
    try:
        manifest = prepare_plan(DEFAULT_MANIFEST)
        print(f"清单准备完成：{len(manifest['jobs'])} 项，开始串行独立规划。", flush=True)
        result = batch.main(["--manifest", str(DEFAULT_MANIFEST), "--phase", "plan"])
        rebuild_precheck_comparison(DEFAULT_MANIFEST)
        return result
    finally:
        if DEFAULT_MANIFEST.is_file():
            manifest = batch.load_manifest(DEFAULT_MANIFEST)
            completed = batch.Journal(manifest["state_dir"]).load()
            state_path = Path(manifest["state_dir"])/"state.json"
            state = batch.read_json(state_path) if state_path.is_file() else {}
            rows = [row for row in completed.values() if row["phase"] == "plan"]
            lines = ["\n## 第 4 步 后台独立规划\n",
                     f"后台记录时间：{batch.utc_now()}。状态：**{state.get('status', 'preparation_incomplete')}**；命令退出码 {result}。\n",
                     "无 proposal 或构造见证提示，规划核心及默认预算未改；每项完成即 fsync。\n",
                     "| 集合 | 请求 | 已记录 | solved | 未 solved |\n",
                     "|---|---:|---:|---:|---:|\n"]
            for cohort in ("precheck", "planning_200"):
                requested = sum(job["phase"] == "plan" and job["cohort"] == cohort for job in manifest["jobs"])
                subset = [row for row in rows if row["cohort"] == cohort]
                solved = sum(row["status"] == "solved" for row in subset)
                lines.append(f"| {cohort} | {requested} | {len(subset)} | {solved} | {len(subset)-solved} |\n")
            lines.extend(["\n逐项状态、扩展数、工时、辅助步骤和原因：`planning_precheck.csv`、`planning_200.csv`。\n",
                          "与串行见证的事后对照：`precheck_comparison.csv`；见证不参与搜索。\n",
                          "可恢复事实日志：`overnight/batch_state/completed.jsonl`；状态：`overnight/batch_state/state.json`。\n",
                          "后台输出：`out/overnight_phase4.log`、`out/overnight_phase4.stderr.log`；重启同一脚本跳过完成ID。\n"])
            with (ROOT/"results/sim_v1_3/overnight_report.md").open("a", encoding="utf-8") as stream:
                stream.write("".join(lines))
                stream.flush()


if __name__ == "__main__":
    raise SystemExit(main())
