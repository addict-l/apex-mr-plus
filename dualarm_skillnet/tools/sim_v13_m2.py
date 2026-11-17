"""Compile a real task with the stopped CoppeliaSim IK/RRT geometry model."""
import json
import argparse
from pathlib import Path
import shutil
import time
import traceback

from apexmr.motion import MotionPlanner
from apexmr.sim_connection import Simulator
from skillnet.domains.blocks import load_problem
from skillnet.pipeline import run
from skillnet.report import write_outputs
from simbridge.compiler import MotionCompiler
from simbridge.frames import Frames
from simbridge.workcell import build


ROOT = Path(__file__).resolve().parents[1]


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                          encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--task", default="tasks/s1_parallel_place.toml")
    parser.add_argument("--label", choices=["m2", "m3_compile"], default="m2")
    args = parser.parse_args()
    if shutil.disk_usage("C:\\").free < 5 * 1024**3:
        raise RuntimeError("C 盘剩余不足 5 GB，按交接单停止")
    task = (ROOT / args.task).resolve()
    task_output = ROOT / "results/sim_v1_3" / task.stem
    output = task_output / f"{args.label}_attempt_{args.attempt}"
    output.mkdir(parents=True, exist_ok=True)
    report = {"milestone": args.label.upper(), "passed": False, "port": 23210}
    owner = Simulator(ROOT, port=23210, timeout=30)
    started = time.monotonic()
    compiler = None
    try:
        problem = load_problem(task, ROOT / "skills/blocks_v1.toml")
        planning_started = time.monotonic()
        result = run(problem)
        report["planning_seconds"] = time.monotonic() - planning_started
        report["planning_status"] = result.status
        if result.status != "solved":
            raise RuntimeError(f"规划失败：{result.status}")
        write_outputs(result, task_output.parent)
        owner.start(headless=True)
        report["pid"] = owner.process.pid
        client, sim = owner.connect()
        cfg = build(sim, client, task, owner.install, output)
        with MotionPlanner(sim, client, cfg) as planner:
            compiler = MotionCompiler(planner, cfg, Frames(task))
            compile_started = time.monotonic()
            tasks = compiler.compile(result.schedule, result.graph)
            report["compile_seconds"] = time.monotonic() - compile_started
            report["task_count"] = len(tasks)
            save(output / "tasks.json", tasks)
            report["passed"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        if getattr(exc, "status", None) is not None:
            report["status"] = exc.status.value
            report["segment"] = exc.segment
            report["failure_evidence"] = exc.evidence
        raise
    finally:
        if compiler is not None:
            save(output / "compile_evidence.json", compiler.evidence)
        owner.close()
        report["closed"] = owner.process is not None and owner.process.poll() is not None
        report["elapsed_seconds"] = time.monotonic() - started
        report["log"] = str(owner.log_path)
        save(output / "m2_validation.json", report)
        print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
