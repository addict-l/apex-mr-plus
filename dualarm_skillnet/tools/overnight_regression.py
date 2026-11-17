"""Run isolated standard-unittest regressions with durable timing evidence."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import time
import traceback
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "out"
RESULTS = ROOT / "results" / "sim_v1_3"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def health():
    """Read machine guards without starting or stopping external processes."""
    command = (
        "@(Get-CimInstance Win32_Process | Where-Object { "
        "$_.Name -match '(?i)coppelia' } | Select-Object ProcessId,Name,CommandLine) "
        "| ConvertTo-Json -Compress"
    )
    checked = subprocess.run(["powershell.exe", "-NoProfile", "-Command", command],
                             capture_output=True, text=True, check=True)
    processes = json.loads(checked.stdout.strip() or "[]")
    if isinstance(processes, dict):
        processes = [processes]
    with socket.socket() as port:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            port.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        try:
            port.bind(("0.0.0.0", 23210))
            port_free = True
        except OSError:
            port_free = False
    free = shutil.disk_usage("C:\\").free
    return {"time": utc_now(), "c_free_bytes": free,
            "c_free_gib": round(free / 1024**3, 3),
            "coppelia_processes": processes, "port_23210_free": port_free,
            "passed": free >= 5 * 1024**3 and not processes and port_free}


class TimedResult(unittest.TextTestResult):
    def __init__(self, *args, record, path, **kwargs):
        super().__init__(*args, **kwargs)
        self.record, self.path = record, path
        self.current = None

    def save(self):
        self.record["tests_run"] = self.testsRun
        write_json(self.path, self.record)

    def startTest(self, test):
        self.current = {"id": test.id(), "status": "running", "seconds": None}
        self.record["per_test"].append(self.current)
        self.started = time.perf_counter()
        super().startTest(test)

    def stopTest(self, test):
        self.current["seconds"] = round(time.perf_counter() - self.started, 6)
        super().stopTest(test)
        self.save()

    def addSuccess(self, test):
        self.current["status"] = "passed"
        super().addSuccess(test)

    def addSkip(self, test, reason):
        self.current.update(status="skipped", reason=reason)
        super().addSkip(test, reason)

    def addFailure(self, test, err):
        self.failure(test, err, "failed")
        super().addFailure(test, err)

    def addError(self, test, err):
        self.failure(test, err, "error")
        super().addError(test, err)

    def addSubTest(self, test, subtest, err):
        if err is not None:
            self.failure(test, err, "failed", subtest.id())
        super().addSubTest(test, subtest, err)

    def addExpectedFailure(self, test, err):
        self.current["status"] = "expected_failure"
        super().addExpectedFailure(test, err)

    def addUnexpectedSuccess(self, test):
        self.current["status"] = "unexpected_success"
        self.record["failures"].append({"id": test.id(), "status": "unexpected_success"})
        super().addUnexpectedSuccess(test)

    def failure(self, test, err, status, subtest=None):
        detail = {"id": test.id(), "status": status,
                  "traceback": "".join(traceback.format_exception(*err))}
        if subtest:
            detail["subtest"] = subtest
        self.record["failures"].append(detail)
        if self.current and self.current["id"] == test.id():
            self.current["status"] = status
        self.save()


def worker(args):
    os.chdir(ROOT)
    # These integrations remain gated during every default and localization run.
    os.environ["SKILLNET_LIVE_SIM"] = "0"
    os.environ["SKILLNET_LIVE_MODEL"] = "0"
    record = {"label": args.label, "pid": os.getpid(), "started": utc_now(),
              "hashseed": os.environ.get("PYTHONHASHSEED", "random/default"),
              "status": "running", "tests_run": 0, "per_test": [],
              "support_tasks": [], "failures": []}
    path = Path(args.json)
    write_json(path, record)
    support = importlib.import_module("tests.test_support_scenarios")
    original = support.run

    def timed_support_run(problem, *positional, **kwargs):
        start = time.perf_counter()
        item = {"task": problem.name, "status": "running", "seconds": None}
        record["support_tasks"].append(item)
        write_json(path, record)
        try:
            answer = original(problem, *positional, **kwargs)
            item["status"] = answer.status
            return answer
        except BaseException as error:
            item.update(status="exception", exception=repr(error))
            raise
        finally:
            item["seconds"] = round(time.perf_counter() - start, 6)
            print(f"\nSUPPORT_TASK {json.dumps(item, ensure_ascii=False)}", flush=True)
            write_json(path, record)

    support.run = timed_support_run
    started = time.perf_counter()
    try:
        loader = unittest.TestLoader()
        suite = (loader.loadTestsFromName(args.module) if args.module else
                 loader.discover("tests", pattern="test*.py", top_level_dir="."))
        record["tests_discovered"] = suite.countTestCases()
        runner = unittest.TextTestRunner(
            verbosity=2, resultclass=lambda *a, **kw: TimedResult(
                *a, record=record, path=path, **kw))
        result = runner.run(suite)
        record.update(status="passed" if result.wasSuccessful() else "failed",
                      tests_run=result.testsRun, skipped=len(result.skipped),
                      failure_count=len(result.failures), error_count=len(result.errors))
        code = 0 if result.wasSuccessful() else 1
    except BaseException:
        record.update(status="harness_error", harness_traceback=traceback.format_exc())
        traceback.print_exc()
        code = 2
    finally:
        support.run = original
        record.update(total_seconds=round(time.perf_counter() - started, 6), finished=utc_now())
        write_json(path, record)
    return code


def failed_modules(record):
    modules = set()
    for failure in record.get("failures", []):
        matched = re.search(r"tests\.test_[A-Za-z0-9_]+", failure["id"])
        if matched:
            modules.add(matched.group())
    return sorted(modules)


def supervise():
    os.chdir(ROOT)
    OUT.mkdir(exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}"
    summary_path = RESULTS / f"overnight_regression_{stamp}.json"
    summary = {"started": utc_now(), "pid": os.getpid(), "stamp": stamp,
               "status": "running", "runs": [], "health": []}

    def save():
        write_json(summary_path, summary)

    def guard(stage):
        record = health()
        record["stage"] = stage
        summary["health"].append(record)
        save()
        if not record["passed"]:
            raise RuntimeError("前置检查未通过：C 盘空间、CoppeliaSim 进程或端口状态异常")

    def run_one(label, seed, module=None):
        guard("before_" + label)
        log = OUT / f"overnight_regression_{stamp}_{label}.log"
        record_path = OUT / f"overnight_regression_{stamp}_{label}.json"
        command = [sys.executable, "-X", "utf8", str(Path(__file__).resolve()),
                   "--worker", "--label", label, "--json", str(record_path)]
        if module:
            command += ["--module", module]
        environment = dict(os.environ, SKILLNET_LIVE_SIM="0", SKILLNET_LIVE_MODEL="0")
        environment.pop("PYTHONHASHSEED", None)
        if seed is not None:
            environment["PYTHONHASHSEED"] = seed
        print(f"启动 {label}，日志 {log.name}", flush=True)
        start = time.perf_counter()
        with log.open("x", encoding="utf-8") as stream:
            process = subprocess.Popen(command, cwd=ROOT, env=environment,
                                       stdout=stream, stderr=subprocess.STDOUT)
            summary["active"] = {"label": label, "pid": process.pid, "log": str(log)}
            save()
            exit_code = process.wait()
        record = json.loads(record_path.read_text(encoding="utf-8")) if record_path.is_file() else {}
        if record.get("status") == "running" or not record:
            record.update(status="crashed", tests_run=record.get("tests_run", 0))
        record.update(exit_code=exit_code, subprocess_seconds=round(time.perf_counter()-start, 6),
                      label=label, module=module, log=str(log), json=str(record_path))
        write_json(record_path, record)
        summary["runs"].append(record)
        summary.pop("active", None)
        save()
        print(f"完成 {label}：{record['status']}，{record.get('tests_run', 0)} 项，"
              f"{record.get('total_seconds', record['subprocess_seconds']):.3f} 秒", flush=True)
        guard("after_" + label)
        return record

    localized = set()
    try:
        guard("initial")
        archive_log = OUT / f"overnight_regression_{stamp}_archives_start.log"
        with archive_log.open("x", encoding="utf-8") as stream:
            archive = subprocess.run([sys.executable, "-X", "utf8", "tools/check_v13_archives.py"],
                                     cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        summary["archive_start"] = {"passed": archive.returncode == 0, "log": str(archive_log)}
        save()
        if archive.returncode:
            raise RuntimeError("归档或受保护文件校验不一致，停止所有后续步骤")
        for label, seed in (("default_1", None), ("default_2", None), ("seed0", "0")):
            record = run_one(label, seed)
            for module in failed_modules(record):
                if module not in localized:
                    localized.add(module)
                    run_one("localize_" + module.rsplit(".", 1)[1], seed, module)
            if seed == "0" and record["status"] == "crashed":
                retry = run_one("seed0_retry", "0")
                if retry["status"] == "crashed":
                    # FIX v1.3.2 permits two whole-suite attempts, then module localization.
                    for testfile in sorted((ROOT / "tests").glob("test*.py")):
                        module = "tests." + testfile.stem
                        if module in localized:
                            continue
                        localized.add(module)
                        located = run_one("crash_localize_" + testfile.stem, "0", module)
                        if located["status"] == "crashed":
                            summary["seed0_crash_module"] = module
                            break
        guard("final")
        summary["status"] = ("passed" if all(r["status"] == "passed" for r in summary["runs"])
                             else "completed_with_failures")
    except Exception as error:
        summary.update(status="stopped", reason=str(error), traceback=traceback.format_exc())
        traceback.print_exc()
    summary["finished"] = utc_now()
    save()
    lines = ["\n## 夜间步骤 1：完整回归复核\n", f"记录：`{summary_path.relative_to(ROOT)}`。状态：**{summary['status']}**。\n",
             "仅运行标准 unittest；核心代码、测试、预算均未修改，真实模型和仿真门控始终关闭。\n",
             "| 轮次 | 状态 | 测试数 | 跳过 | 总秒数 | 日志 |\n",
             "|---|---|---:|---:|---:|---|\n"]
    for run in summary["runs"]:
        lines.append(f"| {run['label']} | {run['status']} | {run.get('tests_run', 0)} | "
                     f"{run.get('skipped', '—')} | {run.get('total_seconds', run['subprocess_seconds']):.3f} | "
                     f"`{Path(run['log']).relative_to(ROOT)}` |\n")
        for task in run.get("support_tasks", []):
            lines.append(f"\n- {run['label']} / `{task['task']}`：{task['status']}，{task['seconds']:.6f} 秒。\n")
    if summary.get("reason"):
        lines.append(f"\n停止原因：{summary['reason']}。\n")
    lines.append("\n每个 JSON 另含逐测试方法计时、setUpClass 的三个底层规划任务计时、异常详情与前后环境检查；失败模块仅独立定位一次。\n")
    with (RESULTS / "overnight_report.md").open("a", encoding="utf-8") as stream:
        stream.write("".join(lines))
    print(f"步骤 1 已写入 overnight_report.md：{summary['status']}", flush=True)
    return 0 if summary["status"] == "passed" else 1


def main():
    parser = argparse.ArgumentParser(description="完整 unittest 串行回归与计时（不启动真实仿真）")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--json", help=argparse.SUPPRESS)
    parser.add_argument("--label", help=argparse.SUPPRESS)
    parser.add_argument("--module", help=argparse.SUPPRESS)
    args = parser.parse_args()
    return worker(args) if args.worker else supervise()


if __name__ == "__main__":
    raise SystemExit(main())
