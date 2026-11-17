"""Serially close the physical phase and hand off to the durable workflow.

Importing this module never starts work. The CLI is an explicit runner; its
optional process wait observes PID plus birth identity and never kills it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import overnight_batch as old
from tools import v133_batch as batch
from tools import v133_workflow as workflow

BASE = batch.BASE
DEFAULT_REQUIRED_STAGE = "d_platform_required_final"
GATES_ALLOWANCE_SECONDS = 2*1200+60


def wait_for_process(path, *, recorder=print, birth=None, sleeper=None):
    """Read only the immutable process record and process identity while waiting."""
    record = old.read_json(old.confined(path))
    pid, expected = record.get("pid"), record.get("birth")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0 or expected is None:
        raise ValueError("等待进程记录必须包含有效的 pid 和 birth；未读取活动状态或终止进程")
    birth, sleeper = birth or old.process_birth, sleeper or time.sleep
    recorder(f"等待外部后台父进程退出：pid={pid}，按 birth 核实身份，不终止进程。")
    while birth(pid) == expected:
        sleeper(15.)
    recorder(f"外部后台父进程已退出或 PID 身份已变化：pid={pid}。")
    return {"pid": pid, "birth": expected, "exited": True}


def committed_stage(base, stage):
    """Inspect the frozen manifest and committed JSONL, never active state.json."""
    path = base/"manifests"/(stage+".json")
    if not path.is_file():
        return None
    try:
        manifest = old.load_manifest(path)
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise workflow.GlobalStop("阶段不可变输入/定义校验失败："+stage+" / "+str(error)) from error
    jobs = manifest.get("jobs", [])
    ids = [job.get("id") for job in jobs]
    if not ids or None in ids or len(set(ids)) != len(ids):
        raise RuntimeError("阶段清单缺少唯一任务 ID："+stage)
    for job in jobs:
        motion, planning = job.get("motion_code_sha256"), job.get("planning_code_sha256")
        real = job.get("kind") in {"qualify", "sim", "gate"} or job.get("phase") in {"sim", "gates"}
        if motion is not None and motion != batch.motion_fingerprint():
            raise workflow.GlobalStop("阶段运动代码指纹已改变，拒绝恢复旧定义："+stage)
        if planning is not None and planning != batch.planning_fingerprint():
            raise workflow.GlobalStop("阶段规划/标定指纹已改变，拒绝恢复旧定义："+stage)
        if real and not job.get("skip_reason"):
            if motion is None or (planning is None and job.get("kind") != "qualify"):
                raise workflow.GlobalStop("真实阶段缺少运动/规划定义指纹："+stage)
            try:
                batch.require_motion_gate(job)
            except RuntimeError as error:
                raise workflow.GlobalStop("真实阶段当前回归门无效："+stage+" / "+str(error)) from error
    directory = old.confined(manifest["state_dir"])
    try:
        records = old.Journal(directory).load()
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise workflow.GlobalStop("阶段提交日志校验失败："+stage+" / "+str(error)) from error
    for job in jobs:
        if job["id"] in records and records[job["id"]].get("definition_sha256") != job["definition_sha256"]:
            raise workflow.GlobalStop("同ID提交记录与当前任务定义SHA不一致："+job["id"])
    if not all(ident in records for ident in ids):
        return None
    stopped = next((records[ident].get("global_stop") for ident in ids if records[ident].get("global_stop")), None)
    return {"status": "global_stop" if stopped else "phase_complete", "global_stop": stopped,
            "requested": len(ids), "recorded": len(ids), "resumed_from_committed": True,
            "passed": all(records[ident].get("passed") is True for ident in ids)}


def run(*, base=BASE, runner=None, preparer=None, checker=None, reporter=None,
        recorder=None, now=None, late=None, required_stage=DEFAULT_REQUIRED_STAGE,
        wait_process_record=None):
    base = old.confined(base)
    if not required_stage or Path(required_stage).name != required_stage:
        raise ValueError("required_stage 必须是一个阶段名称")
    runner = runner or (lambda stage: batch.run_stage(stage, base=base))
    preparer = preparer or batch.prepare_native
    checker, reporter = checker or workflow.machine_check, reporter or workflow.report
    recorder, now = recorder or batch.append_report, now or time.time
    late = late or (lambda: workflow.run_pipeline(base=base))
    path = base/"physical_stage_outcome.json"

    # This lock covers waiting, physical jobs and the late-workflow handoff.
    with old.execution_lock(base/"physical_finish.lock"):
        if wait_process_record is not None:
            wait_for_process(wait_process_record, recorder=recorder)
        outcome = old.read_json(path) if path.is_file() else {"finished": False}
        result = {"status": "running", "required_stage": required_stage}

        def check():
            value = checker()
            if not value.get("passed"):
                raise workflow.GlobalStop(value.get("global_stop") or value.get("reason")
                                          or "归档/保护校验不一致或C盘低于5GiB")

        def stage(name):
            check()
            value = committed_stage(base, name) or runner(name)
            if value.get("global_stop") or value.get("status") == "global_stop":
                raise workflow.GlobalStop(str(value.get("global_stop") or "阶段整体停止："+name))
            if value.get("status") != "phase_complete":
                raise RuntimeError("阶段尚未完整提交，保留断点："+name+" / "+str(value.get("status")))
            # Injected runners may model completion without files. Whenever a
            # real manifest exists, committed JSONL is required before handoff.
            if (base/"manifests"/(name+".json")).exists() and committed_stage(base, name) is None:
                raise RuntimeError("阶段返回完成但提交日志仍缺任务："+name)
            reporter()
            return value

        try:
            if outcome.get("global_stop"):
                raise workflow.GlobalStop(str(outcome["global_stop"]))
            check()
            if outcome.get("finished") is True:
                # A resumed handoff must not bypass immutable identity checks
                # merely because an earlier source version marked D finished.
                committed_stage(base, required_stage)
                committed_stage(base, "d_gates")
            if outcome.get("finished") is not True:
                required = stage(required_stage)
                outcome["required_platform"] = {"stage": required_stage, **required}
                outcome["active_stage"] = "d_gates"
                old.atomic_json(path, outcome)
                recorder(f"专项 {required_stage} 全部提交；失败记录保留，不重试已提交失败任务。")
                existing = committed_stage(base, "d_gates")
                if existing is not None:
                    gates = stage("d_gates")
                elif outcome.get("d_gates", {}).get("status") == "skipped_physical_cap":
                    gates = outcome["d_gates"]
                else:
                    deadline = outcome.get("deadline_epoch")
                    if not isinstance(deadline, (int, float)) or isinstance(deadline, bool):
                        raise RuntimeError("缺少原 D 阶段 deadline_epoch，不创建或延长预算")
                    remaining = deadline-now()
                    if remaining < GATES_ALLOWANCE_SECONDS:
                        gates = {"status": "skipped_physical_cap", "remaining_seconds": remaining,
                                 "reason": "原D阶段剩余不足2×1200+60秒，跳过两项S3门控；不缩短每任务1200秒预算。"}
                    else:
                        check()
                        preparer("d_gates")
                        gates = stage("d_gates")
                check()
                outcome.update(d_gates=gates, finished=True, active_stage=None, updated_at=old.utc_now())
                old.atomic_json(path, outcome)
                recorder("D物理阶段已结束，保留原 fixtures 映射；接续后半程批量。")
                reporter()
            check()
            with old.execution_lock(base/"workflow.lock"):
                result = late()
            if result.get("global_stop") or result.get("status") == "global_stop":
                raise workflow.GlobalStop(str(result.get("global_stop") or result.get("reason") or "后续批量整体停止"))
        except workflow.GlobalStop as error:
            outcome.update(finished=False, global_stop=str(error), physical_finish_status="global_stop",
                           updated_at=old.utc_now())
            old.atomic_json(path, outcome)
            result = {"status": "global_stop", "global_stop": str(error), "reason": str(error)}
            recorder("整体停止，不启动后续独立工作："+str(error))
        except Exception as error:
            outcome.update(physical_finish_status="interrupted", updated_at=old.utc_now())
            old.atomic_json(path, outcome)
            result = {"status": "interrupted", "reason": f"{type(error).__name__}: {error}"}
            recorder("接续中断，保留提交记录和阶段时钟："+result["reason"])
        finally:
            old.atomic_json(base/"physical_finish_result.json", {**result, "updated_at": old.utc_now()})
            reporter()
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="串行等待专项结束、关闭D阶段，再接续v133批量；显式运行器")
    parser.add_argument("--required-stage", default=DEFAULT_REQUIRED_STAGE)
    parser.add_argument("--wait-process-record", type=Path)
    args = parser.parse_args(argv)
    result = run(required_stage=args.required_stage, wait_process_record=args.wait_process_record)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0 if result.get("status") == "finished" else 2


if __name__ == "__main__":
    raise SystemExit(main())
