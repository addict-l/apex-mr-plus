"""Separate v134 qualification, planning, and serial native supervisors."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import overnight_batch as old
from tools import v134_batch as batch
from tools import v134_guard as guard


def durable_event(base, phase, stage, kind, result, cursor_path):
    """Append progress/report independently of the cursor, preserving root data."""
    base = Path(base)
    compact = {key: result.get(key) for key in ("status", "reason", "global_stop", "passed", "requested", "recorded")}
    manifest_path = base / "manifests" / ((stage or "") + ".json")
    if stage and manifest_path.is_file():
        manifest = old.read_json(manifest_path)
        records = batch.read_records(manifest["state_dir"])
        selected = [records[j["id"]] for j in manifest.get("jobs", []) if j["id"] in records]
        from collections import Counter
        compact.update(requested=len(manifest.get("jobs", [])), recorded=len(selected),
            statuses=dict(Counter(r.get("status", "missing") for r in selected)),
            task_failures=sum(r.get("passed") is not True for r in selected))
    attempt = batch.active_attempt(base)
    identity = {"namespace": batch.NAMESPACE, "registration_attempt": attempt,
                "phase": phase, "stage": stage, "kind": kind, "result": compact}
    event_id = old.fingerprint(identity)
    event = {"step": "workflow_" + phase + "_" + (stage or "supervisor") + "_" + kind,
        "event_id": event_id, "experiment_namespace": batch.NAMESPACE, "event": kind,
        "phase": phase, "stage": stage, "status": result.get("status"), "at": old.utc_now(),
        "result": compact, "cursor": str(cursor_path),
        "registration_attempt": attempt,
        "journal": str(batch.state_root(base) / stage / "completed.jsonl") if stage else None}
    with old.execution_lock(base / "progress.lock"):
        path = base / "progress_state.json"
        progress = old.read_json(path) if path.exists() else {}
        steps = progress.setdefault("steps", [])
        if not isinstance(steps, list):
            raise ValueError("root progress.steps不是list；保留原文件，不覆盖")
        saved = next((row for row in steps if isinstance(row, dict) and row.get("event_id") == event_id), None)
        if saved is None:
            steps.append(event)
        else:
            event = saved
        if result.get("status") not in {"phase_complete", "finished"}:
            failures = progress.setdefault("failures", [])
            if isinstance(failures, list) and not any(isinstance(row, dict) and row.get("event_id") == event_id for row in failures):
                failures.append(event)
        progress["workflow_updated_at"] = old.utc_now()
        progress["workflow_latest"] = {"phase": phase, "stage": stage, "event": kind,
            "status": result.get("status"), "cursor": str(cursor_path), "registration_attempt": attempt}
        old.atomic_json(path, progress)
        report = base / "report.md"
        marker = "<!-- workflow-event:" + event_id + " -->"
        existing = report.read_text(encoding="utf-8") if report.exists() else ""
        if marker not in existing:
            title = "阶段结束" if kind == "stage_finished" else "监督器退出"
            text = (f"\n\n{marker}\n## 后台{title}：{phase} / {stage or 'supervisor'}\n\n"
                f"记录 {event['at']}；状态 `{result.get('status')}`。"
                f"已提交 {compact.get('recorded')} / {compact.get('requested')}；"
                f"各任务状态 {compact.get('statuses', {})}。阶段完结不等于验收通过，失败任务不会删除。\n\n"
                f"原因：{result.get('reason') or result.get('global_stop') or '无监督器异常'}。"
                f"证据：`{event['journal'] or cursor_path}`；cursor `{cursor_path}`。\n")
            with report.open("a", encoding="utf-8") as stream:
                stream.write(text); stream.flush()
                import os
                os.fsync(stream.fileno())
    return event


def planning_complete(base):
    return batch.planning_committed(base)


def run(*, base=batch.BASE, project=ROOT, phase="all", ready=None, preparer=None, runner=None, checker=None, reporter=None):
    base, project = batch.version_base(base, project)
    ready = ready or (lambda: batch.selection(base=base, project=project))
    runner = runner or (lambda stage: batch.run_stage(stage, base=base))
    if checker is None:
        protection = guard.Guard(base=base, project=project)
        checker = protection.check
    if reporter is None:
        from tools.v134_summary import rebuild
        reporter = lambda: rebuild(base=base, project=project)
    if preparer is None:
        functions = {"qualify5": batch.prepare_qualify5, "planning200": batch.prepare_planning,
                     "sim20": batch.prepare_fixed_sim20, "walls3": batch.prepare_walls}
        preparer = lambda stage: (functions[stage](base=base, project=project) if stage in functions
                                 else batch.prepare_gates(stage, base=base, project=project))
    stage_sets = {"qualification": ("motion_gate", "qualify5"), "planning": ("motion_gate", "qualify5", "planning200"),
                  "native": ("sim20", "walls3", "gates"),
                  "all": ("motion_gate", "qualify5", "planning200", "sim20", "walls3", "gates")}
    if phase not in stage_sets:
        raise ValueError("未知工作流阶段")
    path = batch.experiment_root(base) / ("workflow_" + phase + ".json")
    cursor = old.read_json(path) if path.exists() else {"status": "ready", "stages": {}, "experiment_namespace": batch.NAMESPACE}
    current_stage = None
    try:
        ready()
        if cursor.get("global_stop") or batch.stop_reason(base):
            raise RuntimeError(cursor.get("global_stop") or batch.stop_reason(base))
        cursor.update(status="running", registration_attempt=batch.active_attempt(base))
        old.atomic_json(path, cursor)
        for stage in stage_sets[phase]:
            current_stage = stage
            if stage in {"sim20", "walls3", "gates"} and not planning_complete(base):
                raise RuntimeError("planning200未全部提交，禁止启动native supervisor")
            checked = checker()
            if not checked.get("passed"):
                cursor["global_stop"] = "v134保护/旧归档校验失败"
                raise RuntimeError(cursor["global_stop"])
            preparer(stage)
            result = runner(stage)
            cursor["stages"][stage] = result
            old.atomic_json(path, cursor)
            durable_event(base, phase, stage, "stage_finished", result, path)
            reporter()
            if result.get("global_stop") or result.get("status") == "global_stop":
                cursor["global_stop"] = result.get("global_stop") or "阶段整体停止"
                raise RuntimeError(cursor["global_stop"])
            if result.get("status") != "phase_complete":
                raise RuntimeError("阶段未完整结束，保留断点：" + stage)
        cursor.update(status="finished", reason=None)
    except Exception as error:
        cursor.update(status="global_stop" if cursor.get("global_stop") else "interrupted", reason=str(error))
        if cursor.get("global_stop"):
            old.atomic_json(base / "global_stop.json", {"global_stop": cursor["global_stop"], "at": old.utc_now()})
    finally:
        cursor["updated_at"] = old.utc_now()
        old.atomic_json(path, cursor)
        durable_event(base, phase, current_stage, "supervisor_exit", cursor, path)
        reporter()
    return cursor


def main(argv=None):
    parser = argparse.ArgumentParser(description="v134分开启动的后台监督器；未完成planning不得起native")
    parser.add_argument("--phase", required=True, choices=("qualification", "planning", "native", "all"))
    args = parser.parse_args(argv)
    with old.execution_lock(batch.BASE / "workflow.lock"):
        result = run(phase=args.phase)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0 if result["status"] == "finished" else 2


if __name__ == "__main__":
    raise SystemExit(main())
