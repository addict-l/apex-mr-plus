"""Matched fixed-query A/B/B/A idle dispatch diagnostic; never runs dynamics.

Launch under motion_v135_control's single-owner hard 1200s deadline. A captured
prefix and matched-query speedup are not a full TPG or assembly certificate.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.overnight_batch import atomic_json, utc_now
from tools.motion_v135_tpg_profile import CodecMetrics, ProfiledPlanner, contact_handles, port_free

BASE = ROOT / "results/sim_v1_3/motion_v135"
PHASES = ("A1", "B1", "B2", "A2")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def code_version():
    files = ("apexmr/sim_connection.py", "apexmr/motion.py", "apexmr/motion_backend.lua",
             "apexmr/geometry.py", "apexmr/tpg.py", "apexmr/PROVENANCE.json", "simbridge/workcell.py",
             "tools/motion_v135_tpg_profile.py", "tools/motion_v135_fast_idle.py")
    return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files}


def check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Fixed-tape diagnostic shared task deadline exhausted")


def delta(after, before):
    if isinstance(after, dict):
        return {key: delta(value, before.get(key, {}) if isinstance(value, dict) else before.get(key, 0))
                for key, value in after.items() if isinstance(value, (int, float, dict)) and not isinstance(value, bool)}
    return after-before


def replay_abba(instance, planner, tape, output, deadline, *, warmup=8, codec=None):
    """Replay identical requests; checkpoint each complete verified phase.

    The external simulator and planner are system boundaries. Every phase uses
    the same ordered tape; no adaptive sample count or time-limited prefix is
    compared. A interrupted phase's journal is preserved in a new attempt.
    """
    if not tape or warmup < 0:
        raise ValueError("A nonempty tape and nonnegative warmup are required")
    for row in tape:
        if not isinstance(row.get("samples"), list) or not row["samples"] or "expected" not in row:
            raise ValueError("Every tape entry needs complete samples and expected answer")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"tape_sha256": digest(tape), "code_sha256": code_version(),
                "phase_order": list(PHASES), "warmup_queries": warmup, "queries_per_phase": len(tape)}
    path = output/"manifest.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("Refusing to resume changed query/code/parameter identity")
    atomic_json(path, manifest)
    original = instance.stopped_scene_snapshot()
    phases = []
    for phase in PHASES:
        committed = output/(phase+".json")
        if committed.exists():
            previous = json.loads(committed.read_text(encoding="utf-8"))
            if previous.get("passed") is not True or previous.get("manifest_sha256") != digest(manifest):
                raise ValueError("Existing phase failed or has a different identity: "+phase)
            phases.append(previous)
            continue
        check(deadline)
        attempt = len(list(output.glob(phase+"_attempt_*.jsonl")))+1
        journal_path = output/f"{phase}_attempt_{attempt:02d}.jsonl"
        row = {"phase": phase, "started_at": utc_now(), "passed": False,
               "manifest_sha256": digest(manifest), "journal": str(journal_path),
               "sample_queries": 0, "sample_states": 0, "warmup_queries": warmup}
        context = instance.fast_idle_stopped(deadline=deadline) if phase.startswith("B") else nullcontext(None)
        idle_receipt = None
        try:
            with context as idle_receipt:
                before = instance.stopped_scene_snapshot()
                row["active_snapshot"] = before
                if phase.startswith("A") and before != original:
                    raise RuntimeError("Baseline app/physics state did not restore")
                for index in range(warmup):
                    check(deadline)
                    item = tape[index % len(tape)]
                    answer = planner.sample_state_pairs(deepcopy(item["samples"]), deepcopy(item.get("options")))
                    if answer != item["expected"]:
                        raise ValueError("Warmup answers differ from frozen tape")
                check(deadline)
                server_before = planner.sample_batch_statistics()
                client_before = deepcopy(getattr(planner, "stats", {}))
                codec_before = deepcopy(codec.stats) if codec is not None else {}
                began = time.perf_counter()
                with journal_path.open("x", encoding="utf-8") as journal:
                    for index, item in enumerate(tape):
                        check(deadline)
                        answer = planner.sample_state_pairs(deepcopy(item["samples"]), deepcopy(item.get("options")))
                        row["sample_queries"] += 1
                        row["sample_states"] += len(item["samples"])
                        record = {"index": index, "query_sha256": digest({"samples": item["samples"], "options": item.get("options")}),
                                  "answer_sha256": digest(answer), "matches": answer == item["expected"]}
                        journal.write(json.dumps(record, ensure_ascii=False)+"\n")
                        journal.flush()
                        if not record["matches"]:
                            raise ValueError("Sample answer differs from frozen tape at "+str(index))
                row["sample_loop_wall_s"] = time.perf_counter()-began
                row["sample_loop_scope"] = "Includes identical per-query hash/journal overhead; planner API and RPC are measured separately"
                row["client"] = delta(deepcopy(getattr(planner, "stats", {})), client_before)
                row["codec"] = delta(deepcopy(codec.stats), codec_before) if codec is not None else {}
                row["codec_available"] = bool(codec is not None and codec.stats.get("available"))
                check(deadline)
                server_after = planner.sample_batch_statistics()
                row["server"] = delta(server_after, server_before)
                row["active_after"] = instance.stopped_scene_snapshot()
                if row["active_after"] != before:
                    raise RuntimeError("Sample replay changed stopped app or physics state")
            row["idle_scope"] = idle_receipt
            row["restored_snapshot"] = instance.stopped_scene_snapshot()
            if row["restored_snapshot"] != original:
                raise RuntimeError("Phase did not restore exact original app/physics state")
            if row["server"].get("batches") != len(tape):
                raise RuntimeError("Server counters do not match fixed replay query count")
            if "samples_checked" in row["server"] and row["server"]["samples_checked"] != row["sample_states"]:
                raise RuntimeError("Server did not check all requested sample states")
            row.update(passed=True, finished_at=utc_now())
            atomic_json(committed, row)
            phases.append(row)
        except BaseException as exc:
            row.update(finished_at=utc_now(), reason=f"{type(exc).__name__}: {exc}",
                       idle_scope=getattr(exc, "receipt", idle_receipt))
            atomic_json(output/f"{phase}_attempt_{attempt:02d}.json", row)
            raise
    a = sum(row["sample_loop_wall_s"] for row in phases if row["phase"].startswith("A"))
    b = sum(row["sample_loop_wall_s"] for row in phases if row["phase"].startswith("B"))
    api_available = all("sample_method_s" in row["client"] for row in phases)
    api_a = sum(row["client"].get("sample_method_s", 0.) for row in phases if row["phase"].startswith("A"))
    api_b = sum(row["client"].get("sample_method_s", 0.) for row in phases if row["phase"].startswith("B"))
    report = {"passed": True, "phases": phases, "manifest": manifest,
              "matched_query_api_ratio_A_over_B": api_a/api_b if api_available and api_b else None,
              "matched_sample_loop_ratio_A_over_B": a/b if b else None,
              "A_drift_A2_over_A1": phases[3]["sample_loop_wall_s"]/phases[0]["sample_loop_wall_s"],
              "B_repeat_B2_over_B1": phases[2]["sample_loop_wall_s"]/phases[1]["sample_loop_wall_s"],
              "scope": "Fixed ordered stopped-scene query tape only; no full TPG, dynamics, or assembly certificate"}
    atomic_json(output/"abba.json", report)
    return report


class TapeCaptured(Exception):
    pass


class CapturePlanner:
    def __init__(self, planner, limit, journal, deadline):
        self.planner, self.limit, self.journal, self.deadline = planner, limit, journal, deadline
        self.tape = []

    def sample_state_pairs(self, samples, options=None):
        check(self.deadline)
        query = {"samples": deepcopy(samples), "options": deepcopy(options)}
        answer = self.planner.sample_state_pairs(samples, options)
        query["expected"] = deepcopy(answer)
        self.tape.append(query)
        self.journal.write(json.dumps(query, ensure_ascii=False, allow_nan=False)+"\n")
        self.journal.flush()
        if len(self.tape) == self.limit:
            raise TapeCaptured()
        return answer

    def __getattr__(self, name):
        return getattr(self.planner, name)


def run(case, reference, output, *, seconds=1200., batches=256, warmup=8):
    # All environment and live imports remain behind this explicit entry point.
    started = time.monotonic()
    if not 20 < seconds <= 1200 or batches <= 0 or warmup < 0:
        raise ValueError("Use20<seconds<=1200, positive fixed batches and nonnegative warmup")
    output, reference = Path(output).resolve(), Path(reference).resolve()
    if not output.is_relative_to(BASE) or not reference.is_relative_to(ROOT/"results/sim_v1_3"):
        raise ValueError("Diagnostic outputs/reference must stay in project result directories")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new case attempt; control skips already committed jobs")
    output.mkdir(parents=True, exist_ok=True)
    deadline = started+seconds-20.  # Shared hard wall includes this cleanup reserve.
    read = lambda path: json.loads(Path(path).read_text(encoding="utf-8"))
    file_sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    manifest = read(ROOT/"results/sim_v1_3/v134/manifests/sim20.json")
    job = next(job for job in manifest["jobs"] if job["id"] == "v134-fixed20-"+case)
    if seconds > float(job["timeout_seconds"]):
        raise ValueError("Diagnostic wall exceeds the unchanged original task budget")
    task = Path(job["task"])
    if file_sha(task) != job.get("prepared_task_sha256", job["source_task_sha256"]):
        raise ValueError("Original task SHA does not match formal manifest")
    old_workcell = read(reference/"workcell.json")
    if old_workcell.get("task") != task.stem.removesuffix(".qualified"):
        raise ValueError("Reference task identity differs from selected original task")
    report = {"case": case, "source_job_id": job["source_job_id"], "task": str(task),
              "task_sha256": file_sha(task), "reference": str(reference),
              "reference_tasks_sha256": file_sha(reference/"tasks.json"),
              "reference_workcell_sha256": file_sha(reference/"workcell.json"),
              "code_sha256": code_version(), "budget_seconds": seconds,
              "original_timeout_seconds": job["timeout_seconds"],
              "cleanup_reserve_s": 20., "batches": batches, "warmup": warmup,
              "started_at": utc_now(), "status": "running", "diagnostic_only": True,
              "formal_run": False, "argv": sys.argv}
    atomic_json(output/"progress.json", report)
    owner = planner = None
    try:
        from apexmr.geometry import SampledConflictOracle
        from apexmr.motion import MotionPlanner
        from apexmr.sim_connection import Simulator
        from apexmr.tpg import build_tpg
        from simbridge.runner import check_environment
        from simbridge.workcell import build
        check(deadline)
        check_environment(23210)
        owner = Simulator(output/"instance", port=23210, timeout=30.)
        owner.start(headless=True)
        report["pid"] = owner.process.pid
        client, sim = owner.connect()
        report["simulator_version"] = sim.getInt32Param(sim.intparam_program_version)
        installed = [owner.executable, owner.install/"lua/sim-1.lua", owner.install/"lua/sim-ce-x.lua",
                     owner.install/"addOns/ZMQ remote API server.lua",
                     owner.install/"models/robots/non-mobile/UR5.ttm"]
        installed.append(Path(sys.modules[type(client).__module__].__file__))
        report["installed_sha256"] = {str(path): file_sha(path) for path in installed}
        check(deadline)
        cfg = build(sim, client, task, owner.install, output/"workcell")
        workcell = read(output/"workcell/workcell.json")
        tasks, mapping = contact_handles(read(reference/"tasks.json"), old_workcell, workcell)
        report["contact_handle_mapping"] = mapping
        atomic_json(output/"tasks_replayed.json", tasks)
        planner = MotionPlanner(sim, client, cfg)
        profiled = ProfiledPlanner(planner, lambda: check(deadline), lambda: None)
        report["before"] = owner.stopped_scene_snapshot()
        report["stage"] = "capture_fixed_tape"
        atomic_json(output/"progress.json", report)
        with (output/"capture_queries.jsonl").open("x", encoding="utf-8") as journal:
            capture = CapturePlanner(profiled, batches, journal, deadline)
            oracle = SampledConflictOracle(capture, cfg, output/"capture_geometry")
            try:
                build_tpg(tasks, oracle, skip_ordered_pairs=True)
            except TapeCaptured:
                pass
        if len(capture.tape) != batches:
            raise RuntimeError("Reference supplied fewer queries than fixed tape length")
        atomic_json(output/"query_tape.json", capture.tape)
        report.update(stage="matched_abba", tape_sha256=digest(capture.tape), captured_queries=batches)
        atomic_json(output/"progress.json", report)
        with CodecMetrics.for_client(client) as codec:
            report["abba"] = replay_abba(owner, profiled, capture.tape, output/"replay", deadline,
                                          warmup=warmup, codec=codec)
        report.update(status="completed", after=owner.stopped_scene_snapshot())
        if report["after"] != report["before"]:
            raise RuntimeError("Final app/physics snapshot changed")
    except Exception as exc:
        report.update(status="timeout" if isinstance(exc, TimeoutError) else "failed",
                      reason=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    finally:
        report.update(elapsed_seconds=time.monotonic()-started, finished_at=utc_now())
        atomic_json(output/"profile.json", report)
        if planner is not None:
            try: planner.close()
            except Exception as exc: report["planner_cleanup_error"] = str(exc)
        if owner is not None:
            try: owner.close()
            except Exception as exc: report["instance_cleanup_error"] = str(exc)
            report["simulator_closed"] = owner.process is not None and owner.process.poll() is not None
            report["simulator_log"] = str(owner.log_path)
        report["port_23210_free"] = port_free()
        report["cleanup_passed"] = bool(report.get("simulator_closed") and report["port_23210_free"])
        report["elapsed_with_cleanup_s"] = time.monotonic()-started
        report["code_unchanged"] = report["code_sha256"] == code_version()
        if (not report["code_unchanged"] or not report["cleanup_passed"]
                or any(key.endswith("cleanup_error") for key in report)):
            report["status"] = "failed"
        atomic_json(output/"profile.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("00", "02"), required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=1200.)
    parser.add_argument("--batches", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=8)
    args = parser.parse_args()
    result = run(args.case, args.reference_dir, args.out, seconds=args.seconds,
                 batches=args.batches, warmup=args.warmup)
    print(json.dumps({key: result.get(key) for key in ("status", "reason", "elapsed_with_cleanup_s")}, ensure_ascii=False))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
