"""Replay frozen reference tasks through the unchanged geometry/TPG pipeline.

No symbolic search or dynamics execution is performed. Invoke only from the
single-owner supervisor; its hard wall must also cover an unresponsive RPC.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager, ExitStack
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import socket
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.overnight_batch import atomic_json, utc_now
from tools.motion_v135_control import version as project_version

BASELINE = ROOT / "results/sim_v1_3/v134/experiments/c_wall_v134/runtime_fix2"
RESULTS = ROOT / "results/sim_v1_3/motion_v135"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def geometry_mode(owner, report, deadline, *, enabled=False):
    """Change only stopped-scene dispatch, retaining a restoration receipt."""
    report["fast_idle_enabled"] = bool(enabled)
    if not enabled:
        yield
        return
    try:
        with owner.fast_idle_stopped(deadline=deadline) as receipt:
            report["fast_idle_scope"] = receipt
            yield
    except BaseException as exc:
        if getattr(exc, "receipt", None) is not None:
            report["fast_idle_scope"] = exc.receipt
        raise
    if receipt.get("restored") is not True:
        raise RuntimeError("Stopped geometry profile idle restoration is unverified")


def port_free():
    try:
        with socket.create_connection(("127.0.0.1", 23210), timeout=.2):
            return False
    except (ConnectionError, OSError):
        return True


def contact_handles(tasks, old_report, new_report):
    """Map only contact-pair handles by unique object alias, never joint values."""
    old_objects = {row["handle"]: row["alias"] for row in old_report["objects"]}
    new_by_alias = {}
    for row in new_report["objects"]:
        new_by_alias.setdefault(row["alias"], []).append(row["handle"])
    remapped = deepcopy(tasks)
    receipts = {}

    def visit(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "contact_exclusions":
                    for pair in item:
                        for index, endpoint in enumerate(pair):
                            if isinstance(endpoint, int) and not isinstance(endpoint, bool):
                                alias = old_objects.get(endpoint)
                                choices = new_by_alias.get(alias, [])
                                if len(choices) != 1:
                                    raise ValueError(f"Cannot uniquely remap contact handle {endpoint}: {alias!r}")
                                pair[index] = choices[0]
                                receipts[str(endpoint)] = {"alias": alias, "new_handle": choices[0]}
                else:
                    visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(remapped)
    return remapped, receipts


class CodecMetrics:
    """Temporarily time the existing bundled client's CBOR functions in-process."""
    def __init__(self, codec):
        self.codec = codec
        self.originals = {}
        self.stats = {"available": codec is not None, "encode_calls": 0, "decode_calls": 0,
                      "encoded_bytes": 0, "decoded_bytes": 0,
                      "encode_seconds": 0., "decode_seconds": 0.,
                      "scope": "Actual existing CBOR dumps/loads only; request contents and codec options unchanged"}

    @classmethod
    def for_client(cls, client):
        module = sys.modules.get(type(client).__module__)
        codec = getattr(module, "cbor", None) if module is not None else None
        if codec is not None and not all(hasattr(codec, name) for name in ("dumps", "loads")):
            codec = None
        return cls(codec)

    def __enter__(self):
        if self.codec is None:
            return self
        if self.originals:
            raise RuntimeError("Codec profiler is already installed")
        for name, prefix in (("dumps", "encode"), ("loads", "decode")):
            original = getattr(self.codec, name)
            self.originals[name] = original
            def timed(*args, _original=original, _prefix=prefix, **kwargs):
                began = time.perf_counter()
                self.stats[_prefix+"_calls"] += 1
                if _prefix == "decode" and args:
                    self.stats["decoded_bytes"] += len(args[0])
                try:
                    result = _original(*args, **kwargs)
                    if _prefix == "encode":
                        self.stats["encoded_bytes"] += len(result)
                    return result
                finally:
                    self.stats[_prefix+"_seconds"] += time.perf_counter()-began
            setattr(self.codec, name, timed)
        return self

    def __exit__(self, *exc):
        for name, function in self.originals.items():
            setattr(self.codec, name, function)
        self.originals = {}
        return False


class ProfiledPlanner:
    """Measure normalization and actual RPC time without changing query contents."""
    def __init__(self, planner, check, save, *, legacy_reset=False):
        self._planner, self._check, self._save = planner, check, save
        self.legacy_reset = bool(legacy_reset)
        self.stats = {"sample_calls": 0, "samples": 0, "sample_method_s": 0.,
                      "rpc_calls": 0, "rpc_roundtrip_including_codec_s": 0.,
                      "rpc_methods": {}, "sample_batch_sizes": {},
                      "python_sample_normalization_s": 0.}
        self._original_call = planner._call
        planner._call = self._call

    def _call(self, function, *args):
        self._check()
        started = time.perf_counter()
        try:
            return self._original_call(function, *args)
        finally:
            elapsed = time.perf_counter() - started
            self.stats["rpc_calls"] += 1
            self.stats["rpc_roundtrip_including_codec_s"] += elapsed
            self.stats["rpc_methods"][function] = self.stats["rpc_methods"].get(function, 0) + 1

    def sample_state_pairs(self, samples, options=None):
        self._check()
        started = time.perf_counter()
        rpc_before = self.stats["rpc_roundtrip_including_codec_s"]
        self.stats["sample_calls"] += 1
        self.stats["samples"] += len(samples)
        key = str(len(samples))
        self.stats["sample_batch_sizes"][key] = self.stats["sample_batch_sizes"].get(key, 0) + 1
        try:
            if self.legacy_reset:
                options = {**(options or {}), "legacy_batch_reset": True}
            return self._planner.sample_state_pairs(samples, options)
        finally:
            elapsed = time.perf_counter() - started
            rpc = self.stats["rpc_roundtrip_including_codec_s"] - rpc_before
            self.stats["sample_method_s"] += elapsed
            self.stats["python_sample_normalization_s"] += max(0., elapsed-rpc)
            self._save()

    def __getattr__(self, name):
        return getattr(self._planner, name)


def run(case, out, seconds, until="graph", *, legacy_reset=False, fast_idle=False,
        controlled_job_result=None):
    from apexmr.geometry import SampledConflictOracle, verify_rollout
    from apexmr.motion import MotionPlanner
    from apexmr.program import parameterize
    from apexmr.sim_connection import Simulator
    from apexmr.tpg import build_tpg
    from simbridge.runner import check_environment
    from simbridge.workcell import build

    output = Path(out).resolve()
    if not output.is_relative_to(RESULTS):
        raise ValueError("诊断输出必须位于 results/sim_v1_3/motion_v135/ 内")
    if seconds <= 0:
        raise ValueError("诊断预算必须大于零")
    if (output/"profile.json").exists() or (output/"progress.json").exists():
        raise FileExistsError("诊断目录已有结果；请使用新目录保留前次尝试")
    check_environment(23210)
    if not port_free():
        raise RuntimeError("23210端口被占用；未接管已有实例")
    output.mkdir(parents=True, exist_ok=True)
    sources = list((BASELINE/"sim20"/case/"simulation").iterdir())
    if len(sources) != 1:
        raise ValueError("基线任务目录不是唯一目录")
    source = sources[0]
    old_workcell = read(source/"workcell.json")
    original_tasks = read(source/"tasks.json")
    manifest = read(ROOT/"results/sim_v1_3/v134/manifests/sim20.json")
    job = next(job for job in manifest["jobs"] if job["id"] == "v134-fixed20-"+case)
    task = Path(job["task"])
    task_sha = sha256(task)
    expected = job.get("prepared_task_sha256", job.get("task_sha256"))
    if expected is not None and task_sha != expected:
        raise ValueError("基线任务SHA不匹配")
    started = time.monotonic()
    deadline = started + seconds
    stage = "start"
    stage_started = started
    timings = {}
    owner = None
    planner = None
    profiled = None
    oracle = None
    graph = None
    codec_metrics = None
    geometry_resources = ExitStack()
    last_server_statistics = 0.
    last_save = 0.
    report = {"case": case, "task": str(task), "task_sha256": task_sha,
              "source_reference": str(source), "source_tasks_sha256": sha256(source/"tasks.json"),
              "source_workcell_sha256": sha256(source/"workcell.json"),
              "diagnostic_only": True, "formal_run": False, "budget_seconds": seconds,
              "until": until, "skip_ordered_pairs": True,
              "implementation_variant": "forced_legacy_reset" if legacy_reset else "exact_world_reuse",
              "fast_idle_enabled": bool(fast_idle), "profile_tool_sha256": sha256(Path(__file__)),
              "project_code_sha256": project_version(),
              "controlled_job_result": str(Path(controlled_job_result).resolve()) if controlled_job_result else None,
              "joint_sample_step_rad": .07, "rollout_step_s": .05,
              "status": "running", "stage": stage,
              "stages_not_executed": ["dynamics_execution"],
              "timing_scope": "RPC timings include codec/transport/server; no claim of isolated serialization time",
              "code_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in
                               [ROOT/"apexmr"/name for name in ("geometry.py", "tpg.py", "program.py", "motion.py", "motion_backend.lua", "sim_connection.py", "PROVENANCE.json")]}}

    def check():
        if time.monotonic() >= deadline:
            raise TimeoutError("TPG诊断共享墙钟预算已耗尽，阶段："+stage)

    def save(force=False):
        nonlocal last_save, last_server_statistics
        now = time.monotonic()
        if force or now-last_save >= 1.:
            # This is an explicitly budgeted RPC. Never attempt it at/after the
            # soft deadline merely to fill a missing final metric snapshot.
            if planner is not None and profiled is not None and now-last_server_statistics >= 5. and deadline-now > .25:
                check()
                last_server_statistics = now
                stats_started = time.monotonic()
                try:
                    report["lua_sample_batch_statistics"] = planner.sample_batch_statistics()
                    report["lua_statistics_snapshot_elapsed_s"] = time.monotonic()-started
                except Exception as exc:
                    report["lua_statistics_query_error"] = f"{type(exc).__name__}: {exc}"
                report["lua_statistics_query_seconds"] = report.get("lua_statistics_query_seconds", 0.)+time.monotonic()-stats_started
                report["lua_statistics_queries"] = report.get("lua_statistics_queries", 0)+1
                check()
                now = time.monotonic()
            if codec_metrics is not None:
                report["codec"] = deepcopy(codec_metrics.stats)
            report.update(stage=stage, elapsed_seconds=now-started,
                          stage_seconds={**timings, stage: now-stage_started})
            if profiled is not None:
                report["rpc"] = deepcopy(profiled.stats)
            if oracle is not None:
                report["geometry"] = deepcopy(oracle.stats)
                report["cache_entries"] = len(oracle.cache)
            atomic_json(output/"progress.json", report)
            last_save = now

    def enter(name):
        nonlocal stage, stage_started
        now = time.monotonic()
        timings[stage] = timings.get(stage, 0.)+now-stage_started
        stage, stage_started = name, now
        save(True)
        check()

    try:
        save(True)
        owner = Simulator(output/"instance", port=23210)
        enter("simulator_start")
        owner.start(headless=True)
        report["simulator_pid"] = owner.process.pid
        enter("simulator_connect")
        client, sim = owner.connect()
        enter("workcell")
        cfg = build(sim, client, task, owner.install, output/"workcell")
        new_workcell = read(output/"workcell/workcell.json")
        tasks, remapping = contact_handles(original_tasks, old_workcell, new_workcell)
        report["contact_handle_remapping"] = remapping
        report["reference_task_count"] = len(tasks)
        report["reference_task_kinds"] = dict(Counter(t["kind"] for t in tasks))
        atomic_json(output/"tasks_replayed.json", tasks)
        enter("motion_backend_setup")
        planner = MotionPlanner(sim, client, cfg)
        profiled = ProfiledPlanner(planner, check, save, legacy_reset=legacy_reset)
        codec_metrics = CodecMetrics.for_client(client)
        codec_metrics.__enter__()
        report["backend_scope"] = deepcopy(planner.info)
        geometry_resources.enter_context(geometry_mode(owner, report, deadline, enabled=fast_idle))
        enter("structure_preview")
        skeleton = build_tpg(tasks, None, skip_ordered_pairs=True)
        report["structural_graph_counts"] = deepcopy(skeleton["counts"])
        report["structural_preview_scope"] = "Counts only: no collision certificate; not used for acceptance or execution"
        del skeleton
        enter("tpg_build")
        report["geometry_mode_counts"] = {}
        oracle = SampledConflictOracle(profiled, cfg, output/"geometry", step=.07)

        def bounded_oracle(first, second):
            check()
            answer = oracle(first, second)
            evidence = answer.get("evidence", {})
            mode = evidence.get("mode_fingerprint")
            if mode is not None:
                counts = report["geometry_mode_counts"]
                counts[mode] = counts.get(mode, 0) + 1
                report["geometry_mode_count"] = len(counts)
            report["last_oracle_pair"] = {"a": first["id"], "b": second["id"],
                                          "reference_indices": [first["reference_index"], second["reference_index"]]}
            save()
            check()
            return answer

        graph = build_tpg(tasks, bounded_oracle, skip_ordered_pairs=True)
        report["graph_counts"] = deepcopy(graph["counts"])
        report["graph_certificate"] = deepcopy(graph["certificate"])
        check()
        atomic_json(output/"tpg_unparameterized.json", graph)
        if until == "rollout":
            enter("parameterization")
            parameterize(graph, vmax=.60, amax=1.20, jmax=8.)
            check()
            atomic_json(output/"tpg.json", graph)
            enter("rollout")
            rollout = verify_rollout(graph, profiled, cfg, step_s=.05, outdir=output/"geometry")
            report["rollout"] = rollout
            check()
            report["status"] = "completed" if rollout["valid"] else "rollout_failed"
        else:
            report["stages_not_executed"] += ["parameterization", "rollout"]
            report["status"] = "completed"
    except TimeoutError as exc:
        report.update(status="timeout", reason=str(exc))
    except Exception as exc:
        report.update(status="failed", reason=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    finally:
        finished = time.monotonic()
        timings[stage] = timings.get(stage, 0.)+finished-stage_started
        restoring = finished
        try:
            geometry_resources.close()
        except BaseException as exc:
            report.update(status="failed", restoration_error=f"{type(exc).__name__}: {exc}")
            report.setdefault("reason", report["restoration_error"])
        if fast_idle:
            timings["geometry_context_restore"] = time.monotonic()-restoring
        report["stage_seconds"] = timings
        report["elapsed_seconds"] = time.monotonic()-started
        report["finished_at"] = utc_now()
        if profiled is not None:
            report["rpc"] = deepcopy(profiled.stats)
        if oracle is not None:
            report["geometry"] = deepcopy(oracle.stats)
            report["cache_entries"] = len(oracle.cache)
            try:
                oracle.save_stats()
            except OSError as exc:
                report["statistics_save_error"] = str(exc)
        if codec_metrics is not None:
            report["codec"] = deepcopy(codec_metrics.stats)
            codec_metrics.__exit__(None, None, None)
            report["codec_restored"] = True
        if report.get("lua_statistics_snapshot_elapsed_s") is not None:
            report["lua_statistics_snapshot_age_s"] = max(0., report["elapsed_seconds"]-report["lua_statistics_snapshot_elapsed_s"])
        # Persist the diagnostic before cleanup so the supervisor can recover it.
        atomic_json(output/"profile.json", report)
        if planner is not None:
            try:
                if profiled is not None:
                    planner._call = profiled._original_call
                planner.close()
            except Exception as exc:
                report["planner_close_error"] = f"{type(exc).__name__}: {exc}"
        if owner is not None:
            try:
                owner.close()
            except Exception as exc:
                report["owner_close_error"] = f"{type(exc).__name__}: {exc}"
            report["simulator_closed"] = owner.process is not None and owner.process.poll() is not None
            report["simulator_log"] = str(owner.log_path)
        report["port_23210_free"] = port_free()
        report["cleanup_passed"] = bool(report.get("simulator_closed") and report["port_23210_free"])
        report["total_wall_seconds_including_cleanup"] = time.monotonic()-started
        atomic_json(output/"profile.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="原参考轨迹的TPG诊断；不做符号搜索或动力学执行")
    parser.add_argument("--case", choices=("00", "02"), required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--seconds", type=float, default=120.)
    parser.add_argument("--until", choices=("graph", "rollout"), default="graph")
    parser.add_argument("--legacy-sample-reset", action="store_true", help="诊断对照：逐样本恢复旧世界状态，所有碰撞查询保持不变")
    parser.add_argument("--fast-idle", action="store_true", help="仅停止场景几何阶段启用并恢复自有实例快速调度")
    parser.add_argument("--controlled-job-result", type=Path, help="外层受控作业结果路径，供正式M5前置审计")
    args = parser.parse_args(argv)
    output = args.out or RESULTS/"m1"/("tpg_profile_"+args.case)
    result = run(args.case, output, args.seconds, args.until,
                 legacy_reset=args.legacy_sample_reset, fast_idle=args.fast_idle,
                 controlled_job_result=args.controlled_job_result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "completed" and result.get("cleanup_passed") else (124 if result["status"] == "timeout" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
