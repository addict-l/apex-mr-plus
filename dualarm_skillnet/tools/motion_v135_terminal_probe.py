"""Exact nominal final-home point diagnostics; never native execution proof.

The default CLI supervises an owned worker with a hard wall-clock limit.
``--worker`` is for an existing project OwnedProcess supervisor only. No
symbolic search, IK connection, RRT, TPG or dynamics is performed here.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import overnight_batch as batch
from tools import motion_v135_control as control

BASE = control.BASE


def build_nominal_terminal(problem, frames):
    """Construct one exact nominal goal arrangement without solving a plan.

    Required filled slots use catalog-compatible parts. Unconstrained parts
    retain their declared initial poses, which is one candidate only. Neither
    identity interchangeability nor point collision establishes robustness to
    physical goal tolerances, alternative unconstrained placements or paths.
    """
    from skillnet.core.logic import Fact

    initial = {fact.args[0]: (fact.predicate, fact.args[1]) for fact in problem.initial
               if fact.predicate in {"at", "installed"}}
    positions, required, required_slots, forbidden_slots, unresolved = dict(initial), set(), set(), set(), []
    fixed, owners = {}, {}
    supported = {"at", "installed", "filled", "free", "arm_at", "empty"}
    for condition in problem.goal:
        fact = condition.fact
        if fact.predicate not in supported:
            unresolved.append("unsupported_goal:"+str(condition))
            continue
        if fact.predicate == "filled":
            (forbidden_slots if condition.negated else required_slots).add(fact.args[0])
        elif not condition.negated and fact.predicate in {"at", "installed"}:
            part, place = fact.args
            value = (fact.predicate, place)
            if part in fixed and fixed[part] != value:
                unresolved.append("contradictory_part_goals:"+part)
            fixed[part] = value
            required.add(part)
            if fact.predicate == "installed":
                required_slots.add(place)
                if place in owners and owners[place] != part:
                    unresolved.append("multiple_required_slot_owners:"+place)
                owners[place] = part
    for part, (kind, place) in fixed.items():
        if (part not in frames.blocks or place not in frames.references
                or kind == "installed" and (place not in frames.slots
                    or frames.blocks[part]["type"] != frames.slots[place]["type"])):
            unresolved.append("incompatible_explicit_goal:"+part)
        else:
            positions[part] = (kind, place)
    for slot in sorted(required_slots):
        if slot in forbidden_slots or slot not in frames.slots:
            unresolved.append("contradictory_or_unknown_slot:"+slot)
            continue
        if slot in owners:
            continue
        candidates = [part for part in sorted(frames.blocks) if part not in fixed
                      and part not in owners.values()
                      and frames.blocks[part]["type"] == frames.slots[slot]["type"]]
        existing = [part for part in candidates if initial.get(part) == ("installed", slot)]
        if not candidates:
            unresolved.append("no_unassigned_compatible_part:"+slot)
            continue
        owners[slot] = (existing or candidates)[0]
    for slot, part in owners.items():
        positions[part] = ("installed", slot)
        required.add(part)
    for part, (kind, place) in positions.items():
        if kind == "installed" and place in forbidden_slots:
            unresolved.append("negative_slot_requires_unspecified_relocation:"+part)
    slot_parts = {}
    for part, (kind, place) in positions.items():
        if kind == "installed":
            slot_parts.setdefault(place, []).append(part)
    unresolved.extend("multiple_nominal_slot_occupants:"+slot
                      for slot, parts in slot_parts.items() if len(parts) != 1)
    missing = set(frames.blocks)-set(positions)
    unresolved.extend("no_declared_final_candidate_pose:"+part for part in sorted(missing))
    facts = set()
    for arm in problem.hooks.arms:
        facts.update((Fact("free", (arm,)), Fact("arm_at", (arm, "home_"+arm))))
    for part, (kind, place) in positions.items():
        facts.add(Fact(kind, (part, place)))
        if kind == "installed":
            facts.add(Fact("filled", (place,)))
    occupied = {place for kind, place in positions.values() if kind == "at"}
    facts.update(Fact("empty", (place,)) for place in frames.locations if place not in occupied)
    violation = problem.hooks.invariant(frozenset(facts))
    if violation:
        unresolved.append("nominal_terminal_invariant:"+violation)
    unmet = [str(condition) for condition in problem.goal
             if (condition.fact in facts) == condition.negated]
    unresolved.extend("unmet_goal:"+value for value in unmet)
    placed = {part: frames.block_pose(part, place) for part, (_, place) in positions.items()
              if part in frames.blocks and place in frames.references}
    groups = {}
    for slot, part in owners.items():
        if part not in fixed:
            group = groups.setdefault(frames.blocks[part]["type"], {"parts": [], "slots": []})
            group["parts"].append(part)
            group["slots"].append(slot)
    equivalence = [{"catalog_type": kind, **group,
        "identity_permutations": math.factorial(len(group["parts"])),
        "scope": "same catalog geometry at these exact required slot poses; selected parts only, with bearing aliases permuted consistently"}
        for kind, group in sorted(groups.items())]
    return {"status": "unknown" if unresolved else "constructed", "exact_nominal_goals": not unresolved,
        "goals": list(map(str, problem.goal)), "facts": sorted(map(str, facts)), "unresolved": unresolved,
        "placed": placed, "held": {}, "required_goal_parts": sorted(required),
        "unconstrained_parts": sorted(set(frames.blocks)-required),
        "assignments": [{"part": part, "kind": kind, "place": place,
                         "required": part in required, "explicit_identity": part in fixed}
                        for part, (kind, place) in sorted(positions.items())],
        "identity_equivalence": equivalence,
        "unconstrained_placement_is_one_candidate": True,
        "robust_to_goal_tolerances": False, "native_execution_passed": False,
        "scope": "One exact nominal terminal arrangement; no path, execution, force or tolerance-neighborhood certificate"}


def _collision_pair(evidence):
    if isinstance(evidence, dict):
        if isinstance(evidence.get("pair"), (list, tuple)) and len(evidence["pair"]) == 2:
            return list(evidence["pair"])
        for key in ("problem", "evidence", "last_answer"):
            if key in evidence:
                pair = _collision_pair(evidence[key])
                if pair:
                    return pair
    return None


def check_home_targets(planner, cfg, frames, terminal, *, deadline=None, progress_callback=None):
    """Use the production point validator at both fixed homes; no motion plan."""
    from simbridge.compiler import CompileFailure, MotionCompiler
    from simbridge.errors import is_transport_failure
    from simbridge.primitives import MotionRequest

    if not terminal.get("exact_nominal_goals"):
        return {"status": "unknown", "reason": "nominal_terminal_not_constructed",
                "home_checks": [], "native_execution_passed": False, "global_infeasibility_proven": False}
    compiler = MotionCompiler(planner, cfg, frames)
    compiler.deadline = deadline
    compiler.q = {arm: list(cfg["arms"][arm]["home_q"]) for arm in ("A", "B")}
    compiler.held, compiler.placed = {}, deepcopy(terminal["placed"])
    contacts = compiler._resting_contacts()
    checks = []
    def part_for(value):
        for part, handle in cfg.get("blocks", {}).items():
            if value == handle:
                return part
        if isinstance(value, str):
            alias = value.rsplit("/", 1)[-1]
            if alias in frames.blocks:
                return alias
        return None
    for arm in ("A", "B"):
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("末态目标点探针共享墙钟预算已耗尽")
        request = MotionRequest("terminal_probe_home_"+arm, arm, "transit",
            target_q=list(cfg["arms"][arm]["home_q"]), placed=deepcopy(compiler.placed),
            payloads={}, contacts=deepcopy(contacts), metadata={"source_action": "home", "final_home": True,
                "diagnostic_only": True, "exact_nominal_terminal": True})
        row = {"arm": arm, "q": deepcopy(compiler.q), "target_q": list(request.target_q),
               "held": {}, "placed": deepcopy(compiler.placed), "contacts": deepcopy(contacts),
               "exact_nominal_goals": True}
        if progress_callback is not None:
            progress_callback({"query": "home_target", "phase": "before", **deepcopy(row)})
        began = time.monotonic()
        try:
            compiler._validate(request, request.target_q)
            row.update(status="target_configuration_valid", pair=None)
        except CompileFailure as failure:
            if is_transport_failure(failure):
                raise
            pair = _collision_pair(failure.evidence)
            row.update(status="target_configuration_collision" if pair else "target_configuration_unknown",
                       detail=failure.detail, evidence=deepcopy(failure.evidence), pair=pair)
        row["elapsed_seconds"] = time.monotonic()-began
        parts = [part_for(value) for value in row.get("pair") or []]
        row["collision_part_ids"] = [part for part in parts if part is not None]
        row["collision_involves_required_goal_part"] = any(part in terminal["required_goal_parts"] for part in parts)
        row["conditional_on_unconstrained_placement"] = bool(row.get("pair")) and not row["collision_involves_required_goal_part"]
        row["scope"] = ("exact nominal required-goal geometry at fixed joint home; not all accepted goal tolerances"
                        if row["collision_involves_required_goal_part"] else
                        "this nominal terminal candidate, including unconstrained parts at their recorded original poses")
        checks.append(row)
        if progress_callback is not None:
            progress_callback({"query": "home_target", "phase": "after", **deepcopy(row)})
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("末态目标点探针共享墙钟预算已耗尽")
    return {"status": "nominal_target_checks_passed" if all(row["status"] == "target_configuration_valid" for row in checks)
            else "nominal_target_collision" if any(row["status"] == "target_configuration_collision" for row in checks) else "unknown",
        "home_checks": checks, "native_execution_passed": False, "global_infeasibility_proven": False,
        "robust_to_goal_tolerances": False, "scope": terminal["scope"]}


def confined_output(path):
    output = Path(path).resolve()
    if not output.is_relative_to(BASE.resolve()) or output == BASE.resolve():
        raise ValueError("探针所有产物必须在 motion_v135 独立尝试目录")
    return output


def task_identity(task):
    """Resolve unchanged fixed-cohort input and its original fixed home q."""
    task = batch.confined(task)
    actual = batch.sha256(task)
    candidates = []
    for cohort in ("sim20", "walls3"):
        manifest = ROOT/"results/sim_v1_3/v134/manifests"/(cohort+".json")
        for job in batch.read_json(manifest)["jobs"]:
            if Path(job["task"]).resolve() == task:
                candidates.append((cohort, manifest, job))
    if len(candidates) != 1:
        raise ValueError("输入必须唯一匹配原固定批次正式 manifest")
    cohort, manifest, job = candidates[0]
    if actual != job.get("prepared_task_sha256"):
        raise ValueError("原正式 manifest 的任务 SHA 不一致")
    sources = list((batch.confined(job["output"])/"simulation").glob("*/workcell.json"))
    if len(sources) != 1:
        raise ValueError("原正式场景缺少唯一 workcell/home 证据")
    original = batch.read_json(sources[0])
    homes = {arm: original["cfg"]["arms"][arm]["home_q"] for arm in ("A", "B")}
    if any(len(row) != 6 or any(isinstance(x, bool) or not isinstance(x, (int, float))
                               or not math.isfinite(x) for x in row) for row in homes.values()):
        raise ValueError("原 home 证据不是六个有限关节值")
    return {"task_file": str(task), "task_sha256": actual, "baseline_cohort": cohort,
        "baseline_job_id": job["id"], "source_job_id": job.get("source_job_id", job["id"]),
        "original_manifest": str(manifest), "original_manifest_sha256": batch.sha256(manifest),
        "original_task_budget_seconds": job["timeout_seconds"],
        "baseline_workcell": str(sources[0]), "baseline_workcell_sha256": batch.sha256(sources[0]),
        "home_q": deepcopy(homes), "home_source": "original formal workcell.cfg.arms.*.home_q"}


def run(task, out, seconds=90.):
    """Worker body; the CLI's project OwnedProcess enforces the hard deadline."""
    from apexmr.motion import MotionPlanner
    from apexmr.sim_connection import Simulator
    from simbridge.frames import Frames
    from simbridge.runner import check_environment
    from simbridge.workcell import build
    from skillnet.domains.blocks import load_problem

    output = confined_output(out)
    if not math.isfinite(seconds) or not 10 < seconds <= 1200:
        raise ValueError("诊断预算必须为 (10,1200] 秒；原正式预算不改变")
    if (output/"probe.json").exists() or (output/"progress.json").exists():
        raise FileExistsError("已有探针证据，保留它并使用新的尝试目录")
    output.mkdir(parents=True, exist_ok=True)
    started, phase_started, phase = time.monotonic(), time.monotonic(), "input"
    deadline = started+seconds-5.
    owner = planner = None
    report = {"status": "running", "diagnostic_only": True, "formal_run": False,
        "native_execution_passed": False, "global_infeasibility_proven": False,
        "hard_limit_seconds": seconds, "cleanup_reserve_seconds": 5., "phase_seconds": {},
        "scope": "Exact nominal terminal point checks only; no path, force, dynamics or all-tolerance infeasibility proof",
        "code_sha256": control.version(), "tool_sha256": batch.sha256(Path(__file__))}

    def enter(name):
        nonlocal phase, phase_started
        now = time.monotonic()
        report["phase_seconds"][phase] = report["phase_seconds"].get(phase, 0.)+now-phase_started
        phase, phase_started = name, now
        report.update(phase=phase, at=batch.utc_now(), elapsed_seconds=now-started)
        batch.atomic_json(output/"progress.json", report)
        if name != "cleanup" and now >= deadline:
            raise TimeoutError("末态点检查共享诊断预算已耗尽")

    try:
        identity = task_identity(task)
        report.update(identity)
        problem = load_problem(identity["task_file"], ROOT/"skills/blocks_v1.toml")
        frames = Frames(problem)
        terminal = build_nominal_terminal(problem, frames)
        batch.atomic_json(output/"nominal_terminal.json", terminal)
        if not terminal["exact_nominal_goals"]:
            report.update(status="unknown", reason="名义终态不能无歧义构造", terminal=terminal)
        else:
            check_environment(23210)
            enter("simulator_start")
            owner = Simulator(output/"instance", port=23210)
            owner.start(headless=True)
            report.update(simulator_pid=owner.process.pid, simulator_birth=batch.process_birth(owner.process.pid),
                          simulator_nonce=owner._instance)
            enter("simulator_connect")
            client, sim = owner.connect()
            enter("workcell")
            cfg = build(sim, client, identity["task_file"], owner.install, output/"workcell")
            report["rebuilt_selected_home_q"] = {arm: deepcopy(cfg["arms"][arm]["home_q"]) for arm in ("A", "B")}
            # This is the original fixed terminal joint target, even if a new
            # workcell bootstrap chose a numerically different initial branch.
            for arm in ("A", "B"):
                cfg["arms"][arm]["home_q"] = list(identity["home_q"][arm])
            enter("home_target_points")
            planner = MotionPlanner(sim, client, cfg)
            def save_point(event):
                import os
                with (output/"home_checks.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"at": batch.utc_now(), **event}, ensure_ascii=False, allow_nan=False)+"\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                report["latest_point"] = event
                batch.atomic_json(output/"progress.json", report)
            report.update(check_home_targets(planner, cfg, frames, terminal, deadline=deadline,
                                             progress_callback=save_point))
            report["terminal"] = terminal
            report["point_evidence_scope"] = "MotionCompiler._validate at original both-home q with original predicates/contacts and one exact nominal world"
    except BaseException as exc:
        from simbridge.errors import is_transport_failure
        report.update(status="timeout" if isinstance(exc, TimeoutError) else "error",
                      reason=f"{type(exc).__name__}: {exc}", remote_interface_failure=is_transport_failure(exc),
                      failure_phase=phase,
                      startup_failure_phase=phase if phase in {"simulator_start", "simulator_connect"} else None)
    finally:
        enter("cleanup")
        batch.atomic_json(output/"probe.json", report)
        if planner is not None:
            try:
                planner.close()
            except Exception as exc:
                report["planner_close_error"] = f"{type(exc).__name__}: {exc}"
        if owner is not None:
            try:
                owner.close()
            except Exception as exc:
                report["owner_close_error"] = f"{type(exc).__name__}: {exc}"
            report.update(simulator_closed=owner.process is not None and owner.process.poll() is not None,
                          simulator_log=str(owner.log_path))
        from tools.overnight_regression import health
        report["cleanup"] = health()
        report["code_unchanged_during_probe"] = (control.version() == report["code_sha256"]
            and batch.sha256(Path(__file__)) == report["tool_sha256"])
        if not report["code_unchanged_during_probe"]:
            report.update(status="invalid_version", reason="探针运行期间源码改变；保留证据但不作同版本诊断结论")
        report.update(at=batch.utc_now(), elapsed_seconds=time.monotonic()-started)
        report["phase_seconds"]["cleanup"] = time.monotonic()-phase_started
        batch.atomic_json(output/"probe.json", report)
        batch.atomic_json(output/"progress.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="原固定任务的精确名义末态点检查；不执行装配")
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=90.)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    output = confined_output(args.out)
    if not math.isfinite(args.seconds) or not 10 < args.seconds <= 1200:
        raise ValueError("诊断预算必须为 (10,1200] 秒")
    if args.worker:
        result = run(args.task, output, args.seconds)
        print(json.dumps({"诊断结论": result["status"], "装配通过": False, "证据": str(output/"probe.json")}, ensure_ascii=False))
        return 0 if result["status"] not in {"error", "timeout", "invalid_version"} and result["cleanup"]["passed"] else 2
    task_identity(args.task)
    command = [sys.executable, "-B", "-X", "utf8", str(Path(__file__).resolve()),
               "--worker", "--task", str(args.task.resolve()), "--out", str(output), "--seconds", str(args.seconds)]
    suffix = hashlib.sha256(str(output).encode()).hexdigest()[:16]
    result = control.run_job("terminal_probe_"+suffix, command, args.seconds, live=True)
    print(json.dumps({"诊断监督状态": result["status"], "装配通过": False, "证据": str(output/"probe.json")}, ensure_ascii=False))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
