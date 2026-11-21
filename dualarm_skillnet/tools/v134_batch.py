"""Explicit, resumable v134 batches; imports never prepare or run experiments."""
from collections import Counter
from copy import deepcopy
import argparse
import csv
import io
import json
import math
from pathlib import Path
import re
import sys
import time
import tomllib
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import overnight_batch as old
from tools import v134_guard

BASE = ROOT / "results/sim_v1_3/v134"
NAMESPACE = "c_wall_v134"
FAMILIES = ("F1", "F2", "F3", "F4", "F5")
D_TASKS = tuple(f"gen_f3_s20261115_{i:03d}" for i in (5, 12, 13, 15, 18))
V11_MODULES = ("tests.test_v11_search", "tests.test_v11_admissibility", "tests.test_v11_ordering")
FINGERPRINT_FILES = ("skillnet/planning/search.py", "skillnet/planning/heuristic.py",
    "skillnet/planning/bounds.py", "skillnet/planning/allocation.py", "skillnet/pipeline.py",
    "simbridge/backend.py", "simbridge/qualifier.py", "simbridge/__main__.py", "simbridge/taskio.py",
    "skillnet/cost/model.py", "skills/blocks_v1.toml")
MOTION_FILES = ("compiler.py", "frames.py", "primitives.py", "backend.py", "workcell.py",
    "runner.py", "qualifier.py", "controller_ext.lua", "executor.py")
sha256 = old.sha256


def version_base(base=BASE, project=ROOT):
    base, project = Path(base).resolve(), Path(project).resolve()
    if base != project / "results/sim_v1_3/v134":
        raise ValueError("批次命名空间必须是独立v134目录")
    return base, project


def output_path(path, base):
    result = Path(path).resolve()
    if not result.is_relative_to(Path(base).resolve()):
        raise ValueError("输出/证据必须在v134新命名空间，禁止写v133")
    return result


def planning_fingerprint(*, project=ROOT):
    project = Path(project)
    result = {name: sha256(project / name) for name in FINGERPRINT_FILES}
    result["calibration.json"] = sha256(project / "results/sim_v1_3/v133/calibration.json")
    return result


def motion_fingerprint(*, project=ROOT):
    return {name: sha256(Path(project) / "simbridge" / name) for name in MOTION_FILES}


def _task_name(path):
    return tomllib.loads(Path(path).read_text(encoding="utf-8-sig"))["name"]


def read_records(directory):
    """Read committed JSONL without constructing a directory or recovery file."""
    directory = Path(directory)
    paths = ([directory / "completed.jsonl"] if (directory / "completed.jsonl").is_file() else [])
    paths += sorted(directory.glob("completed.recovery_*.jsonl"))
    records = {}
    for path in paths:
        for line in path.read_bytes().splitlines(keepends=True):
            if not line.endswith(b"\n"):
                continue
            row = json.loads(line)
            if row["id"] in records and records[row["id"]] != row:
                raise ValueError("冲突的提交记录：" + row["id"])
            records[row["id"]] = row
    return records


def _proof(path, kind, expected, project):
    path = Path(path).resolve()
    source = Path(project) / "results/sim_v1_3/v133"
    if not path.is_relative_to(Path(project).resolve()) or path.is_relative_to(source) or not path.is_file():
        raise ValueError("新namespace回归证明缺失或引用旧v133：" + str(path))
    data = old.read_json(path)
    if data.get("experiment_namespace") != NAMESPACE:
        raise ValueError("回归证明缺少当前experiment namespace命名空间")
    if not {"B", "A", "C", "D"} <= set(data.get("fixes_completed", [])):
        raise ValueError("回归证明必须在B/A/C/D四项修复后生成")
    if data.get("planning_code_sha256", data.get("code_sha256")) != expected:
        raise ValueError("回归证明源码指纹不匹配")
    if kind == "full":
        if data.get("status") != "passed" or data.get("tests_run", 0) <= 0:
            raise ValueError("四修复后的完整默认集合未通过")
        for module in V11_MODULES:
            tests = [r for r in data.get("per_test", []) if r.get("id", "").startswith(module + ".")]
            if not tests or not all(r.get("status") == "passed" for r in tests):
                raise ValueError("v1.1逐测证明缺失或失败：" + module)
    elif data.get("passed") is not True or len(data.get("rows", [])) != 11 or not all(r.get("passed") is True for r in data["rows"]):
        raise ValueError("11原任务makespan/bound独立对照未通过")
    return {"path": str(path), "sha256": sha256(path)}


def active_attempt(base=BASE):
    path = Path(base) / "experiment_selection.json"
    attempt = old.read_json(path).get("registration_attempt") if path.is_file() else None
    if attempt is not None and (not isinstance(attempt, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", attempt)):
        raise ValueError("登记attempt必须是安全的单个目录名")
    return attempt


def experiment_root(base=BASE):
    root = Path(base) / "experiments" / NAMESPACE
    attempt = active_attempt(base)
    return root / attempt if attempt else root


def state_root(base=BASE):
    root = Path(base) / "states"
    attempt = active_attempt(base)
    return root / attempt if attempt else root


def _assert_inactive(base):
    """Called under both supervisor and project execution locks."""
    root = Path(base) / "experiments" / NAMESPACE
    for path in root.rglob("workflow_*.json"):
        if old.read_json(path).get("status") not in {"finished", "interrupted", "global_stop"}:
            raise ValueError("旧workflow尚非终态，禁止重登记："+str(path))
    for path in root.rglob("process_*.json"):
        row = old.read_json(path)
        if row.get("pid"):
            if not row.get("birth"):
                raise ValueError("旧process记录缺birth，无法核实活动状态："+str(path))
            if old.process_birth(row["pid"]) == row["birth"]:
                raise ValueError("旧workflow进程仍活动，禁止重登记："+str(path))
    for path in (Path(base) / "states").rglob("state.json"):
        row = old.read_json(path)
        if row.get("status") == "running" or row.get("active_job"):
            raise ValueError("阶段仍running或有未清理active_job："+str(path))
    stopped = Path(base) / "global_stop.json"
    if stopped.is_file() and old.read_json(stopped).get("global_stop"):
        raise ValueError("真实global_stop尚未解除；重登记不能隐式清除")


def _archive_registration(base, previous_sha, attempt, previous):
    """Copy exact controls before removing active pointers; native paths stay put.

    The immutable index makes an interrupted copy/removal safe to resume. No
    directory is moved, and every unlink is a verified active control file.
    """
    base = Path(base).resolve()
    history = base / "registration_history" / previous_sha
    index_path = history / "index.json"
    if index_path.is_file():
        index = old.read_json(index_path)
        if index.get("previous_selection_sha256") != previous_sha or index.get("new_attempt") != attempt:
            raise ValueError("既有重登记归档属于不同attempt")
    else:
        paths = [base / "experiment_selection.json", *sorted((base / "manifests").glob("*.json"))]
        paths += [p for p in (base / "last_motion_gate.json", base / "qualification_index.json") if p.is_file()]
        retained = set(previous.get("regressions", {}).values())
        for path in paths:
            if path.parent == base / "manifests":
                manifest = old.read_json(path)
                if manifest.get("state_dir"):
                    retained.add(manifest["state_dir"])
                retained.update(j["output"] for j in manifest.get("jobs", []) if j.get("output"))
        cursors = sorted((base / "experiments" / NAMESPACE).rglob("workflow_*.json"))
        processes = sorted((base / "experiments" / NAMESPACE).rglob("process_*.json"))
        paths += cursors + processes
        retained.update(str(p) for p in cursors + processes)
        entries = []
        for path in paths:
            path = path.resolve()
            if not path.is_relative_to(base):
                raise ValueError("归档控制路径超出v134")
            target = history / path.relative_to(base)
            data = path.read_bytes()
            if target.exists() and target.read_bytes() != data:
                raise ValueError("既有归档字节不一致："+str(target))
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_bytes(data)
            entries.append({"original": str(path), "archived": str(target), "sha256": sha256(path),
                "remove_active": path.parent == base / "manifests" or path.name in {"last_motion_gate.json", "qualification_index.json"}})
        index = {"previous_selection_sha256": previous_sha, "new_attempt": attempt,
            "path_mapping": entries, "retained_original_paths": sorted(retained),
            "new_experiment_root": str(base / "experiments" / NAMESPACE / attempt),
            "new_state_root": str(base / "states" / attempt)}
        old.atomic_json(index_path, index)
    # Validate the entire copy before unlinking any active pointer. A resumed
    # transaction accepts absent pointers only because their exact copy exists.
    for entry in index["path_mapping"]:
        source, target = Path(entry["original"]).resolve(), Path(entry["archived"]).resolve()
        if not source.is_relative_to(base) or not target.is_relative_to(history):
            raise ValueError("归档索引路径越界")
        if target != history / source.relative_to(base):
            raise ValueError("归档路径映射不一致")
        if entry["remove_active"] and not (source.parent == base / "manifests" or
                source in {base / "last_motion_gate.json", base / "qualification_index.json"}):
            raise ValueError("禁止移除旧原生输出或state；只能切换active控制指针")
        if not target.is_file() or sha256(target) != entry["sha256"]:
            raise ValueError("归档字节校验失败")
        if source.exists() and sha256(source) != entry["sha256"]:
            raise ValueError("归档后的原始控制文件已变化")
    for entry in index["path_mapping"]:
        if entry["remove_active"]:
            source = Path(entry["original"])
            if source.exists():
                source.unlink()
    return index_path


def _register(full, original11, *, base=BASE, project=ROOT, attempt=None, replace_previous_sha256=None):
    """Explicit after-four-fix registration; never called on import/preparation."""
    base, project = version_base(base, project)
    fingerprint = planning_fingerprint(project=project)
    proofs = {kind: _proof(path, kind, fingerprint, project) for kind, path in (("full", full), ("original11", original11))}
    source = project / "results/sim_v1_3/v133"
    old_selection = old.read_json(source / "experiment_selection.json")
    stage = old_selection.get("stages", {}).get("planning200", "planning200")
    manifest_path = source / "manifests" / (stage + ".json")
    manifest = old.read_json(manifest_path)
    inputs = []
    for job in manifest["jobs"]:
        if job.get("table") != "qualified":
            continue
        path = Path(job["task"]).resolve()
        if not path.is_relative_to(source):
            raise ValueError("资格来源必须是已冻结的v133 IK表")
        inputs.append({"family": job["family"], "index": job["index"], "name": _task_name(path),
            "task": str(path), "sha256": sha256(path), "source_job_id": job["id"]})
    if len(inputs) != 200 or {(r["family"], r["index"]) for r in inputs} != {(f, i) for f in FAMILIES for i in range(40)}:
        raise ValueError("旧资格来源必须五族各40、恰200单元")
    sim_manifest = source / "manifests/sim20.json"
    fixed = old.read_json(sim_manifest).get("jobs", [])
    if len(fixed) != 20 or Counter(r["family"] for r in fixed) != Counter({f: 4 for f in FAMILIES}):
        raise ValueError("固定sim20来源必须为旧清单五族各4")
    baseline_path = base / "baseline.json"
    if not baseline_path.is_file():
        raise ValueError("v134保护baseline缺失")
    value = {"schema_version": 1, "selection_id": NAMESPACE, "experiment_namespace": NAMESPACE,
        "planning_code_sha256": fingerprint, "regressions": {k: v["path"] for k, v in proofs.items()},
        "regression_sha256": {k: v["sha256"] for k, v in proofs.items()}, "fixes_completed": ["B", "A", "C", "D"],
        "baseline_sha256": sha256(baseline_path), "source_planning_manifest": str(manifest_path),
        "source_planning_manifest_sha256": sha256(manifest_path), "source_sim20_manifest": str(sim_manifest),
        "source_sim20_manifest_sha256": sha256(sim_manifest),
        "qualification_inputs": sorted(inputs, key=lambda r: (r["family"], r["index"])),
        "d_tasks": list(D_TASKS), "policy": "IK200 only; D5 new copies; fixed20 plus walls001/009/037; v133 read-only"}
    path = base / "experiment_selection.json"
    if attempt is None:
        if replace_previous_sha256 is not None:
            raise ValueError("替换SHA必须与显式attempt一起提供")
        if path.exists():
            if old.read_json(path) != value:
                raise ValueError("既有v134登记/回归/输入SHA不同，不得覆盖")
            return path
    else:
        if not isinstance(attempt, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", attempt):
            raise ValueError("attempt必须是安全的单个目录名")
        if not isinstance(replace_previous_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", replace_previous_sha256):
            raise ValueError("显式attempt需要精确旧selection SHA256")
        if not path.is_file():
            raise ValueError("没有可替换的旧selection")
        value.update(registration_attempt=attempt, previous_selection_sha256=replace_previous_sha256,
                     registration_history=str(base / "registration_history" / replace_previous_sha256 / "index.json"))
        previous = old.read_json(path)
        if previous == value:
            return path
        if sha256(path) != replace_previous_sha256:
            raise ValueError("旧selection SHA256不匹配，拒绝替换")
        if previous.get("selection_id") != NAMESPACE or previous.get("experiment_namespace") != NAMESPACE:
            raise ValueError("旧selection namespace不匹配")
        if previous.get("registration_attempt") == attempt:
            raise ValueError("不能重用同一attempt覆盖既有运行")
        for kind, proof in proofs.items():
            prior = Path(previous.get("regressions", {}).get(kind, "")).resolve()
            if Path(proof["path"]) == prior:
                raise ValueError("重试需要新回归证明路径，旧full/original11必须留原位")
            if not prior.is_file() or sha256(prior) != previous.get("regression_sha256", {}).get(kind):
                raise ValueError("旧回归证明缺失或变化，不能归档后覆盖")
        if old.read_json(proofs["original11"]["path"]).get("label") != "four_fixes_"+attempt:
            raise ValueError("新11任务证明label必须是four_fixes_"+attempt)
        if Path(proofs["full"]["path"]).stem != "full_"+attempt:
            raise ValueError("新完整回归证明文件名必须是full_"+attempt+".json")
        _assert_inactive(base)
        for target in (base / "experiments" / NAMESPACE / attempt, base / "states" / attempt):
            if target.exists():
                raise ValueError("新attempt目录已经存在，禁止混入既有运行")
        _archive_registration(base, replace_previous_sha256, attempt, previous)
    old.atomic_json(path, value)
    return path


def register(full, original11, *, base=BASE, project=ROOT, attempt=None, replace_previous_sha256=None):
    base, project = version_base(base, project)
    with old.execution_lock(base / "workflow.lock"):
        with old.execution_lock(project / "results/sim_v1_3/overnight_execution.lock"):
            return _register(full, original11, base=base, project=project,
                attempt=attempt, replace_previous_sha256=replace_previous_sha256)


def selection(*, base=BASE, project=ROOT):
    base, project = version_base(base, project)
    path = base / "experiment_selection.json"
    if not path.is_file():
        raise ValueError("c_wall_v134尚未登记，不可采用旧namespace证明")
    data = old.read_json(path)
    if data.get("selection_id") != NAMESPACE or data.get("experiment_namespace") != NAMESPACE:
        raise ValueError("错误或缺失namespace")
    if data.get("planning_code_sha256") != planning_fingerprint(project=project):
        raise ValueError("登记后的源码/标定指纹改变，禁止混批")
    if sha256(base / "baseline.json") != data.get("baseline_sha256"):
        raise ValueError("保护baseline改变")
    for kind in ("full", "original11"):
        proof = _proof(data["regressions"][kind], kind, data["planning_code_sha256"], project)
        if proof["sha256"] != data["regression_sha256"][kind]:
            raise ValueError("登记后的回归证明改变")
    for key in ("source_planning_manifest", "source_sim20_manifest"):
        if sha256(data[key]) != data[key + "_sha256"]:
            raise ValueError("冻结的v133来源manifest改变")
    for row in data["qualification_inputs"]:
        if sha256(row["task"]) != row["sha256"]:
            raise ValueError("冻结的v133资格源改变")
    data["selection_sha256"] = sha256(path)
    return data


def prepare_manifest(stage, jobs, *, base=BASE, project=ROOT):
    base, project = version_base(base, project)
    proof = selection(base=base, project=project)
    attempt = active_attempt(base)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", stage) or stage in {".", ".."}:
        raise ValueError("阶段名不安全")
    frozen = []
    for raw in jobs:
        job = deepcopy(raw)
        job["output"] = str(output_path(job["output"], base))
        job.update(experiment_namespace=NAMESPACE, experiment_selection_sha256=proof["selection_sha256"],
                   planning_code_sha256=proof["planning_code_sha256"])
        if job.get("task"):
            job["task"] = str(Path(job["task"]).resolve())
            job["prepared_task_sha256"] = sha256(job["task"])
        if job.get("kind") in {"sim", "gate", "qualify"}:
            job["motion_code_sha256"] = motion_fingerprint(project=project)
        job.setdefault("timeout_seconds", 60. if job.get("kind") in {"plan", "qualify"} else 1200.)
        frozen.append(job)
    manifest = {"schema_version": 1, "session_id": NAMESPACE + ("_"+attempt if attempt else "") + "_" + stage, "port": 23210,
        "state_dir": str(state_root(base) / stage),
        "report_dir": str(base / "stage_reports" / attempt / stage if attempt else base / "stage_reports" / stage),
        "session_budget_seconds": None, "phase_budgets_seconds": {"plan": None, "sim": 18000., "gates": None}, "jobs": frozen}
    path = base / "manifests" / (stage + ".json")
    if path.exists() and old.read_json(path) != manifest:
        raise ValueError("已准备的阶段定义不同，保留已有结果")
    old.atomic_json(path, manifest)
    old.load_manifest(path)
    return path


def _root(base, stage):
    return experiment_root(base) / stage


def prepare_qualify5(*, base=BASE, project=ROOT):
    proof = selection(base=base, project=project)
    jobs = [{"id": "v134-d5-" + r["name"], "phase": "sim", "kind": "qualify", "family": "F3", "index": r["index"],
             "task": r["task"], "prior_qualification": r["task"], "prior_qualification_sha256": r["sha256"],
             "output": str(_root(base, "qualification") / r["name"]), "unknown_only": True}
            for r in proof["qualification_inputs"] if r["name"] in D_TASKS]
    if len(jobs) != 5:
        raise ValueError("D五任务来源缺失")
    return prepare_manifest("qualify5", jobs, base=base, project=project)


def _d_result_index(*, base=BASE, project=ROOT):
    base, project = version_base(base, project)
    proof = selection(base=base, project=project)
    manifest_path = base / "manifests/qualify5.json"
    manifest = old.load_manifest(manifest_path)
    records = read_records(manifest["state_dir"])
    tasks = {}
    for job in manifest["jobs"]:
        if job.get("experiment_namespace") != NAMESPACE or job.get("experiment_selection_sha256") != proof["selection_sha256"]:
            raise ValueError("D阶段不属于当前封存namespace")
        row = records.get(job["id"], {})
        if row.get("definition_sha256") != job["definition_sha256"] or row.get("status") not in {"completed", "partial"} or row.get("simulator_closed") is not True:
            raise ValueError("D重鉴定尚无完整终态证据：" + job["id"])
        path = output_path(row["qualified_task"], base)
        if not path.is_relative_to(Path(job["output"]).resolve()):
            raise ValueError("D结果不属于对应worker输出目录")
        log_path = output_path(row["log_file"], base)
        log = old.read_json(log_path)
        if (not log_path.is_relative_to(Path(job["output"]).resolve()) or log.get("source_task") != job["task"]
                or log.get("source_sha256") != job["task_sha256"] or log.get("qualified_task") != str(path)
                or log.get("simulator_closed") is not True or not isinstance(log.get("simulator_pid"), int)
                or log["simulator_pid"] <= 0 or log.get("status") != row["status"] or "escalation" not in log):
            raise ValueError("D日志缺少真实来源/升级/关闭证据")
        name = _task_name(path)
        if name in tasks:
            raise ValueError("重复D任务")
        tasks[name] = {"source_task": job["task"], "source_sha256": job["task_sha256"],
            "qualified_task": str(path), "qualified_sha256": sha256(path), "log_file": str(log_path), "log_sha256": sha256(log_path),
            "status": row["status"], "simulator_closed": True, "job_id": job["id"], "definition_sha256": job["definition_sha256"]}
    if set(tasks) != set(D_TASKS):
        raise ValueError("D五任务结果集合不完整")
    return {"experiment_namespace": NAMESPACE, "tasks": tasks, "qualify_manifest_sha256": sha256(manifest_path),
            "qualify_records_sha256": old.fingerprint(sorted(records.values(), key=lambda r: r["id"]))}


def collect_d_results(*, base=BASE, project=ROOT):
    value = _d_result_index(base=base, project=project)
    base = Path(base)
    path = base / "qualification_index.json"
    if path.exists() and old.read_json(path) != value:
        raise ValueError("D索引已存在且不同，禁止覆盖")
    old.atomic_json(path, value)
    return path


def qualified_inputs(*, base=BASE, project=ROOT):
    proof = selection(base=base, project=project)
    path = Path(base) / "qualification_index.json"
    if not path.is_file():
        raise ValueError("D五任务重鉴定索引缺失")
    index = old.read_json(path)
    if index.get("experiment_namespace") != NAMESPACE or set(index.get("tasks", {})) != set(D_TASKS):
        raise ValueError("D五任务索引namespace/任务集合不同")
    if index != _d_result_index(base=base, project=project):
        raise ValueError("D索引与真实升级journal/log不一致")
    result = []
    for original in proof["qualification_inputs"]:
        row = dict(original)
        if row["name"] in D_TASKS:
            replacement = index["tasks"][row["name"]]
            target = output_path(replacement["qualified_task"], base)
            if (replacement.get("source_task") != row["task"] or replacement.get("source_sha256") != row["sha256"]
                    or replacement.get("qualified_sha256") != sha256(target) or replacement.get("simulator_closed") is not True):
                raise ValueError("D资格来源链/SHA或关闭证据不一致")
            before = tomllib.loads(Path(row["task"]).read_text(encoding="utf-8-sig"))
            after = tomllib.loads(target.read_text(encoding="utf-8-sig"))
            for data in (before, after):
                data.get("capabilities", {}).pop("reach", None)
            if before != after:
                raise ValueError("D资格副本改变了reach之外的任务输入")
            row.update(task=str(target), sha256=sha256(target), d_requalified=True)
        else:
            row["d_requalified"] = False
        result.append(row)
    return result


def prepare_planning(*, base=BASE, project=ROOT):
    jobs = [{"id": f"v134-plan-{r['family'].lower()}-{r['index']:03d}", "phase": "plan", "kind": "plan", "table": "qualified",
             **{k: r[k] for k in ("family", "index", "task", "d_requalified")},
             "output": str(_root(base, "planning") / r["family"] / f"{r['index']:03d}")}
            for r in qualified_inputs(base=base, project=project)]
    return prepare_manifest("planning200", jobs, base=base, project=project)


def planning_committed(base=BASE):
    path = Path(base) / "manifests/planning200.json"
    if not path.is_file():
        return False
    manifest = old.load_manifest(path)
    records = read_records(manifest["state_dir"])
    return len(manifest["jobs"]) == 200 and all(records.get(job["id"], {}).get("definition_sha256") == job["definition_sha256"] for job in manifest["jobs"])


def prepare_fixed_sim20(*, base=BASE, project=ROOT):
    if not planning_committed(base):
        raise ValueError("planning200未全部提交，不能提前冻结sim20来源")
    proof = selection(base=base, project=project)
    inputs = {r["name"]: r for r in qualified_inputs(base=base, project=project)}
    jobs = []
    for number, source in enumerate(old.read_json(proof["source_sim20_manifest"])["jobs"]):
        name = _task_name(source["task"])
        if name not in inputs:
            raise ValueError("固定样本不属于冻结200输入")
        row = inputs[name]
        current_plan = _root(base, "planning") / row["family"] / f"{row['index']:03d}" / "plan.json"
        jobs.append({"id": f"v134-fixed20-{number:02d}", "phase": "sim", "kind": "sim", "family": source["family"],
            "task": row["task"], "source_job_id": source["id"], "source_task_sha256": sha256(source["task"]),
            "source_manifest_sha256": proof["source_sim20_manifest_sha256"], "fixed_sample_index": number,
            "current_planning_job_id": f"v134-plan-{row['family'].lower()}-{row['index']:03d}",
            "current_plan_path": str(current_plan), "current_plan_sha256": sha256(current_plan) if current_plan.is_file() else None,
            "output": str(_root(base, "sim20") / f"{number:02d}")})
    return prepare_manifest("sim20", jobs, base=base, project=project)


def prepare_walls(*, base=BASE, project=ROOT):
    inputs = {(r["family"], r["index"]): r for r in qualified_inputs(base=base, project=project)}
    jobs = [{"id": f"v134-wall-{index:03d}", "phase": "sim", "kind": "sim", "family": "F3", "index": index,
             "task": inputs["F3", index]["task"], "output": str(_root(base, "walls3") / f"{index:03d}")}
            for index in (1, 9, 37)]
    return prepare_manifest("walls3", jobs, base=base, project=project)


def prepare_gates(stage="gates", *, base=BASE, project=ROOT):
    tests = old.LIVE_TESTS[:1] if stage == "motion_gate" else old.LIVE_TESTS
    return prepare_manifest(stage, [{"id": "v134-" + stage + "-" + test.rsplit(".", 1)[-1], "phase": "gates", "kind": "gate",
        "test_name": test, "output": str(_root(base, stage) / test.rsplit(".", 1)[-1])} for test in tests], base=base, project=project)


def require_motion_gate(job, *, base=BASE):
    if job.get("kind") not in {"sim", "qualify", "gate"}:
        return
    if job.get("motion_code_sha256") != motion_fingerprint():
        raise ValueError("真实阶段运动源码指纹已变化")
    if job.get("kind") == "gate" and job.get("test_name") == old.LIVE_TESTS[0]:
        return
    path = Path(base) / "last_motion_gate.json"
    proof = old.read_json(path) if path.exists() else {}
    if proof.get("passed") is not True or proof.get("motion_code_sha256") != motion_fingerprint():
        raise ValueError("v134独立普通取放真实回归门尚未通过")
    attempt = active_attempt(base)
    if attempt and (proof.get("registration_attempt") != attempt or
            proof.get("experiment_selection_sha256") != sha256(Path(base) / "experiment_selection.json")):
        raise ValueError("新attempt必须通过自己的新普通取放门，不能复用旧gate")


def stop_reason(base=BASE):
    path = Path(base) / "global_stop.json"
    if path.exists() and old.read_json(path).get("global_stop"):
        return old.read_json(path)["global_stop"]
    rows = []
    directories = {path.parent for path in (Path(base) / "states").rglob("completed*.jsonl")}
    for directory in directories:
        rows += list(read_records(directory).values())
    failures = 0
    for row in sorted(rows, key=lambda r: (r.get("completed_at", ""), r.get("id", ""))):
        if row.get("phase") not in {"sim", "gates"} or row.get("executed") is False:
            continue
        stage = row.get("startup_failure_phase") or row.get("original_failure_phase") or row.get("failure_phase")
        if stage in {"simulator_start", "simulator_connect"}:
            failures += 1
        elif row.get("startup_connect_completed") or row.get("simulator_pid"):
            failures = 0
    return "跨阶段连续三次启动/连接失败" if failures >= 3 else None


def process_factory(command, log_path, start_signal):
    command = list(command)
    command[3] = str(Path(__file__).resolve())
    path = Path(command[-1])
    spec = old.read_json(path)
    require_motion_gate(spec["job"])
    spec["hard_deadline_epoch"] = time.time() + spec["job"]["timeout_seconds"]
    old.atomic_json(path, spec)
    return old.OwnedProcess(command, log_path, start_signal)


def adapt_result(job, result):
    """Recover startup phase after a parent hard timeout from durable telemetry."""
    result = dict(result)
    if job.get("kind") not in {"qualify", "sim", "gate"} or result.get("status") not in {"timeout", "stopped", "worker_failed"}:
        return result
    root = Path(job["output"])
    paths = (list(root.glob("*/qualify_log.json")) if job.get("kind") == "qualify" else
             list(root.glob("simulation/*/summary.json")) + list(root.glob("simulation/*/run_progress.json")))
    for path in paths:
        try:
            progress = old.read_json(path)
        except (OSError, ValueError):
            continue
        phase = (progress.get("startup_failure_phase") or progress.get("original_failure_phase")
                 or progress.get("failure_phase") or progress.get("phase"))
        if phase in {"simulator_start", "simulator_connect"}:
            result.update(startup_failure_phase=phase, original_failure_phase=phase,
                failure_phase=phase, timeout_checkpoint=str(path), parent_failure_phase=result.get("failure_phase"))
        if progress.get("simulator_pid") or progress.get("pid"):
            result["simulator_pid"] = progress.get("simulator_pid") or progress["pid"]
        if phase in {"workcell", "qualification", "prequalification", "real_planning", "compile", "execute", "complete"}:
            result["startup_connect_completed"] = True
        break
    if list(root.glob("simulation/*/workcell.json")):
        result["startup_connect_completed"] = True
    return result


def run_stage(stage, *, base=BASE):
    base, _ = version_base(base)
    proof = selection(base=base)
    manifest = old.load_manifest(base / "manifests" / (stage + ".json"))
    if any(j.get("kind") == "sim" or (j.get("kind") == "gate" and j.get("test_name") != old.LIVE_TESTS[0]) for j in manifest["jobs"]):
        if not planning_committed(base):
            raise ValueError("planning200未全部提交，禁止启动真实任务/非普通门控")
    for job in manifest["jobs"]:
        output_path(job["output"], base)
        if job.get("experiment_selection_sha256") != proof["selection_sha256"] or job.get("planning_code_sha256") != proof["planning_code_sha256"]:
            raise ValueError("阶段与当前封存登记不同")
        require_motion_gate(job, base=base)
    guard = v134_guard.Guard(base=base)
    def preflight(archive):
        try:
            selection(base=base)
        except (OSError, ValueError, KeyError) as error:
            return {"ok": False, "global_stop": True, "reason": "v134封存源码/输入/证明改变：" + str(error)}
        stopped = stop_reason(base)
        if stopped:
            return {"ok": False, "global_stop": True, "reason": stopped}
        environment = old.preflight(archive)
        if not environment["ok"]:
            return environment
        protection = guard.check()
        return {**environment, "ok": protection["passed"], "global_stop": not protection["passed"],
                "reason": None if protection["passed"] else "v134源码/历史/归档保护不一致", "protection": protection}
    phases = {j["phase"] for j in manifest["jobs"]}
    if len(phases) != 1:
        raise ValueError("每阶段必须单一任务类型")
    def adapter(job, result):
        if job.get("kind") == "plan":
            # Publish only earlier durable rows. The current result is not yet
            # committed by run_phase; final rebuild includes it after fsync.
            try:
                from tools.v134_summary import rebuild
                rebuild(base=base)
            except Exception as error:
                result = {**result, "report_warning": "pre-commit snapshot: " + str(error)}
        return adapt_result(job, result)
    result = old.run_phase(manifest, next(iter(phases)), environment_check=preflight,
        process_factory=process_factory, result_adapter=adapter)
    stopped = stop_reason(base)
    if stopped:
        result.update(status="global_stop", global_stop=stopped)
        old.save_state(manifest, result)
    if result.get("global_stop"):
        old.atomic_json(base / "global_stop.json", {"global_stop": result["global_stop"], "at": old.utc_now()})
    records = old.Journal(manifest["state_dir"]).load()
    for job in manifest["jobs"]:
        row = records.get(job["id"], {})
        if job.get("test_name") == old.LIVE_TESTS[0] and row.get("passed") is True and row.get("tests_run") == 1 and row.get("test_skipped") == 0 and row.get("simulator_closed") is True:
            old.atomic_json(base / "last_motion_gate.json", {"passed": True, "motion_code_sha256": motion_fingerprint(),
                "source": job["output"], "id": job["id"], "completed_at": row.get("completed_at"),
                "registration_attempt": active_attempt(base), "experiment_selection_sha256": proof["selection_sha256"]})
    if stage == "qualify5" and result["status"] == "phase_complete":
        collect_d_results(base=base)
    from tools.v134_summary import rebuild
    rebuild(base=base)
    return result


def plan_job(job, deadline):
    from skillnet.core.engine import Engine
    from skillnet.core.logic import Fact
    from skillnet.domains.blocks import load_problem
    from skillnet.planning.search import search
    began = time.monotonic()
    problem = load_problem(job["task"], ROOT / "skills/blocks_v1.toml")
    problem.proposal = None
    engine = Engine(problem)
    result = search(engine, time_limit=max(.001, deadline-time.monotonic()-1.))
    bound = getattr(result, "suboptimality_bound", None)
    if bound is not None and not math.isfinite(bound):
        bound = None
    unreachable = []
    if job.get("d_requalified") and result.status == "no_plan":
        for condition in engine.goal:
            fact = condition.fact
            if not condition.negated and fact.predicate == "filled" and all(problem.oracle.check(Fact("reach", (a, fact.args[0]))).status.value == "INVALID" for a in problem.hooks.arms):
                unreachable.append(str(fact))
    value = {"task": problem.name, "status": result.status, "passed": result.status == "solved", "makespan": result.makespan,
        "bound": bound, "expanded": result.expanded, "generated": result.generated, "elapsed_seconds": time.monotonic()-began,
        "search_elapsed_seconds": result.elapsed, "unknown": sorted(map(str, result.unknown)), "proposal_enabled": False,
        "witness_loaded": False, "first_solution_seconds": getattr(result, "first_solution_seconds", None),
        "first_solution_expanded": getattr(result, "first_solution_expanded", None), "stages": list(getattr(result, "stages", ())),
        "physical_infeasible": bool(unreachable), "physical_infeasible_goal_proof": unreachable,
        "failure_phase": None if result.status == "solved" else "planning", "reason": None if result.status == "solved" else result.status,
        "schedule": [{"action": str(s.action), "skill": s.action.name, "args": list(s.action.args), "start": s.start, "end": s.end} for s in result.steps]}
    old.atomic_json(Path(job["output"]) / "plan.json", value)
    return value


def worker(path):
    spec = old.read_json(path)
    waiting = time.monotonic()
    while not Path(spec["start_signal"]).exists():
        if time.monotonic()-waiting > 30:
            raise RuntimeError("未获父进程所有权握手")
        time.sleep(.03)
    job, began = spec["job"], time.monotonic()
    deadline = began + max(.001, spec["hard_deadline_epoch"]-time.time()-.75)
    try:
        selection()
        if job.get("task") and sha256(job["task"]) != job["task_sha256"]:
            raise ValueError("冻结任务输入SHA改变")
        require_motion_gate(job)
        if job["kind"] == "plan":
            result = plan_job(job, deadline)
        elif job["kind"] == "qualify":
            from simbridge.qualifier import qualify_task
            options = {}
            if job.get("unknown_only"):
                prior = job.get("prior_qualification")
                if (not prior or Path(prior).resolve() != Path(job["task"]).resolve()
                        or sha256(prior) != job.get("prior_qualification_sha256", job["task_sha256"])):
                    raise ValueError("D仅UNKNOWN升级必须提供同路径且SHA匹配的原资格表")
                options["prior_qualification"] = prior
            result = qualify_task(job["task"], out=job["output"], timeout_wall_s=max(.001, deadline-time.monotonic()),
                deadline=deadline, port=23210, **options)
            result.setdefault("passed", result.get("status") == "completed" and result.get("simulator_closed") is True)
        elif job["kind"] == "sim":
            result = old.sim_job(job)
        elif job["kind"] == "gate":
            result = old.gate_job(job)
        else:
            raise ValueError("未知worker种类")
    except BaseException as error:
        result = {"status": "worker_failed", "passed": False, "failure_phase": "worker", "reason": str(error), "traceback": traceback.format_exc()}
    if "phase" in result:
        result["execution_phase"] = result.pop("phase")
    result.update(experiment_namespace=NAMESPACE, experiment_selection_sha256=job["experiment_selection_sha256"],
        worker_wall_seconds=time.monotonic()-began, definition_sha256=job["definition_sha256"], kind=job["kind"])
    if job["kind"] in {"sim", "gate", "qualify"}:
        result["motion_code_sha256"] = motion_fingerprint()
    result.setdefault("wall_seconds", result["worker_wall_seconds"])
    old.atomic_json(spec["result_path"], result)
    return 0 if result.get("passed") else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description="显式v134登记/准备/运行；导入不启动实验")
    parser.add_argument("--register", nargs=2, metavar=("FULL", "ORIGINAL11"))
    parser.add_argument("--attempt", help="显式新attempt目录，例如runtime_fix1")
    parser.add_argument("--replace-previous-sha256", help="必须精确匹配当前旧selection")
    parser.add_argument("--prepare", choices=("motion_gate", "qualify5", "planning200", "sim20", "walls3", "gates"))
    parser.add_argument("--stage")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        return worker(args.worker)
    if args.register:
        print(register(*args.register, attempt=args.attempt,
                       replace_previous_sha256=args.replace_previous_sha256)); return 0
    if args.attempt or args.replace_previous_sha256:
        parser.error("attempt/replace-previous-sha256仅用于显式--register")
    if args.prepare:
        functions = {"qualify5": prepare_qualify5, "planning200": prepare_planning, "sim20": prepare_fixed_sim20, "walls3": prepare_walls}
        print(functions[args.prepare]() if args.prepare in functions else prepare_gates(args.prepare)); return 0
    if args.stage:
        print(json.dumps(run_stage(args.stage), ensure_ascii=False)); return 0
    parser.error("需要显式--register、--prepare或--stage")


if __name__ == "__main__":
    raise SystemExit(main())
