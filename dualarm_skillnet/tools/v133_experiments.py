"""Explicit experiment selection; standard library only, no runner imports."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/"results/sim_v1_3/v133"


def stage_name(value):
    if (not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value)
            or value in {".", ".."}):
        raise ValueError("实验阶段必须是安全的basename："+str(value))
    return value


def _evidence_path(value, project):
    if not isinstance(value, str) or not value:
        raise ValueError("回归证据必须是项目内JSON路径")
    path = Path(value)
    path = (path if path.is_absolute() else project/path).resolve()
    if not path.is_relative_to(project) or path.suffix.lower() != ".json":
        raise ValueError("回归证据路径必须位于项目内且为JSON："+value)
    return path.relative_to(project).as_posix()


def load_selection(base=BASE, *, project=ROOT):
    """Return a validated immutable-by-convention snapshot; absence is identity.

    Selected stages/evidence need not exist yet. Missing evidence is pending,
    never a reason to adopt the earlier experiment. Invalid JSON is rejected.
    """
    base, project = Path(base).resolve(), Path(project).resolve()
    if not base.is_relative_to(project):
        raise ValueError("实验选择目录必须位于项目内")
    path = base/"experiment_selection.json"
    if not path.exists():
        return {"selection_id": "baseline", "stages": {}, "diagnostic_trials": [], "regressions": {},
                "planning_code_sha256": {}, "input_cohort_sha256": None, "source_manifest_sha256": None,
                "selection_present": False, "selection_sha256": None,
                "selection_source": path.relative_to(project).as_posix()}
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8-sig"))
    except (OSError, ValueError, UnicodeError) as error:
        raise ValueError("实验选择文件不可读或损坏，禁止回退旧阶段："+str(error)) from error
    if not isinstance(value, dict):
        raise ValueError("实验选择必须是JSON对象")
    result = deepcopy(value)
    stage_name(result.get("selection_id"))
    mapping = result.get("stages")
    if not isinstance(mapping, dict):
        raise ValueError("实验选择必须提供stages映射")
    for logical, physical in mapping.items():
        stage_name(logical);stage_name(physical)
        if logical == "qualify200" or physical == "qualify200":
            raise ValueError("qualify200资格缓存不可重新映射")
        if physical in mapping and physical != logical:
            raise ValueError("实验选择不允许阶段映射链或环")
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("不同逻辑阶段不可共用一个实际阶段")
    trials = result.get("diagnostic_trials", [])
    if not isinstance(trials, list) or not all(isinstance(trial, str) for trial in trials) or len(set(trials)) != len(trials):
        raise ValueError("diagnostic_trials必须是唯一阶段名列表")
    active_stages = {mapping.get(name, name) for name in ("planning200", "sim20", "gates", "qualify200")}
    for trial in trials:
        stage_name(trial)
        if trial in active_stages or trial in mapping.values():
            raise ValueError("选中的实际阶段不能同时列为旧诊断批次")
    regressions = result.get("regressions", {})
    if not isinstance(regressions, dict) or set(regressions)-{"full", "original11"}:
        raise ValueError("regressions只允许full和original11证据")
    result["regressions"] = {key: _evidence_path(path, project) for key, path in regressions.items()}
    fingerprints = result.get("planning_code_sha256", {})
    if not isinstance(fingerprints, dict):
        raise ValueError("planning_code_sha256必须是字典")
    for key, digest in fingerprints.items():
        source = Path(key)
        source = (source if source.is_absolute() else project/source).resolve()
        if not source.is_relative_to(project):
            raise ValueError("规划指纹来源必须位于项目内："+str(key))
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            raise ValueError("规划指纹必须是SHA256："+str(key))
    for key in ("input_cohort_sha256", "source_manifest_sha256"):
        if key in result and (not isinstance(result[key], str) or not re.fullmatch(r"[0-9a-fA-F]{64}", result[key])):
            raise ValueError(key+"必须是SHA256")
    result.setdefault("planning_code_sha256", {})
    result.setdefault("diagnostic_trials", [])
    result.update(selection_present=True, selection_sha256=hashlib.sha256(raw).hexdigest(),
                  selection_source=path.relative_to(project).as_posix())
    return result


def resolve_stage(logical, base=BASE, *, selection=None, project=ROOT):
    stage_name(logical)
    selected = load_selection(base, project=project) if selection is None else selection
    return selected.get("stages", {}).get(logical, logical)
