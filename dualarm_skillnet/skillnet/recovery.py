"""Build an offline recovery bundle and replan from a symbolic fault snapshot."""

import argparse
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import monotonic

from skillnet.capability.oracle import Status
from skillnet.core.engine import Engine
from skillnet.core.logic import parse_fact
from skillnet.domains.blocks import load_problem
from skillnet.planning.backup_plans import prepare_checkpoints, serialize_steps, try_backups
from skillnet.planning.candidate_graph import CandidateGraph
from skillnet.planning.search import search
from skillnet.planning.taskgraph import build_task_graph


SCHEMA = "skillnet-recovery-v1"
MODEL_FILES = (
    "capability/oracle.py", "core/logic.py", "core/skill.py", "core/library.py",
    "core/engine.py", "domains/blocks.py", "cost/model.py", "planning/search.py",
    "planning/candidate_graph.py", "planning/backup_plans.py",
)


def _json_bytes(data):
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _fingerprint(task, skills):
    package = Path(__file__).resolve().parent
    paths = (Path(task), Path(skills), *(package / name for name in MODEL_FILES))
    return sha256(_json_bytes([path.read_text(encoding="utf-8") for path in paths])).hexdigest()


def _atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
        try:
            temporary.write(data)
            temporary.close()
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)


def _read_object(path):
    obj = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(obj, dict):
        raise ValueError(f"{path} 必须是 JSON 对象")
    return obj


def prepare(task, skills, out, *, time_limit, max_expansions, graph_limit, checkpoints,
            backups, backup_seconds):
    began = monotonic()
    model = _fingerprint(task, skills)
    engine = Engine(load_problem(task, skills))
    graph = CandidateGraph(engine, graph_limit)
    nominal = search(engine, time_limit=time_limit, max_expansions=max_expansions, candidate_graph=graph)
    if nominal.status != "solved":
        return {"status": nominal.status, "task": engine.problem.name, "graph": graph.stats()}
    plans = prepare_checkpoints(engine, nominal, graph, checkpoint_limit=checkpoints,
                                backups=backups, seconds=backup_seconds,
                                max_expansions=min(max_expansions, 15_000))
    graph_bytes = _json_bytes(graph.to_dict())
    digest = sha256(graph_bytes).hexdigest()
    bundle = {
        "schema": SCHEMA, "model": model, "graph_sha256": digest, "graph_limit": graph_limit,
        "task": engine.problem.name, "nominal": _summary(nominal), "checkpoints": plans,
    }
    out = Path(out)
    _atomic_write(out / f"graph-{digest}.json", graph_bytes)
    _atomic_write(out / "plans.json", _json_bytes(bundle))
    return {"status": "solved", "task": engine.problem.name, "bundle": str(out),
            "nominal_makespan": nominal.makespan, "nominal_bound": nominal.suboptimality_bound,
            "checkpoints": len(plans), "backups": sum(len(item["plans"])-1 for item in plans),
            "graph_bytes": len(graph_bytes), "graph": graph.stats(), "elapsed_seconds": monotonic()-began}


def _fault_model(engine, snapshot):
    state_rows = snapshot.get("state")
    updates = snapshot.get("capabilities", {})
    disabled = snapshot.get("disabled_skills", {})
    grounded = snapshot.get("disabled_actions", [])
    protected = snapshot.get("protected", [])
    if (not isinstance(state_rows, list) or not isinstance(updates, dict) or not isinstance(disabled, dict)
            or not isinstance(grounded, list) or not isinstance(protected, list)):
        raise ValueError("故障快照需提供完整的 state 列表，能力和约束字段格式须正确")

    state = frozenset(map(parse_fact, state_rows))
    objects = {name for values in engine.problem.objects.values() for name in values}
    positions = set(engine.problem.objects["loc"]) | set(engine.problem.objects["slot"])
    arms = set(engine.problem.objects["arm"])
    positions.update(f"home_{arm}" for arm in arms)
    objects.update(positions)
    if any(fact.predicate not in engine.dynamic_predicates or not set(fact.args) <= objects for fact in state):
        raise ValueError("故障快照含非动态事实，或包含任务中未登记的对象或位置")
    poses = [fact.args for fact in state if fact.predicate == "arm_at"]
    if (any(len(args) != 2 or args[1] not in positions for args in poses)
            or len(poses) != len(arms) or {args[0] for args in poses} != arms):
        raise ValueError("停稳快照必须给每只机械臂提供一个已登记的 arm_at 位置")

    known_skills = set(engine.library.by_name)
    for arm, skills in disabled.items():
        if (arm not in arms or not isinstance(skills, list)
                or not all(isinstance(skill, str) and skill in known_skills for skill in skills)):
            raise ValueError(f"disabled_skills 包含未知的机械臂或技能：{arm} {skills}")
    excluded = {engine.parse_action(action).key for action in grounded}
    if disabled:
        excluded.update(action.key for action in engine.actions.values()
                        if any(action.name in disabled.get(arm, ()) for arm in action.occupies))

    declared = {fact for action in engine.actions.values() for fact in action.capability} if updates else set()
    changed = set()
    for expression, value in updates.items():
        fact, status = parse_fact(expression), Status(value)
        if fact not in declared:
            raise ValueError(f"故障能力未被任务中的任何候选动作使用：{fact}")
        if engine.oracle.check(fact).status != status:
            engine.oracle.update(fact, status, "故障快照")
            changed.add(fact)
    engine.initial = state
    engine.problem.initial = state
    return frozenset(engine.actions)-excluded, changed, frozenset(map(parse_fact, protected))


def _summary(schedule):
    return {"status": schedule.status, "schedule": serialize_steps(schedule.steps),
            "makespan": schedule.makespan if schedule.status == "solved" else None,
            "suboptimality_bound": schedule.suboptimality_bound,
            "expanded": schedule.expanded, "first_solution_seconds": schedule.first_solution_seconds,
            "unknown": [str(fact) for fact in sorted(schedule.unknown)]}


def recover(task, skills, bundle, snapshot, *, time_limit, max_expansions, compare=False):
    began = monotonic()
    deadline = began + time_limit
    bundle = Path(bundle)
    data = _read_object(bundle / "plans.json")
    if data.get("schema") != SCHEMA or data.get("model") != _fingerprint(task, skills):
        raise ValueError("任务、技能库或规划规则与离线包不匹配，请重新准备")
    engine = Engine(load_problem(task, skills))
    allowed, changed, protected = _fault_model(engine, snapshot)

    replay_deadline = min(deadline, monotonic()+min(.5, time_limit/4))
    result, chosen, attempts, unfinished = try_backups(
        engine, data["checkpoints"], allowed=allowed, protected=protected, cuts=frozenset(),
        deadline=replay_deadline)
    replay_seconds = monotonic()-began
    details = {
        "task": engine.problem.name, "stage": "backup" if result is not None else "graph_search",
        "backup_attempts": attempts, "backup_choice": chosen, "backup_budget_exhausted": unfinished,
        "seconds_through_backup_check": replay_seconds, "graph_loaded": False,
        "excluded_actions": len(engine.actions)-len(allowed), "changed_facts": list(map(str, sorted(changed))),
    }
    if result is None:
        digest = data["graph_sha256"]
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("候选图摘要格式错误")
        graph_bytes = (bundle / f"graph-{digest}.json").read_bytes()
        if sha256(graph_bytes).hexdigest() != digest:
            raise ValueError("候选图校验失败，请重新准备离线包")
        graph = CandidateGraph(engine, data["graph_limit"], saved=json.loads(graph_bytes), changed=changed)
        details["graph_loaded"] = True
        started_search = monotonic()
        result = search(engine, allowed=allowed, protected=protected, candidate_graph=graph,
                        time_limit=max(0., deadline-monotonic()), max_expansions=max_expansions)
        if result.first_solution_seconds is not None:
            result.first_solution_seconds += started_search-began
        details["graph"] = graph.stats()
    else:
        result.first_solution_seconds = replay_seconds
    details.update(_summary(result))
    details["task_graph"] = (build_task_graph(engine, result.steps).to_json()
                             if result.status == "solved" else None)
    details["elapsed_seconds"] = monotonic()-began
    if compare:
        cold_began = monotonic()
        cold = search(engine, allowed=allowed, protected=protected,
                      time_limit=time_limit, max_expansions=max_expansions)
        details["full_search"] = {**_summary(cold), "elapsed_seconds": monotonic()-cold_began}
    return details


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    offline = commands.add_parser("prepare", help="准备正常计划、检查点备份和候选图")
    offline.add_argument("--out", type=Path, required=True)
    offline.add_argument("--graph-limit", type=int, default=5_000)
    offline.add_argument("--checkpoints", type=int, default=8)
    offline.add_argument("--backups", type=int, default=2)
    offline.add_argument("--backup-seconds", type=float, default=1.)
    online = commands.add_parser("recover", help="先检查备份，再使用候选图搜索")
    online.add_argument("--bundle", type=Path, required=True)
    online.add_argument("--snapshot", type=Path, required=True)
    online.add_argument("--compare", action="store_true")
    for command in (offline, online):
        command.add_argument("task", type=Path)
        command.add_argument("--skills", type=Path, default=Path("skills/blocks_v1.toml"))
        command.add_argument("--time-limit", type=float, default=60.)
        command.add_argument("--max-expansions", type=int, default=200_000)
    args = parser.parse_args(argv)
    try:
        if not isfinite(args.time_limit) or args.time_limit <= 0 or args.max_expansions <= 0:
            raise ValueError("搜索预算必须为正")
        if args.command == "prepare":
            if (args.graph_limit <= 0 or args.checkpoints <= 0 or args.backups < 0
                    or not isfinite(args.backup_seconds) or args.backup_seconds <= 0):
                raise ValueError("候选图上限、检查点数量和准备时间必须为正，备份数量可为零")
            result = prepare(args.task, args.skills, args.out, time_limit=args.time_limit,
                             max_expansions=args.max_expansions, graph_limit=args.graph_limit,
                             checkpoints=args.checkpoints, backups=args.backups, backup_seconds=args.backup_seconds)
        else:
            result = recover(args.task, args.skills, args.bundle, _read_object(args.snapshot),
                             time_limit=args.time_limit, max_expansions=args.max_expansions,
                             compare=args.compare)
    except (KeyError, TypeError, ValueError, OSError) as error:
        parser.exit(2, f"错误：{error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if result["status"] == "solved" else 1


if __name__ == "__main__":
    raise SystemExit(main())
