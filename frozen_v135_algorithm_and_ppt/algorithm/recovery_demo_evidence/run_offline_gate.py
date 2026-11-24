"""Offline recovery gate for db77489 on the LEGO bridge joint-action domain.

Everything runs inside recovery_demo/ except read-only inputs from the showcase
assets and the two committed validation receipts. No CoppeliaSim is started.
"""
from collections import Counter
from collections import defaultdict
from argparse import ArgumentParser
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from time import monotonic, perf_counter
import sys
import tomllib
import traceback

HERE = Path(__file__).resolve().parent
SHOWCASE = HERE.parent
VENDOR = HERE / "vendor_db77489" / "dualarm_skillnet"
sys.path.insert(0, str(VENDOR))

from skillnet.capability.oracle import Status                       # noqa: E402
from skillnet.core.engine import Engine                            # noqa: E402
from skillnet.core.logic import Condition, Fact, parse_fact        # noqa: E402
from skillnet.domains.blocks import load_problem                   # noqa: E402
from skillnet.planning.backup_plans import prepare_checkpoints, try_backups, serialize_steps  # noqa: E402
from skillnet.planning.candidate_graph import CandidateGraph       # noqa: E402
from skillnet.planning.search import search                        # noqa: E402
from skillnet.planning.taskgraph import build_task_graph           # noqa: E402
from skillnet import recovery as vendor_recovery                   # noqa: E402

MODEL_FILES = (
    "capability/oracle.py", "core/logic.py", "core/skill.py", "core/library.py",
    "core/engine.py", "domains/blocks.py", "cost/model.py", "planning/search.py",
    "planning/candidate_graph.py", "planning/backup_plans.py",
)

MANIFEST = SHOWCASE / "assets" / "scene_manifest.json"
TEMPLATE_TASK = SHOWCASE / "mode_a" / "task.toml"
SKILLS = VENDOR / "skills" / "blocks_v1.toml"
HAND_MID = (0.0, 0.135, 0.238)
DEMO_TASK = HERE / "task_recovery.toml"
BUNDLE = HERE / "bundle"
RECEIPT = HERE / "offline_gate_receipt.json"

PIERS = ("Orange_01", "Orange_02", "Red_01", "Red_02",
         "Blue_01", "Blue_02", "Green_01", "Green_02")
DECK = "Bridge_deck_12x4"
HANDOVER_BLOCKS = {"Orange_01": ("A", "B"), "Orange_02": ("A", "B"),
                   "Green_02": ("B", "A"), "Green_01": ("B", "A")}
BASE_PIERS = ("Blue_01", "Blue_02", "Red_01", "Red_02")
GATED_PICKS = ("Orange_01", "Orange_02", "Green_01", "Green_02")
OUTER_FIRST = (("Orange_01", "Orange_02"), ("Green_02", "Green_01"))

GRAPH_LIMIT = 200_000
ADAPTER_BEGAN = monotonic()
GLOBAL_BUDGET = 235.0


def json_bytes(data):
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def file_sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def model_fingerprint(task, skills):
    paths = (Path(task), Path(skills), *(VENDOR / "skillnet" / name for name in MODEL_FILES))
    return sha256(json_bytes([path.read_text(encoding="utf-8") for path in paths])).hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(json_bytes(data) + b"\n")
    temporary.replace(path)


def toml_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(float(value))
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(toml_value(item) for item in value) + "]"
    raise TypeError(value)


def generate_task(manifest, template):
    """Rebuild the joint-action task from scene geometry with the shared hand_mid."""
    positions = {name: tuple(row["initial_bottom_m"]) for name, row in manifest["parts"].items()}
    targets = {name: tuple(row["target_bottom_m"]) for name, row in manifest["parts"].items()}
    lines = ['name = "lego_bridge_recovery"',
             'description = "LEGO bridge joint reach/backup/graph-repair offline gate"',
             'arms = ["A", "B"]',
             'fill_all_slots = true', ""]
    for arm in template["arm"]:
        lines += ["[[arm]]", f'id = {toml_value(arm["id"])}',
                  f'home = {toml_value(arm["home"])}', f'speed = {toml_value(arm["speed"])}', ""]
    for name in [*PIERS, DECK]:
        lines += ["[[location]]", f'id = {toml_value("stock_" + name)}', 'kind = "stock"',
                  f'pos = {toml_value(positions[name])}', ""]
    lines += ["[[location]]", 'id = "hand_mid"', 'kind = "handover"',
              f'pos = {toml_value(HAND_MID)}', ""]
    slots = {row["id"]: row for row in template["slot"]}
    for name in [*PIERS, DECK]:
        slot_id = "seat_" + name
        row = slots[slot_id]
        lines += ["[[block]]", f'id = {toml_value(name)}', f'type = {toml_value(row["type"])}',
                  f'at = {toml_value("stock_" + name)}', ""]
        lines += ["[[slot]]", f'id = {toml_value(slot_id)}', f'type = {toml_value(row["type"])}',
                  f'pos = {toml_value(row["pos"])}', f'below = {toml_value(row.get("below", []))}', ""]
    reachable = [*( "stock_" + name for name in PIERS), *( "seat_" + name for name in PIERS),
                 "hand_mid"]
    arm_a = reachable
    arm_b = [*reachable, "stock_" + DECK, "seat_" + DECK]
    lines += ["[capabilities.reach]", 'default = "INVALID"',
              f'valid = {{ A = {toml_value(arm_a)}, B = {toml_value(arm_b)} }}', ""]
    lines += ["[capabilities.liftable]", 'default = "VALID"', ""]
    text = "\n".join(lines).rstrip("\n") + "\n"
    return text, {"positions": positions, "targets": targets, "reach_a": arm_a, "reach_b": arm_b}


def base_action_keys(engine):
    keys = []
    for name in PIERS:
        for arm in ("A", "B"):
            keys.append(("pick", (arm, name, "stock_" + name)))
            keys.append(("insert", (arm, name, "seat_" + name)))
    for name, (giver, taker) in HANDOVER_BLOCKS.items():
        keys.append(("handover", (giver, taker, name, "hand_mid")))
    keys.append(("pick", ("B", DECK, "stock_" + DECK)))
    keys.append(("insert", ("B", DECK, "seat_" + DECK)))
    ordered = tuple(sorted(keys))
    missing = [key for key in ordered if key not in engine.actions]
    if missing:
        raise AssertionError(f"base_allowed references unknown actions: {missing}")
    return ordered


def rebuild_triggers(engine):
    counts = Counter(c.fact for action in engine.actions.values() for c in action.pre
                     if not c.negated and c.fact.predicate in engine.dynamic_predicates)
    engine.trigger_index = defaultdict(list)
    engine.untriggered = []
    for action in engine.actions.values():
        triggers = [c.fact for c in action.pre if not c.negated
                    and c.fact.predicate in engine.dynamic_predicates]
        if triggers:
            engine.trigger_index[min(triggers, key=lambda f: (counts[f], str(f)))].append(action)
        else:
            engine.untriggered.append(action)


def replace_action(engine, key, extra):
    action = engine.actions[key]
    updated = type(action)(action.skill, action.args, action.pre + tuple(extra), action.capability,
                           action.add, action.delete, action.occupies, action.move_to,
                           action.primitives)
    engine.actions[key] = updated
    return updated


def install_contract(engine):
    """Install the shared dynamic gates on every engine copy."""
    filled_conditions = tuple(Condition(Fact("filled", ("seat_" + name,)))
                              for name in BASE_PIERS)
    for action in list(engine.actions.values()):
        if action.name != "pick" or action.args[1] not in GATED_PICKS:
            continue
        if action.args[2] != "stock_" + action.args[1]:
            continue
        replace_action(engine, action.key, filled_conditions)
    replace_action(engine, ("pick", ("A", "Orange_02", "stock_Orange_02")),
                   (Condition(Fact("at", ("Orange_01", "stock_Orange_01")), True),))
    replace_action(engine, ("pick", ("B", "Green_01", "stock_Green_01")),
                   (Condition(Fact("at", ("Green_02", "stock_Green_02")), True),))
    rebuild_triggers(engine)


def build_engine(install):
    engine = Engine(load_problem(DEMO_TASK, SKILLS))
    if install:
        install_contract(engine)
    return engine


def state_facts(rows):
    return frozenset(map(parse_fact, rows))


def find_target_checkpoint(checkpoints):
    required_slots = {"seat_" + name for name in BASE_PIERS}
    for checkpoint in checkpoints:
        facts = state_facts(checkpoint["state"])
        installed = {name for name in BASE_PIERS
                     if Fact("installed", (name, "seat_" + name)) in facts}
        filled = {slot for slot in required_slots if Fact("filled", (slot,)) in facts}
        at_stock = {name for name in (*GATED_PICKS, DECK)
                    if Fact("at", (name, "stock_" + name)) in facts}
        free = all(Fact("free", (arm,)) for arm in ("A", "B"))
        no_held = not any(fact.predicate == "held" for fact in facts)
        if installed == set(BASE_PIERS) and filled == required_slots and at_stock == set(GATED_PICKS) | {DECK} and free and no_held:
            return checkpoint
    return None


def describe_suffix(engine, schedule):
    return [row["action"] for row in schedule]


def suffix_keys(engine, schedule):
    return [engine.parse_action(row["action"]).key for row in schedule]


def valid_capabilities(engine, keys):
    return {key for key in keys if engine.oracle.evaluate(engine.actions[key].capability)[0] == Status.VALID}


def classify_l2(schedule, engine, allowed):
    actions = [engine.parse_action(row["action"]) for row in schedule]
    picks = {(row.args[1],): index for index, row in enumerate(actions) if row.name == "pick"}
    handovers = [(row.args[0], row.args[1], row.args[2]) for row in actions if row.name == "handover"]
    deck_insert = [row for row in actions
                   if row.name == "insert" and row.args[1] == DECK and row.args[0] == "B"]
    order_ok = {}
    for earlier, later in OUTER_FIRST:
        order_ok[f"{earlier}_before_{later}"] = picks.get((earlier,), 10**9) < picks.get((later,), 10**9)
    capability_ok = all(engine.oracle.evaluate(row.capability)[0] == Status.VALID for row in actions)
    return {
        "handovers": handovers,
        "handover_count": len(handovers),
        "handover_targets_met": set(handovers) == {(g, t, p) for p, (g, t) in HANDOVER_BLOCKS.items()},
        "deck_placed_by_b": len(deck_insert) == 1,
        "outer_first": order_ok,
        "capabilities_valid": capability_ok,
        "all_in_allowed": all(row.key in allowed for row in actions),
        "actions": [row.key for row in actions],
    }


def run_gate():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8-sig"))
    template = tomllib.loads(TEMPLATE_TASK.read_text(encoding="utf-8-sig"))
    template_locations = {row["id"]: row for row in template["location"]}
    template_slots = {row["id"]: row for row in template["slot"]}
    for name in [*PIERS, DECK]:
        assert tuple(template_locations["stock_" + name]["pos"]) == tuple(manifest["parts"][name]["initial_bottom_m"])
        assert tuple(template_slots["seat_" + name]["pos"]) == tuple(manifest["parts"][name]["target_bottom_m"])
    task_text, geometry = generate_task(manifest, template)
    DEMO_TASK.write_text(task_text, encoding="utf-8")
    for name in [*PIERS, DECK]:
        assert geometry["positions"][name] == tuple(template_locations["stock_" + name]["pos"])
        assert geometry["targets"][name] == tuple(template_slots["seat_" + name]["pos"])

    engine = build_engine(install=True)
    base_allowed = base_action_keys(engine)
    base_set = frozenset(base_allowed)

    graph = CandidateGraph(engine, GRAPH_LIMIT)
    started = perf_counter()
    nominal = search(engine, allowed=base_set, candidate_graph=graph, time_limit=60.,
                     post_first_solution_limit=2.)
    nominal_seconds = perf_counter() - started
    if nominal.status != "solved":
        raise AssertionError(f"nominal search status={nominal.status}")

    prepared_started = perf_counter()
    checkpoints = prepare_checkpoints(engine, nominal, graph, checkpoint_limit=20, backups=3,
                                      seconds=4.0, max_expansions=15_000, base_allowed=base_allowed)
    prepared_seconds = perf_counter() - prepared_started
    graph_dict = graph.to_dict()
    backups = sum(len(item["plans"]) - 1 for item in checkpoints)
    target = find_target_checkpoint(checkpoints)

    contract = {
        "task_sha256": file_sha(DEMO_TASK),
        "base_allowed": [list(key) for key in base_allowed],
        "script_sha256": file_sha(Path(__file__).resolve()),
    }
    contract_sha = sha256(json_bytes(contract)).hexdigest()
    model = model_fingerprint(DEMO_TASK, SKILLS)
    graph_sha = sha256(json_bytes(graph_dict)).hexdigest()
    read_only = {
        "mode_a_validation": file_sha(SHOWCASE / "mode_a" / "validation.json"),
        "mode_b_validation": file_sha(SHOWCASE / "mode_b" / "validation.json"),
        "mode_b_handover_gate": file_sha(SHOWCASE / "mode_b" / "handover_gate.json"),
    }
    bundle = {
        "schema": "lego-recovery-offline-v1",
        "contract_sha256": contract_sha,
        "model": model,
        "graph_sha256": graph_sha,
        "base_allowed": [list(key) for key in base_allowed],
        "graph_limit": GRAPH_LIMIT,
        "graph": graph_dict,
        "checkpoints": checkpoints,
        "nominal": {"status": nominal.status, "steps": len(nominal.steps),
                    "makespan": nominal.makespan, "search_seconds": nominal_seconds,
                    "schedule": serialize_steps(nominal.steps)},
        "target_checkpoint": target["id"] if target is not None else None,
        "read_only_receipts": read_only,
    }
    write_json(BUNDLE / "plans.json", bundle)
    if target is None:
        raise AssertionError("target settled checkpoint not found in prepare output")
    saved_ids = {item["id"] for item in json.loads((BUNDLE / "plans.json").read_text(encoding="utf-8"))["checkpoints"]}
    assert target["id"] in saved_ids, "target checkpoint missing from bundle/plans.json"

    # ---------------- L1: one grounded action disabled -> backup hit ----------------
    next_step = nominal.steps[target["id"]]
    next_action = str(next_step.action)
    next_key = next_step.action.key
    fault_l1 = {"state": target["state"], "disabled_actions": [next_action],
                "checkpoint": target["id"], "budget_seconds": 30.}
    write_json(HERE / "fault_l1.json", fault_l1)
    l1_engine = build_engine(install=True)
    allowed_l1_fault, changed_l1, protected_l1 = vendor_recovery._fault_model(l1_engine, fault_l1)
    allowed_l1 = frozenset(key for key in base_allowed if key in allowed_l1_fault)
    l1_started = monotonic()
    l1_result, l1_choice, l1_attempts, l1_unfinished = try_backups(
        l1_engine, checkpoints, allowed=allowed_l1, protected=protected_l1,
        cuts=frozenset(), deadline=l1_started + fault_l1["budget_seconds"])
    l1_seconds = monotonic() - l1_started
    l1_suffix = [str(step.action) for step in l1_result.steps] if l1_result is not None else []
    l1_keys = [l1_engine.parse_action(text).key for text in l1_suffix]
    l1_detail = {
        "status": l1_result.status if l1_result is not None else "failed",
        "stage": "backup" if l1_result is not None else "none",
        "backup_choice": l1_choice,
        "backup_attempts": l1_attempts,
        "graph_loaded": False,
        "decision_seconds": l1_seconds,
        "nominal_suffix_rejected": any(
            row["kind"] == "nominal_suffix" and row["status"] != "valid" for row in l1_attempts),
        "suffix_actions": l1_suffix,
        "suffix_allowed": all(key in base_set for key in l1_keys),
        "suffix_excludes_disabled": next_key not in l1_keys,
        "disabled_action": next_action,
        "checkpoint": target["id"],
        "budget_exhausted": l1_unfinished,
    }
    l1_ok = (l1_result is not None and l1_result.status == "solved" and l1_choice["kind"] == "backup"
             and l1_detail["nominal_suffix_rejected"] and l1_detail["suffix_allowed"]
             and l1_detail["suffix_excludes_disabled"] and l1_detail["graph_loaded"] is False)

    # ---------------- L2: symmetric cross-zone reachability loss -> 4 handovers -----
    invalid = {}
    for name in ("Orange_01", "Orange_02"):
        invalid[f"reach(B, stock_{name})"] = "INVALID"
    for name in ("Red_01", "Red_02"):
        invalid[f"reach(B, seat_{name})"] = "INVALID"
    for name in ("Green_01", "Green_02"):
        invalid[f"reach(B, seat_{name})"] = "INVALID"
    for name in ("Blue_01", "Blue_02"):
        invalid[f"reach(A, stock_{name})"] = "INVALID"
    for name in ("Green_01", "Green_02"):
        invalid[f"reach(A, stock_{name})"] = "INVALID"
    for name in ("Orange_01", "Orange_02"):
        invalid[f"reach(A, seat_{name})"] = "INVALID"
    for name in ("Blue_01", "Blue_02"):
        invalid[f"reach(A, seat_{name})"] = "INVALID"
    fault_l2 = {"state": target["state"], "capabilities": invalid,
                "checkpoint": target["id"], "budget_seconds": 60.}
    write_json(HERE / "fault_l2.json", fault_l2)
    l2_engine = build_engine(install=True)
    allowed_l2_fault, changed_l2, protected_l2 = vendor_recovery._fault_model(l2_engine, fault_l2)
    allowed_l2 = frozenset(key for key in base_allowed if key in allowed_l2_fault)
    l2_started = monotonic()
    l2_deadline = l2_started + fault_l2["budget_seconds"]
    replay_deadline = min(l2_deadline, monotonic() + 0.5)
    l2_result, l2_choice, l2_attempts, l2_unfinished = try_backups(
        l2_engine, checkpoints, allowed=allowed_l2, protected=protected_l2,
        cuts=frozenset(), deadline=replay_deadline)
    l2_stage = "backup" if l2_result is not None else "graph_search"
    l2_graph_stats = None
    if l2_result is None:
        l2_graph = CandidateGraph(l2_engine, GRAPH_LIMIT, saved=graph_dict, changed=changed_l2)
        l2_result = search(l2_engine, allowed=allowed_l2, protected=protected_l2,
                           candidate_graph=l2_graph,
                           time_limit=max(0., l2_deadline - monotonic()), max_expansions=200_000)
        l2_graph_stats = l2_graph.stats()
    l2_seconds = monotonic() - l2_started
    l2_schedule = serialize_steps(l2_result.steps) if l2_result is not None else []
    l2_class = classify_l2(l2_schedule, l2_engine, allowed_l2)

    cold_engine = build_engine(install=True)
    cold_allowed, cold_changed, cold_protected = vendor_recovery._fault_model(cold_engine, fault_l2)
    cold_allowed = frozenset(key for key in base_allowed if key in cold_allowed)
    cold_started = monotonic()
    cold = search(cold_engine, allowed=cold_allowed, protected=cold_protected,
                  time_limit=fault_l2["budget_seconds"], max_expansions=200_000)
    cold_seconds = monotonic() - cold_started
    cold_class = classify_l2(serialize_steps(cold.steps), cold_engine, cold_allowed) if cold.status == "solved" else None

    l2_detail = {
        "status": l2_result.status if l2_result is not None else "failed",
        "stage": l2_stage,
        "backup_choice": l2_choice,
        "backup_attempts": l2_attempts,
        "graph_loaded": l2_stage == "graph_search",
        "graph": l2_graph_stats,
        "recovery_seconds": l2_seconds,
        "recovery_makespan": l2_result.makespan if l2_result is not None else None,
        "cold_search": {"status": cold.status, "seconds": cold_seconds,
                        "makespan": cold.makespan if cold.status == "solved" else None,
                        "schedule": serialize_steps(cold.steps),
                        "classification": cold_class},
        "classification": l2_class,
        "budget_exhausted": l2_unfinished,
        "checkpoint": target["id"],
    }
    l2_ok = (l2_result is not None and l2_result.status == "solved"
             and l2_class["handover_targets_met"] and l2_class["handover_count"] == 4
             and l2_class["deck_placed_by_b"] and all(l2_class["outer_first"].values())
             and l2_class["capabilities_valid"] and l2_class["all_in_allowed"]
             and cold.status == "solved")
    write_json(HERE / "l2_recovery_plan.json", l2_detail)

    receipt = {
        "passed": bool(l1_ok and l2_ok and target is not None),
        "schema": "lego-recovery-offline-gate-v1",
        "vendor": {"commit": "db774898ea73b59d4a9327ff49149712f38d2524",
                   "source": str(VENDOR)},
        "contract_sha256": contract_sha,
        "read_only_receipts": read_only,
        "task": {"path": str(DEMO_TASK), "sha256": file_sha(DEMO_TASK),
                 "hand_mid": list(HAND_MID), "base_allowed_actions": len(base_allowed)},
        "nominal": {"status": nominal.status, "steps": len(nominal.steps),
                    "makespan": nominal.makespan, "search_seconds": nominal_seconds,
                    "expanded": nominal.expanded,
                    "schedule": serialize_steps(nominal.steps)},
        "bundle": {"checkpoints": len(checkpoints), "backups": backups,
                   "graph_states": len(graph_dict["states"]),
                   "preparation_seconds": prepared_seconds,
                   "target_checkpoint": target["id"] if target is not None else None,
                   "path": str(BUNDLE / "plans.json")},
        "l1": {"passed": bool(l1_ok), **l1_detail},
        "l2": {"passed": bool(l2_ok), **l2_detail},
        "elapsed_seconds": monotonic() - ADAPTER_BEGAN,
    }
    if target is not None:
        receipt["target_checkpoint_state"] = target["state"]
    write_json(RECEIPT, receipt)
    print(json.dumps({"passed": receipt["passed"],
                      "nominal": {k: receipt["nominal"][k] for k in ("status", "steps", "makespan")},
                      "target_checkpoint": receipt["bundle"]["target_checkpoint"],
                      "checkpoints": receipt["bundle"]["checkpoints"],
                      "backups": receipt["bundle"]["backups"],
                      "l1": {k: l1_detail[k] for k in ("status", "stage", "graph_loaded")},
                      "l1_decision_ms": round(l1_seconds * 1000, 3),
                      "l2": {k: l2_detail[k] for k in ("status", "stage", "recovery_seconds")},
                      "handovers": l2_class["handover_count"],
                      "handover_targets_met": l2_class["handover_targets_met"],
                      "outer_first": l2_class["outer_first"]},
                     ensure_ascii=False, indent=2))
    return 0 if receipt["passed"] else 1


def live_recover(snapshot_path, out_path):
    """Invoke the db77489 L2 recoverer against a live barrier snapshot (no cold search)."""
    bundle = json.loads((BUNDLE / "plans.json").read_text(encoding="utf-8"))
    contract = {"task_sha256": file_sha(DEMO_TASK),
                "base_allowed": bundle["base_allowed"],
                "script_sha256": file_sha(Path(__file__).resolve())}
    if sha256(json_bytes(contract)).hexdigest() != bundle.get("contract_sha256"):
        raise ValueError("adapter contract fingerprint mismatch")
    if sha256(json_bytes(bundle["graph"])).hexdigest() != bundle.get("graph_sha256"):
        raise ValueError("candidate graph fingerprint mismatch")
    if model_fingerprint(DEMO_TASK, SKILLS) != bundle.get("model"):
        raise ValueError("db77489 model fingerprint mismatch")
    snapshot = json.loads(Path(snapshot_path).read_text(encoding="utf-8-sig"))
    engine = build_engine(install=True)
    base = [(name, tuple(args)) for name, args in bundle["base_allowed"]]
    allowed_fault, changed, protected = vendor_recovery._fault_model(engine, snapshot)
    allowed = frozenset(key for key in base if key in allowed_fault)
    budget = float(snapshot.get("budget_seconds", 60.))
    began = monotonic()
    deadline = began + budget
    result, choice, attempts, unfinished = try_backups(
        engine, bundle["checkpoints"], allowed=allowed, protected=protected,
        cuts=frozenset(), deadline=min(deadline, monotonic() + 0.5))
    stage = "backup"
    graph_stats = None
    if result is None:
        stage = "graph_search"
        graph = CandidateGraph(engine, bundle["graph_limit"], saved=bundle["graph"], changed=changed)
        result = search(engine, allowed=allowed, protected=protected, candidate_graph=graph,
                        time_limit=max(0., deadline - monotonic()), max_expansions=200_000)
        graph_stats = graph.stats()
    seconds = monotonic() - began
    payload = {
        "status": result.status,
        "stage": stage,
        "recovery_seconds": seconds,
        "backup_choice": choice,
        "backup_attempts": attempts,
        "graph_loaded": stage == "graph_search",
        "graph": graph_stats,
        "recovery_schedule": serialize_steps(result.steps) if result.status == "solved" else [],
        "task_graph": (build_task_graph(engine, result.steps).to_json()
                       if result.status == "solved" else None),
        "changed_facts": [str(fact) for fact in sorted(changed)],
        "allowed_actions": len(allowed),
        "budget_seconds": budget,
        "budget_exhausted": unfinished,
        "checkpoint_state": snapshot.get("state"),
    }
    write_json(Path(out_path), payload)
    print(json.dumps({"status": payload["status"], "stage": payload["stage"],
                      "recovery_seconds": round(seconds, 6),
                      "graph": graph_stats}, ensure_ascii=False))
    return 0 if payload["status"] == "solved" else 1


def main(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--live-recover", action="store_true")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.live_recover:
        if args.snapshot is None or args.out is None:
            parser.error("--live-recover needs --snapshot and --out")
        return live_recover(args.snapshot, args.out)
    return run_gate()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException:
        write_json(HERE / "offline_gate_failure.json",
                   {"traceback": traceback.format_exc(), "elapsed_seconds": monotonic() - ADAPTER_BEGAN})
        raise
