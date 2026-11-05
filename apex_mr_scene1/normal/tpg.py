"""Multi-modal temporal plan graph for already planned reference motions.

The input is a sequential, geometrically validated reference plan. This module
does not invent task assignments or replace the motion planner. It discretizes
each transit into path fragments, keeps contact skills atomic, and gives every
geometric conflict the direction induced by the reference plan. Runtime motion
release depends on predecessor completion, never on a fixed global timestamp.

The optional collision oracle must check swept fragment pairs, including held
objects and all relative path phases. Avoiding motion overlap does not by
itself certify stationary waiting poses; that verification belongs to the
motion backend and executor. The certificate reports this boundary explicitly.
"""
from __future__ import annotations

from copy import deepcopy
import heapq
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any


EPS = 1e-9
ARMS = ("A", "B")
ConflictOracle = Callable[[dict[str, Any], dict[str, Any]], bool | Mapping[str, Any]]


def _number(value: Any, name: str, minimum: float = 0.) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    answer = float(value)
    if not math.isfinite(answer) or answer < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return answer


def _configuration(q: Any, name: str) -> list[float]:
    if isinstance(q, (str, bytes)) or not isinstance(q, Sequence) or len(q) != 6:
        raise ValueError(f"{name} must contain six joint positions")
    values = [float(v) for v in q]
    if not all(math.isfinite(v) for v in values):
        raise ValueError(f"{name} must contain finite joint positions")
    return values


def _tasks(tasks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    used = set()
    for index, item in enumerate(tasks):
        task = deepcopy(dict(item))
        task_id = task.get("id")
        if not isinstance(task_id, str) or not task_id or task_id in used:
            raise ValueError(f"Task ids must be unique nonempty strings: {task_id!r}")
        used.add(task_id)
        if task.get("arm") not in ARMS:
            raise ValueError(f"Task {task_id}: arm must be A or B")
        if not isinstance(task.get("kind"), str) or not task["kind"]:
            raise ValueError(f"Task {task_id}: kind is required")
        task["duration"] = _number(task.get("duration"), f"{task_id}.duration")
        dependencies = task.get("dependencies", [])
        if isinstance(dependencies, (str, bytes)):
            raise ValueError(f"Task {task_id}: dependencies must be a list of task ids")
        if task_id in dependencies or len(set(dependencies)) != len(dependencies):
            raise ValueError(f"Task {task_id}: self or duplicate dependency")
        task["dependencies"] = list(dependencies)
        task.setdefault("metadata", {})
        if not isinstance(task["metadata"], Mapping):
            raise ValueError(f"Task {task_id}: metadata must be a mapping")
        task["metadata"] = dict(task["metadata"])
        order = task.get("reference_order", index)
        if isinstance(order, bool) or not isinstance(order, (int, float)) or not math.isfinite(order):
            raise ValueError(f"Task {task_id}: reference_order must be finite")
        task["reference_order"] = float(order)
        task["input_index"] = index
        if "q_path" in task and task["q_path"] is not None:
            if not task["q_path"]:
                raise ValueError(f"Task {task_id}: q_path cannot be empty")
            task["q_path"] = [_configuration(q, f"{task_id}.q_path") for q in task["q_path"]]
        if task["kind"] == "transit" and not task.get("q_path"):
            raise ValueError(f"Transit task {task_id} needs its planned q_path")
        result.append(task)
    for task in result:
        missing = set(task["dependencies"]) - used
        if missing:
            raise ValueError(f"Task {task['id']}: missing dependencies {sorted(missing)}")
    return sorted(result, key=lambda t: (t["reference_order"], t["input_index"]))


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((x-y)**2 for x, y in zip(a, b)))


def _fragments(path: list[list[float]], span: float) -> list[list[list[float]]]:
    """Partition the unchanged polyline, inserting collinear samples if needed.

    Every fragment's coordinate-wise bounding span is at most ``span``. All
    original waypoints occur in the returned polylines, so grouping does not
    replace a curved subpath with an unchecked endpoint chord.
    """
    if len(path) == 1:
        return [[path[0], path[0]]]
    dense = [path[0]]
    for start, end in zip(path, path[1:]):
        steps = max(1, math.ceil(max(abs(x-y) for x, y in zip(start, end)) / span))
        dense.extend([[x + (y-x)*i/steps for x, y in zip(start, end)] for i in range(1, steps+1)])
    fragments = []
    current = [dense[0]]
    lower = list(dense[0]); upper = list(dense[0])
    for q in dense[1:]:
        low = [min(x, y) for x, y in zip(lower, q)]
        high = [max(x, y) for x, y in zip(upper, q)]
        if len(current) > 1 and max(b-a for a, b in zip(low, high)) > span + EPS:
            fragments.append(current)
            current = [current[-1]]
            lower = list(current[0]); upper = list(current[0])
            low = [min(x, y) for x, y in zip(lower, q)]
            high = [max(x, y) for x, y in zip(upper, q)]
        current.append(q)
        lower, upper = low, high
    if len(current) == 1:
        current.append(current[0])
    fragments.append(current)
    return fragments


def _graph(nodes: list[dict], edges: list[dict]):
    by_id = {node["id"]: node for node in nodes}
    if len(by_id) != len(nodes):
        raise ValueError("Duplicate TPG node ids")
    successors = {node: set() for node in by_id}
    predecessors = {node: set() for node in by_id}
    for edge in edges:
        source, target = edge["source"], edge["target"]
        if source not in by_id or target not in by_id:
            raise ValueError(f"TPG edge refers to unknown node: {source} -> {target}")
        successors[source].add(target)
        predecessors[target].add(source)
    counts = {node: len(values) for node, values in predecessors.items()}
    ready = [(by_id[node]["reference_index"], node) for node, degree in counts.items() if degree == 0]
    heapq.heapify(ready)
    order = []
    while ready:
        _, node = heapq.heappop(ready)
        order.append(node)
        for target in successors[node]:
            counts[target] -= 1
            if counts[target] == 0:
                heapq.heappush(ready, (by_id[target]["reference_index"], target))
    if len(order) != len(nodes):
        unresolved = [node for node, degree in counts.items() if degree]
        raise ValueError(f"TPG has a dependency cycle: {unresolved[:12]}")
    return by_id, predecessors, successors, order


def _oracle_answer(value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"conflict": value, "evidence": {}}
    if not isinstance(value, Mapping) or not isinstance(value.get("conflict"), bool):
        raise ValueError("Conflict oracle must return bool or mapping with boolean conflict")
    answer = deepcopy(dict(value))
    if not isinstance(answer.get("evidence", {}), Mapping):
        raise ValueError("Conflict oracle evidence must be a mapping")
    answer.setdefault("evidence", {})
    return answer


def build_tpg(tasks: Sequence[Mapping[str, Any]], conflict_oracle: ConflictOracle | None = None,
              *, max_joint_span: float = .10, skip_ordered_pairs: bool = True) -> dict[str, Any]:
    """Build a reference-order multi-modal TPG from real motion/skill tasks.

    Cross-arm task dependencies retain their meaning. A conflicting fragment
    pair is ordered according to the input reference plan; this operation never
    claims makespan optimality. Dependencies contradicting reference order are
    rejected because the supplied plan would not be a feasible reference.

    ``skip_ordered_pairs`` avoids expensive collision calls for fragment pairs
    already guaranteed not to overlap by the current dependency graph. It does
    not waive stationary waiting-state or changed-world validation.
    """
    max_joint_span = _number(max_joint_span, "max_joint_span", 1e-6)
    ordered_tasks = _tasks(list(tasks))
    nodes, edges, task_map, reference_schedule = [], [], {}, []
    last_q: dict[str, list[float]] = {}
    serial_clock = 0.
    for task in ordered_tasks:
        task_id, arm = task["id"], task["arm"]
        q_path = task.get("q_path")
        if not q_path:
            explicit = task.get("q_start", task["metadata"].get("q"))
            q = _configuration(explicit, f"{task_id}.q") if explicit is not None else last_q.get(arm)
            q_path = [q] if q is not None else []
        is_motion = task["kind"] == "transit"
        chunks = _fragments(q_path, max_joint_span) if is_motion else [q_path]
        lengths = [sum(_distance(a, b) for a, b in zip(chunk, chunk[1:])) for chunk in chunks]
        total_length = sum(lengths)
        node_ids = []
        local_clock = serial_clock
        for fragment_index, (chunk, length) in enumerate(zip(chunks, lengths)):
            duration = task["duration"] * (length / total_length if total_length > EPS else 1/len(chunks))
            node_id = f"{task_id}::p{fragment_index:04d}" if is_motion else f"{task_id}::skill"
            node = {
                "id": node_id, "task_id": task_id, "arm": arm, "kind": task["kind"],
                "node_type": "pose" if is_motion else "skill", "part": task.get("part"),
                "duration": duration, "nominal_duration": duration,
                "q_start": deepcopy(chunk[0]) if chunk else None,
                "q_end": deepcopy(chunk[-1]) if chunk else None,
                "q_samples": deepcopy(chunk), "payload": deepcopy(task.get("payload")),
                "metadata": deepcopy(task["metadata"]),
                "mode": deepcopy(task.get("mode", task["metadata"].get("mode", {}))),
                "reference_index": len(nodes), "reference_order": task["reference_order"],
                "reference_start": local_clock, "reference_end": local_clock + duration,
                "fragment_index": fragment_index, "fragment_count": len(chunks),
                "geometry_available": bool(chunk),
            }
            nodes.append(node); node_ids.append(node_id); local_clock += duration
        if q_path:
            last_q[arm] = list(q_path[-1])
        task_map[task_id] = {**deepcopy(task), "node_ids": node_ids,
                             "first_node": node_ids[0], "last_node": node_ids[-1]}
        reference_schedule.append({"id": task_id, "task_id": task_id, "arm": arm,
                                   "kind": task["kind"], "part": task.get("part"),
                                   "start": serial_clock, "end": serial_clock + task["duration"],
                                   "duration": task["duration"]})
        serial_clock += task["duration"]
    by_id = {node["id"]: node for node in nodes}
    edge_keys = set()

    def add_edge(source, target, edge_type, reason, **extra):
        key = (source, target, edge_type, reason)
        if key not in edge_keys:
            edges.append({"source": source, "target": target, "type": edge_type, "reason": reason, **extra})
            edge_keys.add(key)

    last_node = {}
    for node in nodes:
        arm = node["arm"]
        if arm in last_node:
            add_edge(last_node[arm], node["id"], "type1", "same_arm_order", arm=arm)
        last_node[arm] = node["id"]
    for task in ordered_tasks:
        for dependency in task["dependencies"]:
            source = task_map[dependency]["last_node"]
            target = task_map[task["id"]]["first_node"]
            if by_id[source]["reference_index"] >= by_id[target]["reference_index"]:
                raise ValueError(f"Reference order violates dependency {dependency} -> {task['id']}")
            add_edge(source, target, "type1" if task_map[dependency]["arm"] == task["arm"] else "type2",
                     "task_dependency", source_task=dependency, target_task=task["id"])
    _, _, successors, topology = _graph(nodes, edges)
    reachable: dict[str, set[str]] = {node: set() for node in by_id}
    for source in reversed(topology):
        for target in successors[source]:
            reachable[source].add(target)
            reachable[source].update(reachable[target])
    checks = []
    skipped_ordered = 0
    skipped_unavailable = 0
    cross_arm_pairs = 0
    for index, before in enumerate(nodes):
        for after in nodes[index+1:]:
            if before["arm"] == after["arm"]:
                continue
            cross_arm_pairs += 1
            if skip_ordered_pairs and after["id"] in reachable[before["id"]]:
                skipped_ordered += 1
                continue
            if conflict_oracle is None:
                continue
            if not before["geometry_available"] or not after["geometry_available"]:
                skipped_unavailable += 1
                continue
            answer = _oracle_answer(conflict_oracle(deepcopy(before), deepcopy(after)))
            record = {"a": before["id"], "b": after["id"], **answer}
            checks.append(record)
            if not answer["conflict"]:
                continue
            allowed = answer.get("allowed_orders")
            if allowed is not None:
                allowed = [tuple(pair) for pair in allowed]
                if (before["id"], after["id"]) not in allowed:
                    raise ValueError(f"Reference serialization rejected by geometry oracle: {before['id']} -> {after['id']}")
            add_edge(before["id"], after["id"], "type2", "motion_conflict",
                     evidence=deepcopy(answer["evidence"]))
            # Incremental transitive closure. All edges point forward in the
            # reference order, so no reverse reachability can be introduced.
            descendants = {after["id"]} | reachable[after["id"]]
            for source in by_id:
                if source == before["id"] or before["id"] in reachable[source]:
                    reachable[source].update(descendants)
    _, _, _, topology = _graph(nodes, edges)
    plan = {
        "schema_version": 1, "algorithm": "reference_order_multimodal_TPG",
        "nodes": nodes, "edges": edges, "task_map": task_map,
        "reference_schedule": reference_schedule, "reference_makespan": serial_clock,
        "topological_order": topology, "graph_acyclic": True,
        "geometry_checks": checks,
        "counts": {"nodes": len(nodes), "pose_nodes": sum(n["node_type"] == "pose" for n in nodes),
                   "skill_nodes": sum(n["node_type"] == "skill" for n in nodes),
                   "type1_edges": sum(e["type"] == "type1" for e in edges),
                   "task_type2_edges": sum(e["type"] == "type2" and e["reason"] == "task_dependency" for e in edges),
                   "motion_type2_edges": sum(e["reason"] == "motion_conflict" for e in edges),
                   "cross_arm_pairs": cross_arm_pairs, "oracle_queries": len(checks),
                   "already_ordered_pairs_skipped": skipped_ordered,
                   "pairs_without_geometry": skipped_unavailable},
        "certificate": {
            "dag": True, "geometric_reference_paths_unchanged": True,
            "motion_overlap_pairs_checked": conflict_oracle is not None and skipped_unavailable == 0,
            "collision_oracle_supplied": conflict_oracle is not None,
            "all_nodes_have_geometry": all(n["geometry_available"] for n in nodes),
            "waiting_endpoints_verified": False,
            "persistent_support_requires_executor_feedback": True,
            "continuous_collision_guarantee": False,
            "scope": "sampled swept moving-phase conflicts; stationary waits, world changes and contact invariants require external validation",
        },
    }
    nominal = simulate_tpg(plan)
    plan["nominal_schedule"] = nominal["nodes"]
    plan["nominal_task_schedule"] = nominal["task_schedule"]
    plan["nominal_timestamps"] = nominal["timestamps"]
    plan["nominal_makespan"] = nominal["makespan"]
    for node in nodes:
        node.update(nominal["timestamps"][node["id"]])
    return plan


def simulate_tpg(tpg: Mapping[str, Any], delays: Mapping[str, float] | None = None) -> dict[str, Any]:
    """Discrete-event prediction for the unchanged graph, not robot execution.

    Each nonnegative delay extends a node's service time. A task-id delay is
    applied once to its final node. Completion events release successors; start
    times are computed from predecessor acknowledgments, not reference times.
    Actual robot feedback must drive these acknowledgments in the executor.
    """
    nodes = deepcopy(list(tpg["nodes"]))
    edges = list(tpg["edges"])
    by_id, predecessors, successors, topology = _graph(nodes, edges)
    values = dict(delays or {})
    task_map = tpg.get("task_map", {})
    missing = set(values) - set(by_id) - set(task_map)
    if missing:
        raise ValueError(f"Delays refer to unknown nodes/tasks: {sorted(missing)}")
    delay_nodes: dict[str, float] = {}
    for key, value in values.items():
        target = key if key in by_id else task_map[key]["last_node"]
        delay_nodes[target] = delay_nodes.get(target, 0.) + _number(value, f"delay[{key}]")
    remaining = {node: len(values) for node, values in predecessors.items()}
    complete = set()
    running = []
    times = {}
    events = []
    sequence = 0

    def start(node_id, now):
        nonlocal sequence
        node = by_id[node_id]
        duration = float(node["duration"]) + delay_nodes.get(node_id, 0.)
        times[node_id] = {"start": now, "end": now + duration, "duration": duration,
                          "nominal_duration": float(node["duration"]), "delay": delay_nodes.get(node_id, 0.)}
        events.append({"event": "start", "node_id": node_id, "task_id": node["task_id"], "arm": node["arm"], "time": now})
        sequence += 1
        heapq.heappush(running, (now + duration, sequence, node_id))

    for node_id in topology:
        if remaining[node_id] == 0:
            start(node_id, 0.)
    while running:
        now, _, node_id = heapq.heappop(running)
        complete.add(node_id)
        node = by_id[node_id]
        events.append({"event": "finish", "node_id": node_id, "task_id": node["task_id"], "arm": node["arm"], "time": now})
        for target in sorted(successors[node_id], key=lambda n: by_id[n]["reference_index"]):
            remaining[target] -= 1
            if remaining[target] == 0:
                start(target, now)
    if len(complete) != len(nodes):
        raise ValueError("Event simulation could not complete every TPG node")
    rows = [{"id": n["id"], "task_id": n["task_id"], "arm": n["arm"],
             "kind": n["kind"], "node_type": n["node_type"], "part": n.get("part"),
             **times[n["id"]]} for n in nodes]
    task_schedule = []
    for task_id, task in task_map.items():
        parts = [times[node] for node in task["node_ids"]]
        beginning, end = min(v["start"] for v in parts), max(v["end"] for v in parts)
        task_schedule.append({"id": task_id, "task_id": task_id, "arm": task["arm"],
                              "kind": task["kind"], "part": task.get("part"),
                              "start": beginning, "end": end, "duration": end-beginning})
    return {"kind": "discrete_event_prediction", "nodes": rows, "timestamps": times,
            "task_schedule": task_schedule, "events": events,
            "makespan": max((v["end"] for v in times.values()), default=0.),
            "completed": len(complete), "graph_unchanged": True,
            "predecessor_completion_release": True, "delays": values}


__all__ = ["build_tpg", "simulate_tpg"]
