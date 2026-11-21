"""Causal and arm-order dependencies for a completed schedule."""

from dataclasses import dataclass, asdict

from skillnet.core.logic import Fact


@dataclass
class GraphNode:
    id: int
    action: str
    start: float
    end: float
    arms: tuple[str, ...]
    title: str
    purposes: list[str]


@dataclass(frozen=True)
class GraphEdge:
    source: int
    target: int
    kind: str
    label: str


@dataclass
class TaskGraph:
    nodes: list[GraphNode]
    edges: list[GraphEdge]

    def to_json(self) -> dict:
        return {
            "nodes": [asdict(node) for node in self.nodes],
            "edges": [asdict(edge) for edge in self.edges],
        }

    def to_mermaid(self) -> str:
        def escaped(text):
            return text.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")

        lines = ["flowchart TD"]
        for node in self.nodes:
            label = (f"第 {node.id} 步 · {node.start:.2f}–{node.end:.2f} 秒 · 臂 {', '.join(node.arms)}"
                     f"<br/>{escaped(node.title)} · {escaped(node.action)}")
            lines.append(f'  n{node.id}["{label}"]')
        for edge in self.edges:
            lines.append(f'  n{edge.source} -->|"{escaped(edge.label)}"| n{edge.target}')
        return "\n".join(lines) + "\n"


def _basis(engine, fact, state, visiting=frozenset()):
    if fact in visiting:
        return set()
    if fact.predicate in engine.hooks.derived_predicates:
        result = set()
        for dependency in engine.hooks.justify(fact, state):
            result.update(_basis(engine, dependency, state, visiting | {fact}))
        return result
    return {fact}


def build_task_graph(engine, steps) -> TaskGraph:
    ordered = sorted(steps, key=lambda step: step.start)
    nodes = [GraphNode(index, str(step.action), step.start, step.end,
                       tuple(step.action.occupies), step.action.skill.title, [])
             for index, step in enumerate(ordered, 1)]
    edges = []
    producers = {}
    deleters = {}
    state = engine.initial
    events = []
    for index, step in enumerate(ordered, 1):
        events.append((step.start, 1, index, index))
        events.append((step.end, 0, step.action.key, index))

    for _, event_kind, _, index in sorted(events):
        step = ordered[index - 1]
        action = step.action
        if event_kind == 0:
            finished = engine.finish(action, state)
            for fact in state - finished:
                producers.pop(fact, None)
                deleters[fact] = index
            added = set(action.add)
            added.update(Fact("arm_at", (arm, place)) for arm, place in action.move_to.items())
            for fact in added:
                producers[fact] = index
                deleters.pop(fact, None)
            state = finished
            continue
        required = set()
        for condition in action.pre:
            if not condition.negated:
                required.update(_basis(engine, condition.fact, step.state))
        for fact in sorted(required):
            if engine.is_static(fact.predicate) or fact.predicate in engine.hooks.resource_predicates:
                continue
            producer = producers.get(fact)
            if producer is not None and fact in step.state:
                label = str(fact)
                edges.append(GraphEdge(producer, index, "causal", label))
                nodes[producer - 1].purposes.append(f"为第 {index} 步提供 {label}")
        state = engine.start(action, state)
        for fact in action.delete:
            producers.pop(fact, None)
            deleters[fact] = index

    causal_pairs = {(edge.source, edge.target) for edge in edges}
    previous_arm_step = {}
    arm_pairs = {}
    for node in nodes:
        for arm in node.arms:
            previous = previous_arm_step.get(arm)
            if previous is not None and (previous, node.id) not in causal_pairs:
                arm_pairs.setdefault((previous, node.id), []).append(arm)
            previous_arm_step[arm] = node.id
    for (source, target), arms in arm_pairs.items():
        edges.append(GraphEdge(source, target, "arm", "同臂顺序 " + ", ".join(arms)))
        nodes[source - 1].purposes.append(f"衔接第 {target} 步的臂 {', '.join(arms)} 操作")

    for condition in engine.goal:
        if not engine.holds(condition, state):
            continue
        if condition.negated:
            producer = deleters.get(condition.fact)
            if producer is not None:
                nodes[producer - 1].purposes.append(f"达成目标 {condition}")
            continue
        for fact in sorted(_basis(engine, condition.fact, state)):
            producer = producers.get(fact)
            if producer is not None:
                text = f"达成目标 {condition}"
                if text not in nodes[producer - 1].purposes:
                    nodes[producer - 1].purposes.append(text)
    return TaskGraph(nodes, edges)
