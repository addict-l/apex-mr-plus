"""Reusable symbolic action edges from states reached during offline planning."""

from skillnet.capability.oracle import Evidence, Status
from skillnet.core.logic import parse_fact
from skillnet.cost.model import DistanceCost
from skillnet.planning.search import _time


def _state_key(state):
    return "\n".join(str(fact) for fact in sorted(state))


class CandidateGraph:
    def __init__(self, engine, limit, *, saved=None, changed=()):
        self.engine = engine
        self.limit = limit
        self.actions = tuple(engine.actions.values())
        self.action_ids = {action.key: index for index, action in enumerate(self.actions)}
        self.cost = engine.cost if engine.cost is not None else DistanceCost(engine.hooks)
        self.changed = frozenset(changed)
        self.rows = {}
        self.capabilities = {}
        self.saved_rows = {}
        self.saved_capabilities = {}
        if saved is not None:
            if saved["actions"] != [str(action) for action in self.actions]:
                raise ValueError("候选图的动作模型与当前任务不一致")
            self.saved_rows = saved["states"]
            self.saved_capabilities = saved["capabilities"]
        self.reused_states = 0
        self.reused_edges = 0
        self.generated_rows = 0
        self.reused_capabilities = 0
        self.refreshed_capabilities = 0

    def starts(self, state):
        entries = self.rows.get(state)
        if entries is not None:
            return entries
        saved = self.saved_rows.get(_state_key(state)) if self.saved_rows else None
        if saved is not None:
            entries = tuple((self.actions[index], duration) for index, duration in saved)
            self.reused_states += 1
            self.reused_edges += len(entries)
        else:
            entries = []
            for action in self.engine.candidates(state):
                if self.engine.satisfied(action.pre, state):
                    started = self.engine.start(action, state)
                    duration = (None if self.engine.hooks.invariant(started)
                                else _time(self.cost.estimate(action, state)))
                    entries.append((action, duration))
            entries = tuple(entries)
            self.generated_rows += 1
        if len(self.rows) < self.limit:
            self.rows[state] = entries
        return entries

    def evaluate(self, action):
        result = self.capabilities.get(action.key)
        if result is not None:
            return result
        saved = self.saved_capabilities.get(str(self.action_ids[action.key]))
        if saved is not None and not self.changed.intersection(action.capability):
            status, unknown = saved
            result = (Status(status), tuple(
                Evidence(parse_fact(fact), Status.UNKNOWN, "离线能力记录") for fact in unknown))
            self.reused_capabilities += 1
        else:
            result = self.engine.oracle.evaluate(action.capability)
            if saved is not None:
                self.refreshed_capabilities += 1
        self.capabilities[action.key] = result
        return result

    def to_dict(self):
        return {
            "actions": [str(action) for action in self.actions],
            "states": {_state_key(state): [[self.action_ids[action.key], duration]
                                          for action, duration in entries]
                       for state, entries in self.rows.items()},
            "capabilities": {
                str(self.action_ids[key]): [status.value, [str(item.fact) for item in evidence
                                                           if item.status == Status.UNKNOWN]]
                for key, (status, evidence) in self.capabilities.items()
            },
        }

    def stats(self):
        return {"offline_states": len(self.saved_rows), "cached_states": len(self.rows),
                "reused_states": self.reused_states, "reused_edges": self.reused_edges,
                "generated_rows": self.generated_rows, "reused_capabilities": self.reused_capabilities,
                "refreshed_capabilities": self.refreshed_capabilities}
