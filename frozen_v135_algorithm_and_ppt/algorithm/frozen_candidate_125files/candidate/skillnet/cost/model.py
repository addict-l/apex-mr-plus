"""State-dependent estimates of motion and operation duration."""

from math import dist
from typing import Protocol

from skillnet.core.logic import Fact
from skillnet.core.skill import Action


# Fitted only from preserved successful native runs. Complete samples,
# constraints and sparse/unobserved-skill limitations are recorded in
# results/sim_v1_3/v133/calibration.json. No artifact or simulator is loaded
# by the planning core. Extra overhead and travel multipliers are nonnegative
# and >=1 respectively, preserving the old HMax/PDB optimistic cost floors.
CALIBRATION = {
    "pick": {"reference_speed": .25, "fixed_delta": 11.211279200839742,
             "movement_multiplier": 3.826328842745374},
    "place": {"reference_speed": .25, "fixed_delta": 11.082892120837263,
              "movement_multiplier": 3.489599503883876},
    "handover": {"reference_speed": .25, "fixed_delta": 6.651749184536076,
                 "movement_multiplier": 4.473947272460712},
    "insert": {"reference_speed": .25, "fixed_delta": 12.56111375304947,
               "movement_multiplier": 3.6579641733146255},
}

_SIGNATURES = {
    "pick": ((("arm", "arm"), ("block", "block"), ("loc", "loc")),
             ("approach", "grasp", "lift")),
    "place": ((("arm", "arm"), ("block", "block"), ("loc", "loc")),
              ("approach", "lower", "release")),
    "handover": ((("giver", "arm"), ("taker", "arm"), ("block", "block"), ("loc", "loc")),
                 ("approach", "approach", "grasp", "release")),
    "insert": ((("arm", "arm"), ("block", "block"), ("slot", "slot")),
               ("approach", "align", "insert_part", "release")),
}


def _calibration(action):
    signature = (tuple(action.skill.params.items()), tuple(fact.predicate for fact in action.skill.primitives))
    return CALIBRATION.get(action.name, {}) if signature == _SIGNATURES.get(action.name) else {}


class CostModel(Protocol):
    def estimate(self, action: Action, state: frozenset[Fact]) -> float: ...
    def evaluate(self, action: Action, state: frozenset[Fact]) -> float | None: ...


class DistanceCost:
    def __init__(self, hooks):
        self.hooks = hooks

    def estimate(self, action: Action, state: frozenset[Fact]) -> float:
        movement_times = []
        for arm, destination in action.move_to.items():
            origins = [fact.args[1] for fact in state
                       if fact.predicate == "arm_at" and fact.args[0] == arm]
            if len(origins) != 1:
                raise ValueError(f"动作 {action}：臂 {arm} 必须有且只有一个 arm_at 位置")
            speed = self.hooks.speed(arm)
            if speed <= 0:
                raise ValueError(f"动作 {action}：臂 {arm} 的速度必须大于 0")
            duration = dist(self.hooks.position(origins[0]), self.hooks.position(destination)) / speed
            movement_times.append(duration)
        parameters = _calibration(action)
        return (max(movement_times, default=0.0)*parameters.get("movement_multiplier", 1.0)
                + action.op_time + parameters.get("fixed_delta", 0.0))

    def evaluate(self, action: Action, state: frozenset[Fact]) -> float | None:
        return None
