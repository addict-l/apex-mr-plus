"""Real timings, independent worked examples and old lower-bound safety."""
from copy import deepcopy
from dataclasses import replace
import json
import math
from pathlib import Path
from statistics import median
import unittest

from skillnet.core.engine import Engine
from skillnet.core.library import library_from_data
from skillnet.core.logic import parse_fact
from skillnet.cost.model import CALIBRATION, DistanceCost
from skillnet.domains.blocks import load_problem
from skillnet.planning.heuristic import build_heuristic
from tests.test_search import TestHooks, WORK_SKILL
from tools.v133_calibrate_cost import build_calibration


ROOT = Path(__file__).resolve().parents[1]


class CalibratedCostTests(unittest.TestCase):
    def test_real_pick_duration_is_no_longer_underestimated_fourfold(self):
        engine = Engine(load_problem("tasks/s1_parallel_place.toml"))
        action = engine.parse_action("pick(B, blue, stock_r1)")
        estimate = DistanceCost(engine.hooks).estimate(action, engine.initial)
        # Actual simulator seconds, preserved in
        # results/sim_v1_3/m3_attempt_5/s1_parallel_place/summary.json step 1.
        actual = 16.919999999999714
        self.assertGreaterEqual(estimate / actual, .7)
        self.assertLessEqual(estimate / actual, 1.3)

    def test_worked_duration_and_current_speed_are_independent_literals(self):
        engine = Engine(load_problem("tasks/s1_parallel_place.toml"))
        action = engine.parse_action("pick(B, blue, stock_r1)")
        cost = DistanceCost(engine.hooks)
        # Independently worked from the fitted seconds/metre and sqrt(.30) m.
        self.assertAlmostEqual(cost.estimate(action, engine.initial), 21.594345679256858, places=9)
        engine.hooks.speeds["B"] = .5
        self.assertAlmostEqual(cost.estimate(action, engine.initial), 17.4028124400483, places=9)
        changed = replace(action.skill, op_time=action.op_time+7).ground(action.args)
        self.assertAlmostEqual(cost.estimate(changed, engine.initial), 24.4028124400483, places=9)

    def test_same_named_custom_skill_and_unobserved_skills_keep_original_model(self):
        raw = deepcopy(WORK_SKILL)
        raw["name"] = "pick"
        action = library_from_data({"skill": [raw]}).skills[0].ground(("A",))
        self.assertEqual(DistanceCost(TestHooks()).estimate(action, frozenset()), 2.)
        engine = Engine(load_problem("tasks/s3a_cantilever.toml"))
        cost = DistanceCost(engine.hooks)
        for action in engine.actions.values():
            if action.name not in {"detach", "support_on", "support_off", "co_pick", "co_insert"}:
                continue
            travel = []
            for arm, destination in action.move_to.items():
                origin = next(f.args[1] for f in engine.initial if f.predicate == "arm_at" and f.args[0] == arm)
                travel.append(math.dist(engine.hooks.position(origin), engine.hooks.position(destination))/engine.hooks.speed(arm))
            self.assertAlmostEqual(cost.estimate(action, engine.initial), action.op_time+max(travel, default=0.), places=10)

    def test_calibration_artifact_rebuilds_from_sixty_seven_real_unique_steps(self):
        report = build_calibration()
        artifact = json.loads((ROOT/"results/sim_v1_3/v133/calibration.json").read_text(encoding="utf-8"))
        self.assertEqual(report, artifact)
        self.assertEqual(report["total_samples"], 67)
        self.assertEqual(len({(s["source_summary"], s["step"]) for s in report["samples"]}), 67)
        self.assertEqual({name: row["samples"] for name, row in report["skills"].items() if row["samples"]},
                         {"pick": 32, "place": 29, "handover": 3, "insert": 3})
        for name, row in report["skills"].items():
            self.assertGreater(row["reference_speed"], 0.)
            self.assertGreaterEqual(row["fixed_delta"], 0.)
            self.assertGreaterEqual(row["movement_multiplier"], 1.)
            if name in CALIBRATION:
                for key in ("reference_speed", "fixed_delta", "movement_multiplier"):
                    self.assertEqual(CALIBRATION[name][key], row[key])
            else:
                self.assertEqual(row["status"], "uncalibrated")

    def test_old_optimistic_costs_remain_lower_bounds_for_all_original_ground_actions(self):
        for path in sorted((ROOT/"tasks").glob("*.toml")):
            engine = Engine(load_problem(path))
            cost = DistanceCost(engine.hooks)
            for action in engine.actions.values():
                travel = []
                for arm, destination in action.move_to.items():
                    origin = next(f.args[1] for f in engine.initial if f.predicate == "arm_at" and f.args[0] == arm)
                    travel.append(math.dist(engine.hooks.position(origin), engine.hooks.position(destination))/engine.hooks.speed(arm))
                old = action.op_time+max(travel, default=0.)
                self.assertGreaterEqual(cost.estimate(action, engine.initial)+1e-12, old)

    def test_all_training_resource_schedules_meet_real_makespan_interval(self):
        report = json.loads((ROOT/"results/sim_v1_3/v133/calibration.json").read_text(encoding="utf-8"))
        ratios = []
        for source in report["sources"]:
            summary = json.loads((ROOT/source["source_summary"]).read_text(encoding="utf-8"))
            plan = json.loads((ROOT/source["source_plan"]).read_text(encoding="utf-8"))
            engine = Engine(load_problem(ROOT/source["task_file"]))
            cost, end_by_arm, structural = DistanceCost(engine.hooks), {}, 0.
            for step in plan["schedule"]:
                action = engine.parse_action(step["action"])
                state = frozenset(parse_fact(fact) for fact in step["state_before"])
                start = max((end_by_arm.get(arm, 0.) for arm in action.occupies), default=0.)
                if action.structural:
                    start = max(start, structural)
                end = start+cost.estimate(action, state)
                if action.structural:
                    structural = end
                for arm in action.occupies:
                    end_by_arm[arm] = end
            ratio = max(end_by_arm.values())/summary["actual_makespan_seconds"]
            self.assertGreaterEqual(ratio, .7, source["source_summary"])
            self.assertLessEqual(ratio, 1.3, source["source_summary"])
            ratios.append(ratio)
        self.assertEqual(len(ratios), 14)
        self.assertAlmostEqual(median(ratios), 1.0858344240745699, places=9)

    def test_calibrated_fixed_work_is_visible_to_the_actual_search_heuristic(self):
        engine = Engine(load_problem("tasks/s1_parallel_place.toml"))
        bound = build_heuristic(engine, time_limit=0.)
        # Each board goal needs a pick and a place. Their independently fitted
        # fixed terms sum to 13.211279200839742+13.082892120837263 seconds.
        self.assertGreaterEqual(bound(engine.initial), 26.29417131)

    def test_heuristic_preserves_original_callback_actions_and_custom_cost_zero_floor(self):
        engine = Engine(load_problem("tasks/s1_parallel_place.toml"))
        originals = dict(engine.actions)
        effect_time = engine.hooks.optimistic_effect_time
        observed = []
        def inspect(effect, earliest, times, action):
            self.assertIs(action, originals[action.key])
            observed.append(action.key)
            return effect_time(effect, earliest, times, action)
        engine.hooks.optimistic_effect_time = inspect
        build_heuristic(engine, time_limit=0.)(engine.initial)
        self.assertTrue(observed)
        for key, action in engine.actions.items():
            self.assertIs(action, originals[key])
        self.assertEqual(engine.parse_action("pick(B, blue, stock_r1)").op_time, 2.)
        class CustomCost:
            def estimate(self, action, state):
                return .001
        engine.cost = CustomCost()
        self.assertEqual(build_heuristic(engine, time_limit=0.)(engine.initial), 0.)
        class CostSubclass(DistanceCost):
            pass
        engine.cost = CostSubclass(engine.hooks)
        self.assertEqual(build_heuristic(engine, time_limit=0.)(engine.initial), 0.)
        other_hooks = load_problem("tasks/s1_parallel_place.toml").hooks
        engine.cost = DistanceCost(other_hooks)
        # An estimator bound to different scene hooks receives only the old
        # generic operation floor, never a calibrated scene-specific view.
        self.assertAlmostEqual(build_heuristic(engine, time_limit=0.)(engine.initial), 3.999999998, places=10)


if __name__ == "__main__":
    unittest.main()
