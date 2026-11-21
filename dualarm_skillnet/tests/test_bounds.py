import math
import unittest

from skillnet.capability.oracle import CapabilityOracle
from skillnet.core.engine import Engine
from skillnet.core.library import library_from_data
from skillnet.core.logic import parse_condition, parse_fact
from skillnet.core.problem import Problem
from skillnet.domains.blocks import load_problem
from skillnet.planning.bounds import HMaxBound, lower_bound
from skillnet.planning.search import search
from tests.test_blocks import scene
from tests.test_search import TestHooks, work_engine
from skillnet.domains.blocks import problem_from_data


class HMaxBoundTests(unittest.TestCase):
    def test_independent_work_uses_max_and_ignores_structural_mutex(self):
        for structural in (False, True):
            engine = work_engine(structural=structural)
            bound = lower_bound(engine, engine.initial)
            self.assertAlmostEqual(bound, 2.0)
            self.assertLessEqual(bound, search(engine).makespan)

    def test_running_effects_are_available_after_remaining_time(self):
        engine = work_engine()
        first = engine.parse_action("work(A)")
        second = engine.parse_action("work(B)")
        state = engine.start(second, engine.start(first, engine.initial))
        running = ((first.key, 0.5), (second.key, 1.25))
        self.assertEqual(lower_bound(engine, state, running), 1.25)
        self.assertEqual(HMaxBound(engine, goal=[])(state, running), 1.25)

    def test_standard_scenes_have_admissible_initial_bounds(self):
        for name in ("s1_parallel_place", "s1_nearest_arm", "s1_reach_handover", "s2_tower"):
            with self.subTest(task=name):
                engine = Engine(load_problem(f"tasks/{name}.toml"))
                bound = lower_bound(engine, engine.initial)
                self.assertGreater(bound, 0)
                self.assertLessEqual(bound, search(engine).makespan + 1e-8)

    def test_negative_goals_need_completed_deletion_and_movement_effects_are_included(self):
        engine = Engine(load_problem("tasks/s1_nearest_arm.toml"))
        bound = HMaxBound(engine, [parse_condition("arm_at(A, board_c)")])
        self.assertAlmostEqual(bound(engine.initial), 4.0)
        negative = HMaxBound(engine, [parse_condition("not at(red, stock_c)")])
        self.assertAlmostEqual(negative(engine.initial), 2.0)
        moved = HMaxBound(engine, [parse_condition("not arm_at(A, home_A)")])
        self.assertAlmostEqual(moved(engine.initial), 2.0)

    def test_capability_change_rebuilds_convenience_cache(self):
        engine = work_engine(status="UNKNOWN")
        self.assertEqual(lower_bound(engine, engine.initial), math.inf)
        for arm in ("A", "B"):
            engine.oracle.update(parse_fact(f"able({arm})"), "VALID", "测试")
        self.assertAlmostEqual(lower_bound(engine, engine.initial), 2.0)

    def test_custom_cost_has_zero_floor(self):
        engine = work_engine()
        class FasterCost:
            def estimate(self, action, state):
                return 0.25
        engine.cost = FasterCost()
        self.assertEqual(lower_bound(engine, engine.initial), 0.0)
        self.assertLessEqual(lower_bound(engine, engine.initial), search(engine).makespan)

    def test_domain_derived_and_or_times(self):
        hooks = problem_from_data(scene()).hooks
        times = {parse_fact("filled(p1)"): 4.0, parse_fact("filled(tp1)"): 3.0,
                 parse_fact("filled(tp2)"): 7.0, parse_fact("supporting(A, beam)"): 9.0}
        self.assertEqual(hooks.optimistic_time(parse_fact("slot_ready(beam)"), times), 4.0)
        self.assertEqual(hooks.optimistic_time(parse_fact("has_support(beam)"), times), 7.0)
        times[parse_fact("supporting(B, beam)")] = 2.0
        self.assertEqual(hooks.optimistic_time(parse_fact("has_support(beam)"), times), 2.0)
        hooks.slots["cw"]["needs_support_during"] = ["beam"]
        self.assertEqual(hooks.optimistic_time(parse_fact("insert_supported(cw)"), times), 2.0)
        self.assertEqual(hooks.optimistic_time(parse_fact("slot_clear(p1)"), times), 0.0)

    def test_cache_is_bounded(self):
        engine = work_engine()
        bound = HMaxBound(engine, cache_limit=1)
        bound(engine.initial)
        first = engine.parse_action("work(A)")
        bound(engine.start(first, engine.initial), ((first.key, 1.0),))
        self.assertLessEqual(len(bound.cache), 1)

    def test_overhang_effect_requires_support_or_all_anchors(self):
        hooks = problem_from_data(scene()).hooks
        beam = parse_fact("filled(beam)")
        times = {parse_fact("supporting(A, beam)"): 8.0}
        self.assertEqual(hooks.optimistic_effect_time(beam, 5.0, times), 8.0)
        times[parse_fact("filled(cw)")] = 4.0
        self.assertEqual(hooks.optimistic_effect_time(beam, 5.0, times), 5.0)
        times.clear()
        self.assertEqual(hooks.optimistic_effect_time(beam, 5.0, times), math.inf)

    def test_overhang_propagation_wakes_when_support_becomes_reachable(self):
        engine = Engine(load_problem("tasks/s3a_cantilever.toml"))
        bound = HMaxBound(engine)
        self.assertTrue(math.isfinite(bound(engine.initial)))
        self.assertGreater(bound(engine.initial), 0.0)

    def test_inserting_arm_cannot_supply_its_own_overhang_support(self):
        engine = Engine(load_problem("tasks/s3a_platform.toml"))
        action = engine.parse_action("insert(B, beam_block, beam)")
        times = {parse_fact("supporting(B, beam)"): 1.0,
                 parse_fact("filled(tp1)"): 4.0, parse_fact("filled(tp2)"): 9.0}
        self.assertEqual(engine.hooks.optimistic_effect_time(parse_fact("filled(beam)"), 5.0, times, action), 9.0)
        times[parse_fact("supporting(A, beam)")] = 3.0
        self.assertEqual(engine.hooks.optimistic_effect_time(parse_fact("filled(beam)"), 5.0, times, action), 5.0)

    def test_platform_effect_reacts_to_later_platform_facts(self):
        engine = Engine(load_problem("tasks/s3a_platform.toml"))
        bound = HMaxBound(engine)
        self.assertTrue(math.isfinite(bound(engine.initial)))
        state = engine.initial | {parse_fact("filled(p1)")}
        beam_bound = HMaxBound(engine, [parse_condition("filled(beam)")])
        self.assertGreater(beam_bound(state), 7.9)


def deletion_engine(*, restore_arm="A", delete_both=False, readd=False):
    remove = {"name": "remove", "params": {"arm": "arm"},
              "pre": ["remover(?arm)", "free(?arm)", "filled(x)"],
              "delete": ["free(?arm)", "filled(x)"], "add": ["held(?arm)"],
              "occupies": ["?arm"], "op_time": 3.0}
    if delete_both:
        remove["delete"].append("filled(y)")
    if readd:
        remove["add"].append("filled(x)")
    restore = {"name": "restore", "params": {"arm": "arm"},
               "pre": ["restorer(?arm)"], "delete": [], "add": ["free(A)"],
               "occupies": ["?arm"], "op_time": 2.0}
    populate = {"name": "populate", "params": {"arm": "arm"},
                "pre": ["populator(?arm)"], "delete": [], "add": ["filled(x)"],
                "occupies": ["?arm"], "op_time": 4.0}
    goal = ["not filled(x)", "free(A)"]
    if delete_both:
        goal.append("not filled(y)")
    initial = ["free(A)", "free(B)", "filled(x)", "filled(y)"]
    problem = Problem("deletion", "删除下界", {"arm": ("A", "B"), "block": (), "loc": (), "slot": ()},
                      frozenset(map(parse_fact, ["remover(A)", f"restorer({restore_arm})", "populator(B)"])),
                      frozenset(map(parse_fact, initial)), tuple(map(parse_condition, goal)),
                      library_from_data({"skill": [remove, restore, populate]}), TestHooks(), CapabilityOracle())
    return Engine(problem)


class NegativeBoundTests(unittest.TestCase):
    def test_delete_then_restore_same_arm_must_both_complete(self):
        engine = deletion_engine()
        self.assertAlmostEqual(HMaxBound(engine)(engine.initial), 5.0)
        self.assertEqual(search(engine).makespan, 5.0)

    def test_restore_on_other_arm_can_overlap_deletion(self):
        engine = deletion_engine(restore_arm="B")
        self.assertAlmostEqual(HMaxBound(engine)(engine.initial), 3.0)
        self.assertEqual(search(engine).makespan, 3.0)

    def test_one_action_can_delete_multiple_negative_goals(self):
        engine = deletion_engine(delete_both=True)
        self.assertAlmostEqual(HMaxBound(engine)(engine.initial), 5.0)
        self.assertEqual(search(engine).makespan, 5.0)

    def test_delete_then_readd_is_not_a_last_deletion(self):
        engine = deletion_engine(readd=True)
        self.assertEqual(HMaxBound(engine)(engine.initial), math.inf)

    def test_running_future_add_requires_later_cleanup_even_in_coarse_cache(self):
        engine = deletion_engine()
        state = engine.initial - {parse_fact("filled(x)")}
        action = engine.parse_action("populate(B)")
        coarse = HMaxBound(engine, coarse=True)
        for remaining in (1.0, 4.0, 10.0):
            running = ((action.key, remaining),)
            self.assertAlmostEqual(coarse(state, running), remaining + 5.0)
            self.assertAlmostEqual(HMaxBound(engine)(state, running), remaining + 5.0)
        self.assertEqual(len(coarse.cache), 1)


if __name__ == "__main__":
    unittest.main()
