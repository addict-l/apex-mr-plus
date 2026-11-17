"""Admissibility and bounded-fallback checks for the block abstraction."""

from dataclasses import replace
import unittest

from skillnet.core.engine import Engine
from skillnet.core.logic import Fact, parse_condition
from skillnet.domains.blocks import load_problem
from skillnet.planning.abstraction import PatternDatabase, EventPatternDatabase
from skillnet.planning.search import _search as exhaustive_search


class PatternDatabaseTests(unittest.TestCase):
    def test_bound_does_not_exceed_exact_small_task_makespan(self):
        for name in ("s1_parallel_place", "s1_nearest_arm", "s1_reach_handover", "s2_tower"):
            with self.subTest(task=name):
                engine = Engine(load_problem(f"tasks/{name}.toml"))
                database = PatternDatabase(engine)
                result = exhaustive_search(engine)
                self.assertTrue(database.enabled, database.reason)
                self.assertEqual(result.status, "solved")
                self.assertLessEqual(database(engine.initial), result.makespan)

    def test_platform_bound_preserves_required_left_arm_work(self):
        engine = Engine(load_problem("tasks/s3a_platform.toml"))
        database = PatternDatabase(engine, include_positions=False)
        self.assertTrue(database.enabled)
        self.assertLess(database.states, 1000)
        self.assertAlmostEqual(database(engine.initial), 25.0, places=6)

    def test_running_action_is_finished_for_free_and_remaining_is_respected(self):
        engine = Engine(load_problem("tasks/s1_nearest_arm.toml"))
        result = exhaustive_search(engine)
        database = PatternDatabase(engine)
        first = result.steps[0]
        started = engine.start(first.action, first.state)
        running = ((first.action.key, first.end - first.start),)
        estimate = database(started, running)
        self.assertGreaterEqual(estimate, first.end - first.start)
        self.assertLessEqual(estimate, result.makespan)

    def test_negative_location_goal_is_relaxed_when_locations_merge(self):
        engine = Engine(load_problem("tasks/s1_nearest_arm.toml"))
        goal = (parse_condition("not at(red, stock_c)"), parse_condition("at(red, board_c)"))
        database = PatternDatabase(engine, goal=goal)
        self.assertTrue(database.enabled)
        self.assertEqual(database.regions["stock_c"], database.regions["board_c"])
        self.assertEqual(database(engine.initial), 0.0)

    def test_state_and_time_budget_disable_pruning_safely(self):
        engine = Engine(load_problem("tasks/s1_parallel_place.toml"))
        for arguments in ({"max_states": 1}, {"time_limit": 0.0}):
            with self.subTest(arguments=arguments):
                database = PatternDatabase(engine, **arguments)
                self.assertFalse(database.enabled)
                self.assertEqual(database(engine.initial), 0.0)

    def test_position_classes_strengthen_work_bound(self):
        engine = Engine(load_problem("tasks/s3a_platform.toml"))
        work = PatternDatabase(engine, include_positions=False)
        movement = PatternDatabase(engine)
        self.assertTrue(movement.enabled)
        self.assertGreater(movement(engine.initial), work(engine.initial))
        self.assertLess(movement(engine.initial), 54.8)
        self.assertLess(movement.states, 3000)

    def test_custom_cost_disables_abstraction(self):
        class CheaperCost:
            def estimate(self, action, state):
                return 0.1

        engine = Engine(load_problem("tasks/s1_parallel_place.toml"), cost=CheaperCost())
        database = PatternDatabase(engine)
        self.assertFalse(database.enabled)
        self.assertEqual(database(engine.initial), 0.0)

    def test_custom_skill_semantics_disable_abstraction(self):
        problem = load_problem("tasks/s1_nearest_arm.toml")
        skills = list(problem.library.skills)
        skills[0] = replace(skills[0], structural=True)
        problem.library = replace(problem.library, skills=tuple(skills))
        engine = Engine(problem)
        database = PatternDatabase(engine)
        self.assertFalse(database.enabled)
        self.assertEqual(database(engine.initial), 0.0)


class EventPatternDatabaseTests(unittest.TestCase):
    def test_event_bound_is_below_unpruned_small_task_optimum(self):
        engine = Engine(load_problem("tasks/s1_reach_handover.toml"))
        database = EventPatternDatabase(engine)
        result = exhaustive_search(engine)
        self.assertTrue(database.event_enabled, database.event_reason)
        self.assertLessEqual(database(engine.initial), result.makespan)

    def test_ready_insert_can_wait_for_support_after_duration_order_reverses(self):
        engine = Engine(load_problem("tasks/s3a_cantilever.toml"))
        engine.hooks.positions["beam"] = (0.0, 0.0, 0.0)
        engine.hooks.positions["home_A"] = (2.025, 0.0, 0.0)
        engine.hooks.positions["stock_r1"] = (1.8, 0.0, 0.0)
        engine.hooks.positions["stock_r2"] = (1.5, 0.0, 0.0)
        start = set(engine.initial)
        start.difference_update({Fact("free", ("B",)), Fact("arm_at", ("B", "home_B")),
                                 Fact("at", ("pillar", "stock_l1")), Fact("at", ("beam_block", "stock_r1"))})
        start.update({Fact("held", ("B", "beam_block")), Fact("arm_at", ("B", "stock_r1")),
                      Fact("filled", ("p1",)), Fact("installed", ("pillar", "p1"))})
        start = frozenset(start)
        support = engine.parse_action("support_on(A, beam)")
        insert = engine.parse_action("insert(B, beam_block, beam)")
        engine.actions = {support.key: support, insert.key: insert}
        database = EventPatternDatabase(engine, start=start, goal=(parse_condition("filled(beam)"),))
        self.assertTrue(database.event_enabled, database.event_reason)
        node = database.event_ids[(database.project(start), ())]
        self.assertEqual(database.event_distances[node], 10)
        self.assertLessEqual(database(start), 10.2)
        state = engine.start(insert, engine.start(support, start))
        self.assertIsNone(engine.hooks.invariant(state))
        state = engine.finish(support, state)
        self.assertIsNone(engine.hooks.invariant(state))
        state = engine.finish(insert, state)
        self.assertIsNone(engine.hooks.invariant(state))

    def test_event_budget_falls_back_to_work_bound(self):
        engine = Engine(load_problem("tasks/s1_reach_handover.toml"))
        work = PatternDatabase(engine)
        database = EventPatternDatabase(engine, max_states=50)
        self.assertFalse(database.event_enabled)
        self.assertAlmostEqual(database(engine.initial), work(engine.initial))


if __name__ == "__main__":
    unittest.main()
