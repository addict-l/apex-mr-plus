"""Independent admissibility checks at concrete event states."""

from dataclasses import replace
import unittest

from skillnet.core.engine import Engine
from skillnet.core.logic import parse_condition, parse_fact
from skillnet.domains.blocks import load_problem, problem_from_data
from skillnet.planning.abstraction import EventPatternDatabase, PatternDatabase
from skillnet.planning.search import _search


def schedule_states(engine, schedule):
    state = engine.initial
    running = {}
    times = sorted({0.0} | {step.start for step in schedule.steps} | {step.end for step in schedule.steps})
    for current in times:
        for key in sorted(key for key, step in running.items() if step.end == current):
            state = engine.finish(running[key].action, state)
            del running[key]
        remaining = tuple(sorted((key, step.end - current) for key, step in running.items()))
        yield current, state, remaining
        for step in schedule.steps:
            if step.start != current:
                continue
            state = engine.start(step.action, state)
            running[step.action.key] = step
            remaining = tuple(sorted((key, item.end - current) for key, item in running.items()))
            yield current, state, remaining


class AbstractionSafetyTests(unittest.TestCase):
    def check_schedule_bounds(self, engine, database):
        schedule = _search(engine)
        self.assertEqual(schedule.status, "solved")
        checked_running = False
        for current, state, running in schedule_states(engine, schedule):
            with self.subTest(time=current, running=running):
                bound = database(state, running)
                self.assertLessEqual(bound, schedule.makespan - current + 1e-8)
                if running:
                    checked_running = True
        self.assertTrue(checked_running)

    def test_work_and_event_bounds_at_every_optimal_schedule_event(self):
        for name in ("s1_parallel_place", "s1_reach_handover", "s2_tower"):
            with self.subTest(task=name):
                engine = Engine(load_problem(f"tasks/{name}.toml"))
                work = PatternDatabase(engine, time_limit=2.0)
                self.assertTrue(work.enabled, work.reason)
                self.check_schedule_bounds(engine, work)
                event = EventPatternDatabase(engine, time_limit=2.0)
                self.assertTrue(event.event_enabled, event.event_reason)
                self.check_schedule_bounds(engine, event)

    def test_fractional_operation_durations_and_nonunit_quantum(self):
        problem = load_problem("tasks/s1_reach_handover.toml")
        problem.library = replace(problem.library, skills=tuple(
            replace(skill, op_time=skill.op_time + 0.371) for skill in problem.library.skills))
        engine = Engine(problem)
        for quantum in (0.6, 1.0):
            with self.subTest(quantum=quantum):
                database = EventPatternDatabase(engine, quantum=quantum, time_limit=2.0)
                self.assertTrue(database.event_enabled, database.event_reason)
                self.check_schedule_bounds(engine, database)

    def test_too_large_quantum_disables_event_bound_and_keeps_safe_work_bound(self):
        engine = Engine(load_problem("tasks/s1_reach_handover.toml"))
        database = EventPatternDatabase(engine, quantum=100.0, time_limit=2.0)
        self.assertTrue(database.enabled, database.reason)
        self.assertFalse(database.event_enabled)
        self.check_schedule_bounds(engine, database)

    def test_event_budget_failure_keeps_bound_admissible(self):
        engine = Engine(load_problem("tasks/s1_reach_handover.toml"))
        database = EventPatternDatabase(engine, max_states=50, time_limit=2.0)
        self.assertFalse(database.event_enabled)
        self.check_schedule_bounds(engine, database)

    def test_ready_insert_can_wait_for_a_longer_support_action(self):
        data = {
            "name": "wait_ready", "arms": ["A", "B"],
            "arm": [{"id": arm, "home": [0, 0, 0], "speed": 1.0} for arm in ("A", "B")],
            "location": [{"id": "stock", "kind": "stock", "pos": [0, 0, 0]}],
            "block": [{"id": "p", "type": "pillar", "installed_in": "p1"},
                      {"id": "b", "type": "beam", "at": "stock"}],
            "slot": [{"id": "p1", "type": "pillar", "pos": [0, 0, 0]},
                     {"id": "beam", "type": "beam", "pos": [0, 0, 0], "below": ["p1"],
                      "overhang": True, "anchors": ["cw"], "support_point": True},
                     {"id": "cw", "type": "counter", "pos": [0, 0, 0]}],
            "capabilities": {"reach": {"default": "VALID"}, "liftable": {"default": "VALID"}},
        }
        problem = problem_from_data(data)
        durations = {"support_on": 5.0, "insert": 1.1}
        problem.library = replace(problem.library, skills=tuple(
            replace(skill, op_time=durations.get(skill.name, skill.op_time))
            for skill in problem.library.skills))
        problem.initial = (problem.initial - {parse_fact("free(B)"), parse_fact("at(b, stock)")}
                           | {parse_fact("held(B, b)")})
        problem.goal = (parse_condition("filled(beam)"),)
        engine = Engine(problem)
        database = EventPatternDatabase(engine, time_limit=2.0)
        self.assertTrue(database.event_enabled, database.event_reason)
        self.assertTrue(any(any(remaining == 0 for _, remaining in running)
                            and any(remaining > 0 for _, remaining in running)
                            for _, running in database.event_ids))
        self.assertLessEqual(database(engine.initial), 5.0)
        self.check_schedule_bounds(engine, database)


if __name__ == "__main__":
    unittest.main()
