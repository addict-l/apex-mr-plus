"""Independently replay event schedules against the engine contracts."""
import unittest
from pathlib import Path

from skillnet.core.engine import Engine
from skillnet.domains.blocks import load_problem
from skillnet.capability.oracle import Status
from skillnet.cost.model import DistanceCost
from skillnet.planning.search import search


def replay(test, engine, schedule):
    events = []
    for i, step in enumerate(schedule.steps):
        events.extend(((step.start, 1, i), (step.end, 0, i)))
    state = engine.initial
    running = {}
    events.sort()
    index = 0
    while index < len(events):
        time, kind, _ = events[index]
        if kind == 0:
            while index < len(events) and events[index][:2] == (time, 0):
                _, _, i = events[index]
                step = running.pop(i)
                state = engine.finish(step.action, state)
                index += 1
            test.assertIsNone(engine.hooks.invariant(state))
            continue
        _, _, i = events[index]
        step = schedule.steps[i]
        test.assertEqual(step.state, state)
        test.assertTrue(engine.satisfied(step.action.pre, state))
        test.assertEqual(engine.oracle.evaluate(step.action.capability)[0], Status.VALID)
        test.assertAlmostEqual(step.end - step.start, DistanceCost(engine.hooks).estimate(step.action, state), places=7)
        for other in running.values():
            test.assertFalse(set(step.action.occupies) & set(other.action.occupies))
            test.assertFalse(step.action.structural and other.action.structural)
        state = engine.start(step.action, state)
        test.assertIsNone(engine.hooks.invariant(state))
        running[i] = step
        index += 1
    test.assertFalse(running)
    test.assertTrue(engine.satisfied(engine.goal, state))
    return state


class ScheduleReplayTests(unittest.TestCase):
    def test_basic_schedules_replay_to_goal(self):
        for path in sorted(Path("tasks").glob("s[12]_*.toml")):
            with self.subTest(task=path.name):
                engine = Engine(load_problem(path))
                schedule = search(engine)
                self.assertEqual(schedule.status, "solved")
                replay(self, engine, schedule)
