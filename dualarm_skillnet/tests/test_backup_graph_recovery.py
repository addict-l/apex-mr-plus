"""End-to-end regressions for backup selection and candidate-graph repair."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic
import unittest

from skillnet.capability.oracle import Status
from skillnet.core.engine import Engine
from skillnet.core.logic import parse_fact
from skillnet.domains.blocks import load_problem
from skillnet.planning.backup_plans import replay, serialize_steps
from skillnet.planning.candidate_graph import CandidateGraph
from skillnet.planning.search import _search
from skillnet.recovery import prepare, recover


SKILLS = Path("skills/blocks_v1.toml")
FLAT = Path("tasks/s1_nearest_arm.toml")
SEARCH_BUDGET = {"time_limit": 10., "max_expansions": 10_000}


class BackupGraphRecoveryTests(unittest.TestCase):
    def prepare_bundle(self, task, root, graph_limit=1_000):
        bundle = root / "bundle"
        prepared = prepare(task, SKILLS, bundle, graph_limit=graph_limit, checkpoints=8,
                           backups=2, backup_seconds=1., **SEARCH_BUDGET)
        self.assertEqual(prepared["status"], "solved")
        self.assertGreater(prepared["backups"], 0)
        data = json.loads((bundle / "plans.json").read_text(encoding="utf-8"))
        graph = bundle / f'graph-{data["graph_sha256"]}.json'
        return bundle, data, graph

    def test_backup_hit_skips_loading_candidate_graph(self):
        with TemporaryDirectory() as directory:
            bundle, prepared, graph = self.prepare_bundle(FLAT, Path(directory))
            graph.unlink()
            fault = {"state": prepared["checkpoints"][0]["state"],
                     "disabled_skills": {"A": ["place"]}}
            repaired = recover(FLAT, SKILLS, bundle, fault, compare=True, **SEARCH_BUDGET)

        self.assertEqual((repaired["status"], repaired["stage"]), ("solved", "backup"))
        self.assertEqual(repaired["backup_choice"]["kind"], "backup")
        self.assertFalse(repaired["graph_loaded"])
        self.assertEqual(repaired["expanded"], 0)
        self.assertIsNone(repaired["suboptimality_bound"])
        self.assertEqual(repaired["makespan"], repaired["full_search"]["makespan"])

    def test_all_backups_fail_and_cached_graph_search_finds_handover(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            task = root / "handover.toml"
            source = FLAT.read_text(encoding="utf-8")
            task.write_text(source.replace("[capabilities.reach]", """[[location]]
id = "handover_c"
kind = "handover"
pos = [0.0, 0.0, 0.1]

[capabilities.reach]""").replace('"stock_c", "board_c"]', '"stock_c", "board_c", "handover_c"]'),
                            encoding="utf-8")
            bundle, prepared, graph = self.prepare_bundle(task, root, graph_limit=2)
            fault = {"state": prepared["checkpoints"][0]["state"],
                     "disabled_skills": {"A": ["place"], "B": ["pick"]}}
            repaired = recover(task, SKILLS, bundle, fault, compare=True, **SEARCH_BUDGET)
            graph.write_bytes(graph.read_bytes()+b" ")
            with self.assertRaisesRegex(ValueError, "候选图校验失败"):
                recover(task, SKILLS, bundle, fault, **SEARCH_BUDGET)

        self.assertEqual((repaired["status"], repaired["stage"]), ("solved", "graph_search"))
        self.assertTrue(repaired["backup_attempts"])
        self.assertTrue(all(attempt["status"] != "valid" for attempt in repaired["backup_attempts"]))
        self.assertTrue(repaired["graph_loaded"])
        self.assertGreater(repaired["graph"]["reused_states"], 0)
        self.assertGreater(repaired["graph"]["reused_edges"], 0)
        self.assertGreater(repaired["graph"]["generated_rows"], 0)
        self.assertEqual(repaired["makespan"], repaired["full_search"]["makespan"])
        self.assertEqual(repaired["suboptimality_bound"], 1.)
        self.assertIn("handover(A, B, red, handover_c)",
                      [step["action"] for step in repaired["schedule"]])

    def test_only_changed_capability_evidence_is_refreshed(self):
        old = Engine(load_problem(FLAT))
        changed = parse_fact("reach(B, board_c)")
        old.oracle.update(changed, Status.UNKNOWN, "旧")
        offline = CandidateGraph(old, 100)
        b_place = old.parse_action("place(B, red, board_c)")
        a_place = old.parse_action("place(A, red, board_c)")
        self.assertEqual(offline.evaluate(b_place)[0], Status.UNKNOWN)
        offline.evaluate(a_place)
        self.assertEqual(_search(old, candidate_graph=offline).status, "solved")

        current = Engine(load_problem(FLAT))
        online = CandidateGraph(current, 100, saved=offline.to_dict(), changed={changed})
        self.assertEqual(online.evaluate(current.actions[b_place.key])[0], Status.VALID)
        self.assertEqual(online.evaluate(current.actions[a_place.key])[0], Status.VALID)
        self.assertEqual(online.stats()["refreshed_capabilities"], 1)
        self.assertEqual(online.stats()["reused_capabilities"], 1)
        cached = _search(current, candidate_graph=online)
        cold = _search(current)
        self.assertEqual((cached.status, cached.makespan), (cold.status, cold.makespan))

    def test_replay_preserves_parallel_completion_semantics(self):
        engine = Engine(load_problem("tasks/s1_parallel_place.toml"))
        nominal = _search(engine)
        result, status = replay(engine, serialize_steps(nominal.steps),
                                allowed=frozenset(engine.actions), protected=frozenset(),
                                cuts=frozenset(), deadline=monotonic()+3.)
        self.assertEqual(status, "valid")
        self.assertEqual(result.makespan, nominal.makespan)
        self.assertEqual([step.start for step in result.steps], [step.start for step in nominal.steps])
