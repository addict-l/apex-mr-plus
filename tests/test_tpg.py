"""Behavioral checks for node-level TPG safety and event release semantics."""
import math
import unittest

from normal.tpg import build_tpg, simulate_tpg


def q(x):
    return [x, 0., 0., 0., 0., 0.]


def task(name, arm, kind="transit", duration=1., path=None, deps=(), order=0):
    row = dict(id=name, arm=arm, kind=kind, duration=duration,
               dependencies=list(deps), reference_order=order, part="test_part")
    if path is not None:
        row["q_path"] = path
    return row


class TemporalPlanGraphTests(unittest.TestCase):
    def test_path_partition_keeps_polyline_and_duration(self):
        original = [q(0), q(.03), q(.20), q(.11), q(.37)]
        tpg = build_tpg([task("a", "A", duration=4.5, path=original)])
        self.assertGreater(len(tpg["nodes"]), 3)
        self.assertAlmostEqual(sum(n["duration"] for n in tpg["nodes"]), 4.5)
        flattened = [sample for node in tpg["nodes"] for sample in node["q_samples"]]
        for sample in original:
            self.assertTrue(any(max(abs(a-b) for a,b in zip(sample, candidate)) < 1e-12 for candidate in flattened))
        for node in tpg["nodes"]:
            for axis in range(6):
                self.assertLessEqual(max(q[axis] for q in node["q_samples"])-min(q[axis] for q in node["q_samples"]), .10+1e-9)
        self.assertFalse(tpg["certificate"]["collision_oracle_supplied"])

    def test_cross_arm_conflict_uses_reference_order_and_completion(self):
        tasks = [task("a", "A", duration=1.2, path=[q(0),q(.2)], order=0),
                 task("b", "B", duration=.9, path=[q(0),q(.2)], order=1)]
        def oracle(a, b):
            return {"conflict": a["q_end"][0] > .1 and b["q_start"][0] < .1,
                    "evidence": {"test": "crossing_swept_ranges"}}
        tpg = build_tpg(tasks, oracle)
        self.assertTrue(tpg["graph_acyclic"])
        conflicts = [edge for edge in tpg["edges"] if edge["reason"] == "motion_conflict"]
        self.assertTrue(conflicts)
        by_id = {n["id"]:n for n in tpg["nodes"]}
        for edge in conflicts:
            before, after = by_id[edge["source"]], by_id[edge["target"]]
            self.assertLess(before["reference_index"], after["reference_index"])
            self.assertLessEqual(before["end"], after["start"]+1e-9)
        first = tpg["task_map"]["a"]["last_node"]
        delayed = simulate_tpg(tpg, {first:3.})
        for edge in tpg["edges"]:
            self.assertLessEqual(delayed["timestamps"][edge["source"]]["end"], delayed["timestamps"][edge["target"]]["start"]+1e-9)

    def test_support_start_and_release_remain_event_dependencies(self):
        tasks = [task("support", "B", "support_start", .4, [q(.1)], order=0),
                 task("pick", "A", "pick", .3, [q(.0)], order=1),
                 task("place", "A", "place", .6, [q(.0)], deps=("pick","support"), order=2),
                 task("release", "B", "support_end", .2, [q(.1)], deps=("place",), order=3)]
        tpg = build_tpg(tasks, lambda a,b:False)
        execution = simulate_tpg(tpg, {"support":2.7})
        rows = {row["id"]:row for row in execution["task_schedule"]}
        self.assertGreaterEqual(rows["place"]["start"], rows["support"]["end"])
        self.assertGreaterEqual(rows["release"]["start"], rows["place"]["end"])
        self.assertAlmostEqual(rows["support"]["duration"], 3.1)
        self.assertEqual(len([event for event in execution["events"] if event["event"]=="finish"]), len(tpg["nodes"]))

    def test_missing_oracle_does_not_claim_geometry_validation(self):
        tpg=build_tpg([task("a","A",path=[q(0)]),task("b","B",path=[q(0)],order=1)])
        self.assertFalse(tpg["certificate"]["motion_overlap_pairs_checked"])
        self.assertFalse(tpg["certificate"]["waiting_endpoints_verified"])

    def test_conflicting_reference_direction_is_rejected(self):
        tasks=[task("a","A",path=[q(0)]),task("b","B",path=[q(0)],order=1)]
        with self.assertRaisesRegex(ValueError,"serialization rejected"):
            build_tpg(tasks,lambda a,b:{"conflict":True,"allowed_orders":[[b["id"],a["id"]]]})

    def test_backward_reference_dependency_and_unknown_delay_are_errors(self):
        with self.assertRaisesRegex(ValueError,"Reference order"):
            build_tpg([task("a","A",path=[q(0)],deps=("b",)),task("b","B",path=[q(0)],order=1)])
        tpg=build_tpg([task("a","A",path=[q(0)])])
        with self.assertRaisesRegex(ValueError,"unknown"):
            simulate_tpg(tpg,{"not_a_task":1})
        with self.assertRaises(ValueError):
            simulate_tpg(tpg,{"a":-1})


if __name__=="__main__":
    unittest.main()
