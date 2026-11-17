"""Fixed-seed admissibility checks against unpruned concrete event-state UCS."""

import json
from pathlib import Path
import random
from time import monotonic
import unittest

from skillnet.core.engine import Engine
from skillnet.domains.blocks import load_problem
from skillnet.planning.search import _search


TASKS = (
    "s1_parallel_place", "s1_nearest_arm", "s1_reach_handover",
    "s1_far_handover", "s2_tower", "s2_tower_same_type",
    "s3a_platform_demo_needs_support",
)
SEED = 1101
SAMPLES_PER_TASK = 36


class SampledAdmissibilityTests(unittest.TestCase):
    def test_fixed_seed_ucs_states_have_admissible_actual_search_heuristic(self):
        from skillnet.planning.heuristic import build_heuristic

        random_source = random.Random(SEED)
        results = []
        checked = 0
        checked_running = 0
        began = monotonic()
        for name in TASKS:
            engine = Engine(load_problem(f"tasks/{name}.toml"))
            engine.problem.proposal = None
            observed = {}

            def observe(state, running, elapsed):
                observed.setdefault((state, running), elapsed)

            baseline = _search(engine, observer=observe)
            self.assertEqual(baseline.status, "solved", name)
            population = list(observed)
            sampled = random_source.sample(population, min(SAMPLES_PER_TASK, len(population)))
            heuristic = build_heuristic(engine)
            nonzero = False
            solved = 0
            for index, (state, running) in enumerate(sampled):
                with self.subTest(task=name, sample=index, running=running):
                    exact = _search(engine, start=state, initial_running=running)
                    self.assertIn(exact.status, ("solved", "no_plan"),
                                  "真实值续搜必须完成，不能将预算耗尽视为无穷大")
                    estimate = heuristic(state, running)
                    if exact.status == "solved":
                        self.assertLessEqual(estimate, exact.makespan + 1e-6)
                        solved += 1
                    self.assertGreaterEqual(estimate, 0.0)
                    nonzero |= estimate > 1e-8
                    checked += 1
                    checked_running += bool(running)
                    results.append({
                        "task": name, "sample": index,
                        "state": [str(fact) for fact in sorted(state)],
                        "running": [{"action": str(engine.actions[key]), "remaining": remaining}
                                    for key, remaining in running],
                        "elapsed_before_sample": observed[(state, running)],
                        "heuristic": estimate if estimate != float("inf") else "inf",
                        "exact_status": exact.status,
                        "exact_remaining": exact.makespan if exact.status == "solved" else "inf",
                        "exact_expanded": exact.expanded,
                    })
            self.assertTrue(nonzero, f"{name} 的采样必须实际检验非零启发值")
            self.assertGreater(solved, 0, f"{name} 必须包含有限真实剩余时间的采样")
        self.assertGreaterEqual(checked, 200)
        self.assertGreater(checked_running, 0, "必须覆盖有执行中动作的完整事件状态")
        output = Path("out/v1_1_admissibility.json")
        output.parent.mkdir(exist_ok=True)
        output.write_text(json.dumps({
            "seed": SEED, "samples_per_task": SAMPLES_PER_TASK,
            "checked": checked, "checked_running": checked_running,
            "elapsed": monotonic() - began, "results": results,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
