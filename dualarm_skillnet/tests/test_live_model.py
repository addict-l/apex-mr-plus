"""Opt-in integration with the locally installed, real Qwen model."""

import os
import unittest

from skillnet.decompose.ollama import OllamaDecomposer
from skillnet.domains.blocks import load_problem
from skillnet.pipeline import run


@unittest.skipUnless(os.environ.get("SKILLNET_LIVE_MODEL") == "1",
                     "仅在 SKILLNET_LIVE_MODEL=1 时运行真实模型集成测试")
class LiveModelTests(unittest.TestCase):
    def test_real_proposal_is_parsed_and_preserves_the_optimal_plan(self):
        problem = load_problem("tasks/s1_reach_handover.toml")
        problem.proposal = None
        baseline = run(problem)
        adapter = OllamaDecomposer()
        result = run(problem, decomposer=adapter, max_model_rounds=0)
        self.assertEqual(baseline.status, "solved")
        self.assertEqual(result.status, "solved")
        self.assertIsNotNone(result.proposal)
        self.assertIn("Ollama", result.proposal.source)
        self.assertTrue(adapter.calls)
        self.assertTrue(adapter.calls[-1]["json_valid"])
        self.assertLessEqual(abs(result.schedule.makespan - baseline.schedule.makespan), 1e-6)


if __name__ == "__main__":
    unittest.main()
