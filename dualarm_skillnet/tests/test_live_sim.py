"""Explicitly gated end-to-end real-simulator acceptance cases."""
import os
from pathlib import Path
import time
import unittest
import uuid

from simbridge.runner import ROOT, run_task


@unittest.skipUnless(os.environ.get("SKILLNET_LIVE_SIM") == "1",
                     "设置 SKILLNET_LIVE_SIM=1 才启动真实仿真")
class LiveSimTests(unittest.TestCase):
    def check_task(self, name):
        output = ROOT / "results/sim_v1_3/live_tests" / (
            time.strftime("%Y%m%d-%H%M%S")+"-"+uuid.uuid4().hex[:8])
        result = run_task(ROOT / "tasks" / (name+".toml"), out=output)
        self.assertTrue(result["passed"],
                        f"{result.get('failure_phase')}：{result.get('reason')}；证据 {output/Path(name).name}")

    def test_s1_parallel_place(self):
        self.check_task("s1_parallel_place")

    def test_s1_reach_handover(self):
        self.check_task("s1_reach_handover")

    def test_s2_tower(self):
        self.check_task("sim/s2_tower_sim")

    def test_s3a_cantilever(self):
        self.check_task("sim/s3a_cantilever_sim")

    def test_s3a_platform(self):
        self.check_task("sim/s3a_platform_sim")


if __name__ == "__main__":
    unittest.main()
