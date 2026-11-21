"""Build and independently check the first v1.3 real workcell."""
import json
import math
from pathlib import Path
import shutil
import time

from apexmr.sim_connection import Simulator
from simbridge.frames import Frames
from simbridge.workcell import build


ROOT = Path(__file__).resolve().parents[1]


def main():
    if shutil.disk_usage("C:\\").free < 5 * 1024**3:
        raise RuntimeError("C 盘剩余不足 5 GB，按交接单停止")
    output = ROOT / "results/sim_v1_3/s1_parallel_place"
    output.mkdir(parents=True, exist_ok=True)
    owner = Simulator(ROOT, port=23210, timeout=30)
    report = {"milestone": "M1", "passed": False, "port": 23210}
    started = time.monotonic()
    try:
        owner.start(headless=True)
        report["pid"] = owner.process.pid
        client, sim = owner.connect()
        cfg = build(sim, client, ROOT / "tasks/s1_parallel_place.toml",
                    owner.install, output)
        frames = Frames(ROOT / "tasks/s1_parallel_place.toml")
        report["blocks"] = {}
        for name, handle in cfg["blocks"].items():
            expected = frames.block_pose(name)
            measured = sim.getObjectPose(handle, -1)
            error = math.dist(expected[:3], measured[:3])
            if error > 1e-6:
                raise AssertionError(f"{name} 实测位姿不符：{error} m")
            report["blocks"][name] = {"pose": measured, "error_m": error}
        report["home_errors"] = {side: {"m": arm["home_error_m"],
                                                 "rad": arm["home_error_rad"]}
                                   for side, arm in cfg["arms"].items()}
        if not (output / "workcell.ttt").is_file():
            raise AssertionError("缺少 workcell.ttt")
        report["passed"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        owner.close()
        report["closed"] = owner.process is not None and owner.process.poll() is not None
        report["elapsed_seconds"] = time.monotonic() - started
        report["log"] = str(owner.log_path)
        (output / "m1_validation.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
