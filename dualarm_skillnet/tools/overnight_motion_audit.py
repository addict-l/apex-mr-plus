"""Summarize persisted real-run motion evidence without querying a simulator."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scipy.spatial.transform import Rotation
from simbridge.frames import SIDE_GRASP_PITCH


def read(path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def structure(tasks):
    def key(task):
        row = task.get("metadata", {})
        return (task["arm"], task["kind"], task.get("part"), row.get("action"),
                row.get("source_action"), row.get("source_request"))
    keys = {task["id"]: key(task) for task in tasks}
    rows = [(key(task), sorted((keys[dep] for dep in task.get("dependencies", [])), key=str))
            for task in tasks]
    return sorted(rows, key=str)


def audit(directory, baseline=None):
    directory = Path(directory).resolve()
    if not directory.is_relative_to(ROOT):
        raise ValueError("证据目录必须位于项目内")
    tasks = read(directory/"tasks.json", [])
    summary = read(directory/"summary.json", {})
    segments = []
    for task in tasks:
        metadata = task.get("metadata", {})
        path = task.get("q_path", [])
        delta = max((abs(a-b) for first, second in zip(path, path[1:])
                     for a, b in zip(first, second)), default=0.)
        target = metadata.get("target_pose")
        pitch = None
        if metadata.get("request_kind") == "linear" and target:
            z = Rotation.from_quat(target[3:]).apply([0., 0., 1.])[2]
            pitch = math.degrees(math.asin(max(-1., min(1., -float(z)))))
        segments.append({"id": task["id"], "arm": task["arm"],
                         "action": metadata.get("source_action"),
                         "request": metadata.get("source_request"),
                         "request_kind": metadata.get("request_kind"),
                         "samples": len(path), "max_joint_change_rad": delta,
                         "tcp_down_pitch_degrees": pitch,
                         "yaw_rad": metadata.get("yaw_rad"),
                         "grasp_strategy": metadata.get("grasp_strategy"),
                         "linear_certificate": metadata.get("linear_certificate")})
    retreat = [row for row in segments if str(row["action"]).startswith("handover(")
               and row["request_kind"] == "linear"]
    placement = [row for row in segments if str(row["action"]).startswith("place(")
                 and row["request"] == "r2"]
    result = {"scope": "仅汇总该目录实测/编译证据，不进行新仿真或重新判定验收",
              "summary_passed": summary.get("passed", False),
              "failure_phase": summary.get("failure_phase"), "reason": summary.get("reason"),
              "configured_side_pitch_degrees": math.degrees(SIDE_GRASP_PITCH),
              "code_sha256": {name: hashlib.sha256((ROOT/"simbridge"/name).read_bytes()).hexdigest()
                              for name in ("frames.py", "compiler.py", "primitives.py", "backend.py")},
              "task_count": len(tasks), "segments": segments,
              "two_handover_retreats_compiled": len(retreat) == 2,
              "handover_retreats": retreat, "placement_descents": placement,
              "validation": read(directory/"validation.json"),
              "execution_validation": read(directory/"execution_validation.json")}
    if baseline:
        previous = read(Path(baseline)/"tasks.json", [])
        result["parallel_structure_regression"] = {
            "baseline": str(baseline), "baseline_count": len(previous),
            "current_count": len(tasks),
            "same_structure": bool(tasks and previous) and structure(tasks) == structure(previous),
            "catalog_top_only": bool(tasks) and all(
                row["grasp_strategy"] in (None, "catalog_top") for row in segments)}
    (directory/"overnight_motion_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description="汇总已保存的运动与验收证据")
    parser.add_argument("directory", type=Path)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    result = audit(args.directory, args.baseline)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in {"segments", "validation", "execution_validation"}},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
