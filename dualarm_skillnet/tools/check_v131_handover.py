"""Independently summarize the required real handover evidence."""
import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    output = args.directory.resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("证据目录必须位于项目内")
    read = lambda name: json.loads((output/name).read_text(encoding="utf-8"))
    summary, tasks, audit = read("summary.json"), read("tasks.json"), read("validation.json")
    selected = [task for task in tasks if task["metadata"].get("source_skill") == "handover"]
    if len(selected) != 6:
        raise AssertionError("交接必须包含六个运动/握持请求")
    retreats = {}
    for task in selected:
        metadata = task["metadata"]
        if not metadata.get("yaw_selection_reason") or "yaw_rad" not in metadata:
            raise AssertionError("缺少偏航选择证据")
        if metadata["source_request"] in {"r5", "r6"}:
            path = task["q_path"]
            maximum = max((abs(a-b) for left, right in zip(path, path[1:])
                           for a, b in zip(left, right)), default=0.)
            certificate = metadata["linear_certificate"]
            if maximum > .25 or certificate["max_branch_delta_rad"] != .25:
                raise AssertionError("退让关节连续性证书未通过")
            if certificate["position_tolerance_m"] != .002 or certificate["orientation_tolerance_rad"] != .02:
                raise AssertionError("直线阈值发生变化")
            retreats[metadata["source_request"]] = {"arm": task["arm"],
                "max_adjacent_joint_delta_rad": maximum, "certificate": certificate}
    if set(retreats) != {"r5", "r6"}:
        raise AssertionError("缺少两臂退让")
    if not summary.get("passed") or not audit.get("success"):
        raise AssertionError("端到端验收未通过")
    report = {"passed": True, "yaw_rad": selected[0]["metadata"]["yaw_rad"],
              "selection": selected[0]["metadata"]["handover_frame_selection"],
              "retreats": retreats, "final_pose_validation": audit,
              "wall_seconds": summary["wall_seconds"]}
    (output/"handover_revision_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"passed": True, "yaw_rad": report["yaw_rad"], "retreats": retreats},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
