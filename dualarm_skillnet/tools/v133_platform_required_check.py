"""Compare the extra platform fixture using only structure and geometry."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from skillnet.core.engine import Engine
from skillnet.core.logic import Fact
from skillnet.domains.blocks import load_problem
from simbridge.layout import check_layout


ORIGINAL = ROOT/"tasks/sim/s3a_platform_sim.toml"
REQUIRED = ROOT/"tasks/sim/s3a_platform_required_sim.toml"


def check_required_platform(original=ORIGINAL, required=REQUIRED):
    original, required = Path(original), Path(required)
    before = tomllib.loads(original.read_text(encoding="utf-8-sig"))
    after = tomllib.loads(required.read_text(encoding="utf-8-sig"))
    normalized = deepcopy(after)
    descriptions_changed = all(before[field] != after[field]
                               for field in ("name", "description", "instruction"))
    for field in ("name", "description", "instruction"):
        normalized[field] = before[field]
    old_beam = next(row for row in before["slot"] if row["id"] == "beam")
    new_beam = next(row for row in normalized["slot"] if row["id"] == "beam")
    flag_changed = old_beam.get("support_point") is True and new_beam.get("support_point") is False
    new_beam["support_point"] = old_beam["support_point"]
    same_except_declared = normalized == before
    geometry_before, geometry_after = check_layout(original), check_layout(required)
    geometry_preserved = (geometry_before["metrics"] == geometry_after["metrics"]
                          and geometry_before["slot_poses"] == geometry_after["slot_poses"])
    engine = Engine(load_problem(required))
    support_actions = [action.text for action in engine.actions.values()
                       if action.name == "support_on" and action.args[-1] == "beam"]
    base = engine.initial | {Fact("filled", (slot,)) for slot in ("p0", "p1", "beam")}
    partial = {slot: bool(engine.hooks.evaluate("has_support", ("beam",),
                      base | {Fact("filled", (slot,))})) for slot in ("tp1", "tp2")}
    complete = base | {Fact("filled", (slot,)) for slot in ("tp1", "tp2")}
    full_support = bool(engine.hooks.evaluate("has_support", ("beam",), complete))
    missing_platform_rejected = engine.hooks.invariant(base) is not None
    complete_platform_valid = engine.hooks.invariant(complete) is None
    cleanup_goals = all(any(condition.negated and condition.fact == Fact("filled", (slot,))
                            for condition in engine.goal) for slot in ("tp1", "tp2"))
    no_reach_table = "reach" not in after.get("capabilities", {})
    checks = {"metadata_descriptions_changed": descriptions_changed,
              "beam_manual_support_disabled": flag_changed,
              "only_declared_differences": same_except_declared,
              "original_geometry_valid": geometry_before["valid"],
              "required_geometry_valid": geometry_after["valid"],
              "geometry_preserved": geometry_preserved,
              "no_handwritten_reach": no_reach_table,
              "no_grounded_beam_support_on": not support_actions,
              "partial_platform_not_support": not any(partial.values()),
              "complete_platform_is_support": full_support,
              "unanchored_beam_without_platform_rejected": missing_platform_rejected,
              "unanchored_beam_with_complete_platform_valid": complete_platform_valid,
              "both_temporary_slots_must_end_empty": cleanup_goals}
    return {"valid": all(checks.values()), "checks": checks,
            "original": str(original.relative_to(ROOT)), "required": str(required.relative_to(ROOT)),
            "original_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
            "required_sha256": hashlib.sha256(required.read_bytes()).hexdigest(),
            "allowed_text_changes": ["name", "description", "instruction"],
            "only_semantic_change": "slot[beam].support_point: true -> false",
            "grounded_beam_support_actions": support_actions,
            "partial_platform_has_support": partial,
            "geometry": [geometry_before, geometry_after],
            "geometry_identity_scope": "任务内积木、料垫、底板、台面、槽位；不包括依 support_point 选择的工具装配",
            "existing_workcell_tool_selection": {
                "original_horizontal_support": any(row.get("support_point") for row in before["slot"]),
                "required_horizontal_support": any(row.get("support_point") for row in after["slot"]),
                "dispatch_unchanged": True,
                "note": "workcell.build 无手托接触点时沿用原工具装配，未调用水平托板配置；未改生产规则"},
            "search_performed": False, "ik_performed": False, "dynamics_performed": False,
            "scope": "结构和纯几何检查；不证明独立求解、真实可达、托台受力或完整端到端通过"}


def main():
    report = check_required_platform()
    output = ROOT/"results/sim_v1_3/v133/platform_required_layout.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"valid": report["valid"], "checks": report["checks"],
                      "output": str(output.relative_to(ROOT)), "search_performed": False},
                     ensure_ascii=False, indent=2))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
