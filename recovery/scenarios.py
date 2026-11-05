"""Editable synthetic problem-1 fixtures. They specify NO recovery action chain.

All capability entries are assumptions for these small discrete models, not
observations from CoppeliaSim. The force/contact parameters are illustrative.
"""
from copy import deepcopy

from .model import Problem


def _physics(moment=3.):
    return {"gravity_m_s2": 9.81, "loads": [], "contacts": {},
            "connection": {"force_x_limit_N": 100., "force_z_limit_N": 100., "moment_limit_Nm": moment}}


def _empty_capabilities():
    return {arm: {op: {} for op in ("pick", "place", "install", "support", "handover", "work")}
            for arm in ("A", "B")}


def _pick_problem():
    capabilities = _empty_capabilities()
    for arm in ("A", "B"):
        capabilities[arm].update(pick={"cap@source": "VALID", "cap@buffer": "VALID"},
                                 place={"cap@buffer": "VALID"}, install={"fit_cap": "VALID"},
                                 handover={f"cap@{'B' if arm == 'A' else 'A'}": "VALID"})
    return {"name": "direct_swap", "parts": [{"id": "cap", "mass": .06}],
            "slots": ["source", "buffer"], "buffers": ["buffer"],
            "tasks": [{"id": "fit_cap", "kind": "install", "part": "cap", "original_arm": "B",
                       "dependencies": ["beam_ready"], "duration": 3., "insertion_force_N": 8., "load_x_m": .2}],
            "snapshot": {"locations": {"cap": "source"}, "supports": {}, "completed": ["beam_ready"]},
            "capabilities": capabilities, "physics": _physics(),
            "event": {"description": "B loses the declared fit_cap skill; other skills remain available",
                      "updates": [{"fact": "capabilities/B/install/fit_cap", "value": "INVALID"}]}}


def _support_problem():
    data = _pick_problem()
    data["name"] = "support_swap"
    data["snapshot"].update(locations={"cap": "held:B"}, supports={"A": "left"})
    physics = data["physics"] = _physics(.3)
    physics["loads"] = [{"fx_N": 0., "fz_N": -4.5, "x_m": .12, "z_m": 0.}]
    physics["connection"].update(reinforced_by="fit_cap", reinforced_moment_limit_Nm=3.)
    physics["contacts"] = {
        "left": {"arm": "A", "x_m": .18, "z_m": -.03, "normal_limit_N": 40., "friction": .6},
        "right": {"arm": "B", "x_m": .18, "z_m": -.03, "normal_limit_N": 40., "friction": .6}}
    data["capabilities"]["A"]["support"] = {"left": "VALID"}
    data["capabilities"]["B"]["support"] = {"right": "VALID"}
    return data


def _independent_problem():
    caps = _empty_capabilities()
    caps["A"]["work"] = {"block1": "VALID", "block2": "INVALID", "block3": "INVALID"}
    caps["B"]["work"] = {"block1": "INVALID", "block2": "VALID", "block3": "VALID"}
    return {"name": "independent_work", "parts": [], "slots": [], "buffers": [],
            "tasks": [{"id": tid, "kind": "work", "original_arm": arm, "duration": duration,
                       "dependencies": deps, "independent_of_assembly": True}
                      for tid, arm, duration, deps in [("block1", "A", 5., []),
                                                       ("block2", "B", 2., ["block1"]),
                                                       ("block3", "B", 3., [])]],
            "snapshot": {"locations": {}, "supports": {}, "completed": []},
            "capabilities": caps, "physics": _physics(),
            "event": {"description": "A cannot currently complete block1; no substitute is declared",
                      "updates": [{"fact": "capabilities/A/work/block1", "value": "UNKNOWN"}]}}


def cases() -> dict[str, dict]:
    direct = _pick_problem()
    held = deepcopy(direct)
    held["name"] = "held_handover"
    held["snapshot"]["locations"]["cap"] = "held:B"
    support = _support_problem()
    weak = deepcopy(support)
    weak["name"] = "weak_support"
    weak["physics"]["contacts"]["right"]["normal_limit_N"] = 2.
    blocked = deepcopy(support)
    blocked["name"] = "no_buffer"
    blocked["buffers"] = []
    unknown = deepcopy(direct)
    unknown["name"] = "unknown_substitute"
    unknown["event"]["updates"].append({"fact": "capabilities/A/install/fit_cap", "value": "UNKNOWN"})
    alternate = deepcopy(support)
    alternate["name"] = "alternative_contact"
    alternate["physics"]["contacts"]["right"]["normal_limit_N"] = 2.
    alternate["physics"]["contacts"]["right_far"] = {
        "arm": "B", "x_m": .25, "z_m": -.03, "normal_limit_N": 40., "friction": .6}
    alternate["capabilities"]["B"]["support"]["right_far"] = "VALID"
    independent = _independent_problem()
    return {case["name"]: case for case in (independent, direct, held, support, weak, blocked, unknown, alternate)}


def get_case(name: str) -> Problem:
    return Problem(cases()[name])
