"""Motion quality checks for the demo's unwrapped UR5 joint paths.

These checks reject large base/wrist backtracking. They are a presentation
quality policy, separate from collision and dynamics validation.
"""
from __future__ import annotations

import math


def path_metrics(path):
    """Measure actual joint travel without hiding full turns via angle wrapping."""
    q = [list(map(float, row)) for row in path]
    if not q or any(len(row) != 6 or not all(map(math.isfinite, row)) for row in q):
        raise ValueError("a path needs finite six-joint configurations")
    increments = [[b - a for a, b in zip(left, right)] for left, right in zip(q, q[1:])]
    travel = [sum(abs(delta[axis]) for delta in increments) for axis in range(6)]
    net = [abs(b - a) for a, b in zip(q[0], q[-1])]
    return {
        "joint_arc_length_rad": sum(math.sqrt(sum(x * x for x in delta)) for delta in increments),
        "joint_travel_deg": list(map(math.degrees, travel)),
        "joint_excess_deg": [math.degrees(max(0., total - direct)) for total, direct in zip(travel, net)],
    }


def compact_route(path):
    """Allow required reorientation, but reject over 180 degrees of backtracking."""
    metrics = path_metrics(path)
    return all(metrics["joint_excess_deg"][axis] <= 180. for axis in (0, 5))
