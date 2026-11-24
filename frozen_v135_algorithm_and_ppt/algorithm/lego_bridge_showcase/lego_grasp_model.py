"""Rigid grasp conventions for the 8 mm pitch display workcell."""
from itertools import pairwise
import math

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

PIER_DIMS_M = (.0158, .0158, .016)
DECK_DIMS_M = (.0958, .0318, .0032)
PIER_TCP_HEIGHT_M = .0087
DECK_TCP_HEIGHT_M = .002300
JAW_MIN_M = .006
JAW_MAX_M = .040
WALL_CONTACT_TOLERANCE_M = .0001


def matrix(pose):
    """Accept a Coppelia world pose, row-major 3x4, or homogeneous matrix."""
    arr = np.asarray(pose, dtype=float)
    if not np.isfinite(arr).all():
        raise ValueError("pose must be finite")
    if arr.shape == (4, 4):
        return arr.copy()
    out = np.eye(4)
    if arr.shape == (3,):
        out[:3, 3] = arr
    elif arr.size == 12:
        out[:3, :] = arr.reshape(3, 4)
    elif arr.shape == (7,):
        out[:3, 3] = arr[:3]
        out[:3, :3] = Rotation.from_quat(arr[3:]).as_matrix()
    else:
        raise ValueError("unexpected pose shape")
    return out


def to_pose(transform):
    transform = matrix(transform)
    return [*transform[:3, 3], *Rotation.from_matrix(transform[:3, :3]).as_quat()]


def jaw_q(aperture_m):
    if not math.isfinite(aperture_m) or not JAW_MIN_M-1e-12 <= aperture_m <= JAW_MAX_M+1e-12:
        raise ValueError("jaw aperture is outside 6–40 mm")
    return min((JAW_MAX_M-JAW_MIN_M)/2, max(0., (aperture_m-JAW_MIN_M)/2))


def translate(transform, vector):
    out = matrix(transform)
    out[:3, 3] += np.asarray(vector, dtype=float)
    return out


def _grasp(bottom, axis, tip, offset):
    relative = np.eye(4)
    relative[:3, 0] = axis
    relative[:3, 1] = np.cross(tip, axis)
    relative[:3, 2] = tip
    relative[:3, 3] = offset
    return matrix(bottom) @ relative


def deck_grasp(bottom_pose, direction=1, height_m=DECK_TCP_HEIGHT_M):
    if direction not in (-1, 1):
        raise ValueError("jaw direction must be +1 or -1")
    return _grasp(bottom_pose, [0., direction, 0.], [0., 0., -1.],
                  [0., 0., height_m])


def pier_taker_grasp(bottom_pose, direction=1, height_m=PIER_TCP_HEIGHT_M):
    if direction not in (-1, 1):
        raise ValueError("jaw direction must be +1 or -1")
    return _grasp(bottom_pose, [direction, 0., 0.], [0., 0., -1.],
                  [0., 0., height_m])


def pier_giver_grasp(bottom_pose, tilt_deg, incline, height_m=.009, direction=1):
    """Tilt the Y-facing grasp from vertical toward the center of the table."""
    if incline not in (-1, 1) or direction not in (-1, 1) or not 45. <= tilt_deg <= 65.:
        raise ValueError("giver tilt must be 45–65 degrees with a signed direction")
    tilt = math.radians(tilt_deg)
    # Keep the tilted press shoe outboard of the studs. The actual
    # pad overlap on the sidewalls is checked from the native meshes.
    tcp_x = -incline * .005
    return _grasp(bottom_pose, [0., direction, 0.],
                  [incline * math.sin(tilt), 0., -math.cos(tilt)],
                  [tcp_x, 0., height_m])


def bad_y_pier_grasp(bottom_pose):
    return _grasp(bottom_pose, [0., 1., 0.], [0., 0., -1.], [0., 0., PIER_TCP_HEIGHT_M])


def pose_error(expected, actual):
    a, b = matrix(expected), matrix(actual)
    return (float(np.linalg.norm(a[:3, 3]-b[:3, 3])),
            float((Rotation.from_matrix(a[:3, :3]).inv() *
                   Rotation.from_matrix(b[:3, :3])).magnitude()))


def linear_targets(start, goal, *, step_m=.001, angle_step_rad=math.radians(.5)):
    """Yield every straight TCP waypoint except the known current endpoint."""
    if not 0. < step_m <= .01 or not 0. < angle_step_rad <= .05:
        raise ValueError("sweep discretization is too coarse")
    a, b = matrix(start), matrix(goal)
    distance, angle = pose_error(a, b)
    count = max(1, math.ceil(distance/step_m), math.ceil(angle/angle_step_rad))
    rotations = Slerp([0., 1.], Rotation.from_matrix(np.stack([a[:3, :3], b[:3, :3]])))
    for k in range(1, count+1):
        fraction = k/count
        sample = np.eye(4)
        sample[:3, :3] = rotations([fraction]).as_matrix()[0]
        sample[:3, 3] = a[:3, 3]+fraction*(b[:3, 3]-a[:3, 3])
        yield sample


def _clip(polygon, axis, bound, keep_greater):
    if not len(polygon):
        return []
    result = []
    for p, q in pairwise([*polygon, polygon[0]]):
        p_in = bool(p[axis] >= bound if keep_greater else p[axis] <= bound)
        q_in = bool(q[axis] >= bound if keep_greater else q[axis] <= bound)
        if p_in:
            result.append(p)
        if p_in != q_in:
            result.append(p+(q-p)*((bound-p[axis])/(q[axis]-p[axis])))
    return result


def pad_wall_receipt(sim, pad_handles, block_bottom, dimensions, axis):
    """Certify exact face signs, zero inward crossing, and effective footprint.

    The wall is clipped against each actual pad triangle within the calibrated
    0.1 mm air-side face band. Block and pad mesh poses come from CoppeliaSim.
    """
    b = matrix(block_bottom)
    width = float(dimensions[axis])
    if axis not in (0, 1) or len(pad_handles) != 2:
        raise ValueError("expected two real pads on an opposed X or Y face")
    across = 1-axis
    half_face = float(dimensions[across])/2
    height = float(dimensions[2])
    rows = []
    for pad in pad_handles:
        handle = int(pad["handle"] if isinstance(pad, dict) else pad)
        raw = sim.getShapeMesh(handle)
        points = np.asarray(raw[0], dtype=float).reshape(-1, 3)
        polygons = np.asarray(raw[1], dtype=int).reshape(-1, 3)
        if not len(points) or not len(polygons):
            raise ValueError("empty pad mesh")
        rel = np.linalg.inv(b) @ matrix(sim.getObjectMatrix(handle, sim.handle_world))
        local = points @ rel[:3, :3].T+rel[:3, 3]
        sign = 1 if float(np.mean(local[:, axis])) > 0 else -1
        clearance = float(np.min(sign*local[:, axis])-width/2)
        plane = sign*width/2
        clipped = []
        area = 0.
        for face in local[polygons]:
            polygon = [p for p in face]
            polygon = _clip(polygon, axis, plane-sign*1e-8, sign > 0)
            polygon = _clip(polygon, axis, plane+sign*WALL_CONTACT_TOLERANCE_M, sign < 0)
            for clip_axis, bound, greater in ((across, -half_face, True),
                    (across, half_face, False), (2, 0., True), (2, height, False)):
                polygon = _clip(polygon, clip_axis, bound, greater)
            if len(polygon) >= 3:
                planar = np.asarray(polygon)[:, [across, 2]]
                u, v = planar.T
                triangle_area = abs(float(u @ np.roll(v, -1)-v @ np.roll(u, -1)))/2
                if triangle_area > 1e-13:
                    area += triangle_area
                    clipped.extend(planar[:, 1].tolist())
        span = [min(clipped), max(clipped)] if clipped else None
        rows.append({"pad": handle, "sign": sign, "wall_clearance_m": clearance,
                     "contact_band_m": WALL_CONTACT_TOLERANCE_M,
                     "contact_height_from_bottom_m": span, "contact_height_m": span[1]-span[0] if span else 0.,
                     "pad_tip_height_from_bottom_m": float(np.min(local[:, 2])),
                     "overlap_area_m2": area,
                     "valid": -1e-8 <= clearance <= WALL_CONTACT_TOLERANCE_M and area > 1e-10})
    return {"ok": all(row["valid"] for row in rows) and {row["sign"] for row in rows} == {-1, 1},
            "closing_axis": ("X", "Y")[axis], "pads": rows}
