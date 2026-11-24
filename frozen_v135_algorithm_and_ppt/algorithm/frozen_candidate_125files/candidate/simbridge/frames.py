"""Task reference planes and physically stacked block-center/TCP poses."""
from pathlib import Path
import math
import tomllib

from scipy.spatial.transform import Rotation

from .catalog import load_catalog


BOARD_TOP = 0.772
TABLE_TOP = 0.760
APPROACH = 0.10
# Side grasps approach a vertical face from above at this fixed downward pitch.
# A horizontal side grasp leaves the wrist at block height, which collides with
# the baseplate at every board location (v1.3.2, FIX_v1.3.2.md §1).
# Overnight real pick evidence: at 45 degrees the original shoe front plane
# intersects the brick by 0.368 mm. The permitted 60-degree alternative leaves
# 8.091 mm; the same binding and pitch are used for every side-grasp action.
SIDE_GRASP_PITCH = math.radians(60)
SUPPORT_PAD_SIZE = (.05, .06, .008)
SUPPORT_SENSOR_RISE = .014
SUPPORT_PAD_TOP_OFFSET = SUPPORT_SENSOR_RISE + SUPPORT_PAD_SIZE[2]/2
SUPPORT_SENSOR_PALM_Z = .035
SUPPORT_TCP_PALM_Z = .071
SUPPORT_AXIAL_COMPENSATION = SUPPORT_TCP_PALM_Z-SUPPORT_SENSOR_PALM_Z


def task_to_world(position):
    if len(position) != 3:
        raise ValueError("任务坐标必须包含三个数值")
    x, y, z = (float(x) for x in position)
    return [x, y, BOARD_TOP + z]


def world_to_task(position):
    x, y, z = (float(x) for x in position)
    return [x, y, z - BOARD_TOP]


def read_task(task):
    if isinstance(task, dict):
        return task
    # A loaded Problem preserves all scene rows on its domain hooks.
    if hasattr(task, "hooks"):
        hooks = task.hooks
        return {"name": task.name, "location": list(hooks.locations.values()),
                "slot": list(hooks.slots.values()), "block": list(hooks.blocks.values()),
                "arm": [{"id": side, "home": hooks.positions[f"home_{side}"],
                         "speed": hooks.speeds[side]} for side in hooks.arms]}
    return tomllib.loads(Path(task).read_text(encoding="utf-8-sig"))


class Frames:
    def __init__(self, task, catalog=None, *, arm_config=None):
        self.task = read_task(task)
        self.catalog = catalog or load_catalog()
        self.arm_config = arm_config or {}
        self.blocks = {row["id"]: row for row in self.task.get("block", [])}
        self.locations = {row["id"]: row for row in self.task.get("location", [])}
        self.slots = {row["id"]: row for row in self.task.get("slot", [])}
        self.catalog.require(row["type"] for row in (*self.blocks.values(), *self.slots.values()))
        self.references = {name: task_to_world(row["pos"])
                           for name, row in {**self.locations, **self.slots}.items()}
        self._slot_centers = {}
        for name in self.slots:
            self.slot_pose(name)

    def slot_pose(self, name, _visiting=None):
        """A root slot rests on the board; each child rests on its lower top.

        Input slot z values express symbolic layers in old tasks (30 mm),
        while physical heights come exclusively from the catalog. Their exact
        task-to-world reference positions remain available in ``references``.
        """
        if name in self._slot_centers:
            return list(self._slot_centers[name])
        visiting = set(_visiting or ())
        if name in visiting:
            raise ValueError(f"槽位 below 关系成环：{name}")
        visiting.add(name)
        row = self.slots[name]
        lower_tops = [self.slot_pose(lower, visiting)[2]
                      + self.catalog[self.slots[lower]["type"]].size[2] / 2
                      for lower in row.get("below", [])]
        support_top = max(lower_tops, default=BOARD_TOP)
        x, y, _ = self.references[name]
        center = [x, y, support_top + self.catalog[row["type"]].size[2] / 2, 0, 0, 0, 1]
        self._slot_centers[name] = center
        return list(center)

    def block_pose(self, block, place=None):
        row = self.blocks[block]
        place = place or row.get("installed_in") or row.get("at")
        if place in self.slots:
            return self.slot_pose(place)
        if place not in self.locations:
            raise ValueError(f"积木 {block} 缺少有效的初始位置：{place}")
        x, y, z = self.references[place]
        return [x, y, z + self.catalog[row["type"]].size[2] / 2, 0, 0, 0, 1]

    def grasp_pose(self, block, place=None, arm="A"):
        spec = self.catalog[self.blocks[block]["type"]]
        body = self.block_pose(block, place)
        # A transfer needs accessible opposed faces. Use the same side binding
        # from the first pick through transfer and final place, without regrasp.
        if any(row["kind"] == "handover" for row in self.locations.values()):
            return self.tcp_pose(body, self.side_payload(block, arm))
        offset = Rotation.from_quat(body[3:]).apply(spec.grasp_offset)
        quat = (Rotation.from_quat(body[3:])
                * Rotation.from_quat(spec.grasp_quaternion)).as_quat().tolist()
        # A pi world-z rotation keeps the two native grippers opposed.
        if arm == "B":
            quat = (Rotation.from_euler("z", 3.141592653589793)
                    * Rotation.from_quat(quat)).as_quat().tolist()
        return [body[i] + float(offset[i]) for i in range(3)] + quat

    def payload(self, block, arm="A", pose=None, body_pose=None):
        """Measured-mode convention: block transform relative to TCP."""
        body = body_pose or self.block_pose(block)
        tcp = pose or self.grasp_pose(block, arm=arm)
        inverse = Rotation.from_quat(tcp[3:]).inv()
        return {"name": block,
                "offset": inverse.apply([body[i]-tcp[i] for i in range(3)]).tolist(),
                "quaternion": (inverse*Rotation.from_quat(body[3:])).as_quat().tolist()}

    def side_payload(self, block, arm, pitch=SIDE_GRASP_PITCH):
        # Fixed source/destination bindings use opposed north/south faces:
        # A approaches from +world-y, B from -world-y. At handover the body yaw
        # rotates both bindings together to orient retreats towards the bases.
        # The tool z axis points at the face and is pitched downward by
        # ``pitch`` about the finger-closing axis (tool x = world x), so the
        # TCP sits outside and above the face while the block stays level.
        # Matrix columns are the tool axes expressed in the (level) body frame.
        c, s = math.cos(pitch), math.sin(pitch)
        matrix = ([[-1, 0, 0], [0, s, -c], [0, -c, -s]] if arm == "A"
                  else [[1, 0, 0], [0, -s, c], [0, -c, -s]])
        rotation = Rotation.from_matrix(matrix)
        spec = self.catalog[self.blocks[block]["type"]]
        return {"name": block, "offset": [0., 0., spec.size[1]/2+.006],
                "quaternion": rotation.inv().as_quat().tolist()}

    @staticmethod
    def tcp_pose(body_pose, payload):
        body_rotation = Rotation.from_quat(body_pose[3:])
        tcp_rotation = body_rotation*Rotation.from_quat(payload["quaternion"]).inv()
        offset = tcp_rotation.apply(payload["offset"])
        return [body_pose[i]-float(offset[i]) for i in range(3)] + tcp_rotation.as_quat().tolist()

    def handover_poses(self, block, location, giver, taker, giver_payload=None, *, arm_config=None):
        """Choose the airborne body yaw so both grippers retreat toward bases.

        Payload bindings stay fixed. Only the body orientation at the transfer
        point changes; source and destination body poses are unaffected.
        """
        if self.locations[location]["kind"] != "handover":
            raise ValueError(f"位置 {location} 不是交接点")
        arms = self.arm_config if arm_config is None else arm_config
        bases = {}
        for side in (giver, taker):
            base = arms.get(side, {}).get("base_pose")
            if base is None or len(base) not in (3, 7) or not all(math.isfinite(float(x)) for x in base):
                raise ValueError(f"交接偏航选择缺少臂 {side} 的有效世界基座位姿")
            bases[side] = [float(x) for x in base[:3]]
        giver_binding = giver_payload or self.side_payload(block, giver)
        taker_binding = self.side_payload(block, taker)
        center = self.references[location]
        candidates = []
        for index in range(4):
            yaw = index*math.pi/2
            body = center+Rotation.from_euler("z", yaw).as_quat().tolist()
            poses = {giver: self.tcp_pose(body, giver_binding),
                     taker: self.tcp_pose(body, taker_binding)}
            angles = {}
            for side in (giver, taker):
                direction = Rotation.from_quat(poses[side][3:]).apply([0., 0., -1.])
                toward_base = [bases[side][i]-center[i] for i in range(2)]
                denominator = math.hypot(*direction[:2])*math.hypot(*toward_base)
                if denominator <= 1e-12:
                    angles[side] = math.pi
                else:
                    cosine = sum(a*b for a, b in zip(direction[:2], toward_base))/denominator
                    angles[side] = math.acos(max(-1., min(1., cosine)))
            candidates.append({"yaw_rad": yaw, "retreat_angles_rad": angles,
                               "sum_angle_rad": sum(angles.values()),
                               "poses": poses, "body_pose": body})
        inward = [row for row in candidates if all(
            angle <= math.pi/2+1e-12 for angle in row["retreat_angles_rad"].values())]
        if inward:
            chosen = min(inward, key=lambda row: (
                row["retreat_angles_rad"][giver], row["sum_angle_rad"], row["yaw_rad"]))
            reason = "两臂退让均不背离基座的候选中，优先最小化交出臂水平夹角，再比较夹角之和"
        else:
            chosen = min(candidates, key=lambda row: (
                row["sum_angle_rad"], row["retreat_angles_rad"][giver], row["yaw_rad"]))
            reason = "两臂朝向约束冲突，选择退让方向与基座方向的水平夹角之和最小的偏航"
        selection = {"yaw_rad": chosen["yaw_rad"], "reason": reason,
                     "base_positions": bases,
                     "retreat_angles_rad": chosen["retreat_angles_rad"],
                     "candidates": [{key: value for key, value in row.items()
                                     if key not in {"poses", "body_pose"}} for row in candidates]}
        return chosen["poses"], chosen["body_pose"], taker_binding, selection

    def support_point(self, slot):
        """Choose a bearing point near the end opposite the lower column."""
        body = self.slot_pose(slot)
        size = self.catalog[self.slots[slot]["type"]].size
        lower = self.slots[slot].get("below", [])
        if not lower:
            raise ValueError(f"托举槽位 {slot} 缺少下层结构，无法确定悬出端")
        centers = [self.slot_pose(name) for name in lower]
        delta = [body[i]-sum(row[i] for row in centers)/len(centers) for i in (0, 1)]
        if math.hypot(*delta) <= 1e-9:
            # Old symbolic fixtures have coincident centers; their longest
            # body axis supplies an explicit, documented fallback direction.
            axis = 0 if size[0] >= size[1] else 1
            direction = [float(i == axis) for i in (0, 1)]
        else:
            length = math.hypot(*delta)
            direction = [value/length for value in delta]
        edge = min(size[i]/(2*abs(value)) for i, value in enumerate(direction)
                   if abs(value) > 1e-12)
        distance = max(0., edge-.04)
        return [body[0]+direction[0]*distance, body[1]+direction[1]*distance,
                body[2]-size[2]/2]

    def support_pose(self, slot, arm, *, arm_config=None):
        point = self.support_point(slot)
        arms = self.arm_config if arm_config is None else arm_config
        base = arms.get(arm, {}).get("base_pose")
        if base is None:
            raise ValueError(f"托举位姿缺少臂 {arm} 的世界基座位姿")
        delta = [point[i]-base[i] for i in (0, 1)]
        length = math.hypot(*delta)
        if length <= 1e-9:
            raise ValueError("托举支撑点不能与基座水平重合")
        dx, dy = (value/length for value in delta)
        # Local y points down, local z points horizontally into the workcell.
        rotation = Rotation.from_matrix([[dy, 0, dx], [-dx, 0, dy], [0, -1, 0]])
        # The TCP assembly is unchanged. The shoe is 36 mm behind it, so the
        # commanded TCP extends by that amount along tool z to keep contact.
        return [point[0]+SUPPORT_AXIAL_COMPENSATION*dx,
                point[1]+SUPPORT_AXIAL_COMPENSATION*dy,
                point[2]-SUPPORT_PAD_TOP_OFFSET]+rotation.as_quat().tolist()

    @staticmethod
    def support_approach_pose(pose, distance=APPROACH):
        return Frames.support_retreat(pose, distance)

    @staticmethod
    def handover_retreat(pose, arm, distance=APPROACH):
        result = list(pose)
        direction = Rotation.from_quat(pose[3:]).apply([0., 0., 1.])
        for index in range(3):
            result[index] -= distance*float(direction[index])
        return result

    @staticmethod
    def support_retreat(pose, distance=APPROACH):
        result = list(pose)
        direction = Rotation.from_quat(pose[3:]).apply([0., 0., 1.])
        for index in range(3):
            result[index] -= distance*float(direction[index])
        return result

    @staticmethod
    def approach_pose(pose, distance=APPROACH):
        result = list(pose)
        result[2] += distance
        return result

    def home_pose(self, arm):
        row = next(row for row in self.task["arm"] if row["id"] == arm)
        return task_to_world(row["home"]) + ([1, 0, 0, 0] if arm == "A" else [0, 1, 0, 0])
