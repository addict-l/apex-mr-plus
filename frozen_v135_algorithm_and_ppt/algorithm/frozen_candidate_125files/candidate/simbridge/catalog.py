"""Validated physical block definitions, independent of task planning rules."""
from dataclasses import dataclass
import math
from pathlib import Path
import tomllib


DEFAULT_CATALOG = Path(__file__).resolve().parents[1] / "skills" / "blocks_catalog.toml"


def _vector(value, size, label):
    result = tuple(float(x) for x in value)
    if len(result) != size or not all(math.isfinite(x) for x in result):
        raise ValueError(f"{label} 必须包含 {size} 个有限数值")
    return result


@dataclass(frozen=True)
class BlockSpec:
    name: str
    size: tuple[float, float, float]
    mass: float
    color: tuple[float, float, float]
    grasp_offset: tuple[float, float, float]
    grasp_quaternion: tuple[float, float, float, float]
    insert_direction: tuple[float, float, float]


class BlockCatalog:
    def __init__(self, specs):
        self.specs = dict(specs)

    def __getitem__(self, name):
        try:
            return self.specs[name]
        except KeyError:
            raise ValueError(f"物理目录缺少积木类型：{name}") from None

    def require(self, names):
        for name in names:
            self[name]


def load_catalog(path=DEFAULT_CATALOG):
    rows = tomllib.loads(Path(path).read_text(encoding="utf-8-sig"))["type"]
    specs = {}
    for name, row in rows.items():
        size = _vector(row["size"], 3, f"{name}.size")
        mass = float(row["mass"])
        color = _vector(row["color"], 3, f"{name}.color")
        offset = _vector(row["grasp_offset"], 3, f"{name}.grasp_offset")
        quat = _vector(row["grasp_quaternion"], 4, f"{name}.grasp_quaternion")
        direction = _vector(row.get("insert_direction", [0, 0, -1]), 3,
                            f"{name}.insert_direction")
        if min(size) <= 0 or not math.isfinite(mass) or mass <= 0:
            raise ValueError(f"{name} 的尺寸和质量必须大于零")
        if any(x < 0 or x > 1 for x in color):
            raise ValueError(f"{name} 的颜色必须在 0 到 1 之间")
        if abs(sum(x*x for x in quat) - 1) > 1e-6:
            raise ValueError(f"{name} 的抓取四元数必须归一化")
        if abs(sum(x*x for x in direction) - 1) > 1e-6:
            raise ValueError(f"{name} 的插装方向必须归一化")
        specs[name] = BlockSpec(name, size, mass, color, offset, quat, direction)
    return BlockCatalog(specs)
