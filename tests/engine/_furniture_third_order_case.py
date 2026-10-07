"""#559 第二組手算幾何：能量路與路徑表共用，避開預設值才咬得到接錯參數。

三階；聲源後方高沙發只擋「聲源→第一面牆」那段；兩件不同阻抗且隨頻率變、宣告順序跟代號序相反；
對準點≠接收點、第二條曲線、聲速≠343、密度≠1.2。幾何取自第三步物理審查的第三組。
"""
from __future__ import annotations

from aosr.config.frequency_axis import GEOMETRIC_LANE_FREQUENCIES_HZ
from aosr.geometry.furniture import FurnitureBox, FurnitureKind, Vec3
from aosr.geometry.shoebox import Point, Room
from aosr.physics.amplitude import Materials
from aosr.physics.furniture_scene import FurnitureLaneInputs, furniture_scene
from aosr.physics.report_furniture import AbsoluteFurniture
from aosr.physics.report_source import SourceModelKind, SourceModelSpec
from aosr.physics.room_paths import RoomPath
from tests.engine import _furniture_energy_cases as case

ROOM = Room(6.3, 4.1, 2.9)
SOURCE, RECEIVER, AIM = Point(1.15, 0.95, 1.2), Point(4.35, 2.45, 1.05), Point(5.0, 3.0, 1.0)
SPEED, DENSITY = 340.0, 1.5
RHO_C = DENSITY * SPEED
SCATTERING = 0.3
WALLS = {"x0": 900.0, "xL": 2400.0, "y0": 1300.0, "yL": 5200.0, "floor": 700.0, "ceiling": 3100.0}
FREQUENCIES = tuple(GEOMETRIC_LANE_FREQUENCIES_HZ[i] for i in (0, 60, 111, 150, 190, 219))
CURVE = case.CONTROL_CURVES[1]
MODEL = SourceModelSpec(SourceModelKind.ANALYTIC_AXISYMMETRIC_TWO_PARAMETER_V1, CURVE, AIM)
GLASS_DESK = AbsoluteFurniture(furniture_id="glassdesk", kind=FurnitureKind.DESK, material="glass", width_m=1.0,
    depth_m=0.6, height_m=0.04, bottom_center_m=(3.3, 2.0, 0.7), yaw_deg=0.0)
BACK_SOFA = AbsoluteFurniture(furniture_id="back", kind=FurnitureKind.SOFA, material="leather", width_m=0.4,
    depth_m=0.7, height_m=1.5, bottom_center_m=(0.4, 0.95, 0.0), yaw_deg=0.0)


def pieces(frequencies: tuple[float, ...]) -> tuple[tuple[AbsoluteFurniture, tuple[float, ...]], ...]:
    return ((GLASS_DESK, tuple(4000.0 + 17000.0 * 300.0 / (300.0 + f) for f in frequencies)),
            (BACK_SOFA, tuple(800.0 + 0.25 * f for f in frequencies)))


def lane_inputs(frequencies: tuple[float, ...]) -> FurnitureLaneInputs:
    given = pieces(frequencies)
    furniture, margin_m = furniture_scene(tuple(item for item, _ in given),
        (ROOM.Lx, ROOM.Ly, ROOM.Lz), contact_rel=case.CONTACT_REL)
    return FurnitureLaneInputs(furniture, margin_m, given)


def materials(frequencies: tuple[float, ...]) -> Materials:
    return Materials(RHO_C, frequencies, {wall: (complex(value),) * len(frequencies) for wall, value in WALLS.items()})


def crosses_box(start: Vec3, end: Vec3, box: FurnitureBox, margin_m: float) -> bool:
    """考卷自己的分軸夾區間：線段穿進內縮界線的盒子內部才算擋。"""
    enter, leave = 0.0, 1.0
    for axis in range(3):
        low, high = box.minimum_m[axis] + margin_m, box.maximum_m[axis] - margin_m
        step = end[axis] - start[axis]
        if step == 0.0:
            if not low < start[axis] < high:
                return False
            continue
        near, far = sorted(((low - start[axis]) / step, (high - start[axis]) / step))
        enter, leave = max(enter, near), min(leave, far)
        if enter >= leave:
            return False
    return True


def split_blocked(paths: list[RoomPath], inputs: FurnitureLaneInputs
                  ) -> tuple[list[RoomPath], list[RoomPath], list[RoomPath]]:
    """回 (留下, 被擋, 二階以上只被聲源那段擋住)；不呼叫被測的過濾，順序跟輸入相同。"""
    kept: list[RoomPath] = []
    blocked: list[RoomPath] = []
    source_leg_only: list[RoomPath] = []
    for path in paths:
        corners = (SOURCE.as_tuple(), *(bounce.point for bounce in reversed(path.bounces)), RECEIVER.as_tuple())
        hits = [any(crosses_box(start, end, piece.box, inputs.margin_m) for piece in inputs.furniture)
                for start, end in zip(corners, corners[1:])]
        (blocked if any(hits) else kept).append(path)
        if path.order >= 2 and hits[0] and not any(hits[1:]):
            source_leg_only.append(path)
    return kept, blocked, source_leg_only
