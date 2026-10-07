"""#559 能量接線的手算桌面、頻率軸與控制組場景。"""
from __future__ import annotations

from aosr.config.directivity_defaults import TwoParameterCurve
from aosr.config.frequency_axis import GEOMETRIC_LANE_FREQUENCIES_HZ
from aosr.geometry.shoebox import Point, Room, Wall
from aosr.geometry.furniture import FurnitureKind
from aosr.physics import furniture_scene as scene
from aosr.physics.report_furniture import AbsoluteFurniture
from aosr.physics.report_source import SourceModelKind, SourceModelSpec
from tests.engine._directivity import DIRECTIVITY
from tests.engine._precision_contracts import contract_value

ROOM = Room(6.0, 4.0, 3.0)
SOURCE = Point(2.4, 2.0, 1.05)
RECEIVER = Point(3.6, 2.0, 1.05)
FREQUENCIES = (125.0, 500.0, 2000.0)
SPEED = 343.0
DENSITY = 1.2
RHO_C = DENSITY * SPEED
WALLS = {wall: 1646.4 for wall in Wall.all()}
CONTACT_REL = contract_value("furniture_geometry_contact")
OMNI = SourceModelSpec(SourceModelKind.OMNIDIRECTIONAL)
CONTROL_FREQUENCIES = tuple(GEOMETRIC_LANE_FREQUENCIES_HZ[i] for i in (20, 80, 140))
CONTROL_CURVES = (DIRECTIVITY.two_parameter, TwoParameterCurve(
    beta_limit=2.0, beta_corner_hz=700.0, beta_exponent=1.5,
    power_floor_limit_db=-15.0, power_floor_corner_hz=1200.0, power_floor_exponent=2.0))


def desk(identifier: str = "desk") -> AbsoluteFurniture:
    return AbsoluteFurniture(furniture_id=identifier, kind=FurnitureKind.DESK, material="wood",
        width_m=0.8, depth_m=1.6, height_m=0.1,
        bottom_center_m=(3.0, 2.0, 0.5), yaw_deg=0.0)


def lane_inputs(items: tuple[AbsoluteFurniture, ...],
                frequencies: tuple[float, ...] = FREQUENCIES) -> scene.FurnitureLaneInputs:
    furniture, margin = scene.furniture_scene(items, (ROOM.Lx, ROOM.Ly, ROOM.Lz), contact_rel=CONTACT_REL)
    # 手算題用已知逐頻阻抗，不由材料反推產生答案。
    return scene.FurnitureLaneInputs(furniture, margin,
        tuple((item, (1646.4,) * len(frequencies)) for item in items))


def analytic(curve: TwoParameterCurve = DIRECTIVITY.two_parameter) -> SourceModelSpec:
    return SourceModelSpec(SourceModelKind.ANALYTIC_AXISYMMETRIC_TWO_PARAMETER_V1, curve, RECEIVER)
