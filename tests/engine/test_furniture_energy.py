"""家具能量接線：手算同調和、空房晚期不動與兩軸綁定。"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import pytest

from aosr.config.frequency_axis import GEOMETRIC_BAND_FREQUENCIES_HZ, GEOMETRIC_LANE_FREQUENCIES_HZ
from aosr.geometry.furniture import FaceDirection, FurnitureBox, FurnitureKind, Vec3
from aosr.geometry.shoebox import Point, Room
from aosr.physics.amplitude import Materials
from aosr.physics.furniture_paths import single_bounce_furniture_paths
from aosr.physics.furniture_scene import FurnitureLaneInputs, furniture_pressure_with_directivity, furniture_scene
from aosr.physics.geometric_lane import (
    GeometricEarlyResult, GeometricLaneResult, average_geometric_lane_to_bands_with_dense_early, solve_geometric_early_lane, solve_geometric_lane,
)
from aosr.physics.report_furniture import AbsoluteFurniture
from aosr.physics.report_source import SourceModelKind, SourceModelSpec, directivity_to_apply
from aosr.physics.room_paths import image_source_paths
from aosr.physics.source_directivity import SourceModel, apply_pressure_factor
from tests.engine import _furniture_energy_cases as case


def _solve(model: SourceModelSpec, *, furnished: bool = True) -> GeometricLaneResult:
    return solve_geometric_lane(source_model=model, room=case.ROOM, source=case.SOURCE,
        receiver=case.RECEIVER, sound_speed_m_s=case.SPEED, rho_c_pa_s_per_m=case.RHO_C,
        frequencies_hz=case.FREQUENCIES,
        impedance_by_wall={wall.wall_name(): value for wall, value in case.WALLS.items()},
        scattering_by_wall={wall.wall_name(): 0.36 for wall in case.WALLS}, reflection_order_k=1,
        furniture=case.lane_inputs((case.desk(),)) if furnished else None)


@pytest.mark.parametrize("model", (case.OMNI, case.analytic()))
def test_table_reflection_and_blocked_floor_match_hand_coherent_sum(model: SourceModelSpec) -> None:
    """桌面只取上面；地板鏡像線穿過桌板，其他一階牆路都在桌板上方。"""
    inputs = case.lane_inputs((case.desk(),))
    materials = Materials(case.RHO_C, case.FREQUENCIES,
        {wall.wall_name(): (complex(value),) * len(case.FREQUENCIES) for wall, value in case.WALLS.items()})
    paths = image_source_paths(case.ROOM, case.SOURCE, case.RECEIVER, case.SPEED, max_order=1, materials=materials)
    # 手挑地板：不呼叫被測接線的過濾函式。
    floor = next(path for path in paths if path.order == 1 and path.bounces[0].wall == "floor")
    assert floor.bounces[0].point == pytest.approx((3.0, 2.0, 0.0))
    kept = [path for path in paths if path is not floor]
    directed = directivity_to_apply(model)
    if directed is not None:
        kept = apply_pressure_factor(kept, case.RECEIVER, case.FREQUENCIES, SourceModel.TWO_PARAMETER,
            source=case.SOURCE, aim=directed[1], params=directed[0])
    direct = next(path for path in kept if path.order == 0)
    furniture_paths = single_bounce_furniture_paths(case.SOURCE.as_tuple(), case.RECEIVER.as_tuple(),
        inputs.furniture, c=case.SPEED, margin_m=inputs.margin_m)
    assert tuple(path.face_direction for path in furniture_paths) == (FaceDirection.TOP,)
    table = furniture_paths[0]
    assert table.hit == pytest.approx((3.0, 2.0, 0.6))
    pressure = furniture_pressure_with_directivity(table, case.FREQUENCIES, (1646.4,) * len(case.FREQUENCIES),
        rho_c=case.RHO_C, c=case.SPEED,
        model=SourceModel.OMNIDIRECTIONAL if directed is None else SourceModel.TWO_PARAMETER,
        source=case.SOURCE, aim=None if directed is None else directed[1], params=None if directed is None else directed[0])
    actual = _solve(model)
    for i, p_f in enumerate(pressure):
        p_w = sum(path.path_pressure[i] for path in kept if path.order == 1)
        p_r = (1.0 - 0.36) ** 0.5 * p_w + p_f
        assert actual.reflected_energy[i] == pytest.approx(abs(p_r) ** 2, rel=1e-13)
        assert actual.interference_energy[i] == pytest.approx(
            2.0 * (direct.path_pressure[i] * p_r.conjugate()).real, rel=1e-13)
    assert actual.furniture == (case.desk(),)


# 第二題手算：三階、聲源後方高沙發只擋「聲源→第一面牆」那段、兩件不同阻抗、
# 對準點≠接收點、第二條曲線、聲速≠343、密度≠1.2。幾何取自第三步物理審查的第三組。
_ROOM = Room(6.3, 4.1, 2.9)
_SOURCE, _RECEIVER, _AIM = Point(1.15, 0.95, 1.2), Point(4.35, 2.45, 1.05), Point(5.0, 3.0, 1.0)
_SPEED, _DENSITY = 340.0, 1.5
_WALLS = {"x0": 900.0, "xL": 2400.0, "y0": 1300.0, "yL": 5200.0, "floor": 700.0, "ceiling": 3100.0}
_FREQUENCIES = tuple(GEOMETRIC_LANE_FREQUENCIES_HZ[i] for i in (0, 60, 111, 150, 190, 219))
_MODEL = SourceModelSpec(SourceModelKind.ANALYTIC_AXISYMMETRIC_TWO_PARAMETER_V1, case.CONTROL_CURVES[1], _AIM)
# 宣告順序故意跟代號序相反；兩件阻抗不同且隨頻率變。
_GLASS_DESK = AbsoluteFurniture(furniture_id="glassdesk", kind=FurnitureKind.DESK, material="glass", width_m=1.0,
    depth_m=0.6, height_m=0.04, bottom_center_m=(3.3, 2.0, 0.7), yaw_deg=0.0)
_BACK_SOFA = AbsoluteFurniture(furniture_id="back", kind=FurnitureKind.SOFA, material="leather", width_m=0.4,
    depth_m=0.7, height_m=1.5, bottom_center_m=(0.4, 0.95, 0.0), yaw_deg=0.0)


def _pieces(frequencies: tuple[float, ...]) -> tuple[tuple[AbsoluteFurniture, tuple[float, ...]], ...]:
    return ((_GLASS_DESK, tuple(4000.0 + 17000.0 * 300.0 / (300.0 + f) for f in frequencies)),
            (_BACK_SOFA, tuple(800.0 + 0.25 * f for f in frequencies)))


def _crosses_box(start: Vec3, end: Vec3, box: FurnitureBox, margin_m: float) -> bool:
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


# 報表的密軸只走早期那一支；整條真的密軸也跑，守「依頻率軸換做法」這種改壞。
@pytest.mark.parametrize(("solve", "frequencies"), (
    (solve_geometric_lane, _FREQUENCIES),
    (solve_geometric_early_lane, _FREQUENCIES),
    (solve_geometric_early_lane, GEOMETRIC_BAND_FREQUENCIES_HZ),
), ids=("lane-fine", "early-fine", "early-dense"))
def test_two_pieces_third_order_match_hand_coherent_sum(
        solve: Callable[..., GeometricEarlyResult | GeometricLaneResult], frequencies: tuple[float, ...]) -> None:
    rho_c, scattering, pieces = _DENSITY * _SPEED, 0.3, _pieces(frequencies)
    furniture, margin_m = furniture_scene(tuple(item for item, _ in pieces),
        (_ROOM.Lx, _ROOM.Ly, _ROOM.Lz), contact_rel=case.CONTACT_REL)
    inputs = FurnitureLaneInputs(furniture, margin_m, pieces)
    materials = Materials(rho_c, frequencies,
        {wall: (complex(value),) * len(frequencies) for wall, value in _WALLS.items()})
    paths = image_source_paths(_ROOM, _SOURCE, _RECEIVER, _SPEED, max_order=3, materials=materials)
    kept, source_leg_only = [], []
    for path in paths:
        corners = (_SOURCE.as_tuple(), *(bounce.point for bounce in reversed(path.bounces)), _RECEIVER.as_tuple())
        hits = [any(_crosses_box(start, end, piece.box, margin_m) for piece in furniture)
                for start, end in zip(corners, corners[1:])]
        if not any(hits):
            kept.append(path)
        elif path.order >= 2 and hits[0] and not any(hits[1:]):
            source_leg_only.append(path)
    assert source_leg_only, "幾何要有二階以上、只被聲源那段擋住的牆面路徑"
    kept = apply_pressure_factor(kept, _RECEIVER, frequencies, SourceModel.TWO_PARAMETER,
        source=_SOURCE, aim=_AIM, params=case.CONTROL_CURVES[1])
    direct = next(path for path in kept if path.order == 0)
    furniture_paths = single_bounce_furniture_paths(_SOURCE.as_tuple(), _RECEIVER.as_tuple(), furniture,
        c=_SPEED, margin_m=margin_m)
    assert {path.furniture_id for path in furniture_paths} == {"back", "glassdesk"}
    impedances = {item.furniture_id: values for item, values in pieces}
    furniture_pressures = [furniture_pressure_with_directivity(path, frequencies, impedances[path.furniture_id],
        rho_c=rho_c, c=_SPEED, model=SourceModel.TWO_PARAMETER, source=_SOURCE, aim=_AIM,
        params=case.CONTROL_CURVES[1]) for path in furniture_paths]
    actual = solve(source_model=_MODEL, room=_ROOM, source=_SOURCE, receiver=_RECEIVER,
        sound_speed_m_s=_SPEED, rho_c_pa_s_per_m=rho_c, frequencies_hz=frequencies, impedance_by_wall=_WALLS,
        scattering_by_wall={wall: scattering for wall in _WALLS}, reflection_order_k=3, furniture=inputs)
    for i in range(len(frequencies)):
        p_r = sum((1.0 - scattering) ** (path.order / 2.0) * path.path_pressure[i] for path in kept if path.order)
        p_r += sum(pressure[i] for pressure in furniture_pressures)
        p_d = direct.path_pressure[i]
        assert actual.reflected_energy[i] == pytest.approx(abs(p_r) ** 2, rel=1e-12)
        assert actual.interference_energy[i] == pytest.approx(2.0 * (p_d * p_r.conjugate()).real,
            rel=1e-12, abs=1e-12 * abs(p_d) * abs(p_r))


@pytest.mark.parametrize("model", (case.OMNI, case.analytic()))
def test_furniture_leaves_late_energy_bitwise_unchanged(model: SourceModelSpec) -> None:
    plain, furnished = _solve(model, furnished=False), _solve(model)
    assert tuple(value.hex() for value in furnished.late_energy) == tuple(value.hex() for value in plain.late_energy)
    assert furnished.scattering == plain.scattering
    assert furnished.reflected_energy != plain.reflected_energy


def test_blocked_direct_names_furniture_before_totals() -> None:
    blocker = case.desk("blocking-desk").model_copy(update={"height_m": 1.0})
    with pytest.raises(ValueError, match="直達.*blocking-desk"):
        solve_geometric_early_lane(source_model=case.OMNI, room=case.ROOM, source=case.SOURCE,
            receiver=case.RECEIVER, sound_speed_m_s=case.SPEED, rho_c_pa_s_per_m=case.RHO_C,
            frequencies_hz=case.FREQUENCIES,
            impedance_by_wall={wall.wall_name(): value for wall, value in case.WALLS.items()},
            furniture=case.lane_inputs((blocker,)))


def test_band_average_refuses_different_furniture_on_two_axes() -> None:
    fine = _solve(case.OMNI)
    dense = solve_geometric_early_lane(source_model=case.OMNI, room=case.ROOM, source=case.SOURCE,
        receiver=case.RECEIVER, sound_speed_m_s=case.SPEED, rho_c_pa_s_per_m=case.RHO_C,
        frequencies_hz=case.FREQUENCIES,
        impedance_by_wall={wall.wall_name(): value for wall, value in case.WALLS.items()}, reflection_order_k=1)
    with pytest.raises(ValueError, match="細軸.*密軸.*家具"):
        average_geometric_lane_to_bands_with_dense_early(fine, dense,
            reflection_order_k=1, band_centers_hz=case.FREQUENCIES)
    dense = replace(dense, furniture=(case.desk("different"),))
    with pytest.raises(ValueError, match="家具"):
        average_geometric_lane_to_bands_with_dense_early(fine, dense,
            reflection_order_k=1, band_centers_hz=case.FREQUENCIES)
