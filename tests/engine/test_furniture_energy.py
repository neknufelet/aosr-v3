"""家具能量接線：手算同調和、空房晚期不動與兩軸綁定。"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import pytest

from aosr.config.frequency_axis import GEOMETRIC_BAND_FREQUENCIES_HZ
from aosr.geometry.furniture import FaceDirection
from aosr.physics.amplitude import Materials
from aosr.physics.furniture_paths import single_bounce_furniture_paths
from aosr.physics.furniture_scene import furniture_pressure_with_directivity
from aosr.physics.geometric_lane import (
    GeometricEarlyResult, GeometricLaneResult, average_geometric_lane_to_bands_with_dense_early, solve_geometric_early_lane, solve_geometric_lane,
)
from aosr.physics.report_source import SourceModelSpec, directivity_to_apply
from aosr.physics.room_paths import image_source_paths
from aosr.physics.source_directivity import SourceModel, apply_pressure_factor
from tests.engine import _furniture_energy_cases as case, _furniture_third_order_case as third


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


# 報表的密軸只走早期那一支；整條真的密軸也跑，守「依頻率軸換做法」這種改壞。
@pytest.mark.parametrize(("solve", "frequencies"), (
    (solve_geometric_lane, third.FREQUENCIES),
    (solve_geometric_early_lane, third.FREQUENCIES),
    (solve_geometric_early_lane, GEOMETRIC_BAND_FREQUENCIES_HZ),
), ids=("lane-fine", "early-fine", "early-dense"))
def test_two_pieces_third_order_match_hand_coherent_sum(
        solve: Callable[..., GeometricEarlyResult | GeometricLaneResult], frequencies: tuple[float, ...]) -> None:
    inputs = third.lane_inputs(frequencies)
    paths = image_source_paths(third.ROOM, third.SOURCE, third.RECEIVER, third.SPEED, max_order=3,
        materials=third.materials(frequencies))
    kept, _, source_leg_only = third.split_blocked(paths, inputs)
    assert source_leg_only, "幾何要有二階以上、只被聲源那段擋住的牆面路徑"
    kept = apply_pressure_factor(kept, third.RECEIVER, frequencies, SourceModel.TWO_PARAMETER,
        source=third.SOURCE, aim=third.AIM, params=third.CURVE)
    direct = next(path for path in kept if path.order == 0)
    furniture_paths = single_bounce_furniture_paths(third.SOURCE.as_tuple(), third.RECEIVER.as_tuple(),
        inputs.furniture, c=third.SPEED, margin_m=inputs.margin_m)
    assert {path.furniture_id for path in furniture_paths} == {"back", "glassdesk"}
    impedances = {item.furniture_id: values for item, values in third.pieces(frequencies)}
    furniture_pressures = [furniture_pressure_with_directivity(path, frequencies, impedances[path.furniture_id],
        rho_c=third.RHO_C, c=third.SPEED, model=SourceModel.TWO_PARAMETER, source=third.SOURCE, aim=third.AIM,
        params=third.CURVE) for path in furniture_paths]
    actual = solve(source_model=third.MODEL, room=third.ROOM, source=third.SOURCE, receiver=third.RECEIVER,
        sound_speed_m_s=third.SPEED, rho_c_pa_s_per_m=third.RHO_C, frequencies_hz=frequencies,
        impedance_by_wall=third.WALLS, scattering_by_wall={wall: third.SCATTERING for wall in third.WALLS},
        reflection_order_k=3, furniture=inputs)
    for i in range(len(frequencies)):
        p_r = sum((1.0 - third.SCATTERING) ** (path.order / 2.0) * path.path_pressure[i]
                  for path in kept if path.order)
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
