"""幾何路細頻率軸、相位能量接法與頻帶平均的性質考卷。"""

from __future__ import annotations

import math
import re

import pytest

from aosr.config import frequency_axis as frequency_axis_config
from aosr.config.art_lane import ART_N_PER_WALL_DEFAULT
from aosr.geometry.shoebox import Point, Room, Wall
from aosr.materials.response import MATERIAL_SCATTERING_DEFAULT_S
from aosr.physics.amplitude import Materials
from aosr.physics.late_energy import LateEnergyInputs, solve_late_energy
from aosr.physics.room_paths import image_source_paths
from aosr.physics.totals import totals_from_paths


_ROOM = Room(Lx=6.0, Ly=4.0, Lz=3.0)
_SOURCE = Point(x=1.2, y=1.3, z=1.1)
_RECEIVER = Point(x=4.7, y=2.8, z=1.4)
_SOUND_SPEED_M_S = 343.0
_RHO_C_PA_S_PER_M = 411.6


def _constant_walls(value: complex) -> dict[str, complex]:
    return {wall: value for wall in Wall.wall_names()}


def _wall_rows(
    value: complex, frequencies_hz: tuple[float, ...]
) -> dict[str, tuple[complex, ...]]:
    return {
        wall: tuple(value for _frequency in frequencies_hz)
        for wall in Wall.wall_names()
    }


def _constant_scattering(value: float) -> dict[str, float]:
    return {wall: value for wall in Wall.wall_names()}


def test_geometric_axis_extends_the_single_formula_axis_past_fem() -> None:
    """另生幾何軸、漏掉 FEM 前綴或接錯 300 Hz 後第一點時必須紅。"""
    geometric_axis = frequency_axis_config.GEOMETRIC_LANE_FREQUENCIES_HZ
    fem_axis = frequency_axis_config.FEM_LANE_FREQUENCIES_HZ
    first_geometric_only = (
        frequency_axis_config.V3_AXIS_START_HZ
        * 2.0
        ** (
            len(fem_axis)
            / frequency_axis_config.V3_AXIS_POINTS_PER_OCTAVE
        )
    )

    assert geometric_axis[: len(fem_axis)] == fem_axis
    assert geometric_axis[len(fem_axis)] == first_geometric_only
    assert first_geometric_only > frequency_axis_config.FEM_GEOMETRIC_CROSSOVER_CAP_HZ
    assert geometric_axis[-1] <= frequency_axis_config.GEOMETRIC_AXIS_UPPER_HZ
    assert (
        frequency_axis_config.V3_AXIS_START_HZ
        * 2.0
        ** (
            len(geometric_axis)
            / frequency_axis_config.V3_AXIS_POINTS_PER_OCTAVE
        )
        > frequency_axis_config.GEOMETRIC_AXIS_UPPER_HZ
    )


@pytest.mark.parametrize("impedance_multiple", (4.0, 10.0), ids=("flat", "lowabs"))
def test_lane_reuses_existing_coherent_totals_and_late_energy_exactly(
    impedance_multiple: float,
) -> None:
    """改抄鏡像或晚期算法、改相位相加順序、或不是三階時必須紅。"""
    from aosr.physics import geometric_lane

    frequencies_hz = frequency_axis_config.GEOMETRIC_REPORT_OCTAVE_CENTERS_HZ
    impedance = complex(impedance_multiple * _RHO_C_PA_S_PER_M, 0.0)
    impedance_rows = _wall_rows(impedance, frequencies_hz)
    materials = Materials(
        rho_c=_RHO_C_PA_S_PER_M,
        frequencies_hz=frequencies_hz,
        walls=impedance_rows,
    )
    paths = image_source_paths(
        _ROOM,
        _SOURCE,
        _RECEIVER,
        _SOUND_SPEED_M_S,
        max_order=3,
        materials=materials,
    )
    expected_totals = totals_from_paths(paths)
    expected_late = solve_late_energy(
        LateEnergyInputs(
            room=_ROOM,
            rho_c_pa_s_per_m=_RHO_C_PA_S_PER_M,
            frequencies_hz=frequencies_hz,
            impedance_by_wall=impedance_rows,
            n_per_wall=ART_N_PER_WALL_DEFAULT,
            domain_alpha_bar_max=math.inf,
        )
    )

    actual = geometric_lane.solve_geometric_lane(
        room=_ROOM,
        source=_SOURCE,
        receiver=_RECEIVER,
        sound_speed_m_s=_SOUND_SPEED_M_S,
        rho_c_pa_s_per_m=_RHO_C_PA_S_PER_M,
        frequencies_hz=frequencies_hz,
        impedance_by_wall=_constant_walls(impedance),
    )

    assert actual.direct_energy == expected_totals.direct_energy
    assert actual.reflected_energy == expected_totals.reflected_energy
    assert actual.late_energy == tuple(
        band.late_reverberant_energy for band in expected_late.bands
    )


def test_scattering_endpoints_reduce_geometric_energy_exactly() -> None:
    """散射權重兩端若仍混入另一項能量，或接反反射／晚期，必須紅。"""
    from aosr.physics import geometric_lane

    impedances = _constant_walls(complex(4.0 * _RHO_C_PA_S_PER_M, 0.0))

    zero = geometric_lane.solve_geometric_lane(
        room=_ROOM,
        source=_SOURCE,
        receiver=_RECEIVER,
        sound_speed_m_s=_SOUND_SPEED_M_S,
        rho_c_pa_s_per_m=_RHO_C_PA_S_PER_M,
        frequencies_hz=(125.0,),
        impedance_by_wall=impedances,
        scattering_by_wall=_constant_scattering(0.0),
    )
    one = geometric_lane.solve_geometric_lane(
        room=_ROOM,
        source=_SOURCE,
        receiver=_RECEIVER,
        sound_speed_m_s=_SOUND_SPEED_M_S,
        rho_c_pa_s_per_m=_RHO_C_PA_S_PER_M,
        frequencies_hz=(125.0,),
        impedance_by_wall=impedances,
        scattering_by_wall=_constant_scattering(1.0),
    )

    assert zero.scattering == (0.0,)
    assert zero.geometric_energy == (
        zero.direct_energy[0] + zero.reflected_energy[0],
    )
    assert one.scattering == (1.0,)
    assert one.geometric_energy == (
        one.direct_energy[0] + one.late_energy[0],
    )


def test_missing_scattering_uses_the_material_default_curve() -> None:
    """缺省散射若不是 response 的正式常數，或 E_geo 漏任一項，必須紅。"""
    from aosr.physics import geometric_lane

    actual = geometric_lane.solve_geometric_lane(
        room=_ROOM,
        source=_SOURCE,
        receiver=_RECEIVER,
        sound_speed_m_s=_SOUND_SPEED_M_S,
        rho_c_pa_s_per_m=_RHO_C_PA_S_PER_M,
        frequencies_hz=(125.0,),
        impedance_by_wall=_constant_walls(
            complex(4.0 * _RHO_C_PA_S_PER_M, 0.0)
        ),
    )
    expected = (
        actual.direct_energy[0]
        + (1.0 - MATERIAL_SCATTERING_DEFAULT_S) * actual.reflected_energy[0]
        + MATERIAL_SCATTERING_DEFAULT_S * actual.late_energy[0]
    )

    assert actual.scattering == (MATERIAL_SCATTERING_DEFAULT_S,)
    assert actual.geometric_energy == (expected,)


def test_room_scattering_ignores_a_fully_absorbing_walls_scattering() -> None:
    """多材料合成若未以反射能量加權，全吸收牆的 s 會污染房間 s。"""
    from aosr.physics import geometric_lane

    reflective_wall = Wall.FLOOR.wall_name()
    impedances = {
        wall: complex(_RHO_C_PA_S_PER_M, 0.0) for wall in Wall.wall_names()
    }
    impedances[reflective_wall] = complex(4.0 * _RHO_C_PA_S_PER_M, 0.0)
    scattering = _constant_scattering(0.0)
    scattering[reflective_wall] = 0.75

    actual = geometric_lane.solve_geometric_lane(
        room=_ROOM,
        source=_SOURCE,
        receiver=_RECEIVER,
        sound_speed_m_s=_SOUND_SPEED_M_S,
        rho_c_pa_s_per_m=_RHO_C_PA_S_PER_M,
        frequencies_hz=(125.0,),
        impedance_by_wall=impedances,
        scattering_by_wall=scattering,
    )

    assert actual.scattering == (0.75,)


def test_room_scattering_uses_each_reflecting_walls_area_and_energy() -> None:
    """漏掉面積或把 x 牆面積誤寫成另一面尺寸時必須紅。"""
    from aosr.physics import geometric_lane

    floor = Wall.FLOOR.wall_name()
    x0 = Wall.X0.wall_name()
    floor_impedance = complex(3.0 * _RHO_C_PA_S_PER_M, 0.0)
    x0_impedance = complex(2.0 * _RHO_C_PA_S_PER_M, 0.0)
    impedances = _constant_walls(complex(_RHO_C_PA_S_PER_M, 0.0))
    impedances[floor] = floor_impedance
    impedances[x0] = x0_impedance
    scattering = _constant_scattering(0.0)
    scattering[floor] = 0.25
    scattering[x0] = 0.75

    floor_reflection = (
        (floor_impedance - _RHO_C_PA_S_PER_M)
        / (floor_impedance + _RHO_C_PA_S_PER_M)
    )
    x0_reflection = (
        (x0_impedance - _RHO_C_PA_S_PER_M)
        / (x0_impedance + _RHO_C_PA_S_PER_M)
    )
    floor_weight = (6.0 * 4.0) * abs(floor_reflection) ** 2
    x0_weight = (4.0 * 3.0) * abs(x0_reflection) ** 2
    expected = (
        floor_weight * 0.25 + x0_weight * 0.75
    ) / (floor_weight + x0_weight)

    actual = geometric_lane.solve_geometric_lane(
        room=_ROOM,
        source=_SOURCE,
        receiver=_RECEIVER,
        sound_speed_m_s=_SOUND_SPEED_M_S,
        rho_c_pa_s_per_m=_RHO_C_PA_S_PER_M,
        frequencies_hz=(125.0,),
        impedance_by_wall=impedances,
        scattering_by_wall=scattering,
    )

    assert actual.scattering == (expected,)


def test_room_scattering_is_zero_when_no_wall_reflects() -> None:
    """分母為零時若六面 s 相同就提前回 s，這題必須紅。"""
    from aosr.physics import geometric_lane

    actual = geometric_lane.solve_geometric_lane(
        room=_ROOM,
        source=_SOURCE,
        receiver=_RECEIVER,
        sound_speed_m_s=_SOUND_SPEED_M_S,
        rho_c_pa_s_per_m=_RHO_C_PA_S_PER_M,
        frequencies_hz=(125.0,),
        impedance_by_wall=_constant_walls(
            complex(_RHO_C_PA_S_PER_M, 0.0)
        ),
        scattering_by_wall=_constant_scattering(0.75),
    )

    assert actual.scattering == (0.0,)


def test_band_average_preserves_every_constant_energy_component() -> None:
    """帶內平均若漏欄、誤用加總或改動常數序列，這題必須紅。"""
    from aosr.physics import geometric_lane

    fine = geometric_lane.GeometricLaneResult(
        frequencies_hz=(100.0, 125.0, 150.0),
        direct_energy=(2.0, 2.0, 2.0),
        reflected_energy=(3.0, 3.0, 3.0),
        late_energy=(5.0, 5.0, 5.0),
        scattering=(0.25, 0.25, 0.25),
        geometric_energy=(7.0, 7.0, 7.0),
    )

    averaged = geometric_lane.average_geometric_lane_to_bands(
        fine, band_centers_hz=(125.0,)
    )

    assert averaged.direct_energy == (2.0,)
    assert averaged.reflected_energy == (3.0,)
    assert averaged.late_energy == (5.0,)
    assert averaged.scattering == (0.25,)
    assert averaged.geometric_energy == (7.0,)


def test_band_average_is_the_exact_arithmetic_mean_of_distinct_points() -> None:
    """只取中點或改成幾何平均時，五個逐位答案都必須紅。"""
    from aosr.physics import geometric_lane

    fine = geometric_lane.GeometricLaneResult(
        frequencies_hz=(100.0, 125.0, 150.0),
        direct_energy=(1.0, 4.0, 16.0),
        reflected_energy=(2.0, 8.0, 32.0),
        late_energy=(3.0, 12.0, 48.0),
        scattering=(0.125, 0.25, 0.5),
        geometric_energy=(5.0, 20.0, 80.0),
    )

    averaged = geometric_lane.average_geometric_lane_to_bands(
        fine, band_centers_hz=(125.0,)
    )

    assert averaged.direct_energy == ((1.0 + 4.0 + 16.0) / 3.0,)
    assert averaged.reflected_energy == ((2.0 + 8.0 + 32.0) / 3.0,)
    assert averaged.late_energy == ((3.0 + 12.0 + 48.0) / 3.0,)
    assert averaged.scattering == ((0.125 + 0.25 + 0.5) / 3.0,)
    assert averaged.geometric_energy == ((5.0 + 20.0 + 80.0) / 3.0,)


def test_band_average_excludes_a_point_exactly_on_the_upper_edge() -> None:
    """把半開區間上緣誤寫成包含，帶平均就會被上緣異值污染。"""
    from aosr.physics import geometric_lane

    upper_edge = 125.0 * math.sqrt(2.0)
    fine = geometric_lane.GeometricLaneResult(
        frequencies_hz=(125.0, upper_edge),
        direct_energy=(2.0, 100.0),
        reflected_energy=(0.0, 0.0),
        late_energy=(0.0, 0.0),
        scattering=(0.0, 0.0),
        geometric_energy=(2.0, 100.0),
    )

    averaged = geometric_lane.average_geometric_lane_to_bands(
        fine, band_centers_hz=(125.0,)
    )

    assert averaged.direct_energy == (2.0,)
    assert averaged.geometric_energy == (2.0,)


def test_band_average_rejects_a_band_without_fine_axis_points() -> None:
    """空帶若被無聲寫成零、非數或跳過，這題必須紅。"""
    from aosr.physics import geometric_lane

    fine = geometric_lane.GeometricLaneResult(
        frequencies_hz=(125.0,),
        direct_energy=(1.0,),
        reflected_energy=(1.0,),
        late_energy=(1.0,),
        scattering=(0.0,),
        geometric_energy=(2.0,),
    )

    with pytest.raises(ValueError, match="沒有頻點"):
        geometric_lane.average_geometric_lane_to_bands(
            fine, band_centers_hz=(4000.0,)
        )


def test_band_only_impedance_is_rejected_on_the_fine_axis() -> None:
    """六帶材料若被擅自攤到細軸，未拍板的內插規則就會偷進產品。"""
    from aosr.physics import geometric_lane

    octave_band_row = tuple(
        complex(4.0 * _RHO_C_PA_S_PER_M, 0.0)
        for _center in frequency_axis_config.GEOMETRIC_REPORT_OCTAVE_CENTERS_HZ
    )
    impedances = {wall: octave_band_row for wall in Wall.wall_names()}

    with pytest.raises(
        ValueError,
        match=re.escape("只有頻帶值的材料在細軸上怎麼取值待拍（實測資料要內插）"),
    ):
        geometric_lane.solve_geometric_lane(
            room=_ROOM,
            source=_SOURCE,
            receiver=_RECEIVER,
            sound_speed_m_s=_SOUND_SPEED_M_S,
            rho_c_pa_s_per_m=_RHO_C_PA_S_PER_M,
            frequencies_hz=frequency_axis_config.GEOMETRIC_LANE_FREQUENCIES_HZ,
            impedance_by_wall=impedances,
        )
