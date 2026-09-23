"""幾何路細頻率軸、相位能量接法與頻帶平均的性質考卷。"""

from __future__ import annotations

import cmath
import math
import re
from dataclasses import dataclass

import pytest

from aosr.config import frequency_axis as frequency_axis_config
from aosr.config.art_lane import ART_N_PER_WALL_DEFAULT
from aosr.config.three_lane_crossover import REFLECTION_ORDER_K
from aosr.geometry.shoebox import Point, Room, Wall
from aosr.materials.response import MATERIAL_SCATTERING_DEFAULT_S
from aosr.physics.amplitude import Materials
from aosr.physics.late_energy import (
    LateEnergyInputs,
    solve_late_energy,
    solve_late_energy_by_order,
)
from aosr.physics.room_paths import RoomPath, image_source_paths
from aosr.physics.geometric_lane import GeometricEarlyResult, GeometricLaneResult
from aosr.physics.totals import (
    Totals,
    totals_and_pressure_sums_from_paths,
    totals_from_paths,
)
from tests.engine._precision_contracts import contract_value


_ROOM = Room(Lx=6.0, Ly=4.0, Lz=3.0)
_SOURCE = Point(x=1.2, y=1.3, z=1.1)
_RECEIVER = Point(x=4.7, y=2.8, z=1.4)
_SOUND_SPEED_M_S = 343.0
_RHO_C_PA_S_PER_M = 411.6
_DIRECT_TOLERANCE_REL = contract_value("direct_energy_vs_legacy")


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


@dataclass(frozen=True)
class _InterferenceCase:
    result: GeometricLaneResult
    totals: Totals
    direct_pressure: tuple[complex, ...]
    order_pressure: tuple[tuple[complex, ...], ...]
    scattering: float


def _order_pressure(
    paths: list[RoomPath], frequency_count: int
) -> tuple[tuple[complex, ...], ...]:
    """考卷自己把路徑按反射階數分組相加，不呼叫產品那一支逐階和。"""
    return tuple(
        tuple(
            sum(
                (path.path_pressure[index] for path in paths if path.order == order),
                complex(0.0, 0.0),
            )
            for index in range(frequency_count)
        )
        for order in range(1, REFLECTION_ORDER_K + 1)
    )


def _order_scaled_pressure(
    order_pressure: tuple[tuple[complex, ...], ...],
    scattering: float,
    index: int,
) -> complex:
    """考卷自己按 ``(1−s)^{k/2}`` 縮放逐階壓力再相加：``Σ_{k≤K}(1−s)^{k/2}·p_k``。"""
    retained = 1.0 - scattering
    total = complex(0.0, 0.0)
    for order, column in enumerate(order_pressure, start=1):
        total += retained ** (order / 2.0) * column[index]
    return total


@dataclass(frozen=True)
class _DirectFloorAnalyticCase:
    frequencies_hz: tuple[float, ...]
    direct_energy: tuple[float, ...]
    reflected_energy: tuple[float, ...]
    interference_energy: tuple[float, ...]
    coherent_energy: tuple[float, ...]
    product_totals: Totals
    product_interference_energy: tuple[float, ...]
    direct_distance_m: float
    floor_distance_m: float
    floor_reflection: complex


@pytest.fixture(
    scope="module",
    params=(
        (4.0, MATERIAL_SCATTERING_DEFAULT_S),
        (10.0, MATERIAL_SCATTERING_DEFAULT_S),
        (400.0, 0.0),
        (400.0, 1.0),
    ),
    ids=("flat", "low-absorption", "hard-wall-s0", "hard-wall-s1"),
)
def interference_case(request: pytest.FixtureRequest) -> _InterferenceCase:
    """同一份真實路徑結果供三個互相獨立的物理性質使用。"""
    from aosr.physics import geometric_lane

    impedance_multiple, scattering = request.param
    frequencies_hz = frequency_axis_config.GEOMETRIC_LANE_FREQUENCIES_HZ
    impedance = complex(float(impedance_multiple) * _RHO_C_PA_S_PER_M, 0.0)
    impedance_rows = _wall_rows(impedance, frequencies_hz)
    paths = image_source_paths(
        _ROOM,
        _SOURCE,
        _RECEIVER,
        _SOUND_SPEED_M_S,
        max_order=3,
        materials=Materials(
            rho_c=_RHO_C_PA_S_PER_M,
            frequencies_hz=frequencies_hz,
            walls=impedance_rows,
        ),
    )
    direct = next(path for path in paths if path.order == 0)
    result = geometric_lane.solve_geometric_lane(
        room=_ROOM,
        source=_SOURCE,
        receiver=_RECEIVER,
        sound_speed_m_s=_SOUND_SPEED_M_S,
        rho_c_pa_s_per_m=_RHO_C_PA_S_PER_M,
        frequencies_hz=frequencies_hz,
        impedance_by_wall=_constant_walls(impedance),
        scattering_by_wall=_constant_scattering(float(scattering)),
    )
    return _InterferenceCase(
        result=result,
        totals=totals_from_paths(paths),
        direct_pressure=direct.path_pressure,
        order_pressure=_order_pressure(paths, len(frequencies_hz)),
        scattering=float(scattering),
    )


def _energy_roundoff_bound(*values: float) -> float:
    """只用既有直達能量契約界線縮放本題各能量量級。"""
    return _DIRECT_TOLERANCE_REL * sum(abs(value) for value in values)


def _direct_floor_geometry() -> tuple[float, float, float]:
    direct_dx = _RECEIVER.x - _SOURCE.x
    direct_dy = _RECEIVER.y - _SOURCE.y
    direct_dz = _RECEIVER.z - _SOURCE.z
    direct_distance = math.sqrt(
        direct_dx * direct_dx + direct_dy * direct_dy + direct_dz * direct_dz
    )
    floor_image_z = -_SOURCE.z
    floor_dz = _RECEIVER.z - floor_image_z
    floor_distance = math.sqrt(
        direct_dx * direct_dx + direct_dy * direct_dy + floor_dz * floor_dz
    )
    cos_theta = abs(floor_dz) / floor_distance
    return direct_distance, floor_distance, cos_theta


def _analytic_direct_floor_energies(
    frequencies_hz: tuple[float, ...],
    direct_distance: float,
    floor_distance: float,
    floor_reflection: complex,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    direct_energy = []
    reflected_energy = []
    interference_energy = []
    coherent_energy = []
    for frequency_hz in frequencies_hz:
        wave_number = 2.0 * math.pi * frequency_hz / _SOUND_SPEED_M_S
        direct_pressure = cmath.exp(-1j * wave_number * direct_distance) / direct_distance
        floor_pressure = (
            floor_reflection
            * cmath.exp(-1j * wave_number * floor_distance)
            / floor_distance
        )
        direct_energy.append(abs(direct_pressure) ** 2)
        reflected_energy.append(abs(floor_pressure) ** 2)
        interference_energy.append(
            2.0 * (direct_pressure * floor_pressure.conjugate()).real
        )
        coherent_energy.append(abs(direct_pressure + floor_pressure) ** 2)
    return (
        tuple(direct_energy),
        tuple(reflected_energy),
        tuple(interference_energy),
        tuple(coherent_energy),
    )


def _direct_floor_analytic_case(
    frequencies_hz: tuple[float, ...],
) -> _DirectFloorAnalyticCase:
    from aosr.physics import geometric_lane

    impedance = complex(400.0 * _RHO_C_PA_S_PER_M, 0.0)
    materials = Materials(
        rho_c=_RHO_C_PA_S_PER_M,
        frequencies_hz=frequencies_hz,
        walls=_wall_rows(impedance, frequencies_hz),
    )
    enumerated = image_source_paths(
        _ROOM,
        _SOURCE,
        _RECEIVER,
        _SOUND_SPEED_M_S,
        max_order=1,
        materials=materials,
    )
    direct_path = next(path for path in enumerated if path.order == 0)
    floor_path = next(
        path
        for path in enumerated
        if path.order == 1
        and tuple(bounce.wall for bounce in path.bounces)
        == (Wall.FLOOR.wall_name(),)
    )
    product_totals, pressure_sums = totals_and_pressure_sums_from_paths(
        [direct_path, floor_path]
    )

    direct_distance, floor_distance, cos_theta = _direct_floor_geometry()
    z_cos = impedance * cos_theta
    floor_reflection = (z_cos - _RHO_C_PA_S_PER_M) / (
        z_cos + _RHO_C_PA_S_PER_M
    )
    direct_energy, reflected_energy, interference_energy, coherent_energy = (
        _analytic_direct_floor_energies(
            frequencies_hz,
            direct_distance,
            floor_distance,
            floor_reflection,
        )
    )

    return _DirectFloorAnalyticCase(
        frequencies_hz=frequencies_hz,
        direct_energy=direct_energy,
        reflected_energy=reflected_energy,
        interference_energy=interference_energy,
        coherent_energy=coherent_energy,
        product_totals=product_totals,
        product_interference_energy=geometric_lane._interference_energy(
            pressure_sums.direct_pressure,
            pressure_sums.reflected_pressure,
        ),
        direct_distance_m=direct_distance,
        floor_distance_m=floor_distance,
        floor_reflection=floor_reflection,
    )


def _assert_energy_matches_analytic(actual: float, analytic: float) -> None:
    assert abs(actual - analytic) <= _DIRECT_TOLERANCE_REL * abs(analytic)


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
    """改抄鏡像或晚期算法、改相位相加順序、或不是交接階數 K 那一套時必須紅。

    兩條「沒有第二份算法」：直達那一欄逐位等於既有 ``totals_from_paths``；逐階晚期解的
    總量逐位等於既有 ``solve_late_energy``。第三條是分工本身——s=0 時晚期只剩尾巴，
    鏡面那一欄退回既有的同調和（浮點結合律之內）。
    """
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

    expected_orders = solve_late_energy_by_order(
        LateEnergyInputs(
            room=_ROOM,
            rho_c_pa_s_per_m=_RHO_C_PA_S_PER_M,
            frequencies_hz=frequencies_hz,
            impedance_by_wall=impedance_rows,
            n_per_wall=ART_N_PER_WALL_DEFAULT,
            domain_alpha_bar_max=math.inf,
        ),
        max_order=REFLECTION_ORDER_K,
    )

    actual = geometric_lane.solve_geometric_lane(
        room=_ROOM,
        source=_SOURCE,
        receiver=_RECEIVER,
        sound_speed_m_s=_SOUND_SPEED_M_S,
        rho_c_pa_s_per_m=_RHO_C_PA_S_PER_M,
        frequencies_hz=frequencies_hz,
        impedance_by_wall=_constant_walls(impedance),
        scattering_by_wall=_constant_scattering(0.0),
    )

    assert actual.reflection_order_k == REFLECTION_ORDER_K
    assert actual.direct_energy == expected_totals.direct_energy
    # 逐階晚期解沒有另開一份算法：總量逐位等於既有那一支。
    assert tuple(band.late_reverberant_energy for band in expected_orders.bands) == (
        tuple(band.late_reverberant_energy for band in expected_late.bands)
    )
    # s=0：鏡面退回既有同調和，晚期只留 K 階以上的尾巴。
    for index, band in enumerate(expected_orders.bands):
        assert abs(
            actual.reflected_energy[index] - expected_totals.reflected_energy[index]
        ) <= _energy_roundoff_bound(expected_totals.reflected_energy[index])
        assert actual.late_energy[index] == band.tail_energy


def test_early_solver_reuses_coherent_paths_without_solving_late_energy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """密軸另抄鏡像法、漏干涉，或順手計算晚期混響時必須紅。"""
    from aosr.physics import geometric_lane

    frequencies_hz = (100.0, 125.0)
    impedance = complex(4.0 * _RHO_C_PA_S_PER_M, 0.0)
    paths = image_source_paths(
        _ROOM,
        _SOURCE,
        _RECEIVER,
        _SOUND_SPEED_M_S,
        max_order=3,
        materials=Materials(
            rho_c=_RHO_C_PA_S_PER_M,
            frequencies_hz=frequencies_hz,
            walls=_wall_rows(impedance, frequencies_hz),
        ),
    )
    expected = totals_from_paths(paths)

    def late_must_not_run(inputs: LateEnergyInputs, *, max_order: int) -> object:
        raise AssertionError(
            f"密軸不准求晚期混響：{inputs.frequencies_hz!r}（K={max_order}）"
        )

    monkeypatch.setattr(
        geometric_lane, "solve_late_energy_by_order", late_must_not_run
    )
    actual = geometric_lane.solve_geometric_early_lane(
        room=_ROOM,
        source=_SOURCE,
        receiver=_RECEIVER,
        sound_speed_m_s=_SOUND_SPEED_M_S,
        rho_c_pa_s_per_m=_RHO_C_PA_S_PER_M,
        frequencies_hz=frequencies_hz,
        impedance_by_wall=_constant_walls(impedance),
        scattering_by_wall=_constant_scattering(0.0),
    )

    assert actual.direct_energy == expected.direct_energy
    for index, pressure in enumerate(expected.pressure):
        direct = expected.direct_energy[index]
        reflected = expected.reflected_energy[index]
        bound = _energy_roundoff_bound(abs(pressure) ** 2, direct, reflected)
        # s=0：早期那三欄退回既有同調總量（浮點結合律之內）。
        assert abs(actual.reflected_energy[index] - reflected) <= bound
        residual = abs(pressure) ** 2 - direct - reflected
        assert abs(actual.interference_energy[index] - residual) <= bound


@pytest.mark.parametrize(
    ("direct_pressure", "reflected_pressure", "expected"),
    (
        (complex(1.0, 2.0), complex(3.0, 4.0), 22.0),
        (complex(1.0e16, 0.0), complex(1.0, 1.0), 2.0e16),
    ),
    ids=("factor-and-conjugate", "no-energy-subtraction-cancellation"),
)
def test_interference_uses_the_direct_complex_product(
    direct_pressure: complex,
    reflected_pressure: complex,
    expected: float,
) -> None:
    """少係數 2、共軛直達，或改用三個能量相減時必須紅。"""
    from aosr.physics import geometric_lane

    assert geometric_lane._interference_energy(
        (direct_pressure,), (reflected_pressure,)
    ) == (expected,)


def test_early_columns_match_the_independently_scaled_order_sums(
    interference_case: _InterferenceCase,
) -> None:
    """三個早期欄若不是「逐階乘 (1−s)^{k/2} 之後」的直達、反射與交叉項，必須紅。

    考卷自己把路徑按階數分組、自己乘 ``(1−s)^{k/2}``（``_order_pressure`` 與
    ``_order_scaled_pressure``），所以漏掉逐階縮放、把縮放放到能量域（乘 ``1−s``
    而不是壓力乘 ``√(1−s)``）、或干涉沒跟著縮放，三種都紅。
    """
    case = interference_case
    actual = case.result
    for index, direct_pressure in enumerate(case.direct_pressure):
        scaled = _order_scaled_pressure(case.order_pressure, case.scattering, index)
        direct_energy = abs(direct_pressure) ** 2
        reflected_energy = abs(scaled) ** 2
        interference = (
            abs(direct_pressure + scaled) ** 2 - direct_energy - reflected_energy
        )
        bound = _energy_roundoff_bound(
            direct_energy, reflected_energy, abs(direct_pressure + scaled) ** 2
        )
        assert abs(actual.direct_energy[index] - direct_energy) <= bound
        assert abs(actual.reflected_energy[index] - reflected_energy) <= bound
        assert abs(actual.interference_energy[index] - interference) <= bound


def test_direct_and_floor_reflection_match_independent_analytic_solution() -> None:
    """cosθ 是鏡像幾何算的反射路徑對地板法向入射角餘弦。

    這與產品反射係數使用同一定義；相位正負、反射係數、共軛或係數 2 任一錯
    都必須紅。
    """
    case = _direct_floor_analytic_case((173.0, 997.0, 2123.0))

    for index in range(len(case.frequencies_hz)):
        _assert_energy_matches_analytic(
            case.product_totals.direct_energy[index], case.direct_energy[index]
        )
        _assert_energy_matches_analytic(
            case.product_totals.reflected_energy[index], case.reflected_energy[index]
        )
        _assert_energy_matches_analytic(
            case.product_interference_energy[index], case.interference_energy[index]
        )
        _assert_energy_matches_analytic(
            abs(case.product_totals.pressure[index]) ** 2,
            case.coherent_energy[index],
        )


@pytest.mark.parametrize("comb_index", (1, 2, 3, 4), ids=("n1", "n2", "n3", "n4"))
def test_direct_and_floor_reflection_form_analytic_comb_extrema(
    comb_index: int,
) -> None:
    """路徑差相位沒在整數處建設、半整數處破壞，或干涉正負顛倒時必須紅。"""
    direct_distance, floor_distance, _cos_theta = _direct_floor_geometry()
    path_difference = floor_distance - direct_distance
    constructive_hz = comb_index * _SOUND_SPEED_M_S / path_difference
    destructive_hz = (comb_index + 0.5) * _SOUND_SPEED_M_S / path_difference
    case = _direct_floor_analytic_case((constructive_hz, destructive_hz))
    reflection = case.floor_reflection.real
    constructive_energy = (
        1.0 / direct_distance + reflection / floor_distance
    ) ** 2
    destructive_energy = (
        1.0 / direct_distance - reflection / floor_distance
    ) ** 2
    constructive_interference = 2.0 * reflection / (
        direct_distance * floor_distance
    )
    destructive_interference = -constructive_interference

    _assert_energy_matches_analytic(case.coherent_energy[0], constructive_energy)
    _assert_energy_matches_analytic(case.coherent_energy[1], destructive_energy)
    _assert_energy_matches_analytic(
        abs(case.product_totals.pressure[0]) ** 2, constructive_energy
    )
    _assert_energy_matches_analytic(
        abs(case.product_totals.pressure[1]) ** 2, destructive_energy
    )
    _assert_energy_matches_analytic(
        case.interference_energy[0], constructive_interference
    )
    _assert_energy_matches_analytic(
        case.interference_energy[1], destructive_interference
    )
    _assert_energy_matches_analytic(
        case.product_interference_energy[0], constructive_interference
    )
    _assert_energy_matches_analytic(
        case.product_interference_energy[1], destructive_interference
    )
    assert case.interference_energy[0] > 0.0
    assert case.interference_energy[1] < 0.0
    assert case.product_interference_energy[0] > 0.0
    assert case.product_interference_energy[1] < 0.0


def test_geometric_energy_matches_the_coherent_pressure_identity(
    interference_case: _InterferenceCase,
) -> None:
    """漏干涉、或沒有把整個同調場逐階乘上 ``√(1−s)`` 時必須紅。

    決策紙第 5 條的第一項是**模平方**：``|p_direct + Σ_{k≤K}(1−s)^{k/2}·p_k|²``。
    這題把它跟晚期那一欄相加，對上產品的 ``geometric_energy``。
    """
    case = interference_case
    for index, direct_pressure in enumerate(case.direct_pressure):
        scaled = _order_scaled_pressure(case.order_pressure, case.scattering, index)
        coherent = abs(direct_pressure + scaled) ** 2
        late = case.result.late_energy[index]
        expected = coherent + late
        bound = _energy_roundoff_bound(expected, coherent, late)
        assert abs(case.result.geometric_energy[index] - expected) <= bound


def test_geometric_energy_is_nonnegative_for_all_material_cases(
    interference_case: _InterferenceCase,
) -> None:
    """低吸音、近硬牆或散射端點產生負幾何能量時必須紅。"""
    assert all(value >= 0.0 for value in interference_case.result.geometric_energy)


def test_scattering_endpoints_split_geometric_energy_by_order() -> None:
    """s=0 不等於「K 階以內鏡面含干涉＋晚期尾巴」，或 s=1 仍混入反射與干涉時必須紅。

    兩端極限照決策紙第 5 條最後一段。s=1 那一端不逐位比：晚期那一欄是
    ``ΣA_k + (E_late − ΣA_k)``，浮點加減的結合律差別留在捨入量級裡。
    """
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

    late_total = solve_late_energy(
        LateEnergyInputs(
            room=_ROOM,
            rho_c_pa_s_per_m=_RHO_C_PA_S_PER_M,
            frequencies_hz=(125.0,),
            impedance_by_wall=_wall_rows(
                complex(4.0 * _RHO_C_PA_S_PER_M, 0.0), (125.0,)
            ),
            n_per_wall=ART_N_PER_WALL_DEFAULT,
            domain_alpha_bar_max=math.inf,
        )
    ).bands[0].late_reverberant_energy

    # s=0：鏡面那三欄不被散射扣掉，晚期只剩 K 階以上那一段尾巴（嚴格小於總量）。
    assert zero.scattering == (0.0,)
    assert zero.interference_energy[0] != 0.0
    assert zero.geometric_energy == (
        zero.direct_energy[0]
        + zero.reflected_energy[0]
        + zero.interference_energy[0]
        + zero.late_energy[0],
    )
    assert 0.0 < zero.late_energy[0] < late_total

    # s=1：整個反射相關部分等於晚期混響總量，鏡面那兩欄歸零。
    assert one.scattering == (1.0,)
    assert one.reflected_energy[0] == 0.0
    assert one.interference_energy[0] == 0.0
    assert abs(one.late_energy[0] - late_total) <= _energy_roundoff_bound(late_total)
    assert one.geometric_energy == (
        one.direct_energy[0] + one.late_energy[0],
    )


def test_missing_scattering_uses_the_material_default_curve() -> None:
    """缺省散射若不是 response 的正式常數，或幾何能量不是報表四欄相加，必須紅。"""
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
        + actual.reflected_energy[0]
        + actual.interference_energy[0]
        + actual.late_energy[0]
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
        interference_energy=(-1.0, -1.0, -1.0),
        late_energy=(5.0, 5.0, 5.0),
        scattering=(0.25, 0.25, 0.25),
        geometric_energy=(7.0, 7.0, 7.0),
        reflection_order_k=REFLECTION_ORDER_K,
    )

    averaged = geometric_lane.average_geometric_lane_to_bands(
        fine, band_centers_hz=(125.0,)
    )

    assert averaged.direct_energy == (2.0,)
    assert averaged.reflected_energy == (3.0,)
    assert averaged.interference_energy == (-1.0,)
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
        interference_energy=(-1.0, -4.0, -16.0),
        late_energy=(3.0, 12.0, 48.0),
        scattering=(0.125, 0.25, 0.5),
        geometric_energy=(5.0, 20.0, 80.0),
        reflection_order_k=REFLECTION_ORDER_K,
    )

    averaged = geometric_lane.average_geometric_lane_to_bands(
        fine, band_centers_hz=(125.0,)
    )

    assert averaged.direct_energy == ((1.0 + 4.0 + 16.0) / 3.0,)
    assert averaged.reflected_energy == ((2.0 + 8.0 + 32.0) / 3.0,)
    assert averaged.interference_energy == ((-1.0 - 4.0 - 16.0) / 3.0,)
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
        interference_energy=(0.0, 0.0),
        late_energy=(0.0, 0.0),
        scattering=(0.0, 0.0),
        geometric_energy=(2.0, 100.0),
        reflection_order_k=REFLECTION_ORDER_K,
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
        interference_energy=(0.0,),
        late_energy=(1.0,),
        scattering=(0.0,),
        geometric_energy=(2.0,),
        reflection_order_k=REFLECTION_ORDER_K,
    )

    with pytest.raises(ValueError, match="沒有頻點"):
        geometric_lane.average_geometric_lane_to_bands(
            fine, band_centers_hz=(4000.0,)
        )


def test_dense_early_band_average_keeps_late_energy_on_the_fine_axis() -> None:
    """早期項退回細軸、晚期搬到密軸或把分項平均後才合成時必須紅。"""
    from aosr.physics import geometric_lane

    fine = geometric_lane.GeometricLaneResult(
        frequencies_hz=(100.0, 125.0, 150.0),
        direct_energy=(100.0, 100.0, 100.0),
        reflected_energy=(100.0, 100.0, 100.0),
        interference_energy=(100.0, 100.0, 100.0),
        late_energy=(10.0, 20.0, 30.0),
        scattering=(0.4, 0.5, 0.6),
        geometric_energy=(1000.0, 1000.0, 1000.0),
        reflection_order_k=REFLECTION_ORDER_K,
    )
    dense = geometric_lane.GeometricEarlyResult(
        frequencies_hz=(100.0, 125.0, 150.0),
        direct_energy=(1.0, 4.0, 7.0),
        reflected_energy=(2.0, 5.0, 8.0),
        interference_energy=(0.5, -1.0, 2.0),
        scattering=(0.1, 0.2, 0.3),
        reflection_order_k=REFLECTION_ORDER_K,
    )

    actual = geometric_lane.average_geometric_lane_to_bands_with_dense_early(
        fine,
        dense,
        band_centers_hz=(125.0,),
    )

    # 早期三欄已經含散射留存，這裡只相加不再乘 1−s；晚期那一欄同理不再乘 s。
    dense_early = (3.5 + 8.0 + 17.0) / 3.0
    positions = tuple(math.log2(f) for f in fine.frequencies_hz)
    edges = (
        positions[0] - (positions[1] - positions[0]) / 2,
        (positions[0] + positions[1]) / 2,
        (positions[1] + positions[2]) / 2,
        positions[2] + (positions[2] - positions[1]) / 2,
    )
    widths = tuple(right - left for left, right in zip(edges, edges[1:]))
    fine_late_share = sum(w * value for w, value in zip(widths, fine.late_energy)) / sum(widths)
    assert actual.direct_energy == (4.0,)
    assert actual.reflected_energy == (5.0,)
    assert actual.interference_energy == (0.5,)
    assert actual.late_energy == (fine_late_share,)
    assert actual.scattering == (sum((0.1, 0.2, 0.3)) / 3.0,)
    assert actual.geometric_energy == (dense_early + fine_late_share,)


def _interference_band_delta_db(result: GeometricEarlyResult) -> float:
    without_interference = sum(
        direct + reflected
        for direct, reflected in zip(
            result.direct_energy, result.reflected_energy, strict=True
        )
    ) / len(result.frequencies_hz)
    with_interference = sum(
        direct + reflected + interference
        for direct, reflected, interference in zip(
            result.direct_energy,
            result.reflected_energy,
            result.interference_energy,
            strict=True,
        )
    ) / len(result.frequencies_hz)
    return abs(10.0 * math.log10(with_interference / without_interference))


def test_dense_sampling_reduces_flat_room_4000_hz_interference_bias() -> None:
    """4000 Hz 早期項退回 24 點細軸，使干涉帶平均偏差變大時必須紅。"""
    from aosr.physics import geometric_lane

    source = Point(x=1.5, y=1.0, z=1.2)
    receiver = Point(x=4.0, y=3.0, z=1.5)
    center_hz = 4000.0
    lower = center_hz / math.sqrt(2.0)
    upper = center_hz * math.sqrt(2.0)
    fine_frequencies = tuple(
        frequency
        for frequency in frequency_axis_config.GEOMETRIC_LANE_FREQUENCIES_HZ
        if lower <= frequency < upper
    )
    dense_frequencies = frequency_axis_config.geometric_band_frequencies(center_hz)
    impedance = complex(4.0 * _RHO_C_PA_S_PER_M, 0.0)
    fine = geometric_lane.solve_geometric_early_lane(
        room=_ROOM,
        source=source,
        receiver=receiver,
        sound_speed_m_s=_SOUND_SPEED_M_S,
        rho_c_pa_s_per_m=_RHO_C_PA_S_PER_M,
        frequencies_hz=fine_frequencies,
        impedance_by_wall=_constant_walls(impedance),
    )
    dense = geometric_lane.solve_geometric_early_lane(
        room=_ROOM,
        source=source,
        receiver=receiver,
        sound_speed_m_s=_SOUND_SPEED_M_S,
        rho_c_pa_s_per_m=_RHO_C_PA_S_PER_M,
        frequencies_hz=dense_frequencies,
        impedance_by_wall=_constant_walls(impedance),
    )
    assert _interference_band_delta_db(dense) < _interference_band_delta_db(fine)


def test_band_only_impedance_is_rejected_on_the_fine_axis() -> None:
    """報表帶材料若被擅自攤到細軸，未拍板的內插規則就會偷進產品。"""
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
