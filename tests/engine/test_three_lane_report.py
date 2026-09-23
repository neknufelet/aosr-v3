"""第九段三路接合的結構化物理量報表考卷。"""

from __future__ import annotations

import inspect
import math
from functools import partial
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pytest
from numpy.typing import NDArray

from aosr import runtime
from aosr.config.art_lane import ART_N_PER_WALL_DEFAULT
from aosr.config.fem_lane import FEM_ELEMENTS_PER_WAVELENGTH, FEM_MESH_RANDOM_SEED
from aosr.config.frequency_axis import (
    FEM_GEOMETRIC_CROSSOVER_CAP_HZ,
    FEM_LANE_FREQUENCIES_HZ,
    GEOMETRIC_BAND_FREQUENCIES_HZ,
    GEOMETRIC_LANE_FREQUENCIES_HZ,
    GEOMETRIC_REPORT_OCTAVE_CENTERS_HZ,
)
from aosr.config.three_lane_crossover import (
    CROSSOVER_LOWER_FLOOR_HZ,
    REFLECTION_ORDER_K,
)
from aosr.geometry.shoebox import Point, Room, Wall
from aosr.geometry.shoebox_mesh import ShoeboxMesh, generate_shoebox_mesh
from aosr.materials import catalog_absorption
from aosr.physics import three_lane_report
from aosr.physics.crossover import (
    CrossoverWeights,
    crossover_weights,
    eyring_t60_by_band,
    schroeder_frequency_hz,
)
from aosr.physics.fem_helmholtz import WallImpedances, solve_fem_helmholtz
from aosr.physics.geometric_lane import (
    GeometricEarlyResult,
    GeometricLaneResult,
    solve_geometric_early_lane,
    solve_geometric_lane,
)
from aosr.physics.late_decay import LateDecayBand, LateDecayResult, solve_late_decay
from aosr.physics.late_energy import LateEnergyInputs, LateEnergyOrderResult
from aosr.physics.three_lane_report import ThreeLaneBandReport, ThreeLaneReport


ROOM = Room(6.0, 4.0, 3.0)
SOURCE = Point(1.2, 1.3, 1.1)
RECEIVER = Point(4.7, 2.8, 1.4)
SOUND_SPEED_M_S = 343.0
DENSITY_KG_M3 = 1.2
RHO_C_PA_S_PER_M = DENSITY_KG_M3 * SOUND_SPEED_M_S
HARD_CUT_ROOM = Room(1.5, 1.0, 0.75)
HARD_CUT_SOURCE = Point(0.3, 0.325, 0.275)
HARD_CUT_RECEIVER = Point(1.175, 0.7, 0.35)
FEM_COMPARISON_FREQUENCIES_HZ = (
    FEM_LANE_FREQUENCIES_HZ[0],
    FEM_LANE_FREQUENCIES_HZ[-1],
)
T20_ONLY_IMPEDANCE_MULTIPLE = (
    catalog_absorption.normalized_impedance_from_random_incidence_absorption(0.026)
)
UNREACHED_T20_IMPEDANCE_MULTIPLE = (
    catalog_absorption.normalized_impedance_from_random_incidence_absorption(0.02)
)


def _walls(value: object) -> dict[Wall, object]:
    return {wall: value for wall in Wall.all()}


def _named_walls(
    value: complex, frequencies_hz: tuple[float, ...]
) -> dict[str, tuple[complex, ...]]:
    return {
        wall.wall_name(): tuple(value for _frequency in frequencies_hz)
        for wall in Wall.all()
    }


def _random_absorption(impedance: float) -> float:
    return catalog_absorption.complex_random_incidence_absorption(
        impedance / RHO_C_PA_S_PER_M
    )


def _deterministic_fem_energy(frequency_hz: float) -> float:
    return frequency_hz * 3.0 + 7.0


def _report_decay_frequencies() -> tuple[float, ...]:
    root_two = math.sqrt(2.0)
    return tuple(
        frequency
        for frequency in GEOMETRIC_LANE_FREQUENCIES_HZ
        if any(
            center / root_two <= frequency < center * root_two
            for center in GEOMETRIC_REPORT_OCTAVE_CENTERS_HZ
        )
    )


def _constant_late_decay(frequencies_hz: tuple[float, ...]) -> LateDecayResult:
    return LateDecayResult(
        orders_used=256,
        bands=tuple(
            LateDecayBand(
                frequency_hz=frequency,
                t20_s=1.0,
                collision_frequency_hz=1.0,
                slope_db_per_s=-1.0,
                soft_weight_sum=1.0,
                fell_back_to_perron=False,
                perron_t60_s=1.0,
                t30_s=2.0,
                t30_slope_db_per_s=-1.0,
                t30_soft_weight_sum=1.0,
            )
            for frequency in frequencies_hz
        ),
    )


def _solve_directly_on_official_mesh() -> tuple[
    ShoeboxMesh,
    dict[Wall, float],
    NDArray[np.complex128],
]:
    impedance = 4.0 * RHO_C_PA_S_PER_M
    wall_impedances = {wall: impedance for wall in Wall.all()}
    mesh = generate_shoebox_mesh(
        ROOM,
        max_frequency_hz=FEM_GEOMETRIC_CROSSOVER_CAP_HZ,
        elements_per_wavelength=FEM_ELEMENTS_PER_WAVELENGTH,
        sound_speed_m_s=SOUND_SPEED_M_S,
        random_seed=FEM_MESH_RANDOM_SEED,
    )
    runtime.preload_mkl()
    runtime.set_pardiso_threads()
    pressure = solve_fem_helmholtz(
        mesh,
        wall_impedances=wall_impedances,
        source=SOURCE,
        receiver=RECEIVER,
        frequencies_hz=FEM_COMPARISON_FREQUENCIES_HZ,
        density_kg_m3=DENSITY_KG_M3,
        sound_speed_m_s=SOUND_SPEED_M_S,
    )
    return mesh, wall_impedances, pressure


def _fake_fem_energy(
    *,
    room: Room,
    source: Point,
    receiver: Point,
    wall_impedances: Mapping[Wall, float],
    frequencies_hz: tuple[float, ...],
    density_kg_m3: float,
    sound_speed_m_s: float,
) -> tuple[float, ...]:
    del room, source, receiver, wall_impedances, density_kg_m3, sound_speed_m_s
    return tuple(_deterministic_fem_energy(value) for value in frequencies_hz)


@dataclass(frozen=True)
class _TimedReport:
    impedance_multiple: float
    elapsed_s: float
    report: ThreeLaneReport


@dataclass(frozen=True)
class _ExpectedBandMeans:
    fem: float | None
    fem_count: int
    direct: float
    reflected: float
    interference: float
    late: float
    scattering: float
    geometric: float
    fem_contribution: float
    geometric_contribution: float
    total: float
    w_fem: float
    w_geo: float
    t20: float
    t30: float


def _solve_fake_report(
    monkeypatch: pytest.MonkeyPatch,
    impedance_multiple: float,
    *,
    scattering: float | None = None,
    room: Room = ROOM,
    source: Point = SOURCE,
    receiver: Point = RECEIVER,
) -> ThreeLaneReport:
    monkeypatch.setattr(three_lane_report, "_solve_fem_energy", _fake_fem_energy)
    scattering_by_wall = (
        None if scattering is None else {wall: scattering for wall in Wall.all()}
    )
    return three_lane_report.solve_three_lane_report(
        room=room,
        source=source,
        receiver=receiver,
        sound_speed_m_s=SOUND_SPEED_M_S,
        density_kg_m3=DENSITY_KG_M3,
        impedance_by_wall=_walls(impedance_multiple * RHO_C_PA_S_PER_M),
        scattering_by_wall=scattering_by_wall,
    )


@pytest.fixture(params=(4.0, 10.0), ids=("flat", "lowabs"))
def timed_report(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> _TimedReport:
    """兩組材料走完整正式入口；只替換昂貴的 FEM 求解。"""
    multiple = float(request.param)
    started = time.perf_counter()
    report = _solve_fake_report(monkeypatch, multiple)
    return _TimedReport(multiple, time.perf_counter() - started, report)


def _expected_geometric(
    impedance: float,
    *,
    scattering: float | None = None,
) -> GeometricLaneResult:
    scattering_by_wall = (
        None
        if scattering is None
        else {wall.wall_name(): scattering for wall in Wall.all()}
    )
    return solve_geometric_lane(
        room=ROOM,
        source=SOURCE,
        receiver=RECEIVER,
        sound_speed_m_s=SOUND_SPEED_M_S,
        rho_c_pa_s_per_m=RHO_C_PA_S_PER_M,
        frequencies_hz=GEOMETRIC_LANE_FREQUENCIES_HZ,
        impedance_by_wall={wall.wall_name(): complex(impedance) for wall in Wall.all()},
        scattering_by_wall=scattering_by_wall,
    )


def _assert_source_results(timed: _TimedReport, impedance: float) -> None:
    absorption = {
        wall: {
            500.0: _random_absorption(impedance),
            1000.0: _random_absorption(impedance),
        }
        for wall in Wall.all()
    }
    expected_t60 = eyring_t60_by_band(ROOM, absorption)
    expected_f_s = schroeder_frequency_hz(ROOM, expected_t60)
    expected_weights = crossover_weights(GEOMETRIC_LANE_FREQUENCIES_HZ, expected_f_s)
    expected_geometric = _expected_geometric(impedance)
    expected_decay = solve_late_decay(
        LateEnergyInputs(
            room=ROOM,
            rho_c_pa_s_per_m=RHO_C_PA_S_PER_M,
            frequencies_hz=_report_decay_frequencies(),
            impedance_by_wall=_named_walls(
                complex(impedance), _report_decay_frequencies()
            ),
            n_per_wall=ART_N_PER_WALL_DEFAULT,
            domain_alpha_bar_max=math.inf,
        ),
        sound_speed_m_s=SOUND_SPEED_M_S,
    )

    actual = timed.report
    assert actual.eyring_t60_by_band_s == expected_t60
    assert actual.f_s_hz == expected_f_s
    assert actual.full_axis_weights == expected_weights
    assert actual.geometric_lane == expected_geometric
    assert actual.late_decay == expected_decay
    assert "每個細軸頻點" in actual.late_decay_frequency_policy


def _assert_pointwise_stitch(
    report: ThreeLaneReport,
    expected_geometric: GeometricLaneResult,
    expected_weights: CrossoverWeights,
) -> None:
    assert report.fem_frequencies_hz == FEM_LANE_FREQUENCIES_HZ
    assert report.fem_energy == tuple(
        _deterministic_fem_energy(value) for value in FEM_LANE_FREQUENCIES_HZ
    )
    assert tuple(point.frequency_hz for point in report.points) == (
        GEOMETRIC_LANE_FREQUENCIES_HZ
    )
    for index, point in enumerate(report.points):
        expected_fem = (
            _deterministic_fem_energy(point.frequency_hz)
            if point.frequency_hz <= FEM_GEOMETRIC_CROSSOVER_CAP_HZ
            else None
        )
        assert point.fem_energy == expected_fem
        assert point.direct_energy == expected_geometric.direct_energy[index]
        assert point.reflected_energy == expected_geometric.reflected_energy[index]
        assert point.interference_energy == expected_geometric.interference_energy[index]
        assert point.late_energy == expected_geometric.late_energy[index]
        assert point.scattering == expected_geometric.scattering[index]
        assert point.geometric_energy == expected_geometric.geometric_energy[index]
        assert point.w_fem == expected_weights.w_fem[index]
        assert point.w_geo == expected_weights.w_geo[index]
        if point.frequency_hz > FEM_GEOMETRIC_CROSSOVER_CAP_HZ:
            assert point.fem_energy is None
            assert point.total_energy == point.geometric_energy
        else:
            assert point.fem_energy is not None
            assert point.total_energy == (
                point.w_fem * point.fem_energy
                + point.w_geo * point.geometric_energy
            )
        if point.frequency_hz < report.crossover_lower_hz:
            assert point.total_energy == point.fem_energy


def _octave_band_mean(
    frequencies: tuple[float, ...], values: tuple[float, ...], *, lower: float, upper: float
) -> float:
    """獨立用 log2 格邊驗 #435 的帶平均（不呼叫產品程式），避免逐點算術平均暗中回來。"""
    positions = tuple(math.log2(frequency) for frequency in frequencies)
    edges = (
        max(positions[0] - (positions[1] - positions[0]) / 2, math.log2(lower)),
        *(0.5 * (left + right) for left, right in zip(positions, positions[1:])),
        min(positions[-1] + (positions[-1] - positions[-2]) / 2, math.log2(upper)),
    )
    widths = tuple(right - left for left, right in zip(edges, edges[1:]))
    return sum(width * value for width, value in zip(widths, values)) / sum(widths)


def _expected_decay(report: ThreeLaneReport, lower: float, upper: float) -> tuple[float, float]:
    """帶內 T20/T30 的算術平均；兩種報表都在正式細軸上算（#435）。"""
    decay_points = tuple(
        point
        for point in report.late_decay.bands
        if lower <= point.frequency_hz < upper
    )
    t20 = sum(point.t20_s for point in decay_points) / len(decay_points)
    t30 = sum(point.t30_s for point in decay_points if point.t30_s is not None) / len(decay_points)
    return t20, t30


def _expected_band_means(
    report: ThreeLaneReport,
    band: ThreeLaneBandReport,
    dense: GeometricEarlyResult,
    dense_weights: CrossoverWeights,
    fem_by_frequency: Mapping[float, float],
) -> _ExpectedBandMeans:
    root_two = math.sqrt(2.0)
    lower = band.center_frequency_hz / root_two
    upper = band.center_frequency_hz * root_two
    points = tuple(
        point for point in report.points if lower <= point.frequency_hz < upper
    )
    fem_values = tuple(
        energy
        for frequency, energy in fem_by_frequency.items()
        if lower <= frequency < upper
    )
    dense_indices = tuple(
        index
        for index, frequency in enumerate(dense.frequencies_hz)
        if lower <= frequency < upper
    )
    # 逐階分工之後早期三欄與晚期那一欄都已經含散射留存，這裡只相加。
    dense_early = tuple(
        dense.direct_energy[index]
        + dense.reflected_energy[index]
        + dense.interference_energy[index]
        for index in dense_indices
    )
    fine_late = tuple(point.late_energy for point in points)
    octave_mean = partial(_octave_band_mean, lower=lower, upper=upper)
    fine_frequencies = tuple(point.frequency_hz for point in points)
    fem_frequencies = tuple(frequency for frequency in fem_by_frequency if lower <= frequency < upper)
    fem_contribution = octave_mean(
        fine_frequencies,
        tuple(0.0 if point.fem_energy is None else point.w_fem * point.fem_energy for point in points),
    )
    geometric_contribution = sum(
        dense_weights.w_geo[index] * value
        for index, value in zip(dense_indices, dense_early, strict=True)
    ) / len(dense_indices) + octave_mean(
        fine_frequencies, tuple(point.w_geo * point.late_energy for point in points)
    )
    t20, t30 = _expected_decay(report, lower, upper)
    return _ExpectedBandMeans(
        fem=octave_mean(fem_frequencies, fem_values) if fem_values else None,
        fem_count=len(fem_values),
        direct=sum(dense.direct_energy[i] for i in dense_indices) / len(dense_indices),
        reflected=sum(dense.reflected_energy[i] for i in dense_indices)
        / len(dense_indices),
        interference=sum(dense.interference_energy[i] for i in dense_indices)
        / len(dense_indices),
        late=octave_mean(fine_frequencies, fine_late),
        scattering=sum(dense.scattering[i] for i in dense_indices) / len(dense_indices),
        geometric=sum(dense_early) / len(dense_indices) + octave_mean(fine_frequencies, fine_late),
        fem_contribution=fem_contribution,
        geometric_contribution=geometric_contribution,
        total=fem_contribution + geometric_contribution,
        w_fem=octave_mean(fine_frequencies, tuple(point.w_fem for point in points)),
        w_geo=octave_mean(fine_frequencies, tuple(point.w_geo for point in points)),
        t20=t20,
        t30=t30,
    )


def _assert_band_means(report: ThreeLaneReport, impedance: float) -> None:
    fem_by_frequency = dict(
        zip(report.fem_frequencies_hz, report.fem_energy, strict=True)
    )
    dense = solve_geometric_early_lane(
        room=ROOM,
        source=SOURCE,
        receiver=RECEIVER,
        sound_speed_m_s=SOUND_SPEED_M_S,
        rho_c_pa_s_per_m=RHO_C_PA_S_PER_M,
        frequencies_hz=GEOMETRIC_BAND_FREQUENCIES_HZ,
        impedance_by_wall={wall.wall_name(): complex(impedance) for wall in Wall.all()},
    )
    dense_weights = crossover_weights(
        GEOMETRIC_BAND_FREQUENCIES_HZ, report.f_s_hz
    )
    for band in report.bands:
        expected = _expected_band_means(
            report, band, dense, dense_weights, fem_by_frequency
        )
        assert band.fem_energy == expected.fem
        assert band.fem_point_count == expected.fem_count
        assert band.direct_energy == expected.direct
        assert band.reflected_energy == expected.reflected
        assert band.interference_energy == expected.interference
        assert band.late_energy == expected.late
        assert band.scattering == expected.scattering
        assert band.geometric_energy == expected.geometric
        assert band.fem_contribution == expected.fem_contribution
        assert band.geometric_contribution == expected.geometric_contribution
        assert band.total_energy == expected.total
        assert band.fem_contribution + band.geometric_contribution == band.total_energy
        assert band.w_fem == expected.w_fem
        assert band.w_geo == expected.w_geo
        assert band.t20_s == expected.t20
        assert band.t30_s == expected.t30


def test_report_reuses_all_sources_and_all_fine_axis_points_exactly(
    timed_report: _TimedReport,
) -> None:
    """兩組正式報表逐位守住來源、整條細軸、接合公式與頻帶平均。"""
    impedance = timed_report.impedance_multiple * RHO_C_PA_S_PER_M
    _assert_source_results(timed_report, impedance)
    expected_geometric = _expected_geometric(impedance)
    expected_weights = crossover_weights(
        GEOMETRIC_LANE_FREQUENCIES_HZ, timed_report.report.f_s_hz
    )
    _assert_pointwise_stitch(
        timed_report.report, expected_geometric, expected_weights
    )
    _assert_band_means(timed_report.report, impedance)


def test_hard_cut_report_uses_fem_through_cap_and_geometry_above(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """硬切下端若誤用 ``max(f_s, floor)``，本題必須紅。"""
    report = _solve_fake_report(
        monkeypatch,
        400.0,
        room=HARD_CUT_ROOM,
        source=HARD_CUT_SOURCE,
        receiver=HARD_CUT_RECEIVER,
    )

    assert report.crossover_lower_hz == FEM_GEOMETRIC_CROSSOVER_CAP_HZ
    assert report.capped_by_upper_limit is True
    for point in report.points:
        if point.frequency_hz <= FEM_GEOMETRIC_CROSSOVER_CAP_HZ:
            assert point.total_energy == point.fem_energy
        else:
            assert point.total_energy == point.geometric_energy


def test_band_fem_contribution_averages_every_fine_axis_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """抓 FEM 貢獻只除有 FEM 值的子集，重現 250 Hz 跨界帶錯分母。"""
    report = _solve_fake_report(monkeypatch, 4.0)
    crossing_band = next(
        band
        for band in report.bands
        if 0 < band.fem_point_count
        < sum(
            1
            for point in report.points
            if band.center_frequency_hz / math.sqrt(2.0)
            <= point.frequency_hz
            < band.center_frequency_hz * math.sqrt(2.0)
        )
    )
    lower = crossing_band.center_frequency_hz / math.sqrt(2.0)
    upper = crossing_band.center_frequency_hz * math.sqrt(2.0)
    points = tuple(
        point for point in report.points if lower <= point.frequency_hz < upper
    )
    contributions = tuple(
        0.0
        if point.fem_energy is None
        else point.w_fem * point.fem_energy
        for point in points
    )

    assert crossing_band.fem_contribution == _octave_band_mean(
        tuple(point.frequency_hz for point in points), contributions, lower=lower, upper=upper
    )
    assert crossing_band.fem_contribution != (
        sum(contributions) / crossing_band.fem_point_count
    )


def test_real_crossover_band_uses_dense_point_weights_for_early_energy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """真房間若把密軸早期貢獻退回細軸取樣，本題必須紅。"""
    scattering = 0.2
    impedance = 30.0 * RHO_C_PA_S_PER_M
    report = _solve_fake_report(monkeypatch, 30.0, scattering=scattering)
    assert CROSSOVER_LOWER_FLOOR_HZ < report.f_s_hz < FEM_GEOMETRIC_CROSSOVER_CAP_HZ

    center_hz = 250.0
    lower = center_hz / math.sqrt(2.0)
    upper = center_hz * math.sqrt(2.0)
    band = next(item for item in report.bands if item.center_frequency_hz == center_hz)
    fine_points = tuple(
        point for point in report.points if lower <= point.frequency_hz < upper
    )
    dense = solve_geometric_early_lane(
        room=ROOM,
        source=SOURCE,
        receiver=RECEIVER,
        sound_speed_m_s=SOUND_SPEED_M_S,
        rho_c_pa_s_per_m=RHO_C_PA_S_PER_M,
        frequencies_hz=GEOMETRIC_BAND_FREQUENCIES_HZ,
        impedance_by_wall={
            wall.wall_name(): complex(impedance) for wall in Wall.all()
        },
        scattering_by_wall={wall.wall_name(): scattering for wall in Wall.all()},
    )
    dense_weights = crossover_weights(GEOMETRIC_BAND_FREQUENCIES_HZ, report.f_s_hz)
    dense_indices = tuple(
        index
        for index, frequency in enumerate(dense.frequencies_hz)
        if lower <= frequency < upper
    )
    dense_weighted_early = sum(
        dense_weights.w_geo[index]
        * (
            dense.direct_energy[index]
            + dense.reflected_energy[index]
            + dense.interference_energy[index]
        )
        for index in dense_indices
    ) / len(dense_indices)
    fine_weighted_early = sum(
        point.w_geo
        * (
            point.direct_energy
            + point.reflected_energy
            + point.interference_energy
        )
        for point in fine_points
    ) / len(fine_points)
    fine_weighted_late = _octave_band_mean(
        tuple(point.frequency_hz for point in fine_points),
        tuple(point.w_geo * point.late_energy for point in fine_points),
        lower=lower,
        upper=upper,
    )
    expected = dense_weighted_early + fine_weighted_late
    fine_axis_counterfactual = fine_weighted_early + fine_weighted_late

    assert band.geometric_contribution == expected
    assert band.geometric_contribution != fine_axis_counterfactual


def test_band_report_uses_dense_early_fields_and_fine_late_field() -> None:
    """報表若把所有帶欄整批退回細軸平均，密軸的分項與合成必須抓到。"""
    centers = GEOMETRIC_REPORT_OCTAVE_CENTERS_HZ
    fine = GeometricLaneResult(
        frequencies_hz=centers,
        direct_energy=tuple(100.0 for _center in centers),
        reflected_energy=tuple(100.0 for _center in centers),
        interference_energy=tuple(100.0 for _center in centers),
        late_energy=tuple(10.0 for _center in centers),
        scattering=tuple(0.4 for _center in centers),
        geometric_energy=tuple(999.0 for _center in centers),
        reflection_order_k=REFLECTION_ORDER_K,
    )
    fine_weights = CrossoverWeights(
        w_fem=tuple(0.8 for _center in centers),
        w_geo=tuple(0.2 for _center in centers),
        capped_by_upper_limit=False,
    )
    fem_by_frequency = {center: 5.0 for center in centers}
    points = three_lane_report.stitch_energy_points(
        frequencies_hz=centers,
        fem_energy_by_frequency=fem_by_frequency,
        geometric_lane=fine,
        geometric_indices=tuple(range(len(centers))),
        weights=fine_weights,
    )
    dense = GeometricEarlyResult(
        frequencies_hz=centers,
        direct_energy=tuple(1.0 for _center in centers),
        reflected_energy=tuple(2.0 for _center in centers),
        interference_energy=tuple(0.5 for _center in centers),
        scattering=tuple(0.1 for _center in centers),
        reflection_order_k=REFLECTION_ORDER_K,
    )
    dense_weights = CrossoverWeights(
        w_fem=tuple(0.7 for _center in centers),
        w_geo=tuple(0.3 for _center in centers),
        capped_by_upper_limit=False,
    )
    decay = _constant_late_decay(centers)

    actual = three_lane_report._band_reports(
        report_points=points,
        fem_energy_by_frequency=fem_by_frequency,
        geometric_lane=fine,
        dense_early_lane=dense,
        full_axis_weights=fine_weights,
        dense_axis_weights=dense_weights,
        late_decay=decay,
        decay_unavailable_by_center_hz={},
        f_s_hz=200.0,
    )
    expected_geometric_contribution = (
        dense_weights.w_geo[0] * 3.5
        + fine_weights.w_geo[0] * fine.late_energy[0]
    )

    for band in actual:
        assert band.direct_energy == 1.0
        assert band.reflected_energy == 2.0
        assert band.interference_energy == 0.5
        assert band.late_energy == 10.0
        assert band.scattering == 0.1
        assert band.geometric_energy == 3.5 + 10.0
        assert band.fem_contribution == 4.0
        assert band.geometric_contribution == expected_geometric_contribution
        assert band.total_energy == (
            band.fem_contribution + band.geometric_contribution
        )


@pytest.mark.parametrize(
    ("impedance_multiple", "t20_available", "t20_lower", "t30_lower"),
    (
        (T20_ONLY_IMPEDANCE_MULTIPLE, True, None, "-35 dB"),
        (UNREACHED_T20_IMPEDANCE_MULTIPLE, False, "-25 dB", "-35 dB"),
    ),
)
def test_report_marks_only_unreached_decay_windows_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    impedance_multiple: float,
    t20_available: bool,
    t20_lower: str | None,
    t30_lower: str,
) -> None:
    """Paris α=0.026 保留 T20；α=0.02 讓未達下緣欄位失效。"""
    report = _solve_fake_report(monkeypatch, impedance_multiple)

    assert report.points
    for band in report.bands:
        assert math.isfinite(band.total_energy)
        assert math.isfinite(band.geometric_energy)
        assert (band.t20_s is not None) is t20_available
        assert (band.t20_unavailable_reason is None) is t20_available
        if t20_lower is not None:
            assert band.t20_unavailable_reason is not None
            assert t20_lower in band.t20_unavailable_reason
            assert "第 256 階最低" in band.t20_unavailable_reason
        assert band.t30_s is None
        assert band.t30_unavailable_reason is not None
        assert t30_lower in band.t30_unavailable_reason
        assert "第 256 階最低" in band.t30_unavailable_reason


def test_report_does_not_catch_unrelated_late_decay_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """抓報表用 except ValueError 或 except Exception 吞掉非下緣錯誤。"""

    def unrelated_failure(
        inputs: LateEnergyInputs,
        *,
        sound_speed_m_s: float,
    ) -> LateDecayResult:
        del inputs, sound_speed_m_s
        raise ValueError("別種晚期衰減錯誤")

    monkeypatch.setattr(three_lane_report, "solve_late_decay", unrelated_failure)
    with pytest.raises(ValueError, match="別種晚期衰減錯誤"):
        _solve_fake_report(monkeypatch, 4.0)


def test_report_scattering_parameter_reaches_every_geometric_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """非缺省散射若被正式入口吃掉或逐點欄位錯位，本題必須紅。"""
    impedance = 4.0 * RHO_C_PA_S_PER_M
    default = _solve_fake_report(monkeypatch, 4.0)
    scattering = 0.65
    actual = _solve_fake_report(monkeypatch, 4.0, scattering=scattering)
    expected = _expected_geometric(impedance, scattering=scattering)

    assert actual.geometric_lane.geometric_energy != (
        default.geometric_lane.geometric_energy
    )
    for index, point in enumerate(actual.points):
        assert point.scattering == expected.scattering[index]
        assert point.geometric_energy == expected.geometric_energy[index]


def test_stitch_energy_points_rejects_missing_positive_weight_fem_value() -> None:
    """直接入口少了正權重 FEM 值時必須報錯。"""
    frequency = FEM_LANE_FREQUENCIES_HZ[0]
    geometric = GeometricLaneResult(
        frequencies_hz=(frequency,),
        direct_energy=(1.0,),
        reflected_energy=(2.0,),
        interference_energy=(-1.0,),
        late_energy=(3.0,),
        scattering=(0.1,),
        geometric_energy=(4.0,),
        reflection_order_k=REFLECTION_ORDER_K,
    )
    weights = CrossoverWeights(w_fem=(1.0,), w_geo=(0.0,), capped_by_upper_limit=False)

    with pytest.raises(ValueError, match="w_fem > 0"):
        three_lane_report.stitch_energy_points(
            frequencies_hz=(frequency,),
            fem_energy_by_frequency={},
            geometric_lane=geometric,
            geometric_indices=(0,),
            weights=weights,
        )


@pytest.mark.parametrize(
    "invalid_impedance",
    (
        complex(4.0 * RHO_C_PA_S_PER_M, 0.0),
        (4.0 * RHO_C_PA_S_PER_M, 4.0 * RHO_C_PA_S_PER_M),
    ),
    ids=("complex", "frequency-dependent"),
)
def test_report_rejects_impedance_shapes_not_supported_by_fem(
    invalid_impedance: object,
) -> None:
    """複數或隨頻率阻抗若滑進目前只吃實數常數的 FEM 路必須紅。"""
    with pytest.raises(ValueError, match="頻率無關的正有限實數"):
        three_lane_report.solve_three_lane_report(
            room=ROOM,
            source=SOURCE,
            receiver=RECEIVER,
            sound_speed_m_s=SOUND_SPEED_M_S,
            density_kg_m3=DENSITY_KG_M3,
            impedance_by_wall=_walls(invalid_impedance),
        )


def test_report_entry_accepts_no_fem_subset_or_external_rho_c() -> None:
    """正式入口不得開放 FEM 子集，也不得另收會與密度聲速漂開的 rho_c。"""
    parameters = inspect.signature(three_lane_report.solve_three_lane_report).parameters
    assert "fem_frequencies_hz" not in parameters
    assert "rho_c_pa_s_per_m" not in parameters


def test_report_rejects_when_fem_solver_returns_one_fewer_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """正式入口收到缺一點的 FEM 結果時必須直接報錯。"""

    def missing_one_fem_energy(
        *,
        room: Room,
        source: Point,
        receiver: Point,
        wall_impedances: Mapping[Wall, float],
        frequencies_hz: tuple[float, ...],
        density_kg_m3: float,
        sound_speed_m_s: float,
    ) -> tuple[float, ...]:
        values = _fake_fem_energy(
            room=room,
            source=source,
            receiver=receiver,
            wall_impedances=wall_impedances,
            frequencies_hz=frequencies_hz,
            density_kg_m3=density_kg_m3,
            sound_speed_m_s=sound_speed_m_s,
        )
        return values[:-1]

    monkeypatch.setattr(
        three_lane_report, "_solve_fem_energy", missing_one_fem_energy
    )
    with pytest.raises(ValueError, match="有限元素"):
        three_lane_report.solve_three_lane_report(
            room=ROOM,
            source=SOURCE,
            receiver=RECEIVER,
            sound_speed_m_s=SOUND_SPEED_M_S,
            density_kg_m3=DENSITY_KG_M3,
            impedance_by_wall=_walls(4.0 * RHO_C_PA_S_PER_M),
        )


def test_report_averages_decay_from_every_fine_axis_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """抓報表只在帶中心算 T20/T30，而不是把帶內細軸結果逐位平均。"""
    called_frequencies: list[tuple[float, ...]] = []

    def frequency_labeled_decay(
        inputs: LateEnergyInputs,
        *,
        sound_speed_m_s: float,
    ) -> LateDecayResult:
        called_frequencies.append(inputs.frequencies_hz)
        assert sound_speed_m_s == SOUND_SPEED_M_S
        return LateDecayResult(
            orders_used=256,
            bands=tuple(
                LateDecayBand(
                    frequency_hz=frequency,
                    t20_s=frequency,
                    collision_frequency_hz=1.0,
                    slope_db_per_s=-1.0,
                    soft_weight_sum=1.0,
                    fell_back_to_perron=False,
                    perron_t60_s=1.0,
                    t30_s=2.0 * frequency,
                    t30_slope_db_per_s=-1.0,
                    t30_soft_weight_sum=1.0,
                )
                for frequency in inputs.frequencies_hz
            ),
        )

    monkeypatch.setattr(three_lane_report, "solve_late_decay", frequency_labeled_decay)
    report = _solve_fake_report(monkeypatch, 4.0)

    root_two = math.sqrt(2.0)
    expected_calls = [
        tuple(
            frequency
            for frequency in _report_decay_frequencies()
            if center / root_two <= frequency < center * root_two
        )
        for center in GEOMETRIC_REPORT_OCTAVE_CENTERS_HZ
    ]
    assert called_frequencies == expected_calls
    for band in report.bands:
        frequencies = tuple(
            frequency
            for frequency in _report_decay_frequencies()
            if band.center_frequency_hz / root_two
            <= frequency
            < band.center_frequency_hz * root_two
        )
        assert band.t20_s == sum(frequencies) / len(frequencies)
        assert band.t30_s == sum(2.0 * value for value in frequencies) / len(
            frequencies
        )


def test_report_does_not_solve_late_energy_on_the_dense_axis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """密頻率入口若連晚期能量一起重算，替身必須抓到多餘呼叫。"""
    from aosr.physics.late_energy import solve_late_energy_by_order

    called_frequencies: list[tuple[float, ...]] = []

    def record_late_energy(
        inputs: LateEnergyInputs, *, max_order: int
    ) -> LateEnergyOrderResult:
        called_frequencies.append(inputs.frequencies_hz)
        if inputs.frequencies_hz != GEOMETRIC_LANE_FREQUENCIES_HZ:
            raise AssertionError("晚期能量收到密頻率軸")
        assert max_order == REFLECTION_ORDER_K
        return solve_late_energy_by_order(inputs, max_order=max_order)

    monkeypatch.setattr(
        "aosr.physics.geometric_lane.solve_late_energy_by_order", record_late_energy
    )
    _solve_fake_report(monkeypatch, 4.0)

    assert called_frequencies == [GEOMETRIC_LANE_FREQUENCIES_HZ]


def test_solve_fem_energy_matches_direct_solver_on_official_mesh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FEM 包裝在正式網格兩點逐位等於同網格直接求解的能量。"""
    mesh, expected_wall_impedances, pressure = (
        _solve_directly_on_official_mesh()
    )

    def reuse_official_mesh(
        room: Room,
        *,
        max_frequency_hz: float,
        elements_per_wavelength: float,
        sound_speed_m_s: float,
        random_seed: int,
    ) -> ShoeboxMesh:
        assert room == ROOM
        assert max_frequency_hz == FEM_GEOMETRIC_CROSSOVER_CAP_HZ
        assert elements_per_wavelength == FEM_ELEMENTS_PER_WAVELENGTH
        assert sound_speed_m_s == SOUND_SPEED_M_S
        assert random_seed == FEM_MESH_RANDOM_SEED
        return mesh

    def replay_direct_pressure(
        actual_mesh: ShoeboxMesh,
        *,
        wall_impedances: WallImpedances,
        source: Point,
        receiver: Point,
        frequencies_hz: Sequence[float] | NDArray[np.float64],
        density_kg_m3: float,
        sound_speed_m_s: float,
    ) -> NDArray[np.complex128]:
        assert actual_mesh is mesh
        assert wall_impedances == expected_wall_impedances
        assert source == SOURCE
        assert receiver == RECEIVER
        assert tuple(frequencies_hz) == FEM_COMPARISON_FREQUENCIES_HZ
        assert density_kg_m3 == DENSITY_KG_M3
        assert sound_speed_m_s == SOUND_SPEED_M_S
        return pressure

    monkeypatch.setattr(three_lane_report, "generate_shoebox_mesh", reuse_official_mesh)
    monkeypatch.setattr(
        three_lane_report, "solve_fem_helmholtz", replay_direct_pressure
    )
    actual = three_lane_report._solve_fem_energy(
        room=ROOM,
        source=SOURCE,
        receiver=RECEIVER,
        wall_impedances=expected_wall_impedances,
        frequencies_hz=FEM_COMPARISON_FREQUENCIES_HZ,
        density_kg_m3=DENSITY_KG_M3,
        sound_speed_m_s=SOUND_SPEED_M_S,
    )

    assert actual == tuple(float(abs(value) ** 2) for value in pressure)
