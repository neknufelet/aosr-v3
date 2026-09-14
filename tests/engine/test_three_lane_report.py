"""第九段三路接合的結構化物理量報表考卷。"""

from __future__ import annotations

import inspect
import math
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
    GEOMETRIC_LANE_FREQUENCIES_HZ,
    GEOMETRIC_REPORT_OCTAVE_CENTERS_HZ,
)
from aosr.geometry.shoebox import Point, Room, Wall
from aosr.geometry.shoebox_mesh import ShoeboxMesh, generate_shoebox_mesh
from aosr.physics import three_lane_report
from aosr.physics.crossover import (
    CrossoverWeights,
    crossover_weights,
    eyring_t60_by_band,
    schroeder_frequency_hz,
)
from aosr.physics.fem_helmholtz import WallImpedances, solve_fem_helmholtz
from aosr.physics.geometric_lane import (
    GeometricLaneResult,
    average_geometric_lane_to_bands,
    solve_geometric_lane,
)
from aosr.physics.late_decay import solve_late_decay
from aosr.physics.late_energy import LateEnergyInputs
from aosr.physics.three_lane_report import ThreeLaneReport


ROOM = Room(6.0, 4.0, 3.0)
SOURCE = Point(1.2, 1.3, 1.1)
RECEIVER = Point(4.7, 2.8, 1.4)
SOUND_SPEED_M_S = 343.0
DENSITY_KG_M3 = 1.2
RHO_C_PA_S_PER_M = 411.6
FEM_COMPARISON_FREQUENCIES_HZ = (
    FEM_LANE_FREQUENCIES_HZ[0],
    FEM_LANE_FREQUENCIES_HZ[-1],
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


def _normal_absorption(impedance: float) -> float:
    reflection = (impedance - RHO_C_PA_S_PER_M) / (impedance + RHO_C_PA_S_PER_M)
    return 1.0 - reflection * reflection


def _deterministic_fem_energy(frequency_hz: float) -> float:
    return frequency_hz * 3.0 + 7.0


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


def _solve_fake_report(
    monkeypatch: pytest.MonkeyPatch,
    impedance_multiple: float,
    *,
    scattering: float | None = None,
) -> ThreeLaneReport:
    monkeypatch.setattr(three_lane_report, "_solve_fem_energy", _fake_fem_energy)
    scattering_by_wall = (
        None if scattering is None else {wall: scattering for wall in Wall.all()}
    )
    return three_lane_report.solve_three_lane_report(
        room=ROOM,
        source=SOURCE,
        receiver=RECEIVER,
        sound_speed_m_s=SOUND_SPEED_M_S,
        density_kg_m3=DENSITY_KG_M3,
        rho_c_pa_s_per_m=RHO_C_PA_S_PER_M,
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
            500.0: _normal_absorption(impedance),
            1000.0: _normal_absorption(impedance),
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
            frequencies_hz=GEOMETRIC_REPORT_OCTAVE_CENTERS_HZ,
            impedance_by_wall=_named_walls(
                complex(impedance), GEOMETRIC_REPORT_OCTAVE_CENTERS_HZ
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


def _assert_band_means(report: ThreeLaneReport) -> None:
    fem_by_frequency = dict(
        zip(report.fem_frequencies_hz, report.fem_energy, strict=True)
    )
    root_two = math.sqrt(2.0)
    decay_by_frequency = {band.frequency_hz: band for band in report.late_decay.bands}
    geometric_bands = average_geometric_lane_to_bands(report.geometric_lane)
    for band in report.bands:
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
        expected_fem = sum(fem_values) / len(fem_values) if fem_values else None
        expected_total = sum(point.total_energy for point in points) / len(points)
        expected_w_fem = sum(point.w_fem for point in points) / len(points)
        expected_w_geo = sum(point.w_geo for point in points) / len(points)
        decay = decay_by_frequency[band.center_frequency_hz]
        geometric_index = geometric_bands.band_centers_hz.index(
            band.center_frequency_hz
        )

        assert band.fem_energy == expected_fem
        assert band.fem_point_count == len(fem_values)
        assert band.direct_energy == geometric_bands.direct_energy[geometric_index]
        assert band.reflected_energy == geometric_bands.reflected_energy[geometric_index]
        assert band.late_energy == geometric_bands.late_energy[geometric_index]
        assert band.scattering == geometric_bands.scattering[geometric_index]
        assert band.geometric_energy == geometric_bands.geometric_energy[geometric_index]
        assert band.total_energy == expected_total
        assert band.w_fem == expected_w_fem
        assert band.w_geo == expected_w_geo
        assert band.t20_s == decay.t20_s
        assert band.t30_s == decay.t30_s


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
    _assert_band_means(timed_report.report)


def test_hard_cut_report_uses_fem_through_cap_and_geometry_above(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """硬切下端若誤用 ``max(f_s, floor)``，本題必須紅。"""
    report = _solve_fake_report(monkeypatch, 400.0)

    assert report.crossover_lower_hz == FEM_GEOMETRIC_CROSSOVER_CAP_HZ
    assert report.capped_by_upper_limit is True
    for point in report.points:
        if point.frequency_hz <= FEM_GEOMETRIC_CROSSOVER_CAP_HZ:
            assert point.total_energy == point.fem_energy
        else:
            assert point.total_energy == point.geometric_energy


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
        late_energy=(3.0,),
        scattering=(0.1,),
        geometric_energy=(4.0,),
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
            rho_c_pa_s_per_m=RHO_C_PA_S_PER_M,
            impedance_by_wall=_walls(invalid_impedance),
        )


def test_report_entry_has_no_fem_frequency_subset_parameter() -> None:
    """正式入口不得重新開放會讓報表丟點的 FEM 頻點子集。"""
    parameters = inspect.signature(three_lane_report.solve_three_lane_report).parameters
    assert "fem_frequencies_hz" not in parameters


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
            rho_c_pa_s_per_m=RHO_C_PA_S_PER_M,
            impedance_by_wall=_walls(4.0 * RHO_C_PA_S_PER_M),
        )


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
