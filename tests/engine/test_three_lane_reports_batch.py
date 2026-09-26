"""同一候選的有限元素分解與晚期混響只能共用一次。"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from aosr.config.fem_lane import FEM_ELEMENTS_PER_WAVELENGTH, FEM_MESH_RANDOM_SEED
from aosr.config.frequency_axis import (
    FEM_GEOMETRIC_CROSSOVER_CAP_HZ,
    GEOMETRIC_LANE_FREQUENCIES_HZ,
    GEOMETRIC_REPORT_OCTAVE_CENTERS_HZ,
    LowFrequencyAxis,
)
from aosr.geometry.shoebox import Point, Room, Wall
from aosr.geometry.shoebox_mesh import generate_shoebox_mesh
from aosr import runtime
from aosr.physics import fem_helmholtz, geometric_lane, three_lane_report
from aosr.physics.report_source import SourceModelKind, SourceModelSpec
from aosr.physics.late_decay import LateDecayBand, LateDecayResult
from aosr.physics.late_energy import LateEnergyInputs, LateEnergyOrderResult, solve_late_energy_by_order


ROOM = Room(6.0, 4.0, 3.0)
SOURCES = {"left": Point(1.2, 1.3, 1.1), "right": Point(1.8, 2.0, 1.2)}
RECEIVERS = {"front": Point(4.7, 2.8, 1.4), "back": Point(3.8, 2.1, 1.5)}
SOUND_SPEED = 343.0
DENSITY = 1.2
FREQUENCIES = (40.0, 80.0, 160.0)
# 有限元素那幾題用小房間：同一段程式、未知數少很多，幾秒內跑完（正式大小的房間一題要五十幾秒）。
FEM_ROOM = Room(1.6, 1.2, 1.0)
FEM_SOURCES = {"left": Point(0.3, 0.3, 0.4), "right": Point(0.4, 0.8, 0.5)}
FEM_RECEIVERS = {"front": Point(1.2, 0.7, 0.6), "back": Point(1.0, 0.4, 0.3)}
WALLS = {wall: 4.0 * SOUND_SPEED * DENSITY for wall in Wall.all()}


def _fast_late_decay(
    *, room: Room, wall_impedances: Mapping[Wall, float],
    rho_c_pa_s_per_m: float, sound_speed_m_s: float,
) -> three_lane_report._ReportLateDecay:
    """沿用低頻軸考卷的固定衰減欄，省去慢速逐頻衰減求解。"""
    del room, wall_impedances, rho_c_pa_s_per_m, sound_speed_m_s
    root_two = math.sqrt(2.0)
    frequencies = tuple(
        frequency for frequency in GEOMETRIC_LANE_FREQUENCIES_HZ
        if any(
            center / root_two <= frequency < center * root_two
            for center in GEOMETRIC_REPORT_OCTAVE_CENTERS_HZ
        )
    )
    bands = tuple(
        LateDecayBand(
            frequency_hz=frequency, t20_s=frequency / 1000.0,
            collision_frequency_hz=1.0, slope_db_per_s=-1.0,
            soft_weight_sum=1.0, fell_back_to_perron=False, perron_t60_s=1.0,
            t30_s=frequency / 500.0, t30_slope_db_per_s=-1.0,
            t30_soft_weight_sum=1.0,
        )
        for frequency in frequencies
    )
    return three_lane_report._ReportLateDecay(
        result=LateDecayResult(orders_used=1, bands=bands),
        unavailable_by_center_hz={},
    )


@pytest.fixture(scope="module")
def operators() -> fem_helmholtz.P2Operators:
    """正式 300 Hz 網格設定、小房間，只縮短頻率軸控制求解時間。"""
    mesh = generate_shoebox_mesh(
        FEM_ROOM,
        max_frequency_hz=FEM_GEOMETRIC_CROSSOVER_CAP_HZ,
        elements_per_wavelength=FEM_ELEMENTS_PER_WAVELENGTH,
        sound_speed_m_s=SOUND_SPEED,
        random_seed=FEM_MESH_RANDOM_SEED,
    )
    return fem_helmholtz.assemble_p2_operators(mesh)


def _loads(operators: fem_helmholtz.P2Operators) -> dict[str, np.ndarray]:
    return {
        name: fem_helmholtz.point_source_load(operators, point) for name, point in FEM_SOURCES.items()
    }


def _many(operators: fem_helmholtz.P2Operators) -> dict[tuple[str, str], np.ndarray]:
    return fem_helmholtz.solve_frequency_responses_many(
        operators,
        right_hand_sides=_loads(operators),
        receivers=FEM_RECEIVERS,
        frequencies_hz=FREQUENCIES,
        wall_impedances=WALLS,
        density_kg_m3=DENSITY,
        sound_speed_m_s=SOUND_SPEED,
    )


def test_many_fem_responses_equal_individual_solves_bitwise(
    operators: fem_helmholtz.P2Operators,
) -> None:
    """若多源路徑改變求解或取值順序，逐點聲壓會失去逐位相等。"""
    actual = _many(operators)
    loads = _loads(operators)
    assert set(actual) == {(source, receiver) for source in FEM_SOURCES for receiver in FEM_RECEIVERS}
    for (source, receiver), pressure in actual.items():
        individual = fem_helmholtz.solve_frequency_responses(
            operators,
            right_hand_side=loads[source],
            receiver=FEM_RECEIVERS[receiver],
            frequencies_hz=FREQUENCIES,
            wall_impedances=WALLS,
            density_kg_m3=DENSITY,
            sound_speed_m_s=SOUND_SPEED,
        )
        assert tuple(pressure) == tuple(individual)


def test_many_fem_factors_once_per_frequency(
    operators: fem_helmholtz.P2Operators, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """新增聲源只增加 solve，不增加 factor 或 refactor。"""
    runtime.preload_mkl()
    from pydiso import mkl_solver

    real_solver = mkl_solver.MKLPardisoSolver
    calls = {"factor": 0, "refactor": 0, "solve": 0}

    class CountedSolver:
        def __init__(
            self, system: csr_matrix[np.complex128],
            matrix_type: str, factor: bool,
        ) -> None:
            assert factor
            calls["factor"] += 1
            self.solver = real_solver(system, matrix_type=matrix_type, factor=factor)

        def refactor(self, system: csr_matrix[np.complex128]) -> None:
            calls["refactor"] += 1
            self.solver.refactor(system)

        def solve(self, right_hand_side: NDArray[np.complex128]) -> NDArray[np.complex128]:
            assert right_hand_side.shape == (operators.basis.N,)
            calls["solve"] += 1
            return self.solver.solve(right_hand_side)

    monkeypatch.setattr(mkl_solver, "MKLPardisoSolver", CountedSolver)
    actual = _many(operators)
    assert set(actual) == {(source, receiver) for source in FEM_SOURCES for receiver in FEM_RECEIVERS}
    assert calls["factor"] + calls["refactor"] == len(FREQUENCIES)
    assert calls["solve"] == len(FREQUENCIES) * len(FEM_SOURCES)


def _fake_energy(
    frequencies_hz: tuple[float, ...], source: Point, receiver: Point,
) -> tuple[float, ...]:
    return tuple(frequency + source.x * 3.0 + receiver.y for frequency in frequencies_hz)


def _fake_single_fem(
    *, room: Room, source: Point, receiver: Point,
    wall_impedances: Mapping[Wall, float], frequencies_hz: tuple[float, ...],
    density_kg_m3: float, sound_speed_m_s: float,
) -> tuple[float, ...]:
    del room, wall_impedances, density_kg_m3, sound_speed_m_s
    return _fake_energy(frequencies_hz, source, receiver)


def _fake_many_fem(
    *, room: Room, sources: Mapping[str, Point], receivers: Mapping[str, Point],
    wall_impedances: Mapping[Wall, float], frequencies_hz: tuple[float, ...],
    density_kg_m3: float, sound_speed_m_s: float,
) -> dict[tuple[str, str], tuple[float, ...]]:
    del room, wall_impedances, density_kg_m3, sound_speed_m_s
    return {
        (source_name, receiver_name): _fake_energy(frequencies_hz, source, receiver)
        for source_name, source in sources.items()
        for receiver_name, receiver in receivers.items()
    }


def _batch(
    axis: LowFrequencyAxis,
    sources: Mapping[str, Point] = SOURCES,
    receivers: Mapping[str, Point] = RECEIVERS,
) -> dict[tuple[str, str], three_lane_report.ThreeLaneReport]:
    return three_lane_report.solve_three_lane_reports(
        source_model=SourceModelSpec(kind=SourceModelKind.OMNIDIRECTIONAL),
        room=ROOM, sources=sources, receivers=receivers,
        sound_speed_m_s=SOUND_SPEED, density_kg_m3=DENSITY,
        impedance_by_wall=WALLS, low_frequency_axis=axis,
    )


@pytest.mark.parametrize("axis", (LowFrequencyAxis.SEARCH, LowFrequencyAxis.VERIFICATION))
def test_candidate_reports_equal_individual_reports_bitwise(
    axis: LowFrequencyAxis, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """任一欄或任一軸改變，完整報表的 dataclass 相等會失敗。"""
    monkeypatch.setattr(three_lane_report, "_solve_fem_energy", _fake_single_fem)
    monkeypatch.setattr(three_lane_report, "_solve_fem_energies", _fake_many_fem)
    monkeypatch.setattr(three_lane_report, "_solve_report_late_decay", _fast_late_decay)
    actual = _batch(axis)
    assert set(actual) == {(source, receiver) for source in SOURCES for receiver in RECEIVERS}
    for (source, receiver), report in actual.items():
        individual = three_lane_report.solve_three_lane_report(
            source_model=SourceModelSpec(kind=SourceModelKind.OMNIDIRECTIONAL),
            room=ROOM, source=SOURCES[source], receiver=RECEIVERS[receiver],
            sound_speed_m_s=SOUND_SPEED, density_kg_m3=DENSITY,
            impedance_by_wall=WALLS, low_frequency_axis=axis,
        )
        assert report == individual


def test_candidate_reports_solve_late_work_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """晚期能量與衰減的呼叫數不受聲源和接收點數影響。"""
    monkeypatch.setattr(three_lane_report, "_solve_fem_energies", _fake_many_fem)
    calls = {"energy": 0, "decay": 0}

    def count_late(
        inputs: LateEnergyInputs, *, max_order: int,
    ) -> LateEnergyOrderResult:
        calls["energy"] += 1
        return solve_late_energy_by_order(inputs, max_order=max_order)

    def count_decay(
        *, room: Room, wall_impedances: Mapping[Wall, float],
        rho_c_pa_s_per_m: float, sound_speed_m_s: float,
    ) -> three_lane_report._ReportLateDecay:
        calls["decay"] += 1
        return _fast_late_decay(
            room=room, wall_impedances=wall_impedances,
            rho_c_pa_s_per_m=rho_c_pa_s_per_m, sound_speed_m_s=sound_speed_m_s,
        )

    monkeypatch.setattr(geometric_lane, "solve_late_energy_by_order", count_late)
    monkeypatch.setattr(three_lane_report, "_solve_report_late_decay", count_decay)
    reports = _batch(LowFrequencyAxis.SEARCH)
    assert set(reports) == {(source, receiver) for source in SOURCES for receiver in RECEIVERS}
    assert calls["energy"] == len({ROOM})
    assert calls["decay"] == len({ROOM})


@pytest.mark.parametrize("sources,receivers", (({}, RECEIVERS), (SOURCES, {})))
def test_candidate_reports_reject_empty_positions(
    sources: Mapping[str, Point], receivers: Mapping[str, Point],
) -> None:
    with pytest.raises(ValueError):
        _batch(LowFrequencyAxis.SEARCH, sources=sources, receivers=receivers)


def test_candidate_entry_never_falls_back_to_per_pair_fem(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """候選入口只准走「每個頻率分解一次」那條有限元素路；偷偷改走逐份那條（答案一樣、慢十倍）這題會紅。
    主對話突變抓到：逐位相同那題兩條路用同一套替身，看不出候選入口走了哪一條。"""

    def forbidden(**_: object) -> tuple[float, ...]:
        raise AssertionError("候選入口不准走逐份有限元素")

    calls: list[int] = []

    def counted_many(
        *, room: Room, sources: Mapping[str, Point], receivers: Mapping[str, Point],
        wall_impedances: Mapping[Wall, float], frequencies_hz: tuple[float, ...],
        density_kg_m3: float, sound_speed_m_s: float,
    ) -> dict[tuple[str, str], tuple[float, ...]]:
        calls.append(1)
        return _fake_many_fem(
            room=room, sources=sources, receivers=receivers, wall_impedances=wall_impedances,
            frequencies_hz=frequencies_hz, density_kg_m3=density_kg_m3,
            sound_speed_m_s=sound_speed_m_s,
        )

    monkeypatch.setattr(three_lane_report, "_solve_fem_energy", forbidden)
    monkeypatch.setattr(three_lane_report, "_solve_fem_energies", counted_many)
    monkeypatch.setattr(three_lane_report, "_solve_report_late_decay", _fast_late_decay)

    reports = _batch(LowFrequencyAxis.SEARCH)

    assert set(reports) == {(source, receiver) for source in SOURCES for receiver in RECEIVERS}
    assert calls == [1]


def test_candidate_fem_wrapper_equals_per_pair_fem_bitwise_on_a_small_room() -> None:
    """候選入口真正的有限元素接線（建網格、組聲源載荷、傳接收點、聲壓取模平方）要逐位等於逐份那條；
    其他題都用替身換掉它，這一題真跑（找碴席抓到：整段換成必定報錯，其他考卷照樣全綠）。
    小房間、兩個頻點，讓正式網格設定下的有限元素幾秒內跑完。"""
    room = Room(1.6, 1.2, 1.0)
    sources = {"left": Point(0.3, 0.3, 0.4), "right": Point(0.4, 0.8, 0.5)}
    receivers = {"front": Point(1.2, 0.7, 0.6), "back": Point(1.0, 0.4, 0.3)}
    walls = {wall: 4.0 * SOUND_SPEED * DENSITY for wall in Wall.all()}
    frequencies = (140.0, 230.0)

    batch = three_lane_report._solve_fem_energies(
        room=room, sources=sources, receivers=receivers, wall_impedances=walls,
        frequencies_hz=frequencies, density_kg_m3=DENSITY, sound_speed_m_s=SOUND_SPEED,
    )

    assert set(batch) == {(s, r) for s in sources for r in receivers}
    for (source_name, receiver_name), energy in batch.items():
        assert energy == three_lane_report._solve_fem_energy(
            room=room, source=sources[source_name], receiver=receivers[receiver_name],
            wall_impedances=walls, frequencies_hz=frequencies,
            density_kg_m3=DENSITY, sound_speed_m_s=SOUND_SPEED,
        )
