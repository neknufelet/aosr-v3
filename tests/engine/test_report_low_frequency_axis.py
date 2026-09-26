"""#435：搜尋與驗證各是一份完整報表，帶內平均不受取樣密度支配。"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest

from aosr.config.capabilities import load_capabilities
from aosr.config.paths import config_path
from aosr.config.three_lane_crossover import REFLECTION_ORDER_K
from aosr.config.frequency_axis import (
    FEM_LANE_FREQUENCIES_HZ,
    GEOMETRIC_LANE_FREQUENCIES_HZ,
    GEOMETRIC_REPORT_OCTAVE_CENTERS_HZ,
    LowFrequencyAxis,
    VERIFICATION_FEM_FREQUENCIES_HZ,
    VERIFICATION_REPORT_FREQUENCIES_HZ,
    low_frequency_axis_frequencies,
    octave_cells_in_range,
)
from aosr.geometry.shoebox import Point, Room, Wall
from aosr.physics import report_io, report_output, three_lane_report
from aosr.physics.report_source import SourceModelKind, SourceModelSpec
from aosr.physics.crossover import CrossoverWeights
from aosr.physics.geometric_lane import GeometricEarlyResult, GeometricLaneResult
from aosr.physics.late_decay import LateDecayBand, LateDecayResult
from aosr.scoring.timbre import _octave_cells_in_range


def test_verification_axis_is_complete_and_search_axis_is_unchanged() -> None:
    assert VERIFICATION_FEM_FREQUENCIES_HZ == tuple(float(f) for f in range(20, 301))
    assert VERIFICATION_REPORT_FREQUENCIES_HZ == (
        VERIFICATION_FEM_FREQUENCIES_HZ
        + tuple(f for f in GEOMETRIC_LANE_FREQUENCIES_HZ if f > 300.0)
    )
    assert all(
        left < right
        for left, right in zip(
            VERIFICATION_REPORT_FREQUENCIES_HZ,
            VERIFICATION_REPORT_FREQUENCIES_HZ[1:],
        )
    )
    assert low_frequency_axis_frequencies(LowFrequencyAxis.SEARCH) == (
        FEM_LANE_FREQUENCIES_HZ,
        GEOMETRIC_LANE_FREQUENCIES_HZ,
    )


@pytest.mark.parametrize(
    ("frequencies", "bounds"),
    (
        ((50.0, 100.0, 200.0), (50.0, 200.0)),
        (tuple(float(f) for f in range(90, 111)), (90.0, 110.0)),
        ((50.0, 100.0, 200.0), (100.0, 200.0)),
        ((50.0, 100.0, 200.0), (99.0, 101.0)),
        ((100.0, 100.0, 200.0), (100.0, 200.0)),
        ((50.0, 100.0, 200.0), (300.0, 400.0)),
    ),
)
def test_octave_cells_move_preserves_timbre_results(
    frequencies: tuple[float, ...], bounds: tuple[float, float]
) -> None:
    expected = octave_cells_in_range(frequencies, bounds)
    assert tuple(_octave_cells_in_range(np.asarray(frequencies), bounds)) == expected
    assert all(weight == 0.0 for f, weight in zip(frequencies, expected) if not bounds[0] <= f <= bounds[1])
    if frequencies == (50.0, 100.0, 200.0) and bounds == (50.0, 200.0):
        assert expected == (0.5, 1.0, 0.5)
    if frequencies == (50.0, 100.0, 200.0) and bounds == (100.0, 200.0):
        assert expected == (0.0, 0.5, 0.5)
    if bounds == (99.0, 101.0):
        assert expected == (0.0, 1.0, 0.0)
    if frequencies == (100.0, 100.0, 200.0):
        assert expected == (1.0, 1.0, 1.0)


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
    return tuple(f * 3.0 + 7.0 for f in frequencies_hz)


def _fast_late_decay(
    *, room: Room, wall_impedances: Mapping[Wall, float],
    rho_c_pa_s_per_m: float, sound_speed_m_s: float,
) -> three_lane_report._ReportLateDecay:
    """只替換慢速衰減求解；依正式細軸產生逐頻變化的 T20/T30。"""
    del room, wall_impedances, rho_c_pa_s_per_m, sound_speed_m_s
    root_two = math.sqrt(2.0)
    frequencies = tuple(
        f for f in GEOMETRIC_LANE_FREQUENCIES_HZ
        if any(center / root_two <= f < center * root_two for center in GEOMETRIC_REPORT_OCTAVE_CENTERS_HZ)
    )
    bands = tuple(
        LateDecayBand(
            frequency_hz=f, t20_s=f / 1000.0, collision_frequency_hz=1.0,
            slope_db_per_s=-1.0, soft_weight_sum=1.0,
            fell_back_to_perron=False, perron_t60_s=1.0,
            t30_s=f / 500.0, t30_slope_db_per_s=-1.0, t30_soft_weight_sum=1.0,
        )
        for f in frequencies
    )
    return three_lane_report._ReportLateDecay(
        result=LateDecayResult(orders_used=1, bands=bands),
        unavailable_by_center_hz={},
    )


def _inputs(**overrides: object) -> report_io.ReportInput:
    values: dict[str, object] = {
        "room_m": {"Lx": 6.0, "Ly": 4.0, "Lz": 3.0},
        "source_model": {"kind": "omnidirectional"},
        "source_m": {"x": 1.2, "y": 1.3, "z": 1.1},
        "receiver_m": {"x": 4.7, "y": 2.8, "z": 1.4},
        "sound_speed_m_s": 343.0,
        "density_kg_m3": 1.2,
        "impedance_pa_s_per_m_by_wall": {
            wall.wall_name(): 1646.4 for wall in Wall.all()
        },
    }
    values.update(overrides)
    return report_io.load_input_document(
        values, load_capabilities(config_path("capabilities.toml"))
    )


def _solve_with_axis(
    solved: report_io.SolverInputs, axis: LowFrequencyAxis
) -> three_lane_report.ThreeLaneReport:
    return three_lane_report.solve_three_lane_report(
        source_model=SourceModelSpec(kind=SourceModelKind.OMNIDIRECTIONAL),
        room=solved.room,
        source=solved.source,
        receiver=solved.receiver,
        sound_speed_m_s=solved.sound_speed_m_s,
        density_kg_m3=solved.density_kg_m3,
        impedance_by_wall=solved.impedance_by_wall,
        low_frequency_axis=axis,
    )


def test_two_complete_reports_keep_high_points_and_decay_identical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(three_lane_report, "_solve_fem_energy", _fake_fem_energy)
    monkeypatch.setattr(three_lane_report, "_solve_report_late_decay", _fast_late_decay)
    inputs = _inputs()
    solved = report_io.solver_inputs(inputs)
    search = _solve_with_axis(solved, LowFrequencyAxis.SEARCH)
    verification = _solve_with_axis(solved, LowFrequencyAxis.VERIFICATION)
    assert search.low_frequency_axis is LowFrequencyAxis.SEARCH
    assert verification.low_frequency_axis is LowFrequencyAxis.VERIFICATION
    assert verification.fem_frequencies_hz == VERIFICATION_FEM_FREQUENCIES_HZ
    assert tuple(p.frequency_hz for p in verification.points) == VERIFICATION_REPORT_FREQUENCIES_HZ
    # 低頻每一點拿到的有限元素值要是它自己那個頻率的（替身給 3f＋7）；配錯位會把錯的能量送進評分（找碴席抓到）。
    low = tuple(p for p in verification.points if p.fem_energy is not None)
    assert low
    assert all(p.fem_energy == p.frequency_hz * 3.0 + 7.0 for p in low)
    assert tuple(p for p in search.points if p.frequency_hz > 300.0) == tuple(
        p for p in verification.points if p.frequency_hz > 300.0
    )
    assert search.late_decay == verification.late_decay
    assert tuple((b.t20_s, b.t30_s) for b in search.bands) == tuple(
        (b.t20_s, b.t30_s) for b in verification.bands
    )
    assert tuple(
        (b.direct_energy, b.reflected_energy, b.interference_energy, b.scattering)
        for b in search.bands
    ) == tuple(
        (b.direct_energy, b.reflected_energy, b.interference_energy, b.scattering)
        for b in verification.bands
    )


def test_band_geometric_contribution_weights_dense_early_and_fine_late() -> None:
    """密軸誤用細軸權重、晚期搬上密軸或先平均再相乘時必須紅。"""
    fine = GeometricLaneResult(
        source_model=SourceModelSpec(kind=SourceModelKind.OMNIDIRECTIONAL),
        frequencies_hz=(100.0, 125.0, 150.0),
        direct_energy=(100.0, 100.0, 100.0),
        reflected_energy=(100.0, 100.0, 100.0),
        interference_energy=(100.0, 100.0, 100.0),
        late_energy=(10.0, 20.0, 30.0),
        scattering=(0.4, 0.5, 0.6),
        geometric_energy=(1000.0, 1000.0, 1000.0),
        reflection_order_k=REFLECTION_ORDER_K,
    )
    fine_weights = CrossoverWeights(
        w_fem=(0.9, 0.8, 0.7),
        w_geo=(0.1, 0.2, 0.3),
        capped_by_upper_limit=False,
    )
    points = three_lane_report.stitch_energy_points(
        frequencies_hz=fine.frequencies_hz,
        fem_energy_by_frequency={100.0: 2.0, 125.0: 3.0, 150.0: 4.0},
        geometric_lane=fine,
        geometric_indices=(0, 1, 2),
        weights=fine_weights,
    )
    dense = GeometricEarlyResult(
        source_model=SourceModelSpec(kind=SourceModelKind.OMNIDIRECTIONAL),
        frequencies_hz=(100.0, 125.0, 150.0),
        direct_energy=(1.0, 4.0, 7.0),
        reflected_energy=(2.0, 5.0, 8.0),
        interference_energy=(0.5, -1.0, 2.0),
        scattering=(0.1, 0.2, 0.3),
        reflection_order_k=REFLECTION_ORDER_K,
    )
    dense_weights = CrossoverWeights(
        w_fem=(0.5, 0.4, 0.3),
        w_geo=(0.5, 0.6, 0.7),
        capped_by_upper_limit=False,
    )

    fem, geometric = three_lane_report._band_contributions(
        points,
        dense_early=dense,
        dense_indices=(0, 1, 2),
        dense_weights=dense_weights,
        center_hz=125.0,
    )

    left = math.log2(100.0) - (math.log2(125.0) - math.log2(100.0)) / 2
    middle_left = (math.log2(100.0) + math.log2(125.0)) / 2
    middle_right = (math.log2(125.0) + math.log2(150.0)) / 2
    right = math.log2(150.0) + (math.log2(150.0) - math.log2(125.0)) / 2
    widths = (middle_left - left, middle_right - middle_left, right - middle_right)
    def weighted(values: tuple[float, ...]) -> float:
        return sum(w * value for w, value in zip(widths, values)) / sum(widths)
    expected_fem = weighted((0.9 * 2.0, 0.8 * 3.0, 0.7 * 4.0))
    # 早期三欄與晚期那一欄都已經含散射留存：這裡只乘權重，不再乘 1−s 或 s。
    expected_dense_early = (0.5 * 3.5 + 0.6 * 8.0 + 0.7 * 17.0) / 3.0
    expected_fine_late = weighted((0.1 * 10.0, 0.2 * 20.0, 0.3 * 30.0))
    assert fem == expected_fem
    assert geometric == expected_dense_early + expected_fine_late

def test_band_mean_weights_octaves_instead_of_sample_count() -> None:
    center = 125.0
    bounds = (center / math.sqrt(2.0), center * math.sqrt(2.0))
    dense = GeometricEarlyResult(
        source_model=SourceModelSpec(kind=SourceModelKind.OMNIDIRECTIONAL),
        frequencies_hz=(center,), direct_energy=(0.0,), reflected_energy=(0.0,),
        interference_energy=(0.0,), scattering=(0.0,), reflection_order_k=3,
    )
    weights = CrossoverWeights(w_fem=(0.0,), w_geo=(1.0,), capped_by_upper_limit=False)

    def means(axis: tuple[float, ...]) -> tuple[float, float]:
        points = tuple(
            three_lane_report.ThreeLanePoint(
                frequency_hz=f, fem_energy=f / center, direct_energy=0.0,
                reflected_energy=0.0, interference_energy=0.0,
                late_energy=0.0, scattering=0.0, geometric_energy=0.0,
                w_fem=1.0, w_geo=0.0, total_energy=f / center,
            )
            for f in axis if bounds[0] <= f < bounds[1]
        )
        weighted, _ = three_lane_report._band_contributions(
            points, dense_early=dense, dense_indices=(0,), dense_weights=weights,
            center_hz=center,
        )
        raw = sum(p.fem_energy or 0.0 for p in points) / len(points)
        return weighted, raw

    search = means(GEOMETRIC_LANE_FREQUENCIES_HZ)
    verification = means(VERIFICATION_REPORT_FREQUENCIES_HZ)
    assert abs(search[0] - verification[0]) < 0.02
    assert abs(search[1] - verification[1]) > 0.02


def test_axis_identity_changes_fingerprint_and_is_visible_in_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    implicit = _inputs()
    explicit = _inputs(low_frequency_axis=LowFrequencyAxis.SEARCH)
    verification = _inputs(low_frequency_axis=LowFrequencyAxis.VERIFICATION)
    assert report_io.scene_fingerprint(implicit) == report_io.scene_fingerprint(explicit)
    assert report_io.scene_fingerprint(implicit) != report_io.scene_fingerprint(verification)
    monkeypatch.setattr(three_lane_report, "_solve_fem_energy", _fake_fem_energy)
    monkeypatch.setattr(three_lane_report, "_solve_report_late_decay", _fast_late_decay)
    solved = report_io.solver_inputs(verification)
    report = _solve_with_axis(solved, verification.low_frequency_axis)
    output = report_output.output_from_report(report, inputs=verification, with_points=False)
    assert output.top.low_frequency_axis is LowFrequencyAxis.VERIFICATION
    assert output.scene.scene_fingerprint == report_io.scene_fingerprint(verification)
    payload = output.model_dump(mode="json")
    del payload["top"]["low_frequency_axis"]
    with pytest.raises(ValueError, match="low_frequency_axis"):
        report_io.ReportOutput.model_validate(payload)


def test_output_refuses_a_report_computed_on_a_different_axis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """輸入寫驗證軸、報表卻是用搜尋軸算的：輸出要報錯，不能把搜尋報表標成驗證報表。
    拿掉輸出前的軸比對，這題會紅（主對話突變抓到的盲點）。"""
    monkeypatch.setattr(three_lane_report, "_solve_fem_energy", _fake_fem_energy)
    monkeypatch.setattr(three_lane_report, "_solve_report_late_decay", _fast_late_decay)
    verification = _inputs(low_frequency_axis=LowFrequencyAxis.VERIFICATION)
    search_report = _solve_with_axis(report_io.solver_inputs(verification), LowFrequencyAxis.SEARCH)

    with pytest.raises(ValueError, match="低頻軸"):
        report_output.output_from_report(search_report, inputs=verification, with_points=False)


def test_cli_computes_on_the_axis_the_input_file_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """輸入檔寫驗證軸，命令列就要用驗證軸算：輸出標驗證軸、300 Hz 以下逐點每 1 Hz。
    命令列不把輸入檔那一格傳給求解，這題會紅（主對話突變抓到的盲點）。"""
    from aosr.physics import three_lane_report_cli

    document = {
        "room_m": {"Lx": 6.0, "Ly": 4.0, "Lz": 3.0},
        "source_model": {"kind": "omnidirectional"},
        "source_m": {"x": 1.2, "y": 1.3, "z": 1.1},
        "receiver_m": {"x": 4.7, "y": 2.8, "z": 1.4},
        "sound_speed_m_s": 343.0,
        "density_kg_m3": 1.2,
        "impedance_pa_s_per_m_by_wall": {wall.wall_name(): 1646.4 for wall in Wall.all()},
        "low_frequency_axis": LowFrequencyAxis.VERIFICATION.value,
    }
    input_path = tmp_path / "room.json"
    input_path.write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.setattr(three_lane_report, "_solve_fem_energy", _fake_fem_energy)
    monkeypatch.setattr(three_lane_report, "_solve_report_late_decay", _fast_late_decay)

    exit_code = three_lane_report_cli.main(
        [
            str(input_path),
            "--points",
            "--format",
            "json",
            "--capabilities",
            str(config_path("capabilities.toml")),
        ]
    )
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["top"]["low_frequency_axis"] == LowFrequencyAxis.VERIFICATION.value
    low_points = tuple(
        point["frequency_hz"] for point in output["points"] if point["frequency_hz"] <= 300.0
    )
    assert low_points == VERIFICATION_FEM_FREQUENCIES_HZ


def test_solver_inputs_carry_the_axis_from_the_input_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """從轉接入口直接算（不經命令列）也要用輸入檔寫的軸；漏帶那一格這題會紅（找碴席抓到）。"""
    monkeypatch.setattr(three_lane_report, "_solve_fem_energy", _fake_fem_energy)
    monkeypatch.setattr(three_lane_report, "_solve_report_late_decay", _fast_late_decay)
    verification = _inputs(low_frequency_axis=LowFrequencyAxis.VERIFICATION)

    report = three_lane_report.solve_three_lane_report(
        **report_io.solver_inputs(verification)._asdict()
    )

    assert report.low_frequency_axis is LowFrequencyAxis.VERIFICATION
    assert report.fem_frequencies_hz == VERIFICATION_FEM_FREQUENCIES_HZ
