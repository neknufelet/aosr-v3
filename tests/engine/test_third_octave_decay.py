"""#351：由已求解的逐頻衰減，獨立彙整嵌套的 1/3 八度帶。"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace

import pytest
from pydantic import ValidationError

from aosr.config import frequency_axis
from aosr.config.capabilities import load_capabilities
from aosr.config.paths import config_path
from aosr.geometry.shoebox import Point, Room, Wall
from aosr.physics import report_io, three_lane_report
from aosr.physics.report_source import SourceModelKind, SourceModelSpec
from aosr.physics.late_decay import LateDecayBand, LateDecayResult
from aosr.physics.report_output import output_from_report
from aosr.physics.third_octave_decay import (
    ThirdOctaveBand, ThirdOctaveDecay, ThirdOctaveDecayRow, build_third_octave_decay,
    subband_weighted_mean, third_octave_bands,
)


def test_subbands_share_exact_octave_edges_and_display_names_do_not_set_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bands = third_octave_bands()
    for center in frequency_axis.GEOMETRIC_REPORT_OCTAVE_CENTERS_HZ:
        rows = tuple(row for row in bands if row.octave_center_hz == center)
        assert rows[0].lower_hz == center / math.sqrt(2.0)
        assert rows[-1].upper_hz == center * math.sqrt(2.0)
        assert rows[0].upper_hz == rows[1].lower_hz
        assert rows[1].upper_hz == rows[2].lower_hz
    highest = bands[-1]
    assert highest.upper_hz == 8000.0 * math.sqrt(2.0)
    assert highest.upper_hz != pytest.approx(11224.6)
    low_500 = next(row for row in bands if row.nominal_center_hz == 400)
    assert low_500.center_hz == pytest.approx(500.0 * 2 ** (-1.0 / 3.0))
    assert low_500.center_hz != 400.0
    monkeypatch.setattr(
        frequency_axis, "GEOMETRIC_REPORT_THIRD_OCTAVE_NOMINAL_HZ",
        tuple(tuple(1 for _ in range(3)) for _ in bands[::3]),
    )
    renamed = third_octave_bands()
    assert tuple((row.lower_hz, row.upper_hz, row.center_hz) for row in renamed) == tuple(
        (row.lower_hz, row.upper_hz, row.center_hz) for row in bands
    )


def _inputs() -> report_io.ReportInput:
    document = {
        "room_m": {"Lx": 4.5, "Ly": 3.5, "Lz": 2.6},
        "source_model": {"kind": "omnidirectional"},
        "source_m": {"x": 1.0, "y": 2.2, "z": 1.2},
        "receiver_m": {"x": 3.2, "y": 1.9, "z": 1.2},
        "sound_speed_m_s": 343.0,
        "density_kg_m3": 1.2,
        "impedance_pa_s_per_m_by_wall": {
            "floor": 900.0, "ceiling": 1300.0, "x0": 1646.4,
            "xL": 2500.0, "y0": 4000.0, "yL": 6000.0,
        },
    }
    return report_io.load_input_document(
        document, load_capabilities(config_path("capabilities.toml"))
    )


def _fake_fem(
    *, room: Room, source: Point, receiver: Point,
    wall_impedances: Mapping[Wall, float], frequencies_hz: tuple[float, ...],
    density_kg_m3: float, sound_speed_m_s: float,
) -> tuple[float, ...]:
    del room, source, receiver, wall_impedances, density_kg_m3, sound_speed_m_s
    return tuple(frequency * 3.0 + 7.0 for frequency in frequencies_hz)


def _fake_decay(
    *, room: Room, wall_impedances: Mapping[Wall, float],
    rho_c_pa_s_per_m: float, sound_speed_m_s: float,
) -> three_lane_report._ReportLateDecay:
    del room, wall_impedances, rho_c_pa_s_per_m, sound_speed_m_s
    frequencies = tuple(
        frequency for frequency in frequency_axis.GEOMETRIC_LANE_FREQUENCIES_HZ
        if any(center / math.sqrt(2.0) <= frequency < center * math.sqrt(2.0)
               for center in frequency_axis.GEOMETRIC_REPORT_OCTAVE_CENTERS_HZ)
    )
    bands = tuple(
        LateDecayBand(
            frequency_hz=frequency, t20_s=frequency / 1000.0,
            collision_frequency_hz=1.0, slope_db_per_s=-1.0,
            soft_weight_sum=1.0, fell_back_to_perron=False, perron_t60_s=1.0,
            t30_s=frequency / 500.0, t30_slope_db_per_s=-1.0,
            t30_soft_weight_sum=1.0,
        ) for frequency in frequencies
    )
    return three_lane_report._ReportLateDecay(
        result=LateDecayResult(orders_used=1, bands=bands),
        unavailable_by_center_hz={},
    )


@pytest.fixture(scope="module")
def solved_report() -> tuple[three_lane_report.ThreeLaneReport, report_io.ReportInput]:
    inputs = _inputs()
    solved = report_io.solver_inputs(inputs)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(three_lane_report, "_solve_fem_energy", _fake_fem)
        patch.setattr(three_lane_report, "_solve_report_late_decay", _fake_decay)
        report = three_lane_report.solve_three_lane_report(
            source_model=SourceModelSpec(kind=SourceModelKind.OMNIDIRECTIONAL),
            room=solved.room, source=solved.source, receiver=solved.receiver,
            sound_speed_m_s=solved.sound_speed_m_s,
            density_kg_m3=solved.density_kg_m3,
            impedance_by_wall=solved.impedance_by_wall,
            scattering_by_wall=solved.scattering_by_wall,
            reflection_order_k=solved.reflection_order_k,
            low_frequency_axis=solved.low_frequency_axis,
        )
    return report, inputs


def _manual_mean(
    points: tuple[LateDecayBand, ...], bounds: tuple[float, float], *, t30: bool = False,
) -> float:
    """考卷自行以對數相鄰中點積分，不呼叫產品權重函式。"""
    logs = tuple(math.log2(point.frequency_hz) for point in points)
    edges = (max(math.log2(bounds[0]), logs[0] - (logs[1] - logs[0]) / 2.0),) + tuple(
        (left + right) / 2.0 for left, right in zip(logs, logs[1:])
    ) + (min(math.log2(bounds[1]), logs[-1] + (logs[-1] - logs[-2]) / 2.0),)
    weights = tuple(right - left for left, right in zip(edges, edges[1:]))
    values: list[float] = []
    for point in points:
        value = point.t30_s if t30 else point.t20_s
        assert value is not None
        values.append(value)
    return sum(weight * value for weight, value in zip(weights, values, strict=True)) / sum(weights)


def test_decay_uses_independent_log_width_average_and_scene_fingerprint(
    solved_report: tuple[three_lane_report.ThreeLaneReport, report_io.ReportInput],
) -> None:
    report, inputs = solved_report
    result = build_third_octave_decay(report, inputs)
    assert result.scene_fingerprint == report_io.scene_fingerprint(inputs)
    row = next(row for row in result.rows if row.band.nominal_center_hz == 400)
    points = tuple(point for point in report.late_decay.bands
                   if row.band.lower_hz <= point.frequency_hz < row.band.upper_hz)
    assert row.point_count == len(points)
    bounds = (row.band.lower_hz, row.band.upper_hz)
    assert row.t20_s == pytest.approx(_manual_mean(points, bounds))
    assert row.t30_s == pytest.approx(_manual_mean(points, bounds, t30=True))


@pytest.mark.parametrize("nominal,above_center", [(10000, True), (400, False)])
def test_both_halves_of_subband_affect_average_but_outside_does_not(
    solved_report: tuple[three_lane_report.ThreeLaneReport, report_io.ReportInput],
    nominal: int, above_center: bool,
) -> None:
    report, inputs = solved_report
    before = build_third_octave_decay(report, inputs)
    target = next(row for row in before.rows if row.band.nominal_center_hz == nominal)
    assert any(
        target.band.lower_hz <= point.frequency_hz < target.band.upper_hz
        and (point.frequency_hz > target.band.center_hz) == above_center
        for point in report.late_decay.bands
    )
    altered = tuple(
        replace(point, t20_s=point.t20_s * 2.0)
        if target.band.lower_hz <= point.frequency_hz < target.band.upper_hz
        and (point.frequency_hz > target.band.center_hz) == above_center
        else point for point in report.late_decay.bands
    )
    changed = build_third_octave_decay(
        replace(report, late_decay=replace(report.late_decay, bands=altered)), inputs
    )
    changed_row = next(row for row in changed.rows if row.band.nominal_center_hz == nominal)
    assert changed_row.t20_s is not None
    assert target.t20_s is not None
    assert changed_row.t20_s > target.t20_s
    assert changed_row.t30_s == target.t30_s
    assert all(
        old.t20_s == new.t20_s for old, new in zip(before.rows, changed.rows, strict=True)
        if old.band.nominal_center_hz != nominal
    )
    outside = next(point for point in report.late_decay.bands
                   if point.frequency_hz < target.band.lower_hz)
    outside_only = replace(report.late_decay, bands=tuple(
        replace(point, t20_s=point.t20_s * 2.0)
        if point.frequency_hz == outside.frequency_hz else point
        for point in report.late_decay.bands
    ))
    untouched = build_third_octave_decay(replace(report, late_decay=outside_only), inputs)
    untouched_row = next(row for row in untouched.rows if row.band.nominal_center_hz == nominal)
    assert untouched_row.t20_s == target.t20_s


def test_missing_octave_and_t30_only_reason_propagate(
    solved_report: tuple[three_lane_report.ThreeLaneReport, report_io.ReportInput],
) -> None:
    report, inputs = solved_report
    missing_center, t30_center = 500.0, 1000.0
    bands = tuple(
        replace(band, t20_s=None, t20_unavailable_reason="T20 fit failed",
                t30_s=None, t30_unavailable_reason="T30 fit failed")
        if band.center_frequency_hz == missing_center else
        replace(band, t30_s=None, t30_unavailable_reason="T30 only failed")
        if band.center_frequency_hz == t30_center else band
        for band in report.bands
    )
    decay = replace(report.late_decay, bands=tuple(
        point for point in report.late_decay.bands
        if not missing_center / math.sqrt(2.0) <= point.frequency_hz < missing_center * math.sqrt(2.0)
    ))
    result = build_third_octave_decay(replace(report, bands=bands, late_decay=decay), inputs)
    for row in result.rows:
        if row.band.octave_center_hz == missing_center:
            assert (row.t20_s, row.t20_unavailable_reason) == (None, "T20 fit failed")
            assert (row.t30_s, row.t30_unavailable_reason) == (None, "T30 fit failed")
        if row.band.octave_center_hz == t30_center:
            assert row.t20_s is not None
            assert (row.t30_s, row.t30_unavailable_reason) == (None, "T30 only failed")


def test_empty_subband_has_reason_and_models_reject_invalid_cells(
    solved_report: tuple[three_lane_report.ThreeLaneReport, report_io.ReportInput],
) -> None:
    report, inputs = solved_report
    band = next(row for row in third_octave_bands() if row.nominal_center_hz == 400)
    decay = replace(report.late_decay, bands=tuple(
        point for point in report.late_decay.bands
        if not band.lower_hz <= point.frequency_hz < band.upper_hz
    ))
    result = build_third_octave_decay(replace(report, late_decay=decay), inputs)
    row = next(row for row in result.rows if row.band == band)
    assert row.point_count == 0
    assert row.t20_s is None and row.t20_unavailable_reason
    assert row.t30_s is None and row.t30_unavailable_reason
    with pytest.raises(ValidationError):
        ThirdOctaveDecayRow.model_validate({**row.model_dump(), "surplus": 1})
    with pytest.raises(ValidationError):
        ThirdOctaveDecayRow.model_validate({**row.model_dump(), "t20_s": math.inf})
    with pytest.raises(ValidationError):
        ThirdOctaveDecayRow.model_validate({**row.model_dump(), "t20_s": 1.0})


def test_builder_preserves_octave_output_and_rejects_wrong_input(
    solved_report: tuple[three_lane_report.ThreeLaneReport, report_io.ReportInput],
) -> None:
    report, inputs = solved_report
    original_bands = report.bands
    original_decay_points = report.late_decay.bands
    before = output_from_report(report, inputs=inputs, with_points=True)
    build_third_octave_decay(report, inputs)
    after = output_from_report(report, inputs=inputs, with_points=True)
    assert report.bands == original_bands
    assert report.late_decay.bands == original_decay_points
    assert before.model_dump(mode="json") == after.model_dump(mode="json")
    with pytest.raises(ValueError, match="反射階數"):
        build_third_octave_decay(report, inputs.model_copy(update={
            "reflection_order_k": inputs.reflection_order_k + 1
        }))
    with pytest.raises(ValueError, match="低頻軸"):
        build_third_octave_decay(report, inputs.model_copy(update={
            "low_frequency_axis": frequency_axis.LowFrequencyAxis.VERIFICATION
        }))


def test_display_names_follow_iso_266_order_one_per_subband() -> None:
    """標稱帶名照 ISO 266 慣用值、由低到高一個子帶一個（只供顯示，帶界不從它來）。"""
    assert tuple(band.nominal_center_hz for band in third_octave_bands()) == (
        100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000,
        1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000,
    )


def test_point_exactly_on_a_subband_edge_belongs_to_the_upper_subband(
    solved_report: tuple[three_lane_report.ThreeLaneReport, report_io.ReportInput],
) -> None:
    """帶界是半開區間（下界含、上界不含），跟報表八度帶同一種切法：剛好落在界上的點歸上面那一帶。"""
    report, inputs = solved_report
    bands = third_octave_bands()
    lower_band = next(band for band in bands if band.nominal_center_hz == 800)
    upper_band = next(band for band in bands if band.nominal_center_hz == 1000)
    assert lower_band.upper_hz == upper_band.lower_hz
    edge = upper_band.lower_hz
    template = next(point for point in report.late_decay.bands
                    if lower_band.lower_hz <= point.frequency_hz < upper_band.upper_hz)
    probe = replace(template, frequency_hz=edge, t20_s=50.0, t30_s=50.0)
    decay = replace(report.late_decay, bands=tuple(
        sorted((*report.late_decay.bands, probe), key=lambda point: point.frequency_hz)
    ))
    before = build_third_octave_decay(report, inputs)
    after = build_third_octave_decay(replace(report, late_decay=decay), inputs)
    rows_before = {row.band.nominal_center_hz: row for row in before.rows}
    rows_after = {row.band.nominal_center_hz: row for row in after.rows}
    assert rows_after[800].point_count == rows_before[800].point_count
    assert rows_after[800].t20_s == rows_before[800].t20_s
    assert rows_after[1000].point_count == rows_before[1000].point_count + 1
    assert rows_after[1000].unplanned_hz == (edge,)
    assert rows_after[1000].t20_s is None
    assert rows_after[1000].t20_unavailable_cause == "subband_sampling"


def test_band_model_rejects_edges_out_of_order() -> None:
    band = third_octave_bands()[0]
    with pytest.raises(ValidationError, match="依序遞增"):
        type(band).model_validate({**band.model_dump(), "center_hz": band.upper_hz * 2.0})


def test_every_subband_is_one_third_octave_centred_in_log_frequency() -> None:
    """三等分：每個子帶寬 1/3 八度、中心在對數正中。期望帶界用考卷自己的算式，不拿結果的。"""
    # 第 k 個子帶（k＝0、1、2）：下界 中心×2^((2k−3)/6)、上界 中心×2^((2k−1)/6)
    expected = tuple(
        (center * 2 ** ((2 * index - 3) / 6.0), center * 2 ** ((2 * index - 1) / 6.0))
        for center in frequency_axis.GEOMETRIC_REPORT_OCTAVE_CENTERS_HZ
        for index in range(3)
    )
    for band, (lower, upper) in zip(third_octave_bands(), expected, strict=True):
        assert band.lower_hz == pytest.approx(lower, rel=1e-12)
        assert band.upper_hz == pytest.approx(upper, rel=1e-12)
        assert math.log2(band.upper_hz / band.lower_hz) == pytest.approx(1.0 / 3.0, rel=1e-12)
        assert band.center_hz == pytest.approx(math.sqrt(band.lower_hz * band.upper_hz), rel=1e-12)


def test_top_subband_keeps_the_last_fine_axis_point_above_8000_hz(
    solved_report: tuple[three_lane_report.ThreeLaneReport, report_io.ReportInput],
) -> None:
    """不能截掉 8000 Hz 以上：細軸最後一點（約 11.17 kHz）要算進 10 kHz 子帶，只改它、那一帶就要變。"""
    report, inputs = solved_report
    last = frequency_axis.GEOMETRIC_LANE_FREQUENCIES_HZ[-1]
    assert any(point.frequency_hz == last for point in report.late_decay.bands)
    before = next(row for row in build_third_octave_decay(report, inputs).rows
                  if row.band.nominal_center_hz == 10000)
    decay = replace(report.late_decay, bands=tuple(
        replace(point, t20_s=point.t20_s * 3.0) if point.frequency_hz == last else point
        for point in report.late_decay.bands
    ))
    after = next(row for row in build_third_octave_decay(replace(report, late_decay=decay), inputs).rows
                 if row.band.nominal_center_hz == 10000)
    assert before.t20_s is not None and after.t20_s is not None
    assert after.t20_s > before.t20_s


def test_rows_follow_band_table_and_model_rejects_reordered_rows(
    solved_report: tuple[three_lane_report.ThreeLaneReport, report_io.ReportInput],
) -> None:
    report, inputs = solved_report
    result = build_third_octave_decay(report, inputs)
    flattened = tuple(name for names in frequency_axis.GEOMETRIC_REPORT_THIRD_OCTAVE_NOMINAL_HZ for name in names)
    assert tuple(row.band.nominal_center_hz for row in result.rows) == flattened
    with pytest.raises(ValidationError, match="嚴格遞增"):
        ThirdOctaveDecay.model_validate({**result.model_dump(), "rows": tuple(reversed(result.model_dump()["rows"]))})


def test_subband_mean_matches_an_independent_sampling_of_the_same_cells() -> None:
    """另一條路：在對數頻率上密集撒點，每個取樣點歸給對數距離最近、而且在它格子內的逐頻點。"""
    band = next(row for row in third_octave_bands() if row.nominal_center_hz == 1000)
    frequencies = (band.lower_hz * 1.01, band.lower_hz * 1.05, band.center_hz, band.upper_hz * 0.97)
    values = (1.0, 3.0, 2.0, 7.0)
    logs = tuple(math.log2(frequency) for frequency in frequencies)
    lo = max(math.log2(band.lower_hz), logs[0] - (logs[1] - logs[0]) / 2.0)
    hi = min(math.log2(band.upper_hz), logs[-1] + (logs[-1] - logs[-2]) / 2.0)
    samples = 200_000
    total = 0.0
    for step in range(samples):
        position = lo + (hi - lo) * (step + 0.5) / samples
        nearest = min(range(len(logs)), key=lambda index: abs(logs[index] - position))
        total += values[nearest]
    assert subband_weighted_mean(frequencies, values, band) == pytest.approx(total / samples, rel=1e-4)


def test_error_paths_and_model_guards(
    solved_report: tuple[three_lane_report.ThreeLaneReport, report_io.ReportInput],
) -> None:
    report, inputs = solved_report
    with pytest.raises(ValueError, match="不是 ThreeLaneReport"):
        build_third_octave_decay(object(), inputs)  # type: ignore[arg-type]  # expires=2027-03-24 reason=故意傳錯型別驗拒收
    with pytest.raises(ValueError, match="報表缺少"):
        build_third_octave_decay(replace(report, bands=report.bands[1:]), inputs)
    missing_t30 = replace(report.late_decay, bands=tuple(
        replace(point, t30_s=None) if index == 0 else point
        for index, point in enumerate(report.late_decay.bands)
    ))
    with pytest.raises(ValueError, match="逐頻晚期衰減缺值"):
        build_third_octave_decay(replace(report, late_decay=missing_t30), inputs)
    row = build_third_octave_decay(report, inputs).rows[0]
    assert row.t30_s is not None
    with pytest.raises(ValidationError, match="有值則不能有原因"):
        ThirdOctaveDecayRow.model_validate({**row.model_dump(), "t30_unavailable_reason": "x"})
    with pytest.raises(ValidationError):
        ThirdOctaveDecayRow.model_validate({**row.model_dump(), "t20_s": 0.0})
    with pytest.raises(ValidationError):
        ThirdOctaveDecay.model_validate({"scene_fingerprint": "not-a-fingerprint", "rows": ()})
    with pytest.raises(ValidationError, match="依序遞增"):
        ThirdOctaveBand.model_validate({**row.band.model_dump(), "center_hz": row.band.lower_hz})
    with pytest.raises(ValidationError):
        row.t20_s = 1.0



def test_subband_mean_counts_an_exact_edge_point_only_in_the_upper_subband() -> None:
    """牆對會拿整條軸直接呼叫這一支：剛好落在子帶界上的點只算上面那一帶，不能兩帶各算一次。"""
    bands = third_octave_bands()
    lower_band = next(band for band in bands if band.nominal_center_hz == 800)
    upper_band = next(band for band in bands if band.nominal_center_hz == 1000)
    edge = upper_band.lower_hz
    frequencies = (lower_band.center_hz, edge, upper_band.center_hz)
    values = (1.0, 100.0, 1.0)
    assert subband_weighted_mean(frequencies, values, lower_band) == 1.0
    assert subband_weighted_mean(frequencies, values, upper_band) > 1.0


def test_three_subbands_split_the_parent_octave_points_without_gaps_or_overlap(
    solved_report: tuple[three_lane_report.ThreeLaneReport, report_io.ReportInput],
) -> None:
    """三個子帶合起來的逐頻點集合＝父八度帶的集合：不漏點、不重複歸帶、沒有標稱帶名造成的縫。"""
    report, _inputs = solved_report
    frequencies = tuple(point.frequency_hz for point in report.late_decay.bands)
    bands = third_octave_bands()
    for center in frequency_axis.GEOMETRIC_REPORT_OCTAVE_CENTERS_HZ:
        parent = {f for f in frequencies if center / math.sqrt(2.0) <= f < center * math.sqrt(2.0)}
        children = [
            {f for f in frequencies if band.lower_hz <= f < band.upper_hz}
            for band in bands if band.octave_center_hz == center
        ]
        assert set().union(*children) == parent
        assert sum(len(child) for child in children) == len(parent)


def test_building_third_octave_rows_never_calls_a_physics_solver(
    solved_report: tuple[three_lane_report.ThreeLaneReport, report_io.ReportInput],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """不重跑物理：把求解入口全換成一叫就炸的，照樣建得出來，而且跟正常建出來的逐格相同。"""
    report, inputs = solved_report
    expected = build_third_octave_decay(report, inputs)

    def explode(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("1/3 八度彙整不准重跑物理求解")

    for name in ("solve_three_lane_report", "_solve_report_late_decay", "_solve_fem_energy"):
        monkeypatch.setattr(three_lane_report, name, explode)
    assert build_third_octave_decay(report, inputs) == expected
