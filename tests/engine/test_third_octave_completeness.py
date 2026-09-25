"""#493：晚期衰減子帶與頻率軸的結構性完整性。"""

from __future__ import annotations

from dataclasses import replace
from typing import cast

import pytest
from pydantic import ValidationError

from aosr.config.frequency_axis import (
    GEOMETRIC_LANE_FREQUENCIES_HZ, LATE_DECAY_FREQUENCIES_HZ,
    LowFrequencyAxis, low_frequency_axis_frequencies, planned_band_points,
)
from aosr.physics import report_io, three_lane_report
from aosr.physics.third_octave_decay import (
    ThirdOctaveDecayRow, build_third_octave_decay, third_octave_bands,
)

from tests.engine.test_third_octave_decay import solved_report as solved_report


def _row_after_removal(
    solved_report: tuple[three_lane_report.ThreeLaneReport, report_io.ReportInput],
    removed: set[float],
) -> ThirdOctaveDecayRow:
    report, inputs = solved_report
    late_decay = replace(report.late_decay, bands=tuple(
        point for point in report.late_decay.bands if point.frequency_hz not in removed))
    result = build_third_octave_decay(replace(report, late_decay=late_decay), inputs)
    return next(row for row in result.rows if row.band.nominal_center_hz == 1000)


def test_planned_points_keep_axis_order_and_half_open_bounds() -> None:
    band = next(b for b in third_octave_bands() if b.nominal_center_hz == 1000)
    axis = (band.upper_hz, band.center_hz, band.lower_hz)
    assert planned_band_points(axis, band.lower_hz, band.upper_hz) == (
        band.center_hz, band.lower_hz)
    assert LATE_DECAY_FREQUENCIES_HZ is GEOMETRIC_LANE_FREQUENCIES_HZ


def test_axis_identity_changes_only_low_wall_pair_plans() -> None:
    search = low_frequency_axis_frequencies(LowFrequencyAxis.SEARCH)[1]
    verification = low_frequency_axis_frequencies(LowFrequencyAxis.VERIFICATION)[1]
    bands = third_octave_bands()
    for band in bands:
        searched = planned_band_points(search, band.lower_hz, band.upper_hz)
        verified = planned_band_points(verification, band.lower_hz, band.upper_hz)
        assert bool(searched != verified) == (band.nominal_center_hz < 400)
        assert searched == planned_band_points(
            LATE_DECAY_FREQUENCIES_HZ, band.lower_hz, band.upper_hz)


def test_missing_middle_point_prevents_both_decay_averages(
    solved_report: tuple[three_lane_report.ThreeLaneReport, report_io.ReportInput],
) -> None:
    band = next(b for b in third_octave_bands() if b.nominal_center_hz == 1000)
    planned = planned_band_points(LATE_DECAY_FREQUENCIES_HZ, band.lower_hz, band.upper_hz)
    removed = planned[len(planned) // 2]
    row = _row_after_removal(solved_report, {removed})
    assert row.missing_planned_hz == (removed,)
    assert row.unplanned_hz == ()
    assert row.t20_s is None and row.t30_s is None
    assert row.t20_unavailable_cause == row.t30_unavailable_cause == "subband_sampling"


def test_only_upper_half_of_decay_band_is_incomplete(
    solved_report: tuple[three_lane_report.ThreeLaneReport, report_io.ReportInput],
) -> None:
    band = next(b for b in third_octave_bands() if b.nominal_center_hz == 1000)
    planned = planned_band_points(LATE_DECAY_FREQUENCIES_HZ, band.lower_hz, band.upper_hz)
    removed = {frequency for frequency in planned if frequency < band.center_hz}
    row = _row_after_removal(solved_report, removed)
    assert row.missing_planned_hz == tuple(sorted(removed))
    assert row.t20_s is None and row.t30_s is None


def test_deleting_unavailable_point_does_not_restore_decay_value(
    solved_report: tuple[three_lane_report.ThreeLaneReport, report_io.ReportInput],
) -> None:
    report, inputs = solved_report
    band = next(b for b in third_octave_bands() if b.nominal_center_hz == 1000)
    removed = planned_band_points(LATE_DECAY_FREQUENCIES_HZ,
                                  band.lower_hz, band.upper_hz)[0]
    bad = replace(report.late_decay, bands=tuple(
        replace(point, t20_s=cast(float, None)) if point.frequency_hz == removed else point
        for point in report.late_decay.bands))
    with pytest.raises(ValueError, match="逐頻晚期衰減缺值"):
        build_third_octave_decay(replace(report, late_decay=bad), inputs)
    row = _row_after_removal(solved_report, {removed})
    assert row.t20_s is None
    assert row.missing_planned_hz == (removed,)


def test_unplanned_edge_point_is_recorded_and_not_averaged(
    solved_report: tuple[three_lane_report.ThreeLaneReport, report_io.ReportInput],
) -> None:
    report, inputs = solved_report
    band = next(b for b in third_octave_bands() if b.nominal_center_hz == 1000)
    template = report.late_decay.bands[0]
    inserted = replace(template, frequency_hz=band.lower_hz)
    late_decay = replace(report.late_decay, bands=tuple(sorted(
        (*report.late_decay.bands, inserted), key=lambda point: point.frequency_hz)))
    row = next(row for row in build_third_octave_decay(
        replace(report, late_decay=late_decay), inputs).rows if row.band == band)
    assert row.unplanned_hz == (band.lower_hz,)
    assert row.missing_planned_hz == ()
    assert row.t20_s is None and row.t30_s is None


def test_row_validator_rejects_inconsistent_sampling_cause(
    solved_report: tuple[three_lane_report.ThreeLaneReport, report_io.ReportInput],
) -> None:
    row = _row_after_removal(solved_report, set())
    planned = planned_band_points(LATE_DECAY_FREQUENCIES_HZ,
                                  row.band.lower_hz, row.band.upper_hz)
    with pytest.raises(ValidationError):
        ThirdOctaveDecayRow.model_validate({**row.model_dump(),
                                            "missing_planned_hz": (planned[0],)})
    with pytest.raises(ValidationError):
        ThirdOctaveDecayRow.model_validate({**row.model_dump(),
                                            "t20_unavailable_cause": "octave_band"})
    with pytest.raises(ValidationError):
        ThirdOctaveDecayRow.model_validate({**row.model_dump(), "t20_s": None,
                                            "t20_unavailable_reason": "x",
                                            "t20_unavailable_cause": "subband_sampling"})
