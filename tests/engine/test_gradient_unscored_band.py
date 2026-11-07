"""#1032：終點低於計分下界的加註，不進目標。"""
from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from aosr.config.frequency_axis import LowFrequencyAxis, low_frequency_axis_frequencies
from aosr.gradient import endpoint
from aosr.scoring.low_frequency_input import low_frequency_input
from aosr.scoring.receiver_set import primary_cross
from aosr.scoring.stage_one_low_frequency import evaluate_stage_one_low_frequency


def _notes(pressures: NDArray[np.complex128], lower: float = 30.0) -> endpoint.UnscoredBand:
    assert hasattr(endpoint, "below_scoring_notes"), "缺少不計分低頻加註"
    return endpoint.below_scoring_notes(
        pressures, speaker_ids=("left", "right"),
        receiver_set=primary_cross(primary_position_m=(3.0, 2.0, 1.2), offset_m=0.1),
        frequencies_hz=low_frequency_axis_frequencies(LowFrequencyAxis.VERIFICATION)[0],
        low_frequency_axis=LowFrequencyAxis.VERIFICATION, stage_one_range_hz=(20.0, 300.0),
        stage_three_lower_hz=lower)


def _pressures() -> NDArray[np.complex128]:
    frequencies = low_frequency_axis_frequencies(LowFrequencyAxis.VERIFICATION)[0]
    values = np.ones((len(frequencies), 2, 7), dtype=np.complex128)
    # 22 Hz 主位峰、25 Hz 座位平均能量最大；70 Hz 更強，不能先取全段最強再篩。
    values[frequencies.index(22.0), 0, 0] = 10.0
    values[frequencies.index(25.0), 0, 1:] = 8.0
    values[frequencies.index(70.0), 0] = 100.0
    values[frequencies.index(30.0), 0, 1:] = 1000.0  # 上界不含。
    values[frequencies.index(20.0), 1] = 2.0  # 下界含，但端點不是 peak。
    return values


def test_notes_filter_peaks_after_full_stage_one_fit_and_average_energy() -> None:
    pressures = _pressures()
    notes = _notes(pressures)
    assert notes.range_hz == (20.0, 30.0)
    left = notes.by_speaker["left"]
    assert left.strongest_peak is not None and left.strongest_peak.frequency_hz == 22.0
    assert left.seat_mean_max is not None and left.seat_mean_max.frequency_hz == 25.0
    assert left.seat_mean_max.db == 10.0 * math.log10((1.0 + 6.0 * 64.0) / 7.0)
    receivers = primary_cross(primary_position_m=(3.0, 2.0, 1.2), offset_m=0.1)
    frequencies = low_frequency_axis_frequencies(LowFrequencyAxis.VERIFICATION)[0]
    data = low_frequency_input(candidate_id="control", speaker_ids=("left", "right"), receiver_set=receivers,
                               axis=LowFrequencyAxis.VERIFICATION,
                               energies={(speaker, point.receiver_id): tuple(np.abs(pressures[:, s, r]) ** 2)
                                         for s, speaker in enumerate(("left", "right"))
                                         for r, point in enumerate(receivers.points)},
                               frequency_hex=tuple(float(value).hex() for value in frequencies))
    evaluation = evaluate_stage_one_low_frequency(data, scoring_range_hz=(20.0, 300.0))
    payload = evaluation.speakers[0].payload
    assert payload is not None and payload.strongest_peak_index is not None
    assert payload.features[payload.strongest_peak_index].center_frequency_hz == 70.0
    expected = next(feature for feature in payload.features if feature.center_frequency_hz == 22.0)
    assert left.strongest_peak.db == expected.depth_db
    right = notes.by_speaker["right"]
    assert right.strongest_peak is None
    assert right.seat_mean_max is not None and right.seat_mean_max.frequency_hz == 20.0
    assert right.seat_mean_max.db == 10.0 * math.log10(4.0)


def test_empty_unscored_band_has_no_peak_or_maximum() -> None:
    notes = _notes(_pressures(), lower=20.0)
    assert notes.range_hz == (20.0, 20.0)
    assert all(row.strongest_peak is None and row.seat_mean_max is None for row in notes.by_speaker.values())


def test_unscored_improvement_is_baseline_minus_endpoint_and_keeps_missing() -> None:
    baseline = _notes(_pressures())
    design = _notes(_pressures() / 2.0)
    assert hasattr(endpoint, "below_scoring_improvement"), "缺少不計分低頻改善量"
    gain = endpoint.below_scoring_improvement(baseline, design)
    for name in ("left", "right"):
        difference = gain[name].seat_mean_max_db
        assert difference is not None
        assert math.isclose(difference, 10.0 * math.log10(4.0))
    assert gain["right"].strongest_peak_db is None
    assert gain["left"].strongest_peak_db is not None
    assert abs(gain["left"].strongest_peak_db) < 1e-12  # 自己的擬線一起平移。


def test_peak_at_upper_boundary_is_excluded_and_largest_inside_is_selected() -> None:
    frequencies = low_frequency_axis_frequencies(LowFrequencyAxis.VERIFICATION)[0]
    pressure = np.ones_like(_pressures())
    pressure[frequencies.index(22.0), :, 0] = 10.0
    pressure[frequencies.index(26.0), :, 0] = 5.0
    pressure[frequencies.index(30.0), :, 0] = 100.0
    notes = _notes(pressure)
    for row in notes.by_speaker.values():
        assert row.strongest_peak is not None
        assert row.strongest_peak.frequency_hz == 22.0


def test_peak_improvement_has_nonzero_baseline_minus_endpoint_sign() -> None:
    frequencies = low_frequency_axis_frequencies(LowFrequencyAxis.VERIFICATION)[0]
    baseline = _pressures()
    design = baseline.copy()
    design[frequencies.index(22.0), 0, 0] /= 2.0
    base, end = _notes(baseline), _notes(design)
    gain = endpoint.below_scoring_improvement(base, end)["left"].strongest_peak_db
    base_peak, end_peak = base.by_speaker["left"].strongest_peak, end.by_speaker["left"].strongest_peak
    assert base_peak is not None and end_peak is not None and gain is not None
    assert gain == base_peak.db - end_peak.db > 0.0
    assert endpoint.below_scoring_improvement(end, base)["left"].strongest_peak_db == -gain
