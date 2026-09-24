"""顫動回音子帶與整房殘響共用精確帶界的驗收題。"""
from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import pytest

from aosr.config.paths import config_path
from aosr.config.quality_targets import SettingEntry, load_quality_targets
from aosr.physics.third_octave_decay import third_octave_bands
from aosr.scoring.contract import EvaluationState, MetricState, ReasonCode
from aosr.scoring.reflections import ReflectionInput
from aosr.scoring.reflections_contract import ReflectionsAndEchoPayload
from test_reflections import (  # type: ignore[import-not-found]  # expires=2026-10-24 reason=pytest-test-module-path
    _evaluate, _pair, _record, _set_entry, _small_reflection_registry, _vary_pair,
)


def _payload(records: tuple[ReflectionInput, ...]) -> ReflectionsAndEchoPayload:
    result = _evaluate(records)
    assert result.state is EvaluationState.MEASURED
    assert isinstance(result.payload, ReflectionsAndEchoPayload)
    return result.payload


def _manual_log_mean(axis: tuple[float, ...], values: tuple[float, ...],
                     lower: float, upper: float) -> float:
    selected = [(math.log2(frequency), value) for frequency, value in zip(axis, values)
                if lower <= frequency < upper]
    positions = [position for position, _ in selected]
    mids = [(left + right) / 2 for left, right in zip(positions, positions[1:])]
    edges = [max(math.log2(lower), positions[0] - (positions[1] - positions[0]) / 2),
             *mids,
             min(math.log2(upper), positions[-1] + (positions[-1] - positions[-2]) / 2)]
    widths = [right - left for left, right in zip(edges, edges[1:])]
    return sum(width * value for width, (_, value) in zip(widths, selected)) / sum(widths)


def test_wall_bands_share_exact_third_octave_edges_with_decay() -> None:
    payload = _payload(_pair())
    expected = third_octave_bands()
    for pair in payload.wall_pairs:
        for actual, band in zip(pair.bands, expected, strict=True):
            assert actual.nominal_center_hz == band.nominal_center_hz
            assert actual.frequency_hz == pytest.approx(band.center_hz)
            assert actual.lower_hz == pytest.approx(band.lower_hz)
            assert actual.upper_hz == pytest.approx(band.upper_hz)


def test_alert_bands_follow_registry_in_order_and_change_fingerprint(tmp_path: Path) -> None:
    ordinary = _evaluate(_pair())
    assert isinstance(ordinary.payload, ReflectionsAndEchoPayload)
    entry = load_quality_targets(config_path("quality_targets.toml")).purpose(
        "dedicated_two_channel_listening_room").entry(
            "reflections_and_echo.flutter_alert_band_centers_hz")
    assert isinstance(entry, SettingEntry)
    assert isinstance(entry.value, tuple)
    assert ordinary.payload.flutter_alert_band_centers_hz == entry.value
    changed = _evaluate(_pair(), _set_entry(
        tmp_path, "reflections_and_echo.flutter_alert_band_centers_hz", "[400.0, 500.0]"))
    assert changed.settings_fingerprint != ordinary.settings_fingerprint
    assert isinstance(changed.payload, ReflectionsAndEchoPayload)
    assert changed.payload.flutter_alert_band_centers_hz == (400.0, 500.0)


@pytest.mark.parametrize(("nominal", "frequency"), [
    (400, 380.0),
    (10000, 11170.0),
])
def test_both_halves_of_full_alert_bands_change_wall_loss(
    nominal: int, frequency: float,
) -> None:
    axis = (300.0, 350.0, 360.0, 380.0, 420.0, 500.0, 800.0,
            1000.0, 2000.0, 4000.0, 8000.0, 9000.0, 10000.0,
            11170.0, 11330.0)
    records = (_record("left", 1.3, axis=axis), _record("right", 2.5, axis=axis))
    baseline = _payload(_vary_pair(records, tuple(0.5 for _ in axis)))
    varied_values = tuple(0.9 if point == frequency else 0.5 for point in axis)
    varied = _payload(_vary_pair(records, varied_values))
    baseline_pair = next(pair for pair in baseline.wall_pairs if pair.walls == ("x0", "xL"))
    varied_pair = next(pair for pair in varied.wall_pairs if pair.walls == ("x0", "xL"))
    old = next(band for band in baseline_pair.bands if band.nominal_center_hz == nominal)
    new = next(band for band in varied_pair.bands if band.nominal_center_hz == nominal)
    assert old.round_trip_loss_db.value is not None
    assert new.round_trip_loss_db.value is not None
    assert old.round_trip_loss_db.value == pytest.approx(-10.0 * math.log10(0.5))
    expected = _manual_log_mean(axis, varied_values, new.lower_hz, new.upper_hz)
    assert new.round_trip_loss_db.value == pytest.approx(-10.0 * math.log10(expected))
    assert new.round_trip_loss_db.value < old.round_trip_loss_db.value
    for outside in (350.0, 11330.0):
        altered = tuple(0.9 if point == outside else 0.5 for point in axis)
        outside_pair = next(pair for pair in _payload(_vary_pair(records, altered)).wall_pairs
                            if pair.walls == ("x0", "xL"))
        outside_band = next(band for band in outside_pair.bands
                            if band.nominal_center_hz == nominal)
        assert outside_band.round_trip_loss_db.value == pytest.approx(
            old.round_trip_loss_db.value)


def test_room_t20_uses_matching_subband_row() -> None:
    changed = []
    for record in _pair():
        decay = record.third_octave_decay
        assert decay is not None
        rows = tuple(row.model_copy(update={"t20_s": (index + 1) / 10.0})
                     for index, row in enumerate(decay.rows))
        changed.append(replace(record, third_octave_decay=decay.model_copy(
            update={"rows": rows})))
    payload = _payload(tuple(changed))
    assert changed[0].third_octave_decay is not None
    for pair in payload.wall_pairs:
        for risk, row in zip(pair.bands, changed[0].third_octave_decay.rows, strict=True):
            assert risk.room_t20_s.value == pytest.approx(row.t20_s)
            if risk.round_trip_loss_db.value is not None:
                assert pair.round_trip_delay_s.value is not None
                assert risk.decay_duration_s.value == pytest.approx(
                    60.0 / risk.round_trip_loss_db.value * pair.round_trip_delay_s.value)


@pytest.mark.parametrize(("change", "reason"), [
    ("missing", ReasonCode.REFLECTION_SCREEN_OR_WINDOW_MISSING),
    ("mismatch", ReasonCode.REFLECTION_SCREEN_OR_WINDOW_MISMATCH),
])
def test_primary_third_octave_failure_makes_category_unavailable(
    change: str, reason: ReasonCode,
) -> None:
    left, right = _pair()
    assert left.third_octave_decay is not None
    decay = None if change == "missing" else left.third_octave_decay.model_copy(
        update={"scene_fingerprint": "b" * 64})
    result = _evaluate((replace(left, third_octave_decay=decay), right))
    assert result.state is EvaluationState.UNAVAILABLE
    assert result.reason_codes == (reason,)


@pytest.mark.parametrize(("change", "reason"), [
    ("missing", ReasonCode.REFLECTION_SCREEN_OR_WINDOW_MISSING),
    ("mismatch", ReasonCode.REFLECTION_SCREEN_OR_WINDOW_MISMATCH),
])
def test_surrounding_third_octave_failure_stays_local(
    change: str, reason: ReasonCode,
) -> None:
    left, right = _pair()
    around_left = _record("left", 1.3, receiver="around")
    around_right = _record("right", 2.5, receiver="around")
    assert around_left.third_octave_decay is not None
    decay = None if change == "missing" else around_left.third_octave_decay.model_copy(
        update={"scene_fingerprint": "b" * 64})
    result = _evaluate((left, right, replace(around_left, third_octave_decay=decay),
                        around_right))
    assert result.state is EvaluationState.MEASURED
    assert isinstance(result.payload, ReflectionsAndEchoPayload)
    channel = next(channel for channel in result.payload.channels
                   if channel.role == "left" and channel.receiver_id == "around")
    assert channel.state is MetricState.UNAVAILABLE
    assert channel.reason_codes == (reason,)


@pytest.mark.parametrize("receiver", ["main", "around"])
def test_different_third_octave_t20_follows_wall_pair_mismatch_rule(receiver: str) -> None:
    left, right = _pair()
    other = _record("left", 1.3, receiver=receiver)
    assert other.third_octave_decay is not None
    rows = list(other.third_octave_decay.rows)
    target = next(index for index, row in enumerate(rows)
                  if row.band.nominal_center_hz == 400)
    rows[target] = rows[target].model_copy(update={"t20_s": 3.7})
    altered = replace(other, third_octave_decay=other.third_octave_decay.model_copy(
        update={"rows": tuple(rows)}))
    records = (altered, right) if receiver == "main" else (
        left, right, altered, _record("right", 2.5, receiver="around"))
    result = _evaluate(records)
    if receiver == "main":
        assert result.state is EvaluationState.UNAVAILABLE
        assert result.reason_codes == (ReasonCode.REFLECTION_SCREEN_OR_WINDOW_MISMATCH,)
    else:
        assert result.state is EvaluationState.MEASURED
        assert isinstance(result.payload, ReflectionsAndEchoPayload)
        channel = next(channel for channel in result.payload.channels
                       if channel.role == "left" and channel.receiver_id == "around")
        assert channel.reason_codes == (ReasonCode.REFLECTION_SCREEN_OR_WINDOW_MISMATCH,)


def test_octave_report_t20_does_not_replace_supplied_third_octave_decay() -> None:
    left, right = _pair()
    old = tuple(band.model_copy(update={"t20_s": 2.0})
                if band.center_frequency_hz == 1000.0 else band
                for band in right.report.bands)
    changed = replace(right, report=right.report.model_copy(update={"bands": old}))
    result = _evaluate((left, changed))
    assert result.state is EvaluationState.MEASURED


@pytest.mark.parametrize(("field", "value", "message"), [
    ("unit", '"s"', "flutter_alert_band_centers_hz 必須是 Hz"),
    ("value", "400.0", "flutter_alert_band_centers_hz 必須是清單"),
    ("value", "[400.5, 500.0]", "flutter_alert_band_centers_hz 必須是整數標稱帶名"),
])
def test_alert_setting_requires_hz_list(tmp_path: Path, field: str, value: str,
                                        message: str) -> None:
    path = _small_reflection_registry(
        tmp_path, "reflections_and_echo.flutter_alert_band_centers_hz", field, value)
    with pytest.raises(ValueError, match=message):
        _evaluate(_pair(), path)


def test_third_octave_rows_with_other_band_edges_are_rejected() -> None:
    """1/3 八度那一份的子帶帶界不是物理那一刀定的那一套（例如混進 IEC 標稱帶界），要當對不上。"""
    left, right = _pair()
    decay = left.third_octave_decay
    assert decay is not None
    rows = list(decay.rows)
    shifted = rows[-1].band.model_copy(update={"upper_hz": 10000.0 * 2 ** (1.0 / 6.0)})
    rows[-1] = rows[-1].model_copy(update={"band": shifted})
    tampered = replace(left, third_octave_decay=decay.model_copy(update={"rows": tuple(rows)}))
    result = _evaluate((tampered, right))
    assert result.reason_codes == (ReasonCode.REFLECTION_SCREEN_OR_WINDOW_MISMATCH,)

