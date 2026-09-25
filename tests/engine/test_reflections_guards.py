"""#351 評估器突變守門：輸入身分與獨立手算值。"""
from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import pytest

from aosr.config.frequency_axis import GEOMETRIC_LANE_FREQUENCIES_HZ
from aosr.config.quality_targets import SettingEntry, load_quality_targets
from aosr.physics.third_octave_decay import third_octave_bands
from aosr.scoring.contract import EvaluationState
from aosr.scoring.contract_base import Flag, MetricState, ReasonCode
from aosr.scoring.direction_zones import DirectionZone
from aosr.scoring.reflections import ReflectionInput
from aosr.scoring.reflections_contract import ReflectionsAndEchoPayload, ReflectionSource
from test_reflections import (  # type: ignore[import-not-found]  # expires=2026-10-24 reason=pytest-test-module-path
    _AXIS, _SMALL, _evaluate, _pair, _record, _set_entry,
)


def _payload(records: tuple[ReflectionInput, ...], path: Path | None = None
             ) -> ReflectionsAndEchoPayload:
    result = _evaluate(records, path)
    assert result.state is EvaluationState.MEASURED
    assert isinstance(result.payload, ReflectionsAndEchoPayload)
    return result.payload


def _mean_in_log_cells(axis: tuple[float, ...], values: tuple[float, ...],
                       bounds: tuple[float, float], *, upper_inclusive: bool = True) -> float:
    selected = [(math.log2(frequency), value) for frequency, value in zip(axis, values)
                if bounds[0] <= frequency and
                (frequency <= bounds[1] if upper_inclusive else frequency < bounds[1])]
    locations = [location for location, _ in selected]
    middle = [(left + right) / 2.0 for left, right in zip(locations, locations[1:])]
    edges = [max(math.log2(bounds[0]), locations[0] -
                 (locations[1] - locations[0]) / 2.0), *middle,
             min(math.log2(bounds[1]), locations[-1] +
                 (locations[-1] - locations[-2]) / 2.0)]
    widths = [right - left for left, right in zip(edges, edges[1:])]
    return sum(width * value for width, (_, value) in zip(widths, selected)) / sum(widths)


def _table_energy(item: ReflectionInput, chosen: dict[int, tuple[float, ...]]) -> ReflectionInput:
    table = item.report.path_table
    assert table is not None
    rows = tuple(row.model_copy(update={"relative_direct_energy": chosen.get(
        index, tuple(0.0 for _ in table.frequencies_hz))}) if row.order > 0 else row
        for index, row in enumerate(table.rows))
    return replace(item, report=item.report.model_copy(update={
        "path_table": table.model_copy(update={"rows": rows})
    }))


def test_registry_decay_changes_duration_from_independent_loss(tmp_path: Path) -> None:
    path = _set_entry(tmp_path, "reflections_and_echo.flutter_decay_db", "61.0")
    entry = load_quality_targets(path).purpose("dedicated_two_channel_listening_room").entry(
        "reflections_and_echo.flutter_decay_db")
    assert isinstance(entry, SettingEntry)
    assert isinstance(entry.value, (int, float))
    axis = GEOMETRIC_LANE_FREQUENCIES_HZ
    left, right = _record("left", 1.3, axis=axis), _record("right", 2.5, axis=axis)
    assert left.screen is not None
    screen = left.screen
    source = screen.pairs[0].model_copy(update={"round_trip_retained_energy": tuple(
        0.3 + frequency / 100000.0 for frequency in axis)})
    left = replace(left, screen=screen.model_copy(update={"pairs": (source, *screen.pairs[1:])}))
    right_screen = right.screen
    assert right_screen is not None
    right_pair = right_screen.pairs[0].model_copy(update={
        "round_trip_retained_energy": source.round_trip_retained_energy})
    right = replace(right, screen=right_screen.model_copy(update={
        "pairs": (right_pair, *right_screen.pairs[1:])}))
    payload = _payload((left, right), path)
    pair = next(item for item in payload.wall_pairs if item.walls == source.faces)
    band = next(item for item in pair.bands if item.nominal_center_hz == 1000)
    expected_retention = _mean_in_log_cells(axis, source.round_trip_retained_energy,
                                            (band.lower_hz, band.upper_hz),
                                            upper_inclusive=False)
    expected_loss = -10.0 * math.log10(expected_retention)
    expected_delay = 2.0 * source.distance_m / 343.0
    assert band.round_trip_loss_db.value == pytest.approx(expected_loss)
    assert band.decay_duration_s.value == pytest.approx(
        float(entry.value) / expected_loss * expected_delay)


def test_registry_frequency_bounds_change_payload_and_points(tmp_path: Path) -> None:
    path = _set_entry(tmp_path, "reflections_and_echo.frequency_range_hz", "[350.0, 8000.0]")
    payload = _payload(_pair(), path)
    expected = tuple(frequency for frequency in _AXIS if 350.0 <= frequency <= 8000.0)
    assert payload.frequency_range_hz == (350.0, 8000.0)
    for channel in payload.channels:
        for zone in channel.zones:
            assert tuple(point.frequency_hz for point in zone.points) == expected


def test_registry_front_limit_reclassifies_fifty_five_degree_path(tmp_path: Path) -> None:
    left, right = _pair()
    original = _payload((left, right))
    changed = _payload((left, right), _set_entry(
        tmp_path, "direction_zones.front_max_abs_azimuth_deg", "60.0"))
    def y0(payload: ReflectionsAndEchoPayload) -> DirectionZone:
        channel = next(item for item in payload.channels if item.role == "left")
        path = next(item for item in channel.reflections if item.wall_sequence == ("y0",))
        assert 50.0 < abs(path.listening_azimuth_deg) < 60.0
        return path.zone
    assert y0(original) is DirectionZone.LATERAL
    assert changed.zone_limits.front_max_abs_azimuth_deg == 60.0
    assert y0(changed) is DirectionZone.FRONT


def test_each_wall_pair_uses_its_own_frequency_varying_retention() -> None:
    axis = GEOMETRIC_LANE_FREQUENCIES_HZ
    changed = []
    for item in (_record("left", 1.3, axis=axis), _record("right", 2.5, axis=axis)):
        screen = item.screen
        assert screen is not None
        pairs = tuple(pair.model_copy(update={"round_trip_retained_energy": tuple(
            0.19 + index * 0.16 + 0.07 * (frequency / 20000.0)
            for frequency in axis)}) for index, pair in enumerate(screen.pairs))
        changed.append(replace(item, screen=screen.model_copy(update={"pairs": pairs})))
    payload = _payload(tuple(changed))
    screen = changed[0].screen
    assert screen is not None
    for source, risk in zip(screen.pairs, payload.wall_pairs, strict=True):
        assert risk.walls == source.faces
        for band in risk.bands:
            mean = _mean_in_log_cells(axis, source.round_trip_retained_energy,
                                      (band.lower_hz, band.upper_hz),
                                      upper_inclusive=False)
            assert band.round_trip_loss_db.value == pytest.approx(-10.0 * math.log10(mean))


def test_window_total_sums_known_paths_from_all_four_zones() -> None:
    left = _record("left", 2.2, room=_SMALL)
    right = _record("right", 2.6, room=_SMALL)
    table = left.report.path_table
    assert table is not None
    chosen = {1: (DirectionZone.FRONT, 0.11), 2: (DirectionZone.LATERAL, 0.17),
              3: (DirectionZone.VERTICAL, 0.23), 6: (DirectionZone.REAR, 0.29)}
    values = {index: tuple(level * (1.0 + frequency / 10000.0)
                           for frequency in table.frequencies_hz)
              for index, (_, level) in chosen.items()}
    left = _table_energy(left, values)
    window = left.window
    assert window is not None
    empty = tuple(0.0 for _ in table.frequencies_hz)
    extension = tuple(row.model_copy(update={"relative_direct_energy": empty})
                      for row in window.rows)
    left = replace(left, window=window.model_copy(update={"rows": extension}))
    result = _payload((left, right))
    channel = next(item for item in result.channels if item.role == "left")
    for index, (zone, _) in chosen.items():
        path = next(item for item in channel.reflections
                    if item.source is ReflectionSource.PATH_TABLE and item.source_index == index)
        assert path.within_window and path.zone is zone
    position = tuple(frequency for frequency in table.frequencies_hz
                     if 300.0 <= frequency <= 8000.0).index(1000.0)
    energy = sum(values[index][table.frequencies_hz.index(1000.0)] for index in chosen)
    assert channel.total_window_energy_db[position].value == pytest.approx(10.0 * math.log10(energy))


def test_measured_unvalidated_surrounding_channel_flags_category() -> None:
    left, right = _pair()
    around_left = _record("left", 1.3, "around", receiver_y=2.1)
    around_right = _record("right", 2.5, "around", receiver_y=2.1)
    assert around_left.window is not None
    assert around_left.window.coverage == "complete"
    around_left = replace(around_left, window=around_left.window.model_copy(update={
        "validation": "unvalidated"}))
    result = _evaluate((left, right, around_left, around_right))
    assert result.state is EvaluationState.MEASURED
    assert isinstance(result.payload, ReflectionsAndEchoPayload)
    primary = tuple(item for item in result.payload.channels if item.is_primary)
    assert all(item.validation == "validated" for item in primary)
    local = next(item for item in result.payload.channels
                 if item.role == "left" and item.receiver_id == "around")
    assert local.state is MetricState.MEASURED
    assert local.coverage == "complete" and local.validation == "unvalidated"
    assert Flag.UNVALIDATED in result.flags


def test_absent_primary_receiver_returns_reason_instead_of_crashing() -> None:
    records = (_record("left", 1.3, "around"), _record("right", 2.5, "around"))
    result = _evaluate(records)
    assert result.state is EvaluationState.UNAVAILABLE
    assert result.reason_codes == (ReasonCode.RECEIVER_ID_MISMATCH,)


def test_extension_broadband_uses_its_own_frequency_varying_energy() -> None:
    left = _record("left", 2.2, room=_SMALL)
    right = _record("right", 2.6, room=_SMALL)
    window = left.window
    table = left.report.path_table
    assert window is not None and window.rows and table is not None
    values = tuple(0.08 + 0.3 * frequency / 9000.0 for frequency in table.frequencies_hz)
    rows = (window.rows[0].model_copy(update={"relative_direct_energy": values}),
            *window.rows[1:])
    left = replace(left, window=window.model_copy(update={"rows": rows}))
    payload = _payload((left, right))
    channel = next(item for item in payload.channels if item.role == "left")
    path = next(item for item in channel.reflections
                if item.source is ReflectionSource.WINDOW_EXTENSION and item.source_index == 0)
    mean = _mean_in_log_cells(table.frequencies_hz, values, (300.0, 8000.0))
    assert path.broadband_level_db == pytest.approx(10.0 * math.log10(mean))


def test_broadband_end_cells_clip_at_declared_bounds() -> None:
    axis = (250.0, 310.0, 800.0, 2000.0, 7800.0, 9000.0)
    left, right = _record("left", 1.3, axis=axis), _record("right", 2.5, axis=axis)
    values = tuple(0.8 if frequency in (310.0, 7800.0) else
                   0.03 + frequency / 100000.0 for frequency in axis)
    left = _table_energy(left, {1: values})
    payload = _payload((left, right))
    channel = next(item for item in payload.channels if item.role == "left")
    path = next(item for item in channel.reflections
                if item.source is ReflectionSource.PATH_TABLE and item.source_index == 1)
    mean = _mean_in_log_cells(axis, values, (300.0, 8000.0))
    assert path.broadband_level_db == pytest.approx(10.0 * math.log10(mean))


def test_diagnostic_subbands_and_raw_delays_on_full_fine_axis() -> None:
    axis = GEOMETRIC_LANE_FREQUENCIES_HZ
    left, right = _record("left", 1.3, axis=axis), _record("right", 2.5, axis=axis)
    assert left.screen is not None
    screen = left.screen
    pairs = tuple(pair.model_copy(update={"round_trip_retained_energy": tuple(
        0.25 + index * 0.12 + frequency / 100000.0 for frequency in axis)})
        for index, pair in enumerate(screen.pairs))
    left = replace(left, screen=screen.model_copy(update={"pairs": pairs}))
    right_screen = right.screen
    assert right_screen is not None
    right = replace(right, screen=right_screen.model_copy(update={"pairs": tuple(
        pair.model_copy(update={"round_trip_retained_energy": values.round_trip_retained_energy})
        for pair, values in zip(right_screen.pairs, pairs, strict=True))}))
    result = _evaluate((left, right))
    assert isinstance(result.payload, ReflectionsAndEchoPayload)
    diagnostic = tuple(band for band in third_octave_bands()
                       if 100 <= band.nominal_center_hz <= 315)
    for source, risk in zip(pairs, result.payload.wall_pairs, strict=True):
        assert risk.walls == source.faces
        for expected_band in diagnostic:
            band = next(item for item in risk.bands
                        if item.nominal_center_hz == expected_band.nominal_center_hz)
            mean = _mean_in_log_cells(axis, source.round_trip_retained_energy,
                                      (expected_band.lower_hz, expected_band.upper_hz),
                                      upper_inclusive=False)
            assert band.round_trip_loss_db.state is MetricState.MEASURED
            assert band.round_trip_loss_db.value == pytest.approx(-10.0 * math.log10(mean))
        raw = next(item for item in result.raw_quantities
                   if item.name == f"round_trip_delay_{source.faces[0]}_{source.faces[1]}")
        assert raw.value == pytest.approx(2.0 * source.distance_m / 343.0)
        assert raw.unit == "s"


def test_registry_window_milliseconds_use_exact_division(tmp_path: Path) -> None:
    path = _set_entry(tmp_path, "reflections_and_echo.window_upper_ms", "16.1")
    window_s = 16.1 / 1000.0
    records = (_record("left", 1.3, window_s=window_s),
               _record("right", 2.5, window_s=window_s))
    payload = _payload(records, path)
    assert payload.window_upper_s == window_s
    assert payload.window_upper_ms == 16.1


@pytest.mark.parametrize("nominal", [band.nominal_center_hz for band in third_octave_bands()])
@pytest.mark.parametrize("receiver", ["main", "around"])
def test_every_third_octave_t20_mismatch_obeys_receiver_scope(nominal: int,
                                                                receiver: str) -> None:
    left, right = _pair()
    around_left = _record("left", 1.3, "around", receiver_y=2.1)
    around_right = _record("right", 2.5, "around", receiver_y=2.1)
    target = left if receiver == "main" else around_left
    decay = target.third_octave_decay
    assert decay is not None
    rows = tuple(row.model_copy(update={"t20_s": 3.7})
                 if row.band.nominal_center_hz == nominal else row for row in decay.rows)
    changed = replace(target, third_octave_decay=decay.model_copy(update={"rows": rows}))
    records = (changed, right, around_left, around_right) if receiver == "main" else (
        left, right, changed, around_right)
    result = _evaluate(records)
    if receiver == "main":
        assert result.state is EvaluationState.UNAVAILABLE
        assert result.reason_codes == (ReasonCode.REFLECTION_SCREEN_OR_WINDOW_MISMATCH,)
    else:
        assert result.state is EvaluationState.MEASURED
        assert isinstance(result.payload, ReflectionsAndEchoPayload)
        local = next(item for item in result.payload.channels
                     if item.role == "left" and item.receiver_id == "around")
        assert local.state is MetricState.UNAVAILABLE
        assert local.reason_codes == (ReasonCode.REFLECTION_SCREEN_OR_WINDOW_MISMATCH,)


def test_missing_t20_reason_difference_rejects_primary_pair() -> None:
    left, right = _pair()
    changed = []
    for item, reason in ((left, "未達下緣"), (right, "另一個母帶原因")):
        decay = item.third_octave_decay
        assert decay is not None
        rows = tuple(row.model_copy(update={"t20_s": None,
                    "t20_unavailable_reason": reason,
                    "t20_unavailable_cause": "octave_band"})
                     if row.band.nominal_center_hz == 1000 else row for row in decay.rows)
        changed.append(replace(item, third_octave_decay=decay.model_copy(update={"rows": rows})))
    result = _evaluate(tuple(changed))
    assert result.state is EvaluationState.UNAVAILABLE
    assert result.reason_codes == (ReasonCode.REFLECTION_SCREEN_OR_WINDOW_MISMATCH,)


def test_alert_centers_preserve_unsorted_registry_order(tmp_path: Path) -> None:
    path = _set_entry(tmp_path, "reflections_and_echo.flutter_alert_band_centers_hz",
                      "[1000.0, 400.0]")
    payload = _payload(_pair(), path)
    assert payload.flutter_alert_band_centers_hz == (1000, 400)


@pytest.mark.parametrize(("change", "reason"), [
    ("window_scene", ReasonCode.REFLECTION_SCREEN_OR_WINDOW_MISMATCH),
    ("window_source", ReasonCode.REFLECTION_SCREEN_OR_WINDOW_MISMATCH),
    ("window_receiver", ReasonCode.REFLECTION_SCREEN_OR_WINDOW_MISMATCH),
    ("screen_receiver", ReasonCode.REFLECTION_SCREEN_OR_WINDOW_MISMATCH),
    ("window_axis", ReasonCode.FREQUENCY_AXIS_MISMATCH),
    ("table_order", ReasonCode.REFLECTION_SCREEN_OR_WINDOW_MISMATCH),
    ("window_order", ReasonCode.REFLECTION_SCREEN_OR_WINDOW_MISMATCH),
    ("screen_order", ReasonCode.REFLECTION_SCREEN_OR_WINDOW_MISMATCH),
    ("missing_direct", ReasonCode.REFLECTION_SCREEN_OR_WINDOW_MISMATCH),
    ("table_row_short", ReasonCode.FREQUENCY_AXIS_MISMATCH),
    ("window_row_short", ReasonCode.FREQUENCY_AXIS_MISMATCH),
])
def test_each_physical_identity_guard_rejects_its_single_bad_input(
    change: str, reason: ReasonCode,
) -> None:
    if change in ("window_source", "window_row_short"):
        left = _record("left", 1.2, receiver_y=1.8, room=_SMALL)
        right = _record("right", 2.4, receiver_y=1.8, room=_SMALL)
    else:
        left, right = _pair()
    table, window, screen = left.report.path_table, left.window, left.screen
    assert table is not None and window is not None and screen is not None
    if change == "window_scene":
        left = replace(left, window=window.model_copy(update={"scene_fingerprint": "b" * 64}))
    elif change == "window_source":
        assert right.window is not None
        assert window.direct_delay_s == right.window.direct_delay_s
        assert window.scene_fingerprint == right.window.scene_fingerprint
        assert window.source_m != right.window.source_m
        left = replace(left, window=right.window)
    elif change == "window_receiver":
        other = _record("left", 1.3, receiver_y=2.1)
        assert other.window is not None
        left = replace(left, window=window.model_copy(update={
            "receiver_m": other.window.receiver_m}))
    elif change == "screen_receiver":
        other = _record("left", 1.3, receiver_y=2.1)
        assert other.screen is not None
        left = replace(left, screen=screen.model_copy(update={
            "receiver_m": other.screen.receiver_m}))
    elif change == "window_axis":
        shifted = tuple(501.0 if frequency == 500.0 else frequency
                        for frequency in window.frequencies_hz)
        assert len(shifted) == len(window.frequencies_hz)
        left = replace(left, window=window.model_copy(update={"frequencies_hz": shifted}))
    elif change == "table_order":
        left = replace(left, report=left.report.model_copy(update={
            "path_table": table.model_copy(update={"reflection_order_k": 2})}))
    elif change == "window_order":
        left = replace(left, window=window.model_copy(update={"report_order_k": 2}))
    elif change == "screen_order":
        left = replace(left, screen=screen.model_copy(update={"reflection_order_k": 2}))
    elif change == "missing_direct":
        left = replace(left, report=left.report.model_copy(update={
            "path_table": table.model_copy(update={
                "rows": tuple(row for row in table.rows if row.order != 0)})}))
    elif change == "table_row_short":
        rows = tuple(row.model_copy(update={
            "relative_direct_energy": row.relative_direct_energy[:-1]}) if index == 1
            else row for index, row in enumerate(table.rows))
        left = replace(left, report=left.report.model_copy(update={
            "path_table": table.model_copy(update={"rows": rows})}))
    else:
        assert window.rows
        rows = (window.rows[0].model_copy(update={
            "relative_direct_energy": window.rows[0].relative_direct_energy[:-1]}),
            *window.rows[1:])
        left = replace(left, window=window.model_copy(update={"rows": rows}))
    result = _evaluate((left, right))
    assert result.state is EvaluationState.UNAVAILABLE
    assert result.reason_codes == (reason,)


def test_one_ulp_direct_delay_change_is_rejected() -> None:
    left, right = _pair()
    window = left.window
    assert window is not None
    changed = window.model_copy(update={
        "direct_delay_s": math.nextafter(window.direct_delay_s, 1.0)})
    result = _evaluate((replace(left, window=changed), right))
    assert result.state is EvaluationState.UNAVAILABLE
    assert result.reason_codes == (ReasonCode.REFLECTION_SCREEN_OR_WINDOW_MISMATCH,)


def test_one_ulp_registry_window_change_is_rejected() -> None:
    left = _record("left", 1.3, window_s=math.nextafter(15.0 / 1000.0, 1.0))
    result = _evaluate((left, _record("right", 2.5)))
    assert result.state is EvaluationState.UNAVAILABLE
    assert result.reason_codes == (ReasonCode.REFLECTION_SCREEN_OR_WINDOW_MISMATCH,)
