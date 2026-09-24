"""#351 找碴補卷：用手造逐頻能量和幾何獨立核對評估量。"""
from __future__ import annotations

import math
import re
from dataclasses import replace
from pathlib import Path

import pytest

from aosr.config.paths import config_path
from aosr.scoring.contract import CategoryEvaluation, EvaluationState, Flag, MetricState, ReasonCode
from aosr.scoring.direction_zones import DirectionZone
from aosr.scoring.reflections import ReflectionInput
from aosr.scoring.reflections_contract import (
    ReflectionChannel, ReflectionsAndEchoPayload, ReflectionSource,
)
from test_reflections import (  # type: ignore[import-not-found]  # expires=2026-10-24 reason=pytest-test-module-path
    _AXIS, _ROOM, _evaluate, _pair, _record, _third_decay, _vary_pair,
)


_IRREGULAR = (300.0, 500.0, 750.0, 800.0, 1300.0, 2000.0, 4000.0, 8000.0)


def _channel(result: CategoryEvaluation, role: str = "left",
             receiver: str = "main") -> ReflectionChannel:
    payload = result.payload
    assert isinstance(payload, ReflectionsAndEchoPayload)
    return next(item for item in payload.channels
                if item.role == role and item.receiver_id == receiver)


def _energy_rows(item: ReflectionInput, selected: dict[int, dict[float, float]]) -> ReflectionInput:
    table = item.report.path_table
    assert table is not None
    axis = table.frequencies_hz
    rows = tuple(row.model_copy(update={"relative_direct_energy": tuple(
        selected.get(index, {}).get(frequency, 0.0) for frequency in axis
    )}) if row.order > 0 else row for index, row in enumerate(table.rows))
    return replace(item, report=item.report.model_copy(update={
        "path_table": table.model_copy(update={"rows": rows})
    }))


def _manual_octave_mean(axis: tuple[float, ...], energy: tuple[float, ...],
                        bounds: tuple[float, float]) -> float:
    selected = [(math.log2(frequency), value) for frequency, value in zip(axis, energy)
                if bounds[0] <= frequency <= bounds[1]]
    positions = [position for position, _ in selected]
    mids = [(left + right) / 2.0 for left, right in zip(positions, positions[1:])]
    edges = [max(positions[0] - (positions[1] - positions[0]) / 2.0,
                 math.log2(bounds[0])), *mids,
             min(positions[-1] + (positions[-1] - positions[-2]) / 2.0,
                 math.log2(bounds[1]))]
    widths = [right - left for left, right in zip(edges, edges[1:])]
    return sum(width * value for width, (_, value) in zip(widths, selected)) / sum(widths)


def test_strongest_level_uses_chosen_frequency_energy_and_ten_log() -> None:
    left, right = _pair()
    left = _energy_rows(left, {8: {250.0: 0.05, 300.0: 0.1,
                                   800.0: 0.4, 1000.0: 0.01},
                               9: {250.0: 0.05, 300.0: 0.4,
                                   800.0: 0.1, 1000.0: 0.04}})
    result = _evaluate((left, right))
    front = next(zone for zone in _channel(result).zones if zone.zone is DirectionZone.FRONT)
    for frequency, winner, energy in ((300.0, 9, 0.4), (800.0, 8, 0.4),
                                      (1000.0, 9, 0.04)):
        point = next(point for point in front.points if point.frequency_hz == frequency)
        assert point.strongest_path_index is not None
        assert _channel(result).reflections[point.strongest_path_index].source_index == winner
        assert point.strongest_level_db == pytest.approx(10.0 * math.log10(energy))


def test_window_total_ignores_large_outside_reflection() -> None:
    left, right = _pair()
    table = left.report.path_table
    assert table is not None
    direct = next(row.delay_s for row in table.rows if row.order == 0)
    inside = next(index for index, row in enumerate(table.rows)
                  if row.order > 0 and row.delay_s - direct <= 0.015)
    outside = next(index for index, row in enumerate(table.rows)
                   if row.order > 0 and row.delay_s - direct > 0.015)
    left = _energy_rows(left, {inside: {1000.0: 0.25}, outside: {1000.0: 1000.0}})
    result = _evaluate((left, right))
    channel = _channel(result)
    point_index = list(frequency for frequency in _AXIS if 300.0 <= frequency <= 8000.0).index(1000.0)
    assert channel.total_window_energy_db[point_index].value == pytest.approx(
        10.0 * math.log10(0.25))


def test_wall_band_uses_octave_widths_on_irregular_axis() -> None:
    axis = (300.0, 500.0, 750.0, 800.0, 850.0, 1300.0, 2000.0, 4000.0, 8000.0)
    left = _record("left", 1.3, axis=axis)
    right = _record("right", 2.5, axis=axis)
    values = tuple({750.0: 0.9, 800.0: 0.5, 850.0: 0.1}.get(frequency, 0.5)
                   for frequency in axis)
    result = _evaluate(_vary_pair((left, right), values))
    assert isinstance(result.payload, ReflectionsAndEchoPayload)
    pair = result.payload.wall_pairs[0]
    band = next(item for item in pair.bands if item.nominal_center_hz == 800)
    mean = _manual_octave_mean((750.0, 800.0, 850.0), (0.9, 0.5, 0.1),
                               (band.lower_hz, band.upper_hz))
    assert band.round_trip_loss_db.value == pytest.approx(-10.0 * math.log10(mean))


def test_path_broadband_uses_octave_widths_and_only_declared_range() -> None:
    axis = (250.0, *_IRREGULAR, 9000.0)
    left = _record("left", 1.3, axis=axis)
    right = _record("right", 2.5, axis=axis)
    values = {frequency: 0.1 for frequency in _IRREGULAR}
    values.update({250.0: 1000.0, 750.0: 0.9, 800.0: 0.9,
                   1300.0: 0.1, 9000.0: 1000.0})
    left = _energy_rows(left, {1: values})
    result = _evaluate((left, right))
    path = next(path for path in _channel(result).reflections
                if path.source is ReflectionSource.PATH_TABLE and path.source_index == 1)
    mean = _manual_octave_mean(_IRREGULAR, tuple(values[frequency] for frequency in _IRREGULAR),
                               (300.0, 8000.0))
    assert path.broadband_level_db == pytest.approx(10.0 * math.log10(mean))


def test_floor_elevation_is_negative_in_room_and_listening_coordinates() -> None:
    result = _evaluate(_pair())
    floor = next(path for path in _channel(result).reflections
                 if path.wall_sequence == ("floor",))
    assert floor.room_elevation_deg < 0.0
    assert floor.listening_elevation_deg < 0.0


def test_equal_energy_prefers_earlier_delay_over_lower_path_index() -> None:
    left, right = _pair()
    table = left.report.path_table
    assert table is not None
    assert table.rows[8].delay_s > table.rows[9].delay_s
    left = _energy_rows(left, {8: {1000.0: 0.4}, 9: {1000.0: 0.4}})
    result = _evaluate((left, right))
    front = next(zone for zone in _channel(result).zones if zone.zone is DirectionZone.FRONT)
    point = next(point for point in front.points if point.frequency_hz == 1000.0)
    assert point.strongest_path_index is not None
    assert _channel(result).reflections[point.strongest_path_index].source_index == 9


def test_wall_delay_comes_from_geometry_and_t20_uses_its_own_band() -> None:
    changed = []
    for item in _pair():
        bands = tuple(band.model_copy(update={"t20_s": center / 1000.0})
                      for band in item.report.bands
                      for center in (band.center_frequency_hz,))
        report = item.report.model_copy(update={"bands": bands})
        changed.append(replace(item, report=report, third_octave_decay=_third_decay(report)))
    result = _evaluate(tuple(changed))
    assert isinstance(result.payload, ReflectionsAndEchoPayload)
    for pair in result.payload.wall_pairs:
        dimension = {("x0", "xL"): "Lx", ("y0", "yL"): "Ly",
                     ("floor", "ceiling"): "Lz"}[pair.walls]
        expected_delay = 2.0 * _ROOM[dimension] / 343.0
        assert pair.round_trip_delay_s.value == pytest.approx(expected_delay)
        for band in pair.bands:
            source = next(row for row in changed[0].third_octave_decay.rows
                          if row.band.nominal_center_hz == band.nominal_center_hz)
            assert band.room_t20_s.value == pytest.approx(source.t20_s)
            if band.round_trip_loss_db.value is not None:
                assert band.decay_duration_s.value == pytest.approx(
                    60.0 / band.round_trip_loss_db.value * expected_delay)


def test_raw_delays_match_screen_pairs_with_names_and_seconds() -> None:
    left, right = _pair()
    assert left.screen is not None
    result = _evaluate((left, right))
    expected = {f"round_trip_delay_{pair.faces[0]}_{pair.faces[1]}": pair.round_trip_delay_s
                for pair in left.screen.pairs}
    actual = {item.name: item for item in result.raw_quantities}
    for name, delay in expected.items():
        assert name in actual
        assert actual[name].unit == "s"
        assert actual[name].value == pytest.approx(delay)


@pytest.mark.parametrize("baseline_key", [
    "reflections_and_echo.window_upper_ms",
    "reflections_and_echo.frequency_range_hz",
    "reflections_and_echo.flutter_decay_db",
    "reflections_and_echo.flutter_alert_band_centers_hz",
    "direction_zones.vertical_min_abs_elevation_deg",
    "direction_zones.front_max_abs_azimuth_deg",
    "direction_zones.rear_min_abs_azimuth_deg",
])
def test_baseline_flag_depends_on_each_used_setting(tmp_path: Path, baseline_key: str) -> None:
    source = config_path("quality_targets.toml").read_text()
    used = (
        "reflections_and_echo.window_upper_ms", "reflections_and_echo.frequency_range_hz",
        "reflections_and_echo.flutter_decay_db",
        "reflections_and_echo.flutter_alert_band_centers_hz",
        "direction_zones.vertical_min_abs_elevation_deg",
        "direction_zones.front_max_abs_azimuth_deg",
        "direction_zones.rear_min_abs_azimuth_deg",
    )
    for key in used:
        pattern = rf'(?s)(key = "{re.escape(key)}"\n(?:(?!\[\[purpose\.).)*?status = ")baseline(")'
        source, count = re.subn(pattern, lambda match: (
            match.group(1) + "calibrated" + match.group(2) + "\n"
            'source_id = "test-fixture"\nsource_version = "1"\n'
            'locator = "controlled-test-entry"\nconditions = "test only"\n'
            'frequency_range_hz = [300.0, 8000.0]\n'
            'verification_digest = "sha256:' + "0" * 64 + '"'
        ), source, count=1)
        if count == 0 and key not in (
            "reflections_and_echo.window_upper_ms", "reflections_and_echo.frequency_range_hz"
        ):
            raise AssertionError(key)
    path = tmp_path / "quality_targets.toml"
    path.write_text(source)
    assert Flag.BASELINE_SETTINGS not in _evaluate(_pair(), path).flags
    pattern = (rf'(?s)(key = "{re.escape(baseline_key)}"\n'
               rf'(?:(?!\[\[purpose\.).)*?status = ")calibrated(")')
    changed, count = re.subn(pattern, r"\1baseline\2", source, count=1)
    assert count == 1
    path.write_text(changed)
    assert Flag.BASELINE_SETTINGS in _evaluate(_pair(), path).flags


def test_each_channel_keeps_its_computed_order_coverage_and_validation() -> None:
    left = _record("left", 2.2, room={"Lx": 4.5, "Ly": 3.5, "Lz": 2.6})
    right = _record("right", 2.6, room={"Lx": 4.5, "Ly": 3.5, "Lz": 2.6})
    result = _evaluate((left, right))
    for source in (left, right):
        assert source.window is not None
        channel = _channel(result, source.role)
        assert channel.computed_order_k == source.window.computed_order_k
        assert channel.coverage == source.window.coverage
        assert channel.validation == source.window.validation
    ordinary = _evaluate(_pair())
    for source in _pair():
        assert source.window is not None
        channel = _channel(ordinary, source.role)
        assert channel.computed_order_k == source.report.top.reflection_order_k
        assert channel.coverage == source.window.coverage
        assert channel.validation == source.window.validation
    main_left, main_right = _pair()
    around_left = _record("left", 1.3, "around", receiver_y=2.1)
    around_right = _record("right", 2.5, "around", receiver_y=2.1)
    assert around_left.window is not None
    around_left = replace(around_left, window=around_left.window.model_copy(update={
        "coverage": "not_provable", "validation": "unvalidated"
    }))
    local_result = _evaluate((main_left, main_right, around_left, around_right))
    local = _channel(local_result, receiver="around")
    assert local.state is MetricState.UNAVAILABLE
    assert local.computed_order_k == around_left.window.computed_order_k
    assert local.coverage == "not_provable"
    assert local.validation == "unvalidated"


def test_surrounding_point_across_speaker_line_keeps_primary_axis() -> None:
    left, right = _pair()
    around_left = _record("left", 1.3, "around", receiver_x=0.2)
    around_right = _record("right", 2.5, "around", receiver_x=0.2)
    result = _evaluate((left, right, around_left, around_right))
    channel = _channel(result, receiver="around")
    table = around_left.report.path_table
    assert table is not None
    path = next(path for path in channel.reflections if path.source_index == 1)
    direction = table.rows[1].direction_vector
    expected = math.degrees(math.atan2(-direction[1], -direction[0]))
    assert path.listening_azimuth_deg == pytest.approx(expected)
    assert result.state is EvaluationState.MEASURED


@pytest.mark.parametrize("key", [
    "reflections_and_echo.zone_threshold_db.front",
    "reflections_and_echo.zone_threshold_db.lateral",
    "reflections_and_echo.zone_threshold_db.rear",
    "reflections_and_echo.zone_threshold_db.vertical",
])
def test_zone_threshold_does_not_change_evaluator_settings_fingerprint(tmp_path: Path,
                                                                    key: str) -> None:
    """門檻走代價那一層，所以不改第一層評估器的設定指紋。"""
    source = config_path("quality_targets.toml").read_text()
    pattern = rf'(key = "{re.escape(key)}"\nvalue = )-10.0'
    changed, count = re.subn(pattern, r"\g<1>-9.0", source)
    assert count == 1
    path = tmp_path / "quality_targets.toml"
    path.write_text(changed)
    assert _evaluate(_pair(), path).settings_fingerprint == _evaluate(_pair()).settings_fingerprint


@pytest.mark.parametrize("edge", [300.0, 8000.0])
def test_single_in_range_sample_exactly_on_a_range_end_counts(edge: float) -> None:
    """範圍兩端都含：範圍內唯一一個細軸點剛好落在 300 或 8000 Hz 上，照樣算範圍內有點。"""
    axis = (250.0, edge, 9000.0)
    result = _evaluate((_record("left", 1.3, axis=axis), _record("right", 2.5, axis=axis)))
    assert ReasonCode.INSUFFICIENT_COVERAGE not in result.reason_codes
