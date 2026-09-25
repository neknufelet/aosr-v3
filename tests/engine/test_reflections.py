"""#351 反射評估器：用手造報表接真實路徑、篩查與時間窗。"""
from __future__ import annotations

import math
import re
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from aosr.config.capabilities import load_capabilities
from aosr.config.quality_targets import SettingEntry, load_quality_targets
from aosr.config.frequency_axis import LowFrequencyAxis
from aosr.config.paths import config_path
from aosr.physics import report_io
from aosr.physics.reflection_screen import build_reflection_screen
from aosr.physics.reflection_window import build_reflection_window
from aosr.physics.report_path_table import build_path_table
from aosr.physics.report_io import (
    BandRow, CapabilitySection, PathTableSection, PointRow, ReportOutput, SceneSection,
    TopFields,
)
from aosr.physics.room_paths import NUMERICALLY_GUARDED_ORDER_K
from aosr.physics.third_octave_decay import (
    ThirdOctaveDecay, ThirdOctaveDecayRow, third_octave_bands,
)
from aosr.scoring.channel_matching import ChannelComparison, ChannelDefinition, ChannelGroup
from aosr.scoring.contract import CategoryEvaluation, EvaluationState, Flag, MetricState, ReasonCode
from aosr.scoring.direction_zones import DirectionZone
import aosr.scoring.reflections as reflections
from aosr.scoring.reflections_contract import ReflectionsAndEchoPayload, ReflectionSource
from aosr.scoring.reflections import ReflectionInput, evaluate_reflections


_AXIS = (250.0, 300.0, 500.0, 800.0, 1000.0, 1250.0,
         2000.0, 4000.0, 8000.0, 9000.0)
_SCATTERING = tuple(0.2 for _ in _AXIS)
_ROOM = {"Lx": 6.0, "Ly": 4.0, "Lz": 3.0}
_SMALL = {"Lx": 4.5, "Ly": 3.5, "Lz": 2.6}
_WALLS = {"floor": 900.0, "ceiling": 1300.0, "x0": 1646.4,
          "xL": 2500.0, "y0": 4000.0, "yL": 6000.0}


def _band(center: float) -> BandRow:
    return BandRow(
        center_frequency_hz=center, fem_energy=1.0, fem_point_count=1,
        direct_energy=1.0, reflected_energy=0.0, interference_energy=0.0,
        late_energy=0.0, geometric_energy=1.0, fem_contribution=1.0,
        geometric_contribution=0.0, total_energy=1.0, w_fem=1.0, w_geo=0.0,
        f_s_hz=200.0, capped_by_upper_limit=False, t20_s=1.0,
        t20_unavailable_reason=None, t30_s=1.2, t30_unavailable_reason=None,
    )


def _third_decay(report: ReportOutput) -> ThirdOctaveDecay:
    parents = {band.center_frequency_hz: band for band in report.bands}
    parents.update({band.octave_center_hz: _band(band.octave_center_hz)
                    for band in third_octave_bands()
                    if band.octave_center_hz not in parents})
    return ThirdOctaveDecay(
        scene_fingerprint=report.scene.scene_fingerprint,
        rows=tuple(ThirdOctaveDecayRow(
            band=band, point_count=1,
            t20_s=parents[band.octave_center_hz].t20_s,
            t20_unavailable_reason=parents[band.octave_center_hz].t20_unavailable_reason,
            t30_s=parents[band.octave_center_hz].t30_s,
            t30_unavailable_reason=parents[band.octave_center_hz].t30_unavailable_reason,
        ) for band in third_octave_bands()),
    )


def _input(source_y: float, receiver_y: float = 1.9, receiver_x: float = 3.2,
           room: dict[str, float] = _ROOM, order: int = 3) -> report_io.ReportInput:
    return report_io.load_input_document({
        "room_m": room,
        "source_m": {"x": 1.0, "y": source_y, "z": 1.2},
        "receiver_m": {"x": receiver_x, "y": receiver_y, "z": 1.2},
        "sound_speed_m_s": 343.0, "density_kg_m3": 1.2,
        "impedance_pa_s_per_m_by_wall": _WALLS, "reflection_order_k": order,
    }, load_capabilities(config_path("capabilities.toml")))


def _record(role: str, source_y: float, receiver: str = "main", *,
            room: dict[str, float] = _ROOM, receiver_y: float = 1.9,
            receiver_x: float = 3.2,
            window_s: float = 15.0 / 1000.0, order: int = 3,
            axis: tuple[float, ...] = _AXIS) -> ReflectionInput:
    inputs = _input(source_y, receiver_y, receiver_x, room, order)
    solved = report_io.solver_inputs(inputs)
    scattering = tuple(0.2 for _ in axis)
    table = build_path_table(
        room=solved.room, source=solved.source, receiver=solved.receiver,
        sound_speed_m_s=solved.sound_speed_m_s,
        rho_c_pa_s_per_m=solved.density_kg_m3 * solved.sound_speed_m_s,
        impedance_by_wall=solved.impedance_by_wall,
        frequencies_hz=axis, scattering_coefficient=scattering,
        reflection_order_k=inputs.reflection_order_k,
    )
    report = ReportOutput(
        scene=SceneSection(
            scene_fingerprint=report_io.scene_fingerprint(inputs),
            source_m=inputs.source_m, receiver_m=inputs.receiver_m,
        ),
        capability=CapabilitySection(
            frequency_hz=(20.0, 20000.0), outputs=("t20_s",),
            status="experimental", evidence=(),
        ),
        top=TopFields(
            f_s_hz=200.0, crossover_lower_hz=100.0,
            crossover_upper_hz=300.0, capped_by_upper_limit=False,
            reflection_order_k=inputs.reflection_order_k,
            low_frequency_axis=LowFrequencyAxis.SEARCH,
            eyring_t60_by_band_s={"1000.0": 1.0}, room_volume_m3=72.0,
            schroeder_band_count=1,
        ),
        bands=(_band(500.0), _band(1000.0), _band(2000.0), _band(4000.0), _band(8000.0)),
        points=tuple(PointRow(
            frequency_hz=frequency, fem_energy=None,
            direct_energy=1.0, reflected_energy=0.0, interference_energy=0.0,
            late_energy=0.0, scattering=scatter, geometric_energy=1.0,
            w_fem=0.0, w_geo=1.0, total_energy=1.0,
        ) for frequency, scatter in zip(axis, scattering, strict=True)),
        path_table=PathTableSection.model_validate(asdict(table)),
    )
    window = build_reflection_window(
        inputs, frequencies_hz=axis, scattering_coefficient=scattering,
        window_s=window_s,
    )
    return ReflectionInput(
        role=role, receiver_id=receiver, report=report,
        screen=build_reflection_screen(inputs, axis), window=window,
        third_octave_decay=_third_decay(report),
        report_id=f"{role}-{receiver}", engine_commit="engine-commit",
        speaker_id=role,
    )


def _group() -> ChannelGroup:
    return ChannelGroup(
        channels=(ChannelDefinition(role="left", speaker_id="left"),
                  ChannelDefinition(role="right", speaker_id="right")),
        comparisons=(ChannelComparison(left_role="left", right_role="right"),),
        feature_match_tolerance_hz=0.0,
    )


def _evaluate(records: tuple[ReflectionInput, ...], path: Path | None = None) -> CategoryEvaluation:
    return evaluate_reflections(
        _group(), records, primary_receiver_id="main", candidate_id="candidate-a",
        purpose="dedicated_two_channel_listening_room",
        quality_targets_path=path or config_path("quality_targets.toml"),
    )


def _pair() -> tuple[ReflectionInput, ReflectionInput]:
    return _record("left", 1.3), _record("right", 2.5)


def _set_entry(tmp_path: Path, key: str, value: str) -> Path:
    text = config_path("quality_targets.toml").read_text()
    pattern = rf'(key = "{re.escape(key)}"\nvalue = )[^\n]+'
    changed, count = re.subn(pattern, lambda match: match.group(1) + value, text)
    if count != 1:
        raise ValueError(f"登記簿設定 {key} 未唯一命中")
    path = tmp_path / "quality_targets.toml"
    path.write_text(changed)
    return path


def _small_reflection_registry(tmp_path: Path, key: str, field: str, value: str) -> Path:
    source = config_path("quality_targets.toml").read_text()
    needed = (
        "reflections_and_echo.window_upper_ms", "reflections_and_echo.frequency_range_hz",
        "reflections_and_echo.flutter_decay_db",
        "reflections_and_echo.flutter_alert_band_centers_hz",
        "direction_zones.vertical_min_abs_elevation_deg",
        "direction_zones.front_max_abs_azimuth_deg",
        "direction_zones.rear_min_abs_azimuth_deg",
    )
    chunks = source.split("[[purpose.setting]]\n")
    blocks = []
    for name in needed:
        block = next(chunk for chunk in chunks[1:] if chunk.startswith(f'key = "{name}"\n'))
        block = block.split("\n[[purpose.", 1)[0].rstrip()
        if name == key:
            block, count = re.subn(rf"(?m)^{field} = [^\n]+$",
                                   f"{field} = {value}", block)
            if count != 1:
                raise ValueError(f"登記簿欄位 {key}.{field} 未唯一命中")
        blocks.append("[[purpose.setting]]\n" + block)
    header = ('schema_version = 1\n[[purpose]]\n'
              'name = "dedicated_two_channel_listening_room"\n'
              'target = []\nweight = []\nqualification = []\n')
    path = tmp_path / "quality_targets.toml"
    path.write_text(header + "\n".join(blocks) + "\n")
    return path


def _vary_pair(records: tuple[ReflectionInput, ReflectionInput], values: tuple[float, ...]
               ) -> tuple[ReflectionInput, ReflectionInput]:
    changed = []
    for item in records:
        assert item.screen is not None
        pair = item.screen.pairs[0]
        first = pair.model_copy(update={"round_trip_retained_energy": values})
        changed.append(replace(item, screen=item.screen.model_copy(update={
            "pairs": (first, *item.screen.pairs[1:])
        })))
    return changed[0], changed[1]


def test_reflections_measures_without_cost_and_validates_frozen_payload() -> None:
    result = _evaluate(_pair())
    assert result.state is EvaluationState.MEASURED
    assert result.category_cost is None
    assert result.payload is not None
    assert isinstance(result.payload, ReflectionsAndEchoPayload)
    assert result.payload.category == "reflections_and_echo"
    assert {Flag.NO_DIRECTIVITY, Flag.WINDOW_ONLY_DELAY_SCREEN,
            Flag.GEOMETRY_MATERIAL_CONSERVATIVE_SCREEN,
            Flag.BASELINE_SETTINGS} <= set(result.flags)
    assert {pair.walls for pair in result.payload.wall_pairs} == {
        ("x0", "xL"), ("y0", "yL"), ("floor", "ceiling")}
    assert all(math.isfinite(item.value) for item in result.raw_quantities)
    assert CategoryEvaluation.model_validate(result.model_dump(mode="json")) == result


def test_relative_delay_excludes_outside_rows_and_preserves_source_index() -> None:
    left, right = _pair()
    result = _evaluate((left, right))
    assert isinstance(result.payload, ReflectionsAndEchoPayload)
    channel = next(item for item in result.payload.channels if item.role == "left")
    table = left.report.path_table
    assert table is not None
    direct = next(row for row in table.rows if row.order == 0)
    assert any(not path.within_window for path in channel.reflections)
    for path in channel.reflections:
        row = table.rows[path.source_index]
        assert path.relative_direct_delay_s == row.delay_s - direct.delay_s
        assert path.within_window == (row.delay_s - direct.delay_s <= result.payload.window_upper_s)
    assert all(channel.reflections[point.strongest_path_index].within_window
               for zone in channel.zones for point in zone.points
               if point.strongest_path_index is not None)


def test_zero_energy_and_empty_zone_are_explicit_states() -> None:
    left, right = _pair()
    table = left.report.path_table
    assert table is not None
    rows = tuple(row.model_copy(update={
        "relative_direct_energy": tuple(0.0 for _ in _AXIS)
    }) if row.order > 0 else row for row in table.rows)
    left = replace(left, report=left.report.model_copy(update={
        "path_table": table.model_copy(update={"rows": rows})
    }))
    result = _evaluate((left, right))
    assert isinstance(result.payload, ReflectionsAndEchoPayload)
    channel = next(item for item in result.payload.channels if item.role == "left")
    assert any(point.strongest_state is MetricState.NOT_COMPUTABLE
               and point.strongest_reason_codes == (ReasonCode.ZERO_REFLECTION_ENERGY,)
               for zone in channel.zones for point in zone.points)
    assert any(point.strongest_path_index is None
               and point.strongest_reason_codes == (ReasonCode.NO_REFLECTION_IN_ZONE_POINT,)
               for zone in channel.zones for point in zone.points)
    assert all(cell.value is None and cell.reason_codes == (ReasonCode.ZERO_REFLECTION_ENERGY,)
               for cell in channel.total_window_energy_db)


def test_large_finite_reflections_keep_finite_zone_and_window_levels() -> None:
    left, right = _pair()
    table = left.report.path_table
    assert table is not None
    rows = tuple(row.model_copy(update={
        "relative_direct_energy": tuple(1e308 for _ in _AXIS)
    }) if index in (1, 8) else row.model_copy(update={
        "relative_direct_energy": tuple(0.0 for _ in _AXIS)
    }) if row.order > 0 else row for index, row in enumerate(table.rows))
    left = replace(left, report=left.report.model_copy(update={
        "path_table": table.model_copy(update={"rows": rows})
    }))
    result = _evaluate((left, right))
    assert isinstance(result.payload, ReflectionsAndEchoPayload)
    channel = next(item for item in result.payload.channels if item.role == "left")
    front = next(zone for zone in channel.zones if zone.zone is DirectionZone.FRONT)
    point = next(point for point in front.points if point.frequency_hz == 1000.0)
    expected = 10.0 * (math.log10(1e308) + math.log10(2.0))
    assert point.total_energy_db.value == pytest.approx(expected)
    assert next(cell for cell, zone_point in zip(channel.total_window_energy_db, front.points)
                if zone_point.frequency_hz == 1000.0).value == pytest.approx(expected)


@pytest.mark.parametrize(("change", "expected"), [
    ("path_missing", ReasonCode.PATH_TABLE_MISSING),
    ("screen_missing", ReasonCode.REFLECTION_SCREEN_OR_WINDOW_MISSING),
    ("window_missing", ReasonCode.REFLECTION_SCREEN_OR_WINDOW_MISSING),
    ("screen_scene", ReasonCode.REFLECTION_SCREEN_OR_WINDOW_MISMATCH),
    ("screen_source", ReasonCode.REFLECTION_SCREEN_OR_WINDOW_MISMATCH),
    ("report_order", ReasonCode.REFLECTION_SCREEN_OR_WINDOW_MISMATCH),
    ("axis", ReasonCode.FREQUENCY_AXIS_MISMATCH),
    ("scattering", ReasonCode.REFLECTION_SCREEN_OR_WINDOW_MISMATCH),
    ("direct_delay", ReasonCode.REFLECTION_SCREEN_OR_WINDOW_MISMATCH),
    ("window_s", ReasonCode.REFLECTION_SCREEN_OR_WINDOW_MISMATCH),
    ("report_points_axis", ReasonCode.FREQUENCY_AXIS_MISMATCH),
    ("report_points_scattering", ReasonCode.REFLECTION_SCREEN_OR_WINDOW_MISMATCH),
])
def test_mismatched_physical_inputs_have_specific_reason_codes(change: str,
                                                                 expected: ReasonCode) -> None:
    left, right = _pair()
    assert left.screen is not None and left.window is not None
    assert left.report.path_table is not None
    assert left.report.points is not None
    screen, window, report_points = left.screen, left.window, left.report.points
    if change == "path_missing":
        left = replace(left, report=left.report.model_copy(update={"path_table": None}))
    elif change == "screen_missing":
        left = replace(left, screen=None)
    elif change == "window_missing":
        left = replace(left, window=None)
    elif change == "screen_scene":
        left = replace(left, screen=screen.model_copy(update={"scene_fingerprint": "b" * 64}))
    elif change == "screen_source":
        left = replace(left, screen=screen.model_copy(update={"source_m": right.report.scene.source_m}))
    elif change == "report_order":
        left = replace(left, report=left.report.model_copy(update={
            "top": left.report.top.model_copy(update={"reflection_order_k": 2})
        }))
    elif change == "axis":
        left = replace(left, screen=screen.model_copy(update={"frequencies_hz": _AXIS[:-1]}))
    elif change == "scattering":
        left = replace(left, window=window.model_copy(update={
            "scattering_coefficient": tuple(0.1 for _ in _AXIS)
        }))
    elif change == "direct_delay":
        left = replace(left, window=window.model_copy(update={
            "direct_delay_s": window.direct_delay_s + 0.001
        }))
    elif change == "window_s":
        left = replace(left, window=window.model_copy(update={"window_s": 0.01}))
    elif change == "report_points_axis":
        points = tuple(point.model_copy(update={"frequency_hz": 501.0})
                       if point.frequency_hz == 500.0 else point
                       for point in report_points)
        left = replace(left, report=left.report.model_copy(update={"points": points}))
    else:
        points = tuple(point.model_copy(update={"scattering": 0.1})
                       if point.frequency_hz == 500.0 else point
                       for point in report_points)
        left = replace(left, report=left.report.model_copy(update={"points": points}))
    result = _evaluate((left, right))
    assert result.state is EvaluationState.UNAVAILABLE
    assert result.reason_codes == (expected,)


def test_main_failure_invalidates_category_but_surrounding_failure_is_local() -> None:
    left, right = _pair()
    bad_main = replace(left, screen=None)
    assert _evaluate((bad_main, right)).state is EvaluationState.UNAVAILABLE
    around_left = replace(left, receiver_id="around", report_id="left-around", screen=None)
    around_right = replace(right, receiver_id="around", report_id="right-around")
    result = _evaluate((left, right, around_left, around_right))
    assert result.state is EvaluationState.MEASURED
    assert isinstance(result.payload, ReflectionsAndEchoPayload)
    local = next(item for item in result.payload.channels
                 if item.role == "left" and item.receiver_id == "around")
    assert local.state is MetricState.UNAVAILABLE
    assert local.reason_codes == (ReasonCode.REFLECTION_SCREEN_OR_WINDOW_MISSING,)


def test_small_room_fourth_order_enters_window_and_flags_unvalidated() -> None:
    left = _record("left", 2.2, room=_SMALL)
    right = _record("right", 2.6, room=_SMALL)
    result = _evaluate((left, right))
    assert result.state is EvaluationState.MEASURED
    assert Flag.UNVALIDATED in result.flags
    assert isinstance(result.payload, ReflectionsAndEchoPayload)
    assert left.window is not None
    assert any(path.source is ReflectionSource.WINDOW_EXTENSION
               and path.order == left.window.computed_order_k
               and path.within_window
               for channel in result.payload.channels for path in channel.reflections)


def test_small_room_extension_can_be_the_strongest_zone_path() -> None:
    left = _record("left", 2.2, room=_SMALL)
    right = _record("right", 2.6, room=_SMALL)
    table = left.report.path_table
    window = left.window
    assert table is not None and window is not None and window.rows
    rows = tuple(row.model_copy(update={
        "relative_direct_energy": tuple(0.0 for _ in _AXIS)
    }) if row.order > 0 else row for row in table.rows)
    extension = tuple(row.model_copy(update={
        "relative_direct_energy": tuple(1.0 if frequency == 500.0 and index == 0 else 0.0
                                        for frequency in _AXIS)
    }) for index, row in enumerate(window.rows))
    left = replace(left,
        report=left.report.model_copy(update={
            "path_table": table.model_copy(update={"rows": rows})}),
        window=window.model_copy(update={"rows": extension}),
    )
    result = _evaluate((left, right))
    assert isinstance(result.payload, ReflectionsAndEchoPayload)
    channel = next(item for item in result.payload.channels if item.role == "left")
    candidate = next(path for path in channel.reflections
                     if path.source is ReflectionSource.WINDOW_EXTENSION
                     and path.source_index == 0)
    zone = next(zone for zone in channel.zones if zone.zone is candidate.zone)
    point = next(point for point in zone.points if point.frequency_hz == 500.0)
    assert point.strongest_path_index is not None
    assert channel.reflections[point.strongest_path_index] == candidate


def test_input_order_does_not_change_any_output_cell() -> None:
    left, right = _pair()
    assert _evaluate((left, right)).model_dump(mode="json") == _evaluate(
        (right, left)).model_dump(mode="json")


def test_each_frequency_chooses_strongest_then_earlier_path() -> None:
    left, right = _pair()
    table = left.report.path_table
    assert table is not None
    index_early, index_late = 1, 8
    index_first_tie, index_second_tie = 15, 16
    energy = {
        index_early: {300.0: 0.2, 800.0: 0.4, 1000.0: 0.3},
        index_late: {300.0: 0.5, 800.0: 0.1, 1000.0: 0.3},
        index_first_tie: {2000.0: 0.6},
        index_second_tie: {2000.0: 0.6},
    }
    rows = tuple(row.model_copy(update={"relative_direct_energy": tuple(
        energy.get(index, {}).get(frequency, 0.0) for frequency in _AXIS
    )}) if row.order > 0 else row for index, row in enumerate(table.rows))
    left = replace(left, report=left.report.model_copy(update={
        "path_table": table.model_copy(update={"rows": rows})
    }))
    result = _evaluate((left, right))
    assert isinstance(result.payload, ReflectionsAndEchoPayload)
    channel = next(item for item in result.payload.channels if item.role == "left")
    front = next(zone for zone in channel.zones if zone.zone is DirectionZone.FRONT)
    vertical = next(zone for zone in channel.zones if zone.zone is DirectionZone.VERTICAL)
    expected = {300.0: index_late, 800.0: index_early, 1000.0: index_early}
    for point in front.points:
        if point.frequency_hz in expected:
            assert point.strongest_path_index is not None
            assert channel.reflections[point.strongest_path_index].source_index == expected[point.frequency_hz]
    tied = next(point for point in vertical.points if point.frequency_hz == 2000.0)
    assert tied.strongest_path_index is not None
    assert channel.reflections[tied.strongest_path_index].source_index == index_first_tie
    assert tied.total_energy_db.value == pytest.approx(10.0 * math.log10(1.2))


def test_wall_pair_uses_matching_third_octave_t20() -> None:
    records = _vary_pair(_pair(), tuple(
        {800.0: 0.25, 1000.0: 0.5, 1250.0: 0.75}.get(frequency, 0.5)
        for frequency in _AXIS
    ))
    result = _evaluate(records)
    assert isinstance(result.payload, ReflectionsAndEchoPayload)
    pair = next(item for item in result.payload.wall_pairs if item.walls == ("x0", "xL"))
    band = next(item for item in pair.bands if item.frequency_hz == 1000.0)
    decay = load_quality_targets(config_path("quality_targets.toml")).purpose(
        "dedicated_two_channel_listening_room").entry("reflections_and_echo.flutter_decay_db")
    assert isinstance(decay, SettingEntry)
    assert not isinstance(decay.value, tuple)
    assert pair.round_trip_delay_s.value is not None
    expected_loss = -10.0 * math.log10(0.5)
    assert band.round_trip_loss_db.value == pytest.approx(expected_loss)
    assert band.decay_duration_s.value == pytest.approx(
        float(decay.value) / expected_loss * pair.round_trip_delay_s.value)
    assert records[0].third_octave_decay is not None
    expected_t20 = next(row.t20_s for row in records[0].third_octave_decay.rows
                        if row.band.nominal_center_hz == 1000)
    assert band.room_t20_s.value == expected_t20


@pytest.mark.parametrize(("retention", "reason"), [
    (0.0, ReasonCode.ZERO_RETENTION),
    (1.0, ReasonCode.FULL_REFLECTION),
])
def test_wall_pair_noncomputable_retention_keeps_reason(retention: float,
                                                         reason: ReasonCode) -> None:
    records = _vary_pair(_pair(), tuple(retention for _ in _AXIS))
    result = _evaluate(records)
    assert isinstance(result.payload, ReflectionsAndEchoPayload)
    pair = next(item for item in result.payload.wall_pairs if item.walls == ("x0", "xL"))
    band = next(item for item in pair.bands if item.frequency_hz == 1000.0)
    assert band.round_trip_loss_db.reason_codes == (reason,)
    assert band.decay_duration_s.reason_codes == (reason,)
    assert band.round_trip_loss_db.value is None


def test_unavailable_t20_keeps_upstream_decay_reason() -> None:
    changed = []
    for item in _pair():
        bands = tuple(band.model_copy(update={
            "t20_s": None, "t20_unavailable_reason": "未達下緣"
        }) if band.center_frequency_hz == 1000.0 else band for band in item.report.bands)
        decay = item.third_octave_decay
        assert decay is not None
        rows = tuple(row.model_copy(update={
            "t20_s": None, "t20_unavailable_reason": "未達下緣"
        }) if row.band.octave_center_hz == 1000.0 else row for row in decay.rows)
        changed.append(replace(item, report=item.report.model_copy(update={"bands": bands}),
                               third_octave_decay=decay.model_copy(update={"rows": rows})))
    result = _evaluate(tuple(changed))
    assert isinstance(result.payload, ReflectionsAndEchoPayload)
    for pair in result.payload.wall_pairs:
        band = next(item for item in pair.bands if item.frequency_hz == 1000.0)
        assert band.room_t20_s.reason_codes == (ReasonCode.INSUFFICIENT_DECAY_RANGE,)


@pytest.mark.parametrize(("key", "value"), [
    ("reflections_and_echo.window_upper_ms", "16.0"),
    ("reflections_and_echo.frequency_range_hz", "[350.0, 8000.0]"),
    ("direction_zones.vertical_min_abs_elevation_deg", "32.0"),
    ("direction_zones.front_max_abs_azimuth_deg", "45.0"),
    ("direction_zones.rear_min_abs_azimuth_deg", "140.0"),
    ("reflections_and_echo.flutter_decay_db", "61.0"),
])
def test_settings_fingerprint_tracks_only_used_settings(tmp_path: Path, key: str,
                                                       value: str) -> None:
    records = _pair()
    ordinary = _evaluate(records)
    changed = _evaluate(records, _set_entry(tmp_path, key, value))
    assert ordinary.settings_fingerprint != changed.settings_fingerprint
    small = (_record("left", 2.2, room=_SMALL),
             _record("right", 2.6, room=_SMALL))
    assert ordinary.settings_fingerprint == _evaluate(small).settings_fingerprint


def test_listening_axis_rule_identity_is_in_settings_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = _pair()
    before = _evaluate(records)
    monkeypatch.setattr(reflections, "_LISTENING_AXIS_RULE", "loudspeaker_base_bisector.v2")
    after = _evaluate(records)
    assert before.settings_fingerprint != after.settings_fingerprint


def test_receiver_mismatch_scene_mismatch_and_wall_pair_mismatch_are_distinct() -> None:
    left, right = _pair()
    wrong_receiver = replace(right, receiver_id="other")
    assert _evaluate((left, wrong_receiver)).reason_codes == (ReasonCode.RECEIVER_ID_MISMATCH,)
    wrong_scene = replace(right, report=right.report.model_copy(update={
        "scene": right.report.scene.model_copy(update={"scene_fingerprint": "b" * 64})
    }))
    assert _evaluate((left, wrong_scene)).reason_codes == (ReasonCode.SCENE_FINGERPRINT_MISMATCH,)
    assert left.screen is not None and right.screen is not None
    wrong_pair = replace(right, screen=right.screen.model_copy(update={
        "pairs": (left.screen.pairs[1], left.screen.pairs[0], left.screen.pairs[2])
    }))
    assert _evaluate((left, wrong_pair)).reason_codes == (ReasonCode.REFLECTION_SCREEN_OR_WINDOW_MISMATCH,)


def test_undefined_axis_and_short_frequency_axis_are_unavailable() -> None:
    left, right = _pair()
    assert left.report.path_table is not None and left.report.points is not None
    assert left.screen is not None and left.window is not None
    one = ChannelGroup(channels=(ChannelDefinition(role="left", speaker_id="left"),),
                       comparisons=(), feature_match_tolerance_hz=0.0)
    single = evaluate_reflections(
        one, (left,), primary_receiver_id="main", candidate_id="candidate-a",
        purpose="dedicated_two_channel_listening_room",
        quality_targets_path=config_path("quality_targets.toml"),
    )
    assert single.reason_codes == (ReasonCode.LISTENING_AXIS_UNDEFINED,)
    short = left.report.path_table.model_copy(update={
        "frequencies_hz": _AXIS[:-2],
        "scattering_coefficient": _SCATTERING[:-2],
        "rows": tuple(row.model_copy(update={
            "relative_direct_energy": row.relative_direct_energy[:-2]
        }) for row in left.report.path_table.rows),
    })
    left = replace(left, report=left.report.model_copy(update={
                       "path_table": short, "points": left.report.points[:-2]}),
                   screen=left.screen.model_copy(update={"frequencies_hz": _AXIS[:-2]}),
                   window=left.window.model_copy(update={
                       "frequencies_hz": _AXIS[:-2],
                       "scattering_coefficient": _SCATTERING[:-2]}))
    assert _evaluate((left, right)).reason_codes == (ReasonCode.INSUFFICIENT_COVERAGE,)


def test_exact_upper_boundary_is_inside_and_later_outside_path_cannot_win(tmp_path: Path) -> None:
    left, right = _pair()
    table = left.report.path_table
    assert table is not None
    direct = next(row.delay_s for row in table.rows if row.order == 0)
    boundary = table.rows[1].delay_s - direct
    assert boundary == (boundary * 1000.0) / 1000.0
    settings = _set_entry(tmp_path, "reflections_and_echo.window_upper_ms",
                          repr(boundary * 1000.0))
    left = _record("left", 1.3, window_s=boundary)
    right = _record("right", 2.5, window_s=boundary)
    table = left.report.path_table
    assert table is not None
    outside = next(index for index, row in enumerate(table.rows)
                   if row.order > 0 and row.delay_s - direct > boundary)
    rows = tuple(row.model_copy(update={
        "relative_direct_energy": tuple(1e6 for _ in _AXIS)
    }) if index == outside else row for index, row in enumerate(table.rows))
    left = replace(left, report=left.report.model_copy(update={
        "path_table": table.model_copy(update={"rows": rows})
    }))
    result = _evaluate((left, right), settings)
    assert isinstance(result.payload, ReflectionsAndEchoPayload)
    channel = next(item for item in result.payload.channels if item.role == "left")
    boundary_path = next(path for path in channel.reflections if path.source_index == 1)
    outside_path = next(path for path in channel.reflections if path.source_index == outside)
    assert boundary_path.relative_direct_delay_s == result.payload.window_upper_s
    assert boundary_path.within_window
    assert not outside_path.within_window
    assert all(point.strongest_path_index != channel.reflections.index(outside_path)
               for zone in channel.zones for point in zone.points)


def test_window_incomplete_and_primary_k_mismatch_are_unavailable() -> None:
    left, right = _pair()
    assert left.window is not None
    incomplete = replace(left, window=left.window.model_copy(update={"coverage": "not_provable"}))
    result = _evaluate((incomplete, right))
    assert result.reason_codes == (ReasonCode.REFLECTION_WINDOW_INCOMPLETE,)
    high_k = replace(left, report=left.report.model_copy(update={
        "top": left.report.top.model_copy(update={"reflection_order_k": 4})
    }))
    result = _evaluate((high_k, right))
    assert result.reason_codes == (ReasonCode.REFLECTION_SCREEN_OR_WINDOW_MISMATCH,)
    assert Flag.UNVALIDATED not in result.flags


def test_t20_difference_and_cross_channel_axis_mismatch_are_unavailable() -> None:
    left, right = _pair()
    assert right.third_octave_decay is not None
    rows = tuple(row.model_copy(update={"t20_s": 2.0})
                 if row.band.nominal_center_hz == 1000 else row
                 for row in right.third_octave_decay.rows)
    altered = replace(right, third_octave_decay=right.third_octave_decay.model_copy(
        update={"rows": rows}))
    assert _evaluate((left, altered)).reason_codes == (
        ReasonCode.REFLECTION_SCREEN_OR_WINDOW_MISMATCH,)
    table = right.report.path_table
    assert table is not None and right.screen is not None and right.window is not None
    alternate_axis = tuple(frequency + 1.0 if frequency == 500.0 else frequency
                           for frequency in _AXIS)
    altered = replace(right,
        report=right.report.model_copy(update={"path_table": table.model_copy(update={
            "frequencies_hz": alternate_axis})}),
        screen=right.screen.model_copy(update={"frequencies_hz": alternate_axis}),
        window=right.window.model_copy(update={"frequencies_hz": alternate_axis}),
    )
    assert _evaluate((left, altered)).reason_codes == (ReasonCode.FREQUENCY_AXIS_MISMATCH,)


def test_surrounding_point_uses_primary_axis_and_primary_scene_identity() -> None:
    left, right = _pair()
    around_left = _record("left", 1.3, "around", receiver_y=2.1)
    around_right = _record("right", 2.5, "around", receiver_y=2.1)
    result = _evaluate((around_right, right, around_left, left))
    assert isinstance(result.payload, ReflectionsAndEchoPayload)
    assert result.scene_fingerprint == left.report.scene.scene_fingerprint
    assert result.payload.listening_axis_xy == (-1.0, 0.0)
    for channel in result.payload.channels:
        for path in channel.reflections:
            expected = path.room_azimuth_deg + 180.0
            if expected > 180.0:
                expected -= 360.0
            assert path.listening_azimuth_deg == pytest.approx(expected)


def test_role_mismatch_and_k_change_in_scene_fingerprint_are_rejected() -> None:
    left, right = _pair()
    wrong_role = replace(right, role="center")
    assert _evaluate((left, wrong_role)).reason_codes == (ReasonCode.CHANNEL_ROLE_MISMATCH,)
    different_k = _record("right", 2.5, order=4)
    assert different_k.report.scene.scene_fingerprint != left.report.scene.scene_fingerprint
    assert _evaluate((left, different_k)).reason_codes == (ReasonCode.SCENE_FINGERPRINT_MISMATCH,)


def test_surrounding_scene_and_order_failure_stay_local() -> None:
    left, right = _pair()
    around_left = _record("left", 1.3, "around", receiver_y=2.1)
    around_right = _record("right", 2.5, "around", receiver_y=2.1)
    other_scene = replace(around_left, report=around_left.report.model_copy(update={
        "scene": around_left.report.scene.model_copy(update={"scene_fingerprint": "b" * 64})
    }))
    result = _evaluate((left, right, other_scene, around_right))
    assert isinstance(result.payload, ReflectionsAndEchoPayload)
    assert result.scene_fingerprint == left.report.scene.scene_fingerprint
    channel = next(item for item in result.payload.channels if item.role == "left"
                   and item.receiver_id == "around")
    assert channel.reason_codes == (ReasonCode.SCENE_FINGERPRINT_MISMATCH,)
    invalid_order = replace(around_left, report=around_left.report.model_copy(update={
        "top": around_left.report.top.model_copy(update={"reflection_order_k": 8})
    }))
    result = _evaluate((left, right, invalid_order, around_right))
    assert result.state is EvaluationState.MEASURED
    assert isinstance(result.payload, ReflectionsAndEchoPayload)
    channel = next(item for item in result.payload.channels if item.role == "left"
                   and item.receiver_id == "around")
    assert channel.reason_codes == (ReasonCode.REFLECTION_SCREEN_OR_WINDOW_MISMATCH,)


def test_nonpositive_third_octave_t20_is_unavailable_in_wall_band() -> None:
    changed = []
    for item in _pair():
        bands = tuple(band.model_copy(update={"t20_s": 0.0})
                      if band.center_frequency_hz == 1000.0 else band
                      for band in item.report.bands)
        decay = item.third_octave_decay
        assert decay is not None
        rows = tuple(row.model_copy(update={"t20_s": 0.0})
                     if row.band.nominal_center_hz == 1000 else row for row in decay.rows)
        changed.append(replace(item, report=item.report.model_copy(update={"bands": bands}),
                               third_octave_decay=decay.model_copy(update={"rows": rows})))
    result = _evaluate(tuple(changed))
    assert isinstance(result.payload, ReflectionsAndEchoPayload)
    band = next(band for band in result.payload.wall_pairs[0].bands
                if band.frequency_hz == 1000.0)
    assert band.room_t20_s.reason_codes == (ReasonCode.NON_POSITIVE_VALUE,)


def test_registry_rejects_wrong_reflection_setting_unit(tmp_path: Path) -> None:
    path = _small_reflection_registry(
        tmp_path, "reflections_and_echo.window_upper_ms", "unit", '"s"')
    with pytest.raises(ValueError, match="window_upper_ms 必須是 ms"):
        _evaluate(_pair(), path)


def test_registry_rejects_vector_where_scalar_is_required(tmp_path: Path) -> None:
    path = _small_reflection_registry(
        tmp_path, "reflections_and_echo.window_upper_ms", "value", "[15.0, 16.0]")
    with pytest.raises(ValueError, match="window_upper_ms 必須是單一數值"):
        _evaluate(_pair(), path)


def test_registry_rejects_scalar_where_frequency_range_is_required(tmp_path: Path) -> None:
    path = _small_reflection_registry(
        tmp_path, "reflections_and_echo.frequency_range_hz", "value", "300.0")
    with pytest.raises(ValueError, match="frequency_range_hz 必須有兩個端點"):
        _evaluate(_pair(), path)


def test_guarded_primary_order_with_guarded_window_has_no_unvalidated_flag() -> None:
    left = _record("left", 1.3, order=NUMERICALLY_GUARDED_ORDER_K)
    right = _record("right", 2.5, order=NUMERICALLY_GUARDED_ORDER_K)
    assert left.window is not None and right.window is not None
    assert left.window.validation != "unvalidated"
    assert right.window.validation != "unvalidated"
    result = _evaluate((left, right))
    assert result.state is EvaluationState.MEASURED
    assert Flag.UNVALIDATED not in result.flags


def test_duplicate_role_at_same_receiver_is_rejected() -> None:
    left, right = _pair()
    duplicate = replace(left, report_id="other-left-report")
    assert _evaluate((left, right, duplicate)).reason_codes == (
        ReasonCode.CHANNEL_ROLE_MISMATCH,)


def test_role_speaker_mismatch_is_rejected() -> None:
    left, right = _pair()
    wrong_speaker = replace(right, speaker_id="another-speaker")
    assert _evaluate((left, wrong_speaker)).reason_codes == (
        ReasonCode.CHANNEL_ROLE_MISMATCH,)


def test_missing_declared_role_is_rejected() -> None:
    left, _ = _pair()
    assert _evaluate((left,)).reason_codes == (ReasonCode.CHANNEL_ROLE_MISMATCH,)


def test_extra_undeclared_role_is_rejected() -> None:
    left, right = _pair()
    extra = replace(left, role="center", report_id="center-main")
    assert _evaluate((left, right, extra)).reason_codes == (
        ReasonCode.CHANNEL_ROLE_MISMATCH,)


def test_exact_frequency_axis_ends_cover_range() -> None:
    axis = (300.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0)
    result = _evaluate((_record("left", 1.3, axis=axis),
                        _record("right", 2.5, axis=axis)))
    assert result.state is EvaluationState.MEASURED
    assert isinstance(result.payload, ReflectionsAndEchoPayload)


@pytest.mark.parametrize("axis", [
    (300.01, 500.0, 1000.0, 2000.0, 4000.0, 8000.0),
    (300.0, 500.0, 1000.0, 2000.0, 4000.0, 7999.99),
])
def test_frequency_axis_just_short_of_either_end_is_incomplete(
    axis: tuple[float, ...],
) -> None:
    result = _evaluate((_record("left", 1.3, axis=axis),
                        _record("right", 2.5, axis=axis)))
    assert result.reason_codes == (ReasonCode.INSUFFICIENT_COVERAGE,)


@pytest.mark.parametrize("edge", [300.0, 8000.0])
def test_exact_frequency_range_edge_is_counted_in_broadband_and_zones(edge: float) -> None:
    axis = (300.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0)
    left = _record("left", 1.3, axis=axis)
    right = _record("right", 2.5, axis=axis)
    table = left.report.path_table
    assert table is not None
    rows = tuple(row.model_copy(update={
        "relative_direct_energy": tuple(0.25 if frequency == edge else 0.0
                                        for frequency in axis)
    }) if index == 1 else row.model_copy(update={
        "relative_direct_energy": tuple(0.0 for _ in axis)
    }) if row.order > 0 else row for index, row in enumerate(table.rows))
    left = replace(left, report=left.report.model_copy(update={
        "path_table": table.model_copy(update={"rows": rows})
    }))
    result = _evaluate((left, right))
    assert isinstance(result.payload, ReflectionsAndEchoPayload)
    channel = next(item for item in result.payload.channels if item.role == "left")
    path = next(item for item in channel.reflections if item.source_index == 1
                and item.source is ReflectionSource.PATH_TABLE)
    assert path.broadband_state is MetricState.MEASURED
    assert any(point.frequency_hz == edge for zone in channel.zones for point in zone.points)


@pytest.mark.parametrize(("energy_at_lower", "energy_at_upper"), [
    (0.25, 0.0), (0.25, 1.0),
])
def test_third_octave_band_includes_lower_edge_excludes_upper_edge(
    energy_at_lower: float, energy_at_upper: float,
) -> None:
    target = next(band for band in third_octave_bands() if band.nominal_center_hz == 1000)
    lower, upper = target.lower_hz, target.upper_hz
    axis = (300.0, 500.0, lower, upper, 2000.0, 4000.0, 8000.0)
    records = (_record("left", 1.3, axis=axis), _record("right", 2.5, axis=axis))
    values = tuple(energy_at_lower if frequency == lower else
                   energy_at_upper if frequency == upper else 0.0 for frequency in axis)
    result = _evaluate(_vary_pair(records, values))
    assert isinstance(result.payload, ReflectionsAndEchoPayload)
    band = next(item for item in result.payload.wall_pairs[0].bands
                if item.frequency_hz == 1000.0)
    assert band.round_trip_loss_db.value == pytest.approx(-10.0 * math.log10(0.25))


def test_octave_band_without_any_axis_point_is_not_computable() -> None:
    axis = (300.0, 500.0, 1600.0, 2000.0, 4000.0, 8000.0)
    result = _evaluate((_record("left", 1.3, axis=axis),
                        _record("right", 2.5, axis=axis)))
    assert isinstance(result.payload, ReflectionsAndEchoPayload)
    band = next(item for item in result.payload.wall_pairs[0].bands
                if item.frequency_hz == 1000.0)
    assert band.round_trip_loss_db.value is None
    assert band.round_trip_loss_db.reason_codes == (ReasonCode.INSUFFICIENT_COVERAGE,)


def test_surrounding_axis_mismatch_is_local_even_when_each_physics_axis_agrees() -> None:
    left, right = _pair()
    altered_axis = tuple(frequency + 1.0 if frequency == 500.0 else frequency
                         for frequency in _AXIS)
    around_left = _record("left", 1.3, "around", axis=altered_axis)
    around_right = _record("right", 2.5, "around")
    result = _evaluate((left, right, around_left, around_right))
    assert isinstance(result.payload, ReflectionsAndEchoPayload)
    channel = next(item for item in result.payload.channels
                   if item.role == "left" and item.receiver_id == "around")
    assert channel.reason_codes == (ReasonCode.FREQUENCY_AXIS_MISMATCH,)


def test_primary_reports_with_different_scene_fingerprints_are_unavailable() -> None:
    left, right = _pair()
    changed = replace(right, report=right.report.model_copy(update={
        "scene": right.report.scene.model_copy(update={"scene_fingerprint": "b" * 64})
    }))
    assert _evaluate((left, changed)).reason_codes == (
        ReasonCode.SCENE_FINGERPRINT_MISMATCH,)


def test_primary_scene_mismatch_precedes_missing_screen_reason() -> None:
    left, right = _pair()
    without_screen = replace(left, screen=None)
    assert _evaluate((without_screen, right)).reason_codes == (
        ReasonCode.REFLECTION_SCREEN_OR_WINDOW_MISSING,)
    changed = replace(right, report=right.report.model_copy(update={
        "scene": right.report.scene.model_copy(update={"scene_fingerprint": "b" * 64})
    }))
    assert _evaluate((without_screen, changed)).reason_codes == (
        ReasonCode.SCENE_FINGERPRINT_MISMATCH,)


def test_empty_reflection_input_is_rejected() -> None:
    with pytest.raises(ValueError, match="至少要有一份報表"):
        _evaluate(())


def test_three_declared_channels_cannot_define_a_listening_axis() -> None:
    group = ChannelGroup(
        channels=(*_group().channels,
                  ChannelDefinition(role="center", speaker_id="center")),
        comparisons=(*_group().comparisons,
                     ChannelComparison(left_role="left", right_role="center")),
        feature_match_tolerance_hz=0.0,
    )
    records = (*_pair(), _record("center", 3.0))
    result = evaluate_reflections(
        group, records, primary_receiver_id="main", candidate_id="candidate-a",
        purpose="dedicated_two_channel_listening_room",
        quality_targets_path=config_path("quality_targets.toml"),
    )
    assert result.reason_codes == (ReasonCode.LISTENING_AXIS_UNDEFINED,)


@pytest.mark.parametrize("kind", ["t20", "screen"])
def test_surrounding_wall_or_t20_mismatch_stays_local(kind: str) -> None:
    left, right = _pair()
    around_left = _record("left", 1.3, "around", receiver_y=2.1)
    around_right = _record("right", 2.5, "around", receiver_y=2.1)
    if kind == "t20":
        assert around_left.third_octave_decay is not None
        rows = tuple(row.model_copy(update={"t20_s": 2.0})
                     if row.band.nominal_center_hz == 1000 else row
                     for row in around_left.third_octave_decay.rows)
        around_left = replace(around_left, third_octave_decay=around_left.third_octave_decay.model_copy(
            update={"rows": rows}))
    else:
        assert around_left.screen is not None
        pair = around_left.screen.pairs[0]
        changed = pair.model_copy(update={"round_trip_delay_s": pair.round_trip_delay_s + 0.001})
        around_left = replace(around_left, screen=around_left.screen.model_copy(update={
            "pairs": (changed, *around_left.screen.pairs[1:])
        }))
    result = _evaluate((left, right, around_left, around_right))
    assert result.state is EvaluationState.MEASURED
    assert isinstance(result.payload, ReflectionsAndEchoPayload)
    local = next(channel for channel in result.payload.channels
                 if channel.role == "left" and channel.receiver_id == "around")
    assert local.state is MetricState.UNAVAILABLE
    assert local.reason_codes == (ReasonCode.REFLECTION_SCREEN_OR_WINDOW_MISMATCH,)


def test_unavailable_surrounding_validation_does_not_flag_category() -> None:
    left, right = _pair()
    around_left = _record("left", 1.3, "around", receiver_y=2.1)
    around_right = _record("right", 2.5, "around", receiver_y=2.1)
    assert around_left.window is not None
    around_left = replace(around_left, window=around_left.window.model_copy(update={
        "coverage": "not_provable", "validation": "unvalidated"
    }))
    result = _evaluate((left, right, around_left, around_right))
    assert result.state is EvaluationState.MEASURED
    assert Flag.UNVALIDATED not in result.flags
    assert isinstance(result.payload, ReflectionsAndEchoPayload)
    local = next(channel for channel in result.payload.channels
                 if channel.role == "left" and channel.receiver_id == "around")
    assert local.reason_codes == (ReasonCode.REFLECTION_WINDOW_INCOMPLETE,)


def test_unavailable_category_has_no_measured_channel_validation_flag() -> None:
    left, right = _pair()
    assert left.window is not None
    left = replace(left, window=left.window.model_copy(update={
        "coverage": "not_provable", "validation": "unvalidated"
    }))
    result = _evaluate((left, right))
    assert result.state is EvaluationState.UNAVAILABLE
    assert Flag.UNVALIDATED not in result.flags


def test_covered_endpoints_without_in_range_sample_are_insufficient() -> None:
    axis = (250.0, 9000.0)
    result = _evaluate((_record("left", 1.3, axis=axis),
                        _record("right", 2.5, axis=axis)))
    assert result.reason_codes == (ReasonCode.INSUFFICIENT_COVERAGE,)
    assert reflections._broadband(axis, (0.2, 0.4), (300.0, 8000.0)).reason_codes == (
        ReasonCode.INSUFFICIENT_COVERAGE,)
    assert reflections._broadband(_AXIS, tuple(0.0 for _ in _AXIS),
                                  (300.0, 8000.0)).reason_codes == (
        ReasonCode.ZERO_REFLECTION_ENERGY,)


def test_empty_report_band_rows_make_wall_pair_unavailable() -> None:
    left, right = _pair()
    empty = tuple(replace(item, report=item.report.model_copy(update={"bands": ()}))
                  for item in (left, right))
    assert _evaluate(empty).reason_codes == (ReasonCode.BAND_ROW_MISSING,)
