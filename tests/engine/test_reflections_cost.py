"""#351 反射已量輸出的代價與逐區超標考卷。"""
from __future__ import annotations

import math
import re
from pathlib import Path

import pytest

from aosr.config.paths import config_path
from aosr.config.quality_targets import QualityTargets, load_quality_targets
from aosr.scoring.contract import (
    CategoryCost, CategoryEvaluation, EvaluationState, Flag, MetricState, ReasonCode,
)
from aosr.scoring.direction_zones import DirectionZone
from aosr.scoring.reflections_cost import cost_reflections_evaluation, reflections_registry_sources
from aosr.scoring.reflections_contract import MetricCell, ReflectionsAndEchoPayload
from tests.engine import test_reflections as fixtures


def test_measured_reflections_become_costed_with_manual_zone_excess() -> None:
    """若仍缺代價接點，或把負聲級直接當代價，這題會紅。"""
    measured = fixtures._evaluate(fixtures._pair())
    registry = load_quality_targets(config_path("quality_targets.toml"))
    result = cost_reflections_evaluation(
        measured, registry.purpose("dedicated_two_channel_listening_room"), registry.fingerprint
    )
    assert result.state is EvaluationState.COSTED
    assert isinstance(result.category_cost, CategoryCost)
    assert isinstance(result.payload, ReflectionsAndEchoPayload)
    primary = tuple(channel for channel in result.payload.channels if channel.is_primary)
    manual: dict[str, float] = {}
    for channel in primary:
        for zone in channel.zones:
            # 這份考卷的等距 log 軸用相鄰中點切格，兩端切到登記範圍。
            frequencies = [point.frequency_hz for point in zone.points]
            logs = [math.log2(frequency) for frequency in frequencies]
            edges = [math.log2(result.payload.frequency_range_hz[0])]
            edges += [(a + b) / 2 for a, b in zip(logs, logs[1:])]
            edges += [math.log2(result.payload.frequency_range_hz[1])]
            weights = [right - left for left, right in zip(edges, edges[1:])]
            excess = [max(0.0, (point.strongest_level_db or -math.inf) + 10.0)
                      for point in zone.points]
            mean = sum(weight * value for weight, value in zip(weights, excess)) / sum(weights)
            manual[f"{channel.role}.{zone.zone.value}"] = mean / 10.0
    assert result.category_cost.components == pytest.approx(manual)
    expected = sum(manual.values()) / len(manual)
    assert result.category_cost.value == pytest.approx(expected)
    zone_flags = {
        DirectionZone.FRONT: Flag.REFLECTION_FRONT_ABOVE_THRESHOLD,
        DirectionZone.LATERAL: Flag.REFLECTION_LATERAL_ABOVE_THRESHOLD,
        DirectionZone.REAR: Flag.REFLECTION_REAR_ABOVE_THRESHOLD,
        DirectionZone.VERTICAL: Flag.REFLECTION_VERTICAL_ABOVE_THRESHOLD,
    }
    for direction_zone in DirectionZone:
        should_flag = any(manual[f"{channel.role}.{direction_zone.value}"] > 0 for channel in primary)
        assert (zone_flags[direction_zone] in result.flags) is should_flag


def _registry(path: Path) -> QualityTargets:
    return load_quality_targets(path)


def _alter(tmp_path: Path, key: str, field: str, new: str) -> Path:
    source = config_path("quality_targets.toml").read_text()
    pattern = rf'(key = "{re.escape(key)}"\n(?:[^\n]*\n)*?{field} = )[^\n]+'
    changed = re.sub(pattern, lambda match: match.group(1) + new, source, count=1)
    assert changed != source
    path = tmp_path / "quality_targets.toml"
    path.write_text(changed)
    return path


def _cost(evaluation: CategoryEvaluation, registry: QualityTargets) -> CategoryEvaluation:
    return cost_reflections_evaluation(
        evaluation, registry.purpose("dedicated_two_channel_listening_room"), registry.fingerprint
    )


def test_changed_threshold_and_scale_recost_without_changing_evaluator_identity(tmp_path: Path) -> None:
    measured = fixtures._evaluate(fixtures._pair())
    original = _cost(measured, _registry(config_path("quality_targets.toml")))
    assert isinstance(original.category_cost, CategoryCost)
    threshold_path = _alter(tmp_path, "reflections_and_echo.zone_threshold_db.front", "value", "-20.0")
    changed = _cost(measured, _registry(threshold_path))
    assert isinstance(changed.category_cost, CategoryCost)
    assert changed.category_cost.value > original.category_cost.value
    assert changed.settings_fingerprint == original.settings_fingerprint
    scale_path = _alter(tmp_path, "reflections_and_echo.zone_excess_db", "worse_reference", "20.0")
    scaled = _cost(measured, _registry(scale_path))
    assert isinstance(scaled.category_cost, CategoryCost)
    assert scaled.category_cost.value == pytest.approx(original.category_cost.value / 2)


def test_missing_zone_weight_row_is_rejected(tmp_path: Path) -> None:
    source = config_path("quality_targets.toml").read_text()
    marker = 'name = "vertical"\nvalue = 1.0\n'
    renamed = source.replace(marker, 'name = "surround"\nvalue = 1.0\n', 1)
    assert renamed != source  # 找不到就是登記簿改了樣子，這題要先紅在這裡、不能安靜略過
    source = renamed
    path = tmp_path / "quality_targets.toml"
    path.write_text(source)
    with pytest.raises(ValueError, match="DirectionZone"):
        _cost(fixtures._evaluate(fixtures._pair()), _registry(path))


def test_payload_measurement_settings_must_match_registry(tmp_path: Path) -> None:
    measured = fixtures._evaluate(fixtures._pair())
    for key, field, value in (
        ("reflections_and_echo.window_upper_ms", "value", "16.0"),
        ("reflections_and_echo.frequency_range_hz", "value", "[301.0, 8000.0]"),
        ("direction_zones.front_max_abs_azimuth_deg", "value", "41.0"),
        ("reflections_and_echo.flutter_alert_band_centers_hz", "value",
         "[400.0, 500.0, 630.0, 800.0, 1000.0]"),
    ):
        with pytest.raises(ValueError, match="量法不同"):
            _cost(measured, _registry(_alter(tmp_path, key, field, value)))


def test_zone_point_without_reflection_is_zero_but_unknown_reason_is_rejected() -> None:
    measured = fixtures._evaluate(fixtures._pair())
    assert isinstance(measured.payload, ReflectionsAndEchoPayload)
    left = next(channel for channel in measured.payload.channels if channel.role == "left")
    rear = next(zone for zone in left.zones if zone.zone is DirectionZone.REAR)
    assert all(point.strongest_level_db is None and
               ReasonCode.NO_REFLECTION_IN_ZONE_POINT in point.strongest_reason_codes
               for point in rear.points)
    costed = _cost(measured, _registry(config_path("quality_targets.toml")))
    assert isinstance(costed.category_cost, CategoryCost)
    assert costed.category_cost.components["left.rear"] == 0.0
    for reasons in ((ReasonCode.OTHER_ERROR,),
                    (ReasonCode.NO_REFLECTION_IN_ZONE_POINT, ReasonCode.OTHER_ERROR)):
        bad = rear.points[0].model_copy(update={"strongest_reason_codes": reasons})
        bad_rear = rear.model_copy(update={"points": (bad, *rear.points[1:])})
        changed_left = left.model_copy(update={"zones": tuple(
            bad_rear if zone.zone is DirectionZone.REAR else zone for zone in left.zones)})
        payload = measured.payload.model_copy(update={"channels": tuple(
            changed_left if channel.role == "left" else channel for channel in measured.payload.channels)})
        with pytest.raises(ValueError, match="缺值原因"):
            _cost(measured.model_copy(update={"payload": payload}),
                  _registry(config_path("quality_targets.toml")))


def test_hand_set_levels_clip_edge_cells_and_keep_exact_threshold_at_zero() -> None:
    measured = fixtures._evaluate(fixtures._pair())
    assert isinstance(measured.payload, ReflectionsAndEchoPayload)
    changed_channels = []
    for channel in measured.payload.channels:
        zones = []
        for zone in channel.zones:
            levels = {300.0: 0.0, 500.0: -10.0, 800.0: -5.0, 8000.0: 0.0}
            points = tuple(point.model_copy(update={"strongest_level_db":
                           levels.get(point.frequency_hz, -20.0) if point.strongest_level_db is not None
                           else None}) for point in zone.points)
            if channel.role != "left" or zone.zone is not DirectionZone.FRONT:
                points = tuple(point.model_copy(update={"strongest_level_db":
                               -20.0 if point.strongest_level_db is not None else None})
                               for point in zone.points)
            zones.append(zone.model_copy(update={"points": points}))
        changed_channels.append(channel.model_copy(update={"zones": tuple(zones)}))
    payload = measured.payload.model_copy(update={"channels": tuple(changed_channels)})
    result = _cost(measured.model_copy(update={"payload": payload}),
                   _registry(config_path("quality_targets.toml")))
    assert isinstance(result.category_cost, CategoryCost)
    first = math.log2(500.0 / 300.0) / 2
    middle = math.log2(1000.0 / 500.0) / 2
    last = math.log2(8000.0 / 4000.0) / 2
    expected_excess = (first * 10.0 + middle * 5.0 + last * 10.0) / math.log2(8000.0 / 300.0)
    assert result.category_cost.components["left.front"] == pytest.approx(expected_excess / 10.0)
    assert result.category_cost.value == pytest.approx(expected_excess / 80.0)
    assert result.category_cost.component_directions["left.front"] == "above_range"
    assert result.category_cost.component_directions["right.front"] == "within_range"


def test_one_zone_weight_changes_class_cost(tmp_path: Path) -> None:
    measured = fixtures._evaluate(fixtures._pair())
    original = _cost(measured, _registry(config_path("quality_targets.toml")))
    assert isinstance(original.category_cost, CategoryCost)
    source = config_path("quality_targets.toml").read_text()
    old = 'key = "reflections_and_echo.within_category_weights"\n\n[[purpose.weight.item]]\nname = "front"\nvalue = 1.0'
    changed = source.replace(old, old.replace("value = 1.0", "value = 4.0"), 1)
    assert changed != source
    path = tmp_path / "quality_targets.toml"
    path.write_text(changed)
    weighted = _cost(measured, _registry(path))
    assert isinstance(weighted.category_cost, CategoryCost)
    assert weighted.category_cost.value != pytest.approx(original.category_cost.value)
    # 精確加權：前區權重 4、其他 1，分母是權重和 7；權重套錯區（例如套到側區）值就不對。
    costs = original.category_cost.components
    roles = sorted({name.split('.')[0] for name in costs})
    per_channel = [(4.0 * costs[f"{role}.front"] + costs[f"{role}.lateral"] + costs[f"{role}.rear"]
                    + costs[f"{role}.vertical"]) / 7.0 for role in roles]
    assert costs[f"{roles[0]}.front"] != costs[f"{roles[0]}.lateral"]
    assert weighted.category_cost.value == pytest.approx(sum(per_channel) / len(per_channel))


def test_zero_reflection_energy_point_counts_zero_excess() -> None:
    measured = fixtures._evaluate(fixtures._pair())
    assert isinstance(measured.payload, ReflectionsAndEchoPayload)
    left = next(channel for channel in measured.payload.channels if channel.role == "left")
    lateral = next(zone for zone in left.zones if zone.zone is DirectionZone.LATERAL)
    first = lateral.points[0]
    assert first.strongest_level_db is not None and first.strongest_level_db > -10.0
    zero = first.model_copy(update={
        "strongest_level_db": None, "strongest_state": MetricState.NOT_COMPUTABLE,
        "strongest_reason_codes": (ReasonCode.ZERO_REFLECTION_ENERGY,),
        "total_energy_db": MetricCell(value=None, state=MetricState.NOT_COMPUTABLE,
                                      reason_codes=(ReasonCode.ZERO_REFLECTION_ENERGY,)),
    })
    zones = tuple(zone.model_copy(update={"points": (zero, *zone.points[1:])})
                  if zone.zone is DirectionZone.LATERAL else zone for zone in left.zones)
    changed_left = left.model_copy(update={"zones": zones})
    payload = measured.payload.model_copy(update={"channels": tuple(
        changed_left if channel.role == "left" else channel for channel in measured.payload.channels)})
    registry = _registry(config_path("quality_targets.toml"))
    original = _cost(measured, registry)
    result = _cost(measured.model_copy(update={"payload": payload}), registry)
    assert isinstance(original.category_cost, CategoryCost)
    assert isinstance(result.category_cost, CategoryCost)
    assert result.category_cost.components["left.lateral"] < original.category_cost.components["left.lateral"]


def test_cost_entry_rejects_wrong_state_and_payload_type() -> None:
    measured = fixtures._evaluate(fixtures._pair())
    registry = _registry(config_path("quality_targets.toml"))
    costed = _cost(measured, registry)
    with pytest.raises(ValueError, match="measured"):
        _cost(costed, registry)
    with pytest.raises(TypeError, match="ReflectionsAndEchoPayload"):
        _cost(measured.model_copy(update={"payload": None}), registry)


def test_registry_sources_name_all_measurement_and_cost_rows() -> None:
    purpose = _registry(config_path("quality_targets.toml")).purpose(
        "dedicated_two_channel_listening_room")
    sources = dict(reflections_registry_sources(purpose))
    expected = {
        "reflections_and_echo.window_upper_ms", "reflections_and_echo.frequency_range_hz",
        "reflections_and_echo.flutter_alert_band_centers_hz",
        "reflections_and_echo.flutter_decay_db", "reflections_and_echo.zone_excess_db",
        "direction_zones.vertical_min_abs_elevation_deg",
        "direction_zones.front_max_abs_azimuth_deg",
        "direction_zones.rear_min_abs_azimuth_deg",
        *(f"reflections_and_echo.zone_threshold_db.{zone.value}" for zone in DirectionZone),
        *(f"reflections_and_echo.within_category_weights.{zone.value}" for zone in DirectionZone),
    }
    assert set(sources) == expected
    assert sources["reflections_and_echo.zone_excess_db"] == "baseline"


@pytest.mark.parametrize(("key", "field", "value", "match"), [
    ("reflections_and_echo.window_upper_ms", "unit", '"s"', "ms 量法設定"),
    ("reflections_and_echo.window_upper_ms", "value", "[15.0, 16.0]", "單一數值"),
    ("reflections_and_echo.frequency_range_hz", "value", "300.0", "清單"),
])
def test_malformed_reflection_settings_are_rejected(
    tmp_path: Path, key: str, field: str, value: str, match: str,
) -> None:
    """登記簿某一格的單位或形狀不對，代價要明確拒收，不猜著用。"""
    measured = fixtures._evaluate(fixtures._pair())
    with pytest.raises(ValueError, match=match):
        _cost(measured, _registry(_alter(tmp_path, key, field, value)))


def test_zone_weight_table_missing_a_row_or_all_zero_is_rejected(tmp_path: Path) -> None:
    """少一區（不是改名）要明確拒收，不能變成查不到那一區就當掉；四區全 0 也要拒收。"""
    measured = fixtures._evaluate(fixtures._pair())
    source = config_path("quality_targets.toml").read_text()
    row = re.compile(r'\[\[purpose\.weight\.item\]\]\nname = "vertical"\n(?:[^\n]+\n)*\n')
    stripped = row.sub("", source, count=1)
    assert stripped != source
    missing = tmp_path / "missing.toml"
    missing.write_text(stripped)
    with pytest.raises(ValueError, match="DirectionZone"):
        _cost(measured, _registry(missing))
    head, tail = source.split('key = "reflections_and_echo.within_category_weights"', 1)
    table, rest = tail.split("\n# ", 1)
    assert "value = 1.0" in table
    zero = tmp_path / "zero.toml"
    zero.write_text(head + 'key = "reflections_and_echo.within_category_weights"'
                    + table.replace("value = 1.0", "value = 0.0") + "\n# " + rest)
    with pytest.raises(ValueError, match="權重和"):
        _cost(measured, _registry(zero))


def test_zone_excess_target_must_stay_less_is_better(tmp_path: Path) -> None:
    measured = fixtures._evaluate(fixtures._pair())
    path = _alter(tmp_path, "reflections_and_echo.zone_excess_db", "cost_shape", '"beyond_threshold_only"')
    with pytest.raises(ValueError, match="less_is_better"):
        _cost(measured, _registry(path))


def test_registry_source_status_follows_the_registry(tmp_path: Path) -> None:
    """排名表頭寫 calibrated 還是 baseline 看這裡回的狀態，要照登記簿讀，不准寫死今天的值。"""
    key = "reflections_and_echo.zone_threshold_db.front"
    before = dict(reflections_registry_sources(
        _registry(config_path("quality_targets.toml")).purpose("dedicated_two_channel_listening_room")))
    assert before[key] == "calibrated"
    path = _alter(tmp_path, key, "status", '"baseline"')
    after = dict(reflections_registry_sources(
        _registry(path).purpose("dedicated_two_channel_listening_room")))
    assert after[key] == "baseline"
