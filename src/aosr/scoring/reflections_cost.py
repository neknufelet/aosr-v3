"""反射逐區非負超標代價、標記、顫動複核警戒與排名來源。"""
from __future__ import annotations

import json
from typing import Final

from aosr.config.frequency_axis import octave_cells_in_range
from aosr.config.quality_targets import (
    EntryStatus, QualityPurpose, SettingEntry, TargetEntry,
)
from aosr.scoring.contract import (
    CategoryCost, CategoryEvaluation, CostDirection, EvaluationState, Flag, QualityCategory,
    ReasonCode, UnassessedBand,
)
from aosr.scoring.cost_shapes import shape_cost, target, weight_table
from aosr.scoring.direction_zones import DirectionZone, zone_limits
from aosr.scoring.reflections_contract import (
    ReflectionChannel, ReflectionsAndEchoPayload, WallPairBandRisk, WallPairRisk,
    ZoneResult,
)
from aosr.scoring.review_alert import FlutterReviewAlert, ReviewAlert


_PREFIX: Final[str] = "reflections_and_echo."
_TARGET_KEY: Final[str] = _PREFIX + "zone_excess_db"
_WEIGHTS_KEY: Final[str] = _PREFIX + "within_category_weights"
_ZONE_FLAGS: Final[dict[DirectionZone, Flag]] = {
    DirectionZone.FRONT: Flag.REFLECTION_FRONT_ABOVE_THRESHOLD,
    DirectionZone.LATERAL: Flag.REFLECTION_LATERAL_ABOVE_THRESHOLD,
    DirectionZone.REAR: Flag.REFLECTION_REAR_ABOVE_THRESHOLD,
    DirectionZone.VERTICAL: Flag.REFLECTION_VERTICAL_ABOVE_THRESHOLD,
}


def _setting(purpose: QualityPurpose, key: str, unit: str) -> SettingEntry:
    entry = purpose.entry(key)
    if not isinstance(entry, SettingEntry) or entry.unit != unit:
        raise ValueError(f"{key} 必須是 {unit} 量法設定")
    return entry


def _number(entry: SettingEntry) -> float:
    if isinstance(entry.value, tuple):
        raise ValueError(f"{entry.key} 必須是單一數值")
    return float(entry.value)


def _sequence(entry: SettingEntry) -> tuple[float, ...]:
    if not isinstance(entry.value, tuple):
        raise ValueError(f"{entry.key} 必須是清單")
    return tuple(float(value) for value in entry.value)


def _checked_payload(
    evaluation: CategoryEvaluation, purpose: QualityPurpose
) -> ReflectionsAndEchoPayload:
    if evaluation.state is not EvaluationState.MEASURED:
        raise ValueError("反射代價只接 measured 評估")
    payload = evaluation.payload
    if not isinstance(payload, ReflectionsAndEchoPayload):
        raise TypeError("反射代價必須收到 ReflectionsAndEchoPayload")
    limits, _ = zone_limits(purpose)
    centers = _sequence(_setting(purpose, _PREFIX + "flutter_alert_band_centers_hz", "Hz"))
    settings_match = (
        payload.window_upper_ms == _number(_setting(purpose, _PREFIX + "window_upper_ms", "ms"))
        and payload.frequency_range_hz == _sequence(_setting(purpose, _PREFIX + "frequency_range_hz", "Hz"))
        and payload.zone_limits == limits
        and payload.flutter_alert_band_centers_hz == centers
    )
    if not settings_match:
        raise ValueError("評估器與代價設定使用的反射量法不同，代價沒有唯一答案")
    return payload


def _zone_threshold(purpose: QualityPurpose, zone: DirectionZone) -> float:
    return _number(_setting(purpose, _PREFIX + f"zone_threshold_db.{zone.value}", "dB"))


def _zone_excess(zone: ZoneResult, threshold: float, bounds: tuple[float, float]) -> float:
    frequencies = tuple(point.frequency_hz for point in zone.points)
    weights = octave_cells_in_range(frequencies, bounds)
    if not weights or sum(weights) <= 0.0:
        raise ValueError("反射分區沒有可計分的頻率點")
    excesses: list[float] = []
    for point in zone.points:
        if point.strongest_level_db is not None:
            excesses.append(max(0.0, point.strongest_level_db - threshold))
        elif point.strongest_reason_codes and set(point.strongest_reason_codes) <= {
            ReasonCode.NO_REFLECTION_IN_ZONE_POINT, ReasonCode.ZERO_REFLECTION_ENERGY
        }:
            excesses.append(0.0)
        else:
            raise ValueError("反射逐點缺值原因不可當成零超標")
    return sum(weight * excess for weight, excess in zip(weights, excesses, strict=True)) / sum(weights)


def _weights(purpose: QualityPurpose) -> dict[DirectionZone, float]:
    items = weight_table(purpose, _WEIGHTS_KEY).item
    if {item.name for item in items} != {zone.value for zone in DirectionZone}:
        raise ValueError("反射區權重列必須剛好是四個 DirectionZone")
    weights = {DirectionZone(item.name): item.value for item in items}
    if sum(weights.values()) <= 0.0:
        raise ValueError("反射區權重和必須大於零")
    return weights


def _channel_components(
    channel: ReflectionChannel, purpose: QualityPurpose,
    goal: TargetEntry, bounds: tuple[float, float],
) -> tuple[dict[str, float], dict[str, CostDirection], set[DirectionZone]]:
    components: dict[str, float] = {}
    directions: dict[str, CostDirection] = {}
    above: set[DirectionZone] = set()
    for zone in channel.zones:
        excess = _zone_excess(zone, _zone_threshold(purpose, zone.zone), bounds)
        name = f"{channel.role}.{zone.zone.value}"
        components[name] = shape_cost(goal, (excess,))
        directions[name] = "above_range" if excess > 0.0 else "within_range"
        if excess > 0.0:
            above.add(zone.zone)
    return components, directions, above


def _flutter_scan(
    payload: ReflectionsAndEchoPayload, purpose: QualityPurpose
) -> tuple[tuple[FlutterReviewAlert, ...], tuple[UnassessedBand, ...]]:
    decay_db = _number(_setting(purpose, _PREFIX + "flutter_decay_db", "dB"))
    alerts: list[FlutterReviewAlert] = []
    unassessed: dict[float, set[ReasonCode]] = {}
    selected = set(payload.flutter_alert_band_centers_hz)
    for pair in payload.wall_pairs:
        for band in pair.bands:
            if band.nominal_center_hz not in selected:
                continue
            reason = _flutter_reason(band)
            if reason is not None:
                unassessed.setdefault(band.frequency_hz, set()).update(reason)
                continue
            alert = _flutter_alert(pair, band, decay_db)
            if alert is not None:
                alerts.append(alert)
    bands = tuple(
        UnassessedBand(center_frequency_hz=center, reason_codes=tuple(
            code for code in ReasonCode if code in reasons
        ))
        for center, reasons in sorted(unassessed.items())
    )
    return tuple(alerts), bands


def _flutter_reason(band: WallPairBandRisk) -> tuple[ReasonCode, ...] | None:
    if band.room_t20_s.value is None:
        return band.room_t20_s.reason_codes
    if ReasonCode.INSUFFICIENT_COVERAGE in band.round_trip_loss_db.reason_codes:
        return band.round_trip_loss_db.reason_codes
    return None


def _flutter_alert(
    pair: WallPairRisk, band: WallPairBandRisk, decay_db: float
) -> FlutterReviewAlert | None:
    loss = band.round_trip_loss_db
    if ReasonCode.ZERO_RETENTION in loss.reason_codes:
        return None
    duration: float | None
    if ReasonCode.FULL_REFLECTION in loss.reason_codes:
        duration = None
        note = "全反射，持續度無限長，待複核"
    else:
        if loss.value is None or pair.round_trip_delay_s.value is None:
            raise ValueError("顫動損耗或來回延遲不可計算")
        duration = decay_db / loss.value * pair.round_trip_delay_s.value
        if duration != band.decay_duration_s.value:
            raise ValueError("評估器與代價設定使用的顫動衰減量不同")
        if band.room_t20_s.value is None or duration <= band.room_t20_s.value:
            return None
        note = "顫動持續度長過本房同帶 T20，待複核"
    if band.room_t20_s.value is None:
        raise ValueError("顫動警戒缺本房 T20")
    return FlutterReviewAlert(
        category=QualityCategory.REFLECTIONS_AND_ECHO, walls=pair.walls,
        nominal_center_hz=band.nominal_center_hz,
        center_frequency_hz=band.frequency_hz, lower_hz=band.lower_hz,
        upper_hz=band.upper_hz, decay_duration_s=duration,
        room_t20_s=band.room_t20_s.value, decay_db=decay_db, note=note,
    )


def cost_reflections_evaluation(
    evaluation: CategoryEvaluation, purpose: QualityPurpose,
    cost_settings_fingerprint: str,
) -> CategoryEvaluation:
    """只用主位兩支的四區非負超標重算代價，保留原始量。"""
    payload = _checked_payload(evaluation, purpose)
    goal = target(purpose, _TARGET_KEY, "dB")
    if goal.cost_shape != "less_is_better":
        raise ValueError("反射區超標目標必須是 less_is_better")
    weights = _weights(purpose)
    components: dict[str, float] = {}
    directions: dict[str, CostDirection] = {}
    above: set[DirectionZone] = set()
    channel_costs: list[float] = []
    for channel in payload.channels:
        if not channel.is_primary:
            continue
        parts, part_directions, channel_above = _channel_components(
            channel, purpose, goal, payload.frequency_range_hz
        )
        components.update(parts)
        directions.update(part_directions)
        above.update(channel_above)
        channel_costs.append(sum(
            weights[zone] * parts[f"{channel.role}.{zone.value}"] for zone in DirectionZone
        ) / sum(weights.values()))
    _, unassessed = _flutter_scan(payload, purpose)
    category_cost = CategoryCost(
        value=sum(channel_costs) / len(channel_costs), components=components,
        component_directions=directions, unassessed_bands=unassessed,
        cost_settings_fingerprint=cost_settings_fingerprint,
    )
    flags = tuple(flag for flag in evaluation.flags if flag not in _ZONE_FLAGS.values())
    flags += tuple(_ZONE_FLAGS[zone] for zone in DirectionZone if zone in above)
    document = evaluation.model_dump(mode="python")
    document.update(state=EvaluationState.COSTED, category_cost=category_cost, flags=flags)
    return CategoryEvaluation.model_validate(document)


def reflections_registry_sources(purpose: QualityPurpose) -> tuple[tuple[str, EntryStatus], ...]:
    """回排名讀過及核對過的反射登記簿列。"""
    keys = (
        _PREFIX + "window_upper_ms", _PREFIX + "frequency_range_hz",
        *(f"{_PREFIX}zone_threshold_db.{zone.value}" for zone in DirectionZone),
        *(f"direction_zones.{name}" for name in (
            "vertical_min_abs_elevation_deg", "front_max_abs_azimuth_deg",
            "rear_min_abs_azimuth_deg")),
        _PREFIX + "flutter_alert_band_centers_hz", _PREFIX + "flutter_decay_db",
    )
    records = [(key, _setting(purpose, key, "deg" if key.startswith("direction_zones.") else
                              "ms" if key.endswith("window_upper_ms") else
                              "Hz" if key.endswith("frequency_range_hz") or key.endswith("band_centers_hz") else "dB").status)
               for key in keys]
    records.append((_TARGET_KEY, target(purpose, _TARGET_KEY, "dB").status))
    records.extend((f"{_WEIGHTS_KEY}.{item.name}", item.status)
                   for item in weight_table(purpose, _WEIGHTS_KEY).item)
    return tuple(records)


def reflections_review_alerts(
    evaluation: CategoryEvaluation, purpose: QualityPurpose
) -> tuple[ReviewAlert, ...]:
    payload = evaluation.payload
    if not isinstance(payload, ReflectionsAndEchoPayload):
        return ()
    return _flutter_scan(payload, purpose)[0]


def comparison_support(evaluation: CategoryEvaluation) -> str:
    """只記實際計分主位聲道與完整的計分頻率序列。

    頻率要整串列出（照殘響、聲道匹配的前例），只記頭、尾、點數的話，只換一個中間頻點的兩個
    候選會被放進同一張表，其實沒量到同一批頻率（#480）。
    """
    payload = evaluation.payload
    if not isinstance(payload, ReflectionsAndEchoPayload):
        return ""
    primary = tuple(channel for channel in payload.channels if channel.is_primary)
    return json.dumps({
        "channels": [{"role": channel.role, "speaker_id": channel.speaker_id}
                     for channel in primary],
        "primary_receiver_id": payload.primary_receiver_id,
        "scoring_frequencies_hz": [point.frequency_hz for point in primary[0].zones[0].points],
    }, sort_keys=True, separators=(",", ":"))


cost_evaluation = cost_reflections_evaluation
registry_sources = reflections_registry_sources
review_alerts = reflections_review_alerts
