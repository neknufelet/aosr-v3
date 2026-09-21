"""殘響的逐帶容許區間代價、相鄰帶突變代價與排名資料資格。"""

from __future__ import annotations

import json
from enum import StrEnum
from typing import Final

from aosr.config.quality_targets import (
    EntryStatus,
    QualificationEntry,
    QualityPurpose,
    SettingEntry,
    TargetEntry,
    Unit,
)
from aosr.scoring.contract import (
    CategoryCost,
    CategoryEvaluation,
    CostDirection,
    EvaluationState,
    ReasonCode,
    ReverberationPayload,
    UnassessedBand,
)
from aosr.scoring.cost_shapes import (
    shape_cost as _shape_cost,
    target as _target,
    weight_table as _weight_table,
)


_CENTERS_KEY: Final[str] = "reverberation.target_band_centers_hz"
_NOMINAL_KEY: Final[str] = "reverberation.target_t20_nominal_s_by_band"
_TOLERANCE_KEY: Final[str] = "reverberation.target_t20_tolerance_s_by_band"
_LOG_BASE_KEY: Final[str] = "reverberation.adjacent_t20_logarithm_base"
_INTERVAL_EXCESS_KEY: Final[str] = "reverberation.t20_interval_excess_s"
_JUMP_EXCESS_KEY: Final[str] = "reverberation.adjacent_t20_log_ratio_excess"
_WEIGHTS_KEY: Final[str] = "reverberation.within_category_weights"
_MAX_UNAVAILABLE_KEY: Final[str] = "ranking.eligibility.max_unavailable_bands"
_CRITICAL_BANDS_KEY: Final[str] = "ranking.eligibility.critical_bands"
_MIN_VALID_KEY: Final[str] = "ranking.eligibility.min_valid_bands"
_SETTING_KEYS: Final[tuple[str, ...]] = (
    _CENTERS_KEY,
    _NOMINAL_KEY,
    _TOLERANCE_KEY,
    _LOG_BASE_KEY,
)
_TARGET_KEYS: Final[tuple[str, ...]] = (_INTERVAL_EXCESS_KEY, _JUMP_EXCESS_KEY)
_ELIGIBILITY_KEYS: Final[tuple[str, ...]] = (
    _MAX_UNAVAILABLE_KEY,
    _CRITICAL_BANDS_KEY,
    _MIN_VALID_KEY,
)
_SETTING_UNITS: Final[dict[str, Unit]] = {
    _CENTERS_KEY: "Hz",
    _NOMINAL_KEY: "s",
    _TOLERANCE_KEY: "s",
    _LOG_BASE_KEY: "1",
}
_TARGET_UNITS: Final[dict[str, Unit]] = {
    _INTERVAL_EXCESS_KEY: "s",
    _JUMP_EXCESS_KEY: "1",
}
_PRINCIPAL_NAMES: Final[frozenset[str]] = frozenset(
    {"t20_target_interval", "adjacent_t20_jump"}
)


class ReverberationEligibilityReason(StrEnum):
    """三條頻帶資料資格各自的機器原因。"""

    TOO_MANY_UNAVAILABLE_BANDS = "reverberation_too_many_unavailable_bands"
    CRITICAL_BAND_UNAVAILABLE = "reverberation_critical_band_unavailable"
    INSUFFICIENT_VALID_BANDS = "reverberation_insufficient_valid_bands"


def _setting(
    purpose: QualityPurpose, key: str, expected_unit: Unit
) -> SettingEntry:
    entry = purpose.entry(key)
    if not isinstance(entry, SettingEntry):
        raise TypeError(f"{key} 不是量法設定")
    if entry.unit != expected_unit:
        raise ValueError(
            f"{key} 單位應為 {expected_unit}，登記簿寫 {entry.unit}"
        )
    return entry


def _qualification(purpose: QualityPurpose, key: str) -> QualificationEntry:
    entry = purpose.entry(key)
    if not isinstance(entry, QualificationEntry):
        raise TypeError(f"{key} 不是資格規則")
    return entry


def _number(entry: SettingEntry) -> float:
    if isinstance(entry.value, tuple):
        raise TypeError(f"{entry.key} 必須是單一數值")
    return float(entry.value)


def _numbers(entry: SettingEntry) -> tuple[float, ...]:
    if not isinstance(entry.value, tuple):
        raise TypeError(f"{entry.key} 必須是數值清單")
    return tuple(float(value) for value in entry.value)


def _target_intervals(purpose: QualityPurpose) -> dict[float, tuple[float, float]]:
    """由登記簿的同索引中心頻率、名義值與容許量生成逐帶閉區間。"""
    centers = _numbers(_setting(purpose, _CENTERS_KEY, _SETTING_UNITS[_CENTERS_KEY]))
    nominal = _numbers(_setting(purpose, _NOMINAL_KEY, _SETTING_UNITS[_NOMINAL_KEY]))
    tolerance = _numbers(
        _setting(purpose, _TOLERANCE_KEY, _SETTING_UNITS[_TOLERANCE_KEY])
    )
    if not centers or len(centers) != len(nominal) or len(centers) != len(tolerance):
        raise ValueError("殘響中心頻率、名義值與逐帶容許量必須非空且等長")
    if tuple(sorted(centers)) != centers or len(set(centers)) != len(centers):
        raise ValueError("殘響目標中心頻率必須嚴格遞增")
    if any(center <= 0.0 for center in centers):
        raise ValueError("殘響目標中心頻率必須為正")
    if any(value <= 0.0 for value in nominal):
        raise ValueError("殘響 T20 名義值必須為正")
    if any(
        width < 0.0 or value <= width
        for value, width in zip(nominal, tolerance, strict=True)
    ):
        raise ValueError("殘響逐帶容許量必須非負且小於名義值")
    return {
        center: (value - width, value + width)
        for center, value, width in zip(centers, nominal, tolerance, strict=True)
    }


def _principal_weights(purpose: QualityPurpose) -> dict[str, float]:
    weights = {
        item.name: item.value for item in _weight_table(purpose, _WEIGHTS_KEY).item
    }
    if set(weights) != _PRINCIPAL_NAMES:
        raise ValueError(f"{_WEIGHTS_KEY} 的名稱必須剛好是 {sorted(_PRINCIPAL_NAMES)}")
    return weights


def _interval_excess_and_direction(
    value: float, lower: float, upper: float
) -> tuple[float, CostDirection]:
    """回逐帶區間的非負超出量與方向。

    這裡刻意不冒充 ``in_range_best``：殘響的上下限由三張逐帶表同索引生成，不是單一
    ``value ± tolerance`` 目標。這裡只求領域特有的區間超出量；正規化仍交給共用的
    ``less_is_better`` 形狀，避免另寫第二份代價公式。
    """
    if value < lower:
        return lower - value, "below_range"
    if value > upper:
        return value - upper, "above_range"
    return 0.0, "within_range"


def _checked_targets(
    payload: ReverberationPayload, purpose: QualityPurpose
) -> tuple[TargetEntry, TargetEntry]:
    """讀兩條代價目標並核對代價形狀與相鄰帶對數底；對不上就報錯。"""
    interval_target = _target(
        purpose, _INTERVAL_EXCESS_KEY, _TARGET_UNITS[_INTERVAL_EXCESS_KEY]
    )
    jump_target = _target(
        purpose, _JUMP_EXCESS_KEY, _TARGET_UNITS[_JUMP_EXCESS_KEY]
    )
    if interval_target.cost_shape != "less_is_better":
        raise ValueError(f"{_INTERVAL_EXCESS_KEY} 必須是 less_is_better")
    if jump_target.cost_shape != "beyond_threshold_only":
        raise ValueError(f"{_JUMP_EXCESS_KEY} 必須是 beyond_threshold_only")
    if payload.logarithm_base != _number(
        _setting(purpose, _LOG_BASE_KEY, _SETTING_UNITS[_LOG_BASE_KEY])
    ):
        raise ValueError("評估器與代價設定使用的相鄰帶對數底不同")
    return interval_target, jump_target


def _cost_components(
    payload: ReverberationPayload, purpose: QualityPurpose
) -> tuple[
    dict[str, float],
    dict[str, float],
    dict[str, CostDirection],
    tuple[UnassessedBand, ...],
]:
    intervals = _target_intervals(purpose)
    interval_target, jump_target = _checked_targets(payload, purpose)
    components: dict[str, float] = {}
    directions: dict[str, CostDirection] = {}
    unassessed_bands: list[UnassessedBand] = []
    interval_costs: list[float] = []
    payload_centers = {band.center_frequency_hz for band in payload.bands}
    for center in sorted(set(intervals) - payload_centers):
        unassessed_bands.append(
            UnassessedBand(
                center_frequency_hz=center,
                reason_codes=(ReasonCode.BAND_ROW_MISSING,),
            )
        )
    for band in payload.bands:
        if band.center_frequency_hz not in intervals:
            raise ValueError(f"{band.center_frequency_hz:g} Hz 沒有殘響目標區間")
        if band.t20.value is None:
            unassessed_bands.append(
                UnassessedBand(
                    center_frequency_hz=band.center_frequency_hz,
                    reason_codes=band.t20.reason_codes,
                )
            )
            continue
        lower, upper = intervals[band.center_frequency_hz]
        excess, direction = _interval_excess_and_direction(band.t20.value, lower, upper)
        cost = _shape_cost(interval_target, (excess,))
        name = f"t20_target_interval.{band.center_frequency_hz:g}Hz"
        components[name] = cost
        directions[name] = direction
        interval_costs.append(cost)
    jump_costs: list[float] = []
    for change in payload.adjacent_band_changes:
        if change.signed_log_ratio is None:
            continue
        cost = _shape_cost(jump_target, (change.signed_log_ratio,))
        label = f"{change.lower_center_frequency_hz:g}-{change.upper_center_frequency_hz:g}Hz"
        components[f"adjacent_t20_jump.{label}"] = cost
        jump_costs.append(cost)
    aggregates: dict[str, float] = {}
    if interval_costs:
        aggregates["t20_target_interval"] = sum(interval_costs) / len(interval_costs)
    if jump_costs:
        aggregates["adjacent_t20_jump"] = sum(jump_costs) / len(jump_costs)
    components.update(aggregates)
    unassessed_bands.sort(key=lambda band: band.center_frequency_hz)
    return components, aggregates, directions, tuple(unassessed_bands)


def cost_reverberation_evaluation(
    evaluation: CategoryEvaluation,
    purpose: QualityPurpose,
    cost_settings_fingerprint: str,
) -> CategoryEvaluation:
    """把已量殘響升成 ``costed``，但所有 T20 都不可估時維持 ``measured``、不捏造零代價。

    逐帶區間代價只平均真的可估帶，相鄰突變也只平均真的可算配對；兩個類內主項再按登記簿
    權重做加權平均。選平均是為了不讓頻帶數本身放大代價；分母只含真的存在的主項與帶，
    所以不可估資料不會以零混入。T30 與擬合差異只留在 payload 當診斷，完全不讀入代價。
    """
    if evaluation.state is not EvaluationState.MEASURED:
        raise ValueError("殘響代價只接 measured 評估")
    payload = evaluation.payload
    if not isinstance(payload, ReverberationPayload):
        raise TypeError("殘響代價必須收到 ReverberationPayload")
    components, aggregates, directions, unassessed_bands = _cost_components(
        payload, purpose
    )
    if not aggregates:
        return evaluation
    weights = _principal_weights(purpose)
    denominator = sum(weights[name] for name in aggregates)
    if denominator <= 0.0:
        raise ValueError("殘響真的存在的主分項權重和必須大於零")
    value = sum(aggregates[name] * weights[name] for name in aggregates) / denominator
    document = evaluation.model_dump(mode="python")
    document.update(
        state=EvaluationState.COSTED,
        category_cost=CategoryCost(
            value=value,
            components=components,
            component_directions=directions,
            unassessed_bands=unassessed_bands,
            cost_settings_fingerprint=cost_settings_fingerprint,
        ),
    )
    return CategoryEvaluation.model_validate(document)


def reverberation_eligibility_reasons(
    evaluation: CategoryEvaluation, purpose: QualityPurpose
) -> tuple[ReverberationEligibilityReason, ...]:
    """按 T20 頻帶證據判三條排名資格；回原因，不改也不加聲學代價。"""
    payload = evaluation.payload
    if not isinstance(payload, ReverberationPayload):
        return ()
    target_centers = set(_target_intervals(purpose))
    available_centers = target_centers & {
        band.center_frequency_hz for band in payload.bands if band.t20.value is not None
    }
    unavailable_centers = target_centers - available_centers
    max_unavailable = _qualification(purpose, _MAX_UNAVAILABLE_KEY).value
    critical_bands = _qualification(purpose, _CRITICAL_BANDS_KEY).value
    min_valid = _qualification(purpose, _MIN_VALID_KEY).value
    if not isinstance(max_unavailable, int) or not isinstance(min_valid, int):
        raise TypeError("殘響不可估帶上限與最低有效帶數必須是整數")
    if not isinstance(critical_bands, tuple) or not all(
        isinstance(center, float) for center in critical_bands
    ):
        raise TypeError("殘響關鍵頻帶必須是浮點數清單")
    reasons: list[ReverberationEligibilityReason] = []
    if len(unavailable_centers) > max_unavailable:
        reasons.append(ReverberationEligibilityReason.TOO_MANY_UNAVAILABLE_BANDS)
    if any(center not in available_centers for center in critical_bands):
        reasons.append(ReverberationEligibilityReason.CRITICAL_BAND_UNAVAILABLE)
    if len(available_centers) < min_valid:
        reasons.append(ReverberationEligibilityReason.INSUFFICIENT_VALID_BANDS)
    return tuple(reasons)


def comparison_support(evaluation: CategoryEvaluation) -> str:
    """回實際計入代價的 T20 帶與相鄰配對之可讀正規 JSON。

    鍵排序、緊密分隔符與數值排序固定，讓同一評估支撐必定產生同一字串。
    """
    payload = evaluation.payload
    if not isinstance(payload, ReverberationPayload):
        return ""
    document = {
        "adjacent_pairs_hz": sorted(
            [
                change.lower_center_frequency_hz,
                change.upper_center_frequency_hz,
            ]
            for change in payload.adjacent_band_changes
            if change.signed_log_ratio is not None
        ),
        "t20_bands_hz": sorted(
            band.center_frequency_hz
            for band in payload.bands
            if band.t20.value is not None
        ),
    }
    return json.dumps(document, sort_keys=True, separators=(",", ":"))


def reverberation_registry_sources(
    purpose: QualityPurpose,
) -> tuple[tuple[str, EntryStatus], ...]:
    """回排名真的讀過的殘響登記簿列，供表頭判斷校準狀態。"""
    records = [
        *(
            (key, _setting(purpose, key, _SETTING_UNITS[key]).status)
            for key in _SETTING_KEYS
        ),
        *(
            (key, _target(purpose, key, _TARGET_UNITS[key]).status)
            for key in _TARGET_KEYS
        ),
        *((key, _qualification(purpose, key).status) for key in _ELIGIBILITY_KEYS),
    ]
    records.extend(
        (f"{_WEIGHTS_KEY}.{item.name}", item.status)
        for item in _weight_table(purpose, _WEIGHTS_KEY).item
    )
    return tuple(records)


cost_evaluation = cost_reverberation_evaluation
eligibility_reasons = reverberation_eligibility_reasons
eligibility_keys = _ELIGIBILITY_KEYS
registry_sources = reverberation_registry_sources
