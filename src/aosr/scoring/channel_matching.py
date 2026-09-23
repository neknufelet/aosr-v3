"""聲道匹配第一層評估器（票 #350 第一段）：逐接收點先比，再沿聆聽區彙總。"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Final, Literal, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from aosr.config.quality_targets import (
    QualityPurpose,
    SettingEntry,
    Unit,
    load_quality_targets,
)
from aosr.scoring.contract import (
    CONTRACT_SCHEMA_VERSION,
    CategoryEvaluation,
    ChannelBroadbandSupport,
    ChannelComparisonAggregate,
    ChannelComparisonPair,
    ChannelFeatureDifference,
    ChannelFrequencyDifference,
    ChannelIdentity,
    ChannelMatchingPayload,
    ChannelMetricAggregate,
    ChannelPointMatch,
    ChannelPointSources,
    ChannelSourceEvaluation,
    EvaluationState,
    Feature,
    Flag,
    InputProvenance,
    MetricState,
    QualityCategory,
    RawQuantity,
    ReasonCode,
    TimbrePayload,
    in_declared_order,
)
from aosr.scoring.receiver_set import ReceiverPoint, ReceiverRole, ReceiverSet
from aosr.scoring.placement import (
    PlacementMismatchError,
    merge_or_empty,
    merge_placements,
)
from aosr.scoring.timbre import _octave_cells_in_range, _smooth_energy


CHANNEL_MATCHING_EVALUATOR_VERSION: Final[str] = "aosr.scoring.channel_matching.v5"
_PREFIX: Final[str] = "channel_matching."
_BROADBAND_KEY: Final[str] = _PREFIX + "broadband_range_hz"
_SMOOTHING_KEY: Final[str] = "timbre_balance.smoothing_width_octave_ripple"
_SWITCH_KEY: Final[str] = _PREFIX + "direct_time_cost_enabled"
_SETTING_UNITS: Final[dict[str, Unit]] = {
    _BROADBAND_KEY: "Hz",
    _SMOOTHING_KEY: "oct",
    _SWITCH_KEY: "1",
}
FROZEN = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
INPUT = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=True)


class ChannelDefinition(BaseModel):
    """聲道組裡一個角色與實際喇叭身分；只有比較設定明列的角色對會被比較。"""

    model_config = FROZEN

    role: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    speaker_id: str = Field(min_length=1)


class ChannelComparison(BaseModel):
    """設定明列的一個有序角色對；評估器用它尋找同一接收點的兩邊輸入。"""

    model_config = FROZEN

    left_role: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    right_role: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")

    @model_validator(mode="after")
    def _roles_are_distinct(self) -> Self:
        if self.left_role == self.right_role:
            raise ValueError("比較對的兩個角色不可相同")
        return self


class ChannelGroup(BaseModel):
    """可擴充聲道清單與明列比較設定；單聲道可不宣告比較對。

    內容正規化後形成共同指紋。
    """

    model_config = FROZEN

    channels: tuple[ChannelDefinition, ...] = Field(min_length=1)
    comparisons: tuple[ChannelComparison, ...]
    feature_match_tolerance_hz: Annotated[float, Field(ge=0.0)]

    @model_validator(mode="after")
    def _roles_and_pairs_are_unambiguous(self) -> Self:
        roles = [item.role for item in self.channels]
        speakers = [item.speaker_id for item in self.channels]
        if len(roles) != len(set(roles)) or len(speakers) != len(set(speakers)):
            raise ValueError("聲道角色與 speaker_id 都不可重複")
        if not self.comparisons and len(self.channels) != 1:
            raise ValueError("沒有比較對的聲道組只准有一個聲道")
        pairs = [(item.left_role, item.right_role) for item in self.comparisons]
        if len(pairs) != len(set(pairs)):
            raise ValueError("比較對不可重複")
        if any(left not in roles or right not in roles for left, right in pairs):
            raise ValueError("比較對必須引用聲道組內角色")
        return self

    @property
    def fingerprint(self) -> str:
        channels = sorted(
            (item.model_dump(mode="json") for item in self.channels),
            key=lambda item: str(item["role"]),
        )
        comparisons = sorted(
            (item.model_dump(mode="json") for item in self.comparisons),
            key=lambda item: (str(item["left_role"]), str(item["right_role"])),
        )
        canonical = json.dumps(
            {
                "channels": channels,
                "comparisons": comparisons,
                "feature_match_tolerance_hz": self.feature_match_tolerance_hz,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ChannelResponse(BaseModel):
    """同一接收點的一支聲道原始音色結果、線性能量曲線與直達距離。"""

    model_config = INPUT

    role: str = Field(min_length=1)
    timbre_evaluation: CategoryEvaluation
    frequencies_hz: tuple[float, ...] = Field(min_length=2)
    total_energy: tuple[float, ...]
    direct_distance_m: float

    @model_validator(mode="after")
    def _axis_and_raw_evaluation_are_well_formed(self) -> Self:
        if len(self.frequencies_hz) != len(self.total_energy):
            raise ValueError("total_energy 必須與 frequencies_hz 等長")
        if any(
            not math.isfinite(value) or value <= 0.0 for value in self.frequencies_hz
        ):
            raise ValueError("frequencies_hz 必須是有限正頻率")
        if any(
            upper <= lower
            for lower, upper in zip(
                self.frequencies_hz, self.frequencies_hz[1:], strict=False
            )
        ):
            raise ValueError("frequencies_hz 必須嚴格遞增")
        evaluation = self.timbre_evaluation
        if evaluation.category != QualityCategory.TIMBRE_BALANCE:
            raise ValueError("聲道匹配只收原始音色評估")
        if evaluation.state == EvaluationState.COSTED:
            raise ValueError("聲道匹配不收已算代價的音色結果")
        return self


class ChannelPointInput(BaseModel):
    """一個接收點的五個比較身分與全部聲道資料。"""

    model_config = FROZEN

    receiver_id: str = Field(min_length=1)
    receiver_set_fingerprint: str = Field(min_length=1)
    timbre_settings_fingerprint: str = Field(min_length=1)
    listening_area_settings_fingerprint: str = Field(min_length=1)
    channel_group_fingerprint: str = Field(min_length=1)
    responses: tuple[ChannelResponse, ...] = Field(min_length=2)


@dataclass(frozen=True)
class _Settings:
    broadband_range_hz: tuple[float, float]
    smoothing_width_octave: float
    direct_time_cost_enabled: bool
    any_baseline: bool
    registry_fingerprint: str


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


def _number(entry: SettingEntry) -> float:
    if isinstance(entry.value, tuple):
        raise TypeError(f"{entry.key} 必須是單一數值")
    return float(entry.value)


def _range(entry: SettingEntry) -> tuple[float, float]:
    if not isinstance(entry.value, tuple) or len(entry.value) != 2:
        raise TypeError(f"{entry.key} 必須是兩個數的範圍")
    lower, upper = (float(value) for value in entry.value)
    if not 0.0 < lower < upper:
        raise ValueError(f"{entry.key} 必須是遞增正值範圍")
    return lower, upper


def _load_settings(path: str | Path, purpose_name: str) -> _Settings:
    registry = load_quality_targets(path)
    purpose = registry.purpose(purpose_name)
    broadband = _setting(purpose, _BROADBAND_KEY, _SETTING_UNITS[_BROADBAND_KEY])
    smoothing = _setting(purpose, _SMOOTHING_KEY, _SETTING_UNITS[_SMOOTHING_KEY])
    switch = _setting(purpose, _SWITCH_KEY, _SETTING_UNITS[_SWITCH_KEY])
    if switch.value not in (0, 1):
        raise ValueError(f"{switch.key} 必須是 0 或 1")
    # 左右差異曲線用的是音色那一格「起伏的平滑寬度」：0＝看原始曲線（票 #432，老闆拍「用原始檔」），負的紅。
    if _number(smoothing) < 0.0:
        raise ValueError(f"{smoothing.key} 不准是負的")
    used = (broadband, smoothing, switch)
    return _Settings(
        broadband_range_hz=_range(broadband),
        smoothing_width_octave=_number(smoothing),
        direct_time_cost_enabled=bool(switch.value),
        any_baseline=any(item.status == "baseline" for item in used),
        registry_fingerprint=registry.fingerprint,
    )


def _measured_points(receiver_set: ReceiverSet) -> tuple[ReceiverPoint, ...]:
    return tuple(
        point
        for point in receiver_set.points
        if point.role in (ReceiverRole.PRIMARY, ReceiverRole.SURROUNDING)
    )


def _identity_reasons(
    receiver_set: ReceiverSet,
    points: Sequence[ChannelPointInput],
    candidate_id: str,
    timbre_fingerprint: str,
    listening_fingerprint: str,
    group: ChannelGroup,
    scene_fingerprint: str,
) -> tuple[ReasonCode, ...]:
    expected_ids = {point.receiver_id for point in _measured_points(receiver_set)}
    actual_ids = [point.receiver_id for point in points]
    reasons: list[ReasonCode] = []
    if expected_ids - set(actual_ids):
        reasons.append(ReasonCode.MISSING_POINTS)
    if set(actual_ids) - expected_ids or len(actual_ids) != len(set(actual_ids)):
        reasons.append(ReasonCode.RECEIVER_ID_MISMATCH)
    expected_speakers = {item.role: item.speaker_id for item in group.channels}
    upstream_versions = {
        response.timbre_evaluation.evaluator_version
        for point in points
        for response in point.responses
    }
    if len(upstream_versions) > 1:
        reasons.append(ReasonCode.EVALUATOR_VERSION_MISMATCH)
    if any(
        response.timbre_evaluation.scene_fingerprint != scene_fingerprint
        for point in points
        for response in point.responses
    ):
        reasons.append(ReasonCode.SCENE_FINGERPRINT_MISMATCH)
    try:
        merge_placements(
            response.timbre_evaluation.placement
            for point in points
            for response in point.responses
        )
    except PlacementMismatchError:
        reasons.append(ReasonCode.PLACEMENT_MISMATCH)
    for point in points:
        if point.receiver_set_fingerprint != receiver_set.fingerprint:
            reasons.append(ReasonCode.RECEIVER_SET_FINGERPRINT_MISMATCH)
        if point.timbre_settings_fingerprint != timbre_fingerprint:
            reasons.append(ReasonCode.TIMBRE_SETTINGS_FINGERPRINT_MISMATCH)
        if point.listening_area_settings_fingerprint != listening_fingerprint:
            reasons.append(ReasonCode.LISTENING_AREA_SETTINGS_FINGERPRINT_MISMATCH)
        if point.channel_group_fingerprint != group.fingerprint:
            reasons.append(ReasonCode.CHANNEL_GROUP_FINGERPRINT_MISMATCH)
        _response_identity_reasons(
            point, expected_speakers, candidate_id, timbre_fingerprint, reasons
        )
    return in_declared_order(reasons)


def _response_identity_reasons(
    point: ChannelPointInput,
    expected_speakers: dict[str, str],
    candidate_id: str,
    timbre_fingerprint: str,
    reasons: list[ReasonCode],
) -> None:
    roles = [response.role for response in point.responses]
    if set(roles) != set(expected_speakers) or len(roles) != len(set(roles)):
        reasons.append(ReasonCode.CHANNEL_ROLE_MISMATCH)
    for response in point.responses:
        evaluation = response.timbre_evaluation
        if evaluation.candidate_id != candidate_id:
            reasons.append(ReasonCode.CANDIDATE_ID_MISMATCH)
        if evaluation.settings_fingerprint != timbre_fingerprint:
            reasons.append(ReasonCode.TIMBRE_SETTINGS_FINGERPRINT_MISMATCH)
        if evaluation.provenance.receiver_id != point.receiver_id:
            reasons.append(ReasonCode.RECEIVER_ID_MISMATCH)
        if evaluation.provenance.speaker_id != expected_speakers.get(response.role):
            reasons.append(ReasonCode.CHANNEL_ROLE_MISMATCH)


def _settings_fingerprint(
    settings: _Settings,
    receiver_set: ReceiverSet,
    points: Sequence[ChannelPointInput],
    group: ChannelGroup,
    timbre_fingerprint: str,
    listening_fingerprint: str,
) -> str:
    upstream_versions = sorted(
        {
            response.timbre_evaluation.evaluator_version
            for point in points
            for response in point.responses
        }
    )
    canonical = json.dumps(
        {
            "registry": settings.registry_fingerprint,
            "channel_group": group.fingerprint,
            "timbre": timbre_fingerprint,
            "listening_area": listening_fingerprint,
            "receiver_layout": receiver_set.layout_fingerprint,
            "timbre_evaluator_versions": upstream_versions,
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _provenance(
    candidate_id: str, receiver_set: ReceiverSet, group: ChannelGroup
) -> InputProvenance:
    return InputProvenance(
        report_id=f"channel-matching:{candidate_id}",
        engine_commit="multiple-input-reports",
        speaker_id=f"channel-group:{group.fingerprint}",
        receiver_id=f"receiver-set:{receiver_set.fingerprint}",
    )


def _flags(points: Sequence[ChannelPointInput], baseline: bool) -> tuple[Flag, ...]:
    found = [
        flag
        for point in points
        for response in point.responses
        for flag in response.timbre_evaluation.flags
    ]
    if baseline:
        found.append(Flag.BASELINE_SETTINGS)
    return in_declared_order(found)


def _unavailable(
    receiver_set: ReceiverSet,
    points: Sequence[ChannelPointInput],
    candidate_id: str,
    group: ChannelGroup,
    settings_fingerprint: str,
    scene_fingerprint: str,
    reasons: tuple[ReasonCode, ...],
    baseline: bool,
) -> CategoryEvaluation:
    return CategoryEvaluation(
        schema_version=CONTRACT_SCHEMA_VERSION,
        candidate_id=candidate_id,
        scene_fingerprint=scene_fingerprint,
        placement=merge_or_empty(
            response.timbre_evaluation.placement
            for point in points
            for response in point.responses
        ),
        category=QualityCategory.CHANNEL_MATCHING,
        state=EvaluationState.UNAVAILABLE,
        payload=None,
        raw_quantities=(),
        category_cost=None,
        flags=_flags(points, baseline),
        reason_codes=reasons,
        evaluator_version=CHANNEL_MATCHING_EVALUATOR_VERSION,
        settings_fingerprint=settings_fingerprint,
        provenance=_provenance(candidate_id, receiver_set, group),
    )


def _source(response: ChannelResponse) -> ChannelSourceEvaluation:
    evaluation = response.timbre_evaluation
    state: Literal[EvaluationState.MEASURED, EvaluationState.UNAVAILABLE]
    if evaluation.state is EvaluationState.MEASURED:
        state = EvaluationState.MEASURED
    elif evaluation.state is EvaluationState.UNAVAILABLE:
        state = EvaluationState.UNAVAILABLE
    else:
        raise ValueError("聲道匹配的原始音色輸入只能是 measured 或 unavailable")
    payload = (
        evaluation.payload if isinstance(evaluation.payload, TimbrePayload) else None
    )
    return ChannelSourceEvaluation(
        schema_version=evaluation.schema_version,
        candidate_id=evaluation.candidate_id,
        role=response.role,
        speaker_id=evaluation.provenance.speaker_id,
        state=state,
        payload=payload,
        raw_quantities=evaluation.raw_quantities,
        flags=evaluation.flags,
        reason_codes=evaluation.reason_codes,
        evaluator_version=evaluation.evaluator_version,
        settings_fingerprint=evaluation.settings_fingerprint,
        provenance=evaluation.provenance,
    )


def _matched_features(
    left: Sequence[Feature], right: Sequence[Feature], tolerance_hz: float
) -> tuple[set[int], set[int]]:
    candidates = sorted(
        (abs(a.center_frequency_hz - b.center_frequency_hz), left_i, right_i)
        for left_i, a in enumerate(left)
        for right_i, b in enumerate(right)
        if a.kind == b.kind
        and abs(a.center_frequency_hz - b.center_frequency_hz) <= tolerance_hz
    )
    used_left: set[int] = set()
    used_right: set[int] = set()
    for _, left_i, right_i in candidates:
        if left_i not in used_left and right_i not in used_right:
            used_left.add(left_i)
            used_right.add(right_i)
    return used_left, used_right


def _unmatched_features(
    left_role: str,
    right_role: str,
    left: TimbrePayload,
    right: TimbrePayload,
    tolerance_hz: float,
) -> tuple[ChannelFeatureDifference, ...]:
    used_left, used_right = _matched_features(
        left.features, right.features, tolerance_hz
    )
    return (
        *(
            ChannelFeatureDifference(present_in_role=left_role, feature=feature)
            for index, feature in enumerate(left.features)
            if index not in used_left
        ),
        *(
            ChannelFeatureDifference(present_in_role=right_role, feature=feature)
            for index, feature in enumerate(right.features)
            if index not in used_right
        ),
    )


def _data_reason(
    left: ChannelResponse,
    right: ChannelResponse,
    broadband_range: tuple[float, float],
) -> ReasonCode | None:
    evaluations = (left.timbre_evaluation, right.timbre_evaluation)
    if any(
        item.state != EvaluationState.MEASURED
        or not isinstance(item.payload, TimbrePayload)
        for item in evaluations
    ):
        return ReasonCode.CHANNEL_RESULT_UNAVAILABLE
    if left.frequencies_hz != right.frequencies_hz:
        return ReasonCode.FREQUENCY_AXIS_MISMATCH
    energy = (*left.total_energy, *right.total_energy)
    if any(not math.isfinite(value) or value <= 0.0 for value in energy):
        return ReasonCode.NON_POSITIVE_ENERGY
    if any(
        not math.isfinite(item.direct_distance_m) or item.direct_distance_m < 0.0
        for item in (left, right)
    ):
        return ReasonCode.INVALID_DIRECT_DISTANCE
    if not any(
        broadband_range[0] <= frequency <= broadband_range[1]
        for frequency in left.frequencies_hz
    ):
        return ReasonCode.INSUFFICIENT_COVERAGE
    return None


def _point_reason_codes(
    left: ChannelResponse,
    right: ChannelResponse,
    reason: ReasonCode,
) -> tuple[ReasonCode, ...]:
    return tuple(
        dict.fromkeys(
            (
                reason,
                *left.timbre_evaluation.reason_codes,
                *right.timbre_evaluation.reason_codes,
            )
        )
    )


def _difference_curve(
    left: ChannelResponse, right: ChannelResponse, smoothing_width: float
) -> tuple[ChannelFrequencyDifference, ...]:
    frequencies = np.asarray(left.frequencies_hz, dtype=np.float64)
    octaves = np.log2(frequencies)
    left_energy = np.asarray(left.total_energy, dtype=np.float64)
    right_energy = np.asarray(right.total_energy, dtype=np.float64)
    left_relative = left_energy / float(np.max(left_energy))
    right_relative = right_energy / float(np.max(right_energy))
    left_db = 10.0 * np.log10(_smooth_energy(octaves, left_relative, smoothing_width))
    right_db = 10.0 * np.log10(_smooth_energy(octaves, right_relative, smoothing_width))
    return tuple(
        ChannelFrequencyDifference(
            frequency_hz=float(frequency),
            left_minus_right_db=float(left_value - right_value),
        )
        for frequency, left_value, right_value in zip(
            frequencies, left_db, right_db, strict=True
        )
    )


def _measured_point(
    receiver: ReceiverPoint,
    comparison: ChannelComparison,
    left: ChannelResponse,
    right: ChannelResponse,
    settings: _Settings,
    group: ChannelGroup,
    sound_speed_m_s: float,
) -> ChannelPointMatch:
    left_payload = left.timbre_evaluation.payload
    right_payload = right.timbre_evaluation.payload
    if not isinstance(left_payload, TimbrePayload) or not isinstance(
        right_payload, TimbrePayload
    ):
        raise TypeError("可估聲道必須帶 TimbrePayload")
    # 已查第一層：這兩個摘要各自用音色登記簿的傾斜／起伏平滑寬度算完；
    # 這裡才做左減右，不能回頭拿原始能量曲線冒充未平滑摘要。
    # 寬頻音量每八度等權（#450）：加密低頻不會讓低頻在寬頻音量裡多拿票。
    frequencies = np.asarray(left.frequencies_hz, dtype=np.float64)
    weights = _octave_cells_in_range(frequencies, settings.broadband_range_hz)
    left_total = float(np.sum(weights * np.asarray(left.total_energy, dtype=np.float64)))
    right_total = float(np.sum(weights * np.asarray(right.total_energy, dtype=np.float64)))
    return ChannelPointMatch(
        receiver_id=receiver.receiver_id,
        importance=receiver.importance,
        left_role=comparison.left_role,
        right_role=comparison.right_role,
        state=MetricState.MEASURED,
        reason_codes=(),
        reason=None,
        tilt_difference_db_per_octave=(
            left_payload.tilt_db_per_octave - right_payload.tilt_db_per_octave
        ),
        ripple_rms_difference_db=(
            left_payload.residual_rms_db - right_payload.residual_rms_db
        ),
        broadband_level_difference_db=10.0 * math.log10(left_total / right_total),
        direct_time_difference_ms=(
            (left.direct_distance_m - right.direct_distance_m)
            / sound_speed_m_s
            * 1000.0
        ),
        frequency_difference_curve_db=_difference_curve(
            left, right, settings.smoothing_width_octave
        ),
        unmatched_features=_unmatched_features(
            comparison.left_role,
            comparison.right_role,
            left_payload,
            right_payload,
            group.feature_match_tolerance_hz,
        ),
    )


def _point_results(
    receiver_set: ReceiverSet,
    points: Sequence[ChannelPointInput],
    group: ChannelGroup,
    settings: _Settings,
    sound_speed_m_s: float,
) -> tuple[ChannelPointMatch, ...]:
    by_receiver = {point.receiver_id: point for point in points}
    results: list[ChannelPointMatch] = []
    for receiver in _measured_points(receiver_set):
        responses = {
            item.role: item for item in by_receiver[receiver.receiver_id].responses
        }
        for comparison in group.comparisons:
            left = responses[comparison.left_role]
            right = responses[comparison.right_role]
            # 任何一點不可估，整類早在 ``_support_reasons`` 就回不可估了；走到這裡只會是已量。
            results.append(
                _measured_point(
                    receiver, comparison, left, right, settings, group, sound_speed_m_s
                )
            )
    return tuple(results)


def _point_unavailability_reasons(
    receiver_set: ReceiverSet,
    points: Sequence[ChannelPointInput],
    group: ChannelGroup,
    broadband_range: tuple[float, float],
) -> tuple[ReasonCode, ...]:
    by_receiver = {point.receiver_id: point for point in points}
    found: list[ReasonCode] = []
    for receiver in _measured_points(receiver_set):
        responses = {
            item.role: item for item in by_receiver[receiver.receiver_id].responses
        }
        for comparison in group.comparisons:
            reason = _data_reason(
                responses[comparison.left_role],
                responses[comparison.right_role],
                broadband_range,
            )
            if reason is not None:
                found.extend(
                    _point_reason_codes(
                        responses[comparison.left_role],
                        responses[comparison.right_role],
                        reason,
                    )
                )
    if not found:
        return ()
    return tuple(dict.fromkeys((ReasonCode.REQUIRED_CHANNEL_POINT_UNAVAILABLE, *found)))


def _broadband_support(
    response: ChannelResponse, broadband_range: tuple[float, float]
) -> ChannelBroadbandSupport | None:
    frequencies = tuple(
        frequency
        for frequency in response.frequencies_hz
        if broadband_range[0] <= frequency <= broadband_range[1]
    )
    if not frequencies:
        return None
    return ChannelBroadbandSupport(
        lowest_frequency_hz=frequencies[0],
        highest_frequency_hz=frequencies[-1],
        frequency_count=len(frequencies),
    )


def _common_broadband_support(
    points: Sequence[ChannelPointInput], broadband_range: tuple[float, float]
) -> tuple[ChannelBroadbandSupport | None, ReasonCode | None]:
    supports = tuple(
        _broadband_support(response, broadband_range)
        for point in points
        for response in point.responses
    )
    if not supports or any(item is None for item in supports):
        return None, ReasonCode.INSUFFICIENT_COVERAGE
    common = supports[0]
    if any(item != common for item in supports[1:]):
        return None, ReasonCode.FREQUENCY_AXIS_MISMATCH
    return common, None


def _support_reasons(
    receiver_set: ReceiverSet,
    points: Sequence[ChannelPointInput],
    group: ChannelGroup,
    broadband_range: tuple[float, float],
) -> tuple[ChannelBroadbandSupport | None, tuple[ReasonCode, ...]]:
    support, reason = _common_broadband_support(points, broadband_range)
    if reason is not None:
        return None, (reason,)
    return support, _point_unavailability_reasons(
        receiver_set, points, group, broadband_range
    )


def _metric_aggregate(
    points: Sequence[ChannelPointMatch], name: str
) -> ChannelMetricAggregate | None:
    measured = [point for point in points if point.state == MetricState.MEASURED]
    if not measured:
        return None
    total_weight = sum(point.importance for point in measured)
    if total_weight <= 0.0:
        raise ValueError("可估聲道匹配點的權重和必須大於零")
    values = [(point, abs(float(getattr(point, name)))) for point in measured]
    worst, worst_value = max(values, key=lambda item: (item[1], item[0].receiver_id))
    return ChannelMetricAggregate(
        weighted_mean_absolute_difference=sum(
            point.importance * value for point, value in values
        )
        / total_weight,
        worst_absolute_difference=worst_value,
        worst_receiver_id=worst.receiver_id,
    )


def _aggregates(
    results: Sequence[ChannelPointMatch], group: ChannelGroup
) -> tuple[ChannelComparisonAggregate, ...]:
    aggregates: list[ChannelComparisonAggregate] = []
    for comparison in group.comparisons:
        selected = tuple(
            item
            for item in results
            if item.left_role == comparison.left_role
            and item.right_role == comparison.right_role
        )
        measured = tuple(
            item for item in selected if item.state == MetricState.MEASURED
        )
        unavailable = tuple(
            item for item in selected if item.state == MetricState.UNAVAILABLE
        )
        aggregates.append(
            ChannelComparisonAggregate(
                left_role=comparison.left_role,
                right_role=comparison.right_role,
                assessed_receiver_ids=tuple(item.receiver_id for item in measured),
                unavailable_receiver_ids=tuple(
                    item.receiver_id for item in unavailable
                ),
                tilt_difference=_metric_aggregate(
                    selected, "tilt_difference_db_per_octave"
                ),
                ripple_rms_difference=_metric_aggregate(
                    selected, "ripple_rms_difference_db"
                ),
                broadband_level_difference=_metric_aggregate(
                    selected, "broadband_level_difference_db"
                ),
                direct_time_difference=_metric_aggregate(
                    selected, "direct_time_difference_ms"
                ),
            )
        )
    return tuple(aggregates)


def _raw_quantities(payload: ChannelMatchingPayload) -> tuple[RawQuantity, ...]:
    # 型別標成契約那一格的受控單位，不要讓它退化成任意字串（標錯單位就傳得進去了）。
    units: dict[str, Unit] = {
        "tilt_difference": "dB/oct",
        "ripple_rms_difference": "dB",
        "broadband_level_difference": "dB",
        "direct_time_difference": "ms",
    }
    quantities = [
        RawQuantity(
            name="assessed_point_count",
            value=float(
                sum(len(item.assessed_receiver_ids) for item in payload.aggregates)
            ),
            unit="1",
        )
    ]
    for index, aggregate in enumerate(payload.aggregates):
        for name, unit in units.items():
            metric = getattr(aggregate, name)
            if metric is None:
                continue
            prefix = f"comparison.{index}.{name}"
            quantities.extend(
                (
                    RawQuantity(
                        name=prefix + ".weighted_mean_absolute_difference",
                        value=metric.weighted_mean_absolute_difference,
                        unit=unit,
                    ),
                    RawQuantity(
                        name=prefix + ".worst_absolute_difference",
                        value=metric.worst_absolute_difference,
                        unit=unit,
                    ),
                )
            )
    return tuple(quantities)


def _payload(
    receiver_set: ReceiverSet,
    points: Sequence[ChannelPointInput],
    candidate_id: str,
    timbre_fingerprint: str,
    listening_fingerprint: str,
    group: ChannelGroup,
    settings: _Settings,
    results: tuple[ChannelPointMatch, ...],
    broadband_support: ChannelBroadbandSupport,
) -> ChannelMatchingPayload:
    by_receiver = {point.receiver_id: point for point in points}
    return ChannelMatchingPayload(
        category="channel_matching",
        candidate_id=candidate_id,
        receiver_set_fingerprint=receiver_set.fingerprint,
        timbre_settings_fingerprint=timbre_fingerprint,
        listening_area_settings_fingerprint=listening_fingerprint,
        channel_group_fingerprint=group.fingerprint,
        channels=tuple(
            ChannelIdentity(role=item.role, speaker_id=item.speaker_id)
            for item in group.channels
        ),
        comparisons=tuple(
            ChannelComparisonPair(left_role=item.left_role, right_role=item.right_role)
            for item in group.comparisons
        ),
        point_sources=tuple(
            ChannelPointSources(
                receiver_id=receiver.receiver_id,
                channels=tuple(
                    _source(response)
                    for response in by_receiver[receiver.receiver_id].responses
                ),
            )
            for receiver in _measured_points(receiver_set)
        ),
        point_results=results,
        aggregates=_aggregates(results, group),
        broadband_support=broadband_support,
        direct_time_cost_enabled=settings.direct_time_cost_enabled,
    )


def _measured_evaluation(
    payload: ChannelMatchingPayload,
    receiver_set: ReceiverSet,
    points: Sequence[ChannelPointInput],
    candidate_id: str,
    group: ChannelGroup,
    fingerprint: str,
    scene_fingerprint: str,
    baseline: bool,
) -> CategoryEvaluation:
    return CategoryEvaluation(
        schema_version=CONTRACT_SCHEMA_VERSION,
        candidate_id=candidate_id,
        scene_fingerprint=scene_fingerprint,
        placement=merge_placements(
            response.timbre_evaluation.placement
            for point in points
            for response in point.responses
        ),
        category=QualityCategory.CHANNEL_MATCHING,
        state=EvaluationState.MEASURED,
        payload=payload,
        raw_quantities=_raw_quantities(payload),
        category_cost=None,
        flags=_flags(points, baseline),
        reason_codes=(),
        evaluator_version=CHANNEL_MATCHING_EVALUATOR_VERSION,
        settings_fingerprint=fingerprint,
        provenance=_provenance(candidate_id, receiver_set, group),
    )


def _evaluate_measured(
    receiver_set: ReceiverSet,
    points: Sequence[ChannelPointInput],
    candidate_id: str,
    timbre_fingerprint: str,
    listening_fingerprint: str,
    group: ChannelGroup,
    settings: _Settings,
    fingerprint: str,
    scene_fingerprint: str,
    support: ChannelBroadbandSupport,
    sound_speed_m_s: float,
) -> CategoryEvaluation:
    results = _point_results(receiver_set, points, group, settings, sound_speed_m_s)
    payload = _payload(
        receiver_set,
        points,
        candidate_id,
        timbre_fingerprint,
        listening_fingerprint,
        group,
        settings,
        results,
        support,
    )
    return _measured_evaluation(
        payload,
        receiver_set,
        points,
        candidate_id,
        group,
        fingerprint,
        scene_fingerprint,
        settings.any_baseline,
    )


def evaluate_channel_matching(
    receiver_set: ReceiverSet,
    points: Sequence[ChannelPointInput],
    *,
    candidate_id: str,
    scene_fingerprint: str,
    timbre_settings_fingerprint: str,
    listening_area_settings_fingerprint: str,
    channel_group: ChannelGroup,
    purpose: str,
    quality_targets_path: str | Path,
    sound_speed_m_s: float,
) -> CategoryEvaluation:
    """量音色、寬頻音量與直達時間的聲道差；任一該量點不可估就整類拒算。

    身分、場景與擺位核對每個點的每支聲道回應（含沒被任何比較對用到的聲道）。
    """
    if not math.isfinite(sound_speed_m_s) or sound_speed_m_s <= 0.0:
        raise ValueError("sound_speed_m_s 必須是有限正數")
    if not channel_group.comparisons:
        raise ValueError("聲道匹配至少要一組比較對；單聲道的聲道組只給聲道音色彙總用")
    settings = _load_settings(quality_targets_path, purpose)
    fingerprint = _settings_fingerprint(
        settings,
        receiver_set,
        points,
        channel_group,
        timbre_settings_fingerprint,
        listening_area_settings_fingerprint,
    )
    reasons = _identity_reasons(
        receiver_set,
        points,
        candidate_id,
        timbre_settings_fingerprint,
        listening_area_settings_fingerprint,
        channel_group,
        scene_fingerprint,
    )
    support = None
    if not reasons:
        support, reasons = _support_reasons(
            receiver_set, points, channel_group, settings.broadband_range_hz
        )
    if reasons or support is None:
        return _unavailable(
            receiver_set,
            points,
            candidate_id,
            channel_group,
            fingerprint,
            scene_fingerprint,
            reasons or (ReasonCode.INSUFFICIENT_COVERAGE,),
            settings.any_baseline,
        )
    return _evaluate_measured(
        receiver_set,
        points,
        candidate_id,
        timbre_settings_fingerprint,
        listening_area_settings_fingerprint,
        channel_group,
        settings,
        fingerprint,
        scene_fingerprint,
        support,
        sound_speed_m_s,
    )
