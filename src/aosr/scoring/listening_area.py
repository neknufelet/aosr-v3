"""聆聽區穩定性疊層（票 #349 第一段）：疊單點音色結果，只量、不算代價。"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Annotated, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from aosr.scoring.contract import (
    CONTRACT_SCHEMA_VERSION,
    CategoryEvaluation,
    DeviationAggregate,
    DeviationEndpoint,
    EvaluationState,
    Feature,
    Flag,
    InputProvenance,
    ListeningAreaStabilityPayload,
    PeakDipOccurrence,
    QualityCategory,
    RawQuantity,
    ReceiverPointProvenance,
    ReasonCode,
    StabilityComparison,
    TargetDeviationPositionSpread,
    TimbrePayload,
    WorstDeviation,
)
from aosr.scoring.receiver_set import ReceiverPoint, ReceiverRole, ReceiverSet


LISTENING_AREA_EVALUATOR_VERSION: Final[str] = "aosr.scoring.listening_area.v1"
FROZEN = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
Distance = Callable[["ReceiverPointResult", "ReceiverPointResult"], float]
RawUnit = Literal["dB", "dB/oct", "Hz", "oct", "1"]


class ReceiverPointResult(BaseModel):
    """一個接收點的既有音色評估與同點寬頻平均總能量。"""

    model_config = FROZEN

    receiver_id: str = Field(min_length=1)
    receiver_set_fingerprint: str = Field(min_length=1)
    timbre_evaluation: CategoryEvaluation
    broadband_mean_total_energy_db: float


class ListeningAreaSettings(BaseModel):
    """本層會改變量測答案的設定；正規化內容的 SHA-256 是跨層比較身分。"""

    model_config = FROZEN

    feature_match_tolerance_hz: Annotated[float, Field(ge=0.0)]

    @property
    def fingerprint(self) -> str:
        canonical = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class _Sample:
    value: float
    receiver: DeviationEndpoint
    reference: DeviationEndpoint
    weight: float


@dataclass(frozen=True)
class _DiagnosticSpread:
    curve: tuple[TargetDeviationPositionSpread, ...]
    common_frequency_count: int
    discarded_frequency_value_count: int


class _CannotAggregate(Exception):
    def __init__(self, reason: ReasonCode) -> None:
        super().__init__(reason.value)
        self.reason = reason


def _endpoint(point: ReceiverPoint) -> DeviationEndpoint:
    return DeviationEndpoint(
        receiver_id=point.receiver_id,
        direction_relative_to_primary=point.direction_relative_to_primary,
        importance=point.importance,
    )


def _timbre_payload(result: ReceiverPointResult) -> TimbrePayload:
    payload = result.timbre_evaluation.payload
    if not isinstance(payload, TimbrePayload):
        raise _CannotAggregate(ReasonCode.TIMBRE_NOT_MEASURED)
    return payload


def _samples(
    receiver_set: ReceiverSet,
    results: dict[str, ReceiverPointResult],
    distance: Distance,
) -> tuple[tuple[_Sample, ...], tuple[_Sample, ...]]:
    """建立兩組偏差；周圍配對權重取兩端重要性相乘是實作選擇，拍板未規定。"""
    primary = receiver_set.primary
    surrounding = sorted(
        (point for point in receiver_set.points if point.role == ReceiverRole.SURROUNDING),
        key=lambda point: point.receiver_id,
    )
    primary_samples = tuple(
        _Sample(
            value=distance(results[point.receiver_id], results[primary.receiver_id]),
            receiver=_endpoint(point),
            reference=_endpoint(primary),
            weight=point.importance,
        )
        for point in surrounding
    )
    peer_samples = tuple(
        _Sample(
            value=distance(results[right.receiver_id], results[left.receiver_id]),
            receiver=_endpoint(right),
            reference=_endpoint(left),
            weight=left.importance * right.importance,
        )
        for left, right in combinations(surrounding, 2)
    )
    return primary_samples, peer_samples


def _aggregate(samples: Sequence[_Sample]) -> DeviationAggregate:
    if not samples:
        raise _CannotAggregate(ReasonCode.INSUFFICIENT_COVERAGE)
    total_weight = sum(sample.weight for sample in samples)
    if total_weight <= 0.0:
        raise _CannotAggregate(ReasonCode.ZERO_TOTAL_IMPORTANCE)
    worst = max(
        samples,
        key=lambda sample: (
            sample.value,
            sample.receiver.receiver_id,
            sample.reference.receiver_id,
        ),
    )
    return DeviationAggregate(
        weighted_mean_deviation=sum(sample.value * sample.weight for sample in samples)
        / total_weight,
        worst_deviation=WorstDeviation(
            value=worst.value,
            receiver=worst.receiver,
            reference=worst.reference,
        ),
    )


def _comparison(
    receiver_set: ReceiverSet,
    results: dict[str, ReceiverPointResult],
    distance: Distance,
) -> StabilityComparison:
    primary, peers = _samples(receiver_set, results, distance)
    return StabilityComparison(
        primary_to_surrounding=_aggregate(primary),
        surrounding_to_surrounding=_aggregate(peers) if peers else None,
        surrounding_to_surrounding_reason=(
            None if peers else ReasonCode.NO_SURROUNDING_PAIRS
        ),
    )


def _scalar_distance(getter: Callable[[TimbrePayload], float]) -> Distance:
    def distance(left: ReceiverPointResult, right: ReceiverPointResult) -> float:
        return abs(getter(_timbre_payload(left)) - getter(_timbre_payload(right)))

    return distance


def _level_distance(left: ReceiverPointResult, right: ReceiverPointResult) -> float:
    return abs(left.broadband_mean_total_energy_db - right.broadband_mean_total_energy_db)


def _matched_pairs(
    left: Sequence[Feature], right: Sequence[Feature], tolerance_hz: float
) -> tuple[tuple[int, int], ...]:
    candidates = sorted(
        (
            (
                abs(a.center_frequency_hz - b.center_frequency_hz),
                left_index,
                right_index,
            )
            for left_index, a in enumerate(left)
            for right_index, b in enumerate(right)
            if a.kind == b.kind
            and abs(a.center_frequency_hz - b.center_frequency_hz) <= tolerance_hz
        )
    )
    used_left: set[int] = set()
    used_right: set[int] = set()
    matches: list[tuple[int, int]] = []
    for _, left_index, right_index in candidates:
        if left_index in used_left or right_index in used_right:
            continue
        used_left.add(left_index)
        used_right.add(right_index)
        matches.append((left_index, right_index))
    return tuple(matches)


def _feature_distance(tolerance_hz: float) -> Distance:
    def distance(left: ReceiverPointResult, right: ReceiverPointResult) -> float:
        left_features = _timbre_payload(left).features
        right_features = _timbre_payload(right).features
        matches = _matched_pairs(left_features, right_features, tolerance_hz)
        return float(len(left_features) + len(right_features) - 2 * len(matches))

    return distance


def _peak_dip_occurrences(
    receiver_set: ReceiverSet,
    results: dict[str, ReceiverPointResult],
    tolerance_hz: float,
) -> tuple[PeakDipOccurrence, ...]:
    """只從主位峰谷往外找；第一版會漏掉周圍彼此共有、但主位沒有的峰谷。"""
    primary_id = receiver_set.primary.receiver_id
    primary_features = _timbre_payload(results[primary_id]).features
    found: list[list[str]] = [[primary_id] for _ in primary_features]
    comparison_points = sorted(
        (point for point in receiver_set.points if point.role == ReceiverRole.SURROUNDING),
        key=lambda point: point.receiver_id,
    )
    for point in comparison_points:
        point_features = _timbre_payload(results[point.receiver_id]).features
        for primary_index, _ in _matched_pairs(primary_features, point_features, tolerance_hz):
            found[primary_index].append(point.receiver_id)
    return tuple(
        PeakDipOccurrence(
            kind=feature.kind,
            primary_center_frequency_hz=feature.center_frequency_hz,
            receiver_ids=tuple(found[index]),
            occurrence_count=len(found[index]),
        )
        for index, feature in enumerate(primary_features)
    )


def _diagnostic_curve(
    receiver_set: ReceiverSet, results: dict[str, ReceiverPointResult]
) -> _DiagnosticSpread:
    comparison_ids = [
        point.receiver_id
        for point in receiver_set.points
        if point.role in (ReceiverRole.PRIMARY, ReceiverRole.SURROUNDING)
    ]
    curves = [dict(_timbre_payload(results[receiver_id]).deviation_curve) for receiver_id in comparison_ids]
    common_frequencies = set(curves[0])
    for curve in curves[1:]:
        common_frequencies &= set(curve)
    spread = tuple(
        TargetDeviationPositionSpread(
            frequency_hz=frequency,
            spread_db=max(curve[frequency] for curve in curves)
            - min(curve[frequency] for curve in curves),
        )
        for frequency in sorted(common_frequencies)
    )
    discarded = sum(len(set(curve) - common_frequencies) for curve in curves)
    return _DiagnosticSpread(
        curve=spread,
        common_frequency_count=len(common_frequencies),
        discarded_frequency_value_count=discarded,
    )


def _measured_points(receiver_set: ReceiverSet) -> tuple[ReceiverPoint, ...]:
    return tuple(
        point
        for point in receiver_set.points
        if point.role in (ReceiverRole.PRIMARY, ReceiverRole.SURROUNDING)
    )


def _relevant_results(
    receiver_set: ReceiverSet, results: Sequence[ReceiverPointResult]
) -> tuple[ReceiverPointResult, ...]:
    ignored_ids = {
        point.receiver_id for point in receiver_set.points if point.role == ReceiverRole.OTHER_SEAT
    }
    return tuple(result for result in results if result.receiver_id not in ignored_ids)


def _identity_reasons(
    receiver_set: ReceiverSet,
    results: Sequence[ReceiverPointResult],
    candidate_id: str,
    speaker_id: str,
    settings_fingerprint: str,
) -> tuple[ReasonCode, ...]:
    reasons: list[ReasonCode] = []
    expected_ids = {point.receiver_id for point in _measured_points(receiver_set)}
    checked_results = _relevant_results(receiver_set, results)
    actual_ids = [result.receiver_id for result in checked_results]
    if expected_ids - set(actual_ids):
        reasons.append(ReasonCode.MISSING_POINTS)
    if set(actual_ids) - expected_ids:
        reasons.append(ReasonCode.RECEIVER_ID_MISMATCH)
    if len(actual_ids) != len(set(actual_ids)):
        reasons.append(ReasonCode.RECEIVER_ID_MISMATCH)
    for result in checked_results:
        evaluation = result.timbre_evaluation
        if evaluation.candidate_id != candidate_id:
            reasons.append(ReasonCode.CANDIDATE_ID_MISMATCH)
        if evaluation.provenance.speaker_id != speaker_id:
            reasons.append(ReasonCode.SPEAKER_ID_MISMATCH)
        if result.receiver_set_fingerprint != receiver_set.fingerprint:
            reasons.append(ReasonCode.RECEIVER_SET_FINGERPRINT_MISMATCH)
        if evaluation.settings_fingerprint != settings_fingerprint:
            reasons.append(ReasonCode.SETTINGS_FINGERPRINT_MISMATCH)
        if result.receiver_id != evaluation.provenance.receiver_id:
            reasons.append(ReasonCode.RECEIVER_ID_MISMATCH)
        if evaluation.state != EvaluationState.MEASURED or not isinstance(
            evaluation.payload, TimbrePayload
        ):
            reasons.append(ReasonCode.TIMBRE_NOT_MEASURED)
    return tuple(dict.fromkeys(reasons))


def _provenance(
    receiver_set: ReceiverSet,
    results: Sequence[ReceiverPointResult],
    candidate_id: str,
    speaker_id: str,
) -> InputProvenance:
    primary_id = receiver_set.primary.receiver_id
    source = next(
        (result.timbre_evaluation.provenance for result in results if result.receiver_id == primary_id),
        None,
    )
    return InputProvenance(
        report_id=source.report_id if source is not None else f"listening-area:{candidate_id}",
        engine_commit=source.engine_commit if source is not None else "not-available",
        speaker_id=speaker_id,
        receiver_id=primary_id,
    )


def _flags(
    results: Sequence[ReceiverPointResult], extra: Sequence[Flag] = ()
) -> tuple[Flag, ...]:
    return tuple(
        dict.fromkeys(
            [
                *(flag for result in results for flag in result.timbre_evaluation.flags),
                *extra,
            ]
        )
    )


def _unavailable(
    receiver_set: ReceiverSet,
    results: Sequence[ReceiverPointResult],
    candidate_id: str,
    speaker_id: str,
    listening_area_settings_fingerprint: str,
    reasons: tuple[ReasonCode, ...],
) -> CategoryEvaluation:
    return CategoryEvaluation(
        schema_version=CONTRACT_SCHEMA_VERSION,
        candidate_id=candidate_id,
        category=QualityCategory.LISTENING_AREA_STABILITY,
        state=EvaluationState.UNAVAILABLE,
        payload=None,
        raw_quantities=(),
        category_cost=None,
        flags=_flags(_relevant_results(receiver_set, results)),
        reason_codes=reasons,
        evaluator_version=LISTENING_AREA_EVALUATOR_VERSION,
        settings_fingerprint=listening_area_settings_fingerprint,
        provenance=_provenance(receiver_set, results, candidate_id, speaker_id),
    )


def _raw_quantities(payload: ListeningAreaStabilityPayload) -> tuple[RawQuantity, ...]:
    metrics: tuple[tuple[str, StabilityComparison, RawUnit], ...] = (
        ("tilt", payload.tilt_stability, "dB/oct"),
        ("ripple_rms", payload.ripple_rms_stability, "dB"),
        ("overall_level", payload.overall_level_stability, "dB"),
        ("peak_dip_consistency", payload.peak_dip_consistency, "1"),
    )
    quantities: list[RawQuantity] = []
    for name, metric, unit in metrics:
        for comparison_name, summary in (
            ("primary_to_surrounding", metric.primary_to_surrounding),
            ("surrounding_to_surrounding", metric.surrounding_to_surrounding),
        ):
            if summary is None:
                continue
            quantities.extend(
                (
                    RawQuantity(
                        name=f"{name}.{comparison_name}.weighted_mean_deviation",
                        value=summary.weighted_mean_deviation,
                        unit=unit,
                    ),
                    RawQuantity(
                        name=f"{name}.{comparison_name}.worst_deviation",
                        value=summary.worst_deviation.value,
                        unit=unit,
                    ),
                )
            )
    return tuple(quantities)


def _payload(
    receiver_set: ReceiverSet,
    results: dict[str, ReceiverPointResult],
    candidate_id: str,
    speaker_id: str,
    settings_fingerprint: str,
    tolerance_hz: float,
) -> ListeningAreaStabilityPayload:
    diagnostic = _diagnostic_curve(receiver_set, results)
    return ListeningAreaStabilityPayload(
        category="listening_area_stability",
        candidate_id=candidate_id,
        speaker_id=speaker_id,
        receiver_set_fingerprint=receiver_set.fingerprint,
        timbre_settings_fingerprint=settings_fingerprint,
        settings_fingerprint=ListeningAreaSettings(
            feature_match_tolerance_hz=tolerance_hz
        ).fingerprint,
        point_provenance=tuple(
            ReceiverPointProvenance(
                receiver_id=point.receiver_id,
                report_id=results[point.receiver_id].timbre_evaluation.provenance.report_id,
                evaluator_version=results[point.receiver_id].timbre_evaluation.evaluator_version,
                settings_fingerprint=results[
                    point.receiver_id
                ].timbre_evaluation.settings_fingerprint,
            )
            for point in _measured_points(receiver_set)
        ),
        tilt_stability=_comparison(
            receiver_set, results, _scalar_distance(lambda payload: payload.tilt_db_per_octave)
        ),
        ripple_rms_stability=_comparison(
            receiver_set, results, _scalar_distance(lambda payload: payload.residual_rms_db)
        ),
        overall_level_stability=_comparison(receiver_set, results, _level_distance),
        peak_dip_consistency=_comparison(
            receiver_set, results, _feature_distance(tolerance_hz)
        ),
        peak_dip_occurrences=_peak_dip_occurrences(receiver_set, results, tolerance_hz),
        target_deviation_position_spread_curve_db=diagnostic.curve,
        target_deviation_common_frequency_count=diagnostic.common_frequency_count,
        target_deviation_discarded_frequency_value_count=(
            diagnostic.discarded_frequency_value_count
        ),
    )


def _validated_settings(
    candidate_id: str,
    speaker_id: str,
    timbre_settings_fingerprint: str,
    feature_match_tolerance_hz: float,
) -> ListeningAreaSettings:
    for name, value in (
        ("candidate_id", candidate_id),
        ("speaker_id", speaker_id),
        ("timbre_settings_fingerprint", timbre_settings_fingerprint),
    ):
        if not value.strip():
            raise ValueError(f"{name} 不可為空白")
    if not math.isfinite(feature_match_tolerance_hz) or feature_match_tolerance_hz < 0.0:
        raise ValueError("feature_match_tolerance_hz 必須是有限非負數")
    return ListeningAreaSettings(
        feature_match_tolerance_hz=feature_match_tolerance_hz
    )


def _measured_evaluation(
    receiver_set: ReceiverSet,
    results: Sequence[ReceiverPointResult],
    payload: ListeningAreaStabilityPayload,
    candidate_id: str,
    speaker_id: str,
    settings_fingerprint: str,
) -> CategoryEvaluation:
    frequency_flags = (
        (Flag.PARTIAL_FREQUENCY_OVERLAP,)
        if payload.target_deviation_discarded_frequency_value_count > 0
        else ()
    )
    return CategoryEvaluation(
        schema_version=CONTRACT_SCHEMA_VERSION,
        candidate_id=candidate_id,
        category=QualityCategory.LISTENING_AREA_STABILITY,
        state=EvaluationState.MEASURED,
        payload=payload,
        raw_quantities=_raw_quantities(payload),
        category_cost=None,
        flags=_flags(results, frequency_flags),
        reason_codes=(),
        evaluator_version=LISTENING_AREA_EVALUATOR_VERSION,
        settings_fingerprint=settings_fingerprint,
        provenance=_provenance(receiver_set, results, candidate_id, speaker_id),
    )


def evaluate_listening_area(
    receiver_set: ReceiverSet,
    point_results: Sequence[ReceiverPointResult],
    *,
    candidate_id: str,
    speaker_id: str,
    timbre_settings_fingerprint: str,
    feature_match_tolerance_hz: Annotated[float, Field(ge=0.0)],
) -> CategoryEvaluation:
    """完整且四格身分一致才疊；失敗回 unavailable，成功只回 measured。"""
    settings = _validated_settings(
        candidate_id,
        speaker_id,
        timbre_settings_fingerprint,
        feature_match_tolerance_hz,
    )
    reasons = _identity_reasons(
        receiver_set,
        point_results,
        candidate_id,
        speaker_id,
        timbre_settings_fingerprint,
    )
    if reasons:
        return _unavailable(
            receiver_set,
            point_results,
            candidate_id,
            speaker_id,
            settings.fingerprint,
            reasons,
        )
    relevant_results = _relevant_results(receiver_set, point_results)
    by_id = {result.receiver_id: result for result in relevant_results}
    try:
        payload = _payload(
            receiver_set,
            by_id,
            candidate_id,
            speaker_id,
            timbre_settings_fingerprint,
            feature_match_tolerance_hz,
        )
    except _CannotAggregate as exc:
        return _unavailable(
            receiver_set,
            point_results,
            candidate_id,
            speaker_id,
            settings.fingerprint,
            (exc.reason,),
        )
    return _measured_evaluation(
        receiver_set,
        relevant_results,
        payload,
        candidate_id,
        speaker_id,
        settings.fingerprint,
    )
