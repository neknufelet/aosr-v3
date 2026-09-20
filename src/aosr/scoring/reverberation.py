"""殘響第一層評估器（票 #348 第一段）：逐帶只量、不打分。"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import Final, Literal

from aosr.physics.report_io import BandRow, ReportOutput
from aosr.scoring.contract import (
    CONTRACT_SCHEMA_VERSION,
    AdjacentBandChange,
    CategoryEvaluation,
    EvaluationState,
    Flag,
    InputProvenance,
    MetricState,
    ModelValidationStatus,
    QualityCategory,
    RawQuantity,
    ReasonCode,
    ReverberationBand,
    ReverberationMetric,
    ReverberationPayload,
    SchroederPosition,
)


REVERBERATION_EVALUATOR_VERSION: Final[str] = "aosr.scoring.reverberation.v1"
# 這是物理層目前輸出的上游散文，不是機器契約；後續票應改成物理層提供原因代碼。
_UPSTREAM_DECAY_RANGE_PROSE_MARKER: Final[str] = "未達下緣"


def _band_range(center_hz: float) -> tuple[float, float]:
    """算完整八度帶；正式清單目前只到 4 kHz，8 kHz 要等清單擴充才會由引擎產出。"""
    root_two = math.sqrt(2.0)
    return center_hz / root_two, center_hz * root_two


def _model_validation_status(
    report: ReportOutput, band_range_hz: tuple[float, float]
) -> ModelValidationStatus:
    """能力表的狀態只在它宣告的頻率範圍內成立，不拿來抹掉報表資料。"""
    coverage = report.capability.frequency_hz
    if not coverage:
        return ModelValidationStatus.UNCHECKED
    lower, upper = coverage
    if lower > band_range_hz[0] or upper < band_range_hz[1]:
        return ModelValidationStatus.UNCHECKED
    return ModelValidationStatus(report.capability.status)


def _reason_code(reason: str) -> ReasonCode:
    if _UPSTREAM_DECAY_RANGE_PROSE_MARKER in reason:
        return ReasonCode.INSUFFICIENT_DECAY_RANGE
    return ReasonCode.OTHER_ERROR


def _missing_metric(
    *,
    unit: Literal["s", "1"],
    state: MetricState,
    reason_codes: Sequence[ReasonCode],
    reason: str,
) -> ReverberationMetric:
    return ReverberationMetric(
        value=None,
        unit=unit,
        state=state,
        reason_codes=tuple(dict.fromkeys(reason_codes)),
        reason=reason,
    )


def _reported_metric(
    *,
    name: str,
    value: float | None,
    report_reason: str | None,
) -> ReverberationMetric:
    if value is None:
        reason = report_reason or f"{name} 報表沒有值也沒有原因"
        return _missing_metric(
            unit="s",
            state=MetricState.UNAVAILABLE,
            reason_codes=(_reason_code(reason),),
            reason=reason,
        )
    if value <= 0.0:
        return _missing_metric(
            unit="s",
            state=MetricState.UNAVAILABLE,
            reason_codes=(ReasonCode.NON_POSITIVE_VALUE,),
            reason=f"{name} 必須為正，報表值為 {value:g} s",
        )
    return ReverberationMetric(
        value=value,
        unit="s",
        state=MetricState.MEASURED,
        reason_codes=(),
        reason=None,
    )


def _fitting_difference(
    t20: ReverberationMetric, t30: ReverberationMetric
) -> ReverberationMetric:
    if t20.value is not None and t30.value is not None:
        return ReverberationMetric(
            value=t30.value / t20.value,
            unit="1",
            state=MetricState.MEASURED,
            reason_codes=(),
            reason=None,
        )
    missing = tuple(name for name, metric in (("T20", t20), ("T30", t30)) if metric.value is None)
    codes = tuple(code for metric in (t20, t30) for code in metric.reason_codes)
    return _missing_metric(
        unit="1",
        state=MetricState.NOT_COMPUTABLE,
        reason_codes=codes,
        reason=f"{'、'.join(missing)} 不可估，無法計算 T30/T20",
    )


def _schroeder_position(band_range_hz: tuple[float, float], f_s_hz: float) -> SchroederPosition:
    lower, upper = band_range_hz
    if upper < f_s_hz:
        return SchroederPosition.BELOW
    if lower > f_s_hz:
        return SchroederPosition.ABOVE
    return SchroederPosition.CROSSING


def _evaluate_band(report: ReportOutput, band: BandRow) -> ReverberationBand:
    """一列就是該帶的資料；能力表範圍另行決定模型驗證狀態，不充當資料覆蓋。"""
    band_range = _band_range(band.center_frequency_hz)
    t20 = _reported_metric(
        name="T20",
        value=band.t20_s,
        report_reason=band.t20_unavailable_reason,
    )
    t30 = _reported_metric(
        name="T30",
        value=band.t30_s,
        report_reason=band.t30_unavailable_reason,
    )
    return ReverberationBand(
        center_frequency_hz=band.center_frequency_hz,
        band_range_hz=band_range,
        schroeder_position=_schroeder_position(band_range, band.f_s_hz),
        model_validation_status=_model_validation_status(report, band_range),
        t20=t20,
        t30=t30,
        fitting_difference=_fitting_difference(t20, t30),
    )


def _adjacent_change(
    lower: ReverberationBand,
    upper: ReverberationBand,
    logarithm_base: float,
) -> AdjacentBandChange:
    if lower.t20.value is not None and upper.t20.value is not None:
        signed = math.log(upper.t20.value / lower.t20.value, logarithm_base)
        return AdjacentBandChange(
            lower_center_frequency_hz=lower.center_frequency_hz,
            upper_center_frequency_hz=upper.center_frequency_hz,
            signed_log_ratio=signed,
            state=MetricState.MEASURED,
            reason_codes=(),
            reason=None,
        )
    codes = tuple(code for band in (lower, upper) for code in band.t20.reason_codes)
    return AdjacentBandChange(
        lower_center_frequency_hz=lower.center_frequency_hz,
        upper_center_frequency_hz=upper.center_frequency_hz,
        signed_log_ratio=None,
        state=MetricState.NOT_COMPUTABLE,
        reason_codes=tuple(dict.fromkeys(codes)),
        reason="相鄰帶任一端 T20 不可估，無法計算相對變化",
    )


def _raw_quantities(payload: ReverberationPayload) -> tuple[RawQuantity, ...]:
    raw = [RawQuantity(name="band_count", value=float(len(payload.bands)), unit="1")]
    for band in payload.bands:
        label = f"{band.center_frequency_hz:g}Hz"
        for name, metric in (
            ("t20_s", band.t20),
            ("t30_s", band.t30),
            ("fitting_difference", band.fitting_difference),
        ):
            if metric.value is not None:
                raw.append(RawQuantity(name=f"{label}.{name}", value=metric.value, unit=metric.unit))
    for change in payload.adjacent_band_changes:
        if change.signed_log_ratio is not None:
            name = f"{change.lower_center_frequency_hz:g}-{change.upper_center_frequency_hz:g}Hz"
            raw.append(RawQuantity(name=f"{name}.signed_log_ratio", value=change.signed_log_ratio, unit="1"))
    return tuple(raw)


def _settings_fingerprint(logarithm_base: float) -> str:
    identity = f"adjacent_t20_logarithm_base={logarithm_base.hex()}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def evaluate_reverberation(
    report: ReportOutput,
    *,
    candidate_id: str,
    provenance: InputProvenance,
    logarithm_base: float,
) -> CategoryEvaluation:
    """把三路報表逐帶衰減整理成 measured 契約；不讀目標、不算類代價。"""
    bands = tuple(_evaluate_band(report, band) for band in report.bands)
    changes = tuple(
        _adjacent_change(lower, upper, logarithm_base)
        for lower, upper in zip(bands[:-1], bands[1:], strict=True)
    )
    payload = ReverberationPayload(
        category="reverberation",
        bands=bands,
        adjacent_band_changes=changes,
        logarithm_base=logarithm_base,
    )
    flags: list[Flag] = []
    if any(band.schroeder_position == SchroederPosition.CROSSING for band in bands):
        flags.append(Flag.CROSSOVER_BAND)
    if any(
        band.model_validation_status != ModelValidationStatus.VALIDATED
        for band in bands
    ):
        flags.append(Flag.UNVALIDATED)
    if any(ReasonCode.INSUFFICIENT_COVERAGE in band.t20.reason_codes for band in bands):
        flags.append(Flag.DATA_COVERAGE_SHORT)
    return CategoryEvaluation(
        schema_version=CONTRACT_SCHEMA_VERSION,
        candidate_id=candidate_id,
        category=QualityCategory.REVERBERATION,
        state=EvaluationState.MEASURED,
        payload=payload,
        raw_quantities=_raw_quantities(payload),
        category_cost=None,
        flags=tuple(flags),
        reason_codes=(),
        evaluator_version=REVERBERATION_EVALUATOR_VERSION,
        settings_fingerprint=_settings_fingerprint(logarithm_base),
        provenance=provenance,
    )
