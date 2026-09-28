"""從已驗結果挑出結果頁資料；畫面不重算聲學量或門檻。"""
from __future__ import annotations

from datetime import date
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from aosr.config.quality_targets import QualityTargets, SettingEntry, TargetEntry, load_quality_targets
from aosr.reporting.compare import compare_results
from aosr.reporting.display import (
    BASELINE_NOTE, LISTENING_AREA_SCOPE_NOTE, LOW_FREQUENCY_DECAY_NOTE,
    REVERBERATION_ROOM_NOTE, SPATIAL_IMPRESSION_NOTE, level_db,
)
from aosr.reporting.result import (
    SchemeResult, Timings, evaluate_point_timbres, read_registry_settings,
    receiver_point_results,
)
from aosr.scoring.contract import (
    CategoryEvaluation, ListeningAreaChannelsPayload, QualityCategory, ReverberationPayload,
)
from aosr.scoring.listening_area import listening_area_pair_deviations
from aosr.scoring.reflections_contract import ReflectionsAndEchoPayload
from aosr.scoring.review_alert import FlutterReviewAlert, ListeningAreaReviewAlert, PeakDipReviewAlert


FROZEN = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class ViewModel(BaseModel):
    model_config = FROZEN


class FrequencyPoint(ViewModel):
    frequency_hz: float
    level_db: float | None
    frequency_text: str
    level_text: str


class FrequencyResponse(ViewModel):
    role: str
    speaker_id: str
    receiver_id: str
    receiver_role: str
    receiver_label: str
    points: tuple[FrequencyPoint, ...]


class CategoryView(ViewModel):
    category: str
    state: str
    state_label: str
    cost: float | None
    cost_text: str
    flags: tuple[str, ...]
    reason_codes: tuple[str, ...]
    evaluator_version: str | None
    note: str


class AlertView(ViewModel):
    kind: str
    category: str
    speaker_id: str | None
    role: str | None
    receiver_id: str | None
    reference_id: str | None
    fields: tuple[tuple[str, str], ...]
    excess_text: str | None
    baseline_note: str | None


class ReverberationBandView(ViewModel):
    center_frequency_hz: float
    center_text: str
    t20_text: str
    t30_text: str
    t20_state: str
    t30_state: str
    t20_reason_codes: tuple[str, ...]
    t30_reason_codes: tuple[str, ...]
    target_low_text: str | None
    target_high_text: str | None
    schroeder_position: str
    model_validation_status: str


class ReverberationView(ViewModel):
    role: str
    receiver_id: str
    note: str
    bands: tuple[ReverberationBandView, ...]


class ReflectionPathView(ViewModel):
    delay_text: str
    level_text: str
    azimuth_text: str
    elevation_text: str
    zone: str
    wall_sequence: tuple[str, ...]
    within_window: bool


class ReflectionView(ViewModel):
    role: str
    speaker_id: str
    receiver_id: str
    coverage: str
    validation: str
    state: str
    reason_codes: tuple[str, ...]
    flags: tuple[str, ...]
    paths: tuple[ReflectionPathView, ...]


class PairView(ViewModel):
    role: str
    speaker_id: str
    metric: str
    group: str
    receiver_id: str
    reference_id: str
    value: float
    value_text: str
    unit: str
    limit: float
    limit_text: str
    excess_text: str
    over_limit: bool
    baseline_note: str


class SummaryView(ViewModel):
    role: str
    speaker_id: str
    metric: str
    group: str
    weighted_mean_text: str
    worst_receiver_id: str
    worst_reference_id: str
    worst_value_text: str
    unit: str
    limit_text: str
    baseline_note: str


class ListeningAreaView(ViewModel):
    scope_note: str
    summaries: tuple[SummaryView, ...]
    pairs: tuple[PairView, ...]


class ResultView(ViewModel):
    scheme_id: str
    engine_commit: str
    run_date: date
    timings: Timings
    timing_texts: dict[str, str]
    labels: dict[str, str]
    frequency_responses: tuple[FrequencyResponse, ...]
    categories: tuple[CategoryView, ...]
    alerts: tuple[AlertView, ...]
    reverberation: ReverberationView
    reflections: tuple[ReflectionView, ...]
    listening_area: ListeningAreaView


def _text(value: float | None, unit: str = "") -> str:
    return "—" if value is None else f"{value:.4f}{(' ' + unit) if unit else ''}"


def _precise(value: float) -> str:
    """警戒差值保留足夠位數，避免剛超線被四捨五入成等於線。"""
    return f"{value:.12f}"


LABELS = {
    "left": "左聲道", "right": "右聲道", "primary": "主位",
    "surrounding": "周圍點", "other_seat": "其他座位",
    "primary_to_surrounding": "主位對周圍點",
    "surrounding_to_surrounding": "周圍點彼此",
    "tilt": "傾斜差", "ripple_rms": "起伏 RMS 差", "overall_level": "音量差",
    "timbre_balance": "音色平衡", "listening_area_stability": "聆聽區穩定性",
    "low_frequency_decay": "低頻拖尾", "reflections_and_echo": "反射與回聲",
    "reverberation": "殘響", "channel_matching": "聲道匹配",
    "spatial_impression": "空間感", "peak": "峰值警戒", "dip": "谷值警戒",
    "flutter": "牆間顫動警戒", "listening_area_worst_deviation": "聆聽區最差差距",
    "window_only_delay_screen": "只看時間窗",
    "geometry_material_conservative_screen": "幾何與材料的保守篩選",
    "measured": "已量", "costed": "已算代價", "unavailable": "不可估",
    "not_computable": "無法計算", "validated": "已驗證", "unvalidated": "尚未驗證",
    "experimental": "試驗中", "unsupported": "不支援", "unchecked": "未檢查",
    "complete": "完整", "not_provable": "無法證明完整", "missing": "缺資料",
    "front": "前方", "lateral": "側向", "rear": "後方", "vertical": "上下方",
    "below": "交界以下", "above": "交界以上", "crossing": "跨過交界",
    "baseline_settings": "使用暫定基線", "partial_frequency_overlap": "頻率範圍部分重疊",
    "listening_area_peer_group_missing": "周圍點彼此組缺資料",
    "no_directivity": "沒有指向資料",
}


def _frequency_responses(result: SchemeResult) -> tuple[FrequencyResponse, ...]:
    roles = {point.receiver_id: point.role.value for point in result.scheme.receiver_set.points}
    labels = {"primary": "主位", "surrounding": "周圍點", "other_seat": "其他座位"}
    responses: list[FrequencyResponse] = []
    for pair in result.pairs:
        if pair.report.points is None:
            raise ValueError(f"{pair.speaker_id} 到 {pair.receiver_id} 缺逐點頻響")
        responses.append(FrequencyResponse(
            role=pair.role, speaker_id=pair.speaker_id, receiver_id=pair.receiver_id,
            receiver_role=roles[pair.receiver_id], receiver_label=labels[roles[pair.receiver_id]],
            points=tuple(FrequencyPoint(
            frequency_hz=point.frequency_hz, level_db=level_db(point.total_energy),
            frequency_text=_text(point.frequency_hz, "Hz"),
            level_text=_text(level_db(point.total_energy), "dB"),
            ) for point in pair.report.points),
        ))
    return tuple(responses)


def _categories(result: SchemeResult, costs: dict[QualityCategory, float],
                ranked: dict[QualityCategory, CategoryEvaluation]) -> tuple[CategoryView, ...]:
    evaluations = {item.category: item for item in result.candidate.evaluations}
    evaluations.update(ranked)
    notes = {QualityCategory.LOW_FREQUENCY_DECAY: LOW_FREQUENCY_DECAY_NOTE,
             QualityCategory.SPATIAL_IMPRESSION: SPATIAL_IMPRESSION_NOTE}
    labels = {"measured": "已量", "costed": "已算代價", "unavailable": "不可估"}
    return tuple(CategoryView(
        category=category.value,
        state=item.state.value if item else "not_evaluated",
        state_label=labels[item.state.value] if item else "尚未評估",
        cost=costs.get(category) if category in costs else
             (item.category_cost.value if item and item.category_cost else None),
        cost_text=_text(costs.get(category) if category in costs else
                        (item.category_cost.value if item and item.category_cost else None)),
        flags=tuple(flag.value for flag in item.flags) if item else (),
        reason_codes=tuple(code.value for code in item.reason_codes) if item else (),
        evaluator_version=item.evaluator_version if item else None,
        note=notes.get(category, ""),
    ) for category in QualityCategory for item in (evaluations.get(category),))


def _alerts(alerts: tuple[PeakDipReviewAlert | FlutterReviewAlert |
                           ListeningAreaReviewAlert, ...]) -> tuple[AlertView, ...]:
    views: list[AlertView] = []
    for alert in alerts:
        data = alert.model_dump(mode="json", exclude={"note"})
        fields = tuple((key, _text(value) if isinstance(value, float) else str(value))
                       for key, value in data.items() if key not in {
                           "kind", "category", "speaker_id", "role", "receiver_id",
                           "reference_id", "deviation", "limit", "depth_db", "limit_db",
                       })
        excess: float | None = None
        if data["kind"] == "listening_area_worst_deviation":
            fields += (("差值", _precise(data["deviation"])),
                       ("暫定線", _precise(data["limit"])))
            excess = data["deviation"] - data["limit"]
        elif data["kind"] in {"peak", "dip"}:
            fields += (("峰谷量", _precise(data["depth_db"])),
                       ("警戒線", _precise(data["limit_db"])))
            excess = abs(data["depth_db"]) - data["limit_db"]
        views.append(AlertView(
            kind=data["kind"], category=data["category"],
            speaker_id=data.get("speaker_id"), role=data.get("role"),
            receiver_id=data.get("receiver_id"), reference_id=data.get("reference_id"),
            fields=fields, excess_text=_precise(excess) if excess is not None else None,
            baseline_note=BASELINE_NOTE if excess is not None else None,
        ))
    return tuple(views)


def _reverberation(result: SchemeResult, registry: QualityTargets) -> ReverberationView:
    evaluation = next(item for item in result.candidate.evaluations
                      if item.category is QualityCategory.REVERBERATION)
    payload = evaluation.payload
    assert isinstance(payload, ReverberationPayload)
    purpose = registry.purpose(result.scheme.purpose)
    center_entry = purpose.entry("reverberation.target_band_centers_hz")
    nominal_entry = purpose.entry("reverberation.target_t20_nominal_s_by_band")
    tolerance_entry = purpose.entry("reverberation.target_t20_tolerance_s_by_band")
    assert isinstance(center_entry, SettingEntry) and isinstance(nominal_entry, SettingEntry)
    assert isinstance(tolerance_entry, SettingEntry)
    centers, nominal, tolerance = center_entry.value, nominal_entry.value, tolerance_entry.value
    assert isinstance(centers, tuple) and isinstance(nominal, tuple)
    assert isinstance(tolerance, tuple)
    targets = {center: (value - span, value + span) for center, value, span in
               zip(centers, nominal, tolerance, strict=True)}
    bands = tuple(ReverberationBandView(
        center_frequency_hz=band.center_frequency_hz,
        center_text=_text(band.center_frequency_hz, "Hz"),
        t20_text=_text(band.t20.value, "秒"), t30_text=_text(band.t30.value, "秒"),
        t20_state=band.t20.state.value, t30_state=band.t30.state.value,
        t20_reason_codes=tuple(code.value for code in band.t20.reason_codes),
        t30_reason_codes=tuple(code.value for code in band.t30.reason_codes),
        target_low_text=_text(targets[band.center_frequency_hz][0], "秒")
        if band.center_frequency_hz in targets else None,
        target_high_text=_text(targets[band.center_frequency_hz][1], "秒")
        if band.center_frequency_hz in targets else None,
        schroeder_position=band.schroeder_position.value,
        model_validation_status=band.model_validation_status.value,
    ) for band in payload.bands)
    first = result.scheme.channel_group.channels[0].role
    return ReverberationView(role=first, receiver_id=result.scheme.receiver_set.primary.receiver_id,
                             note=REVERBERATION_ROOM_NOTE, bands=bands)


def _reflections(result: SchemeResult) -> tuple[ReflectionView, ...]:
    evaluation = next(item for item in result.candidate.evaluations
                      if item.category is QualityCategory.REFLECTIONS_AND_ECHO)
    payload = evaluation.payload
    assert isinstance(payload, ReflectionsAndEchoPayload)
    return tuple(ReflectionView(
        role=channel.role, speaker_id=channel.speaker_id, receiver_id=channel.receiver_id,
        coverage=channel.coverage, validation=channel.validation, state=channel.state.value,
        reason_codes=tuple(code.value for code in channel.reason_codes),
        flags=tuple(flag.value for flag in evaluation.flags),
        paths=tuple(ReflectionPathView(
            delay_text=_text(path.relative_direct_delay_s * 1000.0, "毫秒"),
            level_text=_text(path.broadband_level_db, "dB"),
            azimuth_text=_text(path.listening_azimuth_deg, "度"),
            elevation_text=_text(path.listening_elevation_deg, "度"),
            zone=path.zone.value, wall_sequence=path.wall_sequence,
            within_window=path.within_window,
        ) for path in channel.reflections),
    ) for channel in payload.channels if channel.is_primary)


_METRICS = (("tilt", "tilt_stability", "tilt_worst_deviation"),
            ("ripple_rms", "ripple_rms_stability", "ripple_rms_worst_deviation"),
            ("overall_level", "overall_level_stability", "overall_level_worst_deviation"))


def _listening_area(result: SchemeResult, path: Path, registry: QualityTargets) -> ListeningAreaView:
    evaluation = next(item for item in result.candidate.evaluations
                      if item.category is QualityCategory.LISTENING_AREA_STABILITY)
    payload = evaluation.payload
    assert isinstance(payload, ListeningAreaChannelsPayload)
    purpose = registry.purpose(result.scheme.purpose)
    settings = read_registry_settings(path, result.scheme.purpose)
    pair_map = {(pair.role, pair.receiver_id): pair for pair in result.pairs}
    timbres = evaluate_point_timbres(result, path)
    summaries: list[SummaryView] = []
    pairs: list[PairView] = []
    for channel in payload.channels:
        points = receiver_point_results(result.scheme, pair_map, timbres, channel.role)
        deviations = listening_area_pair_deviations(
            result.scheme.receiver_set, points, broadband_range_hz=settings.broadband_range_hz)
        for metric, field, target_key in _METRICS:
            target = purpose.entry(f"listening_area_stability.{target_key}")
            assert isinstance(target, TargetEntry) and isinstance(target.value, float | int)
            limit = float(target.value)
            for group in ("primary_to_surrounding", "surrounding_to_surrounding"):
                aggregate = getattr(getattr(channel.payload, field), group)
                if aggregate is not None:
                    worst = aggregate.worst_deviation
                    summaries.append(SummaryView(
                        role=channel.role, speaker_id=channel.speaker_id, metric=metric,
                        group=group, weighted_mean_text=_text(aggregate.weighted_mean_deviation),
                        worst_receiver_id=worst.receiver.receiver_id,
                        worst_reference_id=worst.reference.receiver_id,
                        worst_value_text=_precise(worst.value), unit=target.unit,
                        limit_text=_precise(limit), baseline_note=BASELINE_NOTE,
                    ))
            pairs.extend(PairView(
                role=channel.role, speaker_id=channel.speaker_id, metric=metric,
                group=item.group, receiver_id=item.receiver_id, reference_id=item.reference_id,
                value=item.value, value_text=_precise(item.value), unit=item.unit,
                limit=limit, limit_text=_precise(limit),
                excess_text=_precise(max(0.0, item.value - limit)),
                over_limit=item.value > limit, baseline_note=BASELINE_NOTE,
            ) for item in deviations if item.metric == metric)
    return ListeningAreaView(scope_note=LISTENING_AREA_SCOPE_NOTE,
                             summaries=tuple(summaries), pairs=tuple(pairs))


def build_result_view(result: SchemeResult, *, quality_targets_path: Path) -> ResultView:
    """只接受先經 load_result 驗證的結果；呼叫端負責拒收錯誤。"""
    registry = load_quality_targets(quality_targets_path)
    ranking = compare_results((result,), quality_targets=registry, run_date=result.run_date)
    row = next((item for item in ranking.rankable if item.candidate_id == result.scheme.scheme_id), None)
    costs = {line.identity.category: line.category_cost for line in row.categories} if row else {}
    ranked = {line.identity.category: line.evaluation for line in row.categories} if row else {}
    alert_rows = tuple(ranking.rankable) + tuple(ranking.eliminated)
    alerts = next((item.review_alerts for item in alert_rows
                   if item.candidate_id == result.scheme.scheme_id), ())
    return ResultView(
        scheme_id=result.scheme.scheme_id, engine_commit=result.engine_commit,
        run_date=result.run_date, timings=result.timings,
        timing_texts={name: _text(getattr(result.timings, name), "秒") for name in
                      ("solve_s", "output_s", "evaluate_s", "total_s")},
        labels=LABELS,
        frequency_responses=_frequency_responses(result),
        categories=_categories(result, costs, ranked),
        alerts=_alerts(alerts), reverberation=_reverberation(result, registry),
        reflections=_reflections(result),
        listening_area=_listening_area(result, quality_targets_path, registry),
    )
