"""從已驗結果挑出結果頁資料；畫面不重算聲學量或門檻。"""
from __future__ import annotations

from datetime import date
from math import dist
from pathlib import Path
from typing import cast

from pydantic import BaseModel, ConfigDict

from aosr.config.quality_targets import QualityTargets, TargetEntry, load_quality_targets
from aosr.reporting.compare import compare_results
from aosr.reporting.display import (
    BASELINE_NOTE, LOW_FREQUENCY_DECAY_NOTE,
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
from aosr.scoring.reverberation_cost import target_intervals


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
    status_text: str


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
    over_limit: bool
    excess_text: str
    baseline_note: str


class ListeningAreaView(ViewModel):
    scope_note: str
    state: str
    reason_codes: tuple[str, ...]
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
    frequency_plot_data: dict[str, tuple[tuple[float | None, ...], ...]]
    ranking_status: str
    ranking_reasons: tuple[str, ...]
    missing_categories: tuple[str, ...]
    categories: tuple[CategoryView, ...]
    alerts: tuple[AlertView, ...]
    reverberation: ReverberationView
    reflections: tuple[ReflectionView, ...]
    listening_area: ListeningAreaView


def _text(value: float | None, unit: str = "") -> str:
    if value is None:
        return "—"
    digits = 3 if unit == "秒" else 4
    return f"{value:.{digits}f}{(' ' + unit) if unit else ''}"


def _measure(value: float, unit: str) -> str:
    return f"{value:.3f}" if unit == "dB/oct" else f"{value:.2f}"


def _excess(value: float, unit: str = "") -> str:
    if value <= 0:
        return "未超過"
    digits = 3 if unit == "dB/oct" else 2
    while float(f"{value:.{digits}f}") == 0.0 and digits < 12:
        digits += 1
    if float(f"{value:.{digits}f}") == 0.0:
        return "小於 0.000000000001"
    return f"{value:.{digits}f}"


LABELS = {
    "frequency_axis": "頻率（Hz）", "level_axis": "聲級（dB）",
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
    "rankable": "可排名", "eliminated": "淘汰", "not_evaluated": "未評估",
    "not_comparable": "不可同表比較", "illegal": "方案不合法",
    "insufficient_coverage": "覆蓋範圍不足", "timbre_scoring_range_gap": "音色計分頻段有缺口",
    "missing_points": "缺逐點資料", "non_positive_energy": "能量不是正值",
    "solver_unavailable": "求解不可用", "evaluator_not_implemented": "評估器尚未實作",
    "reflections_evaluation_missing": "缺反射評估", "candidate_id_mismatch": "候選代號不符",
    "speaker_id_mismatch": "喇叭代號不符", "receiver_set_fingerprint_mismatch": "座位配置指紋不符",
    "evaluator_version_mismatch": "評估器版本不符", "scene_fingerprint_mismatch": "房間指紋不符",
    "placement_mismatch": "擺位不符", "settings_fingerprint_mismatch": "設定指紋不符",
    "timbre_settings_fingerprint_mismatch": "音色設定指紋不符",
    "listening_area_settings_fingerprint_mismatch": "聆聽區設定指紋不符",
    "channel_group_fingerprint_mismatch": "聲道組指紋不符",
    "channel_result_unavailable": "聲道結果不可估",
    "required_channel_point_unavailable": "必要聲道位置不可估",
    "channel_role_mismatch": "聲道角色不符", "frequency_axis_mismatch": "頻率軸不符",
    "invalid_direct_distance": "直達距離無效", "receiver_id_mismatch": "座位代號不符",
    "timbre_not_measured": "音色尚未量到", "zero_total_importance": "周圍點重要性總和為零",
    "no_surrounding_pairs": "沒有周圍點配對", "insufficient_decay_range": "衰減範圍不足",
    "band_row_missing": "缺頻帶資料", "non_positive_value": "數值不是正值",
    "other_error": "其他錯誤", "path_table_missing": "缺路徑表",
    "reflection_screen_or_window_missing": "反射篩選或時間窗缺資料",
    "reflection_screen_or_window_mismatch": "反射篩選或時間窗不符",
    "reflection_window_incomplete": "反射時間窗不完整",
    "listening_axis_undefined": "聆聽方向無法定義",
    "no_reflection_in_zone_point": "這個方向沒有反射路徑",
    "zero_reflection_energy": "反射能量為零", "zero_retention": "反射保留率為零",
    "full_reflection": "全反射", "t20_band_unavailable": "本房 T20 頻帶不可估",
    "subband_sampling_incomplete": "子帶取樣不完整", "source_model_mismatch": "聲源模型不符",
    "mandatory_category_missing": "缺必要類別", "mandatory_category_unavailable": "必要類別不可估",
    "cost_not_computed": "代價尚未算出",
    "reverberation_too_many_unavailable_bands": "不可估殘響頻帶太多",
    "reverberation_critical_band_unavailable": "重要殘響頻帶不可估",
    "reverberation_insufficient_valid_bands": "可用殘響頻帶不足",
    "external_floor_failed": "外部底線未過",
    "timbre_peak_beyond_limit": "音色峰值超線",
    "timbre_dip_beyond_limit": "音色谷值超線",
    "listening_area_tilt_primary_to_surrounding_worst_beyond_limit": "主位對周圍點傾斜差超線",
    "listening_area_tilt_surrounding_to_surrounding_worst_beyond_limit": "周圍點彼此傾斜差超線",
    "listening_area_ripple_primary_to_surrounding_worst_beyond_limit": "主位對周圍點起伏差超線",
    "listening_area_ripple_surrounding_to_surrounding_worst_beyond_limit": "周圍點彼此起伏差超線",
    "listening_area_level_primary_to_surrounding_worst_beyond_limit": "主位對周圍點音量差超線",
    "listening_area_level_surrounding_to_surrounding_worst_beyond_limit": "周圍點彼此音量差超線",
    "channel_matching_tilt_worst_beyond_limit": "聲道傾斜差超線",
    "channel_matching_ripple_worst_beyond_limit": "聲道起伏差超線",
    "channel_matching_level_worst_beyond_limit": "聲道音量差超線",
    "channel_matching_direct_time_worst_beyond_limit": "聲道直達時間差超線",
    "data_coverage_short": "資料覆蓋不足", "crossover_band": "跨越頻帶交界",
    "feature_too_narrow": "特徵過窄", "feature_boundary_incomplete": "特徵邊界不完整",
    "feature_narrower_than_axis": "特徵窄於頻率軸",
    "reflection_front_above_threshold": "前方反射超線",
    "reflection_lateral_above_threshold": "側向反射超線",
    "reflection_rear_above_threshold": "後方反射超線",
    "reflection_vertical_above_threshold": "上下反射超線",
    "analytic_directivity_unvalidated": "解析指向性尚未驗證",
    "floor": "地板", "ceiling": "天花", "x0": "x 起點牆", "xL": "x 終點牆",
    "y0": "y 起點牆", "yL": "y 終點牆",
}


def _label(code: str) -> str:
    return LABELS.get(code, "尚無中文標籤")


def _frequency_plot_data(responses: tuple[FrequencyResponse, ...]
                         ) -> dict[str, tuple[tuple[float | None, ...], ...]]:
    plots: dict[str, tuple[tuple[float | None, ...], ...]] = {}
    for role in {item.role for item in responses}:
        group = tuple(item for item in responses if item.role == role)
        axis = tuple(sorted({point.frequency_hz for item in group for point in item.points}))
        values = tuple(tuple({point.frequency_hz: point.level_db for point in item.points}.get(x)
                             for x in axis) for item in group)
        plots[role] = (axis, *values)
    return plots


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


def _alert_fields(data: dict[str, object]) -> tuple[tuple[str, str], ...]:
    kind = data["kind"]
    if kind == "flutter":
        duration = data["decay_duration_s"]
        t20 = float(str(data["room_t20_s"]))
        digits = 5
        if duration is not None:
            while float(f"{float(str(duration)):.{digits}f}") == float(f"{t20:.{digits}f}") and digits < 12:
                digits += 1
        return (("牆對", "、".join(_label(wall) for wall in cast(tuple[str, str], data["walls"]))),
                ("名義中心頻率", f"{data['nominal_center_hz']} Hz"),
                ("中心頻率", f"{float(str(data['center_frequency_hz'])):.2f} Hz"),
                ("持續度", f"{float(str(duration)):.{digits}f} 秒" if duration is not None
                 else "全反射，持續度無限長"),
                ("本房 T20", f"{t20:.{digits}f} 秒"))
    if kind == "listening_area_worst_deviation":
        unit = "dB/oct" if data["metric"] == "tilt" else "dB"
        return (("量", _label(str(data["metric"]))), ("組", _label(str(data["group"]))),
                ("差值", _measure(float(str(data["deviation"])), unit)),
                ("暫定線", _measure(float(str(data["limit"])), unit)))
    return (("中心頻率", f"{float(str(data['center_frequency_hz'])):.2f} Hz"),
            ("峰谷量", _measure(float(str(data["depth_db"])), "dB")),
            ("警戒線", _measure(float(str(data["limit_db"])), "dB")),
            ("寬度", "不可估" if data["width_octave"] is None else
             f"{float(str(data['width_octave'])):.3f} 八度"),
            ("窄於頻率軸", "是" if data["narrower_than_axis"] else "否"))


def _alert_entry(data: dict[str, object], registry: QualityTargets | None,
                 purpose_name: str | None) -> TargetEntry | None:
    """這一筆警戒用的那條線在登記簿的那一列；沒有登記簿時回 None。顫動不看登記簿。"""
    if registry is None or purpose_name is None or data["kind"] == "flutter":
        return None
    kind = data["kind"]
    key = (f"listening_area_stability.{next(target for metric, _, target in _METRICS if metric == data['metric'])}"
           if kind == "listening_area_worst_deviation" else
           "timbre_balance.peak_depth_db" if kind == "peak" else "timbre_balance.dip_depth_db")
    entry = registry.purpose(purpose_name).entry(key)
    if not isinstance(entry, TargetEntry):
        raise ValueError("警戒線登記格式無效")
    return entry


def _alert_baseline(data: dict[str, object], registry: QualityTargets | None,
                    purpose_name: str | None) -> str | None:
    entry = _alert_entry(data, registry, purpose_name)
    if entry is None:
        return BASELINE_NOTE
    return BASELINE_NOTE if entry.status == "baseline" else None


def _alert_excess(data: dict[str, object], excess: float, registry: QualityTargets | None,
                  purpose_name: str | None) -> str:
    """超出多少：單位跟著那條線（顫動是秒），位數跟逐對明細同一套。"""
    if data["kind"] == "flutter":
        unit = "秒"
    else:
        entry = _alert_entry(data, registry, purpose_name)
        unit = entry.unit if entry is not None else ""
    text = _excess(excess, unit)
    return text if text == "未超過" or not unit else f"{text} {unit}"


def _alerts(alerts: tuple[PeakDipReviewAlert | FlutterReviewAlert |
                           ListeningAreaReviewAlert, ...], registry: QualityTargets | None = None,
            purpose_name: str | None = None) -> tuple[AlertView, ...]:
    views: list[AlertView] = []
    for alert in alerts:
        data = alert.model_dump(mode="json", exclude={"note"})
        fields = _alert_fields(data)
        excess: float | None = None
        if data["kind"] == "listening_area_worst_deviation":
            excess = data["deviation"] - data["limit"]
        elif data["kind"] in {"peak", "dip"}:
            excess = abs(data["depth_db"]) - data["limit_db"]
        elif data["kind"] == "flutter":
            excess = (data["decay_duration_s"] - data["room_t20_s"]
                      if data["decay_duration_s"] is not None else None)
        views.append(AlertView(
            kind=data["kind"], category=data["category"],
            speaker_id=data.get("speaker_id"), role=data.get("role"),
            receiver_id=data.get("receiver_id"), reference_id=data.get("reference_id"),
            fields=fields, excess_text=(_alert_excess(data, excess, registry, purpose_name)
                                        if excess is not None else None),
            baseline_note=_alert_baseline(data, registry, purpose_name)
            if data["kind"] != "flutter" else None,
        ))
    return tuple(views)


def _reverberation(result: SchemeResult, registry: QualityTargets) -> ReverberationView:
    evaluation = next(item for item in result.candidate.evaluations
                      if item.category is QualityCategory.REVERBERATION)
    payload = evaluation.payload
    assert isinstance(payload, ReverberationPayload)
    targets = target_intervals(registry.purpose(result.scheme.purpose))
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
    if payload is None:
        return tuple(ReflectionView(
            role=channel.role, speaker_id=channel.speaker_id,
            receiver_id=result.scheme.receiver_set.primary.receiver_id,
            coverage="unavailable", validation="unavailable", state="unavailable",
            reason_codes=tuple(code.value for code in evaluation.reason_codes),
            flags=tuple(flag.value for flag in evaluation.flags), paths=(),
        ) for channel in result.scheme.channel_group.channels)
    if not isinstance(payload, ReflectionsAndEchoPayload):
        raise ValueError("反射資料格式無效")
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
            zone=path.zone.value, wall_sequence=tuple(_label(wall) for wall in path.wall_sequence),
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
    primary = result.scheme.receiver_set.primary.position_m
    radius = max((dist(primary, point.position_m) for point in result.scheme.receiver_set.points
                  if point.role.value == "surrounding"), default=0.0)
    scope = f"周圍點離主位最遠 {radius:.2f} 公尺；其他座位本版未評"
    if payload is None:
        return ListeningAreaView(scope_note=scope, state="unavailable",
                                 reason_codes=tuple(code.value for code in evaluation.reason_codes),
                                 summaries=(), pairs=())
    if not isinstance(payload, ListeningAreaChannelsPayload):
        raise ValueError("聆聽區資料格式無效")
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
                        group=group, weighted_mean_text=_measure(aggregate.weighted_mean_deviation, target.unit),
                        worst_receiver_id=worst.receiver.receiver_id,
                        worst_reference_id=worst.reference.receiver_id,
                        worst_value_text=_measure(worst.value, target.unit), unit=target.unit,
                        limit_text=_measure(limit, target.unit),
                        # 剛好超線時兩格位數一樣看不出超了：另給有沒有超與超出多少（位數夠看得出來）。
                        over_limit=worst.value > limit,
                        excess_text=_excess(worst.value - limit, target.unit),
                        baseline_note=BASELINE_NOTE if target.status == "baseline" else "",
                    ))
            pairs.extend(PairView(
                role=channel.role, speaker_id=channel.speaker_id, metric=metric,
                group=item.group, receiver_id=item.receiver_id, reference_id=item.reference_id,
                value=item.value, value_text=_measure(item.value, item.unit), unit=item.unit,
                limit=limit, limit_text=_measure(limit, item.unit),
                excess_text=_excess(item.value - limit, item.unit),
                over_limit=item.value > limit,
                baseline_note=BASELINE_NOTE if target.status == "baseline" else "",
                status_text=("超過暫定線" if item.value > limit else "未超過") +
                (f"；{BASELINE_NOTE}" if target.status == "baseline" else ""),
            ) for item in deviations if item.metric == metric)
    return ListeningAreaView(scope_note=scope, state=evaluation.state.value,
                             reason_codes=tuple(code.value for code in evaluation.reason_codes),
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
    eliminated = next((item for item in ranking.eliminated
                       if item.candidate_id == result.scheme.scheme_id), None)
    not_evaluated = next((item for item in ranking.not_evaluated
                          if item.candidate_id == result.scheme.scheme_id), None)
    missing = eliminated.missing if eliminated else not_evaluated.missing if not_evaluated else ()
    responses = _frequency_responses(result)
    return ResultView(
        scheme_id=result.scheme.scheme_id, engine_commit=result.engine_commit,
        run_date=result.run_date, timings=result.timings,
        timing_texts={name: _text(getattr(result.timings, name), "秒") for name in
                      ("solve_s", "output_s", "evaluate_s", "total_s")},
        labels=LABELS,
        frequency_responses=responses, frequency_plot_data=_frequency_plot_data(responses),
        ranking_status=ranking.status_of(result.scheme.scheme_id).value,
        ranking_reasons=tuple(reason.value for reason in eliminated.reasons) if eliminated else (),
        missing_categories=tuple(item.category.value for item in missing),
        categories=_categories(result, costs, ranked),
        alerts=_alerts(alerts, registry, result.scheme.purpose),
        reverberation=_reverberation(result, registry),
        reflections=_reflections(result),
        listening_area=_listening_area(result, quality_targets_path, registry),
    )
