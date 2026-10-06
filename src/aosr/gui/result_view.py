"""從已驗結果挑出結果頁資料；畫面不重算聲學量或門檻。"""
from __future__ import annotations

from datetime import date
from math import dist
from pathlib import Path
from typing import cast

from pydantic import BaseModel, ConfigDict

from aosr.config.capabilities import load_capabilities
from aosr.config.paths import config_path
from aosr.gui.capability_view import capability_lists
from aosr.config.quality_targets import QualityTargets, TargetEntry, load_quality_targets
from aosr.gui.labels import LISTENING_POINTS, listening_point_label, speaker_label
from aosr.reporting.compare import compare_results
from aosr.reporting.display import (
    BASELINE_NOTE, LOW_FREQUENCY_DECAY_NOTE,
    REVERBERATION_ROOM_NOTE, SPATIAL_IMPRESSION_NOTE, level_db,
)
from aosr.reporting.evaluation import (
    evaluate_point_timbres, read_registry_settings, receiver_point_results,
)
from aosr.reporting.result import PairResult, SchemeResult, Timings
from aosr.reporting.scheme import Scheme
from aosr.scoring.contract import (
    CategoryEvaluation, CostDirection, ListeningAreaChannelsPayload, QualityCategory,
    ReverberationBand, ReverberationPayload,
)
from aosr.scoring.listening_area import listening_area_pair_deviations
from aosr.scoring.listening_area_contract import DeviationAggregate
from aosr.scoring.ranking_models import MissingCategory
from aosr.scoring.reflections_contract import ReflectionPath, ReflectionsAndEchoPayload
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
    # 旗標的白話，給主表用；flags 原代號與 reason_codes、evaluator_version 留給「技術細節」。
    flags_text: str
    reason_codes: tuple[str, ...]
    evaluator_version: str | None
    note: str


class AlertView(ViewModel):
    """峰谷與聆聽區警戒（跟喇叭、座位有關的那幾種）；牆間顫動另外按牆對合併，見 FlutterGroupView。"""

    kind: str
    # 標題由伺服器拼好：警戒種類、喇叭顯示名；喇叭不明時只有種類，不留空的分隔號。
    heading_text: str
    # 在哪裡：峰谷是一個座位，聆聽區是一對座位；都用顯示名。
    place_text: str
    category: str
    speaker_id: str | None
    role: str | None
    receiver_id: str | None
    reference_id: str | None
    fields: tuple[tuple[str, str], ...]
    excess_text: str | None
    baseline_note: str | None


class FlutterBandView(ViewModel):
    """一對牆在一個頻帶的顫動明細（收在摺疊區裡）。"""

    nominal_text: str
    center_text: str
    duration_text: str
    room_t20_text: str
    excess_text: str


class FlutterGroupView(ViewModel):
    """同一對牆的顫動警戒合成一筆：頻帶範圍、帶數、持續度、本房同帶 T20、超出多少。"""

    walls: tuple[str, str]
    heading_text: str
    summary_text: str
    fields: tuple[tuple[str, str], ...]
    detail_summary_text: str
    bands: tuple[FlutterBandView, ...]


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
    # 跟目標比：within_range／below_range／above_range，T20 量不到是 unavailable，沒登記目標是 no_target。
    verdict: str
    verdict_text: str
    # T20、T30 各自量不量得到（取代原本的「已量、已量」）。
    measured_text: str


class ReverberationView(ViewModel):
    role: str
    receiver_id: str
    note: str
    # 表頭說明：房間統計量那句，加上取自哪支喇叭、哪個座位（顯示名）。
    caption_text: str
    # 「跟目標比」那一欄比的是哪個值、判定從哪裡來。
    compare_note: str
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
    # 「左聲道喇叭 → 主位」：喇叭與座位的顯示名。
    heading_text: str
    coverage: str
    validation: str
    state: str
    reason_codes: tuple[str, ...]
    flags: tuple[str, ...]
    # 注意事項的白話句子，跟各類結果主表同一份 FLAG_TEXTS（兩處不會一處白話、一處短代稱）。
    flags_text: str
    # 時間窗內的路徑直接列；窗外的收進摺疊區，摘要行說有幾條。沒有路徑表時兩句都是空字串。
    window_text: str
    outside_summary_text: str
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
    # 最差那一對的顯示名（主位 ↔ 主位左方）。
    worst_pair_text: str
    worst_value_text: str
    unit: str
    limit_text: str
    over_limit: bool
    excess_text: str
    baseline_note: str


class PairChoiceView(ViewModel):
    """聆聽區一個可選的位置對：主位對周圍點做成按鈕，周圍點彼此放進下拉選單。"""

    role: str
    group: str
    receiver_id: str
    reference_id: str
    # 整對的名字（主位 ↔ 主位前方）：明細標題與下拉選單用。
    text: str
    # 按鈕上的字：主位對周圍點那一組前面已有組名，按鈕只寫周圍點那一端（六顆才排得進一列）；其他同 text。
    button_text: str


class ListeningAreaView(ViewModel):
    scope_note: str
    state: str
    reason_codes: tuple[str, ...]
    summaries: tuple[SummaryView, ...]
    pairs: tuple[PairView, ...]
    pair_choices: tuple[PairChoiceView, ...]


class ResultView(ViewModel):
    not_modeled: tuple[str, ...] = ()
    manual_checks: tuple[str, ...] = ()
    scheme_id: str
    engine_commit: str
    # 主畫面只印前 7 碼，完整的留在 engine_commit。
    engine_commit_text: str
    run_date: date
    timings: Timings
    timing_texts: dict[str, str]
    labels: dict[str, str]
    # 顯示名：喇叭按聲道代號、座位按座位代號；表上沒有的照原代號。
    speaker_names: dict[str, str]
    point_names: dict[str, str]
    frequency_responses: tuple[FrequencyResponse, ...]
    frequency_plot_data: dict[str, tuple[tuple[float | None, ...], ...]]
    ranking_status: str
    ranking_reasons: tuple[str, ...]
    missing_categories: tuple[str, ...]
    # 排名那一行的白話：排名位置、有才說的淘汰原因與擋住排名的類、沒有代價的類不算進總代價。
    ranking_text: str
    cost_note: str
    categories: tuple[CategoryView, ...]
    alerts: tuple[AlertView, ...]
    flutter_groups: tuple[FlutterGroupView, ...]
    reverberation: ReverberationView
    reflections: tuple[ReflectionView, ...]
    listening_area: ListeningAreaView


def _text(value: float | None, unit: str = "") -> str:
    if value is None:
        return "—"
    digits = 3 if unit == "秒" else 4
    return f"{value:.{digits}f}{(' ' + unit) if unit else ''}"


def _fixed(value: float | None, digits: int, unit: str) -> str:
    if value is None:
        return "—"
    shown = f"{value:.{digits}f}"
    if float(shown) == 0.0:
        shown = f"{0.0:.{digits}f}"
    return f"{shown} {unit}"


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
    "tilt": "傾斜差", "ripple_rms": "起伏差（均方根）", "overall_level": "音量差",
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


# 各類結果主表的旗標白話；每一句照設旗標的那段程式寫，不多說。表上沒有的退回 LABELS 的短名。
FLAG_TEXTS = {
    "unvalidated": "有一部分計算尚未驗證",
    "baseline_settings": "用的線是暫定的，尚未正式校準",
    "analytic_directivity_unvalidated": "喇叭指向性用解析近似，尚未獨立驗證",
    "feature_narrower_than_axis": "有峰谷比頻率取樣點的間距還窄",
    "feature_boundary_incomplete": "有峰谷找不到完整邊緣，寬度量不到",
    "feature_too_narrow": "有峰谷窄於設定的最小寬度",
    "data_coverage_short": "資料的頻率範圍不夠寬或中間有缺口",
    "window_only_delay_screen": "反射只看直達音後的時間窗",
    "geometry_material_conservative_screen": "反射用幾何與材料做保守篩選",
    "no_directivity": "沒有喇叭指向資料",
}
COST_NOTE = "代價越低越好，0 表示沒有扣分"


def _flags_text(flags: tuple[str, ...]) -> str:
    return "；".join(FLAG_TEXTS.get(flag, _label(flag)) for flag in flags)


def _speaker_names(result: SchemeResult) -> dict[str, str]:
    return {channel.role: speaker_label(channel.role) for channel in result.scheme.channel_group.channels}


def _point_names(result: SchemeResult) -> dict[str, str]:
    return {point.receiver_id: listening_point_label(point.receiver_id)
            for point in result.scheme.receiver_set.points}


def _point_order(receiver_id: str) -> tuple[int, str]:
    """座位排序：主位、前、後、左、右、上、下；表上沒有的排最後、照代號。"""
    order = list(LISTENING_POINTS)
    return (order.index(receiver_id), "") if receiver_id in order else (len(order), receiver_id)


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
    return frequency_responses_for(result.scheme, result.pairs)


def frequency_responses_for(scheme: Scheme, pairs: tuple[PairResult, ...]) -> tuple[FrequencyResponse, ...]:
    """只轉換存下的報表逐點值，供結果頁與搜尋進度頁共用。"""
    roles = {point.receiver_id: point.role.value for point in scheme.receiver_set.points}
    labels = {"primary": "主位", "surrounding": "周圍點", "other_seat": "其他座位"}
    responses: list[FrequencyResponse] = []
    for pair in pairs:
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
        state_label=labels[item.state.value] if item else
            "不計分" if category is QualityCategory.LOW_FREQUENCY_DECAY else "尚未評估",
        cost=costs.get(category) if category in costs else
             (item.category_cost.value if item and item.category_cost else None),
        cost_text=_text(costs.get(category) if category in costs else
                        (item.category_cost.value if item and item.category_cost else None)),
        flags=tuple(flag.value for flag in item.flags) if item else (),
        flags_text=_flags_text(tuple(flag.value for flag in item.flags)) if item else "",
        reason_codes=tuple(code.value for code in item.reason_codes) if item else (),
        evaluator_version=item.evaluator_version if item else None,
        note=notes.get(category, ""),
    ) for category in QualityCategory for item in (evaluations.get(category),))


def _alert_fields(data: dict[str, object]) -> tuple[tuple[str, str], ...]:
    kind = data["kind"]
    if kind == "listening_area_worst_deviation":
        unit = "dB/oct" if data["metric"] == "tilt" else "dB"
        return (("量", _label(str(data["metric"]))), ("組", _label(str(data["group"]))),
                ("差值", f"{_measure(float(str(data['deviation'])), unit)} {unit}"),
                ("暫定線", f"{_measure(float(str(data['limit'])), unit)} {unit}"))
    depth = float(str(data["depth_db"]))
    return (("中心頻率", f"{float(str(data['center_frequency_hz'])):.2f} Hz"),
            ("峰谷量", f"谷深 {abs(depth):.2f} dB" if kind == "dip" else f"{depth:.2f} dB"),
            ("警戒線", f"{_measure(float(str(data['limit_db'])), 'dB')} dB"),
            ("寬度", "不可估" if data["width_octave"] is None else
             f"{float(str(data['width_octave'])):.3f} 八度"),
            ("窄於頻率軸", "是" if data["narrower_than_axis"] else "否"))


def _alert_entry(data: dict[str, object], registry: QualityTargets | None,
                 purpose_name: str | None) -> TargetEntry | None:
    """這一筆警戒用的那條線在登記簿的那一列；沒有登記簿時回 None。"""
    if registry is None or purpose_name is None:
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
    """超出多少：單位跟著那條線。"""
    entry = _alert_entry(data, registry, purpose_name)
    unit = entry.unit if entry is not None else ""
    text = _excess(excess, unit)
    return text if text == "未超過" or not unit else f"{text} {unit}"


def _alert_place(data: dict[str, object]) -> str:
    """峰谷警戒在一個座位，聆聽區警戒在一對座位；都印顯示名。"""
    receiver = listening_point_label(str(data["receiver_id"]))
    if data.get("reference_id") is None:
        return f"位置：{receiver}"
    return f"位置對：{listening_point_label(str(data['reference_id']))} ↔ {receiver}"


def _alerts(alerts: tuple[PeakDipReviewAlert | ListeningAreaReviewAlert, ...],
            registry: QualityTargets | None = None, purpose_name: str | None = None, *,
            roles: dict[str, str] | None = None) -> tuple[AlertView, ...]:
    views: list[AlertView] = []
    for alert in alerts:
        data = alert.model_dump(mode="json", exclude={"note"})
        excess = (data["deviation"] - data["limit"] if data["kind"] == "listening_area_worst_deviation"
                  else abs(data["depth_db"]) - data["limit_db"])
        speaker_id = str(data["speaker_id"])
        role = data.get("role") or (roles or {}).get(speaker_id)
        views.append(AlertView(
            kind=data["kind"],
            heading_text=f"{_label(data['kind'])}・{speaker_label(role) if role else speaker_id}",
            place_text=_alert_place(data),
            category=data["category"], speaker_id=speaker_id, role=role,
            receiver_id=data.get("receiver_id"), reference_id=data.get("reference_id"),
            fields=_alert_fields(data),
            excess_text=_alert_excess(data, excess, registry, purpose_name),
            baseline_note=_alert_baseline(data, registry, purpose_name),
        ))
    return tuple(views)


def _flutter_digits(alert: FlutterReviewAlert) -> int:
    """毫秒印幾位：持續度跟本房 T20 要看得出不同、超出量不准印成 0，最少一位。"""
    t20 = alert.room_t20_s * 1000.0
    digits = 1
    if alert.decay_duration_s is not None:
        value = alert.decay_duration_s * 1000.0
        excess = (alert.decay_duration_s - alert.room_t20_s) * 1000.0
        while (f"{value:.{digits}f}" == f"{t20:.{digits}f}" or
               float(f"{excess:.{digits}f}") == 0.0) and digits < 12:
            digits += 1
    return digits


def _flutter_excess(alert: FlutterReviewAlert, digits: int) -> str:
    if alert.decay_duration_s is None:
        return "持續度無限長"
    excess = alert.decay_duration_s - alert.room_t20_s
    if excess <= 0:
        return "未超過"
    shown = _fixed(excess * 1000.0, digits, "毫秒")
    return shown if float(shown.split()[0]) > 0 else "小於 0.000000000001 毫秒"


def _flutter_band(alert: FlutterReviewAlert, digits: int) -> FlutterBandView:
    return FlutterBandView(
        nominal_text=f"{alert.nominal_center_hz} Hz",
        center_text=f"{alert.center_frequency_hz:.2f} Hz",
        duration_text=(_fixed(alert.decay_duration_s * 1000.0, digits, "毫秒")
                       if alert.decay_duration_s is not None else "全反射，持續度無限長"),
        room_t20_text=_fixed(alert.room_t20_s * 1000.0, digits, "毫秒"),
        excess_text=_flutter_excess(alert, digits),
    )


def _span(low: str, high: str) -> str:
    """兩端已格式化好的值：一樣就只印一個；同單位的數字印「低–高 單位」，其他印「低 到 高」。"""
    if low == high:
        return low
    low_number, _, low_unit = low.rpartition(" ")
    high_unit = high.rpartition(" ")[2]
    if low_unit == high_unit and low_number.replace(".", "", 1).isdigit():
        return f"{low_number}–{high}"
    return f"{low} 到 {high}"


def _flutter_duration_text(alerts: list[FlutterReviewAlert], digits: int) -> str:
    durations = [alert.decay_duration_s * 1000.0 for alert in alerts if alert.decay_duration_s is not None]
    infinite = len(alerts) - len(durations)
    if not durations:
        return "全反射，持續度無限長"
    shown = _span(_fixed(min(durations), digits, "毫秒"), _fixed(max(durations), digits, "毫秒"))
    return f"{shown}；另有 {infinite} 帶全反射，持續度無限長" if infinite else shown


def _flutter_group(walls: tuple[str, str], alerts: list[FlutterReviewAlert]) -> FlutterGroupView:
    """同一對牆的各帶合成一筆；位數取各帶需要的最多位，逐帶明細才對得齊、分得出。"""
    ordered = sorted(alerts, key=lambda alert: alert.nominal_center_hz)
    digits = max(_flutter_digits(alert) for alert in ordered)
    finite = [alert for alert in ordered if alert.decay_duration_s is not None]

    def excess(alert: FlutterReviewAlert) -> float:
        return cast(float, alert.decay_duration_s) - alert.room_t20_s

    t20s = [alert.room_t20_s * 1000.0 for alert in ordered]
    bands = _span(f"{ordered[0].nominal_center_hz} Hz", f"{ordered[-1].nominal_center_hz} Hz")
    decays = {alert.decay_db for alert in ordered}
    decay = f" {next(iter(decays)):g} dB " if len(decays) == 1 else ""
    walls_text = "、".join(_label(wall) for wall in walls)
    return FlutterGroupView(
        walls=walls, heading_text=f"{_label('flutter')}・{walls_text}",
        summary_text=f"這對牆之間來回反射，衰減{decay}所需的時間（持續度）比本房同一頻帶的 T20 長，待複核",
        fields=(("頻帶", f"{bands}，共 {len(ordered)} 個三分之一八度帶"),
                ("持續度", _flutter_duration_text(ordered, digits)),
                ("本房同帶 T20", _span(_fixed(min(t20s), digits, "毫秒"), _fixed(max(t20s), digits, "毫秒"))),
                ("超出多少", _span(_flutter_excess(min(finite, key=excess), digits),
                                   _flutter_excess(max(finite, key=excess), digits))
                 if finite else "持續度無限長")),
        detail_summary_text=f"逐帶明細（{len(ordered)} 帶）",
        bands=tuple(_flutter_band(alert, digits) for alert in ordered),
    )


def _flutter_groups(alerts: tuple[FlutterReviewAlert, ...]) -> tuple[FlutterGroupView, ...]:
    """牆間顫動按牆對合併：一對牆一筆，照第一次出現的順序。"""
    grouped: dict[tuple[str, str], list[FlutterReviewAlert]] = {}
    for alert in alerts:
        grouped.setdefault(alert.walls, []).append(alert)
    return tuple(_flutter_group(walls, items) for walls, items in grouped.items())


def _alert_sections(alerts: tuple[PeakDipReviewAlert | FlutterReviewAlert | ListeningAreaReviewAlert, ...],
                    registry: QualityTargets, purpose_name: str, roles: dict[str, str]
                    ) -> tuple[tuple[AlertView, ...], tuple[FlutterGroupView, ...]]:
    """警戒分兩區：跟喇叭、座位有關的逐筆列；牆間顫動按牆對合併，排在後面。"""
    seats = tuple(alert for alert in alerts
                  if isinstance(alert, PeakDipReviewAlert | ListeningAreaReviewAlert))
    flutter = tuple(alert for alert in alerts if isinstance(alert, FlutterReviewAlert))
    return _alerts(seats, registry, purpose_name, roles=roles), _flutter_groups(flutter)


VERDICT_TEXTS = {"within_range": "在目標內", "below_range": "低於下限", "above_range": "高於上限",
                 "unavailable": "T20 量不到，無法比", "no_target": "這個頻帶沒有登記目標"}
_COMPARED_VALUE = "「跟目標比」看的是 T20（T30 只當參考，不計入代價）"
EVALUATOR_VERDICT_NOTE = f"{_COMPARED_VALUE}；判定照殘響評分自己逐帶算代價時的結果"
DIRECT_VERDICT_NOTE = (f"{_COMPARED_VALUE}；殘響這一類沒有算出代價，判定是直接拿 T20 跟目標上下限比"
                       "（剛好等於上下限算在目標內）")


def _band_verdict(band: ReverberationBand, targets: dict[float, tuple[float, float]],
                  directions: dict[str, CostDirection] | None) -> str:
    """逐帶判定：有評分自己的方向就用它；沒有才直接比，比法跟評分一樣（閉區間）。"""
    value = band.t20.value
    if value is None:
        return "unavailable"
    if band.center_frequency_hz not in targets:
        return "no_target"
    if directions is not None:
        return directions[f"t20_target_interval.{band.center_frequency_hz:g}Hz"]
    lower, upper = targets[band.center_frequency_hz]
    return "below_range" if value < lower else "above_range" if value > upper else "within_range"


def _measured_text(band: ReverberationBand) -> str:
    if band.t20.state.value == band.t30.state.value == "measured":
        return "T20、T30 都量到"
    return "；".join(
        f"{name} 量到" if metric.state.value == "measured" else
        f"{name} {_label(metric.state.value)}（{'、'.join(_label(code.value) for code in metric.reason_codes)}）"
        for name, metric in (("T20", band.t20), ("T30", band.t30)))


def _reverberation(result: SchemeResult, registry: QualityTargets,
                   costed: CategoryEvaluation | None = None) -> ReverberationView:
    """costed 是排名算過代價的殘響評估；它帶逐帶方向時「跟目標比」照它，否則直接比並在說明寫明。"""
    evaluation = next(item for item in result.candidate.evaluations
                      if item.category is QualityCategory.REVERBERATION)
    payload = evaluation.payload
    assert isinstance(payload, ReverberationPayload)
    targets = target_intervals(registry.purpose(result.scheme.purpose))
    directions = (costed.category_cost.component_directions
                  if costed is not None and costed.category_cost is not None else None)
    bands = tuple(ReverberationBandView(
        center_frequency_hz=band.center_frequency_hz,
        center_text=f"{band.center_frequency_hz:g} Hz",
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
        verdict=verdict, verdict_text=VERDICT_TEXTS[verdict], measured_text=_measured_text(band),
    ) for band in payload.bands for verdict in (_band_verdict(band, targets, directions),))
    first = result.scheme.channel_group.channels[0].role
    primary = result.scheme.receiver_set.primary.receiver_id
    return ReverberationView(
        role=first, receiver_id=primary, note=REVERBERATION_ROOM_NOTE,
        caption_text=f"{REVERBERATION_ROOM_NOTE}；取自{speaker_label(first)} → {listening_point_label(primary)}",
        compare_note=EVALUATOR_VERDICT_NOTE if directions is not None else DIRECT_VERDICT_NOTE,
        bands=bands)


def _reflection_path(path: ReflectionPath) -> ReflectionPathView:
    return ReflectionPathView(
        delay_text=_fixed(path.relative_direct_delay_s * 1000.0, 2, "毫秒"),
        level_text=_fixed(path.broadband_level_db, 1, "dB"),
        azimuth_text=_fixed(path.listening_azimuth_deg, 1, "度"),
        elevation_text=_fixed(path.listening_elevation_deg, 1, "度"),
        zone=path.zone.value, wall_sequence=tuple(_label(wall) for wall in path.wall_sequence),
        within_window=path.within_window,
    )


def _reflection_texts(paths: tuple[ReflectionPathView, ...], window_ms: float) -> tuple[str, str]:
    """時間窗那一句與窗外摺疊區的摘要；窗外沒有路徑時摘要是空字串（不畫摺疊區）。"""
    inside = sum(path.within_window for path in paths)
    outside = len(paths) - inside
    window = f"{window_ms:g} 毫秒"
    return (f"時間窗：直達音後 {window}內；窗內有 {inside} 條反射路徑，列在下表",
            f"時間窗外（晚於直達音 {window}）的反射路徑 {outside} 條，點開看" if outside else "")


def _reflections(result: SchemeResult) -> tuple[ReflectionView, ...]:
    evaluation = next(item for item in result.candidate.evaluations
                      if item.category is QualityCategory.REFLECTIONS_AND_ECHO)
    payload = evaluation.payload
    primary = result.scheme.receiver_set.primary.receiver_id
    flags_text = _flags_text(tuple(flag.value for flag in evaluation.flags))
    if payload is None:
        return tuple(ReflectionView(
            role=channel.role, speaker_id=channel.speaker_id, receiver_id=primary,
            heading_text=f"{speaker_label(channel.role)} → {listening_point_label(primary)}",
            coverage="unavailable", validation="unavailable", state="unavailable",
            reason_codes=tuple(code.value for code in evaluation.reason_codes),
            flags=tuple(flag.value for flag in evaluation.flags), flags_text=flags_text,
            window_text="", outside_summary_text="", paths=(),
        ) for channel in result.scheme.channel_group.channels)
    if not isinstance(payload, ReflectionsAndEchoPayload):
        raise ValueError("反射資料格式無效")
    views: list[ReflectionView] = []
    for channel in (item for item in payload.channels if item.is_primary):
        paths = tuple(_reflection_path(path) for path in
                      sorted(channel.reflections, key=lambda item: item.relative_direct_delay_s))
        window_text, outside_text = _reflection_texts(paths, payload.window_upper_ms)
        views.append(ReflectionView(
            role=channel.role, speaker_id=channel.speaker_id, receiver_id=channel.receiver_id,
            heading_text=f"{speaker_label(channel.role)} → {listening_point_label(channel.receiver_id)}",
            coverage=channel.coverage, validation=channel.validation, state=channel.state.value,
            reason_codes=tuple(code.value for code in channel.reason_codes),
            flags=tuple(flag.value for flag in evaluation.flags), flags_text=flags_text,
            window_text=window_text, outside_summary_text=outside_text, paths=paths,
        ))
    return tuple(views)


_METRICS = (("tilt", "tilt_stability", "tilt_worst_deviation"),
            ("ripple_rms", "ripple_rms_stability", "ripple_rms_worst_deviation"),
            ("overall_level", "overall_level_stability", "overall_level_worst_deviation"))


def _summary_view(role: str, speaker_id: str, metric: str, group: str,
                  aggregate: DeviationAggregate, target: TargetEntry) -> SummaryView:
    assert isinstance(target.value, float | int)
    limit = float(target.value)
    worst = aggregate.worst_deviation
    reference, receiver = worst.reference.receiver_id, worst.receiver.receiver_id
    first, second = sorted((reference, receiver), key=_point_order)
    return SummaryView(
        role=role, speaker_id=speaker_id, metric=metric,
        group=group, weighted_mean_text=_measure(aggregate.weighted_mean_deviation, target.unit),
        worst_receiver_id=receiver, worst_reference_id=reference,
        worst_pair_text=f"{listening_point_label(first)} ↔ {listening_point_label(second)}",
        worst_value_text=_measure(worst.value, target.unit), unit=target.unit,
        limit_text=_measure(limit, target.unit),
        # 剛好超線時兩格位數一樣看不出超了：另給有沒有超與超出多少（位數夠看得出來）。
        over_limit=worst.value > limit,
        excess_text=_excess(worst.value - limit, target.unit),
        baseline_note=BASELINE_NOTE if target.status == "baseline" else "",
    )


_PAIR_GROUPS = ("primary_to_surrounding", "surrounding_to_surrounding")


def _pair_order(choice: PairChoiceView) -> tuple[object, ...]:
    ends = sorted((choice.reference_id, choice.receiver_id), key=_point_order)
    group = _PAIR_GROUPS.index(choice.group) if choice.group in _PAIR_GROUPS else len(_PAIR_GROUPS)
    return (group, *(_point_order(end) for end in ends))


def _pair_choices(pairs: tuple[PairView, ...], primary: str) -> tuple[PairChoiceView, ...]:
    """每支喇叭的每一對座位一個選項（三種量共用同一對）；主位對周圍點在前，座位照顯示順序。"""
    choices: dict[tuple[str, str, str, str], PairChoiceView] = {}
    for pair in pairs:
        key = (pair.role, pair.group, pair.receiver_id, pair.reference_id)
        if key not in choices:
            first, second = sorted((pair.reference_id, pair.receiver_id), key=_point_order)
            text = f"{listening_point_label(first)} ↔ {listening_point_label(second)}"
            ends = {pair.reference_id, pair.receiver_id} - {primary}
            short = (listening_point_label(ends.pop()) if pair.group == "primary_to_surrounding"
                     and primary in (pair.reference_id, pair.receiver_id) and len(ends) == 1 else text)
            choices[key] = PairChoiceView(
                role=pair.role, group=pair.group, receiver_id=pair.receiver_id,
                reference_id=pair.reference_id, text=text, button_text=short)
    return tuple(sorted(choices.values(), key=_pair_order))


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
                                 summaries=(), pairs=(), pair_choices=())
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
                    summaries.append(_summary_view(channel.role, channel.speaker_id, metric, group,
                                                   aggregate, target))
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
                             summaries=tuple(summaries), pairs=tuple(pairs),
                             pair_choices=_pair_choices(tuple(pairs),
                                                        result.scheme.receiver_set.primary.receiver_id))


def _ranking_text(status: str, reasons: tuple[str, ...], missing: tuple[MissingCategory, ...],
                  categories: tuple[CategoryView, ...]) -> str:
    """排名那一行：淘汰原因、擋住排名的類有才說；其餘沒有代價的類照狀態列出，可排名時註明不算進總代價。"""
    parts = [f"排名位置：{_label(status)}"]
    if reasons:
        parts.append(f"淘汰原因：{'、'.join(_label(reason) for reason in reasons)}")
    if missing:
        parts.append("擋住排名的類：" + "、".join(
            f"{_label(item.category.value)}（{_label(item.reason.value)}）" for item in missing))
    blocked = {item.category.value for item in missing}
    uncounted: dict[str, list[str]] = {}
    for item in categories:
        if item.category == QualityCategory.LOW_FREQUENCY_DECAY.value:
            parts.append(LOW_FREQUENCY_DECAY_NOTE)
            continue
        if item.cost is None and item.category not in blocked:
            uncounted.setdefault(item.state_label, []).append(_label(item.category))
    counted = "、不算進總代價" if status == "rankable" else ""
    parts.extend(f"{state}{counted}：{'、'.join(names)}" for state, names in uncounted.items())
    return "；".join(parts)


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
    categories = _categories(result, costs, ranked)
    status = ranking.status_of(result.scheme.scheme_id).value
    reasons = tuple(reason.value for reason in eliminated.reasons) if eliminated else ()
    seat_alerts, flutter_groups = _alert_sections(
        alerts, registry, result.scheme.purpose,
        {channel.speaker_id: channel.role for channel in result.scheme.channel_group.channels})
    capabilities = capability_lists(load_capabilities(config_path("capabilities.toml")))
    return ResultView(
        not_modeled=capabilities["not_modeled"], manual_checks=capabilities["manual_checks"],
        scheme_id=result.scheme.scheme_id, engine_commit=result.engine_commit,
        engine_commit_text=result.engine_commit[:7],
        run_date=result.run_date, timings=result.timings,
        timing_texts={name: _text(getattr(result.timings, name), "秒") for name in
                      ("solve_s", "output_s", "evaluate_s", "total_s")},
        labels=LABELS, speaker_names=_speaker_names(result), point_names=_point_names(result),
        frequency_responses=responses, frequency_plot_data=_frequency_plot_data(responses),
        ranking_status=status, ranking_reasons=reasons,
        missing_categories=tuple(item.category.value for item in missing),
        ranking_text=_ranking_text(status, reasons, missing, categories),
        cost_note=COST_NOTE, categories=categories,
        alerts=seat_alerts, flutter_groups=flutter_groups,
        reverberation=_reverberation(result, registry, ranked.get(QualityCategory.REVERBERATION)),
        reflections=_reflections(result),
        listening_area=_listening_area(result, quality_targets_path, registry),
    )
