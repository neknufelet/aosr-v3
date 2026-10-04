"""搜尋報告骨架：只讀快照與結果，停止、細算、品質各自陳述。"""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from aosr.config.quality_targets import QualityTargets, load_quality_targets
from aosr.reporting.compare import compare_results, comparison_problems
from aosr.reporting.result import PurposeSettings, SchemeResult
from aosr.scoring.ranking_models import CandidateStatus, RankingHeader, RankingResult
from aosr.scoring.recommendation import NotFinalReason, RecommendationStatus, ReviewStatus
from aosr.search.layout_settings import Box, Span
from aosr.search.outer_status import conclusion_message
from aosr.search.report_comparison import (
    PlacementReport, default_precision_contracts_path, placement_report, placement_text,
    rank_lines, read_refinement_rows,
)
from aosr.search.run import RefineStopReason, RoundRecord, SearchStatus, State
from aosr.search.sampler import RankingZone
from aosr.search.store import FROZEN, SearchStore
from aosr.search.timings import SearchTimings, round_text, total_text


class _FrozenModel(BaseModel):
    model_config = FROZEN


class SearchStopReport(_FrozenModel):
    """搜尋停止那一段：逐欄抄搜尋自己的狀態；細算狀態另成一段，不放進這裡。停止設定不代表找到全域最佳。"""

    state: State
    message: str
    asked: int
    computed: int
    illegal: int
    illegal_reasons: dict[str, int]
    excluded: dict[str, int]
    best_trial: int | None
    best_score: float | None
    streak: int
    start_enqueued: bool
    baseline_outcome: str
    baseline_reason_codes: tuple[str, ...]
    round: int = 1
    round_start_trial: int = 0
    rounds: tuple[RoundRecord, ...] = ()
    budget: int
    convergence_run: int
    settings_note: str = "預算與連續未改善數都是暫行的工程停止設定，不代表找到全域最佳"


class RefinementState(StrEnum):
    """本支施工只會回報尚未開始。

    讀回狀態時保留細算自己的五種狀態，之後由細算流程更新。
    """

    NOT_STARTED = "not_started"
    RUNNING = "running"
    STOPPED = "stopped"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class RefinementReport(_FrozenModel):
    state: RefinementState = RefinementState.NOT_STARTED
    message: str = "細算未開始：還沒有任何候選用驗證軸細算"
    stop_reason: RefineStopReason | None = None
    outer_message: str = "未判定"


class CandidateQuality(_FrozenModel):
    """結果未讀回或尚未重排時，區別留空，不冒充排名層的未評估區。"""

    message: str
    zone: CandidateStatus | None = None
    review_status: ReviewStatus | None = None
    recommendation_status: RecommendationStatus | None = None
    not_final_reasons: tuple[NotFinalReason, ...] = ()


class QualityReport(_FrozenModel):
    verdict: Literal["未判定合格"] = "未判定合格"
    reason: str = "合格規則還沒拍板；搜尋設計紙第五節第 1 條要求照品質登記簿判，判法尚未定"
    message: str
    header: RankingHeader | None = None
    original: CandidateQuality
    best: CandidateQuality


class RestrictionsReport(_FrozenModel):
    wall_gap_m: float
    keep_out: tuple[Box, ...]
    speaker_areas: tuple[Box, ...] | None
    listening_range_m: Span | None
    base_angle_deg: Span | None


class UnassessedReport(_FrozenModel):
    items: tuple[str, ...] = ("製作用途", "多人座位", "低頻拖尾", "物件反射", "箱體反射")
    angle_note: str | None = None


class ScopeReport(_FrozenModel):
    scope: Literal["stage_two_subset"] = "stage_two_subset"
    message: str = "第二階段子集"


class ReferenceMeaningsReport(_FrozenModel):
    original: str = "原方案：專案本來的擺法"
    provisional: str = "暫定：品質登記簿裡還沒校準的條目（原因碼 calibration_baseline）"


class SearchReport(_FrozenModel):
    """三件事各有子模型，其餘限制、未評估項目與範圍也各自獨立。"""

    search: SearchStopReport
    refinement: RefinementReport
    ranks: tuple[str, ...]
    placement: PlacementReport
    quality: QualityReport
    restrictions: RestrictionsReport
    unassessed: UnassessedReport
    scope: ScopeReport
    references: ReferenceMeaningsReport
    timings: SearchTimings = SearchTimings()


def _read_result(path: Path) -> SchemeResult | None:
    try:
        return SchemeResult.model_validate_json(path.read_bytes())
    except (OSError, ValueError, UnicodeError):
        return None


def _candidate_quality(result: SchemeResult, ranking: RankingResult,
                       message: str) -> CandidateQuality:
    candidate_id = result.candidate.candidate_id
    zone = ranking.status_of(candidate_id)
    row = next((row for row in ranking.rankable if row.candidate_id == candidate_id), None)
    return CandidateQuality(
        message=message, zone=zone,
        review_status=None if row is None else row.review_status,
        recommendation_status=None if row is None else row.recommendation_status,
        not_final_reasons=() if row is None else row.not_final_reasons,
    )


def _same_settings(registry: QualityTargets, store: SearchStore) -> bool:
    purpose = registry.purpose(store.settings.purpose)
    found = PurposeSettings(purpose=purpose.name, fingerprint=purpose.fingerprint,
                            content=purpose.canonical())
    return found == store.identity.purpose_settings


def _named(text: str, original: SchemeResult | None, best: SchemeResult | None) -> str:
    """排名層原文用方案代號點名；換成原方案、第一名，讀者分得出是哪一份（只換這兩個確切代號）。"""
    for result, name in ((original, "原方案"), (best, "第一名")):
        if result is not None:
            text = text.replace(result.scheme.scheme_id, name)
    return text


def _quality(store: SearchStore, status: SearchStatus, original: SchemeResult | None,
             best: SchemeResult | None, registry: QualityTargets, run_date: date) -> QualityReport:
    original_info = CandidateQuality(message="原方案結果檔讀不回" if original is None else "原方案結果已讀回")
    best_info = CandidateQuality(message=("沒有第一名：沒有任何候選拿到分數" if status.best_trial is None
                                         else "不是最終推薦" if best is not None
                                         else "不是最終推薦；第一名結果檔讀不回"))
    quality = QualityReport(message="列出排名層現有欄位作為依據，尚未判定合格",
                            original=original_info, best=best_info)
    try:
        same = _same_settings(registry, store)
    except KeyError:
        same = False
    if not same:
        return quality.model_copy(update={"message": "評分設定跟搜尋快照不同，這份報告不重排"})
    results = [item for item in (original, best) if item is not None]
    if not results:
        return quality.model_copy(update={"message": "沒有讀得回的結果，這份報告無法重排"})
    problems = comparison_problems(results)
    if problems:
        # 固定身分不相容時 compare_results 拒排整張表；照原文寫哪一項不同，兩份都標不能同表。
        updates = {label: info.model_copy(update={"zone": CandidateStatus.NOT_COMPARABLE})
                   for label, result, info in (("original", original, original_info), ("best", best, best_info))
                   if result is not None}
        message = "結果不能同表，這份報告不重排：" + _named("；".join(problems), original, best)
        return quality.model_copy(update={"message": message, **updates})
    try:
        ranking = compare_results(results, quality_targets=registry, run_date=run_date)
    except ValueError as error:
        # 排名層自己的錯（例如登記簿設定矛盾）不是不能同表；照原文寫，不猜所在區。
        return quality.model_copy(update={"message": "排名失敗，這份報告不重排：" + _named(str(error), original, best)})
    return quality.model_copy(update={
        "header": ranking.header,
        "original": original_info if original is None else _candidate_quality(original, ranking, original_info.message),
        "best": best_info if best is None else _candidate_quality(best, ranking, best_info.message),
    })


def build_report(store: SearchStore, *, quality_targets_path: Path, run_date: date,
                 precision_contracts_path: Path | None = None) -> SearchReport:
    """只讀當次搜尋；結果缺席不妨礙其餘段落，評分設定不同就不重排。"""
    status = SearchStatus.model_validate_json(store.status_path.read_bytes())
    original = _read_result(store.baseline_path)
    best = None if status.best_trial is None else _read_result(store.candidate_path(status.best_trial))
    registry = load_quality_targets(quality_targets_path)
    settings = store.settings
    limits = settings.layout
    rows = read_refinement_rows(store)
    try:
        same_settings = _same_settings(registry, store)
    except KeyError:
        same_settings = False
    return SearchReport(
        search=SearchStopReport(**status.model_dump(exclude={"refine", "outer", "timings"}), budget=settings.budget,
                                convergence_run=settings.convergence_run),
        refinement=RefinementReport(state=RefinementState(status.refine.state), message=status.refine.message,
                                    stop_reason=status.refine.stop_reason, outer_message=conclusion_message(status)),
        ranks=rank_lines(store, status, rows, registry, run_date, _read_result, same_settings=same_settings),
        placement=placement_report(store, status, rows, _read_result,
                                   precision_contracts_path or default_precision_contracts_path()),
        quality=_quality(store, status, original, best, registry, run_date),
        restrictions=RestrictionsReport(**{name: getattr(limits, name) for name in RestrictionsReport.model_fields}),
        unassessed=UnassessedReport(angle_note=None if limits.base_angle_deg is None
                                   else "夾角限制不代表空間感已評估"),
        scope=ScopeReport(scope="stage_two_subset" if original is None else original.scope),
        references=ReferenceMeaningsReport(),
        timings=status.timings,
    )


def _refinement_text(report: RefinementReport) -> str:
    """未開始保留既有文字；其餘照保存的狀態、訊息與停止原因寫。"""
    caveat = ("\n細算完成不代表全域最佳：回饋只在第一名附近找，搜尋取樣沒走到的範圍補不到（#611）"
              if report.outer_message == "細算完成" else "")
    if report.state == RefinementState.NOT_STARTED:
        return "細算做完沒\n" + report.message + "\n外圈結論：" + report.outer_message + caveat
    labels = {RefinementState.RUNNING: "進行中", RefinementState.STOPPED: "已停",
              RefinementState.FAILED: "失敗", RefinementState.INTERRUPTED: "中斷"}
    state = labels[report.state]
    reasons = {"stable": "細算第一名連續一段沒被換掉", "refine_budget": "用完細算上限",
               "candidates_exhausted": "沒有候選可以再細算", "user_stopped": "使用者停止"}
    if report.state == RefinementState.STOPPED and report.stop_reason is not None:
        state += f"（{reasons[report.stop_reason]}）"
    return "\n".join(("細算做完沒", f"狀態：{state}", report.message, f"外圈結論：{report.outer_message}")) + caveat


def _counts_text(counts: dict[str, int]) -> str:
    labels = {"cabinet_outside_room": "箱體越界", "wall_gap": "離牆間隙不足",
              "cabinets_overlap": "箱體重疊", "cabinet_in_keep_out": "箱體進入禁區",
              "seat_in_keep_out": "座位進入禁區", "seat_outside_room": "座位越界",
              "outside_speaker_area": "喇叭超出可用區", "listening_distance_out_of_range": "聆聽距離超出範圍",
              "base_angle_out_of_range": "水平夾角超出範圍", RankingZone.ELIMINATED.value: "淘汰",
              RankingZone.UNASSESSED.value: "未評估", RankingZone.INCOMPARABLE.value: "不能同表"}
    return "、".join(f"{labels.get(key, f'未辨識原因（{key}）')}：{value}"
                    for key, value in counts.items()) or "沒有"


def _search_text(report: SearchStopReport) -> str:
    states = {"running": "進行中", "converged": "達到停止條件", "budget_exhausted": "因預算停止",
              "user_stopped": "使用者停止", "failed": "失敗", "interrupted": "中斷"}
    rounds = (f"目前第 {report.round} 輪", *(
        f"第 {record.round} 輪：{states[record.state]}；問過 {record.asked} 題；"
        f"第一名 {'沒有' if record.best_trial is None else record.best_trial}；{record.message}"
        for record in report.rounds)) if report.rounds or report.round > 1 else ()
    return "\n".join((
        "搜尋停了沒", f"狀態：{states[report.state]}", f"訊息：{report.message}",
        *rounds,
        f"問過 {report.asked} 題；算完 {report.computed} 個；不合法 {report.illegal} 個",
        f"不合法各原因數：{_counts_text(report.illegal_reasons)}",
        f"被淘汰或排除 {sum(report.excluded.values())} 個；各區數：{_counts_text(report.excluded)}",
        f"第一名試算編號：{'沒有' if report.best_trial is None else report.best_trial}；"
        f"篩選分數：{'沒有' if report.best_score is None else report.best_score}",
        f"連續未改善數：{report.streak}；起點排入：{'是' if report.start_enqueued else '否'}",
        f"預算：{report.budget} 題；連續幾個未改善就停：{report.convergence_run}", report.settings_note,
    ))


def _candidate_text(label: str, candidate: CandidateQuality) -> tuple[str, ...]:
    zones = {CandidateStatus.RANKABLE: "可排名", CandidateStatus.ELIMINATED: "淘汰",
             CandidateStatus.NOT_EVALUATED: "未評估", CandidateStatus.NOT_COMPARABLE: "不能同表",
             CandidateStatus.ILLEGAL: "不合法"}
    reasons = {NotFinalReason.REVIEW_PENDING: "複核警戒尚未解除",
               NotFinalReason.EXTERNAL_NOT_CHECKED: "外部底線未檢查",
               NotFinalReason.CALIBRATION_BASELINE: "品質登記簿條目仍為暫定",
               NotFinalReason.NO_FINALIZING_PROCESS: "最終推薦的核定機制尚未建立"}
    lines = [f"{label}：{candidate.message}",
             f"{label}所在區：{'未重排，無區別資料' if candidate.zone is None else zones[candidate.zone]}"]
    if candidate.review_status is not None:
        review = ("目前沒有產生複核警戒；不代表各類都已查完" if candidate.review_status == ReviewStatus.CLEAR
                  else "待複核：有未解除的警戒")
        recommendation = ("不是最終推薦" if candidate.recommendation_status == RecommendationStatus.NOT_FINAL
                          else f"未辨識（{candidate.recommendation_status}）")
        lines.extend((f"{label}複核狀態：{review}", f"{label}推薦狀態：{recommendation}",
                      f"{label}不能當最終推薦的原因：" + "、".join(
                          f"{reason.value}（{reasons[reason]}）" for reason in candidate.not_final_reasons)))
    return tuple(lines)


def _quality_text(report: QualityReport) -> str:
    lines = ["品質合不合格", f"判定：{report.verdict}", f"原因：{report.reason}", report.message]
    header = report.header
    if header is None:
        lines.append("校準狀態與整體驗收：未重排，沒有表頭資料")
    else:
        calibration = "暫定" if header.calibration == "baseline" else "已校準"
        acceptance = {"not_checked": "未檢查", "passed": "通過", "failed": "未通過"}
        lines.extend((f"校準狀態：{calibration}；{header.calibration_note}",
                      f"整體驗收：{acceptance[header.overall_acceptance]}；{header.acceptance_note or '沒有附註'}"))
    lines.extend(_candidate_text("原方案", report.original))
    lines.extend(_candidate_text("第一名", report.best))
    return "\n".join(lines)


def _span_text(span: Span, unit: str) -> str:
    return f"{span.low}～{span.high} {unit}"


def _boxes_text(boxes: tuple[Box, ...] | None) -> str:
    if boxes is None:
        return "未限制"
    # 房間座標照原值寫；哪一軸是前後要看前牆，不在這裡替讀者換成方位字。
    return "；".join(f"房間座標 x {_span_text(box.x, '公尺')}、y {_span_text(box.y, '公尺')}、"
                    f"高度 z {_span_text(box.z, '公尺')}" for box in boxes)


def _restrictions_text(report: RestrictionsReport) -> str:
    return "\n".join((
        "限制", f"離牆間隙：{'未限制' if report.wall_gap_m == 0 else f'{report.wall_gap_m} 公尺'}",
        f"禁區：{_boxes_text(report.keep_out) if report.keep_out else '未限制'}",
        f"喇叭可用區：{_boxes_text(report.speaker_areas)}",
        f"型號適用聆聽距離（喇叭聲學中心到主位的三維距離）：{'未限制' if report.listening_range_m is None else _span_text(report.listening_range_m, '公尺')}",
        f"水平夾角：{'未限制' if report.base_angle_deg is None else _span_text(report.base_angle_deg, '度')}",
    ))


def render_text(report: SearchReport) -> str:
    """純中文段落；受控原因碼同句附中文，兩種參考概念明確分開。"""
    unassessed = "尚未評估\n" + "、".join(report.unassessed.items)
    if report.unassessed.angle_note is not None:
        unassessed += "\n" + report.unassessed.angle_note
    recorded = bool(report.timings.search or report.timings.refine)
    refinement = _refinement_text(report.refinement) + "\n" + round_text(report.timings.refine, "細算", recorded=recorded)
    if recorded:
        refinement += "\n" + total_text(report.timings)
    text = "\n\n".join((
        _search_text(report.search) + "\n" + round_text(report.timings.search, "搜尋", recorded=recorded),
        refinement,
        "名次\n" + "\n".join(report.ranks),
        _quality_text(report.quality), placement_text(report.placement), _restrictions_text(report.restrictions), unassessed,
        "範圍標記\n" + report.scope.message,
        "兩種參考分開寫\n" + report.references.original + "\n" + report.references.provisional,
    ))
    return text + "\n"
