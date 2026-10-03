"""報告的帳本名次與擺位並排：純讀取，不推進搜尋或細算。"""

from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import TypeAlias

from pydantic import BaseModel

from aosr.config.paths import config_path
from aosr.config.placement_standards import PlacementStandard, load_placement_standards
from aosr.config.precision_contracts import load_precision_contracts
from aosr.config.quality_targets import QualityTargets
from aosr.reporting.compare import compare_results, comparison_problems
from aosr.reporting.result import SchemeResult
from aosr.scoring.ranking_models import CandidateStatus
from aosr.scoring.recommendation import ReviewStatus
from aosr.search.ledger import Ledger
from aosr.search.refine import RefineLedger, RefineRow, refine_order
from aosr.search.run import SearchStatus
from aosr.search.standards_check import NOTICE, StandardsChecklist, _render_actual, check_placement_standards
from aosr.search.store import FROZEN, SearchStore

ResultReader: TypeAlias = Callable[[Path], SchemeResult | None]
ZONES = {CandidateStatus.ELIMINATED: "淘汰", CandidateStatus.NOT_EVALUATED: "未評估",
         CandidateStatus.NOT_COMPARABLE: "不能同表", CandidateStatus.ILLEGAL: "不合法"}


class PlacementReport(BaseModel):
    """兩份方案共用條文，結果缺席的一邊仍明示沒有檢查。"""

    model_config = FROZEN
    best_label: str
    best: StandardsChecklist | None
    original: StandardsChecklist | None
    clauses: tuple[PlacementStandard, ...]


def default_precision_contracts_path() -> Path:
    """唯一精度契約登記簿；與工作目錄無關，不在搜尋資料夾另開一份。"""
    return Path(__file__).resolve().parents[3] / "blueprint" / "precision_contracts.toml"


def read_refinement_rows(store: SearchStore) -> tuple[RefineRow, ...]:
    return RefineLedger.read(store.refine_ledger_path)[1] if store.refine_ledger_path.exists() else ()


def scored_refinements(rows: tuple[RefineRow, ...]) -> tuple[RefineRow, ...]:
    """全輪次照總代價小的先；穩定排序保留同分時帳上先出現的，跟細算狀態一致。"""
    return tuple(sorted((row for row in rows if row.outcome == "scored" and row.total_cost is not None),
                        key=lambda row: row.total_cost if row.total_cost is not None else float("inf")))


def _review_evidence(store: SearchStore, rows: tuple[RefineRow, ...], registry: QualityTargets,
                     run_date: date, read: ResultReader, same_settings: bool) -> dict[int | None, str]:
    """用品質段的同一排名層看複核與所在區；檔案缺席、身分不合或排名失敗都不猜。"""
    results = {row.trial_number: read(store.refine_result_path(row.trial_number)) for row in rows}
    evidence = {number: "細算結果檔讀不回" for number, result in results.items() if result is None}
    readable = [result for result in results.values() if result is not None]
    if not readable:
        return evidence
    if not same_settings:
        return evidence | {number: "評分設定跟搜尋快照不同，尚未重排" for number, result in results.items()
                           if result is not None}
    if comparison_problems(readable):
        return evidence | {number: "不能同表" for number, result in results.items() if result is not None}
    try:
        ranking = compare_results(readable, quality_targets=registry, run_date=run_date)
    except ValueError as error:
        return evidence | {number: f"排名失敗：{error}" for number, result in results.items() if result is not None}
    for number, result in results.items():
        if result is None:
            continue
        candidate_id = result.candidate.candidate_id
        zone = ranking.status_of(candidate_id)
        if zone != CandidateStatus.RANKABLE:
            evidence[number] = ZONES[zone]
        elif any(row.candidate_id == candidate_id and row.review_status == ReviewStatus.PENDING
                 for row in ranking.rankable):
            evidence[number] = "尚待確認"
    return evidence


def _rank_line(number: int | None, search_ranks: dict[int, int], refined: tuple[RefineRow, ...],
               rows: tuple[RefineRow, ...], pending: int, evidence: dict[int | None, str]) -> str:
    label = "原方案" if number is None else f"{number} 號"
    row = next((row for row in rows if row.trial_number == number), None)
    note = evidence.get(number)
    outcomes = {"excluded": "淘汰", "not_evaluated": "未評估", "not_comparable": "不能同表"}
    if row is not None and row.outcome != "scored":
        return f"{label}：{outcomes[row.outcome]}。"
    if note in ZONES.values():
        return f"{label}：{note}。"
    search = "" if number is None else f"搜尋排名第 {search_ranks[number]}；"
    if row is None:
        return f"{label}：{search}還沒有細算。"
    rank = next(index for index, item in enumerate(refined, start=1) if item.trial_number == number)
    suffix = "" if number is None else f"另有 {pending} 個方案尚未細算。"
    text = f"{label}：{search}在已細算的 {len(refined)} 個方案中，排名第 {rank}。{suffix}"
    return text if note is None else text + note + "。"


def rank_lines(store: SearchStore, status: SearchStatus, rows: tuple[RefineRow, ...],
               registry: QualityTargets, run_date: date, read: ResultReader, *, same_settings: bool) -> tuple[str, ...]:
    """只點名搜尋第一名、細算第一名與原方案；帳本有分數才給名次。"""
    search_rows = Ledger.read(store.ledger_path)[1] if store.ledger_path.exists() else ()
    order = refine_order(search_rows)
    search_ranks = {number: rank for rank, number in enumerate(order, start=1)}
    refined = scored_refinements(rows)
    selected: list[int | None] = list(order[:1])
    if refined and refined[0].trial_number not in selected and refined[0].trial_number is not None:
        selected.append(refined[0].trial_number)
    selected.append(None)
    refined_numbers = {row.trial_number for row in rows}
    pending = sum(number not in refined_numbers for number in order)
    evidence = _review_evidence(store, rows, registry, run_date, read, same_settings)
    lines = []
    for number in selected:
        if number is not None and number not in search_ranks:
            lines.append(f"{number} 號：沒有搜尋分數。")
        elif number is None and None not in refined_numbers and status.baseline_outcome in ZONES:
            lines.append("原方案：" + ZONES[CandidateStatus(status.baseline_outcome)] + "。")
        else:
            lines.append(_rank_line(number, search_ranks, refined, rows, pending, evidence))
    return tuple(lines)


def placement_report(store: SearchStore, status: SearchStatus, rows: tuple[RefineRow, ...],
                     read: ResultReader, contracts_path: Path) -> PlacementReport:
    """細算過就讀細算結果；第一名取細算帳，沒有細算第一名時用搜尋第一名。"""
    refined = scored_refinements(rows)
    number = refined[0].trial_number if refined else status.best_trial
    label = ("細算第一名（原方案）" if number is None else f"細算第一名（{number} 號）") if refined else (
        "沒有第一名" if number is None else f"搜尋第一名（{number} 號）")
    refined_numbers = {row.trial_number for row in rows}
    def result_path(trial: int | None) -> Path:
        if trial in refined_numbers:
            return store.refine_result_path(trial)
        return store.baseline_path if trial is None else store.candidate_path(trial)

    original = read(result_path(None))
    best = read(result_path(number)) if refined or number is not None else None
    standards = load_placement_standards(config_path("placement_standards.toml"))
    boundary = load_precision_contracts(contracts_path)["placement_standard_boundary"].value
    def checklist(result: SchemeResult | None) -> StandardsChecklist | None:
        return None if result is None else check_placement_standards(
            result.scheme, store.settings.layout.front_wall, standards, boundary_rel=boundary)

    original_table, best_table = checklist(original), checklist(best)
    return PlacementReport(best_label=label, best=best_table, original=original_table, clauses=standards.entries)


def _checklist_side(label: str, checklist: StandardsChecklist | None, clause: PlacementStandard,
                    index: int) -> str:
    """說明在條文標題列印一次；這一邊只列判定、實際值，與沒守時才有的附註。"""
    if checklist is None:
        return f"  {label}：結果檔讀不回，沒檢查"
    row = checklist.rows[index]
    actual = "；".join(_render_actual(item) for item in row.actual)
    note = row.description.removeprefix(clause.description).strip()
    return f"  {label}：{row.verdict.label}；{actual}" + (f"；{note}" if note else "")


def placement_text(report: PlacementReport) -> str:
    """每條一段：標題列（條號、說明）、兩邊判定與實際值各一行並排，原文只印一次。"""
    lines = ["擺位標準檢查表", NOTICE]
    for index, clause in enumerate(report.clauses):
        lines.extend((f"{clause.id} | {clause.standard} {clause.clause}（PDF 頁 {clause.pdf_page}） | {clause.description}",
                      _checklist_side(report.best_label, report.best, clause, index),
                      _checklist_side("原方案", report.original, clause, index),
                      f"  原文：{clause.quote}"))
    return "\n".join(lines)
