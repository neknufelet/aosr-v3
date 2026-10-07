"""搜尋收尾的記憶體換接重評；正式結果、兩本帳與外圈結論只有讀取。"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from aosr.config.capabilities import CapabilityTable, load_capabilities
from aosr.config.crossover_sensitivity import LEGACY_HALF_WIDTH_OCTAVE
from aosr.config.directivity_defaults import DirectivityDefaults, load_directivity_defaults
from aosr.config.frequency_axis import FEM_GEOMETRIC_CROSSOVER_CAP_HZ
from aosr.config.paths import config_path
from aosr.config.quality_targets import QualityTargets, load_quality_targets
from aosr.config.three_lane_crossover import CROSSOVER_LOWER_FLOOR_HZ
from aosr.reporting.evaluation import reevaluate
from aosr.reporting.result import SchemeResult
from aosr.scoring.contract import CandidateEvaluation
from aosr.scoring.ranking import ComparisonIdentity, RankingContext, comparison_identity_of
from aosr.search.crossover_record import (
    CostRow, CrossoverSummary, ExcludedRow, STOPPED_REASON, VariantRecord, fresh_summary, is_stale,
    read_summary, trial_label, write_summary,
)
from aosr.search.outer_status import attachment_skip_reason
from aosr.search.refine import RefineLedger, RefineRow
from aosr.search.refine_run import header_for
from aosr.search.report import _same_settings
from aosr.search.report_comparison import read_refinement_rows, scored_refinements
from aosr.search.run import SearchStatus
from aosr.search.sampler import Excluded, Scored
from aosr.search.scoring import screening_outcome
from aosr.search.store import SearchStore, refine_scheme_id


@dataclass(frozen=True)
class Stitching:
    record: VariantRecord
    lower_hz: float | None = None
    upper_hz: float = FEM_GEOMETRIC_CROSSOVER_CAP_HZ

    def geo_weight(self, frequency: float) -> float:
        if self.lower_hz is None:
            return 0.0 if frequency <= self.upper_hz else 1.0
        if frequency <= self.lower_hz:
            return 0.0
        if frequency >= self.upper_hz:
            return 1.0
        t = math.log2(frequency / self.lower_hz) / math.log2(self.upper_hz / self.lower_hz)
        return t * t * (3.0 - 2.0 * t)


def stitchings(f_s: float) -> tuple[Stitching, ...]:
    cap, floor = FEM_GEOMETRIC_CROSSOVER_CAP_HZ, CROSSOVER_LOWER_FLOOR_HZ
    lower, upper = f_s / 2**LEGACY_HALF_WIDTH_OCTAVE, f_s * 2**LEGACY_HALF_WIDTH_OCTAVE
    band = f"本房 {lower:g}～{min(upper, cap):g} Hz" if lower < cap else f"下端 {lower:g} Hz 已不低於上限"
    return (
        Stitching(VariantRecord(key="hard", label=f"{cap:g} Hz 硬切",
            basis="依據：現行硬切房接法；不混合兩路能量")),
        Stitching(VariantRecord(key="legacy", label="上一代接法",
            basis=f"依據：上一代 f_s 上下各半個八度；對數頻率平滑；{band}",
            truncated=upper > cap, reason_text="下端不低於 300 Hz，做不出交接帶" if lower >= cap else ""), lower, min(upper, cap)),
        Stitching(VariantRecord(key="wide", label=f"{floor:g}～{cap:g} Hz 平滑",
            basis="依據：現行低 f_s 房間的最寬交接帶；對數頻率平滑"), floor, cap),
    )


def restitch(result: SchemeResult, stitching: Stitching) -> SchemeResult:
    """只複製三格；頂欄仍是正式交接身分，這份結果絕不存檔。"""
    pairs = []
    for pair in result.pairs:
        if pair.report.points is None:
            raise ValueError("結果沒有逐點能量")
        points = []
        for point in pair.report.points:
            wg = 1.0 if point.fem_energy is None else stitching.geo_weight(point.frequency_hz)
            wf = 1.0 - wg
            total = point.geometric_energy if point.fem_energy is None else wf * point.fem_energy + wg * point.geometric_energy
            points.append(point.model_copy(update={"w_fem": wf, "w_geo": wg, "total_energy": total}))
        pairs.append(pair.model_copy(update={"report": pair.report.model_copy(update={"points": tuple(points)})}))
    return result.model_copy(update={"pairs": tuple(pairs)})


def same_weights(a: SchemeResult, b: SchemeResult) -> bool:
    return all(tuple((p.w_fem, p.w_geo) for p in left.report.points or ()) ==
               tuple((p.w_fem, p.w_geo) for p in right.report.points or ())
               for left, right in zip(a.pairs, b.pairs, strict=True))


def _read_results(store: SearchStore, rows: tuple[RefineRow, ...]) -> dict[int | None, SchemeResult]:
    if RefineLedger.read(store.refine_ledger_path)[0] != header_for(store):
        raise ValueError("細算帳身分與搜尋快照不同")
    results: dict[int | None, SchemeResult] = {}
    for row in rows:
        label = trial_label(row.trial_number)
        try:
            result = SchemeResult.model_validate_json(store.refine_result_path(row.trial_number).read_bytes())
        except ValidationError as error:
            raise ValueError(f"{label}：結果讀不回：JSON 資料損壞或欄位不完整") from error
        except (OSError, ValueError) as error:
            raise ValueError(f"{label}：結果讀不回：{error}") from error
        for name in ("physics_identity", "program_fingerprint", "purpose_settings"):
            if getattr(result, name) != getattr(store.identity, name):
                raise ValueError(f"{label}：身分不合（{name} 與搜尋快照不同）")
        if result.scheme.scheme_id != refine_scheme_id(store.search_id, row.trial_number):
            raise ValueError(f"{label}：身分不合（方案代號與試算編號不同）")
        if not result.pairs or any(not pair.report.points for pair in result.pairs):
            raise ValueError(f"{label}：結果沒有逐點能量")
        results[row.trial_number] = result
    return results


@dataclass(frozen=True)
class Evaluator:
    store: SearchStore
    registry_path: Path
    registry: QualityTargets
    capabilities: CapabilityTable
    directivity: DirectivityDefaults

    def evaluate(self, result: SchemeResult) -> CandidateEvaluation:
        return reevaluate(result, quality_targets_path=self.registry_path,
                          capabilities=self.capabilities, directivity=self.directivity)

    def pin(self, result: SchemeResult, candidate: CandidateEvaluation) -> tuple[ComparisonIdentity, ...] | None:
        scheme = result.scheme
        return comparison_identity_of(candidate, self.registry, RankingContext(purpose=scheme.purpose,
            receiver_set_fingerprint=scheme.receiver_set.fingerprint, channel_group_fingerprint=scheme.channel_group.fingerprint,
            run_date=result.run_date, engine_version=self.store.identity.program_fingerprint))

    def score(self, result: SchemeResult, candidate: CandidateEvaluation, pinned: tuple[ComparisonIdentity, ...]) -> Scored | Excluded:
        outcome, _ = screening_outcome(candidate, result.scheme, registry=self.registry, run_date=result.run_date,
            engine_version=self.store.identity.program_fingerprint, pinned=pinned)
        if isinstance(outcome, (Scored, Excluded)):
            return outcome
        raise ValueError("重評回傳了不合法擺位")


def _self_check(results: dict[int | None, SchemeResult], rows: tuple[RefineRow, ...], evaluator: Evaluator) -> None:
    candidates = {number: evaluator.evaluate(result) for number, result in results.items()}
    pinned = evaluator.pin(results[None], candidates[None])
    if pinned is None:
        raise ValueError("正式接法的原方案算不出比較身分")
    for row in rows:
        outcome = evaluator.score(results[row.trial_number], candidates[row.trial_number], pinned)
        if not isinstance(outcome, Scored) or row.total_cost is None or outcome.value.hex() != row.total_cost.hex():
            raise ValueError(f"正式接法自檢不等：{trial_label(row.trial_number)} 的總代價與細算帳未逐位相等")


def _variant(stitching: Stitching, results: dict[int | None, SchemeResult], rows: tuple[RefineRow, ...],
             official: int | None, evaluator: Evaluator) -> VariantRecord:
    record = stitching.record
    if record.reason_text:
        return record
    changed = {number: restitch(result, stitching) for number, result in results.items()}
    if all(same_weights(results[n], result) for n, result in changed.items()):
        return record.model_copy(update={"reason_text": "與正式接法逐點權重相同，不算另一種比較"})
    baseline = evaluator.evaluate(changed[None])
    pinned = evaluator.pin(changed[None], baseline)
    if pinned is None:
        return record.model_copy(update={"reason_text": "此接法的原方案算不出比較身分"})
    ranking, excluded = [], []
    for row in rows:
        number = row.trial_number
        candidate = baseline if number is None else evaluator.evaluate(changed[number])
        outcome = evaluator.score(changed[number], candidate, pinned)
        if isinstance(outcome, Scored):
            ranking.append(CostRow(trial_number=number, total_cost=outcome.value))
        else:
            reason = {"eliminated": "淘汰", "unassessed": "未評估", "incomparable": "不能同表"}[outcome.zone.value]
            excluded.append(ExcludedRow(trial_number=number, reason_text=reason))
    ranking.sort(key=lambda r: r.total_cost)  # 同分保留帳上先後，與 scored_refinements 相同。
    rank = next((i for i, r in enumerate(ranking, 1) if r.trial_number == official), None)
    reason = "" if rank is not None and len(ranking) >= 2 else "此接法沒有至少兩列可比較的分數，或正式第一名未進此表"
    record = record.model_copy(update={"tested": not reason, "reason_text": reason,
        "ranking": tuple(ranking), "excluded": tuple(excluded), "official_rank": rank})
    if ranking and ranking[0].trial_number != official:
        first, original = results[ranking[0].trial_number].scheme, results[official].scheme
        record = record.model_copy(update={"speaker_distance_cm": {
            key: math.dist(point.as_tuple(), first.speakers[key].as_tuple()) * 100 for key, point in original.speakers.items()},
            "primary_distance_cm": math.dist(original.receiver_set.primary.position_m, first.receiver_set.primary.position_m) * 100})
    return record


def _compute(store: SearchStore, summary: CrossoverSummary, quality_targets_path: Path) -> CrossoverSummary:
    rows = tuple(row for row in read_refinement_rows(store) if row.outcome == "scored")
    if len(rows) < 2:
        raise ValueError("有分數的細算列少於兩列，做不出比較")
    registry = load_quality_targets(quality_targets_path)
    try:
        same_settings = _same_settings(registry, store)
    except KeyError:
        same_settings = False
    if not same_settings:
        raise ValueError("評分設定與搜尋快照不同")
    results = _read_results(store, rows)
    if None not in results:
        raise ValueError("有分數的細算表缺原方案，無法釘住比較身分")
    frequencies = {pair.report.top.f_s_hz for result in results.values() for pair in result.pairs}
    if len(frequencies) != 1:
        raise ValueError("各列或喇叭座位的 f_s 不同")
    f_s = results[None].pairs[0].report.top.f_s_hz
    if not math.isfinite(f_s) or f_s <= 0:
        raise ValueError("原方案 f_s 不是有限正數")
    evaluator = Evaluator(store, quality_targets_path, registry, load_capabilities(config_path("capabilities.toml")),
                          load_directivity_defaults(config_path("directivity_defaults.toml")))
    _self_check(results, rows, evaluator)
    official = scored_refinements(rows)[0].trial_number
    variants = tuple(_variant(s, results, rows, official, evaluator) for s in stitchings(f_s))
    tested = tuple(v for v in variants if v.tested)
    sensitive = any(v.official_rank is not None and v.ranking[0].total_cost < v.ranking[v.official_rank - 1].total_cost for v in tested)
    verdict, reason = "stable", ""
    if sensitive:
        verdict = "sensitive"
    elif not tested:
        verdict, reason = "unverified", "做不出比較；各接法原因列在下面"
    elif f_s >= FEM_GEOMETRIC_CROSSOVER_CAP_HZ:
        verdict, reason = "unverified", f"300 Hz 到 f_s（{f_s:g} Hz）之間沒有有限元素，已測接法都碰不到那一段"
    return summary.model_copy(update={"completed": True, "state": "done", "verdict": verdict,
        "reason_text": reason, "official_best": official, "variants": variants})


def attach_crossover(store: SearchStore, *, status: SearchStatus, quality_targets_path: Path) -> CrossoverSummary:
    """由 auto 在鎖裡呼叫；中斷與例外由 CLI 獨立出口處理，重評失敗是已完成的未驗證提醒。"""
    summary = fresh_summary(store, status)
    try:
        previous = read_summary(store.path)
    except (OSError, ValueError):
        previous = None
    if previous is not None and previous.completed and not is_stale(previous, summary):
        return previous
    write_summary(store.path, summary)
    reason = attachment_skip_reason(status.outer.conclusion)
    if reason:
        summary = summary.model_copy(update={"completed": True, "state": "skipped", "reason_text": reason})
    else:
        try:
            summary = _compute(store, summary, quality_targets_path)
        except ValueError as error:
            summary = summary.model_copy(update={"completed": True, "state": "done", "reason_text": str(error)})
    write_summary(store.path, summary)
    return summary


def record_crossover_error(store: SearchStore, status: SearchStatus, error: BaseException) -> None:
    summary = fresh_summary(store, status)
    stopped = isinstance(error, KeyboardInterrupt)
    write_summary(store.path, summary.model_copy(update={"state": "stopped" if stopped else "failed",
        "reason_text": STOPPED_REASON if stopped else f"交接敏感度計算失敗：{error}"}))
