"""搜尋收尾的記憶體換接重評；正式結果、兩本帳與外圈結論只有讀取。"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
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


class CrossoverUnverified(Exception):
    """決策紙列出的比較關卡未通過；重評程式本身的錯誤另記失敗。"""


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
            truncated=lower < cap < upper, reason_text="下端不低於 300 Hz，做不出交接帶" if lower >= cap else ""), lower, min(upper, cap)),
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


def _read_result(store: SearchStore, row: RefineRow) -> SchemeResult:
    label = trial_label(row.trial_number)
    try:
        result = SchemeResult.model_validate_json(store.refine_result_path(row.trial_number).read_bytes())
    except ValidationError as error:
        raise CrossoverUnverified(f"{label}：結果讀不回：JSON 資料損壞或欄位不完整") from error
    except (OSError, ValueError) as error:
        raise CrossoverUnverified(f"{label}：結果讀不回：{error}") from error
    for name in ("physics_identity", "program_fingerprint", "purpose_settings"):
        if getattr(result, name) != getattr(store.identity, name):
            raise CrossoverUnverified(f"{label}：身分不合（{name} 與搜尋快照不同）")
    if result.scheme.scheme_id != refine_scheme_id(store.search_id, row.trial_number):
        raise CrossoverUnverified(f"{label}：身分不合（方案代號與試算編號不同）")
    if not result.pairs or any(not pair.report.points for pair in result.pairs):
        raise CrossoverUnverified(f"{label}：結果讀不回：結果沒有逐點能量")
    return result


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


def _self_check(result: SchemeResult, row: RefineRow, evaluator: Evaluator,
                candidate: CandidateEvaluation, pinned: tuple[ComparisonIdentity, ...]) -> None:
    outcome = evaluator.score(result, candidate, pinned)
    if not isinstance(outcome, Scored) or row.total_cost is None or outcome.value.hex() != row.total_cost.hex():
        raise CrossoverUnverified(f"正式接法自檢不等：{trial_label(row.trial_number)} 的總代價與細算帳未逐位相等")


@dataclass(frozen=True)
class Placement:
    speakers: dict[str, tuple[float, float, float]]
    primary: tuple[float, float, float]

    @classmethod
    def of(cls, result: SchemeResult) -> Placement:
        return cls({key: point.as_tuple() for key, point in result.scheme.speakers.items()},
                   result.scheme.receiver_set.primary.position_m)


@dataclass
class VariantScores:
    stitching: Stitching
    pinned: tuple[ComparisonIdentity, ...] | None = None
    identical: bool = True
    reason: str = ""
    ranking: list[CostRow] = field(default_factory=list)
    excluded: list[ExcludedRow] = field(default_factory=list)


def _add_variant(scores: VariantScores, result: SchemeResult, evaluator: Evaluator,
                 official_candidate: CandidateEvaluation, official_pinned: tuple[ComparisonIdentity, ...], *,
                 baseline: bool = False) -> None:
    if scores.stitching.record.reason_text or scores.reason:
        return
    changed = restitch(result, scores.stitching)
    same = same_weights(result, changed)
    scores.identical = scores.identical and same
    candidate = official_candidate if same else evaluator.evaluate(changed)
    if baseline:
        scores.pinned = official_pinned if same else evaluator.pin(changed, candidate)
    if scores.pinned is None:
        raise CrossoverUnverified("此接法的原方案算不出比較身分")
    outcome = evaluator.score(changed, candidate, scores.pinned)
    number = result.origin.trial_number
    if isinstance(outcome, Scored):
        scores.ranking.append(CostRow(trial_number=number, total_cost=outcome.value))
    else:
        reason = {"eliminated": "淘汰", "unassessed": "未評估", "incomparable": "不能同表"}[outcome.zone.value]
        scores.excluded.append(ExcludedRow(trial_number=number, reason_text=reason))


def _variant(scores: VariantScores, official: int | None) -> VariantRecord:
    record = scores.stitching.record
    if record.reason_text or scores.reason:
        return record.model_copy(update={"reason_text": record.reason_text or scores.reason})
    if scores.identical:
        return record.model_copy(update={"reason_text": "與正式接法逐點權重相同，不算另一種比較"})
    ranking, excluded = scores.ranking, scores.excluded
    ranking.sort(key=lambda r: (r.total_cost, r.trial_number != official))
    rank = next((i for i, r in enumerate(ranking, 1) if r.trial_number == official), None)
    reasons = ["此接法可比較的列少於兩列"] if len(ranking) < 2 else []
    if rank is None:
        zone = next(row.reason_text for row in excluded if row.trial_number == official)
        exclusion = "被淘汰" if zone == "淘汰" else zone
        reasons.append(f"正式第一名在此接法{exclusion}，不進此接法的排名")
    return record.model_copy(update={"tested": not reasons, "reason_text": "；".join(reasons),
        "ranking": tuple(ranking), "excluded": tuple(excluded), "official_rank": rank})


def _check_f_s(result: SchemeResult, expected: float) -> None:
    frequencies = tuple(pair.report.top.f_s_hz for pair in result.pairs)
    if any(not math.isfinite(f_s) or f_s <= 0 for f_s in frequencies):
        raise CrossoverUnverified(f"{trial_label(result.origin.trial_number)} f_s 不是有限正數")
    if any(f_s != expected for f_s in frequencies):
        raise CrossoverUnverified("各列或喇叭座位的 f_s 不同")


def _baseline(result: SchemeResult, row: RefineRow, evaluator: Evaluator,
              f_s: float) -> tuple[tuple[ComparisonIdentity, ...], tuple[VariantScores, ...]]:
    candidate = evaluator.evaluate(result)
    pinned = evaluator.pin(result, candidate)
    if pinned is None:
        raise CrossoverUnverified("正式接法的原方案算不出比較身分")
    _self_check(result, row, evaluator, candidate, pinned)
    scores = tuple(VariantScores(s) for s in stitchings(f_s))
    for variant in scores:
        try:
            _add_variant(variant, result, evaluator, candidate, pinned, baseline=True)
        except CrossoverUnverified as error:
            variant.reason = str(error)
    return pinned, scores


def _finish(summary: CrossoverSummary, scores: tuple[VariantScores, ...], official: int | None,
            f_s: float, placements: dict[int | None, Placement]) -> CrossoverSummary:
    variants = tuple(_variant(s, official) for s in scores)
    tested = tuple(v for v in variants if v.tested)
    excluded = tuple(v for v in variants if any(row.trial_number == official for row in v.excluded))
    sensitive = any(v.ranking[0].total_cost < v.ranking[v.official_rank - 1].total_cost
                    for v in tested if v.official_rank is not None)
    reasons = [f"{v.label}：{v.reason_text}" for v in excluded]
    if not tested:
        reasons.append("做不出比較；各接法原因列在下面")
    if f_s >= FEM_GEOMETRIC_CROSSOVER_CAP_HZ and (not sensitive or excluded):
        reasons.append(f"300 Hz 到 f_s（{f_s:g} Hz）之間沒有有限元素，已測接法都碰不到那一段")
    # 看到換人就是敏感（決策紙第 4 條：任何一種已測接法下有別的列更低）；別的接法比不出來只列原因，不蓋掉看到的事實。
    verdict = "sensitive" if sensitive else "unverified" if excluded or not tested or reasons else "stable"
    if verdict == "sensitive":
        variants = tuple(_distances(v, official, placements) for v in variants)
    return summary.model_copy(update={"completed": True, "state": "done", "verdict": verdict,
        "reason_text": "；".join(reasons), "official_best": official, "variants": variants})


def _distances(variant: VariantRecord, official: int | None,
               placements: dict[int | None, Placement]) -> VariantRecord:
    if not variant.tested or variant.ranking[0].trial_number == official:
        return variant
    first, original = placements[variant.ranking[0].trial_number], placements[official]
    return variant.model_copy(update={"speaker_distance_cm": {
        key: math.dist(point, first.speakers[key]) * 100 for key, point in original.speakers.items()},
        "primary_distance_cm": math.dist(original.primary, first.primary) * 100})


def _compute(store: SearchStore, summary: CrossoverSummary, quality_targets_path: Path) -> CrossoverSummary:
    rows = tuple(row for row in read_refinement_rows(store) if row.outcome == "scored")
    if len(rows) < 2:
        raise CrossoverUnverified("有分數的細算列少於兩列，做不出比較")
    registry = load_quality_targets(quality_targets_path)
    try:
        same_settings = _same_settings(registry, store)
    except KeyError as error:
        raise CrossoverUnverified("評分設定與搜尋快照不同") from error
    if not same_settings:
        raise CrossoverUnverified("評分設定與搜尋快照不同")
    if RefineLedger.read(store.refine_ledger_path)[0] != header_for(store):
        raise CrossoverUnverified("細算帳身分與搜尋快照不同")
    baseline = next((row for row in rows if row.trial_number is None), None)
    if baseline is None:
        raise CrossoverUnverified("有分數的細算表缺原方案，無法釘住比較身分")
    evaluator = Evaluator(store, quality_targets_path, registry, load_capabilities(config_path("capabilities.toml")),
                          load_directivity_defaults(config_path("directivity_defaults.toml")))
    result = _read_result(store, baseline)
    f_s = result.pairs[0].report.top.f_s_hz
    _check_f_s(result, f_s)
    pinned, scores = _baseline(result, baseline, evaluator, f_s)
    placements: dict[int | None, Placement] = {None: Placement.of(result)}
    del result
    for row in rows:
        if row.trial_number is None:
            continue
        result = _read_result(store, row)
        _check_f_s(result, f_s)
        official_candidate = evaluator.evaluate(result)
        _self_check(result, row, evaluator, official_candidate, pinned)
        for variant in scores:
            _add_variant(variant, result, evaluator, official_candidate, pinned)
        placements[row.trial_number] = Placement.of(result)
        del result
    official = scored_refinements(rows)[0].trial_number
    return _finish(summary, scores, official, f_s, placements)


def attach_crossover(store: SearchStore, *, status: SearchStatus, quality_targets_path: Path) -> CrossoverSummary:
    """由 auto 在鎖裡呼叫；只有比較關卡記完成的未驗證，重評出錯交給命令列記失敗。"""
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
        except CrossoverUnverified as error:
            summary = summary.model_copy(update={"completed": True, "state": "done", "reason_text": str(error)})
    write_summary(store.path, summary)
    return summary


def record_crossover_error(store: SearchStore, status: SearchStatus, error: BaseException) -> None:
    summary = fresh_summary(store, status)
    try:
        on_disk = read_summary(store.path)
    except (OSError, ValueError):
        on_disk = None
    if on_disk is not None and on_disk.completed and not is_stale(on_disk, summary):
        return
    stopped = isinstance(error, KeyboardInterrupt)
    write_summary(store.path, summary.model_copy(update={"state": "stopped" if stopped else "failed",
        "reason_text": STOPPED_REASON if stopped else f"交接敏感度計算失敗：{type(error).__name__}：{error}"}))
