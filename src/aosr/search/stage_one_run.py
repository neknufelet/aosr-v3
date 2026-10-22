"""固定房間的第一階段搜尋；只按候選合分排名，首批依編號釘比較身分。"""
from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from aosr.config.quality_targets import load_quality_targets
from aosr.reporting.scheme import Scheme
from aosr.reporting.stage_one_result import (
    StageOneComparisonIdentity, StageOneResult, load_stage_one_result, stage_one_comparison_identity,
)
from aosr.search.sampler import RankingZone
from aosr.search import constraints, layout, ledger
from aosr.search.run import (
    CandidateJob, ComputedCandidate, IdentityChanged, IdentityProbe, SearchStatus, _Runner, _adapter,
    _has_candidate_rows, _resume_inputs,
)
from aosr.search.sampler import Excluded, Outcome, Proposal, ReplayMismatch, Scored
from aosr.search.stage_one_worker import StageOneComputed, StageOneComputeFunction, check_stage_one_result
from aosr.search.store import SearchStore, candidate_name, check_search_axis
from aosr.search.timings import WallClock


@dataclass
class _StageOneRunner(_Runner):
    """只替換候選準備與篩選；預算、停止、重播、計時及保存沿用共用迴圈。"""

    stage_compute: StageOneComputeFunction | None = None
    comparison: StageOneComparisonIdentity | None = None

    def baseline(self, *, resume: bool) -> bool:
        self.status = self.status.model_copy(update={"baseline_outcome": "not_computed"})
        if resume:
            if self.status.comparison_trial is None:
                self.restore_missing_comparison()
            else:
                self.restore_comparison()
        self.adapter, enqueued = _adapter(self.store)
        self.status = self.status.model_copy(update={"start_enqueued": enqueued})
        self.save()
        return True

    def _read_comparison(self, number: int, *, label: str | None = None) -> StageOneResult:
        label = label or f"釘住比較身分的試算 {number}"
        try:
            path = self.store.candidate_path(number)
            scheme = Scheme.model_validate_json(self.store.scheme_path_for(path).read_bytes())
            if scheme.scheme_id != f"{self.store.search_id}-trial-{number:06d}":
                raise ValueError("候選代號與釘住的試算編號不同")
            result = load_stage_one_result(path)
            check_stage_one_result(result, CandidateJob(number, scheme, path), self.store.search_id, self.store.identity)
            return result
        except Exception as error:
            raise IdentityChanged(f"{label} 讀回失敗：{error}") from error

    def restore_comparison(self) -> None:
        number = self.status.comparison_trial
        assert number is not None
        self.comparison = stage_one_comparison_identity(self._read_comparison(number))

    def restore_missing_comparison(self) -> None:
        for row in sorted(ledger.read_for(self.store).rows, key=lambda item: item.trial_number):
            if row.outcome != "illegal":
                result = self._read_comparison(row.trial_number, label=f"重推比較身分時讀到的試算 {row.trial_number}")
                self.comparison = stage_one_comparison_identity(result)
                self.status = self.status.model_copy(update={"comparison_trial": row.trial_number})
                return

    def _prepare(self, proposals: Sequence[Proposal], saved: Sequence[ledger.LedgerRow]
                 ) -> tuple[dict[int, Outcome], dict[int, CandidateJob], dict[int, dict[str, float]]]:
        outcomes = {row.trial_number: ledger.row_outcome(row) for row in saved}
        jobs: dict[int, CandidateJob] = {}
        meters: dict[int, dict[str, float]] = {}
        for proposal in proposals:
            number = proposal.trial_number
            if number in outcomes:
                continue
            params = layout.params_from_unit(proposal.params, self.store.settings.layout, project=self.store.project)
            meters[number] = layout.params_to_meters(params)
            placement = layout.place(self.store.project, self.store.settings.layout, params)
            violations = constraints.check(self.store.project, self.store.settings.layout, placement)
            if violations:
                outcomes[number] = constraints.to_illegal(violations)
            else:
                scheme = layout.to_scheme(self.store.project, placement, f"{self.store.search_id}-trial-{number:06d}")
                jobs[number] = CandidateJob(number, scheme, self.store.candidate_path(number))
        return outcomes, jobs, meters

    def _accept(self, result: StageOneComputed, expected: CandidateJob) -> Outcome:
        if result.job != expected:
            raise ValueError("計算交回的工作跟要算的方案不同")
        if not math.isfinite(result.seconds) or result.seconds < 0:
            raise ValueError("計算耗時必須有限且非負")
        if not expected.result_path.is_file():
            raise ValueError("計算沒有保存候選結果檔")
        check_stage_one_result(result.result, expected, self.store.search_id, self.store.identity)
        identity = stage_one_comparison_identity(result.result)
        if self.comparison is None:
            self.comparison = identity
            self.status = self.status.model_copy(update={"comparison_trial": expected.trial_number})
            self.save()  # 先保存釘住的編號再落帳；重播只讀這一份，禁止換人。
        if identity != self.comparison:
            return Excluded(RankingZone.INCOMPARABLE)
        score = result.result.score
        if score.state != "scored":
            return Excluded(RankingZone.UNASSESSED)
        assert score.screening_total is not None
        return Scored(score.screening_total)

    def batch(self, index: int, proposals: Sequence[Proposal], saved: Sequence[ledger.LedgerRow]) -> None:
        outcomes, jobs, meters = self._prepare(proposals, saved)
        reused = {row.trial_number for row in saved}
        by_number = {p.trial_number: p for p in proposals}
        unpinned = self.comparison is None
        for number, outcome in outcomes.items():
            if not unpinned and number not in reused:
                self.record(index, by_number[number], meters[number], outcome, 0., None)
        pending = dict(jobs)
        completed: dict[int, StageOneComputed] = {}
        assert self.stage_compute is not None
        for result in self.stage_compute(tuple(jobs.values()), self.store.settings.max_workers) if jobs else ():
            returned = result.job.trial_number
            if returned is None or returned not in pending:
                raise ValueError("計算交回重複或未要求的試算編號")
            number = returned
            pending.pop(number)
            if unpinned:
                completed[number] = result
            else:
                outcome = self._accept(result, jobs[number])
                self.record(index, by_number[number], meters[number], outcome, result.seconds, candidate_name(number))
                outcomes[number] = outcome
        if pending:
            raise ValueError("計算沒有交回整批候選")
        if unpinned:
            for number in sorted(by_number.keys() - reused):
                if number in completed:
                    result = completed[number]
                    outcomes[number] = self._accept(result, jobs[number])
                    self.record(index, by_number[number], meters[number], outcomes[number], result.seconds, candidate_name(number))
                else:
                    self.record(index, by_number[number], meters[number], outcomes[number], 0., None)
        self.adapter.tell_batch(outcomes)
        self.refresh()
        self.save()


def _runner(store: SearchStore, compute: StageOneComputeFunction, probe: IdentityProbe, registry_path: Path,
            run_date: date, engine_version: str, book: ledger.Ledger, clock: WallClock) -> _StageOneRunner:
    # 基類持有第二階段計算欄；第一階段覆寫 batch 後不會呼叫它，明確拒用而非塞假候選。
    def unused(jobs: Sequence[CandidateJob], workers: int) -> Iterator[ComputedCandidate]:
        raise RuntimeError("第一階段不能呼叫第二階段計算")
    return _StageOneRunner(store, unused, probe, load_quality_targets(registry_path),
        run_date, engine_version, book, SearchStatus(baseline_outcome="not_computed", timed_from_start=True),
        clock, stage_compute=compute)


def _execute(runner: _StageOneRunner, recorded: ledger.LedgerRead | None = None) -> SearchStatus:
    try:
        runner.baseline(resume=recorded is not None)
        index, partial = (0, ()) if recorded is None else runner.replay(recorded)
        return runner.loop(index, partial)
    except (IdentityChanged, ReplayMismatch) as error:
        runner.refresh()
        return runner.save(state="interrupted", message=f"搜尋中斷：{error}")
    except Exception as error:
        runner.refresh()
        runner.save(state="failed", message=f"搜尋失敗：{error}")
        raise


def start_stage_one_search(store: SearchStore, *, compute: StageOneComputeFunction, probe: IdentityProbe,
                           registry_path: Path, run_date: date, engine_version: str) -> SearchStatus:
    """建立候選帳本，排入取樣起點；不做額外求解。"""
    runner = _runner(store, compute, probe, registry_path, run_date, engine_version, ledger.create_for(store), WallClock())
    runner.save()
    return _execute(runner)


def resume_stage_one_search(store: SearchStore, *, compute: StageOneComputeFunction, probe: IdentityProbe,
                            registry_path: Path, run_date: date, engine_version: str) -> SearchStatus:
    """接續先核快照與帳本，再讀釘住的結果；缺檔與身分不合一律中斷。"""
    try:
        previous = SearchStatus.model_validate_json(store.status_path.read_bytes())
    except FileNotFoundError:
        previous = SearchStatus(timed_from_start=not _has_candidate_rows(store))
    except (OSError, ValueError) as error:
        raise ValueError(f"狀態檔讀不回來（{error}），不能判斷這次搜尋停了沒，拒絕接續") from error
    if previous.state != "running":
        raise ValueError(f"搜尋已停止（{previous.state}），不能接續")
    runner = _runner(store, compute, probe, registry_path, run_date, engine_version, ledger.Ledger(store.ledger_path), WallClock())
    runner.status = previous.model_copy(update={"baseline_outcome": "not_computed"})
    try:
        recorded = _resume_inputs(store)
        check_search_axis(store.project)
    except (OSError, ValueError) as error:
        return runner.save(state="interrupted", message=f"快照或帳本讀回失敗：{error}")
    return _execute(runner, recorded)
