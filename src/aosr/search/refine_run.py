"""停下的搜尋用驗證軸細算與重排；只更新細算進度，回饋與外圈另支施工。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Literal

from aosr.config.frequency_axis import LowFrequencyAxis
from aosr.config.quality_targets import QualityTargets, load_quality_targets
from aosr.reporting.scheme import Scheme
from aosr.scoring.contract import CandidateEvaluation
from aosr.scoring.ranking import ComparisonIdentity, RankingContext, comparison_identity_of, rank_candidates
from aosr.search import ledger
from aosr.search.refine import REFINE_LEDGER_VERSION, RefineHeader, RefineLedger, RefineRow, refine_order
from aosr.search.run import (
    CandidateJob, Compute, ComputedCandidate, IdentityChanged, IdentityProbe, RefineState,
    RefineStatus, RefineStopReason, SearchStatus, _check_computed, _check_identity,
    _resume_inputs, _saved_baseline, _write_status,
)
from aosr.search.sampler import Excluded, RankingZone, Scored
from aosr.search.scoring import screening_outcome
from aosr.search.store import SearchStore, refine_result_name, refine_scheme_id


class _BaselineUnavailable(ValueError):
    """原方案缺類，細算不能釘住比較身分。"""


def refinement_status(store: SearchStore) -> SearchStatus:
    """先拒跑再寫入；拒絕時搜尋狀態與細算目錄逐位不動。"""
    status = SearchStatus.model_validate_json(store.status_path.read_bytes())
    if status.state not in ("converged", "budget_exhausted"):
        raise ValueError(f"搜尋狀態是 {status.state}，只准在 converged 或 budget_exhausted 時細算")
    if store.settings.refine is None:
        raise ValueError("搜尋設定沒有 refine（細算設定），拒絕細算")
    if status.refine.state not in ("not_started", "running"):
        raise ValueError(f"細算狀態是 {status.refine.state}，不能接續")
    return status


def header_for(store: SearchStore) -> RefineHeader:
    """指紋沿用搜尋表頭的快照來源；只有細算帳版號不同。"""
    snapshot = ledger.header_for(store)
    fields = {name: getattr(snapshot, name) for name in RefineHeader.model_fields if name != "ledger_version"}
    return RefineHeader.model_validate(fields | {"ledger_version": REFINE_LEDGER_VERSION})


def _job(store: SearchStore, number: int | None) -> CandidateJob:
    project = (store.project if number is None else
               Scheme.model_validate_json(store.scheme_path_for(store.candidate_path(number)).read_bytes()))
    scene = project.scene.model_copy(update={"low_frequency_axis": LowFrequencyAxis.VERIFICATION})
    scheme = Scheme.model_validate(project.model_dump() | {
        "scene": scene.model_dump(), "scheme_id": refine_scheme_id(store.search_id, number),
    })
    return CandidateJob(number, scheme, store.refine_result_path(number))


def _progress(rows: Sequence[RefineRow]) -> RefineStatus:
    best: int | Literal["baseline"] | None = None
    cost: float | None = None
    streak, refined = 0, 0
    for row in rows:
        if row.trial_number is not None:
            refined += 1
        if row.total_cost is not None and (cost is None or row.total_cost < cost):
            best = "baseline" if row.trial_number is None else row.trial_number
            cost, streak = row.total_cost, 0
        elif row.trial_number is not None:
            streak += 1
    return RefineStatus(state="running", refined=refined, best=best, best_total_cost=cost, streak=streak)


def _stop_reason(store: SearchStore, status: RefineStatus, order: Sequence[int]) -> RefineStopReason | None:
    settings = store.settings.refine
    assert settings is not None
    if status.streak >= settings.convergence_run:
        return "stable"
    if status.refined >= settings.budget:
        return "refine_budget"
    if status.refined >= len(order):
        return "candidates_exhausted"
    return None


def _stop_message(status: RefineStatus, reason: RefineStopReason) -> str:
    best = "原方案" if status.best == "baseline" else "沒有可排名的方案" if status.best is None else f"{status.best} 號"
    reasons = {"stable": f"細算第一名 {best} 連續 {status.streak} 個沒被換掉",
               "refine_budget": "用完細算上限", "candidates_exhausted": "沒有候選可以再細算",
               "user_stopped": "使用者停止"}
    return f"細算已停：{reasons[reason]}（暫行設定，不代表細算完成——回饋還沒做）"


@dataclass
class _Refiner:
    store: SearchStore
    compute: Compute
    probe: IdentityProbe
    registry: QualityTargets
    run_date: date
    engine_version: str
    status: SearchStatus
    order: tuple[int, ...] = ()
    rows: list[RefineRow] = field(default_factory=list)
    book: RefineLedger | None = None
    pinned: tuple[ComparisonIdentity, ...] | None = None
    note: str = ""
    loaded: bool = False

    def save(self, *, state: RefineState = "running", message: str = "細算進行中",
             reason: RefineStopReason | None = None) -> SearchStatus:
        progress = (_progress(self.rows) if self.loaded else self.status.refine).model_copy(update={
            "state": state, "stop_reason": reason, "message": message + self.note,
        })
        self.status = self.status.model_copy(update={"refine": progress})
        return _write_status(self.store, self.status)

    def identity(self) -> None:
        _check_identity(self.probe(), self.store.identity, "當前身分")
        try:
            _resume_inputs(self.store)
        except (OSError, ValueError) as error:
            raise IdentityChanged(f"搜尋快照或帳本不同：{error}") from error

    def open_book(self) -> None:
        expected = header_for(self.store)
        if self.store.refine_ledger_path.exists():
            try:
                recorded = RefineLedger.read_status(self.store.refine_ledger_path)
                if recorded.header != expected:
                    different = [name for name in RefineHeader.model_fields
                                 if getattr(recorded.header, name) != getattr(expected, name)]
                    raise ValueError(f"細算帳表頭跟搜尋快照不同：{different}")
                self.rows = list(recorded.rows)
                numbers = [row.trial_number for row in self.rows]
                if numbers and numbers != [None, *self.order[:len(numbers) - 1]]:
                    raise ValueError("細算帳不是原方案與候選細算順序的前段")
                if any(row.round != 1 for row in self.rows):
                    raise ValueError("細算帳輪次不是 1")
            except (OSError, ValueError) as error:
                raise IdentityChanged(f"細算帳讀回失敗：{error}") from error
            self.book = RefineLedger(self.store.refine_ledger_path)
        else:
            self.book = RefineLedger.create(self.store.refine_ledger_path, expected)
        self.loaded = True

    def screen(self, candidate: CandidateEvaluation, job: CandidateJob, seconds: float) -> RefineRow:
        outcome, _ = screening_outcome(candidate, job.scheme, registry=self.registry, run_date=self.run_date,
                                        engine_version=self.engine_version, pinned=self.pinned)
        labels = {RankingZone.ELIMINATED: "excluded", RankingZone.UNASSESSED: "not_evaluated",
                  RankingZone.INCOMPARABLE: "not_comparable"}
        label = "scored" if isinstance(outcome, Scored) else labels[outcome.zone] if isinstance(outcome, Excluded) else "not_evaluated"
        return RefineRow.model_validate({
            "round": 1, "trial_number": job.trial_number, "result_file": refine_result_name(job.trial_number),
            "outcome": label, "total_cost": outcome.value if isinstance(outcome, Scored) else None, "seconds": seconds,
        })

    def pin(self, candidate: CandidateEvaluation, scheme: Scheme) -> None:
        context = RankingContext(purpose=scheme.purpose, receiver_set_fingerprint=scheme.receiver_set.fingerprint,
                                 channel_group_fingerprint=scheme.channel_group.fingerprint,
                                 run_date=self.run_date, engine_version=self.engine_version)
        self.pinned = comparison_identity_of(candidate, self.registry, context)
        if self.pinned is None:
            ranking = rank_candidates([candidate], self.registry, context)
            missing = (tuple(item for row in ranking.eliminated for item in row.missing)
                       + tuple(item for row in ranking.not_evaluated for item in row.missing))
            details = "、".join(f"{item.category.value}（{','.join((item.reason.value, *(code.value for code in item.evaluator_reason_codes)))}）"
                               for item in missing)
            raise _BaselineUnavailable(f"原方案缺類：{details}，所以定不出比較身分")

    def record(self, row: RefineRow, saved: RefineRow | None) -> None:
        if saved is not None:
            if row.model_dump(exclude={"seconds"}) != saved.model_dump(exclude={"seconds"}):
                raise IdentityChanged("重算結果跟細算帳對不上")
            return
        assert self.book is not None
        self.book.append(row)
        self.rows.append(row)

    def batch(self, jobs: Sequence[CandidateJob], saved: dict[int | None, RefineRow]) -> None:
        """核每個交回結果，按細算次序寫完整前段；不讓工作完成先後改變同分與接續。"""
        expected = {job.trial_number: job for job in jobs}
        pending: dict[int | None, ComputedCandidate] = {}
        seen: set[int | None] = set()
        cursor = 0
        for result in self.compute(jobs, self.store.settings.max_workers):
            number = result.job.trial_number
            if number not in expected or number in seen:
                raise ValueError("計算交回重複或未要求的細算編號")
            _check_computed(result, expected[number], self.store.identity)
            seen.add(number)
            pending[number] = result
            while cursor < len(jobs) and jobs[cursor].trial_number in pending:
                item = pending.pop(jobs[cursor].trial_number)
                if item.job.trial_number is None:
                    self.pin(item.candidate, item.job.scheme)
                self.record(self.screen(item.candidate, item.job, item.seconds), saved.get(item.job.trial_number))
                cursor += 1
        if cursor != len(jobs):
            raise ValueError("計算沒有交回整批細算候選")

    def baseline(self, cached: CandidateEvaluation | None) -> None:
        job = _job(self.store, None)
        saved = {row.trial_number: row for row in self.rows}
        if cached is None:
            self.identity()
            self.store.ensure_refine_dir()
            self.batch((job,), saved)
        else:
            self.pin(cached, job.scheme)
            self.record(self.screen(cached, job, saved[None].seconds), saved[None])

    def restore(self) -> dict[int | None, CandidateEvaluation | None]:
        """先核完所有已落帳的快取，才重算讀不回的結果；身分失配不寫新列。"""
        return {row.trial_number: self.read_saved(row.trial_number) for row in self.rows}

    def read_saved(self, number: int | None) -> CandidateEvaluation | None:
        """沿用搜尋的快取核身分；錯誤點名這份細算結果，不把候選誤稱原方案。"""
        try:
            return _saved_baseline(_job(self.store, number), self.store.identity)
        except IdentityChanged as error:
            label = "細算原方案" if number is None else f"細算 {number} 號"
            raise IdentityChanged(str(error).replace("原方案", label, 1)) from error

    def restore_candidates(self, cached: dict[int | None, CandidateEvaluation | None]) -> None:
        for row in tuple(self.rows):
            if row.trial_number is None:
                continue
            job = _job(self.store, row.trial_number)
            candidate = cached[row.trial_number]
            if candidate is None:
                self.identity()
                self.store.ensure_refine_dir()
                self.batch((job,), {row.trial_number: row})
            else:
                self.record(self.screen(candidate, job, row.seconds), row)

    def stop(self, reason: RefineStopReason) -> SearchStatus:
        return self.save(state="stopped", reason=reason, message=_stop_message(_progress(self.rows), reason))

    def loop(self) -> SearchStatus:
        settings = self.store.settings
        assert settings.refine is not None
        while True:
            progress = _progress(self.rows)
            # 半批接續必須補回原批，不在半批上判停止，亦不重新組成 K 個新候選。
            complete = progress.refined % settings.batch_size == 0 or progress.refined >= min(len(self.order), settings.refine.budget)
            reason = _stop_reason(self.store, progress, self.order) if complete else None
            if reason is not None:
                return self.stop(reason)
            self.identity()
            if self.store.refine_stop_path.exists():
                return self.stop("user_stopped")
            start = progress.refined
            end = min((start // settings.batch_size + 1) * settings.batch_size, settings.refine.budget, len(self.order))
            jobs = tuple(_job(self.store, number) for number in self.order[start:end])
            self.store.ensure_refine_dir()
            self.batch(jobs, {})
            self.save()


def refine_search(store: SearchStore, *, compute: Compute, probe: IdentityProbe,
                  registry_path: Path, run_date: date, engine_version: str) -> SearchStatus:
    """只接停下的搜尋；拒跑不寫狀態，執行例外只改 refine（細算）子物件。"""
    previous = refinement_status(store)
    runner = _Refiner(store, compute, probe, load_quality_targets(registry_path), run_date, engine_version, previous)
    try:
        runner.identity()
        runner.order = refine_order(ledger.read_for(store).rows)
        runner.open_book()
        if store.refine_stop_path.exists():
            store.refine_stop_path.unlink()
            runner.note = "；已刪除殘留的細算停止記號"
        runner.save()
        cached = runner.restore()
        runner.baseline(cached.get(None))
        runner.restore_candidates(cached)
        runner.save()
        return runner.loop()
    except IdentityChanged as error:
        return runner.save(state="interrupted", message=f"細算中斷：{error}")
    except _BaselineUnavailable as error:
        return runner.save(state="failed", message=f"細算失敗：{error}")
    except Exception as error:
        runner.save(state="failed", message=f"細算失敗：{error}")
        raise
