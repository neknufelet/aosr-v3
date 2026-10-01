"""第二階段子集搜尋迴圈；計算可注入，主行程獨佔取樣與帳本。

主對話判斷：原方案先算一次釘住比較身分，不經取樣器、不佔試算編號。
原方案排不進表就整次失敗；結果保存在搜尋資料夾，供日後並排使用。
主對話判斷：預算算的是要過的題數（含求解前被過濾的不合法擺法），保證一定會停；
狀態另外分開記「算了幾個」「過濾掉幾個（各原因幾個）」。
最後一批原訂幾個＝min(K, 預算 − 已要的題數)。
主對話判斷：照試算編號順序看「算過的」候選（有分數與排名三區，不含不合法），
第一名（最小篩選分數）連續 convergence_run 個都沒被嚴格超過，這一批結束後停、記「已收斂」。
主對話判斷：停止原因分為已收斂、因預算停止、使用者停止、失敗、中斷。
每批開始前核對身分與停止記號；計算丟例外＝整次搜尋失敗、不重試，寫狀態再往外丟。
行程被砍留下 running（進行中）供接續；重播對不上標中斷。
主對話判斷：單行程與多行程逐位相同。取樣器只在整批算完後照試算編號回報；
收斂與最佳的判斷也照試算編號順序；計算完成的先後只影響帳本列的寫入順序。
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

from aosr.config.quality_targets import QualityTargets, load_quality_targets
from aosr.reporting.scheme import Scheme
from aosr.scoring.contract import CandidateEvaluation
from aosr.scoring.ranking import ComparisonIdentity
from aosr.search import constraints, layout, ledger
from aosr.search.sampler import Excluded, Illegal, Outcome, Proposal, ReplayMismatch, SamplerAdapter, Scored
from aosr.search.scoring import screening_outcome
from aosr.search.store import SearchIdentity, SearchStore, candidate_name


@dataclass(frozen=True)
class CandidateJob:
    """交給計算的方案與結果路徑；None 編號是原方案。"""

    trial_number: int | None
    scheme: Scheme
    result_path: Path


@dataclass(frozen=True)
class ComputedCandidate:
    """計算每完成一個就交回一個；結果檔由計算負責保存。"""

    job: CandidateJob
    candidate: CandidateEvaluation
    seconds: float


Compute: TypeAlias = Callable[[Sequence[CandidateJob], int], Iterator[ComputedCandidate]]
IdentityProbe: TypeAlias = Callable[[], SearchIdentity]
State: TypeAlias = Literal["running", "converged", "budget_exhausted", "user_stopped", "failed", "interrupted"]


class SearchStatus(BaseModel):
    """原子保存的搜尋狀態；細算與品質合格不混入搜尋停止原因。"""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    state: State = "running"
    message: str = "搜尋進行中"
    asked: int = Field(default=0, ge=0)
    computed: int = Field(default=0, ge=0)
    illegal: int = Field(default=0, ge=0)
    illegal_reasons: dict[str, int] = Field(default_factory=dict)
    excluded: dict[str, int] = Field(default_factory=dict)
    best_trial: int | None = None
    best_score: float | None = None
    streak: int = Field(default=0, ge=0)
    start_enqueued: bool = False
    baseline_outcome: str = "pending"


def _write_status(store: SearchStore, status: SearchStatus) -> SearchStatus:
    descriptor, name = tempfile.mkstemp(prefix=".status-", dir=store.path)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(status.model_dump_json() + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(store.status_path)  # 同一個資料夾內換名是原子的：讀的人不會看到寫一半的狀態
    finally:
        temporary.unlink(missing_ok=True)
    return status


def _progress(status: SearchStatus, rows: Sequence[ledger.LedgerRow]) -> SearchStatus:
    """主對話判斷：按編號重建最佳與連續未改善數，非法擺法不進收斂。"""
    computed, illegal, streak = 0, 0, 0
    best_trial: int | None = None
    best_score: float | None = None
    reasons: dict[str, int] = {}
    excluded: dict[str, int] = {}
    for row in sorted(rows, key=lambda item: item.trial_number):
        outcome = ledger.row_outcome(row)
        if isinstance(outcome, Illegal):
            illegal += 1
            for reason in outcome.reason.split("+"):
                reasons[reason] = reasons.get(reason, 0) + 1
            continue
        computed += 1
        if isinstance(outcome, Excluded):
            excluded[outcome.zone.value] = excluded.get(outcome.zone.value, 0) + 1
        if isinstance(outcome, Scored) and (best_score is None or outcome.value < best_score):
            best_trial, best_score, streak = row.trial_number, outcome.value, 0
        else:
            streak += 1
    return status.model_copy(update={"computed": computed, "illegal": illegal, "illegal_reasons": reasons,
                                     "excluded": excluded, "best_trial": best_trial,
                                     "best_score": best_score, "streak": streak})


def _adapter(store: SearchStore) -> tuple[SamplerAdapter, bool]:
    adapter = SamplerAdapter(layout.UNIT_SPACE, store.settings.sampler_settings())
    start = layout.standard_start(store.project, store.settings.layout)
    if start is not None:
        adapter.enqueue(layout.unit_from_params(start, store.settings.layout))
    return adapter, start is not None


def _baseline_job(store: SearchStore) -> CandidateJob:
    scheme = Scheme.model_validate(store.project.model_dump() | {"scheme_id": f"{store.search_id}-baseline"})
    return CandidateJob(None, scheme, store.baseline_path)


def _check_computed(result: ComputedCandidate, expected: CandidateJob) -> None:
    if result.job != expected or result.candidate.candidate_id != expected.scheme.scheme_id:
        raise ValueError("計算交回的工作或候選代號跟要算的方案不同")
    if not math.isfinite(result.seconds) or result.seconds < 0.0:
        raise ValueError("計算耗時必須有限且非負")
    if not expected.result_path.is_file():
        raise ValueError("計算沒有保存候選結果檔")


@dataclass
class _Runner:
    store: SearchStore
    compute: Compute
    probe: IdentityProbe
    registry: QualityTargets
    run_date: date
    engine_version: str
    book: ledger.Ledger
    status: SearchStatus
    adapter: SamplerAdapter = field(init=False)
    pinned: tuple[ComparisonIdentity, ...] | None = None

    def save(self, *, state: State | None = None, message: str | None = None) -> SearchStatus:
        changes: dict[str, object] = {}
        if state is not None:
            changes["state"] = state
        if message is not None:
            changes["message"] = message
        self.status = self.status.model_copy(update=changes)
        if (layout.standard_start(self.store.project, self.store.settings.layout) is None
                and "起點不在搜尋範圍內，沒有排入" not in self.status.message):
            self.status = self.status.model_copy(update={"message": self.status.message + "；起點不在搜尋範圍內，沒有排入"})
        return _write_status(self.store, self.status)

    def baseline(self, *, resume: bool) -> bool:
        job = _baseline_job(self.store)
        if resume and job.result_path.is_file():
            with job.result_path.open(encoding="utf-8") as handle:
                document = json.load(handle)
            if not isinstance(document, dict) or "candidate" not in document:
                raise ValueError("原方案結果檔缺少候選包")
            candidate = CandidateEvaluation.model_validate(document["candidate"])
            if candidate.candidate_id != job.scheme.scheme_id:
                raise ValueError("原方案結果檔的候選代號不同")
        else:
            results = iter(self.compute((job,), self.store.settings.max_workers))
            first = next(results, None)
            if first is None:
                raise ValueError("計算沒有交回原方案")
            _check_computed(first, job)
            if next(results, None) is not None:
                raise ValueError("計算交回重複原方案")
            candidate = first.candidate
        outcome, self.pinned = self.screen(candidate, job.scheme, pinned=None)
        label = "scored" if isinstance(outcome, Scored) else outcome.zone.value if isinstance(outcome, Excluded) else "illegal"
        self.status = self.status.model_copy(update={"baseline_outcome": label})
        if not isinstance(outcome, Scored):
            self.save(state="failed", message="原方案排不進表，定不出這次搜尋的比較身分")
            return False
        self.adapter, enqueued = _adapter(self.store)
        self.status = self.status.model_copy(update={"start_enqueued": enqueued})
        return True

    def screen(self, candidate: CandidateEvaluation, scheme: Scheme, *,
               pinned: tuple[ComparisonIdentity, ...] | None
               ) -> tuple[Outcome, tuple[ComparisonIdentity, ...] | None]:
        return screening_outcome(candidate, scheme, registry=self.registry, run_date=self.run_date,
                                 engine_version=self.engine_version, pinned=pinned)

    def boundary(self) -> bool:
        """每批要題前核身分與停止記號；probe（身分探針）也核評分設定。"""
        if self.probe() != self.store.identity:
            self.save(state="interrupted", message="物理身分、整支程式指紋或評分設定跟搜尋快照不同")
            return False
        if self.store.stop_path.exists():
            self.save(state="user_stopped", message="使用者停止")
            return False
        return True

    def batch(self, index: int, proposals: Sequence[Proposal], saved: Sequence[ledger.LedgerRow]) -> None:
        """不合法先寫；合法每完成一個落帳；回報與收斂等到整批完成。"""
        reused = {row.trial_number: row for row in saved}
        outcomes: dict[int, Outcome] = {}
        jobs: dict[int, CandidateJob] = {}
        pending: dict[int, tuple[Proposal, dict[str, float]]] = {}
        outcome: Outcome
        for proposal in proposals:
            number = proposal.trial_number
            if number in reused:
                outcomes[number] = ledger.row_outcome(reused[number])
                continue
            params = layout.params_from_unit(proposal.params, self.store.settings.layout)
            meters = dict(zip(layout.SEARCH_QUANTITIES, (params.front_distance_m, params.spacing_m, params.listening_distance_m), strict=True))
            placement = layout.place(self.store.project, self.store.settings.layout, params)
            violations = constraints.check(self.store.project, self.store.settings.layout, placement)
            if violations:
                outcome = constraints.to_illegal(violations)
                self.record(index, proposal, meters, outcome, 0.0, None)
                outcomes[number] = outcome
            else:
                scheme = layout.to_scheme(self.store.project, placement, f"{self.store.search_id}-trial-{number:06d}")
                jobs[number] = CandidateJob(number, scheme, self.store.candidate_path(number))
                pending[number] = proposal, meters
        for result in self.compute(tuple(jobs.values()), self.store.settings.max_workers) if jobs else ():
            result_number = result.job.trial_number
            if result_number is None or result_number not in pending:
                raise ValueError("計算交回重複或未要求的試算編號")
            _check_computed(result, jobs[result_number])
            outcome, _ = self.screen(result.candidate, result.job.scheme, pinned=self.pinned)
            proposal, meters = pending.pop(result_number)
            self.record(index, proposal, meters, outcome, result.seconds, candidate_name(result_number))
            outcomes[result_number] = outcome
        if pending:
            raise ValueError("計算沒有交回整批候選")
        self.adapter.tell_batch(outcomes)
        self.refresh()
        self.save()

    def record(self, index: int, proposal: Proposal, meters: dict[str, float], outcome: Outcome,
               seconds: float, result_file: str | None) -> None:
        self.book.append(ledger.row_from_outcome(batch_index=index, proposal=proposal, params_m=meters,
                                                outcome=outcome, seconds=seconds, result_file=result_file))

    def refresh(self) -> None:
        self.status = _progress(self.status, ledger.read_for(self.store).rows)

    def replay(self, recorded: ledger.LedgerRead) -> tuple[int, tuple[ledger.LedgerRow, ...]]:
        settings = self.store.settings
        sizes = {i: min(settings.batch_size, settings.budget - i * settings.batch_size)
                 for i in range((settings.budget + settings.batch_size - 1) // settings.batch_size)}
        for row in recorded.rows:
            if row.batch_index not in sizes or not row.batch_index * settings.batch_size <= row.trial_number < row.batch_index * settings.batch_size + sizes[row.batch_index]:
                raise ReplayMismatch("帳本試算編號或批號超出原訂範圍")
        try:
            history, partial = ledger.replay_history(recorded.header, recorded.rows, batch_sizes=sizes)
        except ValueError as error:
            raise ReplayMismatch(f"帳本批次重播對不上：{error}") from error
        self.adapter.replay(history)
        asked = sum(len(proposals) for proposals, _ in history)
        self.status = _progress(self.status.model_copy(update={"asked": asked}), recorded.rows)
        return len(history), partial

    def loop(self, index: int = 0, partial: Sequence[ledger.LedgerRow] = ()) -> SearchStatus:
        settings = self.store.settings
        while self.status.asked < settings.budget:
            # 未完成批不能提前判收斂；先補齊後才按全批編號判斷。
            if not partial and self.status.best_score is not None and self.status.streak >= settings.convergence_run:
                return self.save(state="converged", message="已收斂")
            if not self.boundary():
                return self.status
            size = min(settings.batch_size, settings.budget - self.status.asked)
            proposals = self.adapter.ask_batch(size)
            try:
                self.check_partial(proposals, partial)
            except ReplayMismatch as error:
                return self.save(state="interrupted", message=f"重播對不上：{error}")
            self.status = self.status.model_copy(update={"asked": self.status.asked + size})
            self.save()
            self.batch(index, proposals, partial)
            if self.status.best_score is not None and self.status.streak >= settings.convergence_run:
                return self.save(state="converged", message="已收斂")
            index, partial = index + 1, ()
        return self.save(state="budget_exhausted", message="因預算停止")

    @staticmethod
    def check_partial(proposals: Sequence[Proposal], partial: Sequence[ledger.LedgerRow]) -> None:
        actual = {proposal.trial_number: {key: value.hex() for key, value in proposal.params.items()}
                  for proposal in proposals}
        for row in partial:
            if actual.get(row.trial_number) != row.unit_params_hex:
                raise ReplayMismatch(f"未完成批試算 {row.trial_number} 的參數逐位重播對不上")


def _new_runner(store: SearchStore, compute: Compute, probe: IdentityProbe, registry_path: Path,
                run_date: date, engine_version: str, book: ledger.Ledger) -> _Runner:
    return _Runner(store, compute, probe, load_quality_targets(registry_path), run_date, engine_version,
                   book, SearchStatus())


def start_search(store: SearchStore, *, compute: Compute, probe: IdentityProbe,
                 registry_path: Path, run_date: date, engine_version: str) -> SearchStatus:
    """新帳本、原方案、釘住比較身分、排入起點，再依批次搜尋。"""
    book = ledger.create_for(store)
    runner = _new_runner(store, compute, probe, registry_path, run_date, engine_version, book)
    runner.save()
    try:
        if not runner.baseline(resume=False):
            return runner.status
        return runner.loop()
    except Exception as error:
        runner.refresh()
        runner.save(state="failed", message=f"搜尋失敗：{error}")
        raise


def _resume_inputs(store: SearchStore) -> ledger.LedgerRead:
    recorded = ledger.read_for(store)  # 先核表頭，不能先讀原方案或重播。
    reopened = SearchStore.open(store.path)
    if (reopened.project != store.project or reopened.settings != store.settings
            or reopened.identity != store.identity or reopened.versions != store.versions):
        raise ValueError("搜尋快照跟開啟時不同")
    ledger.read_for(reopened)
    return recorded


def resume_search(store: SearchStore, *, compute: Compute, probe: IdentityProbe,
                  registry_path: Path, run_date: date, engine_version: str) -> SearchStatus:
    """先核表頭與快照、讀回原方案重排；完整批重播、末批逐位核對後只算缺列。"""
    if store.status_path.is_file():
        previous = SearchStatus.model_validate_json(store.status_path.read_bytes())
        if previous.state != "running":
            raise ValueError(f"搜尋已停止（{previous.state}），不能接續")
    try:
        recorded = _resume_inputs(store)
    except (OSError, ValueError) as error:
        return _write_status(store, SearchStatus(state="interrupted", message=f"快照或帳本讀回失敗：{error}"))
    runner = _new_runner(store, compute, probe, registry_path, run_date, engine_version, ledger.Ledger(store.ledger_path))
    try:
        if not runner.baseline(resume=True):
            return runner.status
        try:
            index, partial = runner.replay(recorded)
        except ReplayMismatch as error:
            runner.refresh()
            return runner.save(state="interrupted", message=f"重播對不上：{error}")
        return runner.loop(index, partial)
    except Exception as error:
        runner.refresh()
        runner.save(state="failed", message=f"搜尋失敗：{error}")
        raise
