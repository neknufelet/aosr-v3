"""第二階段子集搜尋迴圈；計算可注入，主行程獨佔取樣與帳本。

主對話判斷：原方案先算一次釘住比較身分，不經取樣器、不佔試算編號。
原方案排不進表就整次失敗；結果保存在搜尋資料夾，供日後並排使用。
主對話修補判斷：上述拒跑改為只有缺類才失敗；被淘汰的原方案仍釘身分並記原因。
主對話判斷：預算算的是要過的題數（含求解前被過濾的不合法擺法），保證一定會停；
狀態另外分開記「算了幾個」「過濾掉幾個（各原因幾個）」。
最後一批原訂幾個＝min(K, 預算 − 已要的題數)。
主對話判斷：照試算編號順序看「算過的」候選（有分數與排名三區，不含不合法），
第一名（最小篩選分數）連續 convergence_run 個都沒被嚴格超過，這一批結束後停（狀態代碼 converged）。
訊息寫「達到停止條件」不寫「已收斂」（2026-10-03，#585）：這是工程停止設定，不代表找到全域最佳；
預算用完時的第一名只稱「本次預算內最佳」。
主對話判斷：停止原因分為達到停止條件、因預算停止、使用者停止、失敗、中斷。
每批開始前核對身分與停止記號；計算丟例外＝整次搜尋失敗、不重試，寫狀態再往外丟。
子行程明確回報計算中身分變更時改記中斷，保留原因；其餘計算例外沿用上述失敗規則。
行程被砍留下 running（進行中）供接續；重播對不上標中斷。
主對話判斷：單行程與多行程逐位相同。取樣器只在整批算完後照試算編號回報；
收斂與最佳的判斷也照試算編號順序；計算完成的先後只影響帳本列的寫入順序。
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Literal, Self, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator

from aosr.config.quality_targets import QualityTargets, load_quality_targets
from aosr.reporting.result import PurposeSettings
from aosr.reporting.scheme import Scheme
from aosr.scoring.contract import CandidateEvaluation
from aosr.scoring.ranking import ComparisonIdentity, RankingContext, comparison_identity_of, rank_candidates
from aosr.search import constraints, layout, ledger
from aosr.search.sampler import Excluded, Illegal, Outcome, Proposal, ReplayMismatch, SamplerAdapter, Scored
from aosr.search.scoring import screening_outcome
from aosr.search.store import SearchIdentity, SearchStore, candidate_name, check_search_axis


class ComputeFailed(Exception):
    """候選計算失敗；不自動重試或換解法。"""


class IdentityChanged(ComputeFailed):
    """子行程用專用離開碼與固定標記回報計算中身分改變。

    交回的計算身分與搜尋快照不同也中斷，禁止混用。
    """


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
    identity: SearchIdentity


Compute: TypeAlias = Callable[[Sequence[CandidateJob], int], Iterator[ComputedCandidate]]
IdentityProbe: TypeAlias = Callable[[], SearchIdentity]
State: TypeAlias = Literal["running", "converged", "budget_exhausted", "user_stopped", "failed", "interrupted"]
RefineState: TypeAlias = Literal["not_started", "running", "stopped", "failed", "interrupted"]
# 細算停止原因用固定代碼，外圈與報告靠代碼分辨，不比對訊息字串（中文對照在 report.py）。
RefineStopReason: TypeAlias = Literal["stable", "refine_budget", "candidates_exhausted", "user_stopped"]


class RefineStatus(BaseModel):
    """細算的獨立進度；每格都有預設，舊搜尋狀態沒有這個子物件也讀得回。"""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    state: RefineState = "not_started"
    stop_reason: RefineStopReason | None = None
    round: int = Field(default=1, ge=1)
    refined: int = Field(default=0, ge=0)
    best: int | Literal["baseline"] | None = None
    best_total_cost: float | None = None
    streak: int = Field(default=0, ge=0)
    message: str = "細算未開始：還沒有任何候選用驗證軸細算"

    @model_validator(mode="after")
    def _reason_only_when_stopped(self) -> Self:
        # 改回進行中卻殘留舊原因、或已停卻沒原因，讀回時就擋下：外圈靠代碼判斷，不准讀錯。
        if (self.state == "stopped") != (self.stop_reason is not None):
            raise ValueError("stop_reason is required when refinement stopped and forbidden otherwise")
        return self


class RoundRecord(BaseModel):
    """回饋重新開搜尋前，保存剛停止的那一輪。"""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    round: int = Field(ge=1, strict=True)
    state: State
    message: str
    asked: int = Field(ge=0, strict=True)
    best_trial: int | None = Field(ge=0, strict=True)
    best_score: float | None


class SearchStatus(BaseModel):
    """原子保存的搜尋狀態；細算與品質合格不混入搜尋停止原因。

    搜尋停止原因（state）與細算狀態分開記，細算狀態不混進搜尋的 state／message。
    """

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
    baseline_reason_codes: tuple[str, ...] = ()
    refine: RefineStatus = RefineStatus()
    round: int = Field(default=1, ge=1, strict=True)
    round_start_trial: int = Field(default=0, ge=0, strict=True)
    rounds: tuple[RoundRecord, ...] = ()


def _status_message(store: SearchStore, status: SearchStatus) -> SearchStatus:
    """一般保存與快照讀失敗共用起點訊息規則，避免漏句或重複加句。"""
    note = "起點不在搜尋範圍或夾角、耳距限制內，沒有排入"
    if layout.standard_start(store.project, store.settings.layout) is None and note not in status.message:
        return status.model_copy(update={"message": status.message + "；" + note})
    return status


def _saved_baseline(job: CandidateJob, pinned: SearchIdentity) -> CandidateEvaluation | None:
    """主對話判斷：原方案是確定的；結果快取讀不回來、代號不同或不帶身分就重算。

    讀得回而身分跟搜尋快照不同就中斷（跟重算後交回的結果走同一道核對），不拿別版程式算的原方案當比較基準。
    """
    try:
        with job.result_path.open(encoding="utf-8") as handle:
            document: object = json.load(handle)
        if not isinstance(document, dict) or "candidate" not in document:
            return None
        candidate = CandidateEvaluation.model_validate(document["candidate"])
        if candidate.candidate_id != job.scheme.scheme_id:
            return None
        physics, program = document["physics_identity"], document["program_fingerprint"]
        if not isinstance(physics, str) or not isinstance(program, str):
            return None  # 身分欄型別壞掉＝快取讀不回，重算，不把搜尋判成中斷。
        saved = SearchIdentity(physics, program, PurposeSettings.model_validate(document["purpose_settings"]))
    except (OSError, ValueError, UnicodeError, KeyError):
        return None
    _check_identity(saved, pinned, "原方案")
    return candidate


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
        elif row.trial_number >= status.round_start_trial:
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


def _check_identity(found: SearchIdentity, pinned: SearchIdentity, label: str) -> None:
    different = [name for name in ("physics_identity", "program_fingerprint", "purpose_settings")
                 if getattr(found, name) != getattr(pinned, name)]
    if different:
        raise IdentityChanged(f"{label} 跟搜尋快照不同：{'、'.join(different)}")


def _stop_message(convergence_run: int) -> str:
    """停止條件的訊息照實寫：連續幾個沒改善是工程停止設定，不是收斂證明。"""
    return f"達到停止條件：連續 {convergence_run} 個候選沒有嚴格改善（暫行的工程停止設定，不代表找到全域最佳）"


def _budget_message(scored: bool) -> str:
    """預算用完：有第一名才說它是本次預算內最佳；沒有任何候選拿到分數就照實說。"""
    if scored:
        return "因預算停止（第一名是本次預算內最佳，不代表找到全域最佳）"
    return "因預算停止（沒有任何候選拿到分數）"


def _check_computed(result: ComputedCandidate, expected: CandidateJob, identity: SearchIdentity) -> None:
    _check_identity(result.identity, identity, "原方案" if expected.trial_number is None else f"試算 {expected.trial_number}")
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
    pending_enqueues: Mapping[int, tuple[dict[str, float], ...]] = field(default_factory=dict)

    def save(self, *, state: State | None = None, message: str | None = None) -> SearchStatus:
        changes: dict[str, object] = {}
        if state is not None:
            changes["state"] = state
        if message is not None:
            changes["message"] = message
        self.status = self.status.model_copy(update=changes)
        self.status = _status_message(self.store, self.status)
        return _write_status(self.store, self.status)

    def baseline(self, *, resume: bool) -> bool:
        job = _baseline_job(self.store)
        candidate = _saved_baseline(job, self.store.identity) if resume else None
        if candidate is None:
            results = iter(self.compute((job,), self.store.settings.max_workers))
            first = next(results, None)
            if first is None:
                raise ValueError("計算沒有交回原方案")
            _check_computed(first, job, self.store.identity)
            if next(results, None) is not None:
                raise ValueError("計算交回重複原方案")
            candidate = first.candidate
        if not self.pin_baseline(candidate, job.scheme):
            return False
        self.adapter, enqueued = _adapter(self.store)
        self.status = self.status.model_copy(update={"start_enqueued": enqueued})
        return True

    def pin_baseline(self, candidate: CandidateEvaluation, scheme: Scheme) -> bool:
        """主對話修補判斷：淘汰不妨礙釘主表；缺類失敗時指明每一類與原因代碼。"""
        outcome, self.pinned = self.screen(candidate, scheme, pinned=None)
        label = "scored" if isinstance(outcome, Scored) else outcome.zone.value if isinstance(outcome, Excluded) else "illegal"
        reasons: tuple[str, ...] = ()
        if not isinstance(outcome, Scored):
            context = RankingContext(purpose=scheme.purpose, receiver_set_fingerprint=scheme.receiver_set.fingerprint,
                                     channel_group_fingerprint=scheme.channel_group.fingerprint,
                                     run_date=self.run_date, engine_version=self.engine_version)
            ranking = rank_candidates([candidate], self.registry, context)
            reasons = tuple(reason.value for row in ranking.eliminated for reason in row.reasons)
            self.pinned = comparison_identity_of(candidate, self.registry, context)
            missing = (tuple(item for row in ranking.eliminated for item in row.missing)
                       + tuple(item for row in ranking.not_evaluated for item in row.missing))
        else:
            missing = ()
        self.status = self.status.model_copy(update={"baseline_outcome": label, "baseline_reason_codes": reasons})
        if self.pinned is None:
            details = "、".join(f"{item.category.value}（{','.join((item.reason.value, *(code.value for code in item.evaluator_reason_codes)))}）"
                               for item in missing)
            self.save(state="failed", message=f"原方案缺類：{details}，所以定不出比較身分")
            return False
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
            _check_computed(result, jobs[result_number], self.store.identity)
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
        from aosr.search.feedback import replay_enqueues

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
        try:
            enqueues = replay_enqueues(self.store, self.status, recorded.rows, len(history))
        except (OSError, ValueError) as error:
            raise ReplayMismatch(f"回饋事件讀回失敗：{error}") from error
        self.adapter.replay(history, enqueues=enqueues)
        self.pending_enqueues = {index: points for index, points in enqueues.items() if index >= len(history)}
        asked = sum(len(proposals) for proposals, _ in history)
        self.status = _progress(self.status.model_copy(update={"asked": asked}), recorded.rows)
        return len(history), partial

    def loop(self, index: int = 0, partial: Sequence[ledger.LedgerRow] = ()) -> SearchStatus:
        settings = self.store.settings
        while self.status.asked < settings.budget:
            # 未完成批不能提前判收斂；先補齊後才按全批編號判斷。
            if not partial and self.status.best_score is not None and self.status.streak >= settings.convergence_run:
                return self.save(state="converged", message=_stop_message(settings.convergence_run))
            if not self.boundary():
                return self.status
            size = min(settings.batch_size, settings.budget - self.status.asked)
            if index in self.pending_enqueues:
                self.adapter.enqueue_between_batches(self.pending_enqueues[index])
            proposals = self.adapter.ask_batch(size)
            try:
                self.check_partial(proposals, partial)
            except ReplayMismatch as error:
                return self.save(state="interrupted", message=f"重播對不上：{error}")
            self.status = self.status.model_copy(update={"asked": self.status.asked + size})
            self.save()
            self.batch(index, proposals, partial)
            if self.status.best_score is not None and self.status.streak >= settings.convergence_run:
                return self.save(state="converged", message=_stop_message(settings.convergence_run))
            index, partial = index + 1, ()
        if not partial and self.status.best_score is not None and self.status.streak >= settings.convergence_run:
            return self.save(state="converged", message=_stop_message(settings.convergence_run))
        return self.save(state="budget_exhausted", message=_budget_message(self.status.best_score is not None))

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
    except IdentityChanged as error:
        runner.refresh()
        return runner.save(state="interrupted", message=f"搜尋中斷：{error}")
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
    previous = SearchStatus()
    try:
        previous = SearchStatus.model_validate_json(store.status_path.read_bytes())
    except FileNotFoundError:
        pass  # 建帳本後、第一次寫狀態前被砍：沒有上一份狀態，照帳本接。
    except (OSError, ValueError) as error:
        # 壞掉或認不得的狀態檔不准當成「進行中」：已失敗或已停的搜尋會被悄悄重試。
        raise ValueError(f"狀態檔讀不回來（{error}），不能判斷這次搜尋停了沒，拒絕接續") from error
    if previous.state != "running":
        raise ValueError(f"搜尋已停止（{previous.state}），不能接續")
    try:
        recorded = _resume_inputs(store)
    except (OSError, ValueError) as error:
        status = previous.model_copy(update={"state": "interrupted", "message": f"快照或帳本讀回失敗：{error}"})
        return _write_status(store, _status_message(store, status))
    try:
        # 擋驗證軸的關後來才加在建資料夾那一步；之前的程式建的資料夾（快照與帳本都對得上）接續時也要過。
        # 讀回已成功（_resume_inputs 核過重開的專案等於 store.project），所以另寫原因，不混成讀回失敗。
        check_search_axis(store.project)
    except ValueError as error:
        status = previous.model_copy(update={"state": "interrupted", "message": f"拒絕接續：{error}"})
        return _write_status(store, _status_message(store, status))
    runner = _new_runner(store, compute, probe, registry_path, run_date, engine_version, ledger.Ledger(store.ledger_path))
    # 從上一份狀態接：原方案或重播之前就停下時，已要題數、起點排入、原方案判定不會被歸零（只影響顯示）。
    runner.status = previous
    try:
        if not runner.baseline(resume=True):
            return runner.status
        try:
            index, partial = runner.replay(recorded)
        except ReplayMismatch as error:
            runner.refresh()
            return runner.save(state="interrupted", message=f"重播對不上：{error}")
        return runner.loop(index, partial)
    except IdentityChanged as error:
        runner.refresh()
        return runner.save(state="interrupted", message=f"搜尋中斷：{error}")
    except Exception as error:
        runner.refresh()
        runner.save(state="failed", message=f"搜尋失敗：{error}")
        raise
