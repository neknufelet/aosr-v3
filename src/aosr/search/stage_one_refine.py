"""固定選席的一批驗證軸細算；第一階段狀態與搜尋狀態分開保存。"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Literal

from aosr.reporting.stage_one_result import (StageOneComparisonIdentity, StageOneResult,
    identity_differences, stage_one_comparison_identity)
from aosr.search.refine_run import _job
from aosr.search.run import (CandidateJob, ComputeFailed, IdentityChanged, IdentityProbe, SearchStatus,
    _check_identity, _resume_inputs)
from aosr.search.stage_one_refine_record import (PENDING_CHECK, RefinedSeat, StageOneRefineStatus,
    read_stage_one_refined_result, save_stage_one_refine)
from aosr.search.stage_one_refine_stop import StageOneRefineStopped
from aosr.search.stage_one_selection import select_stage_one_candidates
from aosr.search.stage_one_selection_contract import big_peak
from aosr.search.stage_one_worker import StageOneComputeFunction, check_stage_one_result
from aosr.search.store import SearchStore


def _prerequisites(store: SearchStore) -> None:
    if store.settings.stage is None:
        raise ValueError("只收第一階段搜尋資料夾")
    if store.settings.stage_one_refine is None:
        raise ValueError("這個第一階段搜尋沒有設定細算（stage_one_refine），不能細算")
    status = SearchStatus.model_validate_json(store.status_path.read_bytes())
    if status.state not in ("converged", "budget_exhausted"):
        raise ValueError(f"請先把搜尋跑完；目前狀態是 {status.state}")


@dataclass
class _Refiner:
    store: SearchStore
    status: StageOneRefineStatus
    records: dict[int, StageOneResult] = field(default_factory=dict)
    seconds: dict[int, float | None] = field(default_factory=dict)
    note: str = ""

    def save(self, state: Literal["selected", "done", "interrupted", "failed"], message: str) -> StageOneRefineStatus:
        """每次依同一份已核對結果重建；完成順序不決定比較身分或排名。"""
        pinned_trial = min(self.records) if self.records else None
        pinned = stage_one_comparison_identity(self.records[pinned_trial]) if pinned_trial is not None else None
        results = tuple(self._seat(seat.trial_number, seat.error, pinned) for seat in self.status.selection.candidates)
        top = {seat.trial_number for seat in self.status.selection.candidates if "top_n" in seat.roles}
        order = sorted((seat for seat in results if seat.trial_number in top and seat.state == "scored"),
                       key=lambda seat: (seat.score, seat.trial_number))
        ranks = {seat.trial_number: rank for rank, seat in enumerate(order, 1)}
        self.status = self.status.model_copy(update={"state": state, "message": f"{message}{self.note}；{PENDING_CHECK}",
            "comparison_trial": pinned_trial, "comparison_identity": pinned,
            "results": tuple(seat.model_copy(update={"refine_rank": ranks.get(seat.trial_number)}) for seat in results)})
        return save_stage_one_refine(self.store, self.status)

    def _seat(self, number: int, error: str | None, pinned: StageOneComparisonIdentity | None) -> RefinedSeat:
        seat = RefinedSeat(trial_number=number, result_file=self.store.refine_result_path(number).relative_to(self.store.path).as_posix(),
            seconds=self.seconds.get(number))
        if error is not None:
            return seat.model_copy(update={"state": "unreadable", "reason": error})
        result = self.records.get(number)
        if result is None:
            return seat
        differences = identity_differences(pinned, stage_one_comparison_identity(result)) if pinned is not None else ()
        values = {"big_peak_db": big_peak(result)}
        if differences:
            return seat.model_copy(update=values | {"state": "incomparable", "reason": "不能同表：" + "、".join(differences)})
        if result.score.state != "scored":
            return seat.model_copy(update=values | {"state": "unassessed", "reason": "不可估：" +
                "、".join(code.value for code in result.score.reason_codes)})
        return seat.model_copy(update=values | {"state": "scored", "score": result.score.screening_total})

    def restore(self) -> list[CandidateJob]:
        """已核對的結果直接用；壞掉、放錯或缺席的結果重算。"""
        jobs = []
        for seat in self.status.selection.candidates:
            if seat.error is not None:
                continue
            try:
                self.records[seat.trial_number] = read_stage_one_refined_result(self.store, seat.trial_number)
            except (OSError, ValueError, ComputeFailed, IdentityChanged):
                jobs.append(_job(self.store, seat.trial_number))
        return jobs


def refine_stage_one_search(store: SearchStore, *, compute: StageOneComputeFunction, probe: IdentityProbe,
                            registry_path: Path, run_date: date, engine_version: str,
                            keep_stop_marker: bool = False) -> StageOneRefineStatus:
    """前提與選席變動先拒收，開始後僅寫第一階段細算摘要與驗證結果。"""
    if keep_stop_marker and store.refine_stop_path.exists():
        raise StageOneRefineStopped()
    _prerequisites(store)
    selection = select_stage_one_candidates(store, registry_path=registry_path)
    path = store.path / "stage_one_refine.json"
    previous = StageOneRefineStatus.model_validate_json(path.read_bytes()) if path.exists() else None
    if previous is not None and previous.selection != selection:
        raise ValueError("挑席跟上次不同，搜尋帳本或設定被改過")
    status = StageOneRefineStatus(selection=selection, results=(), state="selected", message=PENDING_CHECK)
    runner = _Refiner(store, status, seconds={seat.trial_number: seat.seconds for seat in previous.results} if previous else {})
    try:
        _check_identity(probe(), store.identity, "當前身分")
        try:
            _resume_inputs(store)
        except (OSError, ValueError) as error:
            raise IdentityChanged(f"搜尋快照或帳本不同：{error}") from error
        if engine_version != store.identity.program_fingerprint:
            raise IdentityChanged("engine_version 與搜尋快照的 program_fingerprint 不同")
        if store.refine_stop_path.exists():
            if keep_stop_marker:
                raise StageOneRefineStopped()
            store.refine_stop_path.unlink()
            runner.note = "；已刪除殘留的細算停止記號"
        runner.save("selected", "已挑席")
        jobs = runner.restore()
        if jobs:
            store.refine_dir.mkdir(exist_ok=True)
            _compute_batch(runner, jobs, compute)
        return runner.save("done", "細算完成")
    except StageOneRefineStopped:
        raise
    except (KeyboardInterrupt, IdentityChanged) as error:
        return runner.save("interrupted", f"細算中斷：{error}")
    except Exception as error:
        return runner.save("failed", f"細算失敗：{type(error).__name__}：{error}")


def _compute_batch(runner: _Refiner, jobs: list[CandidateJob], compute: StageOneComputeFunction) -> None:
    """單批兩波由計算器負責；每份完成立即保存秒數，供中斷接續。"""
    expected = {job.trial_number: job for job in jobs}
    completed: set[int] = set()
    for item in compute(jobs, runner.store.settings.max_workers):
        number = item.job.trial_number
        if number is None or number in completed or expected.get(number) != item.job:
            raise ComputeFailed("計算交回重複或非本批的工作")
        check_stage_one_result(item.result, item.job, runner.store.search_id, runner.store.identity)
        result = read_stage_one_refined_result(runner.store, number)
        if result != item.result:
            raise ComputeFailed("交回結果與存檔不同")
        runner.records[number] = result
        runner.seconds[number] = item.seconds
        completed.add(number)
        runner.save("selected", "細算進行中")
    if completed != set(expected):
        raise ComputeFailed("計算未交回全部候選")
