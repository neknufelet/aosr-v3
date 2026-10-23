"""第一階段兩波計算：共用有限元素分片，再只合併與合分。"""
from __future__ import annotations

import math
import sys
import time
from collections.abc import Callable, Generator, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from threading import Lock
from typing import TypeAlias

from aosr.reporting.result import ResultOrigin
from aosr.reporting.stage_one_result import StageOneResult, load_stage_one_result
from aosr.search.run import CandidateJob, ComputeFailed, IdentityChanged
from aosr.search.store import SearchIdentity
from aosr.search.worker import SubprocessCompute, _Active, _check_exit, _stop_all, _watch
from aosr.search.stage_one_time_limit import StageOneTimeLimit, TimeLimitTracker


@dataclass(frozen=True)
class StageOneComputed:
    """完整第一階段結果與第二波候選自己的牆鐘秒數。"""

    job: CandidateJob
    result: StageOneResult
    seconds: float


StageOneComputeFunction: TypeAlias = Callable[[Sequence[CandidateJob], int], Iterator[StageOneComputed]]
_clock = time.monotonic


class CandidateTimeLimit(Exception):
    """搜尋成本已超限，保存觀測快照後暫停整批；不屬於計算或身分失敗。"""

    def __init__(self, record: StageOneTimeLimit) -> None:
        super().__init__("單題搜尋成本超過上限")
        self.record = record


class _StampedQueue(Queue[_Active]):
    """完成通知排進佇列那一刻記牆鐘；父行程多久之後才讀出來，不算進候選秒數。

    鍵用工作物件本身的身分（id）：行程已收掉、通知還在佇列時，pid 可能被下一個候選重用。
    """

    def __init__(self) -> None:
        super().__init__()
        self.ended: dict[int, float] = {}

    def put(self, item: _Active, block: bool = True, timeout: float | None = None) -> None:
        self.ended[id(item)] = _clock()
        super().put(item, block, timeout)


def check_stage_one_result(result: StageOneResult, job: CandidateJob, search_id: str, identity: SearchIdentity) -> None:
    """候選出處、代號與計算快照核對；第一階段合分設定由程式指紋涵蓋。"""
    origin = ResultOrigin(kind="search_candidate", search_id=search_id, trial_number=job.trial_number)
    if result.origin != origin or result.scheme.scheme_id != job.scheme.scheme_id:
        raise ComputeFailed("結果出處或方案代號跟工作不同")
    if result.physics_identity != identity.physics_identity or result.program_fingerprint != identity.program_fingerprint:
        raise IdentityChanged(f"試算 {job.trial_number} 的物理身分或整支程式指紋跟搜尋快照不同")


class StageOneCompute:
    """組合既有子行程計算器，重用分片、啟動、監看與停止規則。"""

    def __init__(self, *, capabilities_path: Path, engine_commit: str, search_id: str, fem_root: Path,
                 identity: SearchIdentity,
                 runner: Sequence[str] = (sys.executable, "-m", "aosr.reporting.scheme_cli", "stage-one"),
                 slice_runner: Sequence[str] = (sys.executable, "-m", "aosr.reporting.scheme_cli", "fem-slice"),
                 poll_s: float = .5, lock_fd: int | None = None, search_cap_s: float | None = None) -> None:
        if search_cap_s is not None and (not math.isfinite(search_cap_s) or search_cap_s <= 0):
            raise ValueError("search_cap_s 必須有限且為正數")
        self.search_cap_s = search_cap_s
        self.identity, self.search_id = identity, search_id
        self.processes = SubprocessCompute(capabilities_path=capabilities_path, engine_commit=engine_commit,
            search_id=search_id, fem_root=fem_root, runner=runner, slice_runner=slice_runner, poll_s=poll_s, lock_fd=lock_fd)

    def _result(self, item: _Active, seconds: float) -> StageOneComputed:
        _check_exit(item, f"試算 {item.job.trial_number}")
        try:
            result = load_stage_one_result(item.job.result_path)
        except (OSError, ValueError) as error:
            raise ComputeFailed(f"試算 {item.job.trial_number} 結果讀回失敗：{error}") from error
        check_stage_one_result(result, item.job, self.search_id, self.identity)
        return StageOneComputed(item.job, result, seconds)

    def __call__(self, jobs: Sequence[CandidateJob], max_workers: int) -> Generator[StageOneComputed, None, None]:
        if type(max_workers) is not int or max_workers < 1:
            raise ValueError("max_workers 必須為正整數")
        if not jobs:
            return
        if any(job.trial_number is None or job.shift_name is not None for job in jobs):
            raise ValueError("第一階段搜尋只收候選試算")
        if self.search_cap_s is not None:
            yield from self._capped(jobs, max_workers, self.search_cap_s)
            return
        parts_by_job = self.processes._parts_by_job(jobs, max_workers)
        pending = iter(zip(jobs, parts_by_job, strict=True))
        active: list[_Active] = []
        starts: dict[int, float] = {}
        completed = _StampedQueue()
        order = Lock()
        exhausted = False
        try:
            while active or not exhausted:
                while len(active) < max_workers and not exhausted:
                    task = next(pending, None)
                    if task is None:
                        exhausted = True
                    else:
                        job, parts = task
                        started = _clock()
                        item = self.processes._start(job, parts)
                        active.append(item)
                        starts[id(item)] = started
                        _watch(item, completed, order)
                if not active:
                    break
                try:
                    finished = completed.get(timeout=self.processes.poll_s)
                except Empty:
                    continue
                result = self._result(finished, completed.ended.pop(id(finished)) - starts.pop(id(finished)))
                _stop_all((finished,))
                active.remove(finished)
                yield result
        finally:
            _stop_all(active)

    def _limited_parts(self, jobs: Sequence[CandidateJob], max_workers: int,
                       tracker: TimeLimitTracker) -> tuple[tuple[Path, ...], ...]:
        def wake(now: float) -> None:
            record = tracker.slices_limit(now)
            if record is not None:
                raise CandidateTimeLimit(record)

        return self.processes._parts_by_job(jobs, max_workers, observe=tracker.observe, wake=wake)

    def _finished(self, item: _Active, seconds: float, record: StageOneTimeLimit | None) -> StageOneComputed:
        """已有超限時跟分片那一段一樣超限優先：失敗的結果不交回，暫停紀錄照存，原錯誤接在例外鏈上。"""
        try:
            return self._result(item, seconds)
        except (ComputeFailed, IdentityChanged) as error:
            if record is None:
                raise
            raise CandidateTimeLimit(record) from error

    def _capped(self, jobs: Sequence[CandidateJob], max_workers: int,
                cap_s: float) -> Generator[StageOneComputed, None, None]:
        tracker = TimeLimitTracker(jobs, cap_s)
        parts = self._limited_parts(jobs, max_workers, tracker)
        pending = iter(zip(jobs, parts, strict=True))
        active: list[_Active] = []
        starts: dict[int, float] = {}
        completed = _StampedQueue()
        order = Lock()
        exhausted = False
        try:
            while active or not exhausted:
                while len(active) < max_workers and not exhausted:
                    if tracker.candidates_limit(active, starts, completed.ended, _clock(), include_queued=True) is not None:
                        break
                    task = next(pending, None)
                    if task is None:
                        exhausted = True
                    else:
                        job, paths = task
                        started = _clock()
                        item = self.processes._start(job, paths)
                        active.append(item)
                        starts[id(item)] = started
                        _watch(item, completed, order)
                if not active:
                    break
                try:
                    finished = completed.get(timeout=self.processes.poll_s)
                except Empty:
                    finished = None
                now = _clock()
                record = tracker.candidates_limit(active, starts, completed.ended, now, finished=finished)
                result = None
                if finished is not None:
                    result = self._finished(finished, completed.ended.pop(id(finished)) - starts.pop(id(finished)), record)
                    tracker.accept(finished, result.seconds)
                    if record is not None:
                        record = tracker.snapshot("candidates", now)
                    _stop_all((finished,))
                    active.remove(finished)
                if record is not None:
                    _stop_all(active)
                if result is not None:
                    yield result
                if record is not None:
                    raise CandidateTimeLimit(record)
        finally:
            _stop_all(active)
