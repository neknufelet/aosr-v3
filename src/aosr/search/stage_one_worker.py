"""第一階段兩波計算：共用有限元素分片，再只合併與合分。"""
from __future__ import annotations

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


@dataclass(frozen=True)
class StageOneComputed:
    """完整第一階段結果與第二波候選自己的牆鐘秒數。"""

    job: CandidateJob
    result: StageOneResult
    seconds: float


StageOneComputeFunction: TypeAlias = Callable[[Sequence[CandidateJob], int], Iterator[StageOneComputed]]


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
                 poll_s: float = .5, lock_fd: int | None = None) -> None:
        self.identity, self.search_id = identity, search_id
        self.processes = SubprocessCompute(capabilities_path=capabilities_path, engine_commit=engine_commit,
            search_id=search_id, fem_root=fem_root, runner=runner, slice_runner=slice_runner, poll_s=poll_s, lock_fd=lock_fd)

    def _result(self, item: _Active, started: float) -> StageOneComputed:
        _check_exit(item, f"試算 {item.job.trial_number}")
        seconds = time.monotonic() - started
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
        parts = self.processes._slices(jobs, max_workers)
        pending = iter(jobs)
        active: list[_Active] = []
        starts: dict[int, float] = {}
        completed: Queue[_Active] = Queue()
        order = Lock()
        exhausted = False
        try:
            while active or not exhausted:
                while len(active) < max_workers and not exhausted:
                    job = next(pending, None)
                    if job is None:
                        exhausted = True
                    else:
                        started = time.monotonic()
                        item = self.processes._start(job, parts)
                        active.append(item)
                        starts[item.process.pid] = started
                        _watch(item, completed, order)
                if not active:
                    break
                try:
                    finished = completed.get(timeout=self.processes.poll_s)
                except Empty:
                    continue
                result = self._result(finished, starts.pop(finished.process.pid))
                _stop_all((finished,))
                active.remove(finished)
                yield result
        finally:
            _stop_all(active)
