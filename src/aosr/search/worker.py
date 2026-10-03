"""產品計算：先共用有限元素分片，再每個候選獨立子行程，完成一個交回一個。

沒有退回機制：失敗就報錯，不重試、不換解法。單緒與網頁一致，避免數值庫
依執行緒數改變最後一位。停止整個行程群組，讓工作派生的子行程也一起停止。
父行程的通知執行緒只等待退出、不求解；完成通知排隊，呼叫端暫停讀取也不打亂順序。
結果檔 timings 與帳本秒數只記第二波候選子行程自己的時間；共用有限元素時間不分攤、不平均。
"""

from __future__ import annotations

import math
import os
import signal
import subprocess
import sys
from collections.abc import Generator, Sequence
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from typing import TypeVar
from uuid import uuid4

from aosr.config.frequency_axis import LowFrequencyAxis, low_frequency_axis_frequencies
from aosr.reporting.result import ResultOrigin, SchemeResult
from aosr.reporting.scheme_cli import PROGRAM_CHANGED_EXIT, PROGRAM_CHANGED_MARKER
from aosr.runtime import child_process_env
from aosr.search.run import CandidateJob, ComputedCandidate, ComputeFailed, IdentityChanged
from aosr.search.store import JSON_SUFFIX, SearchIdentity, SearchStore


@dataclass
class _Active:
    job: CandidateJob
    process: subprocess.Popen[bytes]
    stderr_path: Path
    watcher: Thread | None = None


@dataclass
class _Slice:
    slice_index: int
    process: subprocess.Popen[bytes]
    stderr_path: Path
    watcher: Thread | None = None


_Process = TypeVar("_Process", _Active, _Slice)


def _notify_finished(item: _Process, completed: Queue[_Process]) -> None:
    """持續接收完成通知；不能等呼叫端讀下一筆時才猜退出先後。"""
    item.process.wait()
    completed.put(item)


def _stop_all(active: Sequence[_Active | _Slice]) -> None:
    """即使群組首領已結束仍殺整群，避免派生行程在錯誤或關閉後繼續計算。"""
    for item in active:
        try:
            os.killpg(item.process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    for item in active:
        item.process.wait()
        if item.watcher is not None:
            item.watcher.join()


def _watch(item: _Process, completed: Queue[_Process]) -> None:
    item.watcher = Thread(target=_notify_finished, args=(item, completed), daemon=True)
    item.watcher.start()


def _check_exit(item: _Active | _Slice, label: str) -> None:
    code = item.process.returncode
    if code == 0:
        return
    stderr = item.stderr_path.read_text(encoding="utf-8", errors="replace")
    message = f"{label} 離開碼 {code}：" + "\n".join(stderr.splitlines()[-8:])
    if code == PROGRAM_CHANGED_EXIT and PROGRAM_CHANGED_MARKER in stderr.splitlines():
        raise IdentityChanged(message)
    raise ComputeFailed(message)


class SubprocessCompute:
    """兩波計算介面；取樣、帳本與排名都由呼叫端負責。

    交回的 seconds 僅是第二波結果檔 timings.total_s，共用分片時間不算入候選秒數。
    """

    def __init__(self, *, capabilities_path: Path, engine_commit: str, search_id: str, fem_root: Path,
                 runner: Sequence[str] = (sys.executable, "-m", "aosr.reporting.scheme_cli", "run"),
                 slice_runner: Sequence[str] = (sys.executable, "-m", "aosr.reporting.scheme_cli", "fem-slice"),
                 poll_s: float = 0.5, lock_fd: int | None = None) -> None:
        if not math.isfinite(poll_s) or poll_s <= 0:
            raise ValueError("poll_s 必須有限且為正數")
        self.capabilities_path = capabilities_path
        self.engine_commit = engine_commit
        self.search_id = search_id
        self.fem_root = fem_root
        self.runner = tuple(runner)
        self.slice_runner = tuple(slice_runner)
        self.poll_s = poll_s
        self.lock_fd = lock_fd

    def _spawn(self, command: Sequence[str], stderr_path: Path) -> subprocess.Popen[bytes]:
        with stderr_path.open("wb") as stderr:
            return subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=stderr,
                                    env=child_process_env(threads=1), start_new_session=True,
                                    pass_fds=() if self.lock_fd is None else (self.lock_fd,),
                                    cwd=Path(__file__).resolve().parents[3])

    def _start(self, job: CandidateJob, parts: Sequence[Path] = ()) -> _Active:
        # 子行程的工作目錄是 repo 根，相對路徑在那裡會指到別處，一律先轉成絕對路徑。
        result_path = job.result_path.resolve()
        scheme_path = SearchStore.scheme_path_for(result_path)
        stderr_path = SearchStore.stderr_path_for(result_path)
        scheme_path.write_text(job.scheme.model_dump_json() + "\n", encoding="utf-8")
        command = [*self.runner, str(scheme_path), "--out", str(result_path),
                   "--engine-commit", self.engine_commit, "--search-id", self.search_id,
                   "--capabilities", str(self.capabilities_path.resolve())]
        if job.trial_number is not None:
            command.extend(("--trial-number", str(job.trial_number)))
        if parts:
            command.extend(("--fem-parts", *(str(path) for path in parts)))
        process = self._spawn(command, stderr_path)
        return _Active(job, process, stderr_path)

    def _slices(self, jobs: Sequence[CandidateJob], max_workers: int) -> tuple[Path, ...]:
        """整批先寫方案；全部分片成功、整群收乾淨後才准開始候選報表。"""
        schemes = tuple(SearchStore.scheme_path_for(job.result_path.resolve()) for job in jobs)
        for job, path in zip(jobs, schemes, strict=True):
            path.write_text(job.scheme.model_dump_json() + "\n", encoding="utf-8")
        axis = jobs[0].scheme.scene.low_frequency_axis or LowFrequencyAxis.SEARCH
        count = min(max_workers, len(low_frequency_axis_frequencies(axis)[0]))
        self.fem_root.mkdir(parents=True, exist_ok=True)
        batch = self.fem_root.resolve() / uuid4().hex
        batch.mkdir()
        parts = tuple(batch / f"slice-{index}{JSON_SUFFIX}" for index in range(count))
        active: list[_Slice] = []
        completed: Queue[_Slice] = Queue()
        try:
            for index, part in enumerate(parts):
                command = [*self.slice_runner, *(str(path) for path in schemes),
                           "--slice", str(index), "--slices", str(count), "--out", str(part),
                           "--capabilities", str(self.capabilities_path.resolve()),
                           "--engine-commit", self.engine_commit]
                stderr_path = SearchStore.stderr_path_for(part)
                item = _Slice(index, self._spawn(command, stderr_path), stderr_path)
                active.append(item)
                _watch(item, completed)
            remaining = len(active)
            while remaining:
                try:
                    finished = completed.get(timeout=self.poll_s)
                except Empty:
                    continue
                if finished.process.returncode != 0:
                    _stop_all(active)
                    _check_exit(finished, f"分片 {finished.slice_index}")
                remaining -= 1
            return parts
        finally:
            _stop_all(active)

    def _result(self, item: _Active) -> ComputedCandidate:
        job = item.job
        _check_exit(item, f"試算 {job.trial_number}")
        try:
            result = SchemeResult.model_validate_json(job.result_path.read_bytes())
            origin = ResultOrigin(kind="search_baseline" if job.trial_number is None else "search_candidate",
                                  search_id=self.search_id, trial_number=job.trial_number)
            if result.origin != origin or result.scheme.scheme_id != job.scheme.scheme_id:
                raise ValueError("結果出處或方案代號跟工作不同")
        except (OSError, ValueError) as error:
            raise ComputeFailed(f"試算 {job.trial_number} 結果讀回失敗：{error}") from error
        identity = SearchIdentity(result.physics_identity, result.program_fingerprint, result.purpose_settings)
        return ComputedCandidate(job, result.candidate, result.timings.total_s, identity)

    def __call__(self, jobs: Sequence[CandidateJob], max_workers: int) -> Generator[ComputedCandidate, None, None]:
        if type(max_workers) is not int or max_workers < 1:
            raise ValueError("max_workers 必須為正整數")
        if not jobs:
            return
        parts = self._slices(jobs, max_workers)
        pending = iter(jobs)
        active: list[_Active] = []
        completed: Queue[_Active] = Queue()
        exhausted = False
        try:
            while active or not exhausted:
                while len(active) < max_workers and not exhausted:
                    job = next(pending, None)
                    if job is None:
                        exhausted = True
                    else:
                        item = self._start(job, parts)
                        active.append(item)
                        _watch(item, completed)
                if not active:
                    break
                try:
                    finished = completed.get(timeout=self.poll_s)
                except Empty:
                    continue
                result = self._result(finished)
                _stop_all((finished,))
                active.remove(finished)
                yield result
        finally:
            _stop_all(active)
