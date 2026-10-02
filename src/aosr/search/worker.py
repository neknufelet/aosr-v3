"""產品計算：每個候選獨立子行程，完成一個交回一個。

沒有退回機制：失敗就報錯，不重試、不換解法。單緒與網頁一致，避免數值庫
依執行緒數改變最後一位。停止整個行程群組，讓工作派生的子行程也一起停止。
父行程的通知執行緒只等待退出、不求解；完成通知排隊，呼叫端暫停讀取也不打亂順序。
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
from typing import TYPE_CHECKING

from aosr.reporting.result import ResultOrigin, SchemeResult
from aosr.reporting.scheme_cli import PROGRAM_CHANGED_EXIT, PROGRAM_CHANGED_MARKER
from aosr.runtime import child_process_env
from aosr.search.store import SearchStore

if TYPE_CHECKING:
    from aosr.search.run import CandidateJob, ComputedCandidate


class ComputeFailed(Exception):
    """候選計算失敗；不自動重試或換解法。"""


class IdentityChanged(ComputeFailed):
    """子行程用專用離開碼與固定標記回報計算中身分改變。"""


@dataclass
class _Active:
    job: CandidateJob
    process: subprocess.Popen[bytes]
    stderr_path: Path
    watcher: Thread | None = None


def _notify_finished(item: _Active, completed: Queue[_Active]) -> None:
    """持續接收完成通知；不能等呼叫端讀下一筆時才猜退出先後。"""
    item.process.wait()
    completed.put(item)


def _stop_all(active: Sequence[_Active]) -> None:
    """即使群組首領已結束仍殺整群，避免派生行程在錯誤或關閉後繼續計算。"""
    for item in active:
        try:
            os.killpg(item.process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    for item in active:
        item.process.wait()
        if item.watcher is not None:
            item.watcher.join()


class SubprocessCompute:
    """計算介面；取樣、帳本與排名都由呼叫端負責。"""

    def __init__(self, *, capabilities_path: Path, engine_commit: str, search_id: str,
                 runner: Sequence[str] = (sys.executable, "-m", "aosr.reporting.scheme_cli", "run"),
                 poll_s: float = 0.5) -> None:
        if not math.isfinite(poll_s) or poll_s <= 0:
            raise ValueError("poll_s 必須有限且為正數")
        self.capabilities_path = capabilities_path
        self.engine_commit = engine_commit
        self.search_id = search_id
        self.runner = tuple(runner)
        self.poll_s = poll_s

    def _start(self, job: CandidateJob) -> _Active:
        scheme_path = SearchStore.scheme_path_for(job.result_path)
        stderr_path = SearchStore.stderr_path_for(job.result_path)
        scheme_path.write_text(job.scheme.model_dump_json() + "\n", encoding="utf-8")
        command = [*self.runner, str(scheme_path), "--out", str(job.result_path),
                   "--engine-commit", self.engine_commit, "--search-id", self.search_id,
                   "--capabilities", str(self.capabilities_path)]
        if job.trial_number is not None:
            command.extend(("--trial-number", str(job.trial_number)))
        with stderr_path.open("wb") as stderr:
            process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=stderr,
                                       env=child_process_env(threads=1), start_new_session=True,
                                       cwd=Path(__file__).resolve().parents[3])
        return _Active(job, process, stderr_path)

    def _result(self, item: _Active) -> ComputedCandidate:
        from aosr.search.run import ComputedCandidate

        job, code = item.job, item.process.returncode
        stderr = item.stderr_path.read_text(encoding="utf-8", errors="replace")
        if code != 0:
            message = f"試算 {job.trial_number} 離開碼 {code}：" + "\n".join(stderr.splitlines()[-8:])
            if code == PROGRAM_CHANGED_EXIT and PROGRAM_CHANGED_MARKER in stderr.splitlines():
                raise IdentityChanged(message)
            raise ComputeFailed(message)
        try:
            result = SchemeResult.model_validate_json(job.result_path.read_bytes())
            origin = ResultOrigin(kind="search_baseline" if job.trial_number is None else "search_candidate",
                                  search_id=self.search_id, trial_number=job.trial_number)
            if result.origin != origin or result.scheme.scheme_id != job.scheme.scheme_id:
                raise ValueError("結果出處或方案代號跟工作不同")
        except (OSError, ValueError) as error:
            raise ComputeFailed(f"試算 {job.trial_number} 結果讀回失敗：{error}") from error
        return ComputedCandidate(job, result.candidate, result.timings.total_s)

    def __call__(self, jobs: Sequence[CandidateJob], max_workers: int) -> Generator[ComputedCandidate, None, None]:
        if type(max_workers) is not int or max_workers < 1:
            raise ValueError("max_workers 必須為正整數")
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
                        item = self._start(job)
                        active.append(item)
                        watcher = Thread(target=_notify_finished, args=(item, completed), daemon=True)
                        watcher.start()
                        item.watcher = watcher
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
