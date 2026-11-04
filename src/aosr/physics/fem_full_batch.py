"""完整解批次：輸入原序回收；每個 spawn 工人只建一個自己的 FullSolver。"""
from __future__ import annotations

import multiprocessing
import os
import threading
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack, contextmanager
from functools import partial
from itertools import islice
from multiprocessing.synchronize import Event
from types import TracebackType
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from aosr import runtime
from aosr.physics.fem_reduced import FullOperators, FullSolver
from aosr.physics.fem_reduced_boundary import BoundaryImpedances

if TYPE_CHECKING:
    from pydiso.mkl_solver import MKLPardisoSolver

FullSolveJob = tuple[float, BoundaryImpedances]
# 每行程一份的槽；只由工人行程的初始化函式寫入，父行程不設。
_WORKER_SLOT: object = None


def checked_workers(workers: int) -> int:
    """完整解的工人數必須是正整數；布林也拒收。"""
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers 必須是正整數")
    return workers


def _spawn_ready() -> None:
    """讓執行器在開閘前真的開齊工人。"""


def worker_state[T](kind: type[T]) -> T:
    """取回本工人的工廠產物；槽只在工人行程裡被設。"""
    if not isinstance(_WORKER_SLOT, kind):
        raise RuntimeError("工人區域槽尚未初始化或型別不符")
    return _WORKER_SLOT


def _watch_parent(parent_pid: int) -> None:
    """約每秒偵測父行程死亡（含開閘前），偵測到就退出；卡在原生呼叫時不保證。"""
    tick = threading.Event()
    while os.getppid() == parent_pid:
        tick.wait(1.0)
    os._exit(1)


def _spawn_initialize(factory: Callable[[], object], gate: Event, parent_pid: int) -> None:
    """父行程開齊工人前不准完成任何工作，避免執行器提早重用閒置工人。"""
    global _WORKER_SLOT
    threading.Thread(target=_watch_parent, args=(parent_pid,), daemon=True).start()
    gate.wait()
    _WORKER_SLOT = factory()


@contextmanager
def spawn_pool(workers: int, *, factory: Callable[[], object]) -> Iterator[ProcessPoolExecutor]:
    """以 spawn（獨立啟動）開工人；工人跟父行程同一份環境，這裡不讀寫環境。

    單緒要由父行程（啟動腳本或 child_process_env(threads=1) 開的子行程）自己釘；
    JAX 兩格與 CPU 三格也直接繼承。工人區域槽只在工人行程裡被設。
    執行器以原例外傳回工作錯誤，不重試；正常等待回收，出錯取消可取消的未開始工作並回傳。
    正在算的工作不會被中斷，可能在背景算完；錯誤返回不代表工人已全部收回。
    """
    checked_workers(workers)
    context = multiprocessing.get_context("spawn")
    gate = context.Event()
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=context,
                               initializer=partial(_spawn_initialize, factory, gate, os.getpid()))
    try:
        try:
            ready = [pool.submit(_spawn_ready) for _ in range(workers)]
        finally:
            gate.set()
        for future in ready:
            future.result()
        yield pool
    except BaseException:
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True)


def ordered_map[T, R](pool: ProcessPoolExecutor, function: Callable[[T], R], jobs: Sequence[T], *,
                      pending: int) -> Iterator[R]:
    """最多保留 pending 件未回收工作，照輸入順序回傳，避免完整場在父行程堆滿記憶體。"""
    remaining = iter(jobs)
    futures = deque(pool.submit(function, job) for job in islice(remaining, pending))
    try:
        while futures:
            yield futures.popleft().result()
            for job in islice(remaining, 1):
                futures.append(pool.submit(function, job))
    finally:
        for future in futures:
            future.cancel()


class _WorkerSolver(FullSolver):
    """所有工人以整份工作的第一題為共同稀疏錨點，避免分流漏掉非零位置變更。"""

    def __init__(self, operators: FullOperators, limit: float, first: FullSolveJob) -> None:
        super().__init__(operators, max_relative_residual=limit)
        self._first = first

    def _refactored(self, system: csr_matrix[np.complex128]) -> MKLPardisoSolver:
        # 放在工作本體才組裝：錯誤經 future 原樣回傳，不在行程初始化時被包成 BrokenProcessPool。
        if self._solver is None:
            runtime.preload_mkl()
            runtime.set_pardiso_threads()
            from pydiso.mkl_solver import MKLPardisoSolver as solver_type

            self._anchor = self.system(*self._first)
            self._solver = solver_type(self._anchor, matrix_type="complex_symmetric", factor=False)
        return super()._refactored(system)


def _init_worker(operators: FullOperators, limit: float, first: FullSolveJob) -> _WorkerSolver:
    return _WorkerSolver(operators, limit, first)


def _solve_worker(job: FullSolveJob) -> NDArray[np.complex128]:
    worker = worker_state(_WorkerSolver)
    before = worker.solves
    try:
        return worker.solve(*job)
    except Exception as error:
        # 殘差檢查失敗之前，FullSolver 已完成分解並計一次；保留原例外型別與訊息。
        setattr(error, "_aosr_full_solves", worker.solves - before)
        raise


class FullSolveBatch:
    """給算子、殘差上限與工人數，以 map 解一批（頻率、牆阻抗）。

    ``workers=1`` 只呼叫同一個 ``FullSolver.solve``；多行程只平行完整解，
    ``solver`` 保留父行程的 system 與累計解次，供建模貪婪殘差與帳本沿用。
    同一區段可 map 多輪；工人、稀疏錨點與求解器一直重用，離開區段時收回。
    多行程批次出錯關閉後不可再 map；正在算的工作不會被中斷，可能在背景算完。
    """

    def __init__(self, operators: FullOperators, *, max_relative_residual: float, workers: int = 1) -> None:
        self._workers = checked_workers(workers)
        self._operators = operators
        self._limit = max_relative_residual
        self.solver = FullSolver(operators, max_relative_residual=max_relative_residual)
        self._stack = ExitStack()
        self._pool: ProcessPoolExecutor | None = None
        self._failed = False

    def __enter__(self) -> FullSolveBatch:
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                 traceback: TracebackType | None) -> None:
        self._stack.__exit__(exc_type, exc, traceback)

    def map(self, jobs: Sequence[FullSolveJob]) -> Iterator[NDArray[np.complex128]]:
        """照輸入順序回 (喇叭, N)，每個成功回收的頻點只計一次完整解。"""
        if self._failed:
            raise RuntimeError("完整解批次已因錯誤關閉，不能再次 map")
        if self._workers == 1:
            for job in jobs:
                yield self.solver.solve(*job)
            return
        if not jobs:
            return
        if self._pool is None:
            self._pool = self._stack.enter_context(spawn_pool(self._workers, factory=partial(
                _init_worker, self._operators, self._limit, jobs[0])))
        try:
            for solution in ordered_map(self._pool, _solve_worker, jobs, pending=self._workers):
                self.solver.solves += 1
                yield solution
        except Exception as error:
            self._failed = True
            completed = getattr(error, "_aosr_full_solves", 0)
            if isinstance(completed, int):
                self.solver.solves += completed
            self._stack.__exit__(type(error), error, error.__traceback__)
            raise
