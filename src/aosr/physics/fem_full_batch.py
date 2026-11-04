"""完整解批次：輸入原序回收；每個 spawn 工人只建一個自己的 FullSolver。"""
from __future__ import annotations

from collections.abc import Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from functools import partial
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
    worker = runtime.worker_state(_WorkerSolver)
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
    """

    def __init__(self, operators: FullOperators, *, max_relative_residual: float, workers: int = 1) -> None:
        self._workers = runtime.checked_workers(workers)
        self._operators = operators
        self._limit = max_relative_residual
        self.solver = FullSolver(operators, max_relative_residual=max_relative_residual)
        self._stack = ExitStack()
        self._pool: ProcessPoolExecutor | None = None

    def __enter__(self) -> FullSolveBatch:
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                 traceback: TracebackType | None) -> None:
        self._stack.__exit__(exc_type, exc, traceback)

    def map(self, jobs: Sequence[FullSolveJob]) -> Iterator[NDArray[np.complex128]]:
        """照輸入順序回 (喇叭, N)，每個成功回收的頻點只計一次完整解。"""
        if self._workers == 1:
            for job in jobs:
                yield self.solver.solve(*job)
            return
        if not jobs:
            return
        if self._pool is None:
            self._pool = self._stack.enter_context(runtime.spawn_pool(self._workers, factory=partial(
                _init_worker, self._operators, self._limit, jobs[0])))
        try:
            for solution in runtime.ordered_map(self._pool, _solve_worker, jobs, pending=self._workers):
                self.solver.solves += 1
                yield solution
        except Exception as error:
            completed = getattr(error, "_aosr_full_solves", 0)
            if isinstance(completed, int):
                self.solver.solves += completed
            self._stack.__exit__(type(error), error, error.__traceback__)
            raise
