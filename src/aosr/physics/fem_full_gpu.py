"""cuDSS 單精度分解、CPU 雙精度殘差修正；沿用完整模型組裝與殘差上限（#1030）。"""
from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from aosr.physics import fem_reduced
from aosr.physics.fem_reduced import FullOperators, FullSolver
from aosr.physics.fem_reduced_boundary import BoundaryImpedances

# 2026-11-07：109 房 105,462 自由度實測 1–2 步；沿用實驗的 10 步保護上限。
# 這是失敗截止次數，不是精度門檻；收解仍只看既有 full_solve_max_relative_residual。
MAX_REFINEMENT_STEPS = 10


class SinglePrecisionFactor(Protocol):
    """可替換的單精度分解器；操作順序與 nvmath DirectSolver 相同。"""

    def plan(self) -> None: ...
    def reset_operands(self, *, a: csr_matrix[np.complex64] | None = None,
                       b: NDArray[np.complex64] | None = None) -> None: ...
    def factorize(self) -> None: ...
    def solve(self) -> NDArray[np.complex64]: ...
    def free(self) -> None: ...


FactorFactory = Callable[[csr_matrix[np.complex64], NDArray[np.complex64]], SinglePrecisionFactor]


class NvmathSolverOptions(Protocol):
    """nvmath 1.0.0 選項中本票使用的欄位；上游以整數接收矩陣種類列舉。"""

    sparse_system_type: int


class NvmathMatrixTypes(Protocol):
    """只描述本票使用的複數對稱種類；上游 MatrixType 是 IntEnum。"""

    SYMMETRIC: int


class NvmathAdvanced(Protocol):
    """選用模組的具體介面；只收本票使用的 CPU 單精度運算元與選項。"""

    DirectSolverMatrixType: NvmathMatrixTypes

    def DirectSolver(self, a: csr_matrix[np.complex64], b: NDArray[np.complex64], *,
                     options: NvmathSolverOptions | None = None) -> SinglePrecisionFactor: ...

    # 上游是 dataclass，亦接受位置參數與預設值；此處只開本票使用的具名參數。
    def DirectSolverOptions(self, *, sparse_system_type: int) -> NvmathSolverOptions: ...


def nvmath_factor(a: csr_matrix[np.complex64], b: NDArray[np.complex64]) -> SinglePrecisionFactor:
    """只有選 GPU 並真的解第一題才載入選用依賴；複數對稱不是 Hermitian（共軛對稱）。"""
    if TYPE_CHECKING:
        advanced: NvmathAdvanced
    else:
        import nvmath.sparse.advanced as advanced

    return advanced.DirectSolver(a, b, options=advanced.DirectSolverOptions(
        sparse_system_type=advanced.DirectSolverMatrixType.SYMMETRIC))


class GPUFullSolver(FullSolver):
    """沿用 FullSolver.system；每頻一次分解，全部喇叭以雙精度殘差通過才交回。"""

    def __init__(self, operators: FullOperators, *, max_relative_residual: float,
                 factor_factory: FactorFactory | None = None) -> None:
        super().__init__(operators, max_relative_residual=max_relative_residual)
        self._factory = nvmath_factor if factor_factory is None else factor_factory
        self._factor: SinglePrecisionFactor | None = None
        self._closed = False
        self.refinement_steps: list[int] = []

    def close(self) -> None:
        """釋放顯卡資源；成功、失敗都由擁有者在區段結束時呼叫。"""
        self._closed = True
        if self._factor is not None:
            factor, self._factor = self._factor, None
            factor.free()

    def _factorized(self, system: csr_matrix[np.complex128],
                    loads: NDArray[np.complex128]) -> SinglePrecisionFactor:
        if self._closed:
            raise RuntimeError("完整解顯卡求解器已關閉")
        if self._anchor is not None and not (
                system.shape == self._anchor.shape
                and np.array_equal(system.indptr, self._anchor.indptr)
                and np.array_equal(system.indices, self._anchor.indices)):
            raise ValueError("有限元素矩陣非零位置與錨定矩陣不同")
        a, b = system.astype(np.complex64), np.asfortranarray(loads.astype(np.complex64))
        if self._factor is None:
            self._anchor = system.copy()
            self._factor = self._factory(a, b)
            try:
                self._factor.plan()
            except BaseException:
                self.close()
                raise
        else:
            self._factor.reset_operands(a=a, b=b)
        self._factor.factorize()
        self.solves += 1
        return self._factor

    def solve(self, frequency_hz: float, walls: BoundaryImpedances) -> NDArray[np.complex128]:
        """回 (喇叭, N) complex128；不收斂或非有限數就失敗，不退回 CPU。"""
        system = self.system(frequency_hz, walls)
        loads = np.asfortranarray(np.asarray(list(self._operators.sources.values()), dtype=np.complex128).T)
        factor = self._factorized(system, loads)
        solution = np.asarray(factor.solve(), dtype=np.complex128)
        for steps in range(MAX_REFINEMENT_STEPS + 1):
            residuals = [fem_reduced.relative_residual(system, solution[:, i], loads[:, i])
                         for i in range(loads.shape[1])]
            for name, residual in zip(self._operators.sources, residuals, strict=True):
                if not math.isfinite(residual) or (steps == MAX_REFINEMENT_STEPS and residual > self._limit):
                    raise RuntimeError(f"完整解殘差失常：{frequency_hz} Hz、喇叭 {name}、相對殘差 {residual}")
            if all(residual <= self._limit for residual in residuals):
                self.refinement_steps.append(steps)
                return np.asarray(solution.T, dtype=np.complex128)
            residual_vector = loads - np.asarray(system @ solution, dtype=np.complex128)
            scale = float(np.max(np.abs(residual_vector)))
            factor.reset_operands(b=np.asfortranarray((residual_vector / scale).astype(np.complex64)))
            solution = solution + scale * np.asarray(factor.solve(), dtype=np.complex128)
        raise RuntimeError("完整解修正未收斂")
