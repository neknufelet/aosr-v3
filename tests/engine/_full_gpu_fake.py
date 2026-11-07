"""無顯卡考卷用的單精度稀疏分解器；真算 complex64，不代答完整解。"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import SuperLU, splu


class SingleFactor:
    def __init__(self, a: csr_matrix[np.complex64], b: NDArray[np.complex64]) -> None:
        self.a, self.b = a, b
        self.factor: SuperLU | None = None
        self.plans = 0
        self.factorizations = 0
        self.calls = 0
        self.freed = False

    def plan(self) -> None:
        self.plans += 1

    def reset_operands(self, *, a: csr_matrix[np.complex64] | None = None,
                       b: NDArray[np.complex64] | None = None) -> None:
        if a is not None:
            self.a = a
        if b is not None:
            self.b = b

    def factorize(self) -> None:
        assert self.a.dtype == np.complex64 and self.b.dtype == np.complex64
        self.factor = splu(self.a.tocsc())
        self.factorizations += 1

    def solve(self) -> NDArray[np.complex64]:
        assert self.factor is not None and not self.freed
        self.calls += 1
        return np.asarray(self.factor.solve(self.b), dtype=np.complex64)

    def free(self) -> None:
        assert not self.freed
        self.freed = True
