"""有限元素考卷用的替身解法器：只用 numpy／scipy 的固定算術，每台機器逐位相同。

真的 PARDISO 依 CPU 指令路徑差最後幾位：2026-10-12 雲端 AMD EPYC 9V74 跑出來跟本機
Ryzen 7 5700X 錄的主線能量差末幾位，``MKL_CBWR=COMPATIBLE`` 也消不掉。要跨機器釘
「我們自己的程式（阻抗、組矩陣、右側、取點、能量）跟主線逐位相同」，就把 MKL 換成這個替身；
真的解法器由其他考卷在同一台機器上考。
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix


class StandInSolver:
    """跟 ``pydiso.mkl_solver.MKLPardisoSolver`` 同樣的建構、``refactor``、``solve`` 用法。

    ``solve`` 不是真的解：每一格都吃到整個矩陣與整個右側（各乘一條固定權重再加總），
    再除以對角，所以任何一格矩陣、右側或座位取點變了，每個座位的能量都會變。
    """

    def __init__(self, matrix: csr_matrix[np.complex128], matrix_type: str, factor: bool = True) -> None:
        del matrix_type
        self._matrix: csr_matrix[np.complex128] | None = None
        if factor:
            self.refactor(matrix)

    def refactor(self, matrix: csr_matrix[np.complex128]) -> None:
        self._matrix = matrix.copy()

    def solve(self, load: NDArray[np.complex128]) -> NDArray[np.complex128]:
        if self._matrix is None:
            raise AssertionError("替身解法器還沒 refactor 就 solve")
        matrix = self._matrix
        summary = ((matrix.data * np.linspace(1.0, 2.0, matrix.nnz)).sum() / matrix.nnz
                   + (load * np.linspace(1.0, 2.0, load.size)).sum())
        stand_in: NDArray[np.complex128] = (load + summary) / matrix.diagonal()
        return stand_in
