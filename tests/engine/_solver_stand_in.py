"""有限元素考卷用的替身解法器：不做浮點加總，每台機器逐位相同。

真的 PARDISO 依 CPU 指令路徑差最後幾位：2026-10-12 雲端 AMD EPYC 9V74 跑出來跟本機
Ryzen 7 5700X 錄的主線能量差末幾位，``MKL_CBWR=COMPATIBLE`` 也消不掉。要跨機器釘
「我們自己的程式（阻抗、組矩陣、右側、取點、能量）在控制頻點跟主線逐位相同」，就把 MKL 換成這個替身。

這個替身只用在 ``tests/engine/test_fem_complex.py`` 那三題：一題證明送進解法器的矩陣與右側、之後的
取點與能量，在那一間房、六個控制頻點、六面固定實數阻抗下跟主線錄的能量逐位相同；另兩題比構造牆的逐頻
阻抗列送到兩條正式路與分片路彼此逐位、鑰匙跟著翻，不讀主線答案——都是組裝、接線與流程的回歸。它不證明解法器解得對，
不看解法器種類參數（``matrix_type`` 直接丟掉），也不證明任何聲場。真的解法器在雲端由有容差的考卷考：
FEniCS 凍結答案、剛性解析、空氣層對延長房間（登記簿 ``fem_vs_fenics_frozen``、
``fem_rigid_modal_vs_analytic``、``fem_air_gap_vs_extended_room``）。老闆 2026-10-13 只在這個範圍內
有條件追認這組替身；這不是日後紅燈的修法：真求解的考卷紅了，不准自行換替身或放寬容差，先問。
"""
from __future__ import annotations

import hashlib

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix


class StandInSolver:
    """跟 ``pydiso.mkl_solver.MKLPardisoSolver`` 同樣的建構、``refactor``、``solve`` 用法。

    ``solve`` 不是真的解：把矩陣三個陣列與右側的位元組做 SHA-256，取成一個 [1,2) 的複數加到
    右側每一格，再除以對角。矩陣或右側任何一位變了，雜湊就變，解整個換成另一組不相干的數；
    只用位元組雜湊與逐元素加、除（每格標準捨入），不做浮點加總。之後的取點與能量照被測程式算，
    平方後的能量可能把聲壓的末位差捨入掉，所以這不保證「任何一位都紅」。
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
        digest = hashlib.sha256()
        for array in (matrix.data, matrix.indices, matrix.indptr, load):
            digest.update(np.ascontiguousarray(array).tobytes())
        raw = digest.digest()
        shift = complex(1.0 + int.from_bytes(raw[:6], "big") / 2.0**48,
                        1.0 + int.from_bytes(raw[6:12], "big") / 2.0**48)
        stand_in: NDArray[np.complex128] = (load + shift) / matrix.diagonal()
        return stand_in
