"""MKL 預載邊界與 pydiso 真實求解考卷。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from aosr import runtime

# 這是雙精度、四階且對角占優的小題；實測殘差在機器精度附近。留到 1e-12 是容納
# BLAS 實作差異，但仍比這類良態題可接受的數值誤差嚴四個數量級。
MAX_RELATIVE_RESIDUAL = 1e-12


def test_preload_then_pydiso_solves_complex_symmetric_system() -> None:
    """拿掉 RTLD_GLOBAL 或載錯檔，pydiso import 會在真正解題前失敗。"""
    loaded = runtime.preload_mkl()

    from pydiso.mkl_solver import MKLPardisoSolver

    matrix = csr_matrix(
        np.asarray(
            [
                [4.0 + 0.5j, -1.0 + 0.2j, 0.0, 0.0],
                [-1.0 + 0.2j, 4.5 - 0.1j, -0.5j, 0.0],
                [0.0, -0.5j, 5.0 + 0.3j, -0.7],
                [0.0, 0.0, -0.7, 3.5 + 0.4j],
            ],
            dtype=np.complex128,
        )
    )
    expected = np.asarray(
        [1.0 + 0.5j, -0.25 + 0.75j, 0.5 - 0.2j, 1.25 + 0.1j],
        dtype=np.complex128,
    )
    right_hand_side = np.asarray(matrix @ expected, dtype=np.complex128)
    solver = MKLPardisoSolver(matrix, matrix_type="complex_symmetric", factor=True)
    solved = np.asarray(solver.solve(right_hand_side), dtype=np.complex128)
    reconstructed = np.asarray(matrix @ solved, dtype=np.complex128)
    residual = np.linalg.norm(reconstructed - right_hand_side)
    relative_residual = float(residual / np.linalg.norm(right_hand_side))

    assert loaded == Path(sys.prefix) / "lib" / "libmkl_rt.so.3"
    assert relative_residual < MAX_RELATIVE_RESIDUAL


def test_missing_required_mkl_library_fails_at_the_injected_prefix(
    tmp_path: Path,
) -> None:
    """不得搜尋其他路徑或退回別的 solver。"""
    fake_prefix = tmp_path / "venv-without-mkl"
    expected_path = fake_prefix / "lib" / "libmkl_rt.so.3"

    with pytest.raises(FileNotFoundError, match="required MKL runtime library") as caught:
        runtime.preload_mkl(prefix=fake_prefix)

    assert str(expected_path) in str(caught.value)
