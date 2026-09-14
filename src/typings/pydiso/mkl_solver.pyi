from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

class MKLPardisoSolver:
    def __init__(
        self,
        A: csr_matrix[np.complex128],
        matrix_type: str | int | None = ...,
        factor: bool = ...,
        verbose: bool = ...,
    ) -> None: ...
    def solve(
        self,
        b: NDArray[np.complex128],
        x: NDArray[np.complex128] | None = ...,
        transpose: bool = ...,
    ) -> NDArray[np.complex128]: ...

def set_mkl_pardiso_threads(num_threads: int | None = ...) -> None: ...
def get_mkl_pardiso_max_threads() -> int: ...
