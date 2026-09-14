from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

type _FloatArray = NDArray[np.float64]
type _ComplexArray = NDArray[np.complex128]
type _FieldArray = _FloatArray | _ComplexArray

def dot(u: _FieldArray, v: _FieldArray) -> _FieldArray: ...
def grad(u: _FieldArray) -> _FieldArray: ...
