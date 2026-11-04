from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

type _FloatArray = NDArray[np.float64]
type _IntArray = NDArray[np.int64]
type _ComplexArray = NDArray[np.complex128]
type _Quadrature = tuple[_FloatArray, _FloatArray]
type _FormFunction = Callable[..., complex | float]

class MeshTet:
    facets: _IntArray
    def __init__(
        self,
        doflocs: _FloatArray = ...,
        t: _IntArray = ...,
        _boundaries: dict[str, _IntArray] | None = ...,
        _subdomains: dict[str, _IntArray] | None = ...,
        elem: type[object] = ...,
        affine: bool = ...,
        sort_t: bool = ...,
        validate: bool = ...,
    ) -> None: ...
    def boundary_facets(self) -> _IntArray: ...

class ElementTetP2: ...

class DofsView:
    def __init__(
        self,
        obj: object = ...,
        nodal_ix: _IntArray | slice = ...,
        facet_ix: _IntArray | slice = ...,
        edge_ix: _IntArray | slice = ...,
        interior_ix: _IntArray | slice = ...,
        nodal_rows: _IntArray | slice = ...,
        facet_rows: _IntArray | slice = ...,
        edge_rows: _IntArray | slice = ...,
        interior_rows: _IntArray | slice = ...,
        doflocs: object = ...,
    ) -> None: ...
    def all(self, key: str | None = ...) -> _IntArray: ...

class Basis:
    N: int
    mesh: MeshTet
    elem: ElementTetP2
    doflocs: _FloatArray
    def __init__(
        self,
        mesh: MeshTet,
        elem: ElementTetP2,
        mapping: object | None = ...,
        intorder: int | None = ...,
        elements: object | None = ...,
        quadrature: _Quadrature | None = ...,
        dofs: object | None = ...,
        disable_doflocs: bool = ...,
    ) -> None: ...
    def probes(self, x: _FloatArray) -> csr_matrix[np.float64]: ...
    def get_dofs(
        self,
        facets: _IntArray | None = ...,
        elements: _IntArray | None = ...,
        nodes: _IntArray | None = ...,
        skip: list[str] | None = ...,
    ) -> DofsView: ...

class FacetBasis:
    def __init__(
        self,
        mesh: MeshTet,
        elem: ElementTetP2,
        mapping: object | None = ...,
        intorder: int | None = ...,
        quadrature: _Quadrature | None = ...,
        facets: object | None = ...,
        dofs: object | None = ...,
        side: int = ...,
        disable_doflocs: bool = ...,
    ) -> None: ...

class BilinearForm:
    def __init__(
        self,
        form: _FormFunction | BilinearForm | None = ...,
        dtype: type[np.float64] | type[np.complex64] = ...,
        nthreads: int = ...,
        **params: object,
    ) -> None: ...

class LinearForm:
    def __init__(
        self,
        form: _FormFunction | LinearForm | None = ...,
        dtype: type[np.float64] | type[np.complex64] = ...,
        nthreads: int = ...,
        **params: object,
    ) -> None: ...

def asm(
    form: BilinearForm,
    *args: Basis | FacetBasis,
    to: object = ...,
    **kwargs: object,
) -> csr_matrix[np.float64]: ...
