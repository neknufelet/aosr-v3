"""模態求解邊界：原樣失敗、唯一分解路線、形狀與一般網格個數估計。"""
from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, eigs
from typing import Literal

from aosr import runtime
from aosr.geometry.shoebox import Room, Wall
from aosr.geometry.shoebox_mesh import ShoeboxMesh, generate_shoebox_mesh
from aosr.physics import fem_modal
from aosr.physics.fem_helmholtz import assemble_p2_operators
from aosr.physics.fem_modal import FemModalSpectrum, ModalSolverOptions, solve_fem_modes
from aosr.physics.modal_convention import ModalKind
from tests.engine.test_fem_modal import C, RHO, ROUNDING


@pytest.fixture
def mesh() -> ShoeboxMesh:
    return generate_shoebox_mesh(Room(1.5, 1.4, 1.3), max_frequency_hz=150.0,
                                 elements_per_wavelength=3, sound_speed_m_s=C, random_seed=1)


def _run(mesh: ShoeboxMesh, *, options: ModalSolverOptions = ModalSolverOptions(),
         shifts: tuple[float, ...] | None = None, counts: tuple[int, ...] | None = None,
         retain: bool = False) -> FemModalSpectrum:
    return solve_fem_modes(mesh, wall_impedances={wall: None for wall in Wall.all()},
                           density_kg_m3=RHO, sound_speed_m_s=C, frequency_max_hz=150.0,
                           options=options, shifts_hz=shifts, modes_per_shift=counts,
                           retain_shapes=retain)


def test_weyl_default_and_explicit_shift_counts_agree(mesh: ShoeboxMesh) -> None:
    default = _run(mesh)
    explicit = _run(mesh, shifts=tuple(s.frequency_hz for s in default.shifts),
                     counts=tuple(s.requested for s in default.shifts))
    assert [row.omega for row in default.solutions] == [row.omega for row in explicit.solutions]
    assert any(row.kind is ModalKind.STATIC for row in default.solutions)
    assert default.modal_table


def test_retained_shapes_have_unit_mass_and_static_is_constant(mesh: ShoeboxMesh) -> None:
    spectrum = _run(mesh, retain=True)
    operators = assemble_p2_operators(mesh)
    for row in spectrum.solutions:
        assert row.shape is not None
        assert not row.shape.flags.writeable
        assert np.vdot(row.shape, operators.mass @ row.shape) == pytest.approx(1, rel=ROUNDING)
        if row.kind is ModalKind.STATIC:
            assert np.all(row.shape == row.shape[0])


def test_only_pydiso_symmetric_factor_and_linear_operator(mesh: ShoeboxMesh, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime.preload_mkl()
    from pydiso import mkl_solver
    from pydiso.mkl_solver import MKLPardisoSolver

    factored = []
    iterated = []

    class SpySolver(MKLPardisoSolver):
        def __init__(self, A: csr_matrix[np.complex128], matrix_type: str | int | None = None,
                     factor: bool = True, verbose: bool = False) -> None:
            assert matrix_type == "complex_symmetric" and factor
            assert abs(A - A.T).max() / abs(A).max() < ROUNDING
            factored.append(A.shape)
            super().__init__(A, matrix_type=matrix_type, factor=factor, verbose=verbose)

    def inspect_eigs(operator: LinearOperator, *, k: int, which: Literal["LM"], tol: float,
                     maxiter: int, v0: NDArray[np.complex128]) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
        assert isinstance(operator, LinearOperator)
        assert which == "LM"
        iterated.append(operator.shape)
        return eigs(operator, k=k, which=which, tol=tol, maxiter=maxiter, v0=v0)

    monkeypatch.setattr(mkl_solver, "MKLPardisoSolver", SpySolver)
    monkeypatch.setattr(fem_modal, "eigs", inspect_eigs)
    spectrum = _run(mesh)
    assert factored == iterated == [(2 * spectrum.degrees_of_freedom,) * 2 for _ in spectrum.shifts]


@pytest.mark.parametrize("stage", ["factor", "eigs"])
def test_solver_errors_propagate_unchanged(mesh: ShoeboxMesh, monkeypatch: pytest.MonkeyPatch, stage: str) -> None:
    runtime.preload_mkl()
    from pydiso import mkl_solver

    failure = RuntimeError("原始求解錯誤，不准改寫或採部分結果")

    def fail(*args: object, **kwargs: object) -> None:
        raise failure

    if stage == "factor":
        monkeypatch.setattr(mkl_solver, "MKLPardisoSolver", fail)
    else:
        monkeypatch.setattr(fem_modal, "eigs", fail)
    with pytest.raises(RuntimeError) as caught:
        _run(mesh)
    assert caught.value is failure


@pytest.mark.parametrize("shifts,counts", [((0.0,), (4,)), ((25.0, 25.0), (4, 4)),
                                           ((25.0,), ()), ((25.0,), (0,)), ((math.nan,), (4,))])
def test_invalid_shift_plan_rejected(mesh: ShoeboxMesh, shifts: tuple[float, ...], counts: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        _run(mesh, shifts=shifts, counts=counts)


@pytest.mark.parametrize("value", [0.0, -1.0, math.nan, math.inf])
def test_invalid_physical_inputs_rejected(mesh: ShoeboxMesh, value: float) -> None:
    with pytest.raises(ValueError):
        solve_fem_modes(mesh, wall_impedances={wall: None for wall in Wall.all()},
                        density_kg_m3=value, sound_speed_m_s=C, frequency_max_hz=150.0)


def test_unexpected_growing_branch_is_rejected() -> None:
    # 保證不把成長根以 abs(Im ω) 冒充衰減根。
    from aosr.physics.modal_convention import modal_quantities

    with pytest.raises(ValueError, match="增長"):
        modal_quantities(100 - 10j, zero_rad_s=1e-6)


def test_near_degeneracy_uses_maximum_cross_shift_matching() -> None:
    vector = np.ones(2, dtype=np.complex128)
    roots = [fem_modal._Root(complex(omega), vector, shift, 0.0, 0.0)
             for omega, shift in [(10.0, 0), (10.00000009, 0),
                                   (10.000000055, 1), (10.00000012, 1)]]
    kept = fem_modal._deduplicate(roots, zero=1e-12, tolerance=1e-8)
    assert [row.omega for row in kept] == [roots[0].omega, roots[1].omega]
