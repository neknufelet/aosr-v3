"""#1030：真單精度分解、雙精度殘差修正；雲端不需顯卡或 nvmath。"""
from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from aosr.config.fem_reduced import load_reduced_model_settings
from aosr.config.paths import config_path
from aosr.physics.fem_reduced import FullOperators, FullSolver, relative_residual
from aosr.physics.fem_full_batch import FullSolveBatch
from tests.engine._full_gpu_fake import SingleFactor
from tests.engine._parallel_full_solves_probe import WALLS, _operators

if TYPE_CHECKING:
    from aosr.physics.fem_full_gpu import GPUFullSolver

LIMIT = load_reduced_model_settings(config_path("fem_reduced.toml")).full_solve_max_relative_residual


@pytest.fixture(scope="module")
def operators() -> FullOperators:
    return _operators()


def _small(operators: FullOperators, monkeypatch: pytest.MonkeyPatch,
           factor: type[SingleFactor] = SingleFactor) -> tuple[GPUFullSolver, csr_matrix[np.complex128]]:
    from aosr.physics.fem_full_gpu import GPUFullSolver

    # float32 看不見 1e-8，單精度殘差會誤判為零；兩支負載量級不同。
    system = csr_matrix(np.diag(np.array([1 + 1e-8, 2 + 3e-8, 3 + 2e-8], dtype=np.complex128)))
    loads = {"left": np.array([1, 2, 3], dtype=np.complex128),
             "right": np.array([3, 1, 2], dtype=np.complex128) * 1e-4}
    solver = GPUFullSolver(replace(operators, sources=loads), max_relative_residual=LIMIT, factor_factory=factor)
    monkeypatch.setattr(solver, "system", lambda frequency, walls: system)
    return solver, system


def test_refinement_reaches_double_residual_and_direct_solution(operators: FullOperators,
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    """拿掉修正或把殘差降成單精度必紅；不靠 GPU 逐位重現。"""
    solver, system = _small(operators, monkeypatch)
    try:
        answer = solver.solve(40.0, WALLS[0])
        loads = np.array([[1, 2, 3], [3e-4, 1e-4, 2e-4]], dtype=np.complex128)
        assert answer.dtype == np.complex128
        for load, solution in zip(loads, answer, strict=True):
            assert relative_residual(system, solution, load) <= LIMIT
            reference = np.linalg.solve(system.toarray(), load)
            assert np.linalg.norm(solution - reference) / np.linalg.norm(reference) <= np.linalg.cond(system.toarray()) * LIMIT
        assert solver.refinement_steps[0] > 0
        assert solver.solves == 1
    finally:
        solver.close()


def test_gpu_uses_full_solver_assembly_and_reuses_plan(operators: FullOperators) -> None:
    from aosr.physics.fem_full_gpu import GPUFullSolver

    factors: list[SingleFactor] = []

    def factory(a: csr_matrix[np.complex64], b: NDArray[np.complex64]) -> SingleFactor:
        factor = SingleFactor(a, b)
        factors.append(factor)
        return factor

    solver = GPUFullSolver(operators, max_relative_residual=LIMIT, factor_factory=factory)
    cpu = FullSolver(operators, max_relative_residual=LIMIT)
    try:
        for frequency in (40.0, 63.0):
            system = cpu.system(frequency, WALLS[0])
            assert np.array_equal(solver.system(frequency, WALLS[0]).data, system.data)
            answer = solver.solve(frequency, WALLS[0])
            for load, solution in zip(operators.sources.values(), answer, strict=True):
                assert relative_residual(system, solution, load) <= LIMIT
        assert len(factors) == 1 and factors[0].plans == 1 and factors[0].factorizations == solver.solves == 2
    finally:
        solver.close()
    assert factors[0].freed
    solver.close()


@pytest.mark.parametrize("bad", [0.0, float("nan"), float("inf")])
def test_bad_solution_fails_without_cpu_fallback(operators: FullOperators, monkeypatch: pytest.MonkeyPatch,
                                                bad: float) -> None:
    class BadFactor(SingleFactor):
        def solve(self) -> NDArray[np.complex64]:
            return np.full(self.b.shape, bad, dtype=np.complex64)

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("不准退回 CPU")

    monkeypatch.setattr(FullSolver, "solve", forbidden)
    solver, _ = _small(operators, monkeypatch, BadFactor)
    try:
        with pytest.raises(RuntimeError, match="完整解殘差失常：40.0 Hz、喇叭 left、相對殘差"):
            solver.solve(40.0, WALLS[0])
        assert solver.solves == 1
    finally:
        solver.close()


def test_changed_sparsity_rejected(operators: FullOperators, monkeypatch: pytest.MonkeyPatch) -> None:
    solver, system = _small(operators, monkeypatch)
    try:
        solver.solve(40.0, WALLS[0])
        changed = system.toarray()
        changed[0, 1] = changed[1, 0] = 0.5
        monkeypatch.setattr(solver, "system", lambda frequency, walls: csr_matrix(changed))
        with pytest.raises(ValueError, match="非零位置"):
            solver.solve(63.0, WALLS[0])
        assert solver.solves == 1
    finally:
        solver.close()


def test_gpu_batch_rejects_multiple_workers(operators: FullOperators) -> None:
    with pytest.raises(ValueError, match="gpu.*workers"):
        FullSolveBatch(operators, max_relative_residual=LIMIT, device="gpu", workers=2)


def test_gpu_batch_frees_on_failure(operators: FullOperators, monkeypatch: pytest.MonkeyPatch) -> None:
    from aosr.physics import fem_full_gpu

    factors: list[SingleFactor] = []

    class BrokenFactor(SingleFactor):
        def factorize(self) -> None:
            raise RuntimeError("分解失敗")

    def factory(a: csr_matrix[np.complex64], b: NDArray[np.complex64]) -> SingleFactor:
        factor = BrokenFactor(a, b)
        factors.append(factor)
        return factor

    monkeypatch.setattr(fem_full_gpu, "nvmath_factor", factory)
    with pytest.raises(RuntimeError, match="分解失敗"):
        with FullSolveBatch(operators, max_relative_residual=LIMIT, device="gpu") as batch:
            list(batch.map([(40.0, WALLS[0])]))
    assert factors[0].freed


def test_each_speaker_must_converge(operators: FullOperators, monkeypatch: pytest.MonkeyPatch) -> None:
    """第一支過關不能遮住第二支不收斂；訊息須點名第二支。"""
    class BrokenRight(SingleFactor):
        def solve(self) -> NDArray[np.complex64]:
            answer = super().solve()
            answer[:, 1] = 0
            return answer

    solver, _ = _small(operators, monkeypatch, BrokenRight)
    try:
        with pytest.raises(RuntimeError, match="喇叭 right、相對殘差"):
            solver.solve(40.0, WALLS[0])
    finally:
        solver.close()
