"""#1031 三處浪費的控制組與防退化考卷；控制值在起點 2b1b78f6 單緒實量。

選點逐點釘住（包含解次預算截斷），數值差以雙精度捨入尺度比較；不改既有精度契約。
"""
from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Iterator, Sequence
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from aosr.config.fem_reduced import ReducedModelSettings
from aosr.config.frequency_axis import FEM_GEOMETRIC_CROSSOVER_CAP_HZ
from aosr.geometry.shoebox import Wall
from aosr.physics import fem_reduced_check as check
from aosr.physics.fem_full_batch import FullSolveBatch, FullSolveJob
from aosr.physics.fem_reduced import (
    FullOperators, ReducedMatrices, ReducedModel, evaluate_reduced, export_room_operators, impedance_rows,
    project, relative_residual, system_coefficients, walls_at,
)
from aosr.physics.fem_reduced_boundary import FrequencyBoundaryImpedances
from aosr.physics.fem_reduced_build import Point2, _Build, build_reduced_model
from tests.engine import test_fem_reduced as whole
from tests.engine import test_fem_reduced_zones as zones
from tests.engine._rom_build_control import CONTROL_JSON
from tests.engine._rom_build_reference import legacy_validate
from tests.engine._scheme_cache import bitwise_equal
from tests.engine._zone_rom_case import AXIS, SETTINGS

ROUNDING = 64 * np.finfo(np.float64).eps
Case = tuple[FullOperators, tuple[float, ...], Sequence[FrequencyBoundaryImpedances], ReducedModelSettings]


def _case(name: str) -> Case:
    if name == "whole":
        operators = export_room_operators(whole.ROOM, whole.SOURCES, whole.SEATS,
            density_kg_m3=whole.DENSITY, sound_speed_m_s=whole.SOUND_SPEED,
            physics_identity="phys-v1:" + "0" * 64, max_frequency_hz=FEM_GEOMETRIC_CROSSOVER_CAP_HZ)
        return operators, whole.AXIS, whole.TRAINING, whole._settings()
    settings = SETTINGS if name == "zones" else SETTINGS.model_copy(update={
        "initial_impedance_count": 2, "initial_frequency_count": 2, "max_full_solves": 12, "stop_residual": 1e-12})
    return zones._zoned_operators(), AXIS, zones._zoned_materials(1, 8), settings


@pytest.fixture(scope="module", params=("whole", "zones", "zones_greedy"))
def built(request: pytest.FixtureRequest) -> tuple[str, Case, ReducedModel, NDArray[np.float64], list[list[Point2]]]:
    name = str(request.param)
    case = _case(name)
    batches: list[list[Point2]] = []
    bases: list[NDArray[np.float64]] = []
    original_solve, original_reduce = _Build.solve, _Build.reduce

    def solve(build: _Build, pending: Sequence[Point2]) -> None:
        batches.append(list(pending[:max(0, build.settings.max_full_solves - build.solver.solves)]))
        original_solve(build, pending)
        assert build.seen == {point for batch in batches for point in batch}

    def reduce(build: _Build) -> tuple[NDArray[np.float64], ReducedMatrices]:
        basis, matrices = original_reduce(build)
        bases[:] = [basis]
        return basis, matrices

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_Build, "solve", solve)
        patch.setattr(_Build, "reduce", reduce)
        operators, axis, training, settings = case
        model = build_reduced_model(operators, axis, training, settings=settings, dense_threads=1)
    return name, case, model, bases[0], batches


def test_build_matches_starting_commit_points_and_ledger(
    built: tuple[str, Case, ReducedModel, NDArray[np.float64], list[list[Point2]]],
) -> None:
    """抓排序或停法改變：起點與每輪實解點、快照、秩、停因逐項等於施工前控制組。"""
    name, _, model, _, batches = built
    control = json.loads(CONTROL_JSON)["cases"][name]
    assert [[list(point) for point in batch] for batch in batches] == control["solved_batches"]
    actual = dataclasses.asdict(model.ledger)
    for key in ("full_solves", "snapshots", "rank", "stop_reason"):
        assert actual[key] == control[key]
    assert len(actual["rounds"]) == len(control["rounds"])
    for row, expected in zip(actual["rounds"], control["rounds"], strict=True):
        for key in expected:
            if key in ("max_residual", "median_residual"):
                assert abs(row[key] - expected[key]) <= ROUNDING
            else:
                assert row[key] == expected[key]
    assert set(model.ledger.seconds) == {"snapshot", "svd", "projection", "greedy", "total"}


def test_many_threads_matches_single_thread_points_and_ledger(
    built: tuple[str, Case, ReducedModel, NDArray[np.float64], list[list[Point2]]],
    monkeypatch: pytest.MonkeyPatch, record_property: Callable[[str, object], None],
) -> None:
    """抓多緒改變貪婪路徑；殘差另守多緒 SVD 的捨入差，不動單緒控制組的尺。"""
    name, (operators, axis, training, settings), single, _, batches = built
    actual_batches: list[list[Point2]] = []
    original_solve = _Build.solve

    def solve(build: _Build, pending: Sequence[Point2]) -> None:
        actual_batches.append(list(pending[:max(0, build.settings.max_full_solves - build.solver.solves)]))
        original_solve(build, pending)

    monkeypatch.setattr(_Build, "solve", solve)
    many = build_reduced_model(operators, axis, training, settings=settings, dense_threads=4)
    assert actual_batches == batches
    for key in ("full_solves", "snapshots", "rank", "stop_reason"):
        assert getattr(many.ledger, key) == getattr(single.ledger, key)
    assert len(many.ledger.rounds) == len(single.ledger.rounds)
    for index, (one, four) in enumerate(zip(single.ledger.rounds, many.ledger.rounds, strict=True)):
        for field in dataclasses.fields(one):
            before, after = getattr(one, field.name), getattr(four, field.name)
            if field.name in ("max_residual", "median_residual"):
                relative = 0.0 if before == after else abs(after - before) / abs(before)
                record_property(f"{name}_round_{index}_{field.name}_relative_difference", relative)
                # 多緒 SVD 在近重根方向轉基底，11-07 實量 2.6e-8。
                assert relative <= 1e-6
            else:
                assert after == before


def _legacy_project(operators: FullOperators, basis: NDArray[np.float64]) -> ReducedMatrices:
    def squeeze(matrix: csr_matrix[np.float64]) -> NDArray[np.float64]:
        return np.asarray(basis.T @ np.asarray(matrix @ basis, dtype=np.float64), dtype=np.float64)
    return ReducedMatrices(stiffness=squeeze(operators.stiffness), mass=squeeze(operators.mass),
        boundary=np.asarray([squeeze(matrix) for matrix in operators.boundary]),
        sources=np.asarray([basis.T @ load for load in operators.sources.values()]),
        receivers=np.asarray([row @ basis for row in operators.receivers.values()]))


def test_projection_matches_full_support_control(
    built: tuple[str, Case, ReducedModel, NDArray[np.float64], list[list[Point2]]],
) -> None:
    """抓面片漏行、漏列、排序或轉置錯誤：所有矩陣對起點全域投影只容許捨入差。"""
    _, (operators, _, _, _), model, basis, _ = built
    reference = _legacy_project(operators, basis)
    for field in dataclasses.fields(reference):
        expected = getattr(reference, field.name)
        actual = getattr(model.matrices, field.name)
        if field.name == "boundary":
            assert np.max(np.abs(actual - expected)) <= ROUNDING * np.max(np.abs(expected))
        else:
            assert actual.tobytes() == expected.tobytes()


def test_boundary_projection_does_not_multiply_full_basis(monkeypatch: pytest.MonkeyPatch) -> None:
    """抓面片投影退回全自由度運算；並保留空面與非對角支撐的投影結果。"""
    operators, _, _, _ = _case("zones")
    size = operators.stiffness.shape[0]
    boundary = dict(operators.p2.boundary_mass)
    wall = next(iter(boundary))
    boundary[wall] = csr_matrix(([2.0], ([1], [size - 2])), shape=(size, size))
    other = next(key for key in boundary if key != wall)
    boundary[other] = csr_matrix((size, size), dtype=np.float64)
    operators = dataclasses.replace(operators, p2=dataclasses.replace(operators.p2, boundary_mass=boundary))
    basis = np.random.default_rng(1031).normal(size=(size, 9))
    expected = _legacy_project(operators, basis)
    matrices = operators.boundary
    original = csr_matrix.__matmul__

    def multiply(matrix: csr_matrix[np.float64], values: NDArray[np.float64]) -> NDArray[np.float64]:
        assert not any(matrix is item for item in matrices), "面片不應用整個自由度乘基底"
        return np.asarray(original(matrix, values), dtype=np.float64)

    monkeypatch.setattr(csr_matrix, "__matmul__", multiply)
    actual = project(operators, basis)
    assert np.max(np.abs(actual.boundary - expected.boundary)) <= ROUNDING * np.max(np.abs(expected.boundary))


class _RealProductBasis:
    """在乘法邊界抓整份實基底被升成複數；仍執行真正 numpy 乘法。"""

    def __init__(self, basis: NDArray[np.float64]) -> None:
        self.basis = basis

    def __matmul__(self, other: NDArray[np.float64]) -> NDArray[np.float64]:
        assert not np.iscomplexobj(other), "整份基底不應乘複數座標"
        return self.basis @ other


def test_greedy_residual_avoids_assembly_and_complex_basis(
    built: tuple[str, Case, ReducedModel, NDArray[np.float64], list[list[Point2]]], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """抓完整 CSR 組裝與複數基底暫存回歸；對原定義的 b−A(Vq) 逐喇叭最大範數比。"""
    name, (operators, axis, training, settings), model, basis, _ = built
    if name == "whole":
        # 整面可剛性；混合與全剛性兩種都守零係數，分區仍維持必須有材料。
        mixed = dict(whole.TRAINING[0])
        mixed[Wall.FLOOR] = None
        training = [*training, mixed, {wall: None for wall in Wall.all()}]
    rows = impedance_rows(training, axis, patch_keys=operators.patch_keys)
    with FullSolveBatch(operators, max_relative_residual=settings.full_solve_max_relative_residual, workers=1) as batch:
        build = _Build(operators, axis, rows, settings, batch.solver, batch, 1)
        points = [(0, 0), (len(training) - 1, len(axis) - 1)]
        if name == "whole":
            points.append((len(training) - 2, len(axis) - 1))
        for point in points:
            walls = walls_at(rows[point[0]], point[1])
            k2, coefficients = system_coefficients(axis[point[1]], walls, density_kg_m3=operators.density_kg_m3,
                sound_speed_m_s=operators.sound_speed_m_s, patch_keys=operators.patch_keys)
            m = model.matrices
            q = np.linalg.solve(m.stiffness - k2 * m.mass + np.tensordot(coefficients, m.boundary, axes=1), m.sources.T)
            fields = basis @ q
            system = batch.solver.system(axis[point[1]], walls)
            expected = max(relative_residual(system, fields[:, i], load) for i, load in enumerate(operators.sources.values()))

            def no_system(frequency: float, values: object) -> None:
                raise AssertionError("量殘差不應組裝完整 CSR")

            with monkeypatch.context() as patch:
                patch.setattr(batch.solver, "system", no_system)
                actual = build.residual(cast(NDArray[np.float64], _RealProductBasis(basis)), m, point)
            assert abs(actual - expected) <= ROUNDING


@pytest.mark.parametrize("workers", (1, 2))
def test_validation_report_bitwise_matches_interleaved_control(workers: int, monkeypatch: pytest.MonkeyPatch) -> None:
    """抓留出驗證更動數值或在完整解尚未收齊時求替身；兩支真完整解路徑各對起點逐位。"""
    operators, axis, training, settings = _case("zones")
    model = build_reduced_model(operators, axis, training, settings=settings, dense_threads=1)
    holdouts = zones._zoned_materials(99, 2)
    expected_steps: list[int] = []
    expected = legacy_validate(model, operators, holdouts, workers=workers, refinement_steps=expected_steps)
    consumed = []
    original_map, original_evaluate = FullSolveBatch.map, evaluate_reduced

    def mapped(batch: FullSolveBatch, jobs: Sequence[FullSolveJob]) -> Iterator[NDArray[np.complex128]]:
        for solution in original_map(batch, jobs):
            consumed.append(solution.shape)
            yield solution

    def evaluate(matrices: ReducedMatrices, k_squared: float, coefficients: NDArray[np.complex128]) -> NDArray[np.complex128]:
        if workers > 1:
            assert len(consumed) == len(holdouts) * len(axis), "替身求值擋住完整解回收"
        return original_evaluate(matrices, k_squared, coefficients)

    monkeypatch.setattr(FullSolveBatch, "map", mapped)
    monkeypatch.setattr(check, "evaluate_reduced", evaluate)
    steps: list[int] = []
    actual = check.validate_reduced(model, operators, holdouts, workers=workers, refinement_steps=steps, dense_threads=1)
    assert bitwise_equal(actual, expected)
    assert steps == expected_steps
