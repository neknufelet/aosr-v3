"""#1036 稠密段放開 BLAS、完整解維持原緒數；全部用小盒子房與 CPU。"""
from __future__ import annotations

import importlib
import os
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from types import TracebackType
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from aosr.physics import fem_reduced_build as builder, fem_reduced_check as check
from aosr.physics import dense_threads as dense
from aosr.physics.fem_full_batch import FullSolveBatch, FullSolveJob
from aosr.physics.fem_reduced import FullOperators, FullSolver, ReducedMatrices, evaluate_reduced, project, snapshot_basis
from aosr.physics.fem_reduced_boundary import BoundaryImpedances
from tests.engine._parallel_full_solves_probe import AXIS, SETTINGS, WALLS, _operators

ROUNDING = 64 * np.finfo(np.float64).eps


def _blas_threads() -> list[int]:
    pools: list[dict[str, object]] = importlib.import_module("threadpoolctl").threadpool_info()
    counts = []
    for pool in pools:
        if pool["user_api"] == "blas":
            count = pool["num_threads"]
            assert isinstance(count, int)
            counts.append(count)
    assert counts, "環境必須有能被 threadpoolctl 辨識的 BLAS"
    return counts


@pytest.mark.parametrize("raises", (False, True))
def test_dense_blas_sets_and_restores_real_pools(raises: bool) -> None:
    """抓漏設、限制了非 BLAS 或例外離開漏還原；讀真正函式庫緒數。"""
    pools = importlib.import_module("threadpoolctl")
    with pools.threadpool_limits(limits=1, user_api="blas"):
        before = _blas_threads()
        try:
            with dense.dense_blas_threads(2):
                assert _blas_threads() == [2] * len(before)
                if raises:
                    raise RuntimeError("離開測試")
        except RuntimeError as error:
            assert raises and str(error) == "離開測試"
        assert _blas_threads() == before


@pytest.mark.parametrize("filename", ("core_cpus_list", "thread_siblings_list"))
def test_dense_default_counts_physical_cores(
    filename: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """抓超執行緒被重複計數；親和性只含兄弟之一仍各算一核，排除不可用核。"""
    for cpu in range(16):
        topology = tmp_path / f"cpu{cpu}" / "topology"
        topology.mkdir(parents=True)
        (topology / filename).write_text(f"{cpu % 8},{cpu % 8 + 8}\n", encoding="ascii")
    monkeypatch.setattr(dense, "_CPU_SYSFS", tmp_path, raising=False)
    for allowed, expected in ((set(range(16)), 8), ({0, 8, 2, 10, 7}, 3), (set(range(8, 16)), 8)):
        monkeypatch.setattr(os, "sched_getaffinity", lambda pid: allowed, raising=False)
        assert dense.default_dense_threads() == expected
    # 支援 sysfs 的區間清單；跨兄弟組只按允許的緒去重。
    for cpu in range(4):
        (tmp_path / f"cpu{cpu}" / "topology" / filename).write_text("0-3\n", encoding="ascii")
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 1, 2, 3}, raising=False)
    assert dense.default_dense_threads() == 1


@pytest.mark.parametrize("contents", (None, "invalid", "", "0-", "2-1"))
def test_dense_default_falls_back_to_logical_threads(
    contents: str | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """抓 sysfs 讀不到或壞掉卻猜成半數，與缺親和性 API 時無法開跑。"""
    monkeypatch.setattr(dense, "_CPU_SYSFS", tmp_path, raising=False)
    if contents is not None:
        topology = tmp_path / "cpu0" / "topology"
        topology.mkdir(parents=True)
        (topology / "core_cpus_list").write_text(contents, encoding="ascii")
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 2, 4}, raising=False)
    assert dense.default_dense_threads() == 3
    monkeypatch.delattr(os, "sched_getaffinity")
    monkeypatch.setattr(os, "cpu_count", lambda: 2)
    assert dense.default_dense_threads() == 2
    monkeypatch.setattr(os, "cpu_count", lambda: None)
    assert dense.default_dense_threads() == 1


@pytest.mark.parametrize("invalid", (0, -1, True, False, 1.5, "2"))
def test_dense_threads_reject_invalid_values(invalid: object) -> None:
    """抓 BLAS 將 0、負數或非整數當成別種設定而默默接受。"""
    with pytest.raises(ValueError, match="dense_threads"):
        with dense.dense_blas_threads(cast(int, invalid)):
            pytest.fail("不合法緒數不准進區塊")


def test_entry_points_reject_invalid_dense_threads() -> None:
    """兩入口須在完整解開始前拒收緒數，不能只在最後的稠密段才發現。"""
    operators = _operators()
    model = builder.build_reduced_model(operators, AXIS, WALLS[:2], settings=SETTINGS, dense_threads=1)
    for invalid in (0, -1, True, 1.5, "2"):
        with pytest.raises(ValueError, match="dense_threads"):
            builder.build_reduced_model(operators, AXIS, WALLS[:2], settings=SETTINGS, dense_threads=cast(int, invalid))
        with pytest.raises(ValueError, match="dense_threads"):
            check.validate_reduced(model, operators, WALLS[2:], dense_threads=cast(int, invalid))


@pytest.mark.parametrize("workers", (1, 2))
def test_build_dense_sections_and_snapshot_boundary(
    workers: int, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """抓三段漏包、整個建模被包或還原太晚；探針仍呼叫真正 SVD、投影、殘差與完整解。"""
    operators = _operators()
    pools = importlib.import_module("threadpoolctl")
    monkeypatch.setattr(dense, "default_dense_threads", lambda: 2)
    observed: dict[str, list[list[int]]] = {name: [] for name in ("svd", "project", "greedy", "snapshot")}
    original_svd, original_project = snapshot_basis, project
    original_residual, original_solve = builder._Build.residual, FullSolver.solve
    original_map = FullSolveBatch.map

    def svd(snapshots: Sequence[NDArray[np.complex128]], cutoff: float) -> NDArray[np.float64]:
        observed["svd"].append(_blas_threads())
        return original_svd(snapshots, cutoff)

    def projected(ops: FullOperators, basis: NDArray[np.float64]) -> ReducedMatrices:
        observed["project"].append(_blas_threads())
        return original_project(ops, basis)

    def residual(build: builder._Build, basis: NDArray[np.float64], matrices: ReducedMatrices,
                 point: builder.Point2) -> float:
        observed["greedy"].append(_blas_threads())
        return original_residual(build, basis, matrices, point)

    def solve(solver: FullSolver, frequency: float, walls: BoundaryImpedances) -> NDArray[np.complex128]:
        observed["snapshot"].append(_blas_threads())
        return original_solve(solver, frequency, walls)

    def mapped(batch: FullSolveBatch, jobs: Sequence[FullSolveJob]) -> Iterator[NDArray[np.complex128]]:
        observed["snapshot"].append(_blas_threads())
        for solution in original_map(batch, jobs):
            observed["snapshot"].append(_blas_threads())
            yield solution

    monkeypatch.setattr(builder, "snapshot_basis", svd)
    monkeypatch.setattr(builder, "project", projected)
    monkeypatch.setattr(builder._Build, "residual", residual)
    monkeypatch.setattr(FullSolver, "solve", solve)
    monkeypatch.setattr(FullSolveBatch, "map", mapped)
    with pools.threadpool_limits(limits=1, user_api="blas"):
        builder.build_reduced_model(operators, AXIS, WALLS[:2], settings=SETTINGS, workers=workers)
        assert all(count == 1 for count in _blas_threads())
    for name, rows in observed.items():
        assert rows, name
        assert all(count == (1 if name == "snapshot" else 2) for row in rows for count in row), name
    assert capsys.readouterr().err.splitlines() == [f"完整解 {workers} 行程、裝置 cpu、稠密段 2 緒"]


@pytest.mark.parametrize("workers", (1, 2))
def test_validation_dense_after_full_batch_and_rounding(
    workers: int, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    record_property: Callable[[str, object], None],
) -> None:
    """抓留出完整解緒數被改或未收完就進稠密段；完整聲壓逐位、替身與報告只容捨入差。"""
    operators = _operators()
    pools = importlib.import_module("threadpoolctl")
    default = dense.default_dense_threads()
    model = builder.build_reduced_model(operators, AXIS, WALLS[:2], settings=SETTINGS, dense_threads=1)
    capsys.readouterr()
    original_evaluate = evaluate_reduced
    original_exit = FullSolveBatch.__exit__
    original_solve = FullSolver.solve
    exited = False
    dense_counts: list[list[int]] = []
    full_counts: list[list[int]] = []

    def exit_batch(batch: FullSolveBatch, exc_type: type[BaseException] | None,
                   exc: BaseException | None, traceback: TracebackType | None) -> None:
        nonlocal exited
        original_exit(batch, exc_type, exc, traceback)
        exited = True

    def evaluate(matrices: ReducedMatrices, k2: float, coefficients: NDArray[np.complex128]) -> NDArray[np.complex128]:
        assert exited, "完整解批次須先結束並收回工人"
        dense_counts.append(_blas_threads())
        return original_evaluate(matrices, k2, coefficients)

    def solve(solver: FullSolver, frequency: float, walls: BoundaryImpedances) -> NDArray[np.complex128]:
        full_counts.append(_blas_threads())
        return original_solve(solver, frequency, walls)

    monkeypatch.setattr(FullSolveBatch, "__exit__", exit_batch)
    monkeypatch.setattr(check, "evaluate_reduced", evaluate)
    monkeypatch.setattr(FullSolver, "solve", solve)
    reports = []
    for threads in (1, default):
        exited = False
        with pools.threadpool_limits(limits=1, user_api="blas"):
            reports.append(check.validate_reduced(model, operators, WALLS[2:], workers=workers, dense_threads=threads))
            assert all(count == 1 for count in _blas_threads())
    assert dense_counts and all(count == 1 for row in dense_counts[:len(AXIS)] for count in row)
    assert all(count == default for row in dense_counts[len(AXIS):] for count in row), dense_counts
    if workers == 1:
        assert full_counts and all(count == 1 for row in full_counts for count in row)
    _assert_reports(reports[0], reports[1], record_property)
    assert capsys.readouterr().err.splitlines() == [
        f"完整解 {workers} 行程、裝置 cpu、稠密段 {threads} 緒" for threads in (1, default)]


def _assert_reports(one: check.ValidationReport, many: check.ValidationReport,
                    record_property: Callable[[str, object], None]) -> None:
    assert one.full_pressure.tobytes() == many.full_pressure.tobytes()
    assert one.full_solves == many.full_solves
    assert one.bands == many.bands
    assert one.model_digest == many.model_digest
    assert one.computation_digest == many.computation_digest
    assert one.source_points == many.source_points and one.receiver_points == many.receiver_points
    difference = float(np.max(np.abs(many.reduced_pressure - one.reduced_pressure) / np.abs(one.reduced_pressure)))
    record_property("reduced_pressure_max_relative_difference", difference)
    assert difference <= ROUNDING
    for field in ("frequency_db_by_seat", "band_db_by_seat", "partial_band_db_by_seat"):
        before, after = getattr(one, field), getattr(many, field)
        assert before.keys() == after.keys()
        for seat in before:
            assert abs(after[seat] - before[seat]) <= ROUNDING
        relative = max((abs(after[seat] - before[seat]) / abs(before[seat]) for seat in before if before[seat]), default=0.0)
        record_property(f"{field}_max_relative_difference", relative)
