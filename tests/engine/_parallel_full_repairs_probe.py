"""審查反例本體；只在獨立、單緒、CPU 釘選的測試行程裡操作環境與訊號。"""
from __future__ import annotations

import ctypes
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ProcessPoolExecutor
from functools import partial
from multiprocessing.synchronize import Event
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from aosr import runtime
from aosr.physics import fem_reduced_build as builder
from aosr.physics.fem_reduced_boundary import BoundaryImpedances
from aosr.physics.fem_full_batch import FullSolveBatch, FullSolveJob, _WorkerSolver, _init_worker, _solve_worker
from aosr.physics.fem_reduced import FullOperators, FullSolver
from tests.engine._parallel_full_solves_probe import (
    AXIS, SETTINGS, WALLS, _equal, _normal_time, _operators, _saved_pair,
)


def _until(predicate: Callable[[], bool], seconds: float = 30.0) -> None:
    deadline = time.monotonic() + seconds
    while not predicate():
        assert time.monotonic() < deadline, "hard deadline exceeded"
        time.sleep(0.02)


def _noop() -> None:
    pass


def _anchor_solve(job: FullSolveJob, directory: Path) -> tuple[int, int, NDArray[np.complex128], NDArray[np.complex128]]:
    (directory / f"ready-{os.getpid()}").touch()
    _until(lambda: len(list(directory.glob("ready-*"))) == 2)
    runtime.preload_mkl()
    from pydiso.mkl_solver import MKLPardisoSolver
    initial: list[NDArray[np.complex128]] = []

    def tracked(matrix: csr_matrix[np.complex128], *, matrix_type: str, factor: bool) -> MKLPardisoSolver:
        assert matrix_type == "complex_symmetric" and factor is False
        initial.append(matrix.toarray().copy())
        return MKLPardisoSolver(matrix, matrix_type=matrix_type, factor=factor)

    solver = runtime.worker_state(_WorkerSolver)
    assert solver.solves == 0
    with patch("pydiso.mkl_solver.MKLPardisoSolver", tracked):
        solution = _solve_worker(job)
    assert len(initial) == 1
    return os.getpid(), solver.solves, initial[0], solution


def anchor(directory: Path) -> None:
    operators = _operators()
    jobs = [(AXIS[0], WALLS[0]), (AXIS[3], WALLS[2])]
    with FullSolveBatch(operators, max_relative_residual=1e-8, workers=1) as single:
        expected = list(single.map(jobs))
    with runtime.spawn_pool(2, factory=partial(_init_worker, operators, 1e-8, jobs[0])) as pool:
        first = pool.submit(_anchor_solve, jobs[0], directory)
        _until(lambda: len(list(directory.glob("ready-*"))) == 1)
        second = pool.submit(_anchor_solve, jobs[1], directory)
        results = [first.result(timeout=30), second.result(timeout=30)]
        counts = [pool.submit(_slot_count, directory) for _ in range(2)]
        assert sorted(future.result(timeout=30) for future in counts) == sorted((row[0], 1) for row in results)
    assert results[0][0] != results[1][0]
    for (_, solves, initial, actual), wanted in zip(results, expected, strict=True):
        _equal(initial, single.solver.system(*jobs[0]).toarray(), "actual PARDISO anchor")
        assert solves == 1, "initialization must not count as a solve"
        _equal(actual, wanted, "different first jobs")


def _slot_count(directory: Path) -> tuple[int, int]:
    (directory / f"count-{os.getpid()}").touch()
    _until(lambda: len(list(directory.glob("count-*"))) == 2)
    return os.getpid(), runtime.worker_state(_WorkerSolver).solves


def environment(directory: Path) -> None:
    original = dict(os.environ)
    os.environ.update({"AOSR_KEEP": "present", "AOSR_CHANGE": "old", "AOSR_DELETE": "old"})
    os.environ.pop("MKL_NUM_THREADS", None)
    os.environ["OMP_NUM_THREADS"] = "7"
    start, done = threading.Event(), threading.Event()
    observed: list[str | None] = []

    def update() -> None:
        assert start.wait(30)
        observed.append(os.environ.get("AOSR_KEEP"))
        os.environ["AOSR_CHANGE"] = "new"
        os.environ["AOSR_ADDED"] = "new"
        os.environ.pop("AOSR_DELETE", None)
        done.set()

    submit = ProcessPoolExecutor.submit

    def during_start[T](pool: ProcessPoolExecutor, fn: Callable[[], T]) -> Future[T]:
        start.set()
        assert done.wait(30)
        return submit(pool, fn)

    updater = threading.Thread(target=update)
    updater.start()
    try:
        with patch.object(ProcessPoolExecutor, "submit", during_start):
            with runtime.spawn_pool(2, factory=_noop):
                pass
        updater.join(timeout=15)
        assert observed == ["present"], "unrelated parent environment disappeared during spawn"
        assert os.environ["AOSR_CHANGE"] == os.environ["AOSR_ADDED"] == "new"
        assert "AOSR_DELETE" not in os.environ
        assert "MKL_NUM_THREADS" not in os.environ and os.environ["OMP_NUM_THREADS"] == "7"
    finally:
        updater.join(timeout=15)
        for name in set(os.environ) | set(original):
            if name in original:
                os.environ[name] = original[name]
            else:
                os.environ.pop(name, None)


def _record_pid(directory: Path) -> None:
    (directory / f"worker-{os.getpid()}").touch()


def _middle(directory: Path) -> None:
    with runtime.spawn_pool(2, factory=partial(_record_pid, directory)):
        _until(lambda: len(list(directory.glob("worker-*"))) == 2)
        (directory / "opened").touch()
        time.sleep(120)


def _gated_middle(directory: Path) -> None:
    # 在 initializer 進去、閘門尚未開啟時留下 pid，再停住父行程的第二次提交。
    from aosr import runtime as module

    def stalled[T](pool: ProcessPoolExecutor, fn: Callable[[], T]) -> Future[T]:
        future = submit(pool, fn)
        _until(lambda: len(list(directory.glob("worker-*"))) == 1)
        (directory / "opened").touch()
        time.sleep(120)
        return future

    submit = ProcessPoolExecutor.submit
    with patch.object(module, "_spawn_initialize", partial(_record_before_gate, directory)), \
            patch.object(ProcessPoolExecutor, "submit", stalled):
        _middle(directory)


def _record_before_gate(directory: Path, factory: Callable[[], object], gate: Event, parent_pid: int) -> None:
    _record_pid(directory)
    runtime._spawn_initialize(factory, gate, parent_pid)


def _parent_death(directory: Path, mode: str) -> None:
    # 讓本測試收養孫行程並 waitpid，驗的是行程真的消失，不能把殭屍當作已清理。
    assert ctypes.CDLL(None).prctl(36, 1, 0, 0, 0) == 0
    process = subprocess.Popen([sys.executable, "-m", __spec__.name, mode, str(directory)])
    pids: list[int] = []
    try:
        _until(lambda: (directory / "opened").exists())
        pids = [int(file.name.removeprefix("worker-")) for file in directory.glob("worker-*")]
        assert len(pids) == (2 if mode == "_middle" else 1)
        process.kill()
        assert process.wait(timeout=15) == -signal.SIGKILL

        def reaped() -> bool:
            for pid in pids[:]:
                if os.waitpid(pid, os.WNOHANG)[0] == pid:
                    pids.remove(pid)
            return not pids

        _until(reaped, seconds=15)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=15)
        for pid in pids:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)


def parent_death(directory: Path) -> None:
    _parent_death(directory, "_middle")


def gate_death(directory: Path) -> None:
    _parent_death(directory, "_gated_middle")


def _prefix_solve(solver: FullSolver, frequency: float, walls: BoundaryImpedances, directory: Path) -> NDArray[np.complex128]:
    # 只改殘差上限來注入真實失敗；分解、求解與原例外均走產品路徑。
    if frequency == AXIS[1]:
        (directory / "second-pid").write_text(str(os.getpid()), encoding="utf-8")
        _until(lambda: (directory / "third-done").exists())
        solver._limit = -1.0
    result = _ORIGINAL_SOLVE(solver, frequency, walls)
    if frequency == AXIS[2]:
        (directory / "third-done").write_text(str(os.getpid()), encoding="utf-8")
    if frequency == AXIS[1]:
        raise AssertionError("residual failure did not occur")
    return result


_ORIGINAL_SOLVE = FullSolver.solve


def _prefix_hook(directory: Path) -> Callable[..., NDArray[np.complex128]]:
    def solve(solver: FullSolver, frequency: float, walls: BoundaryImpedances) -> NDArray[np.complex128]:
        return _prefix_solve(solver, frequency, walls, directory)
    return solve


def prefix_error(directory: Path) -> None:
    from aosr.physics import fem_full_batch
    operators = _operators()
    jobs = [(AXIS[i], WALLS[i]) for i in range(3)]
    messages = []
    (directory / "third-done").touch()
    with patch.object(FullSolver, "solve", _prefix_hook(directory)):
        with FullSolveBatch(operators, max_relative_residual=1e-8) as batch:
            with pytest.raises(RuntimeError) as error:
                list(batch.map(jobs))
            assert batch.solver.solves == 2
            messages.append(str(error.value))
    (directory / "third-done").unlink()
    with patch.object(fem_full_batch, "_init_worker", partial(_prefix_factory, directory)):
        with FullSolveBatch(operators, max_relative_residual=1e-8, workers=2) as batch:
            with pytest.raises(RuntimeError) as error:
                list(batch.map(jobs))
            assert batch.solver.solves == 2
            messages.append(str(error.value))
    assert (directory / "third-done").read_text() != (directory / "second-pid").read_text()
    assert messages[0] == messages[1]


def _prefix_factory(directory: Path, operators: FullOperators, limit: float, first: FullSolveJob) -> object:
    from aosr.physics.fem_full_batch import _init_worker
    patch.object(FullSolver, "solve", _prefix_hook(directory)).start()
    return _init_worker(operators, limit, first)


def failed_reuse(directory: Path) -> None:
    jobs = [(AXIS[0], WALLS[0])]
    with FullSolveBatch(_operators(), max_relative_residual=-1.0, workers=2) as batch:
        with pytest.raises(RuntimeError, match="殘差"):
            list(batch.map(jobs))
        assert batch.solver.solves == 1
        for again in (jobs, []):
            with pytest.raises(RuntimeError, match="批次已因錯誤關閉"):
                list(batch.map(again))
            assert batch.solver.solves == 1


def _slow(directory: Path) -> None:
    (directory / f"started-{os.getpid()}").touch()
    _until(lambda: (directory / "release").exists())


def _queued(directory: Path, index: int) -> None:
    (directory / f"queued-{index}").touch()


def cancel(directory: Path) -> None:
    started = 0.0
    try:
        with pytest.raises(ValueError, match="original"):
            with runtime.spawn_pool(2, factory=_noop) as pool:
                running = [pool.submit(_slow, directory) for _ in range(2)]
                _until(lambda: len(list(directory.glob("started-*"))) == 2)
                pending = [pool.submit(_queued, directory, i) for i in range(12)]
                started = time.monotonic()
                raise ValueError("original")
        assert time.monotonic() - started < 5.0, "error waited for unrelated running work"
        assert all(not future.done() for future in running), "slow work finished before error returned"
        _until(lambda: any(future.cancelled() for future in pending))
    finally:
        # 原錯誤已回到呼叫者、取消已生效，才放行；斷言失敗也必須放行以便收尾。
        (directory / "release").touch()
    for future in running:
        future.result(timeout=30)
    _until(lambda: all(future.done() for future in pending))
    for index, future in enumerate(pending):
        if future.cancelled():
            assert not (directory / f"queued-{index}").exists()
        else:
            future.result(timeout=30)


def budget(directory: Path) -> None:
    operators = _operators()
    settings = SETTINGS.model_copy(update={"max_full_solves": 5})
    models = [builder.build_reduced_model(operators, AXIS, WALLS[:2], settings=settings, workers=count)
              for count in (1, 2)]
    assert models[0].ledger.full_solves == models[1].ledger.full_solves == 5
    assert models[0].ledger.snapshots == models[1].ledger.snapshots == 10
    assert any(round.solved == 1 for round in models[0].ledger.rounds)
    assert _normal_time(models[0]).ledger == _normal_time(models[1]).ledger
    _saved_pair(models[0], models[1], directory)


if __name__ == "__main__":
    globals()[sys.argv[1]](Path(sys.argv[2]))
