"""多行程完整解的迷你網格考卷本體；由單緒冷行程載入 numpy 與 JAX。"""
from __future__ import annotations

import dataclasses
import os
import json
from concurrent.futures import Future, ProcessPoolExecutor, wait
import sys
import time
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from numpy.typing import NDArray

from aosr import runtime
from aosr.config.fem_reduced import ReducedModelSettings
from aosr.config.paths import config_path
from aosr.config.physics_constants import load_physics_constants
from aosr.config.porous_model_ranges import load_porous_model_ranges
from aosr.geometry.shoebox import Point, Room, Wall
from aosr.physics import fem_reduced_build as builder
from aosr.physics.fem_reduced import FullOperators, FullSolver, ReducedModel, export_room_operators, snapshot_basis
from aosr.physics.fem_full_batch import FullSolveBatch
from aosr.physics.fem_reduced_check import validate_reduced
from aosr.physics.fem_reduced_receipt import HoldoutRulers, Ruler, save_validation_receipt
from aosr.physics.fem_reduced_store import manifest_path, save_reduced
from tests.engine._precision_contracts import contract_value
from tests.engine.test_wall_surfaces import _CPU_PIN_VARS

AXIS = (40.0, 50.0, 63.0, 80.0)
WALLS = [{wall: complex(1400 + i * 300 + j * 20, -100 - i * 60) for j, wall in enumerate(Wall.all())}
         for i in range(3)]
SETTINGS = ReducedModelSettings(initial_impedance_count=1, initial_frequency_count=2,
    candidates_per_round=6, batch=2, stop_residual=1e-12, max_full_solves=6,
    full_solve_max_relative_residual=1e-8, svd_relative_cutoff=1e-10, greedy_seed=21)


def _operators() -> FullOperators:
    return export_room_operators(Room(1.6, 1.2, 1.0),
        {"left": Point(0.3, 0.3, 0.4), "right": Point(0.4, 0.8, 0.5)},
        {"front": Point(1.2, 0.7, 0.6), "back": Point(1.0, 0.4, 0.3)},
        density_kg_m3=1.2, sound_speed_m_s=343.0, physics_identity="phys-v1:" + "0" * 64,
        max_frequency_hz=80.0)


def _timed[T](label: str, call: Callable[[], T]) -> T:
    start = time.perf_counter()
    result = call()
    sys.stdout.write(f"{label}: {time.perf_counter() - start:.6f} s\n")
    return result


def _equal(left: NDArray[np.complex128] | NDArray[np.float64],
           right: NDArray[np.complex128] | NDArray[np.float64], entry: str) -> None:
    assert np.array_equal(left, right), (
        f"STOP {entry}: different={np.count_nonzero(left != right)}, max_abs={np.max(np.abs(left - right))}")


def _model(operators: FullOperators, workers: int = 1) -> ReducedModel:
    return builder.build_reduced_model(operators, AXIS, WALLS[:2], settings=SETTINGS, workers=workers)


def _normal_time(model: ReducedModel) -> ReducedModel:
    return dataclasses.replace(model, ledger=dataclasses.replace(model.ledger,
        seconds=dict.fromkeys(model.ledger.seconds, 1.0)))


def build(directory: Path) -> None:
    operators = _operators()
    snapshots: list[NDArray[np.complex128]] = []
    original = snapshot_basis

    def capture(rows: list[NDArray[np.complex128]], cutoff: float) -> NDArray[np.float64]:
        snapshots.append(np.asarray(rows).copy())
        return original(rows, cutoff)

    with patch.object(builder, "snapshot_basis", capture):
        single = _timed("build workers=1", lambda: _model(operators))
        split = len(snapshots)
        parallel = _timed("build workers=2", lambda: _model(operators, 2))
    assert isinstance(single, ReducedModel) and isinstance(parallel, ReducedModel)
    assert split >= 2 and len(snapshots) == 2 * split
    for left, right in zip(snapshots[:split], snapshots[split:], strict=True):
        _equal(left, right, "build snapshots")
    assert _normal_time(single).ledger == _normal_time(parallel).ledger
    assert single.ledger.full_solves == 6 and single.ledger.snapshots == 12
    for field in dataclasses.fields(single.matrices):
        _equal(getattr(single.matrices, field.name), getattr(parallel.matrices, field.name), "build " + field.name)
    _saved_pair(single, parallel, directory)


def _saved_pair(single: ReducedModel, parallel: ReducedModel, directory: Path) -> tuple[Path, Path]:
    paths = tuple(directory / name / "model.npz" for name in ("single", "parallel"))
    for model, path in zip((single, parallel), paths, strict=True):
        path.parent.mkdir()
        save_reduced(_normal_time(model), path)
    assert paths[0].read_bytes() == paths[1].read_bytes(), "STOP build npz"
    assert manifest_path(paths[0]).read_bytes() == manifest_path(paths[1]).read_bytes(), "STOP build manifest"
    return paths[0], paths[1]


def validation(directory: Path) -> None:
    operators = _operators()
    model = _model(operators)
    start = time.perf_counter()
    single = validate_reduced(model, operators, WALLS[2:], workers=1)
    sys.stdout.write(f"validation workers=1: {time.perf_counter() - start:.6f} s\n")
    start = time.perf_counter()
    parallel = validate_reduced(model, operators, WALLS[2:], workers=2)
    sys.stdout.write(f"validation workers=2: {time.perf_counter() - start:.6f} s\n")
    _equal(single.full_pressure, parallel.full_pressure, "validation full")
    _equal(single.reduced_pressure, parallel.reduced_pressure, "validation reduced")
    assert single.full_solves == parallel.full_solves == 4
    for field in dataclasses.fields(single):
        if field.name not in ("full_pressure", "reduced_pressure"):
            assert getattr(single, field.name) == getattr(parallel, field.name), field.name
    paths = _saved_pair(model, model, directory)
    rulers = HoldoutRulers(Ruler("fem_reduced_holdout_frequency_db", contract_value("fem_reduced_holdout_frequency_db")),
                          Ruler("fem_reduced_holdout_band_db", contract_value("fem_reduced_holdout_band_db")))
    receipts = [save_validation_receipt(path, report, rulers=rulers, seconds=1.0).read_bytes()
                for path, report in zip(paths, (single, parallel), strict=True)]
    assert receipts[0] == receipts[1], "STOP validation receipt"


def endpoint(directory: Path) -> None:
    from aosr.gradient.endpoint import full_model_pressures
    from aosr.materials.construction_models import ConstantImpedance

    operators = _operators()
    air = load_physics_constants(config_path("physics_constants.toml")).model_copy(
        update={"density_kg_m3": 1.2, "sound_speed_m_s": 343.0})
    ranges = load_porous_model_ranges(config_path("porous_model_ranges.toml"))
    materials = {wall: ConstantImpedance(kind="constant_impedance", impedance_pa_s_m=1400.0)
                 for wall in Wall.all()}
    results = [_timed(f"endpoint workers={workers}", lambda: full_model_pressures(
        operators, materials, AXIS, air=air, ranges=ranges, max_relative_residual=1e-8, workers=workers))
        for workers in (1, 2)]
    assert isinstance(results[0], np.ndarray) and isinstance(results[1], np.ndarray)
    _equal(results[0], results[1], "endpoint")


def batch(directory: Path) -> None:
    operators = _operators()
    jobs = [(AXIS[i], WALLS[i % 3]) for i in (3, 0, 2, 1, 0, 3)]
    direct = FullSolver(operators, max_relative_residual=1e-8)
    expected = [direct.solve(*job) for job in jobs]
    for workers in (1, 2):
        with FullSolveBatch(operators, max_relative_residual=1e-8, workers=workers) as pool:
            assert list(pool.map([])) == [] and pool.solver.solves == 0
            actual = list(pool.map(jobs[:3])) + list(pool.map(jobs[3:]))
            assert pool.solver.solves == direct.solves == len(jobs)
        for left, right in zip(expected, actual, strict=True):
            _equal(left, right, "batch order")


def errors(directory: Path) -> None:
    operators = _operators()
    messages = []
    for workers in (1, 2):
        with FullSolveBatch(operators, max_relative_residual=-1.0, workers=workers) as pool:
            with pytest.raises(RuntimeError, match="完整解殘差失常：40.0 Hz、喇叭 left、相對殘差") as error:
                list(pool.map([(40.0, WALLS[0])]))
            messages.append(str(error.value))
            assert pool.solver.solves == 1
            if workers > 1:
                assert "RemoteTraceback" in type(error.value.__cause__).__name__
            else:
                assert error.value.__cause__ is None
    assert messages[0] == messages[1]
    # 對角剛度讓零頻首題可逆；非零頻加入質量與邊界後，非零位置一定不同。
    diagonal = operators.stiffness.copy()
    diagonal.data.fill(0.0)
    diagonal.setdiag(1.0)
    diagonal.eliminate_zeros()
    operators = dataclasses.replace(operators, p2=dataclasses.replace(operators.p2, stiffness=diagonal))
    for workers in (1, 2):
        with FullSolveBatch(operators, max_relative_residual=1e-8, workers=workers) as pool:
            with pytest.raises(ValueError, match="有限元素矩陣非零位置與錨定矩陣不同"):
                list(pool.map([(0.0, WALLS[0]), (40.0, WALLS[0])]))


def invalid(directory: Path) -> None:
    operators = _operators()
    model = _model(operators)
    from aosr.gradient.endpoint import full_model_pressures

    for workers in (0, -1):
        with pytest.raises(ValueError, match="workers"):
            FullSolveBatch(operators, max_relative_residual=1e-8, workers=workers)
        with pytest.raises(ValueError, match="workers"):
            _model(operators, workers)
        with pytest.raises(ValueError, match="workers"):
            validate_reduced(model, operators, WALLS[2:], workers=workers)
        air = load_physics_constants(config_path("physics_constants.toml"))
        ranges = load_porous_model_ranges(config_path("porous_model_ranges.toml"))
        with pytest.raises(ValueError, match="workers"):
            full_model_pressures(operators, {}, AXIS, air=air, ranges=ranges,
                                 max_relative_residual=1e-8, workers=workers)


def _environment() -> tuple[int, str, dict[str, str | None]]:
    import multiprocessing
    names = (*_CPU_PIN_VARS, "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    return os.getpid(), multiprocessing.get_start_method(), {name: os.environ.get(name) for name in names}


def _record_environment(directory: Path) -> None:
    pid, method, env = _environment()
    (directory / f"{pid}.json").write_text(json.dumps([method, env]), encoding="utf-8")


def environment(directory: Path) -> None:
    from functools import partial
    before = dict(os.environ)
    submit = ProcessPoolExecutor.submit
    first = True

    def staggered[T](pool: ProcessPoolExecutor, function: Callable[[], T]) -> Future[T]:
        # 第一個 no-op 有機會先完成；沒有啟動同步時，執行器會重用它而沒開齊行程。
        nonlocal first
        future = submit(pool, function)
        if first:
            first = False
            wait((future,), timeout=2.0)
            time.sleep(0.05)
        return future

    with patch.object(ProcessPoolExecutor, "submit", staggered):
        with runtime.spawn_pool(2, initializer=partial(_record_environment, directory)) as pool:
            pid, method, env = pool.submit(_environment).result()
            assert pid != os.getpid() and method == "spawn"
    records = list(directory.glob("*.json"))
    assert len(records) == 2, "workers were created after the pinned environment was restored"
    for record in records:
        method, env = json.loads(record.read_text(encoding="utf-8"))
        assert method == "spawn"
        for name in _CPU_PIN_VARS:
            assert env[name] == before[name] and env[name]
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            assert env[name] == "1"
    assert dict(os.environ) == before


def cli(directory: Path) -> None:
    from aosr.gradient.cli import _parser
    parser = _parser()
    assert parser.parse_args(["endpoint", str(directory)]).workers == 1
    assert parser.parse_args(["endpoint", str(directory), "--workers", "2"]).workers == 2
    for workers in ("0", "-1"):
        with pytest.raises(SystemExit) as error:
            parser.parse_args(["endpoint", str(directory), "--workers", workers])
        assert error.value.code == 2


def cli_forwarding(directory: Path) -> None:
    from aosr.gradient import cli as command, run, worker

    with patch.object(run, "read_project"), patch.object(command, "_spawn_and_wait", return_value=7) as spawn:
        assert command.main(["endpoint", str(directory), "--workers", "2"]) == 7
        assert spawn.call_args.args[:2] == (directory.resolve(), "endpoint")
        assert spawn.call_args.kwargs == {"workers": 2}
    with patch("aosr.gradient.cli.subprocess.Popen") as launch:
        launch.return_value.wait.return_value = 7
        assert command._spawn_and_wait(directory, "endpoint", 0, workers=2) == 7
        assert launch.call_args.args[0] == [sys.executable, "-m", "aosr.gradient.worker",
                                          str(directory), "endpoint", "--workers", "2"]
    with patch.object(worker, "run_endpoint") as endpoint_run:
        assert worker.main([str(directory), "endpoint", "--workers", "2"]) == 0
        endpoint_run.assert_called_once_with(directory.resolve(), workers=2)


def zoned(directory: Path) -> None:
    from aosr.gradient.endpoint import full_model_pressures
    from aosr.materials.construction_models import ConstantImpedance
    from tests.engine._zone_rom_case import PARTITIONS

    whole = _operators()
    operators = export_room_operators(Room(1.6, 1.2, 1.0), whole.source_points, whole.receiver_points,
        density_kg_m3=1.2, sound_speed_m_s=343.0, physics_identity=whole.physics_identity,
        max_frequency_hz=80.0, partitions=PARTITIONS)
    walls = [{key: complex(1500 + i * 300 + j * 80, -70 * (i + 1))
              for j, key in enumerate(operators.patch_keys)} for i in range(3)]
    models = [builder.build_reduced_model(operators, AXIS, walls[:2], settings=SETTINGS, workers=workers)
              for workers in (1, 2)]
    assert _normal_time(models[0]).ledger == _normal_time(models[1]).ledger
    _saved_pair(models[0], models[1], directory)
    reports = [validate_reduced(model, operators, walls[2:], workers=workers)
               for model, workers in zip(models, (1, 2), strict=True)]
    _equal(reports[0].full_pressure, reports[1].full_pressure, "zoned validation")
    _equal(reports[0].reduced_pressure, reports[1].reduced_pressure, "zoned reduced")
    materials = {key: ConstantImpedance(kind="constant_impedance", impedance_pa_s_m=1500.0 + j * 80)
                 for j, key in enumerate(operators.patch_keys)}
    air = load_physics_constants(config_path("physics_constants.toml")).model_copy(
        update={"density_kg_m3": 1.2, "sound_speed_m_s": 343.0})
    ranges = load_porous_model_ranges(config_path("porous_model_ranges.toml"))
    pressures = [full_model_pressures(operators, materials, AXIS, air=air, ranges=ranges,
                                    max_relative_residual=1e-8, workers=workers) for workers in (1, 2)]
    _equal(pressures[0], pressures[1], "zoned endpoint")


if __name__ == "__main__":
    globals()[sys.argv[1]](Path(sys.argv[2]))
