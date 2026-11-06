"""顯卡模式的 CPU 考卷：身分相容、環境、精度閘門與顯式裝置常數。"""
from __future__ import annotations

import subprocess

from pathlib import Path
from typing import Literal

import pytest

from aosr.gradient import cli, run
from tests.engine.test_material_gradient_mount import CONTROL_PROJECT, _digest, _project


def test_cpu_project_bytes_and_gpu_identity(tmp_path: Path) -> None:
    project = _project()
    explicit = run.GradientProject.model_validate({**project.model_dump(), "device": "cpu"})
    run.write_json_atomic(tmp_path / run.PROJECT_FILE, explicit)
    assert _digest((tmp_path / run.PROJECT_FILE).read_bytes()) == CONTROL_PROJECT
    def identity(device: Literal["cpu", "gpu"]) -> str:
        return run.gradient_id(scheme_sha256="a" * 64, surrogate_manifest_sha256="b" * 64,
            presets=project.presets, purpose=project.purpose, settings_fingerprint="c" * 64,
            physics_identity="physics-x", objective_settings_fingerprint="d" * 64, device=device)
    assert identity("cpu") == "198075c7e78d6f5eaa13d9a1f0ab5de4"
    assert identity("gpu") != identity("cpu")
    gpu = run.GradientProject.model_validate({**project.model_dump(), "device": "gpu"})
    run.write_json_atomic(tmp_path / run.PROJECT_FILE, gpu)
    assert run.read_project(tmp_path).device == "gpu"


@pytest.mark.parametrize("device", ("cpu", "gpu"))
def test_device_parser(device: str) -> None:
    args = ["start", "--scheme", "/s", "--surrogate", "/m", "--presets", ",".join(["none"] * 6),
            "--purpose", "music", "--root", "/r"]
    assert cli._parser().parse_args(args).device == "cpu"
    assert cli._parser().parse_args([*args, "--device", device]).device == device
    with pytest.raises(SystemExit):
        cli._parser().parse_args([*args, "--device", "tpu"])


@pytest.mark.parametrize(("device", "mode", "platforms"), (
    ("cpu", "start", "cpu"), ("gpu", "start", "cuda,cpu"),
    ("gpu", "resume", "cuda,cpu"), ("gpu", "endpoint", "cpu")))
def test_child_environment(device: Literal["cpu", "gpu"], mode: str, platforms: str,
                           monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class Child:
        def __init__(self, command: list[str], *, env: dict[str, str], **kwargs: object) -> None:
            assert env["JAX_PLATFORMS"] == platforms
            assert env["JAX_ENABLE_X64"].lower() == "true"
        def wait(self) -> int:
            return 0
    monkeypatch.setattr(subprocess, "Popen", Child)
    assert cli._spawn_and_wait(tmp_path, mode, 0, device=device) == 0


def test_precision_gate_accepts_and_rejects() -> None:
    from aosr.gradient.gpu import precision_comparison, require_precision
    from tests.engine._precision_contracts import contract_value

    ruler = contract_value("rom_search_gpu_single_vs_cpu_double")
    for difference in (0.0, ruler / 2, ruler):
        comparison = precision_comparison(cpu_double=0.0, gpu_single=difference, theta=(0.05,))
        require_precision(comparison)
        assert comparison.absolute_difference_db == difference
    with pytest.raises(ValueError, match="objective_total.*dB"):
        require_precision(precision_comparison(cpu_double=0.0, gpu_single=ruler * 2, theta=(0.05,)))
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError):
            precision_comparison(cpu_double=0.0, gpu_single=bad, theta=(0.05,))


@pytest.fixture(scope="module")
def numeric(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    import json
    import os
    import subprocess
    import sys
    from typing import cast
    from aosr.runtime import child_process_env, configure_jax

    env = child_process_env(threads=1)
    configure_jax(platforms="cpu", enable_x64=True, env=env)
    env["PYTHONPATH"] = os.pathsep.join((env["PYTHONPATH"], str(Path(__file__).resolve().parents[2])))
    result = subprocess.run([sys.executable, "-m", "tests.engine._material_gradient_gpu_probe"],
        cwd=tmp_path_factory.mktemp("gpu-single"), env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    return cast(dict[str, object], json.loads(result.stdout))


def test_single_chain_within_registered_ruler(numeric: dict[str, object]) -> None:
    from typing import cast
    from tests.engine._precision_contracts import contract_value
    assert abs(cast(float, numeric["single"]) - cast(float, numeric["double"])) <= contract_value(
        "rom_search_gpu_single_vs_cpu_double")
    assert numeric["dtype"] == numeric["gradient_dtype"] == "float64"
    assert numeric["gradient_finite"] and numeric["gradient_nonzero"] and numeric["double_gradient_finite"]
    assert "float64" not in cast(list[str], numeric["array_dtypes"])
    assert "complex128" not in cast(list[str], numeric["array_dtypes"])


def test_perturbed_constant_exceeds_single_precision_contract(numeric: dict[str, object]) -> None:
    from typing import cast
    from aosr.gradient.gpu import precision_comparison, require_precision
    # 把同一合成題 stiffness[0,0] 加倍；恢復常數或移除產品拒收都會讓本題紅。
    with pytest.raises(ValueError, match="objective_total.*dB"):
        require_precision(precision_comparison(cpu_double=cast(float, numeric["double"]),
            gpu_single=cast(float, numeric["mutated"]), theta=(0.05,)))


def test_single_chain_ledger_row_passes_check_scores(numeric: dict[str, object]) -> None:
    """評分仍在單精度算、只把結果轉雙精度時，真實帳本列的分數恆等式會失敗。"""
    from typing import cast
    from aosr.gradient import material_gradient_ledger as ledger
    from tests.engine._material_gradient_gpu_probe import model
    from tests.engine.test_material_gradient import _objective_constants

    step = cast(dict[str, object], numeric["ledger_step"])
    row = ledger.MaterialGradientRow.model_validate({
        **{key: value for key, value in step.items() if key != "theta"},
        "phase": "descent",
        "theta_hex": [float(value).hex() for value in cast(list[float], step["theta"])],
        "round": None, "candidate": None, "stop_reason": None, "final_trial_step": None,
        "evaluations": None, "snap": None, "jump": None, "decision": None})
    ledger.check_scores(ledger.score_constants(_objective_constants(model())), (row,))


def test_device_constants_are_arguments_and_dots_highest(numeric: dict[str, object]) -> None:
    from typing import cast
    assert numeric["embedded"] == []
    signature = cast(str, numeric["argument_signature"])
    for shape in ("16x16xf32", "6x16x16xf32", "7x16xf32", "2x16xcomplex<f32>"):
        assert shape in signature
    dots = cast(list[str], numeric["dots"])
    assert dots
    assert all("HIGHEST" in line and "DEFAULT" not in line for line in dots)


@pytest.mark.parametrize("mode", ("start", "resume"))
@pytest.mark.parametrize("accepted", (True, False))
def test_preflight_precedes_search_and_records_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                       mode: str, accepted: bool) -> None:
    from typing import cast
    from types import SimpleNamespace
    from aosr.gradient import gpu_preflight
    from aosr.gradient.gpu import precision_comparison
    from tests.engine._precision_contracts import contract_value

    ruler = contract_value("rom_search_gpu_single_vs_cpu_double")
    comparison = precision_comparison(cpu_double=0.0, gpu_single=ruler / 2 if accepted else ruler * 2, theta=(0.05,))
    project = run.GradientProject.model_validate({**_project().model_dump(), "device": "gpu"})
    run.write_json_atomic(tmp_path / run.PROJECT_FILE, project)
    monkeypatch.setattr(gpu_preflight, "measure_start", lambda project, problem: comparison)
    monkeypatch.setattr(run, "load_problem", lambda project: cast(run.GradientProblem, SimpleNamespace(header=None)))
    class SearchReached(Exception):
        pass
    def reached(*args: object, **kwargs: object) -> None:
        raise SearchReached
    monkeypatch.setattr(run, "write_ledger", reached)
    entry = run.run_search if mode == "start" else run.resume_search
    if accepted:
        with pytest.raises(SearchReached):
            entry(tmp_path, should_stop=None)
    else:
        with pytest.raises(ValueError, match="objective_total.*dB"):
            entry(tmp_path, should_stop=None)
    saved = run.read_project(tmp_path).precision_check
    assert saved == comparison
    assert not (tmp_path / run.LEDGER_FILE).exists()
    # resume 對已保存的拒收也必須拒跑，不能藉已有記錄繞過。
    if not accepted:
        with pytest.raises(ValueError, match="objective_total.*dB"):
            run.resume_search(tmp_path, should_stop=None)


def test_cpu_reference_environment_and_same_theta(monkeypatch: pytest.MonkeyPatch) -> None:
    import json
    from aosr.gradient.gpu_preflight import cpu_reference

    project = run.GradientProject.model_validate({**_project().model_dump(), "device": "gpu"})
    theta = (0.071, 0.125)
    def child(command: list[str], *, input: str, env: dict[str, str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert command[-1] == "aosr.gradient.gpu_reference"
        assert env["JAX_PLATFORMS"] == "cpu"
        assert env["JAX_ENABLE_X64"].lower() == "true"
        config, values = input.split("\n")
        assert run.GradientProject.model_validate_json(config) == project
        assert json.loads(values) == [x.hex() for x in theta]
        return subprocess.CompletedProcess(command, 0, stdout=3.25.hex() + "\n")
    monkeypatch.setattr(subprocess, "run", child)
    assert cpu_reference(project, theta) == 3.25


@pytest.mark.parametrize("cached_start", (False, True))
def test_preflight_theta_matches_first_descent_ledger_row(tmp_path: Path, cached_start: bool) -> None:
    import os
    import sys
    from aosr.runtime import child_process_env, configure_jax

    env = child_process_env(threads=1)
    configure_jax(platforms="cpu", enable_x64=True, env=env)
    env["PYTHONPATH"] = os.pathsep.join((env["PYTHONPATH"], str(Path(__file__).resolve().parents[2])))
    source = """
from pathlib import Path
import sys
import pytest
from tests.engine.test_material_gradient_gpu import _assert_preflight_start
with pytest.MonkeyPatch.context() as patch:
    _assert_preflight_start(Path.cwd(), patch, cached_start=sys.argv[1] == 'cached')
"""
    result = subprocess.run([sys.executable, "-c", source, "cached" if cached_start else "fresh"], cwd=tmp_path, env=env,
                            capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr


def _assert_preflight_start(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, cached_start: bool) -> None:
    from types import SimpleNamespace
    from typing import cast
    import jax.numpy as jnp
    from aosr.gradient import gpu_preflight, material_gradient as search, material_gradient_ledger as ledger
    from aosr.gradient.gpu import precision_comparison
    from tests.engine._material_gradient_loop_probe import synthetic_header, synthetic_objective
    from tests.engine.test_material_gradient_max_depth import PRESET, _kept, _settings

    settings = _settings()
    designs = run.designs_for((PRESET,) * 6, settings, kept=_kept(), max_depths={"x0": 0.08})
    objective = synthetic_objective(designs, [0])
    problem = cast(run.GradientProblem, SimpleNamespace(designs=designs, objective=objective))
    project = run.GradientProject.model_validate({**_project().model_dump(), "device": "gpu", "max_depths": {"x0": 0.08}})
    if cached_start:
        project = project.model_copy(update={"precision_check": precision_comparison(
            cpu_double=0.0, gpu_single=0.0, theta=tuple(float(x) for x in search.baseline_theta(designs)))})
    run.write_json_atomic(tmp_path / run.PROJECT_FILE, project)
    reference_points: list[tuple[str, ...]] = []

    def reference(project: run.GradientProject, theta: tuple[float, ...]) -> float:
        reference_points.append(tuple(value.hex() for value in theta))
        return float(objective(jnp.asarray(theta))[0])

    monkeypatch.setattr(gpu_preflight, "cpu_reference", reference)
    gpu_preflight.ensure_start_precision(tmp_path, project, problem)
    result = search.projected_descent(objective, search.baseline_theta(designs), search.design_bounds(designs),
        settings.descent.model_copy(update={"max_steps": 1}), limits=search.depth_limits(designs))
    header = synthetic_header(designs, settings).model_copy(update={"max_depths": {"x0": 0.08}})
    ledger.write_ledger(tmp_path / run.LEDGER_FILE, header, ledger.descent_rows(result, "descent", 0))
    _, rows = ledger.read_ledger(tmp_path / run.LEDGER_FILE)
    saved = run.read_project(tmp_path).precision_check
    assert saved is not None
    assert rows[0].theta()[4] == 0.05 and sum(rows[0].theta()[4:6]) <= 0.08
    assert rows[0].theta_hex != tuple(float(x).hex() for x in search.baseline_theta(designs))
    assert saved.theta_hex == rows[0].theta_hex, (saved.theta_hex, rows[0].theta_hex)
    assert reference_points == [rows[0].theta_hex]
    assert saved.gpu_single == float(objective(jnp.asarray(rows[0].theta()))[0])
    monkeypatch.setattr(gpu_preflight, "measure_start", lambda *args: pytest.fail("起點相符的把關記錄不應重測"))
    gpu_preflight.ensure_start_precision(tmp_path, run.read_project(tmp_path), problem)


def test_create_forwards_persisted_gpu_device(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:

    project = run.GradientProject.model_validate({**_project().model_dump(), "device": "gpu"})
    class Child:
        def __init__(self, command: list[str], *, env: dict[str, str], **kwargs: object) -> None:
            assert env["JAX_PLATFORMS"] == "cuda,cpu"
            assert env["JAX_ENABLE_X64"].lower() == "true"
            assert run.read_project(Path(command[-2])).device == "gpu"
        def wait(self) -> int:
            return 0
    monkeypatch.setattr(subprocess, "Popen", Child)
    assert cli._create_and_run(tmp_path, project) == 0


def test_project_and_candidate_use_single_chain_and_cpu_reference(numeric: dict[str, object]) -> None:
    import math
    from typing import cast
    from tests.engine._precision_contracts import contract_value

    wiring = cast(dict[str, object], numeric["wiring"])
    assert math.isfinite(cast(float, wiring["candidate_value"]))
    assert wiring["candidate_gradient_dtype"] == "float64"
    assert abs(cast(float, wiring["reference"]) - cast(float, wiring["single"])) <= contract_value(
        "rom_search_gpu_single_vs_cpu_double")


@pytest.mark.parametrize("backend", ("cpu", "gpu"))
def test_worker_refuses_gpu_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str) -> None:
    import json
    import jax
    from aosr import runtime
    from aosr.gradient import worker

    project = run.GradientProject.model_validate({**_project().model_dump(), "device": "gpu"})
    run.write_json_atomic(tmp_path / run.PROJECT_FILE, project)
    monkeypatch.setattr(runtime, "jax_settings", lambda: ("cuda,cpu", True))
    monkeypatch.setattr(jax, "default_backend", lambda: backend)
    reached: list[str] = []
    monkeypatch.setattr(worker, "run_search", lambda *args, **kwargs: reached.append("search"))
    code = worker.run_worker(tmp_path, "start", should_stop=lambda: False)
    status = json.loads((tmp_path / run.STATUS_FILE).read_text())
    if backend == "cpu":
        assert code == 1 and reached == []
        assert status["state"] == "failed" and "不能退回 CPU" in status["message"]
    else:
        assert code == 0 and reached == ["search"]
        assert status["state"] == "finished"
