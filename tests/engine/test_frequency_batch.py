"""分頻批次：同程式聲壓及反傳對照、編譯暫存上界、輸入與舊檔相容。"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import cast

import jax
import pytest

from aosr.gradient import cli, run
from aosr.runtime import child_process_env, configure_jax
from tests.engine._frequency_batch_probe import problem
from tests.engine.test_fem_reduced_jax import REVERSE_FORWARD_REL, _decode, _within
from tests.engine.test_material_gradient_mount import CONTROL_PROJECT, _digest, _project
from tests.engine.test_material_gradient_max_depth import _identity


@pytest.fixture(scope="module")
def numeric(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    directory = tmp_path_factory.mktemp("frequency-batch")
    env = child_process_env(threads=1)
    configure_jax(platforms="cpu", enable_x64=True, env=env)
    env["PYTHONPATH"] = os.pathsep.join((env["PYTHONPATH"], str(Path(__file__).resolve().parents[2])))
    result = subprocess.run([sys.executable, "-m", "tests.engine._frequency_batch_probe"], cwd=directory,
                            env=env, capture_output=True, text=True, timeout=100)
    assert result.returncode == 0, result.stdout + result.stderr
    return cast(dict[str, object], json.loads(result.stdout))


@pytest.mark.parametrize("batch", (1, 7, 5, 14, 20))
def test_batched_pressure_value_and_gradient(numeric: dict[str, object], batch: int) -> None:
    """抓頻序、餘數遺失、反傳斷線；同程式基準，用既有聲壓尺及自動微分捨入尺。"""
    import numpy as np

    reference = cast(dict[str, object], numeric["None"])
    actual = cast(dict[str, object], numeric[str(batch)])
    assert _within(_decode(actual["pressure"]), _decode(reference["pressure"]))
    for key in ("value", "gradient"):
        expected, result = _decode(reference[key]), _decode(actual[key])
        assert np.isfinite(result).all()
        assert np.all(np.abs(result - expected) <= REVERSE_FORWARD_REL * np.sum(np.abs(expected)))
    assert np.any(np.abs(_decode(reference["gradient"])) > 0)


def test_compiled_temporary_memory_is_bounded(numeric: dict[str, object]) -> None:
    """抓拿掉重算後反傳保存全部 LU；頻數翻倍只准增加小於一個未分批頻數增量的暫存。"""
    memory = cast(dict[str, int], numeric["memory"])
    sys.stdout.write(json.dumps(memory, sort_keys=True) + "\n")
    for name in ("forward", "gradient"):
        whole, whole_double = memory[f"{name}:14:None"], memory[f"{name}:28:None"]
        batched, batch_double = memory[f"{name}:14:7"], memory[f"{name}:28:7"]
        assert 0 < batched < whole and 0 < batch_double < whole_double
        # 翻倍後分批仍低於原頻數全批暫存，且增加量小於全批的頻數增量。
        assert batch_double < whole
        assert batch_double - batched < whole_double - whole


@pytest.mark.parametrize("bad", (0, -1, 1.5, True, "7"))
def test_rejects_invalid_batch(bad: object) -> None:
    from aosr.physics.fem_reduced_jax import reduced_pressures

    constants, z, rigid = problem(3)
    call = cast(Callable[..., jax.Array], reduced_pressures)
    with pytest.raises(ValueError, match="frequency_batch.*正整數"):
        call(constants, z, rigid_mask=rigid, frequency_batch=bad)
    with pytest.raises(ValueError, match="frequency_batch"):
        run.GradientProject.model_validate({**_project().model_dump(), "frequency_batch": bad})


def test_optional_project_bytes_and_identity(tmp_path: Path) -> None:
    project = _project()
    run.write_json_atomic(tmp_path / run.PROJECT_FILE, project)
    assert _digest((tmp_path / run.PROJECT_FILE).read_bytes()) == CONTROL_PROJECT
    explicit = run.GradientProject.model_validate({**project.model_dump(), "frequency_batch": None})
    run.write_json_atomic(tmp_path / "explicit.json", explicit)
    assert (tmp_path / "explicit.json").read_bytes() == (tmp_path / run.PROJECT_FILE).read_bytes()
    assert _identity() == "198075c7e78d6f5eaa13d9a1f0ab5de4"
    def identity(batch: int | None) -> str:
        return run.gradient_id(scheme_sha256="a" * 64, surrogate_manifest_sha256="b" * 64, presets=project.presets,
            purpose=project.purpose, settings_fingerprint="c" * 64, physics_identity="physics-x",
            objective_settings_fingerprint="d" * 64, frequency_batch=batch)
    assert identity(None) == _identity()
    assert identity(7) != identity(None) and identity(7) != identity(1)
    changed = run.GradientProject.model_validate({**project.model_dump(), "frequency_batch": 7})
    run.write_json_atomic(tmp_path / run.PROJECT_FILE, changed)
    assert run.read_project(tmp_path).frequency_batch == 7


@pytest.mark.parametrize("value", ("0", "-2", "1.5"))
def test_cli_rejects_invalid_batch(value: str) -> None:
    with pytest.raises(SystemExit):
        cli._parser().parse_args(["start", "--scheme", "/s", "--surrogate", "/m", "--presets",
            ",".join(["none"] * 6), "--purpose", "music", "--root", "/r", "--frequency-batch", value])


@pytest.mark.parametrize("batch", (None, 7))
def test_cli_report_optional_batch(tmp_path: Path, capsys: pytest.CaptureFixture[str], batch: int | None) -> None:
    project = run.GradientProject.model_validate({**_project().model_dump(), "frequency_batch": batch})
    run.write_json_atomic(tmp_path / run.PROJECT_FILE, project)
    assert cli.main(["report", "--folder", str(tmp_path)]) == 0
    text = capsys.readouterr().out
    if batch is None:
        assert "替身頻點分批" not in text
    else:
        assert "替身頻點分批：每批 7 點（反傳逐批重算）" in text


@pytest.mark.parametrize("command", ("resume", "endpoint"))
def test_batch_is_only_a_start_option(command: str) -> None:
    with pytest.raises(SystemExit):
        cli._parser().parse_args([command, "/folder", "--frequency-batch", "7"])
