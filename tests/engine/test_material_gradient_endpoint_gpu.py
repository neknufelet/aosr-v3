"""#1030 裝置從命令列到子行程、預設 CPU 相容與選用依賴隔離。"""
from __future__ import annotations

import sys
import subprocess
from pathlib import Path
from typing import Literal, cast
from unittest.mock import patch

import pytest

from aosr import runtime
from aosr.gradient import cli, worker
from aosr.physics import fem_full_gpu
from aosr.physics.fem_full_batch import FullSolveBatch
from tests.engine._parallel_full_solves_probe import WALLS, _operators


def test_endpoint_parser_defaults_cpu_and_accepts_gpu() -> None:
    parser = cli._parser()
    assert parser.parse_args(["endpoint", "/tmp/example"]).device == "cpu"
    assert parser.parse_args(["endpoint", "/tmp/example", "--device", "gpu"]).device == "gpu"
    with pytest.raises(SystemExit):
        parser.parse_args(["endpoint", "/tmp/example", "--device", "tpu"])


def test_endpoint_gpu_rejects_workers_before_reading_folder(tmp_path: Path) -> None:
    args = cli._parser().parse_args(["endpoint", str(tmp_path), "--device", "gpu", "--workers", "2"])
    with pytest.raises(ValueError, match="gpu.*workers"):
        cli._endpoint(args)


def _child_env(folder: Path, mode: str, device: Literal["cpu", "gpu"]) -> dict[str, str]:
    """攔住真正 worker 的啟動，只取產品組出的環境；不跑搜尋或終點。"""
    with patch.object(subprocess, "Popen") as start:
        start.return_value.wait.return_value = 0
        assert cli._spawn_and_wait(folder, mode, 4, device=device) == 0
    return cast(dict[str, str], start.call_args.kwargs["env"])


def test_endpoint_gpu_child_receives_device_and_jax_platform(tmp_path: Path) -> None:
    with patch.object(subprocess, "Popen") as start:
        start.return_value.wait.return_value = 0
        assert cli._spawn_and_wait(tmp_path, "endpoint", 4, device="gpu") == 0
    command = start.call_args.args[0]
    assert command[-2:] == ["--device", "gpu"]
    env = start.call_args.kwargs["env"]
    assert env["JAX_PLATFORMS"] == "cpu" and env["JAX_ENABLE_X64"] == "true"


@pytest.mark.parametrize("mode", ["start", "resume"])
def test_gpu_search_child_keeps_cuda_platform(tmp_path: Path, mode: str) -> None:
    env = _child_env(tmp_path, mode, "gpu")
    assert env["JAX_PLATFORMS"] == "cuda,cpu" and env["JAX_ENABLE_X64"] == "true"


def test_endpoint_worker_gpu_reaches_run_endpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime, "jax_settings", lambda: ("cpu", True))
    with patch.object(worker, "run_endpoint") as run:
        assert worker.main([str(tmp_path), "endpoint", "--device", "gpu"]) == 0
    assert run.call_args.args == (tmp_path.resolve(),)
    assert run.call_args.kwargs == {"workers": 1, "device": "gpu"}


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_endpoint_worker_rejects_cuda_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    device: Literal["cpu", "gpu"],
) -> None:
    monkeypatch.setattr(runtime, "jax_settings", lambda: ("cuda,cpu", True))
    with patch.object(worker, "run_endpoint") as run:
        assert worker.main([str(tmp_path), "endpoint", "--device", device]) == 1
    run.assert_not_called()
    assert "終點子行程要中央處理器＋雙精度" in capsys.readouterr().err


def test_endpoint_gpu_preserves_cpu_search_identity(tmp_path: Path) -> None:
    """用實際子行程環境重算兩份摘要，守住既有 CPU 搜尋的身分閘門。"""
    probe = """
import json
import jax
from aosr.config.capabilities import load_capabilities
from aosr.config.directivity_defaults import load_directivity_defaults
from aosr.config.paths import config_path
from aosr.reporting.physics_identity import physics_identity_parts
parts = physics_identity_parts(
    capabilities=load_capabilities(config_path('capabilities.toml')),
    directivity=load_directivity_defaults(config_path('directivity_defaults.toml')),
)
print(json.dumps({'environment_digest': parts.environment_digest, 'physics_identity': parts.identity}))
"""
    identities = []
    for mode, device in (("start", "cpu"), ("endpoint", "cpu"), ("endpoint", "gpu")):
        env = _child_env(tmp_path, mode, cast(Literal["cpu", "gpu"], device))
        result = subprocess.run([sys.executable, "-c", probe], env=env, cwd=tmp_path,
                                capture_output=True, text=True, check=True)
        identities.append(result.stdout.strip())
    assert identities[1] == identities[0], "CPU 終點必須保留 CPU 搜尋的身分"
    assert identities[2] == identities[0], "GPU 完整解不得改變 CPU 搜尋的身分"


def test_cpu_module_import_and_solve_without_nvmath(monkeypatch: pytest.MonkeyPatch) -> None:
    """選用依賴不存在時，模組能冷載入，CPU 完整解照常真算。"""
    monkeypatch.setitem(sys.modules, "nvmath", None)
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location("isolated_full_gpu", fem_full_gpu.__file__)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module_from_spec(spec))
    operators = _operators()
    with FullSolveBatch(operators, max_relative_residual=1e-9) as batch:
        answers = list(batch.map([(40.0, WALLS[0])]))
    assert answers[0].shape == (len(operators.sources), operators.stiffness.shape[0])


def test_stop_keeps_its_existing_arguments(tmp_path: Path) -> None:
    """endpoint 的新參數不准滲入 stop；有效舊入口仍建立停止記號。"""
    from aosr.gradient import run

    args = cli._parser().parse_args(["stop", str(tmp_path)])
    with patch.object(run, "read_project"):
        assert cli._stop(args) == 0
    assert (tmp_path / run.STOP_FILE).exists()
