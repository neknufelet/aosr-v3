"""#561：每題在單緒、同一組 CPU 釘選的冷行程內比較，沒有跨機器固定答案。"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from aosr import runtime
from tests.engine.test_wall_surfaces import _CPU_PIN_VARS


def _probe(case: str, tmp_path: Path) -> None:
    env = runtime.child_process_env(threads=1)
    runtime.configure_jax(platforms="cpu", enable_x64=True, env=env)
    env.update({name: os.environ[name] for name in _CPU_PIN_VARS if name in os.environ})
    if case == "environment":
        env.update(zip(_CPU_PIN_VARS, ("AVX2,FMA3", "Haswell", "--xla_cpu_multi_thread_eigen=false"), strict=True))
    result = subprocess.run(
        [sys.executable, "-m", "tests.engine._parallel_full_solves_probe", case, str(tmp_path)],
        env=env, capture_output=True, text=True, timeout=100, check=False)
    sys.stdout.write(result.stdout)
    assert result.returncode == 0, result.stdout + result.stderr


def test_build_snapshots_ledger_and_archive_are_bitwise_equal(tmp_path: Path) -> None:
    _probe("build", tmp_path)


def test_validation_pressures_and_receipt_are_bitwise_equal(tmp_path: Path) -> None:
    _probe("validation", tmp_path)


def test_endpoint_full_pressures_are_bitwise_equal(tmp_path: Path) -> None:
    _probe("endpoint", tmp_path)


def test_batch_keeps_order_reuses_solver_and_counts(tmp_path: Path) -> None:
    _probe("batch", tmp_path)


def test_worker_propagates_residual_and_pattern_errors(tmp_path: Path) -> None:
    _probe("errors", tmp_path)


@pytest.mark.parametrize("case", ["invalid", "environment", "cli", "cli_forwarding", "zoned"])
def test_worker_contracts(case: str, tmp_path: Path) -> None:
    _probe(case, tmp_path)
