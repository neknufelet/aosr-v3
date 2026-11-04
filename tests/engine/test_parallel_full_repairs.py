"""#561 審查修補：有期限的冷行程反例，工人最多兩個。"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from aosr import runtime
from tests.engine.test_wall_surfaces import _CPU_PIN_VARS


@pytest.mark.parametrize("case", ["anchor", "environment", "parent_death", "gate_death",
                                  "prefix_error", "cancel", "budget"])
def test_parallel_review_regressions(case: str, tmp_path: Path) -> None:
    env = runtime.child_process_env(threads=1)
    runtime.configure_jax(platforms="cpu", enable_x64=True, env=env)
    env.update({name: os.environ[name] for name in _CPU_PIN_VARS if name in os.environ})
    result = subprocess.run(
        [sys.executable, "-m", "tests.engine._parallel_full_repairs_probe", case, str(tmp_path)],
        env=env, capture_output=True, text=True, timeout=25, check=False)
    sys.stdout.write(result.stdout)
    assert result.returncode == 0, result.stdout + result.stderr
