"""顯卡搜尋接線在 CPU 驗共用裝置常數；沒有顯卡也必跑。"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import cast

import pytest

from aosr.runtime import child_process_env, configure_jax


def probe(tmp_path: Path, mode: str) -> dict[str, object]:
    env = child_process_env(threads=1)
    configure_jax(platforms="cpu", enable_x64=True, env=env)
    env["PYTHONPATH"] = os.pathsep.join((env["PYTHONPATH"], str(Path(__file__).resolve().parents[2])))
    result = subprocess.run([sys.executable, "-m", "tests.engine._gpu_shared_constants_probe", mode],
                            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    return cast(dict[str, object], json.loads(result.stdout))


@pytest.fixture(scope="module")
def sharing(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    return probe(tmp_path_factory.mktemp("shared-constants"), "sharing")


def test_initial_and_candidates_share_constant_leaf_objects(sharing: dict[str, object]) -> None:
    assert sharing["initial_reused"]
    assert all(cast(list[bool], sharing["shared"])), sharing


def test_three_candidates_allocate_less_than_one_constant_copy(sharing: dict[str, object]) -> None:
    assert cast(int, sharing["growth"]) < cast(int, sharing["constant_bytes"]), sharing


@pytest.mark.parametrize("batch", ("full", "16"))
def test_shared_value_and_gradient_match_legacy_bits(tmp_path: Path, batch: str) -> None:
    result = probe(tmp_path, batch)
    assert all(cast(list[bool], result["matches"])), result


@pytest.mark.parametrize("mode", ("accept", "reject"))
def test_round_result_releases_discarded_objectives_and_compiled_functions(tmp_path: Path, mode: str) -> None:
    result = probe(tmp_path, mode)
    assert result["accepted"] == (mode == "accept"), result
    assert cast(int, result["discarded"]) > 0
    assert result["retained"] == 0, result
    assert result["chosen_callable"]
