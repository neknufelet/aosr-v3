"""#562 外部審查：動態幾何的微分順序與混合散射的零保留邊界。"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import cast

import numpy as np
import pytest

from aosr.runtime import child_process_env
from tests.engine.test_transfer_matrix_reference import GRADIENT_ABS, GRADIENT_REL


def _probe(directory: Path, scenario: str, dtype: str, mode: str) -> dict[str, object]:
    output = directory / "result.json"
    env = child_process_env(threads=1).copy()
    env["PYTHONPATH"] = os.pathsep.join((env["PYTHONPATH"], str(Path(__file__).resolve().parents[2])))
    run = subprocess.run([sys.executable, "-m", "tests.engine._jax_gradient_boundary_probe", str(output),
        scenario, dtype, mode], cwd=directory, env=env, capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stdout+run.stderr
    return cast(dict[str, object], json.loads(output.read_text()))


@pytest.mark.parametrize("dtype", ("float32", "float64"))
def test_zoned_gradient_transform_orders_are_bitwise_equal(tmp_path: Path, dtype: str) -> None:
    """抓權重閉包捕獲追蹤陣列；兩種編譯微分順序都要有限且逐位相同。"""
    result = _probe(tmp_path, "late", dtype, "alpha_bar")
    assert result["dtype"] == dtype
    assert np.isfinite(np.asarray(result["inside"], dtype=float)).all()
    assert np.isfinite(np.asarray(result["outside"], dtype=float)).all()
    assert result["bitwise"] is True


@pytest.mark.parametrize("dtype", ("float32", "float64"))
def test_zoned_energy_gradient_is_finite_in_both_transform_orders(tmp_path: Path, dtype: str) -> None:
    """同一個動態幾何接縫也須支援完整能量；每個真正參與的材料梯度均為負。"""
    result = _probe(tmp_path, "late", dtype, "late_reverberant_energy")
    assert result["dtype"] == dtype
    for mode in ("inside", "outside"):
        gradient = np.asarray(result[mode], dtype=float)
        assert np.isfinite(gradient).all()
        assert np.array_equal(gradient[0], np.zeros(2))
        assert np.all(gradient[1:] < 0.)


@pytest.mark.parametrize("dtype", ("float32", "float64"))
@pytest.mark.parametrize("mode", ("eager", "jit"))
def test_mixed_scattering_endpoint_gradient_matches_numpy_and_nearby_limit(tmp_path: Path, dtype: str, mode: str) -> None:
    """抓混合散射 s=1 的 0×∞；保留比例非零的鄰點也要走向 numpy 的零導數極限。"""
    result = _probe(tmp_path, "early", dtype, mode)
    points = cast(list[dict[str, object]], result["points"])
    for point in points:
        gradient = cast(float, point["gradient"])
        central = np.asarray(point["central"], dtype=float)
        assert point["valid"] is True
        assert np.isfinite(cast(float, point["energy"]))
        assert np.isfinite(gradient), point
        # 兩種精度都比對每單位阻抗的導數；雙精度另守材料考卷的參數尺度化尺。
        np.testing.assert_allclose(gradient, central, rtol=GRADIENT_REL, atol=GRADIENT_ABS)
        if dtype == "float64":
            impedance = cast(float, point["impedance"])
            np.testing.assert_allclose(gradient*impedance, central*impedance, rtol=GRADIENT_REL, atol=GRADIENT_ABS)
    endpoint, near, nearer = points
    assert np.array_equal(np.asarray(endpoint["scattering"]), np.ones((1, 1, 3)))
    assert endpoint["energy"] == 0.
    assert endpoint["gradient"] == 0.
    assert np.all(np.asarray(near["scattering"]) < 1.)
    assert np.all(np.asarray(nearer["scattering"]) < 1.)
    assert abs(cast(float, nearer["gradient"])) < abs(cast(float, near["gradient"]))
    endpoint_central = cast(list[float], endpoint["central"])
    assert abs(endpoint_central[1]) < abs(endpoint_central[0])
