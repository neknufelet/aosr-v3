"""矩形晚期 JAX 對 numpy 雙精度，沿用現有契約。"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from aosr.geometry.shoebox import Wall
from aosr.physics.late_energy import solve_late_energy
from aosr.runtime import child_process_env
from tests.engine._precision_contracts import contract_value
from tests.engine.test_late_energy_zones import synthetic


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    directory = tmp_path_factory.mktemp("zone-jax")
    output = directory / "output.json"
    probe = Path(__file__).with_name("_late_zone_jax_probe.py")
    env = child_process_env(threads=1)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    process = subprocess.run([sys.executable, str(probe), str(output)], cwd=directory,
        env=env, capture_output=True, text=True, timeout=60)
    assert process.returncode == 0, process.stdout + process.stderr
    return cast(dict[str, object], json.loads(output.read_text()))


@pytest.mark.parametrize("wall", Wall.all())
def test_zoned_jax_matches_numpy_and_jit(runs: dict[str, object], wall: Wall) -> None:
    """抓 JAX 只用面平均、矩形軸序錯、或沒有以小塊吸音率建反射算子。"""
    bands = solve_late_energy(synthetic(wall, 2, 1.0/3.0, 0.5)).bands
    expected = [[band.alpha_bar for band in bands], [band.raw_reverberant_energy for band in bands],
                [band.eyring_ratio for band in bands], [band.late_reverberant_energy for band in bands]]
    run = cast(dict[str, object], runs[wall.wall_name()])
    for mode in ("eager", "jit"):
        actual = cast(list[list[float]], run[mode])
        np.testing.assert_allclose(actual[:4], expected, rtol=contract_value("late_energy_jax_vs_numpy"), atol=0.0)
        assert all(actual[4])
    gradient = np.asarray(run["gradient"])
    assert np.all(np.isfinite(gradient))
    assert np.all(gradient[6:] < 0.0)
    assert run["dtype"] == "float64"


def test_whole_wall_jax_control_is_bitwise_unchanged(runs: dict[str, object]) -> None:
    """727acf825736dbd2da26280fabb88e97cd16e557 的 CPU 雙精度實量四格控制。"""
    assert runs["control"] == CONTROL


CONTROL = {'alpha_bar': ['0x1.1bc125ee2cc8ap-1', '0x1.1bc125ee2cc8ap-1', '0x1.1bc125ee2cc8ap-1'], 'raw_reverberant_energy': ['0x1.50e31309a7553p-2', '0x1.50e31309a7553p-2', '0x1.50e31309a7553p-2'], 'eyring_ratio': ['0x1.5f393b7bd5ab9p-1', '0x1.5f393b7bd5ab9p-1', '0x1.5f393b7bd5ab9p-1'], 'late_reverberant_energy': ['0x1.ce32a7f0c5402p-3', '0x1.ce32a7f0c5402p-3', '0x1.ce32a7f0c5402p-3']}


def test_each_zone_uses_the_original_material_function_and_exact_frequency_selection(runs: dict[str, object]) -> None:
    """固定、構造、實測與貼原牆四區：逐區吸音率與註記必須等於原整面函式。"""
    assert runs["material_rows_equal"] == [True] * 4
    assert runs["selected_rows_equal"] is True


@pytest.mark.parametrize(("name", "alphas"), (("one", (1.0,) * 4), ("same", (0.7,) * 4),
                                              ("mixed", (0.2, 0.4, 0.6, 0.8))))
def test_jax_weighted_absorption_obeys_participating_bounds(
    runs: dict[str, object], name: str, alphas: tuple[float, ...],
) -> None:
    """牆與小塊的浮點凸組合不得越界；均勻 α 必須恰等於原值。"""
    from tests.engine.test_late_energy_zones_bounds import rounding_problem
    run = cast(dict[str, object], cast(dict[str, object], runs["bounds"])[name])
    bands = solve_late_energy(rounding_problem(alphas)).bands
    expected = [[b.alpha_bar for b in bands], [b.raw_reverberant_energy for b in bands],
                [b.eyring_ratio for b in bands], [b.late_reverberant_energy for b in bands]]
    for mode in ("eager", "jit"):
        averages = cast(dict[str, list[list[float]]], run[mode])
        assert min(alphas) <= averages["walls"][0][0] <= max(alphas)
        patches = np.asarray(averages["patches"])[:4, 0]
        assert np.all(patches >= (alphas[0], alphas[1], alphas[2], alphas[3]))
        assert np.all(patches <= max(alphas))
        energy = cast(dict[str, list[list[float]]], run["energy"])[mode]
        np.testing.assert_allclose(energy[:4], expected,
            rtol=contract_value("late_energy_jax_vs_numpy"), atol=0.0)
