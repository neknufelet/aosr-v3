"""#562 2c：窗內逐路徑能量、四區超標與主位類代價的可微等價考卷。"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import NotRequired, TypedDict, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.typing import NDArray

from aosr.physics.amplitude_jax import bounce_impedances
from aosr.physics.reflection_energy_jax import reflection_path_energy
from aosr.config.quality_targets import SettingEntry, TargetEntry
from aosr.runtime import child_process_env, configure_jax
from aosr.scoring.reflections_jax import (
    ReflectionCostArrays, primary_category_cost, reflection_cost_arrays, reflection_geometry, reflections_arrays,
)
from aosr.scoring import reflections_jax
from tests.engine._precision_contracts import MUTANT_MARGIN, contract_value
from tests.engine._reflections_jax_cases import CASES, case, directional_inputs, numpy_score, purpose, reference
from tests.engine._scheme_cache import shared_json
from tests.engine.test_furniture_reflections import furniture_records
from tests.engine.test_transfer_matrix_reference import GRADIENT_ABS, GRADIENT_REL

_PROBE = Path(__file__).with_name("_reflections_jax_probe.py")
# 沿用 test_stage_one_objective_jax 的無量綱中央差分步長與 test_transfer_matrix_reference 的尺。
_STEP = 1e-5
_SCORE_QUANTITIES = ("zone_excess_db", "zone_cost", "category_cost")


class CaseRun(TypedDict):
    eager: dict[str, list[str]]
    jit: dict[str, list[str]]
    again: dict[str, list[str]]
    single: dict[str, list[str]]
    gradient: list[str]
    scattering_gradient: list[str]
    double_dtypes: list[str]
    single_dtypes: list[str]
    gradient_dtypes: list[str]
    single_gradient_dtypes: list[str]
    single_gradient: list[list[str]]
    bounce_gradient: NotRequired[list[str]]
    bounce_central: NotRequired[list[str]]
    directional_64: NotRequired[dict[str, list[str]]]
    directional_32: NotRequired[dict[str, list[str]]]
    zero_scattering_64: NotRequired[dict[str, list[str]]]


def _decode(values: list[str]) -> NDArray[np.float64]:
    return np.asarray([float.fromhex(v) for v in values], dtype=np.float64)


def _within(actual: NDArray[np.float64], expected: NDArray[np.float64], tolerance: float) -> bool:
    return bool(np.all(np.isfinite(actual)) and np.all(np.abs(actual - expected) <= tolerance))


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory, worker_id: str) -> dict[str, CaseRun]:
    directory = tmp_path_factory.mktemp("reflections-jax")
    output = directory / "output.json"
    env = child_process_env(threads=1)
    configure_jax(platforms="cpu", enable_x64=True, env=env)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join((env["PYTHONPATH"], str(Path(__file__).resolve().parents[2])))
    env["TMPDIR"] = str(directory)
    def produce() -> str:
        process = subprocess.run([sys.executable, str(_PROBE), str(output), str(_STEP)], cwd=directory, env=env,
                                 text=True, capture_output=True, timeout=300)
        assert process.returncode == 0, process.stdout + process.stderr
        return output.read_text(encoding="utf-8")
    result = cast(dict[str, object], json.loads(shared_json(tmp_path_factory, worker_id, "reflections-jax", produce)))
    assert result["x64"] is True and result["backend"] == "cpu"
    return cast(dict[str, CaseRun], result["cases"])


def test_threshold_scale_and_asymmetric_weights_come_from_registry() -> None:
    """改登記簿的門檻、比例尺及非對稱權重，仍照 Python 裁判，抓偷寫預設計分數字。"""
    data = case("room")
    base = purpose()
    goal = base.entry("reflections_and_echo.zone_excess_db")
    assert isinstance(goal, TargetEntry)
    zones = ("front", "lateral", "rear", "vertical")
    offsets = {f"reflections_and_echo.zone_threshold_db.{zone}": (i + 1) * goal.worse_reference
               for i, zone in enumerate(zones)}
    settings = base.model_copy(update={
        "setting": tuple(e.model_copy(update={"value": float(e.value) - offsets[e.key]})
                         if e.key in offsets and isinstance(e.value, float) else e for e in base.setting),
        "target": tuple(e.model_copy(update={"worse_reference": len(data.records) * e.worse_reference})
                        if e.key == goal.key else e for e in base.target),
        "weight": tuple(w.model_copy(update={"item": tuple(i.model_copy(update={"value": (zones.index(i.name) + 1) * i.value})
                         for i in w.item)})
                        if w.key == "reflections_and_echo.within_category_weights" else w for w in base.weight),
    })
    energies = reference(data)["path_energy"]
    expected = numpy_score(data, tuple(energies), settings)
    actual = []
    for i, table in enumerate(data.tables):
        geometry = reflection_geometry(table, data.directions[i], data.axis, settings)
        assert len(set(geometry.thresholds_db)) == len(zones)
        assert len(set(geometry.zone_weights)) == len(zones)
        actual.append(jax.jit(lambda e: reflection_cost_arrays(geometry, e, jnp.asarray(True)))(
            jnp.asarray(energies[i], dtype=jnp.float32)))
    # 四區量互異，包含沒有反射的 rear；交換任何一對門檻或權重都有可見影響。
    assert all(len(set(row)) == len(zones) for row in expected["zone_cost"])
    # 單精度量法檢查沿用 test_amplitude_jax 的 1e-6 相對容差，正式雙精度尺另由子行程判。
    np.testing.assert_allclose(np.stack([r.zone_cost for r in actual]), expected["zone_cost"], rtol=1e-6, atol=0)
    np.testing.assert_allclose(primary_category_cost(jnp.stack([r.channel_cost for r in actual])),
                               expected["category_cost"], rtol=1e-6, atol=0)


@pytest.mark.parametrize("name", CASES)
def test_double_matches_python(runs: dict[str, CaseRun], name: str) -> None:
    """抓窗、方向、門檻、逐點先截零、八度權重、四區與兩聲道平均算錯。"""
    expected = reference(case(name))
    tolerance = contract_value("reflections_cost_jax_vs_python")
    for mode in ("eager", "jit"):
        actual = runs[name][mode]
        for quantity in _SCORE_QUANTITIES:
            truth = expected[quantity].reshape(-1)
            assert _within(_decode(actual[quantity]), truth, tolerance), (name, mode, quantity)
        # 能量平方後另有實量的相對尺；零值仍須精確為零。
        np.testing.assert_allclose(_decode(actual["path_energy"]), expected["path_energy"].reshape(-1),
                                   rtol=contract_value("reflection_path_energy_jax_vs_python"), atol=0)
        assert np.all(_decode(actual["valid"]) == 1)


@pytest.mark.parametrize("name", CASES)
def test_precision_compilation_and_batch(runs: dict[str, CaseRun], name: str) -> None:
    """雙精度與單精度的值／梯度型別都跟輸入走；編譯重跑逐位，單精度只量不登記尺。"""
    run = runs[name]
    assert set(run["double_dtypes"] + run["gradient_dtypes"]) == {"float64"}
    assert set(run["single_dtypes"] + run["single_gradient_dtypes"]) == {"float32"}
    for key in (*_SCORE_QUANTITIES, "path_energy"):
        assert np.array_equal(_decode(run["jit"][key]), _decode(run["again"][key]))
        assert np.all(np.isfinite(_decode(run["single"][key])))
    assert all(np.all(np.isfinite(_decode(g))) for g in run["single_gradient"])


def central_difference(name: str) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    data = case(name)
    coordinates = np.stack([data.z.real, data.z.imag]) / np.abs(data.z)
    gradient = np.zeros_like(coordinates)
    for index in np.ndindex(coordinates.shape):
        plus, minus = coordinates.copy(), coordinates.copy()
        plus[index] += _STEP
        minus[index] -= _STEP
        upper_z = (plus[0] + 1j * plus[1]) * np.abs(data.z)
        lower_z = (minus[0] + 1j * minus[1]) * np.abs(data.z)
        gradient[index] = (reference(data, upper_z)["category_cost"] - reference(data, lower_z)["category_cost"]) / (2 * _STEP)
    scattering_gradient = np.zeros_like(data.scattering)
    for column in range(data.scattering.size):
        plus_s, minus_s = data.scattering.copy(), data.scattering.copy()
        plus_s[column] += _STEP
        minus_s[column] -= _STEP
        scattering_gradient[column] = (reference(data, scatter=plus_s)["category_cost"]
                                       - reference(data, scatter=minus_s)["category_cost"]) / (2 * _STEP)
    return gradient, scattering_gradient


@pytest.mark.parametrize("name", CASES)
def test_gradient_every_coordinate_against_python(runs: dict[str, CaseRun], name: str) -> None:
    """每個牆頻格的實／虛部與每個散射頻格：對既有 Python 評分的中央差分。"""
    expected_z, expected_s = central_difference(name)
    actual_z = _decode(runs[name]["gradient"]).reshape(expected_z.shape)
    actual_s = _decode(runs[name]["scattering_gradient"])
    assert np.any(np.abs(actual_z) > 0) and np.any(np.abs(actual_s) > 0)
    np.testing.assert_allclose(actual_z, expected_z, rtol=GRADIENT_REL, atol=GRADIENT_ABS)
    np.testing.assert_allclose(actual_s, expected_s, rtol=GRADIENT_REL, atol=GRADIENT_ABS)
    if name == "complex_frequency":
        # 逐牆那題對 Python 裁判；這題再讓每個反彈獨立變動，考 3780 個實／虛部座標。
        bounce = _decode(runs[name]["bounce_gradient"])
        central = _decode(runs[name]["bounce_central"])
        assert np.any(bounce != 0) and np.all(np.isfinite(bounce))
        np.testing.assert_allclose(bounce, central, rtol=GRADIENT_REL, atol=GRADIENT_ABS)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0])
def test_bad_energy_only_invalidates_used_zone_and_keeps_other_gradients(bad: float) -> None:
    data = case("room")
    geometry = reflection_geometry(data.tables[0], data.directions[0], data.axis, purpose())
    energy = jnp.asarray(reference(data)["path_energy"][0], dtype=jnp.float32)
    zone, path = np.argwhere(geometry.zone_mask)[0]
    column = int(np.flatnonzero(geometry.frequency_weights > 0)[0])
    energy = energy.at[path, column].set(bad)
    call = jax.jit(lambda e: reflection_cost_arrays(geometry, e, jnp.asarray(True)))
    result = call(energy)
    assert not result.zone_valid[zone] and jnp.isnan(result.zone_excess_db[zone]) and jnp.isnan(result.channel_cost)
    assert np.count_nonzero(~np.asarray(result.zone_valid)) == 1
    gradient = jax.jit(jax.grad(lambda e: jnp.nansum(call(e).zone_cost)))(energy)
    assert np.all(np.isfinite(gradient))


@pytest.mark.parametrize("failure", ["nan_z", "singular_z", "scatter_nan", "scatter_negative", "scatter_above_one",
                                    "factor_nan", "direct_zero", "window", "frequencies"])
def test_dynamic_failure_flags_nan_and_safe_gradients(failure: str) -> None:
    data = case("room")
    table = data.tables[0]
    geometry = reflection_geometry(table, data.directions[0], data.axis, purpose())
    z = bounce_impedances(table, jnp.asarray(data.z, dtype=jnp.complex64))
    scatter = jnp.asarray(data.scattering, dtype=jnp.float32)
    factor = jnp.ones((table.path_count, scatter.size), dtype=z.dtype)
    zone, path = np.argwhere(geometry.zone_mask)[0]
    bounce = int(np.flatnonzero(table.bounce_mask[path])[0])
    column = int(np.flatnonzero(geometry.frequency_weights > 0)[0])
    complete = jnp.asarray(failure != "window")
    if failure == "frequencies":
        geometry = replace(geometry, frequency_coverage=False)
    if failure in ("nan_z", "singular_z"):
        if failure == "singular_z":
            # 用 cos=1 的精確奇異題，免得 -rho/cos 在單精度乘回後只接近奇異。
            cos = table.bounce_cos.copy()
            cos[path, bounce] = 1
            table = replace(table, bounce_cos=cos)
            geometry = replace(geometry, paths=table)
        z = z.at[path, bounce, column].set(jnp.nan if failure == "nan_z" else -data.rho_c)
    if failure.startswith("scatter_"):
        scatter = scatter.at[column].set({"scatter_nan": jnp.nan, "scatter_negative": -1.0, "scatter_above_one": 2.0}[failure])
    if failure == "factor_nan":
        factor = factor.at[path, column].set(jnp.nan)
    if failure == "direct_zero":
        factor = factor.at[table.direct_index(), column].set(0)
    call = jax.jit(lambda values, s, gain: reflections_arrays(geometry, values, s, data.rho_c, complete, gain))
    result = call(z, scatter, factor)
    assert not bool(result.zone_valid[zone]) and bool(jnp.isnan(result.zone_cost[zone]))
    assert bool(jnp.isnan(result.channel_cost))
    if failure in ("nan_z", "singular_z"):
        assert not bool(result.energy_flags.impedance_valid[path, column])
    if failure.startswith("scatter_"):
        assert not bool(result.energy_flags.scattering_valid[column])
    if failure == "direct_zero":
        assert not bool(result.energy_flags.direct_valid[column])
    gradients = jax.jit(jax.grad(lambda values, s, gain: jnp.nansum(call(values, s, gain).zone_cost),
                                argnums=(0, 1, 2)))(z, scatter, factor)
    assert all(np.all(np.isfinite(g)) for g in gradients)


def test_empty_zero_full_scattering_and_ties_are_finite() -> None:
    """空區／全吸收不是錯；最大值並列採 jnp.max 次梯度，沒有擅改為軟最大值。"""
    data = case("room")
    geometry = reflection_geometry(data.tables[0], data.directions[0], data.axis, purpose())
    energy = jnp.zeros((data.tables[0].path_count, data.z.shape[-1]), dtype=jnp.float32)
    def call(e: jax.Array) -> ReflectionCostArrays:
        return reflection_cost_arrays(geometry, e, jnp.asarray(True))
    result = jax.jit(call)(energy)
    assert np.all(result.zone_excess_db == 0) and np.all(result.zone_valid)
    assert np.all(np.isfinite(jax.grad(lambda e: call(e).channel_cost)(energy)))
    zone = next(i for i, row in enumerate(geometry.zone_mask) if np.count_nonzero(row) >= 2)
    paths = np.flatnonzero(geometry.zone_mask[zone])[:2]
    tied = energy.at[paths].set(1)
    gradient = jax.grad(lambda e: call(e).channel_cost)(tied)
    assert np.any(np.asarray(gradient[paths[0]]) > 0)
    assert np.array_equal(gradient[paths[0]], gradient[paths[1]])
    z = bounce_impedances(data.tables[0], jnp.asarray(data.z, dtype=jnp.complex64))
    full = jnp.ones_like(jnp.asarray(data.scattering, dtype=jnp.float32))
    def scatter_call(s: jax.Array) -> jax.Array:
        return reflections_arrays(geometry, z, s, data.rho_c, jnp.asarray(True)).channel_cost
    assert scatter_call(full) == 0
    assert np.all(np.isfinite(jax.grad(scatter_call)(full)))


def test_window_boundary_frequency_exclusion_and_vmap() -> None:
    data = case("room")
    table = data.tables[0]
    geometry = reflection_geometry(table, data.directions[0], data.axis, purpose())
    energy = jnp.asarray(reference(data)["path_energy"][0], dtype=jnp.float32)
    excluded = np.flatnonzero(~np.any(geometry.zone_mask, axis=0))
    bad = energy.at[excluded].set(jnp.nan)
    bad = bad.at[:, np.flatnonzero(geometry.frequency_weights == 0)].set(jnp.nan)
    def call(e: jax.Array) -> ReflectionCostArrays:
        return reflection_cost_arrays(geometry, e, jnp.asarray(True))
    actual = jax.jit(jax.vmap(call))(jnp.stack([energy, bad]))
    assert np.all(actual.zone_valid)
    assert np.array_equal(actual.zone_excess_db[0], actual.zone_excess_db[1])
    assert primary_category_cost(jnp.asarray([1.0, 3.0])) == 2
    assert jnp.isnan(primary_category_cost(jnp.asarray([1.0, jnp.nan])))
    assert np.all(np.isfinite(jax.grad(lambda v: jnp.nansum(primary_category_cost(v)))(jnp.asarray([1.0, jnp.nan]))))


def test_window_upper_boundary_is_inclusive() -> None:
    data = case("room")
    table = data.tables[0]
    entry = purpose().entry("reflections_and_echo.window_upper_ms")
    assert isinstance(entry, SettingEntry)
    assert isinstance(entry.value, float)
    window = entry.value / 1000.0
    path = int(np.flatnonzero(table.order > 0)[0])
    delays = table.delay_s - table.delay_s[table.direct_index()]
    delays[path] = window
    boundary = reflection_geometry(replace(table, delay_s=delays.copy()), data.directions[0], data.axis, purpose())
    assert np.any(boundary.zone_mask[:, path])
    delays[path] = np.nextafter(window, np.inf)
    outside = reflection_geometry(replace(table, delay_s=delays), data.directions[0], data.axis, purpose())
    assert not np.any(outside.zone_mask[:, path])


def test_pressure_factor_direct_normalization_and_unused_bounces() -> None:
    """固定聲源指向性是輸入；直達也要乘它。未用反彈填 NaN／奇異值均不得污染梯度。"""
    data = case("room")
    table = data.tables[0]
    geometry = reflection_geometry(table, data.directions[0], data.axis, purpose())
    z = bounce_impedances(table, jnp.asarray(data.z, dtype=jnp.complex64))
    padded = jnp.where(jnp.asarray(table.bounce_mask)[..., None], z, jnp.asarray(jnp.nan, dtype=z.dtype))
    scatter = jnp.asarray(data.scattering, dtype=jnp.float32)
    gain = jnp.asarray(np.random.default_rng(562).uniform(0.2, 1.0, (table.path_count, scatter.size)), dtype=jnp.float32)
    plain = reflections_arrays(geometry, z, scatter, data.rho_c, jnp.asarray(True))
    actual = reflections_arrays(geometry, padded, scatter, data.rho_c, jnp.asarray(True), gain)
    expected = plain.path_energy * gain ** 2 / gain[table.direct_index()] ** 2
    # 單精度只考物理關係，沿用現有梯度考卷的相對尺，不登記 GPU 尺。
    np.testing.assert_allclose(actual.path_energy, expected, rtol=1e-6, atol=0)
    gradient = jax.grad(lambda values: reflections_arrays(geometry, values, scatter, data.rho_c,
                                                         jnp.asarray(True), gain).channel_cost)(padded)
    assert np.all(np.isfinite(gradient))
    assert np.all(np.asarray(gradient)[~table.bounce_mask] == 0)


def test_zero_scattering_is_valid_and_matches_numpy(runs: dict[str, CaseRun]) -> None:
    """把 s>=0 植成 s>0 應紅；合法的零散射仍有有限值與梯度。"""
    data = case("room")
    table = data.tables[0]
    geometry = reflection_geometry(table, data.directions[0], data.axis, purpose())
    z = bounce_impedances(table, jnp.asarray(data.z, dtype=jnp.complex64))
    zero = jnp.zeros_like(jnp.asarray(data.scattering, dtype=jnp.float32))
    call = jax.jit(lambda s: reflections_arrays(geometry, z, s, data.rho_c, jnp.asarray(True)))
    actual = call(zero)
    expected = reference(data, scatter=np.zeros_like(data.scattering))
    assert np.all(actual.energy_flags.scattering_valid) and np.all(actual.zone_valid)
    assert np.all(np.isfinite(actual.path_energy)) and np.isfinite(actual.channel_cost)
    double = runs["room"]["zero_scattering_64"]
    np.testing.assert_allclose(_decode(double["path_energy"]), expected["path_energy"].reshape(-1),
                               rtol=contract_value("reflection_path_energy_jax_vs_python"), atol=0)
    for quantity in _SCORE_QUANTITIES:
        assert _within(_decode(double[quantity]), expected[quantity].reshape(-1),
                       contract_value("reflections_cost_jax_vs_python")), quantity
    assert np.all(np.isfinite(jax.jit(jax.grad(lambda s: call(s).channel_cost))(zero)))


@pytest.mark.parametrize("zero_source", ["impedance", "factor"])
def test_exact_zero_pressure_has_finite_gradients(zero_source: str) -> None:
    """守住 re²+im²：匹配阻抗或零指向性造成精確零壓力時，反向傳播不能 NaN。"""
    data = case("room")
    table = data.tables[0]
    path = int(np.flatnonzero(table.order == 1)[0])
    bounce = int(np.flatnonzero(table.bounce_mask[path])[0])
    cos = table.bounce_cos.copy()
    cos[path, bounce] = 1
    table = replace(table, bounce_cos=cos)
    z = bounce_impedances(table, jnp.asarray(data.z, dtype=jnp.complex64))
    scatter = jnp.asarray(data.scattering, dtype=jnp.float32)
    factor = jnp.ones((table.path_count, scatter.size), dtype=z.dtype)
    if zero_source == "impedance":
        z = z.at[path, bounce].set(data.rho_c)
    else:
        factor = factor.at[path].set(0)
    call = jax.jit(lambda values, gain: reflection_path_energy(table, values, scatter, data.rho_c, gain))
    result = call(z, factor)
    assert np.all(result.valid) and np.all(result.energy[path] == 0)
    gradients = jax.jit(jax.grad(lambda values, gain: jnp.sum(call(values, gain).energy), argnums=(0, 1)))(z, factor)
    assert all(np.all(np.isfinite(g)) for g in gradients)
    # 零壓力附近能量是正的二次量；梯度的方向導數也要保留。
    # 新版 JAX 的 abs(p)**2 一階梯度雖有限，零點曲率卻會被算成 0。
    def perturbed_energy(step: jax.Array) -> jax.Array:
        if zero_source == "impedance":
            return jnp.sum(call(z.at[path, bounce].add(step * data.rho_c), factor).energy)
        return jnp.sum(call(z, factor.at[path].add(step)).energy)
    curvature = jax.jit(jax.grad(jax.grad(perturbed_energy)))(jnp.asarray(0, dtype=scatter.dtype))
    assert np.isfinite(curvature) and curvature > 0


def test_directional_direct_normalization_matches_numpy(runs: dict[str, CaseRun]) -> None:
    """對既有 numpy 有指向性路徑表；漏乘直達指向性不能被 JAX 對自己的比對放過。"""
    data = case("room")
    energies = []
    for channel, table in enumerate(data.tables):
        factors, energy = directional_inputs(data, channel)
        assert np.all(factors[table.direct_index()] > 0) and np.any(factors[table.direct_index()] < 1)
        energies.append(energy)
    expected = {**numpy_score(data, tuple(energies)), "path_energy": np.stack(energies)}
    actual = runs["room"]["directional_64"]
    np.testing.assert_allclose(_decode(actual["path_energy"]), expected["path_energy"].reshape(-1),
                               rtol=contract_value("reflection_path_energy_jax_vs_python"), atol=0)
    for quantity in _SCORE_QUANTITIES:
        assert _within(_decode(actual[quantity]), expected[quantity].reshape(-1),
                       contract_value("reflections_cost_jax_vs_python")), quantity


def test_approximate_coverage_is_estimable_like_numpy() -> None:
    """把 approximate 誤當未覆蓋應紅；家具近似輸入沿用 numpy 的可估語意。"""
    data = case("room")
    records = furniture_records(data.records)
    approximate = replace(data, records=(records[0], records[1]))
    expected = reference(approximate)
    outputs = []
    for channel, table in enumerate(data.tables):
        window = records[channel].window
        assert window is not None and window.coverage == "approximate"
        coverage_complete = jnp.asarray(reflections_jax.coverage_is_estimable(window.coverage))
        geometry = reflection_geometry(table, data.directions[channel], data.axis, purpose())
        actual = jax.jit(lambda z, s: reflections_arrays(
            geometry, bounce_impedances(table, z), s, data.rho_c, coverage_complete))(
                jnp.asarray(data.z, dtype=jnp.complex64), jnp.asarray(data.scattering, dtype=jnp.float32))
        assert np.all(actual.zone_valid) and np.isfinite(actual.channel_cost)
        np.testing.assert_allclose(actual.zone_cost, expected["zone_cost"][channel], rtol=1e-6, atol=0)
        outputs.append(actual.channel_cost)
    np.testing.assert_allclose(primary_category_cost(jnp.stack(outputs)), expected["category_cost"], rtol=1e-6, atol=0)
    assert reflections_jax.coverage_is_estimable("complete")
    assert not reflections_jax.coverage_is_estimable("not_provable")
    assert not reflections_jax.coverage_is_estimable("missing")


def test_path_energy_mutant_beyond_tolerance_is_red(runs: dict[str, CaseRun]) -> None:
    """能量相對尺的界內捨入過、逐格越界紅；零能量不借絕對地板遮掉錯誤。"""
    tolerance = contract_value("reflection_path_energy_jax_vs_python")
    for name in CASES:
        truth = reference(case(name))["path_energy"].reshape(-1)
        actual = _decode(runs[name]["jit"]["path_energy"])
        np.testing.assert_allclose(actual, truth, rtol=tolerance, atol=0)
        np.testing.assert_allclose(truth * (1 + (1 - MUTANT_MARGIN) * tolerance), truth, rtol=tolerance, atol=0)
        for index in np.flatnonzero(truth):
            mutant = actual.copy()
            mutant[index] = truth[index] * (1 + (1 + MUTANT_MARGIN) * tolerance)
            with pytest.raises(AssertionError):
                np.testing.assert_allclose(mutant, truth, rtol=tolerance, atol=0)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(np.asarray([tolerance]), np.asarray([0.0]), rtol=tolerance, atol=0)


def test_mutant_beyond_tolerance_is_red(runs: dict[str, CaseRun]) -> None:
    """真的 JAX 輸出逐格植錯；類代價與逐區量每一格越界都紅，界內與原值過。"""
    tolerance = contract_value("reflections_cost_jax_vs_python")
    expected = reference(case("room"))
    for quantity in _SCORE_QUANTITIES:
        truth = expected[quantity].reshape(-1)
        actual = _decode(runs["room"]["jit"][quantity])
        assert _within(actual, truth, tolerance)
        assert _within(truth + (1 - MUTANT_MARGIN) * tolerance, truth, tolerance)
        for index in range(actual.size):
            mutant = actual.copy()
            mutant[index] = truth[index] + (1 + MUTANT_MARGIN) * tolerance
            assert not _within(mutant, truth, tolerance), (quantity, index)
