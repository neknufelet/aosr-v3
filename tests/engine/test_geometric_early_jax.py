"""#562：分區命中、逐角阻抗、早期三欄與房間散射的獨立 numpy 對照。"""
from __future__ import annotations

import math
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import cast
import cmath
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aosr.geometry.shoebox import Point, Wall, wall_count_signature
from aosr.physics import amplitude as amp, amplitude_jax as aj
from aosr.physics import geometric_early_jax as gj
from aosr.physics.report_surfaces import rectangle_zone_at
from aosr.physics.room_paths import image_source_paths
from aosr.physics.room_paths import RoomPath
from tests.engine import _geometric_early_jax_cases as case
from aosr.runtime import child_process_env, configure_jax
from tests.engine._precision_contracts import MUTANT_MARGIN, contract_value
from tests.engine._geometric_early_jax_probe import ProbeRun
from tests.engine.test_amplitude_jax import _GRADIENT_REL

_TOLERANCE = contract_value("geometric_early_jax_vs_python")


def test_partition_hits_and_angles_match_numpy_bounce_lookup() -> None:
    """抓：所有反彈誤取首區、交換分區順序、角度重新捨入或去重重排。"""
    sources = case.reflection_sources("zoned")
    paths = image_source_paths(case.ROOM, case.SOURCES[0], case.RECEIVERS[0], case.SPEED, 3)
    table = aj.build_path_table(paths, case.RECEIVERS[0].as_tuple(), case.AXIS,
        rectangle_zones_by_wall=sources.rectangle_zones_by_wall)
    assert table.patch_ids == ("floor", "ceiling", "x0@low", "x0@high", "xL", "y0", "yL")
    for i, path in enumerate(paths):
        expected = []
        for bounce in path.bounces:
            for name in bounce.walls:
                zones = (sources.rectangle_zones_by_wall or {}).get(Wall.from_name(name))
                expected.append(name + ("@"+rectangle_zone_at(name, bounce.point, zones).id if zones else ""))
        actual = [table.patch_ids[int(p)] for p in table.bounce_patch[i, table.bounce_mask[i]]]
        assert sorted(actual) == sorted(expected)
    for patch, name in enumerate(table.patch_ids):
        wall = name.partition("@")[0]
        cosines = tuple(dict.fromkeys(amp.incidence_cos(amp._wall_axis(wall), p.dist_m,
            case.RECEIVERS[0].as_tuple(), p.image) for i, p in enumerate(paths)
            if np.any(table.bounce_mask[i] & (table.bounce_patch[i] == patch))))
        assert table.incidence_cosines[patch].tobytes() == np.asarray(cosines, dtype=np.float64).tobytes()
        assert table.incidence_angles_deg[patch].tobytes() == np.asarray(
            [math.degrees(math.acos(c)) for c in cosines], dtype=np.float64).tobytes()


@pytest.fixture(scope="module")
def measured(tmp_path_factory: pytest.TempPathFactory) -> ProbeRun:
    assert importlib.util.find_spec("aosr.physics.geometric_early_jax") is not None, "缺早期 JAX 入口"
    directory = tmp_path_factory.mktemp("geometric-early")
    env = child_process_env(threads=1).copy()
    configure_jax(platforms="cpu", enable_x64=True, env=env)
    env["PYTHONPATH"] = os.pathsep.join((env["PYTHONPATH"], str(Path(__file__).resolve().parents[2])))
    output = directory / "result.json"
    run = subprocess.run([sys.executable, "-m", "tests.engine._geometric_early_jax_probe", str(output)],
        cwd=directory, env=env, capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, run.stdout + run.stderr
    return cast(ProbeRun, json.loads(output.read_text()))


def test_double_reference_angles_gradients_and_transforms(measured: ProbeRun) -> None:
    """抓逐階縮放、同調相加、指向性、家具、區面積或垂直反射權重漏接。"""
    assert measured["x64"] is True
    for name, record in measured["cases"].items():
        assert record["double_dtypes"] == ["float64"]*4
        assert record["single_dtypes"] == ["float32"]*4
        for key in ("double", "double_jit"):
            for quantity, values in record[key].items():
                expected = np.asarray(record["reference"][quantity])
                assert within_contract(np.asarray(values), expected), (name, key, quantity)
        for single in (record["single"], record["single_jit"]):
            assert all(np.isfinite(values).all() for values in single.values())
    assert measured["legacy_bounces_equal"]
    assert measured["repeat_bitwise"]
    assert measured["vmap_difference"] <= _TOLERANCE
    for name, gradient in measured["gradients"].items():
        automatic, central = np.asarray(gradient["automatic"]), np.asarray(gradient["central"])
        assert np.all(np.abs(automatic-central) <= _GRADIENT_REL*np.abs(central)), (name, automatic, central)


def within_contract(actual: np.ndarray, expected: np.ndarray) -> bool:
    """三欄與 s 都逐格絕對差；零與負干涉沒有相對分母問題，非有限一律紅。"""
    return bool(np.all(np.isfinite(actual) & (np.abs(actual-expected) <= _TOLERANCE)))


def test_mutant_beyond_tolerance_is_red(measured: ProbeRun) -> None:
    """真算六種植錯，逐一證明超尺；另夾尺兩側，含零及負參考。"""
    assert set(measured["mutants"]) == {"wrong_zone", "normal_angle", "lost_order", "equal_area", "lost_directivity", "lost_furniture"}
    assert all(error > _TOLERANCE for error in measured["mutants"].values())
    reference = np.array([0., .25, -.5])
    assert within_contract(reference+(1.-MUTANT_MARGIN)*_TOLERANCE, reference)
    assert not within_contract(reference+(1.+MUTANT_MARGIN)*_TOLERANCE, reference)
    assert not within_contract(np.full(3, np.nan), reference)


def test_whole_path_table_existing_fields_bitwise_unchanged() -> None:
    """抓：整面路徑順序、反彈展開、角餘弦、相位或距離常數變位。"""
    from tests.engine._amplitude_jax_probe import _dense_case
    paths, receiver, _z, _rho_c, axis = _dense_case()
    table = aj.build_path_table(paths, receiver, axis)
    for name, expected in _legacy_whole_arrays(paths, receiver, axis).items():
        assert getattr(table, name).tobytes() == expected.tobytes(), name


def _legacy_whole_arrays(paths: list[RoomPath], receiver: tuple[float, float, float],
                         axis: tuple[float, ...]) -> dict[str, np.ndarray]:
    """起點 bfa62c0 的整面運算快照；同一 CPU 重算，避免把機器的 libm 位元寫死成雲端答案。"""
    bounces = []
    for path in paths:
        row = []
        counts = wall_count_signature(path.identity)
        for wall_index, wall in enumerate(amp.CANONICAL_WALLS):
            if counts.get(wall, 0):
                cos = amp.incidence_cos(amp._wall_axis(wall), path.dist_m, receiver, path.image)
                row.extend([(wall_index, cos)]*counts[wall])
        bounces.append(row)
    shape = (len(paths), max(len(row) for row in bounces))
    walls, cosines, mask = np.zeros(shape, np.int64), np.ones(shape, np.float64), np.zeros(shape, np.bool_)
    phase = np.empty((len(paths), len(axis)), np.complex128)
    for i, (path, row) in enumerate(zip(paths, bounces, strict=True)):
        for j, (wall_index, cos) in enumerate(row):
            walls[i, j], cosines[i, j], mask[i, j] = wall_index, cos, True
        for k, f in enumerate(axis):
            omega = 2.*math.pi*f
            phase[i, k] = cmath.exp(complex(0., -omega*path.delay_s))
    return {"frequencies_hz": np.array(axis, np.float64), "order": np.array([p.order for p in paths], np.int64),
        "dist_m": np.array([p.dist_m for p in paths], np.float64), "delay_s": np.array([p.delay_s for p in paths], np.float64),
        "bounce_wall": walls, "bounce_cos": cosines, "bounce_mask": mask,
        "inverse_distance": np.array([1./p.dist_m for p in paths], np.float64), "phase": phase}


def _flag_inputs() -> tuple[gj.GeometricEarlyTable, tuple[jax.Array, ...], jax.Array, jax.Array]:
    # 聲源與座位的水平座標相同，地板一次反彈恰為垂直入射，cos=1 不靠捨入。
    receiver = Point(case.SOURCES[0].x, case.SOURCES[0].y, case.RECEIVERS[0].z)
    table = gj.build_geometric_early_table(room=case.ROOM, sources=case.SOURCES[:1], receivers=(receiver,),
        source_models=case.models()[:1], sound_speed_m_s=case.SPEED, rho_c_pa_s_per_m=case.RHO*case.SPEED,
        frequencies_hz=(125., 503., 1009.))
    rows = tuple(jnp.full((3, len(a)), 900.+0j, dtype=jnp.complex64) for a in table.incidence_angles_deg)
    return table, rows, jnp.full((6, 3), 900.+0j, dtype=jnp.complex64), jnp.arange(6, dtype=jnp.float32)/10.


@pytest.mark.parametrize("bad", ("normal_nan", "normal_pole", "bounce_nan", "bounce_pole", "scattering_nan", "scattering_range"))
@pytest.mark.parametrize("quantity", ("reflected_energy", "interference_energy", "scattering"))
def test_flags_nan_and_valid_frequency_gradients(bad: str, quantity: str) -> None:
    """抓：編譯時丟錯、旗標不亮、無效頻點不是 NaN，或遮罩反向污染其他頻點。"""
    table, rows, normal, scattering = _flag_inputs()
    row_list = list(rows)
    if bad.startswith("normal"):
        normal = normal.at[0, 0].set(jnp.nan if bad.endswith("nan") else -table.rho_c)
    elif bad.startswith("bounce"):
        p = table.pairs[0].paths
        i, j = np.argwhere(p.bounce_mask & (p.bounce_cos == 1.) & (p.order == 1)[:, None])[0]
        patch, angle = int(p.bounce_patch[i, j]), int(p.bounce_angle[i, j])
        value = jnp.nan if bad.endswith("nan") else -table.rho_c
        row_list[patch] = row_list[patch].at[0, angle].set(value)
        if bad == "bounce_pole":
            cos = jnp.asarray(p.bounce_cos[i, j], dtype=row_list[patch].real.dtype)
            denominator = row_list[patch][0, angle]*cos+table.rho_c
            assert complex(denominator) == 0j
    else:
        scattering = jnp.broadcast_to(scattering[:, None], (6, 3)).at[0, 0].set(
            jnp.nan if bad.endswith("nan") else 2.)
    solve = jax.jit(lambda z, n, s: gj.solve_geometric_early(table, z, n, s))
    result = solve(tuple(row_list), normal, scattering)
    room_s, room_valid = jax.jit(lambda n, s: gj.room_scattering(table, n, s))(normal, scattering)
    assert bool(room_valid[0]) == bad.startswith("bounce")
    if not bad.startswith("bounce"):
        assert np.isnan(room_s[0])
    assert not bool(result.valid[0, 0, 0])
    assert bool(result.reflection_valid[0, 0, 0]) == (not bad.startswith("bounce"))
    assert bool(result.scattering_valid[0, 0, 0]) == bad.startswith("bounce")
    assert np.asarray(result.valid[..., 1:]).all()
    assert np.asarray(result.reflection_valid[..., 1:]).all()
    assert np.asarray(result.scattering_valid[..., 1:]).all()
    assert np.isnan(result.reflected_energy[..., 0]).all()
    assert np.isnan(result.interference_energy[..., 0]).all()
    assert np.isnan(result.scattering[..., 0]).all()
    assert np.isfinite(result.direct_energy).all()
    objective = lambda z, n, s: jnp.nansum(getattr(solve(z, n, s), quantity))
    grads = jax.grad(objective, argnums=(0, 1, 2))(tuple(row_list), normal, scattering)
    assert all(np.isfinite(g).all() for g in (*grads[0], grads[1], grads[2]))
    assert any(np.any(np.asarray(g)[..., 1:] != 0) for g in (grads[1], grads[2]))


def test_directivity_rejects_phase_before_taking_real(monkeypatch: pytest.MonkeyPatch) -> None:
    """抓：日後指向倍率帶相位時，取實部靜靜丟掉聲壓。"""
    def with_phase(paths: list[RoomPath], *args: object, **kwargs: object) -> list[RoomPath]:
        return [replace(p, path_pressure=tuple(z*1j for z in p.path_pressure)) for p in paths]
    monkeypatch.setattr(gj, "apply_pressure_factor", with_phase)
    with pytest.raises(AssertionError, match="指向性倍率"):
        gj.build_geometric_early_table(room=case.ROOM, sources=case.SOURCES[1:], receivers=case.RECEIVERS[:1],
            source_models=case.models()[1:], sound_speed_m_s=case.SPEED, rho_c_pa_s_per_m=case.RHO*case.SPEED,
            frequencies_hz=(125., 503., 1009.))


def test_reference_probe_uses_configured_reflection_order(tmp_path: Path) -> None:
    """抓：改交接階數時探針仍以寫死的三階建立晚期結果。"""
    env = child_process_env(threads=1).copy()
    configure_jax(platforms="cpu", enable_x64=True, env=env)
    env["PYTHONPATH"] = os.pathsep.join((env["PYTHONPATH"], str(Path(__file__).resolve().parents[2])))
    code = """
from aosr.config import three_lane_crossover as config
config.REFLECTION_ORDER_K = 2
from tests.engine import _geometric_early_jax_probe as probe
probe.case.AXIS = (125., 503., 1009.)
columns = probe.reference_columns(probe.case.SOURCES[:1], probe.case.RECEIVERS[:1],
    probe.case.models()[:1], probe.case.reflection_sources('whole', probe.case.AXIS), None)
assert len(columns['reflected_energy']) == 1
assert len(columns['reflected_energy'][0]) == 3
"""
    run = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env,
        capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stdout+run.stderr


@pytest.mark.parametrize("s", (0., 1.))
def test_scattering_endpoints_and_fully_absorbing_room_have_finite_gradients(s: float) -> None:
    """抓：零反射權重除零、s=1 的零保留壓力在反向傳播產生 0×∞。"""
    table, rows, normal, scattering = _flag_inputs()
    scattering = jnp.full_like(scattering, s)
    for n in (normal, jnp.full_like(normal, table.rho_c)):
        result = jax.jit(lambda z: gj.solve_geometric_early(table, rows, z, scattering))(n)
        assert np.asarray(result.valid).all()
        assert np.array_equal(np.asarray(result.scattering), np.full(table.shape, 0. if bool(jnp.all(n == table.rho_c)) else s))
        grad = jax.grad(lambda z: jnp.sum(gj.solve_geometric_early(table, rows, z, scattering).reflected_energy))(n)
        assert np.isfinite(grad).all()
