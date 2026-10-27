"""#562 第二支：鏡像法反射乘積、路徑壓力與總量的 JAX（可微分的陣列算數函式庫）版的考卷。

**雙精度只在子行程量得到。** 同程式的考卷跑在 JAX 預設精度（雙精度關，見 ``tests/engine/conftest.py``），
雙精度輸入會被悄悄降成單精度，所以「JAX 雙精度對現有實作」由 ``_amplitude_jax_probe.py`` 在一個開了
雙精度的冷子行程裡量，回原始數字；契約裁判（:func:`contract_violations`）住這裡。四組案例：兩組三階 63 條
振幅答案的輸入（flat／varied，``blueprint/reference_amplitude_*.json``）、同一間房 220 點幾何路軸配隨頻率變
的阻抗（dense）、兩面構造牆的 Z(f, θ)（construction，對照現有 ``WallReflectionSources.apply``）。

**契約（登記簿 ``amplitude_jax_vs_python``，暫定）。** 同一個基底餵進現有的界線函式：反射乘積每分量絕對差
（``reflection_tolerance``）、每條路徑壓力相對差（``pressure_tolerance``）、總壓力（``total_pressure_tolerance``
的平方相加開根）、反射能量（``reflected_energy_tolerance``，地板取同一個基底）、直達能量
（``direct_energy_tolerance``，同一個基底）。不另寫新公式。

**其他幾題。** ``jax.grad`` 對 Z 跟中央差分比；eager、``jax.jit``、``jax.vmap`` 逐位相同（雙精度在子行程、
單精度在同程式）；單精度那條有限、dtype 跟輸入走；直達反射乘積恰 1；路徑表拒收分格材料、零距離、壞頻率軸
與形狀不對的阻抗；變異考卷證明裁判咬得住界線兩側。只寫 ``tmp_path``（規矩卡 tests-isolated-from-real-env）。
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import aosr
from aosr.physics import amplitude as amp
from aosr.physics import amplitude_jax as aj
from aosr.physics import totals as tot
from aosr.physics.room_paths import RoomPath, image_source_paths, load_room_input
from aosr.runtime import child_process_env, configure_jax
from tests.engine._precision_contracts import MUTANT_MARGIN, contract_value
from tests.engine.test_amplitude import _input_case

_BASE = contract_value("amplitude_jax_vs_python")
_CASES = ("flat", "varied", "dense", "construction")
_CPU_PIN_VARS = ("NPY_DISABLE_CPU_FEATURES", "OPENBLAS_CORETYPE", "XLA_FLAGS")
# 中央差分的步長取 |Z| 的 1e-6，截斷誤差與捨入誤差都在梯度尺度的 1e-6 以下；這是考卷自己的量測容差，不是精度契約。
_GRADIENT_REL = 1e-6


# ── 契約裁判 ──────────────────────────────────────────────────────────────────


def _stub_paths(order: list[int], delay: list[float], refl: np.ndarray, pressure: np.ndarray) -> list[RoomPath]:
    """只帶界線函式要讀的欄位（階數、到達時間、反射乘積、壓力）的參考路徑。"""
    return [RoomPath(index=i, order=o, identity=(0, 1, 0, 1, 0, 1), image=(0.0, 0.0, 0.0), dist_m=1.0,
                     delay_s=t, bounces=(), reflection_product=tuple(complex(v) for v in refl[i]),
                     path_pressure=tuple(complex(v) for v in pressure[i]))
            for i, (o, t) in enumerate(zip(order, delay, strict=True))]


def contract_violations(reference: dict[str, np.ndarray], candidate: dict[str, np.ndarray], order: list[int],
                        delay: list[float], frequencies: tuple[float, ...], base: float) -> tuple[list[str], dict[str, float]]:
    """候選對參考逐格比；回（超界清單、每種量「差÷界線」最大值）。參考值是現有 Python 雙精度實作。

    零值政策：參考的反射乘積恰為 0 時（例如垂直入射打到 Z=ρc 的牆），壓力界線公式的 1/|反射乘積| 那一項沒定義，
    那一格的反射乘積與壓力界線都當 0——差也必須恰為 0 才過；總壓力與反射能量的平方相加只收那個頻點反射乘積
    非零的路徑（零的那幾條已在逐格要求差恰為 0）；反射路徑壓力和恰為 0 時反射能量界線也當 0。
    """
    paths = _stub_paths(order, delay, reference["refl"], reference["pressure"])
    violations: list[str] = []
    worst = dict.fromkeys(("refl", "pressure", "total", "direct_energy", "reflected_energy"), 0.0)

    def judge(kind: str, where: str, diff: float, bound: float) -> None:
        worst[kind] = max(worst[kind], diff / bound if bound else (0.0 if diff == 0.0 else float("inf")))
        if not diff <= bound:
            violations.append(f"{kind}{where} 差 {diff!r} > 界線 {bound!r}")

    for i, path in enumerate(paths):
        for k, f in enumerate(frequencies):
            ref, got = reference["refl"][i, k], candidate["refl"][i, k]
            zero = ref == 0
            bound = 0.0 if zero else amp.reflection_tolerance(f, path.delay_s, abs(ref), base)
            for slot, a, b in (("real", ref.real, got.real), ("imag", ref.imag, got.imag)):
                judge("refl", f"[{i},{k}].{slot}", abs(a - b), bound)
            p_ref = reference["pressure"][i, k]
            rel = 0.0 if zero else amp.pressure_tolerance(f, path.delay_s, abs(ref), base)
            judge("pressure", f"[{i},{k}]", abs(candidate["pressure"][i, k] - p_ref), rel * abs(p_ref))
    for k in range(len(frequencies)):
        live = [p for p in paths if p.reflection_product[k] != 0]
        judge("total", f"[{k}]", abs(candidate["total"][k] - reference["total"][k]),
              tot.total_pressure_tolerance(live, k, frequencies, base))
        direct = reference["direct_energy"][k]
        judge("direct_energy", f"[{k}]", abs(candidate["direct_energy"][k] - direct),
              tot.direct_energy_tolerance(paths, k, frequencies, base) * direct)
        reflected = reference["reflected_energy"][k]
        reflected_bound = 0.0 if reflected == 0 else (
            tot.reflected_energy_tolerance(live, k, frequencies, base, base) * reflected)
        judge("reflected_energy", f"[{k}]", abs(candidate["reflected_energy"][k] - reflected), reflected_bound)
    return violations, worst


# ── 探針資料解碼 ──────────────────────────────────────────────────────────────


def _complex_array(node: object) -> np.ndarray:
    array = np.asarray(node, dtype=object)
    real = np.asarray(np.vectorize(float.fromhex, otypes=[float])(array[..., 0]), dtype=np.float64)
    imag = np.asarray(np.vectorize(float.fromhex, otypes=[float])(array[..., 1]), dtype=np.float64)
    return np.asarray(real + 1j * imag, dtype=np.complex128)


def _real_array(node: object) -> np.ndarray:
    return np.asarray(np.vectorize(float.fromhex, otypes=[float])(np.asarray(node, dtype=object)), dtype=np.float64)


def _decoded(node: object) -> dict[str, np.ndarray]:
    table = cast(dict[str, object], node)
    return {"refl": _complex_array(table["refl"]), "pressure": _complex_array(table["pressure"]),
            "total": _complex_array(table["total"]), "direct_energy": _real_array(table["direct_energy"]),
            "reflected_energy": _real_array(table["reflected_energy"])}


def _probe(directory: Path) -> dict[str, object]:
    env = child_process_env(threads=1).copy()
    env.update({name: os.environ[name] for name in _CPU_PIN_VARS if name in os.environ})
    configure_jax(platforms="cpu", enable_x64=True, env=env)
    env["TMPDIR"], env["PYTHONDONTWRITEBYTECODE"] = str(directory), "1"
    # 保留 runtime 給的來源路徑，只追加考卷根，不蓋掉被測程式來源。
    env["PYTHONPATH"] = os.pathsep.join((env["PYTHONPATH"], str(Path(__file__).resolve().parents[2])))
    output = directory / "result.json"
    result = subprocess.run([sys.executable, str(Path(__file__).with_name("_amplitude_jax_probe.py")),
                             str(aosr.__file__), str(output)], cwd=directory, env=env, capture_output=True,
                            text=True, timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr
    payload = cast(dict[str, object], json.loads(output.read_text(encoding="utf-8")))
    assert payload["error"] == "", payload["error"]
    return cast(dict[str, object], payload["data"])


def _suite_sections(data: dict[str, object]) -> dict[str, list[str]]:
    """子行程那一份資料逐段判；每段回失敗訊息（空＝過）。"""
    sections: dict[str, list[str]] = {name: [] for name in ("x64", "contract", "dtype", "transforms", "gradient")}
    if data.get("x64") is not True:
        sections["x64"].append(f"子行程沒開雙精度：{data.get('x64')!r}")
    for case in _CASES:
        record = cast(dict[str, object], data[case])
        frequencies = tuple(float(f) for f in _real_array(record["frequencies"]))
        delay = [float(t) for t in _real_array(record["delay"])]
        order = cast(list[int], record["order"])
        violations, _worst = contract_violations(_decoded(record["python"]), _decoded(record["jax64"]), order,
                                                 delay, frequencies, _BASE)
        sections["contract"] += [f"{case} {v}" for v in violations[:5]]
        for tag, want in (("jax64", ["complex128"] * 3 + ["float64"] * 2), ("jax32", ["complex64"] * 3 + ["float32"] * 2)):
            got = cast(dict[str, object], record[tag])["dtypes"]
            if got != want:
                sections["dtype"].append(f"{case} {tag} dtype {got!r} ≠ {want!r}")
        single = _decoded(record["jax32"])
        if not all(np.isfinite(v).all() for v in single.values()):
            sections["dtype"].append(f"{case} 單精度那條出現非有限數")
        for tag in ("transforms64", "transforms32"):
            flags = cast(dict[str, bool], record[tag])
            sections["transforms"] += [f"{case} {tag} {name}" for name, same in flags.items() if same is not True]
        if "gradient" in record:
            rows = np.asarray(record["gradient"], dtype=float)
            scale = float(np.abs(rows[:, :2]).max())
            error = np.maximum(np.abs(rows[:, 0] - rows[:, 2]), np.abs(rows[:, 1] - rows[:, 3]))
            if not scale > 0.0 or float(error.max()) > _GRADIENT_REL * scale:
                sections["gradient"].append(f"{case} 梯度對中央差分最大差 {float(error.max())!r}，尺度 {scale!r}")
    return sections


def test_double_precision_suite(tmp_path: Path) -> None:
    """子行程一次量完；併成一題，免得平行工人各開一次子行程。

    抓：JAX 雙精度跟現有實作超出登記的界線（反射乘積、路徑壓力、總壓力、兩種能量；含構造牆逐角 Z 的接法）、
    dtype 沒跟輸入走、單精度出非有限數、jit／vmap 改了位元、梯度跟中央差分對不上。
    """
    sections = _suite_sections(_probe(tmp_path))
    failures = {name: messages for name, messages in sections.items() if messages}
    assert not failures, json.dumps(failures, ensure_ascii=False, indent=1)


# ── 同程式（JAX 預設單精度）──────────────────────────────────────────────────


def _flat_case(tmp_path: Path) -> tuple[list[RoomPath], aj.PathTable, np.ndarray, float]:
    path = tmp_path / "room-flat.json"
    path.write_text(json.dumps(_input_case("flat"), sort_keys=True), encoding="utf-8")
    inp = load_room_input(path)
    assert inp.materials is not None
    paths = image_source_paths(inp.room, inp.source, inp.receiver, inp.sound_speed, inp.max_order, inp.materials)
    table = aj.build_path_table(paths, inp.receiver.as_tuple(), inp.materials.frequencies_hz)
    z_wall = np.array([inp.materials.walls[w] for w in amp.CANONICAL_WALLS], dtype=np.complex64)
    return paths, table, z_wall, inp.materials.rho_c


def test_single_precision_finite_dtype_and_direct_exact(tmp_path: Path) -> None:
    """抓：單精度輸入被升成雙精度或混進別的 dtype、出非有限數、直達反射乘積不是恰好 1、反彈次數跟階數對不上。"""
    paths, table, z_wall, rho_c = _flat_case(tmp_path)
    result = aj.amplitudes(table, aj.bounce_impedances(table, jnp.asarray(z_wall)), rho_c)
    assert [str(v.dtype) for v in result] == ["complex64"] * 3 + ["float32"] * 2
    assert all(bool(jnp.isfinite(v).all()) for v in result)
    direct = table.direct_index()
    assert np.asarray(result.reflection_product)[direct].tolist() == [1 + 0j] * len(table.frequencies_hz)
    assert table.bounce_mask.sum(axis=1).tolist() == [p.order for p in paths]


def test_single_precision_jit_and_vmap_bitwise_equal_eager(tmp_path: Path) -> None:
    """抓：jit 或 vmap 改了運算順序而改動最後一位（單精度、同程式那一條）。"""
    _paths, table, z_wall, rho_c = _flat_case(tmp_path)
    batch = jnp.stack([jnp.asarray(z_wall) * s for s in (1.0, 0.7 + 0.1j, 1.3 - 0.4j)])

    def run(z: jax.Array) -> aj.JaxAmplitudes:
        return aj.amplitudes(table, aj.bounce_impedances(table, z), rho_c)

    eager = [run(batch[i]) for i in range(3)]
    for transformed in (jax.vmap(run)(batch), jax.jit(jax.vmap(run))(batch)):
        for i in range(3):
            assert all(np.asarray(a[i]).tobytes() == np.asarray(b).tobytes() for a, b in zip(transformed, eager[i], strict=True))
    jitted = jax.jit(run)(batch[0])
    assert all(np.asarray(a).tobytes() == np.asarray(b).tobytes() for a, b in zip(jitted, eager[0], strict=True))


def test_path_table_rejects_bad_inputs(tmp_path: Path) -> None:
    """抓：分格材料、零距離、空或非正頻率軸、空路徑、阻抗形狀不對被悄悄收下。"""
    paths, table, z_wall, rho_c = _flat_case(tmp_path)
    receiver = (0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="分格"):
        aj.build_path_table([replace(paths[1], has_patch_materials=True)], receiver, (125.0,))
    with pytest.raises(ValueError, match="dist_m 為 0"):
        aj.build_path_table([replace(paths[1], dist_m=0.0)], receiver, (125.0,))
    for axis in ((), (0.0,), (float("nan"),)):
        with pytest.raises(ValueError, match="頻率軸"):
            aj.build_path_table(paths, receiver, axis)
    with pytest.raises(ValueError, match="沒有路徑"):
        aj.build_path_table([], receiver, (125.0,))
    with pytest.raises(ValueError, match="每面牆阻抗的形狀"):
        aj.bounce_impedances(table, jnp.asarray(z_wall[:5]))
    with pytest.raises(ValueError, match="逐反彈阻抗的形狀"):
        aj.reflection_products(table, jnp.ones((2, 3, 4), dtype=jnp.complex64), rho_c)
    without_direct = aj.build_path_table([p for p in paths if p.order > 0], receiver, (125.0,))
    with pytest.raises(ValueError, match="直達路徑不是恰好一條"):
        without_direct.direct_index()


def _gradient_energy(table: aj.PathTable, rho_c: float) -> Callable[[jax.Array], jax.Array]:
    def energy(z: jax.Array) -> jax.Array:
        result = aj.amplitudes(table, z, rho_c)
        return jnp.sum(jnp.abs(result.total_pressure) ** 2) + jnp.sum(result.reflected_energy)
    return energy


def test_unused_bounce_cells_never_poison_gradient(tmp_path: Path) -> None:
    """抓：沒用到的反彈格在遮罩之前就做除法——那一格填 −ρc（cos 填 1，分母恰為 0）時前向照樣有限，梯度卻整個變 NaN。

    沒用到的格填 −ρc、任意值與正常值三種，eager 與 jit 的梯度都要全部有限、沒用到的格梯度恰為 0、
    有用的格梯度與前向結果跟填正常值時逐位相同。
    """
    _paths, table, z_wall, rho_c = _flat_case(tmp_path)
    base = np.asarray(aj.bounce_impedances(table, jnp.asarray(z_wall)))
    unused = ~table.bounce_mask
    assert unused.any(), "路徑表沒有沒用到的格，這題測不到東西"
    energy = _gradient_energy(table, rho_c)
    runs = {}
    for label, fill in (("normal", complex(4.0 * rho_c, 0.0)), ("minus_rho_c", complex(-rho_c, 0.0)),
                        ("arbitrary", complex(-123.0, 45.0))):
        z = base.copy()
        z[unused] = fill
        eager = np.asarray(jax.grad(energy)(jnp.asarray(z)))
        jitted = np.asarray(jax.jit(jax.grad(energy))(jnp.asarray(z)))
        assert np.isfinite(eager).all() and np.isfinite(jitted).all(), label
        assert (eager[unused] == 0).all() and (jitted[unused] == 0).all(), label
        runs[label] = (eager, jitted, aj.amplitudes(table, jnp.asarray(z), rho_c))
    reference_eager, reference_jit, reference_out = runs["normal"]
    used = table.bounce_mask
    for label, (eager, jitted, out) in runs.items():
        assert eager[used].tobytes() == reference_eager[used].tobytes(), label
        assert jitted[used].tobytes() == reference_jit[used].tobytes(), label
        assert all(np.asarray(a).tobytes() == np.asarray(b).tobytes() for a, b in zip(out, reference_out, strict=True)), label


def test_contract_judge_zero_reference_policy(tmp_path: Path) -> None:
    """抓：參考反射乘積恰為 0（垂直入射打到 Z=ρc 的牆）時裁判除以零炸掉，或把零當成不用比。

    聲源與接收點在同一條垂直線上，地板與天花板一階路徑垂直入射；六面牆 Z=ρc，那兩條的反射乘積恰為 0。
    自己比自己要過；那一格的反射乘積或壓力推一點點要紅。
    """
    path = tmp_path / "room-flat.json"
    path.write_text(json.dumps(_input_case("flat"), sort_keys=True), encoding="utf-8")
    inp = load_room_input(path)
    assert inp.materials is not None
    rho_c, frequencies = inp.materials.rho_c, inp.materials.frequencies_hz
    walls = {wall: (complex(rho_c, 0.0),) * len(frequencies) for wall in amp.CANONICAL_WALLS}
    source = replace(inp.source, z=inp.room.Lz * 0.25)
    receiver = replace(source, z=inp.room.Lz * 0.75)
    paths = image_source_paths(inp.room, source, receiver, inp.sound_speed, 1,
                               amp.Materials(rho_c, frequencies, walls))
    reference = _python_reference(paths)
    zeros = np.argwhere(reference["refl"] == 0)
    assert zeros.size, "沒有反射乘積恰為 0 的格，這題測不到零值政策"
    i, k = (int(v) for v in zeros[0])
    order, delay = [p.order for p in paths], [p.delay_s for p in paths]
    clean, _ = contract_violations(reference, reference, order, delay, frequencies, _BASE)
    assert clean == []
    nudged_refl = {**reference, "refl": reference["refl"].copy()}
    nudged_refl["refl"][i, k] = complex(5e-324, 0.0)
    violations, _ = contract_violations(reference, nudged_refl, order, delay, frequencies, _BASE)
    assert any(v.startswith(f"refl[{i},{k}]") for v in violations), violations
    nudged_pressure = {**reference, "pressure": reference["pressure"].copy()}
    nudged_pressure["pressure"][i, k] = complex(5e-324, 0.0)
    violations, _ = contract_violations(reference, nudged_pressure, order, delay, frequencies, _BASE)
    assert any(v.startswith(f"pressure[{i},{k}]") for v in violations), violations


# ── 變異考卷 ──────────────────────────────────────────────────────────────────


def _python_reference(paths: list[RoomPath]) -> dict[str, np.ndarray]:
    totals = tot.totals_from_paths(paths)
    return {"refl": np.array([p.reflection_product for p in paths]), "pressure": np.array([p.path_pressure for p in paths]),
            "total": np.array(totals.pressure), "direct_energy": np.array(totals.direct_energy),
            "reflected_energy": np.array(totals.reflected_energy)}


def test_amplitude_jax_mutant_beyond_tolerance_is_red(tmp_path: Path) -> None:
    """裁判兩側各推 δ：反射乘積那一格推到界線內 (1−δ) 判綠、界線外 (1+δ) 判紅；壓力那一格同樣兩側各一次。

    界線只有 2^-4x 那麼小，直接加在 0.5 上面會被捨入吃掉，量不到 1±δ。所以受害那一格的參考反射乘積換成
    界線的 8 倍上下（2 的次方），加上去逐位精確；它的 1/|反射乘積| 也讓壓力界線大到捨入可以忽略。
    """
    paths, _table, _z, _rho_c = _flat_case(tmp_path)
    reference = _python_reference(paths)
    order, delay = [p.order for p in paths], [p.delay_s for p in paths]
    frequencies = tuple(float(f) for f in reference_frequencies())
    victim = next(i for i, p in enumerate(paths) if p.order >= 1)
    reference["refl"][victim, 0] = complex(math.ldexp(1.0, math.frexp(_BASE)[1] + 3), 0.0)
    clean, _ = contract_violations(reference, reference, order, delay, frequencies, _BASE)
    assert clean == []
    for factor, red in ((1.0 - MUTANT_MARGIN, False), (1.0 + MUTANT_MARGIN, True)):
        refl = {**reference, "refl": reference["refl"].copy()}
        refl["refl"][victim, 0] += factor * amp.reflection_tolerance(frequencies[0], delay[victim], 1.0, _BASE)
        violations, _ = contract_violations(reference, refl, order, delay, frequencies, _BASE)
        assert bool(violations) is red, (factor, violations)
        pressure = {**reference, "pressure": reference["pressure"].copy()}
        p_ref = reference["pressure"][victim, 0]
        bound = amp.pressure_tolerance(frequencies[0], delay[victim], abs(reference["refl"][victim, 0]), _BASE)
        pressure["pressure"][victim, 0] = p_ref + factor * bound * abs(p_ref)
        violations, _ = contract_violations(reference, pressure, order, delay, frequencies, _BASE)
        assert bool(violations) is red, (factor, violations)


def reference_frequencies() -> list[float]:
    """flat 那組答案的頻率軸，從現有考卷的輸入組裝函式讀。"""
    materials = _input_case("flat")["materials"]
    assert isinstance(materials, dict)
    return [float(f) for f in materials["frequencies_hz"]]
