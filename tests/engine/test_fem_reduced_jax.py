"""替身求值可微路：同一份小房間替身對 numpy、遮罩安全、構造參數到聲壓的梯度。

子行程只求值公開入口；參考聲壓、中央差分與判決都在父行程。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from collections.abc import Callable
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.typing import NDArray

from aosr.geometry.shoebox import Wall
from aosr.physics import fem_reduced_jax as reduced
from aosr.physics.fem_reduced import ReducedModel, evaluate_reduced, system_coefficients
from aosr.physics.fem_reduced_store import save_reduced
from aosr.runtime import child_process_env, configure_jax
from tests.engine._precision_contracts import MUTANT_MARGIN, contract_value
from tests.engine.test_fem_reduced import AXIS, DENSITY, HOLDOUT, SOUND_SPEED, TRAINING
from tests.engine.test_fem_reduced import model as model, operators as operators
from tests.engine.test_transfer_matrix_reference import GRADIENT_ABS, GRADIENT_REL


# 差分相對步長沿用傳遞矩陣考卷；不是產品精度契約。
DIFFERENCE_STEP = 1e-5
# 樣板兩個探點各自的差分相對步長（#956 步長掃）：取本機步長掃觀察到「截斷誤差還跟 h² 一起縮、捨入還沒主導」的最小一格（掃描結果，不是必然結論）。
# 玻璃棉構造的空腔厚度（layer2，空氣層）無量綱導數尺度約 5.5e3，3e-5 時逐頻最差約 1.5e-9；
# 穿孔板厚度（layer1，0.8 mm）尺度只有約 2.5，1e-2 以下就被能量相減的捨入主導，要 3e-2 才回到 h² 區。
# 本次掃描沒有一格能讓兩探點同時落在 h² 區；共用 1e-4 時穿孔板那格已在捨入區，雲端 Xeon 就是這樣紅的。不是產品精度契約。
TEMPLATE_DIFFERENCE_STEPS = {"layer2.thickness_m": 3e-5, "layer1.thickness_m": 3e-2}
# 反向模式（總能量的 grad）對前向模式（逐頻能量的 jvp）加總：同一函式兩種自動微分模式，差的是不同微分模式與運算次序的浮點捨入、
# 不含有限差分截斷。分母用逐頻導數絕對值總和，正負相消時不被放大；本機實量約 1.5e-15，留近三個數量級。
# 不是產品精度契約。
REVERSE_FORWARD_REL = 1e-12
CASES = ("training", "holdout", "one_rigid", "several_rigid")
Record = dict[str, object]


def _decode(node: object) -> np.ndarray:
    row = cast(Record, node)
    real = np.asarray([float.fromhex(v) for v in cast(list[str], row["real"])])
    imag = np.asarray([float.fromhex(v) for v in cast(list[str], row["imag"])])
    return (real + 1j * imag).reshape(cast(list[int], row["shape"]))


def _case(name: str) -> tuple[NDArray[np.complex128], NDArray[np.bool_]]:
    walls = TRAINING[0] if name == "training" else HOLDOUT[0]
    z = np.asarray([walls[w] for w in Wall.all()], dtype=np.complex128)
    mask = np.zeros(len(Wall.all()), dtype=np.bool_)
    if name == "one_rigid":
        mask[2] = True
    if name == "several_rigid":
        mask[[0, 3, 5]] = True
    z[mask] = 0.0
    return z, mask


def _numpy(model: ReducedModel, z: np.ndarray, mask: np.ndarray) -> np.ndarray:
    rows = []
    for index, frequency in enumerate(AXIS):
        walls = {wall: None if mask[w] else complex(z[w, index]) for w, wall in enumerate(Wall.all())}
        k2, coefficients = system_coefficients(frequency, walls, density_kg_m3=DENSITY, sound_speed_m_s=SOUND_SPEED)
        rows.append(evaluate_reduced(model.matrices, k2, coefficients))
    return np.asarray(rows)


def _within(actual: np.ndarray, expected: np.ndarray) -> bool:
    """同一替身、同一阻抗，逐格複數聲壓相對差；零參考只收恰零。"""
    return bool(actual.shape == expected.shape and np.all(np.isfinite(actual))
                and np.all(np.abs(actual - expected) <= contract_value("fem_reduced_jax_vs_numpy") * np.abs(expected)))


@pytest.fixture(scope="module")
def runs(model: ReducedModel, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Record]:
    directory = tmp_path_factory.mktemp("fem-reduced-jax")
    save_reduced(model, directory / "model.npz")
    problem = {name: {"real": _case(name)[0].real.tolist(), "imag": _case(name)[0].imag.tolist(),
                      "mask": _case(name)[1].tolist()} for name in CASES}
    (directory / "problem.json").write_text(json.dumps(problem), encoding="utf-8")
    env = child_process_env(threads=1)
    configure_jax(platforms="cpu", enable_x64=True, env=env)
    env["TMPDIR"], env["PYTHONDONTWRITEBYTECODE"] = str(directory), "1"
    env["PYTHONPATH"] = os.pathsep.join((env["PYTHONPATH"], str(Path(__file__).resolve().parents[2])))
    process = subprocess.run([sys.executable, str(Path(__file__).with_name("_fem_reduced_jax_probe.py")), str(directory), json.dumps(TEMPLATE_DIFFERENCE_STEPS)],
                             cwd=directory, env=env, capture_output=True, text=True, timeout=100)
    assert process.returncode == 0, process.stdout + process.stderr
    return cast(dict[str, Record], json.loads((directory / "output.json").read_text(encoding="utf-8")))


@pytest.mark.parametrize("name", CASES)
def test_double_matches_numpy(model: ReducedModel, runs: dict[str, Record], name: str) -> None:
    """抓組裝符號、六面順序、喇叭座位軸、剛性邊界與精度變動；訓練及留出阻抗都比逐頻聲壓。"""
    z, mask = _case(name)
    expected = _numpy(model, z, mask)
    for key in ("eager", "jit", "vmap"):
        assert _within(_decode(runs[name][key]), expected), (name, key)
    assert _within(_decode(runs[name]["vmap_alternative"]), _numpy(model, z * (1.2 - 0.1j), mask))
    coefficients = _decode(runs[name]["coefficients"])
    expected_coefficients = np.asarray([system_coefficients(
        f, {w: None if mask[i] else complex(z[i, k]) for i, w in enumerate(Wall.all())},
        density_kg_m3=DENSITY, sound_speed_m_s=SOUND_SPEED)[1] for k, f in enumerate(AXIS)])
    # 同一把聲壓契約也守非零係數；零係數必須恰零。
    assert _within(coefficients, expected_coefficients)
    assert np.all(coefficients[:, mask] == 0)


@pytest.mark.parametrize("name", CASES)
def test_transforms_and_dtype(runs: dict[str, Record], name: str) -> None:
    """抓編譯結果重跑掉位、vmap 軸接錯、輸入單精度被提升、輸出或梯度非有限。"""
    row = runs[name]
    eager, compiled = _decode(row["eager"]), _decode(row["jit"])
    assert compiled.tobytes() == _decode(row["jit_again"]).tobytes()
    # 編譯及批次的求解運算順序會變，依審查裁定用尺；同一份編譯重跑逐位比較。
    assert _within(compiled, eager)
    assert _within(_decode(row["vmap"]), eager)
    assert _decode(row["vmap_jit"]).tobytes() == _decode(row["vmap_jit_again"]).tobytes()
    assert _within(_decode(row["vmap_jit"]), eager)
    for key in ("single", "single_coefficients", "single_gradient"):
        assert cast(Record, row[key])["dtype"] == "complex64", key
        assert np.isfinite(_decode(row[key])).all(), key
    assert cast(Record, row["jit"])["dtype"] == "complex128"


@pytest.mark.parametrize("name", CASES)
def test_impedance_gradient_matches_numpy_difference(model: ReducedModel, runs: dict[str, Record], name: str) -> None:
    """抓 Z 實部或虛部斷線；一次推一面全部頻率，各頻率獨立的能量差分對全陣列梯度。"""
    z, mask = _case(name)
    automatic = _decode(runs[name]["gradient"])
    assert np.isfinite(automatic).all() and np.all(automatic[mask] == 0)
    for component, part in ((1.0, np.real), (1.0j, np.imag)):
        central = np.zeros(z.shape)
        scale = np.maximum(np.abs(part(z)), DENSITY * SOUND_SPEED)
        for wall in np.flatnonzero(~mask):
            h = DIFFERENCE_STEP * scale[wall]
            upper, lower = z.copy(), z.copy()
            upper[wall] += component * h
            lower[wall] -= component * h
            plus = np.sum(np.abs(_numpy(model, upper, mask)) ** 2, axis=(1, 2))
            minus = np.sum(np.abs(_numpy(model, lower, mask)) ** 2, axis=(1, 2))
            central[wall] = (plus - minus) / (2.0 * h)
        # 探針分開微分兩個實陣列，imag 存的是對虛部的導數，沒有複數梯度的共軛約定混入。
        # 逐格相對差在導數過零處受能量相減捨入支配。照振幅先例，用無量綱導數的最大尺度；
        # 每面、每分量分開取尺度，避免弱牆被別面的強導數遮住；尺仍從傳遞矩陣考卷讀。
        expected = central * scale
        wall_scale = np.max(np.abs(expected), axis=1, keepdims=True)
        assert np.all(np.abs(part(automatic) * scale - expected) <= GRADIENT_REL * wall_scale + GRADIENT_ABS)
        assert np.all(np.max(np.abs(part(automatic)[~mask] * scale[~mask]), axis=1) > GRADIENT_ABS)


def test_template_chain_matches_numpy_and_has_two_live_gradients(model: ReducedModel, runs: dict[str, Record]) -> None:
    """空腔厚度與穿孔板厚度兩個探點：前向逐頻導數對獨立重建構造的逐頻中央差分，反向總和接前向，兩面兩邊都非零。"""
    row = runs["templates"]
    mask = np.asarray(row["mask"], dtype=bool)
    assert row["mask_from_templates_identical"] is True and mask.tolist() == [False, True, True, True, False, True]
    z = _decode(row["impedances"])
    assert _within(_decode(row["pressure"]), _numpy(model, z, mask))
    assert _within(_decode(row["jit"]), _decode(row["pressure"]))
    assert _decode(row["jit"]).tobytes() == _decode(row["jit_again"]).tobytes()
    probes = cast(list[Record], row["gradients"])
    assert sorted(cast(str, probe["name"]) for probe in probes) == sorted(TEMPLATE_DIFFERENCE_STEPS)
    for probe in probes:
        name = cast(str, probe["name"])
        step = TEMPLATE_DIFFERENCE_STEPS[name]
        assert probe["step"] == step, name
        baseline = float(cast(float, probe["baseline"]))
        # 總能量一個純量相減會只剩有效位數的尾巴（#956：雲端 8.75e-8 誤擋），改逐頻：每個頻點的能量各自差分。
        upper = np.sum(np.abs(_numpy(model, _decode(probe["upper"]), mask)) ** 2, axis=(1, 2))
        lower = np.sum(np.abs(_numpy(model, _decode(probe["lower"]), mask)) ** 2, axis=(1, 2))
        expected = (upper - lower) / (2.0 * step)
        forward = _decode(probe["automatic_per_frequency"])
        assert forward.shape == (len(AXIS),) and np.isfinite(forward).all() and np.all(forward.imag == 0), name
        actual = forward.real * baseline
        # 乘基準值成無量綱；照上面那題，探點內取頻點上最大絕對值當尺度，導數過零的頻點不被捨入支配；尺仍從傳遞矩陣考卷讀。
        scale = np.max(np.abs(expected))
        assert np.all(np.abs(actual - expected) <= GRADIENT_REL * scale + GRADIENT_ABS), name
        assert np.max(np.abs(actual)) > GRADIENT_ABS and scale > GRADIENT_ABS, name
        # 反向模式經前向模式接到獨立 numpy 參考。
        reverse = float(cast(float, probe["automatic"]))
        assert abs(reverse - np.sum(forward.real)) <= REVERSE_FORWARD_REL * np.sum(np.abs(forward.real)), name
    assert cast(Record, row["single"])["dtype"] == "complex64"
    assert np.isfinite(_decode(row["single"])).all()
    assert row["single_gradient_dtype"] == "float32" and row["single_gradient_finite"] is True


def test_all_training_and_holdout_materials(model: ReducedModel, runs: dict[str, Record]) -> None:
    """整批十二組訓練與兩組留出逐頻比對，避免只挑第一組而漏掉其他邊界材料。"""
    pressures = cast(list[Record], runs["materials"]["pressures"])
    for walls, row in zip([*TRAINING, *HOLDOUT], pressures, strict=True):
        z = np.asarray([walls[w] for w in Wall.all()], dtype=np.complex128)
        assert _within(_decode(row), _numpy(model, z, np.zeros(6, dtype=bool)))


def test_rigid_cells_never_poison_gradient(model: ReducedModel) -> None:
    """剛性格填零、NaN、無窮或任意值，除法前先替換；未遮的結果與梯度逐位不受影響。"""
    constants = reduced.reduced_constants(model)
    z, mask = _case("several_rigid")
    mask_array = jnp.asarray(mask)
    def energy(values: jax.Array) -> jax.Array:
        return jnp.sum(jnp.abs(reduced.reduced_pressures(constants, values, rigid_mask=mask_array)) ** 2)
    compiled = jax.jit(jax.grad(energy))
    outputs = []
    for fill in (0j, complex(float("nan"), 0), complex(float("inf"), 0), -123 + 45j):
        values = z.astype(np.complex64)
        values[mask] = fill
        for run in (jax.grad(energy), compiled):
            gradient = np.asarray(run(jnp.asarray(values)))
            assert np.isfinite(gradient).all() and np.all(gradient[mask] == 0)
            outputs.append(gradient)
    for eager, compiled_gradient in zip(outputs[2::2], outputs[3::2], strict=True):
        assert eager.tobytes() == outputs[0].tobytes()
        assert compiled_gradient.tobytes() == outputs[1].tobytes()


def test_single_precision_in_parent_process(model: ReducedModel) -> None:
    """同程式交 complex64：聲壓、係數、梯度維持單精度且有限，jit 重跑不掉位。"""
    constants = reduced.reduced_constants(model)
    values, mask = _case("one_rigid")
    z, rigid = jnp.asarray(values, dtype=jnp.complex64), jnp.asarray(mask)
    def run(values: jax.Array) -> jax.Array:
        return reduced.reduced_pressures(constants, values, rigid_mask=rigid)
    compiled = jax.jit(run)
    result = compiled(z)
    gradient = jax.grad(lambda values: jnp.sum(jnp.abs(run(values)) ** 2))(z)
    for output in (run(z), result, jax.vmap(run)(jnp.stack((z, 1.2 * z))), gradient,
                   reduced.wall_coefficients(constants, z, rigid_mask=rigid)):
        assert output.dtype == jnp.complex64 and np.isfinite(np.asarray(output)).all()
    assert np.asarray(result).tobytes() == np.asarray(compiled(z)).tobytes()


def test_rejects_shapes_and_frequency_mismatch(model: ReducedModel) -> None:
    """抓非六面、缺頻點、批次誤塞、遮罩形狀及常數矩陣形狀錯誤。"""
    constants = reduced.reduced_constants(model)
    none_rigid = jnp.zeros(6, dtype=bool)
    for shape in ((5, len(AXIS)), (6,), (6, len(AXIS) - 1), (2, 6, len(AXIS))):
        with pytest.raises(ValueError, match="形狀"):
            reduced.reduced_pressures(constants, jnp.ones(shape, dtype=jnp.complex64), rigid_mask=none_rigid)
    with pytest.raises(ValueError, match="遮罩"):
        reduced.wall_coefficients(constants, jnp.ones((6, len(AXIS)), dtype=jnp.complex64), rigid_mask=jnp.zeros(5, dtype=bool))
    with pytest.raises(TypeError, match="複數"):
        reduced.reduced_pressures(constants, jnp.ones((6, len(AXIS)), dtype=jnp.int32), rigid_mask=none_rigid)


def test_mask_cannot_be_omitted_and_template_bundle_is_finite(model: ReducedModel) -> None:
    """審查反例：樣板入口給零當剛性佔位，省略遮罩曾靜默算出全 NaN；現在遮罩必填、樣板一包交出。"""
    constants = reduced.reduced_constants(model)
    z, mask = _case("several_rigid")
    values = jnp.asarray(z, dtype=jnp.complex64)
    # 故意漏給遮罩：型別檢查器會擋這種呼叫，所以用不帶簽名的呼叫型別繞過它、讓執行期自己擋。
    for entry in (reduced.reduced_pressures, reduced.wall_coefficients):
        untyped = cast(Callable[..., jax.Array], entry)
        with pytest.raises(TypeError, match="rigid_mask"):
            untyped(constants, values)
    bundle = reduced.TemplateImpedances(values, jnp.asarray(mask))
    pressure = reduced.reduced_pressures(constants, bundle.values, rigid_mask=bundle.rigid_mask)
    assert np.isfinite(np.asarray(pressure)).all()
    # 同一組零佔位、遮罩寫成全 False（就是「忘了遮」的意思）：係數必須非有限，證明遮罩不是裝飾。
    coefficients = reduced.wall_coefficients(constants, values, rigid_mask=jnp.zeros(6, dtype=bool))
    assert not np.isfinite(np.asarray(coefficients)[:, mask]).all()
    with pytest.raises(ValueError, match="矩陣"):
        reduced.reduced_constants(replace(model, matrices=replace(model.matrices, boundary=model.matrices.boundary[:5])))
    with pytest.raises(ValueError, match="頻率軸"):
        reduced.reduced_constants(replace(model, frequencies_hz=()))


def test_template_rejections(runs: dict[str, Record]) -> None:
    """雙精度行程建樣板後，牆數、參數組數、頻率維度、參數鍵及剛性面的多餘參數必須拒收。"""
    errors = cast(dict[str, str], runs["templates"]["rejections"])
    assert all(errors.values()), errors
    assert "六面" in errors["walls"] and "六面" in errors["parameters"]
    assert "頻率" in errors["frequencies"]
    assert "參數鍵" in errors["keys"] and "剛性" in errors["rigid_parameters"]


def test_mutant_beyond_tolerance_is_red() -> None:
    """正負方向都在尺兩側各推 δ；不放寬界線，NaN 與零參考的非零差也不准漏收。"""
    truth = np.ones((2, 2, 3), dtype=np.complex128)
    ruler = contract_value("fem_reduced_jax_vs_numpy")
    for direction in (1.0, -1.0, 1j, -1j):
        for factor, accepted in ((1.0 - MUTANT_MARGIN, True), (1.0 + MUTANT_MARGIN, False)):
            candidate = truth.copy()
            candidate[1, 0, 2] += direction * factor * ruler
            assert _within(candidate, truth) is accepted
    assert not _within(np.full_like(truth, np.nan), truth)
    assert _within(np.zeros_like(truth), np.zeros_like(truth))
    assert not _within(np.ones_like(truth), np.zeros_like(truth))
