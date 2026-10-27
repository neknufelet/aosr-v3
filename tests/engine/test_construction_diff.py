"""可微構造入口（#561）：基準值逐位等於既有入口、梯度對中央差分、jit／vmap、單精度路。

算數全在隔離的雙精度子行程（同 test_construction 的開法），父考卷只讀它寫在 tmp 的結果。
梯度與 jit 的尺沿用傳遞矩陣移植考卷那幾把（從那支考卷 import，不另寫數字）：
梯度比無量綱 p·dL/dp，jit 對直接呼叫比阻抗的相對與絕對兩項；vmap 對逐組呼叫要逐位相同。
單精度只判型別、有限、不報錯；跟雙精度差多少沒有登記尺，量到的數字寫在施工報告。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from pydantic import TypeAdapter

from aosr.config.construction_presets import load_construction_presets
from aosr.config.paths import config_path
from aosr.materials.construction_models import AirLayer, Assembly, PorousLayer, SurfaceMaterial
from aosr.runtime import child_process_env, configure_jax
from tests.engine.test_transfer_matrix_reference import GRADIENT_ABS, GRADIENT_REL, IMPEDANCE_ABS, REFERENCE_REL
from tests.engine.test_wall_surfaces import _CPU_PIN_VARS


PROBE = Path(__file__).with_name("_construction_diff_probe.py")
Bits = dict[str, object]


def _run(mode: str, directory: Path) -> dict[str, object]:
    env = child_process_env(threads=1).copy()
    env.update({name: os.environ[name] for name in _CPU_PIN_VARS if name in os.environ})
    configure_jax(platforms="cpu", enable_x64=True, env=env)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["TMPDIR"] = str(directory)
    output = directory / f"{mode}.json"
    process = subprocess.run([sys.executable, str(PROBE), mode, str(output)], cwd=directory, env=env,
                             capture_output=True, text=True, timeout=600)
    assert process.returncode == 0, process.stdout + process.stderr
    result = cast(dict[str, object], json.loads(output.read_text()))
    assert result["dtype"] == "float64"
    return cast(dict[str, object], result["rows"])


def _presets() -> dict[str, Assembly]:
    adapter: TypeAdapter[SurfaceMaterial] = TypeAdapter(SurfaceMaterial)
    materials = {preset.id: adapter.validate_python(preset.material)
                 for preset in load_construction_presets(config_path("construction_presets.toml"))}
    assert all(isinstance(material, Assembly) for material in materials.values())
    return cast(dict[str, Assembly], materials)


def _decode(bits: object) -> np.ndarray:
    row = cast(dict[str, list[str]], bits)
    real = np.asarray([float.fromhex(value) for value in row["real"]])
    imag = np.asarray([float.fromhex(value) for value in row["imag"]])
    return real + 1j * imag


@pytest.fixture(scope="module")
def identity(tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict[str, object]]:
    return cast(dict[str, dict[str, object]], _run("identity", tmp_path_factory.mktemp("construction-diff")))


def test_baseline_equals_existing_entries_bit_for_bit(identity: dict[str, dict[str, object]]) -> None:
    """每一筆預設在樣板基準值上，垂直與 Paris 角度格都逐位等於 surface_impedance／surface_impedance_at_angles。"""
    assert set(identity) == set(_presets())
    for name, row in identity.items():
        for key in ("normal", "angles"):
            ours, existing = cast(dict[str, object], row[key]), cast(dict[str, object], row[f"{key}_existing"])
            assert ours["dtype"] == existing["dtype"] == "complex128", (name, key)
            assert ours["shape"] == existing["shape"], (name, key)
            np.testing.assert_array_equal(_decode(ours), _decode(existing), err_msg=f"{name}:{key}")


def test_template_exposes_porous_and_cavity_variables(identity: dict[str, dict[str, object]]) -> None:
    """多孔層的厚度與流阻、空腔厚度一定是變數；層號從房間側的一起算。"""
    for name, material in _presets().items():
        names = set(cast(list[str], identity[name]["variables"]))
        for index, layer in enumerate(material.layers, start=1):
            if isinstance(layer, PorousLayer):
                assert {f"layer{index}.thickness_m", f"layer{index}.flow_resistivity_pa_s_m2"} <= names, name
            if isinstance(layer, AirLayer):
                assert f"layer{index}.thickness_m" in names, name


def test_gradients_match_central_difference(tmp_path: Path) -> None:
    """每筆預設的每個變數、垂直與 60 度、80 與 1000 Hz：jit(grad) 對中央差分，斷線或抓錯欄位必須紅。"""
    rows = cast(list[dict[str, object]], _run("gradients", tmp_path))
    assert {cast(str, row["preset"]) for row in rows} == set(_presets())
    for row in rows:
        automatic, central = float(cast(float, row["automatic_scaled"])), float(cast(float, row["central_scaled"]))
        assert np.isfinite(automatic) and np.isfinite(central), row
        np.testing.assert_allclose(automatic, central, rtol=GRADIENT_REL, atol=GRADIENT_ABS, err_msg=str(row))


def test_jit_and_vmap_preserve_direct_results(tmp_path: Path) -> None:
    """vmap 對逐組直接呼叫逐位相同；jit 有編譯器合併運算的捨入，用傳遞矩陣移植考卷 jit 那把尺。"""
    rows = cast(dict[str, dict[str, dict[str, Bits]]], _run("transforms", tmp_path))
    assert set(rows) == set(_presets())
    for name, row in rows.items():
        for key, values in row.items():
            assert values["vmap"]["dtype"] == values["pointwise"]["dtype"] == "complex128"
            np.testing.assert_array_equal(_decode(values["vmap"]), _decode(values["pointwise"]), err_msg=f"{name}:{key}:vmap")
            jit, direct = _decode(values["jit"]), _decode(values["direct"])
            for part in (np.real, np.imag):
                np.testing.assert_allclose(part(jit), part(direct), rtol=REFERENCE_REL, atol=IMPEDANCE_ABS, err_msg=f"{name}:{key}:jit")


def test_single_precision_inputs_stay_single(tmp_path: Path) -> None:
    """float32 參數、頻率、角度進去：阻抗是 complex64、有限、梯度是 float32 且有限；不判跟雙精度差多少。"""
    rows = cast(dict[str, dict[str, object]], _run("single", tmp_path))
    assert set(rows) == set(_presets())
    for name, row in rows.items():
        for key in ("normal", "angles"):
            value = cast(dict[str, object], row[key])
            assert value["dtype"] == "complex64" and value["finite"] is True, (name, key, value)
        assert row["gradient_dtypes"] == ["float32"] and row["gradient_finite"] is True, (name, row)


def test_template_and_evaluation_reject_wrong_inputs(tmp_path: Path) -> None:
    """建樣板照舊被既有檢查擋；固定阻抗不收；變數名、參數鍵、批次值錯了說清楚；只放一個變數時基準上逐位不變。"""
    rows = _run("rejections", tmp_path)
    assert "超出手動試算範圍" in cast(str, rows["existing_check"])
    assert "只收多層構造" in cast(str, rows["constant"])
    assert "layer9.thickness_m" in cast(str, rows["unknown_variable"])
    assert "參數鍵要剛好是" in cast(str, rows["missing_key"])
    assert "參數鍵要剛好是" in cast(str, rows["extra_key"])
    assert "零維陣列" in cast(str, rows["batched_value"])
    assert rows["subset_names"] == [rows["first"]]
    assert rows["subset_bitwise"] is True
