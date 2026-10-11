"""第三步：逐頻逐角度凍結答案、中央差分及未給角度的原路徑。"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from aosr.runtime import child_process_env, configure_jax


ROOT = Path(__file__).resolve().parents[2]
PROBLEM_FILE = ROOT / "blueprint" / "tmm_reference_problem_step3.json"
ANSWER_FILE = ROOT / "blueprint" / "tmm_reference_answers_step3.json"
PROBE = Path(__file__).with_name("_transfer_matrix_step3_probe.py")
CASES = tuple(json.loads(PROBLEM_FILE.read_text())["cases"])
# 同公式雙精度，延續前兩步的阻抗/反射混合尺，吸音只用絕對尺。
# 實量Z角度實/虛最大絕對差7.205e-12/1.092e-11；近零相對差靠絕對項判。
# 大值與不同平台複數運算用相對項，非物理近似精度；吸音最大差9.993e-16。
REFERENCE_REL = 5e-12
IMPEDANCE_ABS = 2e-10
COEFFICIENT_ABS = 2e-14
INTEGRAL_ABS = 2e-14  # 少量多項式積分的雙精度舍入空間；積分與角度格只用絕對尺。
# 同一條式子直接比元素：相位至多約4.4 rad，cos/sin 差幾個 ulp；元素量級由相對項吸收。
MATRIX_REL = 1e-13
MATRIX_ABS = 1e-15
DETERMINANT_SCALED_ABS = 32 * np.finfo(np.float64).eps
DETERMINANT_ABS = 2e-14
# 中央差分相對步長1e-5，O(h²)截斷與複數相减；角度導數以度為單位。
# 首輪縮放導數最大相對差1.726e-8，絕對差4.092e-8（大導數由相對項判）。
GRADIENT_REL = 3e-7
GRADIENT_ABS = 2e-8


def _decode(values: list[str]) -> np.ndarray:
    return np.asarray([float.fromhex(v) for v in values])


def _run(mode: str, directory: Path) -> dict[str, object]:
    env = child_process_env(threads=1).copy()
    configure_jax(platforms="cpu", enable_x64=True, env=env)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["TMPDIR"] = str(directory)
    env["PYTHONPATH"] = os.pathsep.join((str(ROOT), env["PYTHONPATH"]))
    output = directory / f"{mode}.json"
    process = subprocess.run(
        [sys.executable, str(PROBE), mode, str(PROBLEM_FILE), str(output)],
        cwd=directory,
        env=env,
        capture_output=True,
        text=True,
        timeout=360,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    return cast(dict[str, object], json.loads(output.read_text()))


def _assert_values(
    key: str, actual: list[str], expected: list[str], context: str
) -> None:
    absorption = "absorption" in key
    atol = (
        COEFFICIENT_ABS if absorption or key.startswith("reflection") else IMPEDANCE_ABS
    )
    integral = key.startswith(("integral", "angles"))
    if integral:
        atol = INTEGRAL_ABS
    np.testing.assert_allclose(
        _decode(actual),
        _decode(expected),
        rtol=0 if absorption or integral else REFERENCE_REL,
        atol=atol,
        err_msg=context,
    )


@pytest.fixture(scope="module")
def reference_run(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    return _run("reference", tmp_path_factory.mktemp("tmm-step3-reference"))


@pytest.mark.parametrize("case_name", CASES)
def test_step3_spectra_and_scipy_integrals_match_frozen_reference(
    reference_run: dict[str, object], case_name: str
) -> None:
    """kz/Zc,n、cosθ、負載、角度端點、Simpson尾段與Paris權重植錯都在逐格答案紅。"""
    expected = json.loads(ANSWER_FILE.read_text())["cases"][case_name]
    actual = cast(dict[str, dict[str, list[str]]], reference_run["cases"])[case_name]
    assert reference_run["dtype"] == "float64"
    assert set(actual) == set(expected)
    for key, want in expected.items():
        _assert_values(key, actual[key], want, f"{case_name}::{key}")


def test_step3_absorption_comparisons_use_only_absolute_tolerance() -> None:
    """所有種類的吸音都不得靠相對項放過接近1的漂移。"""
    for key in (
        "absorption_hex",
        "absorption_angle_hex",
        "field_absorption_hex",
        "paris_absorption_hex",
        "selected_absorption_hex",
    ):
        with pytest.raises(AssertionError):
            _assert_values(key, [(0.8 + 1e-12).hex()], [(0.8).hex()], key)


def test_integrals_and_angle_grids_use_only_absolute_tolerance() -> None:
    """Simpson 與角度格量級到數十，相對項會放過遠大於實量差的漂移（數值審查）。"""
    for key in ("integral_real_hex", "integral_imag_hex", "angles_deg_hex"):
        with pytest.raises(AssertionError):
            _assert_values(key, [(6.8 + 1e-12).hex()], [(6.8).hex()], key)


def test_step3_uses_frozen_air_and_every_problem_case(
    reference_run: dict[str, object],
) -> None:
    """不釘數量；刪案例或偷換空氣物性時紅。"""
    problem = json.loads(PROBLEM_FILE.read_text())
    assert reference_run["air"] == {
        k: v["value"] for k, v in problem["air_properties"].items()
    }
    assert set(cast(dict[str, object], reference_run["cases"])) == set(problem["cases"])
    assert CASES


def test_oblique_layers_keep_unit_determinant(tmp_path: Path) -> None:
    """厚層的AD−BC以條件數判；同模型1mm直接尺擋整體縮放。"""
    result = _run("determinants", tmp_path)
    for key in ("rows", "conditioned_rows"):
        rows = cast(list[dict[str, object]], result[key])
        assert rows
        for row in rows:
            det = _decode(cast(list[str], row["real_hex"])) + 1j * _decode(
                cast(list[str], row["imag_hex"])
            )
            scale = np.maximum(
                1, _decode(cast(list[str], row["cancellation_scale_hex"]))
            )
            np.testing.assert_allclose(
                (det - 1) / scale,
                0,
                rtol=0,
                atol=DETERMINANT_SCALED_ABS,
                err_msg=str(row["layer"]),
            )
            if key == "conditioned_rows":
                np.testing.assert_allclose(
                    det, 1, rtol=0, atol=DETERMINANT_ABS, err_msg=str(row["layer"])
                )


def test_angle_and_quadrature_sample_gradients_match_central_difference(
    tmp_path: Path,
) -> None:
    """每種層角度及Simpson/梯形/field/Paris取樣點導數斷線或NaN時紅。"""
    rows = cast(list[dict[str, object]], _run("gradients", tmp_path)["rows"])
    assert rows
    for row in rows:
        actual, want = (
            float(cast(float, row["automatic_scaled"])),
            float(cast(float, row["central_scaled"])),
        )
        assert np.isfinite(actual) and np.isfinite(want), row
        np.testing.assert_allclose(
            actual, want, rtol=GRADIENT_REL, atol=GRADIENT_ABS, err_msg=str(row)
        )


def test_step3_jit_and_vmap_keep_angles_dynamic(tmp_path: Path) -> None:
    """角度、角度維度或積分取樣點被轉成Python值時，編譯/向量化與逐點答案不符。"""
    result = _run("transforms", tmp_path)
    for shapes in ("direct_shapes", "mapped_shapes"):
        for shape in cast(list[list[int]], result[shapes]):
            assert shape == result["expected_shape"]
    rows = cast(dict[str, dict[str, dict[str, list[str]]]], result["rows"])
    assert rows
    for name, row in rows.items():
        for transform in ("jit", "vmap"):
            for key, want in row["pointwise"].items():
                _assert_values(
                    key, row[transform][key], want, f"{name}::{transform}::{key}"
                )


def test_invalid_angles_and_grazing_endpoint_are_handled(tmp_path: Path) -> None:
    """單角度90°、非法範圍及編譯後非法角度須拒收；合法近90°和未選支導數須有限。"""
    result = _run("validation", tmp_path)
    assert result["rejected"] == result["expected_rejected"]
    assert result["near_grazing_finite"]
    assert np.all(np.isfinite(np.asarray(result["near_grazing_gradient"])))
    assert np.all(np.isfinite(np.asarray(result["zero_spacing_gradient"])))


def test_omitted_angles_never_enter_oblique_algorithm(tmp_path: Path) -> None:
    """堵住角度算法入口跑完前兩步所有案例；同值異算法也抓得到。"""
    result = _run("normal_path", tmp_path)
    assert cast(dict[str, object], result["step1"])["dtype"] == "float64"
    assert cast(dict[str, object], result["step2"])["dtype"] == "float64"


def test_angular_shapes_backings_and_integration_boundaries(tmp_path: Path) -> None:
    """隨角度负載、錯誤角度軸與互斥選項；純空氣匹配在近掠角亦須成立。"""
    result = _run("boundaries", tmp_path)
    assert result["rejected"] == result["expected_rejected"]
    assert result["matrix_shape"] == result["expected_matrix_shape"]
    rows = cast(dict[str, dict[str, list[str]]], result)
    for key, want in rows["expected"].items():
        _assert_values(key, rows["actual"][key], want, key)
        _assert_values(key, rows["explicit"][key], want, key)


def test_trapezoidal_integral_matches_numpy_directly(tmp_path: Path) -> None:
    """Paris 用的是比值，梯形法少除 2 會被約掉；公開函式本身要直接對 NumPy 的 trapezoid（主對話植錯找到）。"""
    result = _run("trapezoid", tmp_path)
    assert result["dtype"] == "float64"
    rows = cast(list[dict[str, list[str]]], result["rows"])
    assert rows
    for row in rows:
        actual = np.asarray([float.fromhex(v) for v in row["actual"]])
        expected = np.asarray([float.fromhex(v) for v in row["expected"]])
        # 同一條梯形公式、雙精度，只差加總順序的捨入。
        np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-15, err_msg=str(row.get("name")))


def test_layer_matrices_have_the_right_elements_not_just_ratios(tmp_path: Path) -> None:
    """整層乘−1會被A/C、背後負載比值與行列式一起約掉（數值審查找到）；直接看元素。"""
    result = _run("matrix_values", tmp_path)
    assert result["dtype"] == "float64"
    for key in ("identity_rows", "sheet_rows", "air_rows"):
        rows = cast(list[dict[str, object]], result[key])
        assert rows, key
        for row in rows:
            actual, expected = (
                _decode(cast(list[str], row[f"{side}_real"]))
                + 1j * _decode(cast(list[str], row[f"{side}_imag"]))
                for side in ("actual", "expected")
            )
            np.testing.assert_allclose(
                actual, expected, rtol=MATRIX_REL, atol=MATRIX_ABS, err_msg=str(row["name"])
            )
