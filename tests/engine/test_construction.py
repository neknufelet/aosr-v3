"""構造算數考卷：隔離雙精度行程，獨立解析解與既有傳遞矩陣作裁判。

解析解以相對 2e-12、絕對 2e-10 Pa·s/m 比：給複數矩陣乘除與三角函式舍入空間，
避開空腔極點；共振附近用絕對項。不把這個數值舍入尺當成物理模型的精度宣稱。
直接呼叫對照則逐位相同，不用容差放過參數或公式變動。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from scipy.optimize import brentq

from aosr.config.paths import config_path
from aosr.config.physics_constants import load_physics_constants
from aosr.materials.catalog_absorption import complex_random_incidence_absorption
from aosr.runtime import child_process_env, configure_jax


_FREQUENCIES = [100., 250., 500., 1000., 4000.]
_POROUS: dict[str, object] = {"kind": "porous", "thickness_m": 0.05, "flow_resistivity_pa_s_m2": 19100., "material_class": "fibrous"}
_AIR: dict[str, object] = {"kind": "air", "thickness_m": 0.03}
_BOARD: dict[str, object] = {"kind": "board", "surface_mass_kg_m2": 2., "resistance_pa_s_m": 80.}
_MODELS = {
    "komatsu": {}, "miki": {}, "wilson_2015": {"porosity": 0.8, "tortuosity": 1.3},
    "mechel_1976": {"porosity": 0.8},
}
_VARIANTS = [(model, field) for model, extra in _MODELS.items()
             for field in ("thickness_m", "flow_resistivity_pa_s_m2", *extra)]
_VARIANTS += [("air", "thickness_m"), ("board", "surface_mass_kg_m2"), ("board", "resistance_pa_s_m")]


def _request(layers: list[dict[str, object]], *, backing: str = "rigid", mode: str = "normal",
             frequencies: list[float] | None = None, angles: list[float] | None = None, direct: bool = False) -> dict[str, object]:
    return {"mode": mode, "material": {"kind": "assembly", "layers": layers, "backing": backing},
            "frequencies_hz": _FREQUENCIES if frequencies is None else frequencies, "angles_deg": angles, "direct": direct}


def _resonance_hz() -> float:
    air = load_physics_constants(config_path("physics_constants.toml"))
    def residual(f: float) -> float:
        omega = 2 * np.pi * f
        return float(omega * 2 - air.rho_c / np.tan(omega / air.sound_speed_m_s * 0.03))
    return float(brentq(residual, 1., 1000., xtol=1e-12))


def _requests() -> dict[str, dict[str, object]]:
    rows = {"board_cavity": _request([_BOARD, _AIR]),
            "resonance": _request([{**_BOARD, "resistance_pa_s_m": load_physics_constants(config_path("physics_constants.toml")).rho_c}, _AIR], frequencies=[_resonance_hz()]),
            "board_air": _request([_BOARD], backing="air"),
            "board_air_angles": _request([_BOARD], backing="air", angles=[0., 37., 80.]),
            "order_forward": _request([_POROUS, _AIR]), "order_reverse": _request([_AIR, _POROUS]),
            "foam": _request([{**_POROUS, "material_class": "foam"}, _AIR]),
            "diffuse": _request([_BOARD, _POROUS, _AIR], mode="diffuse", direct=True)}
    for model, extra in _MODELS.items():
        layer = {**_POROUS, "model": model, **extra}
        rows[model] = _request([layer, _AIR], direct=True)
        rows[model + "_angles"] = _request([layer, _AIR], angles=[0., 37., 75.], direct=True)
        rows[model + "_diffuse"] = _request([layer, _AIR], mode="diffuse", direct=True)
        rows[model + "_foam"] = _request([{**layer, "material_class": "foam"}, _AIR])
        for field in ("thickness_m", "flow_resistivity_pa_s_m2", *extra):
            rows[model + "_" + field] = _request([{**layer, field: cast(float, layer[field]) * 1.01}, _AIR])
    for kind, base in (("air", _AIR), ("board", _BOARD)):
        backing = "rigid" if kind == "air" else "air"
        rows[kind] = _request([base], backing=backing)
        for field in ("thickness_m",) if kind == "air" else ("surface_mass_kg_m2", "resistance_pa_s_m"):
            rows[kind + "_" + field] = _request([{**base, field: cast(float, base[field]) * 1.01}], backing=backing)
    _boundary_requests(rows)
    return rows


def _boundary_requests(rows: dict[str, dict[str, object]]) -> None:
    for sigma in (6000., 72900., 5999.99, 72900.01):
        rows[f"sigma_{sigma}"] = _request([_AIR, {**_POROUS, "flow_resistivity_pa_s_m2": sigma}])
    # 答案只由論文四個原始數字推導，不用被測範圍屬性生成答案。
    low, high = 100 * 19100 / 72900, 5000 * 19100 / 6000
    rows["frequency_boundaries"] = _request([_POROUS], frequencies=[low * (1 - 1e-6), low, low * (1 + 1e-6), high * (1 - 1e-6), high, high * (1 + 1e-6)])
    negative = {**_POROUS, "thickness_m": 0.001, "flow_resistivity_pa_s_m2": 6000.}
    rows["negative_normal"] = _request([negative], frequencies=[100., 0.01])
    rows["negative_angles"] = _request([negative], frequencies=[100., 0.01], angles=[30., 60.])
    rows["negative_top_air"] = _request([_AIR, negative], frequencies=[100., 0.01], angles=[30., 60.])
    # Miki 在 2.4 Hz 的近掠角為正、0°為負，抓只檢查第一欄的錯。
    rows["negative_later_angle"] = _request([{**negative, "model": "miki"}], frequencies=[100., 2.4], angles=[89.99, 0.])
    for mode, angles in (("normal", None), ("angles", [0., 30., 80.]), ("diffuse", None)):
        rows["constant_" + mode] = {"mode": mode, "material": {"kind": "constant_impedance", "impedance_pa_s_m": 700.},
                                    "frequencies_hz": _FREQUENCIES, "angles_deg": angles}


def _run_requests(requests: dict[str, dict[str, object]], directory: Path, *, x64: bool = True) -> dict[str, dict[str, object]]:
    env = child_process_env(threads=1).copy()
    configure_jax(platforms="cpu", enable_x64=x64, env=env)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["TMPDIR"] = str(directory)
    source, output = directory / "requests.json", directory / "answers.json"
    source.write_text(json.dumps(requests))
    probe = Path(__file__).with_name("_construction_probe.py")
    process = subprocess.run([sys.executable, str(probe), str(source), str(output)], cwd=directory,
                             env=env, capture_output=True, text=True, timeout=180)
    assert process.returncode == 0, process.stdout + process.stderr
    return cast(dict[str, dict[str, object]], json.loads(output.read_text()))


@pytest.fixture(scope="module")
def results(tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict[str, object]]:
    return _run_requests(_requests(), tmp_path_factory.mktemp("construction"))


def _value(row: dict[str, object], key: str = "value") -> np.ndarray:
    assert "error" not in row, row
    data = cast(dict[str, object], row[key])
    return np.asarray(data["real"]) + 1j * np.asarray(data["imag"])


def _notes(row: dict[str, object]) -> list[dict[str, object]]:
    assert "error" not in row, row
    return cast(list[dict[str, object]], row["notes"])


def test_board_cavity_analytic_spectrum(results: dict[str, dict[str, object]]) -> None:
    """抓板壓力跳躍符號、面密度或空腔傳播厚度錯誤，整條頻率逐點比解析解。"""
    air = load_physics_constants(config_path("physics_constants.toml"))
    omega = 2 * np.pi * np.asarray(_FREQUENCIES)
    expected = 80 + 1j * omega * 2 - 1j * air.rho_c / np.tan(omega / air.sound_speed_m_s * 0.03)
    np.testing.assert_allclose(_value(results["board_cavity"]), expected, rtol=2e-12, atol=2e-10)


def test_board_resonance_is_matched(results: dict[str, dict[str, object]]) -> None:
    """抓共振時板與空腔虛部沒有抵消；r=ρc 時吸音必須為一。"""
    air = load_physics_constants(config_path("physics_constants.toml"))
    np.testing.assert_allclose(_value(results["resonance"]), air.rho_c, rtol=2e-12, atol=2e-10)
    np.testing.assert_allclose(_value(results["resonance"], "absorption"), 1., rtol=0, atol=1e-14)


def test_board_with_air_backing(results: dict[str, dict[str, object]]) -> None:
    """抓空氣背牆被默認成剛背，或薄板阻抗沒有加上ρc負載。"""
    air = load_physics_constants(config_path("physics_constants.toml"))
    expected = 80 + 1j * 2 * np.pi * np.asarray(_FREQUENCIES) * 2 + air.rho_c
    np.testing.assert_allclose(_value(results["board_air"]), expected, rtol=2e-12, atol=2e-10)


def test_board_angle_only_broadcasts_sheet(results: dict[str, dict[str, object]]) -> None:
    """抓斜入射把板質量或阻力多乘角度因子；角度只改空氣背後負載。"""
    air = load_physics_constants(config_path("physics_constants.toml"))
    sheet = 80 + 1j * 2 * np.pi * np.asarray(_FREQUENCIES) * 2
    expected = sheet[:, None] + air.rho_c / np.cos(np.deg2rad([0., 37., 80.]))
    np.testing.assert_allclose(_value(results["board_air_angles"]), expected, rtol=2e-12, atol=2e-10)


def test_layer_order_changes_impedance(results: dict[str, dict[str, object]]) -> None:
    """抓層疊排序或把房間側／牆體側當成可交换。"""
    assert not np.allclose(_value(results["order_forward"]), _value(results["order_reverse"]), rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("model", _MODELS)
@pytest.mark.parametrize("suffix", ["", "_angles", "_diffuse"])
def test_assembly_matches_direct_bitwise(results: dict[str, dict[str, object]], model: str, suffix: str) -> None:
    """抓任一模型參數轉接改單位、丟欄位，或垂直／斜入射／無規重複實作走歪。"""
    row = results[model + suffix]
    assert row["value"] == row["direct"], row


@pytest.mark.parametrize("model,field", _VARIANTS)
def test_each_numeric_field_changes_impedance(results: dict[str, dict[str, object]], model: str, field: str) -> None:
    """抓多孔、空氣、板任一數字欄位被忽略；逐個微擾須改變至少一個頻點。"""
    assert np.any(np.abs(_value(results[model]) - _value(results[model + "_" + field])) > 1e-8)


def test_komatsu_sigma_endpoints_and_rejection(results: dict[str, dict[str, object]]) -> None:
    """抓樣品流阻端點被排除、越界自動換模型，或拒收訊息少了兩條出路。"""
    for sigma in (6000., 72900.):
        assert "error" not in results[f"sigma_{sigma}"]
    for sigma in (5999.99, 72900.01):
        message = cast(str, results[f"sigma_{sigma}"]["error"])
        for text in ("第 2 層", str(sigma), "實測阻抗", "明確指定別的多孔模型", "樣品"):
            assert text in message


def test_komatsu_frequency_extrapolation_points(results: dict[str, dict[str, object]]) -> None:
    """抓錯用固定100–5000 Hz、漏上界或把含端點判成外推；答案由論文原始數字推導。"""
    low, high = 100 * 19100 / 72900, 5000 * 19100 / 6000
    notes = _notes(results["frequency_boundaries"])
    assert {n["code"] for n in notes} == {"komatsu_frequency_extrapolation"}
    for note in notes:
        assert note["layer_index"] == 1
        assert note["frequencies_hz"] == [low * (1 - 1e-6), high * (1 + 1e-6)]


def test_foam_annotation_only_for_komatsu(results: dict[str, dict[str, object]]) -> None:
    """抓材料類別被忽略，或將Komatsu範圍規則套到其他三種模型。"""
    assert {n["code"] for n in _notes(results["foam"])} == {"komatsu_material_class_outside_fit"}
    assert _notes(results["order_forward"]) == []
    for model in ("miki", "wilson_2015", "mechel_1976"):
        assert _notes(results[model + "_foam"]) == []


@pytest.mark.parametrize("case,frequency,layer", [
    ("negative_normal", "0.01", "第 1 層"), ("negative_angles", "0.01", "第 1 層"),
    ("negative_top_air", "0.01", "第 2 層"),
    ("negative_later_angle", "2.4", "第 1 層"),
])
def test_real_negative_impedance_rejected(results: dict[str, dict[str, object]], case: str, frequency: str, layer: str) -> None:
    """抓只查第一個頻点或角度、查中間層而非最終表面阻抗，或負實部未拒收。"""
    message = cast(str, results[case]["error"])
    for text in ("實部", frequency, layer, "檢查", "不自動換模型"):
        assert text in message
    if case != "negative_normal":
        assert "角度" in message


def test_constant_impedance_and_closed_absorption(results: dict[str, dict[str, object]]) -> None:
    """抓定值阻抗變成隨頻率／角度改變，或舊晚期吸音路徑被換成離散角度積分。"""
    air = load_physics_constants(config_path("physics_constants.toml"))
    for case, shape in (("constant_normal", (len(_FREQUENCIES),)), ("constant_angles", (len(_FREQUENCIES), 3))):
        np.testing.assert_array_equal(_value(results[case]), np.full(shape, 700. + 0j))
    expected = complex_random_incidence_absorption(700 / air.rho_c)
    np.testing.assert_array_equal(_value(results["constant_diffuse"]).real, np.full(len(_FREQUENCIES), expected))


def test_diffuse_absorption_matches_direct_bitwise(results: dict[str, dict[str, object]]) -> None:
    """抓無規吸音沒用預設Paris完整角度格、板未广播角度，或另寫積分公式。"""
    assert results["diffuse"]["value"] == results["diffuse"]["direct"]


def test_double_precision_results(results: dict[str, dict[str, object]]) -> None:
    """抓回傳單精度、非JAX算數結果或無規吸音誤回複數。"""
    for name, row in results.items():
        if "value" in row:
            dtype = cast(dict[str, object], row["value"])["dtype"]
            assert dtype == ("float64" if "diffuse" in name else "complex128")


def test_single_precision_child_rejected(tmp_path: Path) -> None:
    """抓單精度環境靜默算下去；只給隔離子行程設定，考卷行程不改全域。"""
    requests = {name: row for name, row in _requests().items()
                if name in {"komatsu", "komatsu_angles", "diffuse", "constant_normal", "constant_angles", "constant_diffuse"}}
    rows = _run_requests(requests, tmp_path, x64=False)
    assert rows.keys() == requests.keys()
    for row in rows.values():
        assert "需要雙精度" in cast(str, row["error"])
