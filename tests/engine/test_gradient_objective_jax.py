"""第三階段梯度目標（#561 壬一）：大峰的平滑代理、夾擠、對偵測器、權重 0、梯度、控制組、登記簿與帳本拒收。

子行程（``_gradient_objective_jax_probe.py``）只求值；父行程用既有評估器（``evaluate_stage_one_low_frequency``）
與既有合分（``score_stage_one``）當參考，殘差由評估器交回的直線自己扣，不另抄目標公式。溫度、權重、較差參考
一律從登記簿讀（正式那份、或權重改 0 的暫存副本），考卷不寫那幾個數字。
"""
from __future__ import annotations

import ast
import json
import math
import subprocess
import sys
from functools import cache
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from aosr.config.construction_presets import load_construction_presets
from aosr.config.frequency_axis import LowFrequencyAxis, low_frequency_axis_frequencies
from aosr.config.material_gradient import load_material_gradient_settings
from aosr.config.paths import config_path
from aosr.config.quality_targets import QualityTargets, SettingEntry, TargetEntry, WeightTable, load_quality_targets
from aosr.gradient import material_gradient as gradient_search
from aosr.gradient import material_gradient_ledger as ledger
from aosr.gradient import template_ledger as templates
from aosr.gradient import template_segment as segment
from aosr.gradient.report import PROXY_MISMATCH, proxy_peak_line
from aosr.runtime import child_process_env, configure_jax
from aosr.scoring.gradient_objective_settings import (
    PROXY, PROXY_TARGET_KEY, SETTING_KEYS, TEMPERATURE_KEY, WEIGHTS_KEY, gradient_objective_settings,
)
from aosr.scoring.low_frequency_response import evaluate_low_frequency_response
from aosr.scoring.low_frequency_response_contract import LowFrequencyResponsePayload
from aosr.scoring.low_frequency_response_cost import cost_evaluation
from aosr.scoring.receiver_set import ReceiverRole, ReceiverSet, primary_cross
from aosr.scoring.stage_one_low_frequency import evaluate_stage_one_low_frequency
from aosr.scoring.stage_one_low_frequency_contract import StageOneLowFrequencyEvaluation, StageOneLowFrequencyInput
from aosr.scoring.stage_one_score import PRINCIPAL, ROBUSTNESS, score_stage_one, stage_one_score_settings
from tests.engine._gradient_objective_rows import PURPOSE, registry_objective
from tests.engine._material_gradient_loop_probe import synthetic_header
from tests.engine._material_gradient_templates_probe import kept_walls
from tests.engine._material_gradient_templates_probe import synthetic_header as template_header
from tests.engine._placement import POINT_PLACEMENT
from tests.engine._precision_contracts import contract_value
from tests.engine._stage_one_low_frequency_curves import (
    AXIS_HZ, DIRECTIONS, candidate, demo_curve_db, gaussian_db, speaker, tilted_db,
)
from tests.engine.test_material_gradient_templates import KEPT, _context
from tests.engine.test_ranking_recost import _PROVENANCE
from tests.engine.test_transfer_matrix_reference import GRADIENT_ABS, GRADIENT_REL

FloatArray = NDArray[np.float64]
Record = dict[str, object]
_PROBE = Path(__file__).with_name("_gradient_objective_jax_probe.py")

# 控制組（派工單第 5 條）：主線 a1be64b5 與這一支同值、先在主線算好再釘。加代理三格不准動第一階段設定、
# 整份登記簿（第一階段那一段不收）、第二階段低頻頻響類的兩個指紋；改了就紅。
STAGE_ONE_FINGERPRINT = "2d9c47e3be7e3ffcb29f4c6f57fc17fb55da1c0fb77f7bb25a449cc2673f98ea"
REGISTRY_FINGERPRINT = "2e9bd2d4f0411968b314881aafdfcc4cb64cecb5b1b9e6aa1af06a39455aed32"
LOW_FREQUENCY_COST_FINGERPRINT = "39965da27931909ece3e92e68c8f98caab7aea8ddb2b6a78fcb9b41bd83b52e2"

# 合成殘差：主位曲線＝一條斜線（擬合直線吃掉它）加一個形狀；每支喇叭各一條。
_SPIKE_CASES = ("interior_spike", "right_spike")
_DETECTOR_CASES = (*_SPIKE_CASES, "plateau", "demo")
_CASES = (*_DETECTOR_CASES, "endpoint_max", "flat")
# 溫度往 0 走：登記值乘 2 的負次方（不寫溫度的數字）。
_HALVINGS = 7
# 中央差分以每一格能量本身為尺度，h/E=1e-5（同 test_stage_one_objective_jax）。
_DIFFERENCE_STEP = 1e-5


@cache
def _receivers() -> ReceiverSet:
    return primary_cross(primary_position_m=(3.0, 2.0, 1.2), offset_m=0.1)


def _inside() -> NDArray[np.bool_]:
    lower, upper = registry_objective().stage_one.scored_range_hz
    return np.asarray((AXIS_HZ >= lower) & (AXIS_HZ <= upper))


def _interior() -> NDArray[np.int64]:
    return np.flatnonzero(_inside())[1:-1]


def _bump(indices: NDArray[np.int64] | list[int], height: float) -> FloatArray:
    shape = np.zeros_like(AXIS_HZ)
    shape[np.asarray(indices)] = height
    return shape


def _curves(name: str) -> tuple[tuple[FloatArray, FloatArray], ...]:
    """每支喇叭（主位曲線, 周圍六點的底線）：主位＝一條斜線加一個形狀；周圍跟著底線（不帶那個形狀），
    尖峰只在主位，把主位的峰壓低時座位穩健性也一起變好，「壓低峰會降目標」這一題才看得到三項同向。"""
    base = tilted_db(AXIS_HZ, -1.0, 80.0)
    interior = _interior()
    middle, quarter, three = (interior[len(interior) * k // 4] for k in (2, 1, 3))
    if name == "interior_spike":
        return (base + _bump([middle], 6.0), base), (base + _bump([quarter], 3.0), base)
    if name == "right_spike":
        return (base + _bump([middle], 2.0), base), (base + _bump([three], 7.0), base)
    if name == "plateau":
        return (base + _bump(list(range(middle - 2, middle + 3)), 4.0), base), (base + _bump([quarter], 1.0), base)
    if name == "endpoint_max":
        return (base + _bump([np.flatnonzero(_inside())[-1]], 12.0), base), (base + _bump([middle], 1.0), base)
    if name == "flat":
        level = np.full_like(AXIS_HZ, 75.0)
        return (level, level), (level.copy(), level)
    demo = demo_curve_db()
    other = demo + gaussian_db(AXIS_HZ, 110.0, -5.0, 0.13)
    return (demo, demo), (other, other)


def _surrounding(background: FloatArray) -> dict[str, FloatArray]:
    return {direction: background + (index + 1) * 0.3 + gaussian_db(AXIS_HZ, 40.0 + index * 20.0,
                                                                    (-1.0) ** index * 2.0, 0.12)
            for index, direction in enumerate(DIRECTIONS)}


def _energies(name: str) -> FloatArray:
    """``(喇叭, 座位, F)``，座位照座位組原序。"""
    channels = []
    for primary, background in _curves(name):
        levels = {"main": primary, **_surrounding(background)}
        channels.append([10.0 ** (levels[point.receiver_id] / 10.0) for point in _receivers().points])
    return np.asarray(channels, dtype=np.float64)


def _evaluator_input(energies: FloatArray) -> StageOneLowFrequencyInput:
    speakers = []
    for channel, speaker_id in enumerate(("left", "right")):
        rows = {point.receiver_id: 10.0 * np.log10(energies[channel, index])
                for index, point in enumerate(_receivers().points)}
        speakers.append(speaker(rows["main"], {direction: rows[direction] for direction in DIRECTIONS},
                                speaker_id=speaker_id))
    data = candidate(*speakers)
    # dB 往返有捨入：直接把能量換回考卷那一份，評估器讀的就是交給子行程的同一組數。
    exact = []
    for channel, curves in enumerate(data.speakers):
        by_id = {point.receiver_id: tuple(float(value) for value in energies[channel, index])
                 for index, point in enumerate(_receivers().points)}
        exact.append(curves.model_copy(update={
            "primary": curves.primary.model_copy(update={"fem_energy": by_id["main"]}),
            "surrounding": tuple(seat.model_copy(update={"fem_energy": by_id[seat.receiver_id]})
                                 for seat in curves.surrounding)}))
    return data.model_copy(update={"speakers": tuple(exact)})


def _evaluate(energies: FloatArray) -> StageOneLowFrequencyEvaluation:
    return evaluate_stage_one_low_frequency(_evaluator_input(energies),
                                            scoring_range_hz=registry_objective().stage_one.scored_range_hz)


def _residuals(energies: FloatArray, evaluation: StageOneLowFrequencyEvaluation) -> FloatArray:
    """考卷自己的殘差 ``(喇叭, 計分範圍內)``：主位能量轉 dB、扣評估器交回的直線（斜率、截距）。"""
    inside = _inside()
    primary = next(index for index, point in enumerate(_receivers().points) if point.role == ReceiverRole.PRIMARY)
    rows = []
    for channel, item in enumerate(evaluation.speakers):
        payload = item.payload
        assert payload is not None
        level = 10.0 * np.log10(energies[channel, primary, inside])
        rows.append(level - (payload.own_line_slope_db_per_octave * np.log2(AXIS_HZ[inside])
                             + payload.own_line_intercept_db))
    return np.asarray(rows)


def _numpy_proxy(residuals: FloatArray, temperature: float) -> float:
    interior = residuals[:, 1:-1].ravel()
    top = float(np.max(interior))
    return top + temperature * float(np.log(np.sum(np.exp((interior - top) / temperature))))


def _numpy_objective(energies: FloatArray) -> float:
    settings = registry_objective()
    evaluation = _evaluate(energies)
    score = score_stage_one(evaluation, settings.stage_one)
    assert score.screening_total is not None, score.reason_codes
    proxy = _numpy_proxy(_residuals(energies, evaluation), settings.temperature_db)
    return score.screening_total + settings.proxy_weight * proxy / settings.worse_reference


def _decode(values: object) -> FloatArray:
    return np.asarray([float.fromhex(value) for value in cast(list[str], values)], dtype=np.float64)


def _ruler(scale: float) -> float:
    """兩邊程式（評估器 numpy、目標 JAX）擬合直線的捨入不同：照低頻目標 JAX 對 Python 那份契約的相對尺，
    以 max(|量|, 1 dB) 為尺度。"""
    return contract_value("stage_one_objective_jax_vs_python") * max(abs(scale), 1.0)


def _interior_count() -> int:
    return 2 * len(_interior())


# ---- 子行程 ----

def _zero_weight_registry(directory: Path) -> Path:
    """登記簿抄到暫存、只把代理那一列權重改 0（照登記簿的寫法逐字找那一列；真檔不動）。"""
    source = config_path("quality_targets.toml").read_text(encoding="utf-8")
    weight = registry_objective().proxy_weight
    anchor = f'name = "{PROXY}"\nvalue = {weight!r}\n'
    assert source.count(anchor) == 1, "登記簿裡代理權重那一列的寫法變了，考卷要跟著改"
    path = directory / "quality_targets.zero.toml"
    path.write_text(source.replace(anchor, f'name = "{PROXY}"\nvalue = 0.0\n'), encoding="utf-8")
    return path


def _run_probe(directory: Path) -> Record:
    temperature = registry_objective().temperature_db
    problem = {
        "purpose": PURPOSE, "receiver_set": _receivers().model_dump(mode="json"),
        "frequencies_hz": [float(value).hex() for value in AXIS_HZ],
        "temperatures": [(temperature * 2.0 ** -k).hex() for k in range(_HALVINGS)],
        "registries": {"real": str(config_path("quality_targets.toml")), "zero": str(_zero_weight_registry(directory))},
        "cases": {name: [[[float(value).hex() for value in row] for row in channel] for channel in _energies(name)]
                  for name in _CASES},
    }
    problem_path, output_path = directory / "problem.json", directory / "output.json"
    problem_path.write_text(json.dumps(problem), encoding="utf-8")
    env = child_process_env(threads=1)
    configure_jax(platforms="cpu", enable_x64=True, env=env)
    env["PYTHONDONTWRITEBYTECODE"], env["TMPDIR"] = "1", str(directory)
    process = subprocess.run([sys.executable, str(_PROBE), str(problem_path), str(output_path)],
                             cwd=directory, env=env, capture_output=True, text=True, timeout=300)
    assert process.returncode == 0, process.stdout + process.stderr
    return cast(Record, json.loads(output_path.read_text(encoding="utf-8")))


@pytest.fixture(scope="module")
def probe(tmp_path_factory: pytest.TempPathFactory) -> Record:
    return _run_probe(tmp_path_factory.mktemp("gradient-objective-jax"))


def _case(probe: Record, name: str, registry: str = "real") -> Record:
    return cast(Record, cast(Record, cast(Record, probe[registry])["cases"])[name])


def _field(probe: Record, name: str, kind: str, field: str, registry: str = "real") -> FloatArray:
    return _decode(cast(Record, _case(probe, name, registry)[kind])[field])


# ---- 1. 夾擠 ----

@pytest.mark.parametrize("name", _CASES)
def test_proxy_is_sandwiched_by_interior_max(probe: Record, name: str) -> None:
    """抓代理不是軟最大值（漏了溫度、少乘 T、加了頻率權重、只取一支喇叭）：max r_內部 ≤ P ≤ max r_內部 ＋ T·ln N，
    N＝兩支喇叭的內部點數；r 由考卷從評估器交回的直線自己扣。上界數字等於考卷自己算的 T·ln N。"""
    energies = _energies(name)
    residuals = _residuals(energies, _evaluate(energies))
    top = float(np.max(residuals[:, 1:-1]))
    bound = registry_objective().temperature_db * math.log(_interior_count())
    assert float.fromhex(cast(str, _case(probe, name)["bound"])) == bound
    for kind in ("eager", "jit"):
        proxy = float(_field(probe, name, kind, "big_peak_proxy_db")[0])
        assert top - _ruler(top) <= proxy <= top + bound + _ruler(top), (kind, proxy, top, bound)
    assert np.allclose(_decode(_case(probe, name)["residual"]).reshape(residuals.shape), residuals,
                       rtol=0.0, atol=_ruler(float(np.max(np.abs(residuals)))))


@pytest.mark.parametrize("name", _CASES)
def test_proxy_converges_to_interior_max_as_temperature_goes_to_zero(probe: Record, name: str) -> None:
    """抓溫度沒進代理（或進錯位置）：溫度從登記值一路減半，P − max r_內部 每次不增、都在 [0, T·ln N]，
    最後一個不超過那個溫度的 T·ln N（往 0 收斂）。"""
    energies = _energies(name)
    top = float(np.max(_residuals(energies, _evaluate(energies))[:, 1:-1]))
    proxies = cast(dict[str, str], _case(probe, name)["temperatures"])
    gaps = []
    for encoded, value in proxies.items():
        temperature = float.fromhex(encoded)
        gap = float.fromhex(value) - top
        assert -_ruler(top) <= gap <= temperature * math.log(_interior_count()) + _ruler(top), (temperature, gap)
        gaps.append(gap)
    assert len(gaps) == _HALVINGS
    assert all(later <= earlier + _ruler(top) for earlier, later in zip(gaps, gaps[1:], strict=False))


def test_endpoint_maximum_is_not_followed(probe: Record) -> None:
    """抓代理看得到計分範圍的兩端點（偵測器 ``range(1, len-1)`` 永遠不報端點）：最後一個計分點比內部最大殘差
    高出不只 T·ln N，代理仍不超過內部最大 ＋ T·ln N，所以比端點那一格低。"""
    energies = _energies("endpoint_max")
    residuals = _residuals(energies, _evaluate(energies))
    top, endpoint = float(np.max(residuals[:, 1:-1])), float(np.max(residuals[:, -1]))
    bound = registry_objective().temperature_db * math.log(_interior_count())
    assert endpoint > top + bound + _ruler(endpoint), "題目要讓端點明顯高過內部，這一題才分得出來"
    proxy = float(_field(probe, "endpoint_max", "jit", "big_peak_proxy_db")[0])
    assert proxy <= top + bound + _ruler(top) < endpoint


# ---- 2. 代理對偵測器 ----

@pytest.mark.parametrize("name", _DETECTOR_CASES)
def test_proxy_minus_detector_peak_is_within_bound(probe: Record, name: str) -> None:
    """抓代理跟終點那支偵測器量的不是同一個量：評估器 ``features[strongest_peak_index].depth_db`` 兩支取較大
    （終點 ``worst_big_peak`` 的取法），內部有嚴格局部極大時它等於內部最大殘差，0 ≤ P − 真大峰 ≤ T·ln N。"""
    energies = _energies(name)
    evaluation = _evaluate(energies)
    depths = []
    for item in evaluation.speakers:
        payload = item.payload
        assert payload is not None and payload.strongest_peak_index is not None
        depths.append(payload.features[payload.strongest_peak_index].depth_db)
    peak = max(depths)
    top = float(np.max(_residuals(energies, evaluation)[:, 1:-1]))
    assert abs(peak - top) <= _ruler(top)
    bound = registry_objective().temperature_db * math.log(_interior_count())
    gap = float(_field(probe, name, "jit", "big_peak_proxy_db")[0]) - peak
    assert -_ruler(peak) <= gap <= bound + _ruler(peak), (gap, bound)


def test_report_line_flags_mismatch_without_judging() -> None:
    """抓報告那一行的判法接反：差在 0～上界（各放寬地板）裡不印「代理與偵測器對不上」，代理比真大峰低、
    或高過上界才印；只是多印一句（函式照常交回一行字，不拋錯）。"""
    bound = registry_objective().temperature_db * math.log(_interior_count())
    floor = 1e-9
    assert PROXY_MISMATCH not in proxy_peak_line("終點", 10.0 + bound / 2.0, 10.0, bound, floor)
    assert PROXY_MISMATCH in proxy_peak_line("終點", 10.0 - 1.0, 10.0, bound, floor)
    assert PROXY_MISMATCH in proxy_peak_line("終點", 10.0 + bound + 1.0, 10.0, bound, floor)


# ---- 3. 權重 0 ----

@pytest.mark.parametrize("name", _CASES)
def test_zero_weight_objective_and_gradient_equal_screening_bitwise(probe: Record, name: str) -> None:
    """抓權重沒從登記簿讀、或權重 0 時代理項仍漏進目標：登記簿暫存副本把權重改 0，新合分逐位等於舊合分
    （直接與編譯後），新合分的梯度逐位等於舊合分的梯度；指紋跟正式那份不同（只改這一格就不能接續），
    第一階段設定指紋相同。"""
    zero = cast(Record, probe["zero"])
    assert float.fromhex(cast(str, zero["weight"])) == 0.0
    assert zero["fingerprint"] != cast(Record, probe["real"])["fingerprint"]
    assert zero["stage_one_fingerprint"] == cast(Record, probe["real"])["stage_one_fingerprint"]
    case = _case(probe, name, "zero")
    for kind in ("eager", "jit"):
        arrays = cast(Record, case[kind])
        assert arrays["objective_total"] == arrays["screening_total"], kind
    assert case["gradient"] == case["screening_gradient"]


@pytest.mark.parametrize("name", _CASES)
def test_screening_total_is_bitwise_the_stage_one_function(probe: Record, name: str) -> None:
    """抓舊合分被代理碰到（第一階段那把尺要一位都不動）：正式登記簿與權重 0 的副本，舊合分都逐位等於第一階段
    原函式 ``objective_from_energies`` 的合分（直接對直接、編譯對編譯），舊合分的梯度逐位等於原函式的梯度。"""
    for registry in ("real", "zero"):
        case = _case(probe, name, registry)
        assert cast(Record, case["eager"])["screening_total"] == case["stage_one_eager"], registry
        assert cast(Record, case["jit"])["screening_total"] == case["stage_one_jit"], registry
        assert case["screening_gradient"] == case["stage_one_gradient"], registry


# ---- 4. 梯度 ----

@pytest.mark.parametrize("name", _SPIKE_CASES)
def test_gradient_at_spike_pushes_the_peak_down(probe: Record, name: str) -> None:
    """抓代理的梯度符號反了或斷線：尖峰那一支喇叭主位、尖峰那一個頻點，下降方向（負梯度）對那一格能量是負的
    ——梯度下降會把峰壓低、壓低會降目標；所以新合分與代理對那一格能量的編譯後偏導數是正的、有限、非零
    （派工單寫「符號是負」指的是這個下降方向；偏導數本身是正，見報告「解讀與偏離」）。"""
    energies = _energies(name)
    residuals = _residuals(energies, _evaluate(energies))
    channel, position = np.unravel_index(int(np.argmax(residuals[:, 1:-1])), residuals[:, 1:-1].shape)
    frequency = int(_interior()[position])
    primary = next(index for index, point in enumerate(_receivers().points) if point.role == ReceiverRole.PRIMARY)
    for key in ("gradient", "proxy_gradient"):
        gradient = _decode(_case(probe, name)[key]).reshape(energies.shape)
        value = float(gradient[channel, primary, frequency])
        assert math.isfinite(value) and -value < 0.0, (key, value)


def test_energy_gradient_matches_numpy_central_difference(probe: Record) -> None:
    """抓代理的梯度跟它的值對不上：每個聲道、座位、頻點分開推（考卷的參考＝既有評估器＋既有合分＋考卷自己的
    代理），比無單位 E·∂J/∂E；差分步長與容差同 ``test_stage_one_objective_jax``（傳遞矩陣考卷那把尺）。"""
    energies = _energies("interior_spike")
    automatic = _decode(_case(probe, "interior_spike")["gradient"]).reshape(energies.shape) * energies
    central = np.empty_like(energies)
    for index in np.ndindex(energies.shape):
        plus, minus = energies.copy(), energies.copy()
        step = energies[index] * _DIFFERENCE_STEP
        plus[index] += step
        minus[index] -= step
        central[index] = (_numpy_objective(plus) - _numpy_objective(minus)) / (2.0 * _DIFFERENCE_STEP)
    np.testing.assert_allclose(automatic, central, rtol=GRADIENT_REL, atol=GRADIENT_ABS)


def test_value_matches_numpy_reference(probe: Record) -> None:
    """抓新合分的組法錯（權重或較差參考接錯、少了舊合分）：編譯後新合分對考卷參考（既有合分 ＋ w·考卷代理／ref）
    在低頻目標 JAX 對 Python 那份契約的相對尺內。"""
    for name in _CASES:
        expected = _numpy_objective(_energies(name))
        actual = float(_field(probe, name, "jit", "objective_total")[0])
        assert abs(actual - expected) <= _ruler(expected), (name, actual, expected)


# ---- 5. 控制組 ----

def test_control_fingerprints_unchanged() -> None:
    """抓加代理三格動到第一階段名次或第二階段比較身分：第一階段設定指紋、整份登記簿指紋、第二階段低頻頻響類的
    評估設定指紋與類代價指紋，都等於主線（加三格之前）的值。低頻頻響那條照主對話在主線量的那一組輸入重算。"""
    registry = load_quality_targets(config_path("quality_targets.toml"))
    assert stage_one_score_settings(registry, PURPOSE).fingerprint == STAGE_ONE_FINGERPRINT
    assert registry.fingerprint == REGISTRY_FINGERPRINT
    frequencies = np.asarray(low_frequency_axis_frequencies(LowFrequencyAxis.SEARCH_48)[0])
    primary = [3 * np.sin(np.log2(frequencies) * 4), 5 * np.cos(np.log2(frequencies) * 3)]
    speakers = [speaker(curve, {direction: curve + (index + 1) * (channel + 1) / 3
                                for index, direction in enumerate(DIRECTIONS)},
                        frequencies=frequencies, speaker_id=name)
                for channel, (curve, name) in enumerate(zip(primary, ("left", "right"), strict=True))]
    measured = evaluate_low_frequency_response(
        candidate(*speakers, axis=LowFrequencyAxis.SEARCH_48), registry=registry, purpose=PURPOSE,
        scene_fingerprint="a" * 64, placement=POINT_PLACEMENT, provenance=_PROVENANCE)
    payload = measured.payload
    assert isinstance(payload, LowFrequencyResponsePayload) and payload.settings_fingerprint == STAGE_ONE_FINGERPRINT
    costed = cost_evaluation(measured, registry.stage_one(PURPOSE), registry.fingerprint)
    assert costed.category_cost is not None
    assert costed.category_cost.cost_settings_fingerprint == LOW_FREQUENCY_COST_FINGERPRINT


# ---- 6. 登記簿與寫死的數字 ----

def test_registry_cells_exist_baseline_and_share_reference() -> None:
    """抓三格少一格、不是 baseline、或代理的較差參考跟兩分項不同（權重比例就不是登記的那個）：三個具名鍵都在、
    狀態都是 baseline；代理的較差參考等於主位起伏與座位穩健性兩分項的那一個（讀出來比）。"""
    registry: QualityTargets = load_quality_targets(config_path("quality_targets.toml"))
    section = registry.stage_one(PURPOSE)
    temperature, target, weights = (section.entry(key) for key in SETTING_KEYS)
    assert isinstance(temperature, SettingEntry) and temperature.status == "baseline"
    assert isinstance(target, TargetEntry) and target.status == "baseline"
    assert isinstance(weights, WeightTable) and [item.name for item in weights.item] == [PROXY]
    assert all(item.status == "baseline" for item in weights.item)
    stage_one = stage_one_score_settings(registry, PURPOSE)
    assert target.worse_reference == stage_one.target(PRINCIPAL).worse_reference
    assert target.worse_reference == stage_one.target(ROBUSTNESS).worse_reference
    settings = gradient_objective_settings(registry, PURPOSE)
    assert settings.status == "baseline" and settings.stage_one.fingerprint == STAGE_ONE_FINGERPRINT
    assert (TEMPERATURE_KEY, PROXY_TARGET_KEY, WEIGHTS_KEY) == SETTING_KEYS


def _objective_sources() -> list[Path]:
    root = Path(__file__).resolve().parents[2] / "src" / "aosr"
    return [*sorted((root / "scoring").glob("gradient_objective*.py")), *sorted((root / "gradient").glob("*.py"))]


def test_objective_code_has_no_hardcoded_proxy_numbers() -> None:
    """抓溫度、權重寫死在程式裡（數字只住登記簿）：梯度目標兩支新模組與梯度搜尋整包的每一支 .py，語法樹裡沒有
    等於登記的溫度或權重的浮點常數；兩支新模組也沒有等於代理較差參考的浮點常數。數字從登記簿讀，考卷不寫。"""
    settings = registry_objective()
    found = []
    for path in _objective_sources():
        forbidden = {settings.temperature_db, settings.proxy_weight}
        if path.name.startswith("gradient_objective"):
            forbidden.add(settings.worse_reference)
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Constant) and isinstance(node.value, float) and node.value in forbidden:
                found.append(f"{path.name}:{node.lineno} {node.value!r}")
    assert len(_objective_sources()) > 2
    assert not found, found


# ---- 7. 帳本拒收 ----

def _downgrade(path: Path, target: Path, version: str) -> Path:
    lines = path.read_bytes().splitlines()
    header = json.loads(lines[0])
    header["ledger_version"] = version
    rows = []
    for line in lines[1:]:
        row = json.loads(line)
        row.pop("screening_total")
        row["components"] = row["components"][:2]
        rows.append(row)
    target.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in (header, *rows)), encoding="utf-8")
    return target


def _material_ledger(directory: Path) -> tuple[Path, ledger.MaterialGradientHeader]:
    """一本只有取規格那一列的 v5 帳本（合成標頭，六面兩面剛性）。"""
    settings = load_material_gradient_settings(config_path("material_gradient.toml"))
    presets = load_construction_presets(config_path("construction_presets.toml"))
    designs = gradient_search.wall_designs_from_presets(
        ["glass_wool_48k_50mm_gap50", None, "glass_wool_48k_50mm_gap50", "glass_wool_48k_50mm", None,
         "glass_wool_24k_50mm"], settings, presets)
    snapped = gradient_search.snap_to_specs(gradient_search.baseline_theta(designs), designs)
    header = synthetic_header(designs, settings)
    path = directory / "ledger.jsonl"
    ledger.write_ledger(path, header, (ledger.snap_row(snapped, 3.5, 3.0, (1.0, 2.0, 2.0), 0.0, 0),))
    return path, header


def test_v4_ledger_and_old_objective_header_are_rejected_on_resume(tmp_path: Path) -> None:
    """抓接續照收加代理之前的帳本：v4 帳本（沒有舊合分、只有兩分項）接續時拒收、訊息講「大峰代理」那一句；
    報告用的讀法（``legacy=True``、不給預期標頭）照讀；v5 帳本標頭的評分常數指紋跟這一次不同也拒收、訊息同一句；
    寫 v4 一律拒收。控制組：原檔照預期標頭讀得回、列帶三分項與舊合分。"""
    path, header = _material_ledger(tmp_path)
    read_header, rows = ledger.read_ledger(path, expected_header=header)
    assert read_header.ledger_version == ledger.LEDGER_VERSION and len(rows[0].components) == 3
    assert rows[0].screening_total is not None
    old = _downgrade(path, tmp_path / "v4.jsonl", ledger.LEGACY_LEDGER_VERSION)
    with pytest.raises(ValueError, match=ledger.PROXY_RERUN_TEXT):
        ledger.read_ledger(old)
    with pytest.raises(ValueError, match=ledger.PROXY_RERUN_TEXT):
        ledger.read_ledger(old, expected_header=header)
    with pytest.raises(ValueError, match=ledger.PROXY_RERUN_TEXT):
        ledger.read_ledger(old, expected_header=header, legacy=True)
    legacy_header, legacy_rows = ledger.read_ledger(old, legacy=True)
    assert legacy_header.ledger_version == ledger.LEGACY_LEDGER_VERSION and legacy_rows[0].is_legacy()
    other = header.model_copy(update={"objective_fingerprint": "0" * 64})
    with pytest.raises(ValueError, match="大峰代理"):
        ledger.read_ledger(path, expected_header=other)
    with pytest.raises(ValueError):
        ledger.write_ledger(tmp_path / "again.jsonl", legacy_header, legacy_rows)


def test_v1_template_ledger_and_old_objective_header_are_rejected_on_resume(tmp_path: Path) -> None:
    """抓接續照收加代理之前的換樣板帳本：沒有可換目標的一輪（不求值）寫一本 v2，換回 v1 形狀後，帶起點或預期
    標頭讀都拒收、訊息講「大峰代理」那一句；報告的讀法照讀；v2 標頭的評分常數指紋不同也拒收、訊息同一句。"""
    settings = load_material_gradient_settings(config_path("material_gradient.toml")).model_copy(update={"family": ()})
    context = _context(settings)
    surfaces = tuple((wall, None) for wall in gradient_search.wall_surface_ids())
    start = templates.TemplateStart(surfaces, (), 1.5, 1.25, (1.0, 2.0, 0.5))
    header = template_header(settings, surfaces, kept_walls(KEPT))
    path = tmp_path / "templates.jsonl"
    segment.run_template_rounds(context, path, header, start)
    _, rows = templates.read_template_ledger(path, expected_header=header, settings=settings, start=start)
    assert rows[0].screening_total == 1.25 and len(rows[0].components) == 3
    old = _downgrade(path, tmp_path / "v1.jsonl", templates.LEGACY_TEMPLATE_LEDGER_VERSION)
    with pytest.raises(ValueError, match=ledger.PROXY_RERUN_TEXT):
        templates.read_template_ledger(old)
    with pytest.raises(ValueError, match=ledger.PROXY_RERUN_TEXT):
        templates.read_template_ledger(old, settings=settings, start=start, legacy=True)
    with pytest.raises(ValueError, match=ledger.PROXY_RERUN_TEXT):
        templates.read_template_ledger(old, expected_header=header, legacy=True)
    legacy_header, legacy_rows = templates.read_template_ledger(old, legacy=True)
    assert legacy_header.ledger_version == templates.LEGACY_TEMPLATE_LEDGER_VERSION and legacy_rows[0].is_legacy()
    other = header.model_copy(update={"objective_fingerprint": "0" * 64})
    with pytest.raises(ValueError, match="大峰代理"):
        templates.read_template_ledger(path, expected_header=other)
