"""第三階段梯度目標（#561 壬一）：大峰的平滑代理、夾擠、對偵測器、權重 0、梯度、控制組、登記簿與帳本拒收。

子行程（``_gradient_objective_jax_probe.py``）只求值；父行程用既有評估器（``evaluate_stage_one_low_frequency``）
與既有合分（``score_stage_one``）當參考，殘差由評估器交回的直線自己扣，不另抄目標公式。溫度、權重、較差參考
一律從登記簿讀（正式那份、或權重改 0 的暫存副本），考卷不寫那幾個數字。

夾擠的對象是內部最大殘差：max r_內部 ≤ P ≤ max r_內部 ＋ T·ln N 恆成立；真大峰（偵測器）≤ max r_內部 ≤ P 也恆成立，
P − 真大峰 落在 [0, 上界] 只在內部最大值是嚴格局部極大時（審查的「肩膀」反例：端點旁那一點最大、但右鄰更高）。
代理可以是負的（內部全在擬合直線下），不截零。
"""
from __future__ import annotations

import ast
import hashlib
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
from aosr.gradient.report import LEGACY_ENDPOINT_NOTE, PROXY_MISMATCH, proxy_lines
from aosr.runtime import child_process_env, configure_jax
from aosr.scoring.gradient_objective_jax import gradient_objective_constants
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
from tests.engine._scheme_cache import shared_json
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
# 控制組（審查第 3 條）：主線 a1be64b5 的第一階段原函式（``objective_constants`` ＋ ``objective_from_energies``）對
# 這一支考卷每一組能量算的舊合分（直接、編譯後，``float.hex``）與編譯後梯度（十六進位逐格以逗號接起來的 SHA-256）。
# 在主線那一份原始碼上先算好再釘（考卷不叫 git）；加代理之後舊合分、權重 0 的新合分與梯度都要逐位等於它。
_MAINLINE_SCREENING: dict[str, tuple[str, str, str]] = {
    "interior_spike": ("0x1.8414e09e863d8p+0", "0x1.8414e09e863d8p+0",
                       "911a91d87762815f0a7645e600120ba3142f518ac4e32831c8795de61d7d7196"),
    "right_spike": ("0x1.88acde59e875ep+0", "0x1.88acde59e8762p+0",
                    "5ce4396f160a285a7d320b47bdc85ef58b0c694c3840fc9cf6996532e7c0b625"),
    "plateau": ("0x1.8e64359de60c6p+0", "0x1.8e64359de60c8p+0",
                "0515fff1239fb82e0b6ba609d051ac5165bdb7c9721809fd21ecd22534a6f534"),
    "demo": ("0x1.d4686369efce8p+1", "0x1.d4686369efcecp+1",
             "eabeec4bab0ff9737969e6d4128a00786d31b93e68405cd46541129fe0c813b7"),
    "shoulder": ("0x1.f2428db07725cp+0", "0x1.f2428db077259p+0",
                 "8e1bee09230b76f8899ca086acee4c9fb32e6e6e3be0049b536e1fe88ac215ca"),
    "endpoint_max": ("0x1.c672b32671427p+0", "0x1.c672b32671427p+0",
                     "5d5476821474a36c0c6e935d8c61e3175bc4b3b9c0d33cb242bd9bbd69bba721"),
    "flat": ("0x1.28a274d45e359p+0", "0x1.28a274d45e330p+0",
             "2b49de7fff3922023f45ecef9276fb5ab0e32fcff64d36c14b73a0b9d570f994"),
    "review_shoulder": ("0x1.204bd0187a4c2p+3", "0x1.204bd0187a4c2p+3",
                        "894cd2268c0294fa1082b21b4cbae79b9c08e6ac4e795a50f60e8f6966176f99"),
    "review_negative": ("0x1.fbfbf7ebc7566p+1", "0x1.fbfbf7ebc7558p+1",
                        "2d3a90ccfedd8d1793bde4468eca1fec19c4f2f0964ad62e9a673ba82744cd43"),
}

# 合成殘差：主位曲線＝一條斜線（擬合直線吃掉它）加一個形狀；每支喇叭各一條。
_SPIKE_CASES = ("interior_spike", "right_spike")
# 內部最大值是嚴格局部極大（平台算它的左端點）：偵測器的真大峰等於它。
_STRICT_CASES = (*_SPIKE_CASES, "plateau", "demo")
# 肩膀：端點 ＋12、端點旁 ＋9（內部最大、但右鄰更高，不是局部極大）、中段一個小峰（偵測器讀到的那一個）。
_DETECTOR_CASES = (*_STRICT_CASES, "shoulder")
_CASES = (*_DETECTOR_CASES, "endpoint_max", "flat")
# 審查的兩個 9 點反例（主對話算的數）：計分範圍兩端各一點、對數等距，格子權重就是梯形（端點半格），
# 殘差逐點等於下面的形狀。兩支喇叭、七個座位同一條曲線。
_REVIEW_PATTERNS: dict[str, tuple[float, ...]] = {
    "review_shoulder": (12.0, 9.0, -32.0 / 3.0, 1.0, -32.0 / 3.0, 1.0, -32.0 / 3.0, 9.0, 12.0),
    "review_negative": (12.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 12.0),
}
# 肩膀那一條的嚴格局部極大只有兩個 1 dB（第 3、5 點）：主對話用偵測器量的真大峰。
_REVIEW_SHOULDER_DETECTOR_DB = 1.0
_ALL_CASES = (*_CASES, *_REVIEW_PATTERNS)
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
    if name == "shoulder":
        last = int(np.flatnonzero(_inside())[-1])
        return ((base + _bump([last], 12.0) + _bump([last - 1], 9.0) + _bump([middle], 3.0), base),
                (base + _bump([quarter], 1.0), base))
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


def _review_axis() -> FloatArray:
    lower, upper = registry_objective().stage_one.scored_range_hz
    axis = lower * (upper / lower) ** (np.arange(9, dtype=np.float64) / 8.0)
    axis[0], axis[-1] = lower, upper
    return axis


def _review_energies(name: str) -> FloatArray:
    level = 10.0 ** ((80.0 + np.asarray(_REVIEW_PATTERNS[name])) / 10.0)
    return np.asarray([[level] * len(_receivers().points)] * 2, dtype=np.float64)


def _case_problem(name: str) -> Record:
    review = name in _REVIEW_PATTERNS
    axis, energies = (_review_axis(), _review_energies(name)) if review else (AXIS_HZ, _energies(name))
    return {"frequencies_hz": [float(value).hex() for value in axis],
            "energies": [[[float(value).hex() for value in row] for row in channel] for channel in energies]}


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


def _run_probe(directory: Path) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    temperature = registry_objective().temperature_db
    problem = {
        "purpose": PURPOSE, "receiver_set": _receivers().model_dump(mode="json"),
        "temperatures": [(temperature * 2.0 ** -k).hex() for k in range(_HALVINGS)],
        "registries": {"real": str(config_path("quality_targets.toml")), "zero": str(_zero_weight_registry(directory))},
        "cases": {name: _case_problem(name) for name in _ALL_CASES},
    }
    problem_path, output_path = directory / "problem.json", directory / "output.json"
    problem_path.write_text(json.dumps(problem), encoding="utf-8")
    env = child_process_env(threads=1)
    configure_jax(platforms="cpu", enable_x64=True, env=env)
    env["PYTHONDONTWRITEBYTECODE"], env["TMPDIR"] = "1", str(directory)
    process = subprocess.run([sys.executable, str(_PROBE), str(problem_path), str(output_path)],
                             cwd=directory, env=env, capture_output=True, text=True, timeout=300)
    assert process.returncode == 0, process.stdout + process.stderr
    return output_path.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def probe(tmp_path_factory: pytest.TempPathFactory, worker_id: str) -> Record:
    """一場 pytest 只跑一次子行程，平行的工人共用（照 ``_scheme_cache.shared_json``）；``-n 0`` 用自己的暫存目錄。"""
    directory = (tmp_path_factory.mktemp("gradient-objective-jax") if worker_id == "master"
                 else tmp_path_factory.getbasetemp().parent / "gradient-objective-jax-shared")
    return cast(Record, json.loads(shared_json(tmp_path_factory, worker_id, "gradient-objective-jax",
                                               lambda: _run_probe(directory))))


def _case(probe: Record, name: str, registry: str = "real") -> Record:
    return cast(Record, cast(Record, cast(Record, probe[registry])["cases"])[name])


def _field(probe: Record, name: str, kind: str, field: str, registry: str = "real") -> FloatArray:
    return _decode(cast(Record, _case(probe, name, registry)[kind])[field])


def _scalar(probe: Record, name: str, kind: str, field: str, registry: str = "real") -> float:
    return float(_field(probe, name, kind, field, registry)[0])


def _digest(values: object) -> str:
    return hashlib.sha256(",".join(cast(list[str], values)).encode()).hexdigest()


def _detector_peak(evaluation: StageOneLowFrequencyEvaluation) -> float:
    """終點 ``worst_big_peak`` 的取法：評估器 ``features[strongest_peak_index].depth_db`` 兩支取較大。"""
    depths = []
    for item in evaluation.speakers:
        payload = item.payload
        assert payload is not None and payload.strongest_peak_index is not None
        depths.append(payload.features[payload.strongest_peak_index].depth_db)
    return max(depths)


# ---- 1. 夾擠 ----

@pytest.mark.parametrize("name", _CASES)
def test_proxy_is_sandwiched_by_interior_max(probe: Record, name: str) -> None:
    """抓代理不是軟最大值（漏了溫度、少乘 T、加了頻率權重、只取一支喇叭）：max r_內部 ≤ P ≤ max r_內部 ＋ T·ln N，
    N＝兩支喇叭的內部點數；r 由考卷從評估器交回的直線自己扣。上界數字等於考卷自己算的 T·ln N；目標交回的內部最大
    殘差（終點報告那一格）等於考卷自己取的 max r_內部。"""
    energies = _energies(name)
    residuals = _residuals(energies, _evaluate(energies))
    top = float(np.max(residuals[:, 1:-1]))
    bound = registry_objective().temperature_db * math.log(_interior_count())
    assert float.fromhex(cast(str, _case(probe, name)["bound"])) == bound
    for kind in ("eager", "jit"):
        proxy = _scalar(probe, name, kind, "big_peak_proxy_db")
        assert top - _ruler(top) <= proxy <= top + bound + _ruler(top), (kind, proxy, top, bound)
        assert abs(_scalar(probe, name, kind, "interior_max_residual_db") - top) <= _ruler(top), kind
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
def test_detector_peak_never_exceeds_proxy(probe: Record, name: str) -> None:
    """抓代理比終點那支偵測器量到的真大峰還低（夾錯對象）：真大峰 ≤ max r_內部 ≤ P 恆成立，含肩膀那一組
    （內部最大值不是局部極大、偵測器讀到的是較低的那一個峰）。這一題只斷言真大峰 ≤ P，不斷言差多少。"""
    energies = _energies(name)
    evaluation = _evaluate(energies)
    peak = _detector_peak(evaluation)
    top = float(np.max(_residuals(energies, evaluation)[:, 1:-1]))
    assert peak <= top + _ruler(top)
    for kind in ("eager", "jit"):
        assert peak <= _scalar(probe, name, kind, "big_peak_proxy_db") + _ruler(peak), kind


@pytest.mark.parametrize("name", _STRICT_CASES)
def test_proxy_minus_detector_peak_is_within_bound_when_interior_max_is_strict(probe: Record, name: str) -> None:
    """抓代理跟偵測器量的不是同一個量：內部最大值是嚴格局部極大時（先斷言這個條件：偵測器的真大峰等於
    max r_內部），0 ≤ P − 真大峰 ≤ T·ln N。"""
    energies = _energies(name)
    evaluation = _evaluate(energies)
    peak = _detector_peak(evaluation)
    top = float(np.max(_residuals(energies, evaluation)[:, 1:-1]))
    assert abs(peak - top) <= _ruler(top), "條件不成立：這一組的內部最大值不是嚴格局部極大"
    bound = registry_objective().temperature_db * math.log(_interior_count())
    gap = _scalar(probe, name, "jit", "big_peak_proxy_db") - peak
    assert -_ruler(peak) <= gap <= bound + _ruler(peak), (gap, bound)


def test_shoulder_detector_peak_is_far_below_proxy(probe: Record) -> None:
    """審查的肩膀反例在 48 點搜尋軸上（偵測器只吃宣告軸）：端點旁那一點是內部最大、但右鄰（端點）更高，
    偵測器不報它、讀的是中段那個小峰；代理仍夾著內部最大值，所以 P − 真大峰 超過上界——
    「P − 真大峰 ∈ [0, 上界]」不是恆等式。"""
    energies = _energies("shoulder")
    evaluation = _evaluate(energies)
    residuals = _residuals(energies, evaluation)
    top = float(np.max(residuals[:, 1:-1]))
    position = np.unravel_index(int(np.argmax(residuals[:, 1:-1])), residuals[:, 1:-1].shape)
    assert position[1] == residuals.shape[1] - 3 and residuals[position[0], -1] > top, "題目要讓最大值落在端點旁"
    bound = registry_objective().temperature_db * math.log(_interior_count())
    proxy = _scalar(probe, "shoulder", "jit", "big_peak_proxy_db")
    assert top - _ruler(top) <= proxy
    assert proxy - _detector_peak(evaluation) > bound + _ruler(top)


def _review_bound() -> float:
    return registry_objective().temperature_db * math.log(2 * (len(_review_axis()) - 2))


def test_review_shoulder_counterexample(probe: Record) -> None:
    """主對話的 9 點肩膀：殘差逐點等於 [12, 9, −32/3, 1, −32/3, 1, −32/3, 9, 12]，內部最大 9、代理 ＝ 考卷自己算的
    T·logsumexp（≈ 9.3466）；嚴格局部極大只有 1 dB，P − 1 遠超過上界 T·ln N，P − 內部最大 在 [0, 上界] 裡。"""
    pattern = np.asarray(_REVIEW_PATTERNS["review_shoulder"])
    residual = _decode(_case(probe, "review_shoulder")["residual"]).reshape(2, -1)
    assert np.allclose(residual, pattern, rtol=0.0, atol=_ruler(12.0))
    temperature, bound = registry_objective().temperature_db, _review_bound()
    expected = _numpy_proxy(np.asarray([pattern, pattern]), temperature)
    for kind in ("eager", "jit"):
        proxy = _scalar(probe, "review_shoulder", kind, "big_peak_proxy_db")
        assert abs(_scalar(probe, "review_shoulder", kind, "interior_max_residual_db") - 9.0) <= _ruler(9.0), kind
        assert abs(proxy - expected) <= _ruler(expected), (kind, proxy, expected)
        assert 0.0 <= proxy - 9.0 <= bound + _ruler(9.0), kind
        assert proxy - _REVIEW_SHOULDER_DETECTOR_DB > bound, kind


def test_negative_proxy_is_not_clipped(probe: Record) -> None:
    """抓代理被截在 0（或新合分被假設不小於舊合分）：主對話的 9 點題 [12, 0, …, 0, 12]，擬合後內部全是 −1.5 dB，
    代理 ＝ −1.5 ＋ T·ln N（≈ −0.8402）是負的；新合分 ＝ 舊合分 ＋ w·P／ref 比舊合分小；值與梯度都有限。"""
    residual = _decode(_case(probe, "review_negative")["residual"]).reshape(2, -1)
    assert np.allclose(residual[:, 1:-1], -1.5, rtol=0.0, atol=_ruler(1.5))
    settings = registry_objective()
    expected = -1.5 + _review_bound()
    for kind in ("eager", "jit"):
        proxy = _scalar(probe, "review_negative", kind, "big_peak_proxy_db")
        screening = _scalar(probe, "review_negative", kind, "screening_total")
        total = _scalar(probe, "review_negative", kind, "objective_total")
        assert proxy < 0.0 and abs(proxy - expected) <= _ruler(expected), (kind, proxy, expected)
        assert total < screening, kind
        objective = screening + settings.proxy_weight * proxy / settings.worse_reference
        assert abs(total - objective) <= _ruler(objective), kind
    for key in ("gradient", "proxy_gradient"):
        gradient = _decode(_case(probe, "review_negative")[key])
        assert np.all(np.isfinite(gradient)) and np.any(gradient != 0.0), key


def test_report_lines_flag_mismatch_without_judging() -> None:
    """抓報告的判法接反或比錯對象：兩行——第一行「代理 − 內部最大殘差」在 0～上界（各放寬地板）裡不印
    「代理與內部最大殘差對不上」，代理比內部最大低、或高過上界才印；第二行是真大峰（偵測器），真大峰低於內部最大
    （肩膀）不算對不上。只是多印一句、不拋錯。加代理之前的終點只印一行註記。"""
    bound = registry_objective().temperature_db * math.log(_interior_count())
    floor = 1e-9
    inside = proxy_lines("終點", 10.0 + bound / 2.0, 10.0, 10.0, bound, floor)
    assert len(inside) == 2 and PROXY_MISMATCH not in "".join(inside)
    assert "代理 − 內部最大殘差" in inside[0] and "真大峰" in inside[1]
    assert PROXY_MISMATCH not in "".join(proxy_lines("終點", 10.0 + bound / 2.0, 10.0, 1.0, bound, floor))
    assert PROXY_MISMATCH in proxy_lines("終點", 10.0 - 1.0, 10.0, 10.0, bound, floor)[0]
    assert PROXY_MISMATCH in proxy_lines("終點", 10.0 + bound + 1.0, 10.0, 10.0, bound, floor)[0]
    assert PROXY_MISMATCH not in "".join(proxy_lines("終點", -0.84, -1.5, None, bound, floor))
    assert proxy_lines("終點", None, None, 3.0, None, floor) == [f"  終點代理：{LEGACY_ENDPOINT_NOTE}"]


# ---- 3. 權重 0 ----

@pytest.mark.parametrize("name", _ALL_CASES)
def test_zero_weight_objective_and_gradient_equal_screening_bitwise(probe: Record, name: str) -> None:
    """抓權重沒從登記簿讀、或權重 0 時代理項仍漏進目標：登記簿暫存副本把權重改 0，新合分逐位等於舊合分
    （直接與編譯後），新合分的梯度逐位等於舊合分的梯度；而且都逐位等於主線（加代理之前）釘下的答案
    （``_MAINLINE_SCREENING``）——同一支程式自己比自己抓不到第一階段那把尺被一起改掉。指紋跟正式那份不同
    （只改這一格就不能接續），第一階段設定指紋相同。"""
    zero = cast(Record, probe["zero"])
    assert float.fromhex(cast(str, zero["weight"])) == 0.0
    assert zero["fingerprint"] != cast(Record, probe["real"])["fingerprint"]
    assert zero["stage_one_fingerprint"] == cast(Record, probe["real"])["stage_one_fingerprint"]
    case = _case(probe, name, "zero")
    for kind in ("eager", "jit"):
        arrays = cast(Record, case[kind])
        assert arrays["objective_total"] == arrays["screening_total"], kind
    assert case["gradient"] == case["screening_gradient"]
    eager, jit, gradient = _MAINLINE_SCREENING[name]
    assert cast(Record, case["eager"])["objective_total"] == [eager]
    assert cast(Record, case["jit"])["objective_total"] == [jit]
    assert _digest(case["gradient"]) == gradient


@pytest.mark.parametrize("name", _ALL_CASES)
def test_screening_total_and_gradient_equal_mainline_pins(probe: Record, name: str) -> None:
    """抓加代理時第一階段那把尺被動到：正式登記簿與權重 0 的副本，舊合分（直接、編譯後）與舊合分的梯度都逐位
    等於主線 a1be64b5 釘下的答案。"""
    eager, jit, gradient = _MAINLINE_SCREENING[name]
    for registry in ("real", "zero"):
        case = _case(probe, name, registry)
        assert cast(Record, case["eager"])["screening_total"] == [eager], registry
        assert cast(Record, case["jit"])["screening_total"] == [jit], registry
        assert _digest(case["screening_gradient"]) == gradient, registry


@pytest.mark.parametrize("name", _ALL_CASES)
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
    header.pop("score_constants")
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
    ledger.write_ledger(path, header, (ledger.snap_row(snapped, 3.5, 3.0, (2.0, 4.0, 1.0), 0.0, 0),))
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
    start = templates.TemplateStart(surfaces, (), 4.25, 1.25, (2.0, 4.0, 0.5))
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


# ---- 8. 帳本的分數恆等式（審查第 4 條）----

def _mutations(row: ledger.ScoredRow) -> dict[str, tuple[object, str]]:
    """審查的四種記憶體植錯（每一種一個欄位、拒收訊息要點名的字）：值、舊合分、代理改 999、分項少一格。"""
    value, screening, components = row.scores()
    return {
        "value": ({"value": value + 1.0}, "value"),
        "screening_total": ({"screening_total": screening + 1.0}, "screening_total"),
        "proxy": ({"components": (components[0], components[1], 999.0)}, "代理"),
        "length": ({"components": components[:2]}, "第 0 列"),
    }


@pytest.mark.parametrize("mutation", ["value", "screening_total", "proxy", "length"])
def test_material_ledger_rejects_broken_score_identity(tmp_path: Path, mutation: str) -> None:
    """抓帳本收下對不上分數常數的列：v5 標頭帶分數常數，逐列守 value ＝ screening_total ＋ w_P·P／ref_P 與
    screening_total ＝ Σ w_i·c_i／ref_i（容差是那一格幾個末位，跟 ``_gradient_objective_rows`` 同一個數）；
    植錯後 ``check_rows`` 拒收、訊息點名第幾列與哪一格。控制組：原列照收。"""
    path, header = _material_ledger(tmp_path)
    _, rows = ledger.read_ledger(path, expected_header=header)
    ledger.check_rows(header, rows)
    update, named = _mutations(rows[0])[mutation]
    with pytest.raises(ValueError, match=named):
        ledger.check_rows(header, [rows[0].model_copy(update=cast(dict[str, object], update))])


def _template_ledger(directory: Path) -> tuple[Path, templates.TemplateHeader]:
    """沒有可換目標的一輪（不求值）寫一本 v2 換樣板帳本；起點的分數照合成分數常數對得上兩個恆等式。"""
    settings = load_material_gradient_settings(config_path("material_gradient.toml")).model_copy(update={"family": ()})
    surfaces = tuple((wall, None) for wall in gradient_search.wall_surface_ids())
    start = templates.TemplateStart(surfaces, (), 4.25, 1.25, (2.0, 4.0, 0.5))
    header = template_header(settings, surfaces, kept_walls(KEPT))
    path = directory / "templates.jsonl"
    segment.run_template_rounds(_context(settings), path, header, start)
    return path, header


@pytest.mark.parametrize("mutation", ["value", "screening_total", "proxy", "length"])
def test_template_ledger_rejects_broken_score_identity(tmp_path: Path, mutation: str) -> None:
    """同上一題，換樣板帳本（v2）：植錯後 ``check_template_rows`` 拒收、訊息點名第幾列與哪一格。"""
    path, header = _template_ledger(tmp_path)
    _, rows = templates.read_template_ledger(path, expected_header=header)
    templates.check_template_rows(header, rows)
    update, named = _mutations(rows[0])[mutation]
    broken = [rows[0].model_copy(update=cast(dict[str, object], update)), *rows[1:]]
    with pytest.raises(ValueError, match=named):
        templates.check_template_rows(header, broken)


def test_score_constants_are_copied_from_objective_and_pinned_in_header(tmp_path: Path) -> None:
    """抓標頭的分數常數不是這一次目標用的那一組、或接續不比它：從目標常數抄出來的五格等於登記簿那幾格
    （兩分項權重與較差參考、代理權重、較差參考、溫度）；預期標頭的分數常數改一格，兩本帳本接續都拒收；
    上一版標頭帶分數常數、或這一版不帶，都拒收。"""
    settings = registry_objective()
    constants = ledger.score_constants(gradient_objective_constants(AXIS_HZ, settings, _receivers(), 2))
    stage_one = settings.stage_one
    assert constants.component_weights == (stage_one.weight(PRINCIPAL), stage_one.weight(ROBUSTNESS))
    assert constants.worse_references == (stage_one.target(PRINCIPAL).worse_reference,
                                          stage_one.target(ROBUSTNESS).worse_reference)
    assert (constants.proxy_weight, constants.proxy_worse_reference, constants.proxy_temperature_db) == (
        settings.proxy_weight, settings.worse_reference, settings.temperature_db)
    path, header = _material_ledger(tmp_path)
    assert header.score_constants is not None
    changed = header.score_constants.model_copy(update={"proxy_weight": header.score_constants.proxy_weight * 2.0})
    with pytest.raises(ValueError, match="score_constants"):
        ledger.read_ledger(path, expected_header=header.model_copy(update={"score_constants": changed}))
    with pytest.raises(ValueError):
        ledger.MaterialGradientHeader.model_validate(header.model_dump() | {"score_constants": None})
    with pytest.raises(ValueError):
        ledger.MaterialGradientHeader.model_validate(
            header.model_dump() | {"ledger_version": ledger.LEGACY_LEDGER_VERSION})
    template_path, template = _template_ledger(tmp_path)
    assert template.score_constants is not None
    other = template.score_constants.model_copy(update={"proxy_temperature_db": template.score_constants
                                                        .proxy_temperature_db * 2.0})
    with pytest.raises(ValueError, match="score_constants"):
        templates.read_template_ledger(template_path, expected_header=template.model_copy(
            update={"score_constants": other}))
    with pytest.raises(ValueError):
        templates.TemplateHeader.model_validate(template.model_dump() | {"score_constants": None})
