"""低頻可微目標對現有 numpy 評估與合分；子行程只求值，父行程比對、差分與判決。"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from dataclasses import replace
from functools import cache
from pathlib import Path
from typing import TypedDict, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.typing import NDArray

from aosr.config.paths import config_path
from aosr.config.quality_targets import load_quality_targets
from aosr.runtime import child_process_env, configure_jax
from aosr.scoring.receiver_set import ReceiverRole, ReceiverSet, primary_cross
from aosr.scoring.stage_one_low_frequency import evaluate_stage_one_low_frequency
from aosr.scoring.stage_one_low_frequency_contract import StageOneLowFrequencyInput
from aosr.scoring.stage_one_score import (
    PRINCIPAL, ROBUSTNESS, StageOneScoreSettings, score_stage_one, stage_one_score_settings,
)
from tests.engine._precision_contracts import MUTANT_MARGIN, contract_value
from tests.engine._stage_one_low_frequency_curves import (
    AXIS_HZ, DIRECTIONS, candidate, demo_curve_db, gaussian_db, speaker, tilted_db,
)
from tests.engine.test_transfer_matrix_reference import GRADIENT_ABS, GRADIENT_REL


FloatArray = NDArray[np.float64]
_PROBE = Path(__file__).with_name("_stage_one_objective_jax_probe.py")
_BASE_CASES = ("demo", "tilted", "gaussian", "random_561", "random_562")
# 七座位交進完全相同的曲線、而且不在單位能量（全 7、全用 demo 主位那條）：修前編譯後相同曲線相減
# 剩約 7e-16 的捨入殘差，在近零均方根上放大成 O(1) 的假梯度；座位穩健性的真值恰 0。
_IDENTICAL_CASES = ("identical_seven", "identical_demo")
# 能量約 1（dB 約 0，地板取下限 1 dB，雙精度約 1.4e-14 dB）時第 k 個周圍點差 k·1e-6 dB：
# 比地板高八個數量級，值與梯度都不准被歸零。不用 demo 的 80 dB：那裡 dB 的末位約 1e-14，
# 對 1e-6 的差就是 1e-8 的相對差，量的是 log10 兩種寫法的末位，不是地板。
_NUDGED = "nudged_unit"
_NUDGE_DB = 1e-6
_CASES = (*_BASE_CASES, "reordered", "clipped", "settings", *_IDENTICAL_CASES, _NUDGED)
_QUANTITIES = (PRINCIPAL, ROBUSTNESS, "mean_" + PRINCIPAL, "mean_" + ROBUSTNESS, "screening_total")
# 中央差分以每一格能量本身為尺度，h/E=1e-5；比 E·∂total/∂E，避免大能量掩蓋錯梯度。
_DIFFERENCE_STEP = 1e-5
# 七座位同乘 exp(s)，起伏與座位差都不變，對 s 的梯度數學上恰 0，每一組輸入都成立。雙精度上限 1e-9：
# 修後全部十一組只剩擬合與加總的捨入（本機實量最大約 1.1e-14），上限離它四個數量級以上；
# 修前座位相同的兩組編譯後是 −4.489、−0.987，離上限九個數量級。單精度上限 2^-14（約 6.1e-5）：
# 正常輸入本機實量最大約 2.0e-6（約 17 個 eps），上限留三十倍給換機器；單精度修前最大只到 −1.38e-6、
# 沒有 O(1) 的錯，這條只擋單精度炸開，單精度的修補由「座位穩健性恰 0」那題抓。
_GAIN_GRADIENT_BOUND = 1e-9
_SINGLE_GAIN_GRADIENT_BOUND = 2.0 ** -14
# 沿單一周圍點的增益 exp(t) 走：座位差逐頻同加 10/ln10·t dB，h=1e-7 只推 4.3e-7 dB、
# 小於最小的 1e-6 dB 差，均方根在這一段是直線，中央差分沒有曲率誤差。
_SEAT_GAIN_STEP = 1e-7


class CaseRun(TypedDict):
    eager: dict[str, list[str]]
    jit: dict[str, list[str]]
    jit_again: dict[str, list[str]]
    single: dict[str, list[str]]
    single_eager: dict[str, list[str]]
    gain_gradient: list[str]
    single_gain_gradient: list[str]
    gradient: list[str]
    single_gradient: list[str]
    double_dtypes: list[str]
    single_dtypes: list[str]
    pressure: dict[str, list[str]]
    pressure_energy: dict[str, list[str]]
    single_pressure: dict[str, list[str]]
    single_pressure_energy: dict[str, list[str]]
    vmap: dict[str, list[str]]
    loop: list[dict[str, list[str]]]


@cache
def _settings(name: str = "demo") -> StageOneScoreSettings:
    base = stage_one_score_settings(load_quality_targets(config_path("quality_targets.toml")),
                                    "dedicated_two_channel_listening_room")
    if name == "settings":
        # 故意反轉具名列序、改權重與非 1 參考：抓硬抄正式數字與照列序取錯分項。
        return replace(base, targets=tuple((key, entry.model_copy(update={"worse_reference": 2.5}))
                                           for key, entry in reversed(base.targets)),
                       weights=((ROBUSTNESS, 0.7), (PRINCIPAL, 1.3)))
    if name == "clipped":
        return replace(base, scored_range_hz=(40.3, 199.2))
    return base


@cache
def _receivers(name: str = "demo") -> ReceiverSet:
    base = primary_cross(primary_position_m=(3.0, 2.0, 1.2), offset_m=0.1)
    if name == "reordered":
        return ReceiverSet(points=tuple(base.points[index] for index in (4, 2, 0, 6, 1, 5, 3)))
    return base


def _surrounding(name: str) -> list[int]:
    return [index for index, point in enumerate(_receivers(name).points) if point.role == ReceiverRole.SURROUNDING]


def _energies(name: str) -> FloatArray:
    shape = (2, len(DIRECTIONS) + 1, len(AXIS_HZ))
    if name in _IDENTICAL_CASES:
        curve = np.full(len(AXIS_HZ), 7.0) if name == "identical_seven" else 10.0 ** (demo_curve_db() / 10.0)
        return np.asarray(np.broadcast_to(curve, shape), dtype=np.float64).copy()
    if name == _NUDGED:
        nudged = np.ones(shape, dtype=np.float64)
        for rank, index in enumerate(_surrounding(name)):
            nudged[:, index, :] = 10.0 ** ((rank + 1) * _NUDGE_DB / 10.0)
        return nudged
    if name.startswith("random_"):
        rng = np.random.default_rng(int(name.removeprefix("random_")))
        return np.asarray(np.exp(rng.normal(1.0, 1.2, (2, len(DIRECTIONS) + 1, len(AXIS_HZ)))), dtype=np.float64)
    primary = (tilted_db(AXIS_HZ, -2.0, 70.0) if name == "tilted" else
               gaussian_db(AXIS_HZ, 75.0, 8.0, 0.2) if name == "gaussian" else demo_curve_db())
    channels = []
    for channel in range(2):
        main = primary + channel * (1.5 if name == "tilted" else gaussian_db(AXIS_HZ, 110.0, -5.0, 0.13))
        levels = {"main": main}
        for index, direction in enumerate(DIRECTIONS):
            levels[direction] = main + (index + 1) * 0.3 + gaussian_db(AXIS_HZ, 40.0 + index * 20.0,
                                                                     (-1.0) ** index * 2.0, 0.12)
        channels.append([10.0 ** (levels[point.receiver_id] / 10.0) for point in _receivers(name).points])
    return np.asarray(channels, dtype=np.float64)


def _gradient_energies(name: str) -> FloatArray:
    values = _energies(name)
    if name == "tilted":
        # 純直線的 RMS 在零點無唯一導數；只在梯度題給主位加 0.5 dB 峰，前向題仍是純直線。
        primary = next(index for index, point in enumerate(_receivers(name).points)
                       if point.role == ReceiverRole.PRIMARY)
        values[:, primary, :] *= 10.0 ** (gaussian_db(AXIS_HZ, 80.0, 0.5, 0.2) / 10.0)
    return values


def _numpy_input(name: str, energies: FloatArray) -> StageOneLowFrequencyInput:
    receivers = _receivers(name)
    primary = next(index for index, point in enumerate(receivers.points) if point.role == ReceiverRole.PRIMARY)
    template = speaker(demo_curve_db())
    reference = []
    for channel, values in enumerate(energies):
        rows = {point.receiver_id: tuple(float(value) for value in values[index])
                for index, point in enumerate(receivers.points)}
        reference.append(template.model_copy(update={
            "speaker_id": f"channel-{channel}",
            "primary": template.primary.model_copy(update={"fem_energy": rows[receivers.points[primary].receiver_id]}),
            "surrounding": tuple(seat.model_copy(update={"fem_energy": rows[seat.receiver_id]})
                                 for seat in template.surrounding),
        }))
    return candidate(*reference)


def _numpy_arrays(name: str, energies: FloatArray | None = None) -> dict[str, FloatArray]:
    settings = _settings(name)
    data = _numpy_input(name, _energies(name) if energies is None else energies)
    evaluation = evaluate_stage_one_low_frequency(data, scoring_range_hz=settings.scored_range_hz)
    score = score_stage_one(evaluation, settings)
    assert score.state == "scored", score.reason_codes
    payloads = [item.payload for item in evaluation.speakers]
    assert all(payload is not None for payload in payloads)
    output: dict[str, FloatArray] = {key: np.asarray([getattr(payload, key) for payload in payloads], dtype=np.float64)
                                     for key in (PRINCIPAL, ROBUSTNESS)}
    output.update({"mean_" + item.name: np.asarray([item.value_db]) for item in score.components})
    output["screening_total"] = np.asarray([score.screening_total], dtype=np.float64)
    return output


def _decode(values: list[str]) -> FloatArray:
    return np.asarray([float.fromhex(value) for value in values], dtype=np.float64)


def _contract_fractions(actual: FloatArray, expected: FloatArray, tolerance: float, *, line_zero: bool = False
                        ) -> FloatArray:
    """非零量用相對差；純直線主項以 1 dB 為尺度；真值恰零只接受恰零。"""
    scale = np.ones_like(expected) if line_zero else np.abs(expected)
    difference = np.abs(actual - expected)
    nonzero = scale != 0.0
    fractions = np.full_like(difference, np.inf)
    np.divide(difference, tolerance * scale, out=fractions, where=nonzero)
    return np.where(~nonzero & (difference == 0.0), 0.0, fractions)


def _within_contract(actual: FloatArray, expected: FloatArray, tolerance: float, *, line_zero: bool = False) -> bool:
    return bool(np.all(np.isfinite(actual)) and
                np.all(_contract_fractions(actual, expected, tolerance, line_zero=line_zero) <= 1.0))


def _run_probe(directory: Path) -> dict[str, CaseRun]:
    """序列化原始題目，不把 numpy 答案送入子行程；十六進位往返不丟位元。"""
    problem = {}
    for name in _CASES:
        settings = _settings(name)
        problem[name] = {
            "energies": [[[float(value).hex() for value in row] for row in channel] for channel in _energies(name)],
            "gradient_energies": [[[float(value).hex() for value in row] for row in channel]
                                  for channel in _gradient_energies(name)],
            "frequencies_hz": [float(value).hex() for value in AXIS_HZ],
            "receiver_set": _receivers(name).model_dump(mode="json"),
            "scored_range_hz": [float(value).hex() for value in settings.scored_range_hz],
            "weights": {key: float(value).hex() for key, value in settings.weights},
            "worse_references": {key: float(entry.worse_reference).hex() for key, entry in settings.targets},
        }
    problem_path, output_path = directory / "problem.json", directory / "output.json"
    problem_path.write_text(json.dumps(problem), encoding="utf-8")
    env = child_process_env(threads=1)
    configure_jax(platforms="cpu", enable_x64=True, env=env)
    env["PYTHONDONTWRITEBYTECODE"], env["TMPDIR"] = "1", str(directory)
    process = subprocess.run([sys.executable, str(_PROBE), str(problem_path), str(output_path)],
                             cwd=directory, env=env, capture_output=True, text=True, timeout=180)
    assert process.returncode == 0, process.stdout + process.stderr
    return cast(dict[str, CaseRun], json.loads(output_path.read_text(encoding="utf-8")))


@pytest.fixture(scope="module")
def jax_runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, CaseRun]:
    return _run_probe(tmp_path_factory.mktemp("stage-one-objective-jax"))


def test_identical_flat_curves_have_zero_objective() -> None:
    """抓漏扣主位直線、把共同音量當代價、或把完全相同的座位當成有差。"""
    from aosr.scoring.stage_one_objective_jax import objective_constants, objective_from_energies

    settings = stage_one_score_settings(load_quality_targets(config_path("quality_targets.toml")),
                                        "dedicated_two_channel_listening_room")
    receivers = primary_cross(primary_position_m=(3.0, 2.0, 1.2), offset_m=0.1)
    constants = objective_constants(AXIS_HZ, settings, receivers, speaker_count=2)
    result = objective_from_energies(constants, jnp.ones((2, len(receivers.points), len(AXIS_HZ)), dtype=jnp.float32))
    assert np.array_equal(np.asarray(result.own_line_ripple_rms_db), np.zeros(2))
    assert np.array_equal(np.asarray(result.seat_robustness_rms_db), np.zeros(2))
    assert float(result.screening_total) == 0.0


@pytest.mark.parametrize("name", _CASES)
def test_double_precision_matches_python(jax_runs: dict[str, CaseRun], name: str) -> None:
    """逐聲道兩項、兩項平均與合分：抓扣線、加權、座位角色與設定讀法錯誤。"""
    tolerance = contract_value("stage_one_objective_jax_vs_python")
    expected = _numpy_arrays(name)
    for mode in ("eager", "jit"):
        arrays = jax_runs[name][mode]
        assert isinstance(arrays, dict)
        for quantity in _QUANTITIES:
            actual = _decode(arrays[quantity])
            assert actual.shape == expected[quantity].shape
            assert _within_contract(actual, expected[quantity], tolerance,
                                    line_zero=name == "tilted" and PRINCIPAL in quantity), (name, mode, quantity)


@pytest.mark.parametrize("name", _CASES)
def test_jit_repeats_bitwise_and_eager_matches_contract(jax_runs: dict[str, CaseRun], name: str) -> None:
    """同一份編譯逐位重現；編譯會改變運算合併順序，對未編譯只用登記的尺。"""
    tolerance = contract_value("stage_one_objective_jax_vs_python")
    run = jax_runs[name]
    assert run["jit"] == run["jit_again"]
    for quantity in _QUANTITIES:
        assert _within_contract(_decode(run["jit"][quantity]), _decode(run["eager"][quantity]), tolerance,
                                line_zero=name == "tilted" and PRINCIPAL in quantity), (name, quantity)


def test_pure_line_ripple_is_near_zero_in_both_implementations(jax_runs: dict[str, CaseRun]) -> None:
    """零值政策以 1 dB 為尺度判絕對差；也守兩份各自接近零，不只彼此相近。"""
    tolerance = contract_value("stage_one_objective_jax_vs_python")
    expected = _numpy_arrays("tilted")
    for quantity in (PRINCIPAL, "mean_" + PRINCIPAL):
        assert np.all(np.abs(expected[quantity]) <= tolerance)
        for mode in ("eager", "jit"):
            arrays = jax_runs["tilted"][mode]
            assert isinstance(arrays, dict)
            assert np.all(np.abs(_decode(arrays[quantity])) <= tolerance)


@pytest.mark.parametrize("name", _CASES)
def test_common_gain_has_zero_compiled_gradient(jax_runs: dict[str, CaseRun], name: str) -> None:
    """七座位同乘 exp(s)、s=0 對 s 的編譯梯度：抓近零均方根上捨入殘差被放大的假梯度。"""
    run = jax_runs[name]
    assert abs(_decode(run["gain_gradient"])[0]) <= _GAIN_GRADIENT_BOUND, name
    assert abs(_decode(run["single_gain_gradient"])[0]) <= _SINGLE_GAIN_GRADIENT_BOUND, name


@pytest.mark.parametrize("name", _IDENTICAL_CASES)
def test_identical_seats_have_exact_zero_robustness(jax_runs: dict[str, CaseRun], name: str) -> None:
    """座位完全相同時 Python 給恰 0；編譯與不編譯、雙與單精度都要恰 0，主位起伏不受影響。"""
    tolerance = contract_value("stage_one_objective_jax_vs_python")
    run = jax_runs[name]
    assert np.array_equal(_numpy_arrays(name)[ROBUSTNESS], np.zeros(2))
    for mode in ("eager", "jit", "single", "single_eager"):
        assert np.array_equal(_decode(run[mode][ROBUSTNESS]), np.zeros(2)), (name, mode)
        assert np.array_equal(_decode(run[mode]["mean_" + ROBUSTNESS]), np.zeros(1)), (name, mode)
        assert np.all(np.isfinite(_decode(run[mode][PRINCIPAL]))), (name, mode)
    # 編譯會改變擬合直線那幾個總和的合併順序，主位起伏對不編譯只守登記的尺（實量不逐位）。
    assert _within_contract(_decode(run["jit"][PRINCIPAL]), _decode(run["eager"][PRINCIPAL]), tolerance)


def test_seat_difference_above_floor_keeps_value_and_gradient(jax_runs: dict[str, CaseRun]) -> None:
    """座位差 1e-6 dB 起跳、遠在地板之上：值對 Python 守契約，沿每個周圍點增益的梯度對差分。"""
    tolerance = contract_value("stage_one_objective_jax_vs_python")
    run = jax_runs[_NUDGED]
    expected = _numpy_arrays(_NUDGED)
    assert np.all(expected[ROBUSTNESS] > 0.0)
    for mode in ("eager", "jit"):
        actual = _decode(run[mode][ROBUSTNESS])
        assert np.all(actual > 0.0), mode
        assert _within_contract(actual, expected[ROBUSTNESS], tolerance), mode
    energies = _gradient_energies(_NUDGED)
    automatic = _decode(run["gradient"]).reshape(energies.shape) * energies
    for channel in range(energies.shape[0]):
        for index in _surrounding(_NUDGED):
            plus, minus = energies.copy(), energies.copy()
            plus[channel, index] *= math.exp(_SEAT_GAIN_STEP)
            minus[channel, index] *= math.exp(-_SEAT_GAIN_STEP)
            central = (_numpy_arrays(_NUDGED, plus)["screening_total"][0] -
                       _numpy_arrays(_NUDGED, minus)["screening_total"][0]) / (2.0 * _SEAT_GAIN_STEP)
            directional = float(np.sum(automatic[channel, index]))
            assert directional != 0.0, (channel, index)
            np.testing.assert_allclose(directional, central, rtol=GRADIENT_REL, atol=GRADIENT_ABS,
                                       err_msg=str((channel, index)))


def _central_difference(name: str) -> FloatArray:
    """每個聲道、座位、頻點分開推；每次都走既有評估器與合分，沒有另抄目標公式。"""
    energies = _gradient_energies(name)
    central = np.empty_like(energies)
    for index in np.ndindex(energies.shape):
        plus, minus = energies.copy(), energies.copy()
        step = energies[index] * _DIFFERENCE_STEP
        plus[index] += step
        minus[index] -= step
        upper = _numpy_arrays(name, plus)["screening_total"][0]
        lower = _numpy_arrays(name, minus)["screening_total"][0]
        central[index] = (upper - lower) / (2.0 * _DIFFERENCE_STEP)
    return central


@pytest.mark.parametrize("name", _BASE_CASES)
def test_energy_gradient_matches_numpy_central_difference(jax_runs: dict[str, CaseRun], name: str) -> None:
    """比無單位 E·∂total/∂E；差分用相對步長，容差只從傳遞矩陣考卷匯入。"""
    energies = _gradient_energies(name)
    automatic = _decode(jax_runs[name]["gradient"]).reshape(energies.shape) * energies
    np.testing.assert_allclose(automatic, _central_difference(name), rtol=GRADIENT_REL, atol=GRADIENT_ABS)


def test_scoring_range_excludes_energy_gradients(jax_runs: dict[str, CaseRun]) -> None:
    """範圍外頻點不參與目標，對它們的導數恰零，不能只在顯示結果時裁掉。"""
    lower, upper = _settings("clipped").scored_range_hz
    outside = (AXIS_HZ < lower) | (AXIS_HZ > upper)
    assert np.any(outside)
    for field in ("gradient", "single_gradient"):
        values = jax_runs["clipped"][field]
        assert isinstance(values, list)
        gradient = _decode(values).reshape(_energies("clipped").shape)
        assert np.array_equal(gradient[:, :, outside], np.zeros_like(gradient[:, :, outside]))


@pytest.mark.parametrize("name", _CASES)
def test_pressure_adapter_is_bitwise_energy_definition(jax_runs: dict[str, CaseRun], name: str) -> None:
    """複數聲壓有非零相位；抓漏掉模長平方、乘錯係數或把 (F,S,R) 轉軸轉錯。"""
    run = jax_runs[name]
    assert run["pressure"] == run["pressure_energy"]
    assert run["single_pressure"] == run["single_pressure_energy"]


@pytest.mark.parametrize("name", _CASES)
def test_vmap_matches_individual_cases(jax_runs: dict[str, CaseRun], name: str) -> None:
    """兩個不同頻率排列的批次對逐組求值；抓把批次軸混入聲道或頻率平均。"""
    run = jax_runs[name]
    tolerance = contract_value("stage_one_objective_jax_vs_python")
    for quantity in _QUANTITIES:
        expected = np.asarray([_decode(row[quantity]) for row in run["loop"]])
        actual = _decode(run["vmap"][quantity]).reshape(expected.shape)
        assert _within_contract(actual, expected, tolerance, line_zero=name == "tilted" and PRINCIPAL in quantity)


@pytest.mark.parametrize("name", _CASES)
def test_precision_follows_input_and_single_is_finite(jax_runs: dict[str, CaseRun], name: str) -> None:
    """單雙精度都考輸出與梯度；單精度只量誤差、不借雙精度契約。"""
    run = jax_runs[name]
    assert set(run["double_dtypes"]) == {"float64"}
    assert set(run["single_dtypes"]) == {"float32"}
    for values in run["single"].values():
        assert np.all(np.isfinite(_decode(values)))
    assert np.all(np.isfinite(_decode(run["single_gradient"])))


@pytest.mark.parametrize("shape", [(2, 7), (2, 6, len(AXIS_HZ)), (2, 8, len(AXIS_HZ)),
                                  (2, 7, len(AXIS_HZ) - 1), (1, 7, len(AXIS_HZ)), (2, 7, 0)])
def test_energy_shape_is_checked(shape: tuple[int, ...]) -> None:
    from aosr.scoring.stage_one_objective_jax import objective_constants, objective_from_energies

    constants = objective_constants(AXIS_HZ, _settings(), _receivers(), speaker_count=2)
    with pytest.raises(ValueError, match="形狀"):
        objective_from_energies(constants, jnp.ones(shape, dtype=jnp.float32))


@pytest.mark.parametrize("dtype", [jnp.int32, jnp.bool_, jnp.complex64])
def test_non_floating_energies_are_rejected(dtype: type) -> None:
    from aosr.scoring.stage_one_objective_jax import objective_constants, objective_from_energies

    constants = objective_constants(AXIS_HZ, _settings(), _receivers(), speaker_count=2)
    with pytest.raises(TypeError, match="浮點"):
        objective_from_energies(constants, jnp.ones((2, 7, len(AXIS_HZ)), dtype=dtype))


def test_pressure_shape_and_type_are_checked() -> None:
    from aosr.scoring.stage_one_objective_jax import objective_constants, objective_from_pressures

    constants = objective_constants(AXIS_HZ, _settings(), _receivers(), speaker_count=2)
    with pytest.raises(ValueError, match="形狀"):
        objective_from_pressures(constants, jnp.ones((2, 7, len(AXIS_HZ)), dtype=jnp.complex64))
    with pytest.raises(TypeError, match="複數"):
        objective_from_pressures(constants, jnp.ones((len(AXIS_HZ), 2, 7), dtype=jnp.float32))


@pytest.mark.parametrize("damage", ["missing", "other_seat", "wrong_direction"])
def test_receiver_set_must_be_one_primary_and_six_named_surroundings(damage: str) -> None:
    from aosr.scoring.stage_one_objective_jax import objective_constants

    points = list(_receivers().points)
    if damage == "missing":
        points.pop()
    else:
        change = ({"role": ReceiverRole.OTHER_SEAT} if damage == "other_seat" else
                  {"direction_relative_to_primary": "diagonal"})
        points[-1] = points[-1].model_copy(update=change)
    with pytest.raises(ValueError, match="周圍"):
        objective_constants(AXIS_HZ, _settings(), ReceiverSet(points=tuple(points)), speaker_count=2)


@pytest.mark.parametrize("frequencies", [np.asarray([20.0]), np.asarray([20.0, 20.0]),
                                        np.asarray([0.0, 20.0]), np.asarray([20.0, np.inf]),
                                        np.asarray([[20.0, 100.0]])])
def test_constant_frequency_axis_is_checked(frequencies: FloatArray) -> None:
    from aosr.scoring.stage_one_objective_jax import objective_constants

    with pytest.raises(ValueError, match="頻率|頻點"):
        objective_constants(frequencies, _settings(), _receivers(), speaker_count=2)


def test_exact_zero_norm_has_finite_zero_subgradient() -> None:
    """完全平坦、七點完全一致的最小值：選零次梯度，未選的 sqrt(0) 不准洩漏 NaN。"""
    from aosr.scoring.stage_one_objective_jax import objective_constants, objective_from_energies

    constants = objective_constants(AXIS_HZ, _settings(), _receivers(), speaker_count=2)
    values = jnp.ones((2, 7, len(AXIS_HZ)), dtype=jnp.float32)
    gradient = jax.grad(lambda energies: objective_from_energies(constants, energies).screening_total)(values)
    assert np.array_equal(np.asarray(gradient), np.zeros(values.shape))


def test_mutant_beyond_tolerance_is_red() -> None:
    """逐量界線內外各推 δ，正負兩側都守；真值恰零與純直線的尺度另考。"""
    tolerance = contract_value("stage_one_objective_jax_vs_python")
    for truth in _numpy_arrays("demo").values():
        for sign in (-1.0, 1.0):
            inside = truth * (1.0 + sign * (1.0 - MUTANT_MARGIN) * tolerance)
            outside = truth * (1.0 + sign * (1.0 + MUTANT_MARGIN) * tolerance)
            assert _within_contract(inside, truth, tolerance)
            assert not _within_contract(outside, truth, tolerance)
            assert math.isclose(float(np.max(_contract_fractions(outside, truth, tolerance))),
                                1.0 + MUTANT_MARGIN, rel_tol=MUTANT_MARGIN / 4.0)
    zero = np.zeros(2)
    assert _within_contract(zero, zero, tolerance)
    assert not _within_contract(np.full(2, np.finfo(np.float64).tiny), zero, tolerance)
    for factor, accepted in ((1.0 - MUTANT_MARGIN, True), (1.0 + MUTANT_MARGIN, False)):
        assert _within_contract(np.full(2, factor * tolerance), zero, tolerance, line_zero=True) == accepted
    assert not _within_contract(np.full(2, np.nan), np.ones(2), tolerance)
