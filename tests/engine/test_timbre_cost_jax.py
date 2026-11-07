"""三類代價對真的 numpy 評估器＋代價函式；雙精度只在 CPU 冷子行程開。"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from aosr.config.quality_targets import TargetEntry, load_quality_targets
from aosr.runtime import child_process_env, configure_jax
from aosr.scoring.cost_shapes import shape_cost
from tests.engine._precision_contracts import contract_value
from tests.engine._scheme_cache import shared_json
from tests.engine._timbre_cost_jax_cases import CATEGORIES, PURPOSE, curves, numpy_costs
from tests.engine.test_stage_one_objective_jax import _DIFFERENCE_STEP
from tests.engine.test_transfer_matrix_reference import GRADIENT_ABS, GRADIENT_REL

_CASES = ("curves", "small_room", "reordered", "multiple_pairs", "time_enabled", "zero_weight_worst", "other_seat")
_CONTRACTS = ("timbre_cost_jax_vs_python", "listening_area_cost_jax_vs_python", "channel_matching_cost_jax_vs_python")
_FLAGS = ("energy_zero", "energy_negative", "energy_nan", "energy_inf", "coverage", "gap", "matching_coverage",
          "numerical_failure", "zero_primary_importance", "zero_peer_importance", "invalid_distance", "no_comparisons",
          "invalid_sound_speed", "zero_matching_weight", "zero_matching_importance", "no_broadband_support")


def _probe(directory: Path) -> str:
    env = child_process_env(threads=1).copy()
    configure_jax(platforms="cpu", enable_x64=True, env=env)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["TMPDIR"] = str(directory)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    output = directory / "measurements.json"
    run = subprocess.run([sys.executable, str(Path(__file__).with_name("_timbre_cost_jax_probe.py")), str(output)],
                         cwd=directory, env=env, capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, run.stdout + run.stderr
    return output.read_text()


@pytest.fixture(scope="session")
def measured(tmp_path_factory: pytest.TempPathFactory, worker_id: str) -> dict[str, object]:
    directory = tmp_path_factory.mktemp("timbre-jax")
    text = shared_json(tmp_path_factory, worker_id, "timbre-jax-cold", lambda: _probe(directory))
    result = cast(dict[str, object], json.loads(text))
    assert result["x64"] is True
    return result


def _record(measured: dict[str, object], name: str) -> dict[str, object]:
    return cast(dict[str, dict[str, object]], measured["cases"])[name]


@pytest.mark.parametrize("category", range(len(CATEGORIES)), ids=CATEGORIES)
def test_category_cost_matches_numpy_evaluator_and_cost(measured: dict[str, object], category: int) -> None:
    """抓：類代價加錯診斷項、套錯形狀／範圍／權重、逐聲道及逐配對彙總錯誤。"""
    tolerance = contract_value(_CONTRACTS[category])
    for name in _CASES:
        record = _record(measured, name)
        expected = np.asarray(record["python"])[category]
        for mode in ("eager", "jit"):
            assert abs(np.asarray(record[mode])[category] - expected) <= tolerance, (name, mode, record)
        assert np.all(np.abs(np.asarray(record["batch"])[:, category] - expected) <= tolerance), (name, record)


@pytest.mark.parametrize("name", _CASES)
def test_intermediate_tilt_ripple_and_broadband_match_numpy(measured: dict[str, object], name: str) -> None:
    """抓：類代價容許帶掩蓋中間量錯誤，或左右起伏誤用音色範圍。"""
    record = _record(measured, name)
    tolerance = max(contract_value(key) for key in _CONTRACTS)
    np.testing.assert_allclose(np.asarray(record["metrics64"]), np.asarray(record["metrics_python"]), rtol=0, atol=tolerance)


@pytest.mark.parametrize("name", _CASES)
def test_dtype_follows_input_and_single_precision_is_finite(measured: dict[str, object], name: str) -> None:
    """抓：常數把單精度升成雙精度、輸入雙精度被降級、單精度出 NaN。單精度只量不登記。"""
    record = _record(measured, name)
    assert all(dtype == "float64" for dtype in cast(list[str], record["double_dtypes"]))
    assert all(dtype == "float32" for dtype in cast(list[str], record["single_dtypes"]))
    assert np.isfinite(np.asarray(record["single"])).all()
    assert np.isfinite(np.asarray(record["metrics32"])).all()
    assert cast(dict[str, object], measured["gradient"])["single_finite"] is True


def test_all_six_listening_components_match_numpy(measured: dict[str, object]) -> None:
    """抓：不進類代價的最差值算錯，含零權重最差座位及只有一位周圍點。"""
    for name in _CASES:
        record = _record(measured, name)
        np.testing.assert_allclose(np.asarray(record["listening"]), np.asarray(record["listening_python"]),
                                   rtol=0, atol=contract_value(_CONTRACTS[1]), err_msg=name)


def test_channel_matching_mean_and_worst_components_match_numpy(measured: dict[str, object]) -> None:
    """抓：四量的平均與最差分項算錯，包含幾何直達時間開／關。"""
    for name in _CASES:
        record = _record(measured, name)
        np.testing.assert_allclose(np.asarray(record["matching"]), np.asarray(record["matching_python"]),
                                   rtol=0, atol=contract_value(_CONTRACTS[2]), err_msg=name)


def test_other_seat_does_not_change_any_category_cost(measured: dict[str, object]) -> None:
    """抓：把非零 importance、左右曲線形狀不同的 OTHER_SEAT 誤納入計分。"""
    expected = np.asarray(_record(measured, "curves")["python"])
    other = _record(measured, "other_seat")
    for mode in ("python", "eager", "jit"):
        assert np.all(np.abs(np.asarray(other[mode]) - expected) <= [contract_value(key) for key in _CONTRACTS]), (mode, other)


@pytest.mark.parametrize("name", ("energy_zero", "energy_negative", "energy_nan", "energy_inf",
                                  "distance_negative", "distance_nan", "distance_inf"))
def test_invalid_other_seat_is_ignored_by_costs_flags_and_gradients(measured: dict[str, object], name: str) -> None:
    """抓：其他座位的壞曲線／壞距離毒掉類代價、誤亮距離旗標或污染梯度。"""
    record = cast(dict[str, dict[str, object]], measured["ignored_seat"])[name]
    assert np.isfinite(np.asarray(record["jit"])).all(), (name, record)
    bounds = [contract_value(key) for key in _CONTRACTS]
    for expected in (record["python"], _record(measured, "curves")["python"]):
        assert np.all(np.abs(np.asarray(record["jit"]) - np.asarray(expected)) <= bounds), (name, record)
    for field in ("distance_flag_clear", "gradient_finite", "ignored_gradient_zero"):
        assert record[field] is True, (name, field, record)


def test_enabled_direct_time_components_are_nonzero(measured: dict[str, object]) -> None:
    """抓：啟用設定改寫落空、或直達時間分項被默默省略。probe 另核開關為真。"""
    for field in ("matching", "matching_python"):
        assert (np.asarray(_record(measured, "time_enabled")[field])[:, -1] > 0.0).all(), field
        assert (np.asarray(_record(measured, "curves")[field])[:, -1] == 0.0).all(), field


@pytest.mark.parametrize("channel", (0, 1))
def test_invalid_primary_curve_makes_timbre_category_nan(measured: dict[str, object], channel: int) -> None:
    """抓：刪掉音色 where(valid, timbre, nan)，主位不可估卻回安全值代價。"""
    record = cast(dict[str, dict[str, bool]], measured["invalid_categories"])[f"timbre_{channel}"]
    assert all(record.values()), record


@pytest.mark.parametrize("channel", (0, 1))
def test_one_invalid_surrounding_curve_makes_listening_category_nan(measured: dict[str, object], channel: int) -> None:
    """抓：聆聽區 ~all(valid) 弱化成 ~any(valid)，漏掉單一無效座位。"""
    record = cast(dict[str, dict[str, bool]], measured["invalid_categories"])[f"listening_{channel}"]
    assert all(record.values()), record


def test_missing_peer_group_is_diagnostic_and_keeps_valid_cost(measured: dict[str, object]) -> None:
    """抓：缺周圍彼此配對時憑空補零分組，或把有效單周圍點場景拒算。"""
    small = _record(measured, "small_room")
    assert small["peer_group_missing"] is True
    assert np.isfinite(np.asarray(small["jit"])).all()
    assert _record(measured, "curves")["peer_group_missing"] is False


@pytest.mark.parametrize("name", ("channel", "seat", "frequency", "distance"))
def test_shape_mismatch_is_flagged_nan_without_poisoning_valid_gradient(measured: dict[str, object], name: str) -> None:
    """抓：缺聲道／座位被越界夾值，或錯頻率／距離形狀在編譯內丟錯。"""
    assert all(cast(dict[str, dict[str, bool]], measured["shapes"])[name].values())


def _central_difference() -> str:
    """每格用真的 numpy 評估器重算；比較 E*dC/dE，沿用第一階段的相對步長。"""
    case = curves()
    derivatives = np.empty((len(CATEGORIES), *case.energies.shape))
    for index in np.ndindex(case.energies.shape):
        upper, lower = case.energies.copy(), case.energies.copy()
        step = case.energies[index] * _DIFFERENCE_STEP
        upper[index] += step
        lower[index] -= step
        derivatives[(slice(None), *index)] = (numpy_costs(case, upper) - numpy_costs(case, lower)) / (2.0 * _DIFFERENCE_STEP)
    return json.dumps(derivatives.tolist())


@pytest.fixture(scope="session")
def central(tmp_path_factory: pytest.TempPathFactory, worker_id: str) -> np.ndarray:
    text = shared_json(tmp_path_factory, worker_id, "timbre-jax-numpy-central", _central_difference)
    return np.asarray(json.loads(text))


@pytest.mark.parametrize("category", range(len(CATEGORIES)), ids=CATEGORIES)
def test_energy_gradient_matches_numpy_central_difference(measured: dict[str, object], central: np.ndarray, category: int) -> None:
    """抓：值相等卻斷梯度、符號錯或配對權重導數錯；逐一比較每個聲道／座位／頻點。"""
    automatic = np.asarray(cast(dict[str, object], measured["gradient"])["double"])[category] * curves().energies
    assert np.any(automatic != 0.0)
    np.testing.assert_allclose(automatic, central[category], rtol=GRADIENT_REL, atol=GRADIENT_ABS)


@pytest.mark.parametrize("name", _FLAGS)
def test_invalid_flag_nan_and_other_gradient_is_finite(measured: dict[str, object], name: str) -> None:
    """抓：旗標沒亮、不可估回假數字，或無效支的 NaN 使另一個有效量的梯度壞掉。"""
    record = cast(dict[str, dict[str, bool]], measured["flags"])[name]
    assert all(record.values()), (name, record)


def test_range_cost_and_gradient_are_exactly_zero_including_boundary(measured: dict[str, object]) -> None:
    """抓：帶內漏出推力、容許帶邊界回半個梯度、帶外斷梯度或與中央差分不同。"""
    record = cast(dict[str, object], measured["bands"])
    for field in ("inside_cost", "inside_gradient", "boundary_gradient"):
        assert (np.asarray(record[field]) == 0.0).all(), field
    actual = np.asarray(record["outside_gradient"])
    assert (actual != 0.0).all()
    purpose = load_quality_targets(curves().targets).purpose(PURPOSE)
    expected = []
    for key, value in zip(cast(list[str], record["keys"]), np.asarray(record["outside_values"]), strict=True):
        entry = purpose.entry(key)
        assert isinstance(entry, TargetEntry)
        step = value * _DIFFERENCE_STEP
        expected.append((shape_cost(entry, (value + step,)) - shape_cost(entry, (value - step,))) / (2.0 * step))
    np.testing.assert_allclose(actual, expected, rtol=GRADIENT_REL, atol=GRADIENT_ABS)


@pytest.mark.parametrize("name", ("smoothing", "fit_weights", "ripple_range", "broadband", "category_weights", "matching_scale"))
def test_physical_mutants_are_rejected(measured: dict[str, object], name: str) -> None:
    """實際換錯 JAX 常數再算：平滑、擬線等權、起伏範圍、寬頻等權、類內權重、正規化。"""
    section, case = ("curves_mutants", "curves") if name == "ripple_range" else ("mutants", "small_room")
    actual = np.asarray(cast(dict[str, object], measured[section])[name])
    expected = np.asarray(_record(measured, case)["python"])
    bounds = np.asarray([contract_value(key) for key in _CONTRACTS])
    assert np.any(np.abs(actual - expected) > bounds), (name, actual, expected)


def test_timbre_cost_jax_mutant_beyond_tolerance_is_red(measured: dict[str, object]) -> None:
    """音色權重倍增必須超過音色契約；參考答案來自 numpy 評估器＋代價。"""
    reference = np.asarray(_record(measured, "small_room")["python"])[0]
    mutant = np.asarray(cast(dict[str, object], measured["mutants"])["category_weights"])[0]
    assert abs(mutant - reference) > contract_value(_CONTRACTS[0])


def test_listening_area_cost_jax_mutant_beyond_tolerance_is_red(measured: dict[str, object]) -> None:
    """聆聽區主項權重倍增必須超過聆聽區契約。"""
    reference = np.asarray(_record(measured, "small_room")["python"])[1]
    mutant = np.asarray(cast(dict[str, object], measured["mutants"])["category_weights"])[1]
    assert abs(mutant - reference) > contract_value(_CONTRACTS[1])


def test_channel_matching_cost_jax_mutant_beyond_tolerance_is_red(measured: dict[str, object]) -> None:
    """左右匹配較差參考倍增必須超過左右匹配契約。"""
    reference = np.asarray(_record(measured, "small_room")["python"])[2]
    mutant = np.asarray(cast(dict[str, object], measured["mutants"])["matching_scale"])[2]
    assert abs(mutant - reference) > contract_value(_CONTRACTS[2])
