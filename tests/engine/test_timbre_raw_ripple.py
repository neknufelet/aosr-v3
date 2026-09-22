"""票 #432：起伏與峰谷看原始曲線——寬度 0 的幾條邊界（檢查席點的盲點各一題）。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aosr.scoring import timbre
from aosr.scoring.contract import EvaluationState, Flag, ReasonCode, TimbrePayload
from tests.engine.test_scoring_timbre import (
    _REGISTRY,
    _curve_input,
    _evaluate,
    _gaussian_feature,
    _range_setting,
    _registry_with,
)

_RIPPLE = 'key = "timbre_balance.smoothing_width_octave_ripple"\nvalue = 0.0'
_MIN_WIDTH = 'key = "timbre_balance.feature_min_width_octave"\nvalue = 0.0'
_TILT = 'key = "timbre_balance.smoothing_width_octave_tilt"\nvalue = 0.3333333333333333'


def test_zero_width_smoothing_returns_the_energy_exactly() -> None:
    """寬度 0 走累積和相減會把很小的值捨成 0（再取 log 就爆）；必須逐位元等於輸入。"""
    energy = np.array([1.0, 1e-20, 0.5, 1e-18, 2.0], dtype=np.float64)
    octaves = np.log2(np.geomspace(100.0, 200.0, len(energy)))

    smoothed = timbre._smooth_energy(octaves, energy, 0.0)

    assert smoothed.tolist() == energy.tolist()


def test_positive_width_smoothing_preserves_tiny_energy() -> None:
    """有平滑時也不能用累積和相減，否則大值後面的 1e-30 能量會被捨成 0。"""
    octaves = np.array([0.0, 0.1, 0.2, 0.3], dtype=np.float64)
    energy = np.array([1.0, 1e-30, 2e-30, 3e-30], dtype=np.float64)

    smoothed = timbre._smooth_energy(octaves, energy, 0.1)

    assert smoothed[-1] > 0.0
    assert smoothed[-1] == pytest.approx(3e-30)


def test_one_missing_axis_point_inside_ripple_range_is_a_gap() -> None:
    """不平滑時少交一個軸點就是洞：拿傾斜的 1/3 八度當尺會放過它，少交資料就變成沒有那個峰。"""
    full = _curve_input(_gaussian_feature(300.0, 6.0, 1.0 / 20.0), point_count=481)
    frequencies = list(full.frequencies_hz)
    energies = list(full.total_energy)
    drop = min(range(len(frequencies)), key=lambda i: abs(frequencies[i] - 300.0))
    del frequencies[drop], energies[drop]
    holed = full.model_copy(update={"frequencies_hz": tuple(frequencies), "total_energy": tuple(energies)})

    assert _evaluate(full).state is EvaluationState.MEASURED
    evaluation = _evaluate(holed)
    assert evaluation.state is EvaluationState.UNAVAILABLE
    assert evaluation.reason_codes == (ReasonCode.TIMBRE_SCORING_RANGE_GAP,)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        (_RIPPLE, _RIPPLE.replace("0.0", "-0.1"), "非負"),
        (_MIN_WIDTH, _MIN_WIDTH.replace("0.0", "-0.1"), "非負"),
        (_TILT, _TILT.replace("0.3333333333333333", "0.0"), "為正"),
    ],
    ids=["negative-ripple", "negative-min-width", "zero-tilt"],
)
def test_width_sign_rules(tmp_path: Path, old: str, new: str, message: str) -> None:
    """起伏寬度與最小寬度准 0 不准負；傾斜寬度必須為正（它是對平滑後的走勢擬合）。"""
    with pytest.raises(ValueError, match=message):
        _evaluate(_curve_input(lambda f: np.zeros_like(f)), registry=_registry_with(tmp_path, old, new))


def test_axis_flag_uses_local_spacing_not_global_median() -> None:
    """軸可以不等距：峰附近的軸很密時，用整條軸的中位數會把已經量得準的峰誤標成窄於軸。"""
    coarse = np.geomspace(20.0, 8000.0, 121)
    dense = np.geomspace(250.0, 360.0, 200)
    frequencies = np.unique(np.concatenate([coarse, dense]))
    base = _curve_input(_gaussian_feature(300.0, 6.0, 1.0 / 30.0))
    db = _gaussian_feature(300.0, 6.0, 1.0 / 30.0)(frequencies)
    input_data = base.model_copy(
        update={
            "frequencies_hz": tuple(float(v) for v in frequencies),
            "total_energy": tuple(float(v) for v in 10.0 ** (db / 10.0)),
        }
    )

    evaluation = _evaluate(input_data)

    assert isinstance(evaluation.payload, TimbrePayload)
    peak = max((f for f in evaluation.payload.features if f.kind == "peak"), key=lambda f: f.depth_db)
    assert 290.0 <= peak.center_frequency_hz <= 310.0
    assert Flag.FEATURE_NARROWER_THAN_AXIS not in peak.flags


def test_axis_flag_uses_local_spacing_where_the_axis_is_coarse() -> None:
    """反過來：軸在別處很密、峰所在這一段粗，用整條軸的中位數會漏標粗軸上量不準的峰。"""
    coarse = np.geomspace(20.0, 8000.0, 121)
    dense = np.geomspace(2000.0, 3000.0, 400)
    frequencies = np.unique(np.concatenate([coarse, dense]))
    coarse_step = float(np.log2(8000.0 / 20.0) / 120)
    db = _gaussian_feature(300.0, 6.0, 1.2 * coarse_step)(frequencies)
    input_data = _curve_input(_gaussian_feature(300.0, 6.0, 1.2 * coarse_step)).model_copy(
        update={
            "frequencies_hz": tuple(float(v) for v in frequencies),
            "total_energy": tuple(float(v) for v in 10.0 ** (db / 10.0)),
        }
    )

    evaluation = _evaluate(input_data)

    assert isinstance(evaluation.payload, TimbrePayload)
    peak = max((f for f in evaluation.payload.features if f.kind == "peak"), key=lambda f: f.depth_db)
    assert 280.0 <= peak.center_frequency_hz <= 320.0
    assert Flag.FEATURE_NARROWER_THAN_AXIS in peak.flags


def test_axis_flag_threshold_is_two_points_not_three() -> None:
    """半深度寬度落在兩到三個軸距之間的峰不准標：門檻寫成三會誤標。"""
    step = float(np.log2(8000.0 / 20.0) / 480)
    input_data = _curve_input(_gaussian_feature(300.0, 6.0, 2.5 * step), point_count=481)

    evaluation = _evaluate(input_data)

    assert isinstance(evaluation.payload, TimbrePayload)
    peak = max((f for f in evaluation.payload.features if f.kind == "peak"), key=lambda f: f.depth_db)
    assert peak.width_octave is not None and 2.0 * step <= peak.width_octave < 3.0 * step
    assert Flag.FEATURE_NARROWER_THAN_AXIS not in peak.flags


def test_boundary_incomplete_feature_never_gets_the_axis_flag() -> None:
    """寬度未知（邊界不完整）的特徵沒有寬度可比，不准同時掛「窄於軸解析度」。"""
    lower_hz, _ = _range_setting("timbre_balance.ripple_range_hz")
    center_hz = float(lower_hz) * 2.0 ** (1.0 / 24.0)
    input_data = _curve_input(_gaussian_feature(center_hz, -6.0, 1.0 / 6.0), point_count=481)

    evaluation = _evaluate(input_data)

    assert isinstance(evaluation.payload, TimbrePayload)
    unknown = [f for f in evaluation.payload.features if f.width_octave is None]
    assert unknown
    assert all(Flag.FEATURE_NARROWER_THAN_AXIS not in f.flags for f in unknown)
    assert all(Flag.FEATURE_BOUNDARY_INCOMPLETE in f.flags for f in unknown)


def test_registry_default_is_the_raw_curve() -> None:
    """登記簿裡起伏的平滑寬度是 0（票 #432）；不是 0 就代表有人把它改回去了。"""
    assert _RIPPLE in _REGISTRY.read_text(encoding="utf-8")
