"""票 #446：音色的積分量每八度等權，取樣加密只增加解析度。"""
from __future__ import annotations

import math

import numpy as np
import pytest

from aosr.config.frequency_axis import GEOMETRIC_LANE_FREQUENCIES_HZ
from aosr.scoring import timbre
from aosr.scoring.contract import TimbrePayload
from tests.engine.test_scoring_timbre import _curve_input, _evaluate


def _input_on_axis(frequencies: np.ndarray, db: np.ndarray) -> timbre.TimbreInput:
    base = _curve_input(np.zeros_like)
    return base.model_copy(
        update={
            "frequencies_hz": tuple(float(value) for value in frequencies),
            "total_energy": tuple(float(value) for value in 10.0 ** (db / 10.0)),
        }
    )


def _payload_on_axis(frequencies: np.ndarray, db: np.ndarray) -> TimbrePayload:
    evaluation = _evaluate(_input_on_axis(frequencies, db))
    assert isinstance(evaluation.payload, TimbrePayload)
    return evaluation.payload


def _axis_with_dense_low_frequencies(coarse: np.ndarray) -> np.ndarray:
    start = float(np.log2(coarse[0]))
    stop = float(np.log2(300.0))
    dense_octaves = np.arange(start, stop + 1.0 / 480.0, 1.0 / 240.0)
    dense = 2.0**dense_octaves
    return np.unique(np.concatenate((dense[dense <= 300.0], coarse[coarse > 300.0])))


def _smooth_test_curve(frequencies: np.ndarray) -> np.ndarray:
    octaves = np.log2(frequencies / 1000.0)
    broad_bump = 4.0 * np.exp(-4.0 * math.log(2.0) * (octaves / 1.0) ** 2)
    return np.asarray(-octaves + broad_bump, dtype=np.float64)


def test_octave_weights_sum_to_the_axis_span() -> None:
    """若權重漏算或重算端點，積分權重總和就不再等於這條軸涵蓋的八度跨度。"""
    octaves = np.array([0.0, 0.2, 0.7, 1.8, 2.0], dtype=np.float64)

    weights = timbre._octave_weights(octaves)

    assert float(np.sum(weights)) == pytest.approx(2.0)


def test_octave_weights_use_one_for_a_single_point() -> None:
    """單點沒有鄰居可推代表寬度；若漏掉明定的退路，正規化會除以零。"""
    weights = timbre._octave_weights(np.array([3.0], dtype=np.float64))

    assert weights.tolist() == [1.0]


def test_octave_weights_are_equal_on_an_equal_ratio_axis() -> None:
    """等比頻率軸每點代表相同八度寬；若首尾誤用半票，這題會紅。"""
    octaves = np.linspace(0.0, 2.0, 9)

    weights = timbre._octave_weights(octaves)

    assert np.allclose(weights, weights[0])


def test_octave_weights_give_dense_points_less_weight() -> None:
    """不等距軸的密段每點要少票；若退回每點等權，密段與疏段會一樣重。"""
    octaves = np.array([0.0, 0.01, 0.02, 1.0, 2.0], dtype=np.float64)

    weights = timbre._octave_weights(octaves)

    assert weights[1] < weights[3]


def test_more_low_frequency_samples_do_not_add_low_frequency_weight() -> None:
    """低頻只加密時，三個積分量只准剩插值誤差，不准把新點當成更多票。"""
    coarse = np.asarray(GEOMETRIC_LANE_FREQUENCIES_HZ, dtype=np.float64)
    dense = _axis_with_dense_low_frequencies(coarse)
    coarse_payload = _payload_on_axis(coarse, _smooth_test_curve(coarse))
    dense_payload = _payload_on_axis(dense, _smooth_test_curve(dense))
    tolerance = 0.02  # dB 或 dB/oct；只容納兩條軸離散積分的插值誤差。

    assert abs(dense_payload.tilt_db_per_octave - coarse_payload.tilt_db_per_octave) < tolerance
    assert abs(dense_payload.residual_rms_db - coarse_payload.residual_rms_db) < tolerance
    assert (
        abs(dense_payload.target_deviation_rms_db - coarse_payload.target_deviation_rms_db)
        < tolerance
    )


def test_dense_axis_reveals_a_peak_hidden_between_coarse_points() -> None:
    """加密若真的帶來新資訊，raw 峰谷與起伏仍要改，不能為追求不變而降回粗軸。"""
    coarse = np.asarray(GEOMETRIC_LANE_FREQUENCIES_HZ, dtype=np.float64)
    dense = _axis_with_dense_low_frequencies(coarse)
    left = int(np.searchsorted(coarse, 180.0))
    center_octave = 0.5 * float(np.log2(coarse[left - 1] * coarse[left]))

    def hidden_peak(frequencies: np.ndarray) -> np.ndarray:
        distance = np.abs(np.log2(frequencies) - center_octave)
        return np.asarray(10.0 * np.maximum(1.0 - distance / 0.02, 0.0), dtype=np.float64)

    coarse_payload = _payload_on_axis(coarse, hidden_peak(coarse))
    dense_payload = _payload_on_axis(dense, hidden_peak(dense))
    center_hz = 2.0**center_octave
    coarse_nearby = [
        feature
        for feature in coarse_payload.features
        if abs(math.log2(feature.center_frequency_hz / center_hz)) < 0.02
    ]
    dense_nearby = [
        feature
        for feature in dense_payload.features
        if feature.kind == "peak"
        and abs(math.log2(feature.center_frequency_hz / center_hz)) < 0.02
    ]

    assert not coarse_nearby
    assert dense_nearby
    assert abs(max(feature.depth_db for feature in dense_nearby) - 10.0) < 0.3
    assert dense_payload.residual_rms_db > coarse_payload.residual_rms_db
