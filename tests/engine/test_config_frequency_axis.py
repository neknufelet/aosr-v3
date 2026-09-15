"""v3 共用頻率軸產生器與有限元素路軸的性質考卷。"""

from __future__ import annotations

import math

from aosr.config import frequency_axis as frequency_axis_config


def test_fractional_octave_axis_has_constant_ratio_and_bounded_last_point() -> None:
    """指數分母、首點或終止條件錯誤時，分數八度性質必須紅。"""
    start_hz = 20.0
    stop_hz = 300.0
    per_octave = 24

    axis = frequency_axis_config.frequency_axis(
        "octave_fraction",
        start_hz,
        stop_hz,
        per_octave=per_octave,
    )

    assert axis[0] == start_hz
    assert all(
        value == start_hz * 2.0 ** (index / per_octave)
        for index, value in enumerate(axis)
    )
    assert axis[-1] <= stop_hz
    assert start_hz * 2.0 ** (len(axis) / per_octave) > stop_hz


def test_fractional_octave_axis_includes_point_exactly_at_upper_bound() -> None:
    """分數八度終止條件若從 ``<=`` 變成 ``<``，剛好上限的點必須被抓到。"""
    axis = frequency_axis_config.frequency_axis(
        "octave_fraction",
        20.0,
        80.0,
        per_octave=12,
    )

    assert axis[-1] == 80.0


def test_linear_axis_is_equidistant_and_includes_equal_upper_bound() -> None:
    """線性格若漂移、漏首點或把等於上限排除，這題必須紅。"""
    start_hz = 20.0
    stop_hz = 21.0
    step_hz = 0.25

    axis = frequency_axis_config.frequency_axis(
        "linear",
        start_hz,
        stop_hz,
        step_hz=step_hz,
    )

    assert axis[0] == start_hz
    assert all(right - left == step_hz for left, right in zip(axis, axis[1:]))
    assert axis[-1] == stop_hz
    assert axis[-1] + step_hz > stop_hz


def test_v3_fem_lane_axis_obeys_accepted_resolution_and_cap() -> None:
    """有限元素路軸的起點、公比或 300 Hz 上限接錯時必須紅。"""
    axis = frequency_axis_config.FEM_LANE_FREQUENCIES_HZ
    ratio = 2.0 ** (1.0 / frequency_axis_config.V3_AXIS_POINTS_PER_OCTAVE)

    assert frequency_axis_config.FEM_GEOMETRIC_CROSSOVER_CAP_HZ == 300.0
    assert axis[0] == frequency_axis_config.V3_AXIS_START_HZ
    assert all(
        value
        == frequency_axis_config.V3_AXIS_START_HZ
        * 2.0 ** (index / frequency_axis_config.V3_AXIS_POINTS_PER_OCTAVE)
        for index, value in enumerate(axis)
    )
    assert axis[-1] <= frequency_axis_config.FEM_GEOMETRIC_CROSSOVER_CAP_HZ
    assert axis[-1] * ratio > frequency_axis_config.FEM_GEOMETRIC_CROSSOVER_CAP_HZ


def test_geometric_band_axis_uses_half_open_interval_and_step_multiples() -> None:
    """密軸含錯邊界、不是 0.5 Hz 整數倍或首尾少一點時必須紅。"""
    center_hz = 1000.0
    step_hz = frequency_axis_config.GEOMETRIC_BAND_FREQUENCY_STEP_HZ
    actual = frequency_axis_config.geometric_band_frequencies(center_hz)
    lower = center_hz / math.sqrt(2.0)
    upper = center_hz * math.sqrt(2.0)

    assert actual[0] == 707.5
    assert actual[-1] == 1414.0
    assert actual[0] - step_hz < lower <= actual[0]
    assert actual[-1] < upper <= actual[-1] + step_hz
    assert all(lower <= frequency < upper for frequency in actual)
    assert all(frequency / step_hz == int(frequency / step_hz) for frequency in actual)

    aligned_upper_hz = 1000.0
    upper_aligned = frequency_axis_config.geometric_band_frequencies(
        aligned_upper_hz / math.sqrt(2.0)
    )
    assert upper_aligned[-1] == aligned_upper_hz - step_hz
    assert aligned_upper_hz not in upper_aligned
