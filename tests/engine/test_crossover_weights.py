"""Schroeder 頻率與有限元素／幾何路交接權重的獨立性質考卷。"""
from __future__ import annotations

import math
from collections.abc import Sequence

import pytest

from aosr.config import three_lane_crossover as crossover_config
from aosr.config.frequency_axis import FEM_GEOMETRIC_CROSSOVER_CAP_HZ
from aosr.geometry.shoebox import Room, Wall
from aosr.physics.crossover import (
    CrossoverWeights,
    crossover_weights,
    eyring_t60_by_band,
    schroeder_frequency_hz,
)


def _weights(frequencies_hz: Sequence[float], f_s_hz: float) -> CrossoverWeights:
    return crossover_weights(frequencies_hz, f_s_hz)


def _uniform_absorption(alpha: float) -> dict[Wall, dict[float, float]]:
    return {
        wall: {500.0: alpha, 1000.0: alpha}
        for wall in Wall.all()
    }


def _independent_geo_weight(frequency_hz: float, lower_hz: float) -> float:
    """考卷自寫公式，不呼叫產品的平滑階梯或區間函式。"""
    if frequency_hz <= lower_hz:
        return 0.0
    if frequency_hz >= FEM_GEOMETRIC_CROSSOVER_CAP_HZ:
        return 1.0
    t = math.log2(frequency_hz / lower_hz) / math.log2(
        FEM_GEOMETRIC_CROSSOVER_CAP_HZ / lower_hz
    )
    return t * t * (3.0 - 2.0 * t)


def test_frequency_axis_is_the_only_config_home_of_crossover_cap() -> None:
    """交接設定檔若重新公開第二份有限元素上限，單一權威契約就必須紅。"""
    assert not hasattr(crossover_config, "FEM_GEOMETRIC_CROSSOVER_CAP_HZ")


def test_eyring_uniform_reference_room_matches_hand_calculation_exactly() -> None:
    """手算：V=6·4·3=72、S=2(24+18+12)=108、T=0.161·72/[-108 ln(1-0.3)]。"""
    expected_t60_s = 0.30092759572079847

    actual = eyring_t60_by_band(
        Room(6.0, 4.0, 3.0), _uniform_absorption(0.3)
    )

    assert actual == {500.0: expected_t60_s, 1000.0: expected_t60_s}


def test_eyring_uses_face_area_weighted_absorption() -> None:
    """把牆面積誤當等權平均時，兩個具名頻帶都會與手算答案不同。"""
    values_500 = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
    values_1000 = tuple(reversed(values_500))
    absorption = {
        wall: {500.0: values_500[index], 1000.0: values_1000[index]}
        for index, wall in enumerate(Wall.all())
    }

    actual = eyring_t60_by_band(Room(6.0, 4.0, 3.0), absorption)

    assert actual == {
        500.0: 0.2702478329676631,
        1000.0: 0.2305422524158068,
    }


@pytest.mark.parametrize("invalid_alpha", (1.0, 1.2))
def test_eyring_rejects_absorption_at_or_above_one(invalid_alpha: float) -> None:
    """任一牆、任一頻帶的吸音率到 1 或越界都不准被夾回來。"""
    absorption = _uniform_absorption(0.3)
    absorption[Wall.FLOOR][500.0] = invalid_alpha

    with pytest.raises(ValueError, match="吸音率"):
        eyring_t60_by_band(Room(6.0, 4.0, 3.0), absorption)


@pytest.mark.parametrize("missing_hz", (500.0, 1000.0))
def test_eyring_input_requires_both_schroeder_midbands(missing_hz: float) -> None:
    """六面吸音率原始輸入少 500/1000 任一帶時，第一步就必須報錯。"""
    absorption = _uniform_absorption(0.3)
    for wall in Wall.all():
        del absorption[wall][missing_hz]

    with pytest.raises(ValueError, match=f"{missing_hz:g}"):
        eyring_t60_by_band(Room(6.0, 4.0, 3.0), absorption)


@pytest.mark.parametrize("missing_hz", (500.0, 1000.0))
def test_schroeder_frequency_requires_both_midbands(missing_hz: float) -> None:
    """500 Hz 或 1000 Hz 任缺一帶都必須報錯，不准換帶或退回平均。"""
    t60_by_band = {500.0: 0.4, 1000.0: 0.8}
    del t60_by_band[missing_hz]

    with pytest.raises(ValueError, match=f"{missing_hz:g}"):
        schroeder_frequency_hz(Room(6.0, 4.0, 3.0), t60_by_band)


def test_schroeder_frequency_uses_arithmetic_mean_of_midband_t60() -> None:
    """f_s=2000·sqrt(((0.4+0.8)/2)/72)，其他頻帶不得混入。"""
    actual = schroeder_frequency_hz(
        Room(6.0, 4.0, 3.0),
        {125.0: 9.0, 500.0: 0.4, 1000.0: 0.8, 2000.0: 7.0},
    )

    assert actual == 182.57418583505537


def test_weights_are_bounded_and_exactly_complementary_on_fine_grid() -> None:
    """細網格逐點守 [0,1] 與精確 ``w_fem + w_geo == 1.0``。"""
    frequencies = tuple(
        20.0 + (4000.0 - 20.0) * index / 8192.0
        for index in range(8193)
    )
    actual = _weights(frequencies, 151.0)

    assert all(0.0 <= weight <= 1.0 for weight in (*actual.w_fem, *actual.w_geo))
    assert all(
        w_fem + w_geo == 1.0
        for w_fem, w_geo in zip(actual.w_fem, actual.w_geo, strict=True)
    )


def test_transition_endpoints_are_exact_and_outside_is_single_lane() -> None:
    """平滑段兩端恰為 0/1；段外沒有另一條路殘留。"""
    actual = _weights(
        (149.0, 150.0, 225.0, FEM_GEOMETRIC_CROSSOVER_CAP_HZ, 301.0),
        149.0,
    )

    assert actual.w_geo[:2] == (0.0, 0.0)
    assert 0.0 < actual.w_geo[2] < 1.0
    assert actual.w_geo[3:] == (1.0, 1.0)
    assert actual.w_fem[:2] == (1.0, 1.0)
    assert actual.w_fem[3:] == (0.0, 0.0)


def test_geometric_weight_is_monotonic_in_frequency() -> None:
    """排序頻率上的幾何權重只能不降，有限元素權重只能不升。"""
    frequencies = tuple(float(frequency) for frequency in range(100, 351))
    actual = _weights(frequencies, 151.0)

    assert all(left <= right for left, right in zip(actual.w_geo, actual.w_geo[1:]))
    assert all(left >= right for left, right in zip(actual.w_fem, actual.w_fem[1:]))


@pytest.mark.parametrize(
    ("f_s_hz", "expected_lower_hz"),
    ((149.0, 150.0), (150.0, 150.0), (151.0, 151.0), (299.0, 299.0)),
)
def test_soft_branch_chooses_the_decided_lower_endpoint(
    f_s_hz: float, expected_lower_hz: float
) -> None:
    """149/150/151 與 299 都走軟交接，且下端是 max(f_s,150)。"""
    inside_hz = (expected_lower_hz + FEM_GEOMETRIC_CROSSOVER_CAP_HZ) / 2.0
    actual = _weights(
        (expected_lower_hz, inside_hz, FEM_GEOMETRIC_CROSSOVER_CAP_HZ),
        f_s_hz,
    )

    assert not actual.capped_by_upper_limit
    assert actual.w_geo[0] == 0.0
    assert 0.0 < actual.w_geo[1] < 1.0
    assert actual.w_geo[2] == 1.0


@pytest.mark.parametrize(
    "f_s_hz",
    (FEM_GEOMETRIC_CROSSOVER_CAP_HZ, FEM_GEOMETRIC_CROSSOVER_CAP_HZ + 1.0),
)
def test_hard_branch_keeps_300_hz_in_fem_and_reports_cap(f_s_hz: float) -> None:
    """f_s=300 也算硬切；300 Hz 歸有限元素，剛超過才歸幾何。"""
    actual = _weights(
        (299.0, FEM_GEOMETRIC_CROSSOVER_CAP_HZ, 301.0), f_s_hz
    )

    assert actual.capped_by_upper_limit
    assert actual.w_fem == (1.0, 1.0, 0.0)
    assert actual.w_geo == (0.0, 0.0, 1.0)


def test_smooth_transition_matches_independent_formula_pointwise() -> None:
    """產品的 log2 平滑階梯逐點等於考卷獨立寫出的 t²(3−2t)。"""
    frequencies = (
        150.0,
        175.0,
        200.0,
        225.0,
        250.0,
        275.0,
        FEM_GEOMETRIC_CROSSOVER_CAP_HZ,
    )
    expected_geo = tuple(
        _independent_geo_weight(frequency_hz, 150.0)
        for frequency_hz in frequencies
    )

    actual = _weights(frequencies, 149.0)

    assert actual.w_geo == expected_geo
