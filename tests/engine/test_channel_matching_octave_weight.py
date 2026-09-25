"""聲道匹配的寬頻音量差每八度等權，不讓取樣密度變成票數（票 #450）。"""

from __future__ import annotations

import math

import numpy as np
import pytest

from aosr.config.frequency_axis import frequency_axis
from aosr.scoring.channel_matching_settings import _load_settings
from tests.engine import test_channel_matching as fixtures

_RANGE_HZ = (20.0, 8000.0)  # 等於登記簿 channel_matching.broadband_range_hz；下面那題守兩者一致
_PEAK_HZ = 257.3
_PEAK_HALF_WIDTH_HZ = 1.0


def _coarse_axis() -> tuple[float, ...]:
    """今天的正式細軸：20 Hz 起每八度 24 點，接到 8000 Hz 帶上緣之外。"""
    return frequency_axis("octave_fraction", 20.0, 11166.8, per_octave=24)


def _mixed_axis() -> tuple[float, ...]:
    """驗證軸的形狀（#435）：300 Hz 以下每 1 Hz 一點，以上照正式細軸。"""
    dense = frequency_axis("linear", 20.0, 300.0, step_hz=1.0)
    return (*dense, *(f for f in _coarse_axis() if f > 300.0))


def _broadband_db(
    frequencies: tuple[float, ...],
    left_energy: tuple[float, ...],
    right_energy: tuple[float, ...],
) -> float:
    receivers = fixtures._receivers()
    group = fixtures._group()
    main = fixtures._point(
        receivers,
        group,
        "main",
        left_energy=left_energy,
        right_energy=right_energy,
        left_frequencies_hz=frequencies,
        right_frequencies_hz=frequencies,
    )
    front = fixtures._point(
        receivers,
        group,
        "front",
        left_energy=left_energy,
        right_energy=right_energy,
        left_frequencies_hz=frequencies,
        right_frequencies_hz=frequencies,
    )
    payload = fixtures._payload(fixtures._evaluate(receivers, group, (main, front)))
    result = next(item for item in payload.point_results if item.receiver_id == "main")
    assert result.broadband_level_difference_db is not None
    return result.broadband_level_difference_db


def _pink(frequencies: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(100.0 / f for f in frequencies)


def _with_narrow_peak(frequencies: tuple[float, ...]) -> tuple[float, ...]:
    """平的底加一根 +20 dB、半功率寬 2 Hz 的窄峰；峰頂落在正式細軸兩點中間。"""
    return tuple(
        1.0 + 100.0 / (1.0 + ((f - _PEAK_HZ) / _PEAK_HALF_WIDTH_HZ) ** 2)
        for f in frequencies
    )


def test_range_constant_matches_registry() -> None:
    """這份考卷寫死的範圍要等於評估器從登記簿讀到的那一條；登記簿改了這題會紅，提醒下面幾題跟著看。"""
    settings = _load_settings(fixtures._TARGETS, fixtures._PURPOSE)

    assert settings.broadband_range_hz == _RANGE_HZ


def test_densifying_low_frequencies_keeps_a_smooth_answer() -> None:
    """平滑曲線（左聲道 1/f、右聲道平）從正式細軸換成低頻每 1 Hz 的軸，答案不動。
    每點一票的舊算法會讓低頻多拿約三倍的票、左聲道多出好幾 dB——拿掉加權這題會紅。"""
    coarse, mixed = _coarse_axis(), _mixed_axis()

    on_coarse = _broadband_db(coarse, _pink(coarse), tuple(1.0 for _ in coarse))
    on_mixed = _broadband_db(mixed, _pink(mixed), tuple(1.0 for _ in mixed))

    assert on_mixed == pytest.approx(on_coarse, abs=0.01)


def test_densifying_finds_a_narrow_peak_the_coarse_axis_missed() -> None:
    """一根落在正式細軸兩點中間的窄峰，加密後要找得到：答案變大，而且對到獨立積分的參考值。
    把加密找到的資訊也當成「密度造成的偏差」壓掉（例如每一區間只取一點），這題會紅。"""
    coarse, mixed = _coarse_axis(), _mixed_axis()
    flat_coarse = tuple(1.0 for _ in coarse)
    flat_mixed = tuple(1.0 for _ in mixed)

    on_coarse = _broadband_db(coarse, _with_narrow_peak(coarse), flat_coarse)
    on_mixed = _broadband_db(mixed, _with_narrow_peak(mixed), flat_mixed)

    # 獨立參考：在對數頻率上用 0.01 Hz 細格做梯形積分，跟評估器的格子算法無關。
    fine = np.arange(_RANGE_HZ[0], _RANGE_HZ[1] + 0.005, 0.01)
    octaves = np.log2(fine)
    energy = 1.0 + 100.0 / (1.0 + ((fine - _PEAK_HZ) / _PEAK_HALF_WIDTH_HZ) ** 2)
    reference = 10.0 * math.log10(
        float(np.trapezoid(energy, octaves)) / float(octaves[-1] - octaves[0])
    )

    assert on_mixed > on_coarse + 0.2
    assert on_mixed == pytest.approx(reference, abs=0.02)


def test_points_outside_the_range_never_change_the_answer() -> None:
    """8000 Hz 以上的點不管有多少、能量多大，都不改變範圍內的尺；讓範圍外的點組格子，這題會紅。"""
    coarse = _coarse_axis()
    flat = tuple(1.0 for _ in coarse)
    left = _pink(coarse)
    loud_outside = tuple(1e6 if f > _RANGE_HZ[1] else e for f, e in zip(coarse, left, strict=True))
    trimmed = tuple(f for f in coarse if f <= _RANGE_HZ[1])

    full = _broadband_db(coarse, left, flat)

    assert _broadband_db(coarse, loud_outside, flat) == full
    assert _broadband_db(trimmed, _pink(trimmed), tuple(1.0 for _ in trimmed)) == full


def test_uneven_points_weigh_by_the_octave_they_represent() -> None:
    """100、200、800 Hz 三點：格子寬 1、1.5、2 個八度（兩端各往外延半個鄰距，不伸到 20／8000）。
    左 (4,1,1)、右 (1,1,1)：(4·1+1.5+2)÷(1+1.5+2) = 7.5 ÷ 4.5。每點一票是 6 ÷ 3；
    兩端伸到範圍邊界是 (4·2.82+1.5+4.32)/(2.82+1.5+4.32)——兩種改法這題都會紅。"""
    frequencies = (100.0, 200.0, 800.0)

    result = _broadband_db(frequencies, (4.0, 1.0, 1.0), (1.0, 1.0, 1.0))

    assert result == pytest.approx(10.0 * math.log10(7.5 / 4.5))


@pytest.mark.parametrize(
    ("frequencies", "left", "expected_ratio"),
    (
        # 20 Hz 那一點的格子往下延會出範圍，切掉後寬 0.5 八度、40 Hz 寬 1：(4·0.5+1)/(0.5+1)。
        # 不切到範圍下緣是 (4+1)/2——拿掉那一刀這一例會紅。
        ((20.0, 40.0), (4.0, 1.0), 3.0 / 1.5),
        # 範圍內只剩一點（9000 Hz 在範圍外）：就是那一點的左右比，不能除以零。
        ((100.0, 9000.0), (4.0, 1.0), 4.0),
        # 8000 Hz 那一點的格子往上延會出範圍，切掉後寬 0.5 八度、4000 Hz 寬 1：(1·1+4·0.5)÷(1+0.5)。
        # 不切到範圍上緣是 (1+4)÷2——拿掉那一刀這一例會紅（找碴席抓到的盲點）。
        ((4000.0, 8000.0), (1.0, 4.0), 3.0 / 1.5),
        # 剛好比 8000 Hz 多一點點的點在範圍外；用對數值判斷範圍會把它捨入成 8000 算進去（找碴席抓到）。
        ((4000.0, math.nextafter(8000.0, math.inf)), (1.0, 4.0), 1.0),
        # 兩點近到取對數後分不開：退回每點等權 (4+1)÷2，不能除以零（找碴席抓到）。
        ((20.0, math.nextafter(20.0, math.inf)), (4.0, 1.0), 2.5),
    ),
)
def test_range_edges_clip_cells_and_a_single_point_still_counts(
    frequencies: tuple[float, ...], left: tuple[float, ...], expected_ratio: float
) -> None:
    result = _broadband_db(frequencies, left, (1.0, 1.0))

    assert result == pytest.approx(10.0 * math.log10(expected_ratio))


@pytest.mark.parametrize(
    ("energy", "frequencies"),
    # 下溢那一例要兩點夠近：格子只有約 0.007 八度寬，5e-324 乘上去才會變成 0。
    ((1e308, (100.0, 200.0)), (5e-324, (100.0, 100.5))),
)
def test_extreme_but_valid_energies_still_give_a_finite_answer(
    energy: float, frequencies: tuple[float, ...]
) -> None:
    """左聲道能量極大或極小（仍是有限正值）：寬頻音量差要算得出來、等於 10·log10(energy)，
    不能溢位成無限大或下溢成 0 讓評估器當掉（#455 找碴席抓到，聲道匹配同一支）。"""
    result = _broadband_db(frequencies, (energy, energy), (1.0, 1.0))

    assert result == pytest.approx(10.0 * math.log10(energy))
