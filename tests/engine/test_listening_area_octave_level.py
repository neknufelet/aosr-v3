"""聽音區的整體音量由評估器自己從原始曲線每八度等權算，不讓取樣密度變成票數（票 #455）。"""

from __future__ import annotations

import math

import numpy as np
import pytest

from aosr.config.frequency_axis import frequency_axis
from aosr.scoring.contract import EvaluationState, ReasonCode
from aosr.scoring.listening_area import ReceiverPointResult, evaluate_listening_area
from aosr.scoring.receiver_set import ReceiverSet
from tests.engine import test_listening_area as fixtures

_RANGE_HZ = (20.0, 8000.0)
_PEAK_HZ = 257.3
_PEAK_HALF_WIDTH_HZ = 1.0


def _two_points() -> ReceiverSet:
    """主位加一個周圍點：主位對周圍的加權平均偏差就是兩點整體音量差的絕對值。"""
    return ReceiverSet(points=fixtures._receiver_set().points[:2])


def _coarse_axis() -> tuple[float, ...]:
    return frequency_axis("octave_fraction", 20.0, 11166.8, per_octave=24)


def _mixed_axis() -> tuple[float, ...]:
    """驗證軸的形狀（#435）：300 Hz 以下每 1 Hz 一點，以上照正式細軸。"""
    dense = frequency_axis("linear", 20.0, 300.0, step_hz=1.0)
    return (*dense, *(f for f in _coarse_axis() if f > 300.0))


def _result(
    receivers: ReceiverSet, receiver_id: str, frequencies: tuple[float, ...], energy: tuple[float, ...]
) -> ReceiverPointResult:
    return ReceiverPointResult(
        receiver_id=receiver_id,
        receiver_set_fingerprint=receivers.fingerprint,
        timbre_evaluation=fixtures._timbre(receiver_id, tilt=0.0, ripple=0.0),
        frequencies_hz=frequencies,
        total_energy=energy,
    )


def _level_difference(
    main: tuple[tuple[float, ...], tuple[float, ...]],
    front: tuple[tuple[float, ...], tuple[float, ...]],
) -> float:
    receivers = _two_points()
    evaluation = fixtures._evaluate(
        receivers, (_result(receivers, "main", *main), _result(receivers, "front", *front))
    )
    return fixtures._payload(evaluation).overall_level_stability.primary_to_surrounding.weighted_mean_deviation


def _pink(frequencies: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(100.0 / f for f in frequencies)


def _flat(frequencies: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(1.0 for _ in frequencies)


def test_fixture_range_matches_this_file() -> None:
    """共用考卷工具傳給評估器的範圍要等於這份考卷算參考值用的範圍。"""
    assert fixtures._BROADBAND_RANGE_HZ == _RANGE_HZ


def test_densifying_low_frequencies_keeps_a_smooth_answer() -> None:
    """主位 1/f、周圍點平的，從正式細軸換成低頻每 1 Hz 的軸，兩點音量差不動。
    呼叫端每點平均（舊做法）會讓低頻多拿約三倍的票——改回每點一票這題會紅。"""
    coarse, mixed = _coarse_axis(), _mixed_axis()

    on_coarse = _level_difference((coarse, _pink(coarse)), (coarse, _flat(coarse)))
    on_mixed = _level_difference((mixed, _pink(mixed)), (mixed, _flat(mixed)))

    assert on_mixed == pytest.approx(on_coarse, abs=0.01)


def test_densifying_finds_a_narrow_peak_the_coarse_axis_missed() -> None:
    """主位帶一根落在正式細軸兩點中間的窄峰，加密後音量差要變大、而且對到獨立積分的參考值。"""
    coarse, mixed = _coarse_axis(), _mixed_axis()

    def peaked(frequencies: tuple[float, ...]) -> tuple[float, ...]:
        return tuple(
            1.0 + 100.0 / (1.0 + ((f - _PEAK_HZ) / _PEAK_HALF_WIDTH_HZ) ** 2) for f in frequencies
        )

    on_coarse = _level_difference((coarse, peaked(coarse)), (coarse, _flat(coarse)))
    on_mixed = _level_difference((mixed, peaked(mixed)), (mixed, _flat(mixed)))

    fine = np.arange(_RANGE_HZ[0], _RANGE_HZ[1] + 0.005, 0.01)
    octaves = np.log2(fine)
    energy = 1.0 + 100.0 / (1.0 + ((fine - _PEAK_HZ) / _PEAK_HALF_WIDTH_HZ) ** 2)
    reference = 10.0 * math.log10(
        float(np.trapezoid(energy, octaves)) / float(octaves[-1] - octaves[0])
    )

    assert on_mixed > on_coarse + 0.2
    assert on_mixed == pytest.approx(reference, abs=0.02)


def test_uneven_points_weigh_by_the_octave_they_represent() -> None:
    """100、200、800 Hz 格子寬 1、1.5、2 八度：主位 (4,1,1) 的平均是 7.5 ÷ 4.5，周圍平的是 1。
    每點平均是 6 ÷ 3——改回每點一票這題會紅。"""
    frequencies = (100.0, 200.0, 800.0)

    result = _level_difference((frequencies, (4.0, 1.0, 1.0)), (frequencies, _flat(frequencies)))

    assert result == pytest.approx(10.0 * math.log10(7.5 / 4.5))


def test_points_outside_the_range_never_change_the_answer() -> None:
    """8000 Hz 以上的點能量再大，也不進整體音量。"""
    coarse = _coarse_axis()
    loud = tuple(1e6 if f > _RANGE_HZ[1] else e for f, e in zip(coarse, _pink(coarse), strict=True))

    assert _level_difference((coarse, loud), (coarse, _flat(coarse))) == _level_difference(
        (coarse, _pink(coarse)), (coarse, _flat(coarse))
    )


@pytest.mark.parametrize(
    ("front_axis", "reason"),
    (
        # 兩點範圍內的軸不一樣（一點加密、一點沒有）：整體音量不可比，整類不可估，不硬比。
        ((100.0, 150.0, 200.0), ReasonCode.FREQUENCY_AXIS_MISMATCH),
        # 周圍點在範圍內一點資料都沒有。
        ((9000.0, 10000.0), ReasonCode.INSUFFICIENT_COVERAGE),
    ),
)
def test_points_must_share_one_axis_inside_the_range(
    front_axis: tuple[float, ...], reason: ReasonCode
) -> None:
    receivers = _two_points()
    main_axis = (100.0, 200.0)
    evaluation = fixtures._evaluate(
        receivers,
        (
            _result(receivers, "main", main_axis, _flat(main_axis)),
            _result(receivers, "front", front_axis, _flat(front_axis)),
        ),
    )

    assert evaluation.state is EvaluationState.UNAVAILABLE
    assert reason in evaluation.reason_codes


def test_settings_fingerprint_tracks_the_broadband_range() -> None:
    """範圍沒進本層指紋的話，同一指紋會對到兩種整體音量答案；拿掉那一格這題會紅。"""
    receivers = fixtures._receiver_set()
    results = fixtures._results(receivers)

    def evaluate(bounds: tuple[float, float]) -> str:
        return evaluate_listening_area(
            receivers,
            results,
            candidate_id=fixtures._CANDIDATE,
            speaker_id=fixtures._SPEAKER,
            timbre_settings_fingerprint=fixtures._SETTINGS,
            scene_fingerprint=fixtures._SCENE_FINGERPRINT,
            feature_match_tolerance_hz=10.0,
            broadband_range_hz=bounds,
        ).settings_fingerprint

    assert evaluate((20.0, 8000.0)) != evaluate((20.0, 4000.0))


@pytest.mark.parametrize("bounds", ((8000.0, 20.0), (0.0, 8000.0), (20.0, math.inf)))
def test_broadband_range_must_be_increasing_finite_and_positive(
    bounds: tuple[float, float],
) -> None:
    receivers = fixtures._receiver_set()

    with pytest.raises(ValueError, match="broadband_range_hz"):
        evaluate_listening_area(
            receivers,
            fixtures._results(receivers),
            candidate_id=fixtures._CANDIDATE,
            speaker_id=fixtures._SPEAKER,
            timbre_settings_fingerprint=fixtures._SETTINGS,
            scene_fingerprint=fixtures._SCENE_FINGERPRINT,
            feature_match_tolerance_hz=10.0,
            broadband_range_hz=bounds,
        )


def test_level_is_computed_over_the_range_the_caller_passed() -> None:
    """範圍 20–150 Hz 只剩 100 Hz 那一點：主位 (4,1) 對平的周圍點差 10·log10(4)；20–8000 Hz 兩點等寬是
    10·log10(2.5)。評估器若不用呼叫端傳的範圍（例如寫死 20–8000），這題會紅。"""
    receivers = _two_points()
    axis = (100.0, 200.0)
    results = (
        _result(receivers, "main", axis, (4.0, 1.0)),
        _result(receivers, "front", axis, _flat(axis)),
    )

    def level_difference(bounds: tuple[float, float]) -> float:
        evaluation = evaluate_listening_area(
            receivers,
            results,
            candidate_id=fixtures._CANDIDATE,
            speaker_id=fixtures._SPEAKER,
            timbre_settings_fingerprint=fixtures._SETTINGS,
            scene_fingerprint=fixtures._SCENE_FINGERPRINT,
            feature_match_tolerance_hz=10.0,
            broadband_range_hz=bounds,
        )
        comparison = fixtures._payload(evaluation).overall_level_stability
        return comparison.primary_to_surrounding.weighted_mean_deviation

    assert level_difference((20.0, 150.0)) == pytest.approx(10.0 * math.log10(4.0))
    assert level_difference((20.0, 8000.0)) == pytest.approx(10.0 * math.log10(2.5))


def test_axes_that_differ_only_outside_the_range_are_still_comparable() -> None:
    """兩點只在 8000 Hz 以上不一樣（9000 對 10000 Hz）：範圍內是同一條軸，照算不拒收。
    把檢查改成「整條軸不同就拒收」這題會紅（找碴席抓到的盲點）。"""
    receivers = _two_points()
    main_axis, front_axis = (100.0, 200.0, 9000.0), (100.0, 200.0, 10000.0)
    evaluation = fixtures._evaluate(
        receivers,
        (
            _result(receivers, "main", main_axis, _flat(main_axis)),
            _result(receivers, "front", front_axis, _flat(front_axis)),
        ),
    )

    assert evaluation.state is EvaluationState.MEASURED


@pytest.mark.parametrize(
    ("energy", "axis"),
    # 下溢那一例要兩點夠近：格子只有約 0.007 八度寬，5e-324 乘上去才會變成 0。
    ((1e308, (100.0, 200.0)), (5e-324, (100.0, 100.5))),
)
def test_extreme_but_valid_energies_still_give_a_finite_answer(
    energy: float, axis: tuple[float, ...]
) -> None:
    """輸入驗證收任意有限正值；直接加總時 1e308 溢位、5e-324 下溢，評估器會當掉（找碴席抓到）。
    先除以最大值再加總就算得出來：兩點差等於 |10·log10(energy)|。"""

    result = _level_difference((axis, (energy, energy)), (axis, _flat(axis)))

    assert result == pytest.approx(abs(10.0 * math.log10(energy)))
