"""音色判「資料中間缺一段」用平滑實際那一把半窗（票 #401）。

有平滑時，相鄰兩點離得比平滑半窗（含捨入護欄）還遠，視窗就收不到隔壁那一點、平滑值只剩自己這一側——
判洞跟平滑共用 ``timbre._smoothing_half_width_octave``，不另外寫死 1/6 八度。不平滑（寬度 0）那一支照舊。
曲線直接造在指定的軸上，不跑物理；登記簿改動只發生在 pytest 的暫存目錄。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aosr.config.frequency_axis import frequency_axis
from aosr.scoring import timbre
from aosr.scoring.channel_matching_settings import _load_settings
from aosr.scoring.contract import EvaluationState, Flag, ReasonCode
from tests.engine import test_channel_matching as channel_fixtures
from tests.engine.test_scoring_timbre import _registry_with
from tests.engine.test_timbre_dependency_range import _evaluate, _input_on_axis, _setting

_RIPPLE_WIDTH = 'key = "timbre_balance.smoothing_width_octave_ripple"\nvalue = 0.0'
# 每個寬度各考三種相鄰距離：剛好半窗（不算缺）、超出護欄兩根頭髮、落在半窗與整窗之間（舊的尺放過它）。
_WIDTHS: dict[str, float | None] = {"registry-tilt": None, "one-octave": 1.0, "tenth-octave": 0.1}
_STEPS: dict[str, tuple[float, float, bool]] = {
    # 名字：(半窗的倍數, 再加幾根捨入護欄, 是不是洞)
    "exactly-half": (1.0, 0.0, False),
    "two-guards-past-half": (1.0, 2.0, True),
    "between-half-and-width": (1.5, 0.0, True),
}


def _tilt_width() -> float:
    """正式登記簿的傾斜平滑寬度；起伏寬度是 0，所以有平滑的判洞今天只剩傾斜那一段。"""
    value = _setting("timbre_balance.smoothing_width_octave_tilt").value
    assert isinstance(value, float) and value > 0.0
    return value


def _even_axis(per_octave: int) -> tuple[float, ...]:
    """等距分數八度軸：17 Hz 起（避開 80 Hz 這類範圍邊界剛好落在格點上），上緣蓋過起伏改平滑後外擴的依賴範圍。"""
    return frequency_axis("octave_fraction", 17.0, 12000.0, per_octave=per_octave)


def _half_window_per_octave() -> int:
    """每八度幾點，相鄰距離剛好是傾斜平滑的半窗。"""
    per_octave = round(2.0 / _tilt_width())
    assert 1.0 / per_octave == pytest.approx(0.5 * _tilt_width(), rel=1e-12)
    return per_octave


def _coarse_per_octave() -> int:
    """每八度幾點，相鄰距離落在半窗與整個平滑寬度之間：舊的尺（整個寬度）放過、新的尺（半窗）擋下。"""
    per_octave = _half_window_per_octave() * 2 // 3
    assert 0.5 * _tilt_width() < 1.0 / per_octave < _tilt_width()
    return per_octave


@pytest.mark.parametrize("step_name", list(_STEPS))
@pytest.mark.parametrize("width_name", list(_WIDTHS))
def test_gap_and_smoothing_window_share_one_half_width(width_name: str, step_name: str) -> None:
    """兩點相距多遠算洞，要跟平滑視窗收不收得到隔壁那一點同一個答案；舊的尺是整個寬度，寬一倍。"""
    width = _WIDTHS[width_name] or _tilt_width()
    half_factor, guards, expected_gap = _STEPS[step_name]
    step = half_factor * 0.5 * width + guards * timbre._WINDOW_ROUNDING_OCT
    frequencies = 2.0 ** np.asarray([9.0, 9.0 + step])
    energy = np.asarray([1.0, 4.0])

    smoothed = timbre._smooth_energy(np.log2(frequencies), energy, width)

    assert timbre._has_gap(frequencies, width) is expected_gap
    if expected_gap:
        # 視窗只剩自己一點：平滑值逐位元等於原始值。
        assert smoothed[0] == energy[0] and smoothed[1] == energy[1]
    else:
        # 收得到隔壁：兩點等權平均。
        assert smoothed == pytest.approx([2.5, 2.5], rel=1e-12)


def _windows_miss_each_other(frequencies: np.ndarray, width: float) -> bool:
    """兩點各自的平滑視窗有沒有一邊收不到另一點：收不到的那一點平滑值逐位元等於原始值。"""
    energy = np.asarray([1.0, 4.0])
    smoothed = timbre._smooth_energy(np.log2(frequencies), energy, width)
    return bool(smoothed[0] == energy[0] or smoothed[1] == energy[1])


@pytest.mark.parametrize(
    ("width", "frequencies"),
    [
        (1.0 / 3.0, (128.0, 143.6751421836012)),
        (0.1, (512.0, 530.0556410067908)),
        (1.0, tuple(2.0 ** np.asarray([9.0, 9.0 + timbre._smoothing_half_width_octave(1.0)]))),
    ],
    ids=["tilt-third-octave-one-rounding-past", "tenth-octave-one-rounding-past", "one-octave-exactly-half-plus-guard"],
)
def test_neighbours_the_smoothing_window_reaches_are_not_a_gap(
    width: float, frequencies: tuple[float, float]
) -> None:
    """審查席造的三對：先相減比半窗會多出一個捨入、或剛好相等；平滑兩邊都收得到，就不准判成洞。"""
    axis = np.asarray(frequencies)

    smoothed = timbre._smooth_energy(np.log2(axis), np.asarray([1.0, 4.0]), width)

    assert smoothed == pytest.approx([2.5, 2.5], rel=1e-12)
    assert timbre._has_gap(axis, width) is False


@pytest.mark.parametrize(
    ("width", "frequencies"),
    [(0.1, (248.9811671479444, 257.7614690453565)), (1.0, (214.43518721782374, 303.25715001346305))],
    ids=["tenth-octave-across-256-hz", "one-octave-across-256-hz"],
)
def test_one_sided_window_miss_is_a_gap(width: float, frequencies: tuple[float, float]) -> None:
    """右點的對數跨過 8（256 Hz）那一格，浮點間隔變兩倍：左點的視窗收得到右點、右點的收不到左點。

    只要有一邊收不到，那一點的平滑值就只剩自己——算洞，不准只看其中一邊。
    """
    axis = np.asarray(frequencies)

    smoothed = timbre._smooth_energy(np.log2(axis), np.asarray([1.0, 4.0]), width)

    assert smoothed[0] == pytest.approx(2.5, rel=1e-12)
    assert smoothed[1] == 4.0
    assert timbre._has_gap(axis, width) is True


@pytest.mark.parametrize("width_name", list(_WIDTHS))
@pytest.mark.parametrize("base_octave", [7.0, 9.0, 12.0])
def test_gap_matches_the_smoothing_window_at_floating_point_edges(width_name: str, base_octave: float) -> None:
    """第二點從「剛好半窗＋護欄」往兩側一次挪一個浮點間隔：每一步判洞都要跟平滑收不收得到同一個答案。"""
    width = _WIDTHS[width_name] or _tilt_width()
    edge = 2.0 ** (base_octave + timbre._smoothing_half_width_octave(width))
    second = [edge]
    for _ in range(12):
        second.insert(0, float(np.nextafter(second[0], 0.0)))
        second.append(float(np.nextafter(second[-1], np.inf)))
    answers = set()

    for frequency in second:
        axis = np.asarray([2.0**base_octave, frequency])
        expected = _windows_miss_each_other(axis, width)
        answers.add(expected)
        assert timbre._has_gap(axis, width) is expected, frequency

    assert answers == {True, False}  # 掃過的範圍真的跨過門檻


def test_even_axis_at_exactly_the_half_window_is_complete() -> None:
    """相鄰剛好半窗的等距軸：取對數相減有幾段多出一根頭髮，靠同一個捨入護欄吸收，不算缺。"""
    axis = _even_axis(_half_window_per_octave())
    octaves = np.log2(np.asarray(axis))
    alternating = np.where(np.arange(len(axis)) % 2 == 0, 1.0, 4.0)

    smoothed = timbre._smooth_energy(octaves, alternating, _tilt_width())
    evaluation = _evaluate(_input_on_axis(axis, np.zeros_like))
    settings = _load_settings(channel_fixtures._TARGETS, channel_fixtures._PURPOSE)

    assert np.max(np.diff(octaves)) > 0.5 * _tilt_width()  # 護欄真的有在吸收
    assert bool(np.all(smoothed != alternating))  # 每一點的視窗都收到鄰點
    assert evaluation.state is EvaluationState.MEASURED
    assert evaluation.reason_codes == ()
    assert Flag.DATA_COVERAGE_SHORT not in evaluation.flags
    assert timbre.timbre_ripple_coverage_reason(
        np.asarray(axis), settings.ripple_range_hz, settings.timbre
    ) is None


def test_even_axis_coarser_than_the_half_window_is_unavailable_not_a_score() -> None:
    """相鄰距離在半窗與整窗之間：傾斜平滑每一點都只剩自己，主位音色標不可估，不給分數。

    覆蓋旗標取兩個寬度較窄的那一個（起伏 0），走不平滑那一支：等距軸沒少點，不掛覆蓋不足。
    """
    axis = _even_axis(_coarse_per_octave())
    alternating = np.where(np.arange(len(axis)) % 2 == 0, 1.0, 4.0)

    smoothed = timbre._smooth_energy(np.log2(np.asarray(axis)), alternating, _tilt_width())
    evaluation = _evaluate(_input_on_axis(axis, np.zeros_like))

    assert bool(np.all(smoothed == alternating))
    assert evaluation.state is EvaluationState.UNAVAILABLE
    assert evaluation.reason_codes == (ReasonCode.TIMBRE_SCORING_RANGE_GAP,)
    assert evaluation.payload is None
    assert evaluation.raw_quantities == ()
    assert evaluation.category_cost is None
    assert Flag.DATA_COVERAGE_SHORT not in evaluation.flags


def test_channel_matching_on_the_coarse_axis_reports_the_gap_reason() -> None:
    """左右匹配的起伏差走同一把尺：同一條粗軸回「計分依賴範圍內缺段」，整類不可估。"""
    axis = _even_axis(_coarse_per_octave())
    ones = tuple(1.0 for _ in axis)
    receivers = channel_fixtures._receivers()
    group = channel_fixtures._group()
    points = tuple(
        channel_fixtures._point(
            receivers, group, receiver_id,
            left_energy=ones, right_energy=ones,
            left_frequencies_hz=axis, right_frequencies_hz=axis,
        )
        for receiver_id in ("main", "front")
    )
    settings = _load_settings(channel_fixtures._TARGETS, channel_fixtures._PURPOSE)

    evaluation = channel_fixtures._evaluate(receivers, group, points)

    assert timbre.timbre_ripple_coverage_reason(
        np.asarray(axis), settings.ripple_range_hz, settings.timbre
    ) is ReasonCode.TIMBRE_SCORING_RANGE_GAP
    assert evaluation.state is EvaluationState.UNAVAILABLE
    assert ReasonCode.TIMBRE_SCORING_RANGE_GAP in evaluation.reason_codes
    assert evaluation.category_cost is None


def test_coverage_flag_with_a_smoothed_narrower_width_uses_its_half_window(tmp_path: Path) -> None:
    """覆蓋旗標仍取兩個寬度較窄的那一個；那一個有平滑時，跟依賴範圍一樣用它的半窗判。"""
    width = _tilt_width()
    registry = _registry_with(tmp_path, _RIPPLE_WIDTH, _RIPPLE_WIDTH.replace("0.0", repr(width)))
    axis = _even_axis(_coarse_per_octave())

    evaluation = _evaluate(_input_on_axis(axis, np.zeros_like), registry)

    assert Flag.DATA_COVERAGE_SHORT in evaluation.flags
    assert evaluation.state is EvaluationState.UNAVAILABLE
    assert evaluation.reason_codes == (ReasonCode.TIMBRE_SCORING_RANGE_GAP,)


@pytest.mark.parametrize(
    ("steps", "expected_gap"),
    [
        ((0.1, 0.1, 0.14, 0.1, 0.1), False),
        ((0.1, 0.1, 0.16, 0.1, 0.1), True),
        ((0.1, 0.1, 0.2, 0.1, 0.1), True),
        ((0.2, 0.2, 0.1, 0.1, 0.1), False),
        ((0.25,) * 8, False),
    ],
    ids=["1.4x-neighbour", "1.6x-neighbour", "one-point-missing", "sparse-to-dense", "even-coarse"],
)
def test_unsmoothed_gap_rule_is_unchanged(steps: tuple[float, ...], expected_gap: bool) -> None:
    """寬度 0（起伏、峰谷、第一階段低頻都走這一支）：比左右鄰距較大者大 1.5 倍才是洞，照舊。

    等距的粗軸在這一支不算洞——它看的是鄰距的比例、不看絕對距離，所以上面粗軸那幾題只有傾斜那一段在判。
    """
    frequencies = 2.0 ** (9.0 + np.cumsum(np.asarray((0.0, *steps))))

    assert timbre._has_gap(frequencies, 0.0) is expected_gap
