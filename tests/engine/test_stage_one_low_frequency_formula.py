"""第一階段低音量測的公式考卷（票 #558，老闆第 1 題）：每題答案由構造或獨立算法得出。

主項＝主位曲線扣自己最小平方直線後的加權均方根；頻率權重是範圍內每八度等權、切到計分範圍上下界；
不平滑、範圍外資料不用、缺的頻率不補成假資料。
"""
from __future__ import annotations

import numpy as np
import pytest

from aosr.config.frequency_axis import FEM_LANE_FREQUENCIES_HZ
from aosr.scoring.stage_one_low_frequency import evaluate_stage_one_low_frequency
from aosr.scoring.stage_one_low_frequency_contract import StageOneReasonCode, StageOneSeatCurve
from tests.engine._stage_one_low_frequency_curves import (
    AXIS_HZ,
    SCORING_RANGE_HZ,
    candidate,
    demo_curve_db,
    evaluate,
    gaussian_db,
    independent_own_line_answer,
    measured_payload,
    seat,
    speaker,
    tilted_db,
)


@pytest.mark.parametrize(("slope", "level"), [(-3.0, 95.0), (0.0, 60.0), (2.5, -40.0)])
def test_pure_line_has_zero_ripple_and_reports_its_own_slope(slope: float, level: float) -> None:
    """純直線（任意斜率與音量）扣自己的直線後什麼都不剩；列出的斜率與截距就是造曲線用的那一條。"""
    payload = measured_payload(candidate(speaker(tilted_db(AXIS_HZ, slope, level))))

    assert payload.own_line_ripple_rms_db == pytest.approx(0.0, abs=1e-12)
    assert payload.own_line_slope_db_per_octave == pytest.approx(slope, abs=1e-12)
    assert payload.own_line_intercept_db == pytest.approx(level, abs=1e-9)


def test_ripple_matches_an_independent_weighted_polyfit() -> None:
    """同一條曲線拿 numpy.polyfit 配考卷自己重算、切到上下界的權重，直線與均方根都對得上。"""
    level = demo_curve_db()
    payload = measured_payload(candidate(speaker(level)))
    rms, slope, intercept = independent_own_line_answer(AXIS_HZ, level, SCORING_RANGE_HZ)

    assert rms > 1.0
    assert payload.own_line_ripple_rms_db == pytest.approx(rms, rel=1e-12)
    assert payload.own_line_slope_db_per_octave == pytest.approx(slope, rel=1e-12)
    assert payload.own_line_intercept_db == pytest.approx(intercept, rel=1e-12)


def test_scoring_range_is_the_callers_parameter() -> None:
    """計分範圍照呼叫端給的算：換一段範圍，答案跟著換成那一段的獨立答案。"""
    level = demo_curve_db()
    bounds = (40.0, 200.0)
    result = evaluate(candidate(speaker(level)), bounds)
    payload = measured_payload(candidate(speaker(level)), bounds)
    rms, slope, _ = independent_own_line_answer(AXIS_HZ, level, bounds)

    assert result.scoring_range_hz == bounds
    assert payload.own_line_ripple_rms_db == pytest.approx(rms, rel=1e-12)
    assert payload.own_line_slope_db_per_octave == pytest.approx(slope, rel=1e-12)
    inside = AXIS_HZ[(AXIS_HZ >= bounds[0]) & (AXIS_HZ <= bounds[1])]
    assert payload.data_range_hz == (float(inside[0]), float(inside[-1]))


def test_points_outside_the_scoring_range_change_nothing() -> None:
    """20 Hz 以下、300 Hz 以上塞亂數點（含沒有有限元素值、非正能量的點），整份結果逐位不變。"""
    level = demo_curve_db()
    clean = evaluate(candidate(speaker(level)))
    rng = np.random.default_rng(558)
    below = np.asarray([12.0, 15.5, 19.9])
    above = np.asarray([300.5, 340.0, 1000.0])
    frequencies = np.concatenate((below, AXIS_HZ, above))
    noisy = speaker(np.concatenate((rng.normal(0.0, 30.0, 3), level, rng.normal(0.0, 30.0, 3))),
                    frequencies=frequencies)
    outside = {0: None, 1: 0.0, len(frequencies) - 1: -1.0}

    def with_bad_outside_values(curve: StageOneSeatCurve) -> StageOneSeatCurve:
        energy = tuple(outside.get(index, value) for index, value in enumerate(curve.fem_energy))
        return curve.model_copy(update={"fem_energy": energy})

    noisy = noisy.model_copy(update={
        "primary": with_bad_outside_values(noisy.primary),
        "surrounding": tuple(with_bad_outside_values(item) for item in noisy.surrounding),
    })

    assert evaluate(candidate(noisy)).model_dump() == clean.model_dump()


@pytest.mark.parametrize("how", ["row_dropped", "value_missing"])
def test_missing_axis_point_inside_the_range_is_unavailable(how: str) -> None:
    """範圍中間少一個軸點（整列沒交，或有限元素值是 None）：不可估、帶原因，不內插補值。"""
    level = demo_curve_db()
    drop = int(np.argmin(np.abs(AXIS_HZ - 100.0)))
    if how == "row_dropped":
        keep = np.arange(len(AXIS_HZ)) != drop
        data = candidate(speaker(level[keep], frequencies=AXIS_HZ[keep]))
    else:
        full = speaker(level)

        def holed(curve: StageOneSeatCurve) -> StageOneSeatCurve:
            energy = tuple(None if index == drop else value for index, value in enumerate(curve.fem_energy))
            return curve.model_copy(update={"fem_energy": energy})

        data = candidate(full.model_copy(update={
            "primary": holed(full.primary),
            "surrounding": tuple(holed(item) for item in full.surrounding),
        }))

    result = evaluate(data).speakers[0]

    assert (result.state, result.payload, result.reason_codes) == (
        "unavailable", None, (StageOneReasonCode.SCORING_RANGE_GAP,)
    )


def test_search_axis_ending_below_the_upper_bound_is_measured_and_not_extrapolated() -> None:
    """48 點軸止於 297.71 Hz、24 點軸止於 293.44 Hz：上下界當虛擬點判洞，照樣可估；
    payload 記資料真正蓋到的範圍，不把末點說成 300 Hz。資料只到 250 Hz 才是缺段。"""
    for axis in (AXIS_HZ, np.asarray(FEM_LANE_FREQUENCIES_HZ, dtype=np.float64)):
        payload = measured_payload(candidate(speaker(demo_curve_db(axis), frequencies=axis)))
        assert axis[-1] < SCORING_RANGE_HZ[1]
        assert payload.data_range_hz == (float(axis[0]), float(axis[-1]))

    short = AXIS_HZ <= 250.0
    result = evaluate(candidate(speaker(demo_curve_db()[short], frequencies=AXIS_HZ[short])))
    assert result.speakers[0].reason_codes == (StageOneReasonCode.SCORING_RANGE_GAP,)


def test_narrow_spike_is_scored_unsmoothed_and_listed_as_a_peak() -> None:
    """一個軸點上的窄刺照算（主項等於不平滑的獨立答案），峰谷清單列得出它、摘要指到它。"""
    spike_index = int(np.argmin(np.abs(AXIS_HZ - 120.0)))
    level = tilted_db(AXIS_HZ, -1.0, 70.0)
    level[spike_index] += 6.0
    payload = measured_payload(candidate(speaker(level)))
    rms, _, _ = independent_own_line_answer(AXIS_HZ, level, SCORING_RANGE_HZ)

    assert payload.own_line_ripple_rms_db == pytest.approx(rms, rel=1e-12)
    assert payload.strongest_peak_index is not None
    strongest = payload.features[payload.strongest_peak_index]
    assert (strongest.kind, strongest.center_frequency_hz) == ("peak", float(AXIS_HZ[spike_index]))


def test_peaks_and_dips_of_the_primary_residual_are_listed() -> None:
    """主位殘差的峰谷都列出來（只列、不計分）：55 Hz 的峰、140 Hz 的谷都找得到。"""
    payload = measured_payload(candidate(speaker(demo_curve_db())))
    assert payload.strongest_peak_index is not None and payload.deepest_dip_index is not None
    peak = payload.features[payload.strongest_peak_index]
    dip = payload.features[payload.deepest_dip_index]

    assert peak.kind == "peak" and peak.center_frequency_hz == pytest.approx(55.0, rel=0.02)
    assert dip.kind == "dip" and dip.center_frequency_hz == pytest.approx(140.0, rel=0.02)


def test_ripple_grows_with_the_ripple_and_ties_stay_tied() -> None:
    """排序方向：起伏放大一倍，主項剛好放大一倍（越小越好）；兩份一模一樣的曲線分數完全相同。"""
    line = tilted_db(AXIS_HZ, -2.0, 75.0)
    bump = gaussian_db(AXIS_HZ, 70.0, 4.0, 0.2)
    small = measured_payload(candidate(speaker(line + bump)))
    large = measured_payload(candidate(speaker(line + 2.0 * bump)))
    again = measured_payload(candidate(speaker(line + bump)))

    assert large.own_line_ripple_rms_db == pytest.approx(2.0 * small.own_line_ripple_rms_db, rel=1e-12)
    assert again.model_dump() == small.model_dump()


def test_non_positive_energy_inside_the_range_is_unavailable() -> None:
    """範圍裡有一點能量是 0：取不了 dB，不可估。"""
    full = speaker(demo_curve_db())
    energy = list(full.primary.fem_energy)
    energy[10] = 0.0
    data = candidate(full.model_copy(update={
        "primary": full.primary.model_copy(update={"fem_energy": tuple(energy)})
    }))

    assert evaluate(data).speakers[0].reason_codes == (StageOneReasonCode.NON_POSITIVE_ENERGY,)


def test_seat_axes_that_differ_are_unavailable_not_interpolated() -> None:
    """只有一個周圍點少交一個軸點：七點的軸對不上，不可估，不內插去對齊。"""
    full = speaker(demo_curve_db())
    keep = np.arange(len(AXIS_HZ)) != 40
    shorter = seat("front", "front", AXIS_HZ[keep], demo_curve_db()[keep])
    data = candidate(full.model_copy(update={"surrounding": (shorter, *full.surrounding[1:])}))

    assert evaluate(data).speakers[0].reason_codes == (StageOneReasonCode.FREQUENCY_AXIS_MISMATCH,)


def test_one_unavailable_speaker_leaves_the_other_measured() -> None:
    """左右兩支各自一組：右邊不可估，左邊照算、結果跟只交左邊時一樣。"""
    left = speaker(demo_curve_db(), speaker_id="left")
    keep = AXIS_HZ <= 200.0
    right = speaker(demo_curve_db()[keep], frequencies=AXIS_HZ[keep], speaker_id="right")
    both = evaluate(candidate(left, right))
    alone = evaluate(candidate(left))

    assert {item.speaker_id: item.state for item in both.speakers} == {
        "left": "measured", "right": "unavailable"
    }
    assert both.speakers[0].model_dump() == alone.speakers[0].model_dump()


@pytest.mark.parametrize("bounds", [(300.0, 20.0), (0.0, 300.0), (20.0, float("inf"))])
def test_scoring_range_must_be_increasing_finite_positive(bounds: tuple[float, float]) -> None:
    with pytest.raises(ValueError, match="scoring_range_hz"):
        evaluate_stage_one_low_frequency(candidate(speaker(demo_curve_db())), scoring_range_hz=bounds)
