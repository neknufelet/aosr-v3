"""替身頻率軸夠不夠算這一次的計分（#984，顧問 #971）：``axis_reason_not_measurable`` 沿用低頻評估器的宣告點與缺段判法。

每一例同時拿同一條軸、平坦曲線交給評估器本身（``evaluate_stage_one_low_frequency``），兩邊的理由碼要相同：
公開函式不是另一套規則。合法的每八度 48 點軸止於 297.71 Hz，照樣要過（不是「末點要到 300 Hz」）。
"""
from __future__ import annotations

import numpy as np
import pytest

from aosr.config.frequency_axis import FEM_LANE_FREQUENCIES_HZ, SEARCH_48_FEM_FREQUENCIES_HZ, LowFrequencyAxis
from aosr.scoring.stage_one_low_frequency import axis_reason_not_measurable
from aosr.scoring.stage_one_low_frequency_contract import StageOneReasonCode
from tests.engine._stage_one_low_frequency_curves import SCORING_RANGE_HZ, candidate, evaluate, speaker

SEARCH, SEARCH_48 = LowFrequencyAxis.SEARCH, LowFrequencyAxis.SEARCH_48
_MIDDLE = len(FEM_LANE_FREQUENCIES_HZ) // 2
# 只蓋得到宣告軸一個點的計分範圍：從 24 點軸中間那一點到它往上半格（下一點在一格之外）。
_ONE_POINT_RANGE = (FEM_LANE_FREQUENCIES_HZ[_MIDDLE], FEM_LANE_FREQUENCIES_HZ[_MIDDLE] * 2.0 ** (1.0 / 48.0))
# 計分上界拉到 400 Hz：宣告軸止於 300 Hz 以下，上界當虛擬點判出缺段（同低頻公式考卷那一題）。
_WIDE_RANGE = (SCORING_RANGE_HZ[0], 400.0)

CASES: dict[str, tuple[tuple[float, ...], LowFrequencyAxis, tuple[float, float], StageOneReasonCode | None]] = {
    "advisor_narrow_80_100_125": ((80.0, 100.0, 125.0), SEARCH, SCORING_RANGE_HZ,
                                  StageOneReasonCode.FREQUENCY_AXIS_MISMATCH),
    "declared_minus_middle_point": (FEM_LANE_FREQUENCIES_HZ[:_MIDDLE] + FEM_LANE_FREQUENCIES_HZ[_MIDDLE + 1:], SEARCH,
                                    SCORING_RANGE_HZ, StageOneReasonCode.MISSING_DECLARED_POINT),
    "one_declared_point": ((_ONE_POINT_RANGE[0],), SEARCH, _ONE_POINT_RANGE, StageOneReasonCode.INSUFFICIENT_POINTS),
    "declared_axis_short_of_range": (SEARCH_48_FEM_FREQUENCIES_HZ, SEARCH_48, _WIDE_RANGE,
                                     StageOneReasonCode.SCORING_RANGE_GAP),
    "full_search_48": (SEARCH_48_FEM_FREQUENCIES_HZ, SEARCH_48, SCORING_RANGE_HZ, None),
    "full_search_24": (FEM_LANE_FREQUENCIES_HZ, SEARCH, SCORING_RANGE_HZ, None),
}


@pytest.mark.parametrize("name", tuple(CASES))
def test_axis_reason_matches_declared_point_and_gap_rules(name: str) -> None:
    """四種理由各一例、兩條完整合法軸各一例：理由碼要是預期那一個，而且等於評估器對同一條軸的判決。"""
    frequencies, axis, bounds, expected = CASES[name]
    assert axis_reason_not_measurable(frequencies, axis, bounds) == expected
    flat = np.asarray(frequencies, dtype=np.float64)
    judged = evaluate(candidate(speaker(np.full(len(flat), 70.0), frequencies=flat), axis=axis), bounds).speakers[0]
    assert judged.reason_codes == (() if expected is None else (expected,))


def test_search_48_axis_stops_below_upper_bound_and_still_passes() -> None:
    """控制組的邊界：48 點軸末點在計分上界之下（約 297.71 Hz），只看「末點 ≥ 上界」的判法會把它拒掉。"""
    assert SEARCH_48_FEM_FREQUENCIES_HZ[-1] < SCORING_RANGE_HZ[1]
    assert axis_reason_not_measurable(SEARCH_48_FEM_FREQUENCIES_HZ, SEARCH_48, SCORING_RANGE_HZ) is None


def test_points_outside_scoring_range_are_not_looked_at() -> None:
    """範圍外多交的點不算多出宣告軸的點：完整 24 點軸外加 10 Hz 與 1000 Hz 照樣過。"""
    frequencies = (10.0, *FEM_LANE_FREQUENCIES_HZ, 1000.0)
    assert axis_reason_not_measurable(frequencies, SEARCH, SCORING_RANGE_HZ) is None
