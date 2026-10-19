"""第一階段低音量測（票 #558）：每支喇叭兩個分項，只量、不合分、不排名。

照三階段紙第一階段那條老闆定的量法：
- 主項：主位曲線（dB 對 log2 f）扣它自己的最小平方直線（least squares，最小平方法擬合）後的
  加權均方根；不先平滑、計分範圍外的資料不用。
- 座位穩健性：每個周圍點各自對主位逐頻比 dB，曲線差的加權均方根，再取周圍點等權平均；
  共同聲源與音量基準，不替各位置各自扣斜率或調平；最差方向只提醒、不淘汰。
兩項的頻率權重同一組：音色評估器那套「範圍內每八度等權」的格子，切到計分範圍上下界。

計分範圍由呼叫端給（品質登記簿的條目要到合分那一支才進來），本模組不寫任何頻率或權重數字。
重用 :mod:`aosr.scoring.timbre` 的幾支純函式；不用它的平滑斜率線（``_tilt_and_line``／
``_ripple``），也不走 ``evaluate_timbre`` 那條「每條曲線除以自己最大值」的入口。
判洞只把上下界當虛擬點再判（不要求資料末點剛好到上界：每八度 48 點的搜尋軸止於 297.71 Hz），
範圍中間真的少點照樣不可估；不內插、不外插。
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from aosr.scoring import timbre
from aosr.scoring.contract_base import FrequencyRange
from aosr.scoring.stage_one_low_frequency_contract import (
    IDEAL_SOURCE_ASSUMPTION,
    IDEAL_SOURCE_ASSUMPTION_TEXT,
    STAGE_ONE_LOW_FREQUENCY_EVALUATOR_VERSION,
    SeatDirectionDifference,
    StageOneLowFrequencyEvaluation,
    StageOneLowFrequencyInput,
    StageOneReasonCode,
    StageOneSeatCurve,
    StageOneSpeakerCurves,
    StageOneSpeakerEvaluation,
    StageOneSpeakerPayload,
    WorstSeatDirection,
)

FloatArray = NDArray[np.float64]
# 峰谷不刪窄峰（feature 最小寬度 0）：照音色起伏看原始曲線那一題（票 #432），不是調出來的門檻。
_NO_MIN_FEATURE_WIDTH_OCTAVE = 0.0
# 不平滑時的判洞尺寬度：0 讓 ``timbre._has_gap`` 走「相鄰距離對鄰近間距」那一支（少一點就是洞）。
_RAW_CURVE_WIDTH_OCTAVE = 0.0


def _scoped(
    seat: StageOneSeatCurve, bounds: FrequencyRange
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """計分範圍裡有有限元素值的那幾點；範圍外的點連值都不看，沒有值的點就是少交、留給判洞。"""
    rows = [
        (frequency, energy)
        for frequency, energy in zip(seat.frequencies_hz, seat.fem_energy, strict=True)
        if bounds[0] <= frequency <= bounds[1] and energy is not None
    ]
    return tuple(row[0] for row in rows), tuple(row[1] for row in rows)


def _unavailable(speaker_id: str, reason: StageOneReasonCode) -> StageOneSpeakerEvaluation:
    return StageOneSpeakerEvaluation(
        speaker_id=speaker_id, state="unavailable", payload=None, reason_codes=(reason,)
    )


def _reason_not_measurable(
    frequencies: Sequence[tuple[float, ...]],
    energies: Sequence[tuple[float, ...]],
    bounds: FrequencyRange,
) -> StageOneReasonCode | None:
    """七條曲線能不能量；能量回 None。順序：軸對不上、能量取不了 dB、點太少、範圍裡有洞。"""
    axis = frequencies[0]
    if any(other != axis for other in frequencies[1:]):
        return StageOneReasonCode.FREQUENCY_AXIS_MISMATCH
    if not all(math.isfinite(value) and value > 0.0 for curve in energies for value in curve):
        return StageOneReasonCode.NON_POSITIVE_ENERGY
    # 兩點才定得出一條直線：這是算術上的下限，不是品質門檻。
    if len(axis) < 2:
        return StageOneReasonCode.INSUFFICIENT_POINTS
    with_bounds = np.asarray((bounds[0], *axis, bounds[1]), dtype=np.float64)
    if timbre._has_gap(with_bounds, _RAW_CURVE_WIDTH_OCTAVE):
        return StageOneReasonCode.SCORING_RANGE_GAP
    return None


def _direction_difference(
    seat: StageOneSeatCurve,
    level_db: FloatArray,
    primary_level_db: FloatArray,
    frequencies: FloatArray,
    weights: FloatArray,
) -> SeatDirectionDifference:
    """周圍點減主位的逐頻 dB 差；兩條都是原始 dB，不扣直線、不扣平均。"""
    difference = level_db - primary_level_db
    direction = seat.direction_relative_to_primary
    if direction is None:
        raise ValueError("周圍點必須帶相對主位方向")
    return SeatDirectionDifference(
        receiver_id=seat.receiver_id,
        direction_relative_to_primary=direction,
        report_id=seat.report_id,
        difference_rms_db=timbre._rms(difference, weights),
        difference_curve_db=tuple(
            (float(frequency), float(value))
            for frequency, value in zip(frequencies, difference, strict=True)
        ),
    )


def _evaluate_speaker(
    speaker: StageOneSpeakerCurves, bounds: FrequencyRange
) -> StageOneSpeakerEvaluation:
    seats = (speaker.primary, *speaker.surrounding)
    scoped = [_scoped(seat, bounds) for seat in seats]
    reason = _reason_not_measurable(
        [item[0] for item in scoped], [item[1] for item in scoped], bounds
    )
    if reason is not None:
        return _unavailable(speaker.speaker_id, reason)
    frequencies = np.asarray(scoped[0][0], dtype=np.float64)
    octaves = np.log2(frequencies)
    mask = np.ones(len(frequencies), dtype=np.bool_)
    weights = timbre._weights_in_range(octaves, mask, bounds)
    levels = [10.0 * np.log10(np.asarray(item[1], dtype=np.float64)) for item in scoped]
    primary_level = levels[0]
    slope, intercept = timbre._fit_line(octaves, primary_level, weights)
    residual = primary_level - (slope * octaves + intercept)
    features = timbre._features(frequencies, residual, _NO_MIN_FEATURE_WIDTH_OCTAVE)
    directions = tuple(
        _direction_difference(seat, level, primary_level, frequencies, weights)
        for seat, level in zip(speaker.surrounding, levels[1:], strict=True)
    )
    # 周圍點等權平均（老闆原話「再取六點平均」），不照座位重要度加權。
    robustness = math.fsum(item.difference_rms_db for item in directions) / len(directions)
    worst = max(directions, key=lambda item: item.difference_rms_db)
    payload = StageOneSpeakerPayload(
        primary_report_id=speaker.primary.report_id,
        own_line_ripple_rms_db=timbre._rms(residual, weights),
        seat_robustness_rms_db=robustness,
        directions=directions,
        worst_direction=WorstSeatDirection(
            receiver_id=worst.receiver_id,
            direction_relative_to_primary=worst.direction_relative_to_primary,
            difference_rms_db=worst.difference_rms_db,
        ),
        own_line_slope_db_per_octave=slope,
        own_line_intercept_db=intercept,
        features=features,
        strongest_peak_index=timbre._summary_index(features, "peak"),
        deepest_dip_index=timbre._summary_index(features, "dip"),
        data_range_hz=(float(frequencies[0]), float(frequencies[-1])),
    )
    return StageOneSpeakerEvaluation(
        speaker_id=speaker.speaker_id, state="measured", payload=payload, reason_codes=()
    )


def evaluate_stage_one_low_frequency(
    data: StageOneLowFrequencyInput, *, scoring_range_hz: FrequencyRange
) -> StageOneLowFrequencyEvaluation:
    """量一個候選每支喇叭的主項與座位穩健性；某一支不可估不連累其他支。

    ``scoring_range_hz`` 是計分範圍（Hz，含端點），呼叫端必給、沒有預設。只讀輸入，不改它。
    """
    lower, upper = float(scoring_range_hz[0]), float(scoring_range_hz[1])
    if not (math.isfinite(lower) and math.isfinite(upper) and 0.0 < lower < upper):
        raise ValueError("scoring_range_hz 必須是遞增的有限正頻率範圍")
    bounds = (lower, upper)
    return StageOneLowFrequencyEvaluation(
        evaluator_version=STAGE_ONE_LOW_FREQUENCY_EVALUATOR_VERSION,
        candidate_id=data.candidate_id,
        energy_column="fem_energy",
        scoring_range_hz=bounds,
        primary_receiver_id=data.speakers[0].primary.receiver_id,
        assumption=IDEAL_SOURCE_ASSUMPTION,
        assumption_text=IDEAL_SOURCE_ASSUMPTION_TEXT,
        speakers=tuple(_evaluate_speaker(speaker, bounds) for speaker in data.speakers),
    )
