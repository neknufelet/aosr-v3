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
缺不缺點照輸入宣告的低頻軸身分現算：計分範圍裡每個宣告點、每個座位都要有正值，少一列或是 None
就不可估，不靠相鄰間距互相參照（連刪兩列會互相當參考）。宣告軸本身蓋不蓋得住計分範圍，只把上下界
當虛擬點再判（不要求末點剛好到上界：每八度 48 點的搜尋軸止於 297.71 Hz）；不內插、不外插。
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from aosr.config.frequency_axis import LowFrequencyAxis, low_frequency_axis_frequencies
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


def _declared_in_range(axis: LowFrequencyAxis, bounds: FrequencyRange) -> tuple[float, ...]:
    """宣告軸有限元素那一段落在計分範圍裡（含端點）的頻率。"""
    fem_frequencies, _ = low_frequency_axis_frequencies(axis)
    return tuple(frequency for frequency in fem_frequencies if bounds[0] <= frequency <= bounds[1])


def _scoped(seat: StageOneSeatCurve, bounds: FrequencyRange) -> dict[float, float | None]:
    """計分範圍裡的每一列（含 None）；範圍外的點連值都不看。None 留著，是「明確缺值」的證據。"""
    return {
        frequency: energy
        for frequency, energy in zip(seat.frequencies_hz, seat.fem_energy, strict=True)
        if bounds[0] <= frequency <= bounds[1]
    }


def _unavailable(speaker_id: str, reason: StageOneReasonCode) -> StageOneSpeakerEvaluation:
    return StageOneSpeakerEvaluation(
        speaker_id=speaker_id, state="unavailable", payload=None, reason_codes=(reason,)
    )


def _reason_not_measurable(
    rows: Sequence[dict[float, float | None]],
    declared: tuple[float, ...],
    bounds: FrequencyRange,
) -> StageOneReasonCode | None:
    """七條曲線能不能量；能量回 None。順序：多出宣告軸沒有的頻率、宣告點缺值、能量取不了 dB、
    點太少、宣告軸蓋不住計分範圍。"""
    declared_set = set(declared)
    if any(set(seat_rows) - declared_set for seat_rows in rows):
        return StageOneReasonCode.FREQUENCY_AXIS_MISMATCH
    if any(seat_rows.get(frequency) is None for seat_rows in rows for frequency in declared):
        return StageOneReasonCode.MISSING_DECLARED_POINT
    if not all(
        math.isfinite(value) and value > 0.0
        for seat_rows in rows
        for value in seat_rows.values()
        if value is not None
    ):
        return StageOneReasonCode.NON_POSITIVE_ENERGY
    # 兩點才定得出一條直線：這是算術上的下限，不是品質門檻。
    if len(declared) < 2:
        return StageOneReasonCode.INSUFFICIENT_POINTS
    with_bounds = np.asarray((bounds[0], *declared, bounds[1]), dtype=np.float64)
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
    speaker: StageOneSpeakerCurves, declared: tuple[float, ...], bounds: FrequencyRange
) -> StageOneSpeakerEvaluation:
    seats = (speaker.primary, *speaker.surrounding)
    rows = [_scoped(seat, bounds) for seat in seats]
    reason = _reason_not_measurable(rows, declared, bounds)
    if reason is not None:
        return _unavailable(speaker.speaker_id, reason)
    # 上面已確認每個宣告點都有正值，這裡照宣告軸的順序取，型別上把 None 排掉。
    scoped = [
        tuple(value for value in (seat_rows[frequency] for frequency in declared) if value is not None)
        for seat_rows in rows
    ]
    frequencies = np.asarray(declared, dtype=np.float64)
    octaves = np.log2(frequencies)
    mask = np.ones(len(frequencies), dtype=np.bool_)
    weights = timbre._weights_in_range(octaves, mask, bounds)
    levels = [10.0 * np.log10(np.asarray(item, dtype=np.float64)) for item in scoped]
    primary_level = levels[0]
    slope, intercept = timbre._fit_line(octaves, primary_level, weights)
    residual = primary_level - (slope * octaves + intercept)
    features = timbre._features(frequencies, residual, _NO_MIN_FEATURE_WIDTH_OCTAVE)
    directions = tuple(
        _direction_difference(seat, level, primary_level, frequencies, weights)
        for seat, level in zip(speaker.surrounding, levels[1:], strict=True)
    )
    # 前後左右上下六點等權平均（老闆原話「再取六點平均」），不照座位重要度加權；六個方向由輸入模型守。
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
    declared = _declared_in_range(data.low_frequency_axis, bounds)
    return StageOneLowFrequencyEvaluation(
        evaluator_version=STAGE_ONE_LOW_FREQUENCY_EVALUATOR_VERSION,
        candidate_id=data.candidate_id,
        energy_column="fem_energy",
        low_frequency_axis=data.low_frequency_axis,
        scoring_range_hz=bounds,
        primary_receiver_id=data.speakers[0].primary.receiver_id,
        assumption=IDEAL_SOURCE_ASSUMPTION,
        assumption_text=IDEAL_SOURCE_ASSUMPTION_TEXT,
        speakers=tuple(_evaluate_speaker(speaker, declared, bounds) for speaker in data.speakers),
    )
