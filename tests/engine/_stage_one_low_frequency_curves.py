"""第一階段低音量測三支考卷共用的造曲線工具（票 #558）：曲線全用程式造，不跑整房求解。"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from aosr.config.frequency_axis import SEARCH_48_FEM_FREQUENCIES_HZ
from aosr.scoring.stage_one_low_frequency import evaluate_stage_one_low_frequency
from aosr.scoring.stage_one_low_frequency_contract import (
    StageOneLowFrequencyEvaluation,
    StageOneLowFrequencyInput,
    StageOneSeatCurve,
    StageOneSpeakerCurves,
    StageOneSpeakerPayload,
)

FloatArray = NDArray[np.float64]
# 計分範圍由考卷自己給（評估器不寫預設）：三階段紙第一階段那條的 20～300 Hz。
SCORING_RANGE_HZ = (20.0, 300.0)
# 每八度 48 點的正式擺位搜尋軸，有限元素那一段止於 297.71 Hz。
AXIS_HZ: FloatArray = np.asarray(SEARCH_48_FEM_FREQUENCIES_HZ, dtype=np.float64)
# 主位四周六個方向（前後左右上下），照座位組 ``primary_cross`` 的方向字。
DIRECTIONS = ("front", "back", "left", "right", "up", "down")


def tilted_db(frequencies: FloatArray, slope_db_per_octave: float, level_db: float) -> FloatArray:
    """dB 對 log2 f 的直線：在 1 Hz 處是 ``level_db``。"""
    return np.asarray(level_db + slope_db_per_octave * np.log2(frequencies), dtype=np.float64)


def gaussian_db(
    frequencies: FloatArray, center_hz: float, height_db: float, width_octave: float
) -> FloatArray:
    """對數頻率上的高斯形峰（height 為負就是谷）。"""
    distance = np.log2(frequencies / center_hz)
    return np.asarray(height_db * np.exp(-0.5 * (distance / width_octave) ** 2), dtype=np.float64)


def demo_curve_db(frequencies: FloatArray = AXIS_HZ) -> FloatArray:
    """有斜率、有峰有谷的主位曲線：−2 dB／八度，55 Hz 峰 8 dB，140 Hz 谷 −6 dB。"""
    return (
        tilted_db(frequencies, -2.0, 80.0)
        + gaussian_db(frequencies, 55.0, 8.0, 0.15)
        + gaussian_db(frequencies, 140.0, -6.0, 0.1)
    )


def seat(
    receiver_id: str,
    direction: str | None,
    frequencies: Sequence[float] | FloatArray,
    level_db: Sequence[float] | FloatArray,
) -> StageOneSeatCurve:
    """一個座位的曲線：dB 換回能量（10^(dB/10)）。"""
    energy = tuple(float(10.0 ** (value / 10.0)) for value in level_db)
    return StageOneSeatCurve(
        receiver_id=receiver_id,
        direction_relative_to_primary=direction,
        report_id=f"report-{receiver_id}",
        frequencies_hz=tuple(float(value) for value in frequencies),
        fem_energy=energy,
    )


def speaker(
    primary_db: FloatArray,
    surrounding_db: Mapping[str, FloatArray] | None = None,
    *,
    frequencies: FloatArray = AXIS_HZ,
    speaker_id: str = "left",
) -> StageOneSpeakerCurves:
    """一支喇叭：沒指定的方向用跟主位一模一樣的曲線。"""
    given = surrounding_db or {}
    return StageOneSpeakerCurves(
        speaker_id=speaker_id,
        primary=seat("main", None, frequencies, primary_db),
        surrounding=tuple(
            seat(direction, direction, frequencies, given.get(direction, primary_db))
            for direction in DIRECTIONS
        ),
    )


def candidate(*speakers: StageOneSpeakerCurves) -> StageOneLowFrequencyInput:
    return StageOneLowFrequencyInput(candidate_id="candidate-a", speakers=speakers)


def evaluate(
    data: StageOneLowFrequencyInput, scoring_range_hz: tuple[float, float] = SCORING_RANGE_HZ
) -> StageOneLowFrequencyEvaluation:
    return evaluate_stage_one_low_frequency(data, scoring_range_hz=scoring_range_hz)


def measured_payload(
    data: StageOneLowFrequencyInput, scoring_range_hz: tuple[float, float] = SCORING_RANGE_HZ
) -> StageOneSpeakerPayload:
    """第一支喇叭的 payload；不可估就讓考卷紅在這裡。"""
    result = evaluate(data, scoring_range_hz).speakers[0]
    assert result.state == "measured", result.reason_codes
    assert result.payload is not None
    return result.payload


def independent_weights(frequencies: FloatArray, bounds_hz: tuple[float, float]) -> FloatArray:
    """考卷自己重算的頻率權重：每點的格子以 log2 鄰點中點為界，兩端切到範圍上下界。

    不呼叫評估器或音色模組的權重函式，用來當獨立答案。
    """
    octaves = np.log2(frequencies)
    edges = np.concatenate((
        [np.log2(bounds_hz[0])],
        0.5 * (octaves[1:] + octaves[:-1]),
        [np.log2(bounds_hz[1])],
    ))
    return np.asarray(np.diff(edges), dtype=np.float64)


def independent_own_line_answer(
    frequencies: FloatArray, level_db: FloatArray, bounds_hz: tuple[float, float]
) -> tuple[float, float, float]:
    """用 ``numpy.polyfit``（加權多項式擬合）獨立算自己那條直線與扣線後的加權均方根。"""
    inside = (frequencies >= bounds_hz[0]) & (frequencies <= bounds_hz[1])
    f, y = frequencies[inside], level_db[inside]
    weights = independent_weights(f, bounds_hz)
    x = np.log2(f)
    slope, intercept = np.polyfit(x, y, 1, w=np.sqrt(weights))
    residual = y - (slope * x + intercept)
    rms = float(np.sqrt(np.sum(weights * residual**2) / np.sum(weights)))
    return rms, float(slope), float(intercept)
