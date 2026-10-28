"""音色判缺段的範圍上下界（票 #419）。

依賴範圍兩端判兩次、任一有洞就是洞：①上下界當虛擬點接在範圍內的點兩端；②改接範圍外最近的真實點。
只判①，界不落在格點上時，界內緊貼界那一點缺了，前一點到虛擬界還不到鄰距 1.5 倍就放過（正式細軸 8000 Hz
前那一點）；只判②，界外那一點離得遠時，界內緊貼界的缺點會被當成由疏變密放過。另守對目標偏差讀的範圍
（覆蓋範圍）完整落在不平滑的缺段檢查範圍內——讀實際設定，不要求兩個範圍相等，也不寫死頻率。
曲線直接造在指定的軸上，不跑物理。300 Hz 疏密交界四點是已知缺陷（接續票 #943），不是正確行為。
"""
from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from aosr.config.capabilities import load_capabilities
from aosr.config.frequency_axis import (
    GEOMETRIC_LANE_FREQUENCIES_HZ,
    LowFrequencyAxis,
    low_frequency_axis_frequencies,
)
from aosr.config.paths import config_path
from aosr.config.quality_targets import load_quality_targets
from aosr.reporting.evaluation import LoadedResult, ResultStanding, load_result
from aosr.reporting.result import SchemeResult, save_result
from aosr.scoring import timbre
from aosr.scoring.channel_matching_settings import _load_settings
from aosr.scoring.contract import EvaluationState, Flag, QualityCategory, ReasonCode, TimbrePayload
from tests.engine import _scoring_source_model_control as control
from tests.engine import test_channel_matching as channel_fixtures
from tests.engine._cross_cpu_tolerance import SCORING_ABS, SCORING_REL, assert_within, cell
from tests.engine._directivity import DIRECTIVITY
from tests.engine.test_result_axis_identity import _report
from tests.engine.test_scheme_pipeline import shared_control_result
from tests.engine.test_scoring_timbre import _registry_with
from tests.engine.test_timbre_dependency_range import _PURPOSE, _QUALITY_TARGETS, _evaluate, _input_on_axis
from tests.engine.test_timbre_gap_half_window import _tilt_width

# 合成軸的鄰距（八度），跟正式細軸同一級；被考的是界與缺點的相對位置，不是這個數。
_STEP = 1.0 / 24.0
# 控制組：三種合法軸整條、``_room_curve`` 那條曲線，主線 36970513（評估器 v10，改界判法之前）算出的
# 傾斜、起伏、對目標偏差。完整資料不碰新加的那一次判洞，三個數要跟這份答案在跨處理器評分容差內
# （決策紙 verification-split-stand-in-wiring-and-cross-cpu-tolerance.md 第 6 條③級，tests/engine/_cross_cpu_tolerance.py）。
# 原本逐位比（#955）：外部審查 10-28 在另一顆處理器一般設定下差 4.44e-16～8.88e-16（每八度 48 點與 1 Hz 兩軸），
# 同機 #944 前後的程式逐位相同；本機差 0。
_MAIN_ANSWERS: dict[LowFrequencyAxis, tuple[str, str, str]] = {
    LowFrequencyAxis.SEARCH: ("-0x1.0ea7c8e951319p+0", "0x1.330f0115c3a52p+1", "0x1.ace3ba86ef771p+1"),
    LowFrequencyAxis.SEARCH_48: ("-0x1.0e56db257deeap+0", "0x1.32fbc804db258p+1", "0x1.ace2a019c20bep+1"),
    LowFrequencyAxis.VERIFICATION: ("-0x1.0de7d311c1436p+0", "0x1.32e135b89175bp+1", "0x1.acee1a9ac94dcp+1"),
}
# 已知缺陷（接續票 #943）：軸在 300 Hz 由密變疏，交界缺一點時缺口緊貼較疏那一側，被不平滑判法的
# 「由疏變密不算洞」放過。這是重現資料，不是正確行為；#943 修好後改成必須拒算、並從逐點掃描拿掉排除。
_KNOWN_TRANSITION_DEFECTS: dict[LowFrequencyAxis, tuple[float, ...]] = {
    LowFrequencyAxis.SEARCH_48: (297.71,),
    LowFrequencyAxis.VERIFICATION: (299.0, 300.0, 302.04),
}


def _room_curve(frequencies: np.ndarray) -> np.ndarray:
    """傾斜加兩條正弦：傾斜、起伏、對目標偏差都不是零，刪一點才看得出數字變沒變。"""
    octaves = np.log2(np.asarray(frequencies, dtype=np.float64))
    curve = -1.0 * (octaves - np.log2(1000.0)) + 3.0 * np.sin(2.7 * octaves) + 1.5 * np.sin(11.0 * octaves)
    return np.asarray(curve, dtype=np.float64)


def _settings() -> timbre.TimbreSettings:
    return timbre.load_timbre_settings(_QUALITY_TARGETS, _PURPOSE)


def _report_axis(axis: LowFrequencyAxis) -> tuple[float, ...]:
    return low_frequency_axis_frequencies(axis)[1]


def _merged(ranges: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """把重疊或相接的範圍併成一段；相接處兩邊各自判洞，取或之後接縫上的缺點兩段都看得到。"""
    merged: list[tuple[float, float]] = []
    for low, high in sorted(ranges):
        if merged and low <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], high))
        else:
            merged.append((low, high))
    return merged


@pytest.mark.parametrize(
    "purpose", [item.name for item in load_quality_targets(_QUALITY_TARGETS).purposes]
)
def test_deviation_range_lies_inside_the_unsmoothed_gap_checked_ranges(purpose: str) -> None:
    """對目標偏差讀原始曲線、讀整個覆蓋範圍，它自己不判缺段；靠的是不平滑那幾段依賴範圍把它整段包住。

    今天兩個範圍剛好相等，是兩個暫定值湊上的，不是規則；這一題讀實際設定守「包住」，不要求相等。
    日後若正式決策要把兩個範圍拆開，要同時給偏差自己的資料完整性檢查，不能只刪這一題（老闆 10-27）。
    """
    settings = timbre.load_timbre_settings(_QUALITY_TARGETS, purpose)
    unsmoothed = [
        bounds
        for bounds, width in zip(
            timbre._dependency_ranges(settings), timbre._dependency_gap_widths(settings), strict=True
        )
        if width <= 0.0
    ]
    coverage = settings.coverage_range_hz

    assert unsmoothed, "沒有任何一段依賴範圍走不平滑判法"
    assert any(low <= coverage[0] and coverage[1] <= high for low, high in _merged(unsmoothed)), (
        f"覆蓋範圍 {coverage} 沒被不平滑的缺段檢查範圍 {_merged(unsmoothed)} 整段包住"
    )


def _off_grid_axis(side: str, offset: float) -> tuple[np.ndarray, tuple[float, float], int]:
    """等距軸，指定那一側的界落在格點之間：界內緊貼界那一點離界 ``offset`` 個鄰距。回軸、界、那一點的位置。"""
    octaves = 9.0 + _STEP * np.arange(40)
    if side == "lower":
        inner = 5
        bounds = (octaves[inner] - offset * _STEP, octaves[-6])
    else:
        inner = 34
        bounds = (octaves[5], octaves[inner] + offset * _STEP)
    return 2.0**octaves, (2.0 ** bounds[0], 2.0 ** bounds[1]), inner


@pytest.mark.parametrize("offset", [0.0, 0.25, 0.45])
@pytest.mark.parametrize("side", ["lower", "upper"])
def test_missing_point_next_to_a_bound_is_a_gap(side: str, offset: float) -> None:
    """界內緊貼界那一點缺了就是洞，不管界有沒有落在格點上；整條軸不缺就不是洞。

    0.25、0.45：只判虛擬界時，前一點到界只有鄰距的 1.25、1.45 倍，不到 1.5 倍、放過（正式細軸 8000 Hz
    前那一點是 1.45 倍）。0.0：界剛好在格點上，缺的就是界上那一點。
    """
    frequencies, bounds, inner = _off_grid_axis(side, offset)
    missing = np.delete(frequencies, inner)

    assert timbre._dependency_range_has_gap(frequencies, bounds, 0.0) is False
    assert timbre._dependency_range_has_gap(missing, bounds, 0.0) is True


@pytest.mark.parametrize("side", ["lower", "upper"])
def test_missing_the_first_real_point_outside_a_bound_is_a_gap(side: str) -> None:
    """不平滑時，界外緊鄰那一點缺了，跨界那一步超過鄰距 1.5 倍也算洞：判不出缺的那一點落在界內還是界外，寧可不估（決策紙寫明）。"""
    frequencies, bounds, inner = _off_grid_axis(side, 0.45)
    outer = inner - 1 if side == "lower" else inner + 1

    assert timbre._dependency_range_has_gap(np.delete(frequencies, outer), bounds, 0.0) is True


@pytest.mark.parametrize("side", ["lower", "upper"])
def test_point_exactly_on_a_bound_needs_nothing_beyond_it(side: str) -> None:
    """界上剛好有一點，範圍就收齊了：界外那一點缺了不算洞（界外緊鄰算洞只發生在界落在兩點之間）。"""
    frequencies, bounds, inner = _off_grid_axis(side, 0.0)
    outer = inner - 1 if side == "lower" else inner + 1

    assert timbre._dependency_range_has_gap(np.delete(frequencies, outer), bounds, 0.0) is False


def test_coarse_point_outside_does_not_mask_a_missing_point_inside() -> None:
    """只接界外真實點的話，界外那一點離得遠，界內緊貼界的缺點會被當成由疏變密放過；虛擬界那一次要判得出來。"""
    low = 9.0
    inside = low + 0.1 * _STEP + _STEP * np.concatenate(([0.0], 2.0 + np.arange(20)))
    octaves = np.concatenate(([low - 5.0 * _STEP, low - 2.5 * _STEP], inside, [inside[-1] + _STEP]))
    bounds = (2.0**low, 2.0 ** inside[-1])

    assert timbre._dependency_range_has_gap(2.0**octaves, bounds, 0.0) is True


@pytest.mark.parametrize("side", ["lower", "upper"])
def test_smoothed_range_needs_a_real_neighbour_outside_within_its_half_window(side: str) -> None:
    """有平滑的那一段也一樣：界外最近的真實點要在界內最後一點的半窗裡，只看虛擬界會放過離太遠的。"""
    width = _tilt_width()
    half = timbre._smoothing_half_width_octave(width)
    inside = 9.0 + _STEP * np.arange(20)

    def gap(outside_distance: float) -> bool:
        if side == "lower":
            octaves = np.concatenate(([inside[0] - outside_distance * half], inside))
            bounds = (inside[0] - 0.5 * half, inside[-1])
        else:
            octaves = np.concatenate((inside, [inside[-1] + outside_distance * half]))
            bounds = (inside[0], inside[-1] + 0.5 * half)
        return timbre._dependency_range_has_gap(2.0**octaves, (2.0 ** bounds[0], 2.0 ** bounds[1]), width)

    assert gap(0.9) is False
    assert gap(1.5) is True


def test_ripple_segment_judges_gaps_with_its_own_unsmoothed_ruler(tmp_path: Path) -> None:
    """起伏那段用起伏的尺（不平滑、少一點就是洞），不借傾斜的半窗。

    今天峰谷那段（也不平滑）把起伏範圍整個包住，起伏那段借錯尺也看不出來；這裡把峰谷範圍縮到起伏下界，
    起伏範圍上半只剩起伏自己那一段在判。
    """
    settings = _settings()
    feature, ripple = settings.feature_range_hz, settings.ripple_range_hz
    key = 'key = "timbre_balance.feature_range_hz"\nvalue = '
    registry = _registry_with(
        tmp_path, f"{key}[{feature[0]!r}, {feature[1]!r}]", f"{key}[{feature[0]!r}, {ripple[0]!r}]"
    )
    tilt_high = timbre._dependency_ranges(settings)[0][1]
    lone = min(f for f in GEOMETRIC_LANE_FREQUENCIES_HZ if f > tilt_high)
    assert lone < ripple[1]

    complete = _evaluate(_input_on_axis(GEOMETRIC_LANE_FREQUENCIES_HZ, _room_curve), registry)
    missing = _evaluate(
        _input_on_axis(tuple(f for f in GEOMETRIC_LANE_FREQUENCIES_HZ if f != lone), _room_curve), registry
    )

    assert complete.state is EvaluationState.MEASURED
    assert missing.state is EvaluationState.UNAVAILABLE
    assert missing.reason_codes == (ReasonCode.TIMBRE_SCORING_RANGE_GAP,)


@pytest.mark.parametrize("axis", list(LowFrequencyAxis))
def test_complete_legal_axes_pass_every_gap_check_and_keep_main_answers(axis: LowFrequencyAxis) -> None:
    """三種合法報表軸整條：每一段依賴範圍、左右匹配那一段都不是洞，量出來的三個數跟改動前主線在跨處理器評分容差內。"""
    frequencies = np.asarray(_report_axis(axis))
    settings = _settings()
    channel = _load_settings(channel_fixtures._TARGETS, channel_fixtures._PURPOSE)

    evaluation = _evaluate(_input_on_axis(_report_axis(axis), _room_curve))

    for bounds, width in zip(
        timbre._dependency_ranges(settings), timbre._dependency_gap_widths(settings), strict=True
    ):
        assert timbre._dependency_range_has_gap(frequencies, bounds, width) is False
    assert timbre.timbre_ripple_coverage_reason(frequencies, channel.ripple_range_hz, channel.timbre) is None
    assert evaluation.state is EvaluationState.MEASURED
    assert isinstance(evaluation.payload, TimbrePayload)
    payload = evaluation.payload
    measured = (payload.tilt_db_per_octave, payload.residual_rms_db, payload.target_deviation_rms_db)
    names = ("tilt_db_per_octave", "residual_rms_db", "target_deviation_rms_db")
    assert_within(cell(f"{axis.name} {name}", value, answer, rel=SCORING_REL, abs_=SCORING_ABS)
                  for name, value, answer in zip(names, measured, _MAIN_ANSWERS[axis], strict=True))


@pytest.mark.parametrize("axis", list(LowFrequencyAxis))
def test_every_single_missing_point_the_deviation_reads_is_unavailable(axis: LowFrequencyAxis) -> None:
    """偏差讀的範圍裡任一點單獨缺了，整份音色不可估、不出分數（只扣掉接續票 #943 那幾個已知缺陷）。"""
    frequencies = _report_axis(axis)
    low, high = _settings().coverage_range_hz
    known = _KNOWN_TRANSITION_DEFECTS.get(axis, ())
    slipped = []
    for index, frequency in enumerate(frequencies):
        if not low <= frequency <= high or round(frequency, 2) in known:
            continue
        evaluation = _evaluate(_input_on_axis(frequencies[:index] + frequencies[index + 1:], _room_curve))
        if evaluation.state is not EvaluationState.UNAVAILABLE or evaluation.reason_codes != (
            ReasonCode.TIMBRE_SCORING_RANGE_GAP,
        ):
            slipped.append(frequency)

    assert slipped == []


@pytest.mark.parametrize(
    ("axis", "frequency"),
    [(axis, frequency) for axis, frequencies in _KNOWN_TRANSITION_DEFECTS.items() for frequency in frequencies],
)
def test_known_defect_943_transition_missing_point_is_still_scored(axis: LowFrequencyAxis, frequency: float) -> None:
    """已知缺陷，不是正確行為（接續票 #943）：300 Hz 疏密交界缺一點，今天照算、連覆蓋不足旗標都沒有。

    這一題紅了，表示 #943 修好了：改成「缺點必須拒算」的回歸測試，並把逐點掃描那一題的排除清單拿掉。
    """
    frequencies = _report_axis(axis)
    index = next(i for i, value in enumerate(frequencies) if round(value, 2) == frequency)

    evaluation = _evaluate(_input_on_axis(frequencies[:index] + frequencies[index + 1:], _room_curve))

    assert evaluation.state is EvaluationState.MEASURED
    assert Flag.DATA_COVERAGE_SHORT not in evaluation.flags


def _regression_cases() -> dict[str, Callable[[float], bool]]:
    """票 #419 備料實跑過的刪段、刪點、截斷，界從實際設定算，不寫死頻率；回「留下哪些點」。"""
    settings = _settings()
    tilt_high = timbre._dependency_ranges(settings)[0][1]
    coverage_high = settings.coverage_range_hz[1]
    axis = GEOMETRIC_LANE_FREQUENCIES_HZ
    past_tilt = min(f for f in axis if f > tilt_high)
    last_inside = max(f for f in axis if f <= coverage_high)
    middle = math.sqrt(tilt_high * coverage_high)
    return {
        "segment-between-tilt-dependency-and-coverage-upper": lambda f: not tilt_high < f <= middle,
        "one-point-past-tilt-dependency": lambda f: f != past_tilt,
        "truncated-just-past-tilt-dependency": lambda f: f <= past_tilt,
        "truncated-at-the-last-point-inside-coverage": lambda f: f <= last_inside,
        "last-point-before-coverage-upper": lambda f: f != last_inside,
    }


@pytest.mark.parametrize("case", list(_regression_cases()))
def test_missing_data_the_deviation_reads_is_unavailable_not_a_score(case: str) -> None:
    """偏差讀的範圍裡缺一段、缺一點、截短：音色不可估，沒有酬載、沒有原始量、沒有類代價。"""
    keep = _regression_cases()[case]

    evaluation = _evaluate(
        _input_on_axis(tuple(f for f in GEOMETRIC_LANE_FREQUENCIES_HZ if keep(f)), _room_curve)
    )

    assert evaluation.state is EvaluationState.UNAVAILABLE
    assert evaluation.reason_codes == (ReasonCode.TIMBRE_SCORING_RANGE_GAP,)
    assert evaluation.payload is None
    assert evaluation.raw_quantities == ()
    assert evaluation.category_cost is None


def test_points_past_the_first_one_outside_coverage_do_not_change_the_deviation() -> None:
    """覆蓋外第一個真實點之後的點刪光：照算，對目標偏差與偏差曲線跟整條逐位相同——偏差只讀覆蓋範圍。"""
    axis = GEOMETRIC_LANE_FREQUENCIES_HZ
    first_outside = min(f for f in axis if f >= _settings().coverage_range_hz[1])

    full = _evaluate(_input_on_axis(axis, _room_curve))
    trimmed = _evaluate(_input_on_axis(tuple(f for f in axis if f <= first_outside), _room_curve))

    assert trimmed.state is EvaluationState.MEASURED
    assert isinstance(full.payload, TimbrePayload) and isinstance(trimmed.payload, TimbrePayload)
    assert trimmed.payload.target_deviation_rms_db == full.payload.target_deviation_rms_db
    assert trimmed.payload.deviation_curve == full.payload.deviation_curve


def test_channel_matching_rejects_the_missing_point_next_to_its_upper_bound() -> None:
    """左右匹配的起伏差走同一支判洞：它那一段上界前緊貼的那一點缺了，回「計分依賴範圍內缺段」。"""
    channel = _load_settings(channel_fixtures._TARGETS, channel_fixtures._PURPOSE)
    axis = GEOMETRIC_LANE_FREQUENCIES_HZ
    last_inside = max(f for f in axis if f <= channel.ripple_range_hz[1])
    missing = np.asarray([f for f in axis if f != last_inside])

    assert timbre.timbre_ripple_coverage_reason(
        missing, channel.ripple_range_hz, channel.timbre
    ) is ReasonCode.TIMBRE_SCORING_RANGE_GAP


# 讀回時物理身分跟現在對不上的舊結果（存檔內容不必真舊，只換這一格）。
_OLDER_PHYSICS = "phys-v1:" + "0" * 64


@pytest.fixture(scope="module")
def wall_1(tmp_path_factory: pytest.TempPathFactory, worker_id: str) -> SchemeResult:
    return shared_control_result(tmp_path_factory, worker_id, "wall-1")


def _reload_older_physics(wall_1: SchemeResult, tmp_path: Path, missing: float | None) -> tuple[LoadedResult, bool]:
    """存成物理身分過期的結果，第一組報表逐點表刪掉 ``missing`` 那一點（None 不刪）；回讀回結果與存檔有沒有原封不動。"""
    path = tmp_path / "older-physics.json"
    save_result(wall_1.model_copy(update={"physics_identity": _OLDER_PHYSICS}), path)
    document = json.loads(path.read_text(encoding="utf-8"))
    report = _report(document, 0)
    points = report["points"]
    assert isinstance(points, list)
    report["points"] = [point for point in points if point["frequency_hz"] != missing]
    path.write_text(json.dumps(document), encoding="utf-8")
    before = path.read_bytes()
    loaded = load_result(path, capabilities=load_capabilities(config_path("capabilities.toml")), directivity=DIRECTIVITY,
                         quality_targets_path=control.TARGETS, physics_identity=wall_1.physics_identity)
    return loaded, path.read_bytes() == before


def _states(loaded: LoadedResult) -> dict[QualityCategory, tuple[EvaluationState, tuple[ReasonCode, ...]]]:
    return {evaluation.category: (evaluation.state, evaluation.reason_codes)
            for evaluation in loaded.result.candidate.evaluations}


def test_older_physics_readback_missing_the_point_past_the_bound_is_unavailable_not_a_score(
    wall_1: SchemeResult, tmp_path: Path
) -> None:
    """讀回不核宣告軸的那一支：物理身分過期的舊結果不逐點對宣告軸（#494 只核物理身分相同的），報表曲線直接進音色。
    界外緊鄰那一點缺了（範圍外本來就疏也一樣），邊界接真實鄰點那一次照樣判缺段——決策紙寫明的保守代價；處置是整份標
    要重算、音色與左右匹配標不可估，不變成分數，存檔不改。同一份不刪點是控制組，照量。刪的是正式搜尋軸上依賴範圍上界
    外第一個實點：只判虛擬界時前一點到界不到鄰距 1.5 倍，照算。#419 合併前掃過當時外部工作區全部存檔，沒有這種形狀。
    """
    upper = max(high for _, high in timbre._dependency_ranges(_settings()))
    first_outside = min(f for f in _report_axis(LowFrequencyAxis.SEARCH) if f >= upper)

    complete, complete_untouched = _reload_older_physics(wall_1, tmp_path, None)
    missing, missing_untouched = _reload_older_physics(wall_1, tmp_path, first_outside)

    assert complete.standing is ResultStanding.NEEDS_PHYSICS
    assert missing.standing is ResultStanding.NEEDS_PHYSICS
    assert complete_untouched and missing_untouched
    assert _states(complete)[QualityCategory.TIMBRE_BALANCE] == (EvaluationState.MEASURED, ())
    for category in (QualityCategory.TIMBRE_BALANCE, QualityCategory.CHANNEL_MATCHING):
        state, reasons = _states(missing)[category]
        assert state is EvaluationState.UNAVAILABLE
        assert ReasonCode.TIMBRE_SCORING_RANGE_GAP in reasons
