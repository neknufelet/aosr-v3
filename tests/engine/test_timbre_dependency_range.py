"""音色計分依賴範圍的行為與三處頻率上限一致性考卷（票 #409）。

曲線直接走正式 1/24 八度細軸，不跑物理；登記簿改動只發生在 pytest 的暫存目錄。
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pytest

from aosr.config.capabilities import Capability, load_capabilities
from aosr.config.frequency_axis import (
    GEOMETRIC_LANE_FREQUENCIES_HZ,
    GEOMETRIC_REPORT_OCTAVE_CENTERS_HZ,
)
from aosr.config.paths import config_path
from aosr.config.quality_targets import SettingEntry, load_quality_targets
from aosr.scoring import timbre
from aosr.scoring.contract import (
    CategoryEvaluation,
    EvaluationState,
    Flag,
    InputProvenance,
    ModelValidationStatus,
    ReasonCode,
    TimbrePayload,
)


_QUALITY_TARGETS = config_path("quality_targets.toml")
_CAPABILITIES = config_path("capabilities.toml")
_PURPOSE = "dedicated_two_channel_listening_room"
_PROVENANCE = InputProvenance(
    report_id="dependency-range-test",
    engine_commit="0123456789abcdef",
    speaker_id="left",
    receiver_id="seat-a",
)
DbCurve = Callable[[np.ndarray], np.ndarray]


def _setting(key: str, registry: Path = _QUALITY_TARGETS) -> SettingEntry:
    entry = load_quality_targets(registry).purpose(_PURPOSE).entry(key)
    assert isinstance(entry, SettingEntry)
    return entry


def _range_setting(
    key: str, registry: Path = _QUALITY_TARGETS
) -> tuple[float, float]:
    value = _setting(key, registry).value
    assert isinstance(value, tuple)
    return float(value[0]), float(value[1])


def _expected_dependency_range(
    range_key: str, width_key: str, registry: Path = _QUALITY_TARGETS
) -> tuple[float, float]:
    """從考卷側獨立算半窗期望值，抓把整窗當半窗的變形（捨入護欄小到比對界線量不出來，這裡守不到）。"""
    bounds = _range_setting(range_key, registry)
    width = _setting(width_key, registry).value
    assert isinstance(width, int | float)
    if float(width) <= 0.0:
        # 不平滑沒有視窗，也就沒有護欄：依賴範圍就是計分範圍本身。
        return bounds
    half_width = 0.5 * float(width) + timbre._WINDOW_ROUNDING_OCT
    factor = 2.0**half_width
    return bounds[0] / factor, bounds[1] * factor


def _input_on_axis(
    frequencies_hz: tuple[float, ...], db_at_frequency: DbCurve
) -> timbre.TimbreInput:
    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    db_values = db_at_frequency(frequencies)
    return timbre.TimbreInput(
        candidate_id="candidate-a",
        scene_fingerprint="a" * 64,
        speaker_id="left",
        receiver_id="seat-a",
        source_position_m=(1.2, 1.3, 1.1),
        receiver_position_m=(4.7, 2.8, 1.4),
        frequencies_hz=frequencies_hz,
        total_energy=tuple(float(10.0 ** (value / 10.0)) for value in db_values),
        source_reference="共同聲源功率基準",
        report_flags=(),
        model_validation_status=ModelValidationStatus.VALIDATED,
        model_validation_frequency_range_hz=(frequencies_hz[0], frequencies_hz[-1]),
        provenance=_PROVENANCE,
    )


def _official_input(db_at_frequency: DbCurve) -> timbre.TimbreInput:
    return _input_on_axis(GEOMETRIC_LANE_FREQUENCIES_HZ, db_at_frequency)


def _evaluate(
    input_data: timbre.TimbreInput, registry: Path = _QUALITY_TARGETS
) -> CategoryEvaluation:
    return timbre.evaluate_timbre(
        input_data,
        purpose=_PURPOSE,
        quality_targets_path=registry,
    )


def _keep_frequencies(
    input_data: timbre.TimbreInput, predicate: Callable[[float], bool]
) -> timbre.TimbreInput:
    indices = tuple(
        index
        for index, frequency in enumerate(input_data.frequencies_hz)
        if predicate(frequency)
    )
    return input_data.model_copy(
        update={
            "frequencies_hz": tuple(input_data.frequencies_hz[index] for index in indices),
            "total_energy": tuple(input_data.total_energy[index] for index in indices),
        }
    )


def _payload(input_data: timbre.TimbreInput, registry: Path = _QUALITY_TARGETS) -> TimbrePayload:
    evaluation = _evaluate(input_data, registry)
    assert evaluation.state is EvaluationState.MEASURED
    assert isinstance(evaluation.payload, TimbrePayload)
    return evaluation.payload


def _registry_with_ripple_upper(tmp_path: Path, upper_hz: float) -> Path:
    """造一份起伏上緣較低的合法設定，單獨壓測傾斜平滑半窗。"""
    lower_hz, current_upper_hz = _range_setting("timbre_balance.ripple_range_hz")
    original = _QUALITY_TARGETS.read_text(encoding="utf-8")
    old = (
        'key = "timbre_balance.ripple_range_hz"\n'
        f"value = [{lower_hz:.1f}, {current_upper_hz:.1f}]"
    )
    new = (
        'key = "timbre_balance.ripple_range_hz"\n'
        f"value = [{lower_hz:.1f}, {upper_hz:.1f}]"
    )
    assert old in original
    changed_registry = tmp_path / "quality_targets.toml"
    changed_registry.write_text(original.replace(old, new, 1), encoding="utf-8")
    return changed_registry


def test_missing_upper_smoothing_half_window_rejects_the_measured_counterexample(
    tmp_path: Path,
) -> None:
    """若缺段只看名義計分範圍，刪掉傾斜上緣外資料會把真實傾斜洗淡後照樣進榜。"""
    _tilt_lower_hz, tilt_upper_hz = _range_setting(
        "timbre_balance.tilt_fit_range_hz"
    )
    registry = _registry_with_ripple_upper(tmp_path, tilt_upper_hz)
    _dependency_lower_hz, dependency_upper_hz = _expected_dependency_range(
        "timbre_balance.tilt_fit_range_hz",
        "timbre_balance.smoothing_width_octave_tilt",
        registry,
    )
    full = _official_input(
        lambda frequencies: np.where(
            (frequencies >= tilt_upper_hz)
            & (frequencies <= dependency_upper_hz),
            15.0,
            0.0,
        )
    )

    complete = _evaluate(full, registry)
    truncated = _evaluate(
        _keep_frequencies(full, lambda frequency: frequency <= tilt_upper_hz),
        registry,
    )

    assert complete.state is EvaluationState.MEASURED
    assert isinstance(complete.payload, TimbrePayload)
    assert complete.payload.tilt_db_per_octave > 0.0
    assert truncated.state is EvaluationState.UNAVAILABLE
    assert truncated.reason_codes == (ReasonCode.TIMBRE_SCORING_RANGE_GAP,)


def test_data_ending_just_inside_the_upper_dependency_edge_is_unavailable(
    tmp_path: Path,
) -> None:
    """最後一點落在傾斜計分上界與依賴上界之間：離上界的距離小於缺段門檻，

    只靠「邊界當虛擬點」判不出來，要靠「資料必須完整包住依賴範圍」那半條才擋得住。
    """
    _tilt_lower_hz, tilt_upper_hz = _range_setting(
        "timbre_balance.tilt_fit_range_hz"
    )
    registry = _registry_with_ripple_upper(tmp_path, tilt_upper_hz)
    _dependency_lower_hz, dependency_upper_hz = _expected_dependency_range(
        "timbre_balance.tilt_fit_range_hz",
        "timbre_balance.smoothing_width_octave_tilt",
        registry,
    )
    inside_upper_hz = (tilt_upper_hz + dependency_upper_hz) / 2.0
    full = _official_input(lambda frequencies: np.zeros_like(frequencies))
    kept = _keep_frequencies(full, lambda frequency: frequency <= inside_upper_hz)
    assert kept.frequencies_hz[-1] > tilt_upper_hz

    evaluation = _evaluate(kept, registry)

    assert evaluation.state is EvaluationState.UNAVAILABLE
    assert evaluation.reason_codes == (ReasonCode.TIMBRE_SCORING_RANGE_GAP,)


def test_missing_lower_smoothing_half_window_is_unavailable_without_losing_scored_points() -> None:
    """若下側依賴半窗沒守，刪掉傾斜名義下界以下全部資料仍會錯當成完整輸入。

    用傾斜那一段而不是起伏：起伏不平滑（寬度 0）沒有半窗，拿它當例子只會靠捨入護欄那根頭髮
    誤紅（票 #432 之後這題原本就是靠那根頭髮過的）。
    """
    tilt_lower_hz, _ = _range_setting("timbre_balance.tilt_fit_range_hz")
    tilt_width = _setting("timbre_balance.smoothing_width_octave_tilt").value
    assert isinstance(tilt_width, float) and tilt_width > 0.0
    full = _official_input(np.zeros_like)
    truncated = _keep_frequencies(full, lambda frequency: frequency >= tilt_lower_hz)

    evaluation = _evaluate(truncated)

    full_scored = tuple(frequency for frequency in full.frequencies_hz if frequency >= tilt_lower_hz)
    assert truncated.frequencies_hz == full_scored
    assert evaluation.state is EvaluationState.UNAVAILABLE
    assert evaluation.reason_codes == (ReasonCode.TIMBRE_SCORING_RANGE_GAP,)


def test_complete_official_axis_flat_curve_meets_the_registered_coverage() -> None:
    """正式細軸已包住診斷與計分依賴範圍，不得再掛覆蓋不足。"""
    evaluation = _evaluate(_official_input(np.zeros_like))

    assert evaluation.state is EvaluationState.MEASURED
    assert evaluation.reason_codes == ()
    assert Flag.DATA_COVERAGE_SHORT not in evaluation.flags
    assert Flag.UNVALIDATED not in evaluation.flags


def test_wider_registered_ripple_smoothing_widens_payload_and_rejects_old_data(
    tmp_path: Path,
) -> None:
    """若依賴範圍不是每次從登記簿現算，改寬起伏平滑後輸出與可估性都不會跟著動。"""
    original_payload = _payload(_official_input(np.zeros_like))
    dependency_ranges = (
        original_payload.tilt_dependency_range_hz,
        original_payload.ripple_dependency_range_hz,
    )
    lower_hz = min(bounds[0] for bounds in dependency_ranges)
    upper_hz = max(bounds[1] for bounds in dependency_ranges)
    just_enough_axis = tuple(float(value) for value in np.geomspace(lower_hz, upper_hz, 241))
    just_enough = _input_on_axis(just_enough_axis, np.zeros_like)
    before = _payload(just_enough)

    original = _QUALITY_TARGETS.read_text(encoding="utf-8")
    old = (
        'key = "timbre_balance.smoothing_width_octave_ripple"\n'
        "value = 0.0\n"
        'unit = "oct"'
    )
    new = (
        'key = "timbre_balance.smoothing_width_octave_ripple"\n'
        "value = 1.0\n"
        'unit = "oct"'
    )
    assert old in original
    changed_registry = tmp_path / "quality_targets.toml"
    changed_registry.write_text(original.replace(old, new, 1), encoding="utf-8")

    widened = _evaluate(just_enough, changed_registry)

    assert before.ripple_dependency_range_hz == original_payload.ripple_dependency_range_hz
    assert widened.state is EvaluationState.UNAVAILABLE
    assert widened.reason_codes == (ReasonCode.TIMBRE_SCORING_RANGE_GAP,)
    expected_changed = _expected_dependency_range(
        "timbre_balance.ripple_range_hz",
        "timbre_balance.smoothing_width_octave_ripple",
        changed_registry,
    )
    wider_axis = tuple(
        float(value)
        for value in np.geomspace(expected_changed[0], expected_changed[1], 481)
    )
    changed_full = _payload(_input_on_axis(wider_axis, np.zeros_like), changed_registry)
    assert changed_full.ripple_dependency_range_hz[0] < before.ripple_dependency_range_hz[0]
    assert changed_full.ripple_dependency_range_hz[1] > before.ripple_dependency_range_hz[1]
    assert changed_full.ripple_dependency_range_hz == pytest.approx(expected_changed)


def test_validated_nominal_scoring_range_does_not_cover_dependency_ranges() -> None:
    """若能力判斷只看部分名義計分範圍，少了其餘依賴資料的報表會冒充已驗過。"""
    ripple_lower_hz, _ripple_upper_hz = _range_setting(
        "timbre_balance.ripple_range_hz"
    )
    _tilt_lower_hz, tilt_upper_hz = _range_setting(
        "timbre_balance.tilt_fit_range_hz"
    )
    input_data = _official_input(np.zeros_like).model_copy(
        update={
            "model_validation_frequency_range_hz": (
                ripple_lower_hz,
                tilt_upper_hz,
            )
        }
    )

    evaluation = _evaluate(input_data)

    assert evaluation.state is EvaluationState.MEASURED
    assert Flag.UNVALIDATED in evaluation.flags


def test_registered_dependency_ranges_fit_inside_the_official_frequency_axis() -> None:
    """品質登記簿若只拉高計分範圍、沒先拉正式細軸，完整報表會在合併前變紅。"""
    payload = _payload(_official_input(np.zeros_like))
    expected = (
        _expected_dependency_range(
            "timbre_balance.tilt_fit_range_hz",
            "timbre_balance.smoothing_width_octave_tilt",
        ),
        _expected_dependency_range(
            "timbre_balance.ripple_range_hz",
            "timbre_balance.smoothing_width_octave_ripple",
        ),
    )

    actual = (
        payload.tilt_dependency_range_hz,
        payload.ripple_dependency_range_hz,
    )
    for actual_bounds, expected_bounds in zip(actual, expected, strict=True):
        assert actual_bounds == pytest.approx(expected_bounds)
        lower_hz, upper_hz = actual_bounds
        assert GEOMETRIC_LANE_FREQUENCIES_HZ[0] <= lower_hz
        assert upper_hz <= GEOMETRIC_LANE_FREQUENCIES_HZ[-1]


def test_capability_rows_declaring_the_fine_axis_track_its_rounded_endpoints() -> None:
    """細軸條目的程式判準：宣告的上界越過最高那個報表八度帶中心的能力列。

    逐帶的入口最多宣告到最高的帶中心；宣告得比它高，就是在宣告整條細軸。這類能力列的下界
    要等於正式細軸第一點（寫 20.5 就紅）；表上上界是整數取整，所以不可超過軸末點，
    也不可比軸末點低一個完整 Hz。條目名稱與數量都不寫死。
    """
    table = load_capabilities(_CAPABILITIES)
    fine_axis_rows: tuple[Capability, ...] = tuple(
        capability
        for entry in table.entry
        for capability in entry.capability
        if capability.frequency_hz[1] > max(GEOMETRIC_REPORT_OCTAVE_CENTERS_HZ)
    )

    assert fine_axis_rows
    for capability in fine_axis_rows:
        lower_hz, upper_hz = capability.frequency_hz
        assert lower_hz == GEOMETRIC_LANE_FREQUENCIES_HZ[0]
        assert upper_hz <= GEOMETRIC_LANE_FREQUENCIES_HZ[-1]
        assert GEOMETRIC_LANE_FREQUENCIES_HZ[-1] - upper_hz < 1.0


def test_raw_ripple_data_ending_exactly_at_the_scoring_bounds_is_measured() -> None:
    """起伏不平滑（寬度 0）時，資料剛好量到計分範圍兩端就算完整。

    護欄若仍把依賴上界推高一根頭髮，量到 8000.0 Hz 整的曲線會被判「計分範圍有缺」而不可估——
    量測曲線常常就停在整數上限。
    """
    ripple_width = _setting("timbre_balance.smoothing_width_octave_ripple").value
    assert ripple_width == 0.0
    ripple_lower_hz, ripple_upper_hz = _range_setting("timbre_balance.ripple_range_hz")
    tilt_lower_hz, tilt_upper_hz = _expected_dependency_range(
        "timbre_balance.tilt_fit_range_hz",
        "timbre_balance.smoothing_width_octave_tilt",
    )
    assert ripple_lower_hz <= tilt_lower_hz
    assert tilt_upper_hz <= ripple_upper_hz
    axis = tuple(
        float(value) for value in np.geomspace(ripple_lower_hz, ripple_upper_hz, 481)
    )
    assert axis[0] == ripple_lower_hz
    assert axis[-1] == ripple_upper_hz

    evaluation = _evaluate(_input_on_axis(axis, np.zeros_like))

    assert evaluation.state is EvaluationState.MEASURED
    assert isinstance(evaluation.payload, TimbrePayload)
    assert evaluation.payload.ripple_dependency_range_hz == (ripple_lower_hz, ripple_upper_hz)
    assert Flag.UNVALIDATED not in evaluation.flags
