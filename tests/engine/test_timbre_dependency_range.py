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
    FEM_LANE_FREQUENCIES_HZ,
    GEOMETRIC_LANE_FREQUENCIES_HZ,
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
    """從考卷側獨立算半窗期望值，抓把整窗當半窗或漏掉捨入護欄的變形。"""
    bounds = _range_setting(range_key, registry)
    width = _setting(width_key, registry).value
    assert isinstance(width, int | float)
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


def test_missing_upper_smoothing_half_window_rejects_the_measured_counterexample() -> None:
    """若缺段只看名義計分範圍，刪掉 4000 Hz 以上資料會把真實起伏洗成零後照樣進榜。"""
    full = _official_input(
        lambda frequencies: np.where(
            (frequencies >= 4000.0) & (frequencies <= 4500.0), 15.0, 0.0
        )
    )

    complete = _evaluate(full)
    truncated = _evaluate(_keep_frequencies(full, lambda frequency: frequency <= 4000.0))

    assert complete.state is EvaluationState.MEASURED
    assert isinstance(complete.payload, TimbrePayload)
    assert complete.payload.tilt_db_per_octave > 0.0
    assert complete.payload.residual_rms_db > 0.0
    assert truncated.state is EvaluationState.UNAVAILABLE
    assert truncated.reason_codes == (ReasonCode.TIMBRE_SCORING_RANGE_GAP,)


def test_missing_lower_smoothing_half_window_is_unavailable_without_losing_scored_points() -> None:
    """若下側依賴半窗沒守，刪掉起伏名義下界以下全部資料仍會錯當成完整輸入。"""
    ripple_lower_hz, _ = _range_setting("timbre_balance.ripple_range_hz")
    full = _official_input(np.zeros_like)
    truncated = _keep_frequencies(full, lambda frequency: frequency >= ripple_lower_hz)

    evaluation = _evaluate(truncated)

    full_scored = tuple(frequency for frequency in full.frequencies_hz if frequency >= ripple_lower_hz)
    assert truncated.frequencies_hz == full_scored
    assert evaluation.state is EvaluationState.UNAVAILABLE
    assert evaluation.reason_codes == (ReasonCode.TIMBRE_SCORING_RANGE_GAP,)


def test_complete_official_axis_flat_curve_remains_measured_with_coverage_flag() -> None:
    """依賴範圍不可誤傷今天完整的正式細軸；它只因沒到診斷覆蓋上限而掛覆蓋不足。"""
    evaluation = _evaluate(_official_input(np.zeros_like))

    assert evaluation.state is EvaluationState.MEASURED
    assert evaluation.reason_codes == ()
    assert Flag.DATA_COVERAGE_SHORT in evaluation.flags
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
        "value = 0.16666666666666666\n"
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
    wider_axis = tuple(float(value) for value in np.geomspace(20.0, 8000.0, 481))
    changed_full = _payload(_input_on_axis(wider_axis, np.zeros_like), changed_registry)
    expected_changed = _expected_dependency_range(
        "timbre_balance.ripple_range_hz",
        "timbre_balance.smoothing_width_octave_ripple",
        changed_registry,
    )
    assert changed_full.ripple_dependency_range_hz[0] < before.ripple_dependency_range_hz[0]
    assert changed_full.ripple_dependency_range_hz[1] > before.ripple_dependency_range_hz[1]
    assert changed_full.ripple_dependency_range_hz == pytest.approx(expected_changed)


def test_validated_nominal_scoring_range_does_not_cover_dependency_ranges() -> None:
    """若能力判斷仍看 40–4000 Hz 名義聯集，少了兩側平滑資料的報表會冒充已驗過。"""
    input_data = _official_input(np.zeros_like).model_copy(
        update={"model_validation_frequency_range_hz": (40.0, 4000.0)}
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
    """細軸條目的程式判準：從正式細軸第一點起，且上界越過 FEM 細軸最後一點的能力列。

    這類能力列的下界要等於正式細軸第一點；表上上界是整數取整，所以不可超過軸末點，
    也不可比軸末點低一個完整 Hz。條目名稱與數量都不寫死。
    """
    table = load_capabilities(_CAPABILITIES)
    fine_axis_rows: tuple[Capability, ...] = tuple(
        capability
        for entry in table.entry
        for capability in entry.capability
        if capability.frequency_hz[0] == GEOMETRIC_LANE_FREQUENCIES_HZ[0]
        and capability.frequency_hz[1] > FEM_LANE_FREQUENCIES_HZ[-1]
    )

    assert fine_axis_rows
    for capability in fine_axis_rows:
        lower_hz, upper_hz = capability.frequency_hz
        assert lower_hz == GEOMETRIC_LANE_FREQUENCIES_HZ[0]
        assert upper_hz <= GEOMETRIC_LANE_FREQUENCIES_HZ[-1]
        assert GEOMETRIC_LANE_FREQUENCIES_HZ[-1] - upper_hz < 1.0
