"""音色平衡第一層評估器的行為考卷（票 #345）。

曲線全部用程式造，不跑物理；範圍、平滑寬度、最小寬度一律回登記簿讀，不在這裡另寫一份。
浮點比較用 ``pytest.approx`` 的預設界線（只吸收捨入），不自訂容差。
"""
from __future__ import annotations

import inspect
import math
from pathlib import Path
from typing import Callable

import numpy as np
import pytest

from aosr.config.paths import config_path
from aosr.config.quality_targets import SettingEntry, TargetEntry, load_quality_targets
from aosr.geometry.shoebox import Point
from aosr.physics.report_io import (
    BandRow,
    CapabilitySection,
    PointRow,
    ReportOutput,
    SceneSection,
    TopFields,
)
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


_REGISTRY = config_path("quality_targets.toml")
_PURPOSE = "dedicated_two_channel_listening_room"
_CANDIDATE = "candidate-a"
_SPEAKER = "left"
_RECEIVER = "seat-a"
_SOURCE_POSITION = (1.2, 1.3, 1.1)
_POSITION = (4.7, 2.8, 1.4)
_SCENE_FINGERPRINT = "a" * 64
_PROVENANCE = InputProvenance(
    report_id="report-a",
    engine_commit="0123456789abcdef",
    speaker_id=_SPEAKER,
    receiver_id=_RECEIVER,
)
DbCurve = Callable[[np.ndarray], np.ndarray]


def _setting(key: str) -> SettingEntry:
    entry = load_quality_targets(_REGISTRY).purpose(_PURPOSE).entry(key)
    assert isinstance(entry, SettingEntry)
    return entry


def _range_setting(key: str) -> tuple[float, float]:
    value = _setting(key).value
    assert isinstance(value, tuple)
    lower, upper = value
    return lower, upper


def _scalar_setting(key: str) -> float:
    value = _setting(key).value
    assert isinstance(value, int | float)
    return float(value)


def _curve_input(
    db_at_frequency: DbCurve,
    *,
    upper_hz: float = 8000.0,
    point_count: int = 241,
    report_flags: tuple[Flag, ...] = (),
) -> timbre.TimbreInput:
    frequencies = np.geomspace(20.0, upper_hz, point_count)
    db_values = db_at_frequency(frequencies)
    return timbre.TimbreInput(
        candidate_id=_CANDIDATE,
        scene_fingerprint=_SCENE_FINGERPRINT,
        speaker_id=_SPEAKER,
        receiver_id=_RECEIVER,
        source_position_m=_SOURCE_POSITION,
        receiver_position_m=_POSITION,
        frequencies_hz=tuple(float(value) for value in frequencies),
        total_energy=tuple(float(10.0 ** (value / 10.0)) for value in db_values),
        source_reference="共同聲源功率基準",
        report_flags=report_flags,
        model_validation_status=ModelValidationStatus.VALIDATED,
        model_validation_frequency_range_hz=(20.0, 8000.0),
        provenance=_PROVENANCE,
    )


def _flat_input(level_db: float = 0.0, *, upper_hz: float = 8000.0) -> timbre.TimbreInput:
    return _curve_input(lambda frequencies: np.full_like(frequencies, level_db), upper_hz=upper_hz)


def _without_span(
    input_data: timbre.TimbreInput, lower_hz: float, upper_hz: float
) -> timbre.TimbreInput:
    kept = [
        index
        for index, frequency in enumerate(input_data.frequencies_hz)
        if not lower_hz < frequency < upper_hz
    ]
    return input_data.model_copy(
        update={
            "frequencies_hz": tuple(input_data.frequencies_hz[index] for index in kept),
            "total_energy": tuple(input_data.total_energy[index] for index in kept),
        }
    )


def _evaluate(
    input_data: timbre.TimbreInput,
    *,
    registry: Path = _REGISTRY,
    target_curve: timbre.TargetCurve | None = None,
) -> CategoryEvaluation:
    return timbre.evaluate_timbre(
        input_data, purpose=_PURPOSE, quality_targets_path=registry, target_curve=target_curve
    )


def _payload(
    input_data: timbre.TimbreInput,
    *,
    registry: Path = _REGISTRY,
    target_curve: timbre.TargetCurve | None = None,
) -> TimbrePayload:
    evaluation = _evaluate(input_data, registry=registry, target_curve=target_curve)
    assert evaluation.state == EvaluationState.MEASURED
    assert isinstance(evaluation.payload, TimbrePayload)
    return evaluation.payload


def _gaussian_feature(center_hz: float, depth_db: float, width_octave: float) -> DbCurve:
    """對數頻率上的高斯形 dB 起伏；``width_octave`` 是半深度處的全寬（照票上寬度的定義）。"""

    def curve(frequencies: np.ndarray) -> np.ndarray:
        distance = np.log2(frequencies / center_hz)
        return depth_db * np.exp(-4.0 * math.log(2.0) * (distance / width_octave) ** 2)

    return curve


def _minus_one_db_per_octave(frequencies: np.ndarray) -> np.ndarray:
    return np.asarray(-np.log2(frequencies / frequencies[0]), dtype=np.float64)


def _registry_with(tmp_path: Path, old: str, new: str) -> Path:
    """複製一份登記簿、只改一處文字；改不到就紅，免得考卷靜靜比了兩份一樣的設定。"""
    original = _REGISTRY.read_text(encoding="utf-8")
    assert old in original
    changed_path = tmp_path / "quality_targets.toml"
    changed_path.write_text(original.replace(old, new, 1), encoding="utf-8")
    return changed_path


def test_flat_curve_measures_zero_shape_and_target_deviation() -> None:
    """若斜率、去趨勢或音量正規化漏掉，平直曲線不會三項都剛好回零、清單也不會是空的。"""
    evaluation = _evaluate(_flat_input())
    payload = evaluation.payload

    assert evaluation.state == EvaluationState.MEASURED
    assert evaluation.category_cost is None
    assert evaluation.reason_codes == ()
    assert evaluation.raw_quantities
    assert evaluation.evaluator_version == timbre.TIMBRE_EVALUATOR_VERSION
    assert evaluation.provenance == _PROVENANCE
    assert isinstance(payload, TimbrePayload)
    assert payload.tilt_db_per_octave == 0.0
    assert payload.residual_rms_db == 0.0
    assert payload.target_deviation_rms_db == 0.0
    assert payload.features == ()
    assert payload.strongest_peak_index is None
    assert payload.deepest_dip_index is None
    assert Flag.BASELINE_SETTINGS in evaluation.flags


def test_adding_three_db_keeps_all_shape_quantities_bitwise_equal() -> None:
    """若入口保留絕對音量，整條加 3 dB 會誤改任何一個形狀量。"""
    base = _payload(_flat_input())
    louder = _payload(_flat_input(3.0))

    assert louder.tilt_db_per_octave == base.tilt_db_per_octave
    assert louder.residual_rms_db == base.residual_rms_db
    assert louder.target_deviation_rms_db == base.target_deviation_rms_db
    assert louder.deviation_curve == base.deviation_curve
    assert louder.features == base.features


def test_linear_minus_one_db_per_octave_separates_tilt_from_target() -> None:
    """若傾斜擬合或目標曲線接錯，純 −1 dB/八度直線不會量出 −1、也不會只對傾斜目標歸零。"""
    input_data = _curve_input(_minus_one_db_per_octave)
    flat_target = _payload(input_data, target_curve=timbre.TargetCurve(kind="flat", tilt_db_per_octave=0.0))
    matching_target = _payload(
        input_data, target_curve=timbre.TargetCurve(kind="sloped", tilt_db_per_octave=-1.0)
    )

    assert flat_target.tilt_db_per_octave == pytest.approx(-1.0)
    assert flat_target.target_deviation_rms_db > 0.0
    assert matching_target.target_tilt_db_per_octave == -1.0
    assert matching_target.target_deviation_rms_db == pytest.approx(0.0)


def test_linear_curve_residual_is_only_the_smoothing_width_level_gap(tmp_path: Path) -> None:
    """若起伏分支沒扣擬合線，純傾斜會被當成起伏；扣了之後只剩兩個平滑寬度的準位差。

    能量域平均對斜線會墊高一個跟視窗寬度有關的常數；傾斜與起伏兩支寬度不同，殘差就是
    這兩個常數的差。把兩個寬度改成一樣，這個差歸零、殘差只剩捨入。
    """
    input_data = _curve_input(_minus_one_db_per_octave)
    dip = _payload(_curve_input(_gaussian_feature(100.0, -3.0, 1.0 / 3.0)))
    tilt_width = _setting("timbre_balance.smoothing_width_octave_tilt").value
    equal_widths = _registry_with(
        tmp_path,
        'key = "timbre_balance.smoothing_width_octave_ripple"\nvalue = 0.16666666666666666',
        f'key = "timbre_balance.smoothing_width_octave_ripple"\nvalue = {tilt_width!r}',
    )

    assert _payload(input_data).residual_rms_db < dip.residual_rms_db
    assert _payload(input_data, registry=equal_widths).residual_rms_db == pytest.approx(0.0)


def test_one_third_octave_dip_reports_defined_center_depth_and_width() -> None:
    """若局部極值或半深度內插錯，100 Hz 的已知凹陷不會落在從定義推出的界線內。

    怎麼推：造的是半深度全寬 w＝1/3 八度、深 3 dB 的凹陷。
    - 中心：凹陷對稱、平滑視窗在等比軸上對稱，極小值落在離 100 Hz 最近的格點，
      所以界線是 100 Hz 上下各一格。
    - 深度：能量平均不會比最低點還低，所以不會深過 −3 dB；擬合線被凹陷往下拉、
      平滑把谷填淺，兩者都讓殘差變淺（實測 −2.21 dB：平滑後 −2.79、擬合線在那裡 −0.59），
      下界取設計深度的一半——淺到一半就認不出是這個凹陷。
    - 寬度：寬 b 的平滑視窗讓每個半深度交點最多移 b/2，所以落在 [w−b, w+b]；
      擬合線偏移改了半深度的基準，實測 0.298 八度，在界線內。
    """
    designed_width = 1.0 / 3.0
    input_data = _curve_input(_gaussian_feature(100.0, -3.0, designed_width))
    payload = _payload(input_data)
    frequency_ratio = input_data.frequencies_hz[1] / input_data.frequencies_hz[0]
    smoothing = _scalar_setting("timbre_balance.smoothing_width_octave_ripple")
    matching = tuple(
        feature
        for feature in payload.features
        if feature.kind == "dip"
        and 100.0 / frequency_ratio <= feature.center_frequency_hz <= 100.0 * frequency_ratio
    )

    assert matching
    dip = min(matching, key=lambda feature: feature.depth_db)
    assert -3.0 <= dip.depth_db <= -1.5
    assert dip.width_octave is not None
    assert designed_width - smoothing <= dip.width_octave <= designed_width + smoothing
    assert payload.deepest_dip_index is not None
    assert payload.features[payload.deepest_dip_index].kind == "dip"


def test_deeper_dip_has_larger_residual_rms_without_assigning_a_score() -> None:
    """若殘差均方根對缺陷深度沒有鑑別力，排序層拿不到深淺證據。"""
    width = 1.0 / 3.0
    shallow = _evaluate(_curve_input(_gaussian_feature(100.0, -3.0, width)))
    deep = _evaluate(_curve_input(_gaussian_feature(100.0, -6.0, width)))

    assert isinstance(shallow.payload, TimbrePayload)
    assert isinstance(deep.payload, TimbrePayload)
    assert deep.payload.residual_rms_db > shallow.payload.residual_rms_db
    assert shallow.category_cost is None and deep.category_cost is None


def _two_dips(frequencies: np.ndarray) -> np.ndarray:
    first = _gaussian_feature(100.0, -3.0, 1.0 / 3.0)(frequencies)
    second = _gaussian_feature(1000.0, -5.0, 1.0 / 3.0)(frequencies)
    return np.asarray(first + second, dtype=np.float64)


def _nearest(center_hz: float, expected: tuple[float, ...]) -> float:
    return min(expected, key=lambda value: abs(math.log2(center_hz / value)))


def test_feature_list_is_complete_and_summary_points_at_the_deepest_dip() -> None:
    """清單若被截短（只留第一個）或摘要索引指錯，排名層的峰谷保護就只看得到一半的凹陷。"""
    payload = _payload(_curve_input(_two_dips, point_count=481))
    expected = (100.0, 1000.0)
    deep_dips = [item for item in payload.features if item.kind == "dip" and item.depth_db < -1.0]

    assert {_nearest(item.center_frequency_hz, expected) for item in deep_dips} == set(expected)
    assert payload.deepest_dip_index is not None
    deepest = payload.features[payload.deepest_dip_index]
    assert _nearest(deepest.center_frequency_hz, expected) == 1000.0
    assert deepest.depth_db == min(item.depth_db for item in payload.features)


def test_one_dip_is_one_local_extremum_not_every_point_below_the_line() -> None:
    """若把「線下的每一點」都當成谷，一個凹陷會變成一長串特徵，摘要與保護代價都失真。"""
    payload = _payload(_curve_input(_gaussian_feature(100.0, -3.0, 1.0 / 3.0), point_count=481))
    deep_centers = [
        item.center_frequency_hz
        for item in payload.features
        if item.kind == "dip" and item.depth_db < -1.0
    ]

    assert deep_centers
    assert max(deep_centers) == min(deep_centers)
    centers = [item.center_frequency_hz for item in payload.features]
    assert centers == sorted(centers)


def test_ripple_smoothing_width_is_independent_of_the_tilt_branch(tmp_path: Path) -> None:
    """起伏那一支若偷用傾斜那一支的平滑結果，改起伏的平滑寬度就不會動到殘差。"""
    narrower = _registry_with(
        tmp_path,
        'key = "timbre_balance.smoothing_width_octave_ripple"\nvalue = 0.16666666666666666',
        'key = "timbre_balance.smoothing_width_octave_ripple"\nvalue = 0.08333333333333333',
    )
    input_data = _curve_input(_gaussian_feature(100.0, -3.0, 1.0 / 3.0), point_count=481)

    before = _payload(input_data)
    after = _payload(input_data, registry=narrower)

    assert after.tilt_db_per_octave == before.tilt_db_per_octave
    assert after.residual_rms_db != before.residual_rms_db


def test_missing_wide_dip_is_unavailable_instead_of_perfectly_measured() -> None:
    """若把藏著寬谷的頻段整段拿掉仍照量，少交資料會得到零起伏並進榜。"""
    full = _curve_input(
        lambda frequencies: np.where(
            (frequencies >= 500.0) & (frequencies <= 1000.0), -12.0, 0.0
        )
    )

    complete = _evaluate(full)
    holed = _evaluate(_without_span(full, 400.0, 1200.0))

    assert complete.state is EvaluationState.MEASURED
    assert isinstance(complete.payload, TimbrePayload)
    assert complete.payload.deepest_dip_index is not None
    dip = complete.payload.features[complete.payload.deepest_dip_index]
    assert dip.kind == "dip"
    assert dip.depth_db < 0.0
    assert holed.state is EvaluationState.UNAVAILABLE
    assert holed.reason_codes == (ReasonCode.TIMBRE_SCORING_RANGE_GAP,)
    assert Flag.DATA_COVERAGE_SHORT in holed.flags


def test_hole_inside_scored_range_is_unavailable_and_flagged_short_coverage() -> None:
    """中間的洞不能被當成完整覆蓋；落在計分範圍時要帶旗標並回不可估。"""
    full = _flat_input()
    evaluation = _evaluate(_without_span(full, 2500.0, 5000.0))

    assert evaluation.state is EvaluationState.UNAVAILABLE
    assert evaluation.reason_codes == (ReasonCode.TIMBRE_SCORING_RANGE_GAP,)
    assert Flag.DATA_COVERAGE_SHORT in evaluation.flags
    assert Flag.DATA_COVERAGE_SHORT not in _evaluate(full).flags


def test_hole_outside_scored_ranges_stays_measured_and_flagged() -> None:
    """診斷覆蓋範圍裡的洞若完全在兩個計分範圍外，只掛覆蓋不足旗標。"""
    evaluation = _evaluate(_without_span(_flat_input(), 5000.0, 7000.0))

    assert evaluation.state is EvaluationState.MEASURED
    assert evaluation.reason_codes == ()
    assert Flag.DATA_COVERAGE_SHORT in evaluation.flags


def test_missing_scored_range_lower_boundary_is_unavailable() -> None:
    """只看範圍內相鄰資料點會漏掉下邊界到第一點的缺段；虛擬邊界必須抓到它。"""
    full = _flat_input()
    started_at_200_hz = _without_span(full, 0.0, 200.0)

    evaluation = _evaluate(started_at_200_hz)

    assert evaluation.state is EvaluationState.UNAVAILABLE
    assert evaluation.reason_codes == (ReasonCode.TIMBRE_SCORING_RANGE_GAP,)
    assert Flag.DATA_COVERAGE_SHORT in evaluation.flags


def test_hole_only_inside_the_ripple_range_is_unavailable_too() -> None:
    """起伏範圍比傾斜擬合範圍往低頻多一段；洞只落在那一段（45–75 Hz）也是計分範圍缺段。"""
    holed = _without_span(_flat_input(), 45.0, 75.0)

    evaluation = _evaluate(holed)

    assert evaluation.state is EvaluationState.UNAVAILABLE
    assert evaluation.reason_codes == (ReasonCode.TIMBRE_SCORING_RANGE_GAP,)


def test_narrow_peak_is_kept_and_flagged() -> None:
    """若最小寬度被當成刪除條件，尖峰會從診斷清單消失。"""
    minimum = _scalar_setting("timbre_balance.feature_min_width_octave")
    evaluation = _evaluate(_curve_input(_gaussian_feature(300.0, 6.0, minimum / 8.0), point_count=481))
    assert isinstance(evaluation.payload, TimbrePayload)
    narrow_peaks = tuple(
        feature
        for feature in evaluation.payload.features
        if feature.kind == "peak" and Flag.FEATURE_TOO_NARROW in feature.flags
    )

    assert narrow_peaks
    assert all(feature.width_octave is not None for feature in narrow_peaks)
    assert all(feature.width_octave < minimum for feature in narrow_peaks if feature.width_octave)
    assert Flag.FEATURE_TOO_NARROW in evaluation.flags


def test_dip_touching_ripple_boundary_keeps_unknown_width() -> None:
    """若寬度搜尋越過評估範圍，邊界凹陷會被捏造成完整寬度。"""
    lower_hz, _ = _range_setting("timbre_balance.ripple_range_hz")
    minimum = _scalar_setting("timbre_balance.feature_min_width_octave")
    center_hz = float(lower_hz) * 2.0 ** (minimum / 4.0)
    evaluation = _evaluate(_curve_input(_gaussian_feature(center_hz, -6.0, minimum), point_count=481))
    assert isinstance(evaluation.payload, TimbrePayload)
    boundary_dips = tuple(
        feature
        for feature in evaluation.payload.features
        if feature.kind == "dip"
        and feature.width_octave is None
        and Flag.FEATURE_BOUNDARY_INCOMPLETE in feature.flags
    )

    assert boundary_dips
    assert Flag.FEATURE_BOUNDARY_INCOMPLETE in evaluation.flags


def test_short_data_range_is_flagged_but_remains_measured() -> None:
    """若 coverage 缺口被誤當整條不可估或靜靜截掉，今天 5657 Hz 的報表就無法使用。"""
    evaluation = _evaluate(_flat_input(upper_hz=5657.0))

    assert evaluation.state == EvaluationState.MEASURED
    assert Flag.DATA_COVERAGE_SHORT in evaluation.flags
    assert isinstance(evaluation.payload, TimbrePayload)
    assert evaluation.payload.data_range_hz == (20.0, 5657.0)
    assert evaluation.payload.coverage_range_hz == _range_setting("timbre_balance.coverage_range_hz")


def test_full_data_range_is_not_flagged_short() -> None:
    """若覆蓋判斷寫反，資料蓋滿要求範圍時也會亂標覆蓋不到。"""
    assert Flag.DATA_COVERAGE_SHORT not in _evaluate(_flat_input()).flags


def test_insufficient_intersection_is_unavailable_without_fabricated_payload() -> None:
    """若資料不足被填零，可排序端會把不可估誤認成完美。"""
    evaluation = _evaluate(_curve_input(lambda frequencies: np.zeros_like(frequencies), point_count=6))

    assert evaluation.state == EvaluationState.UNAVAILABLE
    assert evaluation.payload is None
    assert evaluation.raw_quantities == ()
    assert evaluation.category_cost is None
    assert evaluation.reason_codes == (ReasonCode.INSUFFICIENT_COVERAGE,)
    assert evaluation.scene_fingerprint == _SCENE_FINGERPRINT


@pytest.mark.parametrize("bad_energy", [0.0, -1.0, math.inf, math.nan])
def test_non_positive_or_non_finite_energy_is_unavailable(bad_energy: float) -> None:
    """若非正或非有限的線性能量進 log10，輸出會出現非有限值或捏造的零。"""
    input_data = _without_span(_flat_input(), 400.0, 1200.0)
    energies = list(input_data.total_energy)
    energies[len(energies) // 2] = bad_energy
    invalid = timbre.TimbreInput.model_validate(
        {**input_data.model_dump(), "total_energy": tuple(energies)}
    )
    evaluation = _evaluate(invalid)

    assert evaluation.state == EvaluationState.UNAVAILABLE
    assert evaluation.payload is None
    assert evaluation.raw_quantities == ()
    assert evaluation.reason_codes == (ReasonCode.NON_POSITIVE_ENERGY,)


def test_report_flags_are_forwarded_to_the_category_evaluation() -> None:
    """若報表既有標記在評估入口遺失，排名表無法回查資料限制。"""
    flat = _curve_input(np.zeros_like, report_flags=(Flag.CROSSOVER_BAND,))
    evaluation = _evaluate(flat)

    assert Flag.CROSSOVER_BAND in evaluation.flags


def test_default_target_comes_from_registry() -> None:
    """若沒給目標時評估器自己編一條，目標就不是事先給定、也不住登記簿。"""
    entry = load_quality_targets(_REGISTRY).purpose(_PURPOSE).entry(
        "timbre_balance.target_tilt_db_per_octave"
    )
    payload = _payload(_flat_input())

    assert isinstance(entry, TargetEntry)
    assert payload.target_tilt_db_per_octave == entry.value


def test_target_curve_kind_must_match_its_tilt() -> None:
    """若 flat 目標准帶傾斜，同一條目標就有兩種寫法，指紋與報表對不起來。"""
    with pytest.raises(ValueError):
        timbre.TargetCurve(kind="flat", tilt_db_per_octave=-1.0)
    with pytest.raises(ValueError):
        timbre.TargetCurve(kind="sloped", tilt_db_per_octave=0.0)


def test_registry_change_changes_settings_fingerprint(tmp_path: Path) -> None:
    """若評估設定改了而指紋不變，不同量法會被排名層放進同一張表。"""
    changed_path = _registry_with(
        tmp_path,
        'key = "timbre_balance.feature_min_width_octave"\nvalue = 0.16666666666666666',
        'key = "timbre_balance.feature_min_width_octave"\nvalue = 0.125',
    )

    before = _evaluate(_flat_input())
    after = _evaluate(_flat_input(), registry=changed_path)

    assert after.settings_fingerprint != before.settings_fingerprint


def _report_point(frequency_hz: float, total_energy: float) -> PointRow:
    return PointRow(
        frequency_hz=frequency_hz,
        fem_energy=total_energy,
        direct_energy=total_energy,
        reflected_energy=0.0,
        interference_energy=0.0,
        late_energy=0.0,
        scattering=0.0,
        geometric_energy=total_energy,
        w_fem=1.0,
        w_geo=0.0,
        total_energy=total_energy,
    )


def _report_band() -> BandRow:
    return BandRow(
        center_frequency_hz=1000.0,
        fem_energy=1.0,
        fem_point_count=1,
        direct_energy=1.0,
        reflected_energy=0.0,
        interference_energy=0.0,
        late_energy=0.0,
        geometric_energy=1.0,
        fem_contribution=1.0,
        geometric_contribution=0.0,
        total_energy=1.0,
        w_fem=1.0,
        w_geo=0.0,
        f_s_hz=200.0,
        capped_by_upper_limit=False,
        t20_s=1.0,
        t20_unavailable_reason=None,
        t30_s=1.0,
        t30_unavailable_reason=None,
    )


def _minimal_report(points: tuple[PointRow, ...] | None) -> ReportOutput:
    return ReportOutput(
        scene=SceneSection(
            scene_fingerprint=_SCENE_FINGERPRINT,
            source_m=Point(1.2, 1.3, 1.1),
            receiver_m=Point(*_POSITION),
        ),
        capability=CapabilitySection(
            # 刻意跟手造輸入常用的 (20, 8000) 不同：轉接器若把範圍寫死，那一題就會紅。
            frequency_hz=(25.0, 5583.0),
            outputs=("total_energy",),
            status="experimental",
            evidence=(),
        ),
        top=TopFields(
            f_s_hz=200.0,
            crossover_lower_hz=200.0,
            crossover_upper_hz=300.0,
            capped_by_upper_limit=False,
            reflection_order_k=3,
            eyring_t60_by_band_s={"1000.0": 1.0},
            room_volume_m3=72.0,
            schroeder_band_count=1,
        ),
        bands=(_report_band(),),
        points=points,
    )


def _collect(report: ReportOutput) -> timbre.TimbreInput:
    return timbre.timbre_input_from_report(
        report,
        candidate_id=_CANDIDATE,
        speaker_id=_SPEAKER,
        receiver_id=_RECEIVER,
        source_reference="呼叫端給的共同基準",
        provenance=_PROVENANCE,
    )


def test_report_helper_takes_scene_and_placement_only_from_report() -> None:
    """轉接器若仍收呼叫端覆寫，或沒跟著 scene 走，就會把錯場景／錯座標貼進音色輸入。"""
    original = _minimal_report((_report_point(20.0, 1.0), _report_point(40.0, 0.5)))
    report = original.model_copy(
        update={
            "scene": original.scene.model_copy(
                update={
                    "scene_fingerprint": "a" * 64,
                    "source_m": Point(0.1, 0.2, 0.3),
                    "receiver_m": Point(1.0, 2.0, 3.0),
                }
            )
        }
    )
    before = report.model_dump(mode="python")
    collected = _collect(report)

    parameters = inspect.signature(timbre.timbre_input_from_report).parameters
    assert "scene_fingerprint" not in parameters
    assert "source_position_m" not in parameters
    assert "receiver_position_m" not in parameters
    assert collected.frequencies_hz == (20.0, 40.0)
    assert collected.total_energy == (1.0, 0.5)
    assert collected.scene_fingerprint == "a" * 64
    assert collected.source_position_m == (0.1, 0.2, 0.3)
    assert collected.receiver_position_m == (1.0, 2.0, 3.0)
    assert collected.source_reference == "呼叫端給的共同基準"
    assert collected.provenance == _PROVENANCE
    assert collected.candidate_id == _CANDIDATE
    assert collected.report_flags == ()
    assert collected.model_validation_status is ModelValidationStatus.EXPERIMENTAL
    assert collected.model_validation_frequency_range_hz == (25.0, 5583.0)
    assert report.model_dump(mode="python") == before


@pytest.mark.parametrize(
    "declared",
    [(100.0, 3000.0), (60.0, 5000.0), (30.0, 4300.0)],
    ids=["both-ends-short", "only-ripple-lower-end-short", "only-tilt-upper-end-short"],
)
def test_validated_capability_must_cover_every_scoring_range(
    declared: tuple[float, float],
) -> None:
    """若只看 validated 狀態、或只比其中一端、或只比其中一段，超出能力證據的音色會冒充已驗過。

    要包住的是依賴範圍（票 #409）：傾斜約 71–4490 Hz、起伏約 38–4238 Hz。
    (60, 5000) 包得住傾斜那一段、包不住起伏的下端；(30, 4300) 包得住起伏那一段、包不住傾斜的上端——
    只比下界、只比上界、只比其中一段的壞改法各會被其中一個例子抓到。
    """
    evaluation = _evaluate(
        _flat_input().model_copy(
            update={"model_validation_frequency_range_hz": declared}
        )
    )

    assert evaluation.state is EvaluationState.MEASURED
    assert Flag.UNVALIDATED in evaluation.flags
    assert isinstance(evaluation.payload, TimbrePayload)
    assert evaluation.payload.model_validation_status is ModelValidationStatus.VALIDATED
    assert evaluation.payload.model_validation_frequency_range_hz == declared


def test_validated_capability_covering_exactly_the_scored_ranges_is_not_flagged() -> None:
    """端點相等算包得住：宣告剛好包住兩個依賴範圍的驗過報表不掛未驗證。"""
    reference = _payload(_flat_input())
    dependency_ranges = (
        reference.tilt_dependency_range_hz,
        reference.ripple_dependency_range_hz,
    )
    declared = (
        min(bounds[0] for bounds in dependency_ranges),
        max(bounds[1] for bounds in dependency_ranges),
    )
    evaluation = _evaluate(
        _flat_input().model_copy(
            update={"model_validation_frequency_range_hz": declared}
        )
    )

    assert Flag.UNVALIDATED not in evaluation.flags


@pytest.mark.parametrize(
    ("status", "declared"),
    [
        (ModelValidationStatus.VALIDATED, ()),
        (ModelValidationStatus.UNCHECKED, (20.0, 8000.0)),
    ],
)
def test_empty_capability_range_and_unchecked_status_go_together(
    status: ModelValidationStatus, declared: tuple[float, float] | tuple[()]
) -> None:
    """「驗過但範圍是空的」「沒查表卻有範圍」都是自相矛盾的輸入，收進來之前就拒收。"""
    document = _flat_input().model_dump(mode="python")
    document.update(
        model_validation_status=status, model_validation_frequency_range_hz=declared
    )

    with pytest.raises(ValueError, match="若且唯若"):
        timbre.TimbreInput.model_validate(document)


def test_unavailable_evaluation_keeps_unvalidated_capability_flag() -> None:
    """若不可估捷徑漏帶能力標記，排名端會看不出該物理模型仍在試驗中。"""
    original = _flat_input()
    input_data = original.model_copy(
        update={
            "model_validation_status": ModelValidationStatus.EXPERIMENTAL,
            "total_energy": (0.0, *original.total_energy[1:]),
        }
    )

    evaluation = _evaluate(input_data)

    assert evaluation.state is EvaluationState.UNAVAILABLE
    assert Flag.UNVALIDATED in evaluation.flags


def test_report_helper_refuses_report_without_fine_axis() -> None:
    """若報表沒有細軸表卻收成空曲線，評估器會拿到一條假的輸入。"""
    with pytest.raises(ValueError):
        _collect(_minimal_report(None))


def test_tilt_fit_range_rejects_registry_unit_mismatch(tmp_path: Path) -> None:
    """若音色評估器沒聲明 Hz，登記簿把擬合範圍標成 oct 仍會被拿去算。"""
    registry = _registry_with(
        tmp_path,
        'key = "timbre_balance.tilt_fit_range_hz"\nvalue = [80.0, 4000.0]\nunit = "Hz"',
        'key = "timbre_balance.tilt_fit_range_hz"\nvalue = [80.0, 4000.0]\nunit = "oct"',
    )

    with pytest.raises(
        ValueError,
        match="timbre_balance.tilt_fit_range_hz 單位應為 Hz，登記簿寫 oct",
    ):
        _evaluate(_flat_input(), registry=registry)
