"""殘響第一層評估器考卷（票 #348 第一段）：只量、不打分。"""
from __future__ import annotations

import pytest

from aosr.config.art_lane import ART_WLS_T20_LO_DB, ART_WLS_T30_LO_DB
from aosr.config.frequency_axis import (
    GEOMETRIC_LANE_FREQUENCIES_HZ,
    GEOMETRIC_REPORT_OCTAVE_CENTERS_HZ,
)
from aosr.geometry.shoebox import Point
from aosr.physics.late_decay import DecayRangeError
from aosr.physics.report_io import (
    BandRow,
    CapabilitySection,
    ReportOutput,
    SceneSection,
    TopFields,
)
from aosr.scoring.contract import (
    CategoryEvaluation,
    EvaluationState,
    InputProvenance,
    MetricState,
    ReasonCode,
    ReverberationPayload,
)


_PROVENANCE = InputProvenance(
    report_id="three-lane-report",
    engine_commit="unknown",
    speaker_id="left",
    receiver_id="main-seat",
)
_SCENE_FINGERPRINT = "a" * 64


def _decay_range_reason() -> str:
    """直接取物理層的真訊息；上游散文若改字，原因分類考卷必須紅。"""
    return str(
        DecayRangeError(
            lower_db=ART_WLS_T20_LO_DB,
            minimums_by_frequency=((1000.0, -20.0),),
        )
    )


def _band(
    center_hz: float,
    *,
    f_s_hz: float = 200.0,
    t20_s: float | None = 1.0,
    t20_reason: str | None = None,
    t30_s: float | None = 1.2,
    t30_reason: str | None = None,
) -> BandRow:
    return BandRow(
        center_frequency_hz=center_hz,
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
        f_s_hz=f_s_hz,
        capped_by_upper_limit=False,
        t20_s=t20_s,
        t20_unavailable_reason=t20_reason,
        t30_s=t30_s,
        t30_unavailable_reason=t30_reason,
    )


def _report(
    bands: tuple[BandRow, ...],
    *,
    coverage_hz: tuple[float, float] = (20.0, 20000.0),
    validation_status: str = "experimental",
) -> ReportOutput:
    return ReportOutput(
        scene=SceneSection(
            scene_fingerprint=_SCENE_FINGERPRINT,
            source_m=Point(1.2, 1.3, 1.1),
            receiver_m=Point(4.7, 2.8, 1.4),
        ),
        capability=CapabilitySection(
            frequency_hz=coverage_hz,
            outputs=("t20_s", "t30_s"),
            status=validation_status,
            evidence=(),
        ),
        top=TopFields(
            f_s_hz=bands[0].f_s_hz,
            crossover_lower_hz=100.0,
            crossover_upper_hz=300.0,
            capped_by_upper_limit=False,
            reflection_order_k=3,
            eyring_t60_by_band_s={str(band.center_frequency_hz): 1.0 for band in bands},
            room_volume_m3=72.0,
            schroeder_band_count=1,
        ),
        bands=bands,
        points=None,
    )


def _evaluate(report: ReportOutput, *, logarithm_base: float = 2.0) -> CategoryEvaluation:
    from aosr.scoring import reverberation

    return reverberation.evaluate_reverberation(
        report,
        candidate_id="candidate-a",
        provenance=_PROVENANCE,
        logarithm_base=logarithm_base,
    )


def _payload(report: ReportOutput, *, logarithm_base: float = 2.0) -> ReverberationPayload:
    evaluation = _evaluate(report, logarithm_base=logarithm_base)
    assert evaluation.state is EvaluationState.MEASURED
    assert evaluation.category_cost is None
    assert isinstance(evaluation.payload, ReverberationPayload)
    return evaluation.payload


def test_reverberation_contract_keeps_local_states_and_confidence_markers() -> None:
    """拿掉局部狀態或任一可信度標記時，消費端會把有值誤讀成已驗證且可用。"""
    payload = ReverberationPayload.model_validate(
        {
            "category": "reverberation",
            "bands": [
                {
                    "center_frequency_hz": 1000.0,
                    "band_range_hz": [707.0, 1414.0],
                    "schroeder_position": "above",
                    "model_validation_status": "experimental",
                    "t20": {
                        "value": 1.0,
                        "unit": "s",
                        "state": "measured",
                        "reason_codes": [],
                        "reason": None,
                    },
                    "t30": {
                        "value": None,
                        "unit": "s",
                        "state": "unavailable",
                        "reason_codes": ["insufficient_decay_range"],
                        "reason": "未達 T30 擬合下緣",
                    },
                    "fitting_difference": {
                        "value": None,
                        "unit": "1",
                        "state": "not_computable",
                        "reason_codes": ["insufficient_decay_range"],
                        "reason": "T30 不可估，無法計算 T30/T20",
                    },
                }
            ],
            "adjacent_band_changes": [],
            "logarithm_base": 2.0,
        }
    )

    assert payload.bands[0].t20.value == 1.0
    assert payload.bands[0].t30.reason_codes == ("insufficient_decay_range",)
    assert payload.bands[0].schroeder_position == "above"
    assert payload.bands[0].model_validation_status == "experimental"


def test_reverberation_carries_report_scene_fingerprint() -> None:
    """評估器若漏轉報表場景，候選包就無法攔下跨房間的殘響結果。"""
    evaluation = _evaluate(_report((_band(1000.0),)))

    assert evaluation.scene_fingerprint == _SCENE_FINGERPRINT
    assert evaluation.model_dump(mode="python")["placement"] == {
        "speaker_positions_m": (("left", (1.2, 1.3, 1.1)),),
        "receiver_positions_m": (("main-seat", (4.7, 2.8, 1.4)),),
    }


def test_synthetic_eight_kilohertz_band_uses_its_full_octave_range() -> None:
    """今天正式報表只產 125～4000 Hz，這題的 8 kHz 是自造資料；頻帶清單擴充後才會真走到。"""
    payload = _payload(
        _report((_band(8000.0),), coverage_hz=(20.0, 8000.0))
    )

    assert payload.bands[0].band_range_hz == pytest.approx(
        (8000.0 / 2**0.5, 8000.0 * 2**0.5)
    )
    assert payload.bands[0].t20.state is MetricState.MEASURED
    assert ReasonCode.INSUFFICIENT_COVERAGE not in payload.bands[0].t20.reason_codes


def test_formal_report_highest_band_is_not_misclassified_as_uncovered() -> None:
    """細軸上限不是資料上限，正式最高報表帶有值就不得報覆蓋不足。"""
    report = _report(
        tuple(_band(center_hz) for center_hz in GEOMETRIC_REPORT_OCTAVE_CENTERS_HZ),
        coverage_hz=(
            GEOMETRIC_LANE_FREQUENCIES_HZ[0],
            GEOMETRIC_LANE_FREQUENCIES_HZ[-1],
        ),
    )

    highest = _payload(report).bands[-1]

    assert highest.center_frequency_hz == max(GEOMETRIC_REPORT_OCTAVE_CENTERS_HZ)
    assert highest.t20.state is MetricState.MEASURED
    assert ReasonCode.INSUFFICIENT_COVERAGE not in highest.t20.reason_codes


def test_capability_range_limits_model_validation_instead_of_data_coverage() -> None:
    """能力表沒蓋完整帶時只降模型驗證標記，不得抹掉報表已經給的量值。"""
    highest = _payload(
        _report(
            (_band(max(GEOMETRIC_REPORT_OCTAVE_CENTERS_HZ)),),
            coverage_hz=(
                GEOMETRIC_LANE_FREQUENCIES_HZ[0],
                max(GEOMETRIC_REPORT_OCTAVE_CENTERS_HZ),
            ),
            validation_status="validated",
        )
    ).bands[0]

    assert highest.t20.state is MetricState.MEASURED
    assert highest.model_validation_status == "unchecked"


@pytest.mark.parametrize(
    ("report", "reason_code"),
    (
        (
            _report((_band(1000.0, t20_s=0.0),)),
            "non_positive_value",
        ),
        (
            _report(
                (
                    _band(
                        1000.0,
                        t20_s=None,
                        t20_reason=_decay_range_reason(),
                    ),
                )
            ),
            ReasonCode.INSUFFICIENT_DECAY_RANGE,
        ),
        (
            _report(
                (
                    _band(
                        1000.0,
                        t20_s=None,
                        t20_reason="T20 求解發生未分類錯誤",
                    ),
                )
            ),
            ReasonCode.OTHER_ERROR,
        ),
    ),
)
def test_three_unavailable_reason_families_remain_distinct(
    report: ReportOutput, reason_code: ReasonCode | str
) -> None:
    """壞值、衰減範圍與其他錯誤若共用一碼，後續無法決定修資料還是修求解。"""
    metric = _payload(report).bands[0].t20

    assert metric.state is MetricState.UNAVAILABLE
    assert metric.reason_codes == (reason_code,)
    assert metric.reason


def test_band_that_contains_schroeder_frequency_is_marked_crossing() -> None:
    """用中心頻率與 f_s 比會漏掉頻帶範圍真正夾住交界的情形。"""
    payload = _payload(_report((_band(1000.0, f_s_hz=1100.0),)))

    assert payload.bands[0].schroeder_position == "crossing"


def test_above_schroeder_does_not_upgrade_model_validation() -> None:
    """整帶在 f_s 上方仍須保留報表的 experimental，不可推成 validated。"""
    payload = _payload(
        _report((_band(1000.0, f_s_hz=200.0),), validation_status="experimental")
    )

    assert payload.bands[0].schroeder_position == "above"
    assert payload.bands[0].model_validation_status == "experimental"


def test_missing_t20_is_not_replaced_by_valid_t30() -> None:
    """T20 空時拿 T30 補值，會把診斷量偷偷改成殘響長短。"""
    payload = _payload(
        _report(
            (
                _band(
                    1000.0,
                    t20_s=None,
                    t20_reason=_decay_range_reason(),
                    t30_s=1.8,
                ),
            )
        )
    )

    assert payload.bands[0].t20.value is None
    assert payload.bands[0].t20.state is MetricState.UNAVAILABLE
    assert payload.bands[0].t30.value == 1.8


def test_fitting_difference_is_not_computable_when_either_fit_is_missing() -> None:
    """缺一個擬合卻回 1，會把沒有診斷證據偽裝成完全一致。"""
    payload = _payload(
        _report(
            (
                _band(
                    1000.0,
                    t30_s=None,
                    t30_reason=DecayRangeError(
                        lower_db=ART_WLS_T20_LO_DB,
                        minimums_by_frequency=((1000.0, -20.0),),
                    ).reason_for(ART_WLS_T30_LO_DB),
                ),
            )
        )
    )

    assert payload.bands[0].fitting_difference.value is None
    assert payload.bands[0].fitting_difference.state is MetricState.NOT_COMPUTABLE
    assert payload.bands[0].fitting_difference.reason_codes == (
        ReasonCode.INSUFFICIENT_DECAY_RANGE,
    )


def test_fitting_difference_is_t30_divided_by_t20() -> None:
    """顛倒比值或與 T20 平均，診斷方向就會相反或失去定義。"""
    payload = _payload(_report((_band(1000.0, t20_s=1.5, t30_s=1.8),)))

    assert payload.bands[0].fitting_difference.value == pytest.approx(1.2)


def test_adjacent_change_list_scales_with_bands_and_keeps_unavailable_pairs() -> None:
    """清單寫死或跨過不可估中帶，擴帶與缺帶時都會少回應有的一對。"""
    bands = (
        _band(500.0, t20_s=1.0),
        _band(1000.0, t20_s=None, t20_reason=_decay_range_reason()),
        _band(2000.0, t20_s=2.0),
    )
    payload = _payload(_report(bands))

    assert len(payload.adjacent_band_changes) == len(bands) - 1
    assert all(
        change.state is MetricState.NOT_COMPUTABLE
        for change in payload.adjacent_band_changes
    )
    assert all(change.signed_log_ratio is None for change in payload.adjacent_band_changes)


def test_adjacent_signed_log_ratio_is_positive_when_upper_band_is_longer() -> None:
    """正負號若按相反方向取，下游會把往高頻變長解讀成變短。"""
    bands = (
        _band(500.0, t20_s=1.0),
        _band(1000.0, t20_s=2.0),
        _band(2000.0, t20_s=1.0),
    )
    changes = _payload(_report(bands), logarithm_base=2.0).adjacent_band_changes

    assert tuple(change.signed_log_ratio for change in changes) == pytest.approx(
        (1.0, -1.0)
    )
    assert all("severity" not in change.model_dump() for change in changes)
