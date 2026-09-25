"""#493：正式軸控制組與反射子帶的預定座標完整性。"""

from __future__ import annotations

from dataclasses import replace

from aosr.config.frequency_axis import (
    GEOMETRIC_LANE_FREQUENCIES_HZ, LowFrequencyAxis,
    low_frequency_axis_frequencies, planned_band_points,
)
from aosr.config.paths import config_path
from aosr.config.quality_targets import load_quality_targets
from aosr.physics.third_octave_decay import third_octave_bands
from aosr.physics import report_io, three_lane_report
from aosr.physics.third_octave_decay import build_third_octave_decay
from aosr.scoring.contract import CategoryCost, CategoryEvaluation, EvaluationState, QualityCategory, ReasonCode
from aosr.scoring.ranking import CandidateStatus, rank_candidates
from aosr.scoring.recommendation import ReviewStatus
from aosr.scoring.reflections import ReflectionInput
from aosr.scoring.reflections_contract import (
    REFLECTIONS_AND_ECHO_EVALUATOR_VERSION, ReflectionsAndEchoPayload, WallPairBandRisk,
)
from aosr.scoring.reflections_cost import cost_reflections_evaluation, reflections_review_alerts
from aosr.scoring.review_alert import FlutterReviewAlert
from tests.engine import test_reflections as fixtures
from tests.engine import test_reflections_ranking as ranking_fixtures
from tests.engine import test_ranking_recost as recost
from tests.engine.test_third_octave_decay import solved_report as solved_report


_NAMES = (100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250,
          1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000)
_LOSS_DB = (6.112042675129047, 4.883957980528085, 6.293182562678186,
            5.019774703729075, 6.087684800607789, 5.159976314657557,
            5.891472967318623, 5.3048554216211645, 5.703744004637709,
            5.454734936016983, 5.523794724967573, 5.6099724067330206,
            5.351005908110293, 5.770965158344654, 5.184829442755187,
            5.938156418891113, 5.024777838868397, 6.112042675129048,
            4.883957980528086, 6.29318256267818, 5.019774703729075)
_DURATION_S = (0.3434408880967388, 0.4298000459464469, 0.3335554536237963,
               0.4181712304482699, 0.34481505419300157, 0.406809108497002,
               0.3562989045482897, 0.3956988829282729, 0.3680258726066054,
               0.3848262819465718, 0.3800150926940555, 0.3741774847077933,
               0.392286123483798, 0.36373904656073175, 0.4048590966409928,
               0.35349782261604284, 0.4177548603629846, 0.3434408880967387,
               0.42980004594644683, 0.3335554536237965, 0.4181712304482699)
_ROOM_T20_S = (0.705, 0.7062499999999999, 0.708, 0.71, 0.7124999999999999,
               0.71575, 0.72, 0.725, 0.7314999999999999, 0.74, 0.75, 0.7625,
               0.7799999999999999, 0.7999999999999999, 0.825, 0.8574999999999999,
               0.8999999999999999, 0.95, 1.015, 1.1, 1.2)


def _control_records() -> tuple[ReflectionInput, ReflectionInput]:
    records = []
    for role, y in (("left", 1.3), ("right", 2.5)):
        record = fixtures._record(role, y, axis=GEOMETRIC_LANE_FREQUENCIES_HZ)
        decay = record.third_octave_decay
        assert decay is not None
        rows = tuple(row.model_copy(update={
            "t20_s": 0.7 + row.band.nominal_center_hz / 20000.0,
            "t30_s": 0.9 + row.band.nominal_center_hz / 16000.0,
        }) for row in decay.rows)
        records.append(replace(record, third_octave_decay=decay.model_copy(
            update={"rows": rows})))
    retention = tuple(0.2 + (index % 17) / 100
                      for index, _ in enumerate(GEOMETRIC_LANE_FREQUENCIES_HZ))
    return fixtures._vary_pair((records[0], records[1]), retention)


def _control_band_rows(
    records: tuple[ReflectionInput, ...],
) -> tuple[WallPairBandRisk, ...]:
    result = fixtures._evaluate(records)
    assert isinstance(result.payload, ReflectionsAndEchoPayload)
    return next(pair.bands for pair in result.payload.wall_pairs
                if pair.walls == ("x0", "xL"))


def test_formal_axis_control_values_are_bitwise_unchanged() -> None:
    bands = _control_band_rows(_control_records())
    assert tuple(band.nominal_center_hz for band in bands) == _NAMES
    assert tuple(band.round_trip_loss_db.value for band in bands) == _LOSS_DB
    assert tuple(band.decay_duration_s.value for band in bands) == _DURATION_S
    assert tuple(band.room_t20_s.value for band in bands) == _ROOM_T20_S


def test_missing_middle_wall_point_is_unassessed() -> None:
    axis = GEOMETRIC_LANE_FREQUENCIES_HZ
    removed = next(point for point in axis if 900.0 < point < 1000.0)
    records = (fixtures._record("left", 1.3, axis=tuple(p for p in axis if p != removed)),
               fixtures._record("right", 2.5, axis=tuple(p for p in axis if p != removed)))
    band = next(row for row in _control_band_rows(records) if row.nominal_center_hz == 1000)
    assert band.round_trip_loss_db.value is None
    assert band.round_trip_loss_db.reason_codes == (ReasonCode.SUBBAND_SAMPLING_INCOMPLETE,)


def _fine_pair() -> tuple[ReflectionInput, ReflectionInput]:
    axis = GEOMETRIC_LANE_FREQUENCIES_HZ
    return fixtures._record("left", 1.3, axis=axis), fixtures._record("right", 2.5, axis=axis)


def _band(records: tuple[ReflectionInput, ...], nominal: int) -> WallPairBandRisk:
    return next(row for row in _control_band_rows(records) if row.nominal_center_hz == nominal)


def _missing_t20(
    records: tuple[ReflectionInput, ...], nominal: int,
    missing: float, prose: str = "子帶取樣不全",
) -> tuple[ReflectionInput, ...]:
    changed = []
    for record in records:
        decay = record.third_octave_decay
        assert decay is not None
        rows = tuple(row.model_copy(update={
            "t20_s": None, "t30_s": None,
            "t20_unavailable_reason": prose, "t30_unavailable_reason": prose,
            "t20_unavailable_cause": "subband_sampling",
            "t30_unavailable_cause": "subband_sampling",
            "missing_planned_hz": (missing,),
        }) if row.band.nominal_center_hz == nominal else row for row in decay.rows)
        changed.append(replace(record, third_octave_decay=decay.model_copy(
            update={"rows": rows})))
    return tuple(changed)


def _set_t20(
    records: tuple[ReflectionInput, ...], nominal: int, value: float,
) -> tuple[ReflectionInput, ...]:
    changed = []
    for record in records:
        decay = record.third_octave_decay
        assert decay is not None
        rows = tuple(row.model_copy(update={"t20_s": value})
                     if row.band.nominal_center_hz == nominal else row for row in decay.rows)
        changed.append(replace(record, third_octave_decay=decay.model_copy(
            update={"rows": rows})))
    return tuple(changed)


def _cost(records: tuple[ReflectionInput, ...]) -> CategoryEvaluation:
    evaluated = fixtures._evaluate(records)
    registry = load_quality_targets(config_path("quality_targets.toml"))
    return cost_reflections_evaluation(
        evaluated, registry.purpose("dedicated_two_channel_listening_room"),
        registry.fingerprint,
    )


def _unassessed(result: CategoryEvaluation, nominal: int) -> tuple[ReasonCode, ...]:
    assert isinstance(result.category_cost, CategoryCost)
    assert isinstance(result.payload, ReflectionsAndEchoPayload)
    center = next(row.frequency_hz for row in result.payload.wall_pairs[0].bands
                  if row.nominal_center_hz == nominal)
    return next(row.reason_codes for row in result.category_cost.unassessed_bands
                if row.center_frequency_hz == center)


def test_only_upper_half_wall_points_cannot_make_band_measured() -> None:
    band = next(b for b in third_octave_bands() if b.nominal_center_hz == 1000)
    axis = tuple(point for point in GEOMETRIC_LANE_FREQUENCIES_HZ
                 if not band.lower_hz <= point < band.center_hz)
    records = (fixtures._record("left", 1.3, axis=axis),
               fixtures._record("right", 2.5, axis=axis))
    assert _band(records, 1000).round_trip_loss_db.reason_codes == (
        ReasonCode.SUBBAND_SAMPLING_INCOMPLETE,)
    assert _unassessed(_cost(records), 1000) == (
        ReasonCode.SUBBAND_SAMPLING_INCOMPLETE,)


def test_wall_point_causing_full_reflection_cannot_be_deleted_into_clear_band() -> None:
    axis = GEOMETRIC_LANE_FREQUENCIES_HZ
    band = next(b for b in third_octave_bands() if b.nominal_center_hz == 1000)
    point = planned_band_points(axis, band.lower_hz, band.upper_hz)[0]
    complete = fixtures._vary_pair(_fine_pair(), tuple(1.0 for _ in axis))
    assert _band(complete, 1000).round_trip_loss_db.reason_codes == (ReasonCode.FULL_REFLECTION,)
    shortened = tuple(f for f in axis if f != point)
    incomplete = fixtures._vary_pair((fixtures._record("left", 1.3, axis=shortened),
                                      fixtures._record("right", 2.5, axis=shortened)),
                                     tuple(1.0 for _ in shortened))
    assert _band(incomplete, 1000).round_trip_loss_db.reason_codes == (
        ReasonCode.SUBBAND_SAMPLING_INCOMPLETE,)
    assert ReasonCode.SUBBAND_SAMPLING_INCOMPLETE in _unassessed(_cost(incomplete), 1000)


def test_deleting_alert_causing_wall_point_cannot_create_measured_clear_band() -> None:
    axis = GEOMETRIC_LANE_FREQUENCIES_HZ
    band = next(b for b in third_octave_bands() if b.nominal_center_hz == 1000)
    removed = planned_band_points(axis, band.lower_hz, band.upper_hz)[0]
    ordinary = fixtures._vary_pair(_fine_pair(), tuple(0.5 for _ in axis))
    raised = fixtures._vary_pair(_fine_pair(), tuple(
        0.99 if point == removed else 0.5 for point in axis))
    low = _band(ordinary, 1000).decay_duration_s.value
    high = _band(raised, 1000).decay_duration_s.value
    assert low is not None and high is not None and high > low
    threshold = (low + high) / 2.0
    raised_with_t20 = _set_t20(raised, 1000, threshold)
    registry = load_quality_targets(config_path("quality_targets.toml"))
    purpose = registry.purpose("dedicated_two_channel_listening_room")
    assert any(isinstance(alert, FlutterReviewAlert) and alert.nominal_center_hz == 1000
               for alert in reflections_review_alerts(_cost(raised_with_t20), purpose))
    shortened = tuple(point for point in axis if point != removed)
    missing = fixtures._vary_pair((fixtures._record("left", 1.3, axis=shortened),
                                   fixtures._record("right", 2.5, axis=shortened)),
                                  tuple(0.5 for _ in shortened))
    costed = _cost(_set_t20(missing, 1000, threshold))
    assert _unassessed(costed, 1000) == (ReasonCode.SUBBAND_SAMPLING_INCOMPLETE,)
    assert not any(isinstance(alert, FlutterReviewAlert) and alert.nominal_center_hz == 1000
                   for alert in reflections_review_alerts(costed, purpose))


def test_wall_unplanned_point_is_incomplete() -> None:
    band = next(b for b in third_octave_bands() if b.nominal_center_hz == 1000)
    planned = planned_band_points(GEOMETRIC_LANE_FREQUENCIES_HZ,
                                  band.lower_hz, band.upper_hz)
    inserted = (planned[0] + planned[1]) / 2.0
    axis = tuple(sorted((*GEOMETRIC_LANE_FREQUENCIES_HZ, inserted)))
    records = (fixtures._record("left", 1.3, axis=axis),
               fixtures._record("right", 2.5, axis=axis))
    assert _band(records, 1000).round_trip_loss_db.reason_codes == (
        ReasonCode.SUBBAND_SAMPLING_INCOMPLETE,)


def test_verification_axis_has_complete_wall_bands() -> None:
    axis = low_frequency_axis_frequencies(LowFrequencyAxis.VERIFICATION)[1]
    changed = []
    for role, y in (("left", 1.3), ("right", 2.5)):
        record = fixtures._record(role, y, axis=axis)
        top = record.report.top.model_copy(update={"low_frequency_axis": LowFrequencyAxis.VERIFICATION})
        changed.append(replace(record, report=record.report.model_copy(update={"top": top})))
    bands = _control_band_rows(tuple(changed))
    assert bands
    assert all(band.round_trip_loss_db.value is not None for band in bands)


def test_one_missing_band_leaves_other_control_values_and_candidate_cost() -> None:
    axis = GEOMETRIC_LANE_FREQUENCIES_HZ
    band = next(b for b in third_octave_bands() if b.nominal_center_hz == 1000)
    removed = planned_band_points(axis, band.lower_hz, band.upper_hz)[0]
    shortened = tuple(point for point in axis if point != removed)
    base = _control_records()
    records = tuple(replace(fixtures._record(role, y, axis=shortened),
                            third_octave_decay=original.third_octave_decay)
                    for (role, y), original in zip((("left", 1.3), ("right", 2.5)), base, strict=True))
    retention = {point: 0.2 + (index % 17) / 100 for index, point in enumerate(axis)}
    changed = fixtures._vary_pair((records[0], records[1]),
                                  tuple(retention[point] for point in shortened))
    rows = _control_band_rows(changed)
    assert _band(changed, 1000).round_trip_loss_db.value is None
    for row, name, loss, duration, t20 in zip(rows, _NAMES, _LOSS_DB,
                                               _DURATION_S, _ROOM_T20_S, strict=True):
        if name != 1000:
            assert (row.round_trip_loss_db.value, row.decay_duration_s.value,
                    row.room_t20_s.value) == (loss, duration, t20)
    costed = _cost(changed)
    assert costed.state is EvaluationState.COSTED
    assert isinstance(costed.category_cost, CategoryCost)
    assert costed.category_cost.value is not None
    assert _unassessed(costed, 1000) == (ReasonCode.SUBBAND_SAMPLING_INCOMPLETE,)


def test_single_missing_side_and_both_missing_sides_are_unassessed() -> None:
    axis = GEOMETRIC_LANE_FREQUENCIES_HZ
    band = next(b for b in third_octave_bands() if b.nominal_center_hz == 1000)
    removed = planned_band_points(axis, band.lower_hz, band.upper_hz)[0]
    shortened = tuple(point for point in axis if point != removed)
    wall = (fixtures._record("left", 1.3, axis=shortened),
            fixtures._record("right", 2.5, axis=shortened))
    room = _missing_t20(_fine_pair(), 1000, removed)
    for records in (wall, room, _missing_t20(wall, 1000, removed)):
        assert _unassessed(_cost(records), 1000) == (ReasonCode.SUBBAND_SAMPLING_INCOMPLETE,)
    assert _band(wall, 1000).room_t20_s.value is not None
    assert _band(room, 1000).round_trip_loss_db.value is not None


def test_distinct_wall_and_octave_reasons_are_unioned_once() -> None:
    axis = GEOMETRIC_LANE_FREQUENCIES_HZ
    band = next(b for b in third_octave_bands() if b.nominal_center_hz == 1000)
    removed = planned_band_points(axis, band.lower_hz, band.upper_hz)[0]
    shortened = tuple(point for point in axis if point != removed)
    wall = (fixtures._record("left", 1.3, axis=shortened),
            fixtures._record("right", 2.5, axis=shortened))
    changed = []
    for record in wall:
        decay = record.third_octave_decay
        assert decay is not None
        rows = tuple(row.model_copy(update={
            "t20_s": None, "t20_unavailable_reason": "未達下緣",
            "t20_unavailable_cause": "octave_band",
        }) if row.band.nominal_center_hz == 1000 else row for row in decay.rows)
        changed.append(replace(record, third_octave_decay=decay.model_copy(
            update={"rows": rows})))
    expected = {ReasonCode.INSUFFICIENT_DECAY_RANGE,
                ReasonCode.SUBBAND_SAMPLING_INCOMPLETE}
    assert _unassessed(_cost(tuple(changed)), 1000) == tuple(
        code for code in ReasonCode if code in expected)


def test_empty_built_decay_subband_reaches_room_t20_reason(
    solved_report: tuple[three_lane_report.ThreeLaneReport, report_io.ReportInput],
) -> None:
    report, inputs = solved_report
    band = next(b for b in third_octave_bands() if b.nominal_center_hz == 1000)
    late = replace(report.late_decay, bands=tuple(
        point for point in report.late_decay.bands
        if not band.lower_hz <= point.frequency_hz < band.upper_hz))
    built = build_third_octave_decay(replace(report, late_decay=late), inputs)
    row = next(row for row in built.rows if row.band == band)
    assert row.t20_unavailable_cause == "subband_sampling"
    changed = []
    for record in _fine_pair():
        decay = record.third_octave_decay
        assert decay is not None
        rows = tuple(row if old.band == band else old for old in decay.rows)
        changed.append(replace(record, third_octave_decay=decay.model_copy(
            update={"rows": rows})))
    assert _band(tuple(changed), 1000).room_t20_s.reason_codes == (
        ReasonCode.SUBBAND_SAMPLING_INCOMPLETE,)


def test_t20_reason_uses_structural_cause_not_prose() -> None:
    band = next(b for b in third_octave_bands() if b.nominal_center_hz == 1000)
    missing = planned_band_points(GEOMETRIC_LANE_FREQUENCIES_HZ,
                                  band.lower_hz, band.upper_hz)[0]
    for prose in ("任意散文甲", "任意散文乙"):
        records = _missing_t20(_fine_pair(), 1000, missing, prose)
        assert _band(records, 1000).room_t20_s.reason_codes == (
            ReasonCode.SUBBAND_SAMPLING_INCOMPLETE,)


def test_primary_sampling_difference_rejects_pair_and_surrounding_stays_local() -> None:
    left, right = _fine_pair()
    band = next(b for b in third_octave_bands() if b.nominal_center_hz == 1000)
    planned = planned_band_points(GEOMETRIC_LANE_FREQUENCIES_HZ,
                                  band.lower_hz, band.upper_hz)
    changed = _missing_t20((left,), 1000, planned[0])[0]
    other_primary = _missing_t20((right,), 1000, planned[1])[0]
    result = fixtures._evaluate((changed, other_primary))
    assert result.state is EvaluationState.UNAVAILABLE
    assert result.reason_codes == (ReasonCode.REFLECTION_SCREEN_OR_WINDOW_MISMATCH,)
    around = fixtures._record("left", 1.3, "around", axis=GEOMETRIC_LANE_FREQUENCIES_HZ)
    around = _missing_t20((around,), 1000, planned[0])[0]
    other = fixtures._record("right", 2.5, "around", axis=GEOMETRIC_LANE_FREQUENCIES_HZ)
    local = fixtures._evaluate((left, right, around, other))
    assert local.state is EvaluationState.MEASURED
    assert isinstance(local.payload, ReflectionsAndEchoPayload)
    affected = next(channel for channel in local.payload.channels
                    if channel.role == "left" and channel.receiver_id == "around")
    assert affected.reason_codes == (ReasonCode.REFLECTION_SCREEN_OR_WINDOW_MISMATCH,)
    unaffected = next(channel for channel in local.payload.channels
                      if channel.role == "right" and channel.receiver_id == "around")
    assert unaffected.reason_codes == ()


def test_incomplete_band_without_alert_is_clear_but_ranked_as_unassessed() -> None:
    axis = GEOMETRIC_LANE_FREQUENCIES_HZ
    band = next(b for b in third_octave_bands() if b.nominal_center_hz == 1000)
    removed = planned_band_points(axis, band.lower_hz, band.upper_hz)[0]
    shortened = tuple(point for point in axis if point != removed)
    measured = fixtures._evaluate((fixtures._record("left", 1.3, axis=shortened),
                                   fixtures._record("right", 2.5, axis=shortened)))
    registry = recost._registry_for(QualityCategory.REFLECTIONS_AND_ECHO)
    ranking = rank_candidates((ranking_fixtures._candidate(measured),),
                              registry, recost._CONTEXT)
    assert ranking.status_of(measured.candidate_id) is CandidateStatus.RANKABLE
    row = ranking.rankable[0]
    assert row.review_status is ReviewStatus.CLEAR
    assert row.categories[0].unassessed_bands
    assert all(band.reason_codes for band in row.categories[0].unassessed_bands)
    assert any(ReasonCode.SUBBAND_SAMPLING_INCOMPLETE in band.reason_codes
               for band in row.categories[0].unassessed_bands)


def test_evaluator_version_is_v2() -> None:
    assert REFLECTIONS_AND_ECHO_EVALUATOR_VERSION == "aosr.scoring.reflections.v2"
