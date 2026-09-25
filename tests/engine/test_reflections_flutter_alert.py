"""#351 牆對顫動與本房 T20 比較的複核警戒考卷。"""
from __future__ import annotations

from pathlib import Path

import pytest

from aosr.config.frequency_axis import GEOMETRIC_LANE_FREQUENCIES_HZ
from aosr.config.paths import config_path
from aosr.config.quality_targets import QualityPurpose, SettingEntry, load_quality_targets
from aosr.physics.third_octave_decay import third_octave_bands
from aosr.scoring.contract import CategoryCost, CategoryEvaluation, MetricState, QualityCategory, ReasonCode
from aosr.scoring.ranking import CandidateStatus, ExternalAcceptance, ExternalFloors, rank_candidates
from aosr.scoring.recommendation import NotFinalReason, ReviewStatus
from aosr.scoring.reflections_contract import MetricCell, ReflectionsAndEchoPayload
from aosr.scoring.reflections import ReflectionInput
from aosr.scoring.reflections_cost import cost_reflections_evaluation, reflections_review_alerts
from aosr.scoring.review_alert import FlutterReviewAlert
from tests.engine import test_ranking_recost as recost
from tests.engine import test_reflections as fixtures
from tests.engine import test_reflections_cost as costs
from tests.engine import test_reflections_ranking as ranking_fixtures


def _change_band(
    evaluation: CategoryEvaluation, center: int, walls: tuple[str, str] | None,
    replacement: dict[str, MetricCell],
) -> CategoryEvaluation:
    payload = evaluation.payload
    assert isinstance(payload, ReflectionsAndEchoPayload)
    pairs = []
    for pair in payload.wall_pairs:
        if walls is not None and pair.walls != walls:
            pairs.append(pair)
            continue
        bands = tuple(band.model_copy(update=replacement) if band.nominal_center_hz == center else band
                      for band in pair.bands)
        pairs.append(pair.model_copy(update={"bands": bands}))
    return evaluation.model_copy(update={"payload": payload.model_copy(update={"wall_pairs": tuple(pairs)})})


def _purpose() -> QualityPurpose:
    return load_quality_targets(config_path("quality_targets.toml")).purpose(
        recost._PURPOSE)


def _alerts(evaluation: CategoryEvaluation) -> tuple[FlutterReviewAlert, ...]:
    alerts = reflections_review_alerts(evaluation, _purpose())
    return tuple(alert for alert in alerts if isinstance(alert, FlutterReviewAlert))


def _fine_pair() -> tuple[ReflectionInput, ReflectionInput]:
    axis = GEOMETRIC_LANE_FREQUENCIES_HZ
    return fixtures._record("left", 1.3, axis=axis), fixtures._record("right", 2.5, axis=axis)


def test_lower_t20_alerts_but_equal_duration_does_not() -> None:
    measured = fixtures._evaluate(_fine_pair())
    assert isinstance(measured.payload, ReflectionsAndEchoPayload)
    pair = measured.payload.wall_pairs[0]
    band = next(band for band in pair.bands if band.nominal_center_hz == 1000)
    assert band.decay_duration_s.value is not None
    lowered = _change_band(measured, 1000, pair.walls, {"room_t20_s": MetricCell(
        value=band.decay_duration_s.value / 2, state=MetricState.MEASURED, reason_codes=())})
    selected = tuple(alert for alert in _alerts(lowered)
                     if alert.nominal_center_hz == 1000 and alert.walls == pair.walls)
    assert [alert.decay_duration_s for alert in selected] == [band.decay_duration_s.value]
    assert (selected[0].center_frequency_hz, selected[0].lower_hz,
            selected[0].upper_hz) == (band.frequency_hz, band.lower_hz, band.upper_hz)
    assert "待複核" in selected[0].note
    equal = _change_band(measured, 1000, pair.walls, {"room_t20_s": MetricCell(
        value=band.decay_duration_s.value, state=MetricState.MEASURED, reason_codes=())})
    assert not any(alert.nominal_center_hz == 1000 and alert.walls == pair.walls
                   for alert in _alerts(equal))


def test_full_reflection_alerts_and_zero_retention_does_not() -> None:
    measured = fixtures._evaluate(_fine_pair())
    pair_walls = ("x0", "xL")
    full = _change_band(measured, 1000, pair_walls, {
        "round_trip_loss_db": MetricCell(value=None, state=MetricState.NOT_COMPUTABLE,
                                         reason_codes=(ReasonCode.FULL_REFLECTION,)),
        "decay_duration_s": MetricCell(value=None, state=MetricState.NOT_COMPUTABLE,
                                       reason_codes=(ReasonCode.FULL_REFLECTION,)),
    })
    selected = tuple(alert for alert in _alerts(full) if alert.nominal_center_hz == 1000
                     and alert.walls == pair_walls)
    assert [alert.decay_duration_s for alert in selected] == [None]
    assert "全反射，持續度無限長" in selected[0].note
    zero = _change_band(measured, 1000, pair_walls, {
        "round_trip_loss_db": MetricCell(value=None, state=MetricState.NOT_COMPUTABLE,
                                         reason_codes=(ReasonCode.ZERO_RETENTION,)),
        "decay_duration_s": MetricCell(value=None, state=MetricState.NOT_COMPUTABLE,
                                       reason_codes=(ReasonCode.ZERO_RETENTION,)),
    })
    assert not any(alert.nominal_center_hz == 1000 and alert.walls == pair_walls
                   for alert in _alerts(zero))


def test_unavailable_t20_and_missing_band_points_are_unassessed_once() -> None:
    measured = fixtures._evaluate(_fine_pair())
    no_t20 = _change_band(measured, 1000, None, {"room_t20_s": MetricCell(
        value=None, state=MetricState.UNAVAILABLE,
        reason_codes=(ReasonCode.T20_BAND_UNAVAILABLE,))})
    registry = load_quality_targets(config_path("quality_targets.toml"))
    costed = cost_reflections_evaluation(no_t20, registry.purpose(recost._PURPOSE), registry.fingerprint)
    assert isinstance(costed.category_cost, CategoryCost)
    assert isinstance(no_t20.payload, ReflectionsAndEchoPayload)
    center = next(band.frequency_hz for band in no_t20.payload.wall_pairs[0].bands
                  if band.nominal_center_hz == 1000)
    selected = tuple(band for band in costed.category_cost.unassessed_bands
                     if band.center_frequency_hz == center)
    assert [band.reason_codes for band in selected] == [(ReasonCode.T20_BAND_UNAVAILABLE,)]
    assert not any(alert.nominal_center_hz == 1000 for alert in _alerts(no_t20))
    assert isinstance(measured.payload, ReflectionsAndEchoPayload)
    low = next(band.frequency_hz for band in measured.payload.wall_pairs[0].bands
               if band.nominal_center_hz == 400)
    sparse = fixtures._evaluate(fixtures._pair())
    plain = cost_reflections_evaluation(sparse, registry.purpose(recost._PURPOSE), registry.fingerprint)
    assert isinstance(plain.category_cost, CategoryCost)
    assert (low, (ReasonCode.SUBBAND_SAMPLING_INCOMPLETE,)) in {
        (band.center_frequency_hz, band.reason_codes) for band in plain.category_cost.unassessed_bands}


def test_flutter_decay_registry_change_rejects_stale_payload(tmp_path: Path) -> None:
    measured = fixtures._evaluate(_fine_pair())
    path = costs._alter(tmp_path, "reflections_and_echo.flutter_decay_db", "value", "61.0")
    registry = load_quality_targets(path)
    with pytest.raises(ValueError, match="顫動衰減量不同"):
        cost_reflections_evaluation(measured, registry.purpose(recost._PURPOSE), registry.fingerprint)


def test_alert_reaches_rankable_and_eliminated_rows_without_changing_cost() -> None:
    measured = fixtures._evaluate(_fine_pair())
    assert isinstance(measured.payload, ReflectionsAndEchoPayload)
    band = next(band for band in measured.payload.wall_pairs[0].bands if band.nominal_center_hz == 1000)
    assert band.decay_duration_s.value is not None
    alerted = _change_band(measured, 1000, ("x0", "xL"), {"room_t20_s": MetricCell(
        value=band.decay_duration_s.value / 2, state=MetricState.MEASURED, reason_codes=())})
    registry = recost._registry_for(QualityCategory.REFLECTIONS_AND_ECHO)
    clean_result = rank_candidates((ranking_fixtures._candidate(measured),), registry, recost._CONTEXT)
    result = rank_candidates((ranking_fixtures._candidate(alerted),), registry, recost._CONTEXT)
    assert result.status_of(measured.candidate_id) is CandidateStatus.RANKABLE
    row = result.rankable[0]
    assert row.review_status is ReviewStatus.PENDING
    assert NotFinalReason.REVIEW_PENDING in row.not_final_reasons
    assert any(alert.kind == "flutter" for alert in row.review_alerts)
    assert row.total_cost == clean_result.rankable[0].total_cost
    assert row.rank == clean_result.rankable[0].rank
    external = ExternalFloors(declared_by="fixture", verdicts={measured.candidate_id: ExternalAcceptance.FAILED})
    eliminated = rank_candidates((ranking_fixtures._candidate(alerted),), registry,
                                 recost._CONTEXT, external)
    assert eliminated.status_of(measured.candidate_id) is CandidateStatus.ELIMINATED
    assert any(alert.kind == "flutter" for alert in eliminated.eliminated[0].review_alerts)


def test_below_alert_list_is_diagnostic_only() -> None:
    measured = fixtures._evaluate(fixtures._pair())
    full = _change_band(measured, 125, ("x0", "xL"), {
        "round_trip_loss_db": MetricCell(value=None, state=MetricState.NOT_COMPUTABLE,
                                         reason_codes=(ReasonCode.FULL_REFLECTION,)),
        "decay_duration_s": MetricCell(value=None, state=MetricState.NOT_COMPUTABLE,
                                       reason_codes=(ReasonCode.FULL_REFLECTION,)),
    })
    assert not any(alert.nominal_center_hz == 125 for alert in _alerts(full))


def test_flutter_alert_does_not_move_candidate_in_two_row_ranking() -> None:
    measured = fixtures._evaluate(_fine_pair())
    assert isinstance(measured.payload, ReflectionsAndEchoPayload)
    band = next(band for band in measured.payload.wall_pairs[0].bands
                if band.nominal_center_hz == 1000)
    assert band.decay_duration_s.value is not None
    alerted = _change_band(measured, 1000, ("x0", "xL"), {"room_t20_s": MetricCell(
        value=band.decay_duration_s.value / 2, state=MetricState.MEASURED,
        reason_codes=())})
    channels = tuple(channel.model_copy(update={"zones": tuple(
        zone.model_copy(update={"points": tuple(point.model_copy(update={
            "strongest_level_db": point.strongest_level_db + 5.0})
            if point.strongest_level_db is not None else point for point in zone.points)})
        for zone in channel.zones)}) for channel in measured.payload.channels)
    worse = measured.model_copy(update={
        "candidate_id": "candidate-worse",
        "payload": measured.payload.model_copy(update={"channels": channels}),
    })
    registry = recost._registry_for(QualityCategory.REFLECTIONS_AND_ECHO)
    plain = rank_candidates((ranking_fixtures._candidate(measured),
                             ranking_fixtures._candidate(worse)), registry, recost._CONTEXT)
    pending = rank_candidates((ranking_fixtures._candidate(alerted),
                               ranking_fixtures._candidate(worse)), registry, recost._CONTEXT)
    assert [(row.candidate_id, row.rank, row.total_cost) for row in plain.rankable] == [
        (row.candidate_id, row.rank, row.total_cost) for row in pending.rankable]
    assert pending.rankable[0].review_status is ReviewStatus.PENDING


def test_official_axis_covers_both_halves_of_every_flutter_band_equally() -> None:
    registry = load_quality_targets(config_path("quality_targets.toml"))
    purpose = registry.purpose(recost._PURPOSE)
    entry = purpose.entry("reflections_and_echo.flutter_alert_band_centers_hz")
    assert isinstance(entry, SettingEntry)
    assert isinstance(entry.value, tuple)
    selected = set(entry.value)
    counts = []
    seen: set[int] = set()
    for band in third_octave_bands():
        if band.nominal_center_hz not in selected:
            continue
        seen.add(band.nominal_center_hz)
        lower = tuple(frequency for frequency in GEOMETRIC_LANE_FREQUENCIES_HZ
                      if band.lower_hz <= frequency < band.center_hz)
        upper = tuple(frequency for frequency in GEOMETRIC_LANE_FREQUENCIES_HZ
                      if band.center_hz <= frequency <= band.upper_hz)
        assert lower and upper, band.nominal_center_hz
        counts.append(len(lower) + len(upper))
    assert counts and all(count == counts[0] for count in counts[1:])
    assert seen == selected


def test_alert_uses_exact_center_room_t20_and_no_multiplier() -> None:
    """長過一點點就掛（不設倍率）；警戒帶子帶精確中心（不是標稱名）與本房那一帶的 T20。

    1000 Hz 那一帶精確中心剛好等於標稱名，用它驗不出帶錯中心，所以這題用 1250 Hz 帶。
    """
    measured = fixtures._evaluate(_fine_pair())
    assert isinstance(measured.payload, ReflectionsAndEchoPayload)
    pair = measured.payload.wall_pairs[0]
    band = next(band for band in pair.bands if band.nominal_center_hz == 1250)
    assert band.frequency_hz != float(band.nominal_center_hz)
    duration = band.decay_duration_s.value
    assert duration is not None
    t20 = duration * 0.99
    lowered = _change_band(measured, 1250, pair.walls, {"room_t20_s": MetricCell(
        value=t20, state=MetricState.MEASURED, reason_codes=())})
    selected = [alert for alert in _alerts(lowered)
                if alert.nominal_center_hz == 1250 and alert.walls == pair.walls]
    assert [alert.center_frequency_hz for alert in selected] == [band.frequency_hz]
    assert [alert.room_t20_s for alert in selected] == [t20]
    assert [alert.decay_duration_s for alert in selected] == [duration]
