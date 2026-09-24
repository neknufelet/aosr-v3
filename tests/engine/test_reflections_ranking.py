"""#351 反射代價、逐區標記與比較支撐進排名表。"""
from __future__ import annotations

from pathlib import Path

from aosr.config.paths import config_path
from aosr.config.quality_targets import QualityTargets, load_quality_targets
from aosr.scoring.contract import (
    CONTRACT_SCHEMA_VERSION, CandidateEvaluation, CategoryCost, CategoryEvaluation,
    Flag, QualityCategory,
)
from aosr.scoring.ranking import CandidateStatus, rank_candidates
from aosr.scoring.reflections_contract import ReflectionsAndEchoPayload
from aosr.scoring.reflections_cost import comparison_support, cost_reflections_evaluation
from tests.engine import test_ranking_recost as recost
from tests.engine import test_reflections as fixtures
from tests.engine import test_reflections_cost as costs


def _candidate(evaluation: CategoryEvaluation) -> CandidateEvaluation:
    return CandidateEvaluation(
        schema_version=CONTRACT_SCHEMA_VERSION, candidate_id=evaluation.candidate_id,
        scene_fingerprint=evaluation.scene_fingerprint, evaluations=(evaluation,),
    )


def test_measured_reflections_rank_and_keep_method_flags() -> None:
    measured = fixtures._evaluate(fixtures._pair())
    registry = recost._registry_for(QualityCategory.REFLECTIONS_AND_ECHO)
    result = rank_candidates((_candidate(measured),), registry, recost._CONTEXT)
    assert result.status_of(measured.candidate_id) is CandidateStatus.RANKABLE
    row = result.rankable[0]
    assert {Flag.WINDOW_ONLY_DELAY_SCREEN, Flag.NO_DIRECTIVITY} <= set(row.flags)
    assert Flag.REFLECTION_LATERAL_ABOVE_THRESHOLD in row.flags
    assert {Flag.REFLECTION_FRONT_ABOVE_THRESHOLD, Flag.REFLECTION_REAR_ABOVE_THRESHOLD,
            Flag.REFLECTION_VERTICAL_ABOVE_THRESHOLD}.isdisjoint(row.flags)
    cost = row.categories[0].evaluation.category_cost
    assert isinstance(cost, CategoryCost)
    assert row.total_cost < cost.components["left.lateral"]


def test_stale_upstream_zone_flag_is_removed_on_recost() -> None:
    measured = fixtures._evaluate(fixtures._pair())
    registry = recost._registry_for(QualityCategory.REFLECTIONS_AND_ECHO)
    fake = recost._upstream_costed(measured, value=99.0, components={}, fingerprint="old")
    fake = fake.model_copy(update={"flags": (*fake.flags, Flag.REFLECTION_REAR_ABOVE_THRESHOLD)})
    result = rank_candidates((_candidate(fake),), registry, recost._CONTEXT)
    assert result.status_of(measured.candidate_id) is CandidateStatus.RANKABLE
    row = result.rankable[0]
    assert Flag.REFLECTION_REAR_ABOVE_THRESHOLD not in row.flags
    cost = row.categories[0].evaluation.category_cost
    assert isinstance(cost, CategoryCost)
    assert cost.value != 99.0


def test_surrounding_point_level_does_not_change_primary_cost() -> None:
    inputs = (*fixtures._pair(), fixtures._record("left", 1.3, "s1", receiver_y=2.2),
              fixtures._record("right", 2.5, "s1", receiver_y=2.2))
    measured = fixtures._evaluate(inputs)
    assert isinstance(measured.payload, ReflectionsAndEchoPayload)
    registry = load_quality_targets(config_path("quality_targets.toml"))
    purpose = registry.purpose(recost._PURPOSE)
    original = cost_reflections_evaluation(measured, purpose, registry.fingerprint)
    assert isinstance(original.category_cost, CategoryCost)
    channels = []
    for channel in measured.payload.channels:
        if channel.is_primary:
            channels.append(channel)
            continue
        zones = tuple(zone.model_copy(update={"points": tuple(
            point.model_copy(update={"strongest_level_db": 30.0})
            if point.strongest_level_db is not None else point for point in zone.points
        )}) for zone in channel.zones)
        channels.append(channel.model_copy(update={"zones": zones}))
    payload = measured.payload.model_copy(update={"channels": tuple(channels)})
    changed = cost_reflections_evaluation(measured.model_copy(update={"payload": payload}),
                                          purpose, registry.fingerprint)
    assert isinstance(changed.category_cost, CategoryCost)
    assert changed.category_cost.value == original.category_cost.value
    # 周圍點每一區都超標，逐區標記也只看主位：跟沒改之前一模一樣（前、上下兩區主位沒超標，不准冒出來）。
    assert changed.flags == original.flags
    assert Flag.REFLECTION_FRONT_ABOVE_THRESHOLD not in changed.flags


def test_comparison_support_ignores_computed_order_but_keeps_primary_axis() -> None:
    measured = fixtures._evaluate(fixtures._pair())
    assert isinstance(measured.payload, ReflectionsAndEchoPayload)
    channels = tuple(channel.model_copy(update={"computed_order_k": 4, "validation": "unvalidated"})
                     for channel in measured.payload.channels)
    changed = measured.model_copy(update={"payload": measured.payload.model_copy(update={"channels": channels})})
    assert comparison_support(measured) == comparison_support(changed)
    support = comparison_support(measured)
    assert '"scoring_axis_min_hz":300.0' in support
    assert '"scoring_axis_max_hz":8000.0' in support
    assert '"primary_receiver_id":"main"' in support
    assert '"scoring_axis_point_count":' in support
    assert 'computed_order_k' not in support
    import json
    primary = [channel for channel in measured.payload.channels if channel.is_primary]
    assert json.loads(support)["channels"] == [
        {"role": channel.role, "speaker_id": channel.speaker_id} for channel in primary]
    renamed = measured.model_copy(update={"payload": measured.payload.model_copy(update={"channels": tuple(
        channel.model_copy(update={"speaker_id": channel.speaker_id + "-other"})
        for channel in measured.payload.channels)})})
    assert comparison_support(renamed) != support  # 主位喇叭不同的候選不准同表
    first = measured.model_copy(update={"candidate_id": "candidate-order-3"})
    second = changed.model_copy(update={"candidate_id": "candidate-order-4"})
    registry = recost._registry_for(QualityCategory.REFLECTIONS_AND_ECHO)
    result = rank_candidates((_candidate(first), _candidate(second)), registry, recost._CONTEXT)
    assert {row.candidate_id for row in result.rankable} == {
        "candidate-order-3", "candidate-order-4"}


def test_stale_excess_flag_disappears_when_threshold_is_relaxed(tmp_path: Path) -> None:
    measured = fixtures._evaluate(fixtures._pair())
    registry = recost._registry_for(QualityCategory.REFLECTIONS_AND_ECHO)
    costed = cost_reflections_evaluation(measured, registry.purpose(recost._PURPOSE), registry.fingerprint)
    assert Flag.REFLECTION_LATERAL_ABOVE_THRESHOLD in costed.flags
    path = costs._alter(tmp_path, "reflections_and_echo.zone_threshold_db.lateral", "value", "0.0")
    document = load_quality_targets(path).model_dump(mode="json", by_alias=True)
    for row in document["purpose"][0]["qualification"]:
        if row["key"] == "ranking.mandatory_categories":
            row["value"] = [QualityCategory.REFLECTIONS_AND_ECHO.value]
        elif row["key"] == "ranking.optional_categories":
            row["value"] = [name for name in row["value"]
                            if name != QualityCategory.REFLECTIONS_AND_ECHO.value]
    relaxed = QualityTargets.model_validate(document)
    result = rank_candidates((_candidate(costed),), relaxed, recost._CONTEXT)
    assert result.status_of(measured.candidate_id) is CandidateStatus.RANKABLE
    assert Flag.REFLECTION_LATERAL_ABOVE_THRESHOLD not in result.rankable[0].flags


def test_unavailable_reflections_go_through_ranking_without_alerts_or_support() -> None:
    """反射不可估時沒有輸出：排名層照樣收集各類警戒，不能因為這一類沒有輸出就當掉。"""
    from dataclasses import replace

    from aosr.scoring.contract import EvaluationState
    from aosr.scoring.reflections_cost import reflections_review_alerts

    left, right = fixtures._pair()
    unavailable = fixtures._evaluate((replace(left, screen=None), right))
    assert unavailable.state is EvaluationState.UNAVAILABLE and unavailable.payload is None
    registry = recost._registry_for(QualityCategory.REFLECTIONS_AND_ECHO)
    result = rank_candidates((_candidate(unavailable),), registry, recost._CONTEXT)
    assert result.status_of(unavailable.candidate_id) is CandidateStatus.NOT_EVALUATED
    assert reflections_review_alerts(unavailable, registry.purpose("dedicated_two_channel_listening_room")) == ()
    assert comparison_support(unavailable) == ""
