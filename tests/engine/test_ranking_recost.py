"""排名層一律從原始量重算類代價的反例考卷（票 #392）。"""

from __future__ import annotations

from datetime import date
from typing import Final, Literal

import pytest

from aosr.config.paths import config_path
from aosr.config.quality_targets import QualityTargets, load_quality_targets
from aosr.scoring import channel_matching_cost, listening_area_cost, ranking
from aosr.scoring.category_registry import CATEGORY_REGISTRY
from aosr.scoring.contract import (
    CONTRACT_SCHEMA_VERSION,
    CandidateEvaluation,
    CategoryCost,
    CategoryEvaluation,
    EvaluationState,
    InputProvenance,
    QualityCategory,
)
from aosr.scoring.ranking import (
    CandidateStatus,
    EliminationReason,
    RankingContext,
)


_PURPOSE: Final[str] = "dedicated_two_channel_listening_room"
_CANDIDATE: Final[str] = "candidate-recost"
_SCENE_FINGERPRINT: Final[str] = "a" * 64
_PROVENANCE: Final[InputProvenance] = InputProvenance(
    report_id="recost-report",
    engine_commit="fixture-engine",
    speaker_id="speaker-fixture",
    receiver_id="receiver-fixture",
)
_CONTEXT: Final[RankingContext] = RankingContext(
    purpose=_PURPOSE,
    receiver_set_fingerprint="receiver-set-fixture",
    channel_group_fingerprint="channel-group-fixture",
    run_date=date(2026, 9, 21),
    engine_version="engine-fixture",
)


def _registry_for(category: QualityCategory) -> QualityTargets:
    document = load_quality_targets(
        config_path("quality_targets.toml")
    ).model_dump(mode="json", by_alias=True)
    for row in document["purpose"][0]["qualification"]:
        if row["key"] == "ranking.mandatory_categories":
            row["value"] = [category.value]
        elif row["key"] == "ranking.optional_categories":
            row["value"] = [
                name for name in row["value"] if name != category.value
            ]
    return QualityTargets.model_validate(document)


def _candidate(evaluation: CategoryEvaluation) -> CandidateEvaluation:
    return CandidateEvaluation(
        schema_version=CONTRACT_SCHEMA_VERSION,
        candidate_id=_CANDIDATE,
        scene_fingerprint=evaluation.scene_fingerprint,
        evaluations=(evaluation,),
    )


def _upstream_costed(
    measured: CategoryEvaluation,
    *,
    value: float,
    components: dict[str, float],
    fingerprint: str,
) -> CategoryEvaluation:
    document = measured.model_dump(mode="python")
    document.update(
        state=EvaluationState.COSTED,
        category_cost=CategoryCost(
            value=value,
            components=components,
            cost_settings_fingerprint=fingerprint,
        ),
    )
    return CategoryEvaluation.model_validate(document)


def _listening_comparison(worst: float, mean: float = 0.0) -> dict[str, object]:
    def aggregate(receiver_id: str) -> dict[str, object]:
        return {
            "weighted_mean_deviation": mean,
            "worst_deviation": {
                "value": worst,
                "receiver": {
                    "receiver_id": receiver_id,
                    "direction_relative_to_primary": "front",
                    "importance": 1.0,
                },
                "reference": {
                    "receiver_id": "main",
                    "direction_relative_to_primary": None,
                    "importance": 1.0,
                },
            },
        }

    return {
        "primary_to_surrounding": aggregate("front"),
        "surrounding_to_surrounding": aggregate("back"),
        "surrounding_to_surrounding_reason": None,
    }


def _listening_measured(
    *, tilt_worst: float, tilt_mean: float = 0.0
) -> CategoryEvaluation:
    quiet = _listening_comparison(0.0)
    return CategoryEvaluation.model_validate(
        {
            "schema_version": CONTRACT_SCHEMA_VERSION,
            "candidate_id": _CANDIDATE,
            "scene_fingerprint": _SCENE_FINGERPRINT,
            "category": "listening_area_stability",
            "state": "measured",
            "payload": {
                "category": "listening_area_stability",
                "candidate_id": _CANDIDATE,
                "speaker_id": "speaker-fixture",
                "receiver_set_fingerprint": "receiver-set-fixture",
                "timbre_settings_fingerprint": "timbre-settings-fixture",
                "settings_fingerprint": "listening-settings-fixture",
                "point_provenance": [
                    {
                        "receiver_id": receiver,
                        "report_id": f"report-{receiver}",
                        "evaluator_version": "timbre-fixture-v1",
                        "settings_fingerprint": "timbre-settings-fixture",
                    }
                    for receiver in ("main", "front")
                ],
                "tilt_stability": _listening_comparison(tilt_worst, tilt_mean),
                "ripple_rms_stability": quiet,
                "overall_level_stability": quiet,
                "peak_dip_consistency": quiet,
                "peak_dip_occurrences": [],
                "target_deviation_position_spread_curve_db": [],
                "target_deviation_common_frequency_count": 0,
                "target_deviation_discarded_frequency_value_count": 0,
            },
            "raw_quantities": [
                {"name": "tilt_worst", "value": tilt_worst, "unit": "dB/oct"}
            ],
            "category_cost": None,
            "flags": [],
            "reason_codes": [],
            "evaluator_version": "listening-fixture-v1",
            "settings_fingerprint": "listening-settings-fixture",
            "provenance": _PROVENANCE,
        }
    )


_BROADBAND_SUPPORT: Final[dict[str, float | int]] = {
    "lowest_frequency_hz": 100.0,
    "highest_frequency_hz": 200.0,
    "frequency_count": 2,
}


def _channel_summary(mean: float, worst: float) -> dict[str, object]:
    return {
        "weighted_mean_absolute_difference": mean,
        "worst_absolute_difference": worst,
        "worst_receiver_id": "main",
    }


def _channel_measured(
    *, tilt_worst: float, tilt_mean: float = 0.0
) -> CategoryEvaluation:
    return CategoryEvaluation.model_validate(
        {
            "schema_version": CONTRACT_SCHEMA_VERSION,
            "candidate_id": _CANDIDATE,
            "scene_fingerprint": _SCENE_FINGERPRINT,
            "category": "channel_matching",
            "state": "measured",
            "payload": {
                "category": "channel_matching",
                "candidate_id": _CANDIDATE,
                "receiver_set_fingerprint": "receiver-set-fixture",
                "timbre_settings_fingerprint": "timbre-settings-fixture",
                "listening_area_settings_fingerprint": "listening-settings-fixture",
                "channel_group_fingerprint": "channel-group-fixture",
                "channels": [
                    {"role": "left", "speaker_id": "speaker-left"},
                    {"role": "right", "speaker_id": "speaker-right"},
                ],
                "comparisons": [{"left_role": "left", "right_role": "right"}],
                "point_sources": [],
                "point_results": [
                    {
                        "receiver_id": "main",
                        "importance": 1.0,
                        "left_role": "left",
                        "right_role": "right",
                        "state": "measured",
                        "reason_codes": [],
                        "reason": None,
                        "tilt_difference_db_per_octave": 0.0,
                        "ripple_rms_difference_db": 0.0,
                        "broadband_level_difference_db": 0.0,
                        "direct_time_difference_ms": 0.0,
                        "frequency_difference_curve_db": [],
                        "unmatched_features": [],
                    }
                ],
                "aggregates": [
                    {
                        "left_role": "left",
                        "right_role": "right",
                        "assessed_receiver_ids": ["main"],
                        "unavailable_receiver_ids": [],
                        "tilt_difference": _channel_summary(tilt_mean, tilt_worst),
                        "ripple_rms_difference": _channel_summary(0.0, 0.0),
                        "broadband_level_difference": _channel_summary(0.0, 0.0),
                        "direct_time_difference": _channel_summary(0.0, 0.0),
                    }
                ],
                "broadband_support": _BROADBAND_SUPPORT,
                "direct_time_cost_enabled": False,
            },
            "raw_quantities": [
                {"name": "tilt_worst", "value": tilt_worst, "unit": "dB/oct"}
            ],
            "category_cost": None,
            "flags": [],
            "reason_codes": [],
            "evaluator_version": "channel-fixture-v1",
            "settings_fingerprint": "channel-settings-fixture",
            "provenance": _PROVENANCE,
        }
    )


def _input_shape(
    measured: CategoryEvaluation,
    registry: QualityTargets,
    shape: Literal["measured", "complete", "fake"],
) -> CategoryEvaluation:
    if shape == "measured":
        return measured
    if shape == "fake":
        return _upstream_costed(
            measured,
            value=0.0,
            components={},
            fingerprint="stale-upstream-registry",
        )
    registration = CATEGORY_REGISTRY[measured.category]
    return registration.coster(
        measured,
        registry.purpose(_PURPOSE),
        registry.fingerprint,
    )


@pytest.mark.parametrize("shape", ("measured", "complete", "fake"))
def test_listening_area_recosts_every_input_shape_before_floor_check(
    shape: Literal["measured", "complete", "fake"],
) -> None:
    """只要原始最差值超標，空分項的上游代價也不能讓候選逃過淘汰。"""
    registry = _registry_for(QualityCategory.LISTENING_AREA_STABILITY)
    evaluation = _input_shape(_listening_measured(tilt_worst=2.0), registry, shape)

    result = ranking.rank_candidates([_candidate(evaluation)], registry, _CONTEXT)

    assert result.status_of(_CANDIDATE) is CandidateStatus.ELIMINATED
    (row,) = result.eliminated
    assert row.reasons == (
        EliminationReason.LISTENING_AREA_TILT_PRIMARY_TO_SURROUNDING_WORST_BEYOND_LIMIT,
        EliminationReason.LISTENING_AREA_TILT_SURROUNDING_TO_SURROUNDING_WORST_BEYOND_LIMIT,
    )


@pytest.mark.parametrize("shape", ("measured", "complete", "fake"))
def test_channel_matching_recosts_every_input_shape_before_floor_check(
    shape: Literal["measured", "complete", "fake"],
) -> None:
    """聲道最差值超標時，排名層不信任完整或空殼的上游代價。"""
    registry = _registry_for(QualityCategory.CHANNEL_MATCHING)
    evaluation = _input_shape(_channel_measured(tilt_worst=2.0), registry, shape)

    result = ranking.rank_candidates([_candidate(evaluation)], registry, _CONTEXT)

    assert result.status_of(_CANDIDATE) is CandidateStatus.ELIMINATED
    (row,) = result.eliminated
    assert row.reasons == (
        EliminationReason.CHANNEL_MATCHING_TILT_WORST_BEYOND_LIMIT,
    )


@pytest.mark.parametrize(
    "measured",
    (
        _listening_measured(tilt_worst=1.0, tilt_mean=1.0),
        _channel_measured(tilt_worst=1.0, tilt_mean=1.0),
    ),
    ids=lambda item: str(item.category.value),
)
def test_stale_upstream_cost_and_fingerprint_are_replaced(
    measured: CategoryEvaluation,
) -> None:
    """上游的假值與舊指紋都不可出現在排名表；表上的代價等於這一跑從原始量重算的值。"""
    registry = _registry_for(measured.category)
    expected = _input_shape(measured, registry, "complete").category_cost
    assert expected is not None
    assert expected.value > 0.0
    fake = _upstream_costed(
        measured,
        value=99.0,
        components={"invented": 99.0},
        fingerprint="stale-upstream-registry",
    )

    result = ranking.rank_candidates([_candidate(fake)], registry, _CONTEXT)

    assert result.status_of(_CANDIDATE) is CandidateStatus.RANKABLE
    (row,) = result.rankable
    (line,) = row.categories
    assert line.category_cost == expected.value
    assert line.identity.cost_settings_fingerprint == registry.fingerprint
    assert line.evaluation.category_cost == expected


def test_listening_area_floor_rejects_a_missing_required_component() -> None:
    """有原始比較組卻少了最差值分項，是程式錯誤而不是零違規。"""
    registry = _registry_for(QualityCategory.LISTENING_AREA_STABILITY)
    costed = listening_area_cost.cost_listening_area_evaluation(
        _listening_measured(tilt_worst=0.0),
        registry.purpose(_PURPOSE),
        registry.fingerprint,
    )
    assert costed.category_cost is not None
    components = dict(costed.category_cost.components)
    components.pop("tilt_worst_deviation.primary_to_surrounding")
    malformed = _upstream_costed(
        _listening_measured(tilt_worst=0.0),
        value=costed.category_cost.value,
        components=components,
        fingerprint=registry.fingerprint,
    )

    with pytest.raises(ValueError, match="tilt_worst_deviation"):
        listening_area_cost.listening_area_floor_reasons(
            malformed, registry.purpose(_PURPOSE)
        )


def test_channel_floor_rejects_missing_worst_details_for_an_assessed_metric() -> None:
    """有可估的傾斜彙總卻沒有任何 worst 明細時，不准靜靜回空。"""
    registry = _registry_for(QualityCategory.CHANNEL_MATCHING)
    costed = channel_matching_cost.cost_channel_matching_evaluation(
        _channel_measured(tilt_worst=0.0),
        registry.purpose(_PURPOSE),
        registry.fingerprint,
    )
    assert costed.category_cost is not None
    components = {
        name: value
        for name, value in costed.category_cost.components.items()
        if not name.startswith("tilt_difference.worst.")
    }
    malformed = _upstream_costed(
        _channel_measured(tilt_worst=0.0),
        value=costed.category_cost.value,
        components=components,
        fingerprint=registry.fingerprint,
    )

    with pytest.raises(ValueError, match="tilt_difference.*worst"):
        channel_matching_cost.channel_matching_floor_reasons(
            malformed, registry.purpose(_PURPOSE)
        )


def test_measured_channel_matching_cannot_carry_an_unavailable_point() -> None:
    """票 #403：有一點不可估整類就該是不可估；夾著不可估列的已量結果重讀時契約直接拒收。

    這一題原本守「整批沒有可估點時沒有 worst 明細合法」——那種形狀評估器已經產不出來，
    改成守契約不收它；「直達時間開關關著不要求明細」由上面兩題的三種輸入形狀守著。
    """
    document = _channel_measured(tilt_worst=0.0).model_dump(mode="python")
    payload = document["payload"]
    payload["point_results"][0].update(
        state="unavailable",
        reason_codes=["missing_points"],
        reason="fixture has no assessed point",
        tilt_difference_db_per_octave=None,
        ripple_rms_difference_db=None,
        broadband_level_difference_db=None,
        direct_time_difference_ms=None,
    )
    payload["aggregates"][0].update(
        assessed_receiver_ids=[],
        unavailable_receiver_ids=["main"],
        tilt_difference=None,
        ripple_rms_difference=None,
        broadband_level_difference=None,
        direct_time_difference=None,
    )

    with pytest.raises(ValueError, match="整類應回不可估"):
        CategoryEvaluation.model_validate(document)
