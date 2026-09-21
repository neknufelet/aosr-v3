"""聆聽區穩定性疊層的完整性、身分與彙總語意考卷（票 #349）。"""
from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Literal

import pytest

from aosr.config.paths import config_path
from aosr.config.quality_targets import QualityTargets, load_quality_targets
from aosr.scoring import ranking
from aosr.scoring.contract import (
    CONTRACT_SCHEMA_VERSION,
    CandidateEvaluation,
    CategoryEvaluation,
    EvaluationState,
    Feature,
    Flag,
    InputProvenance,
    ListeningAreaStabilityPayload,
    ModelValidationStatus,
    QualityCategory,
    RawQuantity,
    ReasonCode,
    TimbrePayload,
)
from aosr.scoring.listening_area import ReceiverPointResult, evaluate_listening_area
from aosr.scoring.placement import point_placement
from aosr.scoring.ranking import CandidateStatus, RankingContext
from aosr.scoring.receiver_set import ReceiverPoint, ReceiverRole, ReceiverSet


_CANDIDATE = "candidate-a"
_SPEAKER = "left"
_SETTINGS = "timbre-settings-a"
_SCENE_FINGERPRINT = "a" * 64
_SPEAKER_POSITION = (0.2, 0.3, 1.1)
_RECEIVER_POSITIONS = {
    "main": (1.0, 2.0, 1.2),
    "front": (1.1, 2.0, 1.2),
    "back": (0.9, 2.0, 1.2),
}


def _receiver_set(*, front_importance: float = 1.0, back_importance: float = 1.0) -> ReceiverSet:
    return ReceiverSet(
        points=(
            ReceiverPoint(
                receiver_id="main",
                position_m=(1.0, 2.0, 1.2),
                role=ReceiverRole.PRIMARY,
                importance=1.0,
            ),
            ReceiverPoint(
                receiver_id="front",
                position_m=(1.1, 2.0, 1.2),
                role=ReceiverRole.SURROUNDING,
                importance=front_importance,
                direction_relative_to_primary="front",
            ),
            ReceiverPoint(
                receiver_id="back",
                position_m=(0.9, 2.0, 1.2),
                role=ReceiverRole.SURROUNDING,
                importance=back_importance,
                direction_relative_to_primary="back",
            ),
        )
    )


def _feature(kind: Literal["peak", "dip"], center_hz: float) -> Feature:
    return Feature(
        kind=kind,
        center_frequency_hz=center_hz,
        depth_db=-6.0 if kind == "dip" else 6.0,
        width_octave=0.25,
        flags=(),
    )


def _timbre(
    receiver_id: str,
    *,
    tilt: float,
    ripple: float,
    features: Sequence[Feature] = (),
    candidate_id: str = _CANDIDATE,
    speaker_id: str = _SPEAKER,
    settings_fingerprint: str = _SETTINGS,
    deviation_curve: tuple[tuple[float, float], ...] | None = None,
    target_deviation_rms_db: float = 0.0,
    evaluator_version: str = "timbre-fixture-v1",
    scene_fingerprint: str = _SCENE_FINGERPRINT,
) -> CategoryEvaluation:
    payload = TimbrePayload(
        category="timbre_balance",
        tilt_db_per_octave=tilt,
        tilt_fit_range_hz=(80.0, 4000.0),
        tilt_dependency_range_hz=(71.0, 4490.0),
        target_tilt_db_per_octave=0.0,
        target_deviation_rms_db=target_deviation_rms_db,
        deviation_curve=deviation_curve or ((100.0, tilt), (200.0, ripple)),
        residual_rms_db=ripple,
        ripple_range_hz=(40.0, 4000.0),
        ripple_dependency_range_hz=(37.0, 4240.0),
        features=tuple(features),
        strongest_peak_index=None,
        deepest_dip_index=None,
        data_range_hz=(20.0, 8000.0),
        coverage_range_hz=(20.0, 8000.0),
        model_validation_status=ModelValidationStatus.VALIDATED,
        model_validation_frequency_range_hz=(20.0, 8000.0),
    )
    return CategoryEvaluation(
        schema_version=CONTRACT_SCHEMA_VERSION,
        candidate_id=candidate_id,
        scene_fingerprint=scene_fingerprint,
        placement=point_placement(
            speaker_id,
            _SPEAKER_POSITION,
            receiver_id,
            _RECEIVER_POSITIONS.get(receiver_id, (9.0, 9.0, 9.0)),
        ),
        category=QualityCategory.TIMBRE_BALANCE,
        state=EvaluationState.MEASURED,
        payload=payload,
        raw_quantities=(RawQuantity(name="tilt", value=tilt, unit="dB/oct"),),
        category_cost=None,
        flags=(),
        reason_codes=(),
        evaluator_version=evaluator_version,
        settings_fingerprint=settings_fingerprint,
        provenance=InputProvenance(
            report_id=f"report-{receiver_id}",
            engine_commit="engine-fixture",
            speaker_id=speaker_id,
            receiver_id=receiver_id,
        ),
    )


def _results(
    receivers: ReceiverSet,
    *,
    tilts: tuple[float, float, float] = (0.0, 2.0, 8.0),
    ripples: tuple[float, float, float] = (1.0, 2.0, 3.0),
    levels: tuple[float, float, float] = (70.0, 71.0, 74.0),
    features: tuple[Sequence[Feature], Sequence[Feature], Sequence[Feature]] = ((), (), ()),
    candidate_id: str = _CANDIDATE,
    settings_fingerprint: str = _SETTINGS,
    evaluator_version: str = "timbre-fixture-v1",
) -> tuple[ReceiverPointResult, ...]:
    ids = ("main", "front", "back")
    return tuple(
        ReceiverPointResult(
            receiver_id=receiver_id,
            receiver_set_fingerprint=receivers.fingerprint,
            timbre_evaluation=_timbre(
                receiver_id,
                tilt=tilt,
                ripple=ripple,
                features=point_features,
                candidate_id=candidate_id,
                settings_fingerprint=settings_fingerprint,
                evaluator_version=evaluator_version,
            ),
            broadband_mean_total_energy_db=level,
        )
        for receiver_id, tilt, ripple, level, point_features in zip(
            ids, tilts, ripples, levels, features, strict=True
        )
    )


def _evaluate(
    receivers: ReceiverSet,
    results: Sequence[ReceiverPointResult],
    *,
    tolerance_hz: float = 10.0,
    candidate_id: str = _CANDIDATE,
    timbre_settings_fingerprint: str = _SETTINGS,
    scene_fingerprint: str = _SCENE_FINGERPRINT,
) -> CategoryEvaluation:
    return evaluate_listening_area(
        receivers,
        results,
        candidate_id=candidate_id,
        speaker_id=_SPEAKER,
        timbre_settings_fingerprint=timbre_settings_fingerprint,
        scene_fingerprint=scene_fingerprint,
        feature_match_tolerance_hz=tolerance_hz,
    )


def _payload(evaluation: CategoryEvaluation) -> ListeningAreaStabilityPayload:
    assert evaluation.state == EvaluationState.MEASURED
    assert isinstance(evaluation.payload, ListeningAreaStabilityPayload)
    return evaluation.payload


def _translated(receivers: ReceiverSet) -> ReceiverSet:
    translation = (0.1, 0.2, 0.3)
    return ReceiverSet(
        points=tuple(
            point.model_copy(
                update={
                    "position_m": tuple(
                        coordinate + offset
                        for coordinate, offset in zip(
                            point.position_m, translation, strict=True
                        )
                    )
                }
            )
            for point in receivers.points
        )
    )


def _changed_front(receivers: ReceiverSet, change: str) -> ReceiverSet:
    points = list(receivers.points)
    front_index = next(
        index for index, point in enumerate(points) if point.receiver_id == "front"
    )
    front = points[front_index]
    if change == "position":
        x, y, z = front.position_m
        front = front.model_copy(update={"position_m": (x + 0.01, y, z)})
    elif change == "role":
        front = front.model_copy(update={"role": ReceiverRole.OTHER_SEAT})
    else:
        front = front.model_copy(update={"importance": front.importance + 0.5})
    points[front_index] = front
    return ReceiverSet(points=tuple(points))


def _listening_only_registry() -> QualityTargets:
    registry = load_quality_targets(config_path("quality_targets.toml"))
    document = registry.model_dump(mode="json", by_alias=True)
    for row in document["purpose"][0]["qualification"]:
        if row["key"] == "ranking.mandatory_categories":
            row["value"] = ["listening_area_stability"]
        elif row["key"] == "ranking.optional_categories":
            row["value"] = [
                name for name in row["value"] if name != "listening_area_stability"
            ]
    return QualityTargets.model_validate(document)


def _ranking_candidate(evaluation: CategoryEvaluation) -> CandidateEvaluation:
    return CandidateEvaluation(
        schema_version=CONTRACT_SCHEMA_VERSION,
        candidate_id=evaluation.candidate_id,
        scene_fingerprint=evaluation.scene_fingerprint,
        evaluations=(evaluation,),
    )


def _ranking_context(receivers: ReceiverSet) -> RankingContext:
    return RankingContext(
        purpose="dedicated_two_channel_listening_room",
        receiver_set_fingerprint=receivers.fingerprint,
        channel_group_fingerprint="channel-group-fixture",
        run_date=date(2026, 9, 21),
        engine_version="engine-fixture",
    )


@pytest.mark.parametrize(
    ("identity", "reason"),
    (
        ("candidate", ReasonCode.CANDIDATE_ID_MISMATCH),
        ("speaker", ReasonCode.SPEAKER_ID_MISMATCH),
        ("receiver_set", ReasonCode.RECEIVER_SET_FINGERPRINT_MISMATCH),
        ("settings", ReasonCode.SETTINGS_FINGERPRINT_MISMATCH),
    ),
)
def test_any_identity_mismatch_is_unavailable_with_specific_reason(
    identity: str, reason: ReasonCode
) -> None:
    """四個代號任一不一致都要拒絕整批，且原因必須指明是哪一格不合。"""
    receivers = _receiver_set()
    results = list(_results(receivers))
    first = results[0]
    if identity == "candidate":
        changed = first.model_copy(
            update={
                "timbre_evaluation": first.timbre_evaluation.model_copy(
                    update={"candidate_id": "candidate-b"}
                )
            }
        )
    elif identity == "speaker":
        provenance = first.timbre_evaluation.provenance.model_copy(update={"speaker_id": "right"})
        changed = first.model_copy(
            update={
                "timbre_evaluation": first.timbre_evaluation.model_copy(
                    update={"provenance": provenance}
                )
            }
        )
    elif identity == "receiver_set":
        changed = first.model_copy(update={"receiver_set_fingerprint": "different-set"})
    else:
        changed = first.model_copy(
            update={
                "timbre_evaluation": first.timbre_evaluation.model_copy(
                    update={"settings_fingerprint": "different-settings"}
                )
            }
        )
    results[0] = changed

    evaluation = _evaluate(receivers, results)

    assert evaluation.state == EvaluationState.UNAVAILABLE
    assert evaluation.payload is None
    assert reason in evaluation.reason_codes


def test_missing_receiver_result_rejects_the_whole_set() -> None:
    """少一個必量的主位或周圍點時必須回不可彙總，不能拿剩下的點湊數字。"""
    receivers = _receiver_set()
    results = _results(receivers)

    evaluation = _evaluate(receivers, results[:-1])

    assert evaluation.state == EvaluationState.UNAVAILABLE
    assert evaluation.payload is None
    assert ReasonCode.MISSING_POINTS in evaluation.reason_codes


def test_mixed_scene_fingerprints_reject_the_whole_set() -> None:
    """批內一點來自別的場景時不可彙總，原因必須明確指向場景。"""
    receivers = _receiver_set()
    results = list(_results(receivers))
    changed = results[1].timbre_evaluation.model_copy(
        update={"scene_fingerprint": "b" * 64}
    )
    results[1] = results[1].model_copy(update={"timbre_evaluation": changed})

    evaluation = _evaluate(receivers, results)

    assert evaluation.state is EvaluationState.UNAVAILABLE
    assert evaluation.reason_codes == (ReasonCode.SCENE_FINGERPRINT_MISMATCH,)
    assert evaluation.scene_fingerprint == _SCENE_FINGERPRINT


def test_mixed_placement_is_unavailable_and_order_independent() -> None:
    """批內同喇叭代號若落在兩個座標，整類不可估；換輸入順序不得改任何輸出格。"""
    receivers = _receiver_set()
    results = list(_results(receivers))
    changed = results[1]
    changed_placement = changed.timbre_evaluation.placement.model_copy(
        update={"speaker_positions_m": ((_SPEAKER, (9.0, 8.0, 7.0)),)}
    )
    results[1] = changed.model_copy(
        update={
            "timbre_evaluation": changed.timbre_evaluation.model_copy(
                update={"placement": changed_placement}
            )
        }
    )

    forward = _evaluate(receivers, results)
    reversed_input = _evaluate(receivers, tuple(reversed(results)))
    candidate = CandidateEvaluation(
        schema_version=CONTRACT_SCHEMA_VERSION,
        candidate_id=_CANDIDATE,
        scene_fingerprint=_SCENE_FINGERPRINT,
        evaluations=(forward,),
    )

    assert forward == reversed_input
    assert forward.state is EvaluationState.UNAVAILABLE
    assert forward.reason_codes == (ReasonCode.PLACEMENT_MISMATCH,)
    assert forward.placement.speaker_positions_m == ()
    assert forward.placement.receiver_positions_m == ()
    assert candidate.evaluations == (forward,)


def test_two_different_mistakes_give_same_output_in_either_order() -> None:
    """兩點各犯一種錯：原因碼照列舉宣告的順序排，輸入反過來輸出逐格相同；擺位沒衝突就照樣帶著。"""
    receivers = _receiver_set()
    results = list(_results(receivers))
    first, second = results[0], results[1]
    results[0] = first.model_copy(
        update={
            "timbre_evaluation": first.timbre_evaluation.model_copy(
                update={"candidate_id": "wrong-candidate"}
            )
        }
    )
    results[1] = second.model_copy(
        update={
            "timbre_evaluation": second.timbre_evaluation.model_copy(
                update={"settings_fingerprint": "wrong-settings"}
            )
        }
    )

    forward = _evaluate(receivers, results)
    reversed_input = _evaluate(receivers, tuple(reversed(results)))

    assert forward == reversed_input
    assert forward.state is EvaluationState.UNAVAILABLE
    assert set(forward.reason_codes) == {
        ReasonCode.CANDIDATE_ID_MISMATCH,
        ReasonCode.SETTINGS_FINGERPRINT_MISMATCH,
    }
    assert dict(forward.placement.speaker_positions_m) == {_SPEAKER: _SPEAKER_POSITION}
    assert {name for name, _ in forward.placement.receiver_positions_m} == {
        item.receiver_id for item in results
    }


def test_flags_from_different_points_do_not_follow_input_order() -> None:
    """兩點各帶一種旗標：彙總出來的旗標順序不准跟著輸入順序變。"""
    receivers = _receiver_set()
    results = list(_results(receivers))
    for index, flag in ((0, Flag.UNVALIDATED), (1, Flag.DATA_COVERAGE_SHORT)):
        item = results[index]
        results[index] = item.model_copy(
            update={
                "timbre_evaluation": item.timbre_evaluation.model_copy(
                    update={"flags": (*item.timbre_evaluation.flags, flag)}
                )
            }
        )

    forward = _evaluate(receivers, results)
    reversed_input = _evaluate(receivers, tuple(reversed(results)))

    assert forward == reversed_input
    assert {Flag.UNVALIDATED, Flag.DATA_COVERAGE_SHORT} <= set(forward.flags)


def test_wrong_primary_scene_is_unavailable_and_still_fits_expected_scene_candidate() -> None:
    """主位恰好是錯場景時，整類仍要帶預期場景，否則連不可估結果都裝不進候選包。"""
    receivers = _receiver_set()
    results = list(_results(receivers))
    primary = results[0]
    results[0] = primary.model_copy(
        update={
            "timbre_evaluation": primary.timbre_evaluation.model_copy(
                update={"scene_fingerprint": "b" * 64}
            )
        }
    )

    evaluation = _evaluate(receivers, results)
    candidate = CandidateEvaluation(
        schema_version=CONTRACT_SCHEMA_VERSION,
        candidate_id=_CANDIDATE,
        scene_fingerprint=_SCENE_FINGERPRINT,
        evaluations=(evaluation,),
    )

    assert evaluation.state is EvaluationState.UNAVAILABLE
    assert evaluation.reason_codes == (ReasonCode.SCENE_FINGERPRINT_MISMATCH,)
    assert evaluation.scene_fingerprint == _SCENE_FINGERPRINT
    assert candidate.evaluations == (evaluation,)


def test_reordering_same_inputs_does_not_change_unavailable_output() -> None:
    """主位缺席且剩餘輸入混場景時，呼叫端順序不得決定不可估輸出的任何一格。"""
    receivers = _receiver_set()
    front, back = _results(receivers)[1:]
    back = back.model_copy(
        update={
            "timbre_evaluation": back.timbre_evaluation.model_copy(
                update={"scene_fingerprint": "b" * 64}
            )
        }
    )

    forward = _evaluate(receivers, (front, back))
    reversed_input = _evaluate(receivers, (back, front))

    assert forward == reversed_input


def test_missing_other_seat_result_does_not_block_aggregation() -> None:
    """其他座位這一版不量；清單保留它但不送結果，主位與周圍仍應能彙總。"""
    base = _receiver_set()
    receivers = ReceiverSet(
        points=(
            *base.points,
            ReceiverPoint(
                receiver_id="sofa",
                position_m=(2.5, 2.0, 1.2),
                role=ReceiverRole.OTHER_SEAT,
                importance=1.0,
            ),
        )
    )

    evaluation = _evaluate(receivers, _results(receivers))

    assert evaluation.state == EvaluationState.MEASURED
    assert isinstance(evaluation.payload, ListeningAreaStabilityPayload)
    assert {item.receiver_id for item in evaluation.payload.point_provenance} == {
        "main",
        "front",
        "back",
    }


def test_aggregate_retains_each_used_point_provenance_in_receiver_set_order() -> None:
    """彙總若只剩主位報表代號，就無法追查每一份實際疊入的單點音色結果。"""
    base = _receiver_set()
    receivers = ReceiverSet(points=(base.points[2], base.points[0], base.points[1]))

    payload = _payload(_evaluate(receivers, _results(receivers)))

    assert tuple(item.receiver_id for item in payload.point_provenance) == (
        "back",
        "main",
        "front",
    )
    assert tuple(item.report_id for item in payload.point_provenance) == (
        "report-back",
        "report-main",
        "report-front",
    )
    assert {item.evaluator_version for item in payload.point_provenance} == {
        "timbre-fixture-v1"
    }
    assert {item.settings_fingerprint for item in payload.point_provenance} == {_SETTINGS}


def test_listening_area_settings_fingerprint_tracks_feature_tolerance() -> None:
    """峰谷容許寬度若沒進本層指紋，同一指紋會對到兩個不同的峰谷答案。"""
    receivers = _receiver_set()
    features = (
        (_feature("dip", 100.0),),
        (_feature("dip", 106.0),),
        (_feature("dip", 112.0),),
    )
    results = _results(receivers, features=features)

    narrow = _evaluate(receivers, results, tolerance_hz=5.0)
    wide = _evaluate(receivers, results, tolerance_hz=10.0)

    assert narrow.settings_fingerprint == (
        "7b77aae2916dfcda7ee0cc741cce60f0c740ddbd6823678ba1becd3c05b4a5f1"
    )
    assert wide.settings_fingerprint == (
        "f939b8427a2e9c0b457bbcbff057cb1dc7d0790ad78fdc8e0d139a60499f5dca"
    )
    assert narrow.settings_fingerprint != wide.settings_fingerprint
    assert _payload(narrow).settings_fingerprint == narrow.settings_fingerprint
    assert _payload(wide).peak_dip_consistency != _payload(narrow).peak_dip_consistency


@pytest.mark.parametrize(
    ("changed_settings", "changed_version"),
    (("timbre-settings-b", "timbre-fixture-v1"), (_SETTINGS, "timbre-fixture-v2")),
)
def test_listening_area_identity_tracks_upstream_measurement_method(
    changed_settings: str, changed_version: str
) -> None:
    """整批各自一致仍不可把不同音色設定或版本發成同一個對外身分。"""
    receivers = _receiver_set()
    original = _evaluate(receivers, _results(receivers))
    changed_results = _results(
        receivers,
        settings_fingerprint=changed_settings,
        evaluator_version=changed_version,
    )
    changed = _evaluate(
        receivers,
        changed_results,
        timbre_settings_fingerprint=changed_settings,
    )

    assert original.state is EvaluationState.MEASURED
    assert changed.state is EvaluationState.MEASURED
    assert original.settings_fingerprint != changed.settings_fingerprint


def test_listening_area_identity_uses_translation_invariant_receiver_layout() -> None:
    """帶浮點尾數的整批平移不分表，但絕對座標清單仍維持不同身分。"""
    receivers = _receiver_set()
    translated = _translated(receivers)

    original = _evaluate(receivers, _results(receivers))
    moved = _evaluate(translated, _results(translated))

    assert receivers.fingerprint != translated.fingerprint
    assert receivers.layout_fingerprint == translated.layout_fingerprint
    assert original.settings_fingerprint == moved.settings_fingerprint


@pytest.mark.parametrize("change", ("position", "role", "importance"))
def test_listening_area_identity_changes_with_relative_layout_detail(change: str) -> None:
    """位移、角色或重要性任一改變，都必須切開彙總層的比較身分。"""
    receivers = _receiver_set()
    changed_receivers = _changed_front(receivers, change)

    original = _evaluate(receivers, _results(receivers))
    changed = _evaluate(changed_receivers, _results(changed_receivers))

    assert receivers.layout_fingerprint != changed_receivers.layout_fingerprint
    assert original.settings_fingerprint != changed.settings_fingerprint


def test_unvalidated_flag_from_any_point_reaches_the_aggregate() -> None:
    """任何一點的音色是用沒驗過的物理模型量的，疊起來的聆聽區評估也要帶著那個標記。"""
    receivers = _receiver_set()
    results = list(_results(receivers))
    flagged = results[-1].timbre_evaluation.model_copy(
        update={"flags": (Flag.UNVALIDATED,)}
    )
    results[-1] = results[-1].model_copy(update={"timbre_evaluation": flagged})

    evaluation = _evaluate(receivers, results)

    assert evaluation.state is EvaluationState.MEASURED
    assert Flag.UNVALIDATED in evaluation.flags


def test_mixed_upstream_evaluator_versions_are_not_aggregated() -> None:
    """同批只有一點的音色評估器版本不同時，原因必須明說是版本錯位。"""
    receivers = _receiver_set()
    results = list(_results(receivers))
    first = results[0]
    results[0] = first.model_copy(
        update={
            "timbre_evaluation": first.timbre_evaluation.model_copy(
                update={"evaluator_version": "timbre-fixture-v2"}
            )
        }
    )

    evaluation = _evaluate(receivers, results)

    assert evaluation.state is EvaluationState.UNAVAILABLE
    assert evaluation.reason_codes == (ReasonCode.EVALUATOR_VERSION_MISMATCH,)


def test_ranking_splits_legal_aggregates_with_different_upstream_settings() -> None:
    """排名只看彙總對外身分，也能把不同上游音色設定的合法候選分表。"""
    receivers = _receiver_set()
    zeroes = (0.0, 0.0, 0.0)
    equal_levels = (70.0, 70.0, 70.0)
    first = _evaluate(
        receivers,
        _results(
            receivers,
            candidate_id="candidate-a",
            tilts=zeroes,
            ripples=zeroes,
            levels=equal_levels,
        ),
        candidate_id="candidate-a",
    )
    second = _evaluate(
        receivers,
        _results(
            receivers,
            candidate_id="candidate-b",
            settings_fingerprint="timbre-settings-b",
            tilts=zeroes,
            ripples=zeroes,
            levels=equal_levels,
        ),
        candidate_id="candidate-b",
        timbre_settings_fingerprint="timbre-settings-b",
    )

    result = ranking.rank_candidates(
        (_ranking_candidate(first), _ranking_candidate(second)),
        _listening_only_registry(),
        _ranking_context(receivers),
    )

    assert first.settings_fingerprint != second.settings_fingerprint
    assert {result.status_of(candidate) for candidate in ("candidate-a", "candidate-b")} == {
        CandidateStatus.RANKABLE,
        CandidateStatus.NOT_COMPARABLE,
    }


def test_extra_point_with_wrong_set_fingerprint_reports_both_identity_failures() -> None:
    """多送一點不能遮住它同時帶錯的接收點清單指紋。"""
    receivers = _receiver_set()
    results = list(_results(receivers))
    results.append(
        ReceiverPointResult(
            receiver_id="rogue",
            receiver_set_fingerprint="wrong-set",
            timbre_evaluation=_timbre("rogue", tilt=0.0, ripple=0.0),
            broadband_mean_total_energy_db=70.0,
        )
    )

    evaluation = _evaluate(receivers, results)

    assert ReasonCode.RECEIVER_ID_MISMATCH in evaluation.reason_codes
    assert ReasonCode.RECEIVER_SET_FINGERPRINT_MISMATCH in evaluation.reason_codes


def test_missing_primary_uses_synthetic_listening_area_provenance() -> None:
    """主位缺席時若借第一個周圍點的報表，輸入順序會捏造不同的區域出身。"""
    receivers = _receiver_set()
    supplied = _results(receivers)[1:]

    evaluation = _evaluate(receivers, supplied)

    assert evaluation.state == EvaluationState.UNAVAILABLE
    assert evaluation.provenance.report_id == f"listening-area:{_CANDIDATE}"
    assert evaluation.provenance.report_id not in {
        item.timbre_evaluation.provenance.report_id for item in supplied
    }


def test_unmeasured_timbre_result_rejects_the_whole_set() -> None:
    """任一點不是 measured 時都必須拒絕整批，不能把不可估點當成零偏差。"""
    receivers = _receiver_set()
    results = list(_results(receivers))
    measured = results[0].timbre_evaluation
    unavailable = measured.model_copy(
        update={
            "state": EvaluationState.UNAVAILABLE,
            "payload": None,
            "raw_quantities": (),
            "reason_codes": (ReasonCode.INSUFFICIENT_COVERAGE,),
        }
    )
    results[0] = results[0].model_copy(update={"timbre_evaluation": unavailable})

    evaluation = _evaluate(receivers, results)

    assert evaluation.state == EvaluationState.UNAVAILABLE
    assert ReasonCode.TIMBRE_NOT_MEASURED in evaluation.reason_codes
    assert evaluation.scene_fingerprint == _SCENE_FINGERPRINT


def test_weighted_mean_moves_toward_the_more_important_point() -> None:
    """同一組偏差提高近端點的重要性後，加權平均必須往近端點的偏差移動。"""
    equal = _receiver_set()
    front_heavy = _receiver_set(front_importance=3.0)

    equal_value = _payload(_evaluate(equal, _results(equal))).tilt_stability
    weighted_value = _payload(
        _evaluate(front_heavy, _results(front_heavy))
    ).tilt_stability

    assert equal_value.primary_to_surrounding.weighted_mean_deviation == pytest.approx(5.0)
    assert weighted_value.primary_to_surrounding.weighted_mean_deviation == pytest.approx(3.5)
    assert (
        weighted_value.primary_to_surrounding.weighted_mean_deviation
        < equal_value.primary_to_surrounding.weighted_mean_deviation
    )


def test_worst_deviation_uses_raw_value_and_carries_point_context() -> None:
    """最差點必須按原始偏差挑選，不得因低重要性消失，並須帶回方向與重要性。"""
    receivers = _receiver_set(front_importance=10.0, back_importance=0.1)
    summary = _payload(_evaluate(receivers, _results(receivers))).tilt_stability
    worst = summary.primary_to_surrounding.worst_deviation

    assert worst.value == pytest.approx(8.0)
    assert worst.receiver.receiver_id == "back"
    assert worst.receiver.direction_relative_to_primary == "back"
    assert worst.receiver.importance == pytest.approx(0.1)


def test_primary_and_peer_comparisons_are_separate() -> None:
    """主位對周圍與周圍彼此必須各算各的，不能共用或覆蓋同一組數字。"""
    receivers = _receiver_set()
    summary = _payload(
        _evaluate(receivers, _results(receivers, tilts=(0.0, 2.0, 4.0)))
    ).tilt_stability

    assert summary.primary_to_surrounding.weighted_mean_deviation == pytest.approx(3.0)
    assert summary.surrounding_to_surrounding is not None
    assert summary.surrounding_to_surrounding.weighted_mean_deviation == pytest.approx(2.0)


def test_one_surrounding_keeps_primary_comparison_and_marks_peer_comparison_empty() -> None:
    """只有一個周圍點時沒有周圍配對，但算得到的主位比較不能跟著整份作廢。"""
    receivers = ReceiverSet(points=_receiver_set().points[:2])
    results = _results(receivers)[:2]

    evaluation = _evaluate(receivers, results)
    summary = _payload(evaluation).tilt_stability

    assert evaluation.state == EvaluationState.MEASURED
    assert summary.primary_to_surrounding.weighted_mean_deviation == pytest.approx(2.0)
    assert summary.surrounding_to_surrounding is None
    assert summary.surrounding_to_surrounding_reason == ReasonCode.NO_SURROUNDING_PAIRS


def test_peak_dip_occurrence_counts_only_same_kind_within_tolerance() -> None:
    """同種峰谷在容許寬度內才算同一個，超出寬度的坑不得灌進出現點數。"""
    receivers = _receiver_set()
    features = (
        (_feature("dip", 100.0),),
        (_feature("dip", 105.0),),
        (_feature("dip", 112.0), _feature("peak", 100.0)),
    )
    payload = _payload(_evaluate(receivers, _results(receivers, features=features)))
    occurrence = payload.peak_dip_occurrences[0]

    assert occurrence.kind == "dip"
    assert occurrence.primary_center_frequency_hz == pytest.approx(100.0)
    assert occurrence.receiver_ids == ("main", "front")
    assert occurrence.occurrence_count == 2


def test_peak_dip_occurrences_are_anchored_to_primary_features() -> None:
    """第一版只量相對主位變化；周圍彼此共有但主位沒有的坑不得冒充主位事件。"""
    receivers = _receiver_set()
    features = ((), (_feature("dip", 100.0),), (_feature("dip", 102.0),))

    payload = _payload(_evaluate(receivers, _results(receivers, features=features)))

    assert payload.peak_dip_occurrences == ()


def test_target_deviation_position_spread_is_diagnostic_and_no_cost_is_created() -> None:
    """對目標偏差的位置間散布只能留在診斷欄位，整段仍是 measured 且沒有類代價。"""
    receivers = _receiver_set()
    evaluation = _evaluate(receivers, _results(receivers))
    payload = _payload(evaluation)

    assert payload.target_deviation_position_spread_curve_db
    assert evaluation.state == EvaluationState.MEASURED
    assert evaluation.category_cost is None


def test_partial_frequency_overlap_reports_kept_and_discarded_values_with_flag() -> None:
    """頻率格只取交集時，丟掉的資料量與部分重疊標記都必須讓下游看得見。"""
    receivers = ReceiverSet(points=_receiver_set().points[:2])
    results = (
        ReceiverPointResult(
            receiver_id="main",
            receiver_set_fingerprint=receivers.fingerprint,
            timbre_evaluation=_timbre(
                "main", tilt=0.0, ripple=1.0, deviation_curve=((100.0, 0.0), (200.0, 1.0))
            ),
            broadband_mean_total_energy_db=70.0,
        ),
        ReceiverPointResult(
            receiver_id="front",
            receiver_set_fingerprint=receivers.fingerprint,
            timbre_evaluation=_timbre(
                "front", tilt=1.0, ripple=2.0, deviation_curve=((100.0, 1.0), (300.0, 2.0))
            ),
            broadband_mean_total_energy_db=71.0,
        ),
    )

    evaluation = _evaluate(receivers, results)
    payload = _payload(evaluation)

    assert payload.target_deviation_common_frequency_count == 1
    assert payload.target_deviation_discarded_frequency_value_count == 2
    assert Flag.PARTIAL_FREQUENCY_OVERLAP in evaluation.flags


def test_target_deviation_rms_is_not_counted_again_as_position_spread() -> None:
    """對目標偏差 RMS 與傾斜加起伏同源；位置層不得再把它列成第五種散布量。"""
    receivers = _receiver_set()
    results = list(_results(receivers))
    timbre = results[1].timbre_evaluation
    assert isinstance(timbre.payload, TimbrePayload)
    results[1] = results[1].model_copy(
        update={
            "timbre_evaluation": timbre.model_copy(
                update={
                    "payload": timbre.payload.model_copy(
                        update={"target_deviation_rms_db": 99.0}
                    )
                }
            )
        }
    )

    evaluation = _evaluate(receivers, results)

    assert {quantity.name for quantity in evaluation.raw_quantities} == {
        "tilt.primary_to_surrounding.weighted_mean_deviation",
        "tilt.primary_to_surrounding.worst_deviation",
        "tilt.surrounding_to_surrounding.weighted_mean_deviation",
        "tilt.surrounding_to_surrounding.worst_deviation",
        "ripple_rms.primary_to_surrounding.weighted_mean_deviation",
        "ripple_rms.primary_to_surrounding.worst_deviation",
        "ripple_rms.surrounding_to_surrounding.weighted_mean_deviation",
        "ripple_rms.surrounding_to_surrounding.worst_deviation",
        "overall_level.primary_to_surrounding.weighted_mean_deviation",
        "overall_level.primary_to_surrounding.worst_deviation",
        "overall_level.surrounding_to_surrounding.weighted_mean_deviation",
        "overall_level.surrounding_to_surrounding.worst_deviation",
        "peak_dip_consistency.primary_to_surrounding.weighted_mean_deviation",
        "peak_dip_consistency.primary_to_surrounding.worst_deviation",
        "peak_dip_consistency.surrounding_to_surrounding.weighted_mean_deviation",
        "peak_dip_consistency.surrounding_to_surrounding.worst_deviation",
    }
