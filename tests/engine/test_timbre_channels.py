"""主位逐聲道音色彙總、代價、複核警戒（review alert）與排名接線考卷。"""
from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import date
from typing import Final, Literal

import pytest
from pydantic import ValidationError

from aosr.config.paths import config_path
from aosr.config.quality_targets import load_quality_targets
from aosr.scoring.channel_matching import (
    ChannelComparison,
    ChannelDefinition,
    ChannelGroup,
    ChannelPointInput,
    ChannelResponse,
    evaluate_channel_matching,
)
from aosr.scoring.contract import (
    CONTRACT_SCHEMA_VERSION,
    CandidateEvaluation,
    CategoryEvaluation,
    EvaluationState,
    Feature,
    InputProvenance,
    ModelValidationStatus,
    QualityCategory,
    RawQuantity,
    ReasonCode,
    TimbreChannelsPayload,
    TimbrePayload,
)
from aosr.scoring.listening_area import ReceiverPointResult, evaluate_listening_area
from aosr.scoring.placement import point_placement
from aosr.scoring.ranking import (
    CandidateStatus,
    RankingContext,
    rank_candidates,
)
from aosr.scoring.receiver_set import ReceiverPoint, ReceiverRole, ReceiverSet
from aosr.scoring.timbre_cost import cost_timbre_evaluation


_CANDIDATE: Final[str] = "candidate-timbre-channels"
_SCENE: Final[str] = "a" * 64
_SETTINGS: Final[str] = "timbre-settings-fixture"
_PURPOSE: Final[str] = "dedicated_two_channel_listening_room"
_TARGETS = config_path("quality_targets.toml")


def _group(*, mono: bool = False) -> ChannelGroup:
    channels: tuple[ChannelDefinition, ...] = (
        ChannelDefinition(role="left", speaker_id="speaker-left"),
    )
    comparisons: tuple[ChannelComparison, ...] = ()
    if not mono:
        channels += (ChannelDefinition(role="right", speaker_id="speaker-right"),)
        comparisons = (ChannelComparison(left_role="left", right_role="right"),)
    return ChannelGroup(
        channels=channels,
        comparisons=comparisons,
        feature_match_tolerance_hz=10.0,
    )


def _payload(
    *,
    tilt: float = 0.0,
    residual: float = 0.0,
    features: tuple[Feature, ...] = (),
) -> TimbrePayload:
    peak = next(
        (index for index, feature in enumerate(features) if feature.kind == "peak"),
        None,
    )
    dip = next(
        (index for index, feature in enumerate(features) if feature.kind == "dip"),
        None,
    )
    return TimbrePayload(
        category="timbre_balance",
        tilt_db_per_octave=tilt,
        tilt_fit_range_hz=(80.0, 4000.0),
        tilt_dependency_range_hz=(71.0, 4490.0),
        target_tilt_db_per_octave=0.0,
        target_deviation_rms_db=0.0,
        deviation_curve=((100.0, 0.0), (200.0, 0.0)),
        residual_rms_db=residual,
        ripple_range_hz=(40.0, 4000.0),
        ripple_dependency_range_hz=(37.0, 4240.0),
        features=features,
        strongest_peak_index=peak,
        deepest_dip_index=dip,
        data_range_hz=(20.0, 8000.0),
        coverage_range_hz=(20.0, 8000.0),
        model_validation_status=ModelValidationStatus.VALIDATED,
        model_validation_frequency_range_hz=(20.0, 8000.0),
    )


def _single(
    role: str,
    *,
    receiver_id: str = "main",
    payload: TimbrePayload | None = None,
    reason: ReasonCode | None = None,
    candidate_id: str = _CANDIDATE,
    scene_fingerprint: str = _SCENE,
    settings_fingerprint: str = _SETTINGS,
    evaluator_version: str = "timbre-fixture-v1",
    receiver_position_m: tuple[float, float, float] = (4.0, 2.0, 1.2),
) -> CategoryEvaluation:
    speaker_id = f"speaker-{role}"
    speaker_position = (1.0, 1.0 if role == "left" else 3.0, 1.1)
    chosen = _payload() if payload is None and reason is None else payload
    state = EvaluationState.UNAVAILABLE if reason is not None else EvaluationState.MEASURED
    return CategoryEvaluation(
        schema_version=CONTRACT_SCHEMA_VERSION,
        candidate_id=candidate_id,
        scene_fingerprint=scene_fingerprint,
        placement=point_placement(
            speaker_id, speaker_position, receiver_id, receiver_position_m
        ),
        category=QualityCategory.TIMBRE_BALANCE,
        state=state,
        payload=chosen,
        raw_quantities=(
            ()
            if chosen is None
            else (
                RawQuantity(
                    name="tilt_db_per_octave",
                    value=chosen.tilt_db_per_octave,
                    unit="dB/oct",
                ),
                RawQuantity(
                    name="residual_rms_db",
                    value=chosen.residual_rms_db,
                    unit="dB",
                ),
            )
        ),
        category_cost=None,
        flags=(),
        reason_codes=() if reason is None else (reason,),
        evaluator_version=evaluator_version,
        settings_fingerprint=settings_fingerprint,
        provenance=InputProvenance(
            report_id=f"report-{role}-{receiver_id}",
            engine_commit="fixture-engine",
            speaker_id=speaker_id,
            receiver_id=receiver_id,
        ),
    )


def _aggregate(
    group: ChannelGroup,
    evaluations: Mapping[str, CategoryEvaluation],
    *,
    primary_receiver_id: str = "main",
) -> CategoryEvaluation:
    from aosr.scoring.timbre_channels import evaluate_timbre_channels

    return evaluate_timbre_channels(
        group,
        primary_receiver_id,
        evaluations,
        candidate_id=_CANDIDATE,
        scene_fingerprint=_SCENE,
        timbre_settings_fingerprint=_SETTINGS,
    )


def _candidate(evaluation: CategoryEvaluation) -> CandidateEvaluation:
    return CandidateEvaluation(
        schema_version=CONTRACT_SCHEMA_VERSION,
        candidate_id=_CANDIDATE,
        scene_fingerprint=_SCENE,
        evaluations=(evaluation,),
    )


def _context(group: ChannelGroup, receiver_id: str = "main") -> RankingContext:
    return RankingContext(
        purpose=_PURPOSE,
        receiver_set_fingerprint=f"receiver:{receiver_id}",
        channel_group_fingerprint=group.fingerprint,
        run_date=date(2026, 9, 21),
        engine_version="timbre-channels-fixture",
    )


def test_left_good_right_bad_cost_is_mean_and_right_channel_gets_alert() -> None:
    """把平均誤寫成只取左聲道，或警戒只看第一支，都會漏掉右聲道壞峰。"""
    group = _group()
    registry = load_quality_targets(_TARGETS)
    purpose = registry.purpose(_PURPOSE)
    left = _single("left", payload=_payload(tilt=0.2, residual=0.5))
    right = _single(
        "right",
        payload=_payload(
            tilt=1.2,
            residual=2.0,
            features=(
                Feature(
                    kind="peak",
                    center_frequency_hz=1000.0,
                    depth_db=99.0,
                    width_octave=0.5,
                    flags=(),
                ),
            ),
        ),
    )
    left_costed = cost_timbre_evaluation(left, purpose, registry.fingerprint)
    right_costed = cost_timbre_evaluation(right, purpose, registry.fingerprint)
    aggregate = _aggregate(group, {"left": left, "right": right})

    result = rank_candidates(
        (_candidate(aggregate),), registry, _context(group)
    )

    assert left_costed.category_cost is not None
    assert right_costed.category_cost is not None
    row = next(item for item in result.rankable if item.candidate_id == _CANDIDATE)
    settled = row.categories[0].evaluation
    assert settled.category_cost is not None
    assert settled.category_cost.value == pytest.approx(
        (left_costed.category_cost.value + right_costed.category_cost.value) / 2.0
    )
    assert result.status_of(_CANDIDATE) is CandidateStatus.RANKABLE
    alert = next(item for item in row.review_alerts if item.kind == "peak")
    assert alert.speaker_id == "speaker-right"


@pytest.mark.parametrize(
    "kind",
    ["peak", "dip"],
)
def test_review_alert_names_the_second_channel_when_only_it_is_bad(
    kind: Literal["peak", "dip"],
) -> None:
    """壞的是宣告順序排第二的那一支：警戒只看第一支的寫法，峰與谷各自都會漏。"""
    group = _group()
    registry = load_quality_targets(_TARGETS)
    bad = Feature(
        kind=kind,
        center_frequency_hz=1000.0,
        depth_db=99.0 if kind == "peak" else -99.0,
        width_octave=0.5,
        flags=(),
    )
    aggregate = _aggregate(
        group,
        {
            "left": _single("left", payload=_payload(tilt=0.2, residual=0.5)),
            "right": _single(
                "right", payload=_payload(tilt=0.2, residual=0.5, features=(bad,))
            ),
        },
    )

    result = rank_candidates((_candidate(aggregate),), registry, _context(group))

    assert result.status_of(_CANDIDATE) is CandidateStatus.RANKABLE
    row = next(item for item in result.rankable if item.candidate_id == _CANDIDATE)
    alert = next(item for item in row.review_alerts if item.kind == kind)
    assert alert.speaker_id == "speaker-right"


@pytest.mark.parametrize(
    "evaluations",
    (
        {"left": _single("left")},
        {
            "left": _single("left"),
            "right": _single("right"),
            "center": _single("center"),
        },
    ),
    ids=("missing", "extra"),
)
def test_declared_roles_must_match_inputs(
    evaluations: Mapping[str, CategoryEvaluation],
) -> None:
    """漏一個宣告角色或多塞未宣告角色都不能拿現有聲道湊成可估。"""
    result = _aggregate(_group(), evaluations)

    assert result.state is EvaluationState.UNAVAILABLE
    assert ReasonCode.CHANNEL_ROLE_MISMATCH in result.reason_codes


def test_unavailable_required_channel_propagates_reason_and_is_order_invariant() -> None:
    """右聲道不可估時整類失敗，且輸入映射的插入順序不改任何輸出格。"""
    group = _group()
    left = _single("left")
    right = _single("right", reason=ReasonCode.NON_POSITIVE_ENERGY)

    left_first = _aggregate(group, {"left": left, "right": right})
    right_first = _aggregate(group, {"right": right, "left": left})

    assert left_first == right_first
    assert left_first.state is EvaluationState.UNAVAILABLE
    assert left_first.reason_codes == (
        ReasonCode.NON_POSITIVE_ENERGY,
        ReasonCode.REQUIRED_CHANNEL_POINT_UNAVAILABLE,
    )


def test_mono_cost_equals_its_single_channel_cost() -> None:
    """單聲道的算術平均分母若不是實際聲道數，這題會改變既有單支代價。"""
    group = _group(mono=True)
    registry = load_quality_targets(_TARGETS)
    left = _single("left", payload=_payload(tilt=0.8, residual=1.5))
    single = cost_timbre_evaluation(
        left, registry.purpose(_PURPOSE), registry.fingerprint
    )
    aggregate = _aggregate(group, {"left": left})

    result = rank_candidates(
        (_candidate(aggregate),), registry, _context(group)
    )

    assert single.category_cost is not None
    settled = result.rankable[0].categories[0].evaluation
    assert settled.category_cost is not None
    assert settled.category_cost.value == pytest.approx(single.category_cost.value)


def test_every_channel_must_name_the_same_primary_receiver() -> None:
    """兩支原始結果不在同一主位時，不可把兩個位置平均成一個答案。"""
    result = _aggregate(
        _group(),
        {
            "left": _single("left", receiver_id="main"),
            "right": _single("right", receiver_id="other-seat"),
        },
    )

    assert result.state is EvaluationState.UNAVAILABLE
    assert ReasonCode.RECEIVER_ID_MISMATCH in result.reason_codes


def test_measured_output_preserves_each_original_payload_exactly() -> None:
    """彙總不得只留摘要；每支完整 TimbrePayload 要逐格等於輸入。"""
    left = _single("left", payload=_payload(tilt=0.2, residual=0.4))
    right = _single("right", payload=_payload(tilt=0.8, residual=1.4))
    result = _aggregate(_group(), {"right": right, "left": left})

    assert isinstance(result.payload, TimbreChannelsPayload)
    by_role = {item.role: item.payload for item in result.payload.channels}
    assert by_role == {"left": left.payload, "right": right.payload}


def test_candidate_rejects_single_estimable_timbre_but_keeps_unavailable() -> None:
    """候選可估音色只能走聲道彙總；無 payload 的不可估契約上無法辨認出身。"""
    with pytest.raises(ValidationError, match="聲道彙總"):
        _candidate(_single("left"))

    unavailable = _single("left", reason=ReasonCode.SOLVER_UNAVAILABLE)
    assert _candidate(unavailable).evaluations == (unavailable,)


@pytest.mark.parametrize(
    ("change", "expected"),
    (
        ("candidate", ReasonCode.CANDIDATE_ID_MISMATCH),
        ("scene", ReasonCode.SCENE_FINGERPRINT_MISMATCH),
        ("settings", ReasonCode.TIMBRE_SETTINGS_FINGERPRINT_MISMATCH),
        ("version", ReasonCode.EVALUATOR_VERSION_MISMATCH),
        ("speaker", ReasonCode.CHANNEL_ROLE_MISMATCH),
        ("placement", ReasonCode.PLACEMENT_MISMATCH),
    ),
)
def test_aggregator_rejects_each_comparison_identity_mismatch(
    change: str, expected: ReasonCode
) -> None:
    """候選、場景、量法、版本、角色喇叭與批內擺位任一不一致都不可估。"""
    left = _single("left")
    right = _single("right")
    if change == "candidate":
        right = right.model_copy(update={"candidate_id": "other-candidate"})
    elif change == "scene":
        right = right.model_copy(update={"scene_fingerprint": "b" * 64})
    elif change == "settings":
        right = right.model_copy(update={"settings_fingerprint": "other-settings"})
    elif change == "version":
        right = right.model_copy(update={"evaluator_version": "timbre-fixture-v2"})
    elif change == "speaker":
        right = right.model_copy(
            update={
                "provenance": right.provenance.model_copy(
                    update={"speaker_id": "wrong-speaker"}
                )
            }
        )
    else:
        right = right.model_copy(
            update={
                "placement": point_placement(
                    "speaker-right",
                    (1.0, 3.0, 1.1),
                    "main",
                    (9.0, 8.0, 7.0),
                )
            }
        )

    result = _aggregate(_group(), {"left": left, "right": right})

    assert result.state is EvaluationState.UNAVAILABLE
    assert expected in result.reason_codes


def test_ranking_lines_and_identity_expose_roles_group_and_primary_receiver() -> None:
    """診斷名漏角色或比較身分漏聲道組／主位時，候選會被錯誤混在同一張表。"""
    group = _group()
    aggregate = _aggregate(
        group,
        {
            "left": _single("left", payload=_payload(tilt=0.2, residual=0.4)),
            "right": _single("right", payload=_payload(tilt=0.8, residual=1.4)),
        },
    )

    result = rank_candidates(
        (_candidate(aggregate),), load_quality_targets(_TARGETS), _context(group)
    )

    category = result.rankable[0].categories[0]
    assert {line.name.split(".", 1)[0] for line in category.components} == {
        "left",
        "right",
    }
    support = json.loads(category.identity.assessed_support)
    assert support == {
        "channel_group_fingerprint": group.fingerprint,
        "channels": [
            {"role": "left", "speaker_id": "speaker-left"},
            {"role": "right", "speaker_id": "speaker-right"},
        ],
        "primary_receiver_id": "main",
        "timbre_evaluator_version": "timbre-fixture-v1",
    }


def _receivers() -> ReceiverSet:
    return ReceiverSet(
        points=(
            ReceiverPoint(
                receiver_id="main",
                position_m=(4.0, 2.0, 1.2),
                role=ReceiverRole.PRIMARY,
                importance=1.0,
            ),
            ReceiverPoint(
                receiver_id="front",
                position_m=(3.8, 2.0, 1.2),
                role=ReceiverRole.SURROUNDING,
                importance=1.0,
                direction_relative_to_primary="front",
            ),
        )
    )


def test_listening_area_does_not_treat_channel_aggregate_as_single_timbre() -> None:
    """聆聽區入口若只看外層類名，會把聲道彙總錯當成單支音色。"""
    receivers = _receivers()
    group = _group()
    aggregate = _aggregate(
        group, {"left": _single("left"), "right": _single("right")}
    )
    results = tuple(
        ReceiverPointResult(
            receiver_id=point.receiver_id,
            receiver_set_fingerprint=receivers.fingerprint,
            timbre_evaluation=aggregate,
            frequencies_hz=(100.0, 200.0),
            total_energy=(1e7, 1e7),
        )
        for point in receivers.points
    )

    result = evaluate_listening_area(
        receivers,
        results,
        candidate_id=_CANDIDATE,
        speaker_id=aggregate.provenance.speaker_id,
        timbre_settings_fingerprint=aggregate.settings_fingerprint,
        scene_fingerprint=_SCENE,
        feature_match_tolerance_hz=10.0,
        broadband_range_hz=(20.0, 8000.0),
    )

    assert result.state is EvaluationState.UNAVAILABLE
    assert ReasonCode.TIMBRE_NOT_MEASURED in result.reason_codes


def _as_channel_source(
    aggregate: CategoryEvaluation, role: str, receiver_id: str
) -> CategoryEvaluation:
    return aggregate.model_copy(
        update={
            "provenance": InputProvenance(
                report_id=f"aggregate-{role}-{receiver_id}",
                engine_commit="fixture-engine",
                speaker_id=f"speaker-{role}",
                receiver_id=receiver_id,
            )
        }
    )


def test_channel_matching_does_not_treat_channel_aggregate_as_single_timbre() -> None:
    """聲道匹配入口收到彙總 payload 時，必須走既有的必要點不可估路徑。"""
    receivers = _receivers()
    group = _group()
    points: list[ChannelPointInput] = []
    for point in receivers.points:
        singles = {
            role: _single(
                role,
                receiver_id=point.receiver_id,
                receiver_position_m=point.position_m,
            )
            for role in ("left", "right")
        }
        aggregate = _aggregate(
            group, singles, primary_receiver_id=point.receiver_id
        )
        points.append(
            ChannelPointInput(
                receiver_id=point.receiver_id,
                receiver_set_fingerprint=receivers.fingerprint,
                timbre_settings_fingerprint=_SETTINGS,
                listening_area_settings_fingerprint="listening-settings",
                channel_group_fingerprint=group.fingerprint,
                responses=tuple(
                    ChannelResponse(
                        role=role,
                        timbre_evaluation=_as_channel_source(
                            aggregate, role, point.receiver_id
                        ),
                        frequencies_hz=(100.0, 200.0),
                        total_energy=(1.0, 1.0),
                        direct_distance_m=2.0,
                    )
                    for role in ("left", "right")
                ),
            )
        )

    result = evaluate_channel_matching(
        receivers,
        tuple(points),
        candidate_id=_CANDIDATE,
        scene_fingerprint=_SCENE,
        timbre_settings_fingerprint=_SETTINGS,
        listening_area_settings_fingerprint="listening-settings",
        channel_group=group,
        purpose=_PURPOSE,
        quality_targets_path=_TARGETS,
        reflections=None,
        sound_speed_m_s=343.0,
    )

    assert result.state is EvaluationState.UNAVAILABLE
    assert ReasonCode.REQUIRED_CHANNEL_POINT_UNAVAILABLE in result.reason_codes
    assert ReasonCode.CHANNEL_RESULT_UNAVAILABLE in result.reason_codes


def test_missing_declared_role_names_the_required_channel_as_unavailable() -> None:
    """聲道組宣告左右、只交左邊：整類不可估，原因要指名「必要聲道不可估」，不只是角色對不上。"""
    aggregate = _aggregate(_group(), {"left": _single("left")})

    assert aggregate.state is EvaluationState.UNAVAILABLE
    assert ReasonCode.REQUIRED_CHANNEL_POINT_UNAVAILABLE in aggregate.reason_codes
    assert ReasonCode.CHANNEL_ROLE_MISMATCH in aggregate.reason_codes


def test_two_different_mistakes_give_same_output_in_either_mapping_order() -> None:
    """左右各犯一種錯、逐支掃出來的原因碼先後會跟著映射順序走：輸出仍要逐格相同。"""
    group = _group()
    left = _single("left", reason=ReasonCode.NON_POSITIVE_ENERGY)
    right = _single("right").model_copy(
        update={
            "provenance": _single("right").provenance.model_copy(
                update={"speaker_id": "someone-else"}
            )
        }
    )

    left_first = _aggregate(group, {"left": left, "right": right})
    right_first = _aggregate(group, {"right": right, "left": left})

    assert left_first == right_first
    assert left_first.state is EvaluationState.UNAVAILABLE
    assert {
        ReasonCode.NON_POSITIVE_ENERGY,
        ReasonCode.CHANNEL_ROLE_MISMATCH,
    } <= set(left_first.reason_codes)
