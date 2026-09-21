"""逐聲道音色補修的契約、比較身分與排名考卷（票 #425）。"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from typing import Final

import pytest
from pydantic import ValidationError

from aosr.config.paths import config_path
from aosr.config.quality_targets import WeightTable, load_quality_targets
from aosr.scoring.channel_matching import (
    ChannelComparison,
    ChannelDefinition,
    ChannelGroup,
)
from aosr.scoring.contract import (
    CONTRACT_SCHEMA_VERSION,
    CandidateEvaluation,
    CategoryCost,
    CategoryEvaluation,
    EvaluationState,
    Feature,
    Flag,
    InputProvenance,
    ModelValidationStatus,
    QualityCategory,
    RawQuantity,
    ReasonCode,
    TimbreChannel,
    TimbreChannelsPayload,
    TimbrePayload,
)
from aosr.scoring.placement import point_placement
from aosr.scoring.ranking import (
    CandidateStatus,
    NotEvaluatedReason,
    RankingContext,
    _timbre_channel_lines,
    rank_candidates,
)
from aosr.scoring.timbre_channels import evaluate_timbre_channels
from aosr.scoring.timbre_cost import comparison_support, cost_timbre_evaluation


_SCENE: Final[str] = "a" * 64
_SETTINGS: Final[str] = "timbre-settings-contract-fixture"
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
    target_deviation: float = 0.0,
    target_tilt: float = 0.0,
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
        target_tilt_db_per_octave=target_tilt,
        target_deviation_rms_db=target_deviation,
        deviation_curve=((100.0, -target_deviation), (200.0, target_deviation)),
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
    candidate_id: str = "candidate-a",
    payload: TimbrePayload | None = None,
    flags: tuple[Flag, ...] = (),
    report_id: str | None = None,
    evaluator_version: str = "timbre-upstream-v1",
    reason: ReasonCode | None = None,
) -> CategoryEvaluation:
    chosen = _payload() if payload is None and reason is None else payload
    state = EvaluationState.UNAVAILABLE if reason is not None else EvaluationState.MEASURED
    speaker_id = f"speaker-{role}"
    speaker_position = (1.0, 1.0 if role == "left" else 3.0, 1.1)
    quantities = () if chosen is None else (
        RawQuantity(name="tilt", value=chosen.tilt_db_per_octave, unit="dB/oct"),
        RawQuantity(name="residual", value=chosen.residual_rms_db, unit="dB"),
    )
    return CategoryEvaluation(
        schema_version=CONTRACT_SCHEMA_VERSION,
        candidate_id=candidate_id,
        scene_fingerprint=_SCENE,
        placement=point_placement(
            speaker_id, speaker_position, "main", (4.0, 2.0, 1.2)
        ),
        category=QualityCategory.TIMBRE_BALANCE,
        state=state,
        payload=chosen,
        raw_quantities=quantities,
        category_cost=None,
        flags=flags,
        reason_codes=() if reason is None else (reason,),
        evaluator_version=evaluator_version,
        settings_fingerprint=_SETTINGS,
        provenance=InputProvenance(
            report_id=report_id or f"report-{candidate_id}-{role}",
            engine_commit="fixture-engine",
            speaker_id=speaker_id,
            receiver_id="main",
        ),
    )


def _aggregate(
    group: ChannelGroup,
    evaluations: Mapping[str, CategoryEvaluation],
    *,
    candidate_id: str = "candidate-a",
) -> CategoryEvaluation:
    return evaluate_timbre_channels(
        group,
        "main",
        evaluations,
        candidate_id=candidate_id,
        scene_fingerprint=_SCENE,
        timbre_settings_fingerprint=_SETTINGS,
    )


def _candidate(evaluation: CategoryEvaluation) -> CandidateEvaluation:
    return CandidateEvaluation(
        schema_version=CONTRACT_SCHEMA_VERSION,
        candidate_id=evaluation.candidate_id,
        scene_fingerprint=evaluation.scene_fingerprint,
        evaluations=(evaluation,),
    )


def _context(group: ChannelGroup) -> RankingContext:
    return RankingContext(
        purpose=_PURPOSE,
        receiver_set_fingerprint="receiver-set-fixture",
        channel_group_fingerprint=group.fingerprint,
        run_date=date(2026, 9, 21),
        engine_version="timbre-contract-fixture",
    )


def test_context_rejects_other_channel_group_but_accepts_matching_group() -> None:
    """只把 context 指紋抄進表頭而不核對 payload，會讓單聲道候選混入雙聲道榜。"""
    stereo = _group()
    mono = _group(mono=True)
    registry = load_quality_targets(_TARGETS)
    mono_evaluation = _aggregate(mono, {"left": _single("left")})
    stereo_evaluation = _aggregate(
        stereo, {"left": _single("left"), "right": _single("right")}
    )

    rejected = rank_candidates((_candidate(mono_evaluation),), registry, _context(stereo))
    accepted = rank_candidates(
        (_candidate(stereo_evaluation),), registry, _context(stereo)
    )

    assert rejected.status_of("candidate-a") is CandidateStatus.NOT_EVALUATED
    assert {
        item.reason for item in rejected.not_evaluated[0].missing
    } >= {NotEvaluatedReason.CHANNEL_GROUP_FINGERPRINT_MISMATCH}
    assert accepted.status_of("candidate-a") is CandidateStatus.RANKABLE


def test_claimed_fingerprint_cannot_hide_a_different_channel_list() -> None:
    """只信 payload 自報的聲道組指紋，會讓偽造單聲道與真正雙聲道候選進同表。"""
    group = _group()
    genuine = _aggregate(
        group,
        {
            "left": _single("left", candidate_id="genuine"),
            "right": _single("right", candidate_id="genuine"),
        },
        candidate_id="genuine",
    )
    mono = _aggregate(
        _group(mono=True),
        {"left": _single("left", candidate_id="forged")},
        candidate_id="forged",
    )
    assert isinstance(mono.payload, TimbreChannelsPayload)
    forged_payload = TimbreChannelsPayload(
        category="timbre_balance_channels",
        channel_group_fingerprint=group.fingerprint,
        primary_receiver_id=mono.payload.primary_receiver_id,
        timbre_evaluator_version=mono.payload.timbre_evaluator_version,
        timbre_settings_fingerprint=mono.payload.timbre_settings_fingerprint,
        channels=mono.payload.channels,
    )
    forged = mono.model_copy(update={"payload": forged_payload})

    result = rank_candidates(
        (_candidate(genuine), _candidate(forged)),
        load_quality_targets(_TARGETS),
        _context(group),
    )

    assert {
        result.status_of("genuine"), result.status_of("forged")
    } == {CandidateStatus.RANKABLE, CandidateStatus.NOT_COMPARABLE}


def test_each_channel_preserves_provenance_and_flags_in_stable_output() -> None:
    """只留外層聯集會丟掉旗標與 report_id 到底屬於哪一支。"""
    group = _group()
    left = _single(
        "left",
        flags=(Flag.DATA_COVERAGE_SHORT,),
        report_id="left-origin",
    )
    right = _single(
        "right",
        flags=(Flag.NO_DIRECTIVITY,),
        report_id="right-origin",
    )

    left_first = _aggregate(group, {"left": left, "right": right})
    right_first = _aggregate(group, {"right": right, "left": left})

    assert left_first == right_first
    assert isinstance(left_first.payload, TimbreChannelsPayload)
    by_role = {item.role: item for item in left_first.payload.channels}
    assert by_role["left"].provenance == left.provenance
    assert by_role["right"].provenance == right.provenance
    assert by_role["left"].flags == left.flags
    assert by_role["right"].flags == right.flags
    assert left_first.flags == (Flag.NO_DIRECTIVITY, Flag.DATA_COVERAGE_SHORT)


def test_comparison_free_group_is_mono_only_and_never_empty() -> None:
    """把「無比較對」放寬到多聲道，會造出沒有任何比較依據的聲道組。"""
    with pytest.raises(ValidationError, match="比較對"):
        ChannelGroup(
            channels=(
                ChannelDefinition(role="left", speaker_id="speaker-left"),
                ChannelDefinition(role="right", speaker_id="speaker-right"),
            ),
            comparisons=(),
            feature_match_tolerance_hz=10.0,
        )
    assert _group(mono=True).comparisons == ()
    with pytest.raises(ValidationError):
        ChannelGroup(
            channels=(), comparisons=(), feature_match_tolerance_hz=10.0
        )


@pytest.mark.parametrize(
    ("change", "error_marker"),
    (
        ({"channels": ()}, "channels"),
        ({"channel_group_fingerprint": "short"}, "channel_group_fingerprint"),
        ("duplicate_role", "不可重複"),
        ("duplicate_speaker", "不可重複"),
    ),
)
def test_timbre_channels_payload_rejects_bad_direct_data(
    change: dict[str, object] | str, error_marker: str
) -> None:
    """繞過彙總器直接造 payload，不能塞空清單、重複身分或短指紋。"""
    aggregate = _aggregate(
        _group(), {"left": _single("left"), "right": _single("right")}
    )
    assert isinstance(aggregate.payload, TimbreChannelsPayload)
    document = aggregate.payload.model_dump(mode="python")
    if change == "duplicate_role":
        left, right = document["channels"]
        document["channels"] = (left, {**right, "role": "left"})
    elif change == "duplicate_speaker":
        left, right = document["channels"]
        document["channels"] = (
            left,
            {**right, "speaker_id": left["speaker_id"]},
        )
    else:
        assert isinstance(change, dict)
        document.update(change)

    with pytest.raises(ValidationError) as caught:
        TimbreChannelsPayload.model_validate(document)

    assert error_marker in str(caught.value)


@pytest.mark.parametrize(
    ("role", "speaker_id", "location"),
    (
        ("Left Channel", "speaker-left", "role"),
        ("left", "", "speaker_id"),
    ),
)
def test_timbre_channel_rejects_bad_direct_identity(
    role: str, speaker_id: str, location: str
) -> None:
    """繞過 payload 外殼直接造單支列，不能塞壞角色格式或空喇叭代號。"""
    source = _single("left")
    document = {
        "role": role,
        "speaker_id": speaker_id,
        "payload": source.payload,
        "provenance": source.provenance,
        "flags": source.flags,
    }

    with pytest.raises(ValidationError) as caught:
        TimbreChannel.model_validate(document)

    assert location in {str(item) for error in caught.value.errors() for item in error["loc"]}


def test_aggregator_rejects_costed_single_and_channel_aggregate_inputs() -> None:
    """只看外層音色類名會把 COSTED 單支或既有聲道彙總再次當作原始單支。"""
    group = _group()
    registry = load_quality_targets(_TARGETS)
    left = _single("left")
    right = _single("right")
    costed_left = cost_timbre_evaluation(
        left, registry.purpose(_PURPOSE), registry.fingerprint
    )
    costed_result = _aggregate(group, {"left": costed_left, "right": right})
    prior = _aggregate(group, {"left": left, "right": right})
    aggregate_as_left = prior.model_copy(
        update={"provenance": left.provenance, "placement": left.placement}
    )
    aggregate_result = _aggregate(
        group, {"left": aggregate_as_left, "right": right}
    )

    assert costed_result.state is EvaluationState.UNAVAILABLE
    assert ReasonCode.TIMBRE_NOT_MEASURED in costed_result.reason_codes
    assert aggregate_result.state is EvaluationState.UNAVAILABLE
    assert ReasonCode.TIMBRE_NOT_MEASURED in aggregate_result.reason_codes


def test_measured_output_is_order_stable_and_uses_declared_channel_identity() -> None:
    """照 mapping 插入順序輸出或拿角色名冒充喇叭代號，會讓同一聲道組產出漂移。"""
    group = _group()
    version = "upstream-timbre-special-v7"
    left = _single("left", evaluator_version=version)
    right = _single("right", evaluator_version=version)

    forward = _aggregate(group, {"left": left, "right": right})
    reverse = _aggregate(group, {"right": right, "left": left})

    assert forward == reverse
    assert isinstance(forward.payload, TimbreChannelsPayload)
    assert tuple(item.role for item in forward.payload.channels) == (
        "left",
        "right",
    )
    assert {item.role: item.speaker_id for item in forward.payload.channels} == {
        "left": "speaker-left",
        "right": "speaker-right",
    }
    assert forward.payload.timbre_evaluator_version == version


def test_nonplacement_failure_still_preserves_merged_channel_placement() -> None:
    """遇到非擺位型不可估就清空 placement，會丟掉兩支原本一致而可追查的擺位。"""
    left = _single("left", reason=ReasonCode.SOLVER_UNAVAILABLE)
    right = _single("right")

    result = _aggregate(_group(), {"left": left, "right": right})

    assert result.state is EvaluationState.UNAVAILABLE
    assert result.placement.speaker_positions_m == (
        ("speaker-left", (1.0, 1.0, 1.1)),
        ("speaker-right", (1.0, 3.0, 1.1)),
    )
    assert result.placement.receiver_positions_m == (("main", (4.0, 2.0, 1.2)),)


def test_raw_quantities_keep_each_channel_prefix_and_value() -> None:
    """只轉第一支或漏角色前綴，會讓左右原始量互相覆蓋或消失。"""
    result = _aggregate(
        _group(),
        {
            "left": _single("left", payload=_payload(tilt=0.25, residual=0.75)),
            "right": _single("right", payload=_payload(tilt=1.25, residual=1.75)),
        },
    )

    assert {item.name: item.value for item in result.raw_quantities} == {
        "left.tilt": 0.25,
        "left.residual": 0.75,
        "right.tilt": 1.25,
        "right.residual": 1.75,
    }


def test_second_channel_target_tilt_must_match_registry() -> None:
    """只檢查第一支的目標傾斜，會讓右支沿用另一把尺卻照樣算代價。"""
    aggregate = _aggregate(
        _group(),
        {
            "left": _single("left", payload=_payload(target_tilt=0.0)),
            "right": _single("right", payload=_payload(target_tilt=0.5)),
        },
    )
    registry = load_quality_targets(_TARGETS)

    with pytest.raises(ValueError, match="目標傾斜"):
        cost_timbre_evaluation(
            aggregate, registry.purpose(_PURPOSE), registry.fingerprint
        )


def test_channel_components_cost_weight_and_raw_values_stay_with_their_role() -> None:
    """把左右分項串錯、權重未除以聲道數或 raw_value 留空，這題會逐列抓到。"""
    group = _group()
    left_payload = _payload(
        tilt=0.25,
        residual=0.75,
        target_deviation=1.25,
        features=(
            Feature(
                kind="peak",
                center_frequency_hz=1000.0,
                depth_db=1.5,
                width_octave=0.5,
                flags=(),
            ),
        ),
    )
    right_payload = _payload(
        tilt=1.25,
        residual=1.75,
        target_deviation=2.25,
        features=(
            Feature(
                kind="dip",
                center_frequency_hz=2000.0,
                depth_db=-2.5,
                width_octave=0.5,
                flags=(),
            ),
        ),
    )
    left = _single("left", payload=left_payload)
    right = _single("right", payload=right_payload)
    registry = load_quality_targets(_TARGETS)
    purpose = registry.purpose(_PURPOSE)
    aggregate = _aggregate(group, {"left": left, "right": right})
    result = rank_candidates(
        (_candidate(aggregate),), registry, _context(group)
    )
    left_cost = cost_timbre_evaluation(left, purpose, registry.fingerprint)
    right_cost = cost_timbre_evaluation(right, purpose, registry.fingerprint)
    assert left_cost.category_cost is not None
    assert right_cost.category_cost is not None
    lines = {
        item.name: item for item in result.rankable[0].categories[0].components
    }
    for role, expected in (
        ("left", left_cost.category_cost.components),
        ("right", right_cost.category_cost.components),
    ):
        for name, value in expected.items():
            assert lines[f"{role}.{name}"].cost == value
    weights = purpose.entry("timbre_balance.within_category_weights")
    assert isinstance(weights, WeightTable)
    registered = {item.name: item.value for item in weights.item}
    for role in ("left", "right"):
        for name, value in registered.items():
            assert lines[f"{role}.{name}"].weight == value / len(group.channels)
    assert lines["left.target_deviation"].raw_value == left_payload.target_deviation_rms_db
    assert lines["right.target_deviation"].raw_value == right_payload.target_deviation_rms_db
    assert lines["left.peaks_dips"].raw_value == abs(left_payload.features[0].depth_db)
    assert lines["right.peaks_dips"].raw_value == abs(right_payload.features[0].depth_db)


@pytest.mark.parametrize("name", ("tilt", "center.tilt"))
def test_timbre_channel_lines_reject_missing_or_unknown_role_prefix(name: str) -> None:
    """分項名沒有角色前綴或冒用 payload 外角色，不能被排名列接受。"""
    aggregate = _aggregate(
        _group(), {"left": _single("left"), "right": _single("right")}
    )
    assert isinstance(aggregate.payload, TimbreChannelsPayload)
    cost = CategoryCost(
        value=0.0,
        components={name: 0.0},
        cost_settings_fingerprint="cost-settings",
    )

    with pytest.raises(ValueError, match="角色前綴"):
        _timbre_channel_lines(
            cost,
            aggregate.payload,
            load_quality_targets(_TARGETS).purpose(_PURPOSE),
        )


def test_ranking_row_includes_feature_flag_from_second_channel() -> None:
    """排名列只掃第一支 features，會漏掉第二支獨有的特徵旗標。"""
    flagged = Feature(
        kind="peak",
        center_frequency_hz=1000.0,
        depth_db=1.0,
        width_octave=None,
        flags=(Flag.FEATURE_BOUNDARY_INCOMPLETE,),
    )
    group = _group()
    aggregate = _aggregate(
        group,
        {
            "left": _single("left"),
            "right": _single("right", payload=_payload(features=(flagged,))),
        },
    )

    result = rank_candidates(
        (_candidate(aggregate),), load_quality_targets(_TARGETS), _context(group)
    )

    assert Flag.FEATURE_BOUNDARY_INCOMPLETE in result.rankable[0].flags


def test_comparison_support_is_stable_canonical_json_with_channel_identities() -> None:
    """漏逐聲道身分、未排序鍵或輸出空白，會破壞比較身分的正規字串。"""
    group = _group()
    aggregate = _aggregate(
        group, {"left": _single("left"), "right": _single("right")}
    )
    expected = (
        '{"channel_group_fingerprint":"'
        + group.fingerprint
        + '","channels":[{"role":"left","speaker_id":"speaker-left"},'
        '{"role":"right","speaker_id":"speaker-right"}],'
        '"primary_receiver_id":"main",'
        '"timbre_evaluator_version":"timbre-upstream-v1"}'
    )

    first = comparison_support(aggregate)
    second = comparison_support(aggregate)

    assert first == second
    assert first == expected
