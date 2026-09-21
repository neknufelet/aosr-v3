"""把候選主位每支單聲道音色收成一條排名用的完整音色評估。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final

from aosr.scoring.channel_matching import ChannelGroup
from aosr.scoring.contract import (
    CONTRACT_SCHEMA_VERSION,
    CategoryEvaluation,
    EvaluationState,
    Flag,
    InputProvenance,
    QualityCategory,
    RawQuantity,
    ReasonCode,
    TimbreChannel,
    TimbreChannelsPayload,
    TimbrePayload,
    in_declared_order,
)
from aosr.scoring.placement import (
    PlacementMismatchError,
    merge_or_empty,
    merge_placements,
)


TIMBRE_CHANNELS_EVALUATOR_VERSION: Final[str] = "aosr.scoring.timbre_channels.v1"


def _ordered_evaluations(
    channel_group: ChannelGroup,
    evaluations: Mapping[str, CategoryEvaluation],
) -> tuple[CategoryEvaluation, ...]:
    return tuple(
        evaluations[channel.role]
        for channel in channel_group.channels
        if channel.role in evaluations
    )


def _all_evaluations(
    evaluations: Mapping[str, CategoryEvaluation],
) -> tuple[CategoryEvaluation, ...]:
    """額外角色也照角色名固定次序納入旗標與擺位診斷。"""
    return tuple(evaluations[role] for role in sorted(evaluations))


def _identity_reasons(
    channel_group: ChannelGroup,
    primary_receiver_id: str,
    evaluations: Mapping[str, CategoryEvaluation],
    candidate_id: str,
    scene_fingerprint: str,
    timbre_settings_fingerprint: str,
) -> tuple[ReasonCode, ...]:
    expected = {item.role: item.speaker_id for item in channel_group.channels}
    reasons: list[ReasonCode] = []
    if set(evaluations) != set(expected):
        reasons.append(ReasonCode.CHANNEL_ROLE_MISMATCH)
    if set(expected) - set(evaluations):
        reasons.append(ReasonCode.REQUIRED_CHANNEL_POINT_UNAVAILABLE)
    supplied = _all_evaluations(evaluations)
    if any(item.candidate_id != candidate_id for item in supplied):
        reasons.append(ReasonCode.CANDIDATE_ID_MISMATCH)
    if any(item.scene_fingerprint != scene_fingerprint for item in supplied):
        reasons.append(ReasonCode.SCENE_FINGERPRINT_MISMATCH)
    if any(
        item.settings_fingerprint != timbre_settings_fingerprint
        for item in supplied
    ):
        reasons.append(ReasonCode.TIMBRE_SETTINGS_FINGERPRINT_MISMATCH)
    if len({item.evaluator_version for item in supplied}) > 1:
        reasons.append(ReasonCode.EVALUATOR_VERSION_MISMATCH)
    if any(item.provenance.receiver_id != primary_receiver_id for item in supplied):
        reasons.append(ReasonCode.RECEIVER_ID_MISMATCH)
    for role, item in evaluations.items():
        if item.provenance.speaker_id != expected.get(role):
            reasons.append(ReasonCode.CHANNEL_ROLE_MISMATCH)
        if (
            item.category is not QualityCategory.TIMBRE_BALANCE
            or item.state is EvaluationState.COSTED
            or (
                item.state is EvaluationState.MEASURED
                and not isinstance(item.payload, TimbrePayload)
            )
        ):
            reasons.extend(
                (
                    ReasonCode.TIMBRE_NOT_MEASURED,
                    ReasonCode.REQUIRED_CHANNEL_POINT_UNAVAILABLE,
                )
            )
        if item.state is EvaluationState.UNAVAILABLE:
            reasons.extend(item.reason_codes)
            reasons.append(ReasonCode.REQUIRED_CHANNEL_POINT_UNAVAILABLE)
    try:
        merge_placements(item.placement for item in supplied)
    except PlacementMismatchError:
        reasons.append(ReasonCode.PLACEMENT_MISMATCH)
    return in_declared_order(reasons)


def _flags(evaluations: Sequence[CategoryEvaluation]) -> tuple[Flag, ...]:
    return in_declared_order(
        flag for evaluation in evaluations for flag in evaluation.flags
    )


def _provenance(
    candidate_id: str, primary_receiver_id: str, channel_group: ChannelGroup
) -> InputProvenance:
    return InputProvenance(
        report_id=f"timbre-channels:{candidate_id}",
        engine_commit="multiple-input-reports",
        speaker_id=f"channel-group:{channel_group.fingerprint}",
        receiver_id=primary_receiver_id,
    )


def _unavailable(
    channel_group: ChannelGroup,
    primary_receiver_id: str,
    evaluations: Mapping[str, CategoryEvaluation],
    candidate_id: str,
    scene_fingerprint: str,
    timbre_settings_fingerprint: str,
    reasons: tuple[ReasonCode, ...],
) -> CategoryEvaluation:
    supplied = _all_evaluations(evaluations)
    return CategoryEvaluation(
        schema_version=CONTRACT_SCHEMA_VERSION,
        candidate_id=candidate_id,
        scene_fingerprint=scene_fingerprint,
        placement=merge_or_empty(item.placement for item in supplied),
        category=QualityCategory.TIMBRE_BALANCE,
        state=EvaluationState.UNAVAILABLE,
        payload=None,
        raw_quantities=(),
        category_cost=None,
        flags=_flags(supplied),
        reason_codes=reasons,
        evaluator_version=TIMBRE_CHANNELS_EVALUATOR_VERSION,
        settings_fingerprint=timbre_settings_fingerprint,
        provenance=_provenance(candidate_id, primary_receiver_id, channel_group),
    )


def _payload(
    channel_group: ChannelGroup,
    primary_receiver_id: str,
    evaluations: Mapping[str, CategoryEvaluation],
    timbre_settings_fingerprint: str,
) -> TimbreChannelsPayload:
    versions = {item.evaluator_version for item in evaluations.values()}
    timbre_version = next(iter(versions))
    channels: list[TimbreChannel] = []
    for channel in channel_group.channels:
        evaluation = evaluations[channel.role]
        payload = evaluation.payload
        if not isinstance(payload, TimbrePayload):
            raise TypeError("可估的主位聲道必須帶單支 TimbrePayload")
        channels.append(
            TimbreChannel(
                role=channel.role,
                speaker_id=channel.speaker_id,
                payload=payload,
                provenance=evaluation.provenance,
                flags=evaluation.flags,
            )
        )
    return TimbreChannelsPayload(
        category="timbre_balance_channels",
        channel_group_fingerprint=channel_group.fingerprint,
        primary_receiver_id=primary_receiver_id,
        timbre_evaluator_version=timbre_version,
        timbre_settings_fingerprint=timbre_settings_fingerprint,
        channels=tuple(channels),
    )


def _raw_quantities(
    channel_group: ChannelGroup,
    evaluations: Mapping[str, CategoryEvaluation],
) -> tuple[RawQuantity, ...]:
    return tuple(
        RawQuantity(
            name=f"{channel.role}.{quantity.name}",
            value=quantity.value,
            unit=quantity.unit,
        )
        for channel in channel_group.channels
        for quantity in evaluations[channel.role].raw_quantities
    )


def evaluate_timbre_channels(
    channel_group: ChannelGroup,
    primary_receiver_id: str,
    evaluations: Mapping[str, CategoryEvaluation],
    *,
    candidate_id: str,
    scene_fingerprint: str,
    timbre_settings_fingerprint: str,
) -> CategoryEvaluation:
    """核對主位全部宣告聲道；任一缺失、不可估或身分不合就整類不可估。"""
    reasons = _identity_reasons(
        channel_group,
        primary_receiver_id,
        evaluations,
        candidate_id,
        scene_fingerprint,
        timbre_settings_fingerprint,
    )
    if reasons:
        return _unavailable(
            channel_group,
            primary_receiver_id,
            evaluations,
            candidate_id,
            scene_fingerprint,
            timbre_settings_fingerprint,
            reasons,
        )
    ordered = _ordered_evaluations(channel_group, evaluations)
    payload = _payload(
        channel_group,
        primary_receiver_id,
        evaluations,
        timbre_settings_fingerprint,
    )
    return CategoryEvaluation(
        schema_version=CONTRACT_SCHEMA_VERSION,
        candidate_id=candidate_id,
        scene_fingerprint=scene_fingerprint,
        placement=merge_placements(item.placement for item in ordered),
        category=QualityCategory.TIMBRE_BALANCE,
        state=EvaluationState.MEASURED,
        payload=payload,
        raw_quantities=_raw_quantities(channel_group, evaluations),
        category_cost=None,
        flags=_flags(ordered),
        reason_codes=(),
        evaluator_version=TIMBRE_CHANNELS_EVALUATOR_VERSION,
        settings_fingerprint=timbre_settings_fingerprint,
        provenance=_provenance(candidate_id, primary_receiver_id, channel_group),
    )
