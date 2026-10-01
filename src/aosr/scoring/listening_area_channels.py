"""把聆聽區各聲道的完整單支評估彙總成候選的一類。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final

from aosr.scoring.channel_group import ChannelGroup
from aosr.scoring.contract import (
    CONTRACT_SCHEMA_VERSION, CategoryEvaluation, EvaluationState, Flag,
    InputProvenance, ListeningAreaChannel, ListeningAreaChannelsPayload,
    ListeningAreaStabilityPayload, QualityCategory, RawQuantity, ReasonCode,
    in_declared_order,
)
from aosr.scoring.placement import PlacementMismatchError, merge_or_empty, merge_placements
from aosr.scoring.receiver_set import ReceiverSet
from aosr.scoring.source_model_identity import (
    distinct_source_model_fingerprints, unique_source_model_fingerprint,
)


LISTENING_AREA_CHANNELS_EVALUATOR_VERSION: Final[str] = (
    "aosr.scoring.listening_area_channels.v1"
)


def _all(evaluations: Mapping[str, CategoryEvaluation]) -> tuple[CategoryEvaluation, ...]:
    return tuple(evaluations[role] for role in sorted(evaluations))


def _identity_reasons(
    group: ChannelGroup, receivers: ReceiverSet,
    evaluations: Mapping[str, CategoryEvaluation], candidate_id: str,
    scene_fingerprint: str, settings_fingerprint: str,
) -> tuple[ReasonCode, ...]:
    expected = {item.role: item.speaker_id for item in group.channels}
    supplied = _all(evaluations)
    reasons: list[ReasonCode] = []
    if set(evaluations) != set(expected):
        reasons.append(ReasonCode.CHANNEL_ROLE_MISMATCH)
    if set(expected) - set(evaluations):
        reasons.append(ReasonCode.REQUIRED_CHANNEL_POINT_UNAVAILABLE)
    if len(distinct_source_model_fingerprints(
        item.source_model_fingerprint for item in supplied
    )) > 1:
        reasons.append(ReasonCode.SOURCE_MODEL_MISMATCH)
    if any(item.candidate_id != candidate_id for item in supplied):
        reasons.append(ReasonCode.CANDIDATE_ID_MISMATCH)
    if any(item.scene_fingerprint != scene_fingerprint for item in supplied):
        reasons.append(ReasonCode.SCENE_FINGERPRINT_MISMATCH)
    if any(item.settings_fingerprint != settings_fingerprint for item in supplied):
        reasons.append(ReasonCode.LISTENING_AREA_SETTINGS_FINGERPRINT_MISMATCH)
    if len({item.evaluator_version for item in supplied}) > 1:
        reasons.append(ReasonCode.EVALUATOR_VERSION_MISMATCH)
    if any(item.provenance.receiver_id != receivers.primary.receiver_id for item in supplied):
        reasons.append(ReasonCode.RECEIVER_ID_MISMATCH)
    for role, item in evaluations.items():
        payload = item.payload
        if item.provenance.speaker_id != expected.get(role):
            reasons.append(ReasonCode.CHANNEL_ROLE_MISMATCH)
        if isinstance(payload, ListeningAreaStabilityPayload):
            if payload.candidate_id != candidate_id:
                reasons.append(ReasonCode.CANDIDATE_ID_MISMATCH)
            if payload.settings_fingerprint != settings_fingerprint:
                reasons.append(ReasonCode.LISTENING_AREA_SETTINGS_FINGERPRINT_MISMATCH)
            if payload.speaker_id != expected.get(role):
                reasons.append(ReasonCode.CHANNEL_ROLE_MISMATCH)
            if payload.receiver_set_fingerprint != receivers.fingerprint:
                reasons.append(ReasonCode.RECEIVER_SET_FINGERPRINT_MISMATCH)
        if (item.category is not QualityCategory.LISTENING_AREA_STABILITY
                or item.state is not EvaluationState.MEASURED
                or (item.state is EvaluationState.MEASURED
                    and not isinstance(payload, ListeningAreaStabilityPayload))):
            reasons.extend((ReasonCode.CHANNEL_RESULT_UNAVAILABLE,
                            ReasonCode.REQUIRED_CHANNEL_POINT_UNAVAILABLE))
        if item.state is EvaluationState.UNAVAILABLE:
            reasons.extend(item.reason_codes)
            reasons.append(ReasonCode.REQUIRED_CHANNEL_POINT_UNAVAILABLE)
    try:
        merge_placements(item.placement for item in supplied)
    except PlacementMismatchError:
        reasons.append(ReasonCode.PLACEMENT_MISMATCH)
    return in_declared_order(reasons)


def _flags(evaluations: Sequence[CategoryEvaluation]) -> tuple[Flag, ...]:
    return in_declared_order(flag for item in evaluations for flag in item.flags)


def _provenance(candidate_id: str, primary: str, group: ChannelGroup) -> InputProvenance:
    return InputProvenance(
        report_id=f"listening-area-channels:{candidate_id}",
        engine_commit="multiple-input-reports",
        speaker_id=f"channel-group:{group.fingerprint}", receiver_id=primary,
    )


def _unavailable(
    group: ChannelGroup, receivers: ReceiverSet,
    evaluations: Mapping[str, CategoryEvaluation], candidate_id: str,
    scene_fingerprint: str, settings_fingerprint: str,
    reasons: tuple[ReasonCode, ...],
) -> CategoryEvaluation:
    supplied = _all(evaluations)
    return CategoryEvaluation(
        schema_version=CONTRACT_SCHEMA_VERSION, candidate_id=candidate_id,
        scene_fingerprint=scene_fingerprint,
        source_model_fingerprint=unique_source_model_fingerprint(
            item.source_model_fingerprint for item in supplied),
        placement=merge_or_empty(item.placement for item in supplied),
        category=QualityCategory.LISTENING_AREA_STABILITY,
        state=EvaluationState.UNAVAILABLE, payload=None, raw_quantities=(),
        category_cost=None, flags=_flags(supplied), reason_codes=reasons,
        evaluator_version=LISTENING_AREA_CHANNELS_EVALUATOR_VERSION,
        settings_fingerprint=settings_fingerprint,
        provenance=_provenance(candidate_id, receivers.primary.receiver_id, group),
    )


def _payload(
    group: ChannelGroup, receivers: ReceiverSet,
    evaluations: Mapping[str, CategoryEvaluation], settings_fingerprint: str,
) -> ListeningAreaChannelsPayload:
    channels: list[ListeningAreaChannel] = []
    for channel in sorted(group.channels, key=lambda item: item.role):
        evaluation = evaluations[channel.role]
        payload = evaluation.payload
        if not isinstance(payload, ListeningAreaStabilityPayload):
            raise TypeError("可估的聆聽區聲道必須帶單支結果")
        channels.append(ListeningAreaChannel(
            role=channel.role, speaker_id=channel.speaker_id, payload=payload,
            provenance=evaluation.provenance, flags=evaluation.flags,
        ))
    return ListeningAreaChannelsPayload(
        category="listening_area_stability_channels",
        channel_group_fingerprint=group.fingerprint,
        receiver_set_fingerprint=receivers.fingerprint,
        primary_receiver_id=receivers.primary.receiver_id,
        listening_area_evaluator_version=next(iter(
            {item.evaluator_version for item in evaluations.values()})),
        listening_area_settings_fingerprint=settings_fingerprint,
        channels=tuple(channels),
    )


def _raw_quantities(
    group: ChannelGroup, evaluations: Mapping[str, CategoryEvaluation],
) -> tuple[RawQuantity, ...]:
    return tuple(RawQuantity(
        name=f"{channel.role}.{quantity.name}", value=quantity.value,
        unit=quantity.unit,
    ) for channel in sorted(group.channels, key=lambda item: item.role)
        for quantity in evaluations[channel.role].raw_quantities)


def evaluate_listening_area_channels(
    channel_group: ChannelGroup, receiver_set: ReceiverSet,
    evaluations: Mapping[str, CategoryEvaluation], *, candidate_id: str,
    scene_fingerprint: str, listening_area_settings_fingerprint: str,
) -> CategoryEvaluation:
    """核對各聲道；任一單支不可估或身分不合就整類不可估。"""
    reasons = _identity_reasons(
        channel_group, receiver_set, evaluations, candidate_id,
        scene_fingerprint, listening_area_settings_fingerprint,
    )
    if reasons:
        return _unavailable(
            channel_group, receiver_set, evaluations, candidate_id,
            scene_fingerprint, listening_area_settings_fingerprint, reasons,
        )
    ordered = tuple(evaluations[item.role] for item in sorted(
        channel_group.channels, key=lambda channel: channel.role))
    return CategoryEvaluation(
        schema_version=CONTRACT_SCHEMA_VERSION, candidate_id=candidate_id,
        scene_fingerprint=scene_fingerprint,
        source_model_fingerprint=unique_source_model_fingerprint(
            item.source_model_fingerprint for item in ordered),
        placement=merge_placements(item.placement for item in ordered),
        category=QualityCategory.LISTENING_AREA_STABILITY,
        state=EvaluationState.MEASURED, payload=_payload(
            channel_group, receiver_set, evaluations, listening_area_settings_fingerprint),
        raw_quantities=_raw_quantities(channel_group, evaluations), category_cost=None,
        flags=_flags(ordered), reason_codes=(),
        evaluator_version=LISTENING_AREA_CHANNELS_EVALUATOR_VERSION,
        settings_fingerprint=listening_area_settings_fingerprint,
        provenance=_provenance(candidate_id, receiver_set.primary.receiver_id, channel_group),
    )
