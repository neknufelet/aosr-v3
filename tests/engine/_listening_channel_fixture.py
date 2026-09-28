"""既有單支聆聽區考卷裝候選包時使用的單角色彙總原料。"""

from __future__ import annotations

from aosr.scoring.contract import (
    CategoryEvaluation, ListeningAreaChannel, ListeningAreaChannelsPayload,
    ListeningAreaStabilityPayload,
)


def as_listening_channels(
    evaluation: CategoryEvaluation, group_fingerprint: str,
    *, role: str = "left",
) -> CategoryEvaluation:
    """單角色彙總的類代價等於原單支，保留舊考卷的數值控制意義。"""
    payload = evaluation.payload
    if not isinstance(payload, ListeningAreaStabilityPayload):
        raise TypeError("考卷需要已量的單支聆聽區")
    aggregate = ListeningAreaChannelsPayload(
        category="listening_area_stability_channels",
        channel_group_fingerprint=group_fingerprint,
        receiver_set_fingerprint=payload.receiver_set_fingerprint,
        primary_receiver_id=evaluation.provenance.receiver_id,
        listening_area_evaluator_version=evaluation.evaluator_version,
        listening_area_settings_fingerprint=evaluation.settings_fingerprint,
        channels=(ListeningAreaChannel(
            role=role, speaker_id=payload.speaker_id, payload=payload,
            provenance=evaluation.provenance, flags=evaluation.flags,
        ),),
    )
    document = evaluation.model_dump(mode="python")
    document["payload"] = aggregate.model_dump(mode="python")
    return CategoryEvaluation.model_validate(document)
