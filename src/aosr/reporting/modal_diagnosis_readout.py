"""讀回診斷時才對當下參考線算超出量與排序；純函式，不改快取或排名。"""
from __future__ import annotations

import math
from pydantic import BaseModel

from aosr.physics.modal_convention import ModalKind
from aosr.reporting.modal_diagnosis_model import (
    FROZEN, Finite, Index, ModalDiagnosis, ModalDiagnosisState, Nonnegative, Positive,
)
from aosr.scoring.low_frequency_decay_reference import LowFrequencyDecayReference, ReferenceOrigin


class ResonanceReadout(BaseModel):
    """逐共振目標；參考線範圍外的目標與超出量均為 None。"""
    model_config = FROZEN
    mode_index: Index
    frequency_hz: Positive
    t60_s: Positive
    target_t60_s: Positive | None
    target_origin: ReferenceOrigin
    excess_t60_s: Nonnegative | None


class GroupSummary(BaseModel):
    """群只給成員計數、頻率跨度與 T60 範圍，沒有代表成員或單一 T60。"""
    model_config = FROZEN
    group_id: Index
    member_count: Index
    frequency_span_hz: Nonnegative
    t60_range_s: tuple[Positive, Positive]
    exceeding_member_count: Index
    not_assessed_member_count: Index


class SortedResonance(BaseModel):
    """排序依自己的絕對大小；相對值只對同一配對最強共振，零大小沒有有限 dB。"""
    model_config = FROZEN
    mode_index: Index
    resonance_magnitude: Nonnegative
    group_magnitude: Nonnegative
    db_relative_to_pair_strongest_resonance: Finite | None


class GroupProfile(BaseModel):
    """該群在每個成員頻率上的整群大小，不濃縮成一個峰值。"""
    model_config = FROZEN
    group_id: Index
    mode_indices: tuple[Index, ...]
    frequencies_hz: tuple[Positive, ...]
    group_magnitude: tuple[Nonnegative, ...]


class PlacementReadout(BaseModel):
    model_config = FROZEN
    speaker_id: str
    receiver_id: str
    sorted_resonances: tuple[SortedResonance, ...]
    group_profiles: tuple[GroupProfile, ...]


class DiagnosisReadout(BaseModel):
    """未產生診斷的三態不偽造空測量表；原因跟原診斷一起傳出。"""
    model_config = FROZEN
    state: ModalDiagnosisState
    reason_code: str | None
    reason_text: str | None
    resonances: tuple[ResonanceReadout, ...] | None
    groups: tuple[GroupSummary, ...] | None
    placements: tuple[PlacementReadout, ...] | None


def _relative_to_pair_strongest_resonance_db(value: float, strongest: float) -> float | None:
    if value == 0 or strongest == 0:
        return None
    # 用對數差避免大小相除下溢；一般大小保留既有零件的算式。
    ratio = value / strongest
    return 20 * math.log10(ratio) if ratio else 20 * (math.log10(value) - math.log10(strongest))


def read_modal_diagnosis(diagnosis: ModalDiagnosis, reference: LowFrequencyDecayReference) -> DiagnosisReadout:
    """只吃診斷與 load_reference（用途參考線載入）的結果；不讀任何登記簿或求解。"""
    diagnosis = ModalDiagnosis.model_validate(diagnosis.model_dump())
    if diagnosis.state is not ModalDiagnosisState.DIAGNOSED_NOT_SCORED:
        return DiagnosisReadout(state=diagnosis.state, reason_code=diagnosis.reason_code, reason_text=diagnosis.reason_text, resonances=None, groups=None, placements=None)
    room, placement = diagnosis.room_layer, diagnosis.placement_layer
    assert room is not None and placement is not None  # 完整文件已由模型核對。
    resonances = []
    for mode in room.modes:
        if mode.kind is not ModalKind.RESONANCE:
            continue
        assert mode.t60_s is not None
        target = reference.target_at(mode.frequency_hz)
        excess = None if target.t60_s is None else max(0.0, mode.t60_s - target.t60_s)
        resonances.append(ResonanceReadout(mode_index=mode.mode_index, frequency_hz=mode.frequency_hz,
            t60_s=mode.t60_s, target_t60_s=target.t60_s, target_origin=target.origin, excess_t60_s=excess))
    by_index = {row.mode_index: row for row in resonances}
    summaries = []
    for group in room.groups:
        members = tuple(by_index[i] for i in group.member_indices)
        summaries.append(GroupSummary(group_id=group.group_id, member_count=len(members),
            frequency_span_hz=group.upper_hz - group.lower_hz,
            t60_range_s=(min(m.t60_s for m in members), max(m.t60_s for m in members)),
            exceeding_member_count=sum(m.excess_t60_s is not None and m.excess_t60_s > 0 for m in members),
            not_assessed_member_count=sum(m.target_origin is ReferenceOrigin.NOT_ASSESSED for m in members)))
    placements = []
    for pair in placement.pairs:
        strongest = max(pair.resonance_magnitude, default=None)
        magnitudes = dict(zip(pair.resonance_indices, pair.group_magnitude, strict=True))
        rows = tuple(SortedResonance(mode_index=i, resonance_magnitude=own, group_magnitude=whole,
            db_relative_to_pair_strongest_resonance=None if strongest is None else
                _relative_to_pair_strongest_resonance_db(own, strongest))
            for i, own, whole in zip(pair.resonance_indices, pair.resonance_magnitude, pair.group_magnitude, strict=True))
        profiles = tuple(GroupProfile(group_id=g.group_id, mode_indices=g.member_indices,
            frequencies_hz=tuple(by_index[i].frequency_hz for i in g.member_indices),
            group_magnitude=tuple(magnitudes[i] for i in g.member_indices)) for g in room.groups)
        placements.append(PlacementReadout(speaker_id=pair.speaker_id, receiver_id=pair.receiver_id,
            sorted_resonances=tuple(sorted(rows, key=lambda r: (-r.resonance_magnitude, r.mode_index))), group_profiles=profiles))
    return DiagnosisReadout(state=diagnosis.state, reason_code=diagnosis.reason_code, reason_text=diagnosis.reason_text, resonances=tuple(resonances), groups=tuple(summaries), placements=tuple(placements))
