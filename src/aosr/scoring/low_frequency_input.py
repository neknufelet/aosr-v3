"""第一、二階段唯一的有限元素能量組裝路徑：逐位核軸，不改場景、不補能量。"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

from aosr.config.frequency_axis import LowFrequencyAxis, low_frequency_axis_frequencies
from aosr.scoring.contract_base import ReasonCode
from aosr.scoring.receiver_set import ReceiverRole, ReceiverSet
from aosr.scoring.stage_one_low_frequency_contract import (
    StageOneLowFrequencyInput, StageOneSeatCurve, StageOneSpeakerCurves,
)


class LowFrequencyInputError(ValueError):
    """共同組裝拒收的機器原因；第一階段仍是原本的 ValueError 人話。"""

    def __init__(self, message: str, reason: ReasonCode) -> None:
        super().__init__(message)
        self.reason = reason


def low_frequency_input(
    *, candidate_id: str, speaker_ids: Sequence[str], receiver_set: ReceiverSet,
    axis: LowFrequencyAxis, energies: Mapping[tuple[str, str], tuple[float | None, ...]],
    frequency_hex: tuple[str, ...] | Mapping[tuple[str, str], tuple[str, ...]],
) -> StageOneLowFrequencyInput:
    """補軸名字、座位方向與曲線代號；頻率逐位核對，能量原樣交給評估器。"""
    frequencies = low_frequency_axis_frequencies(axis)[0]
    expected_hex = tuple(float(value).hex() for value in frequencies)
    axes = (frequency_hex,) if isinstance(frequency_hex, tuple) else tuple(frequency_hex.values())
    for found in axes:
        if found != expected_hex:
            reason = (ReasonCode.MISSING_DECLARED_POINT if set(found) < set(expected_hex)
                      else ReasonCode.FREQUENCY_AXIS_MISMATCH)
            raise LowFrequencyInputError("宣告的低頻軸與分片有限元素頻率不同，請用同一條軸重算", reason)
    expected = {(speaker, point.receiver_id) for speaker in speaker_ids for point in receiver_set.points}
    if set(energies) != expected:
        raise LowFrequencyInputError("有限元素能量必須剛好包含每支喇叭與每個座位", ReasonCode.MISSING_POINTS)
    allowed = {ReceiverRole.PRIMARY, ReceiverRole.SURROUNDING}
    if any(point.role not in allowed for point in receiver_set.points):
        raise LowFrequencyInputError("座位必須是一個主位加前後左右上下六方向", ReasonCode.LOW_FREQUENCY_SEAT_LAYOUT_INVALID)
    if not isinstance(frequency_hex, tuple) and set(frequency_hex) != expected:
        raise LowFrequencyInputError("每支喇叭與每個座位都必須帶頻率軸", ReasonCode.MISSING_POINTS)
    speakers = []
    for speaker in speaker_ids:
        seats = {point.receiver_id: StageOneSeatCurve(
            receiver_id=point.receiver_id, direction_relative_to_primary=point.direction_relative_to_primary,
            report_id=f"fem-{speaker}-{point.receiver_id}", frequencies_hz=frequencies,
            fem_energy=energies[speaker, point.receiver_id]) for point in receiver_set.points}
        speakers.append(StageOneSpeakerCurves(
            speaker_id=speaker, primary=seats[receiver_set.primary.receiver_id],
            surrounding=tuple(seats[point.receiver_id] for point in receiver_set.points
                              if point.role == ReceiverRole.SURROUNDING)))
    return StageOneLowFrequencyInput(candidate_id=candidate_id, low_frequency_axis=axis, speakers=tuple(speakers))
