"""第二階段存檔的有限元素欄接到共用組裝；不改聲源、家具或牆面。"""
from __future__ import annotations

from pathlib import Path

from aosr.config.frequency_axis import LowFrequencyAxis
from aosr.config.quality_targets import load_quality_targets
from aosr.reporting.result import SchemeResult
from aosr.scoring.contract import CategoryEvaluation
from aosr.scoring.contract_base import InputProvenance, ReasonCode
from aosr.scoring.low_frequency_input import LowFrequencyInputError, low_frequency_input
from aosr.scoring.low_frequency_response import evaluate_low_frequency_response
from aosr.scoring.placement import Placement


def evaluate_result_low_frequency(
    result: SchemeResult, quality_targets_path: Path, scene_fingerprint: str,
) -> CategoryEvaluation:
    """非 None 的有限元素頻率逐位對方案低頻軸；缺點與軸不符都不截、不補。"""
    scheme = result.scheme
    axis = scheme.scene.low_frequency_axis or LowFrequencyAxis.SEARCH
    roles = {channel.role: channel.speaker_id for channel in scheme.channel_group.channels}
    placement = Placement(
        speaker_positions_m=tuple((name, point.as_tuple()) for name, point in scheme.speakers.items()),
        receiver_positions_m=tuple((point.receiver_id, point.position_m) for point in scheme.receiver_set.points))
    provenance = InputProvenance(report_id=f"fem-{scheme.scheme_id}", engine_commit=result.engine_commit,
        speaker_id=next(iter(scheme.speakers)), receiver_id=scheme.receiver_set.primary.receiver_id)
    energies: dict[tuple[str, str], tuple[float | None, ...]] = {}
    axes: dict[tuple[str, str], tuple[str, ...]] = {}
    reason = None
    for pair in result.pairs:
        if pair.report.points is None:
            reason = ReasonCode.MISSING_POINTS
            break
        rows = tuple(point for point in pair.report.points if point.fem_energy is not None)
        axes[roles[pair.role], pair.receiver_id] = tuple(float(point.frequency_hz).hex() for point in rows)
        energies[roles[pair.role], pair.receiver_id] = tuple(point.fem_energy for point in rows)
    data = None
    if reason is None:
        try:
            data = low_frequency_input(candidate_id=scheme.scheme_id, speaker_ids=tuple(scheme.speakers),
                receiver_set=scheme.receiver_set, axis=axis, energies=energies, frequency_hex=axes)
        except LowFrequencyInputError as error:
            reason = error.reason
        except ValueError:
            # 軸與逐對資料已核過；剩下是量法輸入要求的主位加六方向或喇叭×座位缺漏。
            reason = ReasonCode.LOW_FREQUENCY_SEAT_LAYOUT_INVALID
    return evaluate_low_frequency_response(data, registry=load_quality_targets(quality_targets_path),
        purpose=scheme.purpose, scene_fingerprint=scene_fingerprint, placement=placement,
        provenance=provenance, input_reason=reason, candidate_id=scheme.scheme_id)
