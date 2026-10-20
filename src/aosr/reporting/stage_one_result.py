"""第一階段有限元素結果：獨立格式、保留原始能量與理想聲源和牆面假設。"""
from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator

from aosr.config.capabilities import CapabilityTable
from aosr.config.directivity_defaults import DirectivityDefaults
from aosr.config.frequency_axis import LowFrequencyAxis, low_frequency_axis_frequencies
from aosr.config.quality_targets import QualityTargets
from aosr.geometry.shoebox import Wall
from aosr.reporting import fem_slices
from aosr.reporting.fem_slices import FemKey, FemShard
from aosr.reporting.result import ResultOrigin
from aosr.reporting.scheme import FROZEN, Scheme
from aosr.scoring.receiver_set import ReceiverRole
from aosr.scoring.stage_one_low_frequency import evaluate_stage_one_low_frequency
from aosr.scoring.stage_one_low_frequency_contract import (
    STAGE_ONE_SURROUNDING_DIRECTIONS, StageOneLowFrequencyEvaluation, StageOneLowFrequencyInput,
    StageOneSeatCurve, StageOneSpeakerCurves,
)
from aosr.scoring.stage_one_score import StageOneScore, score_stage_one, stage_one_score_settings


def check_stage_one_prerequisites(scheme: Scheme) -> None:
    """第一階段的計算前提；只驗七點的角色與方向，不另驗距離。"""
    if scheme.source_model != "omnidirectional":
        raise ValueError("第一階段只收全向聲源，請把聲源設成全向")
    if scheme.speaker_setup is not None and scheme.speaker_setup.mount == "desk":
        raise ValueError("第一階段只收落地模式，不收桌面模式")
    if scheme.furniture:
        raise ValueError("第一階段不放家具，請移除所有家具")
    if scheme.scene.surface_zones_by_wall is not None:
        raise ValueError("第一階段不收牆面分區或頻變材料，只收六面同一個頻率無關實數阻抗")
    walls = scheme.scene.impedance_pa_s_per_m_by_wall
    if walls is None:
        raise ValueError("第一階段只收六面同一個頻率無關實數阻抗")
    if (set(walls) != {wall.wall_name() for wall in Wall.all()}
            or len(set(walls.values())) != 1):
        raise ValueError("第一階段六面必須使用同一個頻率無關實數阻抗")
    if any(not isinstance(value, (int, float)) or value <= 0 for value in walls.values()):
        raise ValueError("第一階段牆面必須是正的頻率無關實數阻抗")
    points = scheme.receiver_set.points
    primary = [point for point in points if point.role == ReceiverRole.PRIMARY]
    surrounding = [point for point in points if point.role == ReceiverRole.SURROUNDING]
    directions = [point.direction_relative_to_primary for point in surrounding]
    if (len(primary) != 1 or primary[0].direction_relative_to_primary is not None
            or len(points) != len(STAGE_ONE_SURROUNDING_DIRECTIONS) + len(primary)
            or len(directions) != len(set(directions))
            or set(directions) != STAGE_ONE_SURROUNDING_DIRECTIONS):
        raise ValueError("第一階段座位必須剛好是一個主位，加前後左右上下六個方向，不可多也不可少")


def wall_assumption_text(scheme: Scheme) -> str:
    """以方案自己的空氣條件現算阻抗倍數，不在程式寫基準材料數值。"""
    check_stage_one_prerequisites(scheme)
    walls = scheme.scene.impedance_pa_s_per_m_by_wall
    if walls is None:
        raise ValueError("缺少六面固定阻抗")
    multiple = next(iter(walls.values())) / (scheme.scene.density_kg_m3 * scheme.scene.sound_speed_m_s)
    return f"在這個牆面假設下：六面同一個頻率無關實數阻抗 {multiple:.1f} ρc（ρc 是空氣特性阻抗）"


def stage_one_input(
    scheme: Scheme, energies: Mapping[tuple[str, str], tuple[float, ...]], fem_key: FemKey,
) -> StageOneLowFrequencyInput:
    """補軸名字、座位方向與曲線代號；頻率逐位核對，能量原樣交給評估器。"""
    check_stage_one_prerequisites(scheme)
    axis = scheme.scene.low_frequency_axis or LowFrequencyAxis.SEARCH
    frequencies = low_frequency_axis_frequencies(axis)[0]
    if tuple(float(value).hex() for value in frequencies) != fem_key.frequency_hex:
        raise ValueError("宣告的低頻軸與分片有限元素頻率不同，請用同一條軸重算")
    expected = {(speaker, point.receiver_id) for speaker in scheme.speakers for point in scheme.receiver_set.points}
    if set(energies) != expected:
        raise ValueError("有限元素能量必須剛好包含每支喇叭與每個座位")
    speakers = []
    for speaker in scheme.speakers:
        seats = {point.receiver_id: StageOneSeatCurve(
            receiver_id=point.receiver_id, direction_relative_to_primary=point.direction_relative_to_primary,
            report_id=f"fem-{speaker}-{point.receiver_id}", frequencies_hz=frequencies,
            fem_energy=energies[speaker, point.receiver_id]) for point in scheme.receiver_set.points}
        speakers.append(StageOneSpeakerCurves(
            speaker_id=speaker, primary=seats[scheme.receiver_set.primary.receiver_id],
            surrounding=tuple(seats[point.receiver_id] for point in scheme.receiver_set.points
                              if point.role == ReceiverRole.SURROUNDING)))
    return StageOneLowFrequencyInput(candidate_id=scheme.scheme_id, low_frequency_axis=axis, speakers=tuple(speakers))


class StageOneResult(BaseModel):
    """只跑有限元素的第一階段結果；與第二階段結果檔互相拒收。"""

    model_config = FROZEN
    schema_version: Literal["aosr.stage_one_result.v1"]
    scope: Literal["stage_one_fem_only"]
    scheme: Scheme
    origin: ResultOrigin
    engine_commit: str = Field(min_length=1)
    physics_identity: str = Field(pattern=r"^phys-v1:[0-9a-f]{64}$")
    program_fingerprint: str = Field(pattern=r"^calc-v1:[0-9a-f]{64}$")
    low_frequency_axis: LowFrequencyAxis
    curves: tuple[StageOneSpeakerCurves, ...] = Field(min_length=1)
    evaluation: StageOneLowFrequencyEvaluation
    score: StageOneScore
    wall_assumption_text: str
    run_date: date

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        check_stage_one_prerequisites(self.scheme)
        axis = self.scheme.scene.low_frequency_axis or LowFrequencyAxis.SEARCH
        if self.low_frequency_axis != axis or self.evaluation.low_frequency_axis != axis:
            raise ValueError("結果低頻軸與方案不同")
        if self.wall_assumption_text != wall_assumption_text(self.scheme):
            raise ValueError("結果牆面假設與方案不同")
        if {item.speaker_id for item in self.curves} != set(self.scheme.speakers):
            raise ValueError("結果原始曲線的喇叭與方案不同")
        frequencies = tuple(float(value).hex() for value in low_frequency_axis_frequencies(axis)[0])
        points = {point.receiver_id: point.direction_relative_to_primary for point in self.scheme.receiver_set.points}
        for speaker in self.curves:
            seats = (speaker.primary, *speaker.surrounding)
            if ({seat.receiver_id: seat.direction_relative_to_primary for seat in seats} != points
                    or speaker.primary.receiver_id != self.scheme.receiver_set.primary.receiver_id):
                raise ValueError("結果原始曲線的座位與方案不同")
            for seat in seats:
                if (seat.report_id != f"fem-{speaker.speaker_id}-{seat.receiver_id}"
                        or tuple(value.hex() for value in seat.frequencies_hz) != frequencies):
                    raise ValueError("結果原始曲線的代號或頻率與方案不同")
        if (self.evaluation.candidate_id != self.scheme.scheme_id or self.score.candidate_id != self.scheme.scheme_id
                or self.evaluation.primary_receiver_id != self.scheme.receiver_set.primary.receiver_id):
            raise ValueError("結果評估與分數的方案或主位代號不同")
        return self


def run_stage_one(
    scheme: Scheme, *, capabilities: CapabilityTable, directivity: DirectivityDefaults,
    registry: QualityTargets, physics_identity: str, program_fingerprint: str,
    engine_commit: str, run_date: date, origin: ResultOrigin | None = None,
    shards: Sequence[FemShard] | None = None,
) -> StageOneResult:
    """無分片時在本行程一段解完整軸；有分片只合併，拒收不退回求解。"""
    check_stage_one_prerequisites(scheme)
    parts = shards if shards is not None else [fem_slices.solve_slice(
        [scheme], capabilities=capabilities, directivity=directivity, slices=1, slice_index=0,
        physics_identity=physics_identity)]
    if not parts:
        raise ValueError("分片不能為空")
    # 每一片的頻率由合併逐點對方案的軸核，不同就拒收；組輸入時再對 fem_key 核一次。
    energies = fem_slices.energies_from_shards(parts, scheme=scheme, capabilities=capabilities,
        directivity=directivity, physics_identity=physics_identity)
    data = stage_one_input(scheme, energies, parts[0].fem_key)
    settings = stage_one_score_settings(registry, scheme.purpose)
    evaluation = evaluate_stage_one_low_frequency(data, scoring_range_hz=settings.scored_range_hz)
    return StageOneResult(schema_version="aosr.stage_one_result.v1", scope="stage_one_fem_only",
        scheme=scheme, origin=origin or ResultOrigin(kind="run"), engine_commit=engine_commit,
        physics_identity=physics_identity, program_fingerprint=program_fingerprint,
        low_frequency_axis=data.low_frequency_axis, curves=data.speakers, evaluation=evaluation,
        score=score_stage_one(evaluation, settings), wall_assumption_text=wall_assumption_text(scheme), run_date=run_date)


def save_stage_one_result(result: StageOneResult, path: Path) -> None:
    """同目錄完整寫入並同步後原子替換；失敗只清自己的暫存檔。"""
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(result.model_dump_json())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)  # noqa: PTH105  # expires=2026-12-08 reason=同目錄原子替換需明用 os.replace
    finally:
        temporary.unlink(missing_ok=True)


def load_stage_one_result(path: Path) -> StageOneResult:
    """讀回第一階段獨立格式，不接受第二階段結果。"""
    return StageOneResult.model_validate_json(path.read_text(encoding="utf-8"))
