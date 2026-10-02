"""物理身分範圍內的有限元素輸入、連續切段與可核對的 hex 分片。

分片自帶模型，不借結果檔；它只保存有限元素相關條件與各候選自己的取值座標。
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from aosr.config.capabilities import CapabilityTable
from aosr.config.directivity_defaults import DirectivityDefaults
from aosr.config.frequency_axis import low_frequency_axis_frequencies
from aosr.geometry.shoebox import Point, Room, Wall
from aosr.physics import report_io, three_lane_report
from aosr.physics.fem_batch import solve_fem_energies_many
from aosr.reporting.scheme import Scheme
from aosr.reporting.validation import checked_inputs

FROZEN = ConfigDict(frozen=True, extra="forbid")
HexPoint = tuple[str, str, str]
Index = Annotated[int, Field(strict=True, ge=0)]


def _hex_value(text: str) -> float:
    try:
        value = float.fromhex(text)
    except OverflowError as exc:
        raise ValueError("分片 hex 超過有限浮點數範圍") from exc
    if not math.isfinite(value) or value.hex() != text:
        raise ValueError("分片數值必須是有限、標準 float.hex 字串")
    return value


@dataclass(frozen=True)
class FemInputs:
    """已驗方案的有限元素輸入，兩條物理路共用同一份取值。"""

    room: Room
    wall_impedances: Mapping[Wall, float]
    density_kg_m3: float
    sound_speed_m_s: float
    frequencies_hz: tuple[float, ...]
    sources: Mapping[str, Point]
    receivers: Mapping[str, Point]


def fem_inputs(scheme: Scheme, solved: report_io.SolverInputs) -> FemInputs:
    """從驗過的逐對輸入（``report_io.solver_inputs``）取值，跟主流程求解吃的是同一份。

    主對話判斷：不直接讀原始方案的 scene——主流程一向吃驗過的輸入，兩邊各取各的就可能悄悄分岔。
    聲源模型、散射與反射階數不決定有限元素的任何輸入。
    """
    return FemInputs(
        room=solved.room, wall_impedances=three_lane_report._wall_impedances(solved.impedance_by_wall),
        density_kg_m3=solved.density_kg_m3, sound_speed_m_s=solved.sound_speed_m_s,
        frequencies_hz=low_frequency_axis_frequencies(solved.low_frequency_axis)[0], sources=scheme.speakers,
        receivers={point.receiver_id: Point(*point.position_m) for point in scheme.receiver_set.points})


def _checked_fem_inputs(scheme: Scheme, *, capabilities: CapabilityTable,
                        directivity: DirectivityDefaults) -> tuple[Scheme, FemInputs]:
    """先過 ``checked_inputs``，再從第一對驗過的輸入取有限元素的值。"""
    checked, documents = checked_inputs(scheme, capabilities=capabilities, directivity=directivity)
    solved = report_io.solver_inputs(next(iter(documents.values()))[1])
    return checked, fem_inputs(checked, solved)


class FemKey(BaseModel):
    """固定牆序的六面阻抗與整條頻率軸；只有改變矩陣的條件進鑰匙。"""

    model_config = FROZEN
    room_hex: HexPoint
    impedance_hex: tuple[str, ...]
    density_hex: str
    speed_hex: str
    frequency_hex: tuple[str, ...]

    @model_validator(mode="after")
    def _valid_physics(self) -> Self:
        if len(self.impedance_hex) != len(Wall.all()) or not self.frequency_hex:
            raise ValueError("有限元素鑰匙必須含六面阻抗與非空頻率軸")
        values = (*self.room_hex, *self.impedance_hex, self.density_hex, self.speed_hex, *self.frequency_hex)
        if any(_hex_value(value) <= 0.0 for value in values):
            raise ValueError("有限元素鑰匙數值必須為正")
        frequencies = tuple(_hex_value(value) for value in self.frequency_hex)
        if any(left >= right for left, right in zip(frequencies, frequencies[1:])):
            raise ValueError("有限元素頻率軸必須嚴格遞增")
        return self

    @classmethod
    def from_inputs(cls, inputs: FemInputs) -> Self:
        room = inputs.room
        return cls(room_hex=(float(room.Lx).hex(), float(room.Ly).hex(), float(room.Lz).hex()),
                   impedance_hex=tuple(float(inputs.wall_impedances[wall]).hex() for wall in Wall.all()),
                   density_hex=float(inputs.density_kg_m3).hex(), speed_hex=float(inputs.sound_speed_m_s).hex(),
                   frequency_hex=tuple(float(value).hex() for value in inputs.frequencies_hz))


def slice_indices(n: int, slices: int, slice_index: int) -> tuple[int, ...]:
    """連續切軸，餘數分給前段；段長最多差一且不允許空段。"""
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (n, slices, slice_index)):
        raise ValueError("切段參數必須是整數")
    if not 1 <= slices <= n or not 0 <= slice_index < slices:
        raise ValueError("slices 必須在 1..n，slice_index 必須在段數範圍內")
    size, remainder = divmod(n, slices)
    start = slice_index * size + min(slice_index, remainder)
    return tuple(range(start, start + size + (slice_index < remainder)))


class FemPair(BaseModel):
    """JSON 可存讀的一對喇叭與座位；能量是純量 float 的 hex。"""

    model_config = FROZEN
    speaker_id: str
    receiver_id: str
    energy_hex: tuple[str, ...]


class FemCandidate(BaseModel):
    """候選自己的聲源、座位與逐對能量，避免跨候選取值。"""

    model_config = FROZEN
    speakers: dict[str, HexPoint]
    receivers: dict[str, HexPoint]
    energies: tuple[FemPair, ...]

    @model_validator(mode="after")
    def _valid_candidate(self) -> Self:
        if not self.speakers or not self.receivers:
            raise ValueError("分片候選的喇叭與座位不能為空")
        pairs = tuple((pair.speaker_id, pair.receiver_id) for pair in self.energies)
        if len(set(pairs)) != len(pairs) or set(pairs) != {
                (source, receiver) for source in self.speakers for receiver in self.receivers}:
            raise ValueError("分片能量必須恰好包含候選的每一對")
        for coordinates in (*self.speakers.values(), *self.receivers.values()):
            for coordinate in coordinates:
                _hex_value(coordinate)
        if any(math.copysign(1.0, _hex_value(value)) < 0.0
               for pair in self.energies for value in pair.energy_hex):
            raise ValueError("分片能量必須非負")
        return self


class FemShard(BaseModel):
    """一段完整候選批次；模型凍結、拒收額外欄位，格式不依賴結果檔。"""

    model_config = FROZEN
    physics_identity: str = Field(min_length=1)
    fem_key: FemKey
    slices: int = Field(strict=True, ge=1)
    slice_index: Index
    indices: tuple[Index, ...]
    frequency_hex: tuple[str, ...]
    candidates: dict[str, FemCandidate]

    @model_validator(mode="after")
    def _valid_slice(self) -> Self:
        if self.indices != slice_indices(len(self.fem_key.frequency_hex), self.slices, self.slice_index):
            raise ValueError("分片索引不符合連續切段")
        if self.frequency_hex != tuple(self.fem_key.frequency_hex[index] for index in self.indices):
            raise ValueError("分片頻率 hex 與索引不符")
        if not self.candidates:
            raise ValueError("分片候選不能為空")
        if any(len(pair.energy_hex) != len(self.indices)
               for candidate in self.candidates.values() for pair in candidate.energies):
            raise ValueError("分片能量長度與索引不符")
        return self


class ShardIdentityMismatch(ValueError):
    """分片來自另一個物理身分，呼叫端可分辨此拒收原因。"""


def _coordinate_hex(points: Mapping[str, Point]) -> dict[str, HexPoint]:
    return {name: (float(point.x).hex(), float(point.y).hex(), float(point.z).hex())
            for name, point in points.items()}


def solve_slice(
    schemes: Sequence[Scheme], *, capabilities: CapabilityTable, directivity: DirectivityDefaults,
    slices: int, slice_index: int, physics_identity: str,
) -> FemShard:
    """整批先驗再比較有限元素鑰匙；不同就報錯，不偷偷拆批或換求解法。"""
    if not schemes or len({scheme.scheme_id for scheme in schemes}) != len(schemes):
        raise ValueError("方案不能為空，候選代號不可重複")
    inputs: dict[str, FemInputs] = {}
    for scheme in schemes:
        checked, item = _checked_fem_inputs(scheme, capabilities=capabilities, directivity=directivity)
        inputs[checked.scheme_id] = item
    first = next(iter(inputs.values()))
    key = FemKey.from_inputs(first)
    if any(FemKey.from_inputs(item) != key for item in inputs.values()):
        raise ValueError("整批候選的有限元素鑰匙必須相同")
    indices = slice_indices(len(first.frequencies_hz), slices, slice_index)
    energies = solve_fem_energies_many(
        room=first.room, candidates={name: (item.sources, item.receivers) for name, item in inputs.items()},
        wall_impedances=first.wall_impedances, frequencies_hz=first.frequencies_hz,
        density_kg_m3=first.density_kg_m3, sound_speed_m_s=first.sound_speed_m_s,
        frequency_indices=indices)
    return FemShard(
        physics_identity=physics_identity, fem_key=key, slices=slices, slice_index=slice_index,
        indices=indices, frequency_hex=tuple(key.frequency_hex[index] for index in indices),
        candidates={name: FemCandidate(
            speakers=_coordinate_hex(item.sources), receivers=_coordinate_hex(item.receivers),
            energies=tuple(FemPair(speaker_id=source, receiver_id=receiver,
                                  energy_hex=tuple(value.hex() for value in values))
                           for (source, receiver), values in energies[name].items()))
                    for name, item in inputs.items()})


def _checked_candidate(shard: FemShard, scheme: Scheme, inputs: FemInputs,
                       physics_identity: str) -> FemCandidate:
    if shard.physics_identity != physics_identity:
        raise ShardIdentityMismatch("分片物理身分不同")
    # model_copy 與可變容器不能繞過讀回時的完整模型驗證。
    shard = FemShard.model_validate(shard.model_dump())
    if shard.fem_key != FemKey.from_inputs(inputs):
        raise ValueError("分片有限元素鑰匙與方案不同")
    if scheme.scheme_id not in shard.candidates:
        raise ValueError("分片沒有這個候選")
    candidate = shard.candidates[scheme.scheme_id]
    if (candidate.speakers != _coordinate_hex(inputs.sources)
            or candidate.receivers != _coordinate_hex(inputs.receivers)):
        raise ValueError("分片喇叭或座位座標與方案不同")
    return candidate


def energies_from_shards(
    shards: Sequence[FemShard], *, scheme: Scheme, capabilities: CapabilityTable,
    directivity: DirectivityDefaults, physics_identity: str,
) -> dict[tuple[str, str], tuple[float, ...]]:
    """核對身分、鑰匙、座標與完整覆蓋，依索引放回整條軸，檔案順序無關。"""
    if not shards:
        raise ValueError("分片不能為空")
    scheme, inputs = _checked_fem_inputs(scheme, capabilities=capabilities, directivity=directivity)
    n = len(inputs.frequencies_hz)
    values: dict[tuple[str, str], list[float]] = {
        (source, receiver): [0.0] * n for source in inputs.sources for receiver in inputs.receivers}
    seen: set[int] = set()
    slices = shards[0].slices
    for shard in shards:
        candidate = _checked_candidate(shard, scheme, inputs, physics_identity)
        if shard.slices != slices:
            raise ValueError("分片 slices 不一致")
        if seen.intersection(shard.indices):
            raise ValueError("分片索引重疊")
        for index, frequency in zip(shard.indices, shard.frequency_hex, strict=True):
            if frequency != float(inputs.frequencies_hz[index]).hex():
                raise ValueError("分片頻率 hex 與方案不符")
        for pair in candidate.energies:
            for index, value in zip(shard.indices, pair.energy_hex, strict=True):
                values[pair.speaker_id, pair.receiver_id][index] = _hex_value(value)
        seen.update(shard.indices)
    if seen != set(range(n)):
        raise ValueError("分片頻率索引有洞，必須覆蓋整條軸")
    return {pair: tuple(energy) for pair, energy in values.items()}
