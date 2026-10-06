"""模態診斷自己的凍結文件；不借評分契約，未定義的量保留 None。"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from aosr.geometry.shoebox import Room, Wall
from aosr.config.fem_lane import FEM_ELEMENTS_PER_WAVELENGTH, FEM_MESH_RANDOM_SEED
from aosr.config.frequency_axis import FEM_GEOMETRIC_CROSSOVER_CAP_HZ
from aosr.physics.fem_modal import ModalSolverOptions
from aosr.physics.modal_convention import ModalKind

SCHEMA_VERSION: Literal["aosr.modal_diagnosis.v1"] = "aosr.modal_diagnosis.v1"
FROZEN = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False, revalidate_instances="always")
Finite = Annotated[float, Field(allow_inf_nan=False)]
Nonnegative = Annotated[Finite, Field(ge=0)]
Positive = Annotated[Finite, Field(gt=0)]
Index = Annotated[int, Field(strict=True, ge=0)]
PositiveInt = Annotated[int, Field(strict=True, gt=0)]
ModalIdentity = Annotated[str, Field(pattern=r"^modal-v1:[0-9a-f]{64}$")]
HexPoint = tuple[str, str, str]
Position = tuple[Finite, Finite, Finite]
Name = Annotated[str, Field(min_length=1)]


def hex_value(text: str) -> float:
    """鑰匙只接受有限、標準 float.hex，不收 NaN 或另一種拼法。"""
    try:
        value = float.fromhex(text)
    except OverflowError as exc:
        raise ValueError("模態鑰匙 hex 超過有限浮點範圍") from exc
    if not math.isfinite(value) or value.hex() != text:
        raise ValueError("模態鑰匙數值必須為有限、標準 float.hex 字串")
    return value


class ModalDiagnosisState(StrEnum):
    """診斷四態；已診斷不等於已計分，參考線範圍外另在逐共振表示。"""
    DIAGNOSED_NOT_SCORED = "diagnosed_not_scored"
    NOT_COMPUTED = "not_computed"
    FAILED = "failed"
    OUT_OF_SCOPE = "out_of_scope"


class ModalKey(BaseModel):
    """只收決定特徵問題的輸入；固定六面牆序，不收頻率軸或位置。"""
    model_config = FROZEN
    room_hex: HexPoint
    impedance_hex: tuple[str, ...]
    density_hex: str
    speed_hex: str
    mesh_frequency_max_hex: str
    elements_per_wavelength: PositiveInt
    mesh_random_seed: Index
    solver_options: ModalSolverOptions

    @model_validator(mode="after")
    def _valid_physics(self) -> Self:
        if len(self.impedance_hex) != len(Wall.all()):
            raise ValueError("模態鑰匙必須含六面實阻抗")
        values = (*self.room_hex, *self.impedance_hex, self.density_hex, self.speed_hex, self.mesh_frequency_max_hex)
        if any(hex_value(x) <= 0 for x in values):
            raise ValueError("房間、阻抗、密度、聲速與網格上限必須為正")
        if isinstance(self.solver_options.random_seed, bool) or not isinstance(self.solver_options.random_seed, int) or self.solver_options.random_seed < 0:
            raise ValueError("求解種子必須為非負整數")
        return self

    @classmethod
    def from_inputs(cls, *, room: Room, wall_impedances: Mapping[Wall, float],
                    density_kg_m3: float, sound_speed_m_s: float,
                    options: ModalSolverOptions = ModalSolverOptions()) -> Self:
        return cls(room_hex=(float(room.Lx).hex(), float(room.Ly).hex(), float(room.Lz).hex()),
                   impedance_hex=tuple(float(wall_impedances[w]).hex() for w in Wall.all()),
                   density_hex=float(density_kg_m3).hex(), speed_hex=float(sound_speed_m_s).hex(),
                   mesh_frequency_max_hex=float(FEM_GEOMETRIC_CROSSOVER_CAP_HZ).hex(),
                   elements_per_wavelength=FEM_ELEMENTS_PER_WAVELENGTH, mesh_random_seed=FEM_MESH_RANDOM_SEED,
                   solver_options=options)

    @property
    def digest(self) -> str:
        content = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(content.encode()).hexdigest()


class RoomMode(BaseModel):
    """解帳的原始三種類型；靜態的 T60 與 Q 未定義，不進重疊群。"""
    model_config = FROZEN
    mode_index: Index
    kind: ModalKind
    frequency_hz: Nonnegative
    omega_real_rad_s: Nonnegative
    omega_imag_rad_s: Nonnegative
    t60_s: Positive | None
    q: Nonnegative | None
    group_id: Index | None

    @model_validator(mode="after")
    def _kind_fields(self) -> Self:
        if self.kind is ModalKind.RESONANCE:
            if self.frequency_hz <= 0 or self.omega_real_rad_s <= 0 or self.omega_imag_rad_s <= 0:
                raise ValueError("共振必須為正頻率與正衰減；無阻尼共振不能保存有限診斷")
            if self.t60_s is None or self.q is None or self.q <= 0 or self.group_id is None:
                raise ValueError("共振必須有 T60、Q 與群號")
        else:
            if self.group_id is not None or self.frequency_hz != 0 or self.omega_real_rad_s != 0:
                raise ValueError("只有共振可以進群，其他種類頻率為零")
            if self.kind is ModalKind.STATIC and (self.t60_s is not None or self.q is not None or self.omega_imag_rad_s != 0):
                raise ValueError("靜態的 T60 與 Q 必須為 None")
            if self.kind is ModalKind.NONOSCILLATING_DECAY and (self.t60_s is None or self.q != 0 or self.omega_imag_rad_s <= 0):
                raise ValueError("純衰減必須有正 T60，算術 Q 為零")
        return self


class RoomGroup(BaseModel):
    """成員是解帳索引，照頻率排；群沒有代表成員或單一衰減量。"""
    model_config = FROZEN
    group_id: Index
    member_indices: tuple[Index, ...] = Field(min_length=1)
    lower_hz: Positive
    upper_hz: Positive

    @model_validator(mode="after")
    def _bounds(self) -> Self:
        if self.lower_hz > self.upper_hz or len(set(self.member_indices)) != len(self.member_indices):
            raise ValueError("群的頻率界線顛倒或成員重複")
        return self


class CountBand(BaseModel):
    """FemModalCheck（求解自檢）的逐段原始個數與 Weyl 估計，不作裁判。"""
    model_config = FROZEN
    lower_hz: Nonnegative
    upper_hz: Positive
    found_resonances: Index
    weyl_estimate: Nonnegative
    found_minus_weyl: Finite
    rigid_reference_count: Index | None
    found_minus_rigid: int | None


class CheckSummary(BaseModel):
    """保留 #661 保證高度的限制；高度為零時最低 T60 未定義，以 None 表示。"""
    model_config = FROZEN
    guaranteed_decay_rate_rad_s: Nonnegative
    guaranteed_min_t60_s: Positive | None
    count_bands: tuple[CountBand, ...]
    static_count: Index
    zero_mode_continuation_count: Index
    overdamped_count: Index
    unconfirmed_decay_count: Index
    returned_above_limit_count: Index
    weyl_terms: str

    @model_validator(mode="after")
    def _guarantee(self) -> Self:
        if (self.guaranteed_decay_rate_rad_s == 0) != (self.guaranteed_min_t60_s is None):
            raise ValueError("零保證高度的最低 T60 必須為 None")
        return self


class RoomLayer(BaseModel):
    """同一房間材料的解帳、群與自檢；位置與工程目標都不存這裡。

    ``mesh_sha256`` 是解這份帳那張網格的指紋：快取命中時照鑰匙重建的網格指紋不同，就當成沒有快取。
    """
    model_config = FROZEN
    key: ModalKey
    modal_identity: ModalIdentity
    mesh_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    degrees_of_freedom: PositiveInt
    mesh_frequency_max_hz: Positive
    modes: tuple[RoomMode, ...] = Field(min_length=1)
    groups: tuple[RoomGroup, ...]
    solve_seconds: Nonnegative
    check_summary: CheckSummary

    @model_validator(mode="after")
    def _groups_match(self) -> Self:
        if tuple(m.mode_index for m in self.modes) != tuple(range(len(self.modes))):
            raise ValueError("解帳索引必須連續且按索引保存")
        if self.mesh_frequency_max_hz != hex_value(self.key.mesh_frequency_max_hex):
            raise ValueError("房間層網格上限與鑰匙不同")
        if tuple(g.group_id for g in self.groups) != tuple(range(len(self.groups))):
            raise ValueError("群號必須連續且按群號保存")
        seen: set[int] = set()
        for group in self.groups:
            if any(i >= len(self.modes) or i in seen for i in group.member_indices):
                raise ValueError("群成員索引越界或跨群重複")
            members = tuple(self.modes[i] for i in group.member_indices)
            if any(m.kind is not ModalKind.RESONANCE or m.group_id != group.group_id for m in members):
                raise ValueError("群只收共振，而且群號必須相符")
            frequencies = tuple(m.frequency_hz for m in members)
            if frequencies != tuple(sorted(frequencies)) or (group.lower_hz, group.upper_hz) != (frequencies[0], frequencies[-1]):
                raise ValueError("群成員必須照頻率排序，界線必須等於首尾頻率")
            seen.update(group.member_indices)
        if seen != {m.mode_index for m in self.modes if m.kind is ModalKind.RESONANCE}:
            raise ValueError("每個共振必須且只能屬於一群")
        return self


class PlacementPair(BaseModel):
    """同載荷、同單位的絕對大小；索引只對房間層的共振列。"""
    model_config = FROZEN
    speaker_id: Name
    speaker_position_m: Position
    receiver_id: Name
    receiver_position_m: Position
    resonance_indices: tuple[Index, ...]
    resonance_magnitude: tuple[Nonnegative, ...]
    group_magnitude: tuple[Nonnegative, ...]

    @model_validator(mode="after")
    def _aligned(self) -> Self:
        if len(self.resonance_indices) != len(self.resonance_magnitude) or len(self.resonance_indices) != len(self.group_magnitude):
            raise ValueError("共振索引與兩組絕對大小必須等長")
        if len(set(self.resonance_indices)) != len(self.resonance_indices):
            raise ValueError("擺位層共振索引不可重複")
        return self


class PlacementLayer(BaseModel):
    """逐（喇叭、座位）的完整笛卡兒積；同一代號不能有兩個位置。"""
    model_config = FROZEN
    pairs: tuple[PlacementPair, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _complete_pairs(self) -> Self:
        sources: dict[str, Position] = {}
        receivers: dict[str, Position] = {}
        seen: set[tuple[str, str]] = set()
        for pair in self.pairs:
            key = (pair.speaker_id, pair.receiver_id)
            if key in seen:
                raise ValueError("喇叭與座位配對重複")
            if sources.setdefault(pair.speaker_id, pair.speaker_position_m) != pair.speaker_position_m or receivers.setdefault(pair.receiver_id, pair.receiver_position_m) != pair.receiver_position_m:
                raise ValueError("同一喇叭或座位代號的座標不一致")
            seen.add(key)
        if seen != {(s, r) for s in sources for r in receivers}:
            raise ValueError("擺位層缺少喇叭與座位配對")
        return self


class ModalDiagnosis(BaseModel):
    """獨立診斷文件，模型自己驗四態、身分、鑰匙與兩層的一致性。"""
    model_config = FROZEN
    schema_version: Literal["aosr.modal_diagnosis.v1"] = SCHEMA_VERSION
    state: ModalDiagnosisState
    reason_code: str | None = None
    reason_text: str | None = None
    key: ModalKey | None = None
    modal_identity: ModalIdentity | None = None
    room_layer: RoomLayer | None = None
    placement_layer: PlacementLayer | None = None

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.state is ModalDiagnosisState.DIAGNOSED_NOT_SCORED:
            if self.room_layer is None or self.placement_layer is None or self.key is None or self.modal_identity is None:
                raise ValueError("已診斷必須同時有鑰匙、身分與兩層")
            if self.reason_code is not None or self.reason_text is not None:
                raise ValueError("已診斷不帶失敗或範圍外原因")
        if self.state is ModalDiagnosisState.FAILED and (self.reason_text is None or not self.reason_text.strip()):
            raise ValueError("失敗必須保留非空的原始原因文字")
        if self.state is ModalDiagnosisState.OUT_OF_SCOPE and (self.reason_code is None or not self.reason_code.strip()):
            raise ValueError("範圍外必須有非空原因碼")
        if self.state is ModalDiagnosisState.NOT_COMPUTED and (self.room_layer is not None or self.placement_layer is not None):
            raise ValueError("未計算不能有兩層")
        if self.room_layer is not None:
            if self.key != self.room_layer.key or self.modal_identity != self.room_layer.modal_identity:
                raise ValueError("診斷與房間層的鑰匙或模態身分不一致")
        if self.placement_layer is not None:
            if self.room_layer is None:
                raise ValueError("擺位層必須有對應房間層")
            indices = tuple(m.mode_index for m in self.room_layer.modes if m.kind is ModalKind.RESONANCE)
            if any(pair.resonance_indices != indices for pair in self.placement_layer.pairs):
                raise ValueError("擺位層必須與房間層共振列同索引")
        return self
