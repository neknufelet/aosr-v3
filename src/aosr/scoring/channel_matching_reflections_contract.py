"""聲道匹配中的反射左右差診斷契約；不參與整類狀態或代價。"""
from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Self

from pydantic import Field, model_validator

from aosr.geometry.shoebox import WALL_SEQUENCE_ORDER
from aosr.scoring.contract_base import (
    Flag, FrequencyRange, FrozenModel, InputProvenance, MetricState, ReasonCode,
)
from aosr.scoring.direction_zones import DirectionZone
from aosr.scoring.reflections_contract import CONFIRMED_NO_REFLECTION, ReflectionSource


class ReflectionAsymmetryState(StrEnum):
    """一格的雙側有值、單側有值、雙側確認沒有或資料缺失。"""

    MEASURED = "measured"
    ONE_SIDED = "one_sided"
    BOTH_ABSENT = "both_absent"
    UNAVAILABLE = "unavailable"


class ReflectionSide(FrozenModel):
    """最強單條反射的聲級、延遲、路徑來源與報表出身。"""

    role: str = Field(min_length=1)
    speaker_id: str = Field(min_length=1)
    level_db: float
    delay_s: Annotated[float, Field(ge=0.0)]
    path_index: Annotated[int, Field(ge=0)]
    source: ReflectionSource
    source_index: Annotated[int, Field(ge=0)]
    order: Annotated[int, Field(ge=1)]
    wall_sequence: tuple[str, ...] = Field(description=WALL_SEQUENCE_ORDER)
    provenance: InputProvenance

    @model_validator(mode="after")
    def _source_is_consistent(self) -> Self:
        if self.provenance.speaker_id != self.speaker_id:
            raise ValueError("反射出身喇叭代號不一致")
        if len(self.wall_sequence) != self.order:
            raise ValueError("牆序列長度必須等於階數")
        return self


class ReflectionAsymmetryPoint(FrozenModel):
    """一比較對、一接收點、一方向區與一頻點的有號 L−R。"""

    left_role: str = Field(min_length=1)
    right_role: str = Field(min_length=1)
    receiver_id: str = Field(min_length=1)
    zone: DirectionZone
    frequency_hz: Annotated[float, Field(gt=0.0)]
    state: ReflectionAsymmetryState
    left_minus_right_db: float | None
    left: ReflectionSide | None
    right: ReflectionSide | None
    reason_codes: tuple[ReasonCode, ...]

    @model_validator(mode="after")
    def _state_is_consistent(self) -> Self:
        if self.left_role == self.right_role:
            raise ValueError("比較對角色不可相同")
        if self.left is not None and self.left.role != self.left_role:
            raise ValueError("左側角色不一致")
        if self.right is not None and self.right.role != self.right_role:
            raise ValueError("右側角色不一致")
        if self.state is ReflectionAsymmetryState.MEASURED:
            if (self.left is None or self.right is None or self.reason_codes or
                    self.left_minus_right_db != self.left.level_db - self.right.level_db):
                raise ValueError("已量格必須逐位等於左減右")
        elif self.state is ReflectionAsymmetryState.ONE_SIDED:
            if (self.left_minus_right_db is not None or (self.left is None) == (self.right is None)
                    or not self.reason_codes or not set(self.reason_codes) <= CONFIRMED_NO_REFLECTION):
                raise ValueError("單側格必須只有一邊與確認沒有原因")
        elif self.state is ReflectionAsymmetryState.BOTH_ABSENT:
            if (self.left_minus_right_db is not None or self.left is not None or self.right is not None
                    or not self.reason_codes or not set(self.reason_codes) <= CONFIRMED_NO_REFLECTION):
                raise ValueError("雙側皆無格不可帶聲級")
        elif (self.left_minus_right_db is not None or self.left is not None or self.right is not None
              or not self.reason_codes or set(self.reason_codes) <= CONFIRMED_NO_REFLECTION):
            raise ValueError("資料缺失格必須帶真正缺失原因")
        return self


class OneSidedReflection(FrozenModel):
    """單側有反射清單的一條：保留存在側及確認沒有側。"""

    left_role: str
    right_role: str
    receiver_id: str
    zone: DirectionZone
    frequency_hz: float
    present: ReflectionSide
    absent_role: str
    reason_codes: tuple[ReasonCode, ...]


class ReflectionAsymmetry(FrozenModel):
    """只量的原始診斷，和聲道匹配整類的可估性互不牽連。"""

    state: MetricState
    reason_codes: tuple[ReasonCode, ...]
    reflections_evaluator_version: str | None
    reflections_settings_fingerprint: str | None
    source_flags: tuple[Flag, ...]
    frequency_range_hz: FrequencyRange | None
    window_upper_ms: float | None
    primary_receiver_id: str | None
    comparison_order: tuple[tuple[str, str], ...]
    points: tuple[ReflectionAsymmetryPoint, ...]
    one_sided: tuple[OneSidedReflection, ...]

    @model_validator(mode="after")
    def _section_is_consistent(self) -> Self:
        if self.state is MetricState.MEASURED:
            if (self.reason_codes or not self.points or self.frequency_range_hz is None
                    or self.window_upper_ms is None or self.primary_receiver_id is None
                    or self.reflections_evaluator_version is None
                    or self.reflections_settings_fingerprint is None):
                raise ValueError("已量反射診斷必須有上游身分與逐格資料")
        elif self.state is MetricState.UNAVAILABLE:
            if not self.reason_codes or self.points or self.one_sided:
                raise ValueError("不可估反射診斷不得帶逐格資料")
        else:
            raise ValueError("反射診斷只准已量或不可估")
        if self.frequency_range_hz is not None and self.frequency_range_hz[0] >= self.frequency_range_hz[1]:
            raise ValueError("頻率範圍必須遞增")
        if len(self.comparison_order) != len(set(self.comparison_order)):
            raise ValueError("比較對宣告不可重複")
        order = {pair: index for index, pair in enumerate(self.comparison_order)}
        zones = {zone: index for index, zone in enumerate(DirectionZone)}
        keys = []
        expected = []
        for point in self.points:
            pair = (point.left_role, point.right_role)
            if pair not in order or self.frequency_range_hz is None or not self.frequency_range_hz[0] <= point.frequency_hz <= self.frequency_range_hz[1]:
                raise ValueError("比較對或頻率不在宣告範圍")
            keys.append((order[pair], point.receiver_id, zones[point.zone], point.frequency_hz))
            if point.state is ReflectionAsymmetryState.ONE_SIDED:
                present = point.left if point.left is not None else point.right
                assert present is not None
                expected.append(OneSidedReflection(
                    left_role=point.left_role, right_role=point.right_role,
                    receiver_id=point.receiver_id, zone=point.zone,
                    frequency_hz=point.frequency_hz, present=present,
                    absent_role=point.right_role if point.left is not None else point.left_role,
                    reason_codes=point.reason_codes,
                ))
        if keys != sorted(set(keys)):
            raise ValueError("反射診斷鍵重複或排序錯")
        if tuple(expected) != self.one_sided:
            raise ValueError("單側清單必須剛好等於逐格單側資料")
        return self
