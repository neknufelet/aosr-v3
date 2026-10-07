"""單一具體擺法的輸入邊界。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from aosr.config.frequency_axis import LowFrequencyAxis
from aosr.geometry.furniture import FurnitureKind
from aosr.geometry.shoebox import Point, Room
from aosr.physics.report_furniture import (
    AbsoluteFurniture, CardinalYaw, FiniteCoordinate, FurnitureRecord, normalized_furniture,
)
from aosr.scoring.channel_group import ChannelGroup
from aosr.scoring.receiver_set import ReceiverSet


SCHEME_SCHEMA_VERSION = "aosr.scheme.v1"
FROZEN = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class ListenerPlacement(BaseModel):
    """水平位移跟主位走；底面離地是絕對高度，相對直角轉向以聆聽者為基準。"""

    model_config = FROZEN
    forward_m: FiniteCoordinate
    left_m: FiniteCoordinate
    bottom_height_m: FiniteCoordinate
    yaw_deg: CardinalYaw


class RoomPlacement(BaseModel):
    """固定天雲的底面中心房間座標與絕對直角轉向。"""

    model_config = FROZEN
    bottom_center_m: tuple[FiniteCoordinate, FiniteCoordinate, FiniteCoordinate]
    yaw_deg: CardinalYaw


class FurnitureSpec(FurnitureRecord):
    """一件家具只收一種完整擺法；接觸界線的驗證留給 furniture_layout。"""

    placement: ListenerPlacement | RoomPlacement

    @model_validator(mode="after")
    def placement_matches_kind(self) -> Self:
        if self.kind == FurnitureKind.CEILING_CLOUD:
            if not isinstance(self.placement, RoomPlacement):
                raise ValueError(f"家具 {self.furniture_id}：天雲必須用房間座標")
        elif not isinstance(self.placement, ListenerPlacement):
            raise ValueError(f"家具 {self.furniture_id}：沙發、座椅、茶几、書桌必須跟著主位")
        return self


class Scene(BaseModel):
    """只收報表輸入的共用場景欄；選填 None 等於未給，交給報表模型補預設。"""

    model_config = FROZEN

    room_m: Room
    sound_speed_m_s: float = Field(gt=0.0)
    density_kg_m3: float = Field(gt=0.0)
    impedance_pa_s_per_m_by_wall: dict[str, float]
    scattering_by_wall: dict[str, float] | None = None
    reflection_order_k: int | None = None
    low_frequency_axis: LowFrequencyAxis | None = None


class Scheme(BaseModel):
    """兩聲道、兩支喇叭與一組座位的已驗擺法。"""

    model_config = FROZEN

    schema_version: Literal["aosr.scheme.v1"]
    scheme_id: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    scene: Scene
    source_model: Literal["product_default", "omnidirectional"]
    speakers: dict[str, Point]
    channel_group: ChannelGroup
    receiver_set: ReceiverSet
    furniture: tuple[FurnitureSpec, ...] | None = None

    @field_validator("furniture")
    @classmethod
    def _normalized_furniture(cls, value: tuple[FurnitureSpec, ...] | None) -> tuple[FurnitureSpec, ...] | None:
        return normalized_furniture(value)

    @model_validator(mode="after")
    def _valid_layout(self) -> Self:
        if not self.scheme_id.strip() or not self.purpose.strip():
            raise ValueError("scheme_id 與 purpose 不可為空白")
        group = self.channel_group
        if len(group.channels) != 2 or len(group.comparisons) != 1:
            raise ValueError("方案必須剛好兩個聲道角色與一個比較對")
        used = {channel.speaker_id for channel in group.channels}
        for speaker_id in used - self.speakers.keys():
            raise ValueError(f"聲道組引用不存在的喇叭 {speaker_id}")
        for speaker_id in self.speakers.keys() - used:
            raise ValueError(f"未使用的喇叭 {speaker_id}")
        for speaker_id, point in self.speakers.items():
            self._inside(point, f"喇叭 {speaker_id}")
        for receiver in self.receiver_set.points:
            self._inside(Point(*receiver.position_m), f"座位 {receiver.receiver_id}")
        return self

    def _inside(self, point: Point, name: str) -> None:
        room = self.scene.room_m
        if not all(0.0 <= value <= limit for value, limit in zip(
            point.as_tuple(), (room.Lx, room.Ly, room.Lz), strict=True,
        )):
            raise ValueError(f"{name} 必須在房間閉區間內")


def scheme_from_document(document: object) -> Scheme:
    """從 JSON 形狀驗成方案。"""
    return Scheme.model_validate(document)


def speaker_pair_ids(project: Scheme) -> tuple[str, str]:
    """左、右聲道各用哪一支喇叭；搜尋與家具換算共用這一份。"""
    roles = {channel.role: channel.speaker_id for channel in project.channel_group.channels}
    if "left" not in roles or "right" not in roles:
        raise ValueError("方案必須有 left 與 right 兩個聲道角色")
    return roles["left"], roles["right"]


def project_midpoint(project: Scheme) -> tuple[float, float]:
    """兩支喇叭的水平中點。"""
    left_id, right_id = speaker_pair_ids(project)
    left, right = project.speakers[left_id], project.speakers[right_id]
    return (left.x + right.x) / 2.0, (left.y + right.y) / 2.0


def project_facing(project: Scheme) -> tuple[float, float]:
    """聆聽者面向＝兩喇叭中點相對主位、水平分量絕對值較大的那一軸（搜尋與家具換算共用）。

    專案方案只拿來定周圍座位與家具要轉幾個直角，不要求座位剛好在中軸上（客戶現況可能不對稱）。
    兩軸一樣大（含都為零）定不出前牆，拒收；比的是大小不是相等，不設容差。
    前方為 f、左方為 (−f_y, f_x)，即面向逆時針轉 90 度。
    """
    mx, my = project_midpoint(project)
    px, py, _ = project.receiver_set.primary.position_m
    dx, dy = mx - px, my - py
    if abs(dx) > abs(dy):
        return (1.0 if dx > 0.0 else -1.0), 0.0
    if abs(dy) > abs(dx):
        return 0.0, (1.0 if dy > 0.0 else -1.0)
    raise ValueError("專案方案的兩喇叭中點相對主位沒有主要方向（x、y 一樣大），定不出前牆")


def absolute_furniture(scheme: Scheme) -> tuple[AbsoluteFurniture, ...] | None:
    """一份換算規則，所有耳朵採樣點共用主位換算後的場景。

    底面中心＝(主位 x + forward·f_x − left·f_y, 主位 y + forward·f_y + left·f_x, 底面離地)。
    面向基準角：(0,1)→0、(−1,0)→90、(0,−1)→180、(1,0)→270。
    絕對角＝(基準角＋相對角) mod 360；相對 0 度時寬沿聆聽者左右、深沿前後。
    天雲原樣留在房間座標；全部依 furniture_id 排序。
    """
    if scheme.furniture is None:
        return None
    facing = project_facing(scheme) if any(isinstance(item.placement, ListenerPlacement)
                                         for item in scheme.furniture) else None
    px, py, _ = scheme.receiver_set.primary.position_m
    absolute = []
    for item in scheme.furniture:
        placement = item.placement
        if isinstance(placement, ListenerPlacement):
            if facing is None:
                raise ValueError("跟著主位的家具缺少面向")
            fx, fy = facing
            bottom = (px + placement.forward_m * fx - placement.left_m * fy,
                      py + placement.forward_m * fy + placement.left_m * fx, placement.bottom_height_m)
            base = {(0.0, 1.0): 0, (-1.0, 0.0): 90, (0.0, -1.0): 180, (1.0, 0.0): 270}[facing]
            yaw = (base + placement.yaw_deg) % 360
        else:
            bottom, yaw = placement.bottom_center_m, placement.yaw_deg
        absolute.append(AbsoluteFurniture.model_validate(item.model_dump(exclude={"placement"}) |
                                                         {"bottom_center_m": bottom, "yaw_deg": yaw}))
    return normalized_furniture(tuple(absolute))


def expected_pairs(scheme: Scheme) -> tuple[tuple[str, str, str], ...]:
    """依宣告順序列出每支喇叭與每個座位的鍵及角色。"""
    return tuple((channel.speaker_id, receiver.receiver_id, channel.role)
                 for channel in scheme.channel_group.channels
                 for receiver in scheme.receiver_set.points)


def pair_input_document(scheme: Scheme, source: Point, receiver: Point,
                        source_model: object) -> dict[str, object]:
    """一對喇叭與座位的報表輸入文件；值是 None 的選填場景欄不放進去，交給報表模型的預設。

    管線組文件與結果檔的自我一致核對都用這一支，規則只有一份。
    """
    document = scheme.scene.model_dump(mode="json", exclude_none=True)
    document["source_m"] = {"x": source.x, "y": source.y, "z": source.z}
    document["receiver_m"] = {"x": receiver.x, "y": receiver.y, "z": receiver.z}
    document["source_model"] = source_model
    furniture = absolute_furniture(scheme)
    if furniture is not None:
        document["furniture"] = [item.model_dump(mode="json") for item in furniture]
    return document


def load_scheme(path: Path) -> Scheme:
    """由呼叫端指定的路徑讀取方案。"""
    with path.open(encoding="utf-8") as handle:
        return scheme_from_document(json.load(handle))
