"""單一具體擺法的輸入邊界。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from aosr.config.frequency_axis import LowFrequencyAxis
from aosr.geometry.shoebox import Point, Room
from aosr.scoring.channel_group import ChannelGroup
from aosr.scoring.receiver_set import ReceiverSet


SCHEME_SCHEMA_VERSION = "aosr.scheme.v1"
FROZEN = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


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
    return document


def load_scheme(path: Path) -> Scheme:
    """由呼叫端指定的路徑讀取方案。"""
    with path.open(encoding="utf-8") as handle:
        return scheme_from_document(json.load(handle))
