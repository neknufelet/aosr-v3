"""不載入評估器的聲道角色、比較對與共同指紋。"""
from __future__ import annotations

import hashlib
import json
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


FROZEN = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class ChannelDefinition(BaseModel):
    """聲道組裡一個角色與實際喇叭身分；只有比較設定明列的角色對會被比較。"""

    model_config = FROZEN

    role: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    speaker_id: str = Field(min_length=1)


class ChannelComparison(BaseModel):
    """設定明列的一個有序角色對；評估器用它尋找同一接收點的兩邊輸入。"""

    model_config = FROZEN

    left_role: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    right_role: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")

    @model_validator(mode="after")
    def _roles_are_distinct(self) -> Self:
        if self.left_role == self.right_role:
            raise ValueError("比較對的兩個角色不可相同")
        return self


class ChannelGroup(BaseModel):
    """可擴充聲道清單與明列比較設定；單聲道可不宣告比較對。

    內容正規化後形成共同指紋。
    """

    model_config = FROZEN

    channels: tuple[ChannelDefinition, ...] = Field(min_length=1)
    comparisons: tuple[ChannelComparison, ...]
    feature_match_tolerance_hz: Annotated[float, Field(ge=0.0)]

    @model_validator(mode="after")
    def _roles_and_pairs_are_unambiguous(self) -> Self:
        roles = [item.role for item in self.channels]
        speakers = [item.speaker_id for item in self.channels]
        if len(roles) != len(set(roles)) or len(speakers) != len(set(speakers)):
            raise ValueError("聲道角色與 speaker_id 都不可重複")
        if not self.comparisons and len(self.channels) != 1:
            raise ValueError("沒有比較對的聲道組只准有一個聲道")
        pairs = [(item.left_role, item.right_role) for item in self.comparisons]
        if len(pairs) != len(set(pairs)):
            raise ValueError("比較對不可重複")
        if any(left not in roles or right not in roles for left, right in pairs):
            raise ValueError("比較對必須引用聲道組內角色")
        return self

    @property
    def fingerprint(self) -> str:
        channels = sorted(
            (item.model_dump(mode="json") for item in self.channels),
            key=lambda item: str(item["role"]),
        )
        comparisons = sorted(
            (item.model_dump(mode="json") for item in self.comparisons),
            key=lambda item: (str(item["left_role"]), str(item["right_role"])),
        )
        canonical = json.dumps(
            {
                "channels": channels,
                "comparisons": comparisons,
                "feature_match_tolerance_hz": self.feature_match_tolerance_hz,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
