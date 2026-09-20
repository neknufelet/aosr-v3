"""計分層的明列接收點清單、內容指紋與兩種清單產生器（票 #349）。"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from enum import StrEnum
from typing import Annotated, Final, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


FROZEN = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
Position = tuple[float, float, float]
LAYOUT_DISPLACEMENT_DECIMAL_PLACES: Final[int] = 9


def _rounded_displacement(value: float) -> float:
    rounded = round(value, LAYOUT_DISPLACEMENT_DECIMAL_PLACES)
    return 0.0 if rounded == 0.0 else rounded


class ReceiverRole(StrEnum):
    """接收點在聆聽區裡扮演的受控角色。"""

    PRIMARY = "primary"
    SURROUNDING = "surrounding"
    OTHER_SEAT = "other_seat"


class ReceiverPoint(BaseModel):
    """一個計分用接收點；物理求解仍由物理層逐點執行。"""

    model_config = FROZEN

    receiver_id: str = Field(min_length=1)
    position_m: Position
    role: ReceiverRole
    importance: Annotated[float, Field(ge=0.0)]
    direction_relative_to_primary: str | None = None

    @model_validator(mode="after")
    def _identity_and_direction_are_meaningful(self) -> Self:
        if not self.receiver_id.strip():
            raise ValueError("receiver_id 不可為空白")
        if self.role == ReceiverRole.SURROUNDING:
            direction = self.direction_relative_to_primary
            if direction is None or not direction.strip():
                raise ValueError("surrounding 接收點必須帶相對主位方向")
        return self


class ReceiverSet(BaseModel):
    """可重排但不可含糊的一份明列接收點清單。"""

    model_config = FROZEN

    points: tuple[ReceiverPoint, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def _set_structure_is_unambiguous(self) -> Self:
        ids = [point.receiver_id for point in self.points]
        if len(ids) != len(set(ids)):
            raise ValueError("receiver_id 不可重複")
        primaries = [point for point in self.points if point.role == ReceiverRole.PRIMARY]
        if len(primaries) != 1:
            raise ValueError("接收點清單必須剛好有一個 primary")
        if not any(point.role == ReceiverRole.SURROUNDING for point in self.points):
            raise ValueError("接收點清單至少要有主位加一個 surrounding")
        return self

    @property
    def primary(self) -> ReceiverPoint:
        """回傳已由結構驗證保證唯一的主位。"""
        return next(point for point in self.points if point.role == ReceiverRole.PRIMARY)

    @property
    def fingerprint(self) -> str:
        """正規化點序後 JSON 內容的 SHA-256 十六進位指紋。"""
        normalized = sorted(
            (point.model_dump(mode="json") for point in self.points),
            key=lambda point: str(point["receiver_id"]),
        )
        canonical = json.dumps(
            {"points": normalized}, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @property
    def layout_fingerprint(self) -> str:
        """不含絕對座標、位移先正規化的相對佈局 SHA-256 指紋。"""
        primary_position = self.primary.position_m
        normalized = sorted(
            (
                {
                    "receiver_id": point.receiver_id,
                    "role": point.role.value,
                    "importance": point.importance,
                    "direction_relative_to_primary": point.direction_relative_to_primary,
                    "displacement_from_primary_m": tuple(
                        _rounded_displacement(coordinate - primary_coordinate)
                        for coordinate, primary_coordinate in zip(
                            point.position_m, primary_position, strict=True
                        )
                    ),
                }
                for point in self.points
            ),
            key=lambda point: str(point["receiver_id"]),
        )
        canonical = json.dumps(
            {"points": normalized}, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def primary_cross(
    *,
    primary_position_m: Position,
    offset_m: Annotated[float, Field(gt=0.0)],
    primary_importance: Annotated[float, Field(ge=0.0)] = 1.0,
    surrounding_importance: Annotated[float, Field(ge=0.0)] = 1.0,
) -> ReceiverSet:
    """產生主位與前後左右上下六點的普通清單；偏移量完全由呼叫端給。"""
    x, y, z = primary_position_m
    offsets = (
        ("front", (offset_m, 0.0, 0.0)),
        ("back", (-offset_m, 0.0, 0.0)),
        ("left", (0.0, offset_m, 0.0)),
        ("right", (0.0, -offset_m, 0.0)),
        ("up", (0.0, 0.0, offset_m)),
        ("down", (0.0, 0.0, -offset_m)),
    )
    points = [
        ReceiverPoint(
            receiver_id="main",
            position_m=primary_position_m,
            role=ReceiverRole.PRIMARY,
            importance=primary_importance,
        )
    ]
    points.extend(
        ReceiverPoint(
            receiver_id=direction,
            position_m=(x + dx, y + dy, z + dz),
            role=ReceiverRole.SURROUNDING,
            importance=surrounding_importance,
            direction_relative_to_primary=direction,
        )
        for direction, (dx, dy, dz) in offsets
    )
    return ReceiverSet(points=tuple(points))


def _axis_direction(offset: float, positive: str, negative: str) -> str | None:
    if offset > 0.0:
        return positive
    if offset < 0.0:
        return negative
    return None


def rectangular_grid(
    *,
    primary_position_m: Position,
    x_offsets_m: Sequence[float],
    y_offsets_m: Sequence[float],
    surrounding_radius_m: Annotated[float, Field(ge=0.0)],
    primary_importance: Annotated[float, Field(ge=0.0)] = 1.0,
    surrounding_importance: Annotated[float, Field(ge=0.0)] = 1.0,
) -> ReceiverSet:
    """以主位為原點產生格點，明列半徑內的周圍點與半徑外的其他座位。"""
    if not x_offsets_m or not y_offsets_m:
        raise ValueError("矩形格點的兩軸偏移量都不可為空")
    if 0.0 not in x_offsets_m or 0.0 not in y_offsets_m:
        raise ValueError("矩形格點的兩軸偏移量都必須包含主位原點")
    if not math.isfinite(surrounding_radius_m) or surrounding_radius_m < 0.0:
        raise ValueError("surrounding_radius_m 必須是有限非負數")
    x, y, z = primary_position_m
    points: list[ReceiverPoint] = []
    for dx in sorted(set(x_offsets_m)):
        for dy in sorted(set(y_offsets_m)):
            if dx == 0.0 and dy == 0.0:
                points.append(
                    ReceiverPoint(
                        receiver_id="main",
                        position_m=primary_position_m,
                        role=ReceiverRole.PRIMARY,
                        importance=primary_importance,
                    )
                )
                continue
            directions = (
                _axis_direction(dx, "front", "back"),
                _axis_direction(dy, "left", "right"),
            )
            direction = "-".join(part for part in directions if part is not None)
            role = (
                ReceiverRole.SURROUNDING
                if math.hypot(dx, dy) <= surrounding_radius_m
                else ReceiverRole.OTHER_SEAT
            )
            points.append(
                ReceiverPoint(
                    receiver_id=f"grid-{dx:g}-{dy:g}",
                    position_m=(x + dx, y + dy, z),
                    role=role,
                    importance=surrounding_importance,
                    direction_relative_to_primary=(
                        direction if role == ReceiverRole.SURROUNDING else None
                    ),
                )
            )
    return ReceiverSet(points=tuple(points))
