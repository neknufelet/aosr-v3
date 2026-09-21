"""分類評估共用的喇叭與接收點擺位契約。"""
from __future__ import annotations

from collections.abc import Iterable
from struct import pack
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


Identifier = Annotated[str, Field(min_length=1)]
CoordinateM = tuple[float, float, float]
PlacementRow = tuple[Identifier, CoordinateM]


class Placement(BaseModel):
    """兩張凍結 tuple 表；一個代號在同一張表只能出現一次。"""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    speaker_positions_m: tuple[PlacementRow, ...]
    receiver_positions_m: tuple[PlacementRow, ...]

    @model_validator(mode="after")
    def _ids_are_unique(self) -> Self:
        for name, rows in (
            ("speaker_positions_m", self.speaker_positions_m),
            ("receiver_positions_m", self.receiver_positions_m),
        ):
            ids = [identifier for identifier, _ in rows]
            repeated = next(
                (identifier for identifier in ids if ids.count(identifier) > 1),
                None,
            )
            if repeated is not None:
                raise ValueError(f"{name} 重複代號 {repeated}")
        return self


class PlacementMismatchError(ValueError):
    """同一類代號被不同 IEEE-754 座標位元表示指到兩處。"""

    def __init__(self, kind: str, identifier: str) -> None:
        super().__init__(f"{kind} {identifier} 指到不同座標")


def empty_placement() -> Placement:
    """不可估且無一致座標可留時使用的明確空擺位。"""
    return Placement(speaker_positions_m=(), receiver_positions_m=())


def point_placement(
    speaker_id: str,
    source_position_m: CoordinateM,
    receiver_id: str,
    receiver_position_m: CoordinateM,
) -> Placement:
    """建立一份單喇叭、單接收點報表的擺位。"""
    return Placement(
        speaker_positions_m=((speaker_id, source_position_m),),
        receiver_positions_m=((receiver_id, receiver_position_m),),
    )


def _coordinate_bits(coordinate: CoordinateM) -> bytes:
    return pack("!ddd", *coordinate)


def _merge_rows(
    tables: Iterable[tuple[PlacementRow, ...]], kind: str
) -> tuple[PlacementRow, ...]:
    by_id: dict[str, CoordinateM] = {}
    for rows in tables:
        for identifier, coordinate in rows:
            previous = by_id.get(identifier)
            if previous is not None and _coordinate_bits(previous) != _coordinate_bits(
                coordinate
            ):
                raise PlacementMismatchError(kind, identifier)
            by_id[identifier] = coordinate
    return tuple(sorted(by_id.items()))


def merge_placements(placements: Iterable[Placement]) -> Placement:
    """逐位元核對同代號並以代號排序，輸入次序不影響合併結果。"""
    collected = tuple(placements)
    return Placement(
        speaker_positions_m=_merge_rows(
            (item.speaker_positions_m for item in collected), "speaker_id"
        ),
        receiver_positions_m=_merge_rows(
            (item.receiver_positions_m for item in collected), "receiver_id"
        ),
    )


def merge_or_empty(placements: Iterable[Placement]) -> Placement:
    """彙總失去唯一座標時不挑一筆冒充，只留下明確空擺位。"""
    try:
        return merge_placements(placements)
    except PlacementMismatchError:
        return empty_placement()
