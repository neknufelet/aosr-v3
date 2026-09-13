"""多接收點的資料結構；幾何與振幅公式仍留在原本模組。"""
from __future__ import annotations

from dataclasses import dataclass
from string import ascii_letters, digits
from typing import TYPE_CHECKING, Final

from aosr.geometry.shoebox import Point
from aosr.physics.totals import Totals, totals_from_paths

if TYPE_CHECKING:
    from aosr.physics.room_paths import RoomInput, RoomPath

MAX_RECEIVER_ID_LENGTH: Final[int] = 32
RECEIVER_ID_CHARACTERS: Final[frozenset[str]] = frozenset(ascii_letters + digits + "_-")


@dataclass(frozen=True)
class Receiver:
    """一個具名接收點，順序就是輸入清單的順序。"""

    id: str
    point: Point


@dataclass(frozen=True)
class ReceiverResult:
    """一個接收點算完的路徑與總量；無材料時沒有總量。"""

    paths: list[RoomPath]
    totals: Totals | None


def validate_receiver_id(value: object, where: str) -> str:
    """接收點 id 是 1–32 個 ASCII 字母、數字、底線或連字號。"""
    valid = (
        isinstance(value, str)
        and bool(value)
        and len(value) <= MAX_RECEIVER_ID_LENGTH
        and all(character in RECEIVER_ID_CHARACTERS for character in value)
    )
    if not valid:
        raise ValueError(f"{where} 必須是 1 到 32 個 ASCII 字母、數字、_ 或 -：{value!r}")
    assert isinstance(value, str)
    return value


def solve_receivers(inputs: RoomInput) -> dict[str, ReceiverResult]:
    """依輸入順序逐接收點求解；每一點各呼叫一次既有單點求解器。"""
    from aosr.physics.room_paths import image_source_paths

    receivers = inputs.receivers or (Receiver(id="R0", point=inputs.receiver),)
    results: dict[str, ReceiverResult] = {}
    for receiver in receivers:
        try:
            paths = image_source_paths(
                inputs.room,
                inputs.source,
                receiver.point,
                inputs.sound_speed,
                inputs.max_order,
                inputs.materials,
            )
            computed_totals = totals_from_paths(paths) if inputs.materials is not None else None
        except ValueError as exc:
            raise ValueError(f"接收點 {receiver.id}: {exc}") from exc
        results[receiver.id] = ReceiverResult(paths=paths, totals=computed_totals)
    return results
