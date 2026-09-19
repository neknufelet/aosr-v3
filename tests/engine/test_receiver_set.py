"""聆聽區接收點清單的結構、產生器與內容指紋考卷（票 #349）。"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from aosr.scoring.receiver_set import (
    ReceiverPoint,
    ReceiverRole,
    ReceiverSet,
    primary_cross,
    rectangular_grid,
)


def _point(
    receiver_id: str,
    role: ReceiverRole,
    *,
    position_m: tuple[float, float, float],
    importance: float = 1.0,
    direction: str | None = None,
) -> ReceiverPoint:
    return ReceiverPoint(
        receiver_id=receiver_id,
        position_m=position_m,
        role=role,
        importance=importance,
        direction_relative_to_primary=direction,
    )


def _valid_points() -> tuple[ReceiverPoint, ...]:
    return (
        _point("main", ReceiverRole.PRIMARY, position_m=(1.0, 2.0, 1.2)),
        _point(
            "front",
            ReceiverRole.SURROUNDING,
            position_m=(1.1, 2.0, 1.2),
            direction="front",
        ),
    )


@pytest.mark.parametrize("violation", ("duplicate", "missing_primary", "two_primaries", "no_direction"))
def test_receiver_set_rejects_each_declared_structural_violation(violation: str) -> None:
    """代號重複、沒有或多個主位、以及周圍點漏方向都不能成為可彙總清單。"""
    main, front = _valid_points()
    points: tuple[ReceiverPoint, ...]
    if violation == "duplicate":
        points = (main, front.model_copy(update={"receiver_id": "main"}))
    elif violation == "missing_primary":
        points = (front, front.model_copy(update={"receiver_id": "back", "direction_relative_to_primary": "back"}))
    elif violation == "two_primaries":
        points = (main, main.model_copy(update={"receiver_id": "other-main"}), front)
    else:
        points = (main, front.model_copy(update={"direction_relative_to_primary": None}))

    with pytest.raises(ValidationError):
        ReceiverSet(points=points)


def test_fingerprint_ignores_order_but_tracks_coordinate_and_importance() -> None:
    """內容相同的重排必須同指紋，而任一座標或重要性變更都必須改變指紋。"""
    main, front = _valid_points()
    original = ReceiverSet(points=(main, front))
    reordered = ReceiverSet(points=(front, main))
    moved = ReceiverSet(points=(main, front.model_copy(update={"position_m": (1.2, 2.0, 1.2)})))
    reweighted = ReceiverSet(points=(main, front.model_copy(update={"importance": 2.0})))

    assert reordered.fingerprint == original.fingerprint
    assert moved.fingerprint != original.fingerprint
    assert reweighted.fingerprint != original.fingerprint


def test_primary_cross_emits_an_explicit_receiver_set() -> None:
    """主位十字只能產生一般接收點清單，且六個方向都以資料而非特殊型別保留。"""
    receivers = primary_cross(primary_position_m=(1.0, 2.0, 1.2), offset_m=0.1)

    surrounding_directions = {
        point.direction_relative_to_primary
        for point in receivers.points
        if point.role == ReceiverRole.SURROUNDING
    }
    assert surrounding_directions == {"front", "back", "left", "right", "up", "down"}
    assert {point.receiver_id for point in receivers.points if point.role == ReceiverRole.PRIMARY} == {
        "main"
    }


def test_rectangular_grid_emits_the_same_receiver_set_type() -> None:
    """矩形格點也只能產生一般清單，原點是主位且近端格點都有相對方向。"""
    receivers = rectangular_grid(
        primary_position_m=(1.0, 2.0, 1.2),
        x_offsets_m=(-0.1, 0.0, 0.1),
        y_offsets_m=(-0.1, 0.0, 0.1),
        surrounding_radius_m=0.15,
    )

    assert isinstance(receivers, ReceiverSet)
    assert {point.receiver_id for point in receivers.points if point.role == ReceiverRole.PRIMARY} == {
        "main"
    }
    assert all(
        point.direction_relative_to_primary
        for point in receivers.points
        if point.role == ReceiverRole.SURROUNDING
    )


def test_rectangular_grid_radius_changes_roles_for_the_same_points() -> None:
    """周圍半徑若沒參與分類，整房格點會全部冒充主位附近的量測點。"""
    narrow = rectangular_grid(
        primary_position_m=(1.0, 2.0, 1.2),
        x_offsets_m=(0.0, 0.1, 0.3),
        y_offsets_m=(0.0,),
        surrounding_radius_m=0.15,
    )
    wide = rectangular_grid(
        primary_position_m=(1.0, 2.0, 1.2),
        x_offsets_m=(0.0, 0.1, 0.3),
        y_offsets_m=(0.0,),
        surrounding_radius_m=0.35,
    )

    assert {point.receiver_id: point.role for point in narrow.points} == {
        "main": ReceiverRole.PRIMARY,
        "grid-0.1-0": ReceiverRole.SURROUNDING,
        "grid-0.3-0": ReceiverRole.OTHER_SEAT,
    }
    assert {point.receiver_id: point.role for point in wide.points} == {
        "main": ReceiverRole.PRIMARY,
        "grid-0.1-0": ReceiverRole.SURROUNDING,
        "grid-0.3-0": ReceiverRole.SURROUNDING,
    }
