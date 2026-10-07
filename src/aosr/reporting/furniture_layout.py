"""家具第二層驗證：呼叫端明給登記簿的 contact_rel，幾何判斷全部交給第二支。"""
from itertools import combinations

from aosr.geometry.furniture import (
    FurnitureBox, box_within_room, boxes_too_close, contact_margin_m, point_inside_box,
)
from aosr.physics.report_furniture import AbsoluteFurniture
from aosr.reporting.scheme import Scheme, absolute_furniture


def _box(item: AbsoluteFurniture, margin: float) -> FurnitureBox:
    try:
        return FurnitureBox(kind=item.kind, width_m=item.width_m, depth_m=item.depth_m,
                            height_m=item.height_m, bottom_center_m=item.bottom_center_m,
                            yaw_deg=item.yaw_deg, margin_m=margin)
    except ValueError as exc:
        raise ValueError(f"家具 {item.furniture_id}：{exc}") from exc


def furniture_boxes(scheme: Scheme, *, contact_rel: float) -> tuple[FurnitureBox, ...]:
    """底面、房內、件間間隙、喇叭與每席逐一驗；這支尚未接入鏡像法的輸入關。"""
    items = absolute_furniture(scheme)
    if items is None:
        return ()
    room = scheme.scene.room_m
    room_size = (room.Lx, room.Ly, room.Lz)
    margin = contact_margin_m(room_size, contact_rel=contact_rel)
    boxes = tuple(_box(item, margin) for item in items)
    named = tuple(zip(items, boxes, strict=True))
    for item, box in named:
        if not box_within_room(box, room_size, margin_m=margin):
            raise ValueError(f"家具 {item.furniture_id} 超出房間接觸界線")
    for (first, first_box), (second, second_box) in combinations(named, 2):
        if boxes_too_close(first_box, second_box, margin_m=margin):
            raise ValueError(f"家具 {first.furniture_id} 與 {second.furniture_id} 的間隙必須超過接觸界線")
    points = tuple((f"喇叭 {name}", point.as_tuple()) for name, point in scheme.speakers.items()) + tuple(
        (f"座位 {point.receiver_id}", point.position_m) for point in scheme.receiver_set.points)
    for item, box in named:
        for name, position in points:
            if point_inside_box(position, box, margin_m=margin):
                raise ValueError(f"{name} 在家具 {item.furniture_id} 的內部超過接觸界線")
    return boxes
