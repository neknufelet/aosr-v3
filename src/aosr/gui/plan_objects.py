"""圖面的絕對盒子與箱體；沿用方案換算與搜尋的水平朝向。"""
from __future__ import annotations

from typing import cast

from aosr.config.precision_contracts import default_precision_contracts_path, furniture_contact_rel
from aosr.geometry.furniture import contact_margin_m
from aosr.geometry.shoebox import Point
from aosr.physics.furniture_paths import Furniture
from aosr.physics.report_furniture import AbsoluteFurniture
from aosr.reporting.display import furniture_name
from aosr.reporting.furniture_layout import FurnitureLayout, _box, direct_blockers
from aosr.reporting.scheme import ListenerPlacement, Scheme, absolute_furniture
from aosr.search.constraints import _cabinet
from aosr.search.labels import SPEAKER_SETUP
from aosr.search.layout_settings import Cabinet


def _side(polygon: tuple[tuple[float, float], ...], bottom: float, top: float
          ) -> tuple[tuple[float, float], ...]:
    low, high = min(x for x, _ in polygon), max(x for x, _ in polygon)
    return (low, bottom), (high, bottom), (high, top), (low, top)


def _footprint(item: AbsoluteFurniture) -> tuple[tuple[float, float], ...]:
    x, y, _ = item.bottom_center_m
    width, depth = ((item.depth_m, item.width_m) if item.yaw_deg in (90, 270)
                    else (item.width_m, item.depth_m))
    return ((x - width / 2, y - depth / 2), (x + width / 2, y - depth / 2),
            (x + width / 2, y + depth / 2), (x - width / 2, y + depth / 2))


def _furniture(item: AbsoluteFurniture, scheme: Scheme) -> dict[str, object]:
    x, y, bottom = item.bottom_center_m
    polygon = _footprint(item)
    top = bottom + item.height_m
    original = next(spec for spec in scheme.furniture or () if spec.furniture_id == item.furniture_id)
    placement = original.placement
    relation = (f"跟著主位，前方 {placement.forward_m:.2f}、左方 {placement.left_m:.2f} 公尺、"
                f"相對角 {placement.yaw_deg:g} 度" if isinstance(placement, ListenerPlacement)
                else "固定在房間座標")
    detail = (f"{furniture_name(item.kind, item.furniture_id)}：{relation}；"
              f"底面中心 x {x:.2f}、y {y:.2f}、z {bottom:.2f} 公尺、房間角度 {item.yaw_deg:g} 度；"
              f"寬 {item.width_m:.2f} × 深 {item.depth_m:.2f} × 高 {item.height_m:.2f} 公尺；"
              f"底面 {bottom:.2f}、頂面 {top:.2f} 公尺")
    return {"id": item.furniture_id, "key": f"furniture:{item.furniture_id}", "kind": item.kind,
            "polygon": polygon, "side_polygon": _side(polygon, bottom, top),
            "bottom_m": bottom, "top_m": top, "detail_text": detail, "blocked": False}


def _blocked_paths(scheme: Scheme, items: tuple[AbsoluteFurniture, ...]) -> list[dict[str, object]]:
    room = scheme.scene.room_m
    margin = contact_margin_m((room.Lx, room.Ly, room.Lz),
                             contact_rel=furniture_contact_rel(default_precision_contracts_path()))
    try:
        # 不驗房內、家具間隙；擺放錯仍有可畫的盒子。端點在實體內等無法判定的情形交回問題清單。
        layout = FurnitureLayout(tuple(Furniture(item.furniture_id, _box(item, margin)) for item in items), margin)
        blockers = direct_blockers(scheme, layout)
    except ValueError:
        return []
    receivers = {item.receiver_id: item.position_m for item in scheme.receiver_set.points}
    return [{"speaker_id": speaker, "receiver_id": receiver, "furniture_ids": ids,
             "start_m": scheme.speakers[speaker].as_tuple(), "end_m": receivers[receiver]}
            for (speaker, receiver), ids in blockers.items() if ids]


def _cabinets(scheme: Scheme, items: tuple[AbsoluteFurniture, ...]) -> list[dict[str, object]]:
    setup = scheme.speaker_setup
    if setup is None:
        return []
    cabinet = Cabinet.model_validate(setup.cabinet.model_dump())
    tables = [item for item in items if item.kind in {"desk", "coffee_table"}]
    if setup.mount == "desk" and len(tables) != 1:
        raise ValueError("放桌面時定不出承托桌桌面頂")
    primary = Point(*scheme.receiver_set.primary.position_m)
    boxes = []
    for name, point in scheme.speakers.items():
        prism = _cabinet(point, primary, cabinet)
        bottom = (tables[0].bottom_center_m[2] + tables[0].height_m if setup.mount == "desk"
                  else 0.0 if setup.mount == "floor" else prism.low_z)
        top = bottom + cabinet.height_m
        model = SPEAKER_SETUP["representative_model" if setup.representative else "actual_model"]
        detail = (f"{SPEAKER_SETUP['cabinet']}（{name}）：{SPEAKER_SETUP[setup.kind]}、"
                  f"放{SPEAKER_SETUP[setup.mount]}；{model}；"
                  f"寬 {cabinet.width_m:.2f} × 深 {cabinet.depth_m:.2f} × 高 {cabinet.height_m:.3f} 公尺；"
                  f"底面 {bottom:.2f}、頂面 {top:.2f} 公尺；箱體只做碰撞檢查，反射暫不計")
        boxes.append({"id": name, "key": f"cabinet:{name}", "kind": "cabinet", "polygon": prism.polygon,
                      "side_polygon": _side(prism.polygon, bottom, top), "bottom_m": bottom,
                      "top_m": top, "detail_text": detail, "blocked": False})
    return boxes


def plan_objects(scheme: Scheme, *, mark_blockers: bool = False) -> dict[str, object]:
    """舊方案不加任何鍵；家具換算不了由輸入端點回問題，箱體失敗單獨降級。"""
    if not scheme.furniture and scheme.speaker_setup is None:
        return {}
    items = absolute_furniture(scheme) or ()
    furniture = [_furniture(item, scheme) for item in items]
    paths = _blocked_paths(scheme, items) if mark_blockers and items else []
    blocked = {name for path in paths for name in cast(tuple[str, ...], path["furniture_ids"])}
    for item in furniture:
        item["blocked"] = item["id"] in blocked
    notes = []
    try:
        cabinets = _cabinets(scheme, items)
    except ValueError as exc:
        cabinets = []
        notes.append(f"喇叭箱體讀不到：{exc}")
    return {"furniture": furniture, "cabinets": cabinets, "blocked_paths": paths, "drawing_notes": notes}


def changed_furniture_keys(a: Scheme, b: Scheme) -> tuple[str, ...]:
    """只比畫出的絕對盒子；材質與相對欄位本身不影響邊線記號。"""
    def boxes(scheme: Scheme) -> dict[str, tuple[object, ...]]:
        return {item.furniture_id: (_footprint(item), item.bottom_center_m[2], item.height_m)
                for item in absolute_furniture(scheme) or ()}
    first, second = boxes(a), boxes(b)
    return tuple(f"furniture:{name}" for name in sorted(first.keys() | second.keys())
                 if first.get(name) != second.get(name))
