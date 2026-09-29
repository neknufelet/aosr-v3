"""方案的平面圖、側面圖與聆聽區圖面資料。"""
from __future__ import annotations

from typing import cast

from aosr.config.directivity_defaults import DirectivityDefaults
from aosr.geometry.shoebox import Point
from aosr.gui.labels import DIRECTIONS
from aosr.physics.report_source import default_source_model
from aosr.reporting.scheme import Scheme

SPEAKER_MARKERS = {"left": "L", "right": "R"}
SPEAKER_ROLES = {"left": "左聲道", "right": "右聲道"}


def _point_detail(name: str, description: str, point: dict[str, float]) -> str:
    return (f"{name}（{description}）：x {point['x']:.2f}、y {point['y']:.2f}、"
            f"z {point['z']:.2f} 公尺")


def _plan_views(speakers: list[dict[str, object]], receivers: list[dict[str, object]],
                vertical: bool, zoomed: bool = False) -> list[dict[str, object]]:
    groups: dict[tuple[float, float], list[tuple[str, dict[str, object]]]] = {}
    included = [item for item in receivers if item.get("role") in {"primary", "surrounding"}]
    for kind, items in (("speaker", [] if zoomed else speakers),
                        ("receiver", included if zoomed else receivers)):
        for item in items:
            point = cast(dict[str, float], item["point"])
            u, v = point["x"], point["z" if vertical else "y"]
            groups.setdefault((u, v), []).append((kind, item))
    return [{"keys": [str(item["key"]) for _, item in group],
             "kind": group[0][0] if all(kind == group[0][0] for kind, _ in group) else "mixed",
             "marker": "／".join(str(item["marker"]) for _, item in group),
             # 整間房的圖：喇叭與主位印短標記；其他座位只畫圓點不印字（座位一多字會擠出畫面），
             # 名字交給圖例、滑過與點選明細；周圍點在聆聽區虛線框裡、放大圖才畫。
             "drawn": zoomed or any(kind == "speaker" or item.get("role") in {"primary", "other_seat"}
                                    for kind, item in group),
             "caption": "／".join(str(item["marker"]) for kind, item in group
                                  if zoomed or kind == "speaker" or item.get("role") == "primary"),
             "detail_lines": [str(item["detail_text"]) for _, item in group],
             "u": u, "v": v}
            for (u, v), group in groups.items()]


def _listening_zoom(scheme: Scheme, vertical: bool) -> dict[str, list[float]]:
    primary = scheme.receiver_set.primary.position_m
    points = [point.position_m for point in scheme.receiver_set.points
              if point.role.value in {"primary", "surrounding"}]
    axis = 2 if vertical else 1
    du = max(abs(point[0] - primary[0]) for point in points) + 0.15
    dv = max(abs(point[axis] - primary[axis]) for point in points) + 0.15
    return {"u": [primary[0] - du, primary[0] + du],
            "v": [primary[axis] - dv, primary[axis] + dv]}


def plan_for(scheme: Scheme, directivity: DirectivityDefaults) -> dict[str, object]:
    primary = Point(*scheme.receiver_set.primary.position_m)
    roles = {channel.speaker_id: channel.role for channel in scheme.channel_group.channels}
    aim = (default_source_model(primary, directivity).model_dump(mode="json")["aim_m"]
           if scheme.source_model == "product_default" else None)
    room = scheme.scene.room_m
    speakers: list[dict[str, object]] = []
    for speaker_id, point in scheme.speakers.items():
        position = {"x": point.x, "y": point.y, "z": point.z}
        role = roles[speaker_id]
        role_name = SPEAKER_ROLES.get(role, f"聲道 {role}")
        speakers.append({"id": speaker_id, "key": f"speaker:{speaker_id}",
                         "role": role, "role_label": role_name,
                         "point": position, "aim": aim,
                         "marker": SPEAKER_MARKERS.get(role, role),
                         "detail_text": _point_detail(
                             speaker_id, f"{role_name}喇叭" if role in SPEAKER_ROLES else f"{role_name} 喇叭",
                             position)})
    receivers: list[dict[str, object]] = []
    seat_number = 0
    for receiver in scheme.receiver_set.points:
        role = receiver.role.value
        position = dict(zip(("x", "y", "z"), receiver.position_m, strict=True))
        direction = DIRECTIONS.get(receiver.direction_relative_to_primary or "")
        if role == "primary":
            marker, description = "主", "主位"
        elif role == "surrounding":
            marker = direction[0] if direction else receiver.receiver_id
            description = f"周圍點，{direction[1]}" if direction else "周圍點，方向未標示"
        else:
            seat_number += 1
            marker, description = f"座{seat_number}", "其他座位"
        receivers.append({"id": receiver.receiver_id,
                          "key": f"receiver:{receiver.receiver_id}", "role": role,
                          "role_label": {"primary": "主位", "surrounding": "周圍點",
                                         "other_seat": "其他座位"}[role],
                          "point": position, "marker": marker,
                          "detail_text": _point_detail(receiver.receiver_id, description, position)})
    return {
        "room": {"Lx": room.Lx, "Ly": room.Ly, "Lz": room.Lz},
        "speakers": speakers, "receivers": receivers,
        "views": {"plan": _plan_views(speakers, receivers, False),
                  "side": _plan_views(speakers, receivers, True),
                  "zoom_plan": _plan_views(speakers, receivers, False, True),
                  "zoom_side": _plan_views(speakers, receivers, True, True)},
        "listening_zoom": {"plan": _listening_zoom(scheme, False),
                           "side": _listening_zoom(scheme, True)},
    }

