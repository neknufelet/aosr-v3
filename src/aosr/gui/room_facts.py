"""輸入頁第 1 段右卡「這間房現在」的房間、喇叭、主位、家具四句（#1265）：伺服器排字、畫面照印，只排字、不改存檔。

- 前牆與左右只照 ``wall_words``（同一份規則）；定不出前牆就寫座標，不猜哪一面是前牆。
- 喇叭、主位兩句只看房間、喇叭、聲道、座位這四格（``draft_layout``，跟牆名同一份）：別格填錯照樣排得出來，
  這四格本身驗不過回 None（畫面寫「還沒填完」）。房間那一句只要長寬高三格是正數。
- 家具那一句只讀家具清單的種類，名字跟名稱表同一份。
- 房間那一句的寬、深照前牆分，每個數後面寫是長寬高哪一格（欄名是固定軸）。
- 數字照首頁那一句的寫法（``room_words``）：到公分、去掉尾巴的 0；「一樣」「在中線上」都比寫出來的字，不另設容差。
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Final, cast

from aosr.gui.problem_text import speaker_name
from aosr.gui.wall_words import FACING, draft_layout, front_wall
from aosr.reporting.display import FURNITURE_KINDS
from aosr.reporting.scheme import Scheme

# 前牆是哪一面 → 離前牆的距離量哪一軸、從房間另一頭量（終點牆）還是從 0 量（起點牆）。
_DEPTH: Final = {"x0": (0, False), "xL": (0, True), "y0": (1, False), "yL": (1, True)}
NOT_FILLED: Final = None


def metres(value: float) -> str:
    """到公分、去掉尾巴的 0（同首頁 room_words）；-0 寫 0。"""
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return "0" if text == "-0" else text


def _room_sizes(document: dict[str, object]) -> tuple[float, float, float] | None:
    scene = document.get("scene")
    room = scene.get("room_m") if isinstance(scene, Mapping) else None
    if not isinstance(room, Mapping):
        return None
    sizes = [room.get(key) for key in ("Lx", "Ly", "Lz")]
    numbers = [float(size) for size in sizes
               if isinstance(size, int | float) and not isinstance(size, bool) and math.isfinite(size) and size > 0]
    return (numbers[0], numbers[1], numbers[2]) if len(numbers) == 3 else None


def _room_line(sizes: tuple[float, float, float] | None, front: str | None) -> str | None:
    if sizes is None:
        return NOT_FILLED
    lx, ly, lz = sizes
    if front is None:
        return f"長 {metres(lx)} × 寬 {metres(ly)} × 高 {metres(lz)} 公尺"
    # 寬、深照前牆分（從主位看左右、前後），長寬高三格的欄名卻是固定軸（長 Lx、寬 Ly）：每個數後面寫是哪一格，
    # 前牆在 y 那兩面時「寬」對的是「長 Lx」那一格，不寫會跟圖邊那一格對不上。
    across, deep = (f"{metres(ly)}（Ly）", f"{metres(lx)}（Lx）") if _DEPTH[front][0] == 0 else (f"{metres(lx)}（Lx）", f"{metres(ly)}（Ly）")
    return f"寬 {across}× 深 {deep}× 高 {metres(lz)} 公尺"


def _from_front(layout: Scheme, front: str, x: float, y: float) -> float:
    axis, far = _DEPTH[front]
    room = layout.scene.room_m
    value = (x, y)[axis]
    return ((room.Lx, room.Ly)[axis] - value) if far else value


def _leftward(front: str, x: float, y: float) -> float:
    """點往聽的人左手邊多遠（只拿來比相對位置）：面向前牆時左方是面向逆時針轉 90 度。"""
    fx, fy = FACING[front]
    return -fy * x + fx * y


def _each(names: Sequence[str], values: Sequence[str]) -> str:
    """幾支一樣就寫一個數；不一樣就逐支寫名字。"""
    if len(set(values)) == 1:
        return values[0]
    return "、".join(f"{name} {value}" for name, value in zip(names, values, strict=True))


def _ordered_speakers(layout: Scheme) -> list[str]:
    ordered = [channel.speaker_id for channel in layout.channel_group.channels if channel.speaker_id in layout.speakers]
    return list(dict.fromkeys(ordered + sorted(layout.speakers)))


def _speakers_line(document: dict[str, object], layout: Scheme, front: str | None) -> str:
    ids = _ordered_speakers(layout)
    names = [speaker_name(document, speaker_id) for speaker_id in ids]
    points = [layout.speakers[speaker_id] for speaker_id in ids]
    heights = _each(names, [metres(point.z) for point in points])
    if len(points) != 2:
        return "；".join(f"{name}在 x {metres(point.x)}、y {metres(point.y)}，高 {metres(point.z)} 公尺"
                        for name, point in zip(names, points, strict=True))
    apart = metres(math.hypot(points[0].x - points[1].x, points[0].y - points[1].y))
    if front is None:
        return f"兩支相距 {apart} 公尺，高 {heights} 公尺（分不出前牆，位置看「座標數字」）"
    depths = _each(names, [metres(_from_front(layout, front, point.x, point.y)) for point in points])
    return f"兩支在前牆前 {depths} 公尺，相距 {apart} 公尺，高 {heights} 公尺"


def _midline(layout: Scheme, front: str, x: float, y: float) -> str:
    points = list(layout.speakers.values())
    if len(points) != 2:
        return ""
    middle = sum(_leftward(front, point.x, point.y) for point in points) / 2
    offset = _leftward(front, x, y) - middle
    if metres(abs(offset)) == "0":
        return "，在兩支喇叭的中線上"
    return f"，在兩支喇叭的中線{'左' if offset > 0 else '右'}邊 {metres(abs(offset))} 公尺"


def _primary_line(layout: Scheme, front: str | None) -> str:
    x, y, z = layout.receiver_set.primary.position_m
    if front is None:
        return f"在 x {metres(x)}、y {metres(y)}，耳朵高 {metres(z)} 公尺（分不出前牆）"
    return f"離前牆 {metres(_from_front(layout, front, x, y))} 公尺{_midline(layout, front, x, y)}，耳朵高 {metres(z)} 公尺"


def _furniture_line(document: dict[str, object]) -> str:
    items = document.get("furniture")
    kinds = [item.get("kind") for item in items if isinstance(item, Mapping)] if isinstance(items, list) else []
    if not kinds:
        return "沒有家具"
    return "、".join(FURNITURE_KINDS.get(str(kind), "家具") for kind in kinds)


def room_facts(document: object) -> dict[str, str | None]:
    """右卡四句；喇叭、主位（還有前牆）要四格驗得過才寫，否則 None。"""
    raw = cast(dict[str, object], dict(document)) if isinstance(document, Mapping) else {}
    layout = draft_layout(document)
    front = front_wall(layout) if layout is not None else None
    sizes = (layout.scene.room_m.Lx, layout.scene.room_m.Ly, layout.scene.room_m.Lz) if layout is not None else _room_sizes(raw)
    return {"room": _room_line(sizes, front),
            "speakers": NOT_FILLED if layout is None else _speakers_line(raw, layout, front),
            "primary": NOT_FILLED if layout is None else _primary_line(layout, front),
            "furniture": _furniture_line(raw)}
