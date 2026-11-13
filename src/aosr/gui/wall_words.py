"""牆名跟著方案走（#1134）：畫面上把 x 起點牆、x 終點牆、y 起點牆、y 終點牆叫成前牆、後牆、左牆、右牆。

前牆照 ``/search-start`` 表單預設的同一條規則：兩支喇叭水平中點離哪一面牆最近，那一面就是前牆
（``front_wall_guess``，只有這一份）。後牆是它對面；左右是聽的人面向前牆時的左右——
面向逆時針轉 90 度是左方（同 ``reporting.scheme.project_facing`` 的左方定義）。

定不出來就整組退回座標名（``labels.WALLS``），不猜：

- 方案不是一對左右聲道（例如第一階段範本只有一個聲源），``project_facing`` 拒收；
- 兩面牆離喇叭中點一樣近；
- 喇叭靠的那面牆跟主位面向（兩支喇叭中點相對主位的主軸）指的不是同一面——兩套規則說法不同時不挑一套。

這一支只排字、不改存檔，住在 gui 底下（不在計算指紋裡）。
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from aosr.gui.labels import WALLS
from aosr.reporting.scheme import Scheme, project_facing

FRONT: Final = "前牆"
BACK: Final = "後牆"
LEFT: Final = "左牆"
RIGHT: Final = "右牆"
OPPOSITE: Final = {"x0": "xL", "xL": "x0", "y0": "yL", "yL": "y0"}
# 面向前牆時左手邊那一面：面向 f 逆時針轉 90 度是 (−f_y, f_x)。前牆 x0 時面向 −x，左方是 −y，也就是 y0。
LEFT_OF: Final = {"x0": "y0", "xL": "yL", "y0": "xL", "yL": "x0"}
# 面向前牆的單位向量（主位往前牆走的方向）。
FACING: Final = {"x0": (-1.0, 0.0), "xL": (1.0, 0.0), "y0": (0.0, -1.0), "yL": (0.0, 1.0)}


def _distances(scheme: Scheme) -> dict[str, float]:
    points = [scheme.speakers[name] for name in sorted(scheme.speakers)]
    x = sum(point.x for point in points) / len(points)
    y = sum(point.y for point in points) / len(points)
    room = scheme.scene.room_m
    return {"x0": x, "xL": room.Lx - x, "y0": y, "yL": room.Ly - y}


def front_wall_guess(scheme: Scheme) -> str:
    """兩支喇叭中點離哪一面牆最近，就當前牆的預設（只是表單的預設，可以改）。"""
    distances = _distances(scheme)
    return min(distances, key=lambda wall: distances[wall])


def front_wall(scheme: Scheme) -> str | None:
    """畫面上能不能叫前牆：喇叭最近的那面、唯一最近、而且主位正面向它；任一條不成立回 None。"""
    try:
        facing = project_facing(scheme)
    except (ValueError, KeyError):
        return None
    distances = _distances(scheme)
    front = front_wall_guess(scheme)
    if sum(1 for value in distances.values() if value == distances[front]) != 1:
        return None
    return front if FACING[front] == facing else None


def wall_words(scheme: Scheme) -> dict[str, str]:
    """六面牆給人看的名字；地板、天花不變，四面牆定得出前牆時換成前後左右，否則照座標名。"""
    front = front_wall(scheme)
    if front is None:
        return dict(WALLS)
    left = LEFT_OF[front]
    named = {front: FRONT, OPPOSITE[front]: BACK, left: LEFT, OPPOSITE[left]: RIGHT}
    return {wall: named.get(wall, word) for wall, word in WALLS.items()}


def draft_wall_words(document: object) -> dict[str, str]:
    """表單草稿的牆名：前後左右只看房間、喇叭、聲道、座位這四格，所以只拿這四格的原值、其餘換成不會錯的佔位值
    再驗（存檔名稱清空、別格填錯時名字不跟著變回座標名，同一頁的檢查訊息才跟表單同一套叫法）；這四格本身驗不過照座標名。
    佔位的聲速、密度只為了過驗證，不拿來算任何東西。"""
    if not isinstance(document, Mapping):
        return dict(WALLS)
    scene = document.get("scene")
    if not isinstance(scene, Mapping):
        return dict(WALLS)
    probe = {"schema_version": "aosr.scheme.v1", "scheme_id": "wall-words", "purpose": "wall-words",
             "source_model": "omnidirectional",
             **{key: document.get(key) for key in ("speakers", "channel_group", "receiver_set")},
             "scene": {"room_m": scene.get("room_m"), "sound_speed_m_s": 1.0, "density_kg_m3": 1.0,
                       "impedance_pa_s_per_m_by_wall": dict.fromkeys(WALLS, 1.0)}}
    try:
        return wall_words(Scheme.model_validate(probe))
    except ValueError:
        return dict(WALLS)
