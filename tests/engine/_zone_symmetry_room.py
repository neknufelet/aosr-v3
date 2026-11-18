"""左右對稱考卷用的假房間（#1206 第 2 支）：數字全是考卷自己編的，不是哪一間真的房間。

房間 5.0 × 4.0 × 2.7 m。預設前牆是 x 起點那一面（x0）：兩支喇叭靠它、左右各離房間中線 0.8 m，主位在中線上，
所以對稱軸沿 x、落在 y = 2.0（房間正中）；面向前牆時左手邊是 y0。``across_shift_m`` 把兩支喇叭與整組座位一起橫移
（軸跟著離開房間正中），``seat_shift_m`` 只橫移座位（主位離開兩支喇叭中間）。``turned=True`` 把同一間房轉 90 度：
長寬對調、前牆換成 y 終點那一面（yL），考另一個朝向的鏡像分支。

六面先各整面一區（固定阻抗），再照 ``cuts``（牆 → 幾個框）切，框外自動補滿（代號 auto1…，跟網頁切牆同一支補滿）。
"""
from __future__ import annotations

from copy import deepcopy
from typing import cast

from aosr.physics.zone_fill import Box as FillBox, fill_wall
from tests.engine._zone_naming_room import ZONE_NAMING_SCHEME

NAME = "sym-sample"
LENGTH, WIDTH, HEIGHT = 5.0, 4.0, 2.7
WALL_IMPEDANCE = 71500.0
Box = tuple[str, tuple[float, float], tuple[float, float]]
Doc = dict[str, object]
_OFFSETS = {"main": (0.0, 0.0, 0.0), "front": (-0.24, 0.0, 0.0), "back": (0.24, 0.0, 0.0), "left": (0.0, -0.24, 0.0),
            "right": (0.0, 0.24, 0.0), "up": (0.0, 0.0, 0.24), "down": (0.0, 0.0, -0.24)}


def _turn(point: tuple[float, float, float]) -> tuple[float, float, float]:
    """轉 90 度：原本的 (x, y) 變成 (y, 長 − x)；前牆從 x0 變成 yL，左牆從 y0 變成 x0。"""
    x, y, z = point
    return y, LENGTH - x, z


def symmetry_document(cuts: dict[str, list[Box]] | None = None, *, across_shift_m: float = 0.0, seat_shift_m: float = 0.0,
                      turned: bool = False, scheme_id: str = NAME) -> Doc:
    document = deepcopy(ZONE_NAMING_SCHEME)
    document["scheme_id"] = scheme_id
    place = _turn if turned else (lambda point: point)
    left, right = (0.9, 1.2 + across_shift_m, 1.0), (0.9, 2.8 + across_shift_m, 1.0)
    seat = (3.2, 2.0 + across_shift_m + seat_shift_m, 1.2)
    document["speakers"] = {name: dict(zip("xyz", place(point), strict=True)) for name, point in (("left", left), ("right", right))}
    points = cast(list[Doc], cast(Doc, document["receiver_set"])["points"])
    for point in points:
        dx, dy, dz = _OFFSETS[cast(str, point["receiver_id"])]
        point["position_m"] = list(place((seat[0] + dx, seat[1] + dy, seat[2] + dz)))
    scene = cast(Doc, document["scene"])
    scene["room_m"] = {"Lx": WIDTH, "Ly": LENGTH, "Lz": HEIGHT} if turned else {"Lx": LENGTH, "Ly": WIDTH, "Lz": HEIGHT}
    whole = {"area": "whole_wall", "material": {"kind": "constant_impedance", "impedance_pa_s_m": WALL_IMPEDANCE}}
    scene["surface_zones_by_wall"] = {wall: [deepcopy(whole)] for wall in ("floor", "ceiling", "x0", "xL", "y0", "yL")}
    zones = cast(dict[str, list[dict[str, object]]], scene["surface_zones_by_wall"])
    room = cast(dict[str, float], scene["room_m"])
    for wall, boxes in (cuts or {}).items():
        width, height = ((room["Lx"], room["Ly"]) if wall in ("floor", "ceiling")
                         else (room["Ly"] if wall[0] == "x" else room["Lx"], room["Lz"]))
        # 補滿不設最小邊：容差那幾題要挪半公分，補出來的窄條照留（方案格式收得下，網頁切牆才擋窄條）。
        filled = fill_wall(wall, width, height, [FillBox(id=name, u_m=u_m, v_m=v_m) for name, u_m, v_m in boxes], min_side_m=0.0)
        assert not filled.problems, filled.problems
        zones[wall] = [{"area": "rectangle", "id": zone.id, "u_m": list(zone.u_m), "v_m": list(zone.v_m),
                        "material": deepcopy(whole["material"])} for zone in filled.zones]
    return document


def searching(zones: dict[str, list[str]], **fields: object) -> Doc:
    """設計設定：列到的區都由程式最佳化（准用清單不寫，照全部可調材料與不貼）；其餘欄位照 ``fields``。"""
    return {"schema_version": "aosr.zone_design.v1", "scheme_id": NAME,
            "zones": {wall: {zone: {"mode": "search"} for zone in names} for wall, names in zones.items()}, **fields}


# 前牆（x0，u 是 y、v 是 z）左右各一塊，互為鏡像；方案裡先列 b 再列 a（面序 b 在前、字串序 a 在前）。
FRONT_PAIR: list[Box] = [("b", (0.4, 1.5), (0.3, 2.4)), ("a", (2.5, 3.6), (0.3, 2.4))]
# 左右牆（y0／yL，u 是 x、v 是 z）同位置各一塊。
SIDE: list[Box] = [("s", (1.0, 3.0), (0.5, 2.2))]
# 天花（u 是 x、v 是 y）一塊，自己跨在軸上、左右置中。
CENTRED: list[Box] = [("c", (2.0, 3.5), (1.2, 2.8))]
