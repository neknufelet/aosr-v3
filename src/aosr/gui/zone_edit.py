"""網頁立面圖畫框（#1091，#1082 第 1-2 段）：一面牆上「畫的框」送進來，其餘自動補滿、帶上材料，交回這一面的分區。

只核被切的這一面：別面因為改了房間尺寸而超出牆面時，仍要能一面一面修——整份方案一起核的話，
兩面以上對不上就哪一面都切不動。這一面的材料原樣搬過來、不在這裡核參數：半填的數字照舊交正式驗證列原因。

邊的浮點誤差在這裡收：網頁上打 0.4 加寬 0.8 得到 1.2000000000000002，跟另一框的 1.2 差一點點就會留一條
看不見的縫；牆寬 4.41988… 也不在 5 公分格上。所以補滿前，離牆邊或別框的邊不到 ``EDGE_MERGE_TOL`` 的邊
一律拉到那條邊的原值。補滿本身（``zone_fill``）照舊精確比對、不加容差。

材料怎麼跟：畫的框代號跟切之前某一區相同，就沿用那一區的材料（搬了位置也一樣）；新畫的框與自動補的格子，
取「切之前蓋住這一塊中心點的那一區」的材料；一個框都沒有就回整面一區，取切之前蓋住牆中心的那一區。
切之前的分區已經對不上牆面時（例如房間改小），中心點沒被蓋到就取離它最近的那一區，所以「清掉這一面」永遠做得到。
每一塊沿用了誰回在 ``inherited``：第 3 段要靠它把每區做法跟過重編的自動代號。

吸附格距、最小邊是工程暫定（太碎的話網格與替身會爆），只寫在這裡，經 /api/input-defaults 交給網頁。
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from copy import deepcopy
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from aosr.geometry.shoebox import Room
from aosr.physics.report_surfaces import RectangleZone, validate_surface_partition
from aosr.physics.zone_fill import FILL_PREFIX, Box, fill_wall, wall_size_m

SNAP_M = 0.05
MIN_SIDE_M = 0.20
# 演算法容差（同 fem_lane 的 *_TOL：判「是不是同一條邊」，不是精度契約），單位公尺。
EDGE_MERGE_TOL = 1e-6
MAX_BOXES = 100
ZONE_FIELD = "surface_zones_by_wall"
WHOLE_WALL_KEY = "whole_wall"
WallName = Literal["floor", "ceiling", "x0", "xL", "y0", "yL"]
ROOM_ERROR = "先把房間的長、寬、高都填好正數，才能切牆面"
SHAPE_ERROR = "這一面的分區資料不完整，網頁讀不出來；請重新打開方案"


class CutBox(BaseModel):
    """網頁畫的一個框：代號與兩軸公尺座標（軸向同方案的矩形分區）。"""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    u_m: tuple[float, float]
    v_m: tuple[float, float]


class ZoneCut(BaseModel):
    """一次切一面：整份方案（拿房間尺寸與切之前的材料）、哪一面、這一面所有畫的框。"""

    model_config = ConfigDict(extra="forbid")
    scheme: dict[str, object]
    wall: WallName
    boxes: list[CutBox] = Field(max_length=MAX_BOXES)


class ZoneCutError(ValueError):
    """切不下去：白話原因；``thin`` 是補滿後太窄的那幾條（給畫面用紅虛線標出位置）。"""

    def __init__(self, message: str, problems: Sequence[str] = (), thin: Sequence[Box] = ()) -> None:
        super().__init__(message)
        self.problems = tuple(problems) or (message,)
        self.thin = tuple(thin)


def zone_cut_defaults() -> dict[str, object]:
    """網頁畫框要用的數字：吸附格距、最小邊、邊的容差、自動補滿的代號字首。"""
    return {"snap_m": SNAP_M, "min_side_m": MIN_SIDE_M, "edge_merge_tol_m": EDGE_MERGE_TOL,
            "fill_prefix": FILL_PREFIX}


def draft_room(document: dict[str, object]) -> Room:
    scene = document.get("scene")
    room = scene.get("room_m") if isinstance(scene, dict) else None
    if not isinstance(room, dict):
        raise ZoneCutError(ROOM_ERROR)
    values = [room.get(key) for key in ("Lx", "Ly", "Lz")]
    if not all(isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value) and value > 0
               for value in values):
        raise ZoneCutError(ROOM_ERROR)
    lx, ly, lz = (float(cast(float, value)) for value in values)
    return Room(Lx=lx, Ly=ly, Lz=lz)


def _old_entries(document: dict[str, object], wall: str) -> list[dict[str, object]]:
    scene = cast(dict[str, object], document["scene"])
    zones = scene.get(ZONE_FIELD)
    entries = zones.get(wall) if isinstance(zones, dict) else None
    if not isinstance(entries, list) or not entries or not all(
            isinstance(entry, dict) and "material" in entry for entry in entries):
        raise ZoneCutError(SHAPE_ERROR)
    return cast(list[dict[str, object]], entries)


def _old_rectangles(entries: Sequence[dict[str, object]]) -> list[tuple[str, Box, dict[str, object]]]:
    """切之前這一面的矩形區（座標讀不出來的跳過，只拿來找材料）。"""
    found = []
    for entry in entries:
        if entry.get("area") != "rectangle":
            continue
        u, v, identifier = entry.get("u_m"), entry.get("v_m"), entry.get("id")
        pairs = (u, v)
        if not isinstance(identifier, str) or not all(
                isinstance(pair, list | tuple) and len(pair) == 2
                and all(isinstance(n, int | float) and not isinstance(n, bool) and math.isfinite(n) for n in pair)
                for pair in pairs):
            continue
        box = Box(id=identifier, u_m=(float(cast(list[float], u)[0]), float(cast(list[float], u)[1])),
                  v_m=(float(cast(list[float], v)[0]), float(cast(list[float], v)[1])))
        found.append((identifier, box, cast(dict[str, object], entry["material"])))
    return found


def _distance(box: Box, u: float, v: float) -> float:
    du = max(box.u_m[0] - u, 0.0, u - box.u_m[1])
    dv = max(box.v_m[0] - v, 0.0, v - box.v_m[1])
    return math.hypot(du, dv)


def _source_at(entries: Sequence[dict[str, object]], u: float, v: float) -> tuple[str, dict[str, object]]:
    """切之前蓋住 (u, v) 的那一區：照方案順序取第一個蓋住的；都沒蓋到取最近的；沒有矩形就是整面那一區。"""
    rectangles = _old_rectangles(entries)
    if not rectangles:
        return WHOLE_WALL_KEY, cast(dict[str, object], entries[0]["material"])
    identifier, _box, material = min(rectangles, key=lambda item: _distance(item[1], u, v))
    return identifier, material


def _pull(value: float, edges: list[float]) -> float:
    for edge in edges:
        if abs(value - edge) <= EDGE_MERGE_TOL:
            return edge
    edges.append(value)
    return value


def _pulled(boxes: Sequence[CutBox], width: float, height: float) -> list[Box]:
    """離牆邊或先出現的邊不到容差的邊，拉到那條邊的原值；牆邊先排，所以貼牆的框正好等於牆寬。"""
    u_edges, v_edges = [0.0, width], [0.0, height]
    return [Box(id=box.id, u_m=(_pull(box.u_m[0], u_edges), _pull(box.u_m[1], u_edges)),
                v_m=(_pull(box.v_m[0], v_edges), _pull(box.v_m[1], v_edges))) for box in boxes]


def too_small(box: Box) -> bool:
    return min(box.u_m[1] - box.u_m[0], box.v_m[1] - box.v_m[0]) < MIN_SIDE_M - EDGE_MERGE_TOL


def _check_drawn(wall: str, boxes: Sequence[Box]) -> None:
    for box in boxes:
        if box.id.startswith(FILL_PREFIX):
            raise ZoneCutError(f"牆面 {wall} 分區 {box.id}：畫的框代號不能用 {FILL_PREFIX} 開頭（那是自動補滿用的）")
        if too_small(box):
            raise ZoneCutError(f"牆面 {wall} 分區 {box.id}：框的寬或高不到最小邊 {MIN_SIDE_M:g} 公尺")


def cut_wall(document: dict[str, object], wall: WallName, boxes: Sequence[CutBox]) -> dict[str, object]:
    """切一面牆，回 ``zones``（這一面新的分區清單）、``filled_ids``（自動補的代號）、``inherited``（每一塊沿用誰的材料）。"""
    room = draft_room(document)
    entries = _old_entries(document, wall)
    width, height = wall_size_m(room, wall)
    drawn = _pulled(boxes, width, height)
    _check_drawn(wall, drawn)
    if not drawn:
        source, material = _source_at(entries, width / 2, height / 2)
        return {"zones": [{"area": "whole_wall", "material": deepcopy(material)}], "filled_ids": [],
                "inherited": {WHOLE_WALL_KEY: source}}
    # 窄條自己判（最小邊扣掉容差：0.7 − 0.5 在浮點不到 0.2，不能因此說它太窄），補滿只管不重疊、不留空。
    result = fill_wall(wall, width, height, drawn, min_side_m=0.0)
    if result.problems:
        raise ZoneCutError("；".join(result.problems), result.problems)
    thin = [zone for zone in result.zones if zone.id in result.filled_ids and too_small(zone)]
    if thin:
        problems = [f"牆面 {wall}：補滿後剩一條 u {box.u_m[0]:.3f}–{box.u_m[1]:.3f}、v {box.v_m[0]:.3f}–{box.v_m[1]:.3f} 公尺的窄條，"
                    f"比最小邊 {MIN_SIDE_M:g} 公尺還窄；請把畫的框貼齊牆邊或鄰框，或離開至少 {MIN_SIDE_M:g} 公尺" for box in thin]
        raise ZoneCutError("；".join(problems), problems, thin)
    old = {identifier: material for identifier, _box, material in _old_rectangles(entries)}
    zones: list[dict[str, object]] = []
    inherited: dict[str, str] = {}
    for zone in result.zones:
        if zone.id not in result.filled_ids and zone.id in old:
            source, material = zone.id, old[zone.id]
        else:
            source, material = _source_at(entries, sum(zone.u_m) / 2, sum(zone.v_m) / 2)
        inherited[zone.id] = source
        zones.append({"area": "rectangle", "id": zone.id, "u_m": list(zone.u_m), "v_m": list(zone.v_m),
                      "material": deepcopy(material)})
    try:
        # 只核這一面的蓋滿與不重疊（同方案那一把精確判法）；材料參數不在這裡核。
        validate_surface_partition(room, {wall: tuple(RectangleZone.model_construct(
            area="rectangle", id=zone.id, u_m=zone.u_m, v_m=zone.v_m) for zone in result.zones)})
    except ValidationError as exc:
        raise ZoneCutError("；".join(str(error["msg"]) for error in exc.errors())) from exc
    return {"zones": zones, "filled_ids": list(result.filled_ids), "inherited": inherited}


async def zone_cut(request: Request) -> Response:
    try:
        body = ZoneCut.model_validate(await request.json())
    except (ValidationError, ValueError):
        return JSONResponse({"error": "切牆面的請求格式不完整：要有方案、哪一面牆與畫的框"}, status_code=400)
    try:
        return JSONResponse(cut_wall(body.scheme, body.wall, body.boxes))
    except ZoneCutError as exc:
        return JSONResponse({"error": str(exc), "problems": [{"text": text} for text in exc.problems],
                             "thin": [{"u_m": list(box.u_m), "v_m": list(box.v_m)} for box in exc.thin]},
                            status_code=400)


def zone_edit_routes() -> list[Route]:
    return [Route("/api/zone-cut", zone_cut, methods=["POST"])]
