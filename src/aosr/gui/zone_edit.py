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
from functools import partial
from typing import TYPE_CHECKING, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from aosr.geometry.shoebox import Room
from aosr.physics.report_surfaces import RectangleZone, validate_surface_partition
from aosr.reporting.scheme import Scheme
from aosr.physics.zone_fill import FILL_PREFIX, Box, FillResult, fill_wall, wall_size_m

if TYPE_CHECKING:
    from aosr.gui.zone_symmetry import Frame, ZoneFrame

SNAP_M = 0.05
MIN_SIDE_M = 0.20
# 演算法容差（同 fem_lane 的 *_TOL：判「是不是同一條邊」，不是精度契約），單位公尺。
EDGE_MERGE_TOL = 1e-6
RESIZE_EDGE_M = 0.001
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


class MirrorChange(BaseModel):
    """只允許一個畫框的前後差異；補畫時兩個值必須是同一個原框。"""

    model_config = ConfigDict(extra="forbid")
    before: CutBox | None
    after: CutBox | None
    complete: bool = False
    protected_walls: tuple[WallName, ...] = ()


class ZoneCut(BaseModel):
    """一次切一面：整份方案（拿房間尺寸與切之前的材料）、哪一面、這一面所有畫的框。"""

    model_config = ConfigDict(extra="forbid")
    scheme: dict[str, object]
    wall: WallName
    boxes: list[CutBox] = Field(max_length=MAX_BOXES)
    mirror: MirrorChange | None = None


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


type MaterialBox = tuple[Box, dict[str, object]]


def _old_rectangles(entries: Sequence[dict[str, object]]) -> list[MaterialBox]:
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
        found.append((box, cast(dict[str, object], entry["material"])))
    return found


def _distance(box: Box, u: float, v: float) -> float:
    du = max(box.u_m[0] - u, 0.0, u - box.u_m[1])
    dv = max(box.v_m[0] - v, 0.0, v - box.v_m[1])
    return math.hypot(du, dv)


def _source_at(rectangles: Sequence[MaterialBox], u: float, v: float,
               whole: dict[str, object]) -> tuple[str, dict[str, object]]:
    """照方案順序取第一個蓋住 (u, v) 的區；都沒蓋到取最近的；沒有矩形就是整面那一區。"""
    if not rectangles:
        return WHOLE_WALL_KEY, whole
    box, material = min(rectangles, key=lambda item: _distance(item[0], u, v))
    return box.id, material


def _inherited(result: FillResult, rectangles: Sequence[MaterialBox],
               whole: dict[str, object]) -> dict[str, tuple[str, dict[str, object]]]:
    """畫框與尺寸調整用同一份歸屬：原來畫的框認代號，自動補滿的框認中心所在的原區。"""
    old = {box.id: material for box, material in rectangles}
    sources = {}
    for zone in result.zones:
        if zone.id not in result.filled_ids and zone.id in old:
            sources[zone.id] = (zone.id, old[zone.id])
        else:
            sources[zone.id] = _source_at(rectangles, sum(zone.u_m) / 2, sum(zone.v_m) / 2, whole)
    return sources


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


def snap_edges(box: CutBox, others: Sequence[CutBox], width: float, height: float) -> CutBox:
    """網頁送框之前的吸邊：框的每一條邊，離牆邊或這一面別的畫框的邊差大於 0、又不到「最小邊扣容差」，就貼到最近的那一條
    （一樣近留先找到的：0、牆寬或牆高，再照別框的順序）。

    跟 ``src/aosr/gui/static/elevation.js`` 的 ``snapEdges`` 是同一把（畫新框、搬框、打數字改框都先過它才送 ``/api/zone-cut``），
    改一邊要改兩邊；``test_gui_zone_symmetry_browser`` 有一題拿同一批框兩邊各算一次比。伺服器這一份只給
    「左右對稱」列切法時算「照打數字網頁會送出哪個框」，切牆面本身不吸。"""
    reach = MIN_SIDE_M - EDGE_MERGE_TOL

    def snap_one(value: float, targets: Sequence[float]) -> float:
        best, gap = value, reach
        for target in targets:
            distance = abs(value - target)
            if 0 < distance < gap:
                best, gap = target, distance
        return best

    u_targets = [0.0, width, *(edge for other in others for edge in other.u_m)]
    v_targets = [0.0, height, *(edge for other in others for edge in other.v_m)]
    return CutBox(id=box.id, u_m=(snap_one(box.u_m[0], u_targets), snap_one(box.u_m[1], u_targets)),
                  v_m=(snap_one(box.v_m[0], v_targets), snap_one(box.v_m[1], v_targets)))


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
    previous = _old_rectangles(entries)
    whole = cast(dict[str, object], entries[0]["material"])
    if not drawn:
        source, material = _source_at(previous, width / 2, height / 2, whole)
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
    sources = _inherited(result, previous, whole)
    zones: list[dict[str, object]] = []
    inherited: dict[str, str] = {}
    for zone in result.zones:
        source, material = sources[zone.id]
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


def _drawn(document: dict[str, object], wall: str) -> list[CutBox]:
    return [CutBox(id=box.id, u_m=box.u_m, v_m=box.v_m) for box, _ in _old_rectangles(_old_entries(document, wall))
            if not box.id.startswith(FILL_PREFIX)]


class MirrorChangeError(ZoneCutError):
    """送來的差異與牆上的框對不上。"""


def _changed_boxes(body: ZoneCut, change: MirrorChange) -> tuple[str, list[CutBox]]:
    """核對改前、改後及所有旁框；舊框留在原來的次序，免得改到別區的代號或吸邊。"""
    before, after = change.before, change.after
    active = after or before
    if active is None or (before and before.id != active.id) or (change.complete and before != after):
        raise MirrorChangeError("改框前後的代號或補畫的位置對不上，請重新選框")
    old = _drawn(body.scheme, body.wall)
    expected = {box.id: box for box in old}
    if expected.get(active.id) != before:
        raise MirrorChangeError("原來的框已改變，請重新選取這一區")
    if after:
        expected[active.id] = after
    else:
        del expected[active.id]
    if len(body.boxes) != len(expected) or {box.id: box for box in body.boxes} != expected:
        raise MirrorChangeError("送出的切法還改到其他區，請重新選框")
    return active.id, list(expected.values())


def _source_cut(body: ZoneCut, change: MirrorChange, key: str, source: list[CutBox],
                reason: str = "", other: str = "", fix: str = "", *, preview: bool = False) -> dict[str, object]:
    """鏡像不能切時仍套用手畫的那一面；補畫只動鏡像，失敗時兩面都保留。"""
    walls = {} if change.complete else {body.wall: cut_wall(body.scheme, body.wall, source)}
    if preview:
        return {"reason": reason, "other": other}
    verb = "畫" if change.after else "刪"
    if change.after is None:
        fix = "原來選的框已刪除。若也要刪另一邊，請選取那一塊再刪。"
    message = f"另一邊沒有跟著{verb}：{{first}} {reason}。{fix}" if reason else ""
    return {"walls": walls, "message": message, "first": f"{body.wall}@{key}", "other": other}


def _place_partner(peers: list[CutBox], partner_id: str | None, u: tuple[float, float],
                   v: tuple[float, float]) -> tuple[list[CutBox], str]:
    used = {box.id for box in peers}
    key = partner_id or next(f"z{i}" for i in range(1, len(used) + 2) if f"z{i}" not in used)
    mirrored = CutBox(id=key, u_m=u, v_m=v)
    if partner_id is None:
        return [*peers, mirrored], key
    return [mirrored if b.id == partner_id else b for b in peers], key


def _follow_mirror(change: MirrorChange, frame: Frame, shape: ZoneFrame, partner: str | None) -> bool:
    """新畫或補畫會找另一邊；搬動落單框時，只有原先因幾何放不下的框才重試。"""
    from aosr.gui.zone_symmetry import ZoneFrame

    if change.before is None or change.complete:
        return partner is None
    if partner:
        return True
    before = ZoneFrame(shape.surface, shape.wall, change.before.u_m, change.before.v_m)
    return change.after is not None and frame.mirror_refusal(before) is not None


def _try_mirror(body: ZoneCut, change: MirrorChange, *, preview: bool = False) -> dict[str, object]:
    """檢查鏡像畫框。拖曳預覽只回放置判決；放開才切牆、回傳新的分區。"""
    from aosr.gui.zone_symmetry import Frame, ZoneFrame, drawn_partners, same_frame, symmetry_axis

    wall, doc = body.wall, body.scheme
    key, source = _changed_boxes(body, change)
    source_only = partial(_source_cut, body, change, key, source, preview=preview)
    editing = change.before is not None and not change.complete
    try:
        scheme = Scheme.model_validate(doc)
    except ValidationError:
        return source_only() if editing else source_only("的方案還有沒填好的地方，定不出對稱軸", fix="請先檢查方案。")
    axis = symmetry_axis(scheme)
    if isinstance(axis, str):
        return source_only() if editing else source_only(axis, fix="請調整喇叭或主位，再選框按「補畫另一邊」。")
    frame = Frame(scheme, axis)
    active = change.after or change.before
    assert active is not None
    shape = ZoneFrame(f"{wall}@{key}", wall, active.u_m, active.v_m)
    partners = drawn_partners(scheme, frame, added=shape if change.before is None else None)
    partner = partners.get(shape.surface)
    if frame.side(wall) and not axis.centred and change.after:
        distance = axis.across_m if axis.left.endswith("0") else frame.across_length - axis.across_m
        reason = f"左右牆不跟著畫：對稱軸離左牆 {distance:.2f} 公尺，房間正中是 {frame.across_length / 2:.2f} 公尺"
        guidance = "要左右牆跟著畫，到下面『喇叭與座位』，把兩支喇叭和主位左右對稱擺在房間正中兩側"
        reply = source_only(reason)
        return {**reply, "cause": reason, "reason": f"{reason}。{guidance}", "source_edit": editing,
                "message": f"{reason}（軸不在房間正中）。{guidance}。"}
    if not _follow_mirror(change, frame, shape, partner):
        return source_only()
    destination, u, v = frame.mirrored(shape)
    target = cast(WallName, destination)
    if change.after and same_frame(target, u, v, shape):
        reply = source_only()
        if partner:
            reply.update(message="{first_number} 現在跨在對稱軸正中，{other_number} 不再跟著它；要的話自己刪 {other_number}", other=partner)
        return {**reply, "centred": True}
    if change.after and (refused := frame.mirror_refusal(shape)):
        return {**source_only(refused[1]), "reason": refused[1], "refusal": refused[0]}
    return _cut_partner(body, change, source, partners, partner, target, u, v, preview=preview)


def _cut_partner(body: ZoneCut, change: MirrorChange, source: list[CutBox], partners: dict[str, str],
                 partner: str | None, target: WallName, u: tuple[float, float], v: tuple[float, float],
                 *, preview: bool) -> dict[str, object]:
    from aosr.gui.zone_symmetry import ZoneFrame, overlap_area, same_frame

    wall, doc = body.wall, body.scheme
    active = change.after or change.before
    assert active is not None
    source_only = partial(_source_cut, body, change, active.id, source, preview=preview)
    if target in change.protected_walls:
        return source_only("的對面還有實測阻抗草稿", fix="請先套用或取消草稿，再補畫另一邊。")
    try:
        peers = source if target == wall else _drawn(doc, target)
    except ZoneCutError as exc:
        return source_only(f"對面的分區讀不出來（{exc}）", fix="請先檢查對面分區。")
    except ValidationError:
        return source_only("對面的既有框代號無法用來切牆", fix="請先修正對面框的代號，再補畫另一邊。")
    partner_id = partner.partition("@")[2] if partner else None
    boxes = [box for box in peers if box.id != partner_id]
    if change.after:
        for box in boxes:
            peer = ZoneFrame(f"{target}@{box.id}", target, box.u_m, box.v_m)
            if change.before and not change.complete and not partner and peer.surface not in partners and same_frame(target, u, v, peer):
                return {**source_only(), "paired": peer.surface}  # 鏡像位置已有另一塊。
            if overlap_area(u, v, peer):
                reply = source_only("照鏡子翻過去會壓到 {other}", f"{target}@{box.id}",
                                    "要另一邊的話，先把 {other} 改小或刪掉，再選 {first} 按「補畫另一邊」。")
                return {**reply, "blocked": {"wall": target, "u_m": u, "v_m": v}}
        if len(boxes) >= MAX_BOXES:
            return source_only("對面的畫框數量已滿", fix="請先刪除多餘的框，再補畫另一邊。")
        boxes, identifier = _place_partner(peers, partner_id, u, v)
    try:
        other_cut = cut_wall(doc, target, boxes)
    except ZoneCutError as exc:
        return source_only(f"翻過去切不成（{exc}）", fix="請調整框邊後再試。")
    walls = {} if target == wall or change.complete else {wall: cut_wall(doc, wall, source)}
    if preview:
        return {"reason": "", "paired": f"{target}@{identifier}" if change.after else None}
    walls[target] = other_cut
    if change.complete:
        return {"walls": walls, "completed": f"{target}@{identifier}"}
    if change.after is None:
        return {"walls": walls, "deleted_partner": partner}
    return {"walls": walls, "message": "", "paired": f"{target}@{identifier}"}


def cut_mirrored(body: ZoneCut, change: MirrorChange, *, preview: bool = False) -> dict[str, object]:
    """先判原操作。幾何不合時，從放開後的牆面試每種修法，保留能配對或自己置中的。"""
    from aosr.gui.zone_symmetry import Frame, ZoneFrame, symmetry_axis

    reply = _try_mirror(body, change, preview=preview)
    if "refusal" not in reply or not change.after:
        return reply
    wall = body.wall
    # 拒絕鏡像仍會切手畫的一面；舊夥伴留在原位，試修時也要把它當現場的框。
    scene = cast(dict[str, object], body.scheme["scene"])
    walls = dict(cast(dict[str, object], scene[ZONE_FIELD]))
    if not change.complete:
        changes = cast(dict[str, dict[str, object]], reply.get("walls", {}))
        if wall in changes:
            walls[wall] = changes[wall]["zones"]
        else:
            _, source = _changed_boxes(body, change)
            walls[wall] = cut_wall(body.scheme, wall, source)["zones"]
    document = {**body.scheme, "scene": {**scene, ZONE_FIELD: walls}}
    scheme = Scheme.model_validate(document)
    axis = symmetry_axis(scheme)
    assert not isinstance(axis, str)
    frame = Frame(scheme, axis)
    boxes = _drawn(document, wall)
    before = next(box for box in boxes if box.id == change.after.id)
    shape = ZoneFrame(f"{wall}@{before.id}", wall, before.u_m, before.v_m)
    repairs, errors = [], []
    for words, candidate in frame.repair_candidates(shape, str(reply["refusal"])):
        trial = MirrorChange(before=before, after=candidate, protected_walls=change.protected_walls)
        updated = [candidate if box.id == before.id else box for box in boxes]
        try:
            verdict = _try_mirror(ZoneCut(scheme=document, wall=wall, boxes=updated, mirror=trial), trial, preview=True)
        except ZoneCutError as exc:
            errors.append(str(exc))
            continue
        if verdict.get("paired") or (reply["refusal"] == "跨軸" and verdict.get("centred")):
            repairs.append({**candidate.model_dump(), "action": words})
        else:
            errors.append(str(verdict.get("reason") or "沒有對應框").replace("{other}", "別的框"))
    reason = str(reply["reason"])
    # 詳細列保留實際通過的兩種操作；橫幅優先顯示第一個候選的畫面欄位。
    if repairs:
        detail = "，或".join(str(r["action"]) for r in reversed(repairs))
    else:
        causes = ("補滿後會剩窄條" if "窄條" in error else error for error in errors)
        detail = "試過挪和收都不行（" + "；".join(dict.fromkeys(causes)) + "）"
    full = reason + "：" + detail
    if "message" in reply and not repairs:
        # 有試過放得下的修法時，提示列只留原因，修法由畫面用欄位數字寫一次（「改成：…」），不重複講兩種說法。
        reply["message"] = str(reply["message"]).replace(reason, full, 1)
    return {**reply, "reason": full, "cause": reason, "repairs": repairs}


async def zone_cut(request: Request) -> Response:
    try:
        body = ZoneCut.model_validate(await request.json())
    except (ValidationError, ValueError):
        return JSONResponse({"error": "切牆面的請求格式不完整：要有方案、哪一面牆與畫的框"}, status_code=400)
    try:
        if body.mirror is not None:
            return JSONResponse(cut_mirrored(body, body.mirror))
        return JSONResponse(cut_wall(body.scheme, body.wall, body.boxes))
    except ZoneCutError as exc:
        return JSONResponse({"error": str(exc), "problems": [{"text": text} for text in exc.problems],
                             "thin": [{"u_m": list(box.u_m), "v_m": list(box.v_m)} for box in exc.thin]},
                            status_code=400)


async def zone_mirror_check(request: Request) -> Response:
    """拖曳時的鏡像預覽：跟放開後同一套判斷（``cut_mirrored``），只回放不放得下與原因，不切牆、不寫任何東西。"""
    try:
        body = ZoneCut.model_validate(await request.json())
    except (ValidationError, ValueError):
        return JSONResponse({"error": "鏡像預覽的請求格式不完整：要有方案、哪一面牆、畫的框與鏡像"}, status_code=400)
    if body.mirror is None:
        return JSONResponse({"error": "鏡像預覽要帶鏡像那一欄"}, status_code=400)
    try:
        return JSONResponse(cut_mirrored(body, body.mirror, preview=True))
    except MirrorChangeError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except ZoneCutError as exc:
        return JSONResponse({"reason": f"這一面切不成（{exc}）", "source_refused": True})


def _at_moving_edge(value: float, old: float, new: float) -> bool:
    # 邊長先相減，剛好 1 毫米的十進位輸入可能多出最後一位浮點誤差。
    return new != old and abs(value - old) <= RESIZE_EDGE_M + math.ulp(old)


def _resize_wall(entries: list[dict[str, object]], wall: str, width: float, height: float) -> dict[str, object]:
    """方案座標沿房間正軸；每面用仍保留的分區外緣當舊尺寸，縮不下的面可在下一次重新調。"""
    rectangles = _old_rectangles(entries)
    if len(rectangles) != len(entries):
        raise ZoneCutError(SHAPE_ERROR)
    old_w = max(box.u_m[1] for box, _ in rectangles)
    old_h = max(box.v_m[1] for box, _ in rectangles)
    if (width, height) == (old_w, old_h):
        return {}
    resized: list[MaterialBox] = []
    for box, material in rectangles:
        key = box.id
        u0, u1 = box.u_m
        v0, v1 = box.v_m
        new = Box(key, (u0, width if _at_moving_edge(u1, old_w, width) else u1),
                  (v0, height if _at_moving_edge(v1, old_h, height) else v1))
        if u0 >= width or v0 >= height or new.u_m[1] > width or new.v_m[1] > height:
            return {"blocked": key, "reason": "超出目前的牆面"}
        if (width < old_w or height < old_h) and too_small(new):
            return {"blocked": key, "reason": f"寬或高不到最小邊 {MIN_SIDE_M:g} 公尺"}
        if (u0 > 0 and _at_moving_edge(u0, old_w, width)) or (v0 > 0 and _at_moving_edge(v0, old_h, height)):
            return {"blocked": key, "reason": "距牆邊在 1 毫米內，調整後會與鄰區重疊"}
        resized.append((new, material))
    # 跟畫框共用補滿。用調整後的矩形辨認原區，讓自動代號、材料和做法一起沿用；
    # 匯入的方案可能手動細分過補滿格，這些既有接縫也必須保留。
    moved = [box for box, _ in resized]
    drawn = [box for box in moved if not box.id.startswith(FILL_PREFIX)]
    filled = fill_wall(wall, width, height, drawn, min_side_m=0.0)
    bounds = {(box.u_m, box.v_m) for box in moved}
    if {(box.u_m, box.v_m) for box in filled.zones} != bounds:
        filled = fill_wall(wall, width, height, moved, min_side_m=0.0)
    if filled.problems or any((box.u_m, box.v_m) not in bounds for box in filled.zones):
        return {"blocked": moved[0].id, "reason": "調整後的分區無法無縫蓋滿這一面"}
    sources = _inherited(filled, resized, resized[0][1])
    return {"zones": {sources[box.id][0]: {"u_m": box.u_m, "v_m": box.v_m} for box in filled.zones}}


async def zone_resize(request: Request) -> Response:
    try:
        document = await request.json()
        if not isinstance(document, dict):
            raise ZoneCutError(SHAPE_ERROR)
        room = draft_room(document)
        changes = {}
        for wall in ("x0", "xL", "y0", "yL", "floor", "ceiling"):
            entries = _old_entries(document, wall)
            if entries[0].get("area") != "whole_wall":
                changes[wall] = _resize_wall(entries, wall, *wall_size_m(room, wall))
        return JSONResponse(changes)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


def zone_edit_routes() -> list[Route]:
    return [Route("/api/zone-cut", zone_cut, methods=["POST"]),
            Route("/api/zone-mirror-check", zone_mirror_check, methods=["POST"]),
            Route("/api/zone-resize", zone_resize, methods=["POST"])]
