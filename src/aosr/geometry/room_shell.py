"""任意形狀房間的「房間殼」與匯入檢查（票 #1299，第一刀：讀進來、檢查、說人話，不算聲學）。

**房間殼是什麼。** 使用者在 Rhino 畫的聲學模型讀進來之後的樣子：外殼（牆、地板、天花
的全部面）是一組三角形，每個三角形帶它的材料圖層名；房內的造型（天雲、凸出的板子）
各自是一個封閉實體，也是一組帶圖層名的三角形；喇叭與座位各是一個點加一個朝向。
長方體的 :class:`aosr.geometry.shoebox.Room` 那條路完全不碰這一支，兩者各走各的。

**面朝哪邊。** 每個三角形的法線（照頂點順序、右手定則）一律朝空氣那一側：外殼朝房內，
所以整個外殼用面算出來的帶號體積是負的；房內實體朝外，帶號體積是正的。

**這一支只做純幾何。** 不讀檔（讀 Rhino 檔在 :mod:`aosr.geometry.rhino_room`）、不碰物理。
檢查的界線全部由呼叫端傳入（:mod:`aosr.config.room_shell_import`），這裡不寫第二份。
每一項問題都寫成白話、講得出是哪個圖層，擋不擋計算寫在 ``blocking``。

**還沒做的（照票寫明）：** 遮擋判斷（平行面對還沒扣掉彼此看不到的）、縫寬、實體之間或
實體與外殼逐面穿插、外殼凹凸。中間有空腔的造型（外層朝外、內層朝空腔）會被誤判成翻面而擋下；
這種造型很少見，擋下比漏掉安全，之後要支援時改成依巢狀判斷每一塊的朝向。
"""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from aosr.config.room_shell_import import RoomShellImportSettings


Vec3: TypeAlias = tuple[float, float, float]
FloatArray: TypeAlias = NDArray[np.float64]

IN_WALL_SPEAKER = "SPK_INWALL"
SPEAKER_KINDS: tuple[str, ...] = ("SPK_BOOKSHELF", "SPK_FLOOR", IN_WALL_SPEAKER)
MAIN_SEAT = "SEAT_MAIN"
SEAT_KINDS: tuple[str, ...] = (MAIN_SEAT, "SEAT")
# 分量刻意不整齊；擦邊時還有其他方向可試，固定順序讓同一份檔每次得到相同判決。
_RAY_DIRECTIONS: tuple[Vec3, ...] = (
    (0.5773, 0.6129, 0.5397), (-0.4211, 0.8167, -0.3947), (0.7071, -0.2236, -0.6708),
    (-0.8397, -0.4763, 0.2617), (0.3169, -0.9241, 0.2137), (0.1723, 0.3691, -0.9137),
    (-0.2687, 0.1531, 0.9513), (0.8639, 0.2843, -0.4169), (-0.5917, -0.3473, -0.7279),
)


@dataclass(frozen=True)
class LayerTriangle:
    """一個三角形與它的材料圖層名；頂點順序決定法線，法線朝空氣那一側。"""

    layer: str
    a: Vec3
    b: Vec3
    c: Vec3


@dataclass(frozen=True)
class Solid:
    """房內一個封閉的造型（天雲、凸出的板子）；``name`` 只用來在問題裡指出是哪一塊。"""

    name: str
    triangles: tuple[LayerTriangle, ...]


@dataclass(frozen=True)
class Placement:
    """喇叭或座位：種類（元件名）、位置（喇叭是聲學中心、座位是雙耳中間）、朝向。"""

    kind: str
    position: Vec3
    facing: Vec3


@dataclass(frozen=True)
class RoomShell:
    """讀進來的整間房。

    ``units`` 是檔案宣告的長度單位，只認 ``"m"``；``skipped`` 是照規定不讀的東西（只提醒），
    ``unreadable`` 是該讀卻讀不了的東西（會擋）。
    """

    units: str
    shell: tuple[LayerTriangle, ...]
    solids: tuple[Solid, ...]
    speakers: tuple[Placement, ...]
    seats: tuple[Placement, ...]
    skipped: tuple[str, ...] = ()
    unreadable: tuple[str, ...] = ()


@dataclass(frozen=True)
class Problem:
    """一項檢查結果：代號給程式用、``text`` 給人看；``blocking`` 為真代表不能拿去算。"""

    code: str
    blocking: bool
    text: str


@dataclass(frozen=True)
class PlaneGroup:
    """同一個平面（同法線、同位置）上的全部三角形，可能跨好幾個材料圖層。"""

    layers: tuple[str, ...]
    normal: Vec3
    centroid: Vec3
    area_m2: float


@dataclass(frozen=True)
class ParallelPair:
    """兩個朝向彼此、接近平行的平面；``gap_m`` 是兩面中心沿法線的距離。"""

    first: PlaneGroup
    second: PlaneGroup
    angle_deg: float
    gap_m: float


@dataclass(frozen=True)
class ImportReport:
    """匯入檢查的結果：問題清單加上幾個事實（各圖層面積、平面數、平行面對）。"""

    problems: tuple[Problem, ...]
    area_by_layer: tuple[tuple[str, float], ...]
    plane_count: int
    parallel_pairs: tuple[ParallelPair, ...]

    @property
    def ok(self) -> bool:
        """沒有任何會擋計算的問題。"""
        return not any(problem.blocking for problem in self.problems)


def _corners(triangles: Sequence[LayerTriangle]) -> FloatArray:
    """(N, 3, 3) 陣列：每個三角形的三個頂點。"""
    return np.array([(t.a, t.b, t.c) for t in triangles], dtype=np.float64).reshape(-1, 3, 3)


def _cross(corners: FloatArray) -> FloatArray:
    """每個三角形的 (b-a)×(c-a)：方向是法線、長度是兩倍面積。"""
    return np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])


def signed_volume(triangles: Sequence[LayerTriangle]) -> float:
    """封閉網格用面算出來的帶號體積；法線朝外為正、朝內為負。"""
    if not triangles:
        return 0.0
    corners = _corners(triangles)
    # 封閉網格可任選四面體的共同頂點；移到網格第一個頂點，避免測量座標的大數相減吃掉體積。
    corners = corners - corners[0, 0]
    return float(np.einsum("ij,ij->i", corners[:, 0], np.cross(corners[:, 1], corners[:, 2])).sum() / 6.0)


def _weld(triangles: Sequence[LayerTriangle], weld_m: float) -> tuple[NDArray[np.int64], set[str]]:
    """回傳 (N, 3) 頂點編號，以及焊在一起後整群跨度超過容差的圖層。

    用距離分群、不用四捨五入到格子：兩個只差一點點的頂點剛好落在格線兩邊時，格子法會把它們拆開。
    """
    points = _corners(triangles).reshape(-1, 3)
    parent = np.arange(len(points))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    for first, second in cKDTree(points).query_pairs(weld_m):
        a, b = root(int(first)), root(int(second))
        if a != b:
            parent[max(a, b)] = min(a, b)
    roots = np.array([root(i) for i in range(len(points))], dtype=np.int64)
    # 近鄰可以接力：量每一點到該群代表點的距離，超過容差就是接力接出來的（兩兩都在容差內的群不會超過）。
    # 不用外接盒對角線：三點兩兩相距不到容差時，外接盒對角線仍可能超過容差而誤擋。
    # 代價：代表點剛好在一串接力中間時，群內兩點最多差到兩倍容差（0.2 毫米）不會被擋，幾何上無影響。
    reach = np.linalg.norm(points - points[roots], axis=1)
    far = np.zeros(len(points), dtype=np.bool_)
    np.logical_or.at(far, roots, reach > weld_m)
    wide = far[roots]
    ids = roots.reshape(-1, 3)
    layers = {t.layer or "無材料子圖層" for t, affected in zip(triangles, wide[ids].any(axis=1), strict=True) if affected}
    return ids, layers


def _edges(ids: NDArray[np.int64]) -> list[tuple[int, int, int]]:
    """每個三角形的三條有向邊 (三角形序號, 起點, 終點)。"""
    return [(index, int(row[i]), int(row[(i + 1) % 3])) for index, row in enumerate(ids) for i in range(3)]


def _edge_problems(triangles: Sequence[LayerTriangle], ids: NDArray[np.int64], where: str) -> list[Problem]:
    """封閉與朝向一致：每條邊恰好被兩個三角形用、而且兩次方向相反。"""
    edges = _edges(ids)
    undirected = Counter((min(a, b), max(a, b)) for _, a, b in edges)
    directed = Counter((a, b) for _, a, b in edges)
    open_layers = {triangles[i].layer for i, a, b in edges if undirected[(min(a, b), max(a, b))] != 2}
    flipped_layers = {triangles[i].layer for i, a, b in edges
                      if undirected[(min(a, b), max(a, b))] == 2 and directed[(a, b)] != 1}
    problems = []
    if open_layers:
        bad = sum(1 for count in undirected.values() if count != 2)
        problems.append(Problem("not_closed", True, f"{where}沒有封起來：有 {bad} 條邊沒有剛好接兩個面"
                                f"（破洞或多出來的面），在這些圖層附近：{'、'.join(sorted(open_layers))}"))
    if flipped_layers:
        problems.append(Problem("inconsistent_faces", True, f"{where}有面翻反了：相鄰兩個面朝向不一致，"
                                f"在這些圖層：{'、'.join(sorted(flipped_layers))}"))
    return problems


def _components(ids: NDArray[np.int64]) -> list[list[int]]:
    """共用邊連在一起的三角形分成一塊一塊。"""
    owner: dict[tuple[int, int], int] = {}
    parent = list(range(len(ids)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for index, a, b in _edges(ids):
        key = (min(a, b), max(a, b))
        if key in owner:
            first, second = root(owner[key]), root(index)
            parent[max(first, second)] = min(first, second)
        else:
            owner[key] = index
    groups: dict[int, list[int]] = {}
    for index in range(len(ids)):
        groups.setdefault(root(index), []).append(index)
    return list(groups.values())


def _orientation_problems(pieces: Sequence[Sequence[LayerTriangle]], inward: bool,
                          where: str, weld_m: float) -> list[Problem]:
    """每一塊各自判朝向：外殼要朝房內（體積為負）而且只能有一塊，房內實體每一塊都要朝外（體積為正）。"""
    problems = []
    if inward and len(pieces) > 1:
        problems.append(Problem("shell_pieces", True, f"{where}分成 {len(pieces)} 塊互不相連的部分；"
                                "外殼只能是一整塊，柱子、造型請放到 TREAT 圖層"))
    for members in pieces:
        volume = signed_volume(members)
        area = float(np.linalg.norm(_cross(_corners(members)), axis=1).sum() / 2.0)
        layers = "、".join(sorted({t.layer for t in members}))
        if abs(volume) <= weld_m * area:
            problems.append(Problem("zero_volume", True, f"{where}有一塊（{layers}）圍出來的體積是零，"
                                    "看起來不是一個封閉的空間"))
        elif (volume < 0) != inward:
            side = "房間裡面" if inward else "外面（空氣那一側）"
            problems.append(Problem("faces_flipped", True, f"{where}有一塊（{layers}）的面整個翻反了："
                                    f"每個面應該朝{side}"))
    return problems


def _check_mesh(triangles: Sequence[LayerTriangle], inward: bool, where: str,
                weld_m: float) -> tuple[list[Problem], list[list[LayerTriangle]]]:
    """封閉、相鄰一致都過了，才判每一塊的整體朝向；回傳同一組連通塊供內外判斷。"""
    ids, wide_layers = _weld(triangles, weld_m)
    pieces = [[triangles[i] for i in piece] for piece in _components(ids)]
    problems = _edge_problems(triangles, ids, where)
    if wide_layers:
        problems.append(Problem("weld_spread", True, f"{where}有些頂點彼此差一點點但沒對齊，"
                                f"在圖層：{'、'.join(sorted(wide_layers))}。請在 Rhino 用焊接或對齊頂點修正"))
    if not problems:
        problems.extend(_orientation_problems(pieces, inward, where, weld_m))
    return problems, pieces


def _parity(points: FloatArray, corners: FloatArray, direction: FloatArray,
            distance_tol: float) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """沿單位方向射線，分別回傳每個點的奇偶票與這一票是否乾淨。"""
    edge1, edge2 = corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
    normals = np.cross(edge1, edge2)
    twice_area = np.linalg.norm(normals, axis=1)
    h = np.cross(direction, edge2)
    det = np.einsum("ij,ij->i", edge1, h)
    # 重心座標和夾角餘弦都是無因次量，留捨入餘裕才不會把邊上的點當成穩定交點。
    slack = 1e-10
    usable = np.abs(det) > slack * twice_area
    parallel = ~usable & (twice_area > 0)
    inv = np.divide(1.0, det, out=np.zeros_like(det), where=usable)
    result = np.zeros(len(points), dtype=np.bool_)
    clean = np.ones(len(points), dtype=np.bool_)
    for index, point in enumerate(points):
        s = point - corners[:, 0]
        u = inv * np.einsum("ij,ij->i", s, h)
        q = np.cross(s, edge1)
        v = inv * (q @ direction)
        t = inv * np.einsum("ij,ij->i", edge2, q)
        margin = np.minimum(np.minimum(u, v), 1 - (u + v))
        hits = usable & (margin >= 0) & (t > 0)
        result[index] = int(hits.sum()) % 2 == 1
        near_hit = usable & (margin >= -slack) & (t >= -distance_tol)
        grazing = near_hit & ((margin <= slack) | (t <= distance_tol))
        plane_offset = np.abs(np.einsum("ij,ij->i", s, normals))
        clean[index] = not (grazing.any() or (parallel & (plane_offset <= distance_tol * twice_area)).any())
    return result, clean


def _inside_mask(points: FloatArray, triangles: Sequence[LayerTriangle], weld_m: float) -> NDArray[np.bool_]:
    """每個點各自取前三條乾淨射線的多數決。"""
    corners = _corners(triangles)
    count = np.zeros(len(points), dtype=np.int64)
    votes, all_votes = np.zeros_like(count), np.zeros_like(count)
    # 這裡只辨認幾乎從面上出發的射線；距離取焊點容差的萬分之一，保留真正貼牆的判斷尺度。
    distance_tol = weld_m * 1e-4
    for direction in _RAY_DIRECTIONS:
        pending = np.flatnonzero(count < 3)
        if not pending.size:
            break
        unit = np.array(direction) / np.linalg.norm(direction)
        inside, clean = _parity(points[pending], corners, unit, distance_tol)
        all_votes[pending] += inside
        votes[pending] += inside & clean
        count[pending] += clean
    # 極端情況：該點每個方向都不乾淨，才退回全方向的原始奇偶票；有乾淨票就只算那些票。
    return np.where(count > 0, 2 * votes > count, 2 * all_votes > len(_RAY_DIRECTIONS))


def _on_surface_mask(points: FloatArray, triangles: Sequence[LayerTriangle], tolerance_m: float) -> NDArray[np.bool_]:
    """每個點是不是貼在網格表面上（離某個面在容差內、而且投影落在那個面裡）。"""
    corners = _corners(triangles)
    normals = _cross(corners)
    lengths = np.linalg.norm(normals, axis=1)
    keep = lengths > 0
    corners, normals = corners[keep], normals[keep] / lengths[keep, None]
    edge1, edge2 = corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
    d00, d01, d11 = (np.einsum("ij,ij->i", x, y) for x, y in ((edge1, edge1), (edge1, edge2), (edge2, edge2)))
    denominator = d00 * d11 - d01 * d01
    result = np.zeros(len(points), dtype=np.bool_)
    for index, point in enumerate(points):
        offset = point - corners[:, 0]
        distance = np.abs(np.einsum("ij,ij->i", offset, normals))
        d20, d21 = np.einsum("ij,ij->i", offset, edge1), np.einsum("ij,ij->i", offset, edge2)
        v = (d11 * d20 - d01 * d21) / denominator
        w = (d00 * d21 - d01 * d20) / denominator
        slack = 1e-6
        result[index] = bool(((distance <= tolerance_m) & (v >= -slack) & (w >= -slack) & (v + w <= 1 + slack)).any())
    return result


def _placement_problems(room: RoomShell, weld_m: float) -> list[Problem]:
    """喇叭、座位：數量對、朝向有長度、位置在外殼裡。"""
    problems = []
    if not room.speakers:
        problems.append(Problem("no_speaker", True, "模型裡沒有喇叭元件（元件名用 SPK_ 開頭）"))
    main_count = sum(1 for seat in room.seats if seat.kind == MAIN_SEAT)
    if main_count != 1:
        problems.append(Problem("main_seat_count", True, f"主聽位（SEAT_MAIN）要剛好一個，現在有 {main_count} 個"))
    placements = (*room.speakers, *room.seats)
    for item in placements:
        if math.hypot(*item.facing) == 0.0:
            problems.append(Problem("no_facing", True, f"{_label(item)}沒有朝向"))
    if not placements or not room.shell:
        return problems
    points = np.array([item.position for item in placements], dtype=np.float64)
    on_wall = _on_surface_mask(points, room.shell, weld_m)
    inside_shell = _inside_mask(points, room.shell, weld_m) & ~on_wall
    for item, inside, flush in zip(placements, inside_shell, on_wall, strict=True):
        # 嵌牆喇叭的聲學中心本來就在牆面上，貼著外殼也算數；其餘要真的在房內。
        if not (inside or (flush and item.kind == IN_WALL_SPEAKER)):
            problems.append(Problem("placement_outside", True, f"{_label(item)}不在房間裡面"))
    return problems


def _label(item: Placement) -> str:
    x, y, z = item.position
    name = "喇叭" if item.kind in SPEAKER_KINDS else "座位"
    return f"{name} {item.kind}（x {x:.2f}、y {y:.2f}、z {z:.2f} 公尺）"


def _solid_problems(room: RoomShell, settings: RoomShellImportSettings) -> list[Problem]:
    """每個造型自己封閉、朝外、而且整塊在外殼裡（貼著外殼的頂點也算在裡面）。"""
    problems = []
    placements = (*room.speakers, *room.seats)
    centers = np.array([item.position for item in placements], dtype=np.float64)
    for solid in room.solids:
        where = f"造型「{solid.name}」"
        if not solid.triangles:
            # 其他匯入入口也能直接建立造型，空造型沒有任何面可供驗證。
            problems.append(Problem("empty_solid", True, f"{where}沒有面，請在 Rhino 補上封閉網格"))
            continue
        mesh_problems, pieces = _check_mesh(solid.triangles, False, where, settings.weld_m)
        problems.extend(mesh_problems)
        if not room.shell:
            continue
        # 一個造型可以含多個封閉塊；重疊區在兩塊材料裡，分塊判再取聯集才不會把交點抵消。
        buried = np.zeros(len(placements), dtype=np.bool_)
        for piece in pieces:
            buried |= _inside_mask(centers, piece, settings.weld_m)
        for item, inside_solid in zip(placements, buried, strict=True):
            if inside_solid:
                problems.append(Problem("placement_in_solid", True, f"{_label(item)}在{where}裡面"))
        points = _corners(solid.triangles).reshape(-1, 3)
        inside = _inside_mask(points, room.shell, settings.weld_m) | _on_surface_mask(points, room.shell, settings.weld_m)
        if not bool(inside.all()):
            problems.append(Problem("solid_outside", True, f"{where}有一部分跑到房間外面"))
    return problems


def _tiny_problems(triangles: Sequence[LayerTriangle], min_area_m2: float) -> list[Problem]:
    """太小的三角形只提醒不擋：切網格時會重切。"""
    areas = np.linalg.norm(_cross(_corners(triangles)), axis=1) / 2.0
    layers = sorted({t.layer for t, area in zip(triangles, areas, strict=True) if area < min_area_m2})
    if not layers:
        return []
    count = int((areas < min_area_m2).sum())
    return [Problem("tiny_faces", False, f"有 {count} 個面小於 {min_area_m2 * 1e4:g} 平方公分，"
                    f"在這些圖層：{'、'.join(layers)}（不影響計算，只是提醒）")]


def plane_groups(triangles: Sequence[LayerTriangle], angle_deg: float, offset_m: float) -> tuple[PlaneGroup, ...]:
    """把同一個平面上的三角形合成一組；大的先分組，同平面不同圖層合在一起。"""
    corners = _corners(triangles)
    cross = _cross(corners)
    areas = np.linalg.norm(cross, axis=1) / 2.0
    cos_limit = math.cos(math.radians(angle_deg))
    centers = corners.mean(axis=1)
    groups: list[tuple[FloatArray, FloatArray, list[int]]] = []
    for index in np.argsort(-areas, kind="stable"):
        if areas[index] == 0.0:
            continue
        normal = cross[index] / (2.0 * areas[index])
        # 距離量的是這個三角形中心到那一組平面的距離，不是兩邊各自的 n·p 相減：
        # 後者離原點越遠，同樣的角度差就差越多，結果會隨模型擺在哪裡而變。
        for group_normal, group_point, members in groups:
            if (float(group_normal @ normal) >= cos_limit
                    and abs(float(group_normal @ (centers[index] - group_point))) <= offset_m):
                members.append(int(index))
                break
        else:
            groups.append((normal, centers[index], [int(index)]))
    result = []
    for normal, _, members in groups:
        weights = areas[members]
        centroid = (corners[members].mean(axis=1) * weights[:, None]).sum(axis=0) / weights.sum()
        layers = tuple(sorted({triangles[i].layer for i in members}))
        result.append(PlaneGroup(layers, _vec(normal), _vec(centroid), float(weights.sum())))
    return tuple(result)


def _vec(values: FloatArray) -> Vec3:
    return (float(values[0]), float(values[1]), float(values[2]))


def parallel_pairs(groups: Sequence[PlaneGroup], angle_deg: float, min_area_m2: float) -> tuple[ParallelPair, ...]:
    """兩個夠大、朝向彼此（法線相反、各在對方前面）、夾角在界內的平面。"""
    big = [group for group in groups if group.area_m2 >= min_area_m2]
    pairs = []
    for i, first in enumerate(big):
        for second in big[i + 1:]:
            n1, n2 = np.array(first.normal), np.array(second.normal)
            angle = math.degrees(math.acos(max(-1.0, min(1.0, float(-(n1 @ n2))))))
            if angle > angle_deg:
                continue
            between = np.array(second.centroid) - np.array(first.centroid)
            if float(n1 @ between) > 0 and float(n2 @ -between) > 0:
                pairs.append(ParallelPair(first, second, angle, float(n1 @ between)))
    return tuple(pairs)


def check_room_shell(room: RoomShell, settings: RoomShellImportSettings) -> ImportReport:
    """把整間房跑過一次匯入檢查，回問題清單與事實。"""
    problems: list[Problem] = []
    if room.units != "m":
        problems.append(Problem("units", True, f"模型單位是「{room.units}」，要用公尺"))
    if not room.shell:
        problems.append(Problem("no_shell", True, "模型裡沒有房間外殼（ROOM 圖層底下沒有面）"))
    problems.extend(Problem("skipped", False, f"略過：{note}") for note in room.skipped)
    if room.unreadable:
        problems.append(Problem("unreadable", True, f"這些物件讀不了：{'；'.join(room.unreadable)}"))
    every = (*room.shell, *(t for solid in room.solids for t in solid.triangles))
    # 先驗原始座標；非有限數會讓焊點的空間索引拋例外，也會讓射線、面積與方向的比較失去意義。
    invalid_faces = np.flatnonzero(~np.isfinite(_corners(every)).all(axis=(1, 2)))
    if invalid_faces.size:
        layers = sorted({every[i].layer or "無材料子圖層" for i in invalid_faces})
        problems.append(Problem("bad_coordinates", True, f"這些圖層的面有無效的頂點座標：{'、'.join(layers)}"))
    bad_placements = [item for item in (*room.speakers, *room.seats)
                      if not all(math.isfinite(x) for x in (*item.position, *item.facing))]
    problems.extend(Problem("bad_placement", True, f"{_label(item)}的位置或朝向含無效數值") for item in bad_placements)
    if bad_placements or invalid_faces.size:
        return ImportReport(tuple(problems), (), 0, ())
    if room.shell:
        shell_problems, _ = _check_mesh(room.shell, True, "房間外殼", settings.weld_m)
        problems.extend(shell_problems)
    problems.extend(_solid_problems(room, settings))
    problems.extend(_placement_problems(room, settings.weld_m))
    problems.extend(_tiny_problems(every, settings.min_triangle_area_m2) if every else [])
    groups = plane_groups(every, settings.coplanar_angle_deg, settings.coplanar_offset_m)
    if len(groups) > settings.max_planes:
        problems.append(Problem("too_many_planes", True, f"合併同一平面之後還有 {len(groups)} 個面，"
                                f"上限是 {settings.max_planes} 個；請在 Rhino 裡簡化模型"))
    unnamed = sum(1 for triangle in every if not triangle.layer)
    if unnamed:
        problems.append(Problem("no_material_layer", True, f"有 {unnamed} 個面直接放在 ROOM 或 TREAT 上層，"
                                "沒有放進材料子圖層，網頁上沒辦法替它們選材料"))
    areas: dict[str, float] = {}
    for triangle, area in zip(every, np.linalg.norm(_cross(_corners(every)), axis=1) / 2.0, strict=True):
        areas[triangle.layer] = areas.get(triangle.layer, 0.0) + float(area)
    pairs = parallel_pairs(groups, settings.parallel_angle_deg, settings.parallel_min_area_m2)
    return ImportReport(tuple(problems), tuple(sorted(areas.items())), len(groups), pairs)
