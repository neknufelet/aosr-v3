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
實體與外殼逐面穿插、外殼凹凸。
"""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from aosr.config.room_shell_import import RoomShellImportSettings


Vec3: TypeAlias = tuple[float, float, float]
FloatArray: TypeAlias = NDArray[np.float64]

SPEAKER_KINDS: tuple[str, ...] = ("SPK_BOOKSHELF", "SPK_FLOOR", "SPK_INWALL")
MAIN_SEAT = "SEAT_MAIN"
SEAT_KINDS: tuple[str, ...] = (MAIN_SEAT, "SEAT")
# 射線法判斷點在不在封閉網格裡用的方向：三個分量刻意不整齊，避免剛好擦過整齊的邊與頂點。
_RAY_DIRECTION: Vec3 = (0.5773, 0.6129, 0.5397)


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
    return float(np.einsum("ij,ij->i", corners[:, 0], np.cross(corners[:, 1], corners[:, 2])).sum() / 6.0)


def _edge_problems(triangles: Sequence[LayerTriangle], weld_m: float, where: str) -> list[Problem]:
    """封閉與朝向一致：每條邊恰好被兩個三角形用、而且兩次方向相反。"""
    keys = np.rint(_corners(triangles) / weld_m).astype(np.int64)
    undirected: Counter[tuple[tuple[int, ...], tuple[int, ...]]] = Counter()
    directed: Counter[tuple[tuple[int, ...], tuple[int, ...]]] = Counter()
    open_layers: set[str] = set()
    flipped_layers: set[str] = set()
    for triangle, corner in zip(triangles, keys, strict=True):
        points = [tuple(int(v) for v in corner[i]) for i in range(3)]
        for start, end in ((points[0], points[1]), (points[1], points[2]), (points[2], points[0])):
            undirected[(min(start, end), max(start, end))] += 1
            directed[(start, end)] += 1
    for triangle, corner in zip(triangles, keys, strict=True):
        points = [tuple(int(v) for v in corner[i]) for i in range(3)]
        for start, end in ((points[0], points[1]), (points[1], points[2]), (points[2], points[0])):
            if undirected[(min(start, end), max(start, end))] != 2:
                open_layers.add(triangle.layer)
            elif directed[(start, end)] != 1:
                flipped_layers.add(triangle.layer)
    problems = []
    if open_layers:
        bad = sum(1 for count in undirected.values() if count != 2)
        problems.append(Problem("not_closed", True, f"{where}沒有封起來：有 {bad} 條邊沒有剛好接兩個面"
                                f"（破洞或多出來的面），在這些圖層附近：{'、'.join(sorted(open_layers))}"))
    if flipped_layers:
        problems.append(Problem("inconsistent_faces", True, f"{where}有面翻反了：相鄰兩個面朝向不一致，"
                                f"在這些圖層：{'、'.join(sorted(flipped_layers))}"))
    return problems


def _orientation_problem(triangles: Sequence[LayerTriangle], inward: bool, where: str) -> list[Problem]:
    """整個網格朝向對不對：外殼要朝房內（體積為負），房內實體要朝外（體積為正）。"""
    volume = signed_volume(triangles)
    if volume == 0.0:
        return [Problem("zero_volume", True, f"{where}圍出來的體積是零，看起來不是一個封閉的空間")]
    if (volume < 0) != inward:
        side = "房間裡面" if inward else "外面（空氣那一側）"
        return [Problem("faces_flipped", True, f"{where}的面整個翻反了：每個面應該朝{side}")]
    return []


def _inside_mask(points: FloatArray, triangles: Sequence[LayerTriangle]) -> NDArray[np.bool_]:
    """每個點在不在封閉網格裡（射線穿過奇數個面就在裡面）。"""
    corners = _corners(triangles)
    direction = np.array(_RAY_DIRECTION, dtype=np.float64)
    edge1, edge2 = corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
    h = np.cross(direction, edge2)
    det = np.einsum("ij,ij->i", edge1, h)
    usable = np.abs(det) > 1e-15
    inv = np.where(usable, 1.0 / np.where(usable, det, 1.0), 0.0)
    result = np.zeros(len(points), dtype=np.bool_)
    for index, point in enumerate(points):
        s = point - corners[:, 0]
        u = inv * np.einsum("ij,ij->i", s, h)
        q = np.cross(s, edge1)
        v = inv * (q @ direction)
        t = inv * np.einsum("ij,ij->i", edge2, q)
        hits = usable & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0)
        result[index] = int(hits.sum()) % 2 == 1
    return result


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
    """喇叭、座位：數量對、朝向有長度、位置在外殼裡、不在任何造型裡。"""
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
    inside_shell = _inside_mask(points, room.shell) & ~_on_surface_mask(points, room.shell, weld_m)
    for item, inside in zip(placements, inside_shell, strict=True):
        if not inside:
            problems.append(Problem("placement_outside", True, f"{_label(item)}不在房間裡面"))
    for solid in room.solids:
        for item, inside in zip(placements, _inside_mask(points, solid.triangles), strict=True):
            if inside:
                problems.append(Problem("placement_in_solid", True, f"{_label(item)}在造型「{solid.name}」裡面"))
    return problems


def _label(item: Placement) -> str:
    x, y, z = item.position
    name = "喇叭" if item.kind in SPEAKER_KINDS else "座位"
    return f"{name} {item.kind}（x {x:.2f}、y {y:.2f}、z {z:.2f} 公尺）"


def _solid_problems(room: RoomShell, settings: RoomShellImportSettings) -> list[Problem]:
    """每個造型自己封閉、朝外、而且整塊在外殼裡（貼著外殼的頂點也算在裡面）。"""
    problems = []
    for solid in room.solids:
        where = f"造型「{solid.name}」"
        own = _edge_problems(solid.triangles, settings.weld_m, where)
        problems.extend(own or _orientation_problem(solid.triangles, False, where))
        if not room.shell:
            continue
        points = _corners(solid.triangles).reshape(-1, 3)
        inside = _inside_mask(points, room.shell) | _on_surface_mask(points, room.shell, settings.weld_m)
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
    groups: list[tuple[FloatArray, float, list[int]]] = []
    for index in np.argsort(-areas, kind="stable"):
        if areas[index] == 0.0:
            continue
        normal = cross[index] / (2.0 * areas[index])
        offset = float(normal @ corners[index, 0])
        for group_normal, group_offset, members in groups:
            if float(group_normal @ normal) >= cos_limit and abs(group_offset - offset) <= offset_m:
                members.append(int(index))
                break
        else:
            groups.append((normal, offset, [int(index)]))
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
    else:
        own = _edge_problems(room.shell, settings.weld_m, "房間外殼")
        problems.extend(own or _orientation_problem(room.shell, True, "房間外殼"))
    problems.extend(_solid_problems(room, settings))
    problems.extend(_placement_problems(room, settings.weld_m))
    every = (*room.shell, *(t for solid in room.solids for t in solid.triangles))
    problems.extend(_tiny_problems(every, settings.min_triangle_area_m2) if every else [])
    groups = plane_groups(every, settings.coplanar_angle_deg, settings.coplanar_offset_m)
    if len(groups) > settings.max_planes:
        problems.append(Problem("too_many_planes", True, f"合併同一平面之後還有 {len(groups)} 個面，"
                                f"上限是 {settings.max_planes} 個；請在 Rhino 裡簡化模型"))
    problems.extend(Problem("skipped", False, f"略過：{note}") for note in room.skipped)
    if room.unreadable:
        problems.append(Problem("unreadable", True, "這一版只讀網格（Mesh），這些物件讀不了："
                                f"{'、'.join(room.unreadable)}；請在 Rhino 用 Mesh 指令轉成網格再存"))
    unnamed = sum(1 for triangle in every if not triangle.layer)
    if unnamed:
        problems.append(Problem("no_material_layer", True, f"有 {unnamed} 個面直接放在 ROOM 或 TREAT 上層，"
                                "沒有放進材料子圖層，網頁上沒辦法替它們選材料"))
    areas: dict[str, float] = {}
    for triangle, area in zip(every, np.linalg.norm(_cross(_corners(every)), axis=1) / 2.0, strict=True):
        areas[triangle.layer] = areas.get(triangle.layer, 0.0) + float(area)
    pairs = parallel_pairs(groups, settings.parallel_angle_deg, settings.parallel_min_area_m2)
    return ImportReport(tuple(problems), tuple(sorted(areas.items())), len(groups), pairs)
