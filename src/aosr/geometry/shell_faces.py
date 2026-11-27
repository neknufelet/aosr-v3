"""房間殼的「面」與「看不看得到」：高音那一路（鏡像法、晚期能量交換）共用的純幾何。

**面是什麼。** 外殼與房內造型的全部三角形，照「同一個平面」分組（同法線、同位置，跨材料圖層
也算同一面）：鏡像法對每一個**平面**做一次鏡射，反射點落在該平面的哪一個三角形上，就用那個
三角形的材料圖層。法線一律朝空氣那一側（:mod:`aosr.geometry.room_shell` 的約定）。

**看不看得到。** 一條線段被擋，是指它從中間穿過某個三角形（兩端剛好在面上的不算）。外殼躺在
「整個房間都在它同一側」的平面上的三角形擋不到任何房內的線段，不拿來擋；只有造型，以及外殼凹進去、
凸出來的那幾面才拿來擋（外殼是凸的時候就只剩造型）。
繞過板子邊緣的繞射不算（計畫第 9 條，已知限制）。

這一支只做純幾何、只用 numpy；不讀檔、不碰材料與物理。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from aosr.geometry.room_shell import LayerTriangle, RoomShell, _inside_mask


FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]
BoolArray: TypeAlias = NDArray[np.bool_]

# 一次丟進向量化交點計算的「線段 × 三角形」對數上限；只管記憶體，不影響答案。
_PAIRS_PER_CHUNK = 400_000
# 重心座標是無因次量，留一點捨入餘裕，剛好擦過邊的點才不會兩邊都不算。
_BARYCENTRIC_SLACK = 1e-9


@dataclass(frozen=True)
class Occluders:
    """會擋線段的三角形，一個平面一組（造型的每一面、外殼凹凸的每一面），每組帶外接盒與平面序號。

    先用外接盒篩：線段碰不到盒子就不必跟那一組的三角形一個一個比，答案不變、只是快。
    平面序號讓線段跳過「自己兩端所在的那個平面」：從一個平面出發的線段不可能再穿過同一個平面。
    """

    groups: tuple[FloatArray, ...]
    boxes: FloatArray
    planes: IntArray

    @property
    def triangle_count(self) -> int:
        """全部組加起來的三角形數。"""
        return sum(len(group) for group in self.groups)


def make_occluders(groups: tuple[tuple[int, FloatArray], ...]) -> Occluders:
    """把幾組（平面序號, 三角形）包成 :class:`Occluders`；空的組丟掉。"""
    kept = tuple((plane, group) for plane, group in groups if len(group))
    boxes = np.array([(group.reshape(-1, 3).min(axis=0), group.reshape(-1, 3).max(axis=0)) for _plane, group in kept],
                     dtype=np.float64).reshape(-1, 2, 3)
    planes = np.array([plane for plane, _group in kept], dtype=np.int64)
    return Occluders(tuple(group for _plane, group in kept), boxes, planes)


@dataclass(frozen=True)
class ShellFaces:
    """整間房的面：三角形、所屬材料圖層、所屬平面，以及拿來擋線段的三角形。

    ``layers`` 是排序後的材料圖層名，``layer_of`` 是每個三角形在 ``layers`` 裡的序號；
    ``plane_of`` 是每個三角形所屬平面；平面 ``p`` 是 ``plane_normals[p]·x = plane_offsets[p]``，
    法線朝空氣。``occluders`` 是會擋線段的三角形（造型全部；外殼只有不在支撐平面上的那些）。
    同一個平面只收同一個物件（外殼、或某一個造型）的三角形。
    ``corners`` 前 ``shell_triangles`` 個是外殼，其餘是房內造型；``supporting`` 是每個平面是不是
    整個外殼都在它朝空氣那一側（:func:`_supporting_planes`）。
    """

    corners: FloatArray
    layers: tuple[str, ...]
    layer_of: IntArray
    plane_of: IntArray
    plane_normals: FloatArray
    plane_offsets: FloatArray
    occluders: Occluders
    shell_convex: bool
    shell_triangles: int
    supporting: BoolArray

    @property
    def plane_count(self) -> int:
        """平面數。"""
        return int(self.plane_normals.shape[0])

    def triangle_areas(self) -> FloatArray:
        """每個三角形的面積。"""
        return triangle_areas(self.corners)

    def layer_areas(self) -> FloatArray:
        """每個材料圖層的總面積（照 ``layers`` 的順序）。"""
        areas = np.zeros(len(self.layers), dtype=np.float64)
        np.add.at(areas, self.layer_of, self.triangle_areas())
        return areas


def corner_array(triangles: tuple[LayerTriangle, ...]) -> FloatArray:
    """(N, 3, 3)：每個三角形的三個頂點。"""
    return np.array([(t.a, t.b, t.c) for t in triangles], dtype=np.float64).reshape(-1, 3, 3)


def triangle_areas(corners: FloatArray) -> FloatArray:
    """每個三角形的面積。"""
    cross = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    return np.asarray(np.linalg.norm(cross, axis=1) / 2.0, dtype=np.float64)


def _unit_normals(corners: FloatArray) -> tuple[FloatArray, FloatArray]:
    cross = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    twice = np.linalg.norm(cross, axis=1)
    normals = np.divide(cross, twice[:, None], out=np.zeros_like(cross), where=twice[:, None] > 0)
    return normals, twice / 2.0


def _group_planes(corners: FloatArray, objects: IntArray, angle_deg: float, offset_m: float) -> IntArray:
    """大的三角形先開組；夾角與到該組平面的距離都在界內就併進去（同 room_shell.plane_groups 的判法）。

    只在同一個物件（外殼、或同一個造型）裡面合併：貼牆幾毫米的薄板正面若跟牆併成一面，擬合出來的
    平面會落在板子正面後面，反射點就被板子自己擋掉。
    """
    normals, areas = _unit_normals(corners)
    centers = corners.mean(axis=1)
    cos_limit = math.cos(math.radians(angle_deg))
    group_normals: list[FloatArray] = []
    group_points: list[FloatArray] = []
    group_objects: list[int] = []
    plane_of = np.full(len(corners), -1, dtype=np.int64)
    for index in np.argsort(-areas, kind="stable"):
        if areas[index] == 0.0:
            raise ValueError("房間殼裡有面積為零的三角形，請先過匯入檢查")
        for group, (normal, point, owner) in enumerate(zip(group_normals, group_points, group_objects, strict=True)):
            if (owner == objects[index] and float(normal @ normals[index]) >= cos_limit
                    and abs(float(normal @ (centers[index] - point))) <= offset_m):
                plane_of[index] = group
                break
        else:
            plane_of[index] = len(group_normals)
            group_normals.append(normals[index])
            group_points.append(centers[index])
            group_objects.append(int(objects[index]))
    return plane_of


def _fit_planes(corners: FloatArray, plane_of: IntArray, count: int) -> tuple[FloatArray, FloatArray]:
    """每個平面的法線取成員面積加權平均、位置取面積加權重心；畫得剛好共面時就是原平面。"""
    normals, areas = _unit_normals(corners)
    weighted = np.zeros((count, 3), dtype=np.float64)
    np.add.at(weighted, plane_of, normals * areas[:, None])
    plane_normals = weighted / np.linalg.norm(weighted, axis=1)[:, None]
    center_sum = np.zeros((count, 3), dtype=np.float64)
    np.add.at(center_sum, plane_of, corners.mean(axis=1) * areas[:, None])
    area_sum = np.zeros(count, dtype=np.float64)
    np.add.at(area_sum, plane_of, areas)
    offsets = np.einsum("ij,ij->i", plane_normals, center_sum / area_sum[:, None])
    return plane_normals, np.asarray(offsets, dtype=np.float64)


def _supporting_planes(shell: FloatArray, normals: FloatArray, offsets: FloatArray, tol_m: float) -> BoolArray:
    """每個平面：外殼全部頂點都在它朝空氣那一側（或面上）嗎。

    是的話，房內任兩點的連線不可能從中間穿過這個平面（兩端都在同一側），躺在它上面的三角形
    就永遠擋不到東西；外殼每個平面都是這樣，外殼就是凸的。
    """
    points = np.unique(shell.reshape(-1, 3), axis=0)
    side = points @ normals.T - offsets[None, :]
    return np.asarray((side >= -tol_m).all(axis=0), dtype=np.bool_)


def shell_faces(room: RoomShell, angle_deg: float, offset_m: float, tol_m: float) -> ShellFaces:
    """把房間殼整理成高音那一路要的面；共平面界線用匯入檢查同一組設定。"""
    shell = corner_array(room.shell)
    solids = tuple(t for solid in room.solids for t in solid.triangles)
    solid_corners = corner_array(solids)
    corners = np.concatenate((shell, solid_corners)) if len(solids) else shell
    every = (*room.shell, *solids)
    if not len(room.shell):
        raise ValueError("房間殼沒有外殼的面")
    if not np.isfinite(corners).all():
        raise ValueError("房間殼有無效的頂點座標，請先過匯入檢查")
    layers = tuple(sorted({t.layer for t in every}))
    layer_of = np.array([layers.index(t.layer) for t in every], dtype=np.int64)
    objects = np.concatenate([np.zeros(len(shell), dtype=np.int64)]
                             + [np.full(len(solid.triangles), number, dtype=np.int64)
                                for number, solid in enumerate(room.solids, start=1)])
    plane_of = _group_planes(corners, objects, angle_deg, offset_m)
    count = int(plane_of.max()) + 1
    plane_normals, plane_offsets = _fit_planes(corners, plane_of, count)
    supporting = _supporting_planes(shell, plane_normals, plane_offsets, offset_m)
    convex = bool(supporting[np.unique(plane_of[:len(shell)])].all())
    # 擋得到東西的：造型的全部三角形，加上外殼躺在「非支撐平面」上的那些（凹進去、凸出來的部分）。
    shell_planes = set(plane_of[:len(shell)].tolist())
    groups = [(int(plane), corners[plane_of == plane]) for plane in np.unique(plane_of)
              if int(plane) not in shell_planes or not supporting[plane]]
    return ShellFaces(corners, layers, layer_of, plane_of, plane_normals, plane_offsets,
                      make_occluders(tuple(groups)), convex, len(shell), supporting)


def points_inside_shell(points: FloatArray, faces: ShellFaces, tol_m: float) -> BoolArray:
    """每個點是不是在外殼裡面、而且離外殼每一面都超過 ``tol_m``。

    先看支撐平面（整個外殼都在它同一側的那些）：在任何一個的背面或貼著它就不在房內。外殼不是凸的
    時候再用匯入檢查同一支射線奇偶判斷補上凹進去的那幾塊。
    """
    shell_planes = np.unique(faces.plane_of[:faces.shell_triangles])
    support = shell_planes[faces.supporting[shell_planes]]
    heights = points @ faces.plane_normals[support].T - faces.plane_offsets[support][None, :]
    inside = np.asarray((heights > tol_m).all(axis=1), dtype=np.bool_)
    if faces.shell_convex or not inside.any():
        return inside
    triangles = [LayerTriangle("", tuple(a), tuple(b), tuple(c)) for a, b, c in faces.corners[:faces.shell_triangles]]
    return np.asarray(inside & _inside_mask(points, triangles, tol_m), dtype=np.bool_)


def locate_on_planes(points: FloatArray, planes: IntArray, faces: ShellFaces, tol_m: float) -> IntArray:
    """每個點（已在它那個平面上）落在該平面的哪個三角形裡；不在任何一個裡回 -1。

    重心座標的餘裕換成長度：邊長乘無因次餘裕再加上長度容差，剛好落在兩個三角形共邊上的點兩邊都算，
    取序號小的那一個。
    """
    result = np.full(len(points), -1, dtype=np.int64)
    for plane in np.unique(planes):
        members = np.flatnonzero(faces.plane_of == plane)
        which = np.flatnonzero(planes == plane)
        tri = faces.corners[members]
        e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
        d00 = np.einsum("ij,ij->i", e1, e1)
        d01 = np.einsum("ij,ij->i", e1, e2)
        d11 = np.einsum("ij,ij->i", e2, e2)
        denominator = d00 * d11 - d01 * d01
        offset = points[which][:, None, :] - tri[None, :, 0, :]
        d20 = np.einsum("pti,ti->pt", offset, e1)
        d21 = np.einsum("pti,ti->pt", offset, e2)
        v = (d11 * d20 - d01 * d21) / denominator
        w = (d00 * d21 - d01 * d20) / denominator
        slack = _BARYCENTRIC_SLACK + tol_m / np.sqrt(np.maximum(d00, d11))
        inside = (v >= -slack) & (w >= -slack) & (v + w <= 1.0 + slack)
        hit = inside.any(axis=1)
        first = np.argmax(inside, axis=1)
        result[which[hit]] = members[first[hit]]
    return result


def _boxes_touched(starts: FloatArray, ends: FloatArray, box: FloatArray, tol_m: float) -> BoolArray:
    """每條線段碰不碰得到外接盒（盒子各邊放大 ``tol_m``；slab 法）。"""
    direction = ends - starts
    low, high = box[0] - tol_m, box[1] + tol_m
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (low[None, :] - starts) / direction
        t2 = (high[None, :] - starts) / direction
    parallel = direction == 0.0
    inside = (starts >= low[None, :]) & (starts <= high[None, :])
    near = np.where(parallel, np.where(inside, -np.inf, np.inf), np.minimum(t1, t2))
    far = np.where(parallel, np.where(inside, np.inf, -np.inf), np.maximum(t1, t2))
    enter = np.maximum(near.max(axis=1), 0.0)
    leave = np.minimum(far.min(axis=1), 1.0)
    return np.asarray(enter <= leave, dtype=np.bool_)


def segments_hit_triangles(starts: FloatArray, ends: FloatArray, triangles: FloatArray, tol_m: float) -> BoolArray:
    """Möller–Trumbore：每條線段（含兩端外 ``tol_m`` 以內）有沒有碰到這幾個三角形之一。

    兩端剛好落在別的面上也算碰到（例如落地柱子底面正下方的地板點）；線段自己兩端所在的平面由
    呼叫端先排除，不靠這裡的容差。
    """
    blocked = np.zeros(len(starts), dtype=np.bool_)
    e1 = triangles[:, 1] - triangles[:, 0]
    e2 = triangles[:, 2] - triangles[:, 0]
    twice_area = np.linalg.norm(np.cross(e1, e2), axis=1)
    step = max(1, _PAIRS_PER_CHUNK // len(triangles))
    for begin in range(0, len(starts), step):
        a, b = starts[begin:begin + step], ends[begin:begin + step]
        direction = b - a
        length = np.linalg.norm(direction, axis=1)
        h = np.cross(direction[:, None, :], e2[None, :, :])
        det = np.einsum("tk,stk->st", e1, h)
        usable = np.abs(det) > _BARYCENTRIC_SLACK * twice_area[None, :] * length[:, None]
        inv = np.divide(1.0, det, out=np.zeros_like(det), where=usable)
        s = a[:, None, :] - triangles[None, :, 0, :]
        u = inv * np.einsum("stk,stk->st", s, h)
        q = np.cross(s, e1[None, :, :])
        v = inv * np.einsum("stk,sk->st", q, direction)
        t = inv * np.einsum("tk,stk->st", e2, q)
        margin = (tol_m / np.maximum(length, tol_m))[:, None]
        hit = (usable & (u >= -_BARYCENTRIC_SLACK) & (v >= -_BARYCENTRIC_SLACK)
               & (u + v <= 1.0 + _BARYCENTRIC_SLACK) & (t > -margin) & (t < 1.0 + margin))
        blocked[begin:begin + step] = hit.any(axis=1)
    return blocked


def segments_blocked(starts: FloatArray, ends: FloatArray, occluders: Occluders, tol_m: float,
                     start_planes: IntArray | None = None, end_planes: IntArray | None = None) -> BoolArray:
    """每條線段是不是穿過（或一端貼著）任何一個擋的三角形。

    ``start_planes``／``end_planes`` 是每條線段兩端所在的平面（聲源、接收點填 -1）；那兩個平面的三角形
    不拿來擋這一條（從平面上出發的線段不會再穿過同一個平面）。跟三角形共平面的線段不算擋。
    """
    blocked = np.zeros(len(starts), dtype=np.bool_)
    none = np.full(len(starts), -1, dtype=np.int64)
    first = none if start_planes is None else start_planes
    last = none if end_planes is None else end_planes
    for group, box, plane in zip(occluders.groups, occluders.boxes, occluders.planes, strict=True):
        candidates = np.flatnonzero(~blocked & (first != plane) & (last != plane))
        if not len(candidates):
            continue
        candidates = candidates[_boxes_touched(starts[candidates], ends[candidates], box, tol_m)]
        if len(candidates):
            blocked[candidates] = segments_hit_triangles(starts[candidates], ends[candidates], group, tol_m)
    return blocked
