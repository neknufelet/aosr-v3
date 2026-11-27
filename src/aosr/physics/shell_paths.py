"""自訂房間（房間殼）的逐面鏡像法：路徑表每間房算一次，每個候選只換材料重算。

**路徑表（只看幾何）。** 從聲源開始對每一個平面鏡射，鏡像源還在那個平面朝空氣一側才往下鏡射
（背面的鏡像不可能有反射）；每一個鏡像源從接收點往回追：反射點要落在那個平面的某個三角形裡，
每一段線段都不能被造型（天雲、側板）或外殼擋住，最後一段要回得到聲源。過得了的才進路徑表，
擋住的不算（計畫第 9 條）。表裡每條路徑記：階數、總長、每一次反彈的材料圖層與入射角餘弦、
離開聲源的方向（給指向性用）、反射點。

**每個候選（只換材料）。** 反射係數 ``R = (Z·cosθ − ρc)/(Z·cosθ + ρc)`` 跟長方體那一路
（:func:`aosr.physics.amplitude.reflection_coefficient`）同一條公式；入射角是那一次反彈真正的
入射角——長方體裡它就等於舊程式「攤開直線那一軸」的定義。路徑壓力 ``(1/d)·ΠR·exp(−iωd/c)``。

長方體的 :mod:`aosr.physics.room_paths` 完全不碰這一支；這一支也不收長方體。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from aosr.config.shell_lane import SHELL_MAX_IMAGE_SOURCES
from aosr.geometry.shell_faces import (
    BoolArray, FloatArray, IntArray, ShellFaces, locate_on_planes, points_inside_shell, segments_blocked,
)


ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class ShellPathTable:
    """一組聲源、接收點在一間房裡的全部有效路徑（直達排第一，其餘照階數、再照平面序列）。

    ``bounce_layer`` 與 ``bounce_cos`` 是 (路徑數, 最高階數)，從聲源那一側數起；沒有那一次
    反彈的格子 layer 是 -1、cos 是 0。``layers`` 是材料圖層名，序號對 ``bounce_layer``。
    """

    layers: tuple[str, ...]
    max_order: int
    source: tuple[float, float, float]
    receiver: tuple[float, float, float]
    order: IntArray
    dist_m: FloatArray
    bounce_layer: IntArray
    bounce_plane: IntArray
    bounce_cos: FloatArray
    bounce_points: FloatArray
    departure: FloatArray
    blocked_count: int

    @property
    def path_count(self) -> int:
        """有效路徑數（含直達）。"""
        return int(self.order.shape[0])


@dataclass(frozen=True)
class _Level:
    """同一階的全部候選鏡像源：每一個的平面序列（聲源側先）與每一層的鏡像源座標。"""

    chains: IntArray
    images: FloatArray


def _mirror(level: _Level, faces: ShellFaces, tol_m: float) -> _Level:
    """對每一個平面鏡射一次；同一平面連續兩次、或鏡像源不在平面朝空氣那一側的都不鏡射。"""
    last = level.images[:, -1, :]
    heights = last @ faces.plane_normals.T - faces.plane_offsets[None, :]
    allowed = heights > tol_m
    if level.chains.shape[1]:
        allowed[np.arange(len(last)), level.chains[:, -1]] = False
    parent, plane = np.nonzero(allowed)
    mirrored = last[parent] - 2.0 * heights[parent, plane][:, None] * faces.plane_normals[plane]
    chains = np.concatenate((level.chains[parent], plane[:, None]), axis=1)
    images = np.concatenate((level.images[parent], mirrored[:, None, :]), axis=1)
    return _Level(np.asarray(chains, dtype=np.int64), np.asarray(images, dtype=np.float64))


@dataclass(frozen=True)
class _Traced:
    valid: BoolArray
    triangles: IntArray
    cos: FloatArray
    points: FloatArray
    blocked: BoolArray


def _trace(level: _Level, receiver: FloatArray, source: FloatArray, faces: ShellFaces, tol_m: float) -> _Traced:
    """從接收點往回追每一個鏡像源：反射點在面裡、每段不被擋，才是一條有效路徑。"""
    count, order = level.chains.shape
    point = np.repeat(receiver[None, :], count, axis=0)
    valid = np.ones(count, dtype=np.bool_)
    blocked = np.zeros(count, dtype=np.bool_)
    triangles = np.full((count, order), -1, dtype=np.int64)
    cos = np.zeros((count, order), dtype=np.float64)
    points = np.zeros((count, order, 3), dtype=np.float64)
    previous = np.full(count, -1, dtype=np.int64)
    for bounce in range(order - 1, -1, -1):
        plane = level.chains[:, bounce]
        image = level.images[:, bounce + 1, :]
        normal = faces.plane_normals[plane]
        toward = image - point
        along = np.einsum("ij,ij->i", normal, toward)
        safe = np.where(np.abs(along) > 0.0, along, 1.0)
        t = (faces.plane_offsets[plane] - np.einsum("ij,ij->i", normal, point)) / safe
        valid &= (np.abs(along) > 0.0) & (t > 0.0) & (t < 1.0)
        hit = point + t[:, None] * toward
        live = np.flatnonzero(valid)
        found = locate_on_planes(hit[live], plane[live], faces, tol_m)
        triangles[live, bounce] = found
        valid[live[found < 0]] = False
        live = np.flatnonzero(valid)
        stopped = segments_blocked(point[live], hit[live], faces.occluders, tol_m, previous[live], plane[live])
        blocked[live[stopped]] = True
        valid[live[stopped]] = False
        cos[:, bounce] = np.abs(along) / np.linalg.norm(toward, axis=1)
        points[:, bounce, :] = hit
        point = np.where(valid[:, None], hit, point)
        previous = plane
    live = np.flatnonzero(valid)
    stopped = segments_blocked(point[live], np.repeat(source[None, :], len(live), axis=0), faces.occluders, tol_m,
                               previous[live], None)
    blocked[live[stopped]] = True
    valid[live[stopped]] = False
    return _Traced(valid, triangles, cos, points, blocked)


def _check_point(name: str, point: tuple[float, float, float], faces: ShellFaces, tol_m: float) -> FloatArray:
    values = np.asarray(point, dtype=np.float64)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError(f"{name}座標必須是三個有限數")
    if not bool(points_inside_shell(values[None, :], faces, tol_m)[0]):
        raise ValueError(f"{name}不在房間裡面（貼在牆上或在牆外）")
    return values


def shell_path_table(
    faces: ShellFaces,
    source: tuple[float, float, float],
    receiver: tuple[float, float, float],
    *,
    max_order: int,
    tol_m: float,
    max_images: int = SHELL_MAX_IMAGE_SOURCES,
) -> ShellPathTable:
    """算一組聲源、接收點的路徑表（直達＋ ``max_order`` 階以內）；直達被擋就說人話停下來。"""
    if max_order < 1:
        raise ValueError(f"反射階數至少要 1，收到 {max_order}")
    src = _check_point("聲源", source, faces, tol_m)
    rcv = _check_point("接收點", receiver, faces, tol_m)
    if bool(segments_blocked(src[None, :], rcv[None, :], faces.occluders, tol_m)[0]):
        raise ValueError("喇叭到聽位的直線被擋住了（房內造型或凹進去的牆），請挪喇叭或聽位")
    level = _Level(np.zeros((1, 0), dtype=np.int64), src[None, None, :].copy())
    rows: list[tuple[int, IntArray, IntArray, FloatArray, FloatArray, FloatArray]] = []
    blocked_count = 0
    total = 0
    for order in range(1, max_order + 1):
        level = _mirror(level, faces, tol_m)
        total += len(level.chains)
        if total > max_images:
            raise ValueError(f"鏡像源超過 {max_images} 個（{faces.plane_count} 個面、{max_order} 階）："
                             "請在 Rhino 把模型簡化（合併小面）或降低反射階數")
        traced = _trace(level, rcv, src, faces, tol_m)
        blocked_count += int(traced.blocked.sum())
        keep = np.flatnonzero(traced.valid)
        sort = np.lexsort(level.chains[keep].T[::-1])
        keep = keep[sort]
        rows.append((order, level.chains[keep], traced.triangles[keep], traced.cos[keep], traced.points[keep],
                     level.images[keep, -1, :]))
    return _assemble(faces, src, rcv, max_order, rows, blocked_count)


def _assemble(faces: ShellFaces, src: FloatArray, rcv: FloatArray, max_order: int,
              rows: list[tuple[int, IntArray, IntArray, FloatArray, FloatArray, FloatArray]],
              blocked_count: int) -> ShellPathTable:
    """直達加各階有效路徑，補齊成 (路徑數, 最高階數) 的表。"""
    total = 1 + sum(len(chains) for _order, chains, *_rest in rows)
    order = np.zeros(total, dtype=np.int64)
    dist = np.zeros(total, dtype=np.float64)
    layer = np.full((total, max_order), -1, dtype=np.int64)
    plane = np.full((total, max_order), -1, dtype=np.int64)
    cos = np.zeros((total, max_order), dtype=np.float64)
    points = np.zeros((total, max_order, 3), dtype=np.float64)
    departure = np.zeros((total, 3), dtype=np.float64)
    dist[0] = float(np.linalg.norm(rcv - src))
    departure[0] = (rcv - src) / dist[0]
    start = 1
    for k, chains, triangles, cosines, hits, images in rows:
        stop = start + len(chains)
        order[start:stop] = k
        plane[start:stop, :k] = chains
        layer[start:stop, :k] = faces.layer_of[triangles]
        cos[start:stop, :k] = cosines
        points[start:stop, :k] = hits
        # 總長取最後一個鏡像源到接收點的直線距離（同長方體那一路），等於各段相加。
        dist[start:stop] = np.linalg.norm(rcv[None, :] - images, axis=1)
        first = hits[:, 0, :] - src[None, :]
        departure[start:stop] = first / np.linalg.norm(first, axis=1)[:, None]
        start = stop
    return ShellPathTable(faces.layers, max_order, _vec(src), _vec(rcv), order, dist, layer, plane, cos,
                          points, departure, blocked_count)


def _vec(values: FloatArray) -> tuple[float, float, float]:
    return (float(values[0]), float(values[1]), float(values[2]))


def reflection_products(table: ShellPathTable, impedance: ComplexArray, rho_c: float) -> ComplexArray:
    """每條路徑每個頻點的反射乘積 ΠR；``impedance`` 是 (圖層數, 頻點數) 的表面阻抗。"""
    if impedance.ndim != 2 or impedance.shape[0] != len(table.layers):
        raise ValueError("阻抗表的列數要等於材料圖層數")
    product = np.ones((table.path_count, impedance.shape[1]), dtype=np.complex128)
    for bounce in range(table.max_order):
        hit = table.bounce_layer[:, bounce] >= 0
        z_cos = impedance[table.bounce_layer[hit, bounce]] * table.bounce_cos[hit, bounce][:, None]
        product[hit] *= (z_cos - rho_c) / (z_cos + rho_c)
    return product


def path_pressures(table: ShellPathTable, impedance: ComplexArray, rho_c: float, sound_speed_m_s: float,
                   frequencies_hz: FloatArray) -> ComplexArray:
    """每條路徑每個頻點的壓力 ``(1/d)·ΠR·exp(−i·2πf·d/c)``（同長方體那一路的寫法）。"""
    product = reflection_products(table, impedance, rho_c)
    tau = table.dist_m / sound_speed_m_s
    phase = np.exp(-1j * (2.0 * np.pi * frequencies_hz)[None, :] * tau[:, None])
    return np.asarray(product * phase / table.dist_m[:, None], dtype=np.complex128)


def save_path_table(table: ShellPathTable, path: Path) -> None:
    """路徑表存成 numpy 壓縮檔；只存幾何，材料不在裡面。"""
    with path.open("wb") as handle:
        np.savez_compressed(
            handle, layers=np.array(table.layers), max_order=table.max_order,
            source=np.array(table.source), receiver=np.array(table.receiver), order=table.order,
            dist_m=table.dist_m, bounce_layer=table.bounce_layer, bounce_plane=table.bounce_plane,
            bounce_cos=table.bounce_cos, bounce_points=table.bounce_points, departure=table.departure,
            blocked_count=table.blocked_count,
        )


def load_path_table(path: Path) -> ShellPathTable:
    """讀回 :func:`save_path_table` 存的路徑表。"""
    with path.open("rb") as handle, np.load(handle, allow_pickle=False) as data:
        source, receiver = data["source"], data["receiver"]
        return ShellPathTable(
            tuple(str(name) for name in data["layers"]), int(data["max_order"]), _vec(source), _vec(receiver),
            data["order"], data["dist_m"], data["bounce_layer"], data["bounce_plane"], data["bounce_cos"],
            data["bounce_points"], data["departure"], int(data["blocked_count"]),
        )
