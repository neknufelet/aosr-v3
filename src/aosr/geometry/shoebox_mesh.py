"""用 gmsh 把 :class:`~aosr.geometry.shoebox.Room` 切成決定性的四面體網格。

這支住 geometry 層，只拿同層的鞋盒契約與第三方 gmsh/numpy。呼叫端把最高頻率、
每波長格數與聲速明確傳入；特徵長度固定為 ``聲速 / 最高頻率 / 每波長格數``。
gmsh 沒有退回路徑：任何初始化、建模、切網格或取元素錯誤都原樣往上拋。
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence
import math
from typing import TypeAlias

import gmsh
import numpy as np
from numpy.typing import NDArray

from aosr.geometry.shoebox import Room, Wall


GMSH_TRIANGLE_3_TYPE = 2
GMSH_TETRAHEDRON_4_TYPE = 4
MeshRectangle: TypeAlias = tuple[str, tuple[float, float], tuple[float, float]]
WallPartitions: TypeAlias = Mapping[Wall, Sequence[MeshRectangle]]
_FragmentPatch: TypeAlias = tuple[Wall, MeshRectangle, tuple[int, ...]]


@dataclass(frozen=True)
class ShoeboxMesh:
    """線性四面體網格與其六面牆邊界。

    ``nodes`` 是 (N, 3) float64 座標；``tetrahedra`` 是 (M, 4) int64、0 起算；
    ``boundary_triangles`` 是 (K, 3) int64、0 起算。``boundary_wall_indices``
    與三角形逐列對齊，整數值是 ``Wall.all()`` 的位置：0=floor、1=ceiling、
    2=x0、3=xL、4=y0、5=yL。
    ``boundary_zone_ids`` 也逐列對齊：矩形用其代號，未分區面用空字串；
    手動建構舊網格可省略為 None，視為全整面。
    """

    nodes: NDArray[np.float64]
    tetrahedra: NDArray[np.int64]
    boundary_triangles: NDArray[np.int64]
    boundary_wall_indices: NDArray[np.int64]
    boundary_zone_ids: NDArray[np.str_] | None = None


def _checked_partitions(room: Room, partitions: WallPartitions) -> None:
    """純資料入口也核實精確覆蓋；不依賴上層方案驗證或面積和。"""
    for wall, rectangles in partitions.items():
        axes = [axis for axis in range(3) if axis != wall.axis()]
        width, height = (room.length(axis) for axis in axes)
        names = [name for name, _, _ in rectangles]
        if not names or any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError("分區代號必須非空且同面唯一")
        for _, (u0, u1), (v0, v1) in rectangles:
            if not (all(math.isfinite(x) for x in (u0, u1, v0, v1))
                    and 0 <= u0 < u1 <= width and 0 <= v0 < v1 <= height):
                raise ValueError("分區矩形必須有限且在牆面內")
        us = sorted({0.0, width, *(x for _, u, _ in rectangles for x in u)})
        vs = sorted({0.0, height, *(x for _, _, v in rectangles for x in v)})
        for u0, u1 in zip(us, us[1:]):
            for v0, v1 in zip(vs, vs[1:]):
                owners = sum(a <= u0 and u1 <= b and c <= v0 and v1 <= d
                             for _, (a, b), (c, d) in rectangles)
                if owners != 1:
                    raise ValueError("分區必須蓋滿整面且互不重疊")


def _fragment_wall_rectangles(room: Room, partitions: WallPartitions) -> list[_FragmentPatch]:
    """OCC 用貼牆矩形切盒面；共用拓樸讓表面與體網格完全相合。"""
    surfaces = []
    sources = []
    for wall in Wall.all():
        axes = [axis for axis in range(3) if axis != wall.axis()]
        for rectangle in partitions.get(wall, ()):
            _, (u0, u1), (v0, v1) = rectangle
            points = []
            for u, v in ((u0, v0), (u1, v0), (u1, v1), (u0, v1)):
                xyz = [0.0, 0.0, 0.0]
                xyz[wall.axis()] = wall.plane(room)
                xyz[axes[0]], xyz[axes[1]] = u, v
                points.append(gmsh.model.occ.addPoint(xyz[0], xyz[1], xyz[2]))
            lines = [gmsh.model.occ.addLine(points[i], points[(i + 1) % 4]) for i in range(4)]
            loop = gmsh.model.occ.addCurveLoop(lines)
            surfaces.append((2, gmsh.model.occ.addPlaneSurface([loop])))
            sources.append((wall, rectangle))
    _, mapping = gmsh.model.occ.fragment([(3, 1)], surfaces)
    return [(wall, rectangle, tuple(tag for dim, tag in parts if dim == 2))
            for (wall, rectangle), parts in zip(sources, mapping[1:], strict=True)]


def _pin_fragment_nodes(
    room: Room, nodes: NDArray[np.float64], sorted_tags: NDArray[np.int64], patches: Sequence[_FragmentPatch],
) -> None:
    """以 OCC 曲面拓樸的閉區間約束釘回輸入座標，不用幾何容差判邊。

    共用節點同時屬於相鄰矩形，其區間交集在切線方向是單點，因此切線座標精確。
    只有已知曲面上的點受矩形限制；體內節點只受盒子範圍限制。
    """
    lower = np.zeros_like(nodes)
    upper = np.broadcast_to(np.asarray([room.Lx, room.Ly, room.Lz]), nodes.shape).copy()
    for wall, (_, u, v), surfaces in patches:
        axes = [axis for axis in range(3) if axis != wall.axis()]
        bounds = [u, v]
        for surface in surfaces:
            tags, _, _ = gmsh.model.mesh.getNodes(2, surface, includeBoundary=True, returnParametricCoord=False)
            indices = np.searchsorted(sorted_tags, np.asarray(tags, dtype=np.int64))
            lower[indices, wall.axis()] = upper[indices, wall.axis()] = wall.plane(room)
            for axis, (lo, hi) in zip(axes, bounds, strict=True):
                lower[indices, axis] = np.maximum(lower[indices, axis], lo)
                upper[indices, axis] = np.minimum(upper[indices, axis], hi)
    if np.any(lower > upper):
        raise ValueError("分區曲面的共用節點具有互斥座標範圍")
    nodes[:] = np.minimum(np.maximum(nodes, lower), upper)


def _boundary_zone_ids(
    nodes: NDArray[np.float64], triangles: NDArray[np.int64],
    wall_indices: NDArray[np.int64], partitions: WallPartitions,
) -> NDArray[np.str_]:
    """以牆面 u、v 重心定區；閉矩形邊上的頂點合法，未分區面用空字串。"""
    longest = max((len(name) for rectangles in partitions.values() for name, _, _ in rectangles), default=1)
    labels = np.full(len(triangles), "", dtype=f"<U{longest}")
    centers = nodes[triangles].mean(axis=1)
    for index, wall in enumerate(Wall.all()):
        axes = [axis for axis in range(3) if axis != wall.axis()]
        on_wall = wall_indices == index
        for name, (u0, u1), (v0, v1) in partitions.get(wall, ()):
            inside = (on_wall & (u0 < centers[:, axes[0]]) & (centers[:, axes[0]] < u1)
                      & (v0 < centers[:, axes[1]]) & (centers[:, axes[1]] < v1))
            labels[inside] = name
        if wall in partitions and np.any(labels[on_wall] == ""):
            raise ValueError("邊界三角形重心不在任何分區內")
    return labels


def _sorted_nodes_and_tags() -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """按 gmsh node tag 升冪重排座標；排序後的位置就是公開的 0 起算節點編號。"""
    raw_tags, raw_coordinates, _ = gmsh.model.mesh.getNodes(
        returnParametricCoord=False
    )
    tags = np.asarray(raw_tags, dtype=np.int64)
    coordinates = np.asarray(raw_coordinates, dtype=np.float64).reshape((-1, 3))
    order = np.argsort(tags, kind="stable")
    return coordinates[order], tags[order]


def _element_nodes(
    *,
    dimension: int,
    gmsh_element_type: int,
    nodes_per_element: int,
    sorted_node_tags: NDArray[np.int64],
) -> NDArray[np.int64]:
    """取指定線性元素，並把每個 gmsh tag 映成排序後的 0 起算節點編號。"""
    element_types, _, element_node_tags = gmsh.model.mesh.getElements(dimension)
    matching = np.flatnonzero(
        np.asarray(element_types, dtype=np.int32) == gmsh_element_type
    )
    if matching.size == 0:
        raise RuntimeError(
            f"gmsh dimension {dimension} returned no element type {gmsh_element_type}"
        )
    raw = np.asarray(element_node_tags[int(matching[0])], dtype=np.int64).reshape(
        (-1, nodes_per_element)
    )
    indices = np.searchsorted(sorted_node_tags, raw)
    if not np.array_equal(sorted_node_tags[indices], raw):
        raise RuntimeError("gmsh element connectivity references an unknown node tag")
    return np.asarray(indices, dtype=np.int64)


def _orient_tetrahedra_positive(
    nodes: NDArray[np.float64],
    tetrahedra: NDArray[np.int64],
) -> NDArray[np.int64]:
    """統一成正向：det([v1-v0, v2-v0, v3-v0]) > 0；負向列交換前兩點。"""
    oriented = tetrahedra.copy()
    vertices = nodes[oriented]
    signed_six_volumes = np.linalg.det(vertices[:, 1:] - vertices[:, :1])
    negative = signed_six_volumes < 0.0
    first = oriented[negative, 0].copy()
    oriented[negative, 0] = oriented[negative, 1]
    oriented[negative, 1] = first
    return oriented


def _boundary_wall_indices(
    room: Room,
    nodes: NDArray[np.float64],
    triangles: NDArray[np.int64],
) -> NDArray[np.int64]:
    """依三個頂點所在的盒面，標成 ``Wall.all()`` 的整數位置。"""
    vertices = nodes[triangles]
    room_scale = max(1.0, room.Lx, room.Ly, room.Lz)
    plane_tolerance = 16.0 * np.finfo(np.float64).eps * room_scale
    indices = np.full(triangles.shape[0], -1, dtype=np.int64)
    for wall_index, wall in enumerate(Wall.all()):
        coordinates = vertices[:, :, wall.axis()]
        on_wall = np.all(
            np.abs(coordinates - wall.plane(room)) <= plane_tolerance,
            axis=1,
        )
        if np.any(on_wall & (indices >= 0)):
            raise RuntimeError("gmsh returned a boundary triangle on more than one wall")
        indices[on_wall] = wall_index
    if np.any(indices < 0):
        raise RuntimeError("gmsh returned a boundary triangle outside the six shoebox walls")
    return indices


def generate_shoebox_mesh(
    room: Room,
    *,
    max_frequency_hz: float,
    elements_per_wavelength: int,
    sound_speed_m_s: float,
    random_seed: int,
    partitions: WallPartitions | None = None,
) -> ShoeboxMesh:
    """產生一張單緒、固定亂數種子的鞋盒線性四面體網格。

    gmsh node tag 不假設連續或依回傳順序穩定：先按 tag 升冪排序，排序位置成為
    ``0..N-1``，所有元素連接再以同一張表重排。任何 gmsh 錯誤都往上拋；``finally``
    只負責一定關掉 gmsh，不重試、不換演算法。
    """
    characteristic_length = (
        sound_speed_m_s / max_frequency_hz / elements_per_wavelength
    )
    if partitions:
        _checked_partitions(room, partitions)
    gmsh.initialize(readConfigFiles=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0.0)
        gmsh.option.setNumber("General.NumThreads", 1.0)
        gmsh.option.setNumber("Mesh.MaxNumThreads3D", 1.0)
        gmsh.option.setNumber("Mesh.RandomSeed", float(random_seed))
        gmsh.option.setNumber("Mesh.MeshSizeMin", characteristic_length)
        gmsh.option.setNumber("Mesh.MeshSizeMax", characteristic_length)
        gmsh.model.add("shoebox")
        gmsh.model.occ.addBox(0.0, 0.0, 0.0, room.Lx, room.Ly, room.Lz)
        patches = []
        if partitions:
            patches = _fragment_wall_rectangles(room, partitions)
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(3)

        return _read_mesh(room, partitions or {}, patches)
    finally:
        gmsh.finalize()


def _read_mesh(room: Room, partitions: WallPartitions, patches: Sequence[_FragmentPatch]) -> ShoeboxMesh:
    """按原有順序取節點、體元素與面元素；只有分區網格釘回精確 CAD 邊界。"""
    nodes, sorted_tags = _sorted_nodes_and_tags()
    if patches:
        _pin_fragment_nodes(room, nodes, sorted_tags, patches)
    tetrahedra = _element_nodes(
        dimension=3,
        gmsh_element_type=GMSH_TETRAHEDRON_4_TYPE,
        nodes_per_element=4,
        sorted_node_tags=sorted_tags,
    )
    triangles = _element_nodes(
        dimension=2,
        gmsh_element_type=GMSH_TRIANGLE_3_TYPE,
        nodes_per_element=3,
        sorted_node_tags=sorted_tags,
    )
    wall_indices = _boundary_wall_indices(room, nodes, triangles)
    return ShoeboxMesh(
        nodes=nodes,
        tetrahedra=_orient_tetrahedra_positive(nodes, tetrahedra),
        boundary_triangles=triangles,
        boundary_wall_indices=wall_indices,
        boundary_zone_ids=_boundary_zone_ids(nodes, triangles, wall_indices, partitions),
    )
