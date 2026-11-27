"""把房間殼切成四面體網格（票 #1310，任意形狀房間第二刀：低頻有限元素接上）。

**切的是空氣。** 外殼圍出來的空間扣掉每個房內造型（天雲、凸出的板子）就是要算的那塊空氣；
造型是空氣裡的洞，它的表面跟外殼一樣是邊界。每個邊界三角形記得自己來自哪個材料圖層，
之後套阻抗靠的就是這個編號。

**怎麼切。** 焊點後的頂點、邊、三角形逐一變成 gmsh 內建幾何的點、線、平面，相鄰三角形共用
同一條線，所以整個邊界拓樸是封閉而且相合的；再用外殼一個面圈、每個造型一個面圈圍成體，
由 gmsh 切四面體。輸入的大三角形（一面牆常常只有兩個）會被切成目標大小的小三角形，
每個小三角形繼承原面的圖層。

**還沒做的：** 造型貼著外殼（凸出牆面的板子與牆共面）時，空氣不再是「殼扣掉洞」，
這一版不收，gmsh 會切失敗、錯誤原樣往上拋。點音源剛好在造型裡由求解那一層擋。

呼叫端傳入焊點距離、最高頻率、每波長格數與聲速；網格大小固定為
``聲速 / 最高頻率 / 每波長格數``，跟長方體那一支同一個定義。gmsh 錯誤不攔、不退回。
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import gmsh
import numpy as np
from numpy.typing import NDArray

from aosr.geometry.room_shell import LayerTriangle, RoomShell, welded_vertex_ids


_GMSH_TRIANGLE_3_TYPE = 2
_GMSH_TETRAHEDRON_4_TYPE = 4


@dataclass(frozen=True)
class ShellMesh:
    """線性四面體網格與帶圖層編號的邊界。

    ``nodes`` (N, 3) float64；``tetrahedra`` (M, 4) int64、0 起算、一律正向；
    ``boundary_triangles`` (K, 3) int64、0 起算；``boundary_layers`` 逐列對齊，
    是 ``layer_names`` 的位置。``layer_names`` 依第一次出現的次序排，外殼在前、造型在後。
    """

    nodes: NDArray[np.float64]
    tetrahedra: NDArray[np.int64]
    boundary_triangles: NDArray[np.int64]
    boundary_layers: NDArray[np.int64]
    layer_names: tuple[str, ...]

    def volume_m3(self) -> float:
        corners = self.nodes[self.tetrahedra]
        return float(np.linalg.det(corners[:, 1:] - corners[:, :1]).sum() / 6.0)

    def layer_area_m2(self, name: str) -> float:
        index = self.layer_names.index(name)
        corners = self.nodes[self.boundary_triangles[self.boundary_layers == index]]
        cross = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        return float(np.linalg.norm(cross, axis=1).sum() / 2.0)


def _add_closed_surface(triangles: Sequence[LayerTriangle], weld_m: float) -> list[int]:
    """一個封閉網格變成共用點線的平面；回傳逐三角形對齊的面編號，退化（焊成同點）的三角形不收。"""
    ids = welded_vertex_ids(triangles, weld_m)
    points = np.array([c for t in triangles for c in (t.a, t.b, t.c)], dtype=np.float64)
    point_tags = {int(i): gmsh.model.geo.addPoint(float(points[i, 0]), float(points[i, 1]), float(points[i, 2]))
                  for i in np.unique(ids)}
    line_tags: dict[tuple[int, int], int] = {}
    surfaces: list[int] = []
    for row in ids:
        if len({int(v) for v in row}) < 3:
            surfaces.append(-1)
            continue
        loop = []
        for k in range(3):
            a, b = int(row[k]), int(row[(k + 1) % 3])
            key = (min(a, b), max(a, b))
            if key not in line_tags:
                line_tags[key] = gmsh.model.geo.addLine(point_tags[key[0]], point_tags[key[1]])
            loop.append(line_tags[key] if a < b else -line_tags[key])
        surfaces.append(gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(loop)]))
    return surfaces


def _sorted_nodes() -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    raw_tags, raw_coordinates, _ = gmsh.model.mesh.getNodes(returnParametricCoord=False)
    tags = np.asarray(raw_tags, dtype=np.int64)
    order = np.argsort(tags, kind="stable")
    return np.asarray(raw_coordinates, dtype=np.float64).reshape((-1, 3))[order], tags[order]


def _elements(dimension: int, tag: int, element_type: int, width: int,
              sorted_tags: NDArray[np.int64]) -> NDArray[np.int64]:
    types, _, node_tags = gmsh.model.mesh.getElements(dimension, tag)
    rows = [np.asarray(nodes, dtype=np.int64).reshape((-1, width))
            for kind, nodes in zip(types, node_tags, strict=True) if int(kind) == element_type]
    raw = np.concatenate(rows) if rows else np.zeros((0, width), dtype=np.int64)
    indices = np.searchsorted(sorted_tags, raw)
    if raw.size and (int(indices.max()) >= len(sorted_tags) or not np.array_equal(sorted_tags[indices], raw)):
        raise RuntimeError("gmsh 元素指到不存在的節點")
    return np.asarray(indices, dtype=np.int64)


def _positive(nodes: NDArray[np.float64], tetrahedra: NDArray[np.int64]) -> NDArray[np.int64]:
    oriented = tetrahedra.copy()
    corners = nodes[oriented]
    negative = np.linalg.det(corners[:, 1:] - corners[:, :1]) < 0.0
    oriented[negative, 0], oriented[negative, 1] = tetrahedra[negative, 1], tetrahedra[negative, 0]
    return oriented


def generate_room_shell_mesh(
    room: RoomShell, *, weld_m: float, max_frequency_hz: float, elements_per_wavelength: int,
    sound_speed_m_s: float, random_seed: int,
) -> ShellMesh:
    """單緒、固定亂數種子切一張房間殼的空氣網格；呼叫前應先過 :func:`check_room_shell`。"""
    size = sound_speed_m_s / max_frequency_hz / elements_per_wavelength
    pieces: list[Sequence[LayerTriangle]] = [room.shell, *(solid.triangles for solid in room.solids)]
    gmsh.initialize(readConfigFiles=False)
    try:
        for name, value in (("General.Terminal", 0.0), ("General.NumThreads", 1.0),
                            ("Mesh.MaxNumThreads3D", 1.0), ("Mesh.RandomSeed", float(random_seed)),
                            ("Mesh.MeshSizeMin", size), ("Mesh.MeshSizeMax", size),
                            ("Mesh.MeshSizeFromPoints", 0.0), ("Mesh.MeshSizeFromCurvature", 0.0),
                            ("Mesh.MeshSizeExtendFromBoundary", 0.0)):
            gmsh.option.setNumber(name, value)
        gmsh.model.add("room_shell")
        layer_names: list[str] = []
        surface_layer: dict[int, int] = {}
        loops = []
        for triangles in pieces:
            surfaces = _add_closed_surface(triangles, weld_m)
            for surface, triangle in zip(surfaces, triangles, strict=True):
                if surface < 0:
                    continue
                if triangle.layer not in layer_names:
                    layer_names.append(triangle.layer)
                surface_layer[surface] = layer_names.index(triangle.layer)
            loops.append(gmsh.model.geo.addSurfaceLoop([s for s in surfaces if s >= 0]))
        gmsh.model.geo.addVolume(loops)
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(3)
        nodes, sorted_tags = _sorted_nodes()
        tetrahedra = _elements(3, -1, _GMSH_TETRAHEDRON_4_TYPE, 4, sorted_tags)
        blocks = [(_elements(2, surface, _GMSH_TRIANGLE_3_TYPE, 3, sorted_tags), layer)
                  for surface, layer in sorted(surface_layer.items())]
    finally:
        gmsh.finalize()
    boundary = np.concatenate([block for block, _ in blocks])
    layers = np.concatenate([np.full(len(block), layer, dtype=np.int64) for block, layer in blocks])
    return ShellMesh(nodes, _positive(nodes, tetrahedra), boundary, layers, tuple(layer_names))
