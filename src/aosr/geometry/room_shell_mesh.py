"""把房間殼切成四面體網格（票 #1310，任意形狀房間第二刀：低頻有限元素接上）。

**切的是空氣。** 外殼圍出來的空間扣掉每個房內造型（天雲、凸出的板子）就是要算的那塊空氣；
造型是空氣裡的洞，它的表面跟外殼一樣是邊界。每個邊界三角形記得自己來自哪個材料圖層，
之後套阻抗靠的就是這個編號。

**怎麼切。** 焊點後的頂點、邊、三角形逐一變成 gmsh 內建幾何的點、線、平面，相鄰三角形共用
同一條線，所以整個邊界拓樸是封閉而且相合的；再用外殼一個面圈、每個造型一個面圈圍成體，
由 gmsh 切四面體。輸入的大三角形（一面牆常常只有兩個）會被切成目標大小的小三角形，
每個小三角形繼承原面的圖層。

**還沒做的：** 造型貼著外殼（凸出牆面的板子與牆共面）、造型彼此重疊或套在一起時，空氣不再是
「殼扣掉洞」，這一版不收：gmsh 可能切失敗（錯誤原樣往上拋），也可能安靜切出一張不對的網格
（例如零個四面體），所以切完一律核對空氣體積等於「外殼體積扣掉每個造型的體積」，對不上就擋。
「應有體積」用焊點後的代表點算（gmsh 拿到的就是這組點），所以兩邊理論上只差浮點誤差；容差是
「四面體數 × 焊點距離的立方」，每個四面體最多容一個焊點大小的小方塊，不另立門檻數字，少掉任何
一個真的四面體都遠大於它。切之前整間房搬到外殼外框中心附近（切完搬回），遠座標不吃有效位數；
容差另加一項隨座標大小走的浮點捨入，只在極端尺寸時才比焊點那一項大。
點音源剛好在造型裡由求解那一層擋。

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


def _local_points(triangles: Sequence[LayerTriangle], origin: NDArray[np.float64]) -> NDArray[np.float64]:
    """全部角點攤平、搬到以 ``origin`` 為原點的局部座標。"""
    return np.array([c for t in triangles for c in (t.a, t.b, t.c)], dtype=np.float64).reshape((-1, 3)) - origin


def _add_closed_surface(triangles: Sequence[LayerTriangle], weld_m: float, origin: NDArray[np.float64]) -> list[int]:
    """一個封閉網格變成共用點線的平面；回傳逐三角形對齊的面編號，退化（焊成同點）的三角形不收。"""
    ids = welded_vertex_ids(triangles, weld_m)
    points = _local_points(triangles, origin)
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


def _welded_volume(triangles: Sequence[LayerTriangle], weld_m: float,
                   origin: NDArray[np.float64]) -> tuple[float, float]:
    """焊點後（gmsh 實際拿到的點、同一組局部座標）的帶號體積與面積；法線朝外為正。"""
    if not triangles:
        return 0.0, 0.0
    corners = _local_points(triangles, origin)[welded_vertex_ids(triangles, weld_m)]
    cross = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    volume = float(np.einsum("ij,ij->i", corners[:, 0], np.cross(corners[:, 1], corners[:, 2])).sum() / 6.0)
    return volume, float(np.linalg.norm(cross, axis=1).sum() / 2.0)


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
    # 整間房先搬到外殼外框的中心附近再切，切完搬回去：遠離原點的座標（例如測量座標）不讓
    # gmsh 與體積核對吃掉有效位數。同號、相差不到兩倍的浮點數相減是精確的，所以搬過去不失真。
    shell_points = _local_points(room.shell, np.zeros(3))
    origin = (shell_points.min(axis=0) + shell_points.max(axis=0)) / 2.0 if len(shell_points) else np.zeros(3)
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
            surfaces = _add_closed_surface(triangles, weld_m, origin)
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
    measured = [_welded_volume(piece, weld_m, origin) for piece in pieces]
    expected = -measured[0][0] - sum(volume for volume, _ in measured[1:])
    actual = float(np.abs(np.linalg.det(nodes[tetrahedra][:, 1:] - nodes[tetrahedra][:, :1])).sum() / 6.0)
    # 浮點那一項：每個邊界面積元素的體積貢獻帶「最大局部座標 × 機器精度」的捨入，乘上四面體數
    # 給求和次數留餘地；搬到局部座標後它遠小於焊點那一項，只在極端尺寸時接手。
    extent = float(np.abs(nodes).max()) if len(nodes) else 0.0
    rounding = len(tetrahedra) * float(np.finfo(np.float64).eps) * extent * sum(area for _, area in measured)
    if not len(tetrahedra) or abs(actual - expected) > len(tetrahedra) * weld_m**3 + rounding:
        raise ValueError(f"切出來的空氣體積 {actual:.3f} m³ 對不上外殼扣掉造型的 {expected:.3f} m³；"
                         "造型彼此重疊、套在一起或貼著外殼，這一版的低頻網格還不支援")
    boundary = np.concatenate([block for block, _ in blocks])
    layers = np.concatenate([np.full(len(block), layer, dtype=np.int64) for block, layer in blocks])
    oriented = _positive(nodes, tetrahedra)
    return ShellMesh(nodes + origin, oriented, boundary, layers, tuple(layer_names))
