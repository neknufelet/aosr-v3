"""用 gmsh 把 :class:`~aosr.geometry.shoebox.Room` 切成決定性的四面體網格。

這支住 geometry 層，只拿同層的鞋盒契約與第三方 gmsh/numpy。呼叫端把最高頻率、
每波長格數與聲速明確傳入；特徵長度固定為 ``聲速 / 最高頻率 / 每波長格數``。
gmsh 沒有退回路徑：任何初始化、建模、切網格或取元素錯誤都原樣往上拋。
"""
from __future__ import annotations

from dataclasses import dataclass

import gmsh
import numpy as np
from numpy.typing import NDArray

from aosr.geometry.shoebox import Room, Wall


GMSH_TRIANGLE_3_TYPE = 2
GMSH_TETRAHEDRON_4_TYPE = 4


@dataclass(frozen=True)
class ShoeboxMesh:
    """線性四面體網格與其六面牆邊界。

    ``nodes`` 是 (N, 3) float64 座標；``tetrahedra`` 是 (M, 4) int64、0 起算；
    ``boundary_triangles`` 是 (K, 3) int64、0 起算。``boundary_wall_indices``
    與三角形逐列對齊，整數值是 ``Wall.all()`` 的位置：0=floor、1=ceiling、
    2=x0、3=xL、4=y0、5=yL。
    """

    nodes: NDArray[np.float64]
    tetrahedra: NDArray[np.int64]
    boundary_triangles: NDArray[np.int64]
    boundary_wall_indices: NDArray[np.int64]


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
) -> ShoeboxMesh:
    """產生一張單緒、固定亂數種子的鞋盒線性四面體網格。

    gmsh node tag 不假設連續或依回傳順序穩定：先按 tag 升冪排序，排序位置成為
    ``0..N-1``，所有元素連接再以同一張表重排。任何 gmsh 錯誤都往上拋；``finally``
    只負責一定關掉 gmsh，不重試、不換演算法。
    """
    characteristic_length = (
        sound_speed_m_s / max_frequency_hz / elements_per_wavelength
    )
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
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(3)

        nodes, sorted_tags = _sorted_nodes_and_tags()
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
        return ShoeboxMesh(
            nodes=nodes,
            tetrahedra=_orient_tetrahedra_positive(nodes, tetrahedra),
            boundary_triangles=triangles,
            boundary_wall_indices=_boundary_wall_indices(room, nodes, triangles),
        )
    finally:
        gmsh.finalize()
