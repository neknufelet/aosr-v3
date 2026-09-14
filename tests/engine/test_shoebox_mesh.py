"""鞋盒 gmsh 網格的決定性、體積、邊界與內點考卷。"""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest
from numpy.typing import NDArray

from aosr.config.fem_lane import FEM_ELEMENTS_PER_WAVELENGTH, FEM_FMAX_CAP_HZ
from aosr.config.paths import config_path
from aosr.config.physics_constants import load_physics_constants
from aosr.geometry.shoebox import Room, Wall
from aosr.geometry.shoebox_mesh import ShoeboxMesh, generate_shoebox_mesh


REFERENCE_ROOM = Room(Lx=6.0, Ly=4.0, Lz=3.0)
REFERENCE_VOLUME_M3 = 72.0
REFERENCE_BOUNDARY_AREA_M2 = 108.0
REFERENCE_WALL_AREAS_M2 = {
    Wall.FLOOR: 24.0,
    Wall.CEILING: 24.0,
    Wall.X0: 12.0,
    Wall.XL: 12.0,
    Wall.Y0: 18.0,
    Wall.YL: 18.0,
}
SOURCE_M = np.asarray([1.5, 1.0, 1.2], dtype=np.float64)
RECEIVER_M = np.asarray([4.0, 3.0, 1.5], dtype=np.float64)

# 盒面與線性四面體都由同一組 float64 頂點算；1e-12 只包住累加與平面座標的捨入誤差，
# 遠小於 0.228... m 的特徵長度，不足以吞掉一個幾何錯誤。
GEOMETRY_REL_TOL = 1.0e-12
GEOMETRY_ABS_TOL = 1.0e-12


@pytest.fixture(scope="module")
def generated_meshes() -> Iterator[tuple[ShoeboxMesh, ShoeboxMesh, ShoeboxMesh]]:
    """同 seed 兩張、不同 seed 一張；全部走正式函式與真 gmsh。"""
    sound_speed = load_physics_constants(
        config_path("physics_constants.toml")
    ).sound_speed_m_s
    yield (
        generate_shoebox_mesh(
            REFERENCE_ROOM,
            max_frequency_hz=FEM_FMAX_CAP_HZ,
            elements_per_wavelength=FEM_ELEMENTS_PER_WAVELENGTH,
            sound_speed_m_s=sound_speed,
            random_seed=1,
        ),
        generate_shoebox_mesh(
            REFERENCE_ROOM,
            max_frequency_hz=FEM_FMAX_CAP_HZ,
            elements_per_wavelength=FEM_ELEMENTS_PER_WAVELENGTH,
            sound_speed_m_s=sound_speed,
            random_seed=1,
        ),
        generate_shoebox_mesh(
            REFERENCE_ROOM,
            max_frequency_hz=FEM_FMAX_CAP_HZ,
            elements_per_wavelength=FEM_ELEMENTS_PER_WAVELENGTH,
            sound_speed_m_s=sound_speed,
            random_seed=17,
        ),
    )


def _signed_six_volumes(mesh: ShoeboxMesh) -> NDArray[np.float64]:
    vertices = mesh.nodes[mesh.tetrahedra]
    edge_matrices = vertices[:, 1:] - vertices[:, :1]
    return np.asarray(np.linalg.det(edge_matrices), dtype=np.float64)


def _triangle_areas(mesh: ShoeboxMesh) -> NDArray[np.float64]:
    vertices = mesh.nodes[mesh.boundary_triangles]
    return np.asarray(
        0.5
        * np.linalg.norm(
            np.cross(
                vertices[:, 1] - vertices[:, 0],
                vertices[:, 2] - vertices[:, 0],
            ),
            axis=1,
        ),
        dtype=np.float64,
    )


def _strictly_contains(
    mesh: ShoeboxMesh,
    point: NDArray[np.float64],
) -> bool:
    vertices = mesh.nodes[mesh.tetrahedra]
    candidates = vertices[
        np.logical_and(
            np.all(point >= vertices.min(axis=1), axis=1),
            np.all(point <= vertices.max(axis=1), axis=1),
        )
    ]
    for tetrahedron in candidates:
        tail = np.linalg.solve(
            (tetrahedron[1:] - tetrahedron[0]).T,
            point - tetrahedron[0],
        )
        barycentric = np.concatenate(
            (np.asarray([1.0 - tail.sum()], dtype=np.float64), tail)
        )
        if np.all(barycentric > 0.0):
            return True
    return False


def test_same_seed_produces_bitwise_identical_nodes_and_tetrahedra(
    generated_meshes: tuple[ShoeboxMesh, ShoeboxMesh, ShoeboxMesh],
) -> None:
    """漏設單緒、漏設 seed 或不按 tag 重排，都可能讓逐位元重現失效。"""
    first, second, _ = generated_meshes
    assert np.array_equal(first.nodes, second.nodes)
    assert np.array_equal(first.tetrahedra, second.tetrahedra)


def test_tetrahedra_are_positive_and_fill_the_room(
    generated_meshes: tuple[ShoeboxMesh, ShoeboxMesh, ShoeboxMesh],
) -> None:
    """反向或漏掉的四面體，會破壞正向約定或 72 m3 體積守恆。"""
    mesh, _, _ = generated_meshes
    assert mesh.nodes.dtype == np.dtype(np.float64)
    assert mesh.tetrahedra.dtype == np.dtype(np.int64)
    assert mesh.nodes.ndim == 2
    assert mesh.nodes.shape[1] == 3
    assert mesh.tetrahedra.ndim == 2
    assert mesh.tetrahedra.shape[1] == 4
    assert np.all(mesh.tetrahedra >= 0)
    assert np.all(mesh.tetrahedra < mesh.nodes.shape[0])

    signed_six_volumes = _signed_six_volumes(mesh)
    assert np.all(signed_six_volumes > 0.0)
    assert signed_six_volumes.sum() / 6.0 == pytest.approx(
        REFERENCE_VOLUME_M3,
        rel=GEOMETRY_REL_TOL,
        abs=GEOMETRY_ABS_TOL,
    )


def test_boundary_triangles_cover_and_match_all_six_walls(
    generated_meshes: tuple[ShoeboxMesh, ShoeboxMesh, ShoeboxMesh],
) -> None:
    """漏面、重複面或牆索引錯置，都會破壞面積與逐頂點平面檢查。"""
    mesh, _, _ = generated_meshes
    assert mesh.boundary_triangles.dtype == np.dtype(np.int64)
    assert mesh.boundary_wall_indices.dtype == np.dtype(np.int64)
    assert mesh.boundary_triangles.ndim == 2
    assert mesh.boundary_triangles.shape[1] == 3
    assert mesh.boundary_wall_indices.shape == mesh.boundary_triangles.shape[:1]

    areas = _triangle_areas(mesh)
    assert areas.sum() == pytest.approx(
        REFERENCE_BOUNDARY_AREA_M2,
        rel=GEOMETRY_REL_TOL,
        abs=GEOMETRY_ABS_TOL,
    )
    for wall_index, wall in enumerate(Wall.all()):
        mask = mesh.boundary_wall_indices == wall_index
        assert np.any(mask)
        wall_vertices = mesh.nodes[mesh.boundary_triangles[mask]]
        wall_coordinates = wall_vertices[:, :, wall.axis()]
        assert np.allclose(
            wall_coordinates,
            wall.plane(REFERENCE_ROOM),
            rtol=0.0,
            atol=GEOMETRY_ABS_TOL,
        )
        assert areas[mask].sum() == pytest.approx(
            REFERENCE_WALL_AREAS_M2[wall],
            rel=GEOMETRY_REL_TOL,
            abs=GEOMETRY_ABS_TOL,
        )


def test_source_and_receiver_are_strictly_inside_tetrahedra(
    generated_meshes: tuple[ShoeboxMesh, ShoeboxMesh, ShoeboxMesh],
) -> None:
    """網格若在兩個正式點穿洞，嚴格正重心座標找不到承載四面體。"""
    mesh, _, _ = generated_meshes
    assert _strictly_contains(mesh, SOURCE_M)
    assert _strictly_contains(mesh, RECEIVER_M)


def test_changed_seed_changes_the_bitwise_mesh(
    generated_meshes: tuple[ShoeboxMesh, ShoeboxMesh, ShoeboxMesh],
) -> None:
    """控制組：若逐位元斷言只會回綠，seed 1 與 17 也會被誤判相同。"""
    seed_one, _, seed_seventeen = generated_meshes
    same_nodes = np.array_equal(seed_one.nodes, seed_seventeen.nodes)
    same_tetrahedra = np.array_equal(
        seed_one.tetrahedra,
        seed_seventeen.tetrahedra,
    )
    assert not (same_nodes and same_tetrahedra)
