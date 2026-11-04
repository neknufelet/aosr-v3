"""分區網格、P2 邊界與逐區阻抗；小房間避免增加雲端重算成本。"""
from __future__ import annotations

import hashlib
from dataclasses import replace
from functools import lru_cache

import numpy as np
import pytest
from scipy.sparse.linalg import spsolve
from skfem import MeshTet

from aosr.config.paths import config_path
from aosr.config.physics_constants import load_physics_constants
from aosr.config.precision_contracts import default_precision_contracts_path, load_precision_contracts
from aosr.geometry.shoebox import Point, Room, Wall, wall_grid_cell
from aosr.geometry.shoebox_mesh import MeshRectangle, ShoeboxMesh, generate_shoebox_mesh
from aosr.physics import fem_helmholtz as fem
from aosr.physics import report_surfaces
from aosr.physics.fem_reduced import mesh_hash
from aosr.materials.construction_models import ConstantImpedance


ROOM = Room(3.0, 2.0, 1.5)
PHYSICS = load_physics_constants(config_path("physics_constants.toml"))
# 起點 c5817bcdf62402e36cd5f8d3d521e6fb0ab50ffd，2026-11-04 在乾淨樹實量。
CONTROL = {
    "nodes": "042039f693bcd800d617dbad6aa93f578a67f195e4be2c16c9842ea270a607ee",
    "tetrahedra": "80d02746eeb9daa85ce5e4263dda2c93074353f69487a8a64967ef30cac36453",
    "boundary_triangles": "7adeb85648b969a4cdb6c01b7afcb050e18fde13269d8f94d7f1b9ac99e5adc3",
    "boundary_wall_indices": "9102264066118e4c2cd48288eb0b06dd4f29666743efe99177858ca59869709f",
}
OPERATOR_CONTROL = {
    "stiffness": "1ccbbe41228c0866414ec4095c4c09f22b5ebeebf7528222c0afe48a045cc00d",
    "mass": "802a61902044048800efb2a652b3c7eefee0412df1e908c84eef1d7dd30536ba",
    "floor": "4fb3c0263202a51cf268b887ae8b0f63553e5014f65af6fe7c8838d433f22a24",
    "ceiling": "a72456a43605f7fae40cf23a5f63c664a11f5b8509cf946e832a52181788f613",
    "x0": "1a1453a11d76a7081ebf0b5753c991579f5c4a09bfb5cc3e2ccead0ecacf15ed",
    "xL": "e30e4e267a287300cdfdad208dbd102363a53a28842b5d204a1552c1507b52a9",
    "y0": "4fef4e7dd9016ff2fe19878a96d0f6bf41be1c2a219837297886d2ae581b0509",
    "yL": "d124b0b37ba6b37120615b26895fc33fe35a5127fc1dabc3a5abe82597544f62",
}


def _partitions() -> dict[Wall, tuple[MeshRectangle, ...]]:
    result = {}
    for wall in Wall.all():
        u_axis, v_axis = [axis for axis in range(3) if axis != wall.axis()]
        width, height = ROOM.length(u_axis), ROOM.length(v_axis)
        # 不對稱：u 三等分的第一條線、v 二等分；六面都考，避免側牆軸向漏驗。
        us, vs = (0.0, width / 3.0, width), (0.0, height / 2.0, height)
        result[wall] = tuple((f"r{row}c{col}", (us[col], us[col + 1]), (vs[row], vs[row + 1]))
                             for row in range(2) for col in range(2))
    return result


@lru_cache(maxsize=2)
def _mesh(zoned: bool) -> ShoeboxMesh:
    return generate_shoebox_mesh(ROOM, max_frequency_hz=80.0, elements_per_wavelength=5,
                                 sound_speed_m_s=PHYSICS.sound_speed_m_s, random_seed=1,
                                 partitions=_partitions() if zoned else None)


def test_unpartitioned_mesh_and_operators_are_bitwise_control() -> None:
    mesh = _mesh(False)
    for field, expected in CONTROL.items():
        assert hashlib.sha256(getattr(mesh, field).tobytes()).hexdigest() == expected
    assert mesh_hash(mesh) == "1843ddae3375da6e19ef773e72510928c3bcf9cbd67b40f0bd45ebf4de8e96ce"
    operators = fem.assemble_p2_operators(mesh)
    assert not operators.zone_boundary_mass
    for field, matrix in [("stiffness", operators.stiffness), ("mass", operators.mass),
                          *((wall.wall_name(), matrix) for wall, matrix in operators.boundary_mass.items())]:
        raw = matrix.indptr.tobytes() + matrix.indices.tobytes() + matrix.data.tobytes()
        assert hashlib.sha256(raw).hexdigest() == OPERATOR_CONTROL[field]
    assert mesh.boundary_zone_ids is not None
    assert mesh.boundary_zone_ids.tolist() == [""] * len(mesh.boundary_triangles)


def test_partition_mesh_is_bitwise_deterministic() -> None:
    first = _mesh(True)
    second = generate_shoebox_mesh(ROOM, max_frequency_hz=80.0, elements_per_wavelength=5,
                                  sound_speed_m_s=PHYSICS.sound_speed_m_s, random_seed=1,
                                  partitions=_partitions())
    for field in (*CONTROL, "boundary_zone_ids"):
        left, right = getattr(first, field), getattr(second, field)
        assert left.dtype == right.dtype and left.shape == right.shape
        assert left.tobytes() == right.tobytes()


def test_asymmetric_partition_vertices_centroids_and_facets_are_exact() -> None:
    mesh = _mesh(True)
    assert mesh.boundary_zone_ids is not None
    assert mesh.boundary_zone_ids.shape == mesh.boundary_triangles.shape[:1]
    finite = MeshTet(mesh.nodes.T, mesh.tetrahedra.T, sort_t=False)
    facets = fem._boundary_facets(finite, mesh)
    for index, wall in enumerate(Wall.all()):
        triangles = mesh.boundary_triangles[mesh.boundary_wall_indices == index]
        labels = mesh.boundary_zone_ids[mesh.boundary_wall_indices == index]
        axes = [axis for axis in range(3) if axis != wall.axis()]
        vertices = mesh.nodes[triangles][:, :, axes]
        groups = []
        for name, (u0, u1), (v0, v1) in _partitions()[wall]:
            selected = labels == name
            assert np.any(selected)
            owned = vertices[selected]
            assert np.all((u0 <= owned[:, :, 0]) & (owned[:, :, 0] <= u1))
            assert np.all((v0 <= owned[:, :, 1]) & (owned[:, :, 1] <= v1))
            centers = vertices.mean(axis=1)
            inside = ((u0 < centers[:, 0]) & (centers[:, 0] < u1)
                      & (v0 < centers[:, 1]) & (centers[:, 1] < v1))
            assert np.array_equal(selected, inside)
            groups.append(set(facets[wall][selected]))
        assert set.union(*groups) == set(facets[wall])
        assert sum(map(len, groups)) == len(set.union(*groups)) == len(triangles)
        # 另用正式 wall_grid_cell（均勻二等分）獨立核實 u、v 對應的空間軸。
        for point in mesh.nodes[triangles].mean(axis=1):
            row, col = wall_grid_cell(ROOM, wall.wall_name(), tuple(point), rows=2, cols=2)
            assert row == int(point[axes[1]] >= ROOM.length(axes[1]) / 2.0)
            assert col == int(point[axes[0]] >= ROOM.length(axes[0]) / 2.0)


def test_zone_matrices_have_local_support_area_and_exact_wall_sum() -> None:
    mesh = _mesh(True)
    operators = fem.assemble_p2_operators(mesh)
    assert mesh.boundary_zone_ids is not None
    facets = fem._boundary_facets(operators.basis.mesh, mesh)
    # 沿用現有有限元素比較尺；不在此另立容差數字。
    limit = load_precision_contracts(default_precision_contracts_path())["fem_vs_fenics_frozen"].value
    ones = np.ones(operators.basis.N)
    # 同一張網格拿掉區標記，走整面那條組裝；各區相加要跟這份獨立積分一致。
    independent = fem.assemble_p2_operators(replace(mesh, boundary_zone_ids=None))
    assert independent.zone_boundary_mass == {}
    for index, wall in enumerate(Wall.all()):
        reference = independent.boundary_mass[wall]
        difference = abs(operators.boundary_mass[wall] - reference).max()
        assert difference <= limit * abs(reference).max()
        matrices = operators.zone_boundary_mass[wall]
        labels = mesh.boundary_zone_ids[mesh.boundary_wall_indices == index]
        sequence = iter(matrices.values())
        total = next(sequence).copy()
        for matrix in sequence:
            total = total + matrix
        assert (total - operators.boundary_mass[wall]).nnz == 0
        for name, (u0, u1), (v0, v1) in _partitions()[wall]:
            matrix = matrices[name]
            allowed = operators.basis.get_dofs(facets=facets[wall][labels == name]).all()
            rows, cols = matrix.nonzero()
            assert set(rows) <= set(allowed)
            assert set(cols) <= set(allowed)
            expected = (u1 - u0) * (v1 - v0)
            observed = float(np.asarray(ones @ matrix @ ones))
            assert abs(observed - expected) / expected <= limit


def test_same_zone_impedance_matches_whole_wall_and_changed_zone_changes_pressure() -> None:
    operators = fem.assemble_p2_operators(_mesh(True))
    independent = fem.assemble_p2_operators(replace(_mesh(True), boundary_zone_ids=None))
    impedance = complex(PHYSICS.rho_c * 4.0, PHYSICS.rho_c)
    walls = {wall: impedance for wall in Wall.all()}
    zones = {wall: dict.fromkeys(operators.zone_boundary_mass[wall], impedance) for wall in Wall.all()}
    arguments = dict(frequency_hz=37.0, density_kg_m3=PHYSICS.air_density_kg_m3,
                     sound_speed_m_s=PHYSICS.sound_speed_m_s)
    whole = fem.assemble_helmholtz_system(independent, wall_impedances=walls, **arguments)
    zoned = fem.assemble_zoned_helmholtz_system(operators, wall_impedances={}, zone_impedances=zones, **arguments)
    load = fem.point_source_load(operators, Point(0.7, 0.6, 0.5))
    receiver = Point(2.1, 1.3, 0.8)
    reference = fem.interpolate_pressure(independent, receiver,
                                         spsolve(whole, fem.point_source_load(independent, Point(0.7, 0.6, 0.5))))
    observed = fem.interpolate_pressure(operators, receiver, spsolve(zoned, load))
    limit = load_precision_contracts(default_precision_contracts_path())["fem_vs_fenics_frozen"].value
    assert abs(observed - reference) / abs(reference) <= limit
    zones[Wall.X0]["r0c0"] *= 0.5
    changed = fem.assemble_zoned_helmholtz_system(operators, wall_impedances={}, zone_impedances=zones, **arguments)
    assert fem.interpolate_pressure(operators, receiver, spsolve(changed, load)) != observed


def test_reverse_boundary_count_rejects_missing_or_duplicate_triangles() -> None:
    mesh = _mesh(False)
    missing = replace(mesh, boundary_triangles=mesh.boundary_triangles[:-1],
                      boundary_wall_indices=mesh.boundary_wall_indices[:-1], boundary_zone_ids=None)
    with pytest.raises(ValueError, match="邊界 facet 數"):
        fem.assemble_p2_operators(missing)
    triangles = mesh.boundary_triangles.copy()
    triangles[-1] = triangles[0]
    duplicate = replace(mesh, boundary_triangles=triangles)
    with pytest.raises(ValueError, match="邊界"):
        fem.assemble_p2_operators(duplicate)


def test_rectangle_adapter_only_passes_geometry_and_preserves_whole_wall_rejection() -> None:
    material = ConstantImpedance(impedance_pa_s_m=PHYSICS.rho_c)
    zones = {wall.wall_name(): tuple(report_surfaces.RectangleZone(
        area="rectangle", id=name, u_m=u, v_m=v, material=material)
        for name, u, v in rectangles) for wall, rectangles in _partitions().items()}
    assert report_surfaces.mesh_partitions(ROOM, zones) == _partitions()
    with pytest.raises(report_surfaces.RectangleZonesUnsupported):
        report_surfaces.whole_wall_materials(zones)


def test_partition_metadata_changes_mesh_hash_but_empty_metadata_does_not() -> None:
    mesh = _mesh(True)
    assert mesh.boundary_zone_ids is not None
    renamed = replace(mesh, boundary_zone_ids=np.char.add("new_", mesh.boundary_zone_ids))
    assert mesh_hash(renamed) != mesh_hash(mesh)
    whole = _mesh(False)
    assert mesh_hash(whole) == mesh_hash(replace(whole, boundary_zone_ids=None))


def test_partial_partition_keeps_other_walls_whole_and_checks_impedance_keys() -> None:
    partitions = {Wall.X0: _partitions()[Wall.X0]}
    mesh = generate_shoebox_mesh(ROOM, max_frequency_hz=80.0, elements_per_wavelength=5,
                                 sound_speed_m_s=PHYSICS.sound_speed_m_s, random_seed=1,
                                 partitions=partitions)
    operators = fem.assemble_p2_operators(mesh)
    assert set(operators.zone_boundary_mass) == {Wall.X0}
    assert mesh.boundary_zone_ids is not None
    for index, wall in enumerate(Wall.all()):
        if wall != Wall.X0:
            assert np.all(mesh.boundary_zone_ids[mesh.boundary_wall_indices == index] == "")
    arguments = dict(frequency_hz=37.0, density_kg_m3=PHYSICS.air_density_kg_m3,
                     sound_speed_m_s=PHYSICS.sound_speed_m_s)
    walls: dict[Wall, fem.Impedance | None] = {wall: None for wall in Wall.all() if wall != Wall.X0}
    zones = {Wall.X0: dict.fromkeys(operators.zone_boundary_mass[Wall.X0], PHYSICS.rho_c)}
    valid = fem.assemble_zoned_helmholtz_system(operators, wall_impedances=walls, zone_impedances=zones, **arguments)
    assert valid.shape == operators.mass.shape
    invalid: tuple[dict[Wall, dict[str, fem.Impedance]], ...] = ({}, {Wall.X0: {}}, {Wall.X0: {"extra": PHYSICS.rho_c}},
                      {Wall.X0: dict.fromkeys(zones[Wall.X0], 0.0)},
                      {Wall.X0: dict.fromkeys(zones[Wall.X0], -PHYSICS.rho_c)},
                      {Wall.X0: dict.fromkeys(zones[Wall.X0], complex(float("nan"), 0.0))})
    for bad_zones in invalid:
        with pytest.raises(ValueError):
            fem.assemble_zoned_helmholtz_system(operators, wall_impedances=walls,
                                               zone_impedances=bad_zones, **arguments)
    with pytest.raises(ValueError):
        fem.assemble_zoned_helmholtz_system(operators, wall_impedances={}, zone_impedances=zones, **arguments)


@pytest.mark.parametrize("bad", [
    (),
    (("a", (0.0, 1.0), (0.0, 1.0)),),
    (("a", (0.0, ROOM.Lx), (0.0, ROOM.Ly)), ("a", (0.0, ROOM.Lx), (0.0, ROOM.Ly))),
    (("a", (0.0, ROOM.Lx), (0.0, ROOM.Ly)), ("b", (0.0, ROOM.Lx), (0.0, ROOM.Ly))),
    (("a", (0.0, float("inf")), (0.0, ROOM.Ly)),),
])
def test_geometry_partition_rejects_empty_gapped_duplicate_overlapping_or_infinite_data(
    bad: tuple[tuple[str, tuple[float, float], tuple[float, float]], ...],
) -> None:
    with pytest.raises(ValueError):
        generate_shoebox_mesh(ROOM, max_frequency_hz=80.0, elements_per_wavelength=5,
                              sound_speed_m_s=PHYSICS.sound_speed_m_s, random_seed=1,
                              partitions={Wall.FLOOR: bad})


def test_nonbinary_rectangle_edges_have_exact_closed_vertex_membership() -> None:
    room = Room(6.98792656080558, 4.41988145469311, 3.1554034223396052)
    partitions: dict[Wall, tuple[MeshRectangle, ...]] = {}
    for wall in Wall.all():
        if wall == Wall.FLOOR:
            continue
        axes = [axis for axis in range(3) if axis != wall.axis()]
        width, height = (room.length(axis) for axis in axes)
        us, vs = (0.0, width / 3.0, width), (0.0, height / 2.0, height)
        partitions[wall] = tuple((f"r{row}c{col}", (us[col], us[col+1]), (vs[row], vs[row+1]))
                                 for row in range(2) for col in range(2))
    mesh = generate_shoebox_mesh(room, max_frequency_hz=50.0, elements_per_wavelength=5,
                                 sound_speed_m_s=PHYSICS.sound_speed_m_s, random_seed=1, partitions=partitions)
    assert mesh.boundary_zone_ids is not None
    for index, wall in enumerate(Wall.all()):
        axes = [axis for axis in range(3) if axis != wall.axis()]
        expected_names = {name for name, _, _ in partitions.get(wall, ())} or {""}
        assert set(mesh.boundary_zone_ids[mesh.boundary_wall_indices == index]) == expected_names
        for name, (u0, u1), (v0, v1) in partitions.get(wall, ()):
            selected = (mesh.boundary_wall_indices == index) & (mesh.boundary_zone_ids == name)
            vertices = mesh.nodes[mesh.boundary_triangles[selected]]
            assert np.all((u0 <= vertices[:, :, axes[0]]) & (vertices[:, :, axes[0]] <= u1))
            assert np.all((v0 <= vertices[:, :, axes[1]]) & (vertices[:, :, axes[1]] <= v1))
    # 修正幾何的共用節點後，仍必須是一張沒有裂縫的四面體／邊界網格。
    fem.assemble_p2_operators(mesh)
