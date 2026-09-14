"""P2 Helmholtz 組裝、點值與 pydiso 求解的物理考卷。"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from aosr.config.fem_lane import FEM_ELEMENTS_PER_WAVELENGTH
from aosr.config.paths import config_path
from aosr.config.physics_constants import load_physics_constants
from aosr.geometry.shoebox import Point, Room, Wall
from aosr.geometry.shoebox_mesh import ShoeboxMesh, generate_shoebox_mesh
from blueprint.reference_fem_check import ModalInputs, analytical_modal_pressure


ROOM = Room(6.0, 4.0, 3.0)
SOURCE = Point(1.5, 1.0, 1.2)
RECEIVER = Point(4.0, 3.0, 1.5)
PHYSICS = load_physics_constants(config_path("physics_constants.toml"))
TEST_MESH_MAX_FREQUENCY_HZ = 60.0
MATRIX_ABSOLUTE_TOLERANCE = 1.0e-11
SPECTRAL_SAMPLE_DOF_LIMIT = 12
# 這是最高只為 60 Hz 切的快速粗網格，不是第 5 段的正式九點契約網格；2^-5
# 只守住正負號、4π、P2 點值和求解接線的大錯。正式網格的 2^-10 仍由第 5 段考卷守。
COARSE_RIGID_RELATIVE_LIMIT = 2.0**-5


@lru_cache(maxsize=1)
def _mesh() -> ShoeboxMesh:
    return generate_shoebox_mesh(
        ROOM,
        max_frequency_hz=TEST_MESH_MAX_FREQUENCY_HZ,
        elements_per_wavelength=FEM_ELEMENTS_PER_WAVELENGTH,
        sound_speed_m_s=PHYSICS.sound_speed_m_s,
        random_seed=1,
    )


def _rigid_walls() -> dict[Wall, float | None]:
    return {wall: None for wall in Wall.all()}


def _active_positive_sample(matrix: csr_matrix[np.float64]) -> float:
    rows = np.flatnonzero(np.asarray(matrix.getnnz(axis=1)))
    sample = rows[:SPECTRAL_SAMPLE_DOF_LIMIT]
    principal = matrix[sample][:, sample].toarray()
    return float(np.linalg.eigvalsh(principal).min())


def test_p2_operators_are_symmetric_and_mass_samples_are_positive() -> None:
    """漏掉對稱組裝或產生非正質量時，抽查的主子矩陣會直接紅。"""
    from aosr.physics.fem_helmholtz import assemble_p2_operators

    operators = assemble_p2_operators(_mesh())
    matrices = (
        operators.stiffness,
        operators.mass,
        *operators.boundary_mass.values(),
    )
    for matrix in matrices:
        difference = matrix - matrix.T
        largest_asymmetry = (
            0.0 if difference.nnz == 0 else float(np.max(np.abs(difference.data)))
        )
        assert largest_asymmetry < MATRIX_ABSOLUTE_TOLERANCE
    assert _active_positive_sample(operators.mass) > 0.0
    for boundary_mass in operators.boundary_mass.values():
        assert _active_positive_sample(boundary_mass) > 0.0


def test_partition_of_unity_recovers_room_volume_and_each_wall_area() -> None:
    """積分階數太低或牆分錯面時，P2 基底總和不再積回幾何量。"""
    from aosr.physics.fem_helmholtz import assemble_p2_operators

    operators = assemble_p2_operators(_mesh())
    expected_areas = {
        Wall.FLOOR: ROOM.Lx * ROOM.Ly,
        Wall.CEILING: ROOM.Lx * ROOM.Ly,
        Wall.X0: ROOM.Ly * ROOM.Lz,
        Wall.XL: ROOM.Ly * ROOM.Lz,
        Wall.Y0: ROOM.Lx * ROOM.Lz,
        Wall.YL: ROOM.Lx * ROOM.Lz,
    }
    assert float(operators.mass.sum()) == pytest.approx(
        ROOM.Lx * ROOM.Ly * ROOM.Lz,
        abs=MATRIX_ABSOLUTE_TOLERANCE,
    )
    for wall, expected_area in expected_areas.items():
        assert float(operators.boundary_mass[wall].sum()) == pytest.approx(
            expected_area,
            abs=MATRIX_ABSOLUTE_TOLERANCE,
        )


def test_point_source_uses_p2_basis_values_and_four_pi_strength() -> None:
    """退成節點脈衝、P1 點值或漏掉 4π 都無法同時通過。"""
    from aosr.physics.fem_helmholtz import assemble_p2_operators, point_source_load

    operators = assemble_p2_operators(_mesh())
    load = point_source_load(operators, SOURCE)
    dof_locations = operators.basis.doflocs
    quadratic = dof_locations[0] ** 2 + 2.0 * dof_locations[1] * dof_locations[2]
    exact = SOURCE.x**2 + 2.0 * SOURCE.y * SOURCE.z

    assert complex(np.sum(load)) == pytest.approx(4.0 * math.pi)
    assert complex(load @ quadratic) == pytest.approx(4.0 * math.pi * exact)


def test_receiver_interpolation_is_exact_for_a_quadratic_field() -> None:
    """若接收點只取最近節點或用 P1，二次多項式不會精確重建。"""
    from aosr.physics.fem_helmholtz import assemble_p2_operators, interpolate_pressure

    operators = assemble_p2_operators(_mesh())
    xyz = operators.basis.doflocs
    nodal = np.asarray(xyz[0] ** 2 + 1j * xyz[1] * xyz[2], dtype=np.complex128)
    expected = RECEIVER.x**2 + 1j * RECEIVER.y * RECEIVER.z

    assert interpolate_pressure(operators, RECEIVER, nodal) == pytest.approx(expected)


def test_absorbing_boundary_has_positive_discrete_power_term() -> None:
    """e^(+jωt) 下吸音項符號應使 Im(pᴴAp) 為正。"""
    from aosr.physics.fem_helmholtz import (
        assemble_helmholtz_system,
        assemble_p2_operators,
    )

    operators = assemble_p2_operators(_mesh())
    impedance = 4.0 * PHYSICS.rho_c
    walls = {wall: impedance for wall in Wall.all()}
    frequency_hz = 17.0
    system = assemble_helmholtz_system(
        operators,
        frequency_hz=frequency_hz,
        wall_impedances=walls,
        density_kg_m3=PHYSICS.air_density_kg_m3,
        sound_speed_m_s=PHYSICS.sound_speed_m_s,
    )
    pressure = np.ones(operators.basis.N, dtype=np.complex128)
    applied = np.asarray(system @ pressure, dtype=np.complex128)
    observed = float(np.imag(np.sum(np.conjugate(pressure) * applied)))
    omega = 2.0 * math.pi * frequency_hz
    expected = omega * PHYSICS.air_density_kg_m3 * sum(
        float(operators.boundary_mass[wall].sum()) / impedance for wall in Wall.all()
    )

    assert observed == pytest.approx(expected, rel=1.0e-12)
    assert observed > 0.0


def test_point_outside_every_tetrahedron_is_rejected() -> None:
    """scikit-fem 找不到所在四面體時，不得夾回房內或改用最近節點。"""
    from aosr.physics.fem_helmholtz import assemble_p2_operators, point_source_load

    operators = assemble_p2_operators(_mesh())
    with pytest.raises(ValueError, match="outside of the mesh"):
        point_source_load(operators, Point(ROOM.Lx + 1.0, 1.0, 1.0))


def test_rigid_room_coarse_mesh_tracks_the_analytical_modal_pressure() -> None:
    """剛性弱式、4π 與點值任一接反，三個離共振低頻點會一起超界。"""
    from aosr.physics.fem_helmholtz import solve_fem_helmholtz

    frequencies = np.asarray((10.0, 14.0, 18.0), dtype=np.float64)
    solved = solve_fem_helmholtz(
        _mesh(),
        wall_impedances=_rigid_walls(),
        source=SOURCE,
        receiver=RECEIVER,
        frequencies_hz=frequencies,
        density_kg_m3=PHYSICS.air_density_kg_m3,
        sound_speed_m_s=PHYSICS.sound_speed_m_s,
    )
    modal_inputs = ModalInputs(
        room=(ROOM.Lx, ROOM.Ly, ROOM.Lz),
        source=SOURCE.as_tuple(),
        receiver=RECEIVER.as_tuple(),
        sound_speed=PHYSICS.sound_speed_m_s,
    )
    expected = np.asarray(
        [analytical_modal_pressure(float(frequency), 256, modal_inputs) for frequency in frequencies],
        dtype=np.complex128,
    )
    relative = np.abs(solved - expected) / np.abs(expected)

    assert solved.dtype == np.dtype(np.complex128)
    assert solved.shape == frequencies.shape
    assert float(relative.max()) < COARSE_RIGID_RELATIVE_LIMIT
