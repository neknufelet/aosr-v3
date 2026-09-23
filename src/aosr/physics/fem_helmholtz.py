"""用 P2 四面體組裝並求解鞋盒房的頻域 Helmholtz 聲壓。

採 ``e^(+jωt)``：``A = K - k²M + jωρ Σ(B_w/Z_w)``，右側是音源點
P2 基底值乘 ``4π``。一面牆的阻抗以正實數表示，``None`` 表示剛性牆。
PARDISO 同一個 solver 物件在每個後續頻點呼叫 ``refactor``，因此每點重新分解；
解法例外不攔截、不退回其他直接解或疊代解。
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from skfem import Basis, ElementTetP2, FacetBasis, MeshTet, asm
from skfem.models.poisson import laplace, mass

from aosr import runtime
from aosr.geometry.shoebox import Point, Wall
from aosr.geometry.shoebox_mesh import ShoeboxMesh


P2_PRODUCT_INTEGRATION_ORDER = 4
POINT_SOURCE_STRENGTH = 4.0 * math.pi
WallImpedances = Mapping[Wall, float | None]


@dataclass(frozen=True)
class P2Operators:
    """同一張網格上與頻率無關的 P2 基底、K、M 與六面邊界質量。"""

    basis: Basis
    stiffness: csr_matrix[np.float64]
    mass: csr_matrix[np.float64]
    boundary_mass: dict[Wall, csr_matrix[np.float64]]


def _real_csr(matrix: csr_matrix[np.float64]) -> csr_matrix[np.float64]:
    """正規化成排序、合併重複項的 float64 CSR。"""
    result = matrix.tocsr(copy=True).astype(np.float64)
    result.sum_duplicates()
    result.sort_indices()
    return result


def _complex_csr(matrix: csr_matrix[np.complex128]) -> csr_matrix[np.complex128]:
    """轉成 pydiso 要求的可寫、C-contiguous complex128 CSR。"""
    result = matrix.tocsr(copy=True).astype(np.complex128)
    result.sum_duplicates()
    result.sort_indices()
    result.data = np.array(result.data, dtype=np.complex128, order="C", copy=True)
    result.indices = np.array(result.indices, order="C", copy=True)
    result.indptr = np.array(result.indptr, order="C", copy=True)
    return result


def _boundary_facets(
    finite_mesh: MeshTet,
    mesh: ShoeboxMesh,
) -> dict[Wall, NDArray[np.int64]]:
    """把 ShoeboxMesh 三角形逐面映到 scikit-fem 的 facet 編號。"""
    facet_by_vertices = {
        tuple(sorted(int(node) for node in finite_mesh.facets[:, facet])): int(facet)
        for facet in finite_mesh.boundary_facets()
    }
    grouped: dict[Wall, list[int]] = {wall: [] for wall in Wall.all()}
    for triangle, wall_index in zip(
        mesh.boundary_triangles,
        mesh.boundary_wall_indices,
        strict=True,
    ):
        key = tuple(sorted(int(node) for node in triangle))
        try:
            facet = facet_by_vertices[key]
            wall = Wall.all()[int(wall_index)]
        except (KeyError, IndexError) as exc:
            raise ValueError("鞋盒邊界三角形或牆編號不屬於四面體網格") from exc
        grouped[wall].append(facet)
    return {
        wall: np.asarray(facets, dtype=np.int64) for wall, facets in grouped.items()
    }


def assemble_p2_operators(mesh: ShoeboxMesh) -> P2Operators:
    """以四階積分組裝 P2×P2 的 K、M 與每面牆的邊界質量。"""
    finite_mesh = MeshTet(
        np.asarray(mesh.nodes, dtype=np.float64).T,
        np.asarray(mesh.tetrahedra, dtype=np.int64).T,
        sort_t=False,
        validate=True,
    )
    element = ElementTetP2()
    basis = Basis(finite_mesh, element, intorder=P2_PRODUCT_INTEGRATION_ORDER)
    boundary_facets = _boundary_facets(finite_mesh, mesh)
    boundary_mass = {
        wall: _real_csr(
            asm(
                mass,
                FacetBasis(
                    finite_mesh,
                    element,
                    intorder=P2_PRODUCT_INTEGRATION_ORDER,
                    facets=facets,
                ),
            )
        )
        for wall, facets in boundary_facets.items()
    }
    return P2Operators(
        basis=basis,
        stiffness=_real_csr(asm(laplace, basis)),
        mass=_real_csr(asm(mass, basis)),
        boundary_mass=boundary_mass,
    )


def _point_operator(
    operators: P2Operators,
    point: Point,
) -> csr_matrix[np.float64]:
    """回傳該點的精確 P2 取值列；點在網格外時保留上游 ValueError。"""
    coordinates = np.asarray(point.as_tuple(), dtype=np.float64).reshape((3, 1))
    return _real_csr(operators.basis.probes(coordinates).tocsr())


def point_source_load(
    operators: P2Operators,
    source: Point,
) -> NDArray[np.complex128]:
    """組出 ``4π q̄(x_s)`` 的 complex128 P2 點音源載荷。"""
    probe = _point_operator(operators, source)
    return np.array(
        POINT_SOURCE_STRENGTH * probe.toarray().ravel(),
        dtype=np.complex128,
        order="C",
        copy=True,
    )


def interpolate_pressure(
    operators: P2Operators,
    receiver: Point,
    nodal_pressure: NDArray[np.complex128],
) -> complex:
    """以接收點的精確 P2 基底值內插一組自由度聲壓。"""
    probe = _point_operator(operators, receiver)
    return complex((probe @ nodal_pressure)[0])


def _checked_impedances(wall_impedances: WallImpedances) -> dict[Wall, float | None]:
    """要求六面牆恰好各一個正實數阻抗或 ``None`` 剛性記號。"""
    walls = set(Wall.all())
    if set(wall_impedances) != walls:
        raise ValueError("wall_impedances 必須恰好包含 Wall.all() 的六面牆")
    checked: dict[Wall, float | None] = {}
    for wall in Wall.all():
        value = wall_impedances[wall]
        if value is not None and (not math.isfinite(value) or value <= 0.0):
            raise ValueError(f"{wall.wall_name()} 的阻抗必須是正有限實數或 None")
        checked[wall] = value
    return checked


def assemble_helmholtz_system(
    operators: P2Operators,
    *,
    frequency_hz: float,
    wall_impedances: WallImpedances,
    density_kg_m3: float,
    sound_speed_m_s: float,
) -> csr_matrix[np.complex128]:
    """組出一個頻點的複數對稱 ``K-k²M+jωρΣ(B/Z)``。"""
    omega = 2.0 * math.pi * frequency_hz
    wave_number = omega / sound_speed_m_s
    system = (operators.stiffness - wave_number**2 * operators.mass).astype(
        np.complex128
    )
    for wall, impedance in _checked_impedances(wall_impedances).items():
        if impedance is not None:
            coefficient = 1j * omega * density_kg_m3 / impedance
            boundary_term = operators.boundary_mass[wall].astype(np.complex128)
            boundary_term.data *= coefficient
            system = system + boundary_term
    return _complex_csr(system.tocsr())


def solve_frequency_responses(
    operators: P2Operators,
    *,
    right_hand_side: NDArray[np.complex128],
    receiver: Point,
    frequencies_hz: Sequence[float] | NDArray[np.float64],
    wall_impedances: WallImpedances,
    density_kg_m3: float,
    sound_speed_m_s: float,
) -> NDArray[np.complex128]:
    """用同一個 PARDISO 物件逐頻 refactor，回每頻接收點聲壓。"""
    return solve_frequency_responses_many(
        operators,
        right_hand_sides={"source": right_hand_side},
        receivers={"receiver": receiver},
        frequencies_hz=frequencies_hz,
        wall_impedances=wall_impedances,
        density_kg_m3=density_kg_m3,
        sound_speed_m_s=sound_speed_m_s,
    )[("source", "receiver")]


def solve_frequency_responses_many(
    operators: P2Operators,
    *,
    right_hand_sides: Mapping[str, NDArray[np.complex128]],
    receivers: Mapping[str, Point],
    frequencies_hz: Sequence[float] | NDArray[np.float64],
    wall_impedances: WallImpedances,
    density_kg_m3: float,
    sound_speed_m_s: float,
) -> dict[tuple[str, str], NDArray[np.complex128]]:
    """每頻只分解一次，各聲源逐一 solve，再用 P2 列取每個接收點。"""
    runtime.preload_mkl()
    runtime.set_pardiso_threads()
    from pydiso.mkl_solver import MKLPardisoSolver

    probes = {name: _point_operator(operators, point) for name, point in receivers.items()}
    pressures: dict[tuple[str, str], list[complex]] = {
        (source, receiver): [] for source in right_hand_sides for receiver in receivers
    }
    solver: MKLPardisoSolver | None = None
    for frequency in frequencies_hz:
        system = assemble_helmholtz_system(
            operators,
            frequency_hz=float(frequency),
            wall_impedances=wall_impedances,
            density_kg_m3=density_kg_m3,
            sound_speed_m_s=sound_speed_m_s,
        )
        if solver is None:
            solver = MKLPardisoSolver(
                system,
                matrix_type="complex_symmetric",
                factor=True,
            )
        else:
            solver.refactor(system)
        for source_name, right_hand_side in right_hand_sides.items():
            solution = solver.solve(right_hand_side)
            for receiver_name, probe in probes.items():
                pressures[(source_name, receiver_name)].append(
                    complex((probe @ solution)[0])
                )
    return {
        pair: np.asarray(values, dtype=np.complex128)
        for pair, values in pressures.items()
    }


def solve_fem_helmholtz(
    mesh: ShoeboxMesh,
    *,
    wall_impedances: WallImpedances,
    source: Point,
    receiver: Point,
    frequencies_hz: Sequence[float] | NDArray[np.float64],
    density_kg_m3: float,
    sound_speed_m_s: float,
) -> NDArray[np.complex128]:
    """組裝一次 P2 空間，再逐頻重新分解並回傳接收點複數聲壓。"""
    operators = assemble_p2_operators(mesh)
    right_hand_side = point_source_load(operators, source)
    return solve_frequency_responses(
        operators,
        right_hand_side=right_hand_side,
        receiver=receiver,
        frequencies_hz=frequencies_hz,
        wall_impedances=wall_impedances,
        density_kg_m3=density_kg_m3,
        sound_speed_m_s=sound_speed_m_s,
    )
