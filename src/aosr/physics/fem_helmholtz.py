"""用 P2 四面體組裝並求解鞋盒房的頻域 Helmholtz 聲壓。

採 ``e^(+jωt)``：``A = K - k²M + jωρ Σ(B_w/Z_w)``，右側是音源點
P2 基底值乘 ``4π``。每面可用固定實數或逐頻複數阻抗，``None`` 表示整軸剛性牆。
PARDISO 同一個 solver 物件在每個後續頻點呼叫 ``refactor``，因此每點重新分解；
解法例外不攔截、不退回其他直接解或疊代解。
"""
from __future__ import annotations

import math
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import TypeVar

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
Impedance = float | complex
SingleWallImpedances = Mapping[Wall, Impedance | None]
WallImpedances = Mapping[Wall, float | None]
FrequencyWallImpedances = Mapping[Wall, Impedance | None | Sequence[Impedance | None]]


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


def _check_impedance(value: Impedance, wall: Wall, frequency_hz: float | None) -> None:
    """被動且非零；零實部的純電抗可用，不裁切、不改模型或數值型別。"""
    z = complex(value)
    if not (math.isfinite(z.real) and math.isfinite(z.imag)) or z.real < 0 or z == 0:
        raise ValueError(f"{wall.wall_name()} 在頻率 {frequency_hz} Hz 的阻抗須有限、實部 ≥ 0 且非零")


def _checked_impedances(
    wall_impedances: SingleWallImpedances, frequency_hz: float | None = None,
) -> dict[Wall, Impedance | None]:
    """單頻六面閘門；保留原始 float，使既有組裝算式逐位不變。"""
    if set(wall_impedances) != set(Wall.all()):
        raise ValueError("wall_impedances 必須恰好包含 Wall.all() 的六面牆")
    for wall, value in wall_impedances.items():
        if value is not None:
            _check_impedance(value, wall, frequency_hz)
    return {wall: wall_impedances[wall] for wall in Wall.all()}


def checked_impedance_rows(
    wall_impedances: FrequencyWallImpedances, frequencies_hz: Sequence[float] | NDArray[np.float64],
) -> dict[Wall, tuple[Impedance | None, ...]]:
    """全軸先驗，包含未選分片頻點；同面整軸剛性或整軸有值，非零位置不變。"""
    if set(wall_impedances) != set(Wall.all()):
        raise ValueError("wall_impedances 必須恰好包含 Wall.all() 的六面牆")
    if not len(frequencies_hz) or any(not math.isfinite(f) or f <= 0 for f in frequencies_hz):
        raise ValueError("頻率軸必須非空且全為有限正數")
    rows: dict[Wall, tuple[Impedance | None, ...]] = {}
    for wall in Wall.all():
        value = wall_impedances[wall]
        row = (tuple(value) if isinstance(value, Sequence) else (value,) * len(frequencies_hz))
        if len(row) != len(frequencies_hz):
            raise ValueError(f"{wall.wall_name()} 在頻率 {frequencies_hz[0]} Hz 的阻抗列長度與軸不符")
        rigid = row[0] is None
        for frequency, cell in zip(frequencies_hz, row, strict=True):
            if (cell is None) != rigid:
                raise ValueError(f"{wall.wall_name()} 在頻率 {frequency} Hz 混用剛性與有值阻抗")
            if cell is not None:
                _check_impedance(cell, wall, frequency)
        rows[wall] = row
    return rows


def assemble_helmholtz_system(
    operators: P2Operators,
    *,
    frequency_hz: float,
    wall_impedances: SingleWallImpedances,
    density_kg_m3: float,
    sound_speed_m_s: float,
) -> csr_matrix[np.complex128]:
    """組出一個頻點的複數對稱 ``K-k²M+jωρΣ(B/Z)``。"""
    omega = 2.0 * math.pi * frequency_hz
    wave_number = omega / sound_speed_m_s
    system = (operators.stiffness - wave_number**2 * operators.mass).astype(
        np.complex128
    )
    for wall, impedance in _checked_impedances(wall_impedances, frequency_hz).items():
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
    wall_impedances: FrequencyWallImpedances,
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


SourceKey = TypeVar("SourceKey", bound=Hashable)
ReceiverKey = TypeVar("ReceiverKey", bound=Hashable)


def _frequency_indices(n: int, indices: Sequence[int] | None) -> tuple[int, ...]:
    selected = tuple(range(n)) if indices is None else tuple(indices)
    if not selected or any(isinstance(i, bool) or not isinstance(i, int)
                           or not 0 <= i < n for i in selected):
        raise ValueError("frequency_indices 必須是非空、範圍內的整數索引")
    if any(left >= right for left, right in zip(selected, selected[1:])):
        raise ValueError("frequency_indices 必須嚴格遞增")
    return selected


def _response_pairs(
    sources: Mapping[SourceKey, NDArray[np.complex128]],
    receivers: Mapping[ReceiverKey, Point],
    pairs: Sequence[tuple[SourceKey, ReceiverKey]] | None,
) -> tuple[tuple[SourceKey, ReceiverKey], ...]:
    selected = (tuple((source, receiver) for source in sources for receiver in receivers)
                if pairs is None else tuple(pairs))
    if any(source not in sources or receiver not in receivers for source, receiver in selected):
        raise ValueError("pairs 引用了不存在的聲源或座位鍵")
    return tuple(dict.fromkeys(selected))


def solve_frequency_responses_many(
    operators: P2Operators,
    *,
    right_hand_sides: Mapping[SourceKey, NDArray[np.complex128]],
    receivers: Mapping[ReceiverKey, Point],
    frequencies_hz: Sequence[float] | NDArray[np.float64],
    wall_impedances: FrequencyWallImpedances,
    density_kg_m3: float,
    sound_speed_m_s: float,
    frequency_indices: Sequence[int] | None = None,
    pairs: Sequence[tuple[SourceKey, ReceiverKey]] | None = None,
) -> dict[tuple[SourceKey, ReceiverKey], NDArray[np.complex128]]:
    """每頻只分解一次，各聲源逐一 solve，再用 P2 列取每個接收點。

    每段分析都錨定整條軸的 A(f0)，factor=False 後第一個呼叫必為 refactor，
    避免 solve 偷用建構矩陣分解。第零頻 refactor 使用同一 A0 物件。
    每個右側必須一維逐一解，整塊解會改最後一位；只讀 pairs 指定的座位。
    refactor 前核對完整 CSR 非零位置，因為 pydiso 本身只查非零個數。
    """
    rows = checked_impedance_rows(wall_impedances, frequencies_hz)
    indices = _frequency_indices(len(frequencies_hz), frequency_indices)
    selected_pairs = _response_pairs(right_hand_sides, receivers, pairs)
    if any(load.shape != (operators.basis.N,) for load in right_hand_sides.values()):
        raise ValueError("每個右邊向量必須是一維 (N,)")
    runtime.preload_mkl()
    runtime.set_pardiso_threads()
    from pydiso.mkl_solver import MKLPardisoSolver

    probes = {name: _point_operator(operators, point) for name, point in receivers.items()}
    targets = {source: tuple(receiver for key, receiver in selected_pairs if key == source)
               for source in right_hand_sides}
    pressures: dict[tuple[SourceKey, ReceiverKey], list[complex]] = {
        pair: [] for pair in selected_pairs}
    anchor = assemble_helmholtz_system(
        operators, frequency_hz=float(frequencies_hz[0]), wall_impedances={w: row[0] for w, row in rows.items()},
        density_kg_m3=density_kg_m3, sound_speed_m_s=sound_speed_m_s)
    solver = MKLPardisoSolver(anchor, matrix_type="complex_symmetric", factor=False)
    for index in indices:
        system = anchor if index == 0 else assemble_helmholtz_system(
            operators, frequency_hz=float(frequencies_hz[index]), wall_impedances={w: row[index] for w, row in rows.items()},
            density_kg_m3=density_kg_m3, sound_speed_m_s=sound_speed_m_s)
        if not (np.array_equal(system.indptr, anchor.indptr)
                and np.array_equal(system.indices, anchor.indices)):
            raise ValueError("有限元素矩陣非零位置與錨定矩陣不同")
        solver.refactor(system)
        for source, load in right_hand_sides.items():
            solution = solver.solve(load)
            for receiver in targets[source]:
                pressures[source, receiver].append(complex((probes[receiver] @ solution)[0]))
    return {pair: np.asarray(values, dtype=np.complex128)
            for pair, values in pressures.items()}


def solve_fem_helmholtz(
    mesh: ShoeboxMesh,
    *,
    wall_impedances: FrequencyWallImpedances,
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
