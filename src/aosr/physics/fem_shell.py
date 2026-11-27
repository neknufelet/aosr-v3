"""任意形狀房間的低頻有限元素：硬牆模態與逐圖層阻抗的頻率響應（票 #1310）。

跟長方體那一路（:mod:`aosr.physics.fem_helmholtz`）同一套算式與慣例：P2 四面體、四階積分、
``e^(+jωt)``、``A = K - k²M + jωρ Σ_l (B_l / Z_l)``、右側是音源點 P2 基底值乘 ``4π``。
差別只在邊界：長方體每面一個牆名，這裡每個材料圖層一份邊界質量 ``B_l``，``None`` 表示硬牆。

求解用 PARDISO（跟長方體那一路同一個），模態用 scipy 的位移反轉 Lanczos（ARPACK）。
物理量（聲速、密度）一律由呼叫端傳入。
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from skfem import Basis, ElementTetP2, FacetBasis, MeshTet, asm
from skfem.models.poisson import laplace, mass

from aosr import runtime
from aosr.geometry.room_shell_mesh import ShellMesh
from aosr.physics.fem_helmholtz import P2_PRODUCT_INTEGRATION_ORDER, POINT_SOURCE_STRENGTH

Vec3 = tuple[float, float, float]
Impedance = float | complex
LayerImpedances = Mapping[str, Impedance | None | Sequence[Impedance | None]]


@dataclass(frozen=True)
class ShellOperators:
    """同一張網格的 P2 基底、K、M 與每個圖層一份邊界質量。"""

    basis: Basis
    stiffness: csr_matrix[np.float64]
    mass: csr_matrix[np.float64]
    layer_mass: dict[str, csr_matrix[np.float64]]


def _csr(matrix: csr_matrix[np.float64]) -> csr_matrix[np.float64]:
    result = csr_matrix(matrix, dtype=np.float64)
    result.sum_duplicates()
    result.sort_indices()
    return result


def _layer_facets(finite: MeshTet, mesh: ShellMesh) -> dict[str, NDArray[np.int64]]:
    """邊界三角形逐一對到 scikit-fem 的 facet；對不上、重複或漏掉都擋。"""
    boundary = finite.boundary_facets()
    if len(boundary) != len(mesh.boundary_triangles):
        raise ValueError("四面體網格的邊界 facet 數與邊界三角形數不同")
    by_vertices = {tuple(sorted(int(n) for n in finite.facets[:, f])): int(f) for f in boundary}
    grouped: dict[str, list[int]] = {name: [] for name in mesh.layer_names}
    seen: set[int] = set()
    for triangle, layer in zip(mesh.boundary_triangles, mesh.boundary_layers, strict=True):
        facet = by_vertices.get(tuple(sorted(int(n) for n in triangle)))
        if facet is None or facet in seen:
            raise ValueError("邊界三角形不屬於四面體網格的表面，或重複")
        seen.add(facet)
        grouped[mesh.layer_names[int(layer)]].append(facet)
    return {name: np.asarray(facets, dtype=np.int64) for name, facets in grouped.items()}


def assemble_shell_operators(mesh: ShellMesh) -> ShellOperators:
    finite = MeshTet(np.asarray(mesh.nodes, dtype=np.float64).T, np.asarray(mesh.tetrahedra, dtype=np.int64).T,
                     sort_t=False, validate=True)
    element = ElementTetP2()
    basis = Basis(finite, element, intorder=P2_PRODUCT_INTEGRATION_ORDER)
    layer_mass = {
        name: _csr(asm(mass, FacetBasis(finite, element, intorder=P2_PRODUCT_INTEGRATION_ORDER, facets=facets)))
        for name, facets in _layer_facets(finite, mesh).items()
    }
    return ShellOperators(basis, _csr(asm(laplace, basis)), _csr(asm(mass, basis)), layer_mass)


def rigid_mode_frequencies(operators: ShellOperators, *, count: int, sound_speed_m_s: float) -> NDArray[np.float64]:
    """全部硬牆時最低的 ``count`` 個模態頻率（Hz，含 0 Hz 的那一個），由低到高。"""
    # 位移取一個小負數：K 是半正定、有零特徵值，位移在 0 會奇異。
    shift = -((2.0 * math.pi / sound_speed_m_s) ** 2)
    # 起始向量固定，否則 ARPACK 每跑一次隨機起點、末位會飄。
    start = np.ones(operators.stiffness.shape[0], dtype=np.float64)
    values = eigsh(operators.stiffness, k=count, M=operators.mass, sigma=shift, which="LM", v0=start,
                   return_eigenvectors=False)
    wave_numbers = np.sqrt(np.clip(np.sort(np.real(values)), 0.0, None))
    return np.asarray(wave_numbers * sound_speed_m_s / (2.0 * math.pi), dtype=np.float64)


def _probe(operators: ShellOperators, point: Vec3) -> csr_matrix[np.float64]:
    """該點的精確 P2 取值列；點在網格外（房外或造型裡）時 scikit-fem 原樣丟 ValueError。"""
    return csr_matrix(operators.basis.probes(np.asarray(point, dtype=np.float64).reshape((3, 1))))


def _checked(impedances: LayerImpedances, operators: ShellOperators,
             frequencies_hz: Sequence[float]) -> dict[str, tuple[complex | None, ...]]:
    """每個圖層一列逐頻阻抗（給單值就整軸同值）；整列硬牆或整列有值，有值須有限、實部 ≥ 0 且非零。"""
    if set(impedances) != set(operators.layer_mass):
        missing = sorted(set(operators.layer_mass) - set(impedances))
        extra = sorted(set(impedances) - set(operators.layer_mass))
        raise ValueError(f"每個圖層都要給阻抗（None＝硬牆）；少了 {missing}，多了 {extra}")
    if not len(frequencies_hz) or any(not math.isfinite(f) or f <= 0 for f in frequencies_hz):
        raise ValueError("頻率軸必須非空且全為有限正數")
    result: dict[str, tuple[complex | None, ...]] = {}
    for name, value in impedances.items():
        row = tuple(value) if isinstance(value, Sequence) else (value,) * len(frequencies_hz)
        if len(row) != len(frequencies_hz):
            raise ValueError(f"圖層 {name} 的阻抗列長度與頻率軸不同")
        if any(cell is None for cell in row) and any(cell is not None for cell in row):
            raise ValueError(f"圖層 {name} 混用硬牆與有值阻抗")
        cells: list[complex | None] = []
        for cell in row:
            if cell is None:
                cells.append(None)
                continue
            z = complex(cell)
            if not (math.isfinite(z.real) and math.isfinite(z.imag)) or z.real < 0 or z == 0:
                raise ValueError(f"圖層 {name} 的阻抗須有限、實部 ≥ 0 且非零")
            cells.append(z)
        result[name] = tuple(cells)
    return result


def solve_shell_responses(
    operators: ShellOperators, *, layer_impedances: LayerImpedances, source: Vec3, receivers: Sequence[Vec3],
    frequencies_hz: Sequence[float], density_kg_m3: float, sound_speed_m_s: float,
) -> NDArray[np.complex128]:
    """(頻點數, 接收點數) 的複數聲壓；每頻重新分解，跟長方體那一路同一個直接解。

    ``layer_impedances`` 每個圖層給單值（整軸同值）、``None``（硬牆）或與頻率軸等長的一列。
    """
    impedances = _checked(layer_impedances, operators, frequencies_hz)
    load = np.array(POINT_SOURCE_STRENGTH * _probe(operators, source).toarray().ravel(), dtype=np.complex128)
    probes = [_probe(operators, point) for point in receivers]
    runtime.preload_mkl()
    runtime.set_pardiso_threads()
    from pydiso.mkl_solver import MKLPardisoSolver

    result = np.zeros((len(frequencies_hz), len(receivers)), dtype=np.complex128)
    solver = None
    anchor = None
    for row_index, frequency in enumerate(frequencies_hz):
        row_impedances = {name: row[row_index] for name, row in impedances.items()}
        system = _system(operators, row_impedances, float(frequency), density_kg_m3, sound_speed_m_s)
        if solver is None or anchor is None:
            solver = MKLPardisoSolver(system, matrix_type="complex_symmetric", factor=False)
            anchor = system
        # pydiso 只比非零個數，非零位置要自己核，跟長方體那一路同一道關。
        if not (np.array_equal(system.indptr, anchor.indptr) and np.array_equal(system.indices, anchor.indices)):
            raise ValueError("有限元素矩陣非零位置與第一個頻點不同")
        solver.refactor(system)
        pressure = solver.solve(load)
        result[row_index] = [complex((probe @ pressure)[0]) for probe in probes]
    return result


def _system(operators: ShellOperators, impedances: Mapping[str, complex | None], frequency_hz: float,
            density_kg_m3: float, sound_speed_m_s: float) -> csr_matrix[np.complex128]:
    """一個頻點的 ``K - k²M + jωρ Σ B_l/Z_l``；每個圖層矩陣都疊上去（硬牆乘 0），非零位置每頻一樣。"""
    omega = 2.0 * math.pi * frequency_hz
    system = (operators.stiffness - (omega / sound_speed_m_s) ** 2 * operators.mass).astype(np.complex128)
    for name, z in impedances.items():
        coefficient = 0.0 if z is None else 1j * omega * density_kg_m3 / z
        system = system + operators.layer_mass[name].astype(np.complex128) * coefficient
    result = csr_matrix(system, dtype=np.complex128)
    result.sum_duplicates()
    result.sort_indices()
    return csr_matrix((np.array(result.data, order="C"), np.array(result.indices, order="C"),
                       np.array(result.indptr, order="C")), shape=result.shape)
