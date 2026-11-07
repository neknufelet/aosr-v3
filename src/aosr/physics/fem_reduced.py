"""有限元素替身方程式（投影式降階）：完整模型快照 → 實數 SVD 基底 → 投影小矩陣（#561）。

決策紙 ``docs/decisions/fem-reduced-model-jax-evaluation-for-stage-three.md``。公式照 #610 實驗：
``A(ω) = K − k²M + Σ_w (jωρ/Z_w) B_w``，``k² = (ω/c)²``；K、M、各面片 B 跟頻率與材料無關，
所以投影一次 ``Vᵀ K V``、``Vᵀ M V``、``Vᵀ B_w V``、``Vᵀ b``、``c V`` 之後，每個頻點只解 r×r 的小系統。
完整解一律走產品的 :func:`~aosr.physics.fem_helmholtz.assemble_helmholtz_system` 與 pydiso，
每個（材料、頻點）重新分解；殘差超過設定的上限就報錯，不收那個快照、不退回別的解法。
快照拆成 [Re, Im] 兩欄做未加權的實數 SVD（V 是實數，投影保持複數對稱）。
這一支只用 numpy／scipy／pydiso，不碰 JAX；:func:`reduced_pressure` 是純陣列函式，之後可換 ``jax.numpy``。
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import svd
from scipy.sparse import csr_matrix

from aosr import runtime
from aosr.config.fem_lane import FEM_ELEMENTS_PER_WAVELENGTH, FEM_MESH_RANDOM_SEED
from aosr.config.fem_reduced import ReducedModelSettings
from aosr.geometry.shoebox import Point, Room, Wall
from aosr.geometry.shoebox_mesh import ShoeboxMesh, WallPartitions, generate_shoebox_mesh
from aosr.physics.fem_helmholtz import (
    Impedance, P2Operators, assemble_helmholtz_system, assemble_p2_operators,
    assemble_zoned_helmholtz_system, point_source_load,
)

if TYPE_CHECKING:
    from pydiso.mkl_solver import MKLPardisoSolver

from aosr.physics.fem_reduced_boundary import (
    BoundaryImpedances, FrequencyBoundaryImpedances, PatchKeys, checked_patch_rows, patch_values,
)
from aosr.physics.fem_reduced_boundary import (
    FrequencyPatchImpedances as FrequencyPatchImpedances, ImpedanceRows as ImpedanceRows,
    boundary_patches as boundary_patches,
)

_MESH_FIELDS = ("nodes", "tetrahedra", "boundary_triangles", "boundary_wall_indices")


@dataclass(frozen=True)
class FullOperators:
    """一間房的完整模型：P2 算子、每支喇叭的點源右側、每個座位的取值列、網格雜湊與物理身分。"""

    p2: P2Operators
    source_points: Mapping[str, Point]
    receiver_points: Mapping[str, Point]
    sources: Mapping[str, NDArray[np.complex128]]
    receivers: Mapping[str, NDArray[np.float64]]
    density_kg_m3: float
    sound_speed_m_s: float
    mesh_hash: str
    physics_identity: str
    patch_keys: PatchKeys = boundary_patches()

    @property
    def stiffness(self) -> csr_matrix[np.float64]:
        return self.p2.stiffness

    @property
    def mass(self) -> csr_matrix[np.float64]:
        return self.p2.mass

    @property
    def boundary(self) -> tuple[csr_matrix[np.float64], ...]:
        """面片邊界質量，順序由方案的 ``patch_keys`` 指定。"""
        return tuple(self.p2.zone_boundary_mass[wall][zone] if zone else self.p2.boundary_mass[wall]
                     for wall, zone in self.patch_keys)


def operators_digest(operators: FullOperators) -> str:
    """完整模型輸出的 SHA-256；CSR 正規化，不收物理身分、不改輸入。

    物理層不能拿報告層的 ``feed_segment``，此處使用相同的名稱、零位元組、八位元組長度框架。
    浮點陣列先核雙精度，索引統一 int64；所有位元組固定小端與 C 順序。
    """
    digest = hashlib.sha256()

    def feed(label: str, content: bytes) -> None:
        digest.update(label.encode() + b"\0" + len(content).to_bytes(8, "big") + content)

    def array(label: str, values: NDArray[np.generic], dtype: np.dtype[np.generic]) -> None:
        if values.dtype != dtype:
            raise ValueError(f"完整模型 {label} 必須是 {dtype}，收到 {values.dtype}")
        feed(label + ".shape", np.asarray(values.shape, dtype="<i8").tobytes())
        feed(label + ".data", np.ascontiguousarray(values, dtype=dtype.newbyteorder("<")).tobytes())

    for label, matrix in (("K", operators.stiffness), ("M", operators.mass),
                           *((f"B.{index}", item) for index, item in enumerate(operators.boundary))):
        normalized = matrix.copy()
        normalized.sum_duplicates()
        normalized.sort_indices()
        normalized.eliminate_zeros()
        feed(label + ".shape", np.asarray(normalized.shape, dtype="<i8").tobytes())
        feed(label + ".indptr", np.asarray(normalized.indptr, dtype="<i8").tobytes())
        feed(label + ".indices", np.asarray(normalized.indices, dtype="<i8").tobytes())
        array(label, normalized.data, np.dtype(np.float64))
    for label, values, dtype in (("sources", operators.sources, np.dtype(np.complex128)),
                                  ("receivers", operators.receivers, np.dtype(np.float64))):
        feed(label + ".names", json.dumps(sorted(values), ensure_ascii=False).encode())
        for name in sorted(values):
            array(label + "." + name, values[name], dtype)
    feed("patch_keys", json.dumps([[wall.wall_name(), zone] for wall, zone in operators.patch_keys]).encode())
    feed("density", float(operators.density_kg_m3).hex().encode())
    feed("sound_speed", float(operators.sound_speed_m_s).hex().encode())
    feed("mesh_hash", operators.mesh_hash.encode())
    return digest.hexdigest()


@dataclass(frozen=True)
class ReducedMatrices:
    """投影後的小矩陣：K_r、M_r（r×r）、B_r（面片數×r×r）、f_r（喇叭×r）、c_r（座位×r）。"""

    stiffness: NDArray[np.float64]
    mass: NDArray[np.float64]
    boundary: NDArray[np.float64]
    sources: NDArray[np.complex128]
    receivers: NDArray[np.float64]


@dataclass(frozen=True)
class GreedyRound:
    """一輪殘差貪婪：當時的快照數與基底大小、抽了幾個候選、抽到的殘差。

    ``proposed`` 是殘差最大、準備完整解的點數；``solved`` 是之後實際完整解的點數——這輪殘差已在停止值內
    就是 0，解次上限只剩幾次就只解那幾次。
    """

    round: int
    snapshots: int
    rank: int
    candidates: int
    max_residual: float
    median_residual: float
    proposed: int
    solved: int


@dataclass(frozen=True)
class BuildLedger:
    """建模帳：完整解（分解）次數、輪數與每輪殘差、停止原因、最後基底大小、各段秒數、當次設定。"""

    full_solves: int
    snapshots: int
    rank: int
    rounds: tuple[GreedyRound, ...]
    stop_reason: str
    seconds: Mapping[str, float]
    settings: ReducedModelSettings


@dataclass(frozen=True)
class ReducedModel:
    """替身本體與它的出身：頻率軸、空氣、喇叭與座位順序、網格雜湊、物理身分、輸入雜湊、建模帳。"""

    matrices: ReducedMatrices
    source_names: tuple[str, ...]
    receiver_names: tuple[str, ...]
    frequencies_hz: tuple[float, ...]
    density_kg_m3: float
    sound_speed_m_s: float
    mesh_hash: str
    physics_identity: str
    input_hash: str
    ledger: BuildLedger
    patch_keys: PatchKeys = boundary_patches()


def mesh_hash(mesh: ShoeboxMesh) -> str:
    """串名稱、dtype、形狀與 C 順序位元組做 SHA-256；只有分區網格附加區代號。

    整面（None 或全空字串）仍只量原有四欄，保持原雜湊逐位不變。
    """
    digest = hashlib.sha256()
    fields: tuple[str, ...] = _MESH_FIELDS
    if mesh.boundary_zone_ids is not None and np.any(mesh.boundary_zone_ids != ""):
        fields = (*fields, "boundary_zone_ids")
    for name in fields:
        array = np.ascontiguousarray(getattr(mesh, name))
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _probe_row(operators: P2Operators, point: Point) -> NDArray[np.float64]:
    """跟產品取點同一個 ``basis.probes`` 呼叫，攤成 (N,) 實數列。"""
    coordinates = np.asarray(point.as_tuple(), dtype=np.float64).reshape((3, 1))
    return np.asarray(operators.basis.probes(coordinates).toarray().ravel(), dtype=np.float64)


def export_room_operators(
    room: Room, sources: Mapping[str, Point], receivers: Mapping[str, Point], *,
    density_kg_m3: float, sound_speed_m_s: float, physics_identity: str, max_frequency_hz: float,
    partitions: WallPartitions | None = None,
) -> FullOperators:
    """用產品的網格（每波長格數與種子取 ``fem_lane``）與 P2 組裝，交出完整模型的全部算子。"""
    if not sources or not receivers:
        raise ValueError("喇叭與座位都不能是空的")
    mesh = generate_shoebox_mesh(
        room, max_frequency_hz=max_frequency_hz, elements_per_wavelength=FEM_ELEMENTS_PER_WAVELENGTH,
        sound_speed_m_s=sound_speed_m_s, random_seed=FEM_MESH_RANDOM_SEED, partitions=partitions)
    p2 = assemble_p2_operators(mesh)
    return FullOperators(
        p2=p2, source_points=dict(sources), receiver_points=dict(receivers),
        sources={name: point_source_load(p2, point) for name, point in sources.items()},
        receivers={name: _probe_row(p2, point) for name, point in receivers.items()},
        density_kg_m3=density_kg_m3, sound_speed_m_s=sound_speed_m_s,
        mesh_hash=mesh_hash(mesh), physics_identity=physics_identity,
        patch_keys=boundary_patches(None if partitions is None else
                                    {wall: tuple(item[0] for item in zones) for wall, zones in partitions.items()}))


def system_coefficients(
    frequency_hz: float, wall_impedances: BoundaryImpedances, *,
    density_kg_m3: float, sound_speed_m_s: float, patch_keys: PatchKeys = boundary_patches(),
) -> tuple[float, NDArray[np.complex128]]:
    """回 ``k² = (2πf/c)²`` 與面片序的各個 ``jωρ/Z``（剛性為 0）；算式與產品組裝同序。"""
    omega = 2.0 * math.pi * frequency_hz
    wave_number = omega / sound_speed_m_s
    values = patch_values(wall_impedances, patch_keys)
    coefficients = [0j if (value := values[key]) is None else 1j * omega * density_kg_m3 / value
                    for key in patch_keys]
    return wave_number**2, np.asarray(coefficients, dtype=np.complex128)


def reduced_pressure(
    stiffness: NDArray[np.float64], mass: NDArray[np.float64], boundary: NDArray[np.float64],
    sources: NDArray[np.complex128], receivers: NDArray[np.float64],
    k_squared: float, wall_coefficients: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    """純陣列：``(K_r − k²M_r + Σ c_w B_r,w) q = f_r``，回 (喇叭, 座位) 複數聲壓；沒有 IO 與全域狀態。"""
    system = stiffness - k_squared * mass + np.tensordot(wall_coefficients, boundary, axes=1)
    coordinates = np.linalg.solve(system, sources.T)
    return np.asarray((receivers @ coordinates).T, dtype=np.complex128)


def evaluate_reduced(
    model: ReducedMatrices, k_squared: float, wall_coefficients: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    """一個頻點的替身聲壓 (喇叭, 座位)；``wall_coefficients`` 順序與面片邊界矩陣相同。"""
    return reduced_pressure(model.stiffness, model.mass, model.boundary, model.sources,
                            model.receivers, k_squared, wall_coefficients)


class FullSolver:
    """同一個 pydiso 物件錨定第一個矩陣的非零位置，之後每個（材料、頻點）都 ``refactor`` 再逐支喇叭解。"""

    def __init__(self, operators: FullOperators, *, max_relative_residual: float) -> None:
        self._operators = operators
        self._limit = max_relative_residual
        self._anchor: csr_matrix[np.complex128] | None = None
        self._solver: MKLPardisoSolver | None = None
        self.solves = 0

    def system(self, frequency_hz: float, walls: BoundaryImpedances) -> csr_matrix[np.complex128]:
        operators = self._operators
        values = patch_values(walls, operators.patch_keys)
        if operators.patch_keys != boundary_patches():
            zones: dict[Wall, dict[str, Impedance]] = {}
            for (wall, zone), value in values.items():
                if zone:
                    if value is None:
                        raise ValueError("分區一定要有材料阻抗")
                    zones.setdefault(wall, {})[zone] = value
            ordered_p2 = replace(operators.p2, zone_boundary_mass={
                wall: {zone: operators.p2.zone_boundary_mass[wall][zone] for zone in values}
                for wall, values in zones.items()})
            return assemble_zoned_helmholtz_system(
                ordered_p2, frequency_hz=frequency_hz,
                wall_impedances={wall: value for (wall, zone), value in values.items() if not zone},
                zone_impedances=zones, density_kg_m3=operators.density_kg_m3,
                sound_speed_m_s=operators.sound_speed_m_s)
        return assemble_helmholtz_system(
            operators.p2, frequency_hz=frequency_hz,
            wall_impedances={wall: values[(wall, "")] for wall in Wall.all()},
            density_kg_m3=operators.density_kg_m3, sound_speed_m_s=operators.sound_speed_m_s)

    def _refactored(self, system: csr_matrix[np.complex128]) -> MKLPardisoSolver:
        if self._solver is None or self._anchor is None:
            runtime.preload_mkl()
            runtime.set_pardiso_threads()
            from pydiso.mkl_solver import MKLPardisoSolver as solver_type

            self._anchor = system
            self._solver = solver_type(system, matrix_type="complex_symmetric", factor=False)
        if not (np.array_equal(system.indptr, self._anchor.indptr)
                and np.array_equal(system.indices, self._anchor.indices)):
            raise ValueError("有限元素矩陣非零位置與錨定矩陣不同")
        self._solver.refactor(system)
        self.solves += 1
        return self._solver

    def solve(self, frequency_hz: float, walls: BoundaryImpedances) -> NDArray[np.complex128]:
        """回 (喇叭, N) 完整解；任何一支的相對殘差超過上限或不是有限數就報錯。"""
        system = self.system(frequency_hz, walls)
        solver = self._refactored(system)
        rows = []
        for name, load in self._operators.sources.items():
            solution = np.asarray(solver.solve(load), dtype=np.complex128)
            residual = relative_residual(system, solution, load)
            if not math.isfinite(residual) or residual > self._limit:
                raise RuntimeError(f"完整解殘差失常：{frequency_hz} Hz、喇叭 {name}、相對殘差 {residual}")
            rows.append(solution)
        return np.asarray(rows, dtype=np.complex128)


def relative_residual(system: csr_matrix[np.complex128], solution: NDArray[np.complex128],
                      load: NDArray[np.complex128]) -> float:
    """直接形成 ``b − A u`` 再取範數比；不用 Gram 平方差那種會相消的寫法。"""
    product = np.asarray(system @ solution, dtype=np.complex128)
    return float(np.linalg.norm(load - product) / np.linalg.norm(load))


def snapshot_basis(snapshots: Sequence[NDArray[np.complex128]], cutoff: float) -> NDArray[np.float64]:
    """每個複數快照拆 [Re, Im] 兩欄，未加權、不逐欄正規化，SVD 後丟掉小於 ``s₀·cutoff`` 的方向。"""
    stacked = np.asarray(snapshots).T
    columns = np.empty((stacked.shape[0], 2 * stacked.shape[1]), dtype=np.float64, order="F")
    columns[:, 0::2] = stacked.real
    columns[:, 1::2] = stacked.imag
    left, singular, _ = svd(columns, full_matrices=False, check_finite=True, lapack_driver="gesdd")
    rank = int(np.sum(singular > singular[0] * cutoff))
    return np.ascontiguousarray(left[:, :rank])


def project(operators: FullOperators, basis: NDArray[np.float64]) -> ReducedMatrices:
    """``Vᵀ A V`` 用普通轉置；面片只投影其非零列、行的聯集支撐。"""
    def squeeze(matrix: csr_matrix[np.float64]) -> NDArray[np.float64]:
        return np.asarray(basis.T @ np.asarray(matrix @ basis, dtype=np.float64), dtype=np.float64)

    def boundary_squeeze(matrix: csr_matrix[np.float64]) -> NDArray[np.float64]:
        indices = np.union1d(*matrix.nonzero())
        local_basis = basis[indices]
        local_matrix = matrix[indices][:, indices]
        return np.asarray(local_basis.T @ (local_matrix @ local_basis), dtype=np.float64)

    return ReducedMatrices(
        stiffness=squeeze(operators.stiffness), mass=squeeze(operators.mass),
        boundary=np.asarray([boundary_squeeze(matrix) for matrix in operators.boundary], dtype=np.float64),
        sources=np.asarray([basis.T @ load for load in operators.sources.values()], dtype=np.complex128),
        receivers=np.asarray([row @ basis for row in operators.receivers.values()], dtype=np.float64))


def checked_axis(frequencies_hz: Sequence[float]) -> tuple[float, ...]:
    """頻率軸非空、有限正數、嚴格遞增。"""
    axis = tuple(float(value) for value in frequencies_hz)
    if not axis or any(not math.isfinite(value) or value <= 0.0 for value in axis):
        raise ValueError("頻率軸必須非空且全為有限正數")
    if any(left >= right for left, right in zip(axis, axis[1:])):
        raise ValueError("頻率軸必須嚴格遞增")
    return axis


def impedance_rows(sets: Sequence[FrequencyBoundaryImpedances], axis: tuple[float, ...], *,
                   patch_keys: PatchKeys = boundary_patches()) -> tuple[ImpedanceRows, ...]:
    """每組每面片逐頻阻抗都過產品閘門，剛性只限整面面片。"""
    if not sets:
        raise ValueError("材料組不能是空的")
    return tuple(checked_patch_rows(values, axis, patch_keys) for values in sets)


def walls_at(rows: ImpedanceRows, index: int) -> dict[tuple[Wall, str], Impedance | None]:
    return {key: row[index] for key, row in rows.items()}


def _hex_rows(rows: ImpedanceRows, keys: PatchKeys) -> list[object]:
    return [None if row[0] is None else
            [[complex(value).real.hex(), complex(value).imag.hex()] for value in row if value is not None]
            for row in (rows[key] for key in keys)]


def input_hash(operators: FullOperators, axis: tuple[float, ...], training: tuple[ImpedanceRows, ...],
               settings: ReducedModelSettings) -> str:
    """頻率軸、訓練材料、設定、空氣、網格雜湊、物理身分、喇叭與座位座標的正規 JSON 做 SHA-256。"""
    def points(items: Mapping[str, Point]) -> dict[str, list[str]]:
        return {name: [float(value).hex() for value in point.as_tuple()] for name, point in items.items()}

    document = {
        "frequencies_hex": [value.hex() for value in axis],
        "training": [_hex_rows(rows, operators.patch_keys) for rows in training],
        "settings": settings.model_dump(mode="json"),
        "density_hex": float(operators.density_kg_m3).hex(),
        "speed_hex": float(operators.sound_speed_m_s).hex(),
        "mesh_hash": operators.mesh_hash, "physics_identity": operators.physics_identity,
        "sources": points(operators.source_points), "receivers": points(operators.receiver_points),
    }
    if operators.patch_keys != boundary_patches():
        document["patches"] = [[wall.wall_name(), zone] for wall, zone in operators.patch_keys]
    text = json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def even_indices(count: int, size: int) -> tuple[int, ...]:
    """在 0..size−1 上平均挑 ``count`` 個（四捨五入、去重、遞增）。"""
    picked = np.rint(np.linspace(0, size - 1, min(count, size))).astype(np.int64)
    return tuple(int(value) for value in np.unique(picked))


def elapsed_since(start: float) -> float:
    return time.perf_counter() - start
