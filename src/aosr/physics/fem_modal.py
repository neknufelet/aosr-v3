"""P2 阻尼模態：λ=k 的對稱第一伴隨線性化，唯一分解器為 pydiso。

由既有 Helmholtz 組裝在 k=1 的虛部取得 Ct，沒有第二份牆面公式。
A=[[-K,0],[0,-M]]、B=[[jCt,-M],[-M,0]]，z=[p;kp]；
eigs 只看到 (A-σB)^(-1)B 的 LinearOperator，不使用 scipy 的稀疏分解。
任何分解或迭代例外原樣向上傳，包含未收斂；不採部分結果或退回解法。

zero_rad_s 不是有限元素精度契約。以全譜尺度（已求得頻譜與
c sqrt(||K||₁/||M||₁) 的較大值）及後向殘差形成舍入預算 η。
剛性零根為二階 Jordan 根，擾動按 sqrt(η) 放大；阻尼時零根為簡根，
按 η c||K||₁/||Ct||₁ 放大，並以剛性界為上限。已知常數靜態根
精確為零，阻尼時再以各移位最靠近零的返回根實測偏移量的最大值
八倍縮緊簡根界；只採上述舍入界內的根。八倍涵蓋移位間的舍入累積，
線性舍入界為下限。不振盪根跨移位去重另容許此絕對雜訊界。
有阻尼時同一移位在靜態界內若有多根，表示靜態與衰減不可分辨，
明確報 ArithmeticError，不能當成僅一個靜態根回傳。
η=64 max(epsilon, 常數解與所收頻譜的後向殘差)。因此第零步的
剛性約 4e-5 rad/s 裂根及約 1.7e-9 虛部雜訊可吸收，阻尼時不把
約 1e-12 的靜態根與有限純衰減根合併。此界隨網格、頻譜、殘差變動。
靜態模長界與 component_zero_rad_s（線性分量雜訊界）分開；正的阻尼
虛部不清零。弱阻尼的有限 T60 不被 Jordan 界抹掉。raw_omega 保留原值。
靜態叢集只留一列，非零根在舍入內的實部回報零。
solutions 保存三種解，modal_table 只包含共振，形狀可選擇保留。
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import bmat, csr_matrix
from scipy.sparse.linalg import LinearOperator, eigs
from scipy.optimize import linear_sum_assignment

from aosr import runtime
from aosr.geometry.shoebox_mesh import ShoeboxMesh
from aosr.physics.fem_helmholtz import (
    P2Operators, WallImpedances, assemble_helmholtz_system, assemble_p2_operators,
)
from aosr.physics.modal_convention import ModalKind, modal_quantities
from aosr.physics.fem_modal_check import DecayOrigin, FemModalCheck, build_check, count_bands


ComplexVector = NDArray[np.complex128]


@dataclass(frozen=True)
class ModalSolverOptions:
    """第零步的移位、個數與迭代規則；皆為數值求解參數而非精度契約。"""

    shift_spacing_hz: float = 50.0
    count_window_half_width_hz: float = 35.0
    count_safety_factor: float = 1.3
    count_extra: int = 12
    eigs_tolerance: float = 1e-10
    max_iterations: int | None = None
    random_seed: int = 20260919
    dedup_relative_tolerance: float = 1e-8
    pardiso_threads: int = 1

    def __post_init__(self) -> None:
        positive = (self.shift_spacing_hz, self.count_window_half_width_hz,
                    self.count_safety_factor, self.eigs_tolerance,
                    self.dedup_relative_tolerance)
        if any(not math.isfinite(value) or value <= 0 for value in positive):
            raise ValueError("移位、視窗、倍率與數值容差必須為正有限數")
        if isinstance(self.count_extra, bool) or not isinstance(self.count_extra, int) or self.count_extra < 0:
            raise ValueError("額外個數必須為非負整數")
        if self.max_iterations is not None and (isinstance(self.max_iterations, bool)
                or not isinstance(self.max_iterations, int) or self.max_iterations < 1):
            raise ValueError("最大迭代次數必須為正整數")
        if isinstance(self.pardiso_threads, bool) or not isinstance(self.pardiso_threads, int) or self.pardiso_threads < 1:
            raise ValueError("分解執行緒數必須為正整數")


@dataclass(frozen=True)
class FemMode:
    """一個解，頻率 Hz、T60 秒、Q 無量綱；形狀是質量歸一化 P2 自由度。"""

    omega: complex
    raw_omega: complex
    frequency_hz: float
    t60_s: float | None
    q: float | None
    kind: ModalKind
    residual: float
    linearized_residual: float
    shift_index: int
    shift_hz: float
    shape: ComplexVector | None
    above_guaranteed_decay: bool = False
    decay_origin: DecayOrigin | None = None
    continuation_index: tuple[int, int, int] | None = None


@dataclass(frozen=True)
class ModalShift:
    """最遠原始返回根的 ω 平面半徑 rad/s，含負頻、靜態、純衰減與超界根。"""

    frequency_hz: float
    requested: int
    radius_rad_s: float


@dataclass(frozen=True)
class FemModalSpectrum:
    """解帳含靜態與不振盪解；共振表另取，供方法驗證與後續參與計算。"""

    solutions: tuple[FemMode, ...]
    shifts: tuple[ModalShift, ...]
    zero_rad_s: float
    component_zero_rad_s: float
    degrees_of_freedom: int
    check: FemModalCheck | None = None

    @property
    def modal_table(self) -> tuple[FemMode, ...]:
        return tuple(row for row in self.solutions if row.kind is ModalKind.RESONANCE)


@dataclass(frozen=True)
class _Pencil:
    operators: P2Operators
    damping: csr_matrix[np.float64]
    a: csr_matrix[np.complex128]
    b: csr_matrix[np.complex128]
    norms: tuple[float, float, float, float, float]


@dataclass(frozen=True)
class _Root:
    omega: complex
    vector: ComplexVector
    shift: int
    residual: float
    linearized_residual: float


def _csr(matrix: csr_matrix[np.complex128]) -> csr_matrix[np.complex128]:
    result = matrix.tocsr(copy=True).astype(np.complex128)
    result.sum_duplicates()
    result.eliminate_zeros()
    result.sort_indices()
    result.data = np.ascontiguousarray(result.data)
    return result


def _norm(matrix: csr_matrix[np.float64] | csr_matrix[np.complex128]) -> float:
    return float(abs(matrix).sum(axis=0).max())


def _pencil(operators: P2Operators, walls: WallImpedances, rho: float, c: float) -> _Pencil:
    system = assemble_helmholtz_system(operators, frequency_hz=c / (2 * math.pi),
                                     wall_impedances=walls, density_kg_m3=rho,
                                     sound_speed_m_s=c)
    damping = csr_matrix((system.data.imag.copy(), system.indices.copy(), system.indptr.copy()),
                         shape=system.shape, dtype=np.float64)
    k, m = operators.stiffness, operators.mass
    kc, mc, dc = k.astype(np.complex128), m.astype(np.complex128), damping.astype(np.complex128)
    dc.data *= 1j
    a = _csr(cast(csr_matrix[np.complex128], bmat([[-kc, None], [None, -mc]], format="csr")))
    b = _csr(cast(csr_matrix[np.complex128], bmat([[dc, -mc], [-mc, None]], format="csr")))
    return _Pencil(operators, damping, a, b, (_norm(k), _norm(damping), _norm(m), _norm(a), _norm(b)))


def _weyl_count(mesh: ShoeboxMesh, frequency: float, c: float) -> float:
    """Neumann 體積與表面兩項估計；不從網格倒猜長方形尺寸。"""
    vertices = mesh.nodes[mesh.tetrahedra]
    volume = float(np.sum(abs(np.linalg.det(vertices[:, 1:] - vertices[:, :1]))) / 6)
    triangles = mesh.nodes[mesh.boundary_triangles]
    surface = float(np.sum(np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                                  triangles[:, 2] - triangles[:, 0]), axis=1)) / 2)
    k = 2 * math.pi * max(0.0, frequency) / c
    return volume * k**3 / (6 * math.pi**2) + surface * k**2 / (16 * math.pi)


def _shift_plan(mesh: ShoeboxMesh, c: float, cap: float, n: int, options: ModalSolverOptions,
                shifts: Sequence[float] | None, counts: Sequence[int] | None,
                references: Sequence[float] | None) -> tuple[tuple[float, int], ...]:
    centers = tuple(shifts) if shifts is not None else tuple(np.arange(options.shift_spacing_hz / 2,
                                                                      cap, options.shift_spacing_hz))
    if not centers:
        centers = (cap / 2,) if shifts is None else ()
    if not centers or any(not math.isfinite(f) or f <= 0 for f in centers) or len(set(centers)) != len(centers):
        raise ValueError("移位必須非空、不重複且為正有限 Hz；零移位使靜態根奇異")
    if references is not None and any(not math.isfinite(f) or f < 0 for f in references):
        raise ValueError("參考頻率必須為非負有限 Hz")
    if counts is not None:
        if len(counts) != len(centers) or any(isinstance(k, bool) or not isinstance(k, int)
                                             or not 0 < k < n - 1 for k in counts):
            raise ValueError("每移位個數必須逐一對應且為 0<k<2N-1 的整數")
        return tuple(zip(centers, counts, strict=True))
    result = []
    for center in centers:
        half = options.count_window_half_width_hz
        local = (sum(abs(f - center) <= half for f in references) if references is not None
                 else _weyl_count(mesh, center + half, c) - _weyl_count(mesh, center - half, c)
                 + (1 if center <= half else 0))
        count = min(n - 2, math.ceil(options.count_safety_factor * local) + options.count_extra)
        result.append((center, max(1, count)))
    return tuple(result)


def _residual(pencil: _Pencil, k: complex, vector: ComplexVector) -> tuple[float, float]:
    ops = pencil.operators
    p = vector[:ops.basis.N]
    nk, nc, nm, na, nb = pencil.norms
    residual = np.asarray(ops.stiffness @ p) + 1j * k * np.asarray(pencil.damping @ p) - k**2 * np.asarray(ops.mass @ p)
    quadratic = float(np.linalg.norm(residual) / ((nk + abs(k) * nc + abs(k)**2 * nm) * np.linalg.norm(p)))
    linear = float(np.linalg.norm(np.asarray(pencil.a @ vector) - k * np.asarray(pencil.b @ vector))
                   / ((na + abs(k) * nb) * np.linalg.norm(vector)))
    return quadratic, linear


def _shift_roots(pencil: _Pencil, center: float, count: int, c: float,
                 shift: int, options: ModalSolverOptions, v0: ComplexVector) -> tuple[list[_Root], ModalShift]:
    from pydiso.mkl_solver import MKLPardisoSolver

    sigma = 2 * math.pi * center / c
    solver = MKLPardisoSolver(_csr(pencil.a - sigma * pencil.b),
                              matrix_type="complex_symmetric", factor=True)

    def matvec(vector: ComplexVector) -> ComplexVector:
        rhs = np.ascontiguousarray(pencil.b @ np.asarray(vector, dtype=np.complex128).ravel())
        return solver.solve(rhs)

    operator = LinearOperator(pencil.a.shape, matvec=matvec, dtype=np.complex128)
    mu, vectors = eigs(operator, k=count, which="LM", tol=options.eigs_tolerance,
                       maxiter=options.max_iterations or 10 * pencil.a.shape[0], v0=v0)
    roots = []
    for j, value in enumerate(mu):
        wave = complex(sigma + 1 / value)
        vector = np.asarray(vectors[:, j], dtype=np.complex128)
        quadratic, linear = _residual(pencil, wave, vector)
        roots.append(_Root(c * wave, vector, shift, quadratic, linear))
    radius = max(abs(root.omega - c * sigma) for root in roots)
    return roots, ModalShift(center, count, radius)


def _zero_scale(pencil: _Pencil, roots: Sequence[_Root], c: float) -> tuple[float, float]:
    nk, nc, nm, _, _ = pencil.norms
    scale = max(c * math.sqrt(nk / nm), *(abs(root.omega) for root in roots))
    constant = np.ones(pencil.operators.basis.N, dtype=np.complex128)
    static_res = float(np.linalg.norm(pencil.operators.stiffness @ constant) / (nk * np.linalg.norm(constant)))
    eta = 64 * max(np.finfo(float).eps, static_res, *(root.residual for root in roots))
    jordan = math.sqrt(eta) * scale
    simple = eta * c * nk / nc if nc else jordan
    observed = [min(abs(root.omega) for root in roots if root.shift == shift)
                for shift in {root.shift for root in roots}]
    near_zero = [value for value in observed if value <= min(jordan, simple)]
    if nc and near_zero:
        simple = min(simple, 8 * max(near_zero))
    component = eta * scale
    return max(component, min(jordan, simple)), component


def _deduplicate(roots: Sequence[_Root], zero: float, tolerance: float) -> list[_Root]:
    """跨移位一對一去重；同移位的簡併各留一列，靜態叢集例外合為一列。"""
    kept: list[_Root] = []
    static = next((root for root in roots if abs(root.omega) <= zero), None)
    for shift in sorted({root.shift for root in roots}):
        incoming = [root for root in roots if root.shift == shift and abs(root.omega) > zero]
        if not incoming:
            continue
        if not kept:
            kept.extend(incoming)
            continue
        # 虛擬列的代價大於全部合法配對代價之和：先最大化配對數，再最小化距離。
        dummy = len(incoming) + 1.0
        costs = np.full((len(incoming), len(kept) + len(incoming)), dummy)
        for i, root in enumerate(incoming):
            for j, old in enumerate(kept):
                limit = tolerance * max(abs(root.omega), abs(old.omega), 2 * math.pi)
                if abs(root.omega.real) <= zero and abs(old.omega.real) <= zero:
                    limit = max(limit, zero)
                distance = abs(root.omega - old.omega) / limit
                costs[i, j] = distance if distance <= 1 else 2 * dummy
        rows, columns = linear_sum_assignment(costs)
        unmatched = [incoming[int(i)] for i, j in zip(rows, columns, strict=True) if j >= len(kept)]
        kept.extend(unmatched)
    if static is not None:
        kept.append(static)
    return kept


def _row(pencil: _Pencil, root: _Root, zero: float, component_zero: float, c: float,
         shifts: Sequence[ModalShift], retain: bool) -> FemMode:
    raw = root.omega
    clear_imaginary = (pencil.norms[1] == 0 or raw.imag < 0) and abs(raw.imag) <= component_zero
    omega = (0j if abs(raw) <= zero else
             complex(0.0 if abs(raw.real) <= zero else raw.real,
                     0.0 if clear_imaginary else raw.imag))
    quantities = modal_quantities(omega, zero_rad_s=component_zero)
    p = (np.ones(pencil.operators.basis.N, dtype=np.complex128) if omega == 0j
         else root.vector[:pencil.operators.basis.N].copy())
    residual, _ = _residual(pencil, omega / c, np.concatenate((p, omega / c * p)))
    shape = None
    if retain:
        p /= math.sqrt(float(np.vdot(p, pencil.operators.mass @ p).real))
        p.flags.writeable = False
        shape = p
    return FemMode(omega, raw, quantities.frequency_hz, quantities.t60_s, quantities.q,
                   quantities.kind, residual, root.linearized_residual, root.shift,
                   shifts[root.shift].frequency_hz, shape)


def solve_fem_modes(
    mesh: ShoeboxMesh, *, wall_impedances: WallImpedances, density_kg_m3: float,
    sound_speed_m_s: float, frequency_max_hz: float,
    shifts_hz: Sequence[float] | None = None, modes_per_shift: Sequence[int] | None = None,
    reference_frequencies_hz: Sequence[float] | None = None,
    options: ModalSolverOptions = ModalSolverOptions(), retain_shapes: bool = False,
    count_band_edges_hz: Sequence[float] | None = None,
    rigid_reference_frequencies_hz: Sequence[float] | None = None,
) -> FemModalSpectrum:
    """求網格的頻率無關實阻抗模態；不讀產品設定、不推測長寬高。

    給 reference_frequencies_hz 時，每移位個數完全採第零步規則
    ceil(1.3 × ±35 Hz 內參考個數)+12；不給則以網格的 Weyl 兩項估計個數。
    modes_per_shift 可直接覆寫。有限移位不宣稱任意強阻尼全譜完備。
    負頻率鏡像支不輸出；正頻率增長支由慣例函式拒絕，不以絕對值修正。
    check 是返回圓在整段的保證高度與原始個數，不判過關、不增算根。
    count_band_edges_hz 可改報數頻段，預設沿用移位間距；rigid_reference_frequencies_hz
    僅供剛性解析個數對照，與控制移位名額的 reference_frequencies_hz 分開。
    returned_above_limit_count 僅指此次找到的超界共振，不能推論由哪條起點被阻尼推出。
    非振盪來源須另由延拓證據標記，不能按衰減排序推測。
    """
    if any(not math.isfinite(value) or value <= 0 for value in
           (density_kg_m3, sound_speed_m_s, frequency_max_hz)):
        raise ValueError("密度、聲速及頻率上限必須為正有限數")
    # 自檢參數在求解前先驗：正式網格一次要幾分鐘，參數錯不該白算（複查）。
    band_edges = _band_edges(count_band_edges_hz, frequency_max_hz, options)
    if rigid_reference_frequencies_hz is not None and any(
            not math.isfinite(f) or f < 0 for f in rigid_reference_frequencies_hz):
        raise ValueError("剛性參考頻率必須非負有限")
    operators = assemble_p2_operators(mesh)
    pencil = _pencil(operators, wall_impedances, density_kg_m3, sound_speed_m_s)
    plan = _shift_plan(mesh, sound_speed_m_s, frequency_max_hz, pencil.a.shape[0], options,
                       shifts_hz, modes_per_shift, reference_frequencies_hz)
    runtime.preload_mkl()
    runtime.set_pardiso_threads(options.pardiso_threads)
    rng = np.random.default_rng(options.random_seed)
    v0 = np.asarray(rng.standard_normal(pencil.a.shape[0])
                    + 1j * rng.standard_normal(pencil.a.shape[0]), dtype=np.complex128)
    roots: list[_Root] = []
    shifts = []
    for index, (center, count) in enumerate(plan):
        found, record = _shift_roots(pencil, center, count, sound_speed_m_s, index, options, v0)
        roots.extend(found)
        shifts.append(record)
    zero, component_zero = _zero_scale(pencil, roots, sound_speed_m_s)
    if pencil.norms[1] and any(sum(abs(root.omega) <= zero for root in roots if root.shift == index) > 1
                              for index in range(len(shifts))):
        raise ArithmeticError(f"阻尼零根叢集不可分辨靜態與非振盪衰減；zero_rad_s={zero!r}")
    candidates = [root for root in roots if root.omega.real >= -zero
                  and root.omega.real <= 2 * math.pi * frequency_max_hz]
    kept = _deduplicate(candidates, zero, options.dedup_relative_tolerance)
    rows = tuple(sorted((_row(pencil, root, zero, component_zero, sound_speed_m_s, shifts, retain_shapes)
                         for root in kept), key=lambda row: (row.frequency_hz, row.omega.imag)))
    return _checked_spectrum(mesh, rows, shifts, zero, component_zero, int(operators.basis.N),
                             sound_speed_m_s, frequency_max_hz, roots, options,
                             band_edges, rigid_reference_frequencies_hz)


def _band_edges(edges: Sequence[float] | None, cap: float, options: ModalSolverOptions) -> tuple[float, ...]:
    """預設沿用移位間距，最後一帶截在上限；給了就驗：從零嚴格遞增、最後一個等於上限。"""
    band_edges = (tuple(float(edge) for edge in edges) if edges is not None else
                  (0.0, *(float(f) for f in np.arange(options.shift_spacing_hz, cap, options.shift_spacing_hz)), cap))
    if (len(band_edges) < 2 or band_edges[0] != 0 or band_edges[-1] != cap
            or any(not math.isfinite(x) for x in band_edges)
            or any(a >= b for a, b in zip(band_edges, band_edges[1:]))):
        raise ValueError("個數頻段界線必須從零嚴格遞增，最後一個等於頻率上限")
    return band_edges


def _checked_spectrum(mesh: ShoeboxMesh, rows: tuple[FemMode, ...], shifts: Sequence[ModalShift],
                      zero: float, component_zero: float, n: int, c: float, cap: float,
                      roots: Sequence[_Root], options: ModalSolverOptions,
                      band_edges: tuple[float, ...], rigid: Sequence[float] | None) -> FemModalSpectrum:
    """求解結束才作自檢；頻段界線已在求解前驗過。"""
    rows = tuple(replace(row, decay_origin=DecayOrigin.UNCONFIRMED
                         if row.kind is ModalKind.NONOSCILLATING_DECAY else None) for row in rows)
    bands = count_bands(rows, band_edges, tuple(_weyl_count(mesh, f, c) for f in band_edges), rigid)
    above = _deduplicate([root for root in roots if root.omega.real > 2 * math.pi * cap],
                         zero, options.dedup_relative_tolerance)
    check = build_check(rows, shifts, cap, bands, len(above))
    marked = tuple(replace(row, above_guaranteed_decay=row.omega.imag > check.guaranteed_decay_rate_rad_s)
                   for row in rows)
    return FemModalSpectrum(marked, tuple(shifts), zero, component_zero, n, check)
