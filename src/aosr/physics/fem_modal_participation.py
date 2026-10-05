"""頻率無關實阻抗的模態參與；方法驗證用，不接候選或評分。

直接解 D(k)p=b_s，其中 D=K+jkCt−k²M，b_s=4π q_s；q_s 是實 P2
基底在聲源的取值列之轉置。時間、k=ω/c、Ct 的符號沿用既有零件。
第一伴隨線性化 A=[[-K,0],[0,-M]]、B=[[jCt,-M],[-M,0]]，
z_n=[p_n;k_n p_n]。A−kB 的頂左 Schur 補為 −D(k)，所以
D(k)^−1=Σ p_n p_nᵀ / ((k−k_n) N_n)，
N_n=z_nᵀBz_n=p_nᵀ(jCt−2k_nM)p_n=p_nᵀD'(k_n)p_n。
接收點壓力為 Σ R_n/(k−k_n)，R_n=4π p_n(x_s)p_n(x_r)/N_n。
R_n 是 k 平面的殘數；ω 平面的殘數為 c R_n。載荷沒有額外 jωρ。
所有轉置不取共軛；既有 pᴴMp=1 只縮放形狀，不能取代 N_n。

實 K、M、Ct 使正頻根 k_n 的鏡像為 −k_n*，形狀 p_n*、範數 −N_n*。
故一個共振的完整項為 R_n/(k−k_n)−R_n*/(k+k_n*)；純虛衰減及
阻尼靜態根是自己的鏡像，只加一次。剛性靜態是二階 Jordan 根，
另用 −4π p_0(x_s)p_0(x_r)/(k² p_0ᵀMp_0)，不能除以零範數。

共振大小定義為完整雙支項在 f=Re ω_n/(2π) 的絕對值（沿用直接解
單位），每個聲源／接收點對各自以最大值作 0 dB。無阻尼極點為無限，
無限最大值的並列極點為 0 dB、有限項為 −inf；全零對的相對 dB 未定義。
靜態與純衰減留帳不排共振。有限截斷的和只是近似，不宣稱全譜完備。
重根若非 B 雙線性正交則明確拒絕；併根／缺陷根不冒充簡根展開。
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from aosr.geometry.shoebox import Point
from aosr.physics.fem_helmholtz import (
    POINT_SOURCE_STRENGTH, P2Operators, WallImpedances, assemble_helmholtz_system, point_source_load,
)
from aosr.physics.fem_modal import FemMode, FemModalSpectrum
from aosr.physics.modal_convention import ModalKind


ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class ModalParticipation:
    """某聲源→某接收點→某解；非共振的大小與 dB 為 None。

    residue_k 是 k 平面的 Laurent 一階殘數。剛性靜態的一階殘數為零，
    double_pole_k2 另存有號的二階係數，其壓力項為 double_pole_k2/k²。
    """

    source_index: int
    receiver_index: int
    mode_index: int
    mode: FemMode
    residue_k: complex
    resonance_magnitude: float | None
    relative_db: float | None
    double_pole_k2: complex | None = None


def _term(mode: FemMode, residue: complex, k: ComplexArray, c: float, rigid: bool,
          double_pole: complex | None = None) -> ComplexArray:
    if rigid and mode.kind is ModalKind.STATIC:
        if double_pole is None:
            raise ArithmeticError("剛性靜態根缺少二階極點係數")
        return double_pole / k**2
    pole = mode.omega / c
    if np.any(k == pole):
        raise ArithmeticError("實頻率落在無阻尼極點，壓力不定義")
    result = residue / (k - pole)
    if mode.kind is ModalKind.RESONANCE:
        result = result - residue.conjugate() / (k + pole.conjugate())
    return result


@dataclass(frozen=True)
class ModalPositionResult:
    """一組位置的查表結果與實測秒數；壓力陣列軸為聲源、接收點、頻率。"""

    sources: tuple[Point, ...]
    receivers: tuple[Point, ...]
    entries: tuple[ModalParticipation, ...]
    position_seconds: float
    sound_speed_m_s: float
    rigid: bool

    def pressures(self, frequencies_hz: Sequence[float] | NDArray[np.float64]) -> ComplexArray:
        """含負頻鏡像、靜態及純衰減的截斷模態和；不作任何頻域求解。"""
        frequencies = np.asarray(frequencies_hz, dtype=np.float64)
        if frequencies.ndim != 1 or not frequencies.size or np.any(~np.isfinite(frequencies)) or np.any(frequencies <= 0):
            raise ValueError("頻率必須是一維非空、正有限 Hz")
        k = np.asarray(2 * math.pi * frequencies / self.sound_speed_m_s, dtype=np.complex128)
        pressure = np.zeros((len(self.sources), len(self.receivers), len(k)), dtype=np.complex128)
        for row in self.entries:
            if row.mode.kind is ModalKind.RESONANCE and row.mode.omega.imag == 0 and np.any(frequencies == row.mode.frequency_hz):
                raise ArithmeticError("實頻率落在無阻尼極點，壓力不定義")
            pressure[row.source_index, row.receiver_index] += _term(row.mode, row.residue_k, k,
                self.sound_speed_m_s, self.rigid, row.double_pole_k2)
        return pressure


def _relative_db(value: float, strongest: float) -> float | None:
    if strongest == 0:
        return None
    if math.isinf(strongest):
        return 0.0 if math.isinf(value) else -math.inf
    return 20 * math.log10(value / strongest) if value else -math.inf


def _position_values(operators: P2Operators, modes: tuple[FemMode, ...], points: tuple[Point, ...]) -> ComplexArray:
    values = np.empty((len(points), len(modes)), dtype=np.complex128)
    for index, point in enumerate(points):
        if any(not math.isfinite(value) for value in point.as_tuple()):
            raise ValueError("位置座標必須為有限數")
        load = point_source_load(operators, point)
        selected = np.flatnonzero(load)
        weights = load[selected] / POINT_SOURCE_STRENGTH
        for column, mode in enumerate(modes):
            assert mode.shape is not None  # 建立展開時已核對。
            values[index, column] = weights @ mode.shape[selected]
    return values


@dataclass(frozen=True)
class ModalExpansion:
    """同一房的形狀與雙線性範數預先核對一次；換位置只做 P2 插值。"""

    spectrum: FemModalSpectrum
    operators: P2Operators
    norms: tuple[complex, ...]
    sound_speed_m_s: float
    rigid: bool

    def at_positions(self, sources: Sequence[Point], receivers: Sequence[Point]) -> ModalPositionResult:
        """一次查多喇叭、多座位；position_seconds 含插值、全部項與 dB 建表。"""
        started = perf_counter()
        source_points, receiver_points = tuple(sources), tuple(receivers)
        if not source_points or not receiver_points:
            raise ValueError("聲源與接收點清單都必須非空")
        modes = self.spectrum.solutions
        source_values = _position_values(self.operators, modes, source_points)
        receiver_values = _position_values(self.operators, modes, receiver_points)
        entries: list[ModalParticipation] = []
        for source_index, source_row in enumerate(source_values):
            for receiver_index, receiver_row in enumerate(receiver_values):
                residues = POINT_SOURCE_STRENGTH * source_row * receiver_row / np.asarray(self.norms)
                magnitudes = self._magnitudes(residues)
                strongest = max((value for value in magnitudes if value is not None), default=0.0)
                for i, mode in enumerate(modes):
                    magnitude = magnitudes[i]
                    db = None if magnitude is None else _relative_db(magnitude, strongest)
                    double_pole = -complex(residues[i]) if self.rigid and mode.kind is ModalKind.STATIC else None
                    residue = complex(residues[i]) if double_pole is None else 0j
                    entries.append(ModalParticipation(source_index, receiver_index, i, mode, residue, magnitude, db, double_pole))
        return ModalPositionResult(source_points, receiver_points, tuple(entries), perf_counter() - started,
                                   self.sound_speed_m_s, self.rigid)

    def _magnitudes(self, residues: ComplexArray) -> list[float | None]:
        magnitudes: list[float | None] = []
        for mode, residue in zip(self.spectrum.solutions, residues, strict=True):
            if mode.kind is not ModalKind.RESONANCE:
                magnitudes.append(None)
            elif mode.omega.imag == 0:
                magnitudes.append(math.inf if residue else 0.0)
            else:
                k = np.asarray([mode.omega.real / self.sound_speed_m_s], dtype=np.complex128)
                magnitudes.append(float(abs(_term(mode, complex(residue), k, self.sound_speed_m_s, self.rigid)[0])))
        return magnitudes


def _matrix_norm(matrix: csr_matrix[np.float64]) -> float:
    return float(abs(matrix).sum(axis=0).max())


def _check_shape(mode: FemMode, operators: P2Operators, damping: csr_matrix[np.float64], c: float,
                 norms: tuple[float, float, float]) -> tuple[ComplexArray, ComplexArray, ComplexArray]:
    p = mode.shape
    if p is None:
        raise ValueError("模態必須用 retain_shapes=True 求得")
    if p.shape != (operators.basis.N,) or not np.all(np.isfinite(p)) or np.linalg.norm(p) == 0:
        raise ValueError("模態形狀必須是同一 P2 空間的非零有限一維向量")
    k = mode.omega / c
    kp, cp, mp = operators.stiffness @ p, damping @ p, operators.mass @ p
    scale = (norms[0] + abs(k) * norms[1] + abs(k)**2 * norms[2]) * np.linalg.norm(p)
    residual = float(np.linalg.norm(kp + 1j * k * cp - k**2 * mp) / scale)
    # 比對已儲存的後向殘差；這是錯配算子的舍入界，不是 FEM 對真值精度契約。
    budget = 64 * np.finfo(float).eps + 8 * mode.residual
    if not math.isfinite(residual) or residual > budget:
        raise ValueError("形狀、阻抗、密度、聲速必須來自同一網格／算子與特徵問題")
    return np.asarray(p), np.asarray(cp), np.asarray(mp)


def _bilinear_norms(spectrum: FemModalSpectrum, operators: P2Operators, damping: csr_matrix[np.float64],
                    c: float, rigid: bool) -> tuple[complex, ...]:
    norms = (_matrix_norm(operators.stiffness), _matrix_norm(damping), _matrix_norm(operators.mass))
    results = []
    for mode in spectrum.solutions:
        p, cp, mp = _check_shape(mode, operators, damping, c, norms)
        derivative = mp if rigid and mode.kind is ModalKind.STATIC else 1j * cp - 2 * mode.omega / c * mp
        norm = complex(p @ derivative)
        rounding = 64 * np.finfo(float).eps * np.linalg.norm(p) * np.linalg.norm(derivative)
        if not math.isfinite(abs(norm)) or abs(norm) <= rounding:
            raise ArithmeticError("雙線性範數為零或不可分辨：缺陷根不能用簡根展開")
        results.append(norm)
    _check_repeated(spectrum, operators, damping, c)
    return tuple(results)


def _check_repeated(spectrum: FemModalSpectrum, operators: P2Operators, damping: csr_matrix[np.float64], c: float) -> None:
    for i, mode in enumerate(spectrum.solutions):
        for other in spectrum.solutions[i + 1:]:
            if abs(mode.omega - other.omega) > spectrum.component_zero_rad_s:
                continue
            assert mode.shape is not None and other.shape is not None
            derivative = 1j * (damping @ other.shape) - 2 * mode.omega / c * (operators.mass @ other.shape)
            cross = abs(mode.shape @ derivative)
            budget = (64 * np.finfo(float).eps + 8 * max(mode.residual, other.residual))
            if cross > budget * np.linalg.norm(mode.shape) * np.linalg.norm(derivative):
                raise ArithmeticError("重根形狀未作 B 雙線性正交，不能逐模態簡根展開")


def prepare_modal_expansion(spectrum: FemModalSpectrum, operators: P2Operators, *,
                            wall_impedances: WallImpedances, density_kg_m3: float,
                            sound_speed_m_s: float) -> ModalExpansion:
    """核對同一 P2 特徵問題並預算 N_n；Ct 從既有 Helmholtz 在 k=1 組裝。"""
    if any(not math.isfinite(value) or value <= 0 for value in (density_kg_m3, sound_speed_m_s)):
        raise ValueError("密度、聲速必須為正有限數")
    if not spectrum.solutions or spectrum.degrees_of_freedom != operators.basis.N:
        raise ValueError("模態必須非空且使用同一 P2 網格／算子")
    system = assemble_helmholtz_system(operators, frequency_hz=sound_speed_m_s / (2 * math.pi),
        wall_impedances=wall_impedances, density_kg_m3=density_kg_m3, sound_speed_m_s=sound_speed_m_s)
    damping = csr_matrix((system.data.imag.copy(), system.indices.copy(), system.indptr.copy()),
                         shape=system.shape, dtype=np.float64)
    rigid = not np.any(damping.data)
    norms = _bilinear_norms(spectrum, operators, damping, sound_speed_m_s, rigid)
    return ModalExpansion(spectrum, operators, norms, sound_speed_m_s, rigid)
