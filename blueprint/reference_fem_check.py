"""票 #213 的獨立 FEM 解析檢查；只用 Python 標準庫。

解析模態級數保留第 1 段 ``seg1_modal_oracle.py`` 的收斂檢查：x／y 模態以完整
cosine-product 週期逐次加倍，z 軸 Neumann 級數用閉式精確加總。這支不 import donor、
NumPy、JAX 或新引擎；答案檔只提供參考房的幾何與聲速。

兩層契約常數錨在
``docs/decisions/precision-contract-fem-two-layers.md``：相容層 ``2^-12``、剛性牆
20 Hz 以下解析物理層 ``2^-10``。本段只入庫參考答案，相容層常數到第七段 v3 出來才會咬。
本徵頻距離定義為對每個非零解析本徵頻 ``f_n`` 算
``|f-f_n|/f_n``，再取最小者；不是先取 Hz 絕對距離最近者。
"""
from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from typing import Final


FEM_COMPAT_CONTRACT_REL: Final[float] = 2.0 ** -12
FEM_PHYSICS_CONTRACT_REL: Final[float] = 2.0 ** -10
FEM_PHYSICS_FMAX_HZ: Final[float] = 20.0

_CONVERGENCE_REL: Final[float] = 1.0e-6
_MODAL_CUTOFFS: Final[tuple[int, ...]] = (60, 120, 240, 480, 960)
_EIGENFREQUENCY_INDICES_PER_AXIS: Final[int] = 10


@dataclass(frozen=True)
class ModalInputs:
    """參考剛性長方房的解析解輸入，全部由答案檔餵入。"""

    room: tuple[float, float, float]
    source: tuple[float, float, float]
    receiver: tuple[float, float, float]
    sound_speed: float


@dataclass(frozen=True)
class ModalSolution:
    """一個頻點的解析壓力與原 oracle 收斂證據。"""

    pressure: complex
    converged: bool
    previous_n_max: int
    n_max: int
    relative_change: float


@dataclass(frozen=True)
class RelativeJudgment:
    """兩個複數的相對差與契約判決。"""

    relative_error: float
    within_contract: bool


def axis_modal_factors(
    length: float,
    source_coordinate: float,
    receiver_coordinate: float,
    n_max: int,
) -> list[tuple[float, float]]:
    """一軸的 Neumann cosine coupling 與波數平方。"""
    factors: list[tuple[float, float]] = []
    for index in range(n_max + 1):
        epsilon = 1.0 if index == 0 else 2.0
        coupling = (
            epsilon
            * math.cos(index * math.pi * source_coordinate / length)
            * math.cos(index * math.pi * receiver_coordinate / length)
        )
        if abs(coupling) < 1.0e-15:
            coupling = 0.0
        factors.append((coupling, (index * math.pi / length) ** 2))
    return factors


def _cosh_over_sinh(q_value: complex, argument: float, length: float) -> complex:
    """穩定計算實數或純虛數 q 的 ``cosh(q*a)/sinh(q*L)``。"""
    if abs(q_value.real) > 1.0e-14:
        return (
            cmath.exp(q_value * (argument - length))
            + cmath.exp(-q_value * (argument + length))
        ) / (1.0 - cmath.exp(-2.0 * q_value * length))
    return cmath.cosh(q_value * argument) / cmath.sinh(q_value * length)


def zero_mode_pressure(frequency_hz: float, inputs: ModalInputs) -> complex:
    """全域 ``(0,0,0)`` 模態對解析壓力的貢獻。"""
    volume = math.prod(inputs.room)
    wave_number_squared = (
        2.0 * math.pi * frequency_hz / inputs.sound_speed
    ) ** 2
    return complex(-4.0 * math.pi / (volume * wave_number_squared), 0.0)


def analytical_modal_pressure(
    frequency_hz: float,
    n_max: int,
    inputs: ModalInputs,
) -> complex:
    """VAL-1a 的三維剛性房模態 Green function，z 級數用閉式加總。"""
    lx, ly, lz = inputs.room
    x_factors = axis_modal_factors(lx, inputs.source[0], inputs.receiver[0], n_max)
    y_factors = axis_modal_factors(ly, inputs.source[1], inputs.receiver[1], n_max)
    wave_number_squared = (
        2.0 * math.pi * frequency_hz / inputs.sound_speed
    ) ** 2
    z_sum = inputs.source[2] + inputs.receiver[2]
    z_difference = abs(inputs.source[2] - inputs.receiver[2])
    volume = math.prod(inputs.room)
    terms: list[complex] = []
    for x_coupling, kx_squared in x_factors:
        if x_coupling == 0.0:
            continue
        for y_coupling, ky_squared in y_factors:
            xy_coupling = x_coupling * y_coupling / volume
            if xy_coupling == 0.0:
                continue
            q_value = cmath.sqrt(
                complex(kx_squared + ky_squared - wave_number_squared, 0.0)
            )
            inner_z = (
                lz
                * 0.5
                * (
                    _cosh_over_sinh(q_value, lz - z_difference, lz)
                    + _cosh_over_sinh(q_value, lz - z_sum, lz)
                )
                / q_value
            )
            terms.append(xy_coupling * inner_z)
    total = complex(
        math.fsum(term.real for term in terms),
        math.fsum(term.imag for term in terms),
    )
    return 4.0 * math.pi * total


def solve_modal_pressure(frequency_hz: float, inputs: ModalInputs) -> ModalSolution:
    """逐次加倍 x／y cutoff，回第一個通過原 oracle 判準的解析解。"""
    previous_n = _MODAL_CUTOFFS[0]
    previous = analytical_modal_pressure(frequency_hz, previous_n, inputs)
    relative_change = math.inf
    for n_max in _MODAL_CUTOFFS[1:]:
        current = analytical_modal_pressure(frequency_hz, n_max, inputs)
        relative_change = abs(current - previous) / max(abs(current), 1.0e-300)
        if relative_change < _CONVERGENCE_REL:
            return ModalSolution(
                pressure=current,
                converged=True,
                previous_n_max=previous_n,
                n_max=n_max,
                relative_change=relative_change,
            )
        previous_n, previous = n_max, current
    return ModalSolution(
        pressure=previous,
        converged=False,
        previous_n_max=_MODAL_CUTOFFS[-2],
        n_max=_MODAL_CUTOFFS[-1],
        relative_change=relative_change,
    )


def rigid_eigenfrequencies(
    room: tuple[float, float, float],
    sound_speed: float,
    n_each: int = _EIGENFREQUENCY_INDICES_PER_AXIS,
) -> tuple[float, ...]:
    """VAL-1a 的非零剛性房本徵頻，三軸 index 各取 ``range(n_each)``。"""
    lx, ly, lz = room
    values: list[float] = []
    for nx in range(n_each):
        for ny in range(n_each):
            for nz in range(n_each):
                if nx == ny == nz == 0:
                    continue
                values.append(
                    (sound_speed / 2.0)
                    * math.sqrt((nx / lx) ** 2 + (ny / ly) ** 2 + (nz / lz) ** 2)
                )
    return tuple(sorted(values))


def nearest_eigenfrequency_relative(
    frequency_hz: float,
    eigenfrequencies_hz: tuple[float, ...],
) -> tuple[float, float]:
    """依 ``min(|f-f_n|/f_n)`` 回最近本徵頻與該相對距離。"""
    if not eigenfrequencies_hz:
        raise ValueError("eigenfrequencies_hz 不可為空")
    relative_distance, nearest = min(
        (abs(frequency_hz - mode) / mode, mode) for mode in eigenfrequencies_hz
    )
    return nearest, relative_distance


def judge_relative(
    reference: complex,
    actual: complex,
    contract_rel: float,
) -> RelativeJudgment:
    """以 ``|actual-reference|/|reference|`` 判是否落在相對契約內。"""
    relative_error = abs(actual - reference) / max(abs(reference), 1.0e-300)
    return RelativeJudgment(
        relative_error=relative_error,
        within_contract=relative_error <= contract_rel,
    )
