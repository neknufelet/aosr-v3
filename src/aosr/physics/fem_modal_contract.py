"""模態頻譜先核一對一身分，再逐對判實部頻率與虛部衰減。

門檻由呼叫端傳入；配對沿第六步的複數相對距離規則，不拿配對
距離當精度判決。零分量核對兩邊原始根，沿受驗頻譜已有的絕對
雜訊尺度；靜態根另核模長，不用分母下限冒充相對誤差。
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import linear_sum_assignment

from aosr.physics.modal_convention import ModalKind

if TYPE_CHECKING:
    from aosr.physics.fem_modal import FemModalSpectrum


@dataclass(frozen=True)
class ModalReference:
    """外部或半解析答案；原始根與分類後的根都用 rad/s。"""

    omega: complex
    raw_omega: complex
    kind: ModalKind


@dataclass(frozen=True)
class ModalComponentJudgment:
    """非零用答案分量當分母；零值的 relative_error 為 None。"""

    relative_error: float | None
    absolute_error_rad_s: float
    zero_noise_rad_s: float | None
    contract_fraction: float
    within_contract: bool


@dataclass(frozen=True)
class ModalPairJudgment:
    """保留兩邊列號與種類，讓精度植錯能證明配對身分未變。"""

    actual_index: int
    expected_index: int
    kind: ModalKind
    actual_omega: complex
    expected_omega: complex
    frequency: ModalComponentJudgment
    decay: ModalComponentJudgment
    static_zero_fraction: float | None

    @property
    def within_contract(self) -> bool:
        """靜態根亦須在模長零界內；歸零後的值不能遮掉原始雜訊。"""
        return (self.frequency.within_contract and self.decay.within_contract
                and (self.static_zero_fraction is None or self.static_zero_fraction <= 1))


@dataclass(frozen=True)
class ModalContractReport:
    """結構失敗即不在契約內，沒有配對不回傳一份空的綠判決。"""

    points: tuple[ModalPairJudgment, ...]
    structural_errors: tuple[str, ...]
    frequency_rel: float
    decay_rel: float

    @property
    def within_contract(self) -> bool:
        return bool(self.points) and not self.structural_errors and all(p.within_contract for p in self.points)

    @property
    def max_frequency_fraction(self) -> float:
        """只記共振的相對頻率契約用量，絕對零界比例留在逐對結構。"""
        return max((p.frequency.contract_fraction for p in self.points
                    if p.kind is ModalKind.RESONANCE and p.frequency.relative_error is not None),
                   default=0.0 if self.points else math.inf)

    @property
    def max_decay_fraction(self) -> float:
        """只記共振的非零衰減相對契約用量，剛性零衰減不算 0/0。"""
        return max((p.decay.contract_fraction for p in self.points
                    if p.kind is ModalKind.RESONANCE and p.decay.relative_error is not None),
                   default=0.0 if self.points else math.inf)

    @property
    def max_frequency_relative_error(self) -> float | None:
        return max((p.frequency.relative_error for p in self.points
                    if p.kind is ModalKind.RESONANCE and p.frequency.relative_error is not None), default=None)

    @property
    def max_decay_relative_error(self) -> float | None:
        return max((p.decay.relative_error for p in self.points
                    if p.kind is ModalKind.RESONANCE and p.decay.relative_error is not None), default=None)


def _fraction(error: float, bound: float) -> float:
    """絕對零界也可為零：只容許精確零，不生造分母。"""
    return error / bound if bound else (0.0 if error == 0 else math.inf)


def _component(actual: float, expected: float, raw_actual: float, raw_expected: float,
               relative: float, zero: float) -> ModalComponentJudgment:
    error = abs(actual - expected)
    if expected == 0:
        noise = max(abs(actual), abs(raw_actual), abs(raw_expected))
        fraction = _fraction(noise, zero)
        return ModalComponentJudgment(None, error, noise, fraction, noise <= zero)
    relative_error = error / abs(expected)
    fraction = relative_error / relative
    return ModalComponentJudgment(relative_error, error, None, fraction, relative_error <= relative)


def _pairs(spectrum: FemModalSpectrum, expected: Sequence[ModalReference]) -> tuple[tuple[tuple[int, int], ...], tuple[str, ...]]:
    if not spectrum.solutions or not expected:
        return (), ("頻譜或答案為空",)
    if len(spectrum.solutions) != len(expected):
        return (), ("兩邊解數不同",)
    actual = np.asarray([row.omega for row in spectrum.solutions])
    reference = np.asarray([row.omega for row in expected])
    # 這個尺度只供結構配對，沿第六步規則；不進逐分量的相對誤差分母。
    scale = np.maximum(np.maximum(abs(actual[:, None]), abs(reference[None, :])), 1.0)
    distances = abs(actual[:, None] - reference[None, :]) / scale
    rows, columns = linear_sum_assignment(distances)
    errors = []
    if (not np.array_equal(columns, np.argmin(distances[rows], axis=1))
            or not np.array_equal(rows, np.argmin(distances[:, columns], axis=0))):
        errors.append("雙向最近鄰與最佳指派認到不同身分")
    pairs = tuple((int(i), int(j)) for i, j in zip(rows, columns, strict=True))
    if any(spectrum.solutions[i].kind is not expected[j].kind for i, j in pairs):
        errors.append("配對種類不同")
    return pairs, tuple(errors)


def _point(spectrum: FemModalSpectrum, reference: ModalReference,
           i: int, j: int, frequency_rel: float, decay_rel: float) -> ModalPairJudgment:
    actual = spectrum.solutions[i]
    static = reference.kind is ModalKind.STATIC
    zero = spectrum.zero_rad_s if static else spectrum.component_zero_rad_s
    frequency = _component(actual.omega.real, reference.omega.real,
                           actual.raw_omega.real, reference.raw_omega.real, frequency_rel, zero)
    decay = _component(actual.omega.imag, reference.omega.imag,
                       actual.raw_omega.imag, reference.raw_omega.imag, decay_rel, zero)
    static_fraction = (_fraction(max(abs(actual.raw_omega), abs(reference.raw_omega), abs(actual.omega)), zero)
                       if static else None)
    return ModalPairJudgment(i, j, reference.kind, actual.omega, reference.omega,
                             frequency, decay, static_fraction)


def judge_modal_spectrum(spectrum: FemModalSpectrum, expected: Sequence[ModalReference], *,
                         frequency_rel: float, decay_rel: float) -> ModalContractReport:
    """沿原始零值規則核對兩邊；同網格答案與受驗頻譜共用此跑的絕對雜訊界。

    靜態用 zero_rad_s（模長界），其他零分量用 component_zero_rad_s；
    非零分量永遠直接除以答案自身的絕對值。數值與種類不一致的外部
    資料應由呼叫端按唯一慣例載入，這裡只判配對及精度，不重新分類。
    """
    if any(not math.isfinite(value) or value <= 0 for value in (frequency_rel, decay_rel)):
        raise ValueError("兩個相對門檻必須為正有限數")
    if any(not math.isfinite(value) or value < 0 for value in (spectrum.zero_rad_s, spectrum.component_zero_rad_s)):
        raise ValueError("既有絕對零界必須為非負有限數")
    roots = [value for row in (*spectrum.solutions, *expected) for value in (row.omega, row.raw_omega)]
    if any(not math.isfinite(value.real) or not math.isfinite(value.imag) for value in roots):
        raise ValueError("分類根與原始根必須有限")
    pairs, errors = _pairs(spectrum, expected)
    points = tuple(_point(spectrum, expected[j], i, j, frequency_rel, decay_rel) for i, j in pairs)
    return ModalContractReport(points, errors, frequency_rel, decay_rel)
