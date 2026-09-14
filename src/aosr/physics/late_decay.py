"""鞋盒房間晚期衰減 T20 的雙精度計算。

反射算子由 :mod:`aosr.physics.late_energy` 的共用建構器取得；本模組只負責
256 階衰減、精確特徵值尾巴、−5～−25 dB 軟視窗擬合與 2^-20 契約裁判。
擬合無效直接報錯，不回傳 Perron 備援值。本刀不實作 T30。
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from aosr.config.art_lane import (
    ART_NEUMANN_K_MAX,
    ART_WLS_T20_HI_DB,
    ART_WLS_T20_LO_DB,
    ART_WLS_WINDOW_SOFTNESS_DB,
)
from aosr.physics.late_energy import LateEnergyInputs, _reflection_problem


LATE_DECAY_T20_CONTRACT_REL: Final[float] = 2.0**-20
# Frozen donor art-kernel module lines 209-210. These are inherited validity
# constants, not newly selected v3 thresholds; the generator records the full source.
ART_WLS_MIN_WEIGHT: Final[float] = 1e-3
ART_WLS_LOG_FLOOR_DB: Final[float] = -400.0


@dataclass(frozen=True)
class LateDecayBand:
    """單一頻帶的 T20 與可追查擬合中介量。"""

    frequency_hz: float
    t20_s: float
    collision_frequency_hz: float
    slope_db_per_s: float
    soft_weight_sum: float
    fell_back_to_perron: bool
    perron_t60_s: float


@dataclass(frozen=True)
class LateDecayResult:
    """同一組材料所有頻帶的晚期衰減結果。"""

    orders_used: int
    bands: tuple[LateDecayBand, ...]


@dataclass(frozen=True)
class LateDecayBandJudgment:
    """單一頻帶相對凍結上一代 T20 的契約判決。"""

    frequency_hz: float
    actual_t20_s: float
    expected_t20_s: float
    relative_difference: float
    contract_fraction: float
    within_contract: bool


@dataclass(frozen=True)
class LateDecayContractReport:
    """一組材料逐頻帶的 T20 契約判決。"""

    points: tuple[LateDecayBandJudgment, ...]

    @property
    def max_relative_difference(self) -> float:
        return max(point.relative_difference for point in self.points)

    @property
    def max_contract_fraction(self) -> float:
        return max(point.contract_fraction for point in self.points)

    @property
    def within_contract(self) -> bool:
        return all(point.within_contract for point in self.points)


@dataclass(frozen=True)
class _FitArrays:
    """逐頻帶擬合結果。"""

    weight_sum: NDArray[np.float64]
    slope: NDArray[np.float64]
    t20: NDArray[np.float64]


def _sigmoid(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """避免指數溢位的雙精度 sigmoid。"""
    positive = values >= 0.0
    negative_exp = np.exp(np.where(positive, -values, values))
    return np.asarray(
        np.where(positive, 1.0 / (1.0 + negative_exp), negative_exp / (1.0 + negative_exp)),
        dtype=np.float64,
    )


def _collision_frequency(inputs: LateEnergyInputs, sound_speed_m_s: float) -> float:
    if not math.isfinite(sound_speed_m_s) or sound_speed_m_s <= 0.0:
        raise ValueError("sound_speed_m_s 必須是有限正數")
    room = inputs.room
    surface_area = 2.0 * (room.Lx * room.Ly + room.Ly * room.Lz + room.Lx * room.Lz)
    volume = room.Lx * room.Ly * room.Lz
    return sound_speed_m_s * surface_area / (4.0 * volume)


def _exact_roots(transfer: NDArray[np.float64]) -> NDArray[np.float64]:
    """逐頻用 ``numpy.linalg.eigvals`` 取最大模特徵值。"""
    roots = np.asarray(
        [
            np.max(np.abs(np.linalg.eigvals(transfer[:, :, index])))
            for index in range(transfer.shape[2])
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(roots)):
        raise ValueError("反射算子的精確特徵值含非有限值")
    return roots


def _order_decay(
    transfer: NDArray[np.float64],
    areas: NDArray[np.float64],
) -> NDArray[np.float64]:
    patch_count, _other_patch_count, frequency_count = transfer.shape
    area_weight = areas / np.sum(areas)
    energy = np.ones((patch_count, frequency_count), dtype=np.float64)
    orders = np.empty((ART_NEUMANN_K_MAX, frequency_count), dtype=np.float64)
    for index in range(ART_NEUMANN_K_MAX):
        energy = np.einsum("ijf,jf->if", transfer, energy)
        orders[index] = np.einsum("p,pf->f", area_weight, energy)
    return orders


def _decay_level(
    order_energy: NDArray[np.float64],
    roots: NDArray[np.float64],
) -> NDArray[np.float64]:
    valid_root = (roots > 0.0) & (roots < 1.0)
    safe_root = np.where(valid_root, roots, 0.5)
    tail = order_energy[-1] * safe_root / (1.0 - safe_root)
    tail = np.where(valid_root, tail, 0.0)
    edc = np.cumsum(order_energy[::-1], axis=0)[::-1] + tail[None, :]
    initial = edc[0]
    ratio = np.where(initial > 0.0, edc / np.where(initial > 0.0, initial, 1.0), 1.0)
    tiny = np.finfo(np.float64).tiny
    safe_ratio = np.where(ratio > tiny, ratio, 1.0)
    return np.asarray(
        np.where(ratio > tiny, 10.0 * np.log10(safe_ratio), ART_WLS_LOG_FLOOR_DB),
        dtype=np.float64,
    )


def _fit_t20(level_db: NDArray[np.float64], collision_frequency_hz: float) -> _FitArrays:
    order = np.arange(1, ART_NEUMANN_K_MAX + 1, dtype=np.float64)
    time = (order / collision_frequency_hz)[:, None]
    upper = (ART_WLS_T20_HI_DB - level_db) / ART_WLS_WINDOW_SOFTNESS_DB
    lower = (level_db - ART_WLS_T20_LO_DB) / ART_WLS_WINDOW_SOFTNESS_DB
    weights = _sigmoid(upper) * _sigmoid(lower)
    weight_sum = np.sum(weights, axis=0)
    safe_weight = np.where(weight_sum > ART_WLS_MIN_WEIGHT, weight_sum, 1.0)
    time_mean = np.sum(weights * time, axis=0) / safe_weight
    level_mean = np.sum(weights * level_db, axis=0) / safe_weight
    covariance = np.sum(
        weights * (time - time_mean[None, :]) * (level_db - level_mean[None, :]),
        axis=0,
    )
    variance = np.sum(weights * (time - time_mean[None, :]) ** 2, axis=0)
    slope = np.where(variance > 0.0, covariance / np.where(variance > 0.0, variance, 1.0), 0.0)
    invalid = (weight_sum <= ART_WLS_MIN_WEIGHT) | (slope >= 0.0)
    if np.any(invalid):
        bands = np.flatnonzero(invalid).tolist()
        measured = [(float(weight_sum[i]), float(slope[i])) for i in bands]
        raise ValueError(f"T20 擬合無效：頻帶索引 {bands} 的 (權重和, 斜率)={measured}")
    return _FitArrays(weight_sum, slope, np.asarray(-60.0 / slope, dtype=np.float64))


def _perron_t60(roots: NDArray[np.float64], collision_frequency_hz: float) -> NDArray[np.float64]:
    valid = (roots > 0.0) & (roots < 1.0)
    safe_roots = np.where(valid, roots, 0.5)
    raw = np.log(1e-6) / np.log(safe_roots) / collision_frequency_hz
    return np.asarray(np.where(valid, raw, 0.0), dtype=np.float64)


def solve_late_decay_t20(
    inputs: LateEnergyInputs,
    *,
    sound_speed_m_s: float,
) -> LateDecayResult:
    """依凍結定義計算 256 階雙精度 T20；無效擬合直接報錯。"""
    problem = _reflection_problem(inputs)
    collision_frequency = _collision_frequency(inputs, sound_speed_m_s)
    roots = _exact_roots(problem.transfer)
    order_energy = _order_decay(problem.transfer, problem.patches.areas)
    fit = _fit_t20(_decay_level(order_energy, roots), collision_frequency)
    perron = _perron_t60(roots, collision_frequency)
    bands = tuple(
        LateDecayBand(
            frequency_hz=frequency,
            t20_s=float(fit.t20[index]),
            collision_frequency_hz=collision_frequency,
            slope_db_per_s=float(fit.slope[index]),
            soft_weight_sum=float(fit.weight_sum[index]),
            fell_back_to_perron=False,
            perron_t60_s=float(perron[index]),
        )
        for index, frequency in enumerate(inputs.frequencies_hz)
    )
    return LateDecayResult(orders_used=ART_NEUMANN_K_MAX, bands=bands)


def judge_late_decay_t20(
    result: LateDecayResult,
    expected_t20_s: Sequence[float],
) -> LateDecayContractReport:
    """逐頻套用 ``|T20_v3-T20_v2|/|T20_v2| <= 2^-20``。"""
    if not result.bands or len(result.bands) != len(expected_t20_s):
        raise ValueError("v3 結果與上一代 T20 答案的頻帶數不同或為空")
    points = []
    for band, expected_value in zip(result.bands, expected_t20_s, strict=True):
        expected = float(expected_value)
        difference = abs(band.t20_s - expected)
        relative = difference / abs(expected) if expected != 0.0 else math.inf
        fraction = relative / LATE_DECAY_T20_CONTRACT_REL
        finite = math.isfinite(band.t20_s) and math.isfinite(expected)
        points.append(
            LateDecayBandJudgment(
                frequency_hz=band.frequency_hz,
                actual_t20_s=band.t20_s,
                expected_t20_s=expected,
                relative_difference=relative,
                contract_fraction=fraction,
                within_contract=finite
                and expected > 0.0
                and relative <= LATE_DECAY_T20_CONTRACT_REL,
            )
        )
    return LateDecayContractReport(points=tuple(points))
