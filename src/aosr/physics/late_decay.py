"""鞋盒房間晚期衰減 T20、T30 的雙精度計算。

反射算子由 :mod:`aosr.physics.late_energy` 的共用建構器取得；本模組只負責
256 階衰減、精確特徵值尾巴、共用軟視窗擬合與上一代 T20 差距量測。
擬合無效直接報錯，不回傳 Perron 備援值；舊界線分類只作第二類相容紀錄。
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
    ART_WLS_T30_LO_DB,
    ART_WLS_WINDOW_SOFTNESS_DB,
)
from aosr.physics.late_energy import LateEnergyInputs, _reflection_problem


LATE_DECAY_T20_CONTRACT_REL: Final[float] = 2.0**-20
LATE_DECAY_SYNTHETIC_PROPERTY_REL: Final[float] = 2.0**-30
"""T20／T30 對已知斜率單一指數衰減的性質考卷界線。

12 組合成衰減（T60 0.1／0.3／1.0／3.0 秒 × f_e 50／128.625／400 Hz）實測最大相對差：T20 2.96e-16、
T30 2.78e-16，是浮點捨入等級。這份合成考卷只驗「擬合能回到已知 T60」，抓得到 f_e 與斜率換算寫錯
（f_e 差 0.1% 約造成 1e-3），**抓不到視窗寫錯**（純指數衰減對視窗不敏感，下緣改 −20～−45 最大差約
3.7e-16）；視窗錯由正式入口的接線考卷與常數考卷抓。

界線取 2^-30，刻意偏離「實測用掉兩到三成」的慣例：照慣例會落在 2^-49，貼著捨入，不同機器會時紅時綠。
第七段 FEniCS 凍結答案契約同樣取 2^-30（``docs/decisions/fem-contract-fenics-frozen-answers.md``，那張紙
訂界時兩套程式實測 flat 2.0e-14、lowabs 5.0e-14）。容差決定寫在
``docs/decisions/late-decay-t30-property-tolerance-2pow30.md``；老闆在票 #280 授權助理依量測訂。
"""
# Frozen donor art-kernel module lines 209-210. These are inherited validity
# constants, not newly selected v3 thresholds; the generator records the full source.
ART_WLS_MIN_WEIGHT: Final[float] = 1e-3
ART_WLS_LOG_FLOOR_DB: Final[float] = -400.0


@dataclass(frozen=True)
class LateDecayBand:
    """單一頻帶的 T20、選配 T30 與可追查擬合中介量。"""

    frequency_hz: float
    t20_s: float
    collision_frequency_hz: float
    slope_db_per_s: float
    soft_weight_sum: float
    fell_back_to_perron: bool
    perron_t60_s: float
    t30_s: float | None = None
    t30_slope_db_per_s: float | None = None
    t30_soft_weight_sum: float | None = None


@dataclass(frozen=True)
class LateDecayResult:
    """同一組材料所有頻帶的晚期衰減結果。"""

    orders_used: int
    bands: tuple[LateDecayBand, ...]


@dataclass(frozen=True)
class LateDecayBandJudgment:
    """單一頻帶相對凍結上一代 T20 的第二類相容紀錄。"""

    frequency_hz: float
    actual_t20_s: float
    expected_t20_s: float
    relative_difference: float
    contract_fraction: float
    within_contract: bool


@dataclass(frozen=True)
class LateDecayContractReport:
    """一組材料逐頻帶的上一代 T20 差距與舊界線分類。"""

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
    t60_s: NDArray[np.float64]


class DecayRangeError(ValueError):
    """第 256 階衰減仍未到指定擬合下緣。

    報表只捕捉這個型別；其他輸入、矩陣或擬合錯誤仍照常往外丟。
    ``minimums_by_frequency`` 保留每個細軸點的第 256 階最低 dB，讓報表能為
    T20 與 T30 各自產生可追查原因。
    """

    def __init__(
        self,
        *,
        lower_db: float,
        minimums_by_frequency: tuple[tuple[float, float], ...],
    ) -> None:
        self.lower_db = lower_db
        self.minimums_by_frequency = minimums_by_frequency
        super().__init__(self.reason_for(lower_db))

    def reason_for(self, lower_db: float) -> str:
        """用同一份第 256 階量測說明指定視窗為何不可算。"""
        missing = tuple(
            (frequency, minimum)
            for frequency, minimum in self.minimums_by_frequency
            if minimum > lower_db
        )
        if not missing:
            raise ValueError("這份第 256 階量測已到指定擬合下緣")
        fit_name = "T30" if lower_db == ART_WLS_T30_LO_DB else "T20"
        minimum = min(value for _frequency, value in missing)
        frequencies = ", ".join(f"{frequency:g} Hz" for frequency, _value in missing)
        return (
            f"{fit_name} 擬合無效：第 256 階最低 {minimum:.17g} dB，"
            f"未達下緣 {lower_db:g} dB；未達頻點 {frequencies}"
        )


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


def _fit_decay(
    level_db: NDArray[np.float64],
    collision_frequency_hz: float,
    *,
    lower_db: float,
) -> _FitArrays:
    """以共用的 −5 dB 上緣和指定下緣作軟視窗加權直線擬合。"""
    order = np.arange(1, ART_NEUMANN_K_MAX + 1, dtype=np.float64)
    time = (order / collision_frequency_hz)[:, None]
    upper = (ART_WLS_T20_HI_DB - level_db) / ART_WLS_WINDOW_SOFTNESS_DB
    lower = (level_db - lower_db) / ART_WLS_WINDOW_SOFTNESS_DB
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
        fit_name = "T30" if lower_db == ART_WLS_T30_LO_DB else "T20"
        raise ValueError(f"{fit_name} 擬合無效：頻帶索引 {bands} 的 (權重和, 斜率)={measured}")
    return _FitArrays(weight_sum, slope, np.asarray(-60.0 / slope, dtype=np.float64))


def _require_decay_reaches_lower_bound(
    level_db: NDArray[np.float64],
    frequencies_hz: tuple[float, ...],
    *,
    lower_db: float,
) -> None:
    """曲線含精確尾巴仍未到擬合下緣時，逐頻帶直接報錯。"""
    minimums = np.min(level_db, axis=0)
    minimums_by_frequency = tuple(
        (frequency, float(minimum))
        for frequency, minimum in zip(frequencies_hz, minimums, strict=True)
    )
    if not any(minimum > lower_db for _frequency, minimum in minimums_by_frequency):
        return
    raise DecayRangeError(
        lower_db=lower_db,
        minimums_by_frequency=minimums_by_frequency,
    )


def _perron_t60(roots: NDArray[np.float64], collision_frequency_hz: float) -> NDArray[np.float64]:
    valid = (roots > 0.0) & (roots < 1.0)
    safe_roots = np.where(valid, roots, 0.5)
    raw = np.log(1e-6) / np.log(safe_roots) / collision_frequency_hz
    return np.asarray(np.where(valid, raw, 0.0), dtype=np.float64)


def _solve_late_decay(
    inputs: LateEnergyInputs,
    *,
    sound_speed_m_s: float,
    include_t30: bool,
) -> LateDecayResult:
    problem = _reflection_problem(inputs)
    collision_frequency = _collision_frequency(inputs, sound_speed_m_s)
    roots = _exact_roots(problem.transfer)
    order_energy = _order_decay(problem.transfer, problem.patches.areas)
    level_db = _decay_level(order_energy, roots)
    _require_decay_reaches_lower_bound(
        level_db,
        inputs.frequencies_hz,
        lower_db=ART_WLS_T20_LO_DB,
    )
    t20_fit = _fit_decay(
        level_db,
        collision_frequency,
        lower_db=ART_WLS_T20_LO_DB,
    )
    if include_t30:
        _require_decay_reaches_lower_bound(
            level_db,
            inputs.frequencies_hz,
            lower_db=ART_WLS_T30_LO_DB,
        )
        t30_fit = _fit_decay(
            level_db,
            collision_frequency,
            lower_db=ART_WLS_T30_LO_DB,
        )
    else:
        t30_fit = None
    perron = _perron_t60(roots, collision_frequency)
    bands = tuple(
        LateDecayBand(
            frequency_hz=frequency,
            t20_s=float(t20_fit.t60_s[index]),
            collision_frequency_hz=collision_frequency,
            slope_db_per_s=float(t20_fit.slope[index]),
            soft_weight_sum=float(t20_fit.weight_sum[index]),
            fell_back_to_perron=False,
            perron_t60_s=float(perron[index]),
            t30_s=float(t30_fit.t60_s[index]) if t30_fit is not None else None,
            t30_slope_db_per_s=(
                float(t30_fit.slope[index]) if t30_fit is not None else None
            ),
            t30_soft_weight_sum=(
                float(t30_fit.weight_sum[index]) if t30_fit is not None else None
            ),
        )
        for index, frequency in enumerate(inputs.frequencies_hz)
    )
    return LateDecayResult(orders_used=ART_NEUMANN_K_MAX, bands=bands)


def solve_late_decay(
    inputs: LateEnergyInputs,
    *,
    sound_speed_m_s: float,
) -> LateDecayResult:
    """由同一條 256 階衰減曲線回傳 T20 與 T30。"""
    return _solve_late_decay(inputs, sound_speed_m_s=sound_speed_m_s, include_t30=True)


def solve_late_decay_t20(
    inputs: LateEnergyInputs,
    *,
    sound_speed_m_s: float,
) -> LateDecayResult:
    """依凍結定義只計算 T20，維持原有無效判斷與例外行為。"""
    return _solve_late_decay(inputs, sound_speed_m_s=sound_speed_m_s, include_t30=False)


def judge_late_decay_t20(
    result: LateDecayResult,
    expected_t20_s: Sequence[float],
) -> LateDecayContractReport:
    """逐頻量 T20 相對差，並保留舊 ``2^-20`` 分類供閱讀。"""
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
