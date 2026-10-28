"""第一階段低頻目標的 JAX 版：主位扣自己的直線後的起伏，加上六點座位穩健性。

量法與合分照 ``stage_one_low_frequency``、``stage_one_score``；峰谷偵測留給候選確認，
不進可微目標。與材料無關的常數由 numpy 雙精度算一次，頻率權重借 ``timbre`` 唯一那一份。
求值只用 ``jax.numpy``，可編譯、微分與批次求值；常數在運算前轉成輸入能量的精度，
不改全域設定。完整正能量與有限複數聲壓由呼叫端保證，這裡只守形狀與型別。
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import DTypeLike, NDArray

from aosr.scoring import timbre
from aosr.scoring.receiver_set import ReceiverRole, ReceiverSet
from aosr.scoring.stage_one_low_frequency_contract import STAGE_ONE_SURROUNDING_DIRECTIONS
from aosr.scoring.stage_one_score import PRINCIPAL, ROBUSTNESS, StageOneScoreSettings

# 均方根零判的地板倍數：地板＝倍數 × 輸入精度的 eps × 該喇叭所有座位、範圍內頻點 |dB| 的最大值
# （下限 1 dB，能量約 1 時 dB 約 0 也要有地板）。座位穩健性在六座位、F=188 時約 √(6·188)≈34 個
# dB 捨入項疊加，取上面最近的 2 的冪——啟發式的工程餘裕，不是誤差上界。由 dtype 推出，不是登記的數，也不從契約的尺推：
# 編譯後相同曲線相減剩捨入殘差，在近零的均方根上除以 sqrt 會放大成 O(1) 的假梯度。
_RMS_FLOOR_EPS_MULTIPLE = 64.0


class StageOneObjectiveConstants(NamedTuple):
    """可直接交進編譯的樹狀資料；頻點與聲道索引的長度保留輸入形狀，整數不當動態尺寸。

    ``log2_frequencies`` 是整軸；``frequency_indices`` 與 ``frequency_weights`` 只含計分範圍內。
    周圍點依接收點清單的原序取出，與 numpy 輸入收集 ``speaker.surrounding`` 的順序一致。
    兩項設定的順序固定為主項、座位穩健性，以具名查詢取得，不依登記簿列序。
    """

    log2_frequencies: NDArray[np.float64]
    frequency_indices: NDArray[np.int64]
    frequency_weights: NDArray[np.float64]
    primary_index: NDArray[np.int64]
    surrounding_indices: NDArray[np.int64]
    speaker_indices: NDArray[np.int64]
    component_weights: NDArray[np.float64]
    worse_references: NDArray[np.float64]


class StageOneObjectiveArrays(NamedTuple):
    """前兩項形狀為 ``(喇叭,)``，兩項聲道平均與合分為純量；都是與輸入同精度的陣列。"""

    own_line_ripple_rms_db: jax.Array
    seat_robustness_rms_db: jax.Array
    mean_own_line_ripple_rms_db: jax.Array
    mean_seat_robustness_rms_db: jax.Array
    screening_total: jax.Array


def objective_constants(
    frequencies_hz: Sequence[float] | NDArray[np.float64],
    settings: StageOneScoreSettings,
    receiver_set: ReceiverSet,
    speaker_count: int,
) -> StageOneObjectiveConstants:
    """雙精度準備常數，不求解聲壓；頻率有限、正值、嚴格遞增，範圍內至少兩點才可擬合直線。"""
    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    if (frequencies.ndim != 1 or not np.all(np.isfinite(frequencies)) or
            not np.all(frequencies > 0.0) or not np.all(np.diff(frequencies) > 0.0)):
        raise ValueError("頻率軸必須是一維、有限、正值且嚴格遞增")
    inside = (frequencies >= settings.scored_range_hz[0]) & (frequencies <= settings.scored_range_hz[1])
    if np.count_nonzero(inside) < 2:
        raise ValueError("計分範圍內至少要有兩個頻點")
    if isinstance(speaker_count, bool) or not isinstance(speaker_count, int) or speaker_count < 1:
        raise ValueError("喇叭數必須是正整數")
    surrounding = [index for index, point in enumerate(receiver_set.points) if point.role == ReceiverRole.SURROUNDING]
    directions = [receiver_set.points[index].direction_relative_to_primary for index in surrounding]
    if (len(receiver_set.points) != 1 + len(STAGE_ONE_SURROUNDING_DIRECTIONS) or
            len(surrounding) != len(STAGE_ONE_SURROUNDING_DIRECTIONS) or
            set(directions) != STAGE_ONE_SURROUNDING_DIRECTIONS):
        raise ValueError("座位必須是一個主位加前後左右上下六個周圍點")
    primary = next(index for index, point in enumerate(receiver_set.points) if point.role == ReceiverRole.PRIMARY)
    octaves = np.log2(frequencies)
    return StageOneObjectiveConstants(
        log2_frequencies=octaves,
        frequency_indices=np.asarray(np.flatnonzero(inside), dtype=np.int64),
        frequency_weights=timbre._weights_in_range(octaves, inside, settings.scored_range_hz),
        primary_index=np.asarray(primary, dtype=np.int64),
        surrounding_indices=np.asarray(surrounding, dtype=np.int64),
        speaker_indices=np.arange(speaker_count, dtype=np.int64),
        component_weights=np.asarray([settings.weight(name) for name in (PRINCIPAL, ROBUSTNESS)], dtype=np.float64),
        worse_references=np.asarray([settings.target(name).worse_reference for name in (PRINCIPAL, ROBUSTNESS)],
                                    dtype=np.float64),
    )


def _energy_shape(constants: StageOneObjectiveConstants) -> tuple[int, int, int]:
    return (constants.speaker_indices.shape[0], 1 + constants.surrounding_indices.shape[0],
            constants.log2_frequencies.shape[0])


def rms_floor_factor(dtype: DTypeLike) -> float:
    """均方根零判地板的倍率：地板＝倍率 × max(該喇叭所有座位、範圍內頻點 |dB| 的最大值, 1)，單位 dB。

    倍率是 ``_RMS_FLOOR_EPS_MULTIPLE`` × ``dtype`` 的 eps；:func:`objective_from_energies` 與要推分數差上界的
    呼叫端（梯度搜尋終點）共用這一份，不各自再寫一次。
    """
    return _RMS_FLOOR_EPS_MULTIPLE * float(np.finfo(dtype).eps)


def _weighted_rms(values: jax.Array, weights: jax.Array, floor: jax.Array) -> jax.Array:
    """沿最後一軸的加權均方根；不到地板的值與梯度一起歸零，未走的 sqrt 不污染反向傳播。"""
    square = jnp.sum(weights * values * values, axis=-1) / jnp.sum(weights)
    is_zero = square <= floor * floor
    safe_square = jnp.where(is_zero, jnp.ones_like(square), square)
    return jnp.where(is_zero, jnp.zeros_like(square), jnp.sqrt(safe_square))


def objective_from_energies(
    constants: StageOneObjectiveConstants, energies: jax.Array,
) -> StageOneObjectiveArrays:
    """輸入 ``(喇叭, 座位, F)`` 浮點正能量，回逐喇叭兩項、逐聲道平均與第一階段合分。

    主位用 dB 對 log2(f) 的加權平均、協方差／變異數擬合自己的直線；六點各自減主位後
    取加權均方根再等權平均。合分為 Σ 權重 × 聲道平均／較差參考，外層類權重不在這裡乘。
    兩項的均方根低於 ``_RMS_FLOOR_EPS_MULTIPLE`` × eps × 該喇叭 max|dB|（下限 1）時，值與梯度都當 0。
    """
    values = jnp.asarray(energies)
    if not jnp.issubdtype(values.dtype, jnp.floating):
        raise TypeError(f"能量必須是浮點陣列，收到 {values.dtype}")
    expected = _energy_shape(constants)
    if values.shape != expected:
        raise ValueError(f"能量形狀必須是 {expected}（喇叭, 座位, F），收到 {values.shape}")
    dtype = values.dtype
    indices = jnp.asarray(constants.frequency_indices)
    x = jnp.take(jnp.asarray(constants.log2_frequencies, dtype=dtype), indices)
    weights = jnp.asarray(constants.frequency_weights, dtype=dtype)
    levels = 10.0 * jnp.log10(jnp.take(values, indices, axis=-1))
    primary = jnp.take(levels, jnp.asarray(constants.primary_index), axis=1)
    surrounding = jnp.take(levels, jnp.asarray(constants.surrounding_indices), axis=1)
    weight_sum = jnp.sum(weights)
    x_mean = jnp.sum(weights * x) / weight_sum
    y_mean = jnp.sum(weights * primary, axis=-1) / weight_sum
    centered = x - x_mean
    covariance = jnp.sum(weights * centered * (primary - y_mean[:, None]), axis=-1)
    variance = jnp.sum(weights * centered * centered)
    slope = covariance / variance
    intercept = y_mean - slope * x_mean
    residual = primary - (slope[:, None] * x + intercept[:, None])
    scale = jnp.maximum(jnp.max(jnp.abs(levels), axis=(1, 2)), jnp.ones((), dtype=dtype))
    floor = rms_floor_factor(dtype) * scale
    ripple = _weighted_rms(residual, weights, floor)
    robustness = jnp.mean(_weighted_rms(surrounding - primary[:, None, :], weights, floor[:, None]), axis=-1)
    means = jnp.stack((jnp.mean(ripple), jnp.mean(robustness)))
    component_weights = jnp.asarray(constants.component_weights, dtype=dtype)
    references = jnp.asarray(constants.worse_references, dtype=dtype)
    total = jnp.sum(component_weights * (means / references))
    return StageOneObjectiveArrays(ripple, robustness, means[0], means[1], total)


def objective_from_pressures(
    constants: StageOneObjectiveConstants, pressures: jax.Array,
) -> StageOneObjectiveArrays:
    """複數聲壓 ``(F, 喇叭, 座位)`` 取 ``abs(p)**2`` 再轉軸；與有限元素批次能量的定義相同。"""
    values = jnp.asarray(pressures)
    if not jnp.issubdtype(values.dtype, jnp.complexfloating):
        raise TypeError(f"聲壓必須是複數陣列，收到 {values.dtype}")
    speakers, receivers, frequencies = _energy_shape(constants)
    expected = (frequencies, speakers, receivers)
    if values.shape != expected:
        raise ValueError(f"聲壓形狀必須是 {expected}（F, 喇叭, 座位），收到 {values.shape}")
    return objective_from_energies(constants, jnp.transpose(jnp.abs(values) ** 2, (1, 2, 0)))
