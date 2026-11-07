"""Eyring、Schroeder 與三路接合的可微陣列版；交接權重只傳值、不傳梯度。

公式與常數沿用 crossover。數值失效回逐項旗標與 NaN；先替換失效輸入再算，
避免未選分支的無效導數污染其他頻點。f_s 達上限照正式 NumPy 硬切；上限旗標只供報告，
不使接合失效，權重仍擋梯度。
房間尺寸、牆面積與兩個中頻帶的選取由呼叫端準備；不接材料鏈或搜尋。
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from aosr.config.frequency_axis import FEM_GEOMETRIC_CROSSOVER_CAP_HZ
from aosr.config.three_lane_crossover import (
    CROSSOVER_LOWER_FLOOR_HZ, EYRING_COEFFICIENT_S_PER_M, SCHROEDER_COEFFICIENT_SI, SCHROEDER_T60_BANDS_HZ,
)


class CheckedBands(NamedTuple):
    """同形狀的值與失效旗標；旗標為真時該格值為 NaN。"""

    values: jax.Array
    invalid: jax.Array


class CrossoverArrays(NamedTuple):
    """逐頻互補功率權重與數值旗標；上限旗標是每個房間一格。"""

    w_fem: jax.Array
    w_geo: jax.Array
    capped_by_upper_limit: jax.Array
    invalid: jax.Array


def eyring_t60_arrays(wall_areas: NDArray[np.float64] | jax.Array, volume_m3: float | jax.Array,
                     alpha_by_wall: jax.Array) -> CheckedBands:
    """面積等效的逐牆 α(W,F) → Eyring T60(F)，秒；分區先用原交集權重換成整面列。"""
    alpha = jnp.asarray(alpha_by_wall)
    areas, volume = jnp.asarray(wall_areas, dtype=alpha.dtype), jnp.asarray(volume_m3, dtype=alpha.dtype)
    valid_room = jnp.all(jnp.isfinite(areas) & (areas > 0.0)) & jnp.isfinite(volume) & (volume > 0.0)
    valid = jnp.all(jnp.isfinite(alpha) & (alpha >= 0.0) & (alpha < 1.0), axis=0) & valid_room
    safe_alpha = jnp.where(valid[None, :], alpha, 0.5)
    safe_areas, safe_volume = jnp.where(valid_room, areas, 1.0), jnp.where(valid_room, volume, 1.0)
    surface = jnp.sum(safe_areas)
    average = jnp.matmul(safe_areas, safe_alpha, precision=jax.lax.Precision.HIGHEST) / surface
    valid = valid & (average > 0.0) & (average < 1.0)
    safe_average = jnp.where(valid, average, 0.5)
    t60 = EYRING_COEFFICIENT_S_PER_M * safe_volume / (-surface * jnp.log(1.0 - safe_average))
    valid = valid & jnp.isfinite(t60) & (t60 > 0.0)
    return CheckedBands(jnp.where(valid, t60, jnp.nan), ~valid)


def schroeder_frequency_arrays(volume_m3: float | jax.Array, midband_t60_s: jax.Array) -> CheckedBands:
    """依 SCHROEDER_T60_BANDS_HZ 順序給兩帶 T60，回 f_s（Hz）；選頻留在呼叫端。"""
    t60 = jnp.asarray(midband_t60_s)
    if t60.shape != (len(SCHROEDER_T60_BANDS_HZ),):
        raise ValueError("Schroeder 輸入須為設定指定的兩個中頻帶")
    volume = jnp.asarray(volume_m3, dtype=t60.dtype)
    valid = jnp.all(jnp.isfinite(t60) & (t60 > 0.0)) & jnp.isfinite(volume) & (volume > 0.0)
    safe_t60, safe_volume = jnp.where(valid, t60, 1.0), jnp.where(valid, volume, 1.0)
    fs = SCHROEDER_COEFFICIENT_SI * jnp.sqrt(jnp.mean(safe_t60) / safe_volume)
    valid = valid & jnp.isfinite(fs) & (fs > 0.0)
    return CheckedBands(jnp.where(valid, fs, jnp.nan), ~valid)


def crossover_weights_arrays(frequencies_hz: jax.Array, f_s_hz: jax.Array) -> CrossoverArrays:
    """值照 NumPy 的對數平滑／上限硬切；兩組權重的導數恰為零。"""
    fs = jnp.asarray(f_s_hz)
    frequencies = jnp.asarray(frequencies_hz, dtype=fs.dtype)
    valid_fs = jnp.isfinite(fs) & (fs > 0.0)
    valid_frequency = jnp.isfinite(frequencies) & (frequencies > 0.0)
    valid = valid_fs & valid_frequency
    capped = valid_fs & (fs >= FEM_GEOMETRIC_CROSSOVER_CAP_HZ)
    # 硬切時不計算 log(cap/lower)=0 的分母，未選分支也必須可安全反傳。
    lower = jnp.maximum(jnp.where(valid_fs & ~capped, fs, CROSSOVER_LOWER_FLOOR_HZ), CROSSOVER_LOWER_FLOOR_HZ)
    safe_frequency = jnp.where(valid_frequency, frequencies, lower)
    t = jnp.log2(safe_frequency / lower) / jnp.log2(FEM_GEOMETRIC_CROSSOVER_CAP_HZ / lower)
    soft = jnp.where(safe_frequency <= lower, 0.0, jnp.where(safe_frequency >= FEM_GEOMETRIC_CROSSOVER_CAP_HZ,
        1.0, t * t * (3.0 - 2.0 * t)))
    hard = jnp.where(safe_frequency <= FEM_GEOMETRIC_CROSSOVER_CAP_HZ, 0.0, 1.0)
    geo = jnp.where(valid, jnp.where(capped, hard, soft), jnp.nan)
    fem = 1.0 - geo
    return CrossoverArrays(jax.lax.stop_gradient(fem), jax.lax.stop_gradient(geo), capped, ~valid)


def stitched_energy_arrays(weights: CrossoverArrays, fem_energy: jax.Array, geometric_energy: jax.Array) -> CheckedBands:
    """w_fem E_fem + w_geo E_geo；無 FEM 的頻點可給 NaN，但正 FEM 權重缺值必須亮旗標。

    E_geo 是上游三欄早期加晚期那一欄。輸入可帶前置批次軸，最後一軸為頻率。
    f_s 達上限照 NumPy 硬切：頻率 ≤ 上限全 FEM、> 上限全幾何；上限旗標不參與有效性。
    """
    fem, geo = jnp.asarray(fem_energy), jnp.asarray(geometric_energy)
    used_fem, used_geo = weights.w_fem > 0.0, weights.w_geo > 0.0
    valid_fem, valid_geo = jnp.isfinite(fem), jnp.isfinite(geo)
    valid = ~weights.invalid & (~used_fem | valid_fem) & (~used_geo | valid_geo)
    safe_fem, safe_geo = jnp.where(valid_fem & used_fem, fem, 0.0), jnp.where(valid_geo & used_geo, geo, 0.0)
    wf, wg = jnp.where(weights.invalid, 0.0, weights.w_fem), jnp.where(weights.invalid, 0.0, weights.w_geo)
    total = wf * safe_fem + wg * safe_geo
    valid = valid & jnp.isfinite(total)
    return CheckedBands(jnp.where(valid, total, jnp.nan), ~valid)
