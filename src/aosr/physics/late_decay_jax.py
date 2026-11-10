"""晚期 T20 的可微陣列版；幾何先在 CPU 算好，吸音率為逐格 ``(P,F)``。

與 ``late_decay._solve_late_decay(include_t30=False)`` 共用定義：512 階能量（``LATE_DECAY_ORDERS``）、
精確根的尾巴、−5 至 −25 dB 軟視窗擬合。任何失敗逐帶亮旗標並回 NaN，
不拿 Perron 的漸近 T60 代替擬合。輸出與幾何常數的工作精度跟 α 走。
"""
from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from aosr.config.art_lane import (
    ART_WLS_T20_HI_DB,
    ART_WLS_T20_LO_DB,
    ART_WLS_WINDOW_SOFTNESS_DB,
    LATE_DECAY_ORDERS,
)
from aosr.geometry.shoebox import Room
from aosr.physics.late_decay import ART_WLS_LOG_FLOOR_DB, ART_WLS_MIN_WEIGHT
from aosr.physics.late_energy_jax import late_energy_geometry


class LateDecayGeometry(NamedTuple):
    """材料無關的幾何常數，可整包傳進 ``jax.jit``（即時編譯）。"""

    form_factors: NDArray[np.float64]
    patch_areas: NDArray[np.float64]
    collision_frequency_hz: float


class LateDecayArrays(NamedTuple):
    """每格形狀 ``(F,)``；失敗原因逐帶互斥，valid 則相反。

    優先序為吸音無效、數值失敗、衰減不足、權重不足、非負斜率。
    """

    t20_s: jax.Array
    slope_db_per_s: jax.Array
    soft_weight_sum: jax.Array
    valid: jax.Array
    invalid_absorption: jax.Array
    insufficient_decay_range: jax.Array
    insufficient_weight: jax.Array
    nonnegative_slope: jax.Array
    numerical_failure: jax.Array


def late_decay_geometry(room: Room, n_per_wall: int, *, sound_speed_m_s: float) -> LateDecayGeometry:
    """借唯一一份 NumPy 形狀因子；只在編譯外驗幾何與聲速。"""
    if not math.isfinite(sound_speed_m_s) or sound_speed_m_s <= 0.0:
        raise ValueError("sound_speed_m_s 必須是有限正數")
    geometry = late_energy_geometry(room, n_per_wall)
    surface = 2.0 * (room.Lx * room.Ly + room.Ly * room.Lz + room.Lx * room.Lz)
    volume = room.Lx * room.Ly * room.Lz
    collision = sound_speed_m_s * surface / (4.0 * volume)
    return LateDecayGeometry(geometry.form_factors, geometry.patch_areas, collision)


def _exact_roots(form: jax.Array, areas: jax.Array, reflectivity: jax.Array) -> jax.Array:
    """T=WB 與 sqrt(W) B sqrt(W) 相似；非負矩陣的 Perron 根是最大實根。

    B=diag(A)F 為面積加權互易交換矩陣。最後一次對稱平均只去掉由除面積再乘回
    造成的捨入差；不改形狀因子。α=1 在入口已標無效並換安全輸入，sqrt 不碰 0。
    """
    exchange = areas[:, None] * form
    exchange = (exchange + exchange.T) / 2.0
    root_w = jnp.sqrt(reflectivity / areas[:, None]).T
    symmetric = root_w[:, :, None] * exchange * root_w[:, None, :]
    return jnp.asarray(jnp.linalg.eigvalsh(symmetric)[:, -1])


def _order_decay(form: jax.Array, reflection: jax.Array, areas: jax.Array) -> jax.Array:
    weight = areas / jnp.sum(areas)

    def step(energy: jax.Array, unused: None) -> tuple[jax.Array, jax.Array]:
        energy = reflection * jnp.einsum("ij,jf->if", form, energy, precision=jax.lax.Precision.HIGHEST)
        return energy, jnp.einsum("p,pf->f", weight, energy, precision=jax.lax.Precision.HIGHEST)

    initial = jnp.ones(reflection.shape, dtype=reflection.dtype)
    _last, orders = jax.lax.scan(step, initial, None, length=LATE_DECAY_ORDERS)
    return orders


def _decay_level(orders: jax.Array, roots: jax.Array) -> jax.Array:
    valid_root = (roots > 0.0) & (roots < 1.0)
    safe_root = jnp.where(valid_root, roots, 0.5)
    tail = jnp.where(valid_root, orders[-1] * safe_root / (1.0 - safe_root), 0.0)
    edc = jnp.cumsum(orders[::-1], axis=0)[::-1] + tail[None, :]
    initial = edc[0]
    ratio = jnp.where(initial > 0.0, edc / jnp.where(initial > 0.0, initial, 1.0), 1.0)
    above_floor = ratio > np.finfo(orders.dtype).tiny
    safe_ratio = jnp.where(above_floor, ratio, 1.0)
    return jnp.where(above_floor, 10.0 * jnp.log10(safe_ratio), ART_WLS_LOG_FLOOR_DB)


def _fit_decay_arrays(level_db: jax.Array, collision_frequency_hz: jax.Array) -> LateDecayArrays:
    """軟視窗與 NumPy 同式；先遮壞輸入、再除與取倒數，讓其他帶梯度有限。"""
    numerical = ~jnp.all(jnp.isfinite(level_db), axis=0)
    levels = jnp.where(numerical[None, :], 0.0, level_db)
    time = (jnp.arange(1, LATE_DECAY_ORDERS + 1, dtype=levels.dtype) / collision_frequency_hz)[:, None]
    upper = (ART_WLS_T20_HI_DB - levels) / ART_WLS_WINDOW_SOFTNESS_DB
    lower = (levels - ART_WLS_T20_LO_DB) / ART_WLS_WINDOW_SOFTNESS_DB
    weights = jax.nn.sigmoid(upper) * jax.nn.sigmoid(lower)
    weight_sum = jnp.sum(weights, axis=0)
    insufficient_weight = weight_sum <= ART_WLS_MIN_WEIGHT
    safe_weight = jnp.where(insufficient_weight, 1.0, weight_sum)
    time_mean = jnp.sum(weights * time, axis=0) / safe_weight
    level_mean = jnp.sum(weights * levels, axis=0) / safe_weight
    covariance = jnp.sum(weights * (time - time_mean) * (levels - level_mean), axis=0)
    variance = jnp.sum(weights * (time - time_mean)**2, axis=0)
    can_fit = (variance > 0.0) & ~insufficient_weight
    slope = jnp.where(can_fit, covariance / jnp.where(can_fit, variance, 1.0), 0.0)
    nonnegative_slope = ~insufficient_weight & (slope >= 0.0)
    insufficient_range = jnp.min(levels, axis=0) > ART_WLS_T20_LO_DB
    numerical = numerical | ~jnp.isfinite(slope) | ~jnp.isfinite(weight_sum)
    insufficient_range = insufficient_range & ~numerical
    fit_eligible = ~(numerical | insufficient_range)
    insufficient_weight = insufficient_weight & fit_eligible
    nonnegative_slope = nonnegative_slope & fit_eligible
    valid = ~(insufficient_weight | nonnegative_slope | insufficient_range | numerical)
    safe_slope = jnp.where(valid, slope, -1.0)
    t20 = -60.0 / safe_slope
    return LateDecayArrays(jnp.where(valid, t20, jnp.nan), slope, weight_sum, valid,
                           jnp.zeros_like(valid), insufficient_range, insufficient_weight,
                           nonnegative_slope, numerical)


def late_decay_arrays(geometry: LateDecayGeometry, alpha: jax.Array) -> LateDecayArrays:
    """逐格 α（P,F）→逐頻 T20 與原因旗標；jit 內只用陣列判斷，不丟例外。

    α 非有限、不在 [0,1)、或正吸音在工作精度下被捨成零時，整帶無效。
    先把整個無效帶換安全輸入，避免 sqrt、特徵值或擬合的無效分支污染反傳。
    幾何由 ``late_decay_geometry`` 在 CPU 驗好；呼叫端負責形狀 (P,F)。
    """
    values = jnp.asarray(alpha)
    areas = jnp.asarray(geometry.patch_areas, dtype=values.dtype)
    form = jnp.asarray(geometry.form_factors, dtype=values.dtype)
    collision = jnp.asarray(geometry.collision_frequency_hz, dtype=values.dtype)
    invalid = jnp.any(~jnp.isfinite(values) | (values < 0.0) | (values >= 1.0)
                      | ((values > 0.0) & ((1.0 - values) == 1.0)), axis=0)
    safe_alpha = jnp.where(invalid[None, :], 0.5, values)
    reflection = 1.0 - safe_alpha
    roots = _exact_roots(form, areas, reflection)
    orders = _order_decay(form, reflection, areas)
    fit = _fit_decay_arrays(_decay_level(orders, roots), collision)
    numerical = ~invalid & (fit.numerical_failure | ~jnp.isfinite(roots))
    fit_eligible = ~(invalid | numerical)
    valid = fit.valid & fit_eligible
    return fit._replace(t20_s=jnp.where(valid, fit.t20_s, jnp.nan), valid=valid,
                        invalid_absorption=invalid, numerical_failure=numerical,
                        insufficient_decay_range=fit.insufficient_decay_range & fit_eligible,
                        insufficient_weight=fit.insufficient_weight & fit_eligible,
                        nonnegative_slope=fit.nonnegative_slope & fit_eligible)
