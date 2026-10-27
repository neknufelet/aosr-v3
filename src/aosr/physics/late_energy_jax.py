"""鞋盒房間晚期混響能量的 JAX 版：同一套公式，對吸音率可微、可 ``jax.jit``。

公式照 :mod:`aosr.physics.late_energy`（決策紙
``late-energy-random-incidence-reciprocal-cpu-double-reference.md``）：每格反射率
``1−α`` 乘中心點形狀因子得到逐頻反射算子 ``R``，直接解 ``(I−R)y=E``、取
reflected-only 的 ``Ry`` 做面積加權平均，再乘 Eyring 比值。

分兩半：

- 跟材料無關的幾何（分格面積、牆索引、形狀因子）由 :func:`late_energy_geometry`
  借 numpy 版那一份實作、用雙精度算好一次，當常數交進來；不在這裡另寫第二份
  形狀因子。
- 跟材料有關的那一半 :func:`late_energy_arrays` 只用 ``jax.numpy``，吸音率 ``α`` 形狀
  ``(6, F)``。**精度跟輸入走**：``α`` 是 float64 就整條雙精度、float32 就整條單精度；
  幾何常數換成 ``α`` 的精度再參與運算，``α`` 本身不轉型。矩陣乘法一律指定最高精度，
  不吃平台預設（有的顯卡單精度乘法預設會降精度）。

這一層不碰 JAX 全域設定（規矩卡 ``layers-import-downward-only``）：要雙精度，由呼叫端
在開機前經 runtime 那一層打開。jit 裡無法依數值報錯，所以 ``α`` 有限且在 ``[0,1]``
這件事由呼叫端負責，這裡只驗形狀與型別。
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from aosr.config.source_reference import DIFFUSE_MONOPOLE_4PI
from aosr.geometry.shoebox import Room, Wall
from aosr.physics.late_energy import _form_factors, _patch_geometry, _wall_areas

_HIGHEST = jax.lax.Precision.HIGHEST
# Eyring 比值的上夾點跟 numpy 版同一個數：1 減單精度的一個最小刻度，單雙精度都表示得出來。
_EYRING_ALPHA_CEILING = 1.0 - float(np.finfo(np.float32).eps)


class LateEnergyGeometry(NamedTuple):
    """跟材料無關的那一半；NamedTuple 本身是 JAX 的樹狀資料，可以整包交進 jit。"""

    form_factors: NDArray[np.float64]
    patch_areas: NDArray[np.float64]
    wall_index: NDArray[np.int64]
    wall_areas: NDArray[np.float64]


class LateEnergyArrays(NamedTuple):
    """逐頻輸出，每格形狀 ``(F,)``，對應 ``LateEnergyBand`` 同名欄位。"""

    alpha_bar: jax.Array
    raw_reverberant_energy: jax.Array
    eyring_ratio: jax.Array
    late_reverberant_energy: jax.Array


def late_energy_geometry(room: Room, n_per_wall: int) -> LateEnergyGeometry:
    """用 numpy 版唯一那一份分格與形狀因子實作，雙精度算好材料無關的常數。"""
    patches = _patch_geometry(room, n_per_wall)
    return LateEnergyGeometry(
        form_factors=_form_factors(patches),
        patch_areas=patches.areas,
        wall_index=patches.wall_index,
        wall_areas=_wall_areas(room),
    )


def _checked_alpha(geometry: LateEnergyGeometry, alpha: jax.Array) -> jax.Array:
    """只驗形狀與型別（jit 裡也是靜態的）；不轉 ``α`` 的精度。"""
    values = jnp.asarray(alpha)
    if not jnp.issubdtype(values.dtype, jnp.floating):
        raise TypeError(f"吸音率必須是浮點陣列，收到 {values.dtype}")
    wall_count = len(Wall.all())
    if values.ndim != 2 or values.shape[0] != wall_count or values.shape[1] == 0:
        raise ValueError(f"吸音率形狀必須是 ({wall_count}, F) 且 F>0，收到 {values.shape}")
    patch_count = geometry.patch_areas.shape[0]
    if geometry.form_factors.shape != (patch_count, patch_count):
        raise ValueError("形狀因子與分格面積的格數不同")
    if geometry.wall_index.shape != (patch_count,) or geometry.wall_areas.shape != (wall_count,):
        raise ValueError("牆索引或牆面積的形狀不對")
    return values


def _eyring_ratio(alpha_bar: jax.Array) -> jax.Array:
    """``ᾱ/−ln(1−ᾱ)``，ᾱ 先夾進 ``[0, 1−ε32]``；ᾱ=0 回 1，梯度不經過 0/0。"""
    guarded = jnp.clip(alpha_bar, 0.0, _EYRING_ALPHA_CEILING)
    is_zero = guarded == 0.0
    safe = jnp.where(is_zero, jnp.ones_like(guarded), guarded)
    return jnp.where(is_zero, jnp.ones_like(guarded), safe / -jnp.log1p(-safe))


def late_energy_arrays(geometry: LateEnergyGeometry, alpha: jax.Array) -> LateEnergyArrays:
    """逐頻回（面積加權平均吸音率, 原始混響能量, Eyring 比值, 修正後晚期能量）。

    ``alpha`` 形狀 ``(6, F)``，牆的順序照 :meth:`Wall.wall_names`。輸出精度等於 ``alpha``
    的精度。逐頻反射算子 ``R_f[i,j] = (1−α_{w(i),f})·F_ij`` 疊成 ``(F, P, P)`` 一次批次直接解。
    """
    values = _checked_alpha(geometry, alpha)
    dtype = values.dtype
    form_factors = jnp.asarray(geometry.form_factors, dtype=dtype)
    patch_areas = jnp.asarray(geometry.patch_areas, dtype=dtype)
    wall_areas = jnp.asarray(geometry.wall_areas, dtype=dtype)
    patch_count = patch_areas.shape[0]
    frequency_count = values.shape[1]
    total_area = jnp.sum(patch_areas)

    reflectance = (1.0 - values[jnp.asarray(geometry.wall_index), :]).T
    transfer = reflectance[:, :, None] * form_factors[None, :, :]
    system = jnp.eye(patch_count, dtype=dtype)[None, :, :] - transfer
    source = jnp.broadcast_to(jnp.ones(patch_count, dtype=dtype) / total_area, (frequency_count, patch_count))
    field = jnp.linalg.solve(system, source[:, :, None])[:, :, 0]
    reflected = jnp.einsum("fij,fj->fi", transfer, field, precision=_HIGHEST)
    mean_reflected = jnp.matmul(reflected, patch_areas, precision=_HIGHEST) / total_area
    raw = DIFFUSE_MONOPOLE_4PI * 4.0 * mean_reflected

    alpha_bar = jnp.matmul(wall_areas, values, precision=_HIGHEST) / jnp.sum(wall_areas)
    ratio = _eyring_ratio(alpha_bar)
    return LateEnergyArrays(
        alpha_bar=alpha_bar,
        raw_reverberant_energy=raw,
        eyring_ratio=ratio,
        late_reverberant_energy=raw * ratio,
    )
