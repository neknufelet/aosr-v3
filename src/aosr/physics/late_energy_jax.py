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

有效範圍用精度講、不用數字：``α > 0`` 的每一格，反射率 ``1−α`` 要在工作精度（``α`` 的 dtype）
下跟 1 分得開。分不開時（例如單精度下極小的 ``α`` 被捨掉、反射率恰為 1）那一格的吸音丟了，
全部丟光時耗散消失、``I−R`` 近奇異，求解沒有定義，但線性解照樣回一個有限的數、可能是負的。
``α`` 恰為 0 的剛性面是合法輸入，本來就沒有吸音可丟，不算。這裡不夾 ``α`` 下限、不把能量截成 0，
原始數字照算照回，另回逐頻旗標 ``dissipation_resolved``：有一格吸音被捨掉、或晚期能量非有限或
不大於 0，那一頻就是 False，呼叫端要自己查這一格。
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
from aosr.physics.wall_surfaces import ZoneAbsorption, rectangle_patch_weights, validate_zone_absorption

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
    """逐頻輸出，每格形狀 ``(F,)``；前四格對應 ``LateEnergyBand`` 同名欄位。

    ``dissipation_resolved`` 是布林：那一頻 ``α > 0`` 的每一格 ``1−α`` 在工作精度下都小於 1、而且
    晚期能量有限且大於 0 才是 True。False 的那一頻求解沒有定義，前四格照算照回、不准當答案用。
    """

    alpha_bar: jax.Array
    raw_reverberant_energy: jax.Array
    eyring_ratio: jax.Array
    late_reverberant_energy: jax.Array
    dissipation_resolved: jax.Array


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


def late_energy_arrays(geometry: LateEnergyGeometry, alpha: jax.Array, *,
                       alpha_by_patch: jax.Array | None = None) -> LateEnergyArrays:
    """逐頻回（面積加權平均吸音率, 原始混響能量, Eyring 比值, 修正後晚期能量）。

    ``alpha`` 形狀 ``(6, F)``，牆的順序照 :meth:`Wall.wall_names`。輸出精度等於 ``alpha``
    的精度。逐頻反射算子 ``R_f[i,j] = (1−α_{w(i),f})·F_ij`` 疊成 ``(F, P, P)`` 一次批次直接解。

    ``α > 0`` 的格，``1−α`` 要在工作精度下跟 1 分得開；分不開（吸音被捨掉）時求解沒有定義、
    ``dissipation_resolved`` 為 False，呼叫端要自己查這格。``α`` 恰為 0 的剛性面合法、不算。
    旗標只由比較算出，不參與梯度，前四格的算法與它無關。
    """
    values = _checked_alpha(geometry, alpha)
    dtype = values.dtype
    form_factors = jnp.asarray(geometry.form_factors, dtype=dtype)
    patch_areas = jnp.asarray(geometry.patch_areas, dtype=dtype)
    wall_areas = jnp.asarray(geometry.wall_areas, dtype=dtype)
    patch_count = patch_areas.shape[0]
    frequency_count = values.shape[1]
    total_area = jnp.sum(patch_areas)

    if alpha_by_patch is None:
        patch_alpha = values[jnp.asarray(geometry.wall_index), :].T
    else:
        if alpha_by_patch.shape != (patch_count, frequency_count) or alpha_by_patch.dtype != dtype:
            raise ValueError("小塊吸音率的形狀或精度與逐牆列不同")
        patch_alpha = alpha_by_patch.T
    reflectance = 1.0 - patch_alpha
    transfer = reflectance[:, :, None] * form_factors[None, :, :]
    system = jnp.eye(patch_count, dtype=dtype)[None, :, :] - transfer
    source = jnp.broadcast_to(jnp.ones(patch_count, dtype=dtype) / total_area, (frequency_count, patch_count))
    field = jnp.linalg.solve(system, source[:, :, None])[:, :, 0]
    reflected = jnp.einsum("fij,fj->fi", transfer, field, precision=_HIGHEST)
    mean_reflected = jnp.matmul(reflected, patch_areas, precision=_HIGHEST) / total_area
    raw = DIFFUSE_MONOPOLE_4PI * 4.0 * mean_reflected

    alpha_bar = jnp.matmul(wall_areas, values, precision=_HIGHEST) / jnp.sum(wall_areas)
    ratio = _eyring_ratio(alpha_bar)
    late = raw * ratio
    # 看實際參與求解的那一份反射率，不另算 1−α：「工作精度下分不分得開」就是這一份說了算。
    absorption_lost = jnp.any((patch_alpha > 0.0) & (reflectance >= 1.0), axis=1)
    resolved = ~absorption_lost & jnp.isfinite(late) & (late > 0.0)
    return LateEnergyArrays(
        alpha_bar=alpha_bar,
        raw_reverberant_energy=raw,
        eyring_ratio=ratio,
        late_reverberant_energy=late,
        dissipation_resolved=resolved,
    )


class ZonedLateEnergyGeometry(NamedTuple):
    """固定分區的交集權重；區域材料的列序是六面整面列後，按牆序接每面區序。"""

    whole: LateEnergyGeometry
    patch_weights: NDArray[np.float64]
    wall_weights: NDArray[np.float64]


def zoned_late_energy_geometry(room: Room, n_per_wall: int, absorption_by_zone: ZoneAbsorption) -> ZonedLateEnergyGeometry:
    """只用矩形座標生成權重；固定分區大小不參與材料梯度。"""
    whole = late_energy_geometry(room, n_per_wall)
    count = next((len(cell.absorption) for cells in absorption_by_zone.values() for cell in cells), 0)
    validate_zone_absorption(room, absorption_by_zone, count)
    row_count = len(Wall.all()) + sum(len(cells) for cells in absorption_by_zone.values())
    patches = np.zeros((len(whole.patch_areas), row_count), dtype=np.float64)
    walls = np.zeros((len(Wall.all()), row_count), dtype=np.float64)
    patches[np.arange(len(whole.patch_areas)), whole.wall_index] = 1.0
    walls[:, :len(Wall.all())] = np.eye(len(Wall.all()))
    offset = len(Wall.all())
    for index, name in enumerate(Wall.wall_names()):
        cells = absorption_by_zone.get(name, ())
        if not cells:
            continue
        start = index * n_per_wall**2
        patches[start:start+n_per_wall**2, index] = 0.0
        patches[start:start+n_per_wall**2, offset:offset+len(cells)] = rectangle_patch_weights(room, name, cells, n_per_wall)
        areas = [(cell.zone.u_m[1]-cell.zone.u_m[0]) *
                 (cell.zone.v_m[1]-cell.zone.v_m[0]) for cell in cells]
        walls[index, index] = 0.0
        walls[index, offset:offset+len(cells)] = [area / sum(areas) for area in areas]
        offset += len(cells)
    return ZonedLateEnergyGeometry(whole, patches, walls)


def _bounded_weighted_absorption(weights: NDArray[np.float64], values: jax.Array) -> jax.Array:
    """分區凸組合只受正權重區的逐頻界約束；單一區直接取原列。"""
    coefficients = jnp.asarray(weights, dtype=values.dtype)
    active = coefficients > 0.0
    average = jnp.matmul(coefficients, values, precision=_HIGHEST) / jnp.sum(coefficients, axis=1)[:, None]
    lower = jnp.min(jnp.where(active[:, :, None], values[None, :, :], jnp.inf), axis=1)
    upper = jnp.max(jnp.where(active[:, :, None], values[None, :, :], -jnp.inf), axis=1)
    direct = values[jnp.argmax(coefficients, axis=1)]
    return jnp.where(jnp.sum(active, axis=1)[:, None] == 1, direct, jnp.clip(average, lower, upper))


def _bounded_absorption_with_linear_gradient(weights: NDArray[np.float64], values: jax.Array) -> jax.Array:
    """值沿用原浮點凸組合護欄；導數是固定權重平均，同材時也不能被 min/max 平分。

    非負權重的凸組合在數學上永遠在輸入上下界內；夾值只修捨入，不改物理導數。
    幾何權重是固定常數，只對材料吸音率求導。前向沿用原 helper，既有與新分區入口共用這份導數。
    """
    @jax.custom_jvp
    def bounded(alpha: jax.Array) -> jax.Array:
        return _bounded_weighted_absorption(weights, alpha)

    @bounded.defjvp
    def bounded_jvp(primals: tuple[jax.Array], tangents: tuple[jax.Array]) -> tuple[jax.Array, jax.Array]:
        coefficients = jnp.asarray(weights, dtype=primals[0].dtype)
        tangent = jnp.matmul(coefficients, tangents[0], precision=_HIGHEST) / jnp.sum(coefficients, axis=1)[:, None]
        return bounded(primals[0]), tangent

    return bounded(values)


def zoned_late_energy_arrays(geometry: ZonedLateEnergyGeometry, alpha: jax.Array) -> LateEnergyArrays:
    """以固定交集對逐區 α 可微；六面整面列後，按 Wall.wall_names 與區宣告順序接矩形列。

    分區牆的整面輸入列不參與結果；有效範圍與旗標沿用原函式，呼叫端仍須保證 α 有限且在 [0,1]。
    """
    values = jnp.asarray(alpha)
    if not jnp.issubdtype(values.dtype, jnp.floating):
        raise TypeError("吸音率必須是浮點陣列")
    if values.ndim != 2 or values.shape[0] != geometry.wall_weights.shape[1] or values.shape[1] == 0:
        raise ValueError("分區吸音率形狀與幾何權重不同")
    walls = _bounded_absorption_with_linear_gradient(geometry.wall_weights, values)
    patches = _bounded_absorption_with_linear_gradient(geometry.patch_weights, values)
    return late_energy_arrays(geometry.whole, walls, alpha_by_patch=patches)
