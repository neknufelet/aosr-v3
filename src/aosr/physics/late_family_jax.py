"""無規入射吸音、逐階晚期能量與散射晚期欄的 JAX 陣列入口。

構造的角度陣列直接由 transfer_matrix.diffuse_incidence_angles_deg(method='paris') 交入，
沿用其梯形積分；固定阻抗沿用 catalog_absorption 的 Gauss–Legendre 節點與權重。
分區幾何由既有 zoned_late_energy_geometry 準備。家具整件吸音面積為常數輸入，
同一 ε 分配各區剩餘反射率（家具紙第 23 條），不把分區壓成整面再求輻射度。
late_share_energy 尚未乘聲源功率比，呼叫端須乘上該比值才可餵三路接合。
全部輸出跟隨輸入精度；不改全域設定、不接材料鏈或目標函式。
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from aosr.config.source_reference import DIFFUSE_MONOPOLE_4PI
from aosr.config.three_lane_crossover import REFLECTION_ORDER_K
from aosr.materials.catalog_absorption import _PARIS_MAPPED_WEIGHTS as PARIS_MAPPED_WEIGHTS, _PARIS_MU as PARIS_MU
from aosr.materials.transfer_matrix import trapezoidal_integral
from aosr.physics.crossover_jax import CheckedBands
from aosr.physics.late_energy_jax import (
    LateEnergyGeometry, ZonedLateEnergyGeometry,
    _bounded_absorption_with_linear_gradient as _bounded_absorption_with_linear_gradient, late_energy_arrays,
)

_HIGHEST = jax.lax.Precision.HIGHEST


def paris_angular_weights(incidence_angles_deg: jax.Array) -> jax.Array:
    """正式構造積分的 sinθ cosθ 權重；角度由原角度生成器提供，不另造角度格。"""
    theta = jnp.deg2rad(incidence_angles_deg)
    return jnp.sin(theta) * jnp.cos(theta)


def paris_absorption_arrays(impedance_pa_s_m: jax.Array, incidence_angles_deg: jax.Array,
                            rho_c_pa_s_m: float | jax.Array) -> CheckedBands:
    """Z(...,F,Nθ) → α(...,F)，構造的正式 Paris 梯形積分；數值失效逐頻隔離。"""
    z = jnp.asarray(impedance_pa_s_m)
    dtype = z.real.dtype
    angles = jnp.asarray(incidence_angles_deg, dtype=dtype)
    if angles.ndim != 1 or angles.size < 2 or z.shape[-1] != angles.size:
        raise ValueError("Paris 積分角度必須與阻抗最後一軸相同且至少兩點")
    rho = jnp.asarray(rho_c_pa_s_m, dtype=dtype)
    valid_angles = jnp.all(jnp.isfinite(angles) & (angles >= 0.0) & (angles < 90.0)) & jnp.all(jnp.diff(angles) > 0.0)
    # 無效角度整組失效；安全支由原索引正規化到有限角度，僅供未選分支避開零分母。
    safe_angles = jnp.where(valid_angles, angles, jnp.arange(angles.size, dtype=dtype) / angles.size)
    valid_rho = jnp.isfinite(rho) & (rho > 0.0)
    valid = jnp.all(jnp.isfinite(z) & (z.real >= 0.0), axis=-1) & valid_angles & valid_rho
    safe_z, safe_rho = jnp.where(valid[..., None], z, 1.0 + 0j), jnp.where(valid_rho, rho, 1.0)
    theta = jnp.deg2rad(safe_angles)
    projected = safe_z * jnp.cos(theta)
    reflection = (projected - safe_rho) / (projected + safe_rho)
    alpha = 1.0 - jnp.abs(reflection) ** 2
    weights = paris_angular_weights(safe_angles)
    result = trapezoidal_integral(alpha * weights, theta) / trapezoidal_integral(weights, theta)
    valid = valid & jnp.isfinite(result)
    return CheckedBands(jnp.where(valid, result, jnp.nan), ~valid)


def constant_absorption_arrays(impedance_pa_s_m: jax.Array, rho_c_pa_s_m: float | jax.Array) -> CheckedBands:
    """固定阻抗 Z(...,F) 的正式 256 點 Gauss–Legendre 換算，與構造角度積分分開。"""
    z = jnp.asarray(impedance_pa_s_m)
    rho = jnp.asarray(rho_c_pa_s_m, dtype=z.real.dtype)
    valid_rho = jnp.isfinite(rho) & (rho > 0.0)
    valid = jnp.isfinite(z) & (z.real > 0.0) & valid_rho
    safe_z, safe_rho = jnp.where(valid, z, 1.0 + 0j), jnp.where(valid_rho, rho, 1.0)
    zeta = safe_z / safe_rho
    mu, weights = jnp.asarray(PARIS_MU, dtype=z.real.dtype), jnp.asarray(PARIS_MAPPED_WEIGHTS, dtype=z.real.dtype)
    reflection = (zeta[..., None] * mu - 1.0) / (zeta[..., None] * mu + 1.0)
    result = 2.0 * jnp.sum(weights * (1.0 - jnp.abs(reflection) ** 2) * mu, axis=-1)
    valid = valid & jnp.isfinite(result)
    return CheckedBands(jnp.where(valid, result, jnp.nan), ~valid)


class LateOrderArrays(NamedTuple):
    """逐頻總量、前 K 階 (F,K)、尾巴及求解失效旗標。"""

    total_energy: jax.Array
    energy_by_order: jax.Array
    tail_energy: jax.Array
    invalid: jax.Array


def late_energy_by_order_arrays(geometry: LateEnergyGeometry, alpha: jax.Array, *,
                                alpha_by_patch: jax.Array | None = None,
                                max_order: int = REFLECTION_ORDER_K) -> LateOrderArrays:
    """沿用既有總量解，以 T 逐階乘向量求 A_k；K 是編譯時的正整數。"""
    if max_order < 1:
        raise ValueError("max_order 必須至少是一階")
    patches = alpha[jnp.asarray(geometry.wall_index)] if alpha_by_patch is None else alpha_by_patch
    valid = jnp.all(jnp.isfinite(alpha) & (alpha >= 0.0) & (alpha <= 1.0), axis=0)
    valid = valid & jnp.all(jnp.isfinite(patches) & (patches >= 0.0) & (patches <= 1.0), axis=0)
    valid = valid & jnp.any(patches > 0.0, axis=0) & ~jnp.any((patches > 0.0) & ((1.0 - patches) >= 1.0), axis=0)
    safe_alpha, safe_patches = jnp.where(valid[None, :], alpha, 0.5), jnp.where(valid[None, :], patches, 0.5)
    total = late_energy_arrays(geometry, safe_alpha, alpha_by_patch=safe_patches)
    areas = jnp.asarray(geometry.patch_areas, dtype=alpha.dtype)
    surface = jnp.sum(areas)
    operator = (1.0 - safe_patches.T)[:, :, None] * jnp.asarray(geometry.form_factors, dtype=alpha.dtype)[None, :, :]
    term = jnp.broadcast_to(jnp.ones_like(areas) / surface, safe_patches.T.shape)
    orders = []
    for _ in range(max_order):
        term = jnp.einsum("fij,fj->fi", operator, term, precision=_HIGHEST)
        raw = DIFFUSE_MONOPOLE_4PI * 4.0 * jnp.matmul(term, areas, precision=_HIGHEST) / surface
        orders.append(raw * total.eyring_ratio)
    by_order = jnp.stack(orders, axis=-1)
    # 原版 sum(tuple) 從左到右累加，保留同一個尾巴扣法。
    tail = total.late_reverberant_energy - sum(orders)
    magnitude = jax.lax.stop_gradient(jnp.abs(total.late_reverberant_energy))
    rounding = (jnp.nextafter(magnitude, jnp.inf) - magnitude) * (max_order + 1)
    valid = valid & jnp.isfinite(total.late_reverberant_energy) & (total.late_reverberant_energy >= 0.0) & (tail >= -rounding)
    return LateOrderArrays(jnp.where(valid, total.late_reverberant_energy, jnp.nan),
        jnp.where(valid[:, None], by_order, jnp.nan), jnp.where(valid, jnp.maximum(tail, 0.0), jnp.nan), ~valid)


class LateFamilyArrays(NamedTuple):
    """晚期一族陣列；四個逐頻旗標分辨吸音、家具、求解、散射失效。"""

    alpha_by_wall: jax.Array
    alpha_by_patch: jax.Array
    total_energy: jax.Array
    energy_by_order: jax.Array
    tail_energy: jax.Array
    late_share_energy: jax.Array
    invalid_absorption: jax.Array
    invalid_furniture: jax.Array
    invalid_energy: jax.Array
    invalid_scattering: jax.Array


def late_family_arrays(geometry: ZonedLateEnergyGeometry, alpha: jax.Array, scattering: jax.Array, *,
                       furniture_absorption_area_m2: jax.Array | None = None,
                       max_order: int = REFLECTION_ORDER_K) -> LateFamilyArrays:
    """六面整面列後接各面矩形列的 α(R,F)，輸出逐階與散射晚期欄；s(F) 為上游輸入。

    late_share_energy 對應 _late_share_energy，尚未乘聲源功率比；呼叫端須先乘該比值再接合。
    """
    used = jnp.any(jnp.asarray(geometry.wall_weights) > 0.0, axis=0) | jnp.any(jnp.asarray(geometry.patch_weights) > 0.0, axis=0)
    cells_valid = jnp.isfinite(alpha) & (alpha >= 0.0) & (alpha <= 1.0)
    valid_alpha = jnp.all(cells_valid | ~used[:, None], axis=0)
    safe_alpha = jnp.where(cells_valid, alpha, 0.5)
    walls = _bounded_absorption_with_linear_gradient(geometry.wall_weights, safe_alpha)
    patches = _bounded_absorption_with_linear_gradient(geometry.patch_weights, safe_alpha)
    valid_furniture = jnp.ones(alpha.shape[1], dtype=jnp.bool_)
    if furniture_absorption_area_m2 is not None:
        walls, patches, valid_furniture = _furniture_absorption(geometry, walls, patches, furniture_absorption_area_m2)
    usable = valid_alpha & valid_furniture
    result = late_energy_by_order_arrays(geometry.whole, jnp.where(usable[None, :], walls, 0.5),
        alpha_by_patch=jnp.where(usable[None, :], patches, 0.5), max_order=max_order)
    valid_energy = usable & ~result.invalid
    valid_scattering = jnp.isfinite(scattering) & (scattering >= 0.0) & (scattering <= 1.0)
    safe_s = jnp.where(valid_scattering, scattering, 0.5)
    safe_orders = jnp.where(result.invalid[:, None], 0.0, result.energy_by_order)
    share = jnp.where(result.invalid, 0.0, result.tail_energy)
    for order in range(1, max_order + 1):
        share = share + (1.0 - (1.0 - safe_s) ** order) * safe_orders[:, order - 1]
    return LateFamilyArrays(jnp.where(usable[None, :], walls, jnp.nan), jnp.where(usable[None, :], patches, jnp.nan),
        jnp.where(valid_energy, result.total_energy, jnp.nan), jnp.where(valid_energy[:, None], result.energy_by_order, jnp.nan),
        jnp.where(valid_energy, result.tail_energy, jnp.nan), jnp.where(valid_energy & valid_scattering, share, jnp.nan),
        ~valid_alpha, ~valid_furniture, result.invalid, ~valid_scattering)


def _furniture_absorption(geometry: ZonedLateEnergyGeometry, walls: jax.Array, patches: jax.Array,
                          absorption_area_m2: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """整房剩餘反射面積作分母；家具量為常數，不擋牆面吸音對 ε 的導數。"""
    af = jax.lax.stop_gradient(jnp.asarray(absorption_area_m2, dtype=walls.dtype))
    areas = jnp.asarray(geometry.whole.wall_areas, dtype=walls.dtype)
    denominator = jnp.sum(areas[:, None] * (1.0 - walls), axis=0)
    valid = jnp.isfinite(af) & (af >= 0.0) & jnp.isfinite(denominator) & (denominator > 0.0) & (af < denominator)
    fraction = jnp.where(valid, af, 0.0) / jnp.where(valid, denominator, 1.0)
    return walls + (1.0 - walls) * fraction, patches + (1.0 - patches) * fraction, valid
