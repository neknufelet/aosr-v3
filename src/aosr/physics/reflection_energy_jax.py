"""固定幾何、逐反彈阻抗與整房散射 → 相對直達的逐路徑能量（#562 2c）。

照 report_path_table._path_row 的 (1-s)^order |p|²/|p_direct|²；每條各取模平方，沒有路徑間干涉。
可選 pressure_factor 是上游已算好的聲源指向性壓力因子 (P,F)，不是材料鏈。
反射乘積只叫 amplitude_jax.reflection_products，未用反彈格照它的安全遮罩處理。
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from aosr.physics import amplitude_jax as amp


class ReflectionEnergyArrays(NamedTuple):
    """能量與逐路徑／逐頻有效旗標 (...,P,F)；散射與直達旗標 (...,F)。"""

    energy: jax.Array
    valid: jax.Array
    impedance_valid: jax.Array
    scattering_valid: jax.Array
    direct_valid: jax.Array


def reflection_path_energy(
    table: amp.PathTable, z_bounce: jax.Array, scattering: jax.Array, rho_c: float,
    pressure_factor: jax.Array | None = None,
) -> ReflectionEnergyArrays:
    """逐反彈 Z (...,P,B,F)、s (...,F)，回同精度能量；無效格先安全求值再回 NaN。

    幾何與物理介質由上游驗證並固定；動態的非法 Z、奇異反射分母、散射超界、
    指向性非有限或直達能量不為正都不在編譯內丟錯。旗標 True 表示可用。
    """
    z = amp._complex(jnp.asarray(z_bounce))
    dtype = z.real.dtype
    mask = jnp.asarray(table.bounce_mask)[:, :, None]
    cos = jnp.asarray(table.bounce_cos, dtype=dtype)[:, :, None]
    used_z = jnp.where(mask, z, jnp.asarray(rho_c, dtype=z.dtype))
    good_bounce = jnp.isfinite(used_z) & jnp.isfinite(used_z * cos) & ((used_z * cos + rho_c) != 0)
    impedance_valid = jnp.all(~mask | good_bounce, axis=-2)
    safe_z = jnp.where(good_bounce, used_z, jnp.asarray(rho_c, dtype=z.dtype))
    product = amp.reflection_products(table, safe_z, rho_c)
    pressure = amp.path_pressures(table, product)
    factor = jnp.ones_like(pressure) if pressure_factor is None else jnp.asarray(pressure_factor, dtype=z.dtype)
    factor_valid = jnp.isfinite(factor)
    pressure = pressure * jnp.where(factor_valid, factor, jnp.ones((), dtype=z.dtype))
    power = pressure.real ** 2 + pressure.imag ** 2
    direct = power[..., table.direct_index(), :]
    direct_valid = jnp.isfinite(direct) & (direct > 0) & factor_valid[..., table.direct_index(), :]
    safe_direct = jnp.where(direct_valid, direct, jnp.ones((), dtype=dtype))
    s = jnp.asarray(scattering, dtype=dtype)
    scattering_valid = jnp.isfinite(s) & (s >= 0) & (s <= 1)
    safe_s = jnp.where(scattering_valid, s, jnp.zeros((), dtype=dtype))
    energy = (1 - safe_s[..., None, :]) ** jnp.asarray(table.order)[:, None] * power / safe_direct[..., None, :]
    valid = (impedance_valid & factor_valid & direct_valid[..., None, :] & scattering_valid[..., None, :]
             & jnp.isfinite(energy) & (energy >= 0))
    return ReflectionEnergyArrays(jnp.where(valid, energy, jnp.asarray(jnp.nan, dtype=dtype)), valid,
                                  impedance_valid, scattering_valid, direct_valid)
