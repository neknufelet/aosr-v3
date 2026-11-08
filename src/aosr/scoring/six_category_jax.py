"""固定幾何的六類 JAX（可微陣列）組裝；不接材料參數、替身或搜尋。

常數只在 CPU 雙精度準備一次，扁平到每欄都可 np.asarray 後雜湊；用閉包交進 jit。
阻抗順序是 patch_ids；斜入射為 (F, sum(A_patch))，晚期為 (N,F_union,A_paris)。
共六個輸入：四路阻抗、聲壓 (聲源,座位,F_fem)、constant_material 布林 (N,)。
垂直入射阻抗分別為 (N,F_fem)、(N,F_report)。
product_total 用第一階段常數；objective_total 的低頻項原樣呼叫既有含代理入口。
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from aosr.physics import amplitude_jax as amplitude
from aosr.physics import geometric_early_jax as early
from aosr.physics.crossover_jax import eyring_t60_arrays, schroeder_frequency_arrays, crossover_weights_arrays, stitched_energy_arrays
from aosr.physics.late_decay_jax import LateDecayGeometry, late_decay_arrays
from aosr.physics.late_energy_jax import LateEnergyGeometry, ZonedLateEnergyGeometry
from aosr.physics.late_family_jax import constant_absorption_arrays, paris_absorption_arrays, late_family_arrays
from aosr.physics.reflection_energy_jax import reflection_path_energy
from aosr.scoring._six_category_constants import SixCategoryConstants as SixCategoryConstants
from aosr.scoring._six_category_build import build_constants as build_constants
from aosr.scoring.gradient_objective_jax import gradient_objective_from_pressures
from aosr.scoring.reflections_jax import ReflectionGeometry, reflection_cost_arrays, primary_category_cost
from aosr.scoring.reverberation_cost_jax import reverberation_cost_arrays
from aosr.scoring.stage_one_objective_jax import objective_from_pressures
from aosr.scoring.timbre_cost_jax import cost_arrays


class SixCategoryArrays(NamedTuple):
    """六類產品代價／旗標，搜尋低頻原值、兩種合分，另保留衰減與交接診斷。"""

    timbre_cost: jax.Array
    listening_cost: jax.Array
    reflections_cost: jax.Array
    reverberation_cost: jax.Array
    matching_cost: jax.Array
    low_frequency_cost: jax.Array
    timbre_valid: jax.Array
    listening_valid: jax.Array
    reflections_valid: jax.Array
    reverberation_valid: jax.Array
    matching_valid: jax.Array
    low_frequency_valid: jax.Array
    low_frequency_objective: jax.Array
    product_total: jax.Array
    objective_total: jax.Array
    reverberation_band_count: jax.Array
    reverberation_pair_count: jax.Array
    schroeder_frequency_hz: jax.Array
    capped_by_upper_limit: jax.Array


def path_table(c: SixCategoryConstants, index: int) -> amplitude.PathTable:
    start, end = (int(v) for v in c.path_offsets[index:index+2])
    angles = tuple(c.angles_deg[int(a):int(b)] for a, b in zip(c.angle_offsets[:-1], c.angle_offsets[1:], strict=True))
    cosines = tuple(c.cosines[int(a):int(b)] for a, b in zip(c.angle_offsets[:-1], c.angle_offsets[1:], strict=True))
    return amplitude.PathTable(c.frequencies_hz, c.order[start:end], c.dist_m[start:end], c.delay_s[start:end],
        c.bounce_wall[start:end], c.bounce_cos[start:end], c.bounce_mask[start:end], c.inverse_distance[start:end],
        c.phase[start:end], tuple(str(x) for x in c.patch_ids), c.bounce_patch[start:end], c.bounce_angle[start:end], cosines, angles)


def geometric_early_table(c: SixCategoryConstants) -> early.GeometricEarlyTable:
    count = c.direct_distances_m.size
    pairs = []
    for index in range(count):
        start, end = (int(v) for v in c.path_offsets[index:index+2])
        pairs.append(early.EarlyPairTable(path_table(c, index), c.directivity[start:end], c.furniture_pressure[index]))
    shape = (*c.direct_distances_m.shape, len(c.frequencies_hz))
    return early.GeometricEarlyTable(tuple(pairs), shape, c.patch_areas_m2, float(c.rho_c), int(c.reflection_order_k))


def _late(c: SixCategoryConstants, z: jax.Array, scattering: jax.Array, constant_material: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    constant = constant_absorption_arrays(z[..., 0], float(c.rho_c))
    angles = jnp.asarray(c.paris_angles_deg, dtype=z.real.dtype)
    # 正式角度生成器同一個 dtype 護欄：double 的 90 度前一格轉 single 會捨成 90。
    edge = jnp.nextafter(jnp.asarray(90., dtype=z.real.dtype), jnp.asarray(0., dtype=z.real.dtype))
    angles = jnp.minimum(angles, edge)
    paris = paris_absorption_arrays(z, angles, float(c.rho_c))
    alpha = jnp.where(constant_material[:, None], constant.values, paris.values)
    geometry = ZonedLateEnergyGeometry(LateEnergyGeometry(c.form_factors, c.late_patch_areas, c.late_wall_index, c.wall_areas),
                                      c.patch_weights, c.wall_weights)
    s = jnp.zeros(len(c.union_frequencies_hz), dtype=z.real.dtype).at[jnp.asarray(c.report_indices)].set(scattering)
    result = late_family_arrays(geometry, alpha, s, max_order=int(c.reflection_order_k),
        furniture_absorption_area_m2=jnp.asarray(c.furniture_absorption_area_m2, dtype=z.real.dtype) if bool(c.has_furniture) else None)
    return result.late_share_energy[jnp.asarray(c.report_indices)], result.alpha_by_wall, result.alpha_by_patch


def _reflections(c: SixCategoryConstants, rows: tuple[jax.Array, ...], scattering: jax.Array) -> jax.Array:
    channels = []
    offset = 0
    for channel, index in enumerate(c.primary_pair_indices):
        table = path_table(c, int(index))
        start, end = (int(v) for v in c.path_offsets[int(index):int(index)+2])
        z = amplitude.bounce_impedances_by_patch(table, rows)
        energy = reflection_path_energy(table, z, scattering, float(c.rho_c), jnp.asarray(c.directivity[start:end]))
        mask = np.concatenate((c.reflection_zone_mask[:, offset:offset+table.path_count], np.eye(4, dtype=np.bool_)), axis=1)
        geometry = ReflectionGeometry(table, mask, c.reflection_frequency_weights, c.reflection_thresholds_db,
                                      c.reflection_zone_weights, float(c.reflection_worse_reference_db), True)
        values = jnp.concatenate((energy.energy, jnp.asarray(c.reflection_furniture_energy[channel], dtype=energy.energy.dtype)), axis=0)
        channels.append(reflection_cost_arrays(geometry, values, jnp.asarray(c.reflection_coverage[channel])).channel_cost)
        offset += table.path_count
    return primary_category_cost(jnp.stack(channels))


def _reverberation(c: SixCategoryConstants, patches: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    decay = late_decay_arrays(LateDecayGeometry(c.form_factors, c.late_patch_areas, float(c.collision_frequency_hz)),
                              patches[:, jnp.asarray(c.decay_indices)])
    mask = jnp.asarray(c.decay_band_mask)
    valid = jnp.all(~mask | decay.valid, axis=1) & jnp.any(mask, axis=1)
    safe_t20 = jnp.where(decay.valid, decay.t20_s, 0.0)
    means = jnp.sum(jnp.where(mask, safe_t20, 0.0), axis=1)/jnp.maximum(jnp.sum(mask, axis=1), 1)
    result = reverberation_cost_arrays(c.reverb(), means, ~valid)
    return result.category_cost, result.band_count, result.pair_count


def category_totals(costs: jax.Array, valid: jax.Array, low_frequency_objective: jax.Array,
                    weights: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """雙層遮罩：先替掉不可用類再乘權重，最後才標 NaN；單獨可驗每一類旗標。"""
    safe = jnp.where(valid, costs, 0.0)
    product = jnp.sum(weights*safe)
    objective = jnp.sum(weights[:-1]*safe[:-1])+weights[-1]*jnp.where(valid[-1], low_frequency_objective, 0.0)
    return (jnp.where(valid, costs, jnp.nan), jnp.where(jnp.all(valid), product, jnp.nan),
            jnp.where(jnp.all(valid), objective, jnp.nan))


def six_category_costs(c: SixCategoryConstants, impedance_fem: jax.Array, impedance_normal: jax.Array,
                       impedance_early: jax.Array, impedance_paris: jax.Array, fem_pressures: jax.Array,
                       constant_material: jax.Array) -> SixCategoryArrays:
    """五個數值輸入須由呼叫端統一精度；constant_material 只有 ConstantImpedance（固定阻抗）時 True。

    貼構造或 MountedOnWall（原牆上貼構造）由呼叫端傳 False，採 Paris 積分；旗標可隨設計變。
    輸入順序必須照 c.patch_ids（這裡只守旗標長度，無法檢查材料身分）。
    旗標 True 的面片，impedance_paris[n,:,0] 必須是真 Z，且該面片整列阻抗有限。
    任一類無效則兩種合分 NaN。
    """
    if constant_material.shape != (len(c.patch_ids),) or constant_material.dtype != jnp.bool_:
        raise ValueError("constant_material 必須是逐面片的布林 (N,) 陣列")
    rows = tuple(impedance_early[:, int(a):int(b)] for a, b in zip(c.angle_offsets[:-1], c.angle_offsets[1:], strict=True))
    geo = early.solve_geometric_early(geometric_early_table(c), rows, impedance_normal, jnp.asarray(c.scattering_by_patch))
    scattering = geo.scattering[0, 0]
    late, walls, patches = _late(c, impedance_paris, scattering, constant_material)
    t60 = eyring_t60_arrays(c.wall_areas, float(c.volume_m3), walls[:, jnp.asarray(c.midband_indices)])
    fs = schroeder_frequency_arrays(float(c.volume_m3), t60.values)
    weights = crossover_weights_arrays(jnp.asarray(c.frequencies_hz, dtype=impedance_normal.real.dtype), fs.values)
    geometric = geo.direct_energy+geo.reflected_energy+geo.interference_energy+late[None, None, :]*jnp.asarray(c.source_power_ratio, dtype=late.dtype)[:, None, :]
    energy = jnp.abs(fem_pressures)**2
    padded = jnp.pad(energy, ((0, 0), (0, 0), (0, len(c.frequencies_hz)-len(c.fem_frequencies_hz))), constant_values=jnp.nan)
    stitched = stitched_energy_arrays(weights, padded, geometric)
    timbre = cost_arrays(c.timbre(), stitched.values, jnp.asarray(c.direct_distances_m, dtype=energy.dtype))
    reflection = _reflections(c, rows, scattering)
    reverb, bands, pairs = _reverberation(c, patches)
    pressure_valid = jnp.all(jnp.isfinite(fem_pressures) & (energy > 0.0)) & jnp.all(jnp.isfinite(impedance_fem))
    safe_pressure = jnp.where(pressure_valid, fem_pressures, jnp.ones_like(fem_pressures))
    pressures = jnp.transpose(safe_pressure, (2, 0, 1))
    product = objective_from_pressures(c.product(), pressures).screening_total
    objective = gradient_objective_from_pressures(c.objective(), pressures).objective_total
    raw = jnp.stack((timbre.category_costs[0], timbre.category_costs[1], reflection, reverb, timbre.category_costs[2], product))
    valid = jnp.isfinite(raw).at[-1].set(pressure_valid & jnp.isfinite(product) & jnp.isfinite(objective))
    values, product_total, objective_total = category_totals(raw, valid, objective, jnp.asarray(c.category_weights, dtype=energy.dtype))
    return SixCategoryArrays(values[0], values[1], values[2], values[3], values[4], values[5],
        valid[0], valid[1], valid[2], valid[3], valid[4], valid[5], jnp.where(valid[-1], objective, jnp.nan),
        product_total, objective_total,
        bands, pairs, fs.values, weights.capped_by_upper_limit)
