"""早期反射四區超標的 JAX（可微陣列）等價版；不接搜尋、不算顫動或左右差（#562 2c）。

幾何表、到達方向、窗與八度權重在中央處理器雙精度固定；動態輸入是陣列。
每區直接 jnp.max，零反射仍參與四區平均；兩支主位缺任一支時類代價回 NaN。
方向排序固定為 DirectionZone 列舉順序。旗標 True 表示可用，沒有改動全域精度設定。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from aosr.config.frequency_axis import octave_cells_in_range
from aosr.config.quality_targets import QualityPurpose, SettingEntry
from aosr.physics.amplitude_jax import PathTable
from aosr.physics.reflection_energy_jax import ReflectionEnergyArrays, reflection_path_energy
from aosr.scoring.cost_shapes import target
from aosr.scoring.direction_zones import DirectionZone, classify, listening_angles, zone_limits
from aosr.scoring.reflections_cost import _weights, _zone_threshold


ReflectionCoverage = Literal["complete", "approximate", "not_provable", "missing"]
EstimableCoverage = Annotated[jax.Array, "布林陣列 (...)；complete 或 approximate 為 True，其餘為 False"]


def coverage_is_estimable(coverage: ReflectionCoverage) -> bool:
    """編譯外把 numpy 覆蓋狀態轉布林；家具 approximate（近似）同樣可估。

    呼叫端將回傳值轉 jnp.asarray 後傳入 coverage_complete；字串不進 jit（編譯）。
    """
    return coverage in ("complete", "approximate")


@dataclass(frozen=True, eq=False)
class ReflectionGeometry:
    """預建常數；用閉包帶進 jit（編譯），不把這個 Python 物件當追蹤參數。"""

    paths: PathTable
    zone_mask: NDArray[np.bool_]
    frequency_weights: NDArray[np.float64]
    thresholds_db: NDArray[np.float64]
    zone_weights: NDArray[np.float64]
    worse_reference_db: float
    frequency_coverage: bool


class ReflectionCostArrays(NamedTuple):
    """逐區超標 dB 與代價 (...,4)、單聲道代價 (...)，以及逐區有效旗標 (...,4)。"""

    zone_excess_db: jax.Array
    zone_cost: jax.Array
    channel_cost: jax.Array
    zone_valid: jax.Array


class ReflectionArrays(NamedTuple):
    path_energy: jax.Array
    zone_excess_db: jax.Array
    zone_cost: jax.Array
    channel_cost: jax.Array
    zone_valid: jax.Array
    energy_flags: ReflectionEnergyArrays


def _setting(purpose: QualityPurpose, key: str) -> SettingEntry:
    entry = purpose.entry("reflections_and_echo." + key)
    if not isinstance(entry, SettingEntry):
        raise TypeError(f"{key} 必須是量法設定")
    return entry


def reflection_geometry(
    table: PathTable, direction_vectors: NDArray[np.float64], axis_xy: tuple[float, float], purpose: QualityPurpose,
) -> ReflectionGeometry:
    """接上游完整路徑表的陣列，預建窗與方向遮罩；coverage_complete 由上游窗外證明另傳。

    路徑表須含唯一的直達；只有反射且相對延遲 <= 登記時間窗才入區。
    本函式不選材料、不依能量篩路徑；表內可以保留窗外路徑，計分時完全忽略。
    complete（完整）及 approximate（家具近似）皆可估，交由 coverage_is_estimable 轉旗標。
    """
    direct = table.direct_index()
    if direction_vectors.shape != (table.path_count, 3):
        raise ValueError("到達方向形狀必須是 (P,3)")
    window = _setting(purpose, "window_upper_ms").value
    bounds = _setting(purpose, "frequency_range_hz").value
    if isinstance(window, tuple) or not isinstance(bounds, tuple) or len(bounds) != 2:
        raise ValueError("時間窗與頻率範圍的登記型別錯誤")
    limits, _ = zone_limits(purpose)
    zones = tuple(classify(*listening_angles(tuple(vector), axis_xy), limits) for vector in direction_vectors)
    inside = (table.order > 0) & (table.delay_s - table.delay_s[direct] <= float(window) / 1000.0)
    mask = np.asarray([[inside[i] and zone == wanted for i, zone in enumerate(zones)] for wanted in DirectionZone])
    weights = np.asarray(octave_cells_in_range(tuple(table.frequencies_hz), (bounds[0], bounds[1])), dtype=np.float64)
    goal = target(purpose, "reflections_and_echo.zone_excess_db", "dB")
    if goal.cost_shape != "less_is_better":
        raise ValueError("反射超標形狀必須是 less_is_better")
    zone_weights = _weights(purpose)
    coverage = table.frequencies_hz[0] <= bounds[0] and table.frequencies_hz[-1] >= bounds[1] and np.any(weights > 0)
    return ReflectionGeometry(table, mask, weights, np.asarray([_zone_threshold(purpose, z) for z in DirectionZone]),
                              np.asarray([zone_weights[z] for z in DirectionZone]), goal.worse_reference, bool(coverage))


def reflection_cost_arrays(
    geometry: ReflectionGeometry, path_energy: jax.Array, coverage_complete: EstimableCoverage,
) -> ReflectionCostArrays:
    """能量 (...,P,F) → 四區超標；非法格只污染有用到它的區，空區與全零能量都合法回零。

    coverage_complete 是布林陣列 (...)，表示可估：complete 與 approximate 都傳 True；
    not_provable（未證明）或 missing（缺席）傳 False，由 coverage_is_estimable 在編譯外轉換。
    """
    energy = jnp.asarray(path_energy)
    dtype = energy.dtype
    finite = jnp.isfinite(energy) & (energy >= 0)
    safe_energy = jnp.where(finite, energy, jnp.zeros((), dtype=dtype))
    mask = jnp.asarray(geometry.zone_mask)[:, :, None]
    weights = jnp.asarray(geometry.frequency_weights, dtype=dtype)
    # 窗外、直達或範圍外的壞資料不能使別的有效區變不可估。
    used = mask & (weights > 0)
    valid = jnp.all(~used | finite[..., None, :, :], axis=(-2, -1))
    valid = valid & jnp.asarray(coverage_complete)[..., None] & geometry.frequency_coverage
    strongest = jnp.max(jnp.where(mask, safe_energy[..., None, :, :], jnp.zeros((), dtype=dtype)), axis=-2)
    positive = strongest > 0
    safe_peak = jnp.where(positive, strongest, jnp.ones((), dtype=dtype))
    level = 10.0 * jnp.log10(safe_peak)
    thresholds = jnp.asarray(geometry.thresholds_db, dtype=dtype)[:, None]
    excess = jnp.where(positive, jnp.maximum(level - thresholds, 0), jnp.zeros((), dtype=dtype))
    denominator = jnp.sum(weights)
    safe_denominator = jnp.where(denominator > 0, denominator, jnp.ones((), dtype=dtype))
    zone_excess = jnp.sum(excess * weights, axis=-1) / safe_denominator
    zone_cost = zone_excess / jnp.asarray(geometry.worse_reference_db, dtype=dtype)
    zone_weights = jnp.asarray(geometry.zone_weights, dtype=dtype)
    channel_cost = jnp.sum(zone_cost * zone_weights, axis=-1) / jnp.sum(zone_weights)
    nan = jnp.asarray(jnp.nan, dtype=dtype)
    return ReflectionCostArrays(jnp.where(valid, zone_excess, nan), jnp.where(valid, zone_cost, nan),
                                jnp.where(jnp.all(valid, axis=-1), channel_cost, nan), valid)


def reflections_arrays(
    geometry: ReflectionGeometry, z_bounce: jax.Array, scattering: jax.Array, rho_c: float,
    coverage_complete: EstimableCoverage, pressure_factor: jax.Array | None = None,
) -> ReflectionArrays:
    """逐反彈阻抗與散射 → 逐路徑能量與單聲道四區代價；不接材料鏈。

    coverage_complete 是可估布林陣列 (...)：complete（完整）與 approximate（家具近似）
    都為 True；呼叫端可先以 coverage_is_estimable 轉換，其他狀態為 False 並回 NaN。
    """
    energy = reflection_path_energy(geometry.paths, z_bounce, scattering, rho_c, pressure_factor)
    cost = reflection_cost_arrays(geometry, energy.energy, coverage_complete)
    return ReflectionArrays(energy.energy, *cost[:2], cost.channel_cost, cost.zone_valid, energy)


def primary_category_cost(channel_costs: jax.Array) -> jax.Array:
    """上游只交兩支主位 (...,2)；任一支不可用，整類 NaN，不拿另一支湊平均。"""
    valid = jnp.isfinite(channel_costs)
    safe = jnp.where(valid, channel_costs, jnp.zeros((), dtype=channel_costs.dtype))
    return jnp.where(jnp.all(valid, axis=-1), jnp.mean(safe, axis=-1), jnp.asarray(jnp.nan, dtype=channel_costs.dtype))


# 組裝器讀同一份反射量法設定；原私名保留相容。
reflection_setting = _setting
