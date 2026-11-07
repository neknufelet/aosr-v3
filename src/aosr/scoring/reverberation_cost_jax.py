"""殘響逐帶區間與相鄰突變的 JAX 版；只平均可估的原頻帶、原配對。

數字全從正式品質登記簿讀進常數；陣列入口接 T20 與逐帶不可用旗標。
缺帶改變分母是第 11 條指定的離散行為，不跨缺帶重配相鄰項。
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from aosr.config.quality_targets import QualityPurpose
from aosr.scoring.cost_shapes import scalar, target
from aosr.scoring.reverberation_cost import (
    LOG_BASE_KEY,
    _number,
    _principal_weights,
    _setting,
    target_intervals,
)


class ReverberationCostSettings(NamedTuple):
    """編譯外讀好的逐帶界線、正規化尺度、門檻與類內權重。"""

    lower_s: NDArray[np.float64]
    upper_s: NDArray[np.float64]
    interval_scale_s: float
    jump_threshold: float
    jump_scale: float
    logarithm_base: float
    weights: NDArray[np.float64]


class ReverberationCostArrays(NamedTuple):
    """不可用項回 NaN；可用帶與配對數供上游報告分母跳變。"""

    category_cost: jax.Array
    interval_costs: jax.Array
    adjacent_costs: jax.Array
    interval_mean: jax.Array
    adjacent_mean: jax.Array
    band_count: jax.Array
    pair_count: jax.Array
    unavailable_bands: jax.Array
    unavailable_pairs: jax.Array
    unavailable: jax.Array


def reverberation_cost_settings(purpose: QualityPurpose, centers_hz: tuple[float, ...] | None = None) -> ReverberationCostSettings:
    """沿用正式逐帶區間與權重讀取；預設取登記簿全帶，也可指定遞增子集。"""
    intervals = target_intervals(purpose)
    centers = tuple(intervals) if centers_hz is None else centers_hz
    if not centers or tuple(sorted(set(centers))) != centers:
        raise ValueError("殘響頻帶必須非空且嚴格遞增")
    bounds = [intervals[center] for center in centers]
    interval = target(purpose, "reverberation.t20_interval_excess_s", "s")
    jump = target(purpose, "reverberation.adjacent_t20_log_ratio_excess", "1")
    if interval.cost_shape != "less_is_better" or jump.cost_shape != "beyond_threshold_only":
        raise ValueError("殘響代價形狀必須與正式 Python 相同")
    weights = _principal_weights(purpose)
    return ReverberationCostSettings(np.asarray([v[0] for v in bounds]), np.asarray([v[1] for v in bounds]),
                                     interval.worse_reference, scalar(jump), jump.worse_reference,
                                     _number(_setting(purpose, LOG_BASE_KEY, "1")),
                                     np.asarray([weights["t20_target_interval"], weights["adjacent_t20_jump"]]))


def _masked_mean(values: jax.Array, valid: jax.Array) -> tuple[jax.Array, jax.Array]:
    count = jnp.sum(valid)
    mean = jnp.sum(jnp.where(valid, values, 0.0)) / jnp.maximum(count, 1).astype(values.dtype)
    return mean, count


def reverberation_cost_arrays(settings: ReverberationCostSettings, t20_s: jax.Array,
                              unavailable: jax.Array) -> ReverberationCostArrays:
    """True 旗標、非有限及非正 T20 都排除；全無資料或無正權重時類代價 NaN。

    先把無效 T20 換成正有限數，再作對數、除法；最後才把無效輸出換 NaN。
    同時缺失的相鄰主項不參與類內權重分母，和 ``cost_reverberation_evaluation`` 相同。
    """
    values = jnp.asarray(t20_s)
    valid = ~jnp.asarray(unavailable) & jnp.isfinite(values) & (values > 0.0)
    safe = jnp.where(valid, values, 1.0)
    lower = jnp.asarray(settings.lower_s, dtype=values.dtype)
    upper = jnp.asarray(settings.upper_s, dtype=values.dtype)
    interval_scale = jnp.asarray(settings.interval_scale_s, dtype=values.dtype)
    interval = jnp.maximum(0.0, jnp.maximum(lower - safe, safe - upper)) / interval_scale
    pairs = valid[:-1] & valid[1:]
    base = jnp.asarray(settings.logarithm_base, dtype=values.dtype)
    signed = jnp.log(safe[1:] / safe[:-1]) / jnp.log(base)
    threshold = jnp.asarray(settings.jump_threshold, dtype=values.dtype)
    jump_scale = jnp.asarray(settings.jump_scale, dtype=values.dtype)
    adjacent = jnp.maximum(0.0, jnp.abs(signed) - threshold) / jump_scale
    interval_mean, band_count = _masked_mean(interval, valid)
    adjacent_mean, pair_count = _masked_mean(adjacent, pairs)
    exists = jnp.stack((band_count > 0, pair_count > 0))
    weights = jnp.asarray(settings.weights, dtype=values.dtype)
    denominator = jnp.sum(jnp.where(exists, weights, 0.0))
    available = denominator > 0.0
    means = jnp.stack((interval_mean, adjacent_mean))
    numerator = jnp.sum(jnp.where(exists, means * weights, 0.0))
    category = numerator / jnp.where(available, denominator, 1.0)
    return ReverberationCostArrays(jnp.where(available, category, jnp.nan),
                                   jnp.where(valid, interval, jnp.nan), jnp.where(pairs, adjacent, jnp.nan),
                                   jnp.where(band_count > 0, interval_mean, jnp.nan),
                                   jnp.where(pair_count > 0, adjacent_mean, jnp.nan), band_count, pair_count,
                                   ~valid, ~pairs, ~available)
