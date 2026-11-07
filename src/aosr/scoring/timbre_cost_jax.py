"""音色、聆聽區、左右匹配的陣列代價（#562 第 2b 段）。

輸入 (聲道, 座位, 頻率) 的正式逐點總能量與 (聲道, 座位) 直達距離。
CPU 準備常數後可 jit／vmap；不接材料鏈、搜尋或梯度目標。
分項順序見各輸出欄說明；保護項照算、照 numpy 權重只供診斷。
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from aosr.config.quality_targets import TargetEntry, load_quality_targets
from aosr.scoring.channel_group import ChannelGroup
from aosr.scoring.channel_matching_cost import _targets, _weights
from aosr.scoring.channel_matching_settings import _load_settings
from aosr.scoring.listening_area_cost import (
    _LISTENING_AREA_TARGET_KEYS, _LISTENING_AREA_TARGET_UNITS,
    _listening_area_principal_weights,
)
from aosr.scoring.cost_shapes import target
from aosr.scoring.receiver_set import ReceiverRole, ReceiverSet
from aosr.scoring.timbre_cost import TIMBRE_TARGET_UNITS, principal_weights
from aosr.scoring.timbre_jax import ResponseArrays, ResponseConstants, response_arrays, response_constants

_MATCHING_METRICS = ("tilt_difference", "ripple_rms_difference", "broadband_level_difference", "direct_time_difference")
_AREA_METRICS = ("tilt", "ripple_rms", "overall_level")


class ShapeConstants(NamedTuple):
    """三種既有形狀的數值參數；所有產品數字由 TargetEntry 提供。"""

    center: NDArray[np.float64]
    tolerance: NDArray[np.float64]
    worse_reference: NDArray[np.float64]
    linear: NDArray[np.bool_]


def shape_constants(entries: tuple[TargetEntry, ...]) -> ShapeConstants:
    """CPU 登記簿 → 廣播陣列；維持 cost_shapes.shape_cost 的三種形狀。"""
    centers, tolerances = [], []
    for entry in entries:
        if isinstance(entry.value, tuple):
            raise ValueError("這三類的代價目標必須是單值")
        centers.append(entry.value if entry.cost_shape == "in_range_best" else 0.0)
        tolerances.append(entry.value if entry.cost_shape == "beyond_threshold_only" else (entry.tolerance or 0.0))
    return ShapeConstants(np.asarray(centers, dtype=np.float64), np.asarray(tolerances, dtype=np.float64),
                          np.asarray([entry.worse_reference for entry in entries], dtype=np.float64),
                          np.asarray([entry.cost_shape == "less_is_better" for entry in entries]))


def shape_cost(constants: ShapeConstants, values: jax.Array) -> jax.Array:
    """標準代價；範圍型容許帶內恰為零，沒有軟化。"""
    center, tolerance, worse = (jnp.asarray(value, dtype=values.dtype) for value in constants[:3])
    outside = jnp.abs(values - center) - tolerance
    excess = jnp.where(outside > 0.0, outside, 0.0)
    return jnp.where(jnp.asarray(constants.linear), values, excess) / worse


class PairConstants(NamedTuple):
    indices: NDArray[np.int64]
    weights: NDArray[np.float64]


class CostConstants(NamedTuple):
    """頻率、聲道、座位、目標與權重的固定陣列；跟輸入能量分開準備。"""

    responses: ResponseConstants
    input_layout: NDArray[np.bool_]
    primary: NDArray[np.int64]
    measured: NDArray[np.int64]
    primary_pairs: PairConstants
    peer_pairs: PairConstants
    channel_pairs: NDArray[np.int64]
    importance: NDArray[np.float64]
    timbre_shapes: ShapeConstants
    listening_shapes: ShapeConstants
    matching_shapes: ShapeConstants
    timbre_weights: NDArray[np.float64]
    listening_weights: NDArray[np.float64]
    matching_weights: NDArray[np.float64]
    sound_speed_m_s: NDArray[np.float64]


def _seat_pairs(receivers: ReceiverSet) -> tuple[PairConstants, PairConstants]:
    points = receivers.points
    primary = points.index(receivers.primary)
    around = sorted((i for i, point in enumerate(points) if point.role == ReceiverRole.SURROUNDING),
                    key=lambda i: points[i].receiver_id)
    peer_indices = tuple(combinations(around, 2))
    return (PairConstants(np.asarray([(primary, i) for i in around], dtype=np.int64).reshape(-1, 2),
                          np.asarray([points[i].importance for i in around], dtype=np.float64)),
            PairConstants(np.asarray(peer_indices, dtype=np.int64).reshape(-1, 2),
                          np.asarray([points[i].importance * points[j].importance for i, j in peer_indices], dtype=np.float64)))


def cost_constants(frequencies_hz: NDArray[np.float64], receivers: ReceiverSet, group: ChannelGroup, *,
                   quality_targets_path: str | Path, purpose: str, sound_speed_m_s: float) -> CostConstants:
    """CPU 準備；陣列聲道／座位順序就是 group.channels／receivers.points 的順序。"""
    settings = _load_settings(quality_targets_path, purpose)
    registry = load_quality_targets(quality_targets_path).purpose(purpose)
    timbre_weights = principal_weights(registry)
    listening_weights = _listening_area_principal_weights(registry)
    matching_weights = _weights(registry)
    timbre_keys = ("timbre_balance.target_tilt_db_per_octave", "timbre_balance.residual_rms_db")
    area_keys = tuple(f"{metric}_{suffix}" for suffix in ("weighted_mean_deviation", "worst_deviation") for metric in _AREA_METRICS)
    timbre_entries = tuple(target(registry, key, TIMBRE_TARGET_UNITS[key]) for key in timbre_keys)
    area_entries = tuple(target(registry, _LISTENING_AREA_TARGET_KEYS[name],
                                _LISTENING_AREA_TARGET_UNITS[_LISTENING_AREA_TARGET_KEYS[name]]) for name in area_keys)
    matching_entries = tuple(_targets(registry, metric)[i] for i in range(2) for metric in _MATCHING_METRICS)
    roles = [channel.role for channel in group.channels]
    primary_pairs, peers = _seat_pairs(receivers)
    measured = np.asarray([i for i, point in enumerate(receivers.points) if point.role != ReceiverRole.OTHER_SEAT], dtype=np.int64)
    return CostConstants(
        response_constants(frequencies_hz, settings.timbre, matching_range_hz=settings.ripple_range_hz,
                           broadband_range_hz=settings.broadband_range_hz),
        np.zeros((len(group.channels), len(receivers.points)), dtype=np.bool_),
        np.asarray(receivers.points.index(receivers.primary), dtype=np.int64), measured,
        primary_pairs, peers,
        np.asarray([(roles.index(pair.left_role), roles.index(pair.right_role)) for pair in group.comparisons], dtype=np.int64).reshape(-1, 2),
        np.asarray([receivers.points[i].importance for i in measured], dtype=np.float64),
        shape_constants(timbre_entries), shape_constants(area_entries), shape_constants(matching_entries),
        np.asarray([timbre_weights[name] for name in ("tilt", "residual_rms")], dtype=np.float64),
        np.asarray([listening_weights[f"{metric}_weighted_mean_deviation"] for metric in _AREA_METRICS], dtype=np.float64),
        np.asarray([matching_weights[name] if name != "direct_time_difference" or settings.direct_time_cost_enabled else 0.0
                    for name in _MATCHING_METRICS], dtype=np.float64), np.asarray(sound_speed_m_s, dtype=np.float64),
    )


class CostFlags(NamedTuple):
    """True 表示無法計算；缺周圍彼此配對只是診斷，與 numpy 同義。"""

    listening_unavailable: jax.Array
    zero_pair_importance: jax.Array
    peer_group_missing: jax.Array
    invalid_direct_distance: jax.Array
    matching_point_unavailable: jax.Array
    zero_matching_importance: jax.Array
    matching_unavailable: jax.Array
    invalid_sound_speed: jax.Array
    zero_matching_weight: jax.Array
    invalid_input_shape: jax.Array


class CostArrays(NamedTuple):
    """三類順序：音色／聆聽區／左右匹配。

    音色分項 (C,2)＝傾斜、起伏；聆聽區 (C,6)＝三個平均、三個最差。
    左右分項 (2,4)＝平均／最差 × 傾斜、起伏、音量、直達時間；停用直達時該欄為零。
    逐點左右差 (P,S,4) 是帶正負的診斷；任一必要點不可估，整類回 NaN。
    """

    category_costs: jax.Array
    timbre_components: jax.Array
    listening_components: jax.Array
    matching_components: jax.Array
    matching_point_differences: jax.Array
    responses: ResponseArrays
    flags: CostFlags


def _group_deviations(values: jax.Array, pairs: PairConstants) -> tuple[jax.Array, jax.Array, jax.Array]:
    indices = jnp.asarray(pairs.indices)
    weights = jnp.asarray(pairs.weights, dtype=values.dtype)
    deviations = jnp.abs(values[:, indices[:, 0], :] - values[:, indices[:, 1], :])
    total = jnp.sum(weights)
    mean = jnp.sum(deviations * weights[None, :, None], axis=1) / jnp.where(total > 0.0, total, 1.0)
    return mean, jnp.max(deviations, axis=1), total <= 0.0


def _listening(constants: CostConstants, response: ResponseArrays) -> tuple[jax.Array, jax.Array, jax.Array]:
    valid = response.valid & ~response.flags.broadband_coverage
    values = jnp.stack((response.tilt_db_per_octave, response.residual_rms_db, response.broadband_level_db), axis=-1)
    safe = jnp.where(valid[..., None], values, 0.0)
    mean, worst, zero = _group_deviations(safe, constants.primary_pairs)
    group_values = [jnp.concatenate((mean, worst), axis=-1)]
    zero_flags = [zero]
    if constants.peer_pairs.indices.shape[0]:
        peer_mean, peer_worst, peer_zero = _group_deviations(safe, constants.peer_pairs)
        group_values.append(jnp.concatenate((peer_mean, peer_worst), axis=-1))
        zero_flags.append(peer_zero)
    components = jnp.max(shape_cost(constants.listening_shapes, jnp.stack(group_values)), axis=0)
    zero_array = jnp.stack(zero_flags)
    unavailable = ~jnp.all(valid[:, jnp.asarray(constants.measured)], axis=1) | jnp.any(zero_array)
    return jnp.where(unavailable[:, None], jnp.nan, components), unavailable, zero_array


def _matching(constants: CostConstants, response: ResponseArrays, distances: jax.Array
              ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    dtype = response.tilt_db_per_octave.dtype
    invalid_distance = ~jnp.isfinite(distances) | (distances < 0.0)
    valid = response.matching_valid & ~invalid_distance
    metrics = jnp.stack((response.tilt_db_per_octave, response.matching_rms_db, response.broadband_level_db), axis=-1)
    safe_metrics = jnp.where(valid[..., None], metrics, 0.0)
    safe_distance = jnp.where(invalid_distance, 0.0, distances).astype(dtype)
    speed = jnp.asarray(constants.sound_speed_m_s, dtype=dtype)
    speed_valid = jnp.isfinite(speed) & (speed > 0.0)
    time = safe_distance / jnp.where(speed_valid, speed, 1.0) * 1000.0
    metrics = jnp.concatenate((safe_metrics, time[..., None]), axis=-1)[:, jnp.asarray(constants.measured), :]
    valid = valid[:, jnp.asarray(constants.measured)]
    pairs = jnp.asarray(constants.channel_pairs)
    difference = metrics[pairs[:, 0]] - metrics[pairs[:, 1]]
    point_valid = valid[pairs[:, 0]] & valid[pairs[:, 1]]
    deviations = jnp.where(point_valid[..., None], jnp.abs(difference), 0.0)
    weights = jnp.asarray(constants.importance, dtype=dtype)
    total = jnp.sum(jnp.where(point_valid, weights, 0.0), axis=-1)
    mean = jnp.sum(deviations * weights[None, :, None], axis=1) / jnp.where(total > 0.0, total, 1.0)[:, None]
    worst = jnp.max(deviations, axis=1)
    pair_values = jnp.concatenate((mean, worst), axis=-1)
    shaped = shape_cost(constants.matching_shapes, pair_values)
    assessed = jnp.any(point_valid, axis=-1)
    count = jnp.sum(assessed)
    mean_costs = jnp.sum(jnp.where(assessed[:, None], shaped[:, :len(_MATCHING_METRICS)], 0.0), axis=0) / jnp.where(count > 0, count, 1)
    worst_costs = jnp.max(jnp.where(assessed[:, None], shaped[:, len(_MATCHING_METRICS):], 0.0), axis=0)
    zero = assessed & (total <= 0.0)
    unavailable = (count == 0) | jnp.any(zero) | jnp.any(~point_valid) | ~speed_valid
    components = jnp.stack((mean_costs, worst_costs))
    active = jnp.asarray(constants.matching_weights) > 0.0
    components = jnp.where(active, components, 0.0)
    return (jnp.where(unavailable, jnp.nan, components),
            jnp.where(point_valid[..., None], difference, jnp.nan), ~point_valid, zero, unavailable)


def _cost_arrays(constants: CostConstants, energies: jax.Array, direct_distances_m: jax.Array) -> CostArrays:
    """正式逐點能量 → 三類陣列代價；有效項的梯度不經無效項的 NaN。"""
    response = response_arrays(constants.responses, energies)
    primary = jnp.asarray(constants.primary)
    valid = response.valid[:, primary]
    values = jnp.stack((response.tilt_db_per_octave[:, primary], response.residual_rms_db[:, primary]), axis=-1)
    timbre = shape_cost(constants.timbre_shapes, jnp.where(valid[:, None], values, 0.0))
    timbre = jnp.where(valid[:, None], timbre, jnp.nan)
    listening, unavailable, zero = _listening(constants, response)
    if constants.channel_pairs.shape[0]:
        matching, differences, point_unavailable, matching_zero, matching_unavailable = _matching(constants, response, direct_distances_m)
    else:
        matching = jnp.full((2, len(_MATCHING_METRICS)), jnp.nan, dtype=energies.dtype)
        differences = jnp.empty((0, len(constants.measured), len(_MATCHING_METRICS)), dtype=energies.dtype)
        point_unavailable = jnp.ones((0, len(constants.measured)), dtype=jnp.bool_)
        matching_zero, matching_unavailable = jnp.zeros((0,), dtype=jnp.bool_), jnp.asarray(True)
    timbre_cost = jnp.mean(jnp.sum(timbre * jnp.asarray(constants.timbre_weights, dtype=energies.dtype), axis=-1))
    area_cost = jnp.mean(jnp.sum(listening[:, :len(_AREA_METRICS)] * jnp.asarray(constants.listening_weights, dtype=energies.dtype), axis=-1))
    weights = jnp.asarray(constants.matching_weights, dtype=energies.dtype)
    total_weight = jnp.sum(weights)
    matching_cost = jnp.sum(matching[0] * weights) / jnp.where(total_weight > 0.0, total_weight, 1.0)
    speed = jnp.asarray(constants.sound_speed_m_s)
    invalid_speed = ~jnp.isfinite(speed) | (speed <= 0.0)
    matching_cost = jnp.where((total_weight > 0.0) & ~invalid_speed, matching_cost, jnp.nan)
    measured_seats = jnp.zeros(direct_distances_m.shape[1], dtype=jnp.bool_).at[jnp.asarray(constants.measured)].set(True)
    flags = CostFlags(unavailable, zero, jnp.asarray(constants.peer_pairs.indices.shape[0] == 0),
                      measured_seats & (~jnp.isfinite(direct_distances_m) | (direct_distances_m < 0.0)),
                      point_unavailable, matching_zero, matching_unavailable | invalid_speed | (total_weight <= 0.0),
                      invalid_speed, total_weight <= 0.0, jnp.asarray(False))
    return CostArrays(jnp.stack((timbre_cost, area_cost, matching_cost)), timbre, listening,
                      matching, differences, response, flags)


def _invalid_shape(constants: CostConstants, energies: jax.Array) -> CostArrays:
    shape = (*constants.input_layout.shape, constants.responses.octaves.shape[0])
    safe = _cost_arrays(constants, jnp.ones(shape, dtype=energies.dtype),
                        jnp.ones(constants.input_layout.shape, dtype=energies.dtype))
    response = safe.responses
    response = response._replace(
        tilt_db_per_octave=jnp.full_like(response.tilt_db_per_octave, jnp.nan),
        residual_rms_db=jnp.full_like(response.residual_rms_db, jnp.nan),
        matching_rms_db=jnp.full_like(response.matching_rms_db, jnp.nan),
        broadband_level_db=jnp.full_like(response.broadband_level_db, jnp.nan),
        valid=jnp.zeros_like(response.valid), matching_valid=jnp.zeros_like(response.matching_valid),
    )
    flags = safe.flags._replace(invalid_input_shape=jnp.asarray(True),
                                listening_unavailable=jnp.ones_like(safe.flags.listening_unavailable),
                                matching_point_unavailable=jnp.ones_like(safe.flags.matching_point_unavailable),
                                matching_unavailable=jnp.asarray(True))
    return safe._replace(category_costs=jnp.full_like(safe.category_costs, jnp.nan),
                          timbre_components=jnp.full_like(safe.timbre_components, jnp.nan),
                          listening_components=jnp.full_like(safe.listening_components, jnp.nan),
                          matching_components=jnp.full_like(safe.matching_components, jnp.nan),
                          matching_point_differences=jnp.full_like(safe.matching_point_differences, jnp.nan),
                          responses=response, flags=flags)


def cost_arrays(constants: CostConstants, energies: jax.Array, direct_distances_m: jax.Array) -> CostArrays:
    """正式逐點能量 → 三類代價。形狀不符也在 jit 內回旗標與 NaN，不做越界 gather。"""
    expected = (*constants.input_layout.shape, constants.responses.octaves.shape[0])
    if energies.shape != expected or direct_distances_m.shape != constants.input_layout.shape:
        return _invalid_shape(constants, energies)
    return _cost_arrays(constants, energies, direct_distances_m)
