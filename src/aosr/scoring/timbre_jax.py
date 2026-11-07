"""固定頻率軸的音色共用零件；CPU 建矩陣，JAX 只收能量陣列。

能量形狀 (..., F)，精度跟輸入走。不碰全域設定；保留 numpy 的除最大能量。
頻率、設定的驗證在 CPU 準備階段完成；資料不可估則回逐曲線旗標與 NaN。
峰谷、對目標偏差不進產品類代價，這裡不重做那些診斷。
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from aosr.scoring.timbre import (
    TimbreSettings, _dependency_ranges, _dependency_ranges_have_gap, _in_range,
    _octave_cells_in_range, _octave_weights, _smoothing_half_width_octave,
    _weights_in_range, timbre_ripple_coverage_reason,
)

FloatArray = NDArray[np.float64]


class ResponseConstants(NamedTuple):
    """只含固定陣列，可整包傳入 jit；權重借 numpy 原實作，只有一份定義。"""

    octaves: FloatArray
    tilt_smoothing: FloatArray
    ripple_smoothing: FloatArray
    fit_weights: FloatArray
    ripple_weights: FloatArray
    matching_weights: FloatArray
    broadband_weights: FloatArray
    insufficient_coverage: NDArray[np.bool_]
    scoring_range_gap: NDArray[np.bool_]
    matching_coverage: NDArray[np.bool_]


class ResponseFlags(NamedTuple):
    """True 表示不可估；形狀跟曲線數相同，靜態覆蓋旗標是純量。"""

    non_positive_energy: jax.Array
    insufficient_coverage: jax.Array
    scoring_range_gap: jax.Array
    matching_coverage: jax.Array
    numerical_failure: jax.Array
    broadband_coverage: jax.Array


class ResponseArrays(NamedTuple):
    """每條曲線的傾斜、音色起伏、左右專用起伏與寬頻音量及旗標。"""

    tilt_db_per_octave: jax.Array
    residual_rms_db: jax.Array
    matching_rms_db: jax.Array
    broadband_level_db: jax.Array
    valid: jax.Array
    matching_valid: jax.Array
    flags: ResponseFlags


def smoothing_matrix(octaves: FloatArray, width_octave: float) -> FloatArray:
    """CPU 建每列一個平滑窗的常數矩陣，窗界與每八度權重沿用 numpy。"""
    if width_octave <= 0.0:
        return np.eye(len(octaves), dtype=np.float64)
    weights = _octave_weights(octaves)
    half = _smoothing_half_width_octave(width_octave)
    lower = np.searchsorted(octaves, octaves - half, side="left")
    upper = np.searchsorted(octaves, octaves + half, side="right")
    matrix = np.zeros((len(octaves), len(octaves)), dtype=np.float64)
    for row, (start, stop) in enumerate(zip(lower, upper, strict=True)):
        matrix[row, start:stop] = weights[start:stop] / np.sum(weights[start:stop])
    return matrix


def _range_weights(octaves: FloatArray, frequencies: FloatArray, bounds: tuple[float, float]) -> FloatArray:
    mask = _in_range(frequencies, bounds)
    weights = np.zeros_like(octaves)
    weights[mask] = _weights_in_range(octaves, mask, bounds)
    return weights


def response_constants(frequencies_hz: FloatArray, settings: TimbreSettings, *,
                       matching_range_hz: tuple[float, float], broadband_range_hz: tuple[float, float]
                       ) -> ResponseConstants:
    """CPU 準備；頻率軸已知且不隨材料改變。缺覆蓋仍建安全的常數並亮旗標。"""
    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    if frequencies.ndim != 1 or len(frequencies) < 2 or not np.all(np.isfinite(frequencies)):
        raise ValueError("CPU 頻率軸必須是一維、至少兩個有限正值")
    if not np.all(frequencies > 0.0) or not np.all(np.diff(frequencies) > 0.0):
        raise ValueError("CPU 頻率軸必須遞增且為正值")
    dependencies = _dependency_ranges(settings)
    bounds = (*dependencies, settings.tilt_fit_range_hz, settings.ripple_range_hz, settings.feature_range_hz)
    insufficient = any(np.count_nonzero(_in_range(frequencies, span)) < settings.min_points for span in bounds)
    gap = _dependency_ranges_have_gap(frequencies, dependencies, settings)
    octaves = np.log2(frequencies)
    return ResponseConstants(
        octaves, smoothing_matrix(octaves, settings.smoothing_width_octave_tilt),
        smoothing_matrix(octaves, settings.smoothing_width_octave_ripple),
        _range_weights(octaves, frequencies, settings.tilt_fit_range_hz),
        _range_weights(octaves, frequencies, settings.ripple_range_hz),
        _range_weights(octaves, frequencies, matching_range_hz),
        _octave_cells_in_range(frequencies, broadband_range_hz),
        np.asarray(insufficient), np.asarray(gap),
        np.asarray(timbre_ripple_coverage_reason(frequencies, matching_range_hz, settings) is not None),
    )


def _smooth_energy(energy: jax.Array, matrix: jax.Array) -> jax.Array:
    return jnp.matmul(energy, matrix.T, precision=jax.lax.Precision.HIGHEST)


def _fit_line(x: jax.Array, y: jax.Array, weights: jax.Array) -> tuple[jax.Array, jax.Array]:
    total = jnp.sum(weights)
    safe_total = jnp.where(total > 0.0, total, jnp.ones_like(total))
    x_mean = jnp.sum(weights * x) / safe_total
    y_mean = jnp.sum(weights * y, axis=-1) / safe_total
    centered = x - x_mean
    variance = jnp.sum(weights * centered * centered)
    safe_variance = jnp.where(variance > 0.0, variance, jnp.ones_like(variance))
    slope = jnp.sum(weights * centered * (y - y_mean[..., None]), axis=-1) / safe_variance
    return slope, y_mean - slope * x_mean


def _rms(values: jax.Array, weights: jax.Array) -> jax.Array:
    total = jnp.sum(weights)
    mean_square = jnp.sum(weights * values * values, axis=-1) / jnp.where(total > 0.0, total, 1.0)
    # 零曲線的範數取零次梯度；先換 sqrt 輸入，免得 0 * inf 變 NaN。
    positive = mean_square > 0.0
    return jnp.where(positive, jnp.sqrt(jnp.where(positive, mean_square, 1.0)), 0.0)


def _tilt_and_line(octaves: jax.Array, db: jax.Array, weights: jax.Array) -> tuple[jax.Array, jax.Array]:
    return _fit_line(octaves, db, weights)


def _ripple(octaves: jax.Array, db: jax.Array, line: tuple[jax.Array, jax.Array],
            weights: jax.Array) -> jax.Array:
    return _rms(db - (line[0][..., None] * octaves + line[1][..., None]), weights)


def _db(energy: jax.Array) -> jax.Array:
    safe = jnp.where(jnp.isfinite(energy) & (energy > 0.0), energy, 1.0)
    return 10.0 * jnp.log10(safe)


def _octave_mean_level_db(energy: jax.Array, weights: jax.Array) -> jax.Array:
    inside = weights > 0.0
    peak = jnp.max(jnp.where(inside, energy, 0.0), axis=-1)
    safe_peak = jnp.where(peak > 0.0, peak, 1.0)
    # 範圍外先歸零再除，極大範圍外能量不能產生 0 * inf。
    scaled = jnp.where(inside, energy, 0.0) / safe_peak[..., None]
    total = jnp.sum(weights)
    mean = jnp.sum(weights * scaled, axis=-1) / jnp.where(total > 0.0, total, 1.0)
    return _db(safe_peak) + _db(mean)


def response_arrays(constants: ResponseConstants, energies: jax.Array) -> ResponseArrays:
    """能量 (..., F) → 逐曲線四項量；編譯內不丟錯，無效支先換安全值。"""
    invalid = ~jnp.all(jnp.isfinite(energies) & (energies > 0.0), axis=-1)
    safe = jnp.where(invalid[..., None], jnp.ones_like(energies), energies)
    relative = safe / jnp.max(safe, axis=-1, keepdims=True)
    dtype = energies.dtype
    x = jnp.asarray(constants.octaves, dtype=dtype)
    tilt_energy = _smooth_energy(relative, jnp.asarray(constants.tilt_smoothing, dtype=dtype))
    ripple_energy = _smooth_energy(relative, jnp.asarray(constants.ripple_smoothing, dtype=dtype))
    numerical = ~jnp.all((tilt_energy > 0.0) & (ripple_energy > 0.0)
                         & jnp.isfinite(tilt_energy) & jnp.isfinite(ripple_energy), axis=-1)
    line = _tilt_and_line(x, _db(tilt_energy), jnp.asarray(constants.fit_weights, dtype=dtype))
    ripple_db = _db(ripple_energy)
    rms = _ripple(x, ripple_db, line, jnp.asarray(constants.ripple_weights, dtype=dtype))
    matching_rms = _ripple(x, ripple_db, line, jnp.asarray(constants.matching_weights, dtype=dtype))
    level = _octave_mean_level_db(safe, jnp.asarray(constants.broadband_weights, dtype=dtype))
    coverage, gap, matching_coverage = (jnp.asarray(flag) for flag in constants[-3:])
    valid = ~(invalid | coverage | gap | numerical)
    broadband_coverage = ~jnp.any(jnp.asarray(constants.broadband_weights) > 0.0)
    matching_valid = valid & ~matching_coverage & ~broadband_coverage
    return ResponseArrays(
        jnp.where(valid, line[0], jnp.nan), jnp.where(valid, rms, jnp.nan),
        jnp.where(matching_valid, matching_rms, jnp.nan), jnp.where(~invalid & ~broadband_coverage, level, jnp.nan),
        valid, matching_valid, ResponseFlags(invalid, coverage, gap, matching_coverage, numerical, broadband_coverage),
    )
