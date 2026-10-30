"""第三階段梯度目標的 JAX 版（#561 壬一）：第一階段合分不動，加上權重 × 大峰的平滑代理。

老闆 2026-10-30 拍 B：「選 B，是因為它是最小、能直接驗證目前問題的實驗。」第一階段兩分項合分（``screening_total``）
仍由 :func:`aosr.scoring.stage_one_objective_jax.objective_from_pressures` 算、一位都不動；這一支只在它旁邊多算一個
代理、再加起來，舊合分與新合分一起交回，呼叫端（梯度搜尋的帳本、結果、終點、報告）兩個都記。

數學定義（考卷逐條守）：
- 殘差 r_s(i)：喇叭 s 主位的 dB 曲線扣自己的加權最小平方直線，計分範圍內的點；就是第一階段主項均方根底下那個量
  （:func:`~aosr.scoring.stage_one_objective_jax.primary_residual_db`，同一份算式）。
- 內部點：計分範圍內去掉第一與最後一個頻點。峰谷偵測（``timbre._features``）的 ``range(1, len-1)`` 兩端點不算
  局部極值；代理若看得到端點，會在端點最大時報一個偵測器永遠不報的峰。
- 代理 P ＝ T · log Σ_{s,i} exp(r_s(i) / T)，加總跨全部喇叭與全部內部點（一個軟最大值，對應終點各支取較大的
  ``worst_big_peak``），不加頻率權重（偵測器也不加）。用 ``jax.nn.logsumexp`` 的穩定寫法。
- 夾擠的對象是**內部最大殘差**（``interior_max_residual_db``，同一個殘差、同一組內部點、左右取較大）：
  max r_內部 ≤ P ≤ max r_內部 ＋ T·ln N，N＝內部點數 × 喇叭數（:func:`proxy_gap_bound_db`）。
- 對偵測器（``timbre._features`` 的最高峰、終點的真大峰）：真大峰 ≤ max r_內部 ≤ P 恆成立，代理不會低估真大峰；
  只有內部最大值是嚴格局部極大時，真大峰才等於 max r_內部、P − 真大峰 才落在 0～上界。反例（審查 A）：內部最大值
  緊貼更高的端點（肩部），它不是局部極大，偵測器報的是別處較小的峰，代理比真大峰高很多。
- 代理可以是負的：殘差加權平均為 0 是含兩端點的，端點都高時內部殘差可以全是負的，P 跟著是負的、新合分低於舊合分。
  照拍板公式**不截零**。
- 新合分：objective_total ＝ screening_total ＋ w · P ／ ref；T、w、ref 都從登記簿那三格讀
  （:mod:`aosr.scoring.gradient_objective_settings`），這裡不寫數字。

精度跟輸入走：常數在運算前轉成輸入的精度，不改全域設定。完整正能量與有限複數聲壓由呼叫端保證。
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from aosr.scoring.gradient_objective_settings import GradientObjectiveSettings
from aosr.scoring.receiver_set import ReceiverSet
from aosr.scoring.stage_one_objective_jax import (
    StageOneObjectiveArrays, StageOneObjectiveConstants, objective_constants, objective_from_energies,
    objective_from_pressures, primary_residual_db,
)


class GradientObjectiveConstants(NamedTuple):
    """可直接交進編譯的樹狀資料；前八欄與 :class:`StageOneObjectiveConstants` 同名同值（扁平抄進來，
    帳本的評分常數指紋照欄位逐一雜湊，不收巢狀）。

    ``interior_positions`` 是計分範圍內那一串頻點裡的位置（去掉第一與最後一個），三個純量是代理的溫度（dB）、
    權重與較差參考（dB）。
    """

    log2_frequencies: NDArray[np.float64]
    frequency_indices: NDArray[np.int64]
    frequency_weights: NDArray[np.float64]
    primary_index: NDArray[np.int64]
    surrounding_indices: NDArray[np.int64]
    speaker_indices: NDArray[np.int64]
    component_weights: NDArray[np.float64]
    worse_references: NDArray[np.float64]
    interior_positions: NDArray[np.int64]
    proxy_temperature_db: NDArray[np.float64]
    proxy_weight: NDArray[np.float64]
    proxy_worse_reference: NDArray[np.float64]

    def stage_one(self) -> StageOneObjectiveConstants:
        """第一階段那一份常數（原樣交給 ``objective_from_*``）。"""
        return StageOneObjectiveConstants(
            log2_frequencies=self.log2_frequencies, frequency_indices=self.frequency_indices,
            frequency_weights=self.frequency_weights, primary_index=self.primary_index,
            surrounding_indices=self.surrounding_indices, speaker_indices=self.speaker_indices,
            component_weights=self.component_weights, worse_references=self.worse_references)


class GradientObjectiveArrays(NamedTuple):
    """前五欄同 :class:`StageOneObjectiveArrays`（逐喇叭兩項、聲道平均、第一階段合分）；
    ``big_peak_proxy_db`` 是代理（純量，dB），``objective_total`` 是新合分，``interior_max_residual_db`` 是代理夾擠的
    對象（內部最大殘差，純量，dB；報告與終點用，梯度不走它）；都是與輸入同精度的陣列。"""

    own_line_ripple_rms_db: jax.Array
    seat_robustness_rms_db: jax.Array
    mean_own_line_ripple_rms_db: jax.Array
    mean_seat_robustness_rms_db: jax.Array
    screening_total: jax.Array
    big_peak_proxy_db: jax.Array
    objective_total: jax.Array
    interior_max_residual_db: jax.Array


class ObjectiveReading(NamedTuple):
    """一次求值讀成 Python 浮點：新合分、舊合分（第一階段兩分項合分）、三分項（主位起伏、座位穩健性的聲道平均、代理）。"""

    objective_total: float
    screening_total: float
    components: tuple[float, float, float]


def gradient_objective_constants(
    frequencies_hz: Sequence[float] | NDArray[np.float64],
    settings: GradientObjectiveSettings,
    receiver_set: ReceiverSet,
    speaker_count: int,
) -> GradientObjectiveConstants:
    """雙精度準備常數；第一階段那八欄照 :func:`objective_constants`，計分範圍內至少要三個頻點（至少一個內部點）。"""
    base = objective_constants(frequencies_hz, settings.stage_one, receiver_set, speaker_count)
    inside = base.frequency_indices.shape[0]
    if inside < 3:
        raise ValueError("計分範圍內至少要有三個頻點：大峰代理只看去掉兩端之後的內部點")
    return GradientObjectiveConstants(
        *base,
        interior_positions=np.arange(1, inside - 1, dtype=np.int64),
        proxy_temperature_db=np.asarray(settings.temperature_db, dtype=np.float64),
        proxy_weight=np.asarray(settings.proxy_weight, dtype=np.float64),
        proxy_worse_reference=np.asarray(settings.worse_reference, dtype=np.float64),
    )


def proxy_gap_bound_db(constants: GradientObjectiveConstants) -> float:
    """代理比內部最大殘差最多高出的量 T·ln N（dB），N＝內部點數 × 喇叭數：max r_內部 ≤ P ≤ max r_內部 ＋ 這個數。
    對的是內部最大殘差，不是偵測器的真大峰（真大峰只保證不大於 P）。"""
    count = constants.interior_positions.shape[0] * constants.speaker_indices.shape[0]
    return float(constants.proxy_temperature_db) * math.log(count)


def _with_proxy(constants: GradientObjectiveConstants, base: StageOneObjectiveArrays,
                energies: jax.Array) -> GradientObjectiveArrays:
    residual = primary_residual_db(constants.stage_one(), energies)
    dtype = residual.dtype
    interior = jnp.take(residual, jnp.asarray(constants.interior_positions), axis=-1)
    temperature = jnp.asarray(constants.proxy_temperature_db, dtype=dtype)
    proxy = temperature * jax.nn.logsumexp(interior / temperature)
    weight = jnp.asarray(constants.proxy_weight, dtype=dtype)
    reference = jnp.asarray(constants.proxy_worse_reference, dtype=dtype)
    total = base.screening_total + weight * proxy / reference
    return GradientObjectiveArrays(*base, big_peak_proxy_db=proxy, objective_total=total,
                                   interior_max_residual_db=jnp.max(interior))


def gradient_objective_from_energies(
    constants: GradientObjectiveConstants, energies: jax.Array,
) -> GradientObjectiveArrays:
    """``(喇叭, 座位, F)`` 浮點正能量 → 第一階段五欄（``objective_from_energies`` 原樣）、代理與新合分。"""
    return _with_proxy(constants, objective_from_energies(constants.stage_one(), energies), jnp.asarray(energies))


def gradient_objective_from_pressures(
    constants: GradientObjectiveConstants, pressures: jax.Array,
) -> GradientObjectiveArrays:
    """複數聲壓 ``(F, 喇叭, 座位)``：第一階段五欄由 ``objective_from_pressures`` 原樣算（舊合分一位都不動），
    代理用同一個能量定義 ``abs(p)**2``。"""
    base = objective_from_pressures(constants.stage_one(), pressures)
    energies = jnp.transpose(jnp.abs(jnp.asarray(pressures)) ** 2, (1, 2, 0))
    return _with_proxy(constants, base, energies)


def objective_reading(result: GradientObjectiveArrays) -> ObjectiveReading:
    """帳本、結果與報告用的那一組浮點。"""
    return ObjectiveReading(
        float(result.objective_total), float(result.screening_total),
        (float(result.mean_own_line_ripple_rms_db), float(result.mean_seat_robustness_rms_db),
         float(result.big_peak_proxy_db)))
