"""求解器的頻率軸：這一塊唯一那張網格（single source of truth，唯一來源）。

網格是 1/6 倍頻、10 Hz 到 20 480 Hz，外加一個不在格子上的點（84.49 Hz，玻璃窗的
limp-panel 共振修正），共 68 點。要改這張網格得先有一張決策紙：下游每一份存下去的酬載都
按它對齊。

``FREQS_HZ_IDENTITY`` 釘成 float32：身分字串與存檔用的是它，所以**不管載入的時候
``jax_enable_x64``（JAX 的 64 位元開關）是開還是關**，它都要是同一組值。
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike

from aosr.materials.response import FrequencyAxis

FREQS_HZ = jnp.array([
     10.00,  11.22,  12.60,  14.14,  15.87,  17.82,
     20.00,  22.45,  25.20,  28.28,  31.75,  35.64,
     40.00,  44.90,  50.40,  56.57,  63.50,  71.27,
     80.00,  84.49,  89.80, 100.80, 113.14, 126.99, 142.54,
    160.00, 179.60, 201.59, 226.27, 253.98, 285.09,
    320.00, 359.19, 403.19, 452.55, 507.97, 570.17,
    640.00, 718.38, 806.38, 905.10, 1015.94, 1140.35,
   1280.00, 1436.75, 1612.75, 1810.19, 2031.87, 2280.70,
   2560.00, 2873.50, 3225.50, 3620.38, 4063.74, 4561.41,
   5120.00, 5747.01, 6451.01, 7240.76, 8127.49, 9122.81,
  10240.00, 11494.02, 12902.01, 14481.53, 16254.97, 18245.62,
  20480.00,
])

FREQS_HZ_IDENTITY: tuple[float, ...] = tuple(
    float(value) for value in np.asarray(FREQS_HZ, dtype=np.float32)
)

FREQ_AXIS: FrequencyAxis = FrequencyAxis.from_hz(FREQS_HZ)

# 前端顯示用的 8 個倍頻帶中心。是 ``list`` 不是 ``tuple``：呼叫端看得到容器的種類。
CHORAS_BANDS = [63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0]

# 報表用的倍頻帶中心。住在這裡是為了讓硬編碼守門分得清「顯示／報表的頻帶」與「計分常數」。
OCTAVE_BAND_CENTERS_HZ: tuple[float, ...] = (
    63.0,
    125.0,
    250.0,
    500.0,
    1000.0,
    2000.0,
    4000.0,
    8000.0,
    16000.0,
)


def interp_to_bands(
    values: ArrayLike,
    src_freqs_hz: ArrayLike | None = None,
    target_bands: ArrayLike | None = None,
) -> list[float]:
    """把一條 N 點頻譜插到目標頻帶中心上，**在 log10(Hz) 空間**插。

    在對數空間插是因為聽覺與這張網格都是等比的；線性空間插會把低頻那一段壓掉。兩端往外
    不外插而是**夾到端點值**（``np.interp`` 的預設），外插出來的數沒有物理根據。

    :param values: 跟 ``src_freqs_hz`` 對齊的 N 個值。
    :param src_freqs_hz: 來源軸（Hz），預設是這張 68 點網格 :data:`FREQS_HZ`。
    :param target_bands: 目標頻帶中心（Hz），預設是 :data:`CHORAS_BANDS`。
    :returns: 一個目標頻帶一格的 ``list[float]``。
    """
    src_f = np.array(FREQS_HZ) if src_freqs_hz is None else np.asarray(src_freqs_hz)
    tgt_f = np.asarray(CHORAS_BANDS if target_bands is None else target_bands)
    vals = np.asarray(values, dtype=float)
    out = np.interp(np.log10(tgt_f), np.log10(src_f), vals)
    return [float(x) for x in out]
