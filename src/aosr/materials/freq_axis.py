"""上一代正式頻率表與純 Python 常用軸產生器。

住 ``materials`` 層是決策紙 ``engine-first-block-config-shape`` 的決定；上一代的
``list``／``jnp.array`` 在這裡都變成不可變的 tuple。

``FREQS_HZ`` 是相容性參考：逐位搬自上一代原檔手打的兩位小數表，不是
``10 * 2 ** (n / 6)`` 四捨五入的結果。扣掉 84.49 Hz 玻璃板共振補點後，仍有一批
格子的末位數與公式四捨五入值不同；最大相對偏差由考卷量出，不是 v3 契約。v3 通用
產生器只走乾淨公式，不追隨手打偏差，也不暗中插入專案專用補點。

上一代同檔的 ``FREQ_AXIS`` 會建立材料層的 JAX 型別，``interp_to_bands`` 會呼叫
NumPy；兩者都碰到本段禁止的依賴，留在票 #218 第 2 段，不在這個純 Python 模組假搬。
"""

from __future__ import annotations

import math
import struct
from typing import cast

FREQS_HZ: tuple[float, ...] = (
    10.00,
    11.22,
    12.60,
    14.14,
    15.87,
    17.82,
    20.00,
    22.45,
    25.20,
    28.28,
    31.75,
    35.64,
    40.00,
    44.90,
    50.40,
    56.57,
    63.50,
    71.27,
    80.00,
    84.49,
    89.80,
    100.80,
    113.14,
    126.99,
    142.54,
    160.00,
    179.60,
    201.59,
    226.27,
    253.98,
    285.09,
    320.00,
    359.19,
    403.19,
    452.55,
    507.97,
    570.17,
    640.00,
    718.38,
    806.38,
    905.10,
    1015.94,
    1140.35,
    1280.00,
    1436.75,
    1612.75,
    1810.19,
    2031.87,
    2280.70,
    2560.00,
    2873.50,
    3225.50,
    3620.38,
    4063.74,
    4561.41,
    5120.00,
    5747.01,
    6451.01,
    7240.76,
    8127.49,
    9122.81,
    10240.00,
    11494.02,
    12902.01,
    14481.53,
    16254.97,
    18245.62,
    20480.00,
)


def _as_float32(value: float) -> float:
    """只用標準庫重現上一代 identity 常數的 IEEE-754 binary32 轉型。"""
    return cast(float, struct.unpack(">f", struct.pack(">f", value))[0])


FREQS_HZ_IDENTITY: tuple[float, ...] = tuple(_as_float32(value) for value in FREQS_HZ)

CHORAS_BANDS: tuple[float, ...] = (
    63.0,
    125.0,
    250.0,
    500.0,
    1000.0,
    2000.0,
    4000.0,
    8000.0,
)

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


def frequency_axis(
    mode: str,
    f_min_hz: float,
    f_max_hz: float,
    *,
    per_octave: int | None = None,
    step_hz: float | None = None,
) -> tuple[float, ...]:
    """產生含下界、且不超過上界的倍頻或線性頻率軸；倍頻軸不修正表值。"""
    if mode not in ("octave_fraction", "linear"):
        raise ValueError(f"mode 不支援：{mode!r}")
    if not math.isfinite(f_min_hz) or f_min_hz <= 0.0:
        raise ValueError("f_min_hz 必須是大於零的有限值")
    if not math.isfinite(f_max_hz) or f_max_hz <= f_min_hz:
        raise ValueError("f_max_hz 必須是大於 f_min_hz 的有限值")

    if mode == "octave_fraction":
        if isinstance(per_octave, bool) or not isinstance(per_octave, int) or per_octave <= 0:
            raise ValueError("per_octave 必須是正整數")
        if step_hz is not None:
            raise ValueError("step_hz 只適用於 linear 模式")
        values: list[float] = []
        index = 0
        while (value := f_min_hz * 2 ** (index / per_octave)) <= f_max_hz:
            values.append(value)
            index += 1
        return tuple(values)

    if isinstance(step_hz, bool) or not isinstance(step_hz, (int, float)):
        raise ValueError("step_hz 必須是正有限值")
    if not math.isfinite(step_hz) or step_hz <= 0.0:
        raise ValueError("step_hz 必須是正有限值")
    if per_octave is not None:
        raise ValueError("per_octave 只適用於 octave_fraction 模式")
    values = []
    index = 0
    while (value := f_min_hz + index * step_hz) <= f_max_hz:
        values.append(value)
        index += 1
    return tuple(values)
