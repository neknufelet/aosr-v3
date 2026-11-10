"""ISO 9613-1:1993 第 6 節的純音空氣衰減與平均自由路徑能量吸收。

係數函式由呼叫端明給條件；產品房間包裝從設定層讀條件，供晚期注入點共用。
"""
from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from aosr.config.air_absorption_environment import air_absorption_environment
from aosr.geometry.shoebox import Room


def _validate_conditions(temperature_c: float, relative_humidity_percent: float, pressure_pa: float) -> None:
    """拒絕非有限值、絕對零度以下、百分比越界與非正氣壓。"""
    if not math.isfinite(temperature_c) or temperature_c <= -273.15:
        raise ValueError("溫度必須是有限的攝氏值，且高於 -273.15 °C")
    if not math.isfinite(relative_humidity_percent) or not 0.0 <= relative_humidity_percent <= 100.0:
        raise ValueError("相對濕度必須是有限的百分比，且在 0 到 100 之間")
    if not math.isfinite(pressure_pa) or pressure_pa <= 0.0:
        raise ValueError("氣壓必須是有限且大於 0 的 Pa 值")


def air_attenuation_db_per_m(
    frequencies_hz: ArrayLike,
    *,
    temperature_c: float,
    relative_humidity_percent: float,
    pressure_pa: float,
) -> NDArray[np.float64]:
    """依 ISO 9613-1 純音公式回傳逐頻衰減係數，單位 dB/m。

    頻率是一維、有限且正的 Hz 陣列；溫度用攝氏、濕度用百分比、氣壓用 Pa。
    輸入頻率一律轉成 numpy（陣列運算函式庫）的雙精度，不沿用單精度輸入。
    """
    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    if frequencies.ndim != 1:
        raise ValueError("頻率必須是一維陣列")
    if not np.all(np.isfinite(frequencies)) or np.any(frequencies <= 0.0):
        raise ValueError("頻率每一格都必須有限且大於 0 Hz")
    _validate_conditions(temperature_c, relative_humidity_percent, pressure_pa)

    temperature_k = temperature_c + 273.15
    temperature_ratio = temperature_k / 293.15
    pressure_kpa = pressure_pa / 1000.0
    pressure_ratio = pressure_kpa / 101.325
    saturation_exponent = -6.8346 * (273.16 / temperature_k) ** 1.261 + 4.6151
    h = relative_humidity_percent * 10.0**saturation_exponent / pressure_ratio
    relaxation_oxygen = pressure_ratio * (24.0 + 4.04e4 * h * (0.02 + h) / (0.391 + h))
    relaxation_nitrogen = pressure_ratio * temperature_ratio**-0.5 * (
        9.0 + 280.0 * h * math.exp(-4.170 * (temperature_ratio ** (-1.0 / 3.0) - 1.0))
    )
    frequency_squared = frequencies**2
    classical = 1.84e-11 / pressure_ratio * temperature_ratio**0.5
    oxygen = 0.01275 * math.exp(-2239.1 / temperature_k) / (
        relaxation_oxygen + frequency_squared / relaxation_oxygen
    )
    nitrogen = 0.1068 * math.exp(-3352.0 / temperature_k) / (
        relaxation_nitrogen + frequency_squared / relaxation_nitrogen
    )
    return np.asarray(
        8.686 * frequency_squared * (classical + temperature_ratio**-2.5 * (oxygen + nitrogen)), dtype=np.float64,
    )


def air_absorption_fraction(
    frequencies_hz: ArrayLike,
    mean_free_path_m: float,
    *,
    temperature_c: float,
    relative_humidity_percent: float,
    pressure_pa: float,
) -> NDArray[np.float64]:
    """回傳每段平均自由路徑多吸收的能量比例 ε＝1−exp(−m·l)。

    m＝α·ln(10)/10 是能量的每公尺衰減率；用 expm1（保留微小指數差的函式）
    避免弱衰減時直接相減丟失有效位數。平均自由路徑以公尺明給。
    """
    if not math.isfinite(mean_free_path_m) or mean_free_path_m <= 0.0:
        raise ValueError("平均自由路徑必須是有限且大於 0 的公尺值")
    alpha = air_attenuation_db_per_m(
        frequencies_hz, temperature_c=temperature_c,
        relative_humidity_percent=relative_humidity_percent, pressure_pa=pressure_pa,
    )
    m = alpha * np.log(10.0) / 10.0
    return np.asarray(-np.expm1(-m * mean_free_path_m), dtype=np.float64)


def room_air_absorption_fraction(room: Room, frequencies_hz: ArrayLike) -> NDArray[np.float64]:
    """產品晚期一族共用 ε_air；4V/S 與晚期衰減的碰撞頻率用同一個幾何量。"""
    surface = 2.0 * (room.Lx * room.Ly + room.Ly * room.Lz + room.Lx * room.Lz)
    volume = room.Lx * room.Ly * room.Lz
    temperature, humidity, pressure = air_absorption_environment()
    return air_absorption_fraction(frequencies_hz, 4.0 * volume / surface,
        temperature_c=temperature, relative_humidity_percent=humidity, pressure_pa=pressure)


def air_adjusted_absorption(absorption: ArrayLike, fraction: ArrayLike) -> NDArray[np.float64]:
    """家具後的每格剩餘反射率乘同一個空氣存活率；ε＝0 保留原陣列值。"""
    epsilon = np.asarray(fraction, dtype=np.float64)
    if not np.all(np.isfinite(epsilon) & (epsilon >= 0.0) & (epsilon < 1.0)):
        raise ValueError("空氣吸收 ε_air 必須有限且在 [0, 1) 之內")
    alpha = np.asarray(absorption, dtype=np.float64)
    return np.asarray(alpha + (1.0 - alpha) * epsilon, dtype=np.float64)
