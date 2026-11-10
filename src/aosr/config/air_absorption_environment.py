"""晚期空氣吸收的產品條件；每次讀設定，不快取或另存物理數字。"""
from __future__ import annotations

from aosr.config.paths import config_path
from aosr.config.physics_constants import load_physics_constants


def air_absorption_environment() -> tuple[float, float, float]:
    """回攝氏溫度、相對濕度百分比與 Pa 氣壓；舊四欄案例不能默認零。"""
    air = load_physics_constants(config_path("physics_constants.toml"))
    conditions = (air.air_temperature_c, air.air_relative_humidity_percent, air.atmospheric_pressure_pa)
    names = ("air_temperature_c", "air_relative_humidity_percent", "atmospheric_pressure_pa")
    missing = tuple(name for name, value in zip(names, conditions, strict=True) if value is None)
    if missing:
        raise ValueError(f"空氣吸收條件缺值：{', '.join(missing)}")
    temperature, humidity, pressure = conditions
    if temperature is None or humidity is None or pressure is None:
        raise ValueError("空氣吸收條件缺值")
    return temperature, humidity, pressure
