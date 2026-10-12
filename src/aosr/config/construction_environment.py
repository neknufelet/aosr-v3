"""構造計算的設定配方：場景ρ、c加上產品熱黏空氣條件與多孔範圍。

讀檔限於設定層，沒有模組載入時快取，也不另存任何物理數字。
"""
from __future__ import annotations

from aosr.config.paths import config_path
from aosr.config.physics_constants import PhysicsConstants, load_physics_constants
from aosr.config.porous_model_ranges import PorousModelRanges, load_porous_model_ranges


def construction_environment(
    *, density_kg_m3: float, sound_speed_m_s: float,
) -> tuple[PhysicsConstants, PorousModelRanges]:
    """讀產品熱黏常數及Komatsu範圍，以場景ρ、c覆寫空氣條件。"""
    air = load_physics_constants(config_path("physics_constants.toml")).model_copy(
        update={"air_density_kg_m3": density_kg_m3, "sound_speed_m_s": sound_speed_m_s})
    return air, load_porous_model_ranges(config_path("porous_model_ranges.toml"))
