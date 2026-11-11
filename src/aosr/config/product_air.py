"""沒有場景的地方用的空氣條件（#1082 段 2-2b：材料頁的吸音曲線）。

材料頁畫的是「這個材料在產品標準空氣裡長什麼樣」，沒有方案、沒有場景ρ、c；有場景的計算一律走
:func:`aosr.config.construction_environment.construction_environment` 傳場景值，不用這一支。
讀檔限於設定層，沒有模組載入時快取，也不另存任何物理數字。
"""
from __future__ import annotations

from aosr.config.paths import config_path
from aosr.config.physics_constants import PhysicsConstants, load_physics_constants
from aosr.config.porous_model_ranges import PorousModelRanges, load_porous_model_ranges


def product_air_environment() -> tuple[PhysicsConstants, PorousModelRanges]:
    """產品設定檔的空氣條件與多孔模型範圍，原樣、不覆寫。"""
    return (load_physics_constants(config_path("physics_constants.toml")),
            load_porous_model_ranges(config_path("porous_model_ranges.toml")))
