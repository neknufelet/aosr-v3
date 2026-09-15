"""樣本產品程式：物理模組自己去拿產品預設。"""

from aosr.config.physics_constants import load_physics_constants


def run() -> float:
    physics = load_physics_constants("src/aosr/config/data/physics_constants.toml")
    return physics["sound_speed_m_s"]
