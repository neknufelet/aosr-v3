"""樣本產品程式：把讀產品設定的函式取別名匯進來。"""

from aosr.config.physics_constants import load_physics_constants as read_defaults


def run() -> float:
    return read_defaults("src/aosr/config/data/physics_constants.toml")["sound_speed_m_s"]
