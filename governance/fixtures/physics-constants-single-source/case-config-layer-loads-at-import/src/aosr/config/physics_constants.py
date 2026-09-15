"""樣本設定層：模組載入時就把產品預設讀進來包成常數。"""


def load_physics_constants(path: str) -> dict[str, float]:
    return {"sound_speed_m_s": 0.0}


PHYSICS = load_physics_constants("src/aosr/config/data/physics_constants.toml")
