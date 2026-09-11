"""最底層：唯一准碰機器設定的地方。"""
import os

PLATFORMS = "JAX_PLATFORMS"


def pin(where: dict[str, str]) -> None:
    """合規：只有這一層准碰 os.environ 與那一族環境變數名。"""
    where[PLATFORMS] = "cpu"
    os.environ[PLATFORMS] = "cpu"
