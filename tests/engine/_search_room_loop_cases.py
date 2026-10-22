"""房間迴圈的合成範圍與建檔；沿用第一階段的計算替身。"""
from aosr.search.layout_settings import LayoutSettings
from tests.engine._search_store_cases import settings_document

ROOM_KEYS = ("room_Lx", "room_Ly", "room_Lz")
RANGES = {"Lx": {"low": 5.0, "high": 8.0}, "Ly": {"low": 3.5, "high": 5.0},
          "Lz": {"low": 2.6, "high": 3.2}}
NO_START = "起點不在搜尋範圍或夾角、耳距限制內，或專案房間不在房間範圍內、地板面積不足，沒有排入"
ROOM_COLUMN = "房間 長×寬×高（公尺）"


def layout_for(**fields: object) -> LayoutSettings:
    original = LayoutSettings.model_validate(settings_document()["layout"])
    return LayoutSettings.model_validate(original.model_dump() | {"room_range": RANGES} | fields)
