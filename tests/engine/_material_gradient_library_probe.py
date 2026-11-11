"""搜尋吃材料庫的物理探針（#1082 段 2-3a）：雙精度子行程裡叫產品的開跑前檢查，把結果寫成 JSON。

stdin：``{"entries": [使用者材料…], "frequencies_hz": [...]}``；argv[1]：輸出檔。
輸出 ``{"ok": true}`` 或 ``{"refused": 訊息, "type": 例外類別名}``；只叫產品入口，不自己判範圍。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from pydantic import TypeAdapter

from aosr.config.material_gradient import load_material_gradient_settings
from aosr.config.paths import config_path
from aosr.config.product_air import product_air_environment
from aosr.gradient import run
from aosr.materials.library import LibraryEntry
from aosr.materials.search_library import with_user_materials

_ENTRIES: TypeAdapter[tuple[LibraryEntry, ...]] = TypeAdapter(tuple[LibraryEntry, ...])


def main() -> None:
    request = json.loads(sys.stdin.read())
    entries = _ENTRIES.validate_python(request["entries"])
    settings = load_material_gradient_settings(config_path("material_gradient.toml"))
    materials = with_user_materials(settings, run.presets_table(), entries)
    try:
        air, _ = product_air_environment()
        run.check_user_families(materials, tuple(request["frequencies_hz"]), density_kg_m3=air.air_density_kg_m3,
                                sound_speed_m_s=air.sound_speed_m_s)
    except ValueError as error:
        result: dict[str, object] = {"refused": str(error), "type": type(error).__name__}
    else:
        result = {"ok": True}
    Path(sys.argv[1]).write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
