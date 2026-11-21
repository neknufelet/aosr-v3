"""材料庫與網頁考卷共用的自編構造。"""


def construction(backing: str = "rigid", placement: str = "wall", **changes: object) -> dict[str, object]:
    return {"kind": "construction", "id": "demo109_floor_sample", "label": "天花自編樣品", "source": "考卷自編",
            "provenance": "literature", "links": ["https://example.org/test"], "notes": ["自編條件"],
            "layers": [{"kind": "fabric", "surface_mass_kg_m2": 0.19, "flow_resistance_pa_s_m": 180.0}, {"kind": "air", "thickness_m": 0.063}],
            "product": False, "placement": placement, "backing": backing, **changes}
