"""材料的位置與分組，供分區頁的指定貼與最佳化清單共用。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal, TypeAlias, assert_never

from aosr.materials.construction_models import Assembly, ConstantImpedance, Layer
from aosr.materials.library import ConstructionEntry, FixedEntry, LibraryEntry, fixed_surface

SurfaceGroup: TypeAlias = Literal["demo109", "surface_itself", "door_window", "ceiling_build", "floor", "wall"]
Family: TypeAlias = Literal["porous", "resonant", "surface", "custom"]
WALLS: Final = ("x0", "xL", "y0", "yL")
# 材料位置的說明與短名也供材料庫頁使用。
GROUP_WORDS: Final[dict[SurfaceGroup, str]] = {
    "wall": "牆面與天花用的（貼在牆上、或貼在吊頂下面）",
    "floor": "地板用的",
    "ceiling_build": "整片天花構造（描述的是整片天花本身，不是貼上去的處理）",
    "door_window": "門、窗",
    "surface_itself": "牆面、地面本身（整面換掉，不是貼上去的）",
    "demo109": "另一間示範房（109 號房）用的",
}
# 每一組的短名（網頁「還有 N 筆不列出來：…」那一句只列短名，不套括號）；鍵跟 ``GROUP_WORDS`` 同一組、同一個順序。
GROUP_SHORT_WORDS: Final[dict[SurfaceGroup, str]] = {
    "wall": "牆面與天花用的",
    "floor": "地板用的",
    "ceiling_build": "整片天花構造",
    "door_window": "門窗",
    "surface_itself": "牆面地面本身",
    "demo109": "示範房的",
}
_DEMO_PREFIX: Final = "demo109"


@dataclass(frozen=True)
class Placement:
    """一筆材料列在哪幾面（空的就是哪一面都不列）、屬於哪一組、一句白話理由。"""

    group: SurfaceGroup
    surfaces: tuple[str, ...]
    reason: str
    family: Family

    def to_json(self) -> dict[str, object]:
        return {"group": self.group, "surfaces": list(self.surfaces), "reason": self.reason, "family": self.family}


def placement(entry: LibraryEntry, *, builtin: bool = False) -> Placement:
    """設計器存的構造歸個人組；內建構造與舊材料依位置、層種類分組。"""
    group, faces, reason = _location(entry)
    family: Family = "porous"
    if isinstance(entry, ConstructionEntry) and not builtin:
        family = "custom"
    elif group in ("door_window", "surface_itself", "floor", "ceiling_build"):
        family = "surface"
    else:
        layers: tuple[Layer, ...] = ()
        if isinstance(entry, ConstructionEntry):
            layers = entry.layers
        elif isinstance(entry, FixedEntry) and isinstance(entry.material, Assembly):
            layers = entry.material.layers
        if any(layer.kind in ("board", "perforated", "micro_perforated", "slotted") for layer in layers):
            family = "resonant"
    return Placement(group, faces, reason, family)


def legacy_replacement(entry: LibraryEntry) -> bool:
    """舊固定多層構造裡整塊換掉分區的兩種。"""
    return (isinstance(entry, FixedEntry) and isinstance(entry.material, Assembly)
            and _location(entry)[0] in ("door_window", "ceiling_build"))


def _location(entry: LibraryEntry) -> tuple[SurfaceGroup, tuple[str, ...], str]:
    if isinstance(entry, ConstructionEntry):
        surfaces = {"wall": (*WALLS, "ceiling"), "floor": ("floor",), "ceiling_build": ("ceiling",), "door_window": WALLS}
        return (entry.placement, surfaces[entry.placement], GROUP_WORDS[entry.placement])
    material = fixed_surface(entry)
    if entry.id.startswith(_DEMO_PREFIX):
        return ("demo109", (), "代號是 demo109 開頭，是另一間示範房（109 號房）的材料，這間房平常用不到")
    if material is not None and not isinstance(material, Assembly):
        what = "固定阻抗" if isinstance(material, ConstantImpedance) else "實測表"
        floor = any(term in text.lower() for text in (entry.id, entry.label) for term in ("carpet", "floor", "地板", "地毯"))
        # 名字同時講牆與地板（例如「一般硬牆面（磚、粉刷、硬地板平均）」）：六面都能用。
        surfaces = (*WALLS, "ceiling", "floor") if floor and "牆" in entry.label else ("floor",) if floor else (*WALLS, "ceiling")
        return ("surface_itself", surfaces, f"它本身就是一整面的表面阻抗（{what}），只能整面換掉，不是貼上去的")
    backing = material.backing if material is not None else None  # 可最佳化材料沒有背後這一格
    if backing == "air":
        return ("door_window", WALLS, "背後是空氣（兩邊都是空氣），是門、窗這類東西，不是貼在牆上的")
    if backing is not None and backing != "rigid":
        assert_never(backing)  # 多一種背後：先決定它歸哪一組
    if "ceiling" in entry.id or entry.label.startswith("天花"):
        return ("ceiling_build", ("ceiling",), "描述的是整片天花（板加吊頂空腔），貼上去會跟原本的天花疊成兩層")
    if "carpet" in entry.id or "floor" in entry.id:
        return ("floor", ("floor",), "地毯、地板類，只鋪在地板")
    if backing is None:
        return ("wall", (*WALLS, "ceiling"), "可調的吸音材：牆面四面都能貼，天花也能貼在吊頂下面")
    return ("wall", (*WALLS, "ceiling"), "固定成品：牆面四面都能貼，天花也能貼在吊頂下面")
