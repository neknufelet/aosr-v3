"""材料能用在哪幾面（#1176）：立面圖下「這一面每一區的做法」只列這一面能貼的處理材料，其餘分組收起來、各一句理由。

現算、不改材料庫格式：照代號、顯示名與材料種類分組，規則有先後（前面的先吃）：

1. 代號 ``demo109`` 開頭 → 另一間示範房（109 號房）的，平常不列；
2. 固定材料本身是固定阻抗或實測表 → 牆面、地面本身（整面換掉，不是貼上去的），不列；
3. 固定多層構造、背後是空氣（``backing == "air"``）→ 門、窗，不列；
4. 代號含 ``ceiling`` 或顯示名以「天花」開頭 → 整片天花構造，不列成貼上去的處理（它描述的是整片天花，貼上去會疊兩層，#1175）；
5. 代號含 ``carpet`` 或 ``floor`` → 只列在地板；
6. 可最佳化材料、背後剛背的固定多層構造 → 牆面四面與天花（天花是貼在吊頂下面）；
7. 以上都不是 → 其他：每一面都看得到，但另成一組、不混進牆面清單。

這裡只決定「列在哪幾面」；不列的照樣在網頁「顯示全部材料」裡分組看得到，存過的代號也照樣顯示。不碰搜尋規則與評分。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

from aosr.materials.construction_models import Assembly, ConstantImpedance
from aosr.materials.library import FixedEntry, LibraryEntry

SurfaceGroup: TypeAlias = Literal["demo109", "surface_itself", "door_window", "ceiling_build", "floor", "wall", "other"]
WALLS: Final = ("x0", "xL", "y0", "yL")
ALL_SURFACES: Final = ("floor", "ceiling", *WALLS)
# 每一組在網頁上的標題（「顯示全部材料」打開後一組一個標題）；順序就是網頁列的順序。
GROUP_WORDS: Final[dict[SurfaceGroup, str]] = {
    "wall": "牆面與天花用的（貼在牆上、或貼在吊頂下面）",
    "floor": "地板用的",
    "other": "其他（分不出是哪一面用的，自己判斷）",
    "ceiling_build": "整片天花構造（描述的是整片天花本身，不是貼上去的處理）",
    "door_window": "門、窗",
    "surface_itself": "牆面、地面本身（整面換掉，不是貼上去的）",
    "demo109": "另一間示範房（109 號房）用的",
}
_DEMO_PREFIX: Final = "demo109"


@dataclass(frozen=True)
class Placement:
    """一筆材料列在哪幾面（空的就是哪一面都不列）、屬於哪一組、一句白話理由。"""

    group: SurfaceGroup
    surfaces: tuple[str, ...]
    reason: str

    def to_json(self) -> dict[str, object]:
        return {"group": self.group, "surfaces": list(self.surfaces), "reason": self.reason}


def placement(entry: LibraryEntry) -> Placement:
    """照模組說明那七條規則的先後分組。"""
    material = entry.material if isinstance(entry, FixedEntry) else None
    if entry.id.startswith(_DEMO_PREFIX):
        return Placement("demo109", (), "代號是 demo109 開頭，是另一間示範房（109 號房）的材料，這間房平常用不到")
    if material is not None and not isinstance(material, Assembly):
        what = "固定阻抗" if isinstance(material, ConstantImpedance) else "實測表"
        return Placement("surface_itself", (), f"它本身就是一整面的表面阻抗（{what}），只能整面換掉，不是貼上去的")
    if isinstance(material, Assembly) and material.backing == "air":
        return Placement("door_window", (), "背後是空氣（兩邊都是空氣），是門、窗這類東西，不是貼在牆上的")
    if "ceiling" in entry.id or entry.label.startswith("天花"):
        return Placement("ceiling_build", (), "描述的是整片天花（板加吊頂空腔），貼上去會跟原本的天花疊成兩層")
    if "carpet" in entry.id or "floor" in entry.id:
        return Placement("floor", ("floor",), "地毯、地板類，只鋪在地板")
    if material is None:
        return Placement("wall", (*WALLS, "ceiling"), "可調的吸音材：牆面四面都能貼，天花也能貼在吊頂下面")
    if material.backing == "rigid":
        return Placement("wall", (*WALLS, "ceiling"), "固定成品：牆面四面都能貼，天花也能貼在吊頂下面")
    return Placement("other", ALL_SURFACES, "分不出是哪一面用的：每一面都列在「其他」，自己判斷")
