"""材料能用在哪幾面（#1176）：立面圖下「這一面每一區的做法」只列這一面能貼的處理材料，其餘分組收起來、各一句理由。

現算、不改材料庫格式：照代號、顯示名與材料種類分組，規則有先後（前面的先吃）：

1. 代號 ``demo109`` 開頭 → 另一間示範房（109 號房）的，平常不列；
2. 固定材料本身是固定阻抗或實測表 → 牆面、地面本身（整面換掉，不是貼上去的），不列；
3. 固定多層構造、背後是空氣（``backing == "air"``）→ 門、窗，不列；
4. 代號含 ``ceiling`` 或顯示名以「天花」開頭 → 整片天花構造，不列成貼上去的處理（它描述的是整片天花，貼上去會疊兩層，#1175）；
5. 代號含 ``carpet`` 或 ``floor`` → 只列在地板；
6. 剩下的（可最佳化材料、背後剛背的固定多層構造）→ 牆面四面與天花（天花是貼在吊頂下面）。

背後只有剛背與空氣兩種（``Assembly.backing``），所以六條分完就沒有剩的；多一種背後時 ``assert_never`` 讓型別檢查紅，
要先決定它歸哪一組（審查 11-15：原本第七條「其他」走不到，拿掉）。

這裡只決定「列在哪幾面」；不列的照樣在網頁「顯示全部材料」裡分組看得到，存過的代號也照樣顯示。不碰搜尋規則與評分。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal, TypeAlias, assert_never

from aosr.materials.construction_models import Assembly, ConstantImpedance
from aosr.materials.library import LibraryEntry, fixed_surface

SurfaceGroup: TypeAlias = Literal["demo109", "surface_itself", "door_window", "ceiling_build", "floor", "wall"]
WALLS: Final = ("x0", "xL", "y0", "yL")
# 每一組在網頁上的標題（「顯示全部材料」打開後一組一個標題）；順序就是網頁列的順序。
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

    def to_json(self) -> dict[str, object]:
        return {"group": self.group, "surfaces": list(self.surfaces), "reason": self.reason}


def placement(entry: LibraryEntry) -> Placement:
    """照模組說明那六條規則的先後分組。"""
    material = fixed_surface(entry)
    if entry.id.startswith(_DEMO_PREFIX):
        return Placement("demo109", (), "代號是 demo109 開頭，是另一間示範房（109 號房）的材料，這間房平常用不到")
    if material is not None and not isinstance(material, Assembly):
        what = "固定阻抗" if isinstance(material, ConstantImpedance) else "實測表"
        return Placement("surface_itself", (), f"它本身就是一整面的表面阻抗（{what}），只能整面換掉，不是貼上去的")
    backing = material.backing if material is not None else None  # 可最佳化材料沒有背後這一格
    if backing == "air":
        return Placement("door_window", (), "背後是空氣（兩邊都是空氣），是門、窗這類東西，不是貼在牆上的")
    if backing is not None and backing != "rigid":
        assert_never(backing)  # 多一種背後：先決定它歸哪一組
    if "ceiling" in entry.id or entry.label.startswith("天花"):
        return Placement("ceiling_build", (), "描述的是整片天花（板加吊頂空腔），貼上去會跟原本的天花疊成兩層")
    if "carpet" in entry.id or "floor" in entry.id:
        return Placement("floor", ("floor",), "地毯、地板類，只鋪在地板")
    if backing is None:
        return Placement("wall", (*WALLS, "ceiling"), "可調的吸音材：牆面四面都能貼，天花也能貼在吊頂下面")
    return Placement("wall", (*WALLS, "ceiling"), "固定成品：牆面四面都能貼，天花也能貼在吊頂下面")
