"""「鋼板直接貼在棉上」的構造（VPR 這一類）：界線頻率以下的數字只能參考（#1208 第 1 支）。

判斷只有這一支，材料庫頁、分區頁、開跑計畫共用：構造裡任何一層是多孔層而且 ``bonded()`` 成立（填了骨架兩格＝整面黏在
前面薄板上的背襯，#1171）就算。看構造，不看名字。多層構造看每一層；貼在原牆上的構造看貼上去的每一層，再看原牆
（原牆是多層構造或實測表時）；實測表看底構造；固定阻抗沒有層，一律不算。前面再加開槽板、吸音棉的一樣算。
可調 VPR（``TunableLayersEntry``）不歸這一支管：程式最佳化本來就不收它（#1200 第 1 支），:func:`entry_reference_only`
只認固定材料。

界線從品質登記簿第一階段那一段的 ``stage_one_low_frequency.scored_range_hz`` 上界讀（有限元素算到哪裡），這裡不另寫數字；
每次現讀，字句跟著設定檔走。登記簿每一個用途的第一階段那一段上界要一樣（材料沒有用途可問），不一樣就報錯、不挑一個。

畫面上的字只住這裡：網頁從介面回覆拿（``/api/materials`` 的 ``reference_only``、``/api/material-curve`` 命中時的同名那一格），
腳本不另寫一份。這一支不載入 JAX，不算物理，數字一個都不改。
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Final

from aosr.config.paths import config_path
from aosr.config.quality_targets import SettingEntry, load_quality_targets
from aosr.materials.construction_models import Assembly, Layer, MeasuredImpedance, MountedOnWall, PorousLayer
from aosr.materials.library import FixedEntry, LibraryEntry, MeasuredTable

SCORED_RANGE_KEY: Final = "stage_one_low_frequency.scored_range_hz"
_KIND: Final = "這類「鋼板直接貼在棉上」的構造"
_WHY: Final = ("程式算這類構造用的是簡化算法（一維、鋼板當成整塊一起動）。跟 Fraunhofer 的實測比，算出來的共振偏高、頻寬偏窄。"
               "數字沒有改，照舊算")
_WHY_PICK: Final = "「指定貼一種」照舊能選、能貼，「由程式最佳化」不會挑它。"


def _bonded(layers: Iterable[Layer]) -> bool:
    return any(isinstance(layer, PorousLayer) and layer.bonded() for layer in layers)


def reference_only(material: object) -> bool:
    """這一份牆面材料是不是「鋼板直接貼在棉上」的構造；不是牆面材料的東西一律不算。"""
    if isinstance(material, Assembly):
        return _bonded(material.layers)
    if isinstance(material, MountedOnWall):
        return _bonded(material.layers) or reference_only(material.wall)
    if isinstance(material, MeasuredImpedance | MeasuredTable):
        return reference_only(material.base_assembly)
    return False


def entry_reference_only(entry: LibraryEntry) -> bool:
    """材料庫的一筆：只認固定材料（可最佳化材料的多孔層不准黏板；可調 VPR 不歸這一支）。"""
    return isinstance(entry, FixedEntry) and reference_only(entry.material)


def band_hz() -> float:
    """界線頻率（Hz）：第一階段計分範圍的上界，每次從設定檔現讀。"""
    sections = load_quality_targets(config_path("quality_targets.toml")).stage_one_sections
    found: set[float] = set()
    for section in sections:
        entry = section.entry(SCORED_RANGE_KEY)
        if not isinstance(entry, SettingEntry) or not isinstance(entry.value, tuple) or len(entry.value) != 2:
            raise TypeError(f"{SCORED_RANGE_KEY} 不是一段頻率範圍")
        found.add(float(entry.value[1]))
    if len(found) != 1:
        raise ValueError(f"品質登記簿第一階段各段的 {SCORED_RANGE_KEY} 上界要剛好一個值，讀到 {sorted(found)}")
    return found.pop()


def _band() -> str:
    return f"{band_hz():g}"


def sentence() -> str:
    return f"VPR（鋼板共振吸音）{_KIND}：{_band()} Hz 以下的數字只能參考。"


def search_why() -> str:
    return f"{_KIND} {_band()} Hz 以下的數字只能參考，所以不進程式最佳化；要用，把這一區改成「指定貼一種」。"


def plan_refused(zone: str, name: str) -> str:
    """開跑計畫拒收那一句；``zone`` 是 ``zone_limits.zone_words`` 的寫法（好幾區共用同一份清單時用「、」接起來），
    ``name`` 是材料庫顯示名。"""
    return (f"{zone}的准用清單裡有{name}：{_KIND} {_band()} Hz 以下的數字只能參考，所以不進程式最佳化；"
            "請在分區頁取消它，或把這一區改成「指定貼一種」")


def words() -> dict[str, object]:
    """畫面要的每一句（鍵名就是用在哪）：材料庫頁與分區頁都從這一份拿。"""
    band = _band()
    return {
        "band_hz": band_hz(),
        "sentence": sentence(),
        "band_words": f"{band} Hz 以下只能參考",
        "map_words": [f"{band} Hz 以下", "數字只能參考"],
        "map_line": f"{band} Hz 以下數字只能參考",
        "why": f"{_WHY}；{_WHY_PICK}",
        "why_only": f"{_WHY}。",
        "kind_words": f"；這一筆 {band} Hz 以下的數字只能參考，所以「由程式最佳化」不會挑，要用就在分區頁「指定貼一種」。",
        "search_why": search_why(),
        "listed_why": f"這一區的清單裡本來有它，程式不會挑。{search_why()}",
    }


def reset_blocked(name: str) -> str:
    """分區頁「改回跟其他牆面一樣」改不回去時講的那一句；``name`` 是材料庫顯示名。"""
    return f"其他牆面的清單裡有{name}（不進程式最佳化）：先在「其他牆面」取消它，才能改回跟其他牆面一樣。"


def listing_block(entries: Iterable[LibraryEntry]) -> dict[str, object] | None:
    """材料庫清單回覆旁邊那一格：命中的代號（照清單順序）、字句、每一筆「改不回去」的那一句（``reset_blocked``：代號 → 句子）。
    旗標不塞進每一筆材料裡——分區頁會把材料那一筆原樣送回曲線介面，材料頁會把它存回材料庫檔。
    一筆都沒有就回 None，而且不讀界線：沒有這類構造的材料庫不靠品質登記簿那一格。"""
    hits = [entry for entry in entries if entry_reference_only(entry)]
    if not hits:
        return None
    return {"ids": [entry.id for entry in hits], **words(), "reset_blocked": {entry.id: reset_blocked(entry.label) for entry in hits}}
