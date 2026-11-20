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

結果頁（第 2 支）：存的方案裡有這類構造時，「各類結果」表上方多一句、頻率範圍含界線以下的那幾列多一個小標籤
（:func:`result_block`）。哪一類讀登記簿哪幾格寫在 :data:`RESULT_RANGES`，跟評分程式實際讀的那幾格對（每一格旁邊註明是哪一支讀的）。

第三階段頁與「逐區表與改善多少」（第 3 支）：哪幾面命中一律看方案每一區的材料（:func:`scheme_walls`；第三階段的設計設定
不存材料，固定、保留原牆、交給搜尋的區的材料都住在方案裡）。交給搜尋的區不分裝法都看：貼在原牆上時搜尋貼的層背後接它
（src/aosr/gradient/surface_design.py::_backing），貼在硬底上時「不貼」那個候選保留的也是它
（src/aosr/gradient/surface_design.py 的保留原牆設計 ``kept_material``，src/aosr/gradient/material_gradient.py::kept_impedance_rows 算它）。
第三階段頁的句子在 :func:`stage_three_block`，改善表的在 :func:`improvement_block`。
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Final, Literal

from aosr.config.paths import config_path
from aosr.config.quality_targets import QualityPurpose, QualityTargets, SettingEntry, load_quality_targets
from aosr.gui.room_words import FOUR_WALLS
from aosr.gui.wall_words import wall_words
from aosr.materials.construction_models import Assembly, Layer, MeasuredImpedance, MountedOnWall, PorousLayer
from aosr.materials.library import FixedEntry, LibraryEntry, MeasuredTable
from aosr.reporting.scheme import Scheme

SCORED_RANGE_KEY: Final = "stage_one_low_frequency.scored_range_hz"
# 第三階段梯度目標的計分範圍（src/aosr/scoring/gradient_objective_settings.py::gradient_objective_settings_from_section 讀的那一格）。
GRADIENT_RANGE_KEY: Final = "stage_three_gradient.scored_range_hz"
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
    return registry_band_hz(load_quality_targets(config_path("quality_targets.toml")))


def registry_band_hz(registry: QualityTargets) -> float:
    """同一條界線，從已經載入的那一份登記簿讀（結果頁用它自己那一份）。"""
    found: set[float] = set()
    for section in registry.stage_one_sections:
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


# ---- 結果頁「各類結果」（第 2 支） ----

# 一格怎麼讀：``range`` 是一段 [下界, 上界]（剛好兩個數）；``centers`` 是一串遞增的頻帶中心（下界取最低那一個中心）。
_Read = Literal["range", "centers"]
# 那一格住登記簿哪一段：``purpose`` 是用途那一段，``stage_one`` 是同一個用途的第一階段那一段。
_Section = Literal["purpose", "stage_one"]
# 類別（``QualityCategory`` 的值）→ 它計分時讀的每一段頻率（段、鍵名、讀法）。每一格後面註明評分程式哪一支讀它。
# 不計分、尚未評估的類寫空的＝不標。少一類、多一類都報錯（:func:`marked_categories`）。
RESULT_RANGES: Final[dict[str, tuple[tuple[_Section, str, _Read], ...]]] = {
    "timbre_balance": (
        ("purpose", "timbre_balance.coverage_range_hz", "range"),   # src/aosr/scoring/timbre.py::load_timbre_settings
        ("purpose", "timbre_balance.tilt_fit_range_hz", "range"),   # 同上
        ("purpose", "timbre_balance.ripple_range_hz", "range"),     # 同上
        ("purpose", "timbre_balance.feature_range_hz", "range"),    # 同上
    ),
    "listening_area_stability": (
        # 位置之間的整體音量差：src/aosr/reporting/evaluation.py::read_registry_settings（BROADBAND_KEY）
        # → src/aosr/scoring/listening_area.py::evaluate_listening_area
        ("purpose", "channel_matching.broadband_range_hz", "range"),
        # 位置之間的斜率差、起伏差、峰谷對不對得上，拿的是逐點音色評估的結果：src/aosr/scoring/timbre.py::load_timbre_settings
        ("purpose", "timbre_balance.tilt_fit_range_hz", "range"),
        ("purpose", "timbre_balance.ripple_range_hz", "range"),
        ("purpose", "timbre_balance.feature_range_hz", "range"),
    ),
    # src/aosr/scoring/stage_one_score.py::stage_one_settings_from_section（第一階段那一段）
    "low_frequency_response": (("stage_one", SCORED_RANGE_KEY, "range"),),
    "low_frequency_decay": (),  # 不計分：不標
    # src/aosr/scoring/reflections_cost.py::_checked_payload
    "reflections_and_echo": (("purpose", "reflections_and_echo.frequency_range_hz", "range"),),
    # src/aosr/scoring/reverberation_cost.py::target_intervals（逐帶目標的中心頻率）
    "reverberation": (("purpose", "reverberation.target_band_centers_hz", "centers"),),
    "channel_matching": (
        ("purpose", "channel_matching.ripple_range_hz", "range"),      # src/aosr/scoring/channel_matching_settings.py::_load_settings
        ("purpose", "channel_matching.broadband_range_hz", "range"),   # 同上
    ),
    "spatial_impression": (),  # 尚未評估：不標
}
# 有算出代價的狀態（``EvaluationState`` 的值）：已量、已算代價。
_COSTED: Final = ("measured", "costed")


def _lower_hz(section: QualityPurpose, key: str, read: _Read) -> float:
    """那一格的下界（Hz）；登記簿沒有這一格（KeyError）、不是量法設定、或讀不出一段頻率就報錯，不當成沒有。"""
    entry = section.entry(key)
    value = entry.value if isinstance(entry, SettingEntry) else None
    if not isinstance(value, tuple) or (read == "range" and len(value) != 2):
        raise TypeError(f"{key} 讀不出一段頻率範圍")
    return float(value[0])


def marked_categories(states: Mapping[str, str], registry: QualityTargets, purpose: str, band: float) -> list[str]:
    """哪幾類要標「含界線以下」：有算出代價，而且它計分讀的範圍裡有一段的下界嚴格小於界線（剛好等於的不標）。
    ``states`` 是結果頁每一類的狀態（照表上的順序）。範圍只讀登記簿現行的值，照這份結果的用途取。"""
    from aosr.scoring.contract import QualityCategory

    if set(RESULT_RANGES) != {category.value for category in QualityCategory}:
        raise ValueError("RESULT_RANGES 要剛好列出每一個品質類別")
    sections: dict[str, QualityPurpose] = {"purpose": registry.purpose(purpose), "stage_one": registry.stage_one(purpose)}
    lowest = {category: [_lower_hz(sections[section], key, read) for section, key, read in spans]
              for category, spans in RESULT_RANGES.items()}
    return [category for category, state in states.items()
            if state in _COSTED and any(lower < band for lower in lowest[category])]


def result_block(walls: Sequence[str], states: Mapping[str, str], registry: QualityTargets, purpose: str) -> dict[str, object] | None:
    """結果頁回覆旁邊那一格。``walls``：存的方案裡哪幾面有這類構造（給人看的牆名，照面序）。一面都沒有就回 None，
    而且不讀界線、不讀範圍。有：表上方那一句（``sentence``）、「為什麼」（``why_label``、``why``）、小標籤的字（``tag``）、
    要標的類別（``categories``）。"""
    if not walls:
        return None
    edge = registry_band_hz(registry)
    band = f"{edge:g}"
    return {
        "sentence": (f"{'、'.join(walls)}有 VPR（鋼板共振吸音）{_KIND}：{band} Hz 以下的數字只能參考；"
                     f"下表標了「含 {band} Hz 以下」的幾列都受影響。"),
        "why_label": "為什麼",
        "why": f"{_WHY}。",
        "tag": f"含 {band} Hz 以下",
        "categories": marked_categories(states, registry, purpose, edge),
    }


# ---- 第三階段頁與「改善多少」（第 3 支） ----

def scheme_walls(*schemes: Scheme) -> list[str]:
    """這幾份方案裡哪幾面有「鋼板直接貼在棉上」的構造：看那一面每一區的材料（整面一區、矩形分區都是區；貼在原牆上的
    與原牆本身由 :func:`reference_only` 自己看）。只寫牆名、一面一次，面序同結果頁開頭那一句（room_words.treated_words）；
    好幾份取聯集，牆名照第一份。六面只給固定阻抗的方案沒有構造，一面都不算。"""
    hit = {wall for scheme in schemes for wall, zones in (scheme.scene.surface_zones_by_wall or {}).items()
           if any(reference_only(zone.material) for zone in zones)}
    names = wall_words(schemes[0])
    return [names[wall] for wall in (*FOUR_WALLS, "ceiling", "floor") if wall in hit]


def stage_three_block(walls: Sequence[str], registry: QualityTargets, purpose: str) -> dict[str, object] | None:
    """第三階段頁（開跑前的預覽、每一跑的「注意」）那一格；一面都沒有就回 None，而且不讀界線、不讀範圍。
    第三階段只算登記簿那一段（:data:`GRADIENT_RANGE_KEY`）：整段都在界線以下（上界不超過界線）就講這一跑的數字都只能參考，
    超過界線就只講界線以下那一部分。"""
    if not walls:
        return None
    entry = registry.stage_one(purpose).entry(GRADIENT_RANGE_KEY)
    if not isinstance(entry, SettingEntry) or not isinstance(entry.value, tuple) or len(entry.value) != 2:
        raise TypeError(f"{GRADIENT_RANGE_KEY} 不是一段頻率範圍")
    low, high = (float(value) for value in entry.value)
    edge = registry_band_hz(registry)
    head = f"{'、'.join(walls)}有 VPR（鋼板共振吸音）{_KIND}："
    tail = (f"第三階段本來就只算 {low:g}～{high:g} Hz，整段都在 {edge:g} Hz 以下，所以這一跑的數字都只能參考。" if high <= edge
            else f"第三階段算 {low:g}～{high:g} Hz，其中 {edge:g} Hz 以下的部分只能參考。")
    return {"sentence": head + tail, "why_label": "為什麼", "why": f"{_WHY}。"}


def scheme_stage_three_block(scheme: Scheme) -> dict[str, object] | None:
    """一份方案的第三階段那一格（預覽、每一跑共用）；沒命中回 None，而且不讀品質登記簿。"""
    walls = scheme_walls(scheme)
    if not walls:
        return None
    return stage_three_block(walls, load_quality_targets(config_path("quality_targets.toml")), scheme.purpose)


def improvement_block(walls: Sequence[str], costed: Mapping[str, bool], in_total: Mapping[str, str], registry: QualityTargets,
                      purpose: str, labels: Mapping[str, str]) -> dict[str, object] | None:
    """「改善多少」表旁邊那一格：句子、「為什麼」、標籤都是結果頁那一份（:func:`result_block`），不另寫字。
    ``costed``：類別 → 這一列有沒有算出代價（照表上的順序）；``in_total``：算進總分的類別 → 總分那一列印的名字
    （沒有總分列就是空的）；``labels``：類別 → 那一列印的名字。``rows`` 是要標的列（列上印的名字）：被標的類別，
    其中有任何一類算進總分，總分那一列也標。一面都沒有就回 None，而且不讀界線。"""
    block = result_block(walls, {category: "costed" if has else "unavailable" for category, has in costed.items()},
                         registry, purpose)
    if block is None:
        return None
    marked = block.pop("categories")
    assert isinstance(marked, list)
    totals = list(dict.fromkeys(in_total[category] for category in marked if category in in_total))
    return {**block, "rows": [*(labels[category] for category in marked), *totals]}
