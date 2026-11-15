"""真瀏覽器考做法表的材料清單（#1176，老闆 11-15「109示範又不知道是娥麼這也能當成一個材料」「都不知道在講甚麼」）：
選單與勾選表只列這一面能貼的（門、窗、地板、整片吊頂、另一間示範房的不出現），「顯示全部材料」打開才分組列出；
這一區現在放的那一筆就算不是這一面用的也照樣顯示；用語是白話（做法三種、原牆不寫阻抗數字、可調寫得出調什麼）。
材料庫用老闆 11-15 那 12 筆（``_boss_material_library``），房間用 109 號房的長寬高。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from playwright.sync_api import Browser, Locator, Page, expect
from pydantic import TypeAdapter

from aosr.materials.library import LibraryEntry, dump_user_library
from tests.engine._boss_material_library import BOSS_USER_ENTRIES
from tests.engine._gui_cache import gui_startup_identity_memo as gui_startup_identity_memo
from tests.engine._surface_zone_cases import ROOM_109, room109_document
from tests.engine.test_gui_browser import _open, _serve, browser as browser
from tests.engine.test_gui_zone_elevation_browser import _drag, _open_scheme, _write
from tests.engine.test_gui_zone_modes_browser import _choose, _modes, _save

HIDDEN_ON_WALLS = ("own_room_door_wood_solid", "own_room_window_glass_10mm_single", "own_room_ceiling_calsil_6mm_gap250",
                   "perforated_gypsum_8_18r_backed_gw30_ceiling", "carpet_on_concrete", "own_room_floor_wood_on_concrete",
                   "demo109_gw48k_limited", "demo109_measured_gw49_gaborit_s1")
DOOR = "own_room_door_wood_solid"
FLOOR_WOOD = "own_room_floor_wood_on_concrete"


def _boss_library(tmp_path: Path) -> None:
    adapter: TypeAdapter[tuple[LibraryEntry, ...]] = TypeAdapter(tuple[LibraryEntry, ...])
    (tmp_path / "materials.json").write_text(dump_user_library(adapter.validate_python(BOSS_USER_ENTRIES)), encoding="utf-8")


def _values(page: Page, selector: str) -> list[str]:
    return cast(list[str], page.locator(selector).evaluate_all("nodes => nodes.map(node => node.value)"))


def _ids(page: Page, wall: str, key: str) -> list[str]:
    prefix = f"zone-allow-{wall}-{key}-"
    return [value.removeprefix(prefix) for value in cast(list[str], page.locator(f"#zone-allowed-{wall}-{key} input").evaluate_all(
        "nodes => nodes.map(node => node.id)"))]


# 一格字照畫出來的樣子分行（逐字量字的上緣，換了一行上緣就變）。
LINES = """cell => {
    const node = cell.firstChild, lines = [];
    let top = null;
    for (let i = 0; i < node.data.length; i += 1) {
        const range = document.createRange(); range.setStart(node, i); range.setEnd(node, i + 1);
        const rect = range.getBoundingClientRect();
        if (top === null || Math.abs(rect.top - top) > 2) { lines.push(""); top = rect.top; }
        lines[lines.length - 1] += node.data[i];
    }
    return lines.map(line => line.trim());
}"""
# 字色與底色的對比（WCAG 相對亮度）、這一格自己的不透明度。
CONTRAST = """node => {
    const style = getComputedStyle(node);
    const rgb = text => text.match(/[\\d.]+/g).slice(0, 3).map(Number);
    const channel = value => { const c = value / 255; return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4; };
    const luminance = ([r, g, b]) => 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b);
    const [light, dark] = [luminance(rgb(style.color)), luminance(rgb(style.backgroundColor))].sort((a, b) => b - a);
    return {ratio: (light + 0.05) / (dark + 0.05), opacity: Number(style.opacity)};
}"""


def _small(page: Page, wall: str, material: str) -> Locator:
    """勾選表（整面一區）裡那一筆名字底下的小字。"""
    return page.locator("label", has=page.locator(f"#zone-allow-{wall}-whole_wall-{material}")).locator("small")


def test_menus_list_only_what_this_face_can_take_and_show_all_reveals_the_rest(tmp_path: Path, browser: Browser) -> None:
    _write(tmp_path, room109_document())
    _boss_library(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-x0").click()
        assert page.locator("#zone-mode-x0-whole_wall option").all_inner_texts() == ["指定貼一種", "不貼，保留原牆", "交給電腦挑"]
        # 指定貼一種：牆上能貼的固定成品（含老闆的 VPR），門、窗、天花構造、地毯、示範房一筆都沒有；原牆不寫阻抗數字。
        fixed = _values(page, "#zone-fixed-x0-whole_wall option")
        assert "glass_wool_48k_50mm" in fixed and "melamine100_vpr_steel1mm_softfoam100_draft" in fixed
        assert not set(HIDDEN_ON_WALLS) & set(fixed)
        expect(page.locator("#zone-fixed-x0-whole_wall option").first).to_have_text("現在：沒貼東西（原牆原樣）")
        expect(page.locator("#zone-numbers-x0-whole_wall")).to_contain_text("原牆：固定阻抗")

        # 交給電腦挑、打開清單：可調的寫得出調什麼（24K、96K 起點沒有空腔，空腔不調），不列的只寫一行有幾筆、哪幾組。
        _choose(page, "#zone-mode-x0-whole_wall", "search")
        page.locator("#zone-allowed-summary-x0-whole_wall").click()
        listed = _ids(page, "x0", "whole_wall")
        assert listed[:4] == ["glass_wool_48k", "glass_wool_24k", "glass_wool_96k", "none"]
        assert not set(HIDDEN_ON_WALLS) & set(listed)
        expect(page.locator("label", has=page.locator("#zone-allow-x0-whole_wall-glass_wool_48k"))).to_contain_text(
            "可調：厚 5～20 cm、空腔 0～25 cm，加起來最深 30 cm")
        # 空腔不調的最深就是厚度上限 20 cm，材料自己的 30 cm 擋不到：不寫「加起來最深」（審查 11-15 抓到寫 30 cm 是錯的）。
        for wool in ("glass_wool_24k", "glass_wool_96k"):
            expect(page.locator("label", has=page.locator(f"#zone-allow-x0-whole_wall-{wool}")).locator("small")).to_have_text(
                "可調：厚 5～20 cm、空腔不調")
        expect(page.locator("label", has=page.locator("#zone-allow-x0-whole_wall-glass_wool_48k"))).not_to_contain_text("（可調厚度與空腔）")
        # 固定成品那一組標題已經講過整塊貼或不貼：每一筆只留總厚；有薄板（只給面密度、沒有厚度）的註明不含，
        # 不然跟顯示名裡把鋼板算進去的「總厚約 17 cm」打架（審查 11-15）。
        expect(_small(page, "x0", "fabric_pet_a_gap100")).to_have_text("總厚約 10 cm")
        expect(_small(page, "x0", "slotted16_mel50_vpr_steel3_5mm_mel100_draft")).to_have_text("總厚約 16.6 cm（不含薄板）")
        # 不列的那一句只列短組名、不套括號。
        hidden = page.locator("#zone-allowed-hidden-x0-whole_wall")
        expect(hidden).to_have_text("還有 11 筆不是前牆用的、不列出來：地板用的、整片天花構造、門窗、牆面地面本身、示範房的。"
                                    "要選就打開表格下面「顯示全部材料」")
        page.locator("#elevation-panel").screenshot(path=str(tmp_path / "choices-wall.png"))

        # 「顯示全部材料」：不列的那幾組分組列出、各一個標題；示範房要選自己的材料就從這裡勾。
        page.locator("#zone-show-all").check()
        listed = _ids(page, "x0", "whole_wall")
        assert set(HIDDEN_ON_WALLS) <= set(listed)
        expect(page.locator("#zone-allowed-x0-whole_wall")).to_contain_text("不列的：另一間示範房（109 號房）用的")
        expect(page.locator("#zone-allowed-hidden-x0-whole_wall")).to_have_count(0)

        # 地板：只有地毯能鋪；沒有可調材料，照實寫電腦只能從「不貼」開始。
        page.locator("#zone-show-all").uncheck()
        page.locator("#elevation-wall-floor").click()
        fixed = _values(page, "#zone-fixed-floor-whole_wall option")
        assert fixed == ["", "carpet_on_concrete"]
        _choose(page, "#zone-mode-floor-whole_wall", "search")
        page.locator("#zone-allowed-summary-floor-whole_wall").click()
        assert _ids(page, "floor", "whole_wall") == ["none", "carpet_on_concrete"]
        expect(page.locator("#zone-allowed-floor-whole_wall")).to_contain_text("地板沒有可調的材料：電腦只能從「不貼」開始")
        page.locator("#elevation-panel").screenshot(path=str(tmp_path / "choices-floor.png"))
        assert watched.page_errors == [] and watched.console_errors == []


def test_a_zone_holding_a_material_from_another_group_still_shows_it(tmp_path: Path, browser: Browser) -> None:
    """選單只列這一面能貼的，但這一區現在放的就是門（打開「顯示全部材料」選的）：關掉之後選單照樣是它、寫出它不是這一面用的；
    保留原牆的區寫白話、數字收在細節；直接取代整面的（地板本身就是木地板那一筆）照樣顯示、不標紅。"""
    document = room109_document()
    floor = next(entry["material"] for entry in BOSS_USER_ENTRIES if entry["id"] == FLOOR_WOOD)
    cast(dict[str, dict[str, list[dict[str, object]]]], document["scene"])["surface_zones_by_wall"]["floor"][0]["material"] = (
        cast(dict[str, object], floor))
    _write(tmp_path, document)
    _boss_library(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-x0").click()
        page.locator("#zone-show-all").check()
        expect(page.locator(f"#zone-fixed-x0-whole_wall optgroup[label='不列的：門、窗'] option[value={DOOR}]")).to_have_count(1)
        _choose(page, "#zone-fixed-x0-whole_wall", DOOR)
        page.locator("#zone-show-all").uncheck()
        expect(page.locator("#zone-fixed-x0-whole_wall")).to_have_value(DOOR)
        expect(page.locator("#zone-fixed-away-x0-whole_wall")).to_contain_text("不是前牆用的")
        expect(page.locator("#zone-fixed-away-x0-whole_wall")).to_contain_text("背後是空氣")

        _choose(page, "#zone-mode-x0-whole_wall", "keep_original")
        expect(page.locator("#zone-keep-x0-whole_wall")).to_have_text("原牆原樣，不貼東西")
        expect(page.locator("#zone-numbers-x0-whole_wall")).to_contain_text("原牆：固定阻抗")

        # 地板本身就是木地板那一筆（直接取代整面的固定阻抗，老闆房間就是這樣）：選單照樣是它，但它就是地板本身，不標紅說不是地板用的。
        page.locator("#elevation-wall-floor").click()
        expect(page.locator("#zone-fixed-floor-whole_wall")).to_have_value(FLOOR_WOOD)
        expect(page.locator(f"#zone-fixed-floor-whole_wall option[value={FLOOR_WOOD}]")).to_contain_text("整面就是它，不是貼上去的")
        expect(page.locator("#zone-fixed-away-floor-whole_wall")).to_have_count(0)
        assert watched.page_errors == [] and watched.console_errors == []


def _chips(page: Page, wall: str, key: str) -> list[str]:
    prefix = f"zone-chip-{wall}-{key}-"
    return [value.removeprefix(prefix) for value in cast(list[str], page.locator(f"#zone-allowed-chips-{wall}-{key} .zone-chip")
                                                         .evaluate_all("nodes => nodes.map(node => node.id)"))]


def test_search_starts_from_a_ready_list_and_presets_replace_it(tmp_path: Path, browser: Browser) -> None:
    """老闆 11-15「每點一個材料每點一個面都還要重新再勾一次 取消一堆」：改成交給電腦挑直接帶好一份清單（三種內建玻璃棉＋
    不貼），一鍵常用組合整份換掉；這一頁上一次改過的組合，下一區改成交給電腦挑時優先放（這一面用不了的拿掉）；
    地板沒有可調材料，只放「不貼」、不出現玻璃棉的組合。存檔一律寫明確清單。"""
    _write(tmp_path, room109_document())
    _boss_library(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-x0").click()
        _choose(page, "#zone-fixed-x0-whole_wall", "glass_wool_48k_50mm")
        _choose(page, "#zone-mode-x0-whole_wall", "search")
        three = ["glass_wool_48k", "glass_wool_24k", "glass_wool_96k", "none"]
        assert _chips(page, "x0", "whole_wall") == three
        assert _modes(page, "x0")["whole_wall"] == {"mode": "search", "allowed": three}
        expect(page.locator("#zone-chip-x0-whole_wall-glass_wool_24k")).to_contain_text("可調：厚 5～20 cm、空腔不調")
        expect(page.locator("#zone-chip-x0-whole_wall-none")).to_contain_text("保留原牆，電腦可以決定這裡不貼")
        # 常用組合：現在這一份就是「三種玻璃棉都比」（標現在用的）；這一區剛才指定貼 48K 50 mm，多一顆只決定它貼不貼。
        expect(page.locator("#zone-preset-x0-whole_wall-three_wools")).to_be_disabled()
        expect(page.locator("#zone-preset-x0-whole_wall-three_wools")).to_contain_text("現在用的")
        expect(page.locator("#zone-preset-x0-whole_wall-this_or_none")).to_have_text("只讓電腦決定：「48K 玻璃棉 50 mm 貼牆」貼或不貼")
        page.locator("#elevation-panel").screenshot(path=str(tmp_path / "search-ready-list.png"))
        page.locator("#zone-preset-x0-whole_wall-this_or_none").click()
        assert _modes(page, "x0")["whole_wall"]["allowed"] == ["glass_wool_48k_50mm", "none"]
        page.locator("#zone-preset-x0-whole_wall-wool48").click()
        assert _chips(page, "x0", "whole_wall") == ["glass_wool_48k", "none"]
        expect(page.locator("#zone-preset-x0-whole_wall-wool48")).to_be_disabled()
        # 改成不貼再改回來：這一區已經不是那一筆了，「只讓電腦決定：這一筆」那一顆不再出現；清單照上一次的組合。
        _choose(page, "#zone-mode-x0-whole_wall", "keep_original")
        _choose(page, "#zone-mode-x0-whole_wall", "search")
        expect(page.locator("#zone-preset-x0-whole_wall-this_or_none")).to_have_count(0)
        assert _chips(page, "x0", "whole_wall") == ["glass_wool_48k", "none"]

        # 下一面改成交給電腦挑：照這一頁上一次的組合（48K＋不貼）。
        page.locator("#elevation-wall-y0").click()
        _choose(page, "#zone-mode-y0-whole_wall", "search")
        assert _modes(page, "y0")["whole_wall"]["allowed"] == ["glass_wool_48k", "none"]
        # 地板：玻璃棉不是地板用的，上一次的組合拿掉它只剩「不貼」；玻璃棉的組合不出現。
        page.locator("#elevation-wall-floor").click()
        _choose(page, "#zone-mode-floor-whole_wall", "search")
        assert _chips(page, "floor", "whole_wall") == ["none"]
        expect(page.locator("#zone-presets-floor-whole_wall button")).to_have_count(0)
        _save(page, "room109")
        stored = json.loads((tmp_path / "designs" / "room109.json").read_text(encoding="utf-8"))["zones"]
        assert {wall: zones["whole_wall"] for wall, zones in stored.items()} == {
            "x0": {"mode": "search", "allowed": ["glass_wool_48k", "none"]},
            "y0": {"mode": "search", "allowed": ["glass_wool_48k", "none"]},
            "floor": {"mode": "search", "allowed": ["none"]}}
        assert watched.page_errors == [] and watched.console_errors == []


def test_the_ready_list_is_always_the_three_wools_even_with_more_tunable_materials(tmp_path: Path, browser: Browser) -> None:
    """起點要可預期（#1176，跟老闆講好「直接帶好三種玻璃棉＋不貼」）：材料庫多了一筆牆上能用的可調材料，改成交給電腦挑照樣
    只帶三種玻璃棉＋不貼；那一筆在「改清單」裡列著、沒勾，要用自己勾。"""
    _write(tmp_path, room109_document())
    extra = {**next(entry for entry in BOSS_USER_ENTRIES if entry["id"] == "demo109_gw48k_limited"),
             "id": "own_rock_wool_free", "label": "自填岩棉"}
    adapter: TypeAdapter[tuple[LibraryEntry, ...]] = TypeAdapter(tuple[LibraryEntry, ...])
    (tmp_path / "materials.json").write_text(dump_user_library(adapter.validate_python([*BOSS_USER_ENTRIES, extra])),
                                             encoding="utf-8")
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-x0").click()
        _choose(page, "#zone-mode-x0-whole_wall", "search")
        three = ["glass_wool_48k", "glass_wool_24k", "glass_wool_96k", "none"]
        assert _chips(page, "x0", "whole_wall") == three
        page.locator("#zone-allowed-summary-x0-whole_wall").click()
        expect(page.locator("#zone-allow-x0-whole_wall-own_rock_wool_free")).not_to_be_checked()
        page.locator("#zone-allow-x0-whole_wall-own_rock_wool_free").check()
        assert _modes(page, "x0")["whole_wall"]["allowed"] == [*three, "own_rock_wool_free"]
        assert watched.page_errors == [] and watched.console_errors == []


def test_the_bottom_message_follows_the_list_and_clears_on_another_face(tmp_path: Path, browser: Browser) -> None:
    """審查 11-15：底部那一行不准留「先放了 48K、不貼」卻畫著別的清單——改清單（×、常用組合）跟著寫現在的清單，換到別一面
    清掉；「要改就按」指這一區的「改清單」，這一區沒有常用組合（地板）就不提組合。"""
    _write(tmp_path, room109_document())
    _boss_library(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        note = page.locator("#zone-note")
        page.locator("#elevation-wall-x0").click()
        _choose(page, "#zone-mode-x0-whole_wall", "search")
        expect(note).to_contain_text("先放了 48K 玻璃棉、24K 玻璃棉、96K 玻璃棉、不貼；要改就按這一區的「改清單」或常用組合")
        page.locator("#zone-chip-remove-x0-whole_wall-glass_wool_48k").click()
        expect(note).to_contain_text("「整面」：清單改成 24K 玻璃棉、96K 玻璃棉、不貼")
        expect(note).not_to_contain_text("先放了")
        page.locator("#zone-preset-x0-whole_wall-wool48").click()
        expect(note).to_contain_text("清單改成 48K 玻璃棉、不貼")
        page.locator("#elevation-wall-floor").click()
        expect(note).to_have_text("")
        _choose(page, "#zone-mode-floor-whole_wall", "search")
        expect(note).to_contain_text("先放了 不貼；要改就按這一區的「改清單」")
        expect(note).not_to_contain_text("常用組合")
        assert watched.page_errors == [] and watched.console_errors == []


def test_the_last_combination_does_not_follow_into_another_scheme(tmp_path: Path, browser: Browser) -> None:
    """「上一次的組合」只在同一份方案裡帶（審查 11-15）：在這一份改成 48K＋不貼，打開另一份方案再改成交給電腦挑，
    照樣從三種玻璃棉＋不貼開始（同一份方案裡的別一面才照上一次的組合）。"""
    _write(tmp_path, room109_document())
    _write(tmp_path, room109_document("room109-b"))
    _boss_library(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        three = ["glass_wool_48k", "glass_wool_24k", "glass_wool_96k", "none"]
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-x0").click()
        _choose(page, "#zone-mode-x0-whole_wall", "search")
        page.locator("#zone-preset-x0-whole_wall-wool48").click()
        page.locator("#elevation-wall-y0").click()
        _choose(page, "#zone-mode-y0-whole_wall", "search")
        assert _chips(page, "y0", "whole_wall") == ["glass_wool_48k", "none"]  # 同一份方案：照上一次的組合
        _open_scheme(page, "room109-b")
        page.locator("#elevation-wall-x0").click()
        _choose(page, "#zone-mode-x0-whole_wall", "search")
        assert _chips(page, "x0", "whole_wall") == three
        assert watched.page_errors == [] and watched.console_errors == []


def _old_search_design(tmp_path: Path) -> Path:
    design = tmp_path / "designs" / "room109.json"
    design.parent.mkdir(exist_ok=True)
    design.write_text(json.dumps({"schema_version": "aosr.zone_design.v1", "scheme_id": "room109",
                                  "zones": {"x0": {"whole_wall": {"mode": "search"}}}}), encoding="utf-8")
    return design


def test_an_old_zone_without_a_list_shows_and_saves_what_the_server_means(tmp_path: Path, browser: Browser) -> None:
    """舊的交給搜尋區沒寫清單（null，伺服器的意思是材料庫全部可調材料＋不貼，示範房的也算）：打開照這個意思列成明確清單
    （示範房那一筆標不是這一面用的），存檔寫成明確清單；伺服器那一邊 null 的意思不改。"""
    _write(tmp_path, room109_document())
    _boss_library(tmp_path)
    design = _old_search_design(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-x0").click()
        everything = ["glass_wool_48k", "glass_wool_24k", "glass_wool_96k", "demo109_gw48k_limited", "none"]
        assert _chips(page, "x0", "whole_wall") == everything
        expect(page.locator("#zone-chip-x0-whole_wall-demo109_gw48k_limited")).to_contain_text("不是前牆用的")
        assert page.evaluate("zoneDesign.zones.x0.whole_wall") == {"mode": "search"}  # 打開不改設計，存檔才寫
        _save(page, "room109")
        assert json.loads(design.read_text(encoding="utf-8"))["zones"]["x0"]["whole_wall"] == {
            "mode": "search", "allowed": everything}
        assert watched.page_errors == [] and watched.console_errors == []


def test_an_old_zone_without_a_list_stays_without_one_when_the_library_is_unreadable(tmp_path: Path, browser: Browser) -> None:
    """材料庫讀不全（使用者材料庫壞了）：列不出伺服器那份全部可調材料是哪幾種，照講、不猜；存檔照舊不寫清單（不替伺服器縮小範圍）。
    這時候伺服器連換做法都不換（/api/zone-material 照講拒收、選單放回原樣），所以新改成交給電腦挑的起點清單輪不到。"""
    _write(tmp_path, room109_document())
    (tmp_path / "materials.json").write_text("{not json", encoding="utf-8")
    design = _old_search_design(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-x0").click()
        expect(page.locator("#zone-allowed-unknown-x0-whole_wall")).to_contain_text("材料庫沒讀全")
        page.locator("#elevation-wall-y0").click()
        with page.expect_response("**/api/zone-material") as response:
            page.locator("#zone-mode-y0-whole_wall").select_option("search")
        assert response.value.status == 400
        page.wait_for_function("!document.querySelector('main').inert")
        expect(page.locator("#zone-note")).to_contain_text("使用者材料庫讀不進來")
        expect(page.locator("#zone-mode-y0-whole_wall")).to_have_value("fixed")
        _save(page, "room109")
        assert json.loads(design.read_text(encoding="utf-8"))["zones"] == {"x0": {"whole_wall": {"mode": "search"}}}
        # 瀏覽器把那一次 400 記成一條載入失敗；除了它不准有別的錯。
        assert watched.page_errors == [] and [text for text in watched.console_errors if "status of 400" not in text] == []


def test_a_list_of_only_a_fixed_product_offers_buttons_instead_of_a_riddle(tmp_path: Path, browser: Browser) -> None:
    """老闆 11-15 看不懂「准用清單只有固定材料：搜尋要一個起點」：只放了固定成品時紅框白話寫電腦沒得挑，下面直接給按鈕——
    只有一筆就能改成指定貼它（清單跟著拿掉，伺服器不收固定區帶清單）、加上「不貼」、加上 48K 玻璃棉（地板用不了就不出現）。"""
    vpr = "melamine100_vpr_steel1mm_softfoam100_draft"
    _write(tmp_path, room109_document())
    _boss_library(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-x0").click()
        _choose(page, "#zone-mode-x0-whole_wall", "search")
        page.locator("#zone-allowed-summary-x0-whole_wall").click()
        page.locator(f"#zone-allow-x0-whole_wall-{vpr}").check()
        for choice in ("glass_wool_48k", "glass_wool_24k", "glass_wool_96k", "none"):
            page.locator(f"#zone-chip-remove-x0-whole_wall-{choice}").click()
        box = page.locator("#zone-allowed-problem-x0-whole_wall")
        expect(box).to_contain_text("只放了固定成品，電腦沒得挑。")
        expect(box).not_to_contain_text("准用")
        expect(box.locator("button")).to_have_text(
            ["就貼這一筆（改成指定貼一種）", "讓電腦決定貼不貼（加上「不貼」）", "讓電腦拿它跟 48K 玻璃棉比（加上 48K 玻璃棉）"])
        page.locator("#zone-allowed-summary-x0-whole_wall").click()  # 收起勾選表，截紅框
        page.locator("#elevation-panel").screenshot(path=str(tmp_path / "only-fixed.png"))

        page.locator("#zone-fix-add-none-x0-whole_wall").click()
        assert _modes(page, "x0")["whole_wall"]["allowed"] == [vpr, "none"]
        expect(box).to_have_count(0)
        page.locator("#zone-chip-remove-x0-whole_wall-none").click()
        page.locator("#zone-fix-add-wool-x0-whole_wall").click()
        assert _modes(page, "x0")["whole_wall"]["allowed"] == [vpr, "glass_wool_48k"]
        page.locator("#zone-chip-remove-x0-whole_wall-glass_wool_48k").click()
        with page.expect_response("**/api/zone-material"):
            page.locator("#zone-fix-only-x0-whole_wall").click()
        page.wait_for_function("!document.querySelector('main').inert")
        expect(page.locator("#zone-mode-x0-whole_wall")).to_have_value("fixed")
        expect(page.locator("#zone-fixed-x0-whole_wall")).to_have_value(vpr)
        assert _modes(page, "x0")["whole_wall"] == {"mode": "fixed"}
        _save(page, "room109")

        # 地板：48K 玻璃棉不是地板用的，那一顆不出現。
        page.locator("#elevation-wall-floor").click()
        _choose(page, "#zone-mode-floor-whole_wall", "search")
        page.locator("#zone-allowed-summary-floor-whole_wall").click()
        page.locator("#zone-allow-floor-whole_wall-carpet_on_concrete").check()
        page.locator("#zone-chip-remove-floor-whole_wall-none").click()
        expect(page.locator("#zone-allowed-problem-floor-whole_wall button")).to_have_text(
            ["就貼這一筆（改成指定貼一種）", "讓電腦決定貼不貼（加上「不貼」）"])
        assert watched.page_errors == [] and watched.console_errors == []


def _only_fixed(page: Page, wall: str, product: str) -> None:
    """把這一面整面一區改成交給電腦挑、清單只剩一筆固定成品（紅框那個樣子）。"""
    _choose(page, f"#zone-mode-{wall}-whole_wall", "search")
    page.locator(f"#zone-allowed-summary-{wall}-whole_wall").click()
    page.locator(f"#zone-allow-{wall}-whole_wall-{product}").check()
    for choice in ("glass_wool_48k", "glass_wool_24k", "glass_wool_96k", "none"):
        page.locator(f"#zone-chip-remove-{wall}-whole_wall-{choice}").click()
    expect(page.locator(f"#zone-allowed-problem-{wall}-whole_wall")).to_contain_text("只放了固定成品，電腦沒得挑")


def test_send_to_search_waits_until_every_red_box_is_handled(tmp_path: Path, browser: Browser) -> None:
    """紅框（只放了固定成品）還在的時候「拿去搜尋」按不下去，旁邊點名是哪幾區的紅框（別面的也算）；處理好一區少一區，
    全處理好自己恢復。直接叫送出也照樣擋、不存不跳頁。"""
    vpr = "melamine100_vpr_steel1mm_softfoam100_draft"
    _write(tmp_path, room109_document())
    _boss_library(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        front, side = cast(list[str], page.evaluate("[wallNames.x0, wallNames.y0]"))
        page.locator("#elevation-wall-x0").click()
        _only_fixed(page, "x0", vpr)
        expect(page.locator("#zone-to-search")).to_be_disabled()
        expect(page.locator("#zone-to-search-note")).to_have_text(
            f"先處理{front}「整面」的紅框（只放了固定成品，電腦沒得挑），處理好才能拿去搜尋")
        page.locator("#elevation-wall-y0").click()
        _only_fixed(page, "y0", vpr)
        note = page.locator("#zone-to-search-note")
        expect(note).to_contain_text(f"先處理{front}「整面」、{side}「整面」的紅框")
        expect(note).to_have_class("limit-red")
        # 停用的按鈕照樣看得清楚（審查 11-15：白字淺底太淡）：字與底的對比至少 4.5:1、不靠透明度變淡；跟紅字留間隔。
        look = cast(dict[str, float], page.locator("#zone-to-search").evaluate(CONTRAST))
        assert look["ratio"] >= 4.5 and look["opacity"] == 1, look
        gap = cast(float, page.evaluate("document.getElementById('zone-to-search-note').getBoundingClientRect().left"
                                        " - document.getElementById('zone-to-search').getBoundingClientRect().right"))
        assert gap >= 8, gap
        page.locator("#elevation-panel").screenshot(path=str(tmp_path / "send-blocked-by-red-box.png"))
        page.evaluate("sendToSearch()")
        expect(page.locator("#zone-note")).to_contain_text(f"先處理{front}「整面」、{side}「整面」的紅框")
        assert page.url.startswith(base) and "/gradient" not in page.url
        assert not (tmp_path / "designs" / "room109.json").exists()

        page.locator("#zone-fix-add-none-y0-whole_wall").click()
        expect(note).to_have_text(f"先處理{front}「整面」的紅框（只放了固定成品，電腦沒得挑），處理好才能拿去搜尋")
        expect(page.locator("#zone-to-search")).to_be_disabled()
        page.locator("#elevation-wall-x0").click()
        page.locator("#zone-fix-add-none-x0-whole_wall").click()
        expect(page.locator("#zone-to-search")).to_be_enabled()
        expect(note).to_have_text("會先照畫面儲存，再打開第三階段搜尋頁")
        assert watched.page_errors == [] and watched.console_errors == []


def _overflow(page: Page) -> dict[str, float]:
    """做法表、深度上限那一欄的右緣比立面圖面板右緣多出幾 px（0 以下＝塞得進面板）。"""
    return cast(dict[str, float], page.evaluate("""() => {
        const panel = document.getElementById('elevation-panel').getBoundingClientRect().right;
        const table = document.querySelector('.zone-table');
        const depth = table.querySelector('th:last-child');
        return {table: table.getBoundingClientRect().right - panel, depth: depth.getBoundingClientRect().right - panel};
    }"""))


def test_the_front_wall_table_fits_the_panel_with_long_product_names(tmp_path: Path, browser: Browser) -> None:
    """老闆的材料庫有幾筆名字很長的固定成品（VPR 那幾筆）：材料下拉選單不准照最長那一筆把表撐寬、把深度上限那一欄擠出面板；
    指定貼一種與交給電腦挑（方塊、常用組合按鈕、說明一行一行換）兩個樣子都要塞得進立面圖面板。"""
    _write(tmp_path, room109_document())
    _boss_library(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-x0").click()
        _drag(page, ROOM_109["Ly"], ROOM_109["Lz"], (0.02, 0.02), (1.0, 1.2))
        expect(page.locator("#zone-fixed-x0-z1 option", has_text="VPR").first).to_be_attached()
        assert all(value <= 0.5 for value in _overflow(page).values()), _overflow(page)
        _choose(page, "#zone-fixed-x0-z1", "melamine100_vpr_steel1mm_softfoam100_draft")
        _choose(page, "#zone-mode-x0-z1", "search")
        long_preset = page.locator("#zone-preset-x0-z1-this_or_none")
        expect(long_preset).to_contain_text("只讓電腦決定：「VPR")
        page.locator("#zone-allowed-summary-x0-z1").click()
        expect(page.locator("#zone-allowed-x0-z1")).to_be_visible()
        assert all(value <= 0.5 for value in _overflow(page).values()), _overflow(page)
        assert cast(float, long_preset.evaluate(
            "node => node.getBoundingClientRect().right - document.getElementById('elevation-panel').getBoundingClientRect().right"
        )) <= 0.5
        expect(page.locator("#depth-zone-x0-z1")).to_be_in_viewport()
        page.locator("#elevation-panel").screenshot(path=str(tmp_path / "front-fits.png"))
        assert watched.page_errors == [] and watched.console_errors == []


def test_zone_and_size_columns_never_break_inside_a_word(tmp_path: Path, browser: Browser) -> None:
    """審查 11-15：天花整面一區、交給電腦挑打開清單，「區」那一欄的「整面」斷成直的兩行。窄視窗（760）也不准在字中間斷：
    「整面」一行，「大小（公尺）」只准在括號前換行。"""
    _write(tmp_path, room109_document())
    _boss_library(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=760) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-ceiling").click()
        _choose(page, "#zone-mode-ceiling-whole_wall", "search")
        page.locator("#zone-allowed-summary-ceiling-whole_wall").click()
        row = page.locator("#zone-row-ceiling-whole_wall td")
        assert row.nth(0).evaluate(LINES) == ["整面"] and row.nth(1).evaluate(LINES) == ["整面"]
        assert page.locator("#zone-table-ceiling th").nth(1).evaluate(LINES) in (["大小（公尺）"], ["大小", "（公尺）"])
        page.locator("#elevation-panel").screenshot(path=str(tmp_path / "narrow-columns.png"))
        assert watched.page_errors == [] and watched.console_errors == []


def test_an_unknown_name_in_the_list_is_not_counted_as_a_fixed_product(tmp_path: Path, browser: Browser) -> None:
    """紅框只在清單每一種都是材料庫認得的固定成品時出現（#1176）：材料庫找不到的代號（使用者材料庫沒讀全時可能是可調材料）
    不當固定成品——不出紅框、不擋「拿去搜尋」，照「材料庫找不到」那一行講（開跑時伺服器點名）；拿掉它只剩固定成品才出紅框。"""
    _write(tmp_path, room109_document())
    design = tmp_path / "designs" / "room109.json"
    design.parent.mkdir(exist_ok=True)
    design.write_text(json.dumps({"schema_version": "aosr.zone_design.v1", "scheme_id": "room109", "zones": {
        "x0": {"whole_wall": {"mode": "search", "allowed": ["glass_wool_48k_50mm", "ghost_material"]}},
        "y0": {"whole_wall": {"mode": "search", "allowed": ["ghost_material"]}}}}), encoding="utf-8")
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        for wall in ("x0", "y0"):
            page.locator(f"#elevation-wall-{wall}").click()
            expect(page.locator(f"#zone-allowed-missing-{wall}-whole_wall")).to_contain_text("ghost_material")
            expect(page.locator(f"#zone-allowed-problem-{wall}-whole_wall")).to_have_count(0)
            expect(page.locator("#zone-to-search")).to_be_enabled()
        page.locator("#elevation-wall-x0").click()
        page.locator("#zone-chip-remove-x0-whole_wall-ghost_material").click()
        expect(page.locator("#zone-allowed-problem-x0-whole_wall")).to_contain_text("只放了固定成品，電腦沒得挑")
        expect(page.locator("#zone-to-search")).to_be_disabled()
        assert watched.page_errors == [] and watched.console_errors == []


def test_picking_what_the_ceiling_already_is_turns_into_keep_original(tmp_path: Path, browser: Browser) -> None:
    """#1175（老闆 11-15 天花選了跟現況同一筆吊頂，存成兩層吊頂疊在一起）：選到跟這一區原牆一模一樣的多層構造，
    網頁把做法改成「不貼，保留原牆」、說「選的跟原本一樣，當成不貼」，那一區放的還是原牆一份。"""
    ceiling = "own_room_ceiling_calsil_melamine40_on_board_gap210"
    built = next(entry["material"] for entry in BOSS_USER_ENTRIES if entry["id"] == ceiling)
    document = room109_document()
    cast(dict[str, dict[str, list[dict[str, object]]]], document["scene"])["surface_zones_by_wall"]["ceiling"][0]["material"] = (
        cast(dict[str, object], built))
    _write(tmp_path, document)
    _boss_library(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-ceiling").click()
        # 整片天花構造平常不列成貼上去的處理；原牆認得出是哪一筆就寫它的名字。
        assert ceiling not in _values(page, "#zone-fixed-ceiling-whole_wall option")
        expect(page.locator("#zone-fixed-ceiling-whole_wall option").first).to_contain_text(
            "現在：沒貼東西（原牆「天花 矽酸鈣板 6 mm＋美耐皿泡棉 40 mm 平放板背＋21 cm 空腔（剛背）」原樣）")
        page.locator("#zone-show-all").check()
        _choose(page, "#zone-fixed-ceiling-whole_wall", ceiling)
        expect(page.locator("#zone-mode-ceiling-whole_wall")).to_have_value("keep_original")
        expect(page.locator("#zone-note")).to_contain_text("選的跟原本一樣，當成不貼")
        # 那一列自己也寫為什麼（八區的牆上，底部那一行常在畫面外）。
        expect(page.locator("#zone-same-ceiling-whole_wall")).to_contain_text("跟原本一樣，當成不貼")
        assert page.evaluate("structuredClone(scheme.scene.surface_zones_by_wall.ceiling[0].material)") == built
        assert _modes(page, "ceiling") == {"whole_wall": {"mode": "keep_original"}}
        page.locator("#elevation-panel").screenshot(path=str(tmp_path / "same-as-original.png"))
        # 那一區再改做法：那一句拿掉，改回不貼也不再出現。
        _choose(page, "#zone-mode-ceiling-whole_wall", "search")
        _choose(page, "#zone-mode-ceiling-whole_wall", "keep_original")
        expect(page.locator("#zone-same-ceiling-whole_wall")).to_have_count(0)
        assert watched.page_errors == [] and watched.console_errors == []
