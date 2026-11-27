"""真瀏覽器考每一區的材料清單（#1265 單元 3 之二第二趟起舊做法表拆了：改在「這間房貼什麼」的小框選、輸入頁第 2 段右欄讀；#1176，老闆 11-15「109示範又不知道是娥麼這也能當成一個材料」「都不知道在講甚麼」）：
選單與小框只列這一面能貼的（門、窗、地板、整片吊頂、另一間示範房的不出現），指定貼一種的下拉可以打開「顯示全部材料」分組列出；
這一區現在放的那一筆就算不是這一面用的也照樣顯示；用語是白話（做法三種、原牆不寫阻抗數字、可調寫得出調什麼）。
材料庫用自己編的合成 12 筆（``_sample_material_library``），房間用 109 號房的長寬高。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
from playwright.sync_api import Browser, Page, expect
from pydantic import TypeAdapter

from aosr.materials.library import LibraryEntry, dump_user_library
import re

from tests.engine._zone_material_steps import pick_fixed_row, row_options
from tests.engine._input_page_steps import new_from_template
from tests.engine._zone_panel_steps import (apply_and_return, choose_fixed, choose_mode, edit_candidates, expect_mode, fixed_value,
                                            listed, material_values, mode_words, open_alone, open_panel)
from tests.engine._sample_material_library import SAMPLE_USER_ENTRIES
from tests.engine._gui_cache import gui_startup_identity_memo as gui_startup_identity_memo
from tests.engine._surface_zone_cases import room109_document
from tests.engine._zone_naming_room import ZONE_NAMING_SCHEME, write_zone_naming_library
from tests.engine.test_gui_browser import _open, _serve, browser as browser
from tests.engine.test_gui_zone_elevation_browser import _open_scheme, _write
from tests.engine.test_gui_zone_modes_browser import _modes, _save

HIDDEN_ON_WALLS = ("hall_door_hollow_core", "side_window_glass_12mm_single", "ceiling_gypsum9mm_gap300",
                   "perforated_gypsum_8_18r_backed_gw30_ceiling", "carpet_on_concrete", "joist_floor_plank_fixed",
                   "demo109_rockwool60k_capped", "demo109_measured_pet35_synthetic_b7")
DOOR = "hall_door_hollow_core"
FLOOR_WOOD = "joist_floor_plank_fixed"


PLAIN_PRODUCT = "fabric_pet_a_gap100"  # 內建的一筆固定成品：牆面能貼、不是哪一種可調玻璃棉包含的、也不是「鋼板直接貼在棉上」的構造


def _sample_library(tmp_path: Path, *extra: dict[str, object]) -> None:
    adapter: TypeAdapter[tuple[LibraryEntry, ...]] = TypeAdapter(tuple[LibraryEntry, ...])
    (tmp_path / "materials.json").write_text(dump_user_library(adapter.validate_python([*SAMPLE_USER_ENTRIES, *extra])), encoding="utf-8")


def _values(page: Page, selector: str) -> list[str]:
    return cast(list[str], page.locator(selector).evaluate_all("nodes => nodes.map(node => node.value)"))


# 字色與底色的對比（WCAG 相對亮度）、這一格自己的不透明度。
CONTRAST = """node => {
    const style = getComputedStyle(node);
    const rgb = text => text.match(/[\\d.]+/g).slice(0, 3).map(Number);
    const channel = value => { const c = value / 255; return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4; };
    const luminance = ([r, g, b]) => 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b);
    const [light, dark] = [luminance(rgb(style.color)), luminance(rgb(style.backgroundColor))].sort((a, b) => b - a);
    return {ratio: (light + 0.05) / (dark + 0.05), opacity: Number(style.opacity)};
}"""


def test_menus_list_only_what_this_face_can_take_and_show_all_reveals_the_rest(tmp_path: Path, browser: Browser) -> None:
    _write(tmp_path, room109_document())
    _sample_library(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-x0").click()
        assert mode_words(page, "x0", "whole_wall") == ["指定貼一種", "不貼，保留原樣", "由程式最佳化"]
        # 指定貼一種：牆上能貼的固定成品（含合成的 VPR），門、窗、天花構造、地毯、示範房一筆都沒有；原牆不寫阻抗數字。
        fixed = material_values(page, "x0", "whole_wall")
        assert "glass_wool_48k_50mm" in fixed and "pet60_vpr_steel1p2mm_foam90_sample" in fixed
        assert not set(HIDDEN_ON_WALLS) & set(fixed)
        expect(page.locator("#zone-numbers-x0-whole_wall")).to_contain_text("原牆：固定阻抗")
        # 舊表「顯示全部材料」那一段：小框那一題蓋得到（test_gui_zone_panel_show_all_browser.py 的
        # test_the_panel_lists_only_what_this_face_can_take_and_show_all_reveals_the_rest，乙）。
        choose_mode(page, "x0", "whole_wall", "search")
        assert _chips(page, "x0", "whole_wall") == ["glass_wool_48k", "glass_wool_24k", "glass_wool_96k", "none"]
        # 舊表方塊上「可調：…」那幾個字跟著表走：48K、24K 兩句同字在那一頁清單卡（test_gui_zone_overview_browser.py 的
        # test_a_design_with_a_search_rule_shows_the_list_card_and_the_zones_that_chose_differently，乙）；96K 那一句、
        # 「不含（可調厚度與空腔）」沒有題逐字考（丙，回報第 5 節）。
        open_panel(page, "x0", "whole_wall")
        page.locator("#zone-panel-picker-search").focus()
        for choice in HIDDEN_ON_WALLS:
            expect(page.locator(f"#zone-panel-pick-{choice}")).to_have_count(0)
        expect(page.locator("#zone-panel-pick-fabric_pet_a_gap100 .zv-description")).to_have_text("整塊貼、固定 10 cm")
        expect(page.locator("#zone-panel-pick-slotted18_pet45_vpr_steel2p25mm_foam70_sample .zv-description")).to_have_text(
            "整塊貼、固定 13.3 cm（不含薄板）")
        page.locator("#zone-panel-cancel").click()
        page.locator("#zone-page-close").click()
        page.locator("#elevation-wall-floor").click()
        assert material_values(page, "floor", "whole_wall") == ["", "carpet_on_concrete"]
        choose_mode(page, "floor", "whole_wall", "search")
        assert _chips(page, "floor", "whole_wall") == ["none"]
        open_panel(page, "floor", "whole_wall")
        expect(page.locator("#zone-panel-form")).to_contain_text("這裡沒有可調的材料：程式只能從「不貼」開始")
        page.locator("#zone-panel-picker-search").focus()
        expect(page.locator("#zone-panel-picker button")).to_have_count(1)
        expect(page.locator("#zone-panel-pick-carpet_on_concrete")).to_be_enabled()
        assert watched.page_errors == [] and watched.console_errors == []


def test_a_zone_holding_a_material_from_another_group_still_shows_it(tmp_path: Path, browser: Browser) -> None:
    """選單只列這一面能貼的，但這一區現在放的就是門（打開「顯示全部材料」選的）：關掉之後選單照樣是它、寫出它不是這一面用的；
    保留原牆的區寫白話、數字收在細節；直接取代整面的（地板本身就是木地板那一筆）照樣顯示、不標紅。"""
    document = room109_document()
    floor = next(entry["material"] for entry in SAMPLE_USER_ENTRIES if entry["id"] == FLOOR_WOOD)
    cast(dict[str, dict[str, list[dict[str, object]]]], document["scene"])["surface_zones_by_wall"]["floor"][0]["material"] = (
        cast(dict[str, object], floor))
    _write(tmp_path, document)
    _sample_library(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-x0").click()
        # 舊表「顯示全部材料」分組、選了門之後下拉照樣是它與「不是前牆用的…背後是空氣」：小框同名題蓋得到（乙）。
        choose_fixed(page, "x0", "whole_wall", DOOR, show_all=True)
        assert fixed_value(page, "x0", "whole_wall") == DOOR

        choose_mode(page, "x0", "whole_wall", "keep_original")
        # 舊表「不貼」那一格（#zone-keep-）→ 右欄那一區那一句的開頭（同一支 keepWords）。
        expect(page.locator("#wall-zone-what-x0-whole_wall")).to_have_text(re.compile(r"^原牆原樣，不貼東西。"))
        expect(page.locator("#zone-numbers-x0-whole_wall")).to_contain_text("原牆：固定阻抗")

        # 地板本身就是木地板那一筆（直接取代整面的固定阻抗，有的房間地板就是這樣）：選單照樣是它，但它就是地板本身，不標紅說不是地板用的。
        page.locator("#elevation-wall-floor").click()
        # 待主對話定（回報第 8 節）：舊表下拉停在 FLOOR_WOOD（地板本身就是那一筆）；小框的材料下拉不停在它（那一區沒貼東西，
        # 下拉停在「選要貼哪一種」）。這裡改讀右欄那一句寫出那一筆的名字（認得出是哪一筆），沒有放寬成不考。
        label = cast(str, page.evaluate("id => libraryEntry(id).label", FLOOR_WOOD))
        expect(page.locator("#wall-zone-what-floor-whole_wall")).to_contain_text(label)
        # 舊表下拉那一項的「整面就是它，不是貼上去的」跟著表走（丙）；不標紅由小框同名題（#zone-panel-away-floor-whole_wall 沒有）考（乙）。
        assert watched.page_errors == [] and watched.console_errors == []


def _chips(page: Page, wall: str, key: str) -> list[str]:
    """舊做法表那一格的方塊 → 小框的候選列（照畫出來的順序；只看，不套用）。"""
    return listed(page, wall, key)


def test_search_starts_from_a_ready_list_and_panel_edits_carry_to_the_next_face(tmp_path: Path, browser: Browser) -> None:
    """老闆 11-15「每點一個材料每點一個面都還要重新再勾一次 取消一堆」：改成由程式最佳化直接帶好一份清單（三種內建玻璃棉＋
    不貼），在小框調成同一份清單；這一頁上一次改過的組合，下一區改成由程式最佳化時優先放（這一面用不了的拿掉）；
    地板沒有可調材料，只放「不貼」、不出現玻璃棉的組合。存檔一律寫明確清單。"""
    _write(tmp_path, room109_document())
    _sample_library(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-x0").click()
        choose_fixed(page, "x0", "whole_wall", "glass_wool_48k_50mm")
        choose_mode(page, "x0", "whole_wall", "search")
        three = ["glass_wool_48k", "glass_wool_24k", "glass_wool_96k", "none"]
        assert _chips(page, "x0", "whole_wall") == three
        assert _modes(page, "x0")["whole_wall"] == {"mode": "search", "allowed": three}
        # 舊表方塊上的說明字跟著表走：24K 那一句同字在那一頁清單卡（test_gui_zone_overview_browser.py 同上，乙）；
        # 「不貼」方塊「保留原樣，程式可以決定這裡不貼」只有舊表寫（丙）。
        expect(page.locator("[id^=zone-preset]")).to_have_count(0)
        # 原本的「這一筆或不貼」：先讓小框加得進固定 48K，再加不貼，保留存檔的先後。
        edit_candidates(page, "x0", "whole_wall", remove=("glass_wool_48k", "none"), add=("glass_wool_48k_50mm",))
        edit_candidates(page, "x0", "whole_wall", remove=("glass_wool_24k", "glass_wool_96k"), add=("none",))
        assert _modes(page, "x0")["whole_wall"]["allowed"] == ["glass_wool_48k_50mm", "none"]
        # 回到「48K 或不貼」：先留下 48K，再加不貼。
        edit_candidates(page, "x0", "whole_wall", remove=("none",), add=("glass_wool_48k",), only="glass_wool_48k")
        edit_candidates(page, "x0", "whole_wall", add=("none",))
        assert _chips(page, "x0", "whole_wall") == ["glass_wool_48k", "none"]
        # 改成不貼再改回來：這一區已經不是那一筆了，「只讓程式決定：這一筆」那一顆不再出現；清單照上一次的組合。
        choose_mode(page, "x0", "whole_wall", "keep_original")
        choose_mode(page, "x0", "whole_wall", "search")
        expect(page.locator("#zone-preset-x0-whole_wall-this_or_none")).to_have_count(0)
        assert _chips(page, "x0", "whole_wall") == ["glass_wool_48k", "none"]

        # 下一面改成由程式最佳化：照這一頁上一次的組合（48K＋不貼）。
        page.locator("#elevation-wall-y0").click()
        choose_mode(page, "y0", "whole_wall", "search")
        assert _modes(page, "y0")["whole_wall"]["allowed"] == ["glass_wool_48k", "none"]
        # 地板：玻璃棉不是地板用的，上一次的組合拿掉它只剩「不貼」；玻璃棉的組合不出現。
        page.locator("#elevation-wall-floor").click()
        choose_mode(page, "floor", "whole_wall", "search")
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
    """起點要可預期（#1176，跟老闆講好「直接帶好三種玻璃棉＋不貼」）：材料庫多了一筆牆上能用的可調材料，改成由程式最佳化照樣
    只帶三種玻璃棉＋不貼；那一筆要用就到小框加進來。"""
    _write(tmp_path, room109_document())
    extra = {**next(entry for entry in SAMPLE_USER_ENTRIES if entry["id"] == "demo109_rockwool60k_capped"),
             "id": "own_rock_wool_free", "label": "自填岩棉"}
    adapter: TypeAdapter[tuple[LibraryEntry, ...]] = TypeAdapter(tuple[LibraryEntry, ...])
    (tmp_path / "materials.json").write_text(dump_user_library(adapter.validate_python([*SAMPLE_USER_ENTRIES, extra])),
                                             encoding="utf-8")
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-x0").click()
        choose_mode(page, "x0", "whole_wall", "search")
        three = ["glass_wool_48k", "glass_wool_24k", "glass_wool_96k", "none"]
        assert _chips(page, "x0", "whole_wall") == three  # 也就是沒有 own_rock_wool_free
        edit_candidates(page, "x0", "whole_wall", add=("own_rock_wool_free",))
        assert _modes(page, "x0")["whole_wall"]["allowed"] == [*three, "own_rock_wool_free"]
        assert watched.page_errors == [] and watched.console_errors == []


def test_the_bottom_message_follows_the_list_and_clears_on_another_face(tmp_path: Path, browser: Browser) -> None:
    """套用小框與換面後都不殘留上一份清單的提示。"""
    _write(tmp_path, room109_document())
    _sample_library(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        # 舊表底下那一行（#zone-note）跟著表拆了：小框的回話寫在那一頁結果列上面那一行（#zone-page-note），回輸入頁前讀；
        # 舊表改做法那一句（「先放了…；要改就按這一列的「改這一區 ▸」」）只有舊表那一路會講，跟著表走（丙）。
        note = page.locator("#zone-page-note")
        page.locator("#elevation-wall-x0").click()
        choose_mode(page, "x0", "whole_wall", "search")
        edit_candidates(page, "x0", "whole_wall", remove=("glass_wool_48k",), said="1 區改成「由程式最佳化」")
        assert _chips(page, "x0", "whole_wall") == ["glass_wool_24k", "glass_wool_96k", "none"]
        page.locator("#elevation-wall-floor").click()
        page.locator("#zone-page-open").click()
        expect(page.locator("#zone-map")).to_be_visible()
        expect(note).to_have_text("")  # 換面就拿掉上一份清單的回話
        page.locator("#zone-page-close").click()
        choose_mode(page, "floor", "whole_wall", "search")
        assert watched.page_errors == [] and watched.console_errors == []


def test_the_last_combination_does_not_follow_into_another_scheme(tmp_path: Path, browser: Browser) -> None:
    """「上一次的組合」只在同一份方案裡帶（審查 11-15）：在這一份改成 48K＋不貼，打開另一份方案再改成由程式最佳化，
    照樣從三種玻璃棉＋不貼開始（同一份方案裡的別一面才照上一次的組合）；從範本開一份新的也一樣從頭。"""
    _write(tmp_path, room109_document())
    _write(tmp_path, room109_document("room109-b"))
    _sample_library(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        three = ["glass_wool_48k", "glass_wool_24k", "glass_wool_96k", "none"]
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-x0").click()
        choose_mode(page, "x0", "whole_wall", "search")
        edit_candidates(page, "x0", "whole_wall", only="glass_wool_48k")
        page.locator("#elevation-wall-y0").click()
        choose_mode(page, "y0", "whole_wall", "search")
        assert _chips(page, "y0", "whole_wall") == ["glass_wool_48k", "none"]  # 同一份方案：照上一次的組合
        _open_scheme(page, "room109-b")
        page.locator("#elevation-wall-x0").click()
        choose_mode(page, "x0", "whole_wall", "search")
        assert _chips(page, "x0", "whole_wall") == three
        # 這一份也改成 48K＋不貼，再從範本開一份新的（#1265 單元 5b 取代載入範本）：一樣從頭。
        edit_candidates(page, "x0", "whole_wall", only="glass_wool_48k")
        new_from_template(page, "fresh-room")
        page.locator("#elevation-wall-x0").click()
        choose_mode(page, "x0", "whole_wall", "search")
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
    _sample_library(tmp_path)
    design = _old_search_design(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-x0").click()
        everything = ["glass_wool_48k", "glass_wool_24k", "glass_wool_96k", "demo109_rockwool60k_capped", "none"]
        assert _chips(page, "x0", "whole_wall") == everything
        # 舊表方塊上「不是前牆用的」那一句：小框候選列沒有對應的字（回報第 8 節，待主對話定要不要補到小框），這一行跟著表走。
        assert page.evaluate("zoneDesign.zones.x0.whole_wall") == {"mode": "search"}  # 打開不改設計，存檔才寫
        _save(page, "room109")
        assert json.loads(design.read_text(encoding="utf-8"))["zones"]["x0"]["whole_wall"] == {
            "mode": "search", "allowed": everything}
        assert watched.page_errors == [] and watched.console_errors == []


def test_an_old_zone_without_a_list_stays_without_one_when_the_library_is_unreadable(tmp_path: Path, browser: Browser) -> None:
    """材料庫讀不全（使用者材料庫壞了）：列不出伺服器那份全部可調材料是哪幾種，照講、不猜；存檔照舊不寫清單（不替伺服器縮小範圍）。
    這時候伺服器連換做法都不換（/api/zone-material 照講拒收、選單放回原樣），所以新改成由程式最佳化的起點清單輪不到。"""
    _write(tmp_path, room109_document())
    (tmp_path / "materials.json").write_text("{not json", encoding="utf-8")
    design = _old_search_design(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-x0").click()
        # 舊表那一格（#zone-allowed-unknown-）→ 右欄那一區那一句（改字：「程式從…裡選」→「從…裡挑」）。
        expect(page.locator("#wall-zone-what-x0-whole_wall")).to_contain_text("從全部可調材料與「不貼」裡挑（材料庫沒讀全，列不出是哪幾種）")
        page.locator("#elevation-wall-y0").click()
        # 換做法改從小框（舊表那一路打 /api/zone-material，小框打 /api/zone-material/batch）。
        open_alone(page, "y0", "whole_wall")
        page.locator("#zone-panel-mode-search").check()
        with page.expect_response("**/api/zone-material/batch") as response:
            page.locator("#zone-panel-apply").click()
        assert response.value.status == 400
        page.wait_for_function("() => !document.getElementById('zone-page').inert && !inputBusy")
        expect(page.locator("#zone-page-note")).to_contain_text("使用者材料庫讀不進來")
        page.locator("#zone-page-close").click()
        expect_mode(page, "y0", "whole_wall", "fixed")
        _save(page, "room109")
        assert json.loads(design.read_text(encoding="utf-8"))["zones"] == {"x0": {"whole_wall": {"mode": "search"}}}
        # 瀏覽器把那一次 400 記成一條載入失敗；除了它不准有別的錯。
        assert watched.page_errors == [] and [text for text in watched.console_errors if "status of 400" not in text] == []


def _open_only_fixed(page: Page, *walls: str) -> None:
    """載入考卷自編的舊清單。小框會擋下只剩固定成品的草稿，紅框考的是既存資料。"""
    design = {"schema_version": "aosr.zone_design.v1", "scheme_id": "room109",
              "zones": {wall: {"whole_wall": {"mode": "search", "allowed": ["carpet_on_concrete" if wall == "floor" else PLAIN_PRODUCT]}}
                        for wall in walls}}
    page.route("**/api/designs/room109", lambda route: route.fulfill(json={"design": design, "problems": []}), times=1)
    before = page.evaluate("zoneDesignGeneration")
    _open_scheme(page, "room109")
    page.wait_for_function("before => zoneDesignGeneration > before && !inputBusy && zoneLimitsShown === zoneLimitsGeneration", arg=before)


def test_a_fixed_only_list_points_to_the_panel_for_all_three_repairs(tmp_path: Path, browser: Browser) -> None:
    _write(tmp_path, room109_document())
    _sample_library(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        for addition in ("none", "glass_wool_48k"):
            _open_only_fixed(page, "x0")
            page.locator("#elevation-wall-x0").click()
            # 舊表那一區的紅框（#zone-allowed-problem-）跟著表走；那一頁的紅框與三條修法由 test_gui_zone_picker_browser.py
            # 與 test_gui_zone_material_groups_browser.py 的「只放固定成品」那幾題考（乙）。輸入頁還在的是頁底那顆鈕旁邊點名紅框。
            send = page.locator("#zone-to-search-note")
            expect(send).to_contain_text("的紅框（只放了固定成品，程式沒得選）")
            edit_candidates(page, "x0", "whole_wall", add=(addition,))
            assert _modes(page, "x0")["whole_wall"]["allowed"] == [PLAIN_PRODUCT, addition]
            expect(send).not_to_contain_text("紅框")
        _open_only_fixed(page, "x0")
        page.locator("#elevation-wall-x0").click()
        open_panel(page, "x0", "whole_wall")
        pick_fixed_row(page, PLAIN_PRODUCT)
        apply_and_return(page)
        expect_mode(page, "x0", "whole_wall", "fixed")
        assert fixed_value(page, "x0", "whole_wall") == PLAIN_PRODUCT
        assert _modes(page, "x0")["whole_wall"] == {"mode": "fixed"}
        _save(page, "room109")
        _open_only_fixed(page, "floor")
        page.locator("#elevation-wall-floor").click()
        expect(page.locator("#zone-to-search-note")).to_contain_text("的紅框（只放了固定成品，程式沒得選）")
        open_panel(page, "floor", "whole_wall")
        page.locator("#zone-panel-picker-search").focus()
        expect(page.locator("#zone-panel-pick-glass_wool_48k")).to_have_count(0)
        page.locator("#zone-panel-picker-search").press("Escape")
        row_options(page, "none")
        page.locator("#zone-panel-addto-none").click()
        apply_and_return(page)
        assert _modes(page, "floor")["whole_wall"]["allowed"] == ["carpet_on_concrete", "none"]
        assert watched.page_errors == [] and watched.console_errors == []


def test_send_to_search_waits_until_every_red_box_is_handled(tmp_path: Path, browser: Browser) -> None:
    """紅框（只放了固定成品）還在的時候「拿去搜尋」按不下去，旁邊點名是哪幾區的紅框（別面的也算）；處理好一區少一區，
    全處理好自己恢復。直接叫送出也照樣擋、不存不跳頁。"""
    _write(tmp_path, room109_document())
    _sample_library(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_only_fixed(page, "x0")
        front, side = cast(list[str], page.evaluate("[wallNames.x0, wallNames.y0]"))
        page.locator("#elevation-wall-x0").click()
        expect(page.locator("#zone-to-search")).to_be_disabled()
        expect(page.locator("#zone-to-search-note")).to_have_text(
            f"先處理{front}「整面」的紅框（只放了固定成品，程式沒得選），處理好才能拿去搜尋")
        _open_only_fixed(page, "x0", "y0")
        page.locator("#elevation-wall-y0").click()
        note = page.locator("#zone-to-search-note")
        expect(note).to_contain_text(f"先處理{front}「整面」、{side}「整面」的紅框")
        expect(note).to_have_class("limit-red")
        # 停用的按鈕照樣看得清楚（審查 11-15：白字淺底太淡）：字與底的對比至少 4.5:1、不靠透明度變淡；跟紅字留間隔。
        # 找按鈕和量顏色寫在同一段裡：伺服器回應晚到會整塊重畫、換一顆新按鈕，先拿到的那顆量到一半就不在頁面上了（顏色變空字串）。
        look = cast(dict[str, float], page.evaluate(f"({CONTRAST})(document.getElementById('zone-to-search'))"))
        assert look["ratio"] >= 4.5 and look["opacity"] == 1, look
        gap = cast(float, page.evaluate("document.getElementById('zone-to-search-note').getBoundingClientRect().top"
                                        " - document.getElementById('zone-to-search').getBoundingClientRect().bottom"))
        assert gap >= 8, gap
        page.locator("#elevation-panel").screenshot(path=str(tmp_path / "send-blocked-by-red-box.png"))
        page.evaluate("sendToSearch()")
        # 舊表底下那一行拆了：送出那一路的回話寫在頁底那顆鈕底下（searchNote → #zone-to-search-note）。
        expect(page.locator("#zone-to-search-note")).to_contain_text(f"先處理{front}「整面」、{side}「整面」的紅框")
        assert page.url.startswith(base) and "/gradient" not in page.url
        assert not (tmp_path / "designs" / "room109.json").exists()

        edit_candidates(page, "y0", "whole_wall", add=("none",))
        expect(note).to_have_text(f"先處理{front}「整面」的紅框（只放了固定成品，程式沒得選），處理好才能拿去搜尋")
        expect(page.locator("#zone-to-search")).to_be_disabled()
        page.locator("#elevation-wall-x0").click()
        edit_candidates(page, "x0", "whole_wall", add=("none",))
        expect(page.locator("#zone-to-search")).to_be_enabled()
        expect(note).to_have_text("交出黃色的 2 區；下一頁按「開始搜尋」才開始算")
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
            # 舊表「清單裡有材料庫找不到的：…」那一行跟著表走（丙）；右欄那一區那一句照樣點名它。紅框沒有＝頁底不點名紅框。
            expect(page.locator(f"#wall-zone-what-{wall}-whole_wall")).to_contain_text("ghost_material")
            expect(page.locator("#zone-to-search-note")).not_to_contain_text("紅框")
            expect(page.locator("#zone-to-search")).to_be_enabled()
        page.locator("#elevation-wall-x0").click()
        open_panel(page, "x0", "whole_wall")
        row_options(page, "ghost_material")
        page.locator("#zone-panel-remove-ghost_material").click()
        expect(page.locator("#zone-panel-problem")).to_contain_text("只放了固定成品，程式沒得選")
        expect(page.locator("#zone-panel-apply")).to_be_disabled()
        assert watched.page_errors == [] and watched.console_errors == []


def test_picking_what_the_ceiling_already_is_turns_into_keep_original(tmp_path: Path, browser: Browser) -> None:
    """#1175（老闆 11-15 天花選了跟現況同一筆吊頂，存成兩層吊頂疊在一起）：選到跟這一區原牆一模一樣的多層構造，
    網頁把做法改成「不貼，保留原樣」、說「選的跟原本一樣，當成不貼」，那一區放的還是原本那一份。"""
    ceiling = "ceiling_fibrecement8mm_wool30_on_board_gap180"
    built = next(entry["material"] for entry in SAMPLE_USER_ENTRIES if entry["id"] == ceiling)
    document = room109_document()
    cast(dict[str, dict[str, list[dict[str, object]]]], document["scene"])["surface_zones_by_wall"]["ceiling"][0]["material"] = (
        cast(dict[str, object], built))
    _write(tmp_path, document)
    _sample_library(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-ceiling").click()
        # 整片天花構造平常不列成貼上去的處理（小框同名題蓋得到，乙）；原牆認得出是哪一筆就寫它的名字
        # （舊表下拉第一項「現在：沒貼東西（…）」→ 右欄那一區那一句，改字：拿掉「現在：」）。
        expect(page.locator("#wall-zone-what-ceiling-whole_wall")).to_contain_text(
            "沒貼東西（原本的天花「天花 纖維水泥板 8 mm＋岩棉毯 30 mm 平鋪在板背＋18 cm 空腔（剛背）」原樣）")
        choose_fixed(page, "ceiling", "whole_wall", ceiling, show_all=True, said_part="選的跟原本一樣，當成不貼")
        expect_mode(page, "ceiling", "whole_wall", "keep_original")
        # 那一列自己也寫為什麼（舊表那一列的新家：第 2 段右欄那一區，id 照舊）。
        expect(page.locator("#zone-same-ceiling-whole_wall")).to_contain_text("跟原本一樣，當成不貼")
        assert page.evaluate("structuredClone(scheme.scene.surface_zones_by_wall.ceiling[0].material)") == built
        assert _modes(page, "ceiling") == {"whole_wall": {"mode": "keep_original"}}
        page.locator("#elevation-panel").screenshot(path=str(tmp_path / "same-as-original.png"))
        # 那一區再改做法：那一句拿掉，改回不貼也不再出現。
        choose_mode(page, "ceiling", "whole_wall", "search")
        choose_mode(page, "ceiling", "whole_wall", "keep_original")
        expect(page.locator("#zone-same-ceiling-whole_wall")).to_have_count(0)
        assert watched.page_errors == [] and watched.console_errors == []



@pytest.mark.parametrize("change", ["kind", "preset"])
def test_the_same_as_original_remark_goes_once_the_original_ceiling_changes(tmp_path: Path, browser: Browser, change: str) -> None:
    """「選的跟原本一樣，當成不貼」那一句（#1265 單元 3 之二收尾追加第 5 條）：只在那一區現在跟記下它的時候一樣才寫——
    之後在輸入頁「換原本的牆」把天花換成固定阻抗（kind）、或換一套構造預設（preset），原本那一面變了，那一句就不寫
    （做法照樣是不貼，主句照新的原本那一面講）。"""
    ceiling = "ceiling_fibrecement8mm_wool30_on_board_gap180"
    built = next(entry["material"] for entry in SAMPLE_USER_ENTRIES if entry["id"] == ceiling)
    document = room109_document()
    cast(dict[str, dict[str, list[dict[str, object]]]], document["scene"])["surface_zones_by_wall"]["ceiling"][0]["material"] = (
        cast(dict[str, object], built))
    _write(tmp_path, document)
    _sample_library(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-ceiling").click()
        choose_fixed(page, "ceiling", "whole_wall", ceiling, show_all=True, said_part="選的跟原本一樣，當成不貼")
        same = page.locator("#zone-same-ceiling-whole_wall")
        expect(same).to_contain_text("跟原本一樣，當成不貼")
        page.locator("#edit-wall-original").click()
        if change == "kind":
            page.locator("#material-kind-ceiling").select_option("constant_impedance")
        else:
            page.locator("#preset-ceiling").select_option("glass_wool_48k_50mm_gap50")
        page.wait_for_function("!document.querySelector('main').inert && !inputBusy && zoneLimitsShown === zoneLimitsGeneration")
        assert page.evaluate("structuredClone(scheme.scene.surface_zones_by_wall.ceiling[0].material)") != built
        expect_mode(page, "ceiling", "whole_wall", "keep_original")
        expect(same).to_have_count(0)
        assert watched.page_errors == [] and watched.console_errors == []

def test_the_same_as_original_remark_goes_while_a_layer_of_the_original_ceiling_is_typed(tmp_path: Path, browser: Browser) -> None:
    """#1265 單元 3 之二收尾第二趟 B2：「換原本的牆」直接改天花構造的某一層（厚度），原本那一面就變了——右欄那一句
    「選的…跟原本一樣，當成不貼」要馬上收，不等按「檢查」、不等限制重判；也不准打斷打字（那一格還是同一格、焦點與字都在）。"""
    ceiling = "ceiling_fibrecement8mm_wool30_on_board_gap180"
    built = next(entry["material"] for entry in SAMPLE_USER_ENTRIES if entry["id"] == ceiling)
    document = room109_document()
    cast(dict[str, dict[str, list[dict[str, object]]]], document["scene"])["surface_zones_by_wall"]["ceiling"][0]["material"] = (
        cast(dict[str, object], built))
    _write(tmp_path, document)
    _sample_library(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-ceiling").click()
        choose_fixed(page, "ceiling", "whole_wall", ceiling, show_all=True, said_part="選的跟原本一樣，當成不貼")
        same = page.locator("#zone-same-ceiling-whole_wall")
        expect(same).to_contain_text("跟原本一樣，當成不貼")
        page.wait_for_function("!document.querySelector('main').inert && !inputBusy && zoneLimitsShown === zoneLimitsGeneration")
        page.locator("#edit-wall-original").click()
        before = page.evaluate("zoneLimitsGeneration")
        layer = page.locator("#layer-ceiling-1-thickness_m")
        handle = layer.element_handle()
        layer.fill("0.05")
        expect(same).to_have_count(0)
        assert page.evaluate("zoneLimitsGeneration") == before  # 不是重判限制收掉的
        assert handle.evaluate("node => node.isConnected && document.activeElement === node && node.value === '0.05'")
        layer.press("Tab")
        expect(same).to_have_count(0)
        expect_mode(page, "ceiling", "whole_wall", "keep_original")
        assert watched.page_errors == [] and watched.console_errors == []


def test_floor_and_ceiling_are_never_called_the_original_wall(tmp_path: Path, browser: Browser) -> None:
    """#1176 第 B 支（用語表）：做法名稱六面一律「不貼，保留原樣」；地板叫「原本的地板」、天花叫「原本的天花」，不叫原牆——
    表上每一格、底部那一行（伺服器回的那一句也照面叫）、原方案那一列都算。假房間 zone-naming-sample：地板本身是木地板、靠前牆鋪地毯，
    天花本身是那一套吊頂（另外掛一片雲板、一條吊櫃，這裡考後面那塊沒貼東西的）。牆面照舊叫原牆。"""
    _write(tmp_path, ZONE_NAMING_SCHEME)
    write_zone_naming_library(tmp_path)
    wood = "木地板 架在龍骨上（考卷假設）"
    built = "天花 礦纖板＋泡棉 45 mm＋22 cm 空腔（考卷假設）"
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, "zone-naming-sample")
        page.locator("#elevation-wall-floor").click()
        # 舊表（#elevation-zones）拆了：表上每一格 → 第 2 段右欄清單與「每一區的位置與大小」（都在 #elevation-list 裡）；
        # 底部那一行（伺服器回的那一句）→ 小框套用後那一頁結果列上面那一行，回輸入頁前讀（said_part）。
        zones = page.locator("#elevation-list")
        # 舊表改做法那一路伺服器回的「：這一區放回原本的地板」只有舊表那一路會講（丙）；小框的回話（#zone-page-note）不准叫原牆，
        # 原題考的三處（地板改不貼、地板貼地毯、天花改不貼）照考，回話逐字比，免得空的那一句也算過（#1265 單元 3 之二收尾追加設計洞）。
        said = page.locator("#zone-page-note")
        choose_mode(page, "floor", "rug", "keep_original")
        expect(said).to_have_text("1 區改成「不貼，保留原樣」")
        expect(said).not_to_contain_text("原牆")
        expect(page.locator("#wall-zone-what-floor-rug")).to_have_text(re.compile(rf"^原本的地板「{re.escape(wood)}」原樣，不貼東西。"))
        choose_mode(page, "floor", "auto2", "search")
        # 舊表「現在：…，程式選了」那一格 → 右欄那一句（改字：「現在存的是「…」」）；「不貼」方塊上的字跟著表走（丙）。
        expect(page.locator("#wall-zone-what-floor-auto2")).to_contain_text(f"現在存的是「沒貼東西（原本的地板「{wood}」原樣）」")
        # 「貼在原本的地板上（最後一層背後接原本的地板）」也是舊表那一路伺服器回的那一句（丙）。
        choose_fixed(page, "floor", "rug", "carpet_on_concrete")
        expect(said).to_have_text("1 區改成「指定貼一種：地毯直接鋪混凝土（文獻擬合）」")
        expect(said).not_to_contain_text("原牆")
        assert mode_words(page, "floor", "rug") == ["指定貼一種", "不貼，保留原樣", "由程式最佳化"]
        assert "原牆" not in zones.inner_text()
        page.locator("#elevation-panel").screenshot(path=str(tmp_path / "floor-words.png"))

        page.locator("#elevation-wall-ceiling").click()
        # 舊表下拉第一項「現在：沒貼東西（…）」→ 右欄那一句（改字：拿掉「現在：」）。
        expect(page.locator("#wall-zone-what-ceiling-auto1")).to_contain_text(f"沒貼東西（原本的天花「{built}」原樣）")
        choose_mode(page, "ceiling", "auto1", "keep_original")  # 「：這一區放回原本的天花」同上（丙）
        expect(said).to_have_text("1 區改成「不貼，保留原樣」")
        expect(said).not_to_contain_text("原牆")
        expect(page.locator("#wall-zone-what-ceiling-auto1")).to_have_text(re.compile(rf"^原本的天花「{re.escape(built)}」原樣，不貼東西。"))
        expect(page.locator("#zone-numbers-ceiling-auto1")).to_contain_text("原本的天花：")
        choose_mode(page, "ceiling", "auto1", "search")
        expect(page.locator("#wall-zone-what-ceiling-auto1")).to_contain_text(f"現在存的是「沒貼東西（原本的天花「{built}」原樣）」")
        assert "原牆" not in zones.inner_text()
        page.locator("#elevation-panel").screenshot(path=str(tmp_path / "ceiling-words.png"))

        # 牆面照舊叫原牆。
        page.locator("#elevation-wall-yL").click()
        expect(page.locator("#wall-zone-what-yL-auto1")).to_contain_text("沒貼東西（原牆原樣）")
        assert watched.page_errors == [] and watched.console_errors == []


def test_a_tunable_vpr_in_the_library_never_shows_on_the_zone_page(tmp_path: Path, browser: Browser) -> None:
    """#1200 第 1 支：材料庫有一筆可調 VPR（電腦挑它還沒接上）：分區頁的勾選表、選單、「顯示全部材料」不列的那幾組都沒有它，
    頁面不壞；地板那一面（它不是地板用的）也一樣。"""
    from tests.engine.test_material_library_vpr import VPR, WOOL, _write_library

    _write(tmp_path, room109_document())
    _write_library(tmp_path / "materials.json", with_vpr=True)
    _old_search_design(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-x0").click()
        assert _chips(page, "x0", "whole_wall") == ["glass_wool_48k", "glass_wool_24k", "glass_wool_96k", WOOL, "none"]
        assert page.evaluate(f"[zoneLibrary.length > 0, zoneLibrary.some(entry => entry.id === '{VPR}')]") == [True, False]
        assert page.evaluate(f"['x0', 'floor'].flatMap(wall => hiddenGroups(wall, zoneLibrary).flatMap(([, items]) => items))"
                             f".some(entry => entry.id === '{VPR}')") is False
        assert "鋼板 VPR（自編）" not in page.locator("body").inner_text()
        assert watched.page_errors == [] and watched.console_errors == []
