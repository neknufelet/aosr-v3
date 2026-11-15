"""真瀏覽器考做法表的材料清單（#1176，老闆 11-15「109示範又不知道是娥麼這也能當成一個材料」「都不知道在講甚麼」）：
選單與勾選表只列這一面能貼的（門、窗、地板、整片吊頂、另一間示範房的不出現），「顯示全部材料」打開才分組列出；
這一區現在放的那一筆就算不是這一面用的也照樣顯示；用語是白話（做法三種、原牆不寫阻抗數字、可調寫得出調什麼）。
材料庫用老闆 11-15 那 12 筆（``_boss_material_library``），房間用 109 號房的長寬高。"""
from __future__ import annotations

from pathlib import Path
from typing import cast

from playwright.sync_api import Browser, Page, expect
from pydantic import TypeAdapter

from aosr.materials.library import LibraryEntry, dump_user_library
from tests.engine._boss_material_library import BOSS_USER_ENTRIES
from tests.engine._gui_cache import gui_startup_identity_memo as gui_startup_identity_memo
from tests.engine._surface_zone_cases import room109_document
from tests.engine.test_gui_browser import _open, _serve, browser as browser
from tests.engine.test_gui_zone_elevation_browser import _open_scheme, _write
from tests.engine.test_gui_zone_modes_browser import _choose

HIDDEN_ON_WALLS = ("own_room_door_wood_solid", "own_room_window_glass_10mm_single", "own_room_ceiling_calsil_6mm_gap250",
                   "perforated_gypsum_8_18r_backed_gw30_ceiling", "carpet_on_concrete", "own_room_floor_wood_on_concrete",
                   "demo109_gw48k_limited", "demo109_measured_gw49_gaborit_s1")
DOOR = "own_room_door_wood_solid"


def _boss_library(tmp_path: Path) -> None:
    adapter: TypeAdapter[tuple[LibraryEntry, ...]] = TypeAdapter(tuple[LibraryEntry, ...])
    (tmp_path / "materials.json").write_text(dump_user_library(adapter.validate_python(BOSS_USER_ENTRIES)), encoding="utf-8")


def _values(page: Page, selector: str) -> list[str]:
    return cast(list[str], page.locator(selector).evaluate_all("nodes => nodes.map(node => node.value)"))


def _ids(page: Page, wall: str, key: str) -> list[str]:
    prefix = f"zone-allow-{wall}-{key}-"
    return [value.removeprefix(prefix) for value in cast(list[str], page.locator(f"#zone-allowed-{wall}-{key} input").evaluate_all(
        "nodes => nodes.map(node => node.id)"))]


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
        page.locator("#zone-limit-x0-whole_wall").check()
        listed = _ids(page, "x0", "whole_wall")
        assert listed[:4] == ["glass_wool_48k", "glass_wool_24k", "glass_wool_96k", "none"]
        assert not set(HIDDEN_ON_WALLS) & set(listed)
        expect(page.locator("label", has=page.locator("#zone-allow-x0-whole_wall-glass_wool_48k"))).to_contain_text(
            "可調：厚 5～20 cm、空腔 0～25 cm，加起來最深 30 cm")
        expect(page.locator("label", has=page.locator("#zone-allow-x0-whole_wall-glass_wool_24k"))).to_contain_text(
            "可調：厚 5～20 cm、空腔不調")
        expect(page.locator("label", has=page.locator("#zone-allow-x0-whole_wall-glass_wool_48k"))).not_to_contain_text("（可調厚度與空腔）")
        expect(page.locator("label", has=page.locator("#zone-allow-x0-whole_wall-fabric_pet_a_gap100"))).to_contain_text(
            "固定成品：整塊貼或不貼，不能調")
        hidden = page.locator("#zone-allowed-hidden-x0-whole_wall")
        expect(hidden).to_contain_text("還有 11 筆不是前牆用的")
        expect(hidden).to_contain_text("門、窗")
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
        page.locator("#zone-limit-floor-whole_wall").check()
        assert _ids(page, "floor", "whole_wall") == ["none", "carpet_on_concrete"]
        expect(page.locator("#zone-allowed-floor-whole_wall")).to_contain_text("地板沒有可調的材料：電腦只能從「不貼」開始")
        page.locator("#elevation-panel").screenshot(path=str(tmp_path / "choices-floor.png"))
        assert watched.page_errors == [] and watched.console_errors == []


def test_a_zone_holding_a_material_from_another_group_still_shows_it(tmp_path: Path, browser: Browser) -> None:
    """選單只列這一面能貼的，但這一區現在放的就是門（打開「顯示全部材料」選的）：關掉之後選單照樣是它、寫出它不是這一面用的；
    保留原牆的區寫白話、數字收在細節。"""
    _write(tmp_path, room109_document())
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
        assert watched.page_errors == [] and watched.console_errors == []
