"""真瀏覽器考做法表的材料清單（#1176，老闆 11-15「109示範又不知道是娥麼這也能當成一個材料」「都不知道在講甚麼」）：
選單與小框只列這一面能貼的（門、窗、地板、整片吊頂、另一間示範房的不出現），指定貼一種的下拉可以打開「顯示全部材料」分組列出；
這一區現在放的那一筆就算不是這一面用的也照樣顯示；用語是白話（做法三種、原牆不寫阻抗數字、可調寫得出調什麼）。
材料庫用自己編的合成 12 筆（``_sample_material_library``），房間用 109 號房的長寬高。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from playwright.sync_api import Browser, Page, expect
from pydantic import TypeAdapter

from aosr.materials.library import LibraryEntry, dump_user_library
from tests.engine._zone_panel_steps import apply_and_return, edit_candidates, open_panel
from tests.engine._bonded_backing_cases import MELAMINE_100, STEEL_2_5MM
from tests.engine._sample_material_library import SAMPLE_USER_ENTRIES
from tests.engine._gui_cache import gui_startup_identity_memo as gui_startup_identity_memo
from tests.engine._surface_zone_cases import ROOM_109, room109_document
from tests.engine._zone_naming_room import ZONE_NAMING_SCHEME, write_zone_naming_library
from tests.engine.test_gui_browser import _load_template, _open, _serve, browser as browser
from tests.engine.test_gui_zone_elevation_browser import _drag, _open_scheme, _write
from tests.engine.test_gui_zone_modes_browser import _choose, _modes, _write_room109_design, _save

HIDDEN_ON_WALLS = ("hall_door_hollow_core", "side_window_glass_12mm_single", "ceiling_gypsum9mm_gap300",
                   "perforated_gypsum_8_18r_backed_gw30_ceiling", "carpet_on_concrete", "joist_floor_plank_fixed",
                   "demo109_rockwool60k_capped", "demo109_measured_pet35_synthetic_b7")
DOOR = "hall_door_hollow_core"
FLOOR_WOOD = "joist_floor_plank_fixed"


PLAIN_PRODUCT = "fabric_pet_a_gap100"  # 內建的一筆固定成品：牆面能貼、不是哪一種可調玻璃棉包含的、也不是「鋼板直接貼在棉上」的構造


# 考卷自己編的一筆長名一般固定成品（鋼板前面一層棉、背襯沒黏住，不是「鋼板直接貼在棉上」的構造）：名字 60 個字以上，
# 用來量固定下拉與唯讀方塊的版面。
LONG_PLAIN = "own_long_name_panel"
LONG_PLAIN_LABEL = "自編長名對照板（考卷用）＋前面自編吸音棉 100 mm（鋼板 2.5 mm、後面自編軟泡棉 100 mm 沒有黏住，總厚約 20 cm）"
assert len(LONG_PLAIN_LABEL) >= 60
LONG_PLAIN_ENTRY: dict[str, object] = {
    "id": LONG_PLAIN, "label": LONG_PLAIN_LABEL, "provenance": "model", "source": "合成：考卷自己編的數字", "links": [], "notes": [],
    "kind": "fixed", "material": {"kind": "assembly", "backing": "rigid", "layers": [MELAMINE_100, STEEL_2_5MM, MELAMINE_100]}}


def _sample_library(tmp_path: Path, *extra: dict[str, object]) -> None:
    adapter: TypeAdapter[tuple[LibraryEntry, ...]] = TypeAdapter(tuple[LibraryEntry, ...])
    (tmp_path / "materials.json").write_text(dump_user_library(adapter.validate_python([*SAMPLE_USER_ENTRIES, *extra])), encoding="utf-8")


def _values(page: Page, selector: str) -> list[str]:
    return cast(list[str], page.locator(selector).evaluate_all("nodes => nodes.map(node => node.value)"))


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


def test_menus_list_only_what_this_face_can_take_and_show_all_reveals_the_rest(tmp_path: Path, browser: Browser) -> None:
    _write(tmp_path, room109_document())
    _sample_library(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-x0").click()
        assert page.locator("#zone-mode-x0-whole_wall option").all_inner_texts() == ["指定貼一種", "不貼，保留原樣", "由程式最佳化"]
        # 指定貼一種：牆上能貼的固定成品（含合成的 VPR），門、窗、天花構造、地毯、示範房一筆都沒有；原牆不寫阻抗數字。
        fixed = _values(page, "#zone-fixed-x0-whole_wall option")
        assert "glass_wool_48k_50mm" in fixed and "pet60_vpr_steel1p2mm_foam90_sample" in fixed
        assert not set(HIDDEN_ON_WALLS) & set(fixed)
        expect(page.locator("#zone-fixed-x0-whole_wall option").first).to_have_text("現在：沒貼東西（原牆原樣）")
        expect(page.locator("#zone-numbers-x0-whole_wall")).to_contain_text("原牆：固定阻抗")

        page.locator("#zone-show-all").check()
        assert set(HIDDEN_ON_WALLS) - {"demo109_rockwool60k_capped"} <= set(_values(page, "#zone-fixed-x0-whole_wall option"))
        page.locator("#zone-show-all").uncheck()
        _choose(page, "#zone-mode-x0-whole_wall", "search")
        assert _chips(page, "x0", "whole_wall") == ["glass_wool_48k", "glass_wool_24k", "glass_wool_96k", "none"]
        expect(page.locator("#zone-chip-x0-whole_wall-glass_wool_48k small")).to_have_text(
            "可調：厚 5～20 cm、空腔 0～25 cm，加起來最深 30 cm")
        expect(page.locator("#zone-chip-x0-whole_wall-glass_wool_48k")).not_to_contain_text("（可調厚度與空腔）")
        for wool in ("glass_wool_24k", "glass_wool_96k"):
            expect(page.locator(f"#zone-chip-x0-whole_wall-{wool} small")).to_have_text("可調：厚 5～20 cm、空腔：沒有（直接貼牆）")
        page.locator("#zone-show-all").check()
        expect(page.locator("#zone-material-x0-whole_wall input")).to_have_count(0)
        open_panel(page, "x0", "whole_wall")
        page.locator("#zone-panel-add").click()
        for choice in HIDDEN_ON_WALLS:
            expect(page.locator(f"#zone-panel-pick-{choice}")).to_have_count(0)
        expect(page.locator("#zone-panel-pick-fabric_pet_a_gap100 .zv-description")).to_have_text("整塊貼、固定 10 cm")
        expect(page.locator("#zone-panel-pick-slotted18_pet45_vpr_steel2p25mm_foam70_sample .zv-description")).to_have_text(
            "整塊貼、固定 13.3 cm（不含薄板）")
        page.locator("#zone-panel-cancel").click()
        page.locator("#zone-page-close").click()
        page.locator("#zone-show-all").uncheck()
        page.locator("#elevation-wall-floor").click()
        assert _values(page, "#zone-fixed-floor-whole_wall option") == ["", "carpet_on_concrete"]
        _choose(page, "#zone-mode-floor-whole_wall", "search")
        assert _chips(page, "floor", "whole_wall") == ["none"]
        open_panel(page, "floor", "whole_wall")
        expect(page.locator("#zone-panel-form")).to_contain_text("這裡沒有可調的材料：程式只能從「不貼」開始")
        page.locator("#zone-panel-add").click()
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

        # 地板本身就是木地板那一筆（直接取代整面的固定阻抗，有的房間地板就是這樣）：選單照樣是它，但它就是地板本身，不標紅說不是地板用的。
        page.locator("#elevation-wall-floor").click()
        expect(page.locator("#zone-fixed-floor-whole_wall")).to_have_value(FLOOR_WOOD)
        expect(page.locator(f"#zone-fixed-floor-whole_wall option[value={FLOOR_WOOD}]")).to_contain_text("整面就是它，不是貼上去的")
        expect(page.locator("#zone-fixed-away-floor-whole_wall")).to_have_count(0)
        assert watched.page_errors == [] and watched.console_errors == []


def _chips(page: Page, wall: str, key: str) -> list[str]:
    prefix = f"zone-chip-{wall}-{key}-"
    return [value.removeprefix(prefix) for value in cast(list[str], page.locator(f"#zone-allowed-chips-{wall}-{key} .zone-chip")
                                                         .evaluate_all("nodes => nodes.map(node => node.id)"))]


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
        _choose(page, "#zone-fixed-x0-whole_wall", "glass_wool_48k_50mm")
        _choose(page, "#zone-mode-x0-whole_wall", "search")
        three = ["glass_wool_48k", "glass_wool_24k", "glass_wool_96k", "none"]
        assert _chips(page, "x0", "whole_wall") == three
        assert _modes(page, "x0")["whole_wall"] == {"mode": "search", "allowed": three}
        expect(page.locator("#zone-chip-x0-whole_wall-glass_wool_24k")).to_contain_text("可調：厚 5～20 cm、空腔：沒有（直接貼牆）")
        expect(page.locator("#zone-chip-x0-whole_wall-none")).to_contain_text("保留原樣，程式可以決定這裡不貼")
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
        _choose(page, "#zone-mode-x0-whole_wall", "keep_original")
        _choose(page, "#zone-mode-x0-whole_wall", "search")
        expect(page.locator("#zone-preset-x0-whole_wall-this_or_none")).to_have_count(0)
        assert _chips(page, "x0", "whole_wall") == ["glass_wool_48k", "none"]

        # 下一面改成由程式最佳化：照這一頁上一次的組合（48K＋不貼）。
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
        _choose(page, "#zone-mode-x0-whole_wall", "search")
        three = ["glass_wool_48k", "glass_wool_24k", "glass_wool_96k", "none"]
        assert _chips(page, "x0", "whole_wall") == three
        expect(page.locator("#zone-chip-x0-whole_wall-own_rock_wool_free")).to_have_count(0)
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
        note = page.locator("#zone-note")
        page.locator("#elevation-wall-x0").click()
        _choose(page, "#zone-mode-x0-whole_wall", "search")
        expect(note).to_have_text("前牆「整面」：由程式最佳化，先放了 48K 玻璃棉、24K 玻璃棉、96K 玻璃棉、不貼；要改就按這一列的「改這一區 ▸」")
        edit_candidates(page, "x0", "whole_wall", remove=("glass_wool_48k",))
        expect(note).to_have_text("1 區改成「由程式最佳化」")
        assert _chips(page, "x0", "whole_wall") == ["glass_wool_24k", "glass_wool_96k", "none"]
        page.locator("#elevation-wall-floor").click()
        expect(note).to_have_text("")
        _choose(page, "#zone-mode-floor-whole_wall", "search")
        expect(note).to_have_text("地板「整面」：由程式最佳化，先放了 不貼；要改就按這一列的「改這一區 ▸」")
        assert watched.page_errors == [] and watched.console_errors == []


def test_the_last_combination_does_not_follow_into_another_scheme(tmp_path: Path, browser: Browser) -> None:
    """「上一次的組合」只在同一份方案裡帶（審查 11-15）：在這一份改成 48K＋不貼，打開另一份方案再改成由程式最佳化，
    照樣從三種玻璃棉＋不貼開始（同一份方案裡的別一面才照上一次的組合）；載入範本也一樣從頭。"""
    _write(tmp_path, room109_document())
    _write(tmp_path, room109_document("room109-b"))
    _sample_library(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        three = ["glass_wool_48k", "glass_wool_24k", "glass_wool_96k", "none"]
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-x0").click()
        _choose(page, "#zone-mode-x0-whole_wall", "search")
        edit_candidates(page, "x0", "whole_wall", only="glass_wool_48k")
        page.locator("#elevation-wall-y0").click()
        _choose(page, "#zone-mode-y0-whole_wall", "search")
        assert _chips(page, "y0", "whole_wall") == ["glass_wool_48k", "none"]  # 同一份方案：照上一次的組合
        _open_scheme(page, "room109-b")
        page.locator("#elevation-wall-x0").click()
        _choose(page, "#zone-mode-x0-whole_wall", "search")
        assert _chips(page, "x0", "whole_wall") == three
        # 這一份也改成 48K＋不貼，再載入範本（resetZoneDesign）：一樣從頭。
        edit_candidates(page, "x0", "whole_wall", only="glass_wool_48k")
        _load_template(page)
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
    _sample_library(tmp_path)
    design = _old_search_design(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-x0").click()
        everything = ["glass_wool_48k", "glass_wool_24k", "glass_wool_96k", "demo109_rockwool60k_capped", "none"]
        assert _chips(page, "x0", "whole_wall") == everything
        expect(page.locator("#zone-chip-x0-whole_wall-demo109_rockwool60k_capped")).to_contain_text("不是前牆用的")
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
        expect(page.locator("#zone-allowed-unknown-x0-whole_wall")).to_have_text("程式從全部可調材料與「不貼」裡選（材料庫沒讀全，列不出是哪幾種）")
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
            box = page.locator("#zone-allowed-problem-x0-whole_wall")
            expect(box).to_have_attribute("role", "alert")
            expect(box.locator("p").first).to_have_text("只放了固定成品，程式沒得選。")
            expect(box.locator("p").last).to_have_text(
                "程式要從一個能調的東西開始：加一種可調的材料（例如玻璃棉），或加上「也可以不貼」；要的就是這一筆、不必讓程式選，就把做法改成「指定貼一種」。要改清單就按這一列的「改這一區 ▸」。")
            expect(box.locator("button")).to_have_count(0)
            edit_candidates(page, "x0", "whole_wall", add=(addition,))
            assert _modes(page, "x0")["whole_wall"]["allowed"] == [PLAIN_PRODUCT, addition]
            expect(box).to_have_count(0)
        _open_only_fixed(page, "x0")
        page.locator("#elevation-wall-x0").click()
        open_panel(page, "x0", "whole_wall")
        page.locator("#zone-panel-material").select_option(PLAIN_PRODUCT)
        apply_and_return(page)
        expect(page.locator("#zone-mode-x0-whole_wall")).to_have_value("fixed")
        expect(page.locator("#zone-fixed-x0-whole_wall")).to_have_value(PLAIN_PRODUCT)
        assert _modes(page, "x0")["whole_wall"] == {"mode": "fixed"}
        _save(page, "room109")
        _open_only_fixed(page, "floor")
        page.locator("#elevation-wall-floor").click()
        floor_box = page.locator("#zone-allowed-problem-floor-whole_wall")
        expect(floor_box).to_contain_text("只放了固定成品，程式沒得選。")
        expect(floor_box.locator("button")).to_have_count(0)
        open_panel(page, "floor", "whole_wall")
        page.locator("#zone-panel-add").click()
        expect(page.locator("#zone-panel-pick-glass_wool_48k")).to_have_count(0)
        page.locator("#zone-panel-picker-search").press("Escape")
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
        expect(page.locator("#zone-note")).to_contain_text(f"先處理{front}「整面」、{side}「整面」的紅框")
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


def _overflow(page: Page) -> dict[str, float]:
    """做法表、深度上限那一欄的右緣比立面圖面板右緣多出幾 px（0 以下＝塞得進面板）。"""
    return cast(dict[str, float], page.evaluate("""() => {
        const panel = document.getElementById('elevation-panel').getBoundingClientRect().right;
        const table = document.querySelector('.zone-table');
        const depth = table.querySelector('th:last-child');
        return {table: table.getBoundingClientRect().right - panel, depth: depth.getBoundingClientRect().right - panel};
    }"""))


def test_the_front_wall_table_fits_the_panel_with_long_product_names(tmp_path: Path, browser: Browser) -> None:
    """材料庫有幾筆名字很長的固定成品（合成的 VPR 那幾筆）：材料下拉選單不准照最長那一筆把表撐寬、把深度上限那一欄擠出面板；
    指定貼一種與由程式最佳化（方塊、說明一行一行換）兩個樣子都要塞得進立面圖面板。"""
    _write_room109_design(tmp_path, {"left_right_symmetry": "off"})  # 這一題不考左右對稱，明寫關著
    _sample_library(tmp_path, LONG_PLAIN_ENTRY)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-x0").click()
        _drag(page, ROOM_109["Ly"], ROOM_109["Lz"], (0.02, 0.02), (1.0, 1.2))
        expect(page.locator("#zone-fixed-x0-z1 option", has_text="VPR").first).to_be_attached()
        assert all(value <= 0.5 for value in _overflow(page).values()), _overflow(page)
        _choose(page, "#zone-fixed-x0-z1", "pet60_vpr_steel1p2mm_foam90_sample")
        assert all(value <= 0.5 for value in _overflow(page).values()), _overflow(page)
        # 切換前後以同一筆自編長名成品量固定下拉與候選方塊。
        _choose(page, "#zone-fixed-x0-z1", LONG_PLAIN)
        assert all(value <= 0.5 for value in _overflow(page).values()), _overflow(page)
        _choose(page, "#zone-mode-x0-z1", "search")
        edit_candidates(page, "x0", "z1", add=(LONG_PLAIN,), only=LONG_PLAIN)
        long_chip = page.locator(f"#zone-chip-x0-z1-{LONG_PLAIN}")
        expect(long_chip.locator("strong")).to_have_text(LONG_PLAIN_LABEL)
        assert all(value <= 0.5 for value in _overflow(page).values()), _overflow(page)
        assert cast(float, long_chip.evaluate(
            "node => node.getBoundingClientRect().right - document.getElementById('elevation-panel').getBoundingClientRect().right"
        )) <= 0.5
        page.locator("#zone-name-x0-z1").scroll_into_view_if_needed()
        expect(page.locator("#depth-zone-x0-z1")).to_be_in_viewport()
        page.locator("#elevation-panel").screenshot(path=str(tmp_path / "front-fits.png"))
        assert watched.page_errors == [] and watched.console_errors == []


def test_zone_and_size_columns_never_break_inside_a_word(tmp_path: Path, browser: Browser) -> None:
    """審查 11-15：天花整面一區、由程式最佳化打開清單，「區」那一欄的「整面」斷成直的兩行。窄視窗（760）也不准在字中間斷：
    「整面」一行，「大小（公尺）」只准在括號前換行。"""
    _write(tmp_path, room109_document())
    _sample_library(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=760) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-ceiling").click()
        _choose(page, "#zone-mode-ceiling-whole_wall", "search")
        row = page.locator("#zone-row-ceiling-whole_wall td")
        # 「區」那一格是「① 整面」（號碼在前面的粗字，名字是它後面那一段字）。
        assert page.locator("#zone-label-ceiling-whole_wall").evaluate(LINES.replace("cell.firstChild", "cell.lastChild")) == ["整面"]
        assert row.nth(1).evaluate(LINES) == ["整面"]
        assert page.locator("#zone-table-ceiling th").nth(1).evaluate(LINES) in (["大小（公尺）"], ["大小", "（公尺）"])
        # 兩欄不准斷字之後最窄寬了一點：這麼窄的視窗整張表照樣塞得進面板。
        assert all(value <= 0.5 for value in _overflow(page).values()), _overflow(page)
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
            expect(page.locator(f"#zone-allowed-missing-{wall}-whole_wall")).to_have_text(
                "清單裡有材料庫找不到的：ghost_material（照原樣留著，開跑時伺服器會點名；要拿掉就按這一列的「改這一區 ▸」）")
            expect(page.locator(f"#zone-allowed-problem-{wall}-whole_wall")).to_have_count(0)
            expect(page.locator("#zone-to-search")).to_be_enabled()
        page.locator("#elevation-wall-x0").click()
        open_panel(page, "x0", "whole_wall")
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
        # 整片天花構造平常不列成貼上去的處理；原牆認得出是哪一筆就寫它的名字。
        assert ceiling not in _values(page, "#zone-fixed-ceiling-whole_wall option")
        expect(page.locator("#zone-fixed-ceiling-whole_wall option").first).to_contain_text(
            "現在：沒貼東西（原本的天花「天花 纖維水泥板 8 mm＋岩棉毯 30 mm 平鋪在板背＋18 cm 空腔（剛背）」原樣）")
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
        zones = page.locator("#elevation-zones")
        _choose(page, "#zone-mode-floor-rug", "keep_original")
        expect(page.locator("#zone-note")).to_contain_text("：這一區放回原本的地板")
        expect(page.locator("#zone-keep-floor-rug")).to_have_text(f"原本的地板「{wood}」原樣，不貼東西")
        _choose(page, "#zone-mode-floor-auto2", "search")
        expect(page.locator("#zone-search-now-floor-auto2")).to_contain_text(f"現在：沒貼東西（原本的地板「{wood}」原樣），程式選了")
        expect(page.locator("#zone-chip-floor-auto2-none")).to_contain_text("保留原樣，程式可以決定這裡不貼")
        page.locator("#zone-mode-floor-rug").select_option("fixed")  # 改回指定貼一種不必問伺服器
        _choose(page, "#zone-fixed-floor-rug", "carpet_on_concrete")
        expect(page.locator("#zone-note")).to_contain_text("貼在原本的地板上（最後一層背後接原本的地板）")
        expect(page.locator("#zone-mode-floor-rug option")).to_have_text(["指定貼一種", "不貼，保留原樣", "由程式最佳化"])
        assert "原牆" not in zones.inner_text() + page.locator("#zone-note").inner_text()
        page.locator("#elevation-panel").screenshot(path=str(tmp_path / "floor-words.png"))

        page.locator("#elevation-wall-ceiling").click()
        expect(page.locator("#zone-fixed-ceiling-auto1 option").first).to_have_text(f"現在：沒貼東西（原本的天花「{built}」原樣）")
        _choose(page, "#zone-mode-ceiling-auto1", "keep_original")
        expect(page.locator("#zone-keep-ceiling-auto1")).to_have_text(f"原本的天花「{built}」原樣，不貼東西")
        expect(page.locator("#zone-numbers-ceiling-auto1")).to_contain_text("原本的天花：")
        expect(page.locator("#zone-note")).to_contain_text("：這一區放回原本的天花")
        _choose(page, "#zone-mode-ceiling-auto1", "search")
        expect(page.locator("#zone-search-now-ceiling-auto1")).to_contain_text(f"現在：沒貼東西（原本的天花「{built}」原樣）")
        assert "原牆" not in zones.inner_text() + page.locator("#zone-note").inner_text()
        page.locator("#elevation-panel").screenshot(path=str(tmp_path / "ceiling-words.png"))

        # 牆面照舊叫原牆。
        page.locator("#elevation-wall-yL").click()
        expect(page.locator("#zone-fixed-yL-auto1 option").first).to_have_text("現在：沒貼東西（原牆原樣）")
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
