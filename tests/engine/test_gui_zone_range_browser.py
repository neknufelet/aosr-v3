"""真瀏覽器考「這間房貼什麼」那一頁改每一區自己的可調範圍（#1199 範圍第 2 支）：小框選了由程式最佳化，勾著的可調材料
每一列底下兩對格子（棉厚兩個下拉、空腔兩格），套用只寫選中的區（設計檔 ``ZoneDesign.ranges``，公尺）；改回材料頁的值就不寫；
各區不同時空著、不動就照舊；超出材料頁、收窄後放不下照伺服器的原句擋套用；右欄清單卡照範圍分張、寫出每一組的範圍；
第三階段頁的預覽寫出這一區實際用的範圍；左右對稱那一塊寫出兩區範圍不同那一句；沒動範圍時存檔跟第 1 支逐位相同（控制組）。

資料全是考卷自己編的：區名考卷那間假房間（``_zone_naming_room`` 的 zone-naming-sample，前牆 yL 五區）與左右對稱考卷那間假房間
（``_zone_symmetry_room``），材料庫是合成那 12 筆（``_sample_material_library``）；可調的三種玻璃棉是產品內建的那幾筆
（材料頁的範圍：棉厚 5、10、20 cm，48K 空腔 0～25 cm、24K 與 96K 沒有空腔）。編的範圍都含 48K 的起跑值（棉 5 cm、空腔 5 cm），
畫面不靠起跑值在不在範圍裡的那一句。截圖存 ``AOSR_SHOTS``（有給才存）。"""
from __future__ import annotations

import os
from pathlib import Path
from typing import cast

from playwright.sync_api import Browser, Page, expect

from aosr.gui.zone_design import DesignSettings, write_design
from aosr.reporting.scheme import Scheme
from tests.engine._gui_cache import gui_startup_identity_memo as gui_startup_identity_memo
from tests.engine._zone_naming_room import ZONE_NAMING_SCHEME
from tests.engine._zone_symmetry_room import FRONT_PAIR, NAME as SYMMETRY, symmetry_document
from tests.engine.test_gui_browser import _open, _serve, browser as browser
from tests.engine.test_gui_zone_elevation_browser import _open_scheme, _write
from tests.engine.test_gui_zone_material_pick_browser import _TEXT_STYLES
from tests.engine.test_gui_zone_overview_browser import SAMPLE, WOOLS
from tests.engine.test_gui_zone_overview_tasks_browser import (
    _apply, _click_zone, _library, _open_overview, _save, _send_to_search, _settled, _stored)

WOOL, THIN = "glass_wool_48k", "glass_wool_24k"
FRONT = ("z1", "z2", "z3", "auto1", "auto2")
SEARCH: dict[str, object] = {"mode": "search", "allowed": WOOLS}
# 兩份編的範圍（公尺）。
NARROW = {"porous_m": [0.05, 0.1], "cavity_m": [0.0, 0.15]}
SHALLOW = {"porous_m": [0.05, 0.2], "cavity_m": [0.0, 0.1]}
Doc = dict[str, object]


def _design(ranges: dict[str, Doc] | None = None, scheme_id: str = SAMPLE) -> Doc:
    """前牆五區都由程式最佳化（三種玻璃棉＋不貼）；``ranges``：區代號 → 那一區的 ranges。"""
    zones = {key: {**SEARCH, **({"ranges": (ranges or {})[key]} if key in (ranges or {}) else {})} for key in FRONT}
    return {"schema_version": "aosr.zone_design.v1", "scheme_id": scheme_id, "zones": {"yL": zones}}


def _prepare(tmp_path: Path, design: Doc | None, scheme: Doc = ZONE_NAMING_SCHEME) -> Path:
    """方案、合成材料庫與設計檔（用產品自己的寫法存，跟網頁存出來的同一種排法，才比得出逐位相同）。回設計檔的路徑。"""
    _write(tmp_path, scheme)
    _library(tmp_path)
    path = tmp_path / "designs" / f"{scheme['scheme_id']}.json"
    if design is not None:
        path.parent.mkdir(exist_ok=True)
        write_design(tmp_path, DesignSettings.model_validate(design), Scheme.model_validate(scheme))
    return path


def _shot(page: Page, name: str) -> None:
    folder = os.environ.get("AOSR_SHOTS")
    if folder:
        page.screenshot(path=str(Path(folder) / f"{name}.png"), full_page=True)


def _cell(page: Page, choice: str, part: str, end: str) -> str:
    return f"#zone-panel-range-{choice}-{part}-{end}"


def _close_and_save(page: Page) -> None:
    page.locator("#zone-page-close").click()
    expect(page.locator("#zone-page")).to_be_hidden()
    _save(page, design=True)


def test_a_narrowed_range_is_written_in_metres_into_the_picked_zones_only(tmp_path: Path, browser: Browser) -> None:
    """選前牆兩區、由程式最佳化：48K 那一列有棉厚兩個下拉（列材料的產品厚度、帶出材料頁的 5～20 cm）與空腔兩格（0～25 cm），
    24K 沒有空腔、只有棉厚。棉厚改到 10 cm、空腔改到 15 cm 套用、存檔：只有這兩區多一格 ranges，數字是公尺；其餘三區一個字都不多。"""
    _prepare(tmp_path, _design())
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        _click_zone(page, "yL/z1")
        _click_zone(page, "yL/z2")
        page.locator("#zone-panel-mode-search").check()
        expect(page.locator("#zone-panel-range-note")).to_have_text("範圍從材料頁帶來，可以改（只改這幾區，不改材料頁）材料頁 ▸")
        expect(page.locator("#zone-panel-materials")).to_have_attribute("href", "/materials")
        options = page.locator(_cell(page, WOOL, "porous", "low") + " option")
        assert options.evaluate_all("items => items.map(item => [item.textContent, item.value])") == [
            ["5", "0.05"], ["10", "0.1"], ["20", "0.2"]]
        for part, end, value in (("porous", "low", "0.05"), ("porous", "high", "0.2"), ("cavity", "low", "0"), ("cavity", "high", "25")):
            expect(page.locator(_cell(page, WOOL, part, end))).to_have_value(value)
        expect(page.locator(f"#zone-panel-range-{THIN}-porous")).to_be_visible()
        expect(page.locator(f"#zone-panel-range-{THIN}-cavity")).to_have_count(0)
        expect(page.locator(f"#zone-panel-range-{WOOL}")).to_contain_text("棉厚從")
        page.locator(_cell(page, WOOL, "porous", "high")).select_option("0.1")
        page.locator(_cell(page, WOOL, "cavity", "high")).fill("15")
        expect(page.locator("#zone-panel-apply")).to_be_enabled()
        _apply(page)
        _close_and_save(page)
        assert watched.page_errors == [] and watched.console_errors == []
    zones, _ = _stored(tmp_path)
    assert zones["yL"] == {"z1": {**SEARCH, "ranges": {WOOL: {"porous_m": [0.05, 0.1], "cavity_m": [0.0, 0.15]}}},
                           "z2": {**SEARCH, "ranges": {WOOL: {"porous_m": [0.05, 0.1], "cavity_m": [0.0, 0.15]}}},
                           **{key: SEARCH for key in ("z3", "auto1", "auto2")}}


def test_one_zone_panel_offers_cells_only_under_ticked_tunable_materials(tmp_path: Path, browser: Browser) -> None:
    """只選一區：那一行寫「只改這一區」；拿掉 96K 的勾，96K 那一列就沒有格子（勾回來又有）；固定成品、也可以不貼沒有格子。
    小框裡新加的字不小於 12 px、對比至少 4.5（截一張：一區改範圍的小框）。"""
    _prepare(tmp_path, _design())
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        _click_zone(page, "yL/z3")
        expect(page.locator("#zone-panel-range-note")).to_contain_text("只改這一區，不改材料頁")
        page.locator("#zone-panel-allow-glass_wool_96k").uncheck()
        expect(page.locator("#zone-panel-range-glass_wool_96k")).to_have_count(0)
        assert page.locator("#zone-panel .zv-mrange").evaluate_all("rows => rows.map(row => row.dataset.choice)") == [WOOL, THIN]
        page.locator("#zone-panel-allow-glass_wool_96k").check()
        expect(page.locator("#zone-panel-range-glass_wool_96k")).to_be_visible()
        page.locator(_cell(page, WOOL, "porous", "high")).select_option("0.1")
        page.locator(_cell(page, WOOL, "cavity", "high")).fill("15")
        rows = cast(list[dict[str, object]], page.locator("#zone-panel").evaluate(_TEXT_STYLES))
        assert rows and all(cast(float, row["size"]) >= 12 and cast(float, row["contrast"]) >= 4.5 for row in rows), rows
        page.locator("#zone-panel").scroll_into_view_if_needed()
        _shot(page, "1-one-zone-panel")
        assert watched.page_errors == [] and watched.console_errors == []


def test_setting_a_range_back_to_the_material_page_drops_it_and_the_file_returns_to_no_ranges(tmp_path: Path, browser: Browser) -> None:
    """z1 存著自己的範圍（棉 5～10、空腔 0～15）：小框帶出那一份；改回材料頁的 20 cm、25 cm 套用、存檔，那一區不再有 ranges，
    整份設計檔逐位等於從來沒寫範圍的那一份（產品自己的寫法存的）。"""
    plain = tmp_path / "plain"
    plain.mkdir()
    expected = _prepare(plain, _design()).read_bytes()
    path = _prepare(tmp_path, _design({"z1": {WOOL: NARROW}}))
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        _click_zone(page, "yL/z1")
        for part, end, value in (("porous", "low", "0.05"), ("porous", "high", "0.1"), ("cavity", "low", "0"), ("cavity", "high", "15")):
            expect(page.locator(_cell(page, WOOL, part, end))).to_have_value(value)
        page.locator(_cell(page, WOOL, "porous", "high")).select_option("0.2")
        page.locator(_cell(page, WOOL, "cavity", "high")).fill("25")
        _apply(page)
        _close_and_save(page)
        assert watched.page_errors == [] and watched.console_errors == []
    assert path.read_bytes() == expected
    assert b"ranges" not in path.read_bytes()


def test_zones_with_different_ranges_show_empty_cells_and_keep_their_own_unless_both_ends_are_set(tmp_path: Path, browser: Browser) -> None:
    """z1、z2 的 48K 範圍不同：選這兩區，48K 兩對格子都空著、旁邊寫「各區不同；不動它，各區照目前的」。只改最深套用，兩區各自的範圍
    照舊；棉厚只填一格不准套用（寫為什麼）；兩格都填（5～10）套用，兩區棉厚一起換、空腔各留自己的（截一張：多區各區不同）。"""
    _prepare(tmp_path, _design({"z1": {WOOL: NARROW}, "z2": {WOOL: SHALLOW}}))
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        _click_zone(page, "yL/z1")
        _click_zone(page, "yL/z2")
        for part in ("porous", "cavity"):
            expect(page.locator(f"#zone-panel-range-{WOOL}-{part}-mixed")).to_have_text("各區不同；不動它，各區照目前的")
            for end in ("low", "high"):
                expect(page.locator(_cell(page, WOOL, part, end))).to_have_value("")
        # 選著區時右欄的清單卡收起來：清單一樣、範圍不同的兩張也寫出各自的範圍，不是只差區名。
        expect(page.locator(".zv-list[data-zones='yL/z1'] .zv-ranges")).to_have_text("這一區自己的範圍：48K 玻璃棉（棉 5～10 cm、空腔 0～15 cm）")
        expect(page.locator(".zv-list[data-zones='yL/z2'] .zv-ranges")).to_have_text("這一區自己的範圍：48K 玻璃棉（棉 5～20 cm、空腔 0～10 cm）")
        expect(page.locator(".zv-list[data-zones*='yL/z3'] .zv-ranges")).to_have_count(0)
        _shot(page, "2-zones-differ")
        page.locator("#zone-panel-depth").fill("28")
        _apply(page)
        zones = cast(dict[str, dict[str, Doc]], page.evaluate("() => zoneDesign.zones"))
        assert zones["yL"]["z1"] == {**SEARCH, "ranges": {WOOL: NARROW}, "max_depth_m": 0.28}
        assert zones["yL"]["z2"] == {**SEARCH, "ranges": {WOOL: SHALLOW}, "max_depth_m": 0.28}
        _click_zone(page, "yL/z1")
        _click_zone(page, "yL/z2")
        page.locator(_cell(page, WOOL, "porous", "high")).select_option("0.1")
        expect(page.locator("#zone-panel-apply")).to_be_disabled()
        expect(page.locator("#zone-panel-problem")).to_contain_text("各區不同，要改就「從」「到」兩格都填")
        page.locator(_cell(page, WOOL, "porous", "low")).select_option("0.05")
        _apply(page)
        zones = cast(dict[str, dict[str, Doc]], page.evaluate("() => zoneDesign.zones"))
        assert zones["yL"]["z1"]["ranges"] == {WOOL: {"porous_m": [0.05, 0.1], "cavity_m": [0.0, 0.15]}}
        assert zones["yL"]["z2"]["ranges"] == {WOOL: {"porous_m": [0.05, 0.1], "cavity_m": [0.0, 0.1]}}
        assert all("ranges" not in zones["yL"][key] for key in ("z3", "auto1", "auto2"))
        assert watched.page_errors == [] and watched.console_errors == []


def test_widening_past_the_material_page_or_from_above_to_is_refused_on_the_spot(tmp_path: Path, browser: Browser) -> None:
    """空腔打到 30 cm（材料頁到 25 cm）按套用：伺服器的限制檢查說超出材料頁、只准收窄，那一句原樣寫在小框，套用鈕停用，
    什麼都沒寫；改回 25 又能套用。棉厚從 20 到 10：當場寫「從」要不大於「到」、不能套用（截一張：放大被擋）。"""
    _prepare(tmp_path, _design())
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        before = page.evaluate("() => JSON.stringify(zoneDesign)")
        _click_zone(page, "yL/z1")
        page.locator(_cell(page, WOOL, "cavity", "high")).fill("30")
        page.locator("#zone-panel-apply").click()
        problem = page.locator("#zone-panel-problem")
        expect(problem).to_contain_text("可調範圍：空腔 0～0.3 公尺超出材料頁的範圍（0～0.25 公尺）；這裡只准收窄，要放寬請到材料頁改那份材料")
        expect(page.locator("#zone-panel-apply")).to_be_disabled()
        _shot(page, "3-widening-refused")
        assert page.evaluate("() => JSON.stringify(zoneDesign)") == before
        page.locator(_cell(page, WOOL, "cavity", "high")).fill("25")
        expect(problem).to_be_hidden()
        expect(page.locator("#zone-panel-apply")).to_be_enabled()
        page.locator(_cell(page, WOOL, "porous", "low")).select_option("0.2")
        page.locator(_cell(page, WOOL, "porous", "high")).select_option("0.1")
        expect(problem).to_have_text("48K 玻璃棉的棉厚：「從」要不大於「到」")
        expect(page.locator("#zone-panel-apply")).to_be_disabled()
        assert page.evaluate("() => JSON.stringify(zoneDesign)") == before
        assert watched.page_errors == [] and watched.console_errors == []


def test_a_range_that_no_longer_fits_the_deepest_is_refused_with_the_server_sentence(tmp_path: Path, browser: Browser) -> None:
    """z1 最深 12 cm、48K 空腔從 8 cm 起（最薄棉 5 cm 加 8 cm 放不進 12 cm）：套用前伺服器判紅，照它那一句（收窄後的最淺排法）寫在小框、
    不套用；空腔從 0 起就放得下、套用後寫進去。"""
    design = _design()
    cast(dict[str, dict[str, Doc]], design["zones"])["yL"]["z1"] = {**SEARCH, "max_depth_m": 0.12}
    _prepare(tmp_path, design)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        _click_zone(page, "yL/z1")
        page.locator(_cell(page, WOOL, "cavity", "low")).fill("8")
        page.locator(_cell(page, WOOL, "cavity", "high")).fill("15")
        page.locator("#zone-panel-apply").click()
        expect(page.locator("#zone-panel-problem")).to_contain_text("這是照這一區的可調範圍收窄後的最淺排法（最薄棉加空腔下界），請調小範圍的下限或放寬上限")
        expect(page.locator("#zone-panel-apply")).to_be_disabled()
        page.locator(_cell(page, WOOL, "cavity", "low")).fill("0")
        _apply(page)
        zones = cast(dict[str, dict[str, Doc]], page.evaluate("() => zoneDesign.zones"))
        assert zones["yL"]["z1"]["ranges"] == {WOOL: {"porous_m": [0.05, 0.2], "cavity_m": [0.0, 0.15]}}
        assert watched.page_errors == [] and watched.console_errors == []


def test_list_cards_split_by_range_and_read_out_the_range_each_group_uses(tmp_path: Path, browser: Browser) -> None:
    """前牆五區清單一樣、z1 z2 有自己的範圍：清單卡分兩張（三區那張在前、照材料頁寫；兩區那張寫「這幾區自己的範圍」），範圍不算
    另一種做法——另外指定那一列前牆五區還是同一組。在範圍那張卡拿掉 48K，兩區的範圍跟著拿掉、存檔沒有 ranges（截一張：右欄卡）。"""
    _prepare(tmp_path, _design({"z1": {WOOL: NARROW}, "z2": {WOOL: NARROW}}))
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        cards = page.locator("#zone-page .zv-list").evaluate_all("cards => cards.map(card => [card.id, card.dataset.zones.split(' ').sort()])")
        assert cards == [["zone-list", ["yL/auto1", "yL/auto2", "yL/z3"]], ["zone-list-2", ["yL/z1", "yL/z2"]]], cards
        expect(page.locator(f"#zone-list-checks label[data-choice={WOOL}] .zv-range")).to_have_text(
            "可調：厚 5～20 cm、空腔 0～25 cm，加起來最深 30 cm")
        ranged = page.locator(f"#zone-list-2-checks label[data-choice={WOOL}] .zv-range")
        expect(ranged).to_have_text("這幾區自己的範圍：棉 5～10 cm、空腔 0～15 cm（材料頁是棉 5～20 cm、空腔 0～25 cm）")
        expect(ranged).to_have_attribute("data-ranged", "true")
        expect(page.locator(f"#zone-list-2-checks label[data-choice={THIN}] .zv-range")).to_have_text("可調：厚 5～20 cm、空腔：沒有（直接貼牆）")
        groups = page.locator("#zone-special .zv-group").evaluate_all("groups => groups.map(group => group.dataset.zones.split(' ').sort())")
        assert ["yL/auto1", "yL/auto2", "yL/z1", "yL/z2", "yL/z3"] in groups, groups
        page.locator("#zone-list-2").scroll_into_view_if_needed()
        _shot(page, "4-cards-read-ranges")
        page.locator(f"#zone-list-2-check-{WOOL}").click()
        _settled(page)
        expect(page.locator("#zone-page .zv-range[data-ranged]")).to_have_count(0)
        _close_and_save(page)
        assert watched.page_errors == [] and watched.console_errors == []
    zones, _ = _stored(tmp_path)
    assert zones["yL"]["z1"] == zones["yL"]["z2"] == {"mode": "search", "allowed": [THIN, "glass_wool_96k", "none"]}


def test_switching_a_ranged_zone_away_from_the_program_drops_its_range(tmp_path: Path, browser: Browser) -> None:
    """有範圍的一區改成「不貼，保留原樣」：範圍只給由程式最佳化的區，那一格跟著拿掉，存得進去（伺服器不收帶範圍的非搜尋區）。"""
    _prepare(tmp_path, _design({"z1": {WOOL: NARROW}}))
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        _click_zone(page, "yL/z1")
        page.locator("#zone-panel-mode-keep_original").check()
        _apply(page)
        _close_and_save(page)
        assert watched.page_errors == [] and watched.console_errors == []
    zones, _ = _stored(tmp_path)
    assert zones["yL"]["z1"] == {"mode": "keep_original"}


def test_the_stage_three_preview_shows_the_range_the_zone_uses(tmp_path: Path, browser: Browser) -> None:
    """z3 改範圍（48K 棉 5～10、空腔 0～15；24K 棉 10～20）套用、按「拿去搜尋」：第三階段頁的預覽表那一區的准用材料底下
    一種一行「48K 玻璃棉…：棉 5～10 cm、空腔 0～15 cm」「24K 玻璃棉…：棉 10～20 cm」（不調空腔的不寫空腔）；
    沒寫範圍的區沒有這一行（截一張：第三階段預覽）。"""
    _prepare(tmp_path, _design())
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        _click_zone(page, "yL/z3")
        page.locator(_cell(page, WOOL, "porous", "high")).select_option("0.1")
        page.locator(_cell(page, WOOL, "cavity", "high")).fill("15")
        page.locator(_cell(page, THIN, "porous", "low")).select_option("0.1")
        _apply(page)
        _send_to_search(page)
        row = page.locator("#plan-table tr[data-surface='yL@z3']")
        expect(row.locator(".range-line")).to_have_count(2)
        expect(row.locator(f".range-line[data-family={WOOL}]")).to_contain_text("：棉 5～10 cm、空腔 0～15 cm")
        expect(row.locator(f".range-line[data-family={THIN}]")).to_contain_text("：棉 10～20 cm")
        assert row.locator(f".range-line[data-family={THIN}]").inner_text().rsplit("：", 1)[1] == "棉 10～20 cm"
        assert page.locator("#plan-table tr[data-surface='yL@z1'] .range-line").count() == 0
        row.scroll_into_view_if_needed()
        _shot(page, "5-stage-three-preview")
        assert watched.page_errors == [] and watched.console_errors == []


def test_a_mirrored_pair_with_different_ranges_says_why_in_the_symmetry_box(tmp_path: Path, browser: Browser) -> None:
    """左右對稱照「這一區最深」那一套：只選前牆一對裡的一區改範圍，只改那一區，「左右對稱」那一塊寫伺服器那一句
    「兩區可調的範圍不一樣，各自調；兩區設成一樣就配成一對」；另一區也設成一樣，又配成一對。"""
    design: Doc = {"schema_version": "aosr.zone_design.v1", "scheme_id": SYMMETRY, "zones": {"x0": {"b": SEARCH, "a": SEARCH}}}
    _prepare(tmp_path, design, symmetry_document({"x0": FRONT_PAIR}))
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, SYMMETRY)
        page.locator("#zone-page-open").click()
        _settled(page)
        expect(page.locator("#zone-symmetry")).to_have_attribute("data-state", "paired")
        expect(page.locator("#zone-symmetry-pairs .zv-sym-pair")).to_have_count(1)
        for zone in ("x0/b", "x0/a"):
            _click_zone(page, zone)
            page.locator(_cell(page, WOOL, "porous", "high")).select_option("0.1")
            page.locator(_cell(page, WOOL, "cavity", "high")).fill("15")
            _apply(page)
            expect(page.locator("#zone-symmetry")).to_have_attribute("data-state", "paired")
            if zone == "x0/b":
                expect(page.locator("#zone-symmetry-pairs .zv-sym-pair")).to_have_count(0)
                expect(page.locator("#zone-symmetry-missed")).to_contain_text("兩區可調的範圍不一樣，各自調；兩區設成一樣就配成一對")
                zones = cast(dict[str, dict[str, Doc]], page.evaluate("() => zoneDesign.zones"))
                assert zones["x0"]["b"]["ranges"] == {WOOL: NARROW} and "ranges" not in zones["x0"]["a"]
        expect(page.locator("#zone-symmetry-pairs .zv-sym-pair")).to_have_count(1)
        expect(page.locator("#zone-symmetry-missed")).to_have_count(0)
        assert watched.page_errors == [] and watched.console_errors == []


# 控制組：沒動範圍的兩條路照第 1 支的前端（14963634 的 src/aosr/gui/static）跑一次存下來的設計檔，逐位釘住。
_CONTROL_PLAIN = (
    b'{"schema_version":"aosr.zone_design.v1","scheme_id":"zone-naming-sample","zones":{"yL":{"z1":{"mode"'
    b':"search","allowed":["glass_wool_48k","glass_wool_96k","none"],"max_depth_m":0.25},"z2":{"mode":"sea'
    b'rch","allowed":["glass_wool_48k","glass_wool_96k","none"],"max_depth_m":0.25},"z3":{"mode":"search",'
    b'"allowed":["glass_wool_48k","glass_wool_24k","none"]},"auto1":{"mode":"search","allowed":["glass_woo'
    b'l_48k","glass_wool_24k","none"]},"auto2":{"mode":"search","allowed":["glass_wool_48k","glass_wool_24'
    b'k","none"]}},"x0":{"whole_wall":{"mode":"search","allowed":["glass_wool_48k","glass_wool_24k","none"'
    b']}}},"original_scheme_id":null}')
_CONTROL_KEPT = (
    b'{"schema_version":"aosr.zone_design.v1","scheme_id":"zone-naming-sample","zones":{"yL":{"z1":{"mode"'
    b':"search","allowed":["glass_wool_48k","glass_wool_24k","glass_wool_96k","none"],"ranges":{"glass_woo'
    b'l_48k":{"porous_m":[0.05,0.1],"cavity_m":[0.0,0.15]}}},"z2":{"mode":"search","allowed":["glass_wool_'
    b'48k","glass_wool_24k","glass_wool_96k","none"]},"z3":{"mode":"search","allowed":["glass_wool_48k","g'
    b'lass_wool_24k","glass_wool_96k","none"],"max_depth_m":0.3},"auto1":{"mode":"search","allowed":["glas'
    b's_wool_48k","glass_wool_24k","glass_wool_96k","none"]},"auto2":{"mode":"search","allowed":["glass_wo'
    b'ol_48k","glass_wool_24k","glass_wool_96k","none"]}}},"original_scheme_id":null}')


def test_control_without_touching_ranges_the_saved_design_is_byte_for_byte_part_one(tmp_path: Path, browser: Browser) -> None:
    """控制組（只用第 1 支就有的畫面）：前牆兩區拿掉 24K、最深 25 cm；左牆改由程式最佳化；清單卡第一張拿掉 96K；存檔。
    設計檔逐位等於第 1 支的前端跑同一條路存出來的那一份。"""
    path = _prepare(tmp_path, _design())
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        _click_zone(page, "yL/z1")
        _click_zone(page, "yL/z2")
        page.locator("#zone-panel-mode-search").check()
        page.locator(f"#zone-panel-allow-{THIN}").uncheck()
        page.locator("#zone-panel-depth").fill("25")
        _apply(page)
        _click_zone(page, "x0/whole_wall")
        page.locator("#zone-panel-mode-search").check()
        _apply(page)
        page.locator("#zone-list-check-glass_wool_96k").click()
        _settled(page)
        _close_and_save(page)
        assert watched.page_errors == [] and watched.console_errors == []
    assert path.read_bytes() == _CONTROL_PLAIN


def test_control_a_range_already_stored_survives_edits_elsewhere_byte_for_byte(tmp_path: Path, browser: Browser) -> None:
    """控制組：z1 存著自己的範圍，只改 z3 的最深（30 cm）就存檔：z1 那一份原封不動，整份逐位等於第 1 支的前端存出來的。"""
    path = _prepare(tmp_path, _design({"z1": {WOOL: NARROW}}))
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        _click_zone(page, "yL/z3")
        page.locator("#zone-panel-depth").fill("30")
        _apply(page)
        _close_and_save(page)
        assert watched.page_errors == [] and watched.console_errors == []
    assert path.read_bytes() == _CONTROL_KEPT


def test_when_the_range_check_cannot_reach_the_server_nothing_is_applied_and_the_page_is_not_left_locked(tmp_path: Path, browser: Browser) -> None:
    """套用前的限制檢查連不上（請求被中斷）：提示列說範圍沒檢查成、這次沒有套用，設計一個字都沒改；頁面沒被鎖著（還能再按套用），
    連線回來再按一次就套用進去。"""
    _prepare(tmp_path, _design())
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        before = page.evaluate("() => JSON.stringify(zoneDesign)")
        _click_zone(page, "yL/z1")
        page.locator(_cell(page, WOOL, "cavity", "high")).fill("15")
        page.route("**/api/zone-limits", lambda route: route.abort())
        page.locator("#zone-panel-apply").click()
        expect(page.locator("#zone-page-note")).to_contain_text("範圍沒檢查成，這次沒有套用")
        assert page.evaluate("() => [document.querySelector('main').inert, document.getElementById('zone-page').inert]") == [False, False]
        assert page.evaluate("() => JSON.stringify(zoneDesign)") == before
        page.unroute("**/api/zone-limits")
        _apply(page)
        zones = cast(dict[str, dict[str, Doc]], page.evaluate("() => zoneDesign.zones"))
        assert zones["yL"]["z1"]["ranges"] == {WOOL: {"porous_m": [0.05, 0.2], "cavity_m": [0.0, 0.15]}}
        assert watched.page_errors == []
