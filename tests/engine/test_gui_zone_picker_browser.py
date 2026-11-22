"""在分區頁以實際選區、候選清單與存檔驗證選材小框。"""
from pathlib import Path
from typing import cast

from playwright.sync_api import Browser, expect

from tests.engine._construction_gui_server import real_server
from tests.engine._gui_cache import gui_startup_identity_memo as gui_startup_identity_memo
from tests.engine.test_gui_browser import _open, _serve, browser as browser
from tests.engine.test_gui_vpr_reference import HIT
from tests.engine.test_gui_vpr_reference_browser import (
    COMPUTED_MS, _click_zone as click_reference_zone, _no_library, _open_page, _prepare as prepare_reference,
)
from tests.engine.test_gui_zone_material_pick_browser import (
    FLOOR_ONLY, ONLY_48K, PLAIN_PRODUCT, _TEXT_STYLES, _add_candidate, _boss_room, _design,
)
from tests.engine.test_gui_zone_overview_browser import SAMPLE, WOOLS
from tests.engine.test_gui_zone_overview_tasks_browser import (
    _ALLOWED, _apply, _click_zone, _open_overview, _prepare, _send_to_search, _stored,
)


def test_picker_lists_only_selected_candidates_and_filters_both_groups(tmp_path: Path, browser: Browser) -> None:
    _boss_room(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _open_overview(page)
        _click_zone(page, "yL/z3")
        rows = page.locator("#zone-panel-list > [data-choice]")
        assert rows.evaluate_all("rows => rows.map(row => row.dataset.choice)") == [WOOLS[0], WOOLS[1], "none"]
        expect(page.locator("#zone-panel-picker")).to_have_count(0)
        page.locator("#zone-panel-add").click()
        search = page.locator("#zone-panel-picker-search")
        expect(search).to_be_focused()
        elsewhere = page.evaluate("""zoneLibrary.filter(e => (e.kind === 'optimizable' || fixedCandidate(e)) &&
            e.backing !== 'zone_wall' && !e.placement.surfaces.includes('yL')).length""")
        expect(page.locator("#zone-panel-picker")).to_contain_text(f"只能貼在別面的 {elsewhere} 筆不列")
        search.fill("玻璃")
        for group in ("參數構造", "成品"):
            options = page.get_by_role("group", name=group, exact=True).locator("button")
            assert options.count() > 0
            for option in options.all():
                expect(option.locator("mark").first).to_have_text("玻璃")
                expect(option.locator(".zv-description")).to_be_visible()
        search.fill("NoMatch")
        expect(page.locator("#zone-panel-picker")).to_contain_text("沒有符合「NoMatch」的")
        search.press("Escape")
        expect(search).to_have_count(0)
        page.locator("#zone-panel-add").click()
        search.fill("96k")
        search.press("Enter")
        expect(search).to_have_count(0)
        assert rows.evaluate_all("rows => rows.map(row => row.dataset.choice)") == WOOLS
        assert watched.page_errors == []


def test_only_and_undo_restore_each_zones_draft_and_ranges(tmp_path: Path, browser: Browser) -> None:
    _boss_room(tmp_path)
    design = tmp_path / "designs" / f"{SAMPLE}.json"
    original = design.read_bytes()
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        for zone in ("z3", "auto2", "auto1"):
            _click_zone(page, f"yL/{zone}")
        page.locator("#zone-panel-range-toggle-glass_wool_48k").click()
        high = page.locator("#zone-panel-range-glass_wool_48k-porous-high")
        high.select_option("0.1")
        rows = page.locator("#zone-panel-list > [data-choice]")
        before = rows.all_inner_texts()
        page.locator("#zone-panel-only-glass_wool_24k").click()
        notice = page.locator("#zone-panel-notice")
        for words in ("③", "④", "⑤", "48K 玻璃棉（2 區有）", "96K 玻璃棉（1 區有）", "「也可以不貼」沒動"):
            expect(notice).to_contain_text(words)
        first_step = rows.all_inner_texts()
        expect(page.locator("#zone-panel-none-row")).to_have_attribute("data-state", "all")
        page.locator("#zone-panel-remove-none").click()
        expect(page.locator("#zone-panel-none-row")).to_have_attribute("data-state", "none")
        assert design.read_bytes() == original
        page.locator("#zone-panel-undo").click()
        assert rows.all_inner_texts() == first_step
        expect(page.locator("#zone-panel-none-row")).to_have_attribute("data-state", "all")
        expect(notice).to_contain_text("已只留")
        page.locator("#zone-panel-undo").click()
        assert rows.all_inner_texts() == before
        expect(high).to_have_value("0.1")
        expect(notice).to_have_count(0)
        assert design.read_bytes() == original
        assert watched.page_errors == []


def test_partial_none_and_material_actions_save_only_the_affected_zones(tmp_path: Path, browser: Browser) -> None:
    _boss_room(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        _click_zone(page, "yL/auto2")
        page.locator("#zone-panel-remove-none").click()
        _apply(page)
        for zone in ("z3", "auto2", "auto1"):
            _click_zone(page, f"yL/{zone}")
        expect(page.locator("#zone-panel-addto-none")).to_have_text("加到 1 區")
        page.locator("#zone-panel-remove-glass_wool_48k").click()
        notice = page.locator("#zone-panel-notice")
        for words in ("③", "④", "48K 玻璃棉", "「也可以不貼」沒動"):
            expect(notice).to_contain_text(words)
        expect(notice).not_to_contain_text("⑤")
        page.locator("#zone-panel-addto-glass_wool_24k").click()
        expect(notice).to_contain_text("④")
        page.locator("#zone-panel-remove-none").click()
        expect(page.locator("#zone-panel-none-row")).to_contain_text("沒有")
        expect(page.locator("#zone-panel-remove-none")).to_have_count(0)
        expect(page.locator("#zone-panel-addto-none")).to_have_text("加上")
        page.locator("#zone-panel-addto-none").click()
        expect(page.locator("#zone-panel-addto-none")).to_have_count(0)
        _apply(page)
        _send_to_search(page)
        assert watched.page_errors == []
    zones, _ = _stored(tmp_path)
    for zone in ("z3", "auto2", "auto1"):
        expected = [WOOLS[1], "none"] + ([WOOLS[2]] if zone == "auto2" else [])
        setting = cast(dict[str, object], zones["yL"][zone])
        assert sorted(cast(list[str], setting["allowed"])) == sorted(expected)


def test_mixed_methods_name_the_actual_starting_list_and_previous_product(tmp_path: Path, browser: Browser) -> None:
    _boss_room(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        page.locator("#zone-whole-yL").click()
        panel = page.locator("#zone-panel")
        expect(panel).to_contain_text("現在：2 區指定貼、3 區由程式最佳化")
        page.locator("#zone-panel-mode-search").check()
        expect(panel).to_contain_text("①② 原本不是由程式最佳化，先放：48K 玻璃棉、24K 玻璃棉、也可以不貼")
        expect(panel).to_contain_text("①② 原本指定貼的 自編前牆吸音板 沒放進清單")
        assert watched.page_errors == []


def test_picker_width_and_readability_in_four_panel_states(tmp_path: Path, browser: Browser) -> None:
    _boss_room(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        page.set_viewport_size({"width": 1440, "height": 960})
        _open_overview(page)
        _click_zone(page, "yL/z3")
        panel = page.locator("#zone-panel")
        assert abs(panel.evaluate("p => p.getBoundingClientRect().width") - 472) <= 1
        for state in ("single", "multiple", "range", "picker"):
            if state == "multiple":
                _click_zone(page, "yL/auto2")
            elif state == "range":
                page.locator("#zone-panel-range-toggle-glass_wool_48k").click()
            elif state == "picker":
                page.locator("#zone-panel-add").click()
            samples = panel.evaluate(_TEXT_STYLES)
            assert samples and all(row["size"] >= 13 and row["contrast"] >= 4.5 for row in samples), (state, samples)
            if state == "multiple":
                markers = [row for row in samples if "border" in row]
                assert {row["stroke"] for row in markers} == {"solid", "dashed"}
                assert all(row["border"] >= 3 for row in markers), markers
            assert page.evaluate("document.documentElement.scrollWidth <= 1440")
        assert watched.page_errors == []


def test_only_names_the_face_that_cannot_take_it_and_keeps_that_faces_none(tmp_path: Path, browser: Browser) -> None:
    _prepare(tmp_path, _design({"yL": {"z3": {"mode": "search", "allowed": ONLY_48K}},
                                "floor": {"auto2": {"mode": "search", "allowed": [FLOOR_ONLY, WOOLS[0], "none"]}}}))
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        _click_zone(page, "yL/z3")
        _click_zone(page, "floor/auto2")
        page.locator("#zone-panel-only-glass_wool_48k").click()
        notice = page.locator("#zone-panel-notice")
        expect(notice).to_contain_text("②那一面貼不了 48K 玻璃棉")
        expect(notice).to_contain_text("②只剩「也可以不貼」")
        _apply(page)
        assert page.evaluate(_ALLOWED) == {"floor/auto2": ["none"], "yL/z3": ONLY_48K}


def test_missing_candidate_can_be_removed_and_a_product_alone_blocks_apply(tmp_path: Path, browser: Browser) -> None:
    missing = "sample_missing"
    _prepare(tmp_path, _design({"yL": {"z1": {"mode": "search", "allowed": [missing, PLAIN_PRODUCT]}}}))
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        _click_zone(page, "yL/z1")
        expect(page.locator(f"#zone-panel-row-{missing}")).to_contain_text("照原樣留著，開跑時伺服器會點名")
        page.locator(f"#zone-panel-remove-{missing}").click()
        expect(page.locator("#zone-panel-apply")).to_be_disabled()
        expect(page.locator("#zone-panel-problem")).to_contain_text("只放了固定成品，程式沒得選。")
        page.locator("#zone-panel-undo").click()
        expect(page.locator("#zone-panel-apply")).to_be_enabled()


def test_add_to_missing_zone_preserves_an_existing_candidate_on_an_ineligible_face(tmp_path: Path, browser: Browser) -> None:
    _prepare(tmp_path, _design({
        "yL": {"z3": {"mode": "search", "allowed": ["none"]}},
        "floor": {"auto1": {"mode": "search", "allowed": ONLY_48K},
                  "auto2": {"mode": "search", "allowed": ["none"]}},
    }))
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        for zone in ("yL/z3", "floor/auto1", "floor/auto2"):
            _click_zone(page, zone)
        expect(page.locator("#zone-panel-row-glass_wool_48k")).to_contain_text("2 區那一面貼不了這一種，不會加")
        add = page.locator("#zone-panel-addto-glass_wool_48k")
        expect(add).to_have_text("加到 1 區")
        add.click()
        _apply(page)
        _send_to_search(page)
    zones, _ = _stored(tmp_path)
    assert zones == {"floor": {"auto1": {"mode": "search", "allowed": ONLY_48K},
                                "auto2": {"mode": "search", "allowed": ["none"]}},
                     "yL": {"z3": {"mode": "search", "allowed": ["none", WOOLS[0]]}}}


def test_adding_the_previously_pasted_product_clears_its_reminder_until_undo(tmp_path: Path, browser: Browser) -> None:
    _boss_room(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        page.locator("#zone-whole-yL").click()
        page.locator("#zone-panel-mode-search").check()
        missing = page.locator("#zone-panel .zv-start-note").filter(has_text="沒放進清單")
        expect(missing).to_contain_text("①② 原本指定貼的 自編前牆吸音板")
        _add_candidate(page, "sample_front_panel")
        expect(missing).to_have_count(0)
        expect(page.locator("#zone-panel-row-sample_front_panel")).to_have_attribute("data-state", "all")
        expect(page.locator("#zone-panel .zv-start-note")).to_have_text(
            "①② 原本不是由程式最佳化，先放：48K 玻璃棉、24K 玻璃棉、也可以不貼")
        page.locator("#zone-panel-undo").click()
        expect(missing).to_contain_text("①② 原本指定貼的 自編前牆吸音板")


def test_pasted_reference_explanation_meets_the_panel_font_and_contrast_minimum(tmp_path: Path, browser: Browser) -> None:
    prepare_reference(tmp_path)
    with real_server(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_page(page)
        click_reference_zone(page, "x0/a")
        page.locator("#zone-panel-material").select_option(HIT)
        note = page.locator("#zone-panel-ref-note")
        expect(note).to_be_visible(timeout=COMPUTED_MS)
        note.locator("summary").click()
        expect(note.locator("details p")).to_be_visible()
        samples = page.locator("#zone-panel").evaluate(_TEXT_STYLES)
        assert any(row["words"] == "藍底：" for row in samples)
        assert [row for row in samples if row["size"] < 13 or row["contrast"] < 4.5] == []


def test_unread_library_explains_why_the_panel_cannot_list_candidates(tmp_path: Path, browser: Browser) -> None:
    _prepare(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _no_library(page)
        _open_overview(page)
        _click_zone(page, "yL/z1")
        page.locator("#zone-panel-mode-search").check()
        expect(page.locator("#zone-panel-search-from")).to_have_text("程式從全部可調材料與「不貼」裡選（材料庫沒讀全，列不出是哪幾種）")
        expect(page.locator("#zone-panel-list")).to_have_count(0)
        expect(page.locator("#zone-panel-add")).to_have_count(0)


def test_stored_covered_product_stays_listed_and_can_be_removed_independently(tmp_path: Path, browser: Browser) -> None:
    product = "glass_wool_48k_50mm"
    _prepare(tmp_path, _design({"yL": {"z3": {"mode": "search", "allowed": [WOOLS[0], product, "none"]}}}))
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        _click_zone(page, "yL/z3")
        row = page.locator(f"#zone-panel-row-{product}")
        expect(row).to_have_attribute("data-covered", WOOLS[0])
        expect(row.locator(".zv-covered-why")).to_contain_text("48K 玻璃棉 已經包含它")
        page.locator(f"#zone-panel-remove-{product}").click()
        expect(row).to_have_count(0)
        page.locator("#zone-panel-add").click()
        expect(page.locator(f"#zone-panel-pick-{product}")).to_be_disabled()
        _apply(page)
        assert page.evaluate(_ALLOWED) == {"yL/z3": ONLY_48K}
