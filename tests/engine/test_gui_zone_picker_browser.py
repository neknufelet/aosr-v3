"""在分區頁以實際選區、候選清單與存檔驗證選材小框。"""
import json
from pathlib import Path
from typing import cast

import pytest
from playwright.sync_api import Browser, expect

from tests.engine._construction_cases import construction
from tests.engine._sample_material_library import SAMPLE_USER_ENTRIES
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
        for state in ("single", "multiple", "range", "picker", "floor"):
            if state == "multiple":
                _click_zone(page, "yL/auto2")
            elif state == "range":
                page.locator("#zone-panel-range-toggle-glass_wool_48k").click()
            elif state == "picker":
                page.locator("#zone-panel-add").click()
            elif state == "floor":
                page.locator("#zone-panel-cancel").click()
                _click_zone(page, "floor/auto1")
                page.locator("#zone-panel-mode-search").check()
                expect(panel.locator(".zv-form-note")).to_contain_text("這裡沒有可調的材料")
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


DISCARDED = "換了選區，剛才清單上還沒套用的改動放掉了；要的話再做一次，會套到全部選中的區"


@pytest.mark.parametrize("action", ["remove-glass_wool_24k", "only-glass_wool_48k"])
def test_changing_selection_discards_the_old_list_action(tmp_path: Path, browser: Browser, action: str) -> None:
    _boss_room(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        _click_zone(page, "yL/z3")
        before = page.evaluate(_ALLOWED)
        page.locator(f"#zone-panel-{action}").click()
        _click_zone(page, "yL/auto1")
        expect(page.locator("#zone-panel-row-glass_wool_24k")).to_be_visible()
        expect(page.locator("#zone-panel-notice")).to_have_text(DISCARDED)
        expect(page.locator("#zone-panel-undo")).to_have_count(0)
        _apply(page)
        assert page.evaluate(_ALLOWED) == before


@pytest.mark.parametrize("change", ["add", "panel-drop", "wall-whole"])
def test_changing_selection_in_fixed_mode_reports_the_discarded_list(
    tmp_path: Path, browser: Browser, change: str,
) -> None:
    allowed = {"z1": ONLY_48K, "z2": ["none"], "z3": WOOLS, "auto1": [WOOLS[1], "none"], "auto2": ONLY_48K}
    _prepare(tmp_path, _design({"yL": {key: {"mode": "search", "allowed": ids} for key, ids in allowed.items()}}))
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        _click_zone(page, "yL/z3")
        if change == "panel-drop":
            _click_zone(page, "yL/auto1")
        page.locator("#zone-panel-remove-glass_wool_24k").click()
        page.locator("#zone-panel-mode-fixed").check()
        if change == "panel-drop":
            page.locator("#zone-panel-drop-yL-auto1").click()
        elif change == "wall-whole":
            page.locator("#zone-whole-yL").click()
        else:
            _click_zone(page, "yL/auto1")
        _click_zone(page, "yL/auto2")
        page.locator("#zone-panel-mode-search").check()
        expect(page.locator("#zone-panel-notice")).to_have_text(DISCARDED)
        expect(page.locator("#zone-panel-undo")).to_have_count(0)
        expect(page.locator("#zone-panel-row-glass_wool_24k")).to_be_visible()
        _apply(page)
        assert page.evaluate(_ALLOWED) == {f"yL/{key}": ids for key, ids in allowed.items()}


@pytest.mark.parametrize("change", ["drop", "whole"])
def test_selection_change_drops_range_only_draft_but_keeps_depth_and_mode(
    tmp_path: Path, browser: Browser, change: str,
) -> None:
    _boss_room(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        _click_zone(page, "yL/z3")
        _click_zone(page, "yL/auto1")
        page.locator("#zone-panel-depth").fill("16")
        toggle = page.locator("#zone-panel-range-toggle-glass_wool_48k")
        toggle.click()
        high = page.locator("#zone-panel-range-glass_wool_48k-porous-high")
        high.select_option("0.1")
        if change == "whole":
            page.locator("#zone-panel-whole").click()
        else:
            _click_zone(page, "yL/auto1")
        expect(page.locator("#zone-panel-notice")).to_have_text(DISCARDED)
        expect(page.locator("#zone-panel-depth")).to_have_value("16")
        expect(page.locator("#zone-panel-mode-search")).to_be_checked()
        expect(toggle).to_have_attribute("aria-expanded", "false")
        toggle.click()
        expect(high).to_have_value("0.2")


def test_undo_restores_removed_ranges_and_keeps_later_edits_on_surviving_rows(tmp_path: Path, browser: Browser) -> None:
    _boss_room(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        _click_zone(page, "yL/z3")
        page.locator("#zone-panel-range-toggle-glass_wool_24k").click()
        removed = page.locator("#zone-panel-range-glass_wool_24k-porous-high")
        removed.select_option("0.1")
        page.locator("#zone-panel-remove-glass_wool_24k").click()
        other = page.locator("#zone-panel-range-toggle-glass_wool_48k")
        other.click()
        survivor = page.locator("#zone-panel-range-glass_wool_48k-cavity-high")
        survivor.fill("13")
        page.locator("#zone-panel-undo").click()
        expect(removed).to_have_value("0.1")
        expect(other).to_have_attribute("aria-expanded", "false")
        other.click()
        expect(survivor).to_have_value("13")


def test_previous_reference_reminder_gives_the_disabled_options_reason(tmp_path: Path, browser: Browser) -> None:
    prepare_reference(tmp_path)
    with real_server(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_page(page)
        click_reference_zone(page, "x0/a")
        page.locator("#zone-panel-material").select_option(HIT)
        _apply(page)
        click_reference_zone(page, "x0/a")
        page.locator("#zone-panel-mode-search").check()
        page.locator("#zone-panel-add").click()
        option = page.locator(f"#zone-panel-pick-{HIT}")
        expect(option).to_be_disabled()
        note = page.locator(".zv-start-note").filter(has_text="沒放進清單")
        expect(note).to_contain_text(option.locator(".zv-covered-why").inner_text())
        expect(note).not_to_contain_text("要用就從")


def test_starting_list_names_the_other_face_that_cannot_take_the_previous_product(tmp_path: Path, browser: Browser) -> None:
    _boss_room(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        _click_zone(page, "yL/z1")
        _click_zone(page, "floor/auto1")
        page.locator("#zone-panel-mode-search").check()
        note = page.locator(".zv-start-note").filter(has_text="原本指定貼的 自編前牆吸音板")
        expect(note).to_have_text("① 原本指定貼的 自編前牆吸音板 沒放進清單，選中的另一面（地板）貼不了這一筆")


def _append_material(tmp_path: Path, entry: dict[str, object]) -> None:
    path = tmp_path / "materials.json"
    data = json.loads(path.read_bytes())
    data["materials"].append(entry)
    path.write_text(json.dumps(data))


def test_range_description_names_fixed_parts_and_the_tighter_depth_limit(tmp_path: Path, browser: Browser) -> None:
    wool = "sample_capped_wool"
    _prepare(tmp_path, _design({"yL": {"z3": {"mode": "search", "allowed": [wool, "none"]}}}))
    _append_material(tmp_path, {**SAMPLE_USER_ENTRIES[0], "id": wool, "label": "自編可調空腔"})
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        _click_zone(page, "yL/z3")
        words = page.locator(f"#zone-panel-row-{wool} .zv-range")
        expected = "棉厚 7.5 cm 不調；給程式調：空腔 0–12 cm"
        expect(words).to_have_text(expected + "，兩個加起來最多 18 cm（材料本身）")
        depth = page.locator("#zone-panel-depth")
        depth.fill("20")
        expect(words).to_have_text(expected + "，兩個加起來最多 18 cm（材料本身）")
        depth.fill("12")
        expect(words).to_have_text(expected + "，兩個加起來最多 12 cm（這一區最深）")
        page.locator(f"#zone-panel-remove-{wool}").click()
        page.locator("#zone-panel-add").click()
        expect(page.locator(f"#zone-panel-pick-{wool} .zv-description")).to_have_text(
            "棉厚 7.5 cm 不調；給程式調：空腔 0–12 cm")


def test_depth_limit_is_shown_only_when_it_restricts_the_stack(tmp_path: Path, browser: Browser) -> None:
    _boss_room(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        _click_zone(page, "yL/z3")
        direct = page.locator("#zone-panel-row-glass_wool_24k .zv-range")
        cavity = page.locator("#zone-panel-row-glass_wool_48k .zv-range")
        direct_words = "空腔：沒有（直接貼牆）；給程式調：棉厚 5–20 cm"
        cavity_words = "給程式調：棉厚 5–20 cm、空腔 0–25 cm"
        expect(direct).to_have_text(direct_words)
        depth = page.locator("#zone-panel-depth")
        depth.fill("15")
        expect(cavity).to_have_text(cavity_words + "，兩個加起來最多 15 cm（這一區最深）")
        expect(direct).to_have_text(direct_words + "，總厚最多 15 cm（這一區最深）")
        for limit in ("20", "25"):
            depth.fill(limit)
            expect(direct).to_have_text(direct_words)
            expect(cavity).to_have_text(cavity_words + f"，兩個加起來最多 {limit} cm（這一區最深）")
        _click_zone(page, "yL/auto2")
        expect(cavity).to_have_text(cavity_words + "，兩個加起來最多 25 cm（這幾區最深）")


def test_rows_only_offer_range_and_only_buttons_when_they_can_act(tmp_path: Path, browser: Browser) -> None:
    still = "one_thickness"
    _prepare(tmp_path, _design({"yL": {"z3": {"mode": "search", "allowed": [still, FLOOR_ONLY, "none"]}}}))
    _append_material(tmp_path, {**SAMPLE_USER_ENTRIES[0], "id": still, "label": "自編單一厚度",
                     "porous_thickness": {"adjustable": True, "choices_m": [0.075], "range_m": [0.075, 0.075], "initial_m": 0.075},
                     "cavity": {"adjustable": False, "range_m": [0.1, 0.1], "initial_m": 0.1}, "suggested_total_depth_m": 0.175})
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        _click_zone(page, "yL/z3")
        for entry in (still, FLOOR_ONLY):
            expect(page.locator(f"#zone-panel-remove-{entry}")).to_be_visible()
        expect(page.locator(f"#zone-panel-range-toggle-{still}")).to_have_count(0)
        expect(page.locator(f"#zone-panel-only-{FLOOR_ONLY}")).to_have_count(0)
        words = page.locator(f"#zone-panel-row-{still} .zv-range")
        expect(words).to_have_text("棉厚 7.5 cm 不調；空腔 10 cm 不調")


def test_cross_face_footer_counts_only_candidates_omitted_for_placement(tmp_path: Path, browser: Browser) -> None:
    _prepare(tmp_path, _design({"yL": {"z3": {"mode": "search", "allowed": ONLY_48K}},
                                "floor": {"auto1": {"mode": "search", "allowed": ["none"]}}}))
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        _click_zone(page, "yL/z3")
        _click_zone(page, "floor/auto1")
        page.locator("#zone-panel-remove-glass_wool_48k").click()
        page.locator("#zone-panel-add").click()
        omitted = page.evaluate("""zoneLibrary.filter(e => (e.kind === 'optimizable' || fixedCandidate(e)) &&
            e.id !== 'glass_wool_48k' && ['yL', 'floor'].some(w => !e.placement.surfaces.includes(w))).length""")
        expect(page.locator(".zv-picker-foot")).to_have_text(f"選中的面有一面貼不了的 {omitted} 筆不列")


def test_picker_can_restore_removed_wall_candidate_across_wall_and_floor(tmp_path: Path, browser: Browser) -> None:
    _prepare(tmp_path, _design({"yL": {"z3": {"mode": "search", "allowed": ONLY_48K}},
                                "floor": {"auto1": {"mode": "search", "allowed": ["none"]}}}))
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        _click_zone(page, "yL/z3")
        _click_zone(page, "floor/auto1")
        page.locator("#zone-panel-remove-glass_wool_48k").click()
        page.locator("#zone-panel-add").click()
        option = page.locator("#zone-panel-pick-glass_wool_48k")
        expect(option).to_be_enabled()
        expect(page.locator("#zone-panel")).not_to_contain_text("這裡沒有可調的材料")
        option.click()
        _apply(page)
        assert page.evaluate(_ALLOWED) == {"yL/z3": ONLY_48K, "floor/auto1": ["none"]}


def test_stored_original_wall_construction_explains_why_and_only_allows_removal(tmp_path: Path, browser: Browser) -> None:
    entry = "sample_wall_back"
    _prepare(tmp_path, _design({"yL": {"z3": {"mode": "search", "allowed": [entry, "none"]}}}))
    _append_material(tmp_path, construction(backing="zone_wall", id=entry))
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        _click_zone(page, "yL/z3")
        row = page.locator(f"#zone-panel-row-{entry}")
        expect(row).to_contain_text("背後要接這一區的原牆，不能整塊當候選")
        expect(row.locator("button, summary")).to_have_count(1)
        page.locator(f"#zone-panel-remove-{entry}").click()
        expect(row).to_have_count(0)


def test_floor_without_tunable_material_explains_the_search_start(tmp_path: Path, browser: Browser) -> None:
    _prepare(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        _click_zone(page, "floor/auto1")
        page.locator("#zone-panel-mode-search").check()
        note = page.locator("#zone-panel .zv-form-note")
        expect(note).to_be_visible()
        expect(note).to_contain_text("這裡沒有可調的材料")
        size = note.evaluate("node => parseFloat(getComputedStyle(node).fontSize)")
        assert size >= 13
        page.locator("#zone-panel-add").click()
        expect(page.locator("#zone-panel")).to_contain_text("這裡沒有可調的材料：程式只能從「不貼」開始，可以再勾固定成品讓它比")


def test_disabled_apply_uses_the_main_button_colours(tmp_path: Path, browser: Browser) -> None:
    _boss_room(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        page.locator("#zone-whole-yL").click()
        apply = page.locator("#zone-panel-apply")
        expect(apply).to_be_disabled()
        expect(apply).to_have_css("background-color", "rgb(143, 155, 156)")
        expect(apply).to_have_css("color", "rgb(255, 255, 255)")


def test_picker_partial_candidate_is_added_only_to_the_remaining_zone(tmp_path: Path, browser: Browser) -> None:
    _boss_room(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        for key in ("z3", "auto2", "auto1"):
            _click_zone(page, f"yL/{key}")
        before = page.evaluate(_ALLOWED)
        page.locator("#zone-panel-add").click()
        option = page.locator("#zone-panel-pick-glass_wool_48k")
        expect(option).to_contain_text("2 區已有")
        option.click()
        _apply(page)
        assert page.evaluate(_ALLOWED) == {**before, "yL/auto1": [*before["yL/auto1"], WOOLS[0]]}


def test_click_outside_closes_picker_and_keeps_the_selection(tmp_path: Path, browser: Browser) -> None:
    _boss_room(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        _click_zone(page, "yL/z3")
        page.locator("#zone-panel-add").click()
        expect(page.locator("#zone-panel-picker-search")).to_be_focused()
        page.locator("#zone-panel h2").click()
        expect(page.locator("#zone-panel-picker")).to_have_count(0)
        expect(page.locator("#zone-panel-add")).to_have_attribute("aria-expanded", "false")


def test_enter_skips_the_first_disabled_match(tmp_path: Path, browser: Browser) -> None:
    _boss_room(tmp_path)
    _append_material(tmp_path, construction(product=True, id="keyboard_sample", label="48K 玻璃棉 鍵盤樣品"))
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        _click_zone(page, "yL/z3")
        page.locator("#zone-panel-add").click()
        search = page.locator("#zone-panel-picker-search")
        search.fill("48K 玻璃棉")
        options = page.locator("#zone-panel-picker button")
        expect(options.first).to_be_disabled()
        expect(page.locator("#zone-panel-pick-keyboard_sample")).to_be_enabled()
        search.press("Enter")
        expect(search).to_have_count(0)
        expect(page.locator("#zone-panel-row-keyboard_sample")).to_be_visible()


def test_notice_undo_open_layers_and_mixed_range_text_meet_readability_minimum(tmp_path: Path, browser: Browser) -> None:
    layer = "sample_layered"
    common = {"mode": "search", "allowed": [WOOLS[0], layer, "none"]}
    _prepare(tmp_path, _design({"yL": {"z3": common, "auto2": {**common, "ranges": {WOOLS[0]: {"porous_m": [0.05, 0.1]}}}}}))
    _append_material(tmp_path, construction(id=layer))
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        _click_zone(page, "yL/z3")
        _click_zone(page, "yL/auto2")
        page.locator("#zone-panel-remove-none").click()
        page.locator(f"#zone-panel-layers-{layer} summary").click()
        page.locator("#zone-panel-range-toggle-glass_wool_48k").click()
        samples = page.locator("#zone-panel").evaluate(_TEXT_STYLES)
        for words in ("拿掉 也可以不貼", "復原", "每一層現在還不能調", "各區不同；不動它", "③ 5–20、④ 5–10 cm"):
            assert any(words in row["words"] for row in samples), words
        assert [row for row in samples if row["size"] < 13 or row["contrast"] < 4.5] == []


@pytest.mark.parametrize("depth", ["19.96", "0.04"])
def test_depth_caption_uses_the_same_millimetres_as_apply(tmp_path: Path, browser: Browser, depth: str) -> None:
    _boss_room(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        _click_zone(page, "yL/z3")
        page.locator("#zone-panel-depth").fill(depth)
        line = page.locator("#zone-panel-row-glass_wool_24k .zv-range")
        expect(line).to_contain_text("棉厚")
        expect(line).not_to_contain_text("最多")


def test_starting_list_groups_two_pasted_zones_by_the_ineligible_wall(tmp_path: Path, browser: Browser) -> None:
    _boss_room(tmp_path)
    path = tmp_path / "materials.json"
    library = json.loads(path.read_bytes())
    entry = next(row for row in library["materials"] if row["id"] == "sample_front_panel")
    entry["id"] = "sample_floor_panel"
    path.write_text(json.dumps(library))
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        for zone in ("yL/z1", "yL/z2"):
            _click_zone(page, zone)
        page.locator("#zone-panel-mode-search").check()
        reason = page.locator(".zv-start-note").filter(has_text="沒放進清單")
        expect(reason).to_have_text("①② 原本指定貼的 自編前牆吸音板 沒放進清單，前牆那一面貼不了這一筆")
