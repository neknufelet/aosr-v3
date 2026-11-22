"""真瀏覽器驗證分區的起始候選清單，以及整塊候選在逐區表、小框與清單卡顯示的代價。"""
from __future__ import annotations

import json
from pathlib import Path

from playwright.sync_api import Browser, Locator, Page, expect

from tests.engine._construction_cases import construction
from tests.engine._gui_cache import gui_startup_identity_memo as gui_startup_identity_memo
from tests.engine.test_gui_browser import _open, _serve, browser as browser
from tests.engine.test_gui_zone_elevation_browser import _open_scheme
from tests.engine.test_gui_zone_material_pick_browser import PLAIN_PRODUCT, _TEXT_STYLES, _design, _pick_search
from tests.engine.test_gui_zone_modes_browser import _choose
from tests.engine.test_gui_zone_overview_browser import SAMPLE, WOOLS
from tests.engine.test_gui_zone_overview_tasks_browser import (
    _apply, _click_zone, _open_overview, _prepare, _save, _stored)


def _prepare_candidates(tmp_path: Path) -> None:
    _prepare(tmp_path)
    path = tmp_path / "materials.json"
    library = json.loads(path.read_bytes())
    tunable = next(entry for entry in library["materials"] if entry["kind"] == "optimizable")
    panel = construction("rigid", id="sample_rigid", label="自編剛背構造")
    product = {"id": "sample_product", "kind": "fixed", "label": "自編吸音成品", "provenance": "model",
               "source": "考卷自編", "material": {"kind": "assembly", "backing": "rigid", "layers": panel["layers"]}}
    library["materials"].extend([tunable | {"id": "sample_tunable", "label": "自編可調岩棉"}, panel, product])
    path.write_text(json.dumps(library), encoding="utf-8")


def _open_candidate_table(page: Page) -> Locator:
    page.locator("#elevation-wall-yL").click()
    details = page.locator("#zone-allowed-box-yL-z3")
    if not details.evaluate("box => box.open"):
        details.locator("summary").click()
    return page.locator("#zone-allowed-yL-z3")


def test_new_search_zone_saves_explicit_builtin_candidates(tmp_path: Path, browser: Browser) -> None:
    """補既有行為：自編可調材料、剛背構造、固定成品都能用，首次改由程式最佳化仍只存內建玻璃棉和不貼。"""
    _prepare_candidates(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, SAMPLE)
        page.locator("#elevation-wall-yL").click()
        _choose(page, "#zone-mode-yL-z3", "search")
        table = _open_candidate_table(page)
        for choice in ("sample_tunable", "sample_rigid", "sample_product"):
            box = table.locator(f"#zone-allow-yL-z3-{choice}")
            expect(box).to_be_visible()
            expect(box).to_be_enabled()
        _save(page, design=True)
    zones, _ = _stored(tmp_path)
    assert zones["yL"]["z3"] == {"mode": "search", "allowed": WOOLS}


def _expect_candidate_cost(surface: Locator, present: bool) -> None:
    expect(surface).to_be_visible()
    note = surface.locator(".zone-candidate-cost")
    if not present:
        expect(note).to_have_count(0)
        return
    expect(note).to_be_visible()
    words = note.inner_text()
    for phrase in ("整塊貼上去的材料", "每筆都要放進訓練", "再多核對一次", "清單、深度上限、裝法或方案有改",
                   "就要重建替身", "第三階段搜尋頁", "「預覽轉好的參數」", "是否需要重建及原因"):
        assert phrase in words
    assert not any(char.isdecimal() for char in words)
    style = next(row for row in surface.evaluate(_TEXT_STYLES) if row["words"] == words)
    assert style["size"] >= 12 and style["contrast"] >= 4.5, style


def test_whole_candidate_cost_follows_the_choice_in_all_three_views(tmp_path: Path, browser: Browser) -> None:
    _prepare_candidates(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        for choice, checked in (("sample_product", True), ("sample_product", False), ("sample_rigid", True)):
            _click_zone(page, "yL/z3")
            _pick_search(page)
            if checked:
                page.locator("#zone-panel-add").click()
                page.locator(f"#zone-panel-pick-{choice}").click()
            else:
                page.locator(f"#zone-panel-remove-{choice}").click()
            form = page.locator("#zone-panel-form")
            _expect_candidate_cost(form, checked)
            if checked and choice == "sample_product":
                product = form.locator("#zone-panel-row-sample_product")
                expect(product).to_have_attribute("data-state", "all")
            _apply(page)
            card = page.locator("#zone-list")
            _expect_candidate_cost(card, checked)
            if checked and choice == "sample_product":
                expect(card.locator("#zone-list-check-sample_product")).to_be_checked()
            page.locator("#zone-page-close").click()
            _expect_candidate_cost(_open_candidate_table(page), checked)
            page.locator("#zone-page-open").click()
        assert watched.page_errors == [] and watched.console_errors == []


def test_only_tunable_candidates_and_none_show_no_cost_in_all_three_views(tmp_path: Path, browser: Browser) -> None:
    _prepare_candidates(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        _click_zone(page, "yL/z3")
        _pick_search(page)
        page.locator("#zone-panel-add").click()
        page.locator("#zone-panel-pick-sample_tunable").click()
        expect(page.locator("#zone-panel-none-row")).to_have_attribute("data-state", "all")
        _expect_candidate_cost(page.locator("#zone-panel-form"), False)
        _apply(page)
        _expect_candidate_cost(page.locator("#zone-list"), False)
        page.locator("#zone-page-close").click()
        _expect_candidate_cost(_open_candidate_table(page), False)
        assert watched.page_errors == [] and watched.console_errors == []


def test_panel_shows_cost_when_only_one_selected_zone_has_a_fixed_product(tmp_path: Path, browser: Browser) -> None:
    _prepare(tmp_path, _design({"yL": {
        "z1": {"mode": "search", "allowed": ["none"]},
        "z2": {"mode": "search", "allowed": ["none", PLAIN_PRODUCT]},
    }}))
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        _click_zone(page, "yL/z1")
        _click_zone(page, "yL/z2")
        form = page.locator("#zone-panel-form")
        product = form.locator(f"#zone-panel-row-{PLAIN_PRODUCT}")
        expect(product).to_have_attribute("data-state", "some")
        _expect_candidate_cost(form, True)
        form.locator(f"#zone-panel-addto-{PLAIN_PRODUCT}").click()
        expect(product).to_have_attribute("data-state", "all")
        form.locator(f"#zone-panel-remove-{PLAIN_PRODUCT}").click()
        _expect_candidate_cost(form, False)
