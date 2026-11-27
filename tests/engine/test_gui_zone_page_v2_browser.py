"""在自編房間，以實際視窗命中測試與存檔驗分區頁。"""
from pathlib import Path
from typing import cast

import pytest
from playwright.sync_api import Browser, Page, expect

from tests.engine._construction_cases import construction
from tests.engine._gui_cache import gui_startup_identity_memo as gui_startup_identity_memo
from tests.engine.test_gui_browser import _open, _serve, browser as browser
from tests.engine.test_gui_zone_material_pick_browser import _design
from tests.engine.test_gui_zone_overview_browser import WOOLS
from tests.engine.test_gui_zone_overview_tasks_browser import (
    PRODUCT, _apply, _click_zone, _materials_before, _open_overview, _prepare, _save, _send_to_search, _stored,
)


def _saved_settings(tmp_path: Path) -> dict[str, dict[str, dict[str, object]]]:
    settings, _ = _stored(tmp_path)
    return cast(dict[str, dict[str, dict[str, object]]], settings)


def _open_zone(page: Page) -> None:
    page.set_viewport_size({"width": 1440, "height": 900})
    _open_overview(page)
    _click_zone(page, "yL/z1")


def test_apply_visible_in_every_mode(tmp_path: Path, browser: Browser) -> None:
    _prepare(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_zone(page)
        for mode in ("search", "fixed", "keep_original"):
            page.locator(f"#zone-panel-mode-{mode}").check()
            hit = page.locator("#zone-panel-apply").evaluate("""button => {
                const r = button.getBoundingClientRect(), x = r.x + r.width / 2, y = r.y + r.height / 2;
                return {y, hit: document.elementFromPoint(x, y) === button};
            }""")
            assert 0 <= hit["y"] < 900 and hit["hit"], (mode, hit)
        expect(page.locator("#zone-panel")).to_be_visible()


def test_fixed_row_writes_chosen_material(tmp_path: Path, browser: Browser) -> None:
    _prepare(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        entry = construction("rigid", id="sample_layered_panel", label="試驗室布面板")
        assert page.request.put(f"{base}/api/materials/sample_layered_panel", data=entry).ok
        page.reload()
        _open_zone(page)
        page.locator("#zone-panel-mode-fixed").check()
        page.get_by_placeholder("找材料").fill("試驗室")
        expect(page.locator("#zone-panel-material [role=radio]:visible")).to_have_count(1)
        page.locator("#zone-panel-material-sample_layered_panel").click()
        _apply(page)
        page.locator("#zone-page-close").click()
        _save(page, design=True)
    _, surfaces = _stored(tmp_path)
    settings = _saved_settings(tmp_path)
    assert settings["yL"]["z1"]["mode"] == "fixed"
    assert surfaces["yL"]["z1"]["layers"] == entry["layers"]


def test_candidate_checks_write_exact_list(tmp_path: Path, browser: Browser) -> None:
    _prepare(tmp_path, _design({"yL": {"z1": {"mode": "search", "allowed": ["none"]}}}))
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_zone(page)
        for choice in [WOOLS[0], WOOLS[1], WOOLS[0]]:
            page.locator(f"#zone-panel-pick-{choice}").click()
        _apply(page)
        page.locator("#zone-page-close").click()
        _save(page, design=True)
    settings = _saved_settings(tmp_path)
    assert sorted(cast(list[str], settings["yL"]["z1"]["allowed"])) == sorted(["none", WOOLS[1]])


def test_result_button_applies_before_search(tmp_path: Path, browser: Browser) -> None:
    _prepare(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_zone(page)
        page.locator("#zone-panel-mode-search").check()
        page.locator("#zone-panel-depth").fill("24")
        expect(page.locator("#zone-page-search")).to_have_text("套用並拿去搜尋")
        _send_to_search(page)
    settings = _saved_settings(tmp_path)
    assert settings["yL"]["z1"]["mode"] == "search"
    assert settings["yL"]["z1"]["max_depth_m"] == 0.24


@pytest.mark.parametrize(("mode", "backing"), [("search", "rigid"), ("fixed", "zone_wall")])
def test_drawer_saves_into_selected_zones(tmp_path: Path, browser: Browser, mode: str, backing: str) -> None:
    _prepare(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_zone(page)
        _click_zone(page, "yL/z2")
        page.locator(f"#zone-panel-mode-{mode}").check()
        page.locator("#zone-panel-depth").fill("15")
        before = page.request.get(f"{base}/api/materials").json()["materials"]
        page.locator("#zone-panel-build").click()
        frame = page.frame_locator("#zone-builder iframe")
        expect(frame.locator("nav")).to_be_hidden()
        expect(frame.locator(f'[name="backing"][value="{backing}"]')).to_be_checked()
        original = frame.locator('[name="backing"][value="zone_wall"]')
        if mode == "search":
            expect(original).to_be_disabled()
            expect(original.locator("xpath=ancestor::label")).to_contain_text("程式挑還不收")
        else:
            expect(original).to_be_enabled()
        expect(frame.locator("#zone-depth")).to_have_text("貼在 前牆 ①② 只用到 15 cm 以內")
        # 保持內嵌框的視窗身分，僅換掉來源，以免另一道檢查掩蓋同源檢查失效。
        refreshed = page.evaluate("""id => {
            const fetch = window.fetch;
            let count = 0;
            window.fetch = (...args) => { if (args[0] === '/api/materials') count++; return fetch(...args); };
            try {
                dispatchEvent(new MessageEvent('message', {origin: 'https://example.invalid',
                    source: document.querySelector('#zone-builder iframe').contentWindow,
                    data: {type: 'aosr-material-saved', id}}));
                return count;
            } finally { window.fetch = fetch; }
        }""", PRODUCT)
        assert refreshed == 0
        expect(page.locator("#zone-builder")).to_be_visible()
        frame.locator("#name").fill("自編試驗棉")
        frame.locator('#palette [data-kind="porous"]').click()
        button = frame.locator("#save")
        expect(button).to_have_text("存入材料庫並加入候選" if mode == "search" else "存入材料庫並選它")
        button.click()
        expect(page.locator("#zone-builder")).to_have_count(0)
        after = page.request.get(f"{base}/api/materials").json()["materials"]
        assert len(after) == len(before) + 1
        entry, = [row["entry"] for row in after if row["entry"]["label"] == "自編試驗棉"]
        assert entry["backing"] == backing
        prefix = "pick" if mode == "search" else "material"
        expect(page.locator(f"#zone-panel-{prefix}-{entry['id']}")).to_have_attribute("aria-checked", "true")
        _apply(page)
        page.locator("#zone-page-close").click()
        _save(page, design=True)
    _, surfaces = _stored(tmp_path)
    settings = _saved_settings(tmp_path)
    original_surfaces = _materials_before()["yL"]
    for key in ("z1", "z2"):
        assert settings["yL"][key]["mode"] == mode
        if mode == "search":
            assert surfaces["yL"][key] == original_surfaces[key]["wall"]
            assert sorted(cast(list[str], settings["yL"][key]["allowed"])) == sorted([*WOOLS, entry["id"]])
        else:
            assert surfaces["yL"][key]["kind"] == "mounted_on_wall"
            assert surfaces["yL"][key]["wall"] == original_surfaces[key]["wall"]
            assert surfaces["yL"][key]["layers"] == entry["layers"]


def test_candidate_options_follow_pointer_and_keyboard(tmp_path: Path, browser: Browser) -> None:
    _prepare(tmp_path, _design({"yL": {"z1": {"mode": "search", "allowed": [WOOLS[0], "none"]}}}))
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_zone(page)
        row = page.locator(f"#zone-panel-row-{WOOLS[0]}")
        toggle = row.locator("summary")
        page.locator("#zone-panel h2").hover()
        expect(toggle).to_have_css("pointer-events", "none")
        row.hover()
        expect(toggle).to_have_css("pointer-events", "auto")
        page.locator("#zone-panel h2").hover()
        row.get_by_role("checkbox").focus()
        page.keyboard.press("Tab")
        expect(toggle).to_be_focused()
        expect(toggle).to_have_css("clip-path", "none")
        page.keyboard.press("Enter")
        expect(row.locator(".zv-remove")).to_be_visible()
