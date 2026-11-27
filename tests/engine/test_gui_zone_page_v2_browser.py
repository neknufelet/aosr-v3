"""在自編房間，以實際視窗命中測試與存檔驗分區頁。"""
from pathlib import Path
from typing import cast

import pytest
from playwright.sync_api import Browser, Page, Route, expect

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
            reason = original.locator("xpath=ancestor::label").locator("small")
            expect(reason).to_have_text("由程式最佳化還不收")
            expect(reason).to_be_in_viewport()
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


def test_unpasted_zone_requires_a_real_fixed_material(tmp_path: Path, browser: Browser) -> None:
    _prepare(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        _click_zone(page, "yL/auto1")
        page.locator("#zone-panel-mode-fixed").check()
        rows = page.locator("#zone-panel-material")
        expect(rows.get_by_text("選要貼哪一種")).to_have_count(0)
        expect(rows.locator("[aria-checked=true]")).to_have_count(0)
        expect(page.locator("#zone-panel-apply")).to_be_disabled()
        rows.get_by_role("radio").first.click()
        expect(rows.locator("[aria-checked=true]")).to_have_count(1)
        expect(page.locator("#zone-panel-apply")).to_be_enabled()


def test_segment_shows_only_its_label_and_remains_keyboard_operable(tmp_path: Path, browser: Browser) -> None:
    _prepare(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        _click_zone(page, "yL/auto1")
        option = page.locator("#zone-panel-option-keep_original")
        expect(option).to_have_text("保留原樣")
        expect(option.locator(".zv-sw")).to_have_count(0)
        radio = option.get_by_role("radio")
        expect(radio).to_have_css("width", "1px")
        radio.focus()
        expect(radio).to_be_focused()
        radio.press("ArrowRight")
        fixed = page.locator("#zone-panel-mode-fixed")
        expect(fixed).to_be_checked()
        expect(fixed).to_be_focused()
        page.keyboard.press("ArrowRight")
        expect(page.locator("#zone-panel-mode-search")).to_be_focused()
        page.keyboard.press("ArrowLeft")
        expect(fixed).to_be_focused()


def test_fixed_material_arrows_select_and_wrap(tmp_path: Path, browser: Browser) -> None:
    _prepare(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_zone(page)
        page.locator("#zone-panel-mode-fixed").check()
        rows = page.get_by_role("radiogroup", name="指定貼哪一種").get_by_role("radio")
        rows.first.focus()
        for key, index in (("ArrowDown", 1), ("ArrowLeft", 0), ("ArrowUp", -1), ("ArrowRight", 0)):
            page.keyboard.press(key)
            expect(rows.nth(index)).to_be_focused()
            expect(rows.nth(index)).to_be_checked()


def test_fixed_form_uses_short_label_and_depth_placeholder(tmp_path: Path, browser: Browser) -> None:
    _prepare(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_zone(page)
        page.locator("#zone-panel-mode-fixed").check()
        expect(page.locator(".zv-show-all")).to_have_text("顯示全部材料")
        expect(page.locator("#zone-panel-depth")).to_have_attribute("placeholder", "空白＝不限")
        expect(page.locator("#zone-panel-deepest")).not_to_contain_text("空白＝不限")


def test_original_method_is_beside_the_zone_heading(tmp_path: Path, browser: Browser) -> None:
    _prepare(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        _click_zone(page, "yL/auto1")
        page.locator("#zone-panel-mode-fixed").check()
        expect(page.locator(".zv-ehead #zone-panel-now-keep_original")).to_be_visible()
        expect(page.locator(".zv-panel-body .zv-now")).to_have_count(0)


@pytest.mark.parametrize("close_with_key", [False, True])
def test_drawer_asks_before_discarding_layers(tmp_path: Path, browser: Browser, close_with_key: bool) -> None:
    _prepare(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_zone(page)
        page.locator("#zone-panel-mode-fixed").check()
        page.locator("#zone-panel-build").click()
        frame = page.frame_locator("#zone-builder iframe")
        frame.locator('#palette [data-kind="porous"]').click()
        for choice in ("繼續組", "放棄"):
            if close_with_key:
                page.keyboard.press("Escape")
            else:
                page.locator("#zone-builder-close").click()
            expect(page.get_by_text("放棄這個新構造？", exact=True)).to_be_visible()
            page.get_by_role("button", name=choice, exact=True).click()
            if choice == "繼續組":
                expect(frame.locator("article[data-kind=porous]")).to_be_visible()
        expect(page.locator("#zone-builder")).to_have_count(0)


@pytest.mark.parametrize(("mode", "failures"), [("fixed", 1), ("search", 2)])
def test_saved_material_recovers_after_library_failure(tmp_path: Path, browser: Browser, mode: str, failures: int) -> None:
    _prepare(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_zone(page)
        page.locator(f"#zone-panel-mode-{mode}").check()
        page.locator("#zone-panel-build").click()
        frame = page.frame_locator("#zone-builder iframe")
        frame.locator("#name").fill("自編補讀構造")
        frame.locator('#palette [data-kind="porous"]').click()
        reads = []

        def fail_library(route: Route) -> None:
            if route.request.method == "GET" and route.request.frame == page.main_frame:
                reads.append(route.request.url)
                if len(reads) <= failures:
                    route.fulfill(status=503, json={"error": "考卷扣住材料庫"})
                    return
            route.continue_()

        page.route("**/api/materials", fail_library)
        frame.locator("#save").click()
        expect(page.locator("#zone-builder [role=status]")).to_contain_text("已存入材料庫")
        page.locator("#zone-builder-close").click()
        expect(page.locator("#zone-builder")).to_have_count(0)
        if failures == 2:
            notice = page.locator("#zone-panel").get_by_role("status").filter(has_text="已存入材料庫")
            expect(notice).to_be_visible()
            expect(notice).to_contain_text("考卷扣住材料庫")
            page.locator("#zone-panel").get_by_role("button", name="重試").click()
        row = page.locator("#zone-panel .zv-material").filter(has_text="自編補讀構造")
        expect(row).to_have_attribute("aria-checked", "true")
        assert len(reads) == failures + 1


@pytest.mark.parametrize("mode", ["fixed", "search"])
def test_drawer_reread_preserves_the_later_pick(tmp_path: Path, browser: Browser, mode: str) -> None:
    _prepare(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_zone(page)
        page.locator(f"#zone-panel-mode-{mode}").check()
        page.locator("#zone-panel-build").click()
        frame = page.frame_locator("#zone-builder iframe")
        frame.locator("#name").fill("自編延遲回覆板")
        frame.locator('#palette [data-kind="porous"]').click()
        page.route("**/api/materials", lambda r: r.fulfill(status=503, json={"error": "考卷暫停讀取"})
                   if r.request.frame == page.main_frame else r.continue_())
        frame.locator("#save").click()
        expect(page.locator("#zone-builder [role=status]")).to_contain_text("已存入材料庫")
        page.unroute("**/api/materials")
        held: list[Route] = []
        page.route("**/api/materials", lambda r: held.append(r))
        with page.expect_request("**/api/materials"):
            page.locator("#zone-builder-close").click()
        prefix = "material" if mode == "fixed" else "pick"
        chosen = page.locator(f"#zone-panel-{prefix}-{PRODUCT}")
        chosen.click()
        with page.expect_response("**/api/materials"):
            held[0].continue_()
        new_row = page.get_by_role("radio" if mode == "fixed" else "checkbox", name="自編延遲回覆板")
        expect(new_row).to_have_attribute("aria-checked", "false")
        expect(chosen).to_have_attribute("aria-checked", "true")


def test_drawer_rebuilds_default_candidates_after_initial_library_failure(tmp_path: Path, browser: Browser) -> None:
    _prepare(tmp_path, _design({"yL": {"z1": {"mode": "search"}}}))
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        page.route("**/api/materials", lambda r: r.fulfill(status=503, json={"error": "考卷初讀失敗"}))
        _open_zone(page)
        expect(page.locator("#zone-panel-search-from")).to_contain_text("材料庫沒讀全")
        page.unroute("**/api/materials")
        page.locator("#zone-panel-build").click()
        frame = page.frame_locator("#zone-builder iframe")
        frame.locator("#name").fill("自編初讀恢復棉")
        frame.locator('#palette [data-kind="porous"]').click()
        frame.locator("#save").click()
        expect(page.locator("#zone-builder")).to_have_count(0)
        expect(page.get_by_role("checkbox", name="自編初讀恢復棉", exact=False)).to_be_checked()
        for choice in WOOLS:
            expect(page.locator(f"#zone-panel-pick-{choice}")).to_be_checked()


@pytest.mark.parametrize("field", ["name", "backing"])
def test_escape_preserves_drawer_edits_before_first_layer(tmp_path: Path, browser: Browser, field: str) -> None:
    _prepare(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_zone(page)
        page.locator("#zone-panel-mode-fixed").check()
        page.locator("#zone-panel-build").click()
        frame = page.frame_locator("#zone-builder iframe")
        if field == "name":
            frame.locator("#name").fill("自編未加層")
        else:
            frame.locator('[name="backing"][value="rigid"]').check()
        frame.locator("#name").press("Escape")
        expect(page.locator("#zone-builder-resume")).to_be_visible()
        page.locator("#zone-builder-resume").click()
        if field == "name":
            expect(frame.locator("#name")).to_have_value("自編未加層")
        else:
            expect(frame.locator('[name="backing"][value="rigid"]')).to_be_checked()


def test_composing_escape_in_drawer_leaves_the_editor_open(tmp_path: Path, browser: Browser) -> None:
    _prepare(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_zone(page)
        page.locator("#zone-panel-mode-fixed").check()
        page.locator("#zone-panel-build").click()
        name = page.frame_locator("#zone-builder iframe").locator("#name")
        name.fill("組字中")
        name.dispatch_event("keydown", {"key": "Escape", "isComposing": True})
        expect(name).to_be_visible()
        expect(page.locator("#zone-builder-resume")).to_be_hidden()
