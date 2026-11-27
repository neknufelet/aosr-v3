"""就地切區。房間、材料與擺位使用考卷自編的假方案。"""
from __future__ import annotations

from pathlib import Path

import pytest
from playwright.sync_api import Browser, Page, expect

from tests.engine._gui_cache import gui_startup_identity_memo as gui_startup_identity_memo
from tests.engine.test_gui_browser import _open, _serve, browser as browser
from tests.engine.test_gui_zone_cut_link_browser import _cut_front_inside_auto2, _said
from tests.engine.test_gui_zone_elevation_browser import _cut_by, _drag, _ready_pixel
from tests.engine.test_gui_zone_mirror_draw_browser import END, FIRST, H, SECOND, START, W, _drawn, _matches_mirror, _room
from tests.engine.test_gui_zone_overview_tasks_browser import _SELECTED, _click_zone, _open_overview, _prepare, _settled


def _enter(page: Page) -> None:
    url = page.url
    page.locator("#zone-cut-yL").click()
    assert page.url == url
    expect(page.locator("#zone-page")).to_be_visible()
    expect(page.locator("#zone-map")).to_be_hidden()
    expect(page.locator("#zone-page #elevation-svg")).to_have_attribute("data-wall", "yL")


def test_finished_cut_exposes_all_new_zones_for_setting(tmp_path: Path, browser: Browser) -> None:
    with _room(tmp_path, browser) as page:
        page.locator("#zone-page-open").click()
        _settled(page)
        front = page.locator("#zone-map [data-wall=yL][data-zone]")
        expect(front).to_have_count(1)
        _enter(page)
        _drag(page, W, H, START, END)
        _settled(page)
        # 完成那一下就要看見伺服器補滿後的整面，不靠下一次判決或另開分區頁。
        keys = page.locator("#elevation .zone").evaluate_all("zones => zones.map(z => z.dataset.zoneId).sort()")
        assert "z1" in keys and "auto1" in keys
        page.get_by_role("button", name="切好了").click()
        assert front.evaluate_all("zones => zones.map(z => z.dataset.zone).sort()") == keys
        _click_zone(page, "yL/z1")
        page.locator("#zone-panel-mode-search").check()
        page.locator("#zone-panel-apply").click()
        _settled(page)
        expect(page.locator("#zone-map [data-wall=yL][data-zone=z1]")).to_have_attribute("data-shown", "search")


@pytest.mark.parametrize("on", [True, False])
def test_mirror_switch_controls_the_other_box(tmp_path: Path, browser: Browser, on: bool) -> None:
    with _room(tmp_path, browser) as page:
        page.locator("#zone-page-open").click()
        _enter(page)
        toggle = page.locator("#elevation-symmetry-on" if on else "#elevation-symmetry-off")
        toggle.click()
        expect(toggle).to_have_attribute("aria-pressed", "true")
        _drag(page, W, H, START, END)
        boxes = _drawn(page)
        assert set(boxes) == ({"z1", "z2"} if on else {"z1"})
        if on:
            _matches_mirror(page, "yL", boxes["z1"], boxes["z2"])
        page.get_by_role("button", name="切好了").click()
        expect(page.locator("#zone-map [data-wall=yL][data-zone=z2]")).to_have_count(int(on))


def test_move_resize_and_clear_use_the_same_editor(tmp_path: Path, browser: Browser) -> None:
    with _room(tmp_path, browser, (FIRST, SECOND)) as page:
        panel = page.locator("#elevation-panel").element_handle()
        page.locator("#zone-page-open").click()
        _enter(page)
        assert panel is not None and panel.evaluate("p => !!p.closest('#zone-page')")
        _drag(page, W, H, (0.9, 0.75), (1.1, 0.75))
        moved = _drawn(page)
        assert moved["z1"]["u_m"] == pytest.approx([0.75, 1.55])
        _drag(page, W, H, (1.55, 1.8), (1.8, 2.0))
        resized = _drawn(page)
        assert resized["z1"]["u_m"] == pytest.approx([0.75, 1.8])
        assert resized["z1"]["v_m"] == pytest.approx([0, 2])
        _matches_mirror(page, "yL", resized["z1"], resized["z2"])
        _cut_by(page, "#elevation-clear")
        assert not _drawn(page)
        page.get_by_role("button", name="切好了").click()
        expect(page.locator("#zone-map [data-wall=yL][data-zone=whole_wall]")).to_be_visible()
        page.locator("#zone-page-close").click()
        assert panel.evaluate("p => !!p.closest('#wall-section')")
        expect(page.locator("#elevation-frame")).to_be_visible()


def test_escape_during_drag_discards_it_and_returns(tmp_path: Path, browser: Browser) -> None:
    with _room(tmp_path, browser) as page:
        page.locator("#zone-page-open").click()
        _enter(page)
        pixel = _ready_pixel(page, W, H)
        requests: list[str] = []
        page.on("request", lambda request: requests.append(request.url))
        page.mouse.move(*pixel(START))
        page.mouse.down()
        page.mouse.move(*pixel(END), steps=6)
        assert page.evaluate("elevationDrawing()")
        page.keyboard.press("Escape")
        page.mouse.up()
        expect(page.locator("#zone-map")).to_be_visible()
        _settled(page)
        assert not any(url.endswith("/api/zone-cut") for url in requests)
        assert not _drawn(page)
        _enter(page)
        page.keyboard.press("Escape")
        expect(page.locator("#zone-cut-yL")).to_be_focused()


def test_selection_is_reconciled_only_when_finished(tmp_path: Path, browser: Browser) -> None:
    _prepare(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        _settled(page)
        _click_zone(page, "yL/auto2")
        _click_zone(page, "yL/z1")
        panel = page.locator("#zone-panel")
        page.locator("#zone-panel-mode-search").check()
        _enter(page)
        _cut_front_inside_auto2(page)
        _settled(page)
        expect(panel).to_be_visible()
        expect(panel).to_have_attribute("data-zones", "yL/auto2 yL/z1")
        assert _said(page) == ""
        page.keyboard.press("Escape")
        assert page.evaluate(_SELECTED) == ["yL/z1"]
        expect(page.locator("#zone-panel-mode-search")).to_be_checked()
        assert _said(page) == "剛才選的區有 1 區那一面重切過，已經不是原來那一塊，不再算選著，要的話重新選"


def test_all_faces_fit_and_done_can_be_hit(tmp_path: Path, browser: Browser) -> None:
    with _room(tmp_path, browser) as page:
        page.set_viewport_size({"width": 1440, "height": 900})
        page.locator("#zone-page-open").click()
        for face in ("yL", "y0", "x0", "xL", "floor", "ceiling"):
            page.locator(f"#zone-cut-{face}").click()
            svg = page.locator("#zone-page #elevation-svg")
            expect(svg).to_have_attribute("data-wall", face)
            expect(svg).to_be_in_viewport(ratio=1)
            done = page.get_by_role("button", name="切好了")
            assert done.evaluate("""button => { const r = button.getBoundingClientRect();
                return button === document.elementFromPoint(r.x + r.width/2, r.y + r.height/2); }""")
            done.click()
            expect(page.locator(f"#zone-cut-{face}")).to_be_focused()
