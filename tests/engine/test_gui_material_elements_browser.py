"""只讀元件分頁，從實際端點核對畫面、鍵盤與網路請求。"""
from pathlib import Path
from urllib.parse import urlsplit

from playwright.sync_api import Browser, expect

from aosr.gui.material_names import family_display_name
from aosr.materials.library import builtin_entries
from tests.engine.test_gui_browser import _open, _serve, browser as browser


def test_element_tab_navigation_readonly_rows_and_phone_width(tmp_path: Path, browser: Browser) -> None:
    with _serve(tmp_path) as base, _open(browser, "about:blank", viewport_width=390) as watched:
        page = watched.page
        page.route("**/api/material-curve*", lambda route: route.fulfill(json={
            "status": "drawn", "frequencies_hz": [80, 160], "absorption": [.2, .4], "warnings": [], "spans": []}))
        page.goto(f"{base}/materials", wait_until="networkidle")
        expect(page.get_by_role("tab")).to_have_count(3)
        tabs = page.get_by_role("tab")
        # 先讓目前可見的列真的畫完，避免把上一頁已排出的請求誤算在切頁上。
        unfinished = page.locator('[role="tabpanel"]:visible .material-item:visible .preview:not(:has(svg))')
        expect(unfinished).to_have_count(0)
        tabs.first.focus()
        requests: list[str] = []
        page.on("request", lambda request: requests.append(urlsplit(request.url).path))
        page.keyboard.press("ArrowLeft")
        expect(tabs.last).to_be_focused()
        expect(tabs.last).to_have_attribute("aria-selected", "true")
        page.wait_for_load_state("networkidle")
        assert "/api/material-curve" not in requests
        for key, index in (("ArrowRight", 0), ("End", 2), ("Home", 0), ("ArrowRight", 1)):
            page.keyboard.press(key)
            expect(tabs.nth(index)).to_be_focused()
            expect(tabs.nth(index)).to_have_attribute("aria-selected", "true")
        expect(unfinished).to_have_count(0)
        requests.clear()
        page.keyboard.press("ArrowRight")
        panel = page.get_by_role("tabpanel")
        rows = page.request.get(f"{base}/api/material-elements").json()["elements"]
        expect(tabs.last).to_have_text(f"元件 {len(rows)} 種")
        expect(panel.get_by_role("article")).to_have_count(len(rows))
        with_templates = sum(bool(row["templates"]) for row in rows)
        # #1351 新外框：頂列本身就是 <header>，說明句也搬進左欄收起來；直接看填那一句的元素。
        expect(page.locator("#element-description")).to_contain_text(f"有內建樣板的 {with_templates} 種，另列內建沒有的種類")
        family_names = {e.id: family_display_name(e.label) for e in builtin_entries()}
        for article, row in zip(panel.get_by_role("article").all(), rows, strict=True):
            expect(article).to_contain_text(row["name"])
            if row["templates"]:
                expect(article).to_contain_text("用這組物性的內建樣板：" + "、".join(row["templates"]))
            else:
                expect(article).to_contain_text("可填的參數：" + row["parameters"][0])
            for family in row["families"]:
                expect(article).to_contain_text(f"搜尋家族：{family_names[family]}（{family}）")
        expect(panel.locator("button,input,select,svg.curve")).to_have_count(0)
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        assert panel.evaluate("p => p.scrollWidth <= p.clientWidth")
        assert "/api/material-curve" not in requests
        assert watched.page_errors == [] and watched.console_errors == []


def test_return_to_editor_after_resizing_on_element_tab(tmp_path: Path, browser: Browser) -> None:
    with _serve(tmp_path) as base, _open(browser, "about:blank", viewport_width=390) as watched:
        page = watched.page
        page.route("**/api/material-curve*", lambda route: route.fulfill(json={
            "status": "drawn", "frequencies_hz": [80, 160], "absorption": [.2, .4], "warnings": [], "spans": []}))
        page.goto(f"{base}/materials")
        page.locator("#panel-fixed button.open").first.click()
        expect(page.locator("#curve-chart canvas")).to_be_visible()
        page.locator("#tab-elements").click()
        expect(page.locator("#editor")).to_be_hidden()
        page.set_viewport_size({"width": 1440, "height": 900})
        page.locator("#tab-fixed").click()
        expect(page.locator("#curve-chart canvas")).to_be_visible()
        assert page.locator("#curve-chart").evaluate("chart => chart.clientWidth === chart.firstElementChild.offsetWidth")
        assert watched.console_errors == [] and watched.page_errors == []


def test_element_catalog_loads_when_entry_listing_fails(tmp_path: Path, browser: Browser) -> None:
    with _serve(tmp_path) as base, _open(browser, "about:blank") as watched:
        page = watched.page
        page.route("**/api/materials", lambda route: route.fulfill(status=503, json={"error": "自編讀取錯誤"}))
        page.goto(f"{base}/materials")
        page.locator("#tab-elements").click()
        expect(page.get_by_role("article").first).to_be_visible()
