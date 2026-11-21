"""從材料庫實際進設計器、存入暫存庫、重新開啟並到分區選材。"""
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import cast

from playwright.sync_api import Browser, Locator, Page, Response, Route, expect

from tests.engine._construction_gui_server import real_server
from tests.engine.test_gui_browser import _open, browser as browser
from tests.engine.test_gui_zone_overview_tasks_browser import _click_zone, _open_overview, _prepare


@contextmanager
def _designer(folder: Path, browser: Browser, width: int = 1440) -> Iterator[tuple[Page, str]]:
    with real_server(folder) as base, _open(browser, f"{base}/materials", viewport_width=width) as watched:
        page = watched.page
        page.locator("#tab-optimizable").click()
        new = page.get_by_role("button", name="新增構造", exact=True)
        expect(new).to_be_visible()
        new.click()
        expect(page.get_by_label("名稱", exact=True)).to_be_editable()
        yield page, base
        assert (watched.page_errors, watched.console_errors) == ([], [])


def _add(page: Page, *kinds: str) -> None:
    for kind in kinds:
        page.locator(f'#palette button[data-kind="{kind}"]').click()


def _save(page: Page, tab: str) -> None:
    page.get_by_role("button", name="儲存", exact=True).click()
    expect(page.locator("#save-message")).to_contain_text(f"已存到「{tab}」頁")


def _list_row(page: Page, base: str, tab: str, name: str) -> Locator:
    page.goto(f"{base}/materials")
    page.locator(f"#tab-{tab}").click()
    return page.get_by_role("tabpanel").get_by_role("article").filter(has=page.get_by_text(name, exact=True))


def _fields(page: Page, index: int) -> dict[str, str]:
    row = page.locator("#stack article").nth(index)
    head = row.locator("button.layer-select")
    if head.get_attribute("aria-expanded") == "false":
        head.click()
    return cast(dict[str, str], row.locator("[name]").evaluate_all("nodes => Object.fromEntries(nodes.map(n => [n.name, n.value]))"))


def test_build_move_remove_save_and_reopen(tmp_path: Path, browser: Browser) -> None:
    with _designer(tmp_path, browser) as (page, base):
        name = "自編疊層甲"
        page.get_by_label("名稱", exact=True).fill(name)
        _add(page, "fabric", "porous", "air")
        _fields(page, 2)
        page.get_by_role("button", name="往房間側移").click()
        _fields(page, 0)
        page.get_by_role("button", name="拿掉").click()
        page.locator('input[name="backing"][value="air"]').check()
        page.get_by_role("radio", name="牆", exact=True).check()
        values = [_fields(page, i) for i in range(2)]
        page.locator('input[name="thickness_m"]').fill("63.7")
        values[1] = _fields(page, 1)
        _save(page, "參數構造")
        row = _list_row(page, base, "optimizable", name)
        row.get_by_role("link", name=name, exact=True).click()
        expect(page.get_by_text("改這一筆，已經貼在分區上的不會跟著變")).to_be_visible()
        expect(page.get_by_label("名稱", exact=True)).to_have_value(name)
        expect(page.locator('input[name="backing"]:checked')).to_have_value("air")
        expect(page.get_by_role("radio", name="牆", exact=True)).to_be_checked()
        assert page.locator("#stack article").evaluate_all("rows => rows.map(r => r.dataset.kind)") == ["air", "porous"]
        assert [_fields(page, i) for i in range(2)] == values
        entries = page.request.get(f"{base}/api/materials").json()["materials"]
        stored, = [r["entry"] for r in entries if r["entry"]["label"] == name]
        assert stored["layers"][1]["thickness_m"] == .0637
        assert stored["kind"] == "construction" and stored["placement"] == "wall"


def _curve_for_thickness(response: Response, meters: float) -> bool:
    if not response.url.endswith("/api/material-curve"):
        return False
    document = response.request.post_data_json
    return isinstance(document, dict) and bool(document.get("layers")) and document["layers"][0].get("thickness_m") == meters


def test_change_thickness_updates_cutaway_curve_and_total(tmp_path: Path, browser: Browser) -> None:
    with _designer(tmp_path, browser) as (page, _):
        _add(page, "porous", "air")
        _fields(page, 0)
        thickness = page.locator('input[name="thickness_m"]')
        with page.expect_response(lambda r: _curve_for_thickness(r, .0371)) as before:
            thickness.fill("37.1")
        assert before.value.json()["status"] == "drawn"
        curve = page.locator("#curve polyline")
        expect(curve).to_be_visible()
        points = curve.get_attribute("points")
        slab = page.locator('#figure g[data-layer="0"] rect')
        width = float(slab.get_attribute("width") or "0")
        total = page.locator("#total").inner_text()
        with page.expect_response(lambda r: _curve_for_thickness(r, .0837)) as after:
            thickness.fill("83.7")
        assert after.value.json()["status"] == "drawn"
        assert points
        expect(curve).not_to_have_attribute("points", points)
        assert float(slab.get_attribute("width") or "0") > width
        expect(page.locator("#total")).not_to_have_text(total)
        page.locator('#figure g[data-layer="1"] rect').click()
        expect(page.locator("#stack article").nth(1).locator(".layer-select")).to_have_attribute("aria-expanded", "true")
        held: list[Route] = []
        page.route("**/api/material-curve", lambda route: held.append(route), times=1)
        with page.expect_request("**/api/material-curve"):
            thickness.fill("71.2")
        with page.expect_response(lambda r: r.url.endswith("/api/material-curve")) as invalid:
            thickness.fill("-13")
        expect(page.locator("#curve")).to_have_text(invalid.value.json()["message"])
        with page.expect_event("requestfinished", predicate=lambda r: r == held[0].request):
            held[0].fulfill(json=after.value.json())
        page.evaluate("new Promise(done => requestAnimationFrame(() => requestAnimationFrame(done)))")
        expect(page.locator("#curve")).to_have_text(invalid.value.json()["message"])
        expect(curve).to_have_count(0)


def test_product_copy_is_a_separate_editable_construction(tmp_path: Path, browser: Browser) -> None:
    with _designer(tmp_path, browser) as (page, base):
        name = "自編完成面"
        page.get_by_label("名稱", exact=True).fill(name)
        _add(page, "fabric")
        page.get_by_label("這是買來的或照抄的，不拆").check()
        _save(page, "成品")
        row = _list_row(page, base, "fixed", name)
        row.get_by_role("button", name="打開").click()
        expect(page.get_by_label("名稱", exact=True)).to_be_disabled()
        expect(page.get_by_role("button", name="儲存")).to_be_hidden()
        entries = f"{base}/api/materials"
        original, = [r["entry"] for r in page.request.get(entries).json()["materials"] if r["entry"]["label"] == name]
        page.get_by_role("button", name="複製到設計器").click()
        expect(page.get_by_label("名稱", exact=True)).to_have_value(name + "（複製）")
        expect(page.get_by_label("這是買來的或照抄的，不拆")).not_to_be_checked()
        _add(page, "air")
        _save(page, "參數構造")
        saved = {r["entry"]["label"]: r["entry"] for r in page.request.get(entries).json()["materials"]}
        assert original["product"] and saved[name] == original
        copy = saved[name + "（複製）"]
        assert copy["id"] != original["id"] and not copy["product"] and len(copy["layers"]) == 2
        expect(_list_row(page, base, "optimizable", copy["label"])).to_be_visible()


def test_saved_rigid_construction_is_selectable_in_zone(tmp_path: Path, browser: Browser) -> None:
    _prepare(tmp_path)
    with _designer(tmp_path, browser) as (page, base):
        name = "自編剛背牆材"
        page.get_by_label("名稱", exact=True).fill(name)
        _add(page, "porous")
        _save(page, "參數構造")
        page.goto(base)
        _open_overview(page)
        _click_zone(page, "yL/z3")
        page.locator("#zone-panel-mode-fixed").check()
        select = page.locator("#zone-panel-material")
        option = select.locator("option").filter(has_text=name)
        value = option.get_attribute("value")
        assert value
        select.select_option(value)
        expect(select).to_have_value(value)


def test_phone_stack_fields_figure_and_save_fit(tmp_path: Path, browser: Browser) -> None:
    with _designer(tmp_path, browser, 390) as (page, _):
        _add(page, "fabric", "porous", "air")
        _fields(page, 1)
        switches = page.get_by_role("switch")
        expect(switches).to_have_count(3)
        for switch in switches.all():
            expect(switch).to_be_disabled()
            expect(switch).not_to_be_checked()
        expect(page.get_by_text("第二批才能給程式調", exact=True)).to_have_count(3)
        expect(page.locator("#curve polyline")).to_be_visible(timeout=30000)
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        assert page.locator("section").evaluate_all("nodes => nodes.every(n => n.scrollWidth <= n.clientWidth)")
        _save(page, "參數構造")
