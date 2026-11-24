"""#1266、#1267：假房間改尺寸後比對落地檔，從搜尋與結果頁實際點回同一份。"""
import json
from copy import deepcopy
from pathlib import Path

import pytest
from playwright.sync_api import Browser, Page, Route, expect

from aosr.reporting.result import SchemeResult
from aosr.reporting.scheme import Scheme
from tests.engine._input_page_steps import open_fold
from tests.engine._gui_cache import gui_startup_identity_memo as gui_startup_identity_memo
from tests.engine._zone_naming_room import ZONE_NAMING_SCHEME
from tests.engine.test_gui_browser import RUN_ID, _open, _save, _serve, browser as browser, result as result
from tests.engine.test_gui_gradient_browser import _serve as _serve_gradient
from tests.engine.test_gui_gradient_jobs import FakeLauncher
from tests.engine.test_gui_zone_elevation_browser import _open_scheme, _write, _zones
from tests.engine.test_gui_zone_overview_browser import SAMPLE, SAMPLE_DESIGN
from tests.engine.test_gui_zone_overview_tasks_browser import _open_overview, _send_to_search


def _files(tmp_path: Path) -> tuple[Path, Path]:
    _write(tmp_path, ZONE_NAMING_SCHEME)
    design = tmp_path / "designs" / f"{SAMPLE}.json"
    design.parent.mkdir()
    design.write_text(json.dumps(SAMPLE_DESIGN), encoding="utf-8")
    return tmp_path / "schemes" / design.name, design


@pytest.mark.parametrize(("axis", "value"), [("Lz", 3.2), ("Ly", 5.8)])
def test_resize_saves_each_original_zone(tmp_path: Path, browser: Browser, axis: str, value: float) -> None:
    path, design = _files(tmp_path)
    before = json.loads(path.read_text())
    settings = json.loads(design.read_text())
    expected = deepcopy(before["scene"]["surface_zones_by_wall"])
    for wall, zones in expected.items():
        axes = ("Lx", "Ly") if wall in ("floor", "ceiling") else ("Ly" if wall.startswith("x") else "Lx", "Lz")
        for zone in zones:
            if zone["area"] == "rectangle" and axis in axes:
                ends = zone[("u_m", "v_m")[axes.index(axis)]]
                if ends[1] == before["scene"]["room_m"][axis]:
                    ends[1] = value
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, SAMPLE)
        page.locator("#zone-table-fold > summary").click()
        page.locator("#room-" + axis).fill(str(value))
        page.locator("#room-" + axis).press("Tab")
        page.wait_for_function("!pendingRoomResize()")
        assert json.loads(path.read_text()) == before  # 調整只動網頁上的草稿：沒按儲存，檔案一個字都不變
        page.locator("#save").click()
        expect(page.locator("#messages")).to_have_text("方案已儲存")
        page.wait_for_function("!inputBusy")
        saved = json.loads(path.read_text())
        Scheme.model_validate(saved)  # 方案驗證逐格核六面無重疊、無空隙
        assert saved["scene"]["room_m"][axis] == value
        assert saved["scene"]["surface_zones_by_wall"] == expected
        assert json.loads(design.read_text())["zones"] == settings["zones"]
        if axis == "Lz":
            # #1265 單元 3 起分區的牆每一區列在第 2 段右欄「每一區的位置與大小」（選中那一面才列；寫寬 × 高，不寫 u／v）：
            # 先點前牆再看第一區（z1，原本落地到頂）的高跟著變成 3.2。
            page.locator("#elevation-wall-yL").click()
            open_fold(page, "zone-numbers-fold")  # 摺疊預設收著：點開、看得見才算（審查：收著時 to_contain_text 也讀得到字）
            expect(page.locator("#zoned-list-yL #zone-size-yL-z1")).to_be_visible()
            expect(page.locator("#zoned-list-yL #zone-size-yL-z1")).to_contain_text("× 3.2")
            page.locator("#elevation-panel").screenshot(path=tmp_path / "front-saved.png")


def test_shrink_keeps_blocked_wall_and_names_its_zone(tmp_path: Path, browser: Browser) -> None:
    path, _design = _files(tmp_path)
    before = path.read_bytes()
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, SAMPLE)
        page.locator("#zone-table-fold > summary").click()
        page.locator("#elevation-wall-xL").click()
        right = _zones(page, "xL")
        page.locator("#room-Ly").fill("5.25")  # 右牆窗旁 auto4 剩 0.15 公尺；地毯放得下
        page.locator("#room-Ly").press("Tab")
        size = page.locator("#elevation-size")
        expect(size).to_contain_text("0.2 公尺")
        label = page.evaluate("zoneLabels('xL').get('auto4')")
        for word in ("右牆", label["numberText"], label["name"], "先"):
            expect(size).to_contain_text(word)
        assert _zones(page, "xL") == right
        assert next(z for z in _zones(page, "floor") if z["id"] == "rug")["v_m"] == [3.6, 5.25]
        page.locator("#save").click()
        expect(page.locator("#messages")).to_contain_text("超出牆面範圍")
        expect(size).to_be_visible()
        assert path.read_bytes() == before
        size.screenshot(path=tmp_path / "right-blocked.png")
        page.locator("#room-Ly").fill("5.8")
        page.locator("#save").click()
        expect(page.locator("#messages")).to_have_text("方案已儲存")
        expect(size).not_to_contain_text("0.2 公尺")


def test_save_locks_the_form_while_resizing(tmp_path: Path, browser: Browser) -> None:
    path, _design = _files(tmp_path)
    before = json.loads(path.read_text())
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, SAMPLE)
        held: list[Route] = []
        page.route("**/api/zone-resize", lambda route: held.append(route))
        page.locator("#room-Lz").fill("3.2")
        page.locator("#save").click()
        for _ in range(200):  # 攔到的那一筆要等瀏覽器送出才進得了清單：最多等 10 秒
            if held:
                break
            page.wait_for_timeout(50)
        assert held
        try:
            assert page.locator("main").evaluate("node => node.inert")
        finally:
            held[0].continue_()
        expect(page.locator("#messages")).to_have_text("方案已儲存")
        assert json.loads(path.read_text())["speakers"] == before["speakers"]


def _returned(page: Page, name: str = SAMPLE) -> None:
    page.get_by_role("link", name="回方案輸入頁").click()
    expect(page.locator("#save-id")).to_have_value(name)
    for axis, value in (("Lx", "5.2"), ("Ly", "5.5"), ("Lz", "3.15")):
        expect(page.locator("#room-" + axis)).to_have_value(value)
    expect(page.locator("#zone-page")).to_be_hidden()
    expect(page.locator("#scheme-menu-button")).to_be_in_viewport()


def test_return_from_gradient_opens_saved_input(tmp_path: Path, browser: Browser) -> None:
    _files(tmp_path)
    with _serve_gradient(tmp_path, FakeLauncher()) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        _send_to_search(page)
        expect(page.locator("#form-block")).to_be_visible()
        _returned(page)
        page.screenshot(path=tmp_path / "returned-input.png")


def test_return_from_result_opens_saved_input(tmp_path: Path, browser: Browser, result: SchemeResult) -> None:
    name = result.scheme.scheme_id
    _write(tmp_path, ZONE_NAMING_SCHEME | {"scheme_id": name})
    _save(tmp_path, result)
    with _serve(tmp_path) as base, _open(browser, f"{base}/results/{RUN_ID}") as watched:
        expect(watched.page.locator("#page-title")).to_contain_text(name)
        _returned(watched.page, name)


def test_missing_scheme_link_explains_at_open_button(tmp_path: Path, browser: Browser) -> None:
    _files(tmp_path)
    with _serve(tmp_path) as base, _open(browser, f"{base}/?scheme=missing-room") as watched:
        page = watched.page
        expect(page.locator("#open-note")).to_contain_text("沒有打開「missing-room」：已存的方案裡沒有這一份")
        expect(page.locator("#save-id")).not_to_have_value("missing-room")
        expect(page.locator("#open-note")).to_be_in_viewport()
