"""真瀏覽器考「這間房貼什麼」每一面的「切區 ▸」（#1262）：畫框在「房間方案輸入」的立面圖，這一頁只選區、選貼什麼；
按「切區 ▸」回輸入頁、打開那一面的立面圖。切過之後回來，選到的區換了一塊長方形就不再算選著（自動補滿的區會重新編號，
同一個代號可能指到另一塊）。牆窄的時候，牆名那一行的字讓出右邊兩顆鈕，不疊在一起。

房間用區名考卷那間假房間（``_zone_naming_room``，數字是考卷自己編的）；窄房間是同一份改的，尺寸也是這裡自己編的。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from playwright.sync_api import Browser, Page, expect

from tests.engine._gui_cache import gui_startup_identity_memo as gui_startup_identity_memo
from tests.engine._zone_naming_room import ZONE_NAMING_SCHEME
from tests.engine.test_gui_browser import _open, _serve, browser as browser
from tests.engine.test_gui_zone_elevation_browser import _drag, _write
from tests.engine.test_gui_zone_overview_tasks_browser import _SELECTED, _click_zone, _open_overview, _prepare, _settled

FACES = ["ceiling", "floor", "x0", "xL", "y0", "yL"]
WALLS = ["x0", "xL", "y0", "yL"]
FRONT_SIZE = (5.2, 3.15)  # 假房間前牆（yL）在立面圖上的寬與高（公尺）
# 每一面的牆名那一行與那一面的鈕各占多大（視窗像素：左、右、上、下）。
_ROWS = """() => [...document.querySelectorAll('#zone-page .zv-face-tools')].map(tools => {
    const wall = tools.dataset.wall, title = document.getElementById(`zone-face-title-${wall}`);
    const box = item => { const rect = item.getBoundingClientRect(); return [rect.left, rect.right, rect.top, rect.bottom]; };
    return {wall, title: box(title), tools: box(tools), aside: title.querySelector('tspan')?.textContent ?? null,
            map: box(document.getElementById('zone-map'))}; })"""


def _shape(page: Page, zone: str) -> list[float] | None:
    """那一區在分區頁圖上畫成哪一塊長方形（圖上的 x、y、寬、高）；圖上沒有這一區回 None。"""
    return cast(list[float] | None, page.evaluate("""id => { const [wall, key] = id.split('/');
        const fill = document.querySelector(`#zone-map [data-wall='${wall}'][data-zone='${key}'] .zv-fill`);
        if (!fill) return null; const box = fill.getBBox(); return [box.x, box.y, box.width, box.height]; }""", zone))


def _cut_front_inside_auto2(page: Page) -> None:
    """前牆（yL）的自動補滿格 auto2（兩塊吸音中間、上半那一格）裡拖一個框；鏡不鏡像都落在那一格裡。"""
    _drag(page, *FRONT_SIZE, (2.25, 2.25), (2.95, 2.95))


def test_every_face_has_a_cut_link_that_opens_its_elevation(tmp_path: Path, browser: Browser) -> None:
    """六面都有「切區 ▸」：按了回輸入頁、那一面的立面圖打開而且在視窗裡；切一個框再回這一頁，看得到多出來的那一區。"""
    _prepare(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        _settled(page)
        cuts = page.locator("#zone-page [id^='zone-cut-']")
        assert sorted(cuts.evaluate_all("buttons => buttons.map(button => button.id)")) == [f"zone-cut-{face}" for face in FACES]
        expect(cuts.first).to_have_text("切區 ▸")
        expect(page.locator("#zone-cut-yL")).to_have_attribute("aria-label", "前牆切區：回「房間方案輸入」在立面圖畫框")
        # 「選整面」照舊只有四面牆有（地板、天花沒有）。
        assert page.locator("#zone-page [id^='zone-whole-']").count() == 4
        # 地板：回輸入頁、地板那一面打開。
        page.locator("#zone-cut-floor").click()
        expect(page.locator("#zone-page")).to_be_hidden()
        expect(page.locator("#elevation-wall-floor")).to_have_attribute("aria-pressed", "true")
        expect(page.locator("#elevation-frame")).to_be_in_viewport()
        # 從頁面最上面那顆（方案那一列）回分區頁，換按前牆：立面圖換成前牆，捲到立面圖那一塊
        # （不是捲回打開分區頁之前的地方：那時捲在最上面，立面圖在視窗外）。
        page.evaluate("window.scrollTo(0, 0)")
        expect(page.locator("#elevation-frame")).not_to_be_in_viewport()
        page.locator("#scheme-zone-open").click()
        expect(page.locator("#zone-map")).to_be_visible()
        before = page.locator("#zone-map [data-wall='yL'][data-zone]").count()
        page.locator("#zone-cut-yL").click()
        expect(page.locator("#elevation-wall-yL")).to_have_attribute("aria-pressed", "true")
        expect(page.locator("#elevation-wall-floor")).to_have_attribute("aria-pressed", "false")
        expect(page.locator("#elevation-frame")).to_be_in_viewport()
        _cut_front_inside_auto2(page)
        page.locator("#zone-page-open").click()
        expect(page.locator("#zone-map")).to_be_visible()
        _settled(page)
        expect(page.locator("#zone-map [data-wall='yL'][data-zone='z4']")).to_have_count(1)
        assert page.locator("#zone-map [data-wall='yL'][data-zone]").count() > before


def test_a_selection_whose_rectangle_changed_after_cutting_is_dropped(tmp_path: Path, browser: Browser) -> None:
    """選了前牆 auto2、小框還開著，回輸入頁在那一格裡切一個框再回來：auto2 這個代號若換成另一塊長方形，就不再算選著，
    小框收起來（不然小框的改動會套到另一塊上）。對照：選右牆一區、切的是前牆，那一區沒動，回來照樣選著、小框還在。"""
    _prepare(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        _settled(page)
        _click_zone(page, "yL/auto2")
        expect(page.locator("#zone-panel")).to_be_visible()
        page.locator("#zone-panel-mode-keep_original").check()
        shape = _shape(page, "yL/auto2")
        # 走原本那條路（不靠新鈕）：回輸入頁、自己點前牆、切框、按「設定每一區貼什麼 ▸」回來。
        page.locator("#zone-page-close").click()
        page.locator("#elevation-wall-yL").click()
        _cut_front_inside_auto2(page)
        page.locator("#zone-page-open").click()
        expect(page.locator("#zone-map")).to_be_visible()
        _settled(page)
        after = _shape(page, "yL/auto2")
        assert after is not None and after != shape  # 這一刀切下去，auto2 這個代號確實換了一塊（不然這一題考不到）
        assert page.evaluate(_SELECTED) == []
        expect(page.locator("#zone-panel")).to_have_count(0)
        # 對照：選右牆一區，切前牆（右牆不動），回來照樣選著。
        _click_zone(page, "xL/auto3")
        expect(page.locator("#zone-panel")).to_be_visible()
        page.locator("#zone-cut-yL").click()
        _drag(page, *FRONT_SIZE, (2.35, 0.2), (2.85, 0.6))
        page.locator("#zone-page-open").click()
        expect(page.locator("#zone-map")).to_be_visible()
        _settled(page)
        assert page.evaluate(_SELECTED) == ["xL/auto3"]
        expect(page.locator("#zone-panel")).to_be_visible()


def _narrow_scheme() -> dict[str, object]:
    """假房間改一份窄的：前後牆窄、側牆長，六面各一整面（尺寸、喇叭、座位都是這裡自己編的）。"""
    scheme = json.loads(json.dumps(ZONE_NAMING_SCHEME))
    scene = scheme["scene"]
    scene["room_m"] = {"Lx": 3.4, "Ly": 7.8, "Lz": 2.7}
    plain = {"kind": "constant_impedance", "impedance_pa_s_m": 50000.0}
    scene["surface_zones_by_wall"] = {face: [{"area": "whole_wall", "material": plain}] for face in FACES}
    scheme["speakers"] = {"left": {"x": 0.9, "y": 7.1, "z": 0.9}, "right": {"x": 2.5, "y": 7.1, "z": 0.9}}
    for point in scheme["receiver_set"]["points"]:
        offset = [axis - centre for axis, centre in zip(point["position_m"], [2.45, 2.5, 1.42], strict=True)]
        point["position_m"] = [round(1.7 + offset[0], 3), round(4.4 + offset[1], 3), round(1.42 + offset[2], 3)]
    return cast(dict[str, object], scheme)


def _check_rows(page: Page) -> list[dict[str, object]]:
    rows = cast(list[dict[str, list[float]]], page.evaluate(_ROWS))
    assert sorted(cast(str, row["wall"]) for row in rows) == FACES
    for row in rows:
        tools, map_box = row["tools"], row["map"]
        assert map_box[0] - 1 <= tools[0] and tools[1] <= map_box[1] + 1, row  # 鈕不跑出圖外
        for other in rows:  # 哪一面的名字那一行都不跟哪一組鈕疊（含隔壁那一面的）
            title = other["title"]
            apart = title[1] <= tools[0] - 4 or tools[1] <= title[0] or title[3] <= tools[2] or tools[3] <= title[2]
            assert apart, (other["wall"], title, row["wall"], tools)
    return cast(list[dict[str, object]], rows)


def test_face_titles_make_room_for_their_buttons(tmp_path: Path, browser: Browser) -> None:
    """牆名那一行的「面向它站…」放得下就照寫，放不下改短句，再放不下就不寫；量的是畫出來的字與鈕，不是估的。
    假房間（前後牆五公尺多）與窄房間（前後牆三公尺多）各看一次：哪一面的字都不壓到鈕。"""
    _prepare(tmp_path)
    narrow = _narrow_scheme()
    narrow["scheme_id"] = "zone-narrow-sample"
    _write(tmp_path, narrow)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        _settled(page)
        rows = {row["wall"]: row for row in _check_rows(page)}
        assert any(row["aside"] for wall, row in rows.items() if wall in WALLS)  # 寬的那幾面還寫得下一句
        page.locator("#zone-page-close").click()
        page.locator("#scheme-list").select_option("zone-narrow-sample")
        page.locator("#open-scheme").click()
        page.wait_for_function("name => document.getElementById('save-id').value === name", arg="zone-narrow-sample")
        page.wait_for_function("!document.querySelector('main').inert")
        page.locator("#zone-page-open").click()
        expect(page.locator("#zone-map")).to_be_visible()
        _settled(page)
        narrow_rows = {row["wall"]: row for row in _check_rows(page)}
        assert page.locator("#zone-page [id^='zone-cut-']").count() == 6
