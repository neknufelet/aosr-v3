"""真瀏覽器考「這間房貼什麼」每一面的「切區 ▸」（#1262）：畫框在「房間方案輸入」的立面圖，這一頁只選區、選貼什麼；
按「切區 ▸」回輸入頁、打開那一面的立面圖。切過之後回來，選到的區代號不在了、或換了一塊長方形，就不再算選著
（自動補滿的區會重新編號，同一個代號可能指到另一塊）。牆窄的時候，牆名那一行的字讓出右邊兩顆鈕，不疊在一起。

房間用區名考卷那間假房間（``_zone_naming_room``，數字是考卷自己編的）；窄房間是同一份改的，尺寸也是這裡自己編的。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from playwright.sync_api import Browser, Page, expect

from tests.engine._input_page_steps import open_scheme
from tests.engine._gui_cache import gui_startup_identity_memo as gui_startup_identity_memo
from tests.engine._zone_naming_room import ZONE_NAMING_SCHEME
from tests.engine.test_gui_browser import _open, _serve, browser as browser
from tests.engine.test_gui_zone_elevation_browser import _drag, _write
from tests.engine.test_gui_zone_overview_tasks_browser import _SELECTED, _click_zone, _open_overview, _prepare, _settled

FACES = ["ceiling", "floor", "x0", "xL", "y0", "yL"]
WALLS = ["x0", "xL", "y0", "yL"]
FRONT_SIZE = (5.2, 3.15)  # 假房間前牆（yL）在立面圖上的寬與高（公尺）
# 每一面的鈕、牆名那一行（旁白寫了什麼）、圖上每一段字各占多大（視窗像素：左、右、上、下）。
_ROWS = """() => { const box = item => { const rect = item.getBoundingClientRect(); return [rect.left, rect.right, rect.top, rect.bottom]; };
    return {map: box(document.getElementById('zone-map')),
            texts: [...document.querySelectorAll('#zone-map text')].map(text => ({words: text.textContent, box: box(text)})),
            rows: [...document.querySelectorAll('#zone-page .zv-face-tools')].map(tools => {
              const wall = tools.dataset.wall, title = document.getElementById(`zone-face-title-${wall}`);
              return {wall, tools: box(tools), aside: title.querySelector('tspan')?.textContent ?? null}; })}; }"""


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
        # #1265 單元 5c 拿掉 #home 兩張大卡後整頁短了，1100 高的視窗捲到最上面也看得到立面圖：視窗改 900 高（老闆用的 1440×900
        # 那一種的高度），「捲到最上面時立面圖在視窗外」這個前提才成立。
        page.set_viewport_size({"width": 1400, "height": 900})
        page.evaluate("window.scrollTo(0, 0)")
        expect(page.locator("#elevation-frame")).not_to_be_in_viewport()
        # #1265 單元 3 拿掉方案那一列那顆（派工單細節裁決 5）：頁面最上面改按頂上導覽列的「這間房貼什麼」（同一支 openZoneOverview）。
        page.locator("#nav-zones").click()
        expect(page.locator("#zone-map")).to_be_visible()
        before = page.locator("#zone-map [data-wall='yL'][data-zone]").count()
        page.locator("#zone-cut-yL").click()
        expect(page.locator("#elevation-wall-yL")).to_have_attribute("aria-pressed", "true")
        expect(page.locator("#elevation-wall-yL")).to_be_focused()  # 鍵盤從那一面的牆名鈕接著走
        expect(page.locator("#elevation-wall-floor")).to_have_attribute("aria-pressed", "false")
        expect(page.locator("#elevation-frame")).to_be_in_viewport()
        _cut_front_inside_auto2(page)
        page.locator("#zone-page-open").click()
        expect(page.locator("#zone-map")).to_be_visible()
        _settled(page)
        expect(page.locator("#zone-map [data-wall='yL'][data-zone='z4']")).to_have_count(1)
        assert page.locator("#zone-map [data-wall='yL'][data-zone]").count() > before


def _said(page: Page) -> str:
    return cast(str, page.evaluate("document.getElementById('zone-page-said')?.textContent ?? ''"))


def _back_to_overview(page: Page) -> None:
    page.locator("#zone-page-open").click()
    expect(page.locator("#zone-map")).to_be_visible()
    _settled(page)


def test_a_selection_whose_rectangle_changed_after_cutting_is_dropped(tmp_path: Path, browser: Browser) -> None:
    """切過那一面再回來：選到的區代號不在了、或同一個代號換成另一塊長方形（自動補滿的區重新編號），就不再算選著，
    訊息列講一句（不然小框還沒套用的改動會套到另一塊上）；同一面沒動到的區、別一面的區照樣選著。
    小框的草稿只在一區都不剩時才丟，那時那一句才加「還沒套用的改動放掉了」。"""
    _prepare(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        _settled(page)
        # 一、同一面兩區：auto2 被切到、z1（左邊那一大塊）沒動。走原本那條路（不靠新鈕）：回輸入頁、自己點前牆、切框、回來。
        _click_zone(page, "yL/auto2")
        _click_zone(page, "yL/z1")
        expect(page.locator("#zone-panel")).to_be_visible()
        page.locator("#zone-panel-mode-search").check()  # 小框動過（兩區原本都不是由程式最佳化）
        shape = _shape(page, "yL/auto2")
        page.locator("#zone-page-close").click()
        page.locator("#elevation-wall-yL").click()
        _cut_front_inside_auto2(page)
        _back_to_overview(page)
        after = _shape(page, "yL/auto2")
        assert after is not None and after != shape  # 這一刀切下去，auto2 這個代號確實換了一塊（不然這一題考不到）
        assert page.evaluate(_SELECTED) == ["yL/z1"]
        expect(page.locator("#zone-panel")).to_be_visible()
        expect(page.locator("#zone-panel-mode-search")).to_be_checked()  # 還有一區選著：草稿留著
        assert _said(page) == "剛才選的區有 1 區那一面重切過，已經不是原來那一塊，不再算選著，要的話重新選"
        # 二、整面一區被切開（代號不在了），小框動過：一區都不剩，草稿也丟，那一句多半句。
        _click_zone(page, "yL/z1")
        expect(page.locator("#zone-panel")).to_have_count(0)
        assert _said(page) == ""  # 下一次按東西就收掉
        _click_zone(page, "x0/whole_wall")
        page.locator("#zone-panel-mode-search").check()
        page.locator("#zone-cut-x0").click()
        _drag(page, 5.5, 3.15, (2.0, 1.0), (3.0, 2.0))
        _back_to_overview(page)
        expect(page.locator("#zone-map [data-wall='x0'][data-zone='z1']")).to_have_count(1)
        assert page.evaluate(_SELECTED) == []
        expect(page.locator("#zone-panel")).to_have_count(0)
        assert _said(page) == "剛才選的區有 1 區那一面重切過，已經不是原來那一塊，不再算選著；小框還沒套用的改動放掉了，要的話重新選"
        # 三、對照：選右牆一區，切前牆（右牆不動），回來照樣選著、不講那一句。
        _click_zone(page, "xL/auto3")
        expect(page.locator("#zone-panel")).to_be_visible()
        expect(page.locator("#zone-panel-mode-search")).not_to_be_checked()  # 上一段丟掉的草稿沒有帶過來
        page.locator("#zone-cut-yL").click()
        _drag(page, *FRONT_SIZE, (2.35, 0.2), (2.85, 0.6))
        _back_to_overview(page)
        assert page.evaluate(_SELECTED) == ["xL/auto3"]
        expect(page.locator("#zone-panel")).to_be_visible()
        assert "重切過" not in _said(page)


def test_all_dropped_with_an_untouched_panel_says_nothing_about_changes(tmp_path: Path, browser: Browser) -> None:
    """選了左牆整面、小框開著但沒動過，切開那一面再回來：一區都不剩，那一句不提「還沒套用的改動」（本來就沒有）。"""
    _prepare(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        _settled(page)
        _click_zone(page, "x0/whole_wall")
        expect(page.locator("#zone-panel")).to_be_visible()
        page.locator("#zone-cut-x0").click()
        _drag(page, 5.5, 3.15, (2.0, 1.0), (3.0, 2.0))
        _back_to_overview(page)
        assert page.evaluate(_SELECTED) == []
        expect(page.locator("#zone-panel")).to_have_count(0)
        assert _said(page) == "剛才選的區有 1 區那一面重切過，已經不是原來那一塊，不再算選著，要的話重新選"


def test_changing_room_size_keeps_the_selection(tmp_path: Path, browser: Browser) -> None:
    """只改房間尺寸、沒重切：選的區（整面一區、鏡像面的長方形）照樣選著，也不講「重切過」那一句。
    畫面座標會跟著房間尺寸變（先改高：整面一區變；再改右牆那個方向的長：鏡像面的長方形變），比的是方案裡的座標。"""
    _prepare(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        _settled(page)
        _click_zone(page, "x0/whole_wall")
        _click_zone(page, "xL/auto3")
        chosen = sorted(cast(list[str], page.evaluate(_SELECTED)))
        assert chosen == ["x0/whole_wall", "xL/auto3"]
        page.locator("#zone-page-close").click()
        page.locator("#room-Lz").fill("3.2")
        page.locator("#room-Lz").press("Tab")
        page.wait_for_function("!pendingRoomResize() && !document.querySelector('main').inert")
        _back_to_overview(page)
        assert sorted(cast(list[str], page.evaluate(_SELECTED))) == chosen
        assert "重切過" not in _said(page)
        expect(page.locator("#zone-panel")).to_be_visible()
        # 再改右牆那個方向的長度：右牆是鏡像面，xL/auto3 的畫面座標跟著變，方案座標沒變，照樣選著。
        page.locator("#zone-page-close").click()
        page.locator("#room-Ly").fill("5.6")
        page.locator("#room-Ly").press("Tab")
        page.wait_for_function("!pendingRoomResize() && !document.querySelector('main').inert")
        _back_to_overview(page)
        assert sorted(cast(list[str], page.evaluate(_SELECTED))) == chosen
        assert "重切過" not in _said(page)


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


def _check_rows(page: Page) -> dict[str, str | None]:
    """每一組鈕都在圖裡、不壓到圖上任何一段字（牆名那一行、隔壁那一面的、門窗、喇叭的字都算）；回每一面旁白寫了什麼。"""
    found = cast(dict[str, object], page.evaluate(_ROWS))
    rows = cast(list[dict[str, object]], found["rows"])
    map_box = cast(list[float], found["map"])
    texts = cast(list[dict[str, object]], found["texts"])
    assert sorted(cast(str, row["wall"]) for row in rows) == FACES
    for row in rows:
        tools = cast(list[float], row["tools"])
        assert map_box[0] - 1 <= tools[0] and tools[1] <= map_box[1] + 1 and tools[3] <= map_box[3] + 1, row  # 鈕不跑出圖外
        for text in texts:
            box = cast(list[float], text["box"])
            apart = box[1] <= tools[0] - 4 or tools[1] <= box[0] or box[3] <= tools[2] or tools[3] <= box[2]
            assert apart, (text["words"], box, row["wall"], tools)
    return {cast(str, row["wall"]): cast(str | None, row["aside"]) for row in rows}


def _switch_to(page: Page, scheme_id: str) -> None:
    page.locator("#zone-page-close").click()
    open_scheme(page, scheme_id)
    page.wait_for_function("name => document.getElementById('save-id').value === name", arg=scheme_id)
    page.wait_for_function("!document.querySelector('main').inert")
    page.locator("#zone-page-open").click()
    expect(page.locator("#zone-map")).to_be_visible()
    _settled(page)


def _back_door_scheme() -> dict[str, object]:
    """假房間改一份：後牆（y0）靠右開一扇門（借右牆那扇門的構造；位置是這裡自己編的）。地板小圖下緣外寫「門」，
    「切區 ▸」不准壓到它。"""
    scheme = json.loads(json.dumps(ZONE_NAMING_SCHEME))
    scheme["scheme_id"] = "zone-rear-opening-sample"
    zones = scheme["scene"]["surface_zones_by_wall"]
    door = next(zone for zone in zones["xL"] if zone["id"] == "door")["material"]
    plain = {"kind": "constant_impedance", "impedance_pa_s_m": 71500.0}
    zones["y0"] = [
        {"area": "rectangle", "id": "door", "u_m": [3.9, 4.8], "v_m": [0.0, 2.05], "material": door},
        {"area": "rectangle", "id": "auto1", "u_m": [0.0, 3.9], "v_m": [0.0, 3.15], "material": plain},
        {"area": "rectangle", "id": "auto2", "u_m": [4.8, 5.2], "v_m": [0.0, 3.15], "material": plain},
        {"area": "rectangle", "id": "auto3", "u_m": [3.9, 4.8], "v_m": [2.05, 3.15], "material": plain}]
    return cast(dict[str, object], scheme)


def test_face_titles_make_room_for_their_buttons(tmp_path: Path, browser: Browser) -> None:
    """牆名那一行的「面向它站…」放得下就照寫，放不下改短句，再放不下就不寫；量的是畫出來的字與鈕，不是估的。
    三間房各看一次，哪一組鈕都不壓到圖上的字：假房間（前後牆五公尺多：四面都換短句）、窄房間（前後牆三公尺多：
    前後牆不寫、側牆寫得下整句）、後牆靠右有門的假房間（地板小圖下緣外的「門」字不被「切區 ▸」蓋住）。"""
    _prepare(tmp_path)
    narrow = _narrow_scheme()
    narrow["scheme_id"] = "zone-narrow-sample"
    _write(tmp_path, narrow)
    _write(tmp_path, _back_door_scheme())
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        _settled(page)
        asides = _check_rows(page)
        assert all((asides[wall] or "").startswith("左手邊") for wall in WALLS), asides
        assert (asides["floor"], asides["ceiling"]) == ("從上往下看", "方向同地板圖")
        _switch_to(page, "zone-narrow-sample")
        asides = _check_rows(page)
        assert asides["yL"] is None and asides["y0"] is None, asides
        assert all((asides[wall] or "").startswith("面向它站") for wall in ("x0", "xL")), asides
        _switch_to(page, "zone-rear-opening-sample")
        _check_rows(page)
        doors = page.locator("#zone-map [data-face='floor'] text.zv-tiny", has_text="門")
        expect(doors).to_have_count(2)  # 右牆、後牆各一扇（後牆那扇的字在小圖下緣外）
