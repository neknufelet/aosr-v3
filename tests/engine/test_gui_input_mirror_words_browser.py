"""兩處字（#1265 單元 3 之二第 A 條）：

- 輸入頁第 2 段「兩邊一起畫」開關底下那一句講成鏡子（示意圖 input-page-v3 第 2 段）：離左牆幾公尺、在不在房間正中都照
  伺服器算的對稱軸（zone_symmetry.symmetry_axis）；立面圖上那條線的字是「鏡子：兩支喇叭的中線」。
- 首頁「挑材料」沒有帶得過去的那一份時那一句小字：指的兩顆（「換一個方案」、「這間房貼什麼」）頂條上真的有。

房間是考卷自己編的：109 號房改寬 5 公尺，兩支喇叭與座位一起左右挪，一間挪到正中、一間不在正中。"""
from __future__ import annotations

import os
from pathlib import Path
from typing import cast

import pytest
from playwright.sync_api import Browser, Route, expect

from aosr.gui.home_view import NO_SCHEME_MATERIALS_NOTE
from aosr.gui.zone_symmetry import Axis, symmetry_axis
from aosr.reporting.scheme import Scheme
from tests.engine._gui_cache import gui_startup_identity_memo as gui_startup_identity_memo
from tests.engine._surface_zone_cases import room109_document
from tests.engine.test_gui_browser import _assert_quiet, _open, _serve, browser as browser
from tests.engine.test_gui_input_clear_undo_browser import _release
from tests.engine.test_gui_zone_elevation_browser import _drag, _open_scheme, _write

WIDTH = 5.0        # 房間寬（左牆到右牆）
SPREAD = 0.7       # 兩支喇叭各離中線多遠
Doc = dict[str, object]


def _mirror_room(middle: float) -> Doc:
    """兩支喇叭的中點與每一個座位都挪到離左牆 middle 公尺（前牆 x0、左牆 y0：左右是 y 方向）。"""
    document = room109_document()
    scene = cast(dict[str, dict[str, float]], document["scene"])
    scene["room_m"]["Ly"] = WIDTH
    speakers = cast(dict[str, dict[str, float]], document["speakers"])
    speakers["left"]["y"], speakers["right"]["y"] = middle - SPREAD, middle + SPREAD
    points = cast(list[dict[str, list[float]]], cast(Doc, document["receiver_set"])["points"])
    shift = middle - points[0]["position_m"][1]
    for point in points:
        point["position_m"][1] += shift
    return document


def _axis(document: Doc) -> Axis:
    axis = symmetry_axis(Scheme.model_validate(document))
    assert isinstance(axis, Axis), axis
    return axis


@pytest.mark.parametrize("middle", [WIDTH / 2, 1.7], ids=["centred", "off-centre"])  # 喇叭、座位落在的數都避開派工單不准用的那幾個
def test_the_mirror_sentence_says_where_the_mirror_is(tmp_path: Path, browser: Browser, middle: float) -> None:
    document = _mirror_room(middle)
    axis = _axis(document)
    assert axis.front == "x0" and axis.left == "y0" and axis.centred is (middle == WIDTH / 2)
    where = "" if axis.centred else "；不在正中，左右牆不跟著畫"  # #1265 說明字瘦身：在正中不多講
    _write(tmp_path, document)
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-x0").click()
        page.wait_for_function("zoneLimitsGeneration > 0 && zoneLimitsShown === zoneLimitsGeneration")
        expect(page.locator("#elevation-symmetry-on")).to_have_attribute("aria-pressed", "true")
        expect(page.locator("#elevation-axis-words")).to_have_text(
            f"鏡子：喇叭中線，離左牆 {axis.across_m:.2f} 公尺{where}")
        label = page.locator("#elevation .mirror-axis text")
        expect(label).to_have_text("鏡子：兩支喇叭的中線")
        # 那幾個字整串落在立面圖裡（線靠右時字不跑出圖外）。
        inside = label.evaluate("""text => { const box = text.getBoundingClientRect(), svg = text.ownerSVGElement.getBoundingClientRect();
            return box.left >= svg.left && box.right <= svg.right && box.top >= svg.top && box.bottom <= svg.bottom; }""")
        assert inside is True
        expect(page.locator("#elevation-axis-words")).not_to_contain_text("對稱軸")
        shots = Path(os.environ.get("AOSR_SHOTS") or tmp_path)  # 給了資料夾就把截圖存去那裡（證據），沒給存暫存
        if axis.centred:
            page.locator("#elevation-panel .card.elev").screenshot(path=str(shots / "u3b-mirror-words.png"))
        _assert_quiet(watched)


def test_while_the_mirror_is_being_checked_the_sentence_says_so_in_mirror_words(tmp_path: Path, browser: Browser) -> None:
    """第二趟決定第 6 條：限制檢查（順便回鏡子在哪）還在路上時，開關底下那一句寫「正在確認鏡子在哪…」，不寫「對稱軸」；
    回來之後寫鏡子在哪。用畫一個框讓檢查重問一次，攔住那一次。"""
    document = _mirror_room(WIDTH / 2)
    _write(tmp_path, document)
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-x0").click()
        page.wait_for_function("zoneLimitsGeneration > 0 && zoneLimitsShown === zoneLimitsGeneration")
        words = page.locator("#elevation-axis-words")
        expect(words).to_contain_text("鏡子：喇叭中線")
        held: list[Route] = []
        page.route("**/api/zone-limits", lambda route: held.append(route))
        _drag(page, WIDTH, cast(dict[str, dict[str, float]], document["scene"])["room_m"]["Lz"], (0.3, 0.3), (1.2, 1.4))
        page.wait_for_function("!document.querySelector('main').inert && !inputBusy")
        assert held, "限制檢查應該還在路上"
        expect(words).to_have_text("正在確認鏡子在哪…")
        _release(held)  # 被新一次蓋掉而中止的那幾個不用再放
        page.unroute("**/api/zone-limits")
        page.wait_for_function("zoneLimitsShown === zoneLimitsGeneration && zoneLimitsReceived === zoneLimitsGeneration")
        expect(words).to_contain_text("鏡子：喇叭中線")
        _assert_quiet(watched)


def test_the_home_note_points_at_buttons_that_are_really_on_the_bar(tmp_path: Path, browser: Browser) -> None:
    """還沒算過任何一份：原本首頁「挑材料」底下那一句（NO_SCHEME_MATERIALS_NOTE）講「換一個方案」「這間房貼什麼」兩顆；
    #1265 單元 5c 拿掉 #home 兩張大卡，那一句不再上頁面（/api/home 還排它，home_view 的題守字）。這裡照舊守它講的兩顆
    都在頂條上、看得到，按頂上的「這間房貼什麼」真的進那一頁。"""
    _write(tmp_path, room109_document())
    with _serve(tmp_path) as base, _open(browser, f"{base}/", viewport_width=1440) as watched:
        page = watched.page
        page.wait_for_function("document.body.dataset.inputReady === 'true'")
        expect(page.locator("#home-materials")).to_have_count(0)
        assert "設定每一區貼什麼" not in NO_SCHEME_MATERIALS_NOTE
        assert "換一個方案" in NO_SCHEME_MATERIALS_NOTE and "這間房貼什麼" in NO_SCHEME_MATERIALS_NOTE
        expect(page.locator("#scheme-menu-button")).to_contain_text("換一個方案")
        expect(page.locator("#scheme-menu-button")).to_be_visible()
        nav = page.locator("#input-bars nav #nav-zones")
        expect(nav).to_have_text("這間房貼什麼")
        expect(nav).to_be_visible()
        shots = Path(os.environ.get("AOSR_SHOTS") or tmp_path)
        page.locator("#input-bars").screenshot(path=str(shots / "u3b-home-note.png"))
        nav.click()
        expect(page.locator("#zone-page")).to_be_visible()
        _assert_quiet(watched)
