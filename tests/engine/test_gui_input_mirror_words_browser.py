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
from playwright.sync_api import Browser, Page, Route, expect

from aosr.gui.home_view import NO_SCHEME_MATERIALS_NOTE
from aosr.gui.zone_design import DesignSettings, write_design
from aosr.gui.zone_symmetry import Axis, symmetry_axis
from aosr.reporting.scheme import Scheme
from tests.engine._gui_cache import gui_startup_identity_memo as gui_startup_identity_memo
from tests.engine._input_page_steps import reveal
from tests.engine._surface_zone_cases import room109_document
from tests.engine._zone_symmetry_room import HEIGHT, LENGTH, NAME, searching, symmetry_document
from tests.engine.test_gui_browser import _assert_quiet, _open, _serve, browser as browser
from tests.engine.test_gui_input_clear_undo_browser import _release
from tests.engine.test_gui_zone_elevation_browser import _drag, _open_scheme, _write, _zones
from tests.engine.test_gui_zone_mirror_draw_browser import _drawn, _matches_mirror
from tests.engine.test_gui_zone_overview_tasks_browser import _library
from tests.engine.test_gui_zone_symmetry_browser import _settled

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
    where = "" if axis.centred else "；不在正中"  # #1265 說明字瘦身：在正中不多講
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


def test_invalid_draft_names_the_field_then_mirrors_again(tmp_path: Path, browser: Browser) -> None:
    document = _mirror_room(2.4)
    cast(dict[str, object], document["scene"])["room_m"] = {"Lx": 6, "Ly": WIDTH, "Lz": 3}
    _write(tmp_path, document)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-xL").click()
        expect(page.locator("#elevation-symmetry-on")).to_have_attribute("aria-pressed", "true")
        field = reveal(page, "#speaker-left-x")
        original = field.input_value()
        with page.expect_response("**/api/plan") as invalid:
            field.fill("")
            field.press("Tab")
        assert invalid.value.status == 422
        assert invalid.value.json()["problems"][0]["text"] == "左聲道喇叭 x 座標：空著沒填"
        _drag(page, WIDTH, 3, (0.35, 0.3), (1.15, 1.2))
        problem = "先改左聲道喇叭 x 座標：空著沒填"
        expect(page.locator("#elevation-note")).to_have_text("另一邊沒有跟著畫：" + problem)
        assert set(_drawn(page, "xL")) == {"z1"}
        page.locator("#zone-page-open").click()
        expect(page.locator("#zone-symmetry")).to_contain_text(problem)
        page.locator("#zone-page-close").click()
        with page.expect_response("**/api/plan") as repaired:
            field.fill(original)
            field.press("Tab")
        assert repaired.value.status == 200
        _drag(page, WIDTH, 3, (0.6, 1.65), (1.3, 2.45))
        boxes = _drawn(page, "xL")
        assert set(boxes) == {"z1", "z2", "z3"}
        _matches_mirror(page, "xL", boxes["z2"], boxes["z3"])


def _write_offset_window_room(tmp_path: Path) -> None:
    document = symmetry_document({"y0": [("window", (2.25, 3.5), (0.0, 1.7))]}, across_shift_m=-0.1)
    speakers = cast(dict[str, dict[str, float]], document["speakers"])
    speakers["left"]["y"], speakers["right"]["y"] = 1.3, 2.5
    walls = cast(dict[str, list[Doc]], cast(Doc, document["scene"])["surface_zones_by_wall"])
    walls["y0"][0]["material"] = {"kind": "assembly", "layers": [
        {"kind": "board", "surface_mass_kg_m2": 12.5, "resistance_pa_s_m": 0.0}], "backing": "air"}
    _write(tmp_path, document)
    _library(tmp_path)
    (tmp_path / "designs").mkdir()
    write_design(tmp_path, DesignSettings.model_validate(searching({"yL": ["whole_wall"]})), Scheme.model_validate(document))


def test_offset_window_and_opposite_wall_in_both_views(tmp_path: Path, browser: Browser) -> None:
    """左牆移窗，右牆仍是整面；在材料頁置中後，可在輸入頁補上同位置的框。"""
    _write_offset_window_room(tmp_path)
    shots = Path(os.environ.get("AOSR_SHOTS") or tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, NAME)
        page.locator("#elevation-wall-y0").click()
        page.set_viewport_size({"width": 1440, "height": 900})
        page.locator("#elevation-symmetry-off").click()
        page.locator("#elevation-symmetry-on").click()
        expect(page.locator("#elevation-symmetry-title")).to_have_text("兩邊一起畫（只切左牆）")
        expect(page.locator("#elevation-axis-words")).to_contain_text("離左牆 1.90 公尺")

        def preview() -> None:
            expect(page.locator("#elevation-banner")).to_have_text("放開後，只切左牆")

        _drag(page, LENGTH, HEIGHT, (2.05, 0.85), (2.45, 0.85), during=preview)
        _settled(page)
        expect(page.locator("#elevation-wall-count-y0")).to_have_text("窗＋3 區")
        expect(page.locator("#elevation-wall-count-yL")).to_have_text("整面")
        _check_offset_result(page, shots)
        page.locator("#elevation-panel").screenshot(path=shots / "after-input.png")
        page.locator("#zone-page-open").click()
        left, right = (page.locator(f"#zone-map .zv-zone[data-wall={face}]") for face in ("y0", "yL"))
        expect(left).to_have_count(4)
        expect(left.locator(".zv-open-label")).to_have_text("窗")
        expect(right).to_have_count(1)
        expect(page.locator("#zone-symmetry")).to_contain_text("左牆 4 區、右牆 1 區")
        expect(page.locator("#zone-symmetry-missed")).to_contain_text("左右牆不配對，各自調")
        page.screenshot(path=shots / "after-materials.png")
        page.locator("#zone-cut-yL").click()
        expect(page.locator("#elevation-symmetry-title")).to_have_text("左右對稱（只切右牆）")
        expect(page.locator("#elevation-axis-words")).to_be_visible()
        page.screenshot(path=shots / "after-right.png")
        page.get_by_role("button", name="將喇叭與座位一起置中").click()
        _settled(page)
        expect(page.locator("#speaker-left-y")).to_have_value("1.4")
        expect(page.locator("#speaker-right-y")).to_have_value("2.6")
        expect(page.locator("#receiver-main-y")).to_have_value("2")
        expect(page.locator("#receiver-left-y")).to_have_value("1.76")
        next_step = "已置中。要補上既有的框，請在原牆選取那一塊，按「補畫另一邊」。"
        expect(page.locator("#elevation-banner")).to_have_text(next_step)
        page.locator("#zone-finish-cut").click()
        expect(right).to_have_count(1)
        expect(page.locator("#zone-page-note")).to_have_text(next_step)
        _complete_window_frame(page)
        page.screenshot(path=shots / "after-complete.png")
        _assert_quiet(watched)


def _check_offset_result(page: Page, shots: Path) -> None:
    expect(page.locator("#elevation-banner")).to_have_text("左牆切成 4 區，右牆仍是整面")
    expect(page.locator("#elevation-note")).to_be_empty()
    expect(page.locator("#elevation-selected")).not_to_contain_text("主位和兩支喇叭的中點")
    expect(page.locator("#elevation-panel")).not_to_contain_text("都加上")
    expect(page.get_by_role("button", name="將喇叭與座位一起置中")).to_be_visible()
    page.locator("#elevation-axis-words").evaluate("e => window.scrollBy(0, e.getBoundingClientRect().top - 180)")
    page.screenshot(path=shots / "input-offset.png")


def _complete_window_frame(page: Page) -> None:
    page.locator("#zone-page-close").click()
    page.locator("#elevation-wall-y0").click()
    page.locator("#elevation [data-zone-id=window] > rect").click()
    page.locator("#elevation-complete").click()
    page.locator("#zone-page-open").click()
    expect(page.locator("#zone-map .zv-zone[data-wall=y0]")).to_have_count(4)
    expect(page.locator("#zone-map .zv-zone[data-wall=yL]")).to_have_count(4)
    mirror, = _drawn(page, "yL").values()
    source = _drawn(page, "y0")["window"]
    assert (mirror["u_m"], mirror["v_m"]) == (source["u_m"], source["v_m"])


@pytest.mark.parametrize("undo", [False, True], ids=["cut", "undo-clear"])
def test_delete_pair_before_limits_reply(tmp_path: Path, browser: Browser, undo: bool) -> None:
    frame = ("z1", (1.35, 2.7), (0.45, 1.55))
    _write(tmp_path, symmetry_document({"y0": [frame], "yL": [frame]} if undo else {}))
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, NAME)
        page.locator("#elevation-wall-y0").click()
        _settled(page)
        if undo:
            # 清掉左牆後，先收下「右框落單」的完整配對表。復原切法時這張表就過期了。
            with page.expect_response("**/api/zone-limits"):
                page.locator("#elevation-clear").click()
            _settled(page)
            assert _zones(page, "y0")[0]["area"] == "whole_wall"
        held: list[Route] = []
        page.route("**/api/zone-limits", lambda route: held.append(route))
        with page.expect_request("**/api/zone-limits"):
            if undo:
                page.locator("#elevation-undo-clear").click()
                page.locator("#elevation [data-zone-id=z1] > rect").click()
            else:
                with page.expect_response("**/api/zone-cut") as cut:
                    _drag(page, LENGTH, HEIGHT, (1.15, 0.45), (2.45, 1.55))
                assert cut.value.json()["paired"] == "yL@z1"
        opposite = _drawn(page, "yL")
        assert set(opposite) == {"z1"}
        field = reveal(page, "#speaker-left-x")
        with page.expect_response("**/api/plan") as invalid:
            field.fill("")
            field.press("Tab")
        assert invalid.value.status == 422
        assert held
        with page.expect_response("**/api/zone-cut") as deleted:
            page.locator("#elevation-delete").click()
        try:
            edit = "只改了選取的框，另一邊沒跟著改" if undo else "只改了左牆，右牆沒跟著改"
            warning = edit + "：先改左聲道喇叭 x 座標：空著沒填"
            assert deleted.value.json()["message"] == warning
            expect(page.locator("#elevation-banner")).to_have_text(warning)
            payload = deleted.value.request.post_data_json
            assert isinstance(payload, dict)
            assert payload["mirror"].get("previous_pair") == (None if undo else "left")
            assert _zones(page, "y0")[0]["area"] == "whole_wall"
            assert _drawn(page, "yL") == opposite
        finally:
            page.unroute("**/api/zone-limits")
            _release(held)


@pytest.mark.parametrize("delete", [False, True], ids=["move", "delete"])
def test_invalid_draft_reports_side_edit(tmp_path: Path, browser: Browser, delete: bool) -> None:
    frame = ("paired", (1.25, 2.65), (0.4, 1.6))
    _write(tmp_path, symmetry_document({"y0": [frame], "yL": [frame]}))
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, NAME)
        page.locator("#elevation-wall-y0").click()
        page.locator("#elevation [data-zone-id=paired] > rect").click()
        expect(page.locator(".elevation-pair-status")).to_contain_text("左右對稱")
        opposite = _zones(page, "yL")
        field = reveal(page, "#speaker-left-x")
        with page.expect_response("**/api/plan") as invalid:
            field.fill("")
            field.press("Tab")
        assert invalid.value.status == 422
        if delete:
            page.locator("#elevation-delete").click()
        else:
            bottom = page.locator("#elevation-bottom")
            bottom.fill("0.65")
            bottom.press("Tab")
        _settled(page)
        if delete:
            assert _zones(page, "y0")[0]["area"] == "whole_wall"
        else:
            assert _drawn(page, "y0")["paired"]["v_m"] == pytest.approx([0.65, 1.85])
        assert _zones(page, "yL") == opposite
        expect(page.locator("#elevation-banner")).to_have_text("只改了左牆，右牆沒跟著改：先改左聲道喇叭 x 座標：空著沒填")


def test_changing_placement_expires_centring_in_both_views(tmp_path: Path, browser: Browser) -> None:
    _write_offset_window_room(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        page.set_viewport_size({"width": 1440, "height": 900})
        _open_scheme(page, NAME)
        page.locator("#elevation-wall-y0").click()
        page.get_by_role("button", name="將喇叭與座位一起置中").click()
        expect(page.locator("#elevation-banner")).to_contain_text("已置中")
        for name, value in [("speaker-left-y", "1.25"), ("speaker-right-y", "2.45"), ("receiver-main-y", "1.85")]:
            field = reveal(page, "#" + name)
            field.fill(value)
            field.press("Tab")
        expect(page.locator("#elevation-axis-words")).to_have_text("鏡子：喇叭中線，離左牆 1.85 公尺；不在正中")
        expect(page.locator("body")).not_to_contain_text("已置中", use_inner_text=True)
        page.locator("#zone-page-open").click()
        expect(page.locator("#zone-symmetry").get_by_text("鏡子不在正中，左牆 4 區、右牆 1 區，各自切區。", exact=True)).to_be_visible()
        expect(page.locator("#zone-page-note")).to_be_empty()
        expect(page.locator("body")).not_to_contain_text("已置中", use_inner_text=True)
        page.locator("#zone-symmetry").scroll_into_view_if_needed()
        page.screenshot(path=Path(os.environ.get("AOSR_SHOTS") or tmp_path) / "placement-changed.png")
