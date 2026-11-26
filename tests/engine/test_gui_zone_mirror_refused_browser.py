"""#1274 在放開滑鼠前就看得到鏡像放不下的原因。"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
from playwright.sync_api import Browser, Page, expect

from aosr.gui.zone_edit import CutBox, WallName
from aosr.gui.zone_symmetry import Frame, symmetry_axis
from aosr.reporting.scheme import Scheme
from tests.engine._gui_cache import gui_startup_identity_memo as gui_startup_identity_memo
from tests.engine._input_page_steps import reveal
from tests.engine.test_gui_browser import browser as browser
from tests.engine.test_gui_zone_elevation_browser import _cut_by, _drag
from tests.engine.test_gui_zone_mirror_draw_browser import (
    BLOCK, END, FRONT, H, OTHER, START, W, _drawn, _matches_mirror, _room,
)
from tests.engine.test_gui_zone_symmetry_browser import _settled

RIGHT_START, RIGHT_END = (4.25, 0.02), (5.19, 2.0)
PREFIX = "放開後另一邊不會跟著畫："


def _outside(page: Page, wall: str) -> float:
    scheme = Scheme.model_validate(page.evaluate("scheme"))
    axis = symmetry_axis(scheme)
    assert not isinstance(axis, str)
    return round(Frame(scheme, axis).sizes[wall][0] - 2 * axis.across_m, 2)


def _top(page: Page) -> None:
    page.set_viewport_size({"width": 1160, "height": 720})
    page.locator("#elevation-svg").evaluate("e => e.scrollIntoView({block:'start'})")


def _refusal(page: Page, *, opposite: bool = False) -> str:
    preview = page.locator(".mirror-preview")
    if opposite:
        assert preview.count() == 0
        expect(page.locator("#elevation-note")).to_contain_text(PREFIX)
    else:
        expect(preview).to_contain_text("放不下")
        rect = preview.locator("rect")
        expect(rect).to_have_css("stroke", "rgb(165, 39, 20)")
        expect(rect).not_to_have_css("stroke-dasharray", "none")
    text = page.locator("#elevation-note").inner_text()
    assert text.startswith(PREFIX), text
    return text.removeprefix(PREFIX)


@pytest.mark.parametrize("wall", ["yL", "y0"])
def test_outside_preview_and_selection(tmp_path: Path, browser: Browser, wall: WallName) -> None:
    with _room(tmp_path, browser, wall=wall) as page:
        _top(page)
        distance = f"{_outside(page, wall):.2f} 公尺"
        start, end = RIGHT_START, RIGHT_END
        if wall == "y0":  # 站在後牆前面看，左緣在房間的右邊。
            start, end = (W - start[0], start[1]), (W - end[0], end[1])
        reasons: list[str] = []

        def during() -> None:
            reason = _refusal(page)
            assert f"超出牆面，左牆那一側多出 {distance}" in reason
            # 修法只寫一次，用畫面上選取那一列的欄位名稱與數字（後牆左右顛倒也照畫面講）。
            assert "。改成：離左緣 " in reason and f"（整塊往對稱軸挪 {distance}）" in reason
            assert "往內收" not in reason
            reasons.append(reason)
            page.screenshot(path=tmp_path / "outside-dragging.png")

        with page.expect_response("**/api/zone-cut") as released:
            _drag(page, W, H, start, end, during=during)
        _settled(page)
        cause = reasons[0].split("。改成：")[0]
        assert cause in released.value.json()["message"] and "改成" not in released.value.json()["message"]
        note = page.locator("#elevation-note")
        expect(note).to_contain_text(reasons[0])
        assert note.evaluate("e => e.previousElementSibling.id === 'elevation' && e.nextElementSibling.id === 'elevation-selected'")
        assert note.evaluate("e => e.getBoundingClientRect().top >= 0 && e.getBoundingClientRect().bottom <= innerHeight")
        page.screenshot(path=tmp_path / "released.png")
        assert set(_drawn(page, wall)) == {"z1"}
        page.locator("#elevation [data-zone-id=z1] > rect").click()
        expect(page.locator("#elevation-selected")).to_contain_text("這一塊翻過去放不下：" + reasons[0])
        expect(page.locator("#elevation-complete")).to_have_count(0)
        page.locator("#elevation-panel").screenshot(path=tmp_path / "selected.png")


def test_blocking_box(tmp_path: Path, browser: Browser) -> None:
    with _room(tmp_path, browser, (BLOCK, OTHER)) as page:
        _top(page)
        label = page.evaluate("zoneCalled('yL','block')")

        def during() -> None:
            assert f"壓到 {label}" in _refusal(page)
            page.screenshot(path=tmp_path / "blocked-dragging.png")

        _drag(page, W, H, START, END, during=during)
        _settled(page)
        expect(page.locator("#elevation-note")).to_contain_text(f"壓到 {label}")
        expect(page.locator("#elevation-complete")).to_be_visible()


def test_off_centre_crossing(tmp_path: Path, browser: Browser) -> None:
    with _room(tmp_path, browser) as page:
        reasons: list[str] = []

        def during() -> None:
            reason = _refusal(page)
            assert "跨在對稱軸上但沒有置中" in reason and "0.15 公尺" in reason
            reasons.append(reason)

        _drag(page, W, H, (2.0, 0.02), (2.6, 1.8), during=during)
        _settled(page)
        expect(page.locator("#elevation-note")).to_contain_text(reasons[0])
        expect(page.locator("#elevation-selected")).to_contain_text("這一塊沒有另一邊：" + reasons[0])
        expect(page.locator("#elevation-complete")).to_have_count(0)


def test_fitting_pair(tmp_path: Path, browser: Browser) -> None:
    with _room(tmp_path, browser) as page:
        def during() -> None:
            expect(page.locator(".mirror-preview text")).to_have_text(["放開後", "這裡跟著出現"])

        _drag(page, W, H, START, END, during=during)
        zones = _drawn(page)
        _matches_mirror(page, FRONT, zones["z1"], zones["z2"])


def test_opposite_wall_obstacle(tmp_path: Path, browser: Browser) -> None:
    # 左右牆同寬同高、框在對面不翻座標；合法拖曳能在對面放不下的情形是該處有別框。
    obstacle = CutBox(id="block", u_m=(0.8, 1.6), v_m=(0.5, 1.5))
    with _room(tmp_path, browser, centred=True, wall="x0", opposite=(obstacle,)) as page:
        label = page.evaluate("zoneCalled('xL','block')")

        def during() -> None:
            assert f"壓到 {label}" in _refusal(page, opposite=True)

        _drag(page, 5.5, H, START, END, during=during)
        expect(page.locator("#elevation-note")).to_contain_text(label)
        assert set(_drawn(page, "xL")) == {"block"}


def test_move_at_suggested_distance(tmp_path: Path, browser: Browser) -> None:
    with _room(tmp_path, browser) as page:
        _drag(page, W, H, RIGHT_START, RIGHT_END)
        reason = page.locator("#elevation-note").inner_text()
        move = re.search(r"往對稱軸挪 ([0-9.]+) 公尺", reason)
        assert move is not None, reason
        shift = float(move[1])
        assert shift == _outside(page, FRONT)
        _settled(page)
        _top(page)
        grab = (4.75, 1.0)
        _drag(page, W, H, grab, (grab[0] - shift, grab[1]))
        _settled(page)
        zones = _drawn(page)
        _matches_mirror(page, FRONT, zones["z1"], zones["z2"])
        assert zones["z1"]["u_m"] == pytest.approx([RIGHT_START[0] - shift, W - shift])
        page.screenshot(path=tmp_path / "repaired.png")


def test_thin_remainder_is_refused_while_dragging(tmp_path: Path, browser: Browser) -> None:
    """對面切完會剩窄條：拖著時就講放不下，跟放開後同一句（審查 #1）。"""
    with _room(tmp_path, browser) as page:
        reasons: list[str] = []

        def during() -> None:
            reason = _refusal(page)
            assert "窄條" in reason, reason
            reasons.append(reason)

        _drag(page, W, H, (2.8, 0.02), (4.8, 1.8), during=during)
        _settled(page)
        expect(page.locator("#elevation-note")).to_contain_text(reasons[0])
        assert set(_drawn(page)) == {"z1"}


def test_wide_box_is_told_to_shrink_not_to_move(tmp_path: Path, browser: Browser) -> None:
    """框寬到整塊挪過去會跨軸：只講收靠牆那一邊；照著把寬度改小，兩邊就都在（審查 #2）。"""
    with _room(tmp_path, browser) as page:
        _drag(page, W, H, (2.6, 0.02), RIGHT_END)  # 離軸 0.15：收完兩塊中間還留 0.30，不會剩窄條
        _settled(page)
        distance = _outside(page, FRONT)
        reason = page.locator("#elevation-note").inner_text()
        assert f"把框靠右牆那一邊往內收 {distance:.2f} 公尺" in reason, reason
        assert "整塊往對稱軸挪" not in reason, reason
        page.locator("#elevation [data-zone-id=z1] > rect").click()
        width = float(page.locator("#elevation-width").input_value())
        page.locator("#elevation-width").fill(f"{width - distance:.2f}")
        _cut_by(page, "press:Enter")
        _settled(page)
        zones = _drawn(page)
        _matches_mirror(page, FRONT, zones["z1"], zones["z2"])


def test_editing_a_box_whose_mirror_still_does_not_fit_says_so(tmp_path: Path, browser: Browser) -> None:
    """落單、因為翻過去放不下才落單的框：改了還是放不下，提示列照樣講（不是安靜的「切成 N 區」）。"""
    with _room(tmp_path, browser) as page:
        _drag(page, W, H, RIGHT_START, RIGHT_END)
        _settled(page)
        page.locator("#elevation [data-zone-id=z1] > rect").click()
        page.locator("#elevation-height").fill("1.1")
        _cut_by(page, "press:Enter")
        _settled(page)
        expect(page.locator("#elevation-note.notice")).to_contain_text("另一邊沒有跟著畫")
        assert set(_drawn(page)) == {"z1"} and _drawn(page)["z1"]["v_m"] == pytest.approx([0, 1.1])


def _zoom(page: Page) -> None:
    page.set_viewport_size({"width": 1440, "height": 900})
    page.evaluate("document.documentElement.style.zoom = '125%'")
    page.locator("#elevation-svg").evaluate("e => e.scrollIntoView({block:'end'})")


def _banner_in_view(page: Page) -> str:
    banner = page.locator("#elevation-banner")
    expect(banner).to_be_visible()
    assert banner.evaluate("e => { const r=e.getBoundingClientRect(); return r.top>=0 && r.bottom<=innerHeight && r.left>=0 && r.right<=innerWidth }")
    return banner.inner_text()


def _apply_repair(page: Page, *, selected: bool = False) -> None:
    selector = "#elevation-selected" if selected else "#elevation-banner"
    button = page.locator(selector).get_by_role("button", name="照這樣改")
    expect(button).to_be_visible()
    calls = []
    page.on("request", lambda req: calls.append(req.post_data_json) if req.url.endswith("/api/zone-cut") else None)
    with page.expect_response("**/api/zone-cut") as cut:
        button.click()
    assert cut.value.ok, cut.value.text()
    _settled(page)
    assert len(calls) == 1
    assert calls[0]["mirror"]["before"]["id"] == calls[0]["mirror"]["after"]["id"]


def test_repair_banner_stays_visible(tmp_path: Path, browser: Browser) -> None:
    with _room(tmp_path, browser) as page:
        _zoom(page)

        def during() -> None:
            expect(page.locator("#elevation-banner")).to_contain_text("改成")
            assert "超出牆面" in _banner_in_view(page)
            page.screenshot(path=tmp_path / "dragging.png")

        _drag(page, W, H, RIGHT_START, RIGHT_END, during=during)
        _settled(page)
        text = _banner_in_view(page)
        assert "超出牆面" in text and "照這樣改" in text
        page.screenshot(path=tmp_path / "released.png")
        page.locator("#elevation-svg").evaluate("e => window.scrollBy(0, e.getBoundingClientRect().top + 100)")
        assert "照這樣改" in _banner_in_view(page)


def test_repair_narrow_gap(tmp_path: Path, browser: Browser) -> None:
    with _room(tmp_path, browser) as page:
        _zoom(page)
        _drag(page, W, H, (2.5, 0.02), RIGHT_END)
        _settled(page)
        assert "把框靠右牆那一邊往內收 0.30 公尺，兩邊就都放得下" not in page.locator("#elevation-note").inner_text()
        _apply_repair(page)
        boxes = _drawn(page)
        _matches_mirror(page, FRONT, boxes["z1"], boxes["z2"])
        expect(page.locator("#elevation-banner.ok")).to_contain_text("改好了：兩邊都在")
        page.screenshot(path=tmp_path / "pair.png")


def _resize_keeping_speakers(page: Page, width: float) -> None:
    """改房間長度，對稱軸（兩支喇叭的中線）留在原地：改尺寸時喇叭與聆聽點會照比例跟著房間走（#1265），這幾題考的是軸不在正中的牆，
    所以改完把喇叭與聆聽點的 x 都打回原值（只打回喇叭，主位就不在兩支喇叭中間了）。"""
    ids = page.evaluate("[...Object.keys(scheme.speakers).map(n => `speaker-${n}-x`),"
                        " ...scheme.receiver_set.points.map(p => `receiver-${p.receiver_id}-x`)]")
    keep = {field: page.evaluate(f"numericInput('{field}')") for field in ids}
    page.locator("#room-Lx").fill(str(width))
    page.locator("#room-Lx").press("Tab")
    page.evaluate("pendingRoomResize()")
    for field, x in keep.items():
        box = reveal(page, f"#{field}")
        box.fill(str(x))
        box.press("Tab")
    page.wait_for_function("document.getElementById('check-state').dataset.kind !== 'pending'")


def test_repair_fractional_wall(tmp_path: Path, browser: Browser) -> None:
    width = 5.207
    with _room(tmp_path, browser) as page:
        _resize_keeping_speakers(page, width)
        _settled(page)
        _drag(page, width, H, RIGHT_START, RIGHT_END)
        _settled(page)
        _apply_repair(page)
        boxes = _drawn(page)
        _matches_mirror(page, FRONT, boxes["z1"], boxes["z2"])


def test_repair_none_fits(tmp_path: Path, browser: Browser) -> None:
    obstacle = CutBox(id="occupied", u_m=(0, 1.3), v_m=(0, 2.0))
    with _room(tmp_path, browser, (obstacle,)) as page:
        _drag(page, W, H, RIGHT_START, RIGHT_END)
        _settled(page)
        expect(page.locator("#elevation-note")).to_contain_text("試過挪和收都不行")
        expect(page.locator("#elevation-banner")).to_contain_text("壓到")
        expect(page.get_by_role("button", name="照這樣改")).to_have_count(0)


def test_repair_crossing_centres_box(tmp_path: Path, browser: Browser) -> None:
    with _room(tmp_path, browser) as page:
        _drag(page, W, H, (2.0, 0.02), (2.6, 1.8))
        _settled(page)
        expect(page.locator("#elevation-selected")).to_contain_text("這一塊沒有另一邊：跨在對稱軸上但沒有置中")
        _apply_repair(page, selected=True)
        boxes = _drawn(page)
        assert set(boxes) == {"z1"}
        _matches_mirror(page, FRONT, boxes["z1"], boxes["z1"])
        expect(page.locator("#elevation-selected")).to_contain_text("這一塊跨在鏡子正中，自己對稱")  # 第二趟決定第 6 條：叫鏡子


def test_repair_side_message_and_switch(tmp_path: Path, browser: Browser) -> None:
    with _room(tmp_path, browser, wall="x0") as page:
        _zoom(page)
        words = "左右牆不跟著畫：對稱軸離左牆 2.45 公尺，房間正中是 2.60 公尺"

        def during() -> None:
            expect(page.locator("#elevation-banner")).to_contain_text(words)
            page.screenshot(path=tmp_path / "side.png")

        _drag(page, 5.5, H, START, END, during=during)
        _settled(page)
        expect(page.locator("#elevation-banner")).to_contain_text(words)
        expect(page.locator("#elevation-note")).to_contain_text("『房間和擺位』的『座標數字』")
        expect(page.locator("#elevation-note")).not_to_contain_text("補畫另一邊")
        page.locator("#elevation-wall-xL").click()
        expect(page.locator("#elevation-note")).to_be_empty()
        expect(page.locator("#elevation-banner")).to_be_hidden()


def test_repair_back_wall_fields(tmp_path: Path, browser: Browser) -> None:
    with _room(tmp_path, browser, wall="y0") as page:
        _zoom(page)
        _drag(page, W, H, (0.02, 0.02), (1.5, 2.6))
        _settled(page)
        banner = page.locator("#elevation-banner")
        expect(banner).to_contain_text("改成：離左緣 0.30、寬 1.50")
        page.screenshot(path=tmp_path / "back.png")
        page.locator("#elevation [data-zone-id=z1] > rect").click()
        expect(page.locator("#elevation-selected").get_by_role("button", name="照這樣改")).to_be_visible()
        _apply_repair(page)
        assert float(page.locator("#elevation-left").input_value()) == 0.30
        assert float(page.locator("#elevation-width").input_value()) == 1.50
        boxes = _drawn(page, "y0")
        _matches_mirror(page, "y0", boxes["z1"], boxes["z2"])


def test_repair_pair_waits_for_limits(tmp_path: Path, browser: Browser) -> None:
    with _room(tmp_path, browser) as page:
        _drag(page, W, H, RIGHT_START, RIGHT_END)
        _settled(page)
        _drag(page, W, H, (4.75, 1.0), (4.45, 1.0))
        _settled(page)
        expect(page.locator(".elevation-pair-status")).to_contain_text("左右對稱")
        pending = []
        page.route("**/api/zone-limits", lambda route: pending.append(route))
        _drag(page, W, H, (4.45, 1.0), (4.75, 1.0))
        expect(page.locator("#elevation-note")).to_contain_text("超出牆面")
        assert "左右對稱" not in page.locator(".elevation-pair-status").inner_text()
        assert pending
        for route in pending:
            route.continue_()
        page.unroute("**/api/zone-limits")
        _settled(page)
        expect(page.locator(".elevation-pair-status")).to_contain_text("超出牆面")


def test_drawing_preview_is_the_box_that_lands(tmp_path: Path, browser: Browser) -> None:
    """拖著畫的虛線框就是吸過格點與牆邊、放開後要交出去的那一塊；鏡像預覽照同一塊算，眼睛看到的跟落地的一樣。"""
    with _room(tmp_path, browser) as page:
        frame = page.locator("#elevation-frame")
        seen: list[float] = []

        def during() -> None:
            seen.append(float(page.locator("#elevation .zone-preview").get_attribute("x") or "nan"))

        _drag(page, W, H, (0.12, 0.02), (1.0, 1.8), during=during)
        _settled(page)
        assert seen and seen[-1] == pytest.approx(float(frame.get_attribute("x") or "nan"))
        page.locator("#elevation [data-zone-id=z1] > rect").click()
        assert float(page.locator("#elevation-left").input_value()) == 0


def test_banner_goes_when_symmetry_is_switched_off_elsewhere(tmp_path: Path, browser: Browser) -> None:
    """分區頁那邊關掉左右對稱：立面圖的說明條與「照這樣改」一起收掉（按了只會改一邊）。"""
    with _room(tmp_path, browser) as page:
        _drag(page, W, H, (2.0, 0.02), (2.6, 1.8))
        _settled(page)
        page.locator("#elevation [data-zone-id=z1] > rect").click()
        banner = page.locator("#elevation-banner")
        expect(banner.get_by_role("button", name="照這樣改")).to_be_visible()
        page.evaluate("window.zoneSymmetry.set(false)")
        _settled(page)
        expect(banner).to_be_hidden()


def test_repair_leaves_a_full_strip_at_the_wall(tmp_path: Path, browser: Browser) -> None:
    """軸只偏一點、框貼牆角：照多出的量挪或收，牆邊都剩窄條；改法要多挪一個最小邊，不能說「都不行」。"""
    width = 5.0  # 軸離左牆 2.45、正中 2.50：貼右牆角的框翻過去只多出 0.10
    with _room(tmp_path, browser) as page:
        _resize_keeping_speakers(page, width)
        _settled(page)
        _drag(page, width, H, RIGHT_START, (4.99, 2.0))
        _settled(page)
        note = page.locator("#elevation-note")
        expect(note).to_contain_text("多出 0.10 公尺")
        expect(note).not_to_contain_text("試過挪和收都不行")
        expect(page.locator("#elevation-banner")).to_contain_text("（整塊往對稱軸挪 0.30 公尺）")
        _apply_repair(page)
        boxes = _drawn(page)
        _matches_mirror(page, FRONT, boxes["z1"], boxes["z2"])


def test_side_wall_says_which_numbers_to_change(tmp_path: Path, browser: Browser) -> None:
    """左右牆不跟著畫時，說到「喇叭與座位」哪一格加多少；照著加完，對稱軸就在房間正中。"""
    with _room(tmp_path, browser, wall="x0") as page:
        _drag(page, 5.5, H, START, END)
        _settled(page)
        expect(page.locator("#elevation-note")).to_contain_text("把兩支喇叭、主位與周圍點的 x 都加上 0.15，對稱軸就在房間正中")
        page.evaluate("""() => {
          for (const field of document.querySelectorAll('#speakers input[id$="-x"], #receivers input[id$="-x"]')) {
            field.value = (Number(field.value) + 0.15).toFixed(4);
            field.dispatchEvent(new Event("input", {bubbles: true}));
            field.dispatchEvent(new Event("change", {bubbles: true}));
          }
        }""")
        _settled(page)
        expect(page.locator("#elevation-axis-words")).to_contain_text("2.60")
        expect(page.locator("#elevation-axis-words")).not_to_contain_text("不在正中")
