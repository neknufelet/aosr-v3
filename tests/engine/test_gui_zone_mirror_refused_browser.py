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
            assert f"把框靠右牆那一邊往內收 {distance}" in reason and f"或整塊往對稱軸挪 {distance}" in reason
            reasons.append(reason)
            page.screenshot(path=tmp_path / "outside-dragging.png")

        with page.expect_response("**/api/zone-cut") as released:
            _drag(page, W, H, start, end, during=during)
        _settled(page)
        assert reasons[0] in released.value.json()["message"]
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
        expect(page.locator("#elevation-selected")).to_contain_text("這一塊翻過去放不下：" + reasons[0])
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
