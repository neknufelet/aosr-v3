"""#1265：改房間長寬高，喇叭與聆聽點跟著房間走（老闆 11-26 把參考房改成 2.5 × 4 × 3，主位馬上跑到房外：「一看房間就跑出去」）。

長、寬照比例挪（原本在房裡的改完還在房裡、相對位置不變）；高度不照比例（耳朵與喇叭高度跟天花板無關），新高度放不下時才照比例降；
主位周圍那幾個點跟著主位平移、間距不縮。比的是上一次收下的尺寸：中間打空白或打錯不算。數字都是預設參考房的（考卷自己的範本）。"""
from __future__ import annotations

from pathlib import Path

from playwright.sync_api import Browser, Page, expect

from tests.engine._input_page_steps import open_example_scheme, reveal
from tests.engine.test_gui_browser import _assert_quiet, _open, _serve, browser as browser

# 預設參考房：6 × 4 × 3；喇叭 (1.0, 1.4／2.6, 1.2)；主位 (3.2, 2.0, 1.2)，周圍點離主位 0.1。
SHRINK = (("room-Lx", "2.5"), ("room-Ly", "3"))


def _commit(page: Page, field: str, value: str) -> None:
    box = reveal(page, f"#{field}")
    box.fill(value)
    box.blur()


def _value(page: Page, field: str) -> float:
    return float(reveal(page, f"#{field}").input_value())


def test_shrinking_the_room_carries_speakers_and_listeners_along_and_the_check_still_passes(tmp_path: Path,
                                                                                         browser: Browser) -> None:
    with _serve(tmp_path) as base, _open(browser, "about:blank", viewport_width=1440) as watched:
        page = watched.page
        page.set_viewport_size({"width": 1440, "height": 900})
        open_example_scheme(page, base, tmp_path)
        check = page.locator("#check-state")
        expect(check).to_have_text("檢查通過")
        for field, value in SHRINK:
            _commit(page, field, value)
        expect(check).to_have_text("檢查通過")
        assert _value(page, "speaker-left-x") == 0.417 and _value(page, "speaker-right-x") == 0.417
        assert _value(page, "speaker-left-y") == 1.05 and _value(page, "speaker-right-y") == 1.95
        assert _value(page, "receiver-main-x") == 1.333 and _value(page, "receiver-main-y") == 1.5
        # 周圍點跟著主位平移：前後左右仍離主位 0.1。
        assert _value(page, "receiver-front-x") == 1.233 and _value(page, "receiver-back-x") == 1.433
        assert _value(page, "receiver-left-y") == 1.4 and _value(page, "receiver-right-y") == 1.6
        # 高度沒改：z 全部照舊。
        assert _value(page, "speaker-left-z") == 1.2 and _value(page, "receiver-up-z") == 1.3
        page.wait_for_function("shownPlan && shownPlan.room.Lx === 2.5 && shownPlan.room.Ly === 3")
        assert page.evaluate("shownPlan.outside_keys ?? []") == []
        _assert_quiet(watched)


def test_a_low_ceiling_brings_heights_down_only_when_they_no_longer_fit(tmp_path: Path, browser: Browser) -> None:
    with _serve(tmp_path) as base, _open(browser, "about:blank", viewport_width=1440) as watched:
        page = watched.page
        page.set_viewport_size({"width": 1440, "height": 900})
        open_example_scheme(page, base, tmp_path)
        check = page.locator("#check-state")
        _commit(page, "room-Lz", "2.4")
        expect(check).to_have_text("檢查通過")
        assert _value(page, "speaker-left-z") == 1.2 and _value(page, "receiver-main-z") == 1.2
        _commit(page, "room-Lz", "1")
        expect(check).to_have_text("檢查通過")
        # 1.2 放不下 1 公尺高：照比例（1 / 2.4）降；主位上下兩點跟著主位平移、間距 0.1 不變。
        assert _value(page, "speaker-left-z") == 0.5 and _value(page, "receiver-main-z") == 0.5
        assert _value(page, "receiver-up-z") == 0.6 and _value(page, "receiver-down-z") == 0.4
        _assert_quiet(watched)


def test_a_blank_or_bad_size_in_between_does_not_count_and_growing_back_restores_the_layout(tmp_path: Path,
                                                                                          browser: Browser) -> None:
    with _serve(tmp_path) as base, _open(browser, "about:blank", viewport_width=1440) as watched:
        page = watched.page
        page.set_viewport_size({"width": 1440, "height": 900})
        open_example_scheme(page, base, tmp_path)
        check = page.locator("#check-state")
        _commit(page, "room-Lx", "")
        expect(check).not_to_have_text("檢查通過")
        assert _value(page, "receiver-main-x") == 3.2
        _commit(page, "room-Lx", "3")
        expect(check).to_have_text("檢查通過")
        assert _value(page, "receiver-main-x") == 1.6 and _value(page, "speaker-left-x") == 0.5
        _commit(page, "room-Lx", "6")
        expect(check).to_have_text("檢查通過")
        assert _value(page, "receiver-main-x") == 3.2 and _value(page, "speaker-left-x") == 1.0
        assert _value(page, "receiver-front-x") == 3.1


def test_a_speaker_already_outside_stays_put_while_the_room_grows_to_fit_it(tmp_path: Path, browser: Browser) -> None:
    """先把左喇叭搬到房外（x 7.2）、再把房間加長到 7.5 裝得下它：左喇叭不動、檢查通過；房裡的照比例挪。"""
    with _serve(tmp_path) as base, _open(browser, "about:blank", viewport_width=1440) as watched:
        page = watched.page
        page.set_viewport_size({"width": 1440, "height": 900})
        open_example_scheme(page, base, tmp_path)
        check = page.locator("#check-state")
        _commit(page, "speaker-left-x", "7.2")
        expect(check).not_to_have_text("檢查通過")
        _commit(page, "room-Lx", "7.5")
        expect(check).to_have_text("檢查通過")
        assert _value(page, "speaker-left-x") == 7.2 and _value(page, "speaker-right-x") == 1.25
        assert _value(page, "receiver-main-x") == 4.0
