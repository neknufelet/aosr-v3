"""房間方案輸入的頂條、選單和頁底。方案、結果與計算行程都是考卷的臨時資料。"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from playwright.sync_api import Browser, expect

from aosr.reporting.result import SchemeResult
from tests.engine._input_page_steps import archive_scheme, close_results, new_from_template, open_example_scheme, open_save_as, open_scheme, reveal, scheme_open_button
from tests.engine.test_scheme_pipeline import _run_control
from tests.engine.test_gui_archive import _bundle, _saved
from tests.engine.test_gui_browser import _assert_quiet, _copy_runner, _open, _serve, browser, result

ROOM = "studio-west"


def _studio(result: SchemeResult) -> SchemeResult:
    return _run_control(result.scheme.model_copy(update={"scheme_id": ROOM}))


def _one_yellow_zone(root: Path) -> None:
    folder = root / "designs"
    folder.mkdir(exist_ok=True)
    (folder / f"{ROOM}.json").write_text(json.dumps({
        "schema_version": "aosr.zone_design.v1", "scheme_id": ROOM,
        "zones": {"floor": {"whole_wall": {"mode": "search", "allowed": ["none", "glass_wool_48k"]}}}}))


def test_saved_idle_and_sticky_results_jump(tmp_path: Path, browser: Browser, result: SchemeResult) -> None:
    _saved(tmp_path, _studio(result))
    _one_yellow_zone(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        page.set_viewport_size({"width": 1440, "height": 900})
        open_scheme(page, ROOM)
        expect(page.locator("#save-id")).to_have_value(ROOM)
        expect(page.locator("#save-state")).to_have_text("都存好了（還沒算過）")
        expect(page.locator("#next-title")).to_have_text("改好了就選一個往下走")
        expect(page.locator("#zone-to-search-note")).to_contain_text("交出橘色的 1 區")
        page.locator("#save").click()
        expect(page.locator("#save-feedback")).to_have_text("方案沒有變動")
        page.locator("#show-results").click()
        expect(page.locator("#results-title")).to_be_in_viewport()
        close_results(page)
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        expect(page.locator("#scheme-menu-button")).to_be_in_viewport()
        expect(page.locator("#calculate")).to_be_in_viewport()
        _assert_quiet(watched)


def test_dirty_copy_cancel_and_save(tmp_path: Path, browser: Browser, result: SchemeResult) -> None:
    path = _saved(tmp_path, _studio(result))
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        open_example_scheme(page, base, tmp_path)
        open_scheme(page, ROOM)
        expect(page.locator("#save-id")).to_have_value(ROOM)
        page.locator("#room-Lz").fill("3.7")
        expect(page.locator("#save-state")).to_have_text("有改動還沒存：房間高度。按右邊「儲存」")
        open_save_as(page)
        page.locator("#save-as-id").fill("無效名字")
        expect(page.locator("#save-as-note")).to_contain_text("中文")
        page.locator("#cancel-save-as").click()
        expect(page.locator("#save-as-id")).to_be_hidden()
        page.locator("#save").click()
        expect(page.locator("#save-state")).to_have_text("都存好了（還沒算過）")
        assert json.loads(path.read_text())["scene"]["room_m"]["Lz"] == 3.7
        _assert_quiet(watched)


def test_locked_then_edited_requires_a_copy(tmp_path: Path, browser: Browser, result: SchemeResult) -> None:
    room = _studio(result)
    _saved(tmp_path, room)
    _bundle(tmp_path, room, "1" * 32)
    _one_yellow_zone(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        open_scheme(page, ROOM)
        expect(page.locator("#save-id")).to_have_value(ROOM)
        expect(page.locator("#save-state")).to_have_text("算過了，內容鎖住留著比較")
        expect(page.locator("#lock-bar")).to_contain_text("之後拿來跟別份比")
        expect(page.locator("#save")).to_be_hidden()
        expect(page.locator("#calculate")).to_be_enabled()
        expect(page.locator("#zone-to-search")).to_be_enabled()
        reveal(page, "#speaker-left-x").fill("1.62")
        expect(page.locator("#calculate")).to_be_disabled()
        expect(page.locator("#zone-to-search")).to_be_disabled()
        expect(page.locator("#next-note")).to_have_text("改了以後先按右上「另存新名字」存成新的一份，才能再算")
        _assert_quiet(watched)


def test_menu_keyboard_archive_and_template(tmp_path: Path, browser: Browser, result: SchemeResult) -> None:
    _saved(tmp_path, _studio(result))
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        trigger = page.locator("#scheme-menu-button")
        trigger.press("ArrowDown")
        row = page.get_by_role("menuitem", name=ROOM, exact=True)
        expect(row).to_be_focused()
        expect(row).to_contain_text("存的，還沒算過")
        page.keyboard.press("ArrowDown")
        expect(page.get_by_role("menuitem", name=f"封存 {ROOM}")).to_be_focused()
        page.keyboard.press("Escape")
        expect(trigger).to_be_focused()
        expect(page.get_by_role("menu")).to_be_hidden()
        open_scheme(page, ROOM)
        expect(page.locator("#save-id")).to_have_value(ROOM)
        page.on("dialog", lambda dialog: dialog.accept())
        archive_scheme(page, ROOM)
        expect(page.locator("#scheme-archive-notice")).to_be_visible()
        trigger.click()
        page.locator("#show-archived").click()
        expect(page.locator("#archived-packages-list")).to_contain_text(ROOM)
        new_from_template(page, "stage-copy", "stage_one")
        expect(page.locator("#open-note")).to_have_text("已從範本「第一階段範本」開一份新的「stage-copy」，存好了")
        expect(page.locator("#save-state")).to_have_text("都存好了（還沒算過）")
        expect(page.locator("#scheme-archive-notice")).to_be_hidden()
        _assert_quiet(watched)


def test_running_pill_and_stop(tmp_path: Path, browser: Browser, result: SchemeResult) -> None:
    room = _studio(result)
    _saved(tmp_path, room)
    with _serve(tmp_path, _copy_runner(tmp_path, room, wait=True)) as base, _open(browser, base) as watched:
        page = watched.page
        open_scheme(page, ROOM)
        expect(page.locator("#save-id")).to_have_value(ROOM)
        page.locator("#calculate").click()
        expect(page.locator("#save-state")).to_have_text("正在算，這一份已鎖住")
        expect(page.locator("#lock-bar")).to_contain_text("這一份正在算")
        expect(page.locator("#next-title")).to_contain_text("正在算這一份：已跑")
        expect(page.locator("#calculate")).to_be_hidden()
        expect(page.locator("#zone-to-search")).to_be_disabled()
        page.locator("#stop").click()
        expect(page.locator("#next-title")).to_have_text("沒算完：已停止")
        _assert_quiet(watched)


def test_done_opens_this_result(tmp_path: Path, browser: Browser, result: SchemeResult) -> None:
    room = _studio(result)
    _saved(tmp_path, room)
    with _serve(tmp_path, _copy_runner(tmp_path, room)) as base, _open(browser, base) as watched:
        page = watched.page
        open_scheme(page, ROOM)
        expect(page.locator("#save-id")).to_have_value(ROOM)
        page.locator("#calculate").click()
        link = page.get_by_role("link", name="算好了 ▸ 看結果")
        expect(link).to_be_visible()
        expect(page.locator("#save-state")).to_have_text("算過了，內容鎖住留著比較")
        href = link.get_attribute("href")
        assert href and href.startswith("/results/")
        link.click()
        page.wait_for_url(base + href)
        expect(page.locator("#page-title")).to_contain_text(ROOM)


def test_failed_shows_reason_log_and_retry(tmp_path: Path, browser: Browser, result: SchemeResult) -> None:
    _saved(tmp_path, _studio(result))
    runner = (sys.executable, "-c", "import sys; sys.exit('測試房的計算被中斷')")
    with _serve(tmp_path, runner) as base, _open(browser, base) as watched:
        page = watched.page
        open_scheme(page, ROOM)
        expect(page.locator("#save-id")).to_have_value(ROOM)
        page.locator("#calculate").click()
        expect(page.locator("#next-title")).to_have_text("沒算完：測試房的計算被中斷")
        expect(page.locator("#save-state")).to_have_text("都存好了（還沒算過）")
        page.get_by_text("看出錯紀錄", exact=True).click()
        expect(page.locator("#run-log-text")).to_have_text("測試房的計算被中斷")
        expect(page.locator("#save")).to_be_visible()
        with page.expect_response(lambda r: r.url.endswith("/api/runs") and r.request.method == "POST") as retry:
            page.get_by_role("button", name="再算一次").click()
        assert retry.value.ok
        _assert_quiet(watched)


def test_switching_dirty_scheme_can_cancel_discard_or_save(tmp_path: Path, browser: Browser, result: SchemeResult) -> None:
    first = _saved(tmp_path, _studio(result))
    _saved(tmp_path, result)
    other = result.scheme.scheme_id
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        open_scheme(page, ROOM)
        expect(page.locator("#save-id")).to_have_value(ROOM)
        page.locator("#room-Lz").fill("3.6")
        original = first.read_bytes()
        scheme_open_button(page, other).click()
        page.get_by_role("dialog").get_by_text("取消").click()
        expect(page.locator("#room-Lz")).to_have_value("3.6")
        scheme_open_button(page, other).click()
        page.get_by_role("dialog").get_by_text("不存走").click()
        expect(page.locator("#save-id")).to_have_value(other)
        assert first.read_bytes() == original
        open_scheme(page, ROOM)
        expect(page.locator("#save-id")).to_have_value(ROOM)
        page.locator("#room-Lz").fill("3.8")
        scheme_open_button(page, other).click()
        page.get_by_role("dialog").get_by_text("存了再走").click()
        expect(page.locator("#save-id")).to_have_value(other)
        assert json.loads(first.read_text())["scene"]["room_m"]["Lz"] == 3.8
        _assert_quiet(watched)


def test_menu_lists_newest_five_then_opens_older_scheme(tmp_path: Path, browser: Browser, result: SchemeResult) -> None:
    names = [f"studio-{number}" for number in range(6)]
    for index, name in enumerate(names):
        room = result.model_copy(update={"scheme": result.scheme.model_copy(update={"scheme_id": name})})
        saved = _saved(tmp_path, room)
        when = 1700000000 + index
        os.utime(saved, (when, when))
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        page.locator("#scheme-menu-button").click()
        visible = page.locator("#home-schemes .home-open:visible b")
        expect(visible).to_have_text(names[:0:-1])
        page.get_by_role("menuitem", name="看全部方案（6 份）").click()
        expect(visible).to_have_text(list(reversed(names)))
        page.get_by_role("menuitem", name=names[0], exact=True).click()
        expect(page.locator("#save-id")).to_have_value(names[0])
        _assert_quiet(watched)
