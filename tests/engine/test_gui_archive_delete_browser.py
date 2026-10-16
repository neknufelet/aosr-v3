"""永久刪除的確認、取消、清單更新與窄畫面。"""
from __future__ import annotations

from pathlib import Path

import pytest
from playwright.sync_api import Browser, Dialog, expect

from aosr.reporting.result import SchemeResult
from tests.engine._gui_cache import gui_load_result_memo, gui_startup_identity_memo
from tests.engine.test_gui_archive import _bundle, _saved, _tree
from tests.engine.test_gui_archive_browser import _assert_text_on_one_line, _confirm, _package_row, _row
from tests.engine.test_gui_archive_delete import _damage
from tests.engine.test_gui_browser import _assert_quiet, _open, _serve, browser, result

WARNING = "這個資料夾沒有備份，刪了救不回來"


def test_package_delete_cancel_then_accept(tmp_path: Path, browser: Browser, result: SchemeResult) -> None:
    _saved(tmp_path, result)
    ids = ["a" * 32, "b" * 32]
    for run_id in ids:
        _bundle(tmp_path, result, run_id)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        package = page.request.post(f"{base}/api/schemes/{result.scheme.scheme_id}/archive", data={}).json()
        page.reload(wait_until="networkidle")
        page.locator("#archived summary").click()
        row = _package_row(page, result.scheme.scheme_id)
        button = row.get_by_role("button", name="永久刪除", exact=True)
        expect(button).to_be_visible()
        messages: list[str] = []
        requests: list[str] = []
        page.on("request", lambda request: requests.append(request.url) if request.url.endswith("/delete") else None)

        def cancel(dialog: Dialog) -> None:
            messages.append(dialog.message)
            dialog.dismiss()

        page.once("dialog", cancel)
        button.click()
        expect(row).to_be_visible()
        assert requests == []
        folder = tmp_path / "archive" / "packages" / package["package_id"]
        assert folder.is_dir()
        assert result.scheme.scheme_id in messages[0] and f"{len(ids)} 筆結果" in messages[0]
        assert WARNING in messages[0] and package["archived_text"] in messages[0]
        _confirm(page)
        button.click()
        expect(row).not_to_be_attached()
        expect(page.locator("#archived")).to_be_hidden()
        expect(page.locator("#archived summary")).to_have_text("已封存（0 包方案、0 筆舊的單筆封存）")
        assert not folder.exists()
        _assert_quiet(watched)


@pytest.mark.parametrize("width", [1400, 390])
def test_clear_keeps_skipped_packages_and_tables_fit(
        tmp_path: Path, browser: Browser, result: SchemeResult, width: int) -> None:
    _bundle(tmp_path / "archive", result, "a" * 32)
    _saved(tmp_path, result)
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=width) as watched:
        page = watched.page
        complete = page.request.post(f"{base}/api/schemes/{result.scheme.scheme_id}/archive", data={}).json()
        _saved(tmp_path, result)
        incomplete = page.request.post(f"{base}/api/schemes/{result.scheme.scheme_id}/archive", data={}).json()
        parent = tmp_path / "archive" / "packages"
        _damage(parent / incomplete["package_id"], "incomplete")
        broken = parent / ("f" * 32)
        broken.mkdir()
        (broken / "manifest.json").write_text("{")
        page.reload(wait_until="networkidle")
        page.locator("#archived summary").click()
        for package_id in (incomplete["package_id"], broken.name):
            row = page.locator(f'#archived-packages-list tr[data-package-id="{package_id}"]')
            expect(row.get_by_role("button", name="永久刪除")).not_to_be_attached()
        for selector in ("#archived-packages", "#legacy-archive table"):
            assert page.locator(selector).evaluate("""table => {
              const card = table.closest('section');
              return table.getBoundingClientRect().right <= card.getBoundingClientRect().right
                && card.scrollWidth <= card.clientWidth;
            }""")
        if width == 390:
            expect(page.locator(".archive-scroll-note")).to_be_visible()
            for button in page.locator("#archived button.archive-danger").all():
                button.scroll_into_view_if_needed()
                assert button.evaluate("""button => {
                  const box = button.getBoundingClientRect(), card = button.closest('section').getBoundingClientRect();
                  return box.left >= card.left && box.right <= card.right;
                }""")
        for date in page.locator("#archived .finished-date").all():
            _assert_text_on_one_line(date)
        messages = _confirm(page)
        page.get_by_role("button", name="全部清除封存", exact=True).click()
        expect(page.locator(f'tr[data-package-id="{complete["package_id"]}"]')).not_to_be_attached()
        expect(_row(page, "a" * 32, True)).not_to_be_attached()
        expect(page.locator("#messages")).to_have_class("notice")
        expect(page.locator("#messages")).to_contain_text("跳過")
        assert WARNING in messages[0] and "1 包方案（共 0 筆結果）和 1 筆舊的單筆封存" in messages[0]
        assert "另有 2 包" in messages[0]
        expect(page.get_by_role("button", name="全部清除封存", exact=True)).to_be_hidden()
        _assert_quiet(watched)


def test_legacy_confirmation_and_last_row_hides_archive(tmp_path: Path, browser: Browser, result: SchemeResult) -> None:
    run_id = "a" * 32
    _bundle(tmp_path / "archive", result, run_id)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        page.locator("#archived summary").click()
        messages = _confirm(page)
        _row(page, run_id, True).get_by_role("button", name="永久刪除", exact=True).click()
        expect(page.locator("#archived")).to_be_hidden()
        assert WARNING in messages[0] and result.scheme.scheme_id in messages[0] and "算完" in messages[0]
        _assert_quiet(watched)


def test_clear_cancel_then_accept_hides_empty_archive(
        tmp_path: Path, browser: Browser, result: SchemeResult) -> None:
    _bundle(tmp_path / "archive", result, "a" * 32)
    _saved(tmp_path, result)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        page.request.post(f"{base}/api/schemes/{result.scheme.scheme_id}/archive", data={})
        page.reload(wait_until="networkidle")
        page.locator("#archived summary").click()
        before = _tree(tmp_path)
        sent: list[str] = []
        messages: list[str] = []
        page.on("request", lambda request: sent.append(request.url)
                if request.method == "POST" and request.url.endswith(("/delete", "/clear")) else None)

        def cancel(dialog: Dialog) -> None:
            messages.append(dialog.message)
            dialog.dismiss()

        page.once("dialog", cancel)
        _row(page, "a" * 32, True).get_by_role("button", name="永久刪除").click()
        page.once("dialog", cancel)
        page.get_by_role("button", name="全部清除封存", exact=True).click()
        assert sent == [] and _tree(tmp_path) == before
        assert all(WARNING in message for message in messages)
        expect(page.locator("#archived")).to_be_visible()
        _confirm(page)
        page.get_by_role("button", name="全部清除封存", exact=True).click()
        expect(page.locator("#archived")).to_be_hidden()
        listing = page.request.get(f"{base}/api/archive").json()
        assert listing["packages"] == [] and listing["results"] == []
        _assert_quiet(watched)
