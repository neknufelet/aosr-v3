"""輸入頁家具提示、正式存檔與直達問題；畫面句照施工單，問題答案照決策紙第 12 條。"""
from pathlib import Path

import pytest
from playwright.sync_api import Browser

from aosr.reporting.scheme import Scheme
from tests.engine import _furniture_cases as schemes
from tests.engine._gui_cache import gui_startup_identity_memo as gui_startup_identity_memo
from tests.engine.test_gui_browser import _assert_quiet, _open, _serve, browser as browser
from tests.engine.test_scheme_furniture import _validation_document


@pytest.mark.parametrize("furnished", [False, True])
def test_input_furniture_notice_is_visible_paragraph_and_save_preserves_furniture(
    browser: Browser, tmp_path: Path, furnished: bool,
) -> None:
    # 有家具那一份放兩件（座位沙發加天雲）：只放一件的話，件數寫死成 1 也看不出來。
    document = schemes.document(schemes.relative_item(), schemes.cloud_item()) if furnished else schemes.document()
    document["scheme_id"] = "loaded"
    scheme = Scheme.model_validate(document)
    assert len(scheme.furniture or ()) == (2 if furnished else 0)
    (tmp_path / "schemes").mkdir()
    path = tmp_path / "schemes" / "loaded.json"
    path.write_text(scheme.model_dump_json())
    with _serve(tmp_path) as url, _open(browser, url) as watched:
        page = watched.page
        assert not page.locator("#furniture-notice").is_visible()
        page.locator("#scheme-list").select_option("loaded")
        page.locator("#open-scheme").click()
        page.wait_for_function("document.getElementById('save-id').value === 'loaded'")
        notice = page.locator("#furniture-notice")
        assert notice.is_visible() == furnished
        # 空的段落高度是 0，is_visible 本來就回 False；沒家具時要真的藏起來（hidden），畫面不多一行空白。
        assert notice.evaluate("el => el.hidden") is not furnished
        assert notice.evaluate("el => el.tagName") == "P"
        if furnished:
            assert scheme.furniture is not None
            assert notice.inner_text() == (
                f"這份方案有 {len(scheme.furniture)} 件家具：這一頁還不能顯示或修改家具，平面圖也還沒畫出家具；"
                "存檔與計算照方案檔裡的家具算。")
        else:
            assert notice.inner_text() == ""
        with page.expect_response(lambda response: response.url.endswith("/api/schemes/loaded")
                                  and response.request.method == "PUT") as saved:
            page.locator("#save").click()
        assert saved.value.ok
        assert Scheme.model_validate_json(path.read_text()).furniture == scheme.furniture
        _assert_quiet(watched)
        page.screenshot(path=str(tmp_path / f"input-{'furniture' if furnished else 'plain'}.png"), full_page=True)
        page.locator("#scheme-section").screenshot(path=str(tmp_path / "input-scheme-section.png"))


def test_input_blocked_direct_paths_are_separate_pair_lines(browser: Browser, tmp_path: Path) -> None:
    scheme = Scheme.model_validate(_validation_document("both") | {"scheme_id": "blocked"})
    (tmp_path / "schemes").mkdir()
    (tmp_path / "schemes" / "blocked.json").write_text(scheme.model_dump_json())
    with _serve(tmp_path) as url, _open(browser, url) as watched:
        page = watched.page
        page.locator("#scheme-list").select_option("blocked")
        page.locator("#open-scheme").click()
        page.wait_for_function("document.getElementById('save-id').value === 'blocked'")
        page.locator("#check").click()
        page.wait_for_function("document.getElementById('messages').textContent.includes('直達路徑被家具')")
        assert page.locator("#messages").inner_text().splitlines() == [
            "左聲道喇叭 → 主位：不符合擺位要求：直達路徑被家具 desk 擋住",
            "左聲道喇叭 → 座位 side：不符合擺位要求：直達路徑被家具 desk 擋住",
        ]
        assert "沒有聲音" not in page.locator("#messages").inner_text()
        # 平面圖驗證照設計拒收為 422；只准這一種網路錯，頁面程式不能拋例外。
        assert all("422" in error for error in watched.console_errors), watched.console_errors
        assert not watched.page_errors
        page.screenshot(path=str(tmp_path / "input-blocked.png"), full_page=True)
        page.locator("#messages").screenshot(path=str(tmp_path / "input-blocked-problems.png"))


def test_input_furniture_notice_is_cleared_when_a_plain_scheme_is_opened_next(browser: Browser, tmp_path: Path) -> None:
    (tmp_path / "schemes").mkdir()
    for name, document in (("furnished", schemes.document(schemes.relative_item(), schemes.cloud_item())),
                           ("plain", schemes.document())):
        (tmp_path / "schemes" / f"{name}.json").write_text(
            Scheme.model_validate(document | {"scheme_id": name}).model_dump_json())
    with _serve(tmp_path) as url, _open(browser, url) as watched:
        page = watched.page
        notice = page.locator("#furniture-notice")
        for name, shown in (("furnished", True), ("plain", False)):
            page.locator("#scheme-list").select_option(name)
            page.locator("#open-scheme").click()
            page.wait_for_function(f"document.getElementById('save-id').value === '{name}'")
            assert notice.is_visible() is shown and notice.evaluate("el => el.hidden") is not shown
            assert ("這份方案有 2 件家具" in notice.inner_text()) is shown
        assert notice.inner_text() == ""
        _assert_quiet(watched)
