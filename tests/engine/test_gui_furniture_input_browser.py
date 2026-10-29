"""輸入頁家具提示、正式存檔與直達問題；畫面句照施工單，問題答案照決策紙第 12 條。"""
from pathlib import Path

from playwright.sync_api import Browser
from typing import cast

from aosr.reporting.scheme import Scheme
from tests.engine import _furniture_cases as schemes
from tests.engine._gui_cache import gui_startup_identity_memo as gui_startup_identity_memo
from tests.engine.test_gui_browser import _assert_quiet, _assert_text_is_formatted, _open, _serve, browser as browser
from tests.engine.test_gui_furniture_problem_text import assert_chinese_lines
from tests.engine import _speaker_setup_cases as speaker_cases


def test_input_bad_furniture_plan_shows_chinese_problem_lines(browser: Browser, tmp_path: Path) -> None:
    # 有毛病的真方案：左右高度不符、書桌出房間；不攔網路、不替伺服器編答案。
    document = speaker_cases.document("desk", left_z=0.93501, right_z=0.93501)
    document["scheme_id"] = "bad-furniture"
    item = cast(list[dict[str, object]], document["furniture"])[0]
    item["furniture_id"] = "桌板"
    cast(dict[str, object], item["placement"])["forward_m"] = 10
    (tmp_path / "schemes").mkdir()
    (tmp_path / "schemes" / "bad-furniture.json").write_text(Scheme.model_validate(document).model_dump_json())
    with _serve(tmp_path) as url, _open(browser, url) as watched:
        page = watched.page
        page.locator("#scheme-list").select_option("bad-furniture")
        page.locator("#open-scheme").click()
        page.wait_for_function("document.getElementById('messages').textContent.includes('超出房間接觸界線')")
        lines = page.locator("#messages").inner_text().splitlines()
        assert_chinese_lines(lines)
        assert lines == [
            "左聲道喇叭 z 座標、右聲道喇叭 z 座標：高度 0.93501 m 跟擺法推出值不同：桌面頂 0.73 m＋聲學中心離箱底 0.205 m＝0.935 m",
            "第 1 件家具（書桌，桌板）：超出房間接觸界線"]
        _assert_text_is_formatted(page)
        assert all("422" in error for error in watched.console_errors), watched.console_errors
        assert not watched.page_errors
        page.locator("#messages").screenshot(path=str(tmp_path / "furniture-problems.png"))


def test_input_furniture_notice_is_visible_paragraph_and_save_preserves_furniture(
    browser: Browser, tmp_path: Path,
) -> None:
    # 有家具那一份放兩件（座位沙發加天雲）：只放一件的話，件數寫死成 1 也看不出來。
    # 沒家具那一格（提示藏起來、存檔不多出家具）併進下面「先開有家具、再開沒家具」那一題（#972）。
    document = schemes.document(schemes.relative_item(), schemes.cloud_item())
    document["scheme_id"] = "loaded"
    scheme = Scheme.model_validate(document)
    assert scheme.furniture is not None and len(scheme.furniture) > 1
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
        assert notice.is_visible()
        assert notice.evaluate("el => el.hidden") is False
        assert notice.evaluate("el => el.tagName") == "P"
        # 第八支第三步之二已提供編輯器：提示改為固定角色可改、其他原件唯讀保留。
        assert notice.inner_text() == (
            f"這份方案有 {len(scheme.furniture)} 件家具：下方可修改本頁支援的家具；"
            "其餘唯讀列出並原樣保留。平面圖與側面圖照方案檔畫。")
        with page.expect_response(lambda response: response.url.endswith("/api/schemes/loaded")
                                  and response.request.method == "PUT") as saved:
            page.locator("#save").click()
        assert saved.value.ok
        assert Scheme.model_validate_json(path.read_text()).furniture == scheme.furniture
        _assert_quiet(watched)
        page.screenshot(path=str(tmp_path / "input-furniture.png"), full_page=True)
        page.locator("#scheme-section").screenshot(path=str(tmp_path / "input-scheme-section.png"))


def test_input_furniture_notice_is_cleared_when_a_plain_scheme_is_opened_next(browser: Browser, tmp_path: Path) -> None:
    (tmp_path / "schemes").mkdir()
    for name, document in (("furnished", schemes.document(schemes.relative_item(), schemes.cloud_item())),
                           ("plain", schemes.document())):
        (tmp_path / "schemes" / f"{name}.json").write_text(
            Scheme.model_validate(document | {"scheme_id": name}).model_dump_json())
    with _serve(tmp_path) as url, _open(browser, url) as watched:
        page = watched.page
        notice = page.locator("#furniture-notice")
        assert not notice.is_visible()
        for name, shown in (("furnished", True), ("plain", False)):
            page.locator("#scheme-list").select_option(name)
            page.locator("#open-scheme").click()
            page.wait_for_function(f"document.getElementById('save-id').value === '{name}'")
            assert notice.is_visible() is shown and notice.evaluate("el => el.hidden") is not shown
            assert notice.evaluate("el => el.tagName") == "P"
            assert ("這份方案有 2 件家具" in notice.inner_text()) is shown
        # 空的段落高度是 0，is_visible 本來就回 False；沒家具時要真的藏起來（hidden），畫面不多一行空白。
        assert notice.inner_text() == ""
        # 沒家具那一份存回去，家具照原樣是沒有：前一份的家具不准被帶進來（#972 由提示題的沒家具那一格併入）。
        plain = tmp_path / "schemes" / "plain.json"
        before = Scheme.model_validate_json(plain.read_text())
        with page.expect_response(lambda response: response.url.endswith("/api/schemes/plain")
                                  and response.request.method == "PUT") as saved:
            page.locator("#save").click()
        assert saved.value.ok
        assert before.furniture is None
        assert Scheme.model_validate_json(plain.read_text()).furniture == before.furniture
        _assert_quiet(watched)
