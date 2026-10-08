"""真家具結果／比較頁與 PNG，連同無家具控制組截圖；答案用決策紙原文。"""
from __future__ import annotations

from pathlib import Path

import pytest
from playwright.sync_api import Browser

from aosr.reporting.result import SchemeResult, save_result
from tests.engine._furniture_scheme_results import scheme_pair as scheme_pair
from tests.engine._gui_cache import (
    gui_load_result_memo as gui_load_result_memo, gui_startup_identity_memo as gui_startup_identity_memo,
)
from tests.engine.test_gui_browser import (
    RUN_ID, _assert_quiet, _assert_text_is_formatted, _open, _save, _serve, browser as browser,
)
from tests.engine.test_gui_compare_browser import _has_lines
from tests.engine.test_gui_compare_view import _renamed
from tests.engine.test_scheme_pipeline import shared_control_result


REASON = ("已含家具一次反射、遮擋與有限尺寸鏡面修正；未含家具與牆之間的多次反射、"
          "完整繞射，以及家具吸音對整房殘響的影響")
FLUTTER = "顫動警戒第一版只看三對牆，家具形成的平行面未評估"


@pytest.mark.parametrize("furnished", [False, True])
def test_real_furniture_result_page_and_plain_control(
    browser: Browser, tmp_path: Path, scheme_pair: tuple[SchemeResult, ...], furnished: bool,
) -> None:
    _save(tmp_path, scheme_pair[1 if furnished else 0])
    with _serve(tmp_path) as url, _open(browser, f"{url}/results/{RUN_ID}") as watched:
        page = watched.page
        page.wait_for_selector("#content", state="visible")
        _has_lines(page)
        text = page.locator("body").inner_text()
        if furnished:
            assert "原本牆面覆蓋條件成立；家具僅一次反射、混合反射未納入" in text
            assert "家具反射只驗證公式實作一致，實際家具精度未驗證" in text
            page.locator("#reflections details").evaluate_all("items => items.forEach(item => item.open = true)")
            assert "沙發（seat）＋頂面" in page.locator("#reflections").inner_text()
            assert "沙發（seat）" in page.locator("#furniture").inner_text()
            assert "布面；估計，非本件實測" in text
            assert "未知（計算時用相鄰頻帶延伸代算）：63、8000 Hz" in text
            before, after = text.split(REASON)
            assert REASON not in before + after
            assert {element.inner_text() for element in page.locator("#furniture-notes p").all()} == {
                "透射未算", "喇叭指向性往下的方向尚未獨立驗證，桌面反射強度靠這個假設",
                "遮擋邊界上的反射會突然出現或消失"}
            assert FLUTTER in page.locator("#alerts > p").all_inner_texts()
            assert FLUTTER not in page.locator("#alerts h3").all_inner_texts()
            assert "家具模型：近似" in page.locator("#ranking-approximation").inner_text()
            assert page.locator("#ranking-approximation").evaluate("el => el.tagName") == "P"
            assert page.locator("#frequency-furniture-note a").get_attribute("href") == "#furniture"
            assert "未包含家具吸音" in page.locator("#reverb-note").inner_text()
            assert "未包含家具吸音" in page.locator("#categories").inner_text()
        else:
            assert not page.locator("#furniture").is_visible()
            assert REASON not in text and FLUTTER not in text
        _assert_text_is_formatted(page)
        _assert_quiet(watched)
        page.screenshot(path=str(tmp_path / f"result-{'furniture' if furnished else 'plain'}.png"), full_page=True)


@pytest.mark.parametrize("furnished", [False, True])
def test_real_furniture_compare_page_exports_and_plain_control(
    browser: Browser, tmp_path: Path, scheme_pair: tuple[SchemeResult, ...], furnished: bool,
    tmp_path_factory: pytest.TempPathFactory, worker_id: str,
) -> None:
    results = scheme_pair if furnished else tuple(shared_control_result(tmp_path_factory, worker_id, name)
                                                   for name in ("wall-1", "wall-2"))
    (tmp_path / "results").mkdir()
    for letter, result in zip(("a", "b"), results, strict=True):
        save_result(result, tmp_path / "results" / f"{letter * 32}.json")
    with _serve(tmp_path) as url, _open(browser, f"{url}/compare/{'a' * 32}/{'b' * 32}") as watched:
        page = watched.page
        page.wait_for_selector("#content", state="visible")
        _has_lines(page)
        if furnished:
            assert "兩份方案設定相同" not in page.locator("#summary-text").inner_text()
            assert "沙發（seat）" in page.locator("#changes").inner_text()
            assert "兩者計算涵蓋範圍不同" in page.locator("#categories").inner_text()
            assert REASON in page.locator("#notes p").all_inner_texts()
            before, after = page.locator("body").inner_text().split(REASON)
            assert REASON not in before + after
            for row in page.locator("#categories tr").all()[1:]:
                cells = row.locator("td").all_inner_texts()
                assert "近似" not in cells[1]
                assert ("近似" in cells[3]) == (cells[0] in {"音色平衡", "聆聽區穩定性", "反射與回聲", "聲道匹配"})
            assert page.locator("#overlay-furniture-note").inner_text() == "家具模型：近似"
            page.evaluate("""() => { window.paintedText = []; const original = CanvasRenderingContext2D.prototype.fillText;
                CanvasRenderingContext2D.prototype.fillText = function(text, ...args) {
                    window.paintedText.push(String(text)); return original.call(this, text, ...args); }; }""")
            with page.expect_download() as event:
                page.locator("#download-png").click()
            event.value.save_as(tmp_path / "compare-furniture-export.png")
            assert "家具模型：近似" in page.evaluate("window.paintedText.join('')")
            response = page.request.get(f"{url}/api/compare/{'a' * 32}/{'b' * 32}/export/curves")
            assert "家具模型：近似" in response.text()
        else:
            assert not page.locator("#overlay-furniture-note").is_visible()
            assert REASON not in page.locator("body").inner_text()
        _assert_text_is_formatted(page)
        _assert_quiet(watched)
        page.screenshot(path=str(tmp_path / f"compare-{'furniture' if furnished else 'plain'}.png"), full_page=True)


def test_furniture_flutter_paragraph_is_printed_before_no_alerts_return(
    browser: Browser, tmp_path: Path, scheme_pair: tuple[SchemeResult, ...], monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("aosr.gui.result_view._alert_sections", lambda *args: ((), ()))
    _save(tmp_path, scheme_pair[1])
    with _serve(tmp_path) as url, _open(browser, f"{url}/results/{RUN_ID}") as watched:
        page = watched.page
        page.wait_for_selector("#content", state="visible")
        assert page.locator("#alerts > p").all_inner_texts() == [FLUTTER, "沒有警戒"]
        assert not page.locator("#alerts h3").all()
        _assert_text_is_formatted(page)
        _assert_quiet(watched)


def test_two_furnished_results_show_ranking_notice_in_its_own_paragraph(
    browser: Browser, tmp_path: Path, scheme_pair: tuple[SchemeResult, ...],
) -> None:
    furnished = scheme_pair[1]
    (tmp_path / "results").mkdir()
    for letter, result in zip(("a", "b"), (furnished, _renamed(furnished, "other-furnished")), strict=True):
        save_result(result, tmp_path / "results" / f"{letter * 32}.json")
    with _serve(tmp_path) as url, _open(browser, f"{url}/compare/{'a' * 32}/{'b' * 32}") as watched:
        page = watched.page
        page.wait_for_selector("#content", state="visible")
        assert page.locator("#table-verdict").inner_text() == "兩份總代價相同，分不出哪一份比較好"
        assert "家具模型：近似" in page.locator("#ranking-approximation").inner_text()
        assert page.locator("#ranking-approximation").evaluate("el => el.tagName") == "P"
        _assert_text_is_formatted(page)
        _assert_quiet(watched)
