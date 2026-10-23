"""第一階段真網頁：伺服器字串逐格呈現、兩種寬度與第二階段相容。"""
from pathlib import Path
import json

import pytest
from playwright.sync_api import Browser

from aosr.runtime import configure_calculation_jax
from tests.engine._gui_cache import gui_startup_identity_memo
from tests.engine.test_gui_browser import _assert_quiet, _assert_text_is_formatted, _open, _serve, browser
from tests.engine.test_stage_one_shortlist import _finished
from aosr.search.stage_one_shortlist import RANK_AND_QUALITY
from tests.engine._search_room_loop_cases import layout_for
from tests.engine.test_gui_stage_one_view import second_stage
from tests.engine._stage_one_post_check_cases import prepared
from tests.engine.test_search_stage_one_post_check import run


@pytest.mark.parametrize("width", [1440, 900])
def test_stage_one_table_scope_pause_and_screenshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                  browser: Browser, width: int) -> None:
    configure_calculation_jax()
    layout = layout_for(room_range={"Lx": {"low": 7.9, "high": 8.1}, "Ly": {"low": 7.9, "high": 8.1},
        "Lz": {"low": 2.6, "high": 2.7}})
    store = _finished(tmp_path, monkeypatch, budget=8, top_n=5, layout=layout)
    raw = json.loads(store.status_path.read_bytes())
    # 本題只驗搜尋投影；保留真細算文件，但搜尋尚無最佳結果可供三張圖讀回。
    raw.update(state="paused_time_limit", best_trial=None, best_score=None, message="單題時間上限暫停")
    store.status_path.write_text(json.dumps(raw))
    with _serve(tmp_path) as base, _open(browser, f"{base}/searches/{store.search_id}", viewport_width=width) as watched:
        page = watched.page
        page.locator("#shortlist table").wait_for()
        data = page.request.get(f"{base}/api/searches/{store.search_id}").json()
        block = next(block for block in data["blocks"] if block["key"] == "shortlist")
        assert page.locator("#shortlist th").all_text_contents() == block["table"]["headings"]
        assert page.locator("#shortlist tr").evaluate_all("nodes => nodes.slice(1).map(n => [...n.cells].map(c => c.textContent))") == block["table"]["rows"]
        assert page.locator("#search-scope").inner_text() == "第一階段：房間尺寸與喇叭、座位擺位一起搜尋，只算 20～300 Hz 有限元素"
        pause = page.locator("#stage p", has_text="搜尋：因單題時間上限暫停")
        assert "notice" in (pause.get_attribute("class") or "")
        # 清單的固定說明句帶「未確認」三個字，但不是出錯，不准標紅（主對話看截圖抓到）。
        explanation = page.locator("#shortlist p", has_text=RANK_AND_QUALITY)
        assert "未確認" in RANK_AND_QUALITY and "notice" not in (explanation.get_attribute("class") or "")
        assert page.locator("body").evaluate("e => e.scrollWidth <= window.innerWidth")
        assert page.locator("#search-content section").evaluate_all("nodes => nodes.every(e => e.scrollWidth <= e.clientWidth)")
        wrapper = page.locator("#shortlist .table-scroll")
        assert wrapper.evaluate("e => e.scrollWidth > e.clientWidth")
        # 清單是全寬一塊，跟目前階段同寬，不是擠進兩欄的半格（植錯存活補的）。
        widths = page.evaluate("() => ['#stage', '#shortlist'].map(id => document.querySelector(id).getBoundingClientRect().width)")
        assert widths[0] == widths[1]
        page.screenshot(path=str(tmp_path / f"stage-one-{width}.png"), full_page=True)
        wrapper.evaluate("e => { e.scrollLeft = e.scrollWidth; }")
        page.locator("#shortlist").screenshot(path=str(tmp_path / f"stage-one-ratios-{width}.png"))
        _assert_text_is_formatted(page)
        _assert_quiet(watched)


def test_old_python_without_optional_fields_keeps_second_stage_scope(tmp_path: Path, browser: Browser) -> None:
    store = second_stage(tmp_path, False)
    with _serve(tmp_path) as base, _open(browser, f"{base}/searches/{store.search_id}", viewport_width=1440) as watched:
        page = watched.page
        page.locator("#stage").wait_for()
        data = page.request.get(f"{base}/api/searches/{store.search_id}").json()
        assert "scope_text" not in data
        assert all("table" not in block for block in data["blocks"])
        assert page.locator("#search-scope").inner_text() == "第二階段子集：房間與材料固定，只搜尋對稱的喇叭與座位擺位。"
        page.screenshot(path=str(tmp_path / "stage-two-1440.png"), full_page=True)
        _assert_text_is_formatted(page)
        _assert_quiet(watched)


def test_post_check_reason_words_do_not_turn_red(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, browser: Browser) -> None:
    # 入選後檢查的原因是寫入端原文（例如跳過時寫「讀不到細算結果」）：伺服器沒標警示就不准因字面標紅（審查反例）。
    configure_calculation_jax()
    store = prepared(tmp_path, monkeypatch)
    run(store, tmp_path / "child")
    path = store.path / "stage-one-check" / "summary.json"
    raw = json.loads(path.read_bytes())
    raw["seats"][0]["reason_text"] = "考卷：原因原文寫讀不到也失敗過，但這一席照常診斷"
    path.write_text(json.dumps(raw))
    with _serve(tmp_path) as base, _open(browser, f"{base}/searches/{store.search_id}") as watched:
        page = watched.page
        line = page.locator("#modal p", has_text="考卷：原因原文寫讀不到也失敗過")
        line.wait_for()
        data = page.request.get(f"{base}/api/searches/{store.search_id}").json()
        modal = next(block for block in data["blocks"] if block["key"] == "modal")
        assert modal["red"] == "warning_only" and modal["warning"] is False
        assert "notice" not in (line.get_attribute("class") or "")
        _assert_quiet(watched)
