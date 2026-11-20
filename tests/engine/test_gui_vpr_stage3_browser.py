"""真瀏覽器考第三階段頁與「改善多少」的提醒（#1208 第 3 支）：預覽那一段說明、每一跑「注意」那一條、改善表的說明與標籤
出現在哪、「為什麼」點得開；沒命中時都沒有。

第三階段頁照現有那一檔的做法：開服務那一層是假的（不真跑搜尋），輪詢用瀏覽器假時鐘快轉。改善表走正式網頁行程
（``real_server``）：設計是第 2 支瀏覽器考卷那一份真算的結果（後牆整面是這類構造，同一次 pytest 只算一次），按「算原牆基準」
讓正式行程自己真算基準。截圖存 ``AOSR_SHOTS``（有給才存）。
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import cast
from urllib.parse import urlsplit

import httpx
from playwright.sync_api import Browser, Page, expect

from tests.engine._bonded_backing_cases import BONDED_VPR
from tests.engine._construction_gui_server import real_server
from tests.engine._gui_cache import gui_startup_identity_memo as gui_startup_identity_memo
from tests.engine.test_gui_browser import _open, browser as browser
from tests.engine.test_gui_gradient_browser import NOTE, _data, _serve, _shot, _tick
from tests.engine.test_gui_gradient_jobs import NAME, FakeLauncher
from tests.engine.test_gui_vpr_results_browser import HIT_RUN, READ_BACK_MS, SENTENCE, WHY, _put, hit_text as hit_text
from tests.engine.test_gui_vpr_stage3 import STAGE_THREE, _put as _put_zone

LINE = "前牆" + STAGE_THREE
# 改善表的列（有算出代價的六類加總分；低頻拖尾不計分、空間感尚未評估，兩邊都沒有數字就不出列）與預期被標的（寫死）。
ROWS = ["音色平衡", "聆聽區穩定性", "低頻頻響", "反射與回聲", "殘響", "聲道匹配", "總分"]
MARKED = ["音色平衡", "聆聽區穩定性", "低頻頻響", "殘響", "聲道匹配", "總分"]
_NOTE_STYLE = "node => { const s = getComputedStyle(node); return [s.borderLeftStyle, s.listStyleType]; }"


def _ready(page: Page) -> None:
    """頁面開好之後把瀏覽器的時鐘停住：這一頁每 5 秒自己重讀一次、把這一跑的清單整個重畫，考卷連讀幾步的中間被它換掉
    就讀到舊的那一個。停住之後只有考卷自己快轉（``_tick``）才會重讀。"""
    expect(page.locator("#surrogate-state")).to_contain_text("已經有替身")
    page.clock.pause_at(datetime.now() + timedelta(seconds=1))


def _preview(page: Page) -> None:
    """按「預覽」，等這一次的回覆回來、而且畫上去了才回來。開頁時就畫過一次預覽，所以預覽那一塊裡的字不能當成等到了：
    按下去那一刻訊息先換成「正在預覽…」，回覆畫上去的同一步才換成「預覽好了」。"""
    plan = f"/api/gradient/schemes/{NAME}/plan"
    with page.expect_response(lambda response: urlsplit(response.url).path == plan and response.request.method == "POST") as landed:
        page.locator("#preview").click()
    landed.value.finished()
    expect(page.locator("#form-message")).to_have_text("預覽好了，看下面。")


def _opens(page: Page, note: str) -> None:
    """那一塊：左邊細灰線、沒有藍色小方塊、「為什麼」點開才有原因。"""
    box = page.locator(note)
    assert cast(list[str], box.evaluate(_NOTE_STYLE))[0] == "solid" and box.locator(".ref-sw").count() == 0
    expect(box.locator("summary")).to_have_text("為什麼")
    expect(box.locator("details p")).to_be_hidden()
    box.locator("summary").click()
    expect(box.locator("details p")).to_have_text(WHY)
    expect(box.locator("details p")).to_be_visible()


def test_the_stage_three_page_says_it_above_the_preview_and_in_the_notes_of_the_run(browser: Browser, tmp_path: Path) -> None:
    """有命中：預覽那一塊最上面（摘要那一句前面）一段說明，再按一次預覽不會多一段；開跑後這一跑的「注意」第一條就是它
    ——連一行紀錄都還沒有時「注意」也要出來；之後搜尋印了注意行，排在它後面。"""
    data_dir = _data(tmp_path)
    _put_zone(data_dir, "x0", "r0_c1", BONDED_VPR)
    launcher = FakeLauncher()
    with _serve(data_dir, launcher) as url, _open(browser, f"{url}/gradient?scheme={NAME}", fake_clock=True) as watched:
        page = watched.page
        _ready(page)
        _preview(page)
        expect(page.locator("#plan-reference .ref-say")).to_have_text(LINE)
        _preview(page)
        expect(page.locator("#plan-reference .ref-say")).to_have_text(LINE)
        expect(page.locator("#plan-summary")).to_contain_text("往下走步數上限")
        order = page.evaluate("() => [...document.getElementById('plan-block').children].map(node => node.id || node.tagName)")
        assert cast(list[str], order)[:3] == ["H2", "plan-reference", "plan-summary"] and page.locator("#plan-reference").count() == 1
        _opens(page, "#plan-reference")
        _shot(page, "vpr-stage3-preview")
        page.locator("#start").click()
        expect(page.locator("#job-state")).to_contain_text("開跑前核對中")
        expect(page.locator("#job-notes")).to_be_visible()
        expect(page.locator("#job-notes li")).to_have_count(1)
        expect(page.locator("#job-reference .ref-say")).to_have_text(LINE)
        assert cast(list[str], page.locator("#job-reference").evaluate(_NOTE_STYLE)) == ["solid", "none"]
        job_id = launcher.launched[-1][0].split("-")[2]
        (data_dir / "gradient" / job_id / "steps" / "0.log").write_text(NOTE + "\n", encoding="utf-8")
        _tick(page)
        expect(page.locator("#job-notes li")).to_have_count(2)
        assert page.locator("#job-notes li").nth(0).get_attribute("id") == "job-reference"
        expect(page.locator("#job-notes li").nth(1)).to_have_text(NOTE)
        _opens(page, "#job-reference")
        _shot(page, "vpr-stage3-job")
        assert (watched.console_errors, watched.page_errors) == ([], [])


def test_the_stage_three_page_without_such_a_construction_has_neither(browser: Browser, tmp_path: Path) -> None:
    """沒命中：預覽沒有那一段、開跑後沒有紀錄行時「注意」整塊照舊藏著；搜尋印了注意行就只有那一行。"""
    data_dir = _data(tmp_path)
    launcher = FakeLauncher()
    with _serve(data_dir, launcher) as url, _open(browser, f"{url}/gradient?scheme={NAME}", fake_clock=True) as watched:
        page = watched.page
        _ready(page)
        _preview(page)
        expect(page.locator("#plan-summary")).to_contain_text("往下走步數上限")
        order = page.evaluate("() => [...document.getElementById('plan-block').children].map(node => node.id || node.tagName)")
        assert cast(list[str], order)[:2] == ["H2", "plan-summary"]
        _shot(page, "vpr-stage3-plain-preview")
        page.locator("#start").click()
        expect(page.locator("#job-state")).to_contain_text("開跑前核對中")
        expect(page.locator("#job-progress li").first).to_be_visible()
        expect(page.locator("#job-notes")).to_be_hidden()
        job_id = launcher.launched[-1][0].split("-")[2]
        (data_dir / "gradient" / job_id / "steps" / "0.log").write_text(NOTE + "\n", encoding="utf-8")
        _tick(page)
        expect(page.locator("#job-notes li")).to_have_text([NOTE])
        assert page.locator(".ref-note, .ref-tag").count() == 0
        _shot(page, "vpr-stage3-plain-job")
        assert (watched.console_errors, watched.page_errors) == ([], [])


def test_the_improvement_table_gets_the_block_above_it_and_tags_five_rows_and_the_total(
        tmp_path: Path, browser: Browser, hit_text: str) -> None:
    """設計的後牆整面是這類構造（原牆基準那一面照舊是它）：改善表前面一塊說明，五類與總分那幾列類別名後面一個小標籤，
    反射與回聲那一列沒有；別的欄沒有標籤。"""
    _put(tmp_path, HIT_RUN, hit_text)
    with real_server(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        page.goto(f"{base}/results/{HIT_RUN}", wait_until="domcontentloaded")
        expect(page.locator("#zone-result")).to_be_visible(timeout=READ_BACK_MS)
        expect(page.locator("#zone-state")).to_contain_text("原牆基準還沒算")
        assert page.locator("#zone-result .ref-note, #zone-result .ref-tag").count() == 0  # 還沒有改善表：這一節沒有說明
        page.locator("#zone-baseline-start").click()
        expect(page.locator("#zone-improvement tr.zone-total")).to_be_visible(timeout=600_000)
        reply = json.loads(httpx.get(f"{base}/api/results/{HIT_RUN}/zones", trust_env=False, timeout=120).text)
        assert reply["improvement"]["reference_only"] == {
            "sentence": SENTENCE, "why_label": "為什麼", "why": WHY, "tag": "含 300 Hz 以下", "rows": MARKED}
        seen = cast(dict[str, object], page.evaluate("""() => {
          const rows = [...document.querySelectorAll('#zone-improvement table tr')].slice(1);
          return {children: [...document.getElementById('zone-improvement').children].map(child => child.id || child.tagName),
                  say: document.querySelector('#zone-improvement-reference .ref-say').textContent,
                  names: rows.map(row => row.cells[0].firstChild.textContent),
                  tags: rows.map(row => [...row.cells[0].querySelectorAll('.ref-tag')].map(tag => tag.textContent)),
                  elsewhere: rows.flatMap(row => [1, 2, 3].map(index => row.cells[index].querySelectorAll('.ref-tag').length))};
        }"""))
        assert seen["children"] == ["zone-improvement-reference", "TABLE"] and seen["say"] == SENTENCE
        assert seen["names"] == ROWS and set(cast(list[int], seen["elsewhere"])) == {0}
        tags = cast(list[list[str]], seen["tags"])
        assert [name for name, row in zip(ROWS, tags, strict=True) if row] == MARKED
        assert [row for row in tags if row] == [["含 300 Hz 以下"]] * 6
        fill = page.locator("#zone-improvement .ref-tag").first.evaluate("node => getComputedStyle(node).backgroundColor")
        assert fill == "rgba(60, 120, 210, 0.16)"
        _opens(page, "#zone-improvement-reference")
        if os.environ.get("AOSR_SHOTS"):
            page.locator("#zone-result").screenshot(path=str(Path(os.environ["AOSR_SHOTS"]) / "vpr-stage3-improvement.png"))
        assert (watched.console_errors, watched.page_errors) == ([], [])
