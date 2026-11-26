"""#1265 輸入頁說明字瘦身（老闆 11-26「這種解釋性的東西很痛苦」）：狀態只寫狀態，能從版面看出來的刪，非留不可的縮到最短。

從範本開一份新的之後，派工單那張表上「刪」的幾句都不在頁面上（元素一起拿掉的，元素也不在），「改成」的幾句照新字；
第 1 段右卡兩列的列名寫「喇叭位置」「主位位置」（這份方案現在擺的固定位置，不是範圍或限制）。方案是範本本身，伺服器是考卷自己起的。"""
from __future__ import annotations

from pathlib import Path

from playwright.sync_api import Browser, expect

from tests.engine._input_page_steps import new_from_template
from tests.engine.test_gui_browser import _assert_quiet, _open, _serve, browser as browser

# 表上「刪」的那幾句（各取一段夠獨特的字）：開好範本後整頁的字裡一句都不准有。
GONE = (
    "邊上的數字是房間尺寸",
    "算好後，結果就出現在這一行",
    "工作／聆聽姿勢切換會一起改",
    "「改善多少」從這裡算起",
    "牆上切好的區會跟著牆邊長或縮",
    "只列出來看；要改貼什麼",
    "不用先按儲存",
    "黃色的區照它們現在存的材料算",
    "六面一起看、一起改",
    "一起算聲音的",
    "聲源模型、散射係數",
    "灰色虛線格子上也能拖",
    "另一邊照鏡子出現",
    "站在房間裡看這一面",
    "房間預設沒設上限",
    "設定要交給程式的區",
    "沒畫到的地方自動補成一區",
    "兩支喇叭的中線（兩邊一起畫開著時）",
)
# 字被刪、元素一起拿掉的那幾個：頁面上不准還有。
GONE_ELEMENTS = ("#room-step-title .say", "#fact-primary small", "#fact-original small", "#scheme-section p.hint",
                 "#elevation-list p.say", "#zone-methods-row small", "#calculate-note")
# 表上「改成」的那幾句：元素 → 新字。
NEW_WORDS = {
    "#fact-depth-text": "總厚度上限：沒設",
    "#coordinates-summary": "座標數字",
    "#advanced-fold summary": "進階",
    "#wall-step-title .say": "在牆上拖框切一區",
    "#elevation-axis-words": "鏡子：喇叭中線，離左牆 2.00 公尺",
    "#zone-to-search-note": "先到「這間房貼什麼」選區",
    "#elevation-size": "前牆 寬 4 × 高 3 公尺",
    "#elevation-depth": "總厚度上限：沒設",
    "#fact-speakers .k": "喇叭位置",
    "#fact-primary .k": "主位位置",
}
LEGEND = ("實線：畫的框；虛線：自動補的區", "點劃線：鏡子")


def test_after_opening_a_template_the_explaining_sentences_are_gone_and_the_short_ones_read_as_listed(
        tmp_path: Path, browser: Browser) -> None:
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        page.wait_for_function("document.body.dataset.inputStart === 'empty'")
        new_from_template(page, "lean-words")
        page.wait_for_function("zoneLimitsGeneration > 0 && zoneLimitsShown === zoneLimitsGeneration")
        expect(page.locator("#elevation-size b")).to_have_text("前牆")  # 開好範本時選中的是前牆
        # 還沒算過：只留左邊那個標籤，右邊那一格沒有字。
        expect(page.locator("#room-result-title")).to_have_text("還沒算過")
        expect(page.locator("#room-result-text")).to_have_text("")
        for selector, words in NEW_WORDS.items():
            expect(page.locator(selector)).to_have_text(words)
        expect(page.locator("#zone-to-search")).to_be_disabled()
        legend = page.locator("#elevation-legend")
        for words in LEGEND:
            expect(legend.get_by_text(words, exact=True)).to_have_count(1)
        for selector in GONE_ELEMENTS:
            expect(page.locator(selector)).to_have_count(0)
        # 收著的摺疊裡、藏著的元素裡也算：整份文件的字（textContent）都不准有。
        text = page.evaluate("document.body.textContent")
        assert [words for words in GONE if words in text] == []
        expect(page.locator("#next-note")).to_have_text("")
        _assert_quiet(watched)
