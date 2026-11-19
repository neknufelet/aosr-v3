"""真瀏覽器考「鋼板直接貼在棉上」的構造（#1208 第 1 支）在材料庫頁與分區頁看到什麼。

資料全是自編的：材料是 ``test_gui_vpr_reference`` 那兩筆（同樣的鋼板與棉，一筆背襯黏住、一筆沒有），房間是左右對稱考卷
那間假房間（5.0 × 4.0 × 2.7 m）。畫面上的字逐字跟伺服器那一份（``reference_only.words``）比；那一份的原文另由函式層考卷釘住。
曲線要雙精度，材料庫頁與面板曲線那幾題的伺服器用正式網頁入口起在獨立行程（``real_server``）。
"""
from __future__ import annotations

from pathlib import Path
from typing import cast

from playwright.sync_api import Browser, Page, expect

from aosr.gui import reference_only
from tests.engine._construction_gui_server import real_server
from tests.engine.test_gui_browser import _open, browser as browser
from tests.engine.test_gui_vpr_reference import HIT, PLAIN_ID, _write_user_library

COMPUTED_MS = 60_000  # 等伺服器算完曲線的上限（冷行程第一次算一種新構造要編譯）
WORDS = reference_only.words()
# 藍底該到哪裡：橫軸是對數軸，界線頻率落在圖寬的幾分之幾（從頁面那張圖自己的橫軸頭尾算，不用它的換算函式）。
_BAND = """() => {
  const over = document.querySelector('#curve-chart .u-over'), band = document.getElementById('curve-ref-band');
  const word = document.getElementById('curve-ref-band-word'), {min, max} = plot.scales.x;
  const box = element => { const rect = element.getBoundingClientRect(); return [rect.left, rect.top, rect.width, rect.height]; };
  return {over: box(over), band: box(band), word: box(word), min, max, fill: getComputedStyle(band).backgroundColor};
}"""


def _open_material(page: Page, material_id: str) -> None:
    page.locator(f'#user-list tr[data-id="{material_id}"] button.open').click()
    expect(page.locator("#curve-chart canvas")).to_have_count(1, timeout=COMPUTED_MS)


def test_the_material_page_shades_the_band_and_says_it_three_ways(tmp_path: Path, browser: Browser) -> None:
    """材料庫頁打開命中的那一筆：曲線圖界線以下淡藍底、頂端內側一行小字；圖例一行「藍底：」加那一句，「為什麼」點開才有原因；
    「種類」那一行後面接一句。原本的橘底外推照留。打開沒黏住的那一筆：這三樣一樣都沒有。"""
    import math

    _write_user_library(tmp_path)
    with real_server(tmp_path) as base, _open(browser, f"{base}/materials") as watched:
        page = watched.page
        _open_material(page, HIT)
        expect(page.locator("#curve-ref-band")).to_be_visible()
        found = cast(dict[str, object], page.evaluate(_BAND))
        over, band, word = (cast(list[float], found[key]) for key in ("over", "band", "word"))
        share = math.log(cast(float, WORDS["band_hz"]) / cast(float, found["min"])) / math.log(cast(float, found["max"]) / cast(float, found["min"]))
        assert abs(band[0] - over[0]) < 1 and abs(band[1] - over[1]) < 1 and abs(band[3] - over[3]) < 1, found
        assert abs(band[2] - over[2] * share) < 2 and 0.2 < share < 0.6, (found, share)
        assert found["fill"] == "rgba(60, 120, 210, 0.16)"
        expect(page.locator("#curve-ref-band-word")).to_have_text(cast(str, WORDS["band_words"]))
        assert abs(word[0] - band[0]) < 1 and abs(word[1] - band[1]) < 1 and word[2] <= band[2], found  # 小字在藍底頂端內側、不比藍底寬
        note = page.locator("#curve-key li#curve-ref-note")
        expect(note).to_be_visible()
        expect(note.locator(".ref-say")).to_have_text(cast(str, WORDS["sentence"]))
        assert note.inner_text().startswith("藍底：" + cast(str, WORDS["sentence"]))
        expect(note.locator("summary")).to_have_text("為什麼")
        assert note.locator("summary").evaluate("node => getComputedStyle(node, '::after').content") == '" ▸"'
        expect(note.locator("details p")).to_be_hidden()
        note.locator("summary").click()
        expect(note.locator("details p")).to_have_text(cast(str, WORDS["why_only"]))
        expect(note.locator("details p")).to_be_visible()
        expect(page.locator("#editor-kind")).to_have_text("種類：固定材料（整塊選或不選）" + cast(str, WORDS["kind_words"]))
        expect(page.locator('#curve-key li[data-kind="extrapolation"]')).to_be_visible()  # 泡棉的外推橘底照留
        # 按重畫：藍底、圖例、那一句都只有一份（不疊第二份）。
        page.locator("#curve-redraw").click()
        expect(page.locator("#curve-redraw")).to_be_enabled(timeout=COMPUTED_MS)
        for selector in ("#curve-chart .ref-band", "#curve-chart .ref-band-word", "#curve-key .ref-note", "#editor-kind .ref-kind"):
            expect(page.locator(selector)).to_have_count(1)
        # 換開同樣的鋼板與棉、背襯沒黏住的那一筆：沒有藍底、沒有那一行圖例、「種類」那一行照舊。
        _open_material(page, PLAIN_ID)
        expect(page.locator("#curve-basis")).to_be_visible(timeout=COMPUTED_MS)
        for selector in (".ref-band", ".ref-band-word", ".ref-note", ".ref-kind"):
            expect(page.locator(selector)).to_have_count(0)
        expect(page.locator("#editor-kind")).to_have_text("種類：固定材料（整塊選或不選）")
        assert (watched.console_errors, watched.page_errors) == ([], [])
