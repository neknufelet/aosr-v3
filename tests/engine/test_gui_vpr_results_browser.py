"""真瀏覽器考結果頁標「含 300 Hz 以下」（#1208 第 2 支）：說明塊與小標籤出現在哪、「為什麼」點得開；沒命中時兩樣都沒有。

結果是考卷自己真算的小房間（分區結果頁考卷那一間）：有命中的那一份後牆整面換成黏住鋼板的背襯（``_bonded_backing_cases``），
沒命中的就是分區結果頁那一份。讀回構造結果要雙精度，伺服器用正式網頁入口起在獨立行程（``real_server``）；
那個行程的資料夾裡沒有使用者材料庫——存的方案有命中、現在材料庫沒有那一筆，照樣標。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import httpx
import pytest
from playwright.sync_api import Browser, expect

from tests.engine._bonded_backing_cases import BONDED_VPR
from tests.engine._construction_gui_server import real_server
from tests.engine._gui_cache import gui_startup_identity_memo as gui_startup_identity_memo
from tests.engine._zone_result_cases import design_scheme, real_design, real_result
from tests.engine.test_gui_browser import _open, browser as browser
from aosr.reporting.scheme import Scheme

HIT_RUN, PLAIN_RUN = "1" * 32, "2" * 32
READ_BACK_MS = 180_000  # 冷的網頁行程第一次讀回構造結果要編譯重評
SENTENCE = ("後牆有 VPR（鋼板共振吸音）這類「鋼板直接貼在棉上」的構造：300 Hz 以下的數字只能參考；"
            "下表標了「含 300 Hz 以下」的幾列都受影響。")
WHY = ("程式算這類構造用的是簡化算法（一維、鋼板當成整塊一起動）。跟 Fraunhofer 的實測比，算出來的共振偏高、頻寬偏窄。"
       "數字沒有改，照舊算。")
# 表上八列的類別名（照表的順序）與預期被標的五列（寫死）。
ROWS = ["音色平衡", "聆聽區穩定性", "低頻頻響", "低頻拖尾", "反射與回聲", "殘響", "聲道匹配", "空間感"]
MARKED = ["音色平衡", "聆聽區穩定性", "低頻頻響", "殘響", "聲道匹配"]
_SEEN = """() => {
  const rows = [...document.querySelectorAll('#categories table tr')].slice(1);
  const box = document.getElementById('categories'), note = document.getElementById('categories-reference');
  return {names: rows.map(row => row.cells[0].textContent),
          tags: rows.map(row => [...row.querySelectorAll('.ref-tag')].map(tag => tag.textContent)),
          elsewhere: [...rows].flatMap(row => [0, 1, 2, 4].map(index => row.cells[index].querySelectorAll('.ref-tag').length)),
          first: rows.map(row => row.cells[3].firstElementChild?.className ?? null),
          cells: rows.map(row => row.cells[3].textContent),
          children: [...box.children].map(child => child.id || child.tagName),
          note: note ? {say: note.querySelector('.ref-say').textContent, swatches: note.querySelectorAll('.ref-sw').length} : null};
}"""


@pytest.fixture(scope="module")
def hit_text(tmp_path_factory: pytest.TempPathFactory, worker_id: str) -> str:
    """有命中的那一份（真算，同一次 pytest 只算一次，本機約 58 秒）：分區結果頁那一份方案，後牆整面換成這類構造。"""
    document = design_scheme().model_dump(mode="json")
    zones = cast(dict[str, object], cast(dict[str, object], document["scene"])["surface_zones_by_wall"])
    zones["xL"] = [{"area": "whole_wall", "material": BONDED_VPR}]
    return real_result(tmp_path_factory, worker_id, "vpr-result-design", Scheme.model_validate(document | {"scheme_id": "vpr-design"}))


@pytest.fixture(scope="module")
def plain_text(tmp_path_factory: pytest.TempPathFactory, worker_id: str) -> str:
    return real_design(tmp_path_factory, worker_id)


def _put(folder: Path, run_id: str, text: str) -> None:
    (folder / "results").mkdir(exist_ok=True)
    (folder / "results" / f"{run_id}.json").write_text(text, encoding="utf-8")


def test_a_result_with_such_a_construction_shows_the_block_above_the_table_and_tags_five_rows(
        tmp_path: Path, browser: Browser, hit_text: str) -> None:
    """有命中的結果頁：「各類結果」表格前面一塊說明（那一句、沒有藍色小方塊、「為什麼」點開才有原因）；
    五列的「注意事項」最前面一個淡藍底小標籤、後面照舊接原本的字，另外三列沒有；別的欄沒有標籤。"""
    _put(tmp_path, HIT_RUN, hit_text)
    assert not (tmp_path / "materials").exists()
    with real_server(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        page.goto(f"{base}/results/{HIT_RUN}", wait_until="domcontentloaded")
        expect(page.locator("#categories table")).to_be_visible(timeout=READ_BACK_MS)
        reply = httpx.get(f"{base}/api/results/{HIT_RUN}", trust_env=False, timeout=60).json()
        served = cast(dict[str, object], reply["reference_only"])
        assert served == {"sentence": SENTENCE, "why_label": "為什麼", "why": WHY, "tag": "含 300 Hz 以下",
                          "categories": ["timbre_balance", "listening_area_stability", "low_frequency_response", "reverberation",
                                         "channel_matching"]}
        seen = cast(dict[str, object], page.evaluate(_SEEN))
        assert seen["names"] == ROWS
        assert seen["children"] == ["categories-reference", "TABLE"]  # 說明塊在表格前面、同一個框裡
        assert seen["note"] == {"say": SENTENCE, "swatches": 0}
        tags = cast(list[list[str]], seen["tags"])
        assert [name for name, row in zip(ROWS, tags, strict=True) if row] == MARKED
        assert [row for row in tags if row] == [["含 300 Hz 以下"]] * 5 and set(cast(list[int], seen["elsewhere"])) == {0}
        flags = {cast(str, row["category"]): cast(str, row["flags_text"]) for row in cast(list[dict[str, object]], reply["categories"])}
        codes = dict(zip(ROWS, flags, strict=True))
        for name, first, cell in zip(ROWS, cast(list["str | None"], seen["first"]), cast(list[str], seen["cells"]), strict=True):
            if name in MARKED:
                assert first == "ref-tag" and cell == f"含 300 Hz 以下 {flags[codes[name]]}", (name, cell)
            else:
                assert cell == flags[codes[name]], (name, cell)
        tag = page.locator("#categories .ref-tag").first
        style = cast(dict[str, str], tag.evaluate(
            "node => { const s = getComputedStyle(node); return {fill: s.backgroundColor, wrap: s.whiteSpace, radius: s.borderTopLeftRadius, edge: s.borderTopStyle}; }"))
        assert style == {"fill": "rgba(60, 120, 210, 0.16)", "wrap": "nowrap", "radius": "4px", "edge": "solid"}
        note = page.locator("#categories-reference")
        assert cast(str, note.evaluate("node => getComputedStyle(node).borderLeftStyle")) == "solid"
        summary = note.locator("summary")
        expect(summary).to_have_text("為什麼")
        assert summary.evaluate("node => getComputedStyle(node, '::after').content") == '" ▸"'
        expect(note.locator("details p")).to_be_hidden()
        summary.click()
        expect(note.locator("details p")).to_be_visible()
        expect(note.locator("details p")).to_have_text(WHY)
        assert summary.evaluate("node => getComputedStyle(node, '::after').content") == '" ▾"'
        assert (watched.console_errors, watched.page_errors) == ([], [])


def test_a_result_without_such_a_construction_has_neither(tmp_path: Path, browser: Browser, plain_text: str) -> None:
    """沒命中的結果頁（貼的是一般的棉）：介面回覆沒有那一格，頁面沒有說明塊、沒有小標籤，表格那個框裡只有表格。"""
    _put(tmp_path, PLAIN_RUN, plain_text)
    with real_server(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        page.goto(f"{base}/results/{PLAIN_RUN}", wait_until="domcontentloaded")
        expect(page.locator("#categories table")).to_be_visible(timeout=READ_BACK_MS)
        assert "reference_only" not in json.loads(httpx.get(f"{base}/api/results/{PLAIN_RUN}", trust_env=False, timeout=60).text)
        seen = cast(dict[str, object], page.evaluate(_SEEN))
        assert seen["names"] == ROWS and seen["children"] == ["TABLE"] and seen["note"] is None
        assert seen["tags"] == [[]] * 8 and page.locator(".ref-tag, .ref-note").count() == 0
        assert (watched.console_errors, watched.page_errors) == ([], [])
