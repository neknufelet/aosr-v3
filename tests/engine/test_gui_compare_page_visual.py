"""比較頁的版面：寬螢幕摘要與「改了哪裡」左右並排、疊圖佔滿整頁寬；窄螢幕照閱讀順序上下排。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
from playwright.sync_api import Browser, Page, Route
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from aosr.reporting.result import SchemeResult
from tests.engine.test_gui_browser import _assert_quiet, _open, _serve, browser
from tests.engine.test_gui_compare_browser import A_ID, B_ID, _data, _has_lines
from tests.engine.test_gui_compare_routes import _files
from tests.engine.test_scheme_pipeline import shared_control_result


@pytest.fixture(scope="module")
def pair(tmp_path_factory: pytest.TempPathFactory, worker_id: str) -> tuple[SchemeResult, SchemeResult]:
    return (shared_control_result(tmp_path_factory, worker_id, "wall-1"),
            shared_control_result(tmp_path_factory, worker_id, "wall-2"))


def _box(page: Page, selector: str) -> dict[str, float]:
    box = page.locator(selector).bounding_box()
    assert box is not None, selector
    return {"left": box["x"], "top": box["y"], "right": box["x"] + box["width"],
            "bottom": box["y"] + box["height"], "width": box["width"]}


def _chart_width(page: Page) -> float:
    return cast(float, page.evaluate("() => plot.width"))


def test_wide_screen_puts_changes_beside_summary_and_chart_full_width(
        tmp_path: Path, browser: Browser, pair: tuple[SchemeResult, SchemeResult]) -> None:
    # 寬螢幕：摘要在左、改了哪裡與核對在右，同一列、不重疊；疊圖區塊從左欄左緣到右欄右緣整頁寬，
    # 圖本身也撐滿那一塊，不再被右欄擠窄（老闆 1440 寬的螢幕）。
    with _serve(tmp_path) as base:
        _files(tmp_path, pair[0], A_ID)
        _files(tmp_path, pair[1], B_ID)
        with _open(browser, f"{base}/compare/{A_ID}/{B_ID}", viewport_width=1440) as watched:
            page = watched.page
            _has_lines(page)
            summary, changes = _box(page, "#compare-summary"), _box(page, "#compare-changes")
            chart, content = _box(page, "#compare-chart"), _box(page, "#content")
            assert abs(summary["top"] - changes["top"]) < 1
            assert summary["right"] < changes["left"]
            assert page.locator("#compare-check #fingerprints").is_visible()
            check = _box(page, "#compare-check")
            assert chart["top"] > max(summary["bottom"], changes["bottom"], check["bottom"])
            assert abs(chart["left"] - summary["left"]) < 1 and abs(chart["right"] - changes["right"]) < 1
            assert abs(chart["width"] - content["width"]) < 1
            inner = cast(float, page.evaluate(
                "() => document.getElementById('chart').clientWidth"))
            assert abs(_chart_width(page) - inner) < 1
            assert _chart_width(page) > 0.9 * chart["width"]
            # 兩個總代價框上下排、一樣寬、左緣對齊（摘要欄只有半頁寬，左右並排放不下）。
            box_a, box_b = _box(page, "#table-a"), _box(page, "#table-b")
            assert abs(box_a["left"] - box_b["left"]) < 1 and abs(box_a["width"] - box_b["width"]) < 1
            assert box_b["top"] >= box_a["bottom"]
            # 視窗改大小，圖跟著區塊寬度重畫（不是停在原來的寬度）。
            wide = _chart_width(page)
            page.set_viewport_size({"width": 1100, "height": 1100})
            try:
                page.wait_for_function(
                    "() => Math.abs(plot.width - document.getElementById('chart').clientWidth) < 1",
                    timeout=5000)
            except PlaywrightTimeoutError:
                pass  # 沒重畫就落到下面的斷言，紅在「圖寬跟區塊寬對不上」。
            narrow_inner = cast(float, page.evaluate("() => document.getElementById('chart').clientWidth"))
            assert abs(_chart_width(page) - narrow_inner) < 1
            assert _chart_width(page) < wide
            _assert_quiet(watched)


def test_narrow_screen_stacks_summary_changes_then_chart(
        tmp_path: Path, browser: Browser, pair: tuple[SchemeResult, SchemeResult]) -> None:
    # 手機寬：按閱讀順序上下排（摘要、改了哪裡、疊圖），圖不超出畫面；按鈕換行不擠出去。
    with _serve(tmp_path) as base:
        _files(tmp_path, pair[0], A_ID)
        _files(tmp_path, pair[1], B_ID)
        with _open(browser, f"{base}/compare/{A_ID}/{B_ID}", viewport_width=390) as watched:
            page = watched.page
            _has_lines(page)
            summary, changes = _box(page, "#compare-summary"), _box(page, "#compare-changes")
            check, chart = _box(page, "#compare-check"), _box(page, "#compare-chart")
            assert summary["bottom"] <= changes["top"] and changes["bottom"] <= check["top"]
            assert check["bottom"] <= chart["top"]
            assert chart["right"] <= 390
            assert _chart_width(page) <= chart["width"]
            buttons = page.locator("#position-buttons button").all()
            assert all(cast(dict[str, float], button.bounding_box())["x"]
                       + cast(dict[str, float], button.bounding_box())["width"] <= 390 for button in buttons)
            # 分項表不擠成一字一行：表頭與各格都只佔一行，放不下就在那一塊裡左右捲，整頁不橫捲。
            # 「改了哪裡」也一樣：項目名不被 A、B 兩欄擠成一字一行。
            lines = cast(list[list[object]], page.evaluate(
                LINES_PER_CELL_JS, "#categories th, #categories td, #changes th, #changes td"))
            assert [item for item in lines if item[1] != 1] == []
            assert cast(int, page.evaluate("() => document.documentElement.scrollWidth")) <= 390
            _assert_quiet(watched)


# 每一格的字排成幾行（用 Range 量字的外框，一行一個框）；空格不算。
LINES_PER_CELL_JS = """selector => [...document.querySelectorAll(selector)]
  .filter((cell) => cell.textContent.trim())
  .map((cell) => {
    const range = document.createRange();
    range.selectNodeContents(cell);
    const tops = new Set([...range.getClientRects()].map((rect) => Math.round(rect.top)));
    return [cell.textContent, tops.size];
  })"""


def test_better_side_is_marked_by_weight_not_only_colour(
        tmp_path: Path, browser: Browser, pair: tuple[SchemeResult, SchemeResult]) -> None:
    # 比較好的那一份：總代價框加粗、加框，分項表那一格代價加粗；不只靠顏色（色弱也看得出來）。
    with _serve(tmp_path) as base:
        _files(tmp_path, pair[0], A_ID)
        _files(tmp_path, pair[1], B_ID)
        with _open(browser, f"{base}/compare/{A_ID}/{B_ID}", viewport_width=1440) as watched:
            page = watched.page
            _has_lines(page)
            data = _data(page, base)
            assert data.table.better in {"a", "b"}
            other = {"a": "b", "b": "a"}[data.table.better]
            weight = "el => Number(getComputedStyle(el).fontWeight)"
            border = "el => parseFloat(getComputedStyle(el).borderTopWidth)"
            better, plain = page.locator(f"#table-{data.table.better}"), page.locator(f"#table-{other}")
            assert better.evaluate(weight) >= 700 > plain.evaluate(weight)
            assert better.evaluate(border) > plain.evaluate(border)
            cells = page.locator("#categories td.better").all()
            assert {cell.inner_text() for cell in cells} == {
                getattr(row, row.better).cost_text for row in data.categories if row.better in {"a", "b"}}
            assert all(cell.evaluate(weight) >= 700 for cell in cells)
            _assert_quiet(watched)


# 在頁面裡把「改了哪裡」從 1 列一列一列加上去，每加一列觸發一次改視窗大小（頁面重新擺核對那一卡），
# 然後量：頁面擺的位置、三種擺法（右欄、左欄、兩欄底下整排）各自兩欄自然高度差、頁面擺法的高度差。
PLACEMENT_SWEEP_JS = """rowsWanted => {
  const $ = (id) => document.getElementById(id);
  const table = document.querySelector('#changes table');
  const template = table.rows[1].cloneNode(true);
  while (table.rows.length > 1) table.rows[1].remove();
  const gap = () => {
    const [left, right] = ['compare-left', 'compare-right'].map((id) => $(id).getBoundingClientRect());
    return Math.abs(left.height - right.height);
  };
  const results = [];
  for (let count = 1; count <= rowsWanted; count++) {
    table.append(template.cloneNode(true));
    window.dispatchEvent(new Event('resize'));
    const check = $('compare-check'), home = check.parentElement;
    const chosen = gap();
    const gaps = ['compare-right', 'compare-left', 'compare-top'].map((id) => { $(id).append(check); return gap(); });
    home.append(check);
    results.push({count, where: home.id, chosen, gaps});
  }
  return results;
}"""


def test_check_card_goes_where_the_two_columns_come_out_most_even(
        tmp_path: Path, browser: Browser, pair: tuple[SchemeResult, SchemeResult]) -> None:
    # 老闆換牆面材料一次就改 6 列：「座位與聲道設定核對」固定放右欄時，摘要卡下面留一大片空白。
    # 改了幾處都一樣：核對那一卡擺在右欄、左欄或兩欄底下整排，挑兩欄高度最接近的那一種；
    # 改得少時接在「改了哪裡」下面，改得多時接在摘要下面，中間兩欄差不多高時整排放在兩欄底下。
    with _serve(tmp_path) as base:
        _files(tmp_path, pair[0], A_ID)
        _files(tmp_path, pair[1], B_ID)
        with _open(browser, f"{base}/compare/{A_ID}/{B_ID}", viewport_width=1440) as watched:
            page = watched.page
            _has_lines(page)
            sweep = cast(list[dict[str, object]], page.evaluate(PLACEMENT_SWEEP_JS, 30))
            for item in sweep:
                gaps = cast(list[float], item["gaps"])
                assert cast(float, item["chosen"]) <= min(gaps) + 1, item
            assert {item["where"] for item in sweep} == {"compare-right", "compare-left", "compare-top"}
            assert (sweep[0]["where"], sweep[-1]["where"]) == ("compare-right", "compare-left")
            _assert_quiet(watched)


# 把「改了哪裡」複製到 rowsWanted 列（老闆一次改很多處），觸發改視窗大小讓頁面重新擺核對那一卡；
# 量上排每一張卡的高度，再強制每張卡照內容高度（不往下撐）量一次。兩次一樣，卡片才是照自己的內容高度。
NATURAL_HEIGHT_JS = """rowsWanted => {
  const table = document.querySelector('#changes table');
  const template = table.rows[1];
  while (table.rows.length <= rowsWanted) table.append(template.cloneNode(true));
  window.dispatchEvent(new Event('resize'));
  const cards = ['compare-summary', 'compare-changes', 'compare-check'];
  const heights = () => cards.map((id) => document.getElementById(id).getBoundingClientRect().height);
  const shown = heights();
  const style = document.createElement('style');
  style.textContent = '.compare-column>section{flex-grow:0!important}#compare-top>section{align-self:start!important}';
  document.head.append(style);
  const natural = heights();
  style.remove();
  return {where: document.getElementById('compare-check').parentElement.id, shown, natural};
}"""


def test_cards_keep_their_own_height_when_many_things_changed(
        tmp_path: Path, browser: Browser, pair: tuple[SchemeResult, SchemeResult]) -> None:
    # 改了 26 處：核對那一卡擺到左欄摘要下面，以前被撐到跟右欄長表一樣高，一句話底下一大片空白。
    # 每張卡（摘要、改了哪裡、核對）照內容自己的高度，不往下撐。
    with _serve(tmp_path) as base:
        _files(tmp_path, pair[0], A_ID)
        _files(tmp_path, pair[1], B_ID)
        with _open(browser, f"{base}/compare/{A_ID}/{B_ID}", viewport_width=1440) as watched:
            page = watched.page
            _has_lines(page)
            measured = cast(dict[str, object], page.evaluate(NATURAL_HEIGHT_JS, 26))
            assert measured["where"] == "compare-left"
            shown, natural = cast(list[float], measured["shown"]), cast(list[float], measured["natural"])
            assert all(abs(left - right) < 1 for left, right in zip(shown, natural, strict=True)), measured
            _assert_quiet(watched)


def test_equal_totals_verdict_is_not_green(
        tmp_path: Path, browser: Browser, pair: tuple[SchemeResult, SchemeResult]) -> None:
    # 兩份總代價相同、分不出好壞（伺服器那一邊的判法在 test_equal_totals_print_no_rank）：
    # 那一句不用綠色粗體（不像在報好消息），總代價框都不加重。頁面只照伺服器給的字排版，這裡把回應換成相同那一種。
    with _serve(tmp_path) as base:
        _files(tmp_path, pair[0], A_ID)
        _files(tmp_path, pair[1], B_ID)
        with _open(browser, f"{base}/compare/{A_ID}/{B_ID}", viewport_width=1440) as watched:
            page = watched.page
            same = {"a_text": "總代價 4.395（越低越好）", "b_text": "總代價 4.395（越低越好）",
                    "verdict_text": "兩份總代價相同，分不出哪一份比較好", "better": "same"}

            def equal_totals(route: Route) -> None:
                body = route.fetch().json()
                body["table"].update(same)
                route.fulfill(status=200, content_type="application/json", body=json.dumps(body))
            page.route(f"**/api/compare/{A_ID}/{B_ID}", equal_totals)
            page.reload(wait_until="networkidle")
            _has_lines(page)
            verdict = page.locator("#table-verdict")
            assert verdict.inner_text() == same["verdict_text"]
            # 跟摘要第一句（一般字）同樣粗細、同樣顏色。
            style = cast(dict[str, str], verdict.evaluate(
                "el => ({weight: getComputedStyle(el).fontWeight, color: getComputedStyle(el).color,"
                " plain: getComputedStyle(document.getElementById('summary-text')).color})"))
            assert int(style["weight"]) < 700
            assert style["color"] == style["plain"]
            assert not page.locator("#totals .better").all()
            _assert_quiet(watched)
