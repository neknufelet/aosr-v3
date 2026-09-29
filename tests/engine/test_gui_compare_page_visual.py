"""比較頁的版面：寬螢幕摘要與「改了哪裡」左右並排、疊圖佔滿整頁寬；窄螢幕照閱讀順序上下排。"""
from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from playwright.sync_api import Browser, Page

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
            assert page.locator("#compare-changes #fingerprints").is_visible()
            assert chart["top"] > max(summary["bottom"], changes["bottom"])
            assert abs(chart["left"] - summary["left"]) < 1 and abs(chart["right"] - changes["right"]) < 1
            assert abs(chart["width"] - content["width"]) < 1
            inner = cast(float, page.evaluate(
                "() => document.getElementById('chart').clientWidth"))
            assert abs(_chart_width(page) - inner) < 1
            assert _chart_width(page) > 0.9 * chart["width"]
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
            chart = _box(page, "#compare-chart")
            assert summary["bottom"] <= changes["top"] and changes["bottom"] <= chart["top"]
            assert chart["right"] <= 390
            assert _chart_width(page) <= chart["width"]
            buttons = page.locator("#position-buttons button").all()
            assert all(cast(dict[str, float], button.bounding_box())["x"]
                       + cast(dict[str, float], button.bounding_box())["width"] <= 390 for button in buttons)
            _assert_quiet(watched)


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
