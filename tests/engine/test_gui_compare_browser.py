"""比較頁畫面與伺服器資料的一致性。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
from playwright.sync_api import Browser, Page, Route

from aosr.config.paths import config_path
from aosr.gui.compare_view import CompareView, OverlaySeries
from aosr.reporting.result import SchemeResult, reevaluate
from tests.engine.test_gui_browser import (
    _assert_quiet, _assert_text_is_formatted, _open, _serve, browser)
from tests.engine.test_gui_compare_routes import _files
from tests.engine.test_gui_compare_view import moved_primary_result
from tests.engine.test_scheme_pipeline import shared_control_result

A_ID = "b" * 32
B_ID = "c" * 32


@pytest.fixture(scope="module")
def pair(tmp_path_factory: pytest.TempPathFactory, worker_id: str) -> tuple[SchemeResult, SchemeResult]:
    return (shared_control_result(tmp_path_factory, worker_id, "wall-1"),
            shared_control_result(tmp_path_factory, worker_id, "wall-2"))


def _has_lines(page: Page, selector: str = "#chart") -> None:
    page.wait_for_function("""selector => {
      const canvas = document.querySelector(`${selector} canvas`);
      if (!canvas) return false;
      const data = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
      for (let i = 0; i < data.length; i += 4) {
        const high = Math.max(data[i], data[i + 1], data[i + 2]);
        const low = Math.min(data[i], data[i + 1], data[i + 2]);
        if (data[i + 3] > 0 && high - low > 60) return true;
      }
      return false;
    }""", arg=selector, timeout=10_000)


def _legend(page: Page) -> set[str]:
    # 圖例整列都讀、不跳第一格：比較頁的圖例只列兩條線，uPlot 預設的英文「Value」那格若又冒出來就紅。
    return {item.strip() for item in page.locator("#chart .u-legend .u-series").all_inner_texts()}


def _drawn_series(page: Page) -> list[list[object]]:
    # 名字、縱軸刻度、虛線三者綁在同一條線上：A 實線、B 虛線，兩條共用同一把 y 刻度（不准給 B 一把自己的尺）。
    return cast(list[list[object]], page.evaluate(
        "() => plot.series.slice(1).map((s) => [s.label, s.scale, (s.dash || []).length > 0])"))


def _pressed(page: Page) -> list[str | None]:
    return [button.get_attribute("aria-pressed") for button in page.locator("#pair-buttons button").all()]


def _data(page: Page, base: str) -> CompareView:
    response = page.request.get(f"{base}/api/compare/{A_ID}/{B_ID}")
    assert response.ok
    data = response.json()
    # 資料端點省略可空的代價與評估器版本；測試讀回模型時補回空格，不改顯示欄位。
    for row in data["categories"]:
        for side in ("a", "b"):
            row[side].setdefault("cost", None)
            row[side].setdefault("evaluator_version", None)
    return CompareView.model_validate(data)


def _selected(data: CompareView, keys: tuple[str, ...]) -> list[OverlaySeries]:
    return [next(item for item in data.overlay.series if item.key == key)
            for key in keys]


def test_compare_page_draws_default_pair(tmp_path: Path, browser: Browser,
                                         pair: tuple[SchemeResult, SchemeResult]) -> None:
    with _serve(tmp_path) as base:
        _files(tmp_path, pair[0], A_ID)
        _files(tmp_path, pair[1], B_ID)
        with _open(browser, f"{base}/compare/{A_ID}/{B_ID}") as watched:
            page = watched.page
            _has_lines(page)
            data = _data(page, base)
            assert _legend(page) == {item.legend_text for item in
                                     _selected(data, data.overlay.default_keys)}
            a_line, b_line = _selected(data, data.overlay.default_keys)
            assert _drawn_series(page) == [[a_line.legend_text, "y", False], [b_line.legend_text, "y", True]]
            assert sorted(page.evaluate("() => Object.keys(plot.scales)")) == ["x", "y"]
            assert _pressed(page)[0] == "true"
            assert set(_pressed(page)[1:]) <= {"false"}
            # 每一格字要掛在對的那一邊，不只是出現在頁面上。
            assert page.locator("#table-a").inner_text() == f"A：{data.table.a_text}"
            assert page.locator("#table-b").inner_text() == f"B：{data.table.b_text}"
            assert page.locator("#identity-a").inner_text().startswith(f"A：{data.a.scheme_id}；")
            assert page.locator("#identity-b").inner_text().startswith(f"B：{data.b.scheme_id}；")
            assert [row.locator("td").all_inner_texts() for row in page.locator("#changes tr").all()[1:]] == [
                [change.label, change.a_text, change.b_text] for change in data.changes]
            text = page.locator("body").inner_text()
            assert data.summary_text in text
            assert all(value in text for value in (data.table.a_text, data.table.b_text,
                       data.table.reason_text, data.table.calibration_text))
            assert all(value in text for change in data.changes
                       for value in (change.label, change.a_text, change.b_text))
            category_rows = [row.locator("td").all_inner_texts() for row in
                             page.locator("#categories tr").all()[1:]]
            assert category_rows == [[item.label, item.a.state_label, item.a.cost_text,
                                      item.b.state_label, item.b.cost_text,
                                      "；".join(dict.fromkeys(note for note in (item.a.note, item.b.note,
                                                                                item.comparison_text) if note))]
                                     for item in data.categories]
            _assert_text_is_formatted(page)
            _assert_quiet(watched)


def test_compare_plot_data_equals_server_levels(tmp_path: Path, browser: Browser,
                                                pair: tuple[SchemeResult, SchemeResult]) -> None:
    with _serve(tmp_path) as base:
        _files(tmp_path, pair[0], A_ID)
        _files(tmp_path, pair[1], B_ID)
        with _open(browser, f"{base}/compare/{A_ID}/{B_ID}") as watched:
            page = watched.page
            _has_lines(page)
            data = _data(page, base)
            drawn = page.evaluate("() => plot.data")
            expected = _selected(data, data.overlay.default_keys)
            assert drawn[0] == list(data.overlay.frequency_hz)
            assert drawn[1:] == [list(item.levels_db) for item in expected]
            _assert_quiet(watched)


def test_switching_pair_changes_both_lines(tmp_path: Path, browser: Browser,
                                           pair: tuple[SchemeResult, SchemeResult]) -> None:
    with _serve(tmp_path) as base:
        _files(tmp_path, pair[0], A_ID)
        _files(tmp_path, pair[1], B_ID)
        with _open(browser, f"{base}/compare/{A_ID}/{B_ID}") as watched:
            page = watched.page
            data = _data(page, base)
            second = data.overlay.pairs[1]
            page.get_by_role("button", name=second.label).click()
            _has_lines(page)
            assert _legend(page) == {item.legend_text for item in
                                     _selected(data, (second.a_key, second.b_key))}
            a_line, b_line = _selected(data, (second.a_key, second.b_key))
            assert _drawn_series(page) == [[a_line.legend_text, "y", False], [b_line.legend_text, "y", True]]
            assert _pressed(page)[1] == "true"
            assert _pressed(page)[0] == "false"
            assert page.evaluate("() => plot.data")[1:] == [list(item.levels_db) for item in
                _selected(data, (second.a_key, second.b_key))]
            _assert_quiet(watched)


def test_split_tables_show_same_status_and_mark_categories(
        tmp_path: Path, browser: Browser, pair: tuple[SchemeResult, SchemeResult],
        tmp_path_factory: pytest.TempPathFactory, worker_id: str) -> None:
    # 只改主位高度 → 不同表：兩邊狀態一樣、分項表標出身分不同的那幾類，畫面不暗示哪一份壞了或比較好。
    moved = moved_primary_result(tmp_path_factory, worker_id)
    with _serve(tmp_path) as base:
        _files(tmp_path, pair[0], A_ID)
        _files(tmp_path, moved, B_ID)
        with _open(browser, f"{base}/compare/{A_ID}/{B_ID}") as watched:
            page = watched.page
            _has_lines(page)
            data = _data(page, base)
            assert not data.table.same_table
            assert page.locator("#table-a").inner_text() == page.locator("#table-b").inner_text().replace("B：", "A：")
            marked = [row.locator("td").all_inner_texts()[0] for row in page.locator("#categories tr").all()[1:]
                      if "兩邊身分不同，代價不能直接比" in row.inner_text()]
            assert set(marked) == {item.label for item in data.categories if item.comparison_text}
            assert marked
            assert data.table.reason_text in page.locator("body").inner_text()
            _assert_quiet(watched)


def test_problems_show_reasons_without_rerun(tmp_path: Path, browser: Browser,
                                             pair: tuple[SchemeResult, SchemeResult]) -> None:
    altered = pair[0].model_copy(update={"engine_commit": "e53bfae" + "f" * 33})
    altered = altered.model_copy(update={
        "candidate": reevaluate(altered, quality_targets_path=config_path("quality_targets.toml"))})
    with _serve(tmp_path) as base:
        _files(tmp_path, pair[0], A_ID)
        _files(tmp_path, altered, B_ID)
        with _open(browser, f"{base}/compare/{A_ID}/{B_ID}") as watched:
            page = watched.page
            response = page.request.get(f"{base}/api/compare/{A_ID}/{B_ID}")
            assert response.status == 409
            for reason in response.json()["problems"]:
                assert reason in page.locator("#rejection").inner_text()
            # 整頁都不准有重算按鈕（塞進原因那一格也不行）：代號重複這類問題重算解不了。
            assert not page.get_by_role("button", name="用現在的引擎重算這一份").all()
            assert page.locator("#reject-reason").is_visible()
            assert watched.page_errors == []
            assert all("409" in error for error in watched.console_errors)


def test_rejected_side_offers_rerun_for_that_side(tmp_path: Path, browser: Browser,
                                                   pair: tuple[SchemeResult, SchemeResult]) -> None:
    with _serve(tmp_path) as base:
        _files(tmp_path, pair[0], A_ID)
        _files(tmp_path, pair[1], B_ID)
        path = next(path for path in (tmp_path / "results").iterdir() if path.stem == B_ID)
        document = json.loads(path.read_text())
        document["engine_commit"] = "forged"
        path.write_text(json.dumps(document))
        with _open(browser, f"{base}/compare/{A_ID}/{B_ID}") as watched:
            page = watched.page
            response = page.request.get(f"{base}/api/compare/{A_ID}/{B_ID}")
            assert response.status == 409
            text = page.locator("#rejection").inner_text()
            assert "B 讀回被拒收" in text
            assert response.json()["reason"] in text
            requested: list[str] = []
            def capture(route: Route) -> None:
                requested.append(route.request.url)
                assert route.request.method == "POST"
                route.fulfill(status=200, content_type="application/json", body='{"run_id":"started"}')
            page.route("**/api/results/*/rerun", capture)
            page.get_by_role("button", name="用現在的引擎重算這一份").click()
            page.locator("#rerun-state").filter(has_text="started").wait_for()
            assert requested == [f"{base}{response.json()['rerun_url']}"]
            assert watched.page_errors == []
            assert all("409" in error for error in watched.console_errors)


def test_home_page_picks_a_and_b_then_opens_compare(tmp_path: Path, browser: Browser,
                                                      pair: tuple[SchemeResult, SchemeResult]) -> None:
    with _serve(tmp_path) as base:
        _files(tmp_path, pair[0], A_ID)
        _files(tmp_path, pair[1], B_ID)
        with _open(browser, f"{base}/") as watched:
            page = watched.page
            rows = page.locator("#results-list tr")
            rows.filter(has_text="wall-1").get_by_role("button", name="選為 A").click()
            assert page.locator("#compare-link").is_hidden()
            rows.filter(has_text="wall-2").get_by_role("button", name="選為 B").click()
            link = page.get_by_role("link", name="比較")
            assert link.get_attribute("href") == f"/compare/{A_ID}/{B_ID}"
            link.click()
            _has_lines(page)
            _assert_quiet(watched)
