"""比較頁畫面與伺服器資料的一致性。"""
from __future__ import annotations

import json
import math
import struct
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
from tests.engine.test_gui_compare_view import moved_primary_result, shorter_room_result
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
    # 兩份平面圖與共用比例是網頁層另外附的兩格，不在比較資料模型裡；驗模型前先拿掉。
    data.pop("plans", None)
    data.pop("plan_scale_room", None)
    # 資料端點省略可空的代價與評估器版本；測試讀回模型時補回空格，不改顯示欄位。
    for row in data["categories"]:
        for side in ("a", "b"):
            row[side].setdefault("cost", None)
            row[side].setdefault("evaluator_version", None)
    return CompareView.model_validate(data)


def test_compare_plans_side_by_side_on_one_scale(
        tmp_path: Path, browser: Browser, pair: tuple[SchemeResult, SchemeResult],
        tmp_path_factory: pytest.TempPathFactory, worker_id: str) -> None:
    shorter = shorter_room_result(tmp_path_factory, worker_id)
    # 兩種順序都開：大房在 A 或在 B，只拿某一邊的房間當比例的寫法總有一種對不上。
    for a_result, b_result in ((pair[0], shorter), (shorter, pair[0])):
        with _serve(tmp_path / a_result.scheme.scheme_id) as base:
            _files(tmp_path / a_result.scheme.scheme_id, a_result, A_ID)
            _files(tmp_path / a_result.scheme.scheme_id, b_result, B_ID)
            with _open(browser, f"{base}/compare/{A_ID}/{B_ID}") as watched:
                page = watched.page
                page.locator("#plan-a-xy g[data-keys]").first.wait_for()
                for side in ("a", "b"):
                    for plane in ("xy", "xz"):
                        assert page.locator(f"#plan-{side}-{plane} g[data-keys]").count() > 0
                rooms = [result.scheme.scene.room_m for result in (a_result, b_result)]
                for plane, height in (("xy", "Ly"), ("xz", "Lz")):
                    # 平面圖、側面圖都量：A、B 對調或各用自己的房間，比值就對不上。
                    width_texts = [page.locator(f"#plan-{side}-{plane} rect").first.get_attribute("width")
                                   for side in ("a", "b")]
                    assert all(value is not None for value in width_texts)
                    widths = [float(value) for value in width_texts if value is not None]
                    assert math.isclose(widths[0] / widths[1], rooms[0].Lx / rooms[1].Lx, rel_tol=1e-6)
                    # 共用比例取較大那間房：大房剛好塞滿畫框（取小的話大房會超出 520 像素）。
                    largest = max(rooms[0].Lx, rooms[1].Lx), max(getattr(room, height) for room in rooms)
                    scale = min(520 / largest[0], 320 / largest[1])
                    assert math.isclose(max(widths), largest[0] * scale, rel_tol=1e-6)
                _assert_quiet(watched)


def test_changed_points_are_ringed(
        tmp_path: Path, browser: Browser, pair: tuple[SchemeResult, SchemeResult],
        tmp_path_factory: pytest.TempPathFactory, worker_id: str) -> None:
    moved = moved_primary_result(tmp_path_factory, worker_id)
    with _serve(tmp_path) as base:
        _files(tmp_path, pair[0], A_ID)
        _files(tmp_path, moved, B_ID)
        with _open(browser, f"{base}/compare/{A_ID}/{B_ID}") as watched:
            page = watched.page
            page.locator("#plan-a-xy g[data-keys]").first.wait_for()
            data = page.request.get(f"{base}/api/compare/{A_ID}/{B_ID}").json()
            expected = set(data["changed_keys"])
            assert expected == {f"receiver:{pair[0].scheme.receiver_set.primary.receiver_id}"}
            for side in ("a", "b"):
                for plane in ("xy", "xz"):
                    groups = page.locator(f"#plan-{side}-{plane} g[data-keys]")
                    ringed = set(page.locator(f"#plan-{side}-{plane} circle.changed-ring").evaluate_all(
                        "nodes => nodes.flatMap(node => node.dataset.keys.split(' '))"))
                    assert ringed == expected
                    assert groups.count() > len(ringed)
            # 點清單各掛在自己那一欄：主位 z 兩邊不同，對調就對不上。
            for side in ("a", "b"):
                plan = data["plans"][side]
                assert page.locator(f"#plan-{side}-legend li").all_inner_texts() == [
                    f"{item['marker']}－{item['detail_text']}" for item in plan["speakers"] + plan["receivers"]]
            key = next(iter(expected))
            page.locator(f"#plan-a-xy g[data-keys~='{key}']").click()
            assert page.locator("#plan-a-detail").inner_text()
            _assert_quiet(watched)


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


# 匯出前掛在頁面上：只記「沒掛進頁面的那張 canvas」（匯出用的）畫了哪些字、哪些線、貼了哪張圖。
EXPORT_RECORDER_JS = """() => {
  window.exportTexts = [];
  window.exportStrokes = [];
  window.exportImages = [];
  const getContext = HTMLCanvasElement.prototype.getContext;
  HTMLCanvasElement.prototype.getContext = function(...args) {
    const context = getContext.apply(this, args);
    if (!this.isConnected) context.exportCanvas = true;
    return context;
  };
  const original = CanvasRenderingContext2D.prototype.fillText;
  CanvasRenderingContext2D.prototype.fillText = function(text, ...rest) {
    if (this.exportCanvas) window.exportTexts.push(String(text));
    return original.call(this, text, ...rest);
  };
  const stroke = CanvasRenderingContext2D.prototype.stroke;
  CanvasRenderingContext2D.prototype.stroke = function(...args) {
    if (this.exportCanvas) window.exportStrokes.push({
      color: this.strokeStyle, dash: this.getLineDash(), width: this.lineWidth});
    return stroke.apply(this, args);
  };
  const drawImage = CanvasRenderingContext2D.prototype.drawImage;
  CanvasRenderingContext2D.prototype.drawImage = function(image, ...rest) {
    if (this.exportCanvas) window.exportImages.push({
      chart: image === document.querySelector('#chart canvas'), dx: rest[0], dy: rest[1],
      height: this.canvas.height});
    return drawImage.call(this, image, ...rest);
  };
}"""


@pytest.mark.parametrize("device_scale_factor", [1, 2])
def test_png_export_is_png_with_legend_strip(tmp_path: Path, browser: Browser,
                                             pair: tuple[SchemeResult, SchemeResult],
                                             device_scale_factor: float) -> None:
    # 一般螢幕與兩倍倍率的高解析度螢幕各考一次：字、位置跟著倍率放大，線寬跟圖上一樣，虛線照 uPlot 不乘倍率。
    with _serve(tmp_path) as base:
        _files(tmp_path, pair[0], A_ID)
        _files(tmp_path, pair[1], B_ID)
        with _open(browser, f"{base}/compare/{A_ID}/{B_ID}", device_scale_factor) as watched:
            page = watched.page
            data = _data(page, base)
            selected_pair = data.overlay.pairs[-1]
            page.get_by_role("button", name=selected_pair.label).click()
            _has_lines(page)
            source_width, source_height = page.evaluate("""() => {
              const canvas = document.querySelector('#chart canvas');
              return [canvas.width, canvas.height];
            }""")
            page.evaluate(EXPORT_RECORDER_JS)
            with page.expect_download() as event:
                page.get_by_role("button", name="下載曲線圖片（PNG）").click()
            download = event.value
            which = "-".join(selected_pair.a_key.split(":")[1:])
            assert download.suggested_filename == f"compare-{A_ID[:8]}-{B_ID[:8]}-{which}.png"
            content = download.path().read_bytes()
            assert content[:8] == b"\x89PNG\r\n\x1a\n"
            width, height = struct.unpack(">II", content[16:24])
            assert width == source_width
            assert height > source_height
            # 曲線真的貼上去：來源就是圖上那張 canvas，貼在標題下方、圖例上方。
            images = page.evaluate("() => window.exportImages")
            assert [image["chart"] for image in images] == [True]
            assert images[0]["dx"] == 0
            assert 0 < images[0]["dy"] < height - source_height
            # 字照畫的順序：標題、A 圖例、B 圖例、音量基準但書；圖例的字跟線的顏色與線型同一個順序綁在一起。
            a_line, b_line = _selected(data, (selected_pair.a_key, selected_pair.b_key))
            assert page.evaluate("() => window.exportTexts") == [
                f"A：{data.a.scheme_id}　B：{data.b.scheme_id}　{selected_pair.label}",
                a_line.legend_text, b_line.legend_text, data.level_note]
            # 圖例顏色、線型、線寬跟圖上兩條線一樣（uPlot 畫完後 stroke 是函式，要呼叫；線寬乘倍率，虛線不乘）。
            assert page.evaluate("""() => window.exportStrokes.map((line) => [
              line.color, line.dash, line.width])""") == page.evaluate("""() =>
              plot.series.slice(1).map((line, index) => [line.stroke(plot, index + 1),
                                                          line.dash || [], line.width * devicePixelRatio])""")
            _assert_quiet(watched)


def test_export_links_point_to_both_csv(tmp_path: Path, browser: Browser,
                                        pair: tuple[SchemeResult, SchemeResult]) -> None:
    with _serve(tmp_path) as base:
        _files(tmp_path, pair[0], A_ID)
        _files(tmp_path, pair[1], B_ID)
        with _open(browser, f"{base}/compare/{A_ID}/{B_ID}") as watched:
            page = watched.page
            page.get_by_role("link", name="下載頻響資料（CSV）").wait_for()
            # 連結字對網址逐條比：兩條互換也要抓得到。
            links = {link.inner_text(): link.get_attribute("href")
                     for link in page.locator("#compare-exports a").all()}
            assert links == {"下載頻響資料（CSV）": f"/api/compare/{A_ID}/{B_ID}/export/curves",
                             "下載摘要與分項（CSV）": f"/api/compare/{A_ID}/{B_ID}/export/summary"}
            assert page.locator("#level-note").inner_text() == _data(page, base).level_note
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
