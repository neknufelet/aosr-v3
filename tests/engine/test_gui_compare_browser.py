"""比較頁畫面與伺服器資料的一致性。"""
from __future__ import annotations

import json
import math
import re
import struct
import sys
from pathlib import Path
from typing import cast

import pytest
from playwright.sync_api import Browser, Page, Route

from aosr.gui.compare_view import CATEGORY_HEADINGS, CompareView, OverlayPair, OverlaySeries
from aosr.reporting.compare import comparison_problems
from aosr.reporting.result import SchemeResult, save_result
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


def _pressed(page: Page, group: str) -> dict[str, str | None]:
    """聲道或位置那一組按鈕（group 是 channel 或 position）：按鈕字對到有沒有按下。"""
    return {button.inner_text(): button.get_attribute("aria-pressed")
            for button in page.locator(f"#{group}-buttons button").all()}


def _names(data: CompareView) -> tuple[dict[str, str], dict[str, str]]:
    return ({item.key: item.label for item in data.overlay.channels},
            {item.key: item.label for item in data.overlay.positions})


def _choose(page: Page, data: CompareView, chosen: OverlayPair) -> None:
    """用聲道切換與位置按鈕挑一對：先按聲道再按位置。按名字要 exact，不然「主位」會按到「主位前方」。"""
    channels, positions = _names(data)
    page.locator("#channel-buttons").get_by_role("button", name=channels[chosen.channel], exact=True).click()
    page.locator("#position-buttons").get_by_role("button", name=positions[chosen.position], exact=True).click()


def _assert_pressed(page: Page, data: CompareView, chosen: OverlayPair) -> None:
    """選到的那一對：它的聲道、位置按下，同組其他都沒按下。"""
    channels, positions = _names(data)
    assert _pressed(page, "channel") == {label: str(key == chosen.channel).lower()
                                         for key, label in channels.items()}
    assert _pressed(page, "position") == {label: str(key == chosen.position).lower()
                                          for key, label in positions.items()}


def _data(page: Page, base: str) -> CompareView:
    response = page.request.get(f"{base}/api/compare/{A_ID}/{B_ID}")
    assert response.ok
    data = response.json()
    # 兩份平面圖與共用比例是網頁層另外附的兩格，不在比較資料模型裡；驗模型前先拿掉。
    data.pop("plans", None)
    data.pop("plan_scale_room", None)
    data.pop("outdated_schemes", None)
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


def test_opening_home_results_and_compare_pages_starts_no_calculation(
        tmp_path: Path, browser: Browser, pair: tuple[SchemeResult, SchemeResult]) -> None:
    """網頁程式一載入就自己送請求：打開首頁、兩份結果頁、比較頁，以及會掛出重算按鈕的兩張頁
    （舊格式結果被拒收、兩份計算指紋不同不能比），等網路靜下來，計算資料夾不准多一筆。"""
    old_id, other_id = "d" * 32, "e" * 32
    script = tmp_path / "must-not-run.py"
    script.write_text("raise SystemExit(9)\n")
    with _serve(tmp_path, (sys.executable, str(script))) as base:
        _files(tmp_path, pair[0], A_ID)
        _files(tmp_path, pair[1], B_ID)
        # 拒收頁要帶得出有效方案，重算按鈕才真的起得了計算；只改格式版本讓它被當舊格式拒收。
        document = json.loads(pair[1].model_dump_json())
        document["schema_version"] = "aosr.scheme_result.v2"
        (tmp_path / "results" / f"{old_id}.json").write_text(json.dumps(document))
        other = pair[1].model_copy(update={"calculation_fingerprint": "calc-v1:" + "2" * 64})
        save_result(other, tmp_path / "results" / f"{other_id}.json")
        before = sorted(path.name for path in (tmp_path / "runs").iterdir())
        for url in ("/", f"/results/{A_ID}", f"/results/{B_ID}", f"/compare/{A_ID}/{B_ID}"):
            with _open(browser, base + url) as watched:
                # 首頁沒有拒收區塊；locator 找不到東西時 is_visible 回假，一起涵蓋。
                assert not watched.page.locator("#rejection").is_visible(), url
                _assert_quiet(watched)
        for url in (f"/results/{old_id}", f"/compare/{A_ID}/{other_id}"):
            with _open(browser, base + url) as watched:
                # 伺服器回 409，瀏覽器會記一條載入失敗；這兩頁本來就該顯示拒收，不查瀏覽器錯誤訊息。
                assert watched.page.locator("#rejection").is_visible(), url
                assert watched.page_errors == [], url
        assert sorted(path.name for path in (tmp_path / "runs").iterdir()) == before


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


def _assert_summary_sentences(page: Page, data: CompareView) -> None:
    """摘要那幾句各在自己的位置：哪一份比較好、複核警戒與還不是最終推薦、兩份都尚未評估的類；
    三項設定核對都相同時核對卡只寫一句話，不畫三列「相同」的表。"""
    assert page.locator("#table-verdict").inner_text() == data.table.verdict_text
    assert "decided" in str(page.locator("#table-verdict").get_attribute("class"))
    assert page.locator("#table-review").inner_text() == data.table.review_text
    assert "不能當最終推薦" in data.table.review_text
    assert page.locator("#pending-text").inner_text() == data.pending_text
    # （考卷的兩份都是舊程式算的，下面另有幾句舊程式的說明；核對那一句是第一段。）
    assert data.fingerprints_text
    assert page.locator("#fingerprints p").first.inner_text() == data.fingerprints_text
    assert not page.locator("#fingerprints table").all()


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
            default = next(item for item in data.overlay.pairs
                           if (item.a_key, item.b_key) == data.overlay.default_keys)
            _assert_pressed(page, data, default)
            # 每一格字要掛在對的那一邊，不只是出現在頁面上。
            assert page.locator("#table-a").inner_text() == f"A：{data.table.a_text}"
            assert page.locator("#table-b").inner_text() == f"B：{data.table.b_text}"
            # 頁首主畫面：哪個方案、哪天算的、算多久，加一句「計算版本：兩份相同」（伺服器判）；
            # 計算指紋與程式提交代號是老闆看不懂的碼，只放在摺起來的技術細節，打開才看得到。
            for side, identity in (("a", data.a), ("b", data.b)):
                assert page.locator(f"#identity-{side}").inner_text() == (
                    f"{side.upper()}：{identity.scheme_id}；計算日期 {identity.run_date}；"
                    f"計算時間（全程）{identity.total_text}")
            assert page.locator("#version-text").inner_text() == data.version_text == "兩份相同"
            technical = page.locator("#identity-technical")
            assert technical.evaluate("el => el.open") is False
            assert not re.search(r"[0-9a-f]{7,}", page.locator("header").inner_text())
            codes = technical.text_content() or ""
            assert all(code in codes for code in (data.a.fingerprint_text, data.b.fingerprint_text,
                                                  data.a.engine_text, data.b.engine_text))
            # 碼的名字跟結果頁、方案輸入頁同一套：計算指紋前 12 碼、程式提交代號。
            assert f"計算指紋前 12 碼 {data.a.fingerprint_text}、程式提交代號 {data.a.engine_text}" in codes
            technical.locator("summary").click()
            assert data.a.fingerprint_text in page.locator("header").inner_text()
            # 回去的連結跟其他頁同一套字：輸入頁叫方案輸入頁。
            assert page.get_by_role("link", name="回方案輸入頁").get_attribute("href") == "/"
            assert [row.locator("td").all_inner_texts() for row in page.locator("#changes tr").all()[1:]] == [
                [change.label, change.a_text, change.b_text] for change in data.changes]
            text = page.locator("body").inner_text()
            assert data.summary_text in text
            assert all(value in text for value in (data.table.a_text, data.table.b_text,
                       data.table.reason_text, data.table.calibration_text))
            _assert_summary_sentences(page, data)
            assert all(value in text for change in data.changes
                       for value in (change.label, change.a_text, change.b_text))
            # 分項表跟摘要 CSV 同一套欄位；說明欄每一類都空著時不畫（CSV 照樣有）；「哪一份較好」照伺服器給的字。
            assert not [item for item in data.categories if item.note_text]
            assert page.locator("#categories th").all_inner_texts() == list(CATEGORY_HEADINGS[:-1])
            category_rows = [row.locator("td").all_inner_texts() for row in
                             page.locator("#categories tr").all()[1:]]
            assert category_rows == [[item.label, item.a.state_label, item.a.cost_text,
                                      item.b.state_label, item.b.cost_text, item.better_text]
                                     for item in data.categories]
            # 比較好的那一邊（伺服器判）加上標記：總代價框與分項表那一格代價。
            assert {side for side in ("a", "b")
                    if "better" in str(page.locator(f"#table-{side}").get_attribute("class"))} == (
                {data.table.better} & {"a", "b"})
            marked = [[index for index, cell in enumerate(row.locator("td").all())
                       if "better" in str(cell.get_attribute("class"))]
                      for row in page.locator("#categories tr").all()[1:]]
            assert marked == [[{"a": 2, "b": 4}[item.better]] if item.better in {"a", "b"} else []
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
    if (!this.isConnected) { context.exportCanvas = true; window.exportCanvas = this; }
    return context;
  };
  const original = CanvasRenderingContext2D.prototype.fillText;
  CanvasRenderingContext2D.prototype.fillText = function(text, x, y, ...rest) {
    if (this.exportCanvas) window.exportTexts.push({text: String(text), x, y, font: this.font,
      width: this.measureText(String(text)).width});
    return original.call(this, text, x, y, ...rest);
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


def _assert_export_texts(texts: list[dict[str, object]], data: CompareView, selected_pair: OverlayPair,
                         *, top: float, bottom: float, width: int, height: int, scale: float) -> None:
    """匯出圖上的字：照畫的順序接起來等於標題、A 圖例、B 圖例、但書（換行不掉字，字跟線同一個順序）；
    每一筆都在畫布內、字級跟著倍率；標題在曲線上方，圖例與但書在曲線下方。"""
    a_line, b_line = _selected(data, (selected_pair.a_key, selected_pair.b_key))
    title = f"A：{data.a.scheme_id}　B：{data.b.scheme_id}　{selected_pair.label}"
    assert "".join(str(item["text"]) for item in texts) == (
        title + a_line.legend_text + b_line.legend_text + data.level_note)
    for item in texts:
        x, y, measured = (float(cast(float, item[key])) for key in ("x", "y", "width"))
        assert 0 <= x and x + measured <= width, item
        size = float(str(item["font"]).split("px")[0])
        assert size >= 9 * scale, item
        # 上下也要在畫布內：基線加字腳（約四分之一字級）不超過圖片底部；區塊高度是另一段算的，最容易對不上。
        assert y + size * 0.25 <= height, item
    title_count = 0
    joined = ""
    for item in texts:
        if joined == title:
            break
        joined += str(item["text"])
        title_count += 1
    assert all(float(cast(float, item["y"])) < top for item in texts[:title_count])
    assert all(float(cast(float, item["y"])) > bottom for item in texts[title_count:])


# 匯出圖上曲線那一段有沒有曲線顏色（照 LINES_DRAWN 的判法：紅綠藍最大減最小超過 60）。
EXPORT_HAS_CURVE_JS = """([top, bottom]) => {
  const canvas = window.exportCanvas;
  const data = canvas.getContext("2d").getImageData(0, top, canvas.width, bottom - top).data;
  for (let i = 0; i < data.length; i += 4) {
    if (Math.max(data[i], data[i + 1], data[i + 2]) - Math.min(data[i], data[i + 1], data[i + 2]) > 60) return true;
  }
  return false;
}"""


@pytest.mark.parametrize(("device_scale_factor", "viewport_width"), [(1, 1400), (2, 1400), (2, 390)])
def test_png_export_is_png_with_legend_strip(tmp_path: Path, browser: Browser,
                                             pair: tuple[SchemeResult, SchemeResult],
                                             device_scale_factor: float, viewport_width: int) -> None:
    # 一般螢幕、兩倍倍率、兩倍倍率的窄畫面各考一次：字與位置跟著倍率放大、放不下就換行不切掉，
    # 線寬跟圖上一樣，虛線照 uPlot 不乘倍率。
    with _serve(tmp_path) as base:
        _files(tmp_path, pair[0], A_ID)
        _files(tmp_path, pair[1], B_ID)
        with _open(browser, f"{base}/compare/{A_ID}/{B_ID}", device_scale_factor,
                   viewport_width) as watched:
            page = watched.page
            data = _data(page, base)
            selected_pair = data.overlay.pairs[-1]
            _choose(page, data, selected_pair)
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
            assert page.evaluate(EXPORT_HAS_CURVE_JS, [images[0]["dy"], images[0]["dy"] + source_height])
            _assert_export_texts(page.evaluate("() => window.exportTexts"), data, selected_pair,
                                 top=images[0]["dy"], bottom=images[0]["dy"] + source_height,
                                 width=width, height=height, scale=device_scale_factor)
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
            _choose(page, data, second)
            _has_lines(page)
            assert _legend(page) == {item.legend_text for item in
                                     _selected(data, (second.a_key, second.b_key))}
            a_line, b_line = _selected(data, (second.a_key, second.b_key))
            assert _drawn_series(page) == [[a_line.legend_text, "y", False], [b_line.legend_text, "y", True]]
            _assert_pressed(page, data, second)
            assert page.evaluate("() => plot.data")[1:] == [list(item.levels_db) for item in
                _selected(data, (second.a_key, second.b_key))]
            _assert_quiet(watched)


def test_every_pair_is_reachable_by_channel_and_position(tmp_path: Path, browser: Browser,
                                                         pair: tuple[SchemeResult, SchemeResult]) -> None:
    # 聲道切換加位置按鈕取代一整排配對按鈕：每一對都挑得到，A、B 兩條線一起換，按鈕寫顯示名。
    with _serve(tmp_path) as base:
        _files(tmp_path, pair[0], A_ID)
        _files(tmp_path, pair[1], B_ID)
        with _open(browser, f"{base}/compare/{A_ID}/{B_ID}") as watched:
            page = watched.page
            _has_lines(page)
            data = _data(page, base)
            channels, positions = _names(data)
            assert set(_pressed(page, "channel")) == set(channels.values())
            assert set(_pressed(page, "position")) == set(positions.values())
            assert not page.locator("#pair-buttons").all()
            for chosen in reversed(data.overlay.pairs):
                _choose(page, data, chosen)
                a_line, b_line = _selected(data, (chosen.a_key, chosen.b_key))
                assert _drawn_series(page) == [[a_line.legend_text, "y", False],
                                               [b_line.legend_text, "y", True]]
                assert page.evaluate("() => plot.data")[1:] == [list(a_line.levels_db),
                                                                list(b_line.levels_db)]
                _assert_pressed(page, data, chosen)
            _assert_quiet(watched)


def test_missing_combination_is_disabled_and_channel_switch_keeps_a_pair(
        tmp_path: Path, browser: Browser, pair: tuple[SchemeResult, SchemeResult]) -> None:
    # 某個聲道在某個位置沒有兩份都有的線：那個位置按鈕在那個聲道下不能按；換到那個聲道時退回它有的那一對。
    with _serve(tmp_path) as base:
        _files(tmp_path, pair[0], A_ID)
        _files(tmp_path, pair[1], B_ID)
        with _open(browser, f"{base}/compare/{A_ID}/{B_ID}") as watched:
            page = watched.page
            data = _data(page, base)
            dropped = next(item for item in data.overlay.pairs
                           if item.channel == "right" and item.position != "primary")

            def fewer_pairs(route: Route) -> None:
                body = route.fetch().json()
                body["overlay"]["pairs"] = [item for item in body["overlay"]["pairs"]
                                            if (item["channel"], item["position"]) !=
                                            (dropped.channel, dropped.position)]
                route.fulfill(status=200, content_type="application/json", body=json.dumps(body))
            page.route(f"**/api/compare/{A_ID}/{B_ID}", fewer_pairs)
            page.reload(wait_until="networkidle")
            _has_lines(page)
            channels, positions = _names(data)
            left_there = next(item for item in data.overlay.pairs
                              if item.channel == "left" and item.position == dropped.position)
            _choose(page, data, left_there)
            page.locator("#channel-buttons").get_by_role(
                "button", name=channels["right"], exact=True).click()
            fallback = next(item for item in data.overlay.pairs
                            if item.channel == "right" and item != dropped)
            _assert_pressed(page, data, fallback)
            assert page.locator("#position-buttons").get_by_role(
                "button", name=positions[dropped.position], exact=True).is_disabled()
            assert _legend(page) == {item.legend_text for item in
                                     _selected(data, (fallback.a_key, fallback.b_key))}
            _assert_quiet(watched)


def test_channel_switch_keeps_the_position(tmp_path: Path, browser: Browser,
                                          pair: tuple[SchemeResult, SchemeResult]) -> None:
    # 換聲道時位置不動：看著「左聲道・主位前方」按右聲道，要換成「右聲道・主位前方」，不是跳回那個聲道的第一對。
    with _serve(tmp_path) as base:
        _files(tmp_path, pair[0], A_ID)
        _files(tmp_path, pair[1], B_ID)
        with _open(browser, f"{base}/compare/{A_ID}/{B_ID}") as watched:
            page = watched.page
            _has_lines(page)
            data = _data(page, base)
            channels, _ = _names(data)
            first = {channel: next(item for item in data.overlay.pairs if item.channel == channel)
                     for channel in channels}
            # 挑一個兩個聲道都有、而且不是任一聲道第一對的位置：退回第一對的寫法就分得出來。
            position = next(key for key in (item.position for item in data.overlay.pairs)
                            if all(any(item.channel == channel and item.position == key
                                       for item in data.overlay.pairs) for channel in channels)
                            and key not in {item.position for item in first.values()})
            by_channel = {channel: next(item for item in data.overlay.pairs
                                        if item.channel == channel and item.position == position)
                          for channel in channels}
            for start, target in (("left", "right"), ("right", "left")):
                _choose(page, data, by_channel[start])
                page.locator("#channel-buttons").get_by_role(
                    "button", name=channels[target], exact=True).click()
                chosen = by_channel[target]
                _assert_pressed(page, data, chosen)
                assert _legend(page) == {item.legend_text for item in
                                         _selected(data, (chosen.a_key, chosen.b_key))}
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
                      if "評分條件不同，這一類代價不能直接比" in row.inner_text()]
            assert set(marked) == {item.label for item in data.categories if item.comparison_text}
            assert marked
            assert data.table.reason_text in page.locator("body").inner_text()
            # 不同表不判哪一份比較好：那一句與複核那句都不出現，兩個總代價框都不加標記。
            assert page.locator("#table-verdict").is_hidden()
            assert page.locator("#table-review").is_hidden()
            assert not page.locator("#totals .better").all()
            # 有說明的類：分項表畫說明欄（跟 CSV 同一套欄位）。
            assert page.locator("#categories th").all_inner_texts() == list(CATEGORY_HEADINGS)
            # 設定核對有不同：畫表，每一格寫白話（不同時說哪一類改了），不印雜湊。
            assert [row.locator("td").all_inner_texts() for row in page.locator("#fingerprints tr").all()[1:]] == [
                [item.label, item.text] for item in data.fingerprints]
            assert not re.search(r"[0-9a-f]{7}", page.locator("#compare-top").inner_text())
            _assert_quiet(watched)


def test_problems_show_reasons_and_outdated_side_rerun(
        tmp_path: Path, browser: Browser, pair: tuple[SchemeResult, SchemeResult],
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("aosr.gui.app.calculation_fingerprint", lambda **kwargs:
                        pair[0].calculation_fingerprint)
    altered = pair[0].model_copy(update={"calculation_fingerprint": "calc-v1:" + "1" * 64})
    with _serve(tmp_path) as base:
        _files(tmp_path, altered, A_ID)
        _files(tmp_path, pair[0], B_ID)
        with _open(browser, f"{base}/compare/{A_ID}/{B_ID}") as watched:
            page = watched.page
            response = page.request.get(f"{base}/api/compare/{A_ID}/{B_ID}")
            assert response.status == 409
            assert any("計算指紋" in reason for reason in response.json()["problems"])
            # 主畫面是 A、B 的白話；伺服器的原句（含雜湊）收在摺起來的技術細節裡。
            assert page.locator("#rejection-title").inner_text() == "這兩份不能直接比較"
            shown = page.locator("#reject-reason").inner_text()
            assert "A 和 B 的計算版本不同" in shown
            assert not re.search(r"[0-9a-f]{7,}|計算指紋|第 1 份", shown)
            folded = page.locator("#reject-reason details.technical")
            assert folded.evaluate("el => el.open") is False
            assert all(reason in (folded.text_content() or "") for reason in response.json()["problems"])
            assert "A（wall-1）是用舊程式算的" in page.locator("#reject-reason").inner_text()
            assert "B（wall-1）是用舊程式算的" not in page.locator("#reject-reason").inner_text()
            assert not page.locator("#rerun-holder button").all()
            assert page.locator("#reject-reason").is_visible()
            assert watched.page_errors == []
            assert all("409" in error for error in watched.console_errors)


def test_duplicate_scheme_and_repairable_fingerprint_buttons(
        tmp_path: Path, browser: Browser, pair: tuple[SchemeResult, SchemeResult],
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("aosr.gui.app.calculation_fingerprint", lambda **kwargs:
                        pair[0].calculation_fingerprint)
    with _serve(tmp_path) as base:
        _files(tmp_path, pair[0], A_ID)
        _files(tmp_path, pair[0], B_ID)
        with _open(browser, f"{base}/compare/{A_ID}/{B_ID}") as watched:
            page = watched.page
            page.locator("#rejection").wait_for(state="visible")
            assert "A 和 B 的方案代號相同" in page.locator("#reject-reason").inner_text()
            assert "候選代號重複" in (page.locator("#reject-reason details.technical").text_content() or "")
            assert not page.locator("#rerun-holder button").all()
        different = pair[1].model_copy(update={"calculation_fingerprint": "calc-v1:" + "1" * 64})
        _files(tmp_path, different, B_ID)
        with _open(browser, f"{base}/compare/{A_ID}/{B_ID}") as watched:
            page = watched.page
            page.locator("#rejection").wait_for(state="visible")
            assert "B（wall-2）是用舊程式算的" in page.locator("#reject-reason").inner_text()
            assert page.get_by_role("button", name="用現在的程式重算 B 這一份").is_visible()
            assert not page.get_by_role("button", name="用現在的程式重算 A 這一份").all()


def test_successful_comparison_still_names_both_old_results(
        tmp_path: Path, browser: Browser, pair: tuple[SchemeResult, SchemeResult],
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("aosr.gui.app.calculation_fingerprint", lambda **kwargs:
                        "calc-v1:" + "1" * 64)
    with _serve(tmp_path) as base:
        _files(tmp_path, pair[0], A_ID)
        _files(tmp_path, pair[1], B_ID)
        with _open(browser, f"{base}/compare/{A_ID}/{B_ID}") as watched:
            page = watched.page
            page.locator("#content").wait_for(state="visible")
            text = page.locator("#fingerprints").inner_text()
            assert "A（wall-1）是用舊程式算的" in text
            assert "B（wall-2）是用舊程式算的" in text
            assert "兩份是同一版舊程式算的，彼此可以比較" in text


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
            # 主畫面一句白話說是哪一份、怎麼辦；伺服器的技術原因收在摺起來的技術細節。
            assert page.locator("#rejection-title").inner_text() == "B 那一份現在的程式讀不了"
            text = page.locator("#rejection").inner_text()
            assert "B 那份結果檔的內容跟現在的程式對不上" in text
            assert "讀回被拒收" not in text and response.json()["reason"] not in text
            assert response.json()["reason"] in (
                page.locator("#reject-reason details.technical").text_content() or "")
            requested: list[str] = []
            # 第一次按：結果檔在頁面開著時被移走（伺服器 404 只回計算代號）→ 印一句白話、不印代號，按鈕還能再按；
            # 第二次按：檔案讀不懂（伺服器回英文原因）→ 白話一句，英文收進摺起來的重算技術細節；第三次按：開始重算。
            english = "Expecting value: line 1 column 1 (char 0)"
            replies = [(404, json.dumps({"error": B_ID})), (400, json.dumps({"error": english})),
                       (200, '{"run_id":"started"}')]
            def capture(route: Route) -> None:
                requested.append(route.request.url)
                assert route.request.method == "POST"
                status, body = replies.pop(0)
                route.fulfill(status=status, content_type="application/json", body=body)
            page.route("**/api/results/*/rerun", capture)
            button = page.get_by_role("button", name="用現在的程式重算 B 這一份")
            button.click()
            page.locator("#rerun-state").filter(has_text="重算沒有開始").wait_for()
            assert page.locator("#rerun-state").inner_text() == (
                "重算沒有開始：這份結果檔找不到（可能已被移走），請回方案輸入頁的結果清單重新選")
            assert button.is_enabled()
            button.click()
            page.locator("#rerun-state").filter(has_text="這份結果檔讀不了").wait_for()
            assert english not in page.locator("#rejection").inner_text()
            assert english in (page.locator("#rerun-detail details.technical").text_content() or "")
            assert button.is_enabled()
            button.click()
            page.locator("#rerun-state").filter(has_text="已開始重算").wait_for()
            assert not page.locator("#rerun-detail > *").all()
            rerun_url = f"{base}{response.json()['rerun_url']}"
            assert requested == [rerun_url, rerun_url, rerun_url]
            # 開始之後按鈕停用（不會重複開好幾份），不印計算代號，寫明算完去哪裡找。
            assert button.is_disabled()
            state = page.locator("#rerun-state").inner_text()
            assert "started" not in state and not re.search(r"[0-9a-f]{32}", state)
            assert "回方案輸入頁的結果清單" in state
            assert watched.page_errors == []
            assert all(any(code in error for code in ("409", "404", "400")) for error in watched.console_errors)


def test_rerun_that_fails_the_check_prints_the_plain_problem_lines(
        tmp_path: Path, browser: Browser, pair: tuple[SchemeResult, SchemeResult]) -> None:
    # 重算被現在的檢查擋下（B 那份的方案裡地板阻抗是負的）：逐條印伺服器寫好的白話（表單上的中文欄名加說明），
    # 一條一行，不印英文路徑；開頭一句寫去哪裡改。按鈕不停用，改好之後還能再按。
    with _serve(tmp_path) as base:
        _files(tmp_path, pair[0], A_ID)
        _files(tmp_path, pair[1], B_ID)
        path = next(path for path in (tmp_path / "results").iterdir() if path.stem == B_ID)
        document = json.loads(path.read_text())
        document["scheme"]["scene"]["impedance_pa_s_per_m_by_wall"]["floor"] = -1.0
        path.write_text(json.dumps(document))
        with _open(browser, f"{base}/compare/{A_ID}/{B_ID}", viewport_width=1440) as watched:
            page = watched.page
            button = page.get_by_role("button", name="用現在的程式重算 B 這一份")
            with page.expect_response(f"**/api/results/{B_ID}/rerun") as answer:
                button.click()
            assert answer.value.status == 422
            problems = [item["text"] for item in answer.value.json()["problems"]]
            assert problems
            state = page.locator("#rerun-state")
            state.filter(has_text="過不了現在的檢查").wait_for()
            lines = state.inner_text().split("\n")
            assert lines == ["這份結果的方案過不了現在的檢查；請在方案輸入頁打開這個方案、"
                             "改好下面幾項，另存新名字再算：", *problems]
            assert any(line.startswith("地板阻抗：") for line in problems)
            assert not re.search(r"scene|impedance|[a-z]+_[a-z]+", state.inner_text())
            assert button.is_enabled()
            assert watched.page_errors == []
            assert all("409" in error or "422" in error for error in watched.console_errors)


# 平面圖上畫出來的記號長什麼樣（A 的平面圖）與說明裡的小圖長什麼樣：填色、線色、虛線、圓圈。
DRAWN_MARKS_JS = """() => {
  const svg = document.getElementById('plan-a-xy');
  const dot = (prefix) => svg.querySelector(`g[data-keys^="${prefix}"] circle:not(.changed-ring)`);
  const zone = [...svg.querySelectorAll('rect')].find((rect) => rect.getAttribute('stroke-dasharray'));
  const ring = svg.querySelector('.changed-ring');
  const line = svg.querySelector('line');
  return {speaker: dot('speaker:').getAttribute('fill'), seat: dot('receiver:').getAttribute('fill'),
          aim: line && line.getAttribute('stroke'), zone: zone && zone.getAttribute('stroke'),
          ring: ring && [ring.getAttribute('stroke'), ring.getAttribute('fill')]};
}"""
KEY_MARKS_JS = """() => {
  const mark = (name, selector, attribute) => document.querySelector(
    `#plan-key li[data-mark="${name}"] svg ${selector}`).getAttribute(attribute);
  return {speaker: mark('speaker', 'circle', 'fill'), seat: mark('seat', 'circle', 'fill'),
          aim: mark('aim', 'line', 'stroke'), zone: mark('zone', 'rect[stroke-dasharray]', 'stroke'),
          ring: [mark('ring', 'circle[fill="none"]', 'stroke'), 'none']};
}"""


def _shown_marks(page: Page) -> set[str]:
    return {str(item.get_attribute("data-mark")) for item in page.locator("#plan-key li").all()
            if item.is_visible()}


def test_plan_key_says_what_the_marks_on_the_plans_are(
        tmp_path: Path, browser: Browser, pair: tuple[SchemeResult, SchemeResult],
        tmp_path_factory: pytest.TempPathFactory, worker_id: str) -> None:
    # 位置並排下面那段說明以前寫「有外框的那一格」：主位外面的虛線方框其實是聆聽區的範圍，改過的點是外面多一個圓圈。
    # 說明一個記號一行，前面的小圖跟圖上畫的同一個顏色、同一種線；圖上沒畫的記號（沒改動就沒有圓圈、
    # 全向聲源沒有指向線）不列。
    moved = moved_primary_result(tmp_path_factory, worker_id)
    with _serve(tmp_path / "moved") as base:
        _files(tmp_path / "moved", pair[0], A_ID)
        _files(tmp_path / "moved", moved, B_ID)
        with _open(browser, f"{base}/compare/{A_ID}/{B_ID}", viewport_width=1440) as watched:
            page = watched.page

            plans: dict[str, dict[str, list[dict[str, str]]]] = {}

            def aimed(route: Route) -> None:
                # 測試方案都是全向聲源（沒有指向線）；把兩份喇叭都改成對準主位，才畫得出指向線來比顏色。
                body = route.fetch().json()
                plans.update(body["plans"])
                for plan in body["plans"].values():
                    primary = next(item["point"] for item in plan["receivers"] if item["role"] == "primary")
                    for speaker in plan["speakers"]:
                        speaker["aim"] = primary
                route.fulfill(status=200, content_type="application/json", body=json.dumps(body))
            page.route(f"**/api/compare/{A_ID}/{B_ID}", aimed)
            page.reload(wait_until="networkidle")
            page.locator("#plan-a-xy .changed-ring").first.wait_for()
            assert _data(page, base).changed_keys
            assert _shown_marks(page) == {"speaker", "aim", "seat", "zone", "ring"}
            assert page.evaluate(KEY_MARKS_JS) == page.evaluate(DRAWN_MARKS_JS)
            texts = {str(item.get_attribute("data-mark")): item.inner_text()
                     for item in page.locator("#plan-key li").all()}
            assert texts["zone"].startswith("虛線方框是聆聽區")
            assert texts["ring"].startswith("外面多套一個圓圈的點：兩份不一樣的喇叭或座位")
            section = page.locator("section").filter(has=page.locator("#plan-key")).inner_text()
            assert "外框" not in section
            # 沒有標字的藍點：說明寫是其他座位或有改到的周圍點；拿四張圖上真的畫出來、沒有字的點對伺服器給的座位角色。
            assert "沒有標字的是其他座位，或有改到的周圍點" in texts["seat"]
            roles = {item["key"]: item["role"] for plan in plans.values() for item in plan["receivers"]}
            changed = set(_data(page, base).changed_keys)
            unlabeled = cast(list[list[str]], page.evaluate(
                "() => [...document.querySelectorAll('.compare-drawings svg g[data-keys]')]"
                ".filter((g) => !g.querySelector('text')).map((g) => g.dataset.keys.split(' '))"))
            assert all(roles[key] == "other_seat" or (roles[key] == "surrounding" and key in changed)
                       for keys in unlabeled for key in keys)
            # 併成一點的例子寫條件，不寫成固定的事（L、R 的 x 座標和高度不一樣時側面圖不會併）。
            assert "L 和 R 的 x 座標和高度都一樣時，側面圖上會併成 L／R" in section
            _assert_quiet(watched)
    # 只改牆面材料：沒有改動的點就沒有圓圈；全向聲源沒有指向線。說明也不列這兩種。
    with _serve(tmp_path / "walls") as base:
        _files(tmp_path / "walls", pair[0], A_ID)
        _files(tmp_path / "walls", pair[1], B_ID)
        with _open(browser, f"{base}/compare/{A_ID}/{B_ID}", viewport_width=1440) as watched:
            page = watched.page
            page.locator("#plan-a-xy g[data-keys]").first.wait_for()
            assert not page.locator(".compare-drawings svg .changed-ring, .compare-drawings svg line").all()
            assert _shown_marks(page) == {"speaker", "seat", "zone"}
            _assert_quiet(watched)


def test_old_format_side_shows_plain_sentence_and_rerun(tmp_path: Path, browser: Browser,
                                                       pair: tuple[SchemeResult, SchemeResult]) -> None:
    # 舊格式那一份：畫面寫伺服器給的白話（句首是哪一份），不貼欄位名 schema_version；重算按鈕寫明是哪一份。
    with _serve(tmp_path) as base:
        _files(tmp_path, pair[0], A_ID)
        _files(tmp_path, pair[1], B_ID)
        path = next(path for path in (tmp_path / "results").iterdir() if path.stem == B_ID)
        document = json.loads(path.read_text())
        document["schema_version"] = "aosr.scheme_result.v2"
        path.write_text(json.dumps(document))
        with _open(browser, f"{base}/compare/{A_ID}/{B_ID}") as watched:
            page = watched.page
            response = page.request.get(f"{base}/api/compare/{A_ID}/{B_ID}")
            assert response.status == 409 and response.json()["reason_kind"] == "old_format"
            page.locator("#rejection").wait_for(state="visible")
            assert page.locator("#reject-reason").inner_text() == response.json()["reason_text"]
            assert page.locator("#rejection-title").inner_text() == "要先重算才能比較"
            assert "schema_version" not in page.locator("#rejection").inner_text()
            assert page.get_by_role("button", name="用現在的程式重算 B 這一份").is_visible()
            assert watched.page_errors == []
            assert all("409" in error for error in watched.console_errors)


def test_rejection_page_says_each_problem_in_plain_words(
        tmp_path: Path, browser: Browser, pair: tuple[SchemeResult, SchemeResult]) -> None:
    # 不能直接比的原因拿比較層真的拒收理由餵頁面（同一個方案代號、用途不同、聲道設定不同、計算指紋不同，
    # 再加一句頁面不認得的）：主畫面每種一句 A、B 的白話，不印英文欄名、雜湊、「第 1 份」與「計算指紋」；
    # 原句收在摺起來的技術細節。
    scheme = pair[0].scheme
    altered = pair[0].model_copy(update={
        "scheme": scheme.model_copy(update={
            "purpose": "other_purpose",
            "channel_group": scheme.channel_group.model_copy(update={"feature_match_tolerance_hz": 99.0})}),
        "calculation_fingerprint": "calc-v1:" + "1" * 64})
    problems = [*comparison_problems((pair[0], altered)), "第 2 份 wall-1 的 some_field 不同：abcdef1／1234567"]
    body = {"problems": problems, "rerun_urls": {}, "outdated_sides": ["b"],
            "outdated_schemes": {"b": "wall-1"}}
    with _serve(tmp_path) as base:
        _files(tmp_path, pair[0], A_ID)
        _files(tmp_path, pair[1], B_ID)
        with _open(browser, f"{base}/compare/{A_ID}/{B_ID}", viewport_width=1440) as watched:
            page = watched.page
            page.route(f"**/api/compare/{A_ID}/{B_ID}", lambda route: route.fulfill(
                status=409, content_type="application/json", body=json.dumps(body)))
            page.reload(wait_until="networkidle")
            page.locator("#rejection").wait_for(state="visible")
            assert page.locator("#rejection-title").inner_text() == "這兩份不能直接比較"
            items = page.locator("#reject-reason > ul li").all_inner_texts()
            assert {item.split("；")[0].split("（")[0] for item in items} == {
                "A 和 B 的計算版本不同", "A 和 B 的方案代號相同", "A 和 B 的方案用途不同",
                "A 和 B 的聲道設定不同", "A 和 B 有一項固定設定對不上，不能直接比較"}
            shown = page.locator("#rejection").inner_text()
            assert "B（wall-1）是用舊程式算的（計算版本跟現在不同）" in shown
            assert not re.search(r"[0-9a-f]{7,}|[a-z]+_[a-z]+|計算指紋|第 1 份|purpose|候選代號", shown)
            folded = page.locator("#reject-reason details.technical")
            assert folded.evaluate("el => el.open") is False
            assert all(problem in (folded.text_content() or "") for problem in problems)
            assert not page.locator("#rerun-holder button").all()
            assert watched.page_errors == []
            assert all("409" in error for error in watched.console_errors)


def test_page_that_cannot_load_prints_a_plain_sentence(
        tmp_path: Path, browser: Browser, pair: tuple[SchemeResult, SchemeResult]) -> None:
    # 讀不到（連不上伺服器、伺服器回的不是資料）：主畫面一句白話說怎麼辦，瀏覽器的英文錯誤收進技術細節。
    with _serve(tmp_path) as base:
        _files(tmp_path, pair[0], A_ID)
        _files(tmp_path, pair[1], B_ID)
        with _open(browser, f"{base}/compare/{A_ID}/{B_ID}", viewport_width=1440) as watched:
            page = watched.page
            for reply, sentence, raw in (
                    (lambda route: route.abort(), "連不上網頁伺服器；確認伺服器開著，再重新整理這一頁", "TypeError"),
                    (lambda route: route.fulfill(status=500, content_type="text/html", body="<html>內部錯誤</html>"),
                     "網頁伺服器的回應讀不懂（回應 500）；請重新整理這一頁", "SyntaxError")):
                page.unroute(f"**/api/compare/{A_ID}/{B_ID}")
                page.route(f"**/api/compare/{A_ID}/{B_ID}", reply)
                page.reload(wait_until="networkidle")
                page.locator("#rejection").wait_for(state="visible")
                assert page.locator("#rejection-title").inner_text() == "比較讀取失敗"
                assert page.locator("#reject-reason > p").all_inner_texts() == [sentence]
                assert raw not in page.locator("#rejection").inner_text()
                folded = page.locator("#reject-reason details.technical")
                assert folded.evaluate("el => el.open") is False
                assert raw in (folded.text_content() or "")
                assert page.locator("#loading").is_hidden()
            assert watched.page_errors == []


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
