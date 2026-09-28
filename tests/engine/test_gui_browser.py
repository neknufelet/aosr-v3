"""畫面考卷：用真的無頭瀏覽器開本機網頁，驗事先想得到的那幾種壞法。

程式驗得出來的只有事先想得到的：每一頁打得開、瀏覽器沒報錯、圖上真的有線、點了有反應、
畫面上沒有沒格式化的數字或 JS 的空值字樣。事先想不到的（字擠在一起、難看）靠人截圖親眼看，不在這裡。
瀏覽器（無頭 Chromium：Chrome 的核心、不開視窗）缺席就紅、不跳過：雲端 verify 在考卷前那一步先裝好，
本機第一次要先跑 `uv run playwright install --only-shell chromium`。
"""
from __future__ import annotations

import re
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import uvicorn
from playwright.sync_api import Browser, ConsoleMessage, Error, Page, sync_playwright

from aosr.gui.app import GuiSettings, create_app
from aosr.reporting.result import SchemeResult, save_result
from tests.engine.test_scheme_pipeline import shared_control_result

RUN_ID = "c" * 32
# 畫布上「有顏色」的點：紅綠藍三色最大減最小超過 60。軸線、格線、字是灰黑色，差值接近 0；
# results.js 的曲線顏色前幾色差值都在 60 以上。uPlot（畫圖的函式庫）沒給線條顏色就一條線都不畫，
# 那時候這個數是 0——第二步截圖抓到的就是這種。
LINES_DRAWN_JS = """() => {
  const canvas = document.querySelector("#chart canvas");
  if (!canvas) return false;
  const data = canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data;
  for (let i = 0; i < data.length; i += 4) {
    const high = Math.max(data[i], data[i + 1], data[i + 2]);
    const low = Math.min(data[i], data[i + 1], data[i + 2]);
    if (data[i + 3] > 0 && high - low > 60) return true;
  }
  return false;
}"""
# 伺服器給的數字都已格式化成固定位數；最細的是超出量，極小時最多印到 12 位（result_view._excess）。
# JS 直接印一個浮點原值通常是 15～17 位有效數字，所以十三位以上小數就是有一格繞過伺服器
# （第二步截圖抓過十幾位小數的表）。
LONG_DECIMAL = re.compile(r"\d\.\d{13,}")
# 圖例一條線的標籤是「喇叭・位置代號（角色）」，取出位置代號。
LEGEND_RECEIVER = re.compile(r"・(.+?)（")
# JS 讀到不存在的欄位、或把物件直接印出來時會出現的字樣；Python 的字典樣子表示伺服器轉印了原物件。
# 頁面自己會接住載入時的例外、把訊息寫進畫面（結果頁寫進拒收區塊、輸入頁寫進訊息列），
# 那種錯瀏覽器不會記成 JS 例外，所以成功的頁面另外要查拒收區塊藏著、該有字的格子有字。
RAW_VALUES = re.compile(r"undefined|NaN|\[object Object\]|\bnull\b|\bNone\b|\{'")


@dataclass
class Watched:
    """一頁加上它開著期間瀏覽器吐出來的錯誤。"""

    page: Page
    console_errors: list[str] = field(default_factory=list)
    page_errors: list[str] = field(default_factory=list)


@pytest.fixture(scope="module")
def result(tmp_path_factory: pytest.TempPathFactory, worker_id: str) -> SchemeResult:
    return shared_control_result(tmp_path_factory, worker_id, "wall-1")


@pytest.fixture(scope="module")
def browser() -> Iterator[Browser]:
    with sync_playwright() as playwright:
        launched = playwright.chromium.launch()
        try:
            yield launched
        finally:
            launched.close()


@contextmanager
def _serve(data_dir: Path) -> Iterator[str]:
    """在迴圈位址挑一個空的埠起真的伺服器（跟 python -m aosr.gui 同一個 uvicorn），用完就關。"""
    app = create_app(GuiSettings(engine_commit="a" * 40, data_dir=data_dir))
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 30
    while not server.started:
        if not thread.is_alive() or time.monotonic() > deadline:
            raise RuntimeError("本機網頁沒有起來")
        time.sleep(0.05)
    port = int(server.servers[0].sockets[0].getsockname()[1])
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=30)


@contextmanager
def _open(browser: Browser, url: str) -> Iterator[Watched]:
    page = browser.new_page(viewport={"width": 1400, "height": 1100})
    watched = Watched(page)

    def on_console(message: ConsoleMessage) -> None:
        if message.type == "error":
            watched.console_errors.append(message.text)

    def on_page_error(error: Error) -> None:
        watched.page_errors.append(str(error))

    page.on("console", on_console)
    page.on("pageerror", on_page_error)
    try:
        page.goto(url, wait_until="networkidle")
        yield watched
    finally:
        page.close()


def _save(data_dir: Path, result: SchemeResult) -> None:
    # 伺服器起來時才建子資料夾；結果檔要在打開頁面前就在，所以這裡先建。
    (data_dir / "results").mkdir(parents=True, exist_ok=True)
    save_result(result, data_dir / "results" / f"{RUN_ID}.json")


def _wait_for_lines(page: Page) -> None:
    # 等到畫布上真的出現曲線顏色；等不到就逾時紅（uPlot 在下一個微任務才畫，不能點完立刻讀）。
    page.wait_for_function(LINES_DRAWN_JS, timeout=10_000)


def _legend_labels(page: Page) -> list[str]:
    # 第一格是 x 軸（頻率）那一條，不是曲線。
    return [text.rstrip("-").strip() for text in page.locator("#chart .u-legend .u-series").all_inner_texts()[1:]]


def _assert_text_is_formatted(page: Page) -> None:
    text = page.locator("body").inner_text()
    assert not LONG_DECIMAL.findall(text), LONG_DECIMAL.findall(text)
    assert not RAW_VALUES.findall(text), RAW_VALUES.findall(text)


def _assert_quiet(watched: Watched) -> None:
    assert watched.console_errors == []
    assert watched.page_errors == []


def test_input_page_draws_plan_on_open_and_check_button_answers(tmp_path: Path,
                                                                browser: Browser) -> None:
    with _serve(tmp_path) as base, _open(browser, f"{base}/") as watched:
        page = watched.page
        # 平面圖與側面圖一打開就畫（第二步截圖抓過「要先存檔才畫」）。
        for svg in ("#plan-xy", "#plan-xz"):
            page.locator(f"{svg} circle").first.wait_for()
            assert page.locator(f"{svg} rect").count() > 0, svg
        # 六面牆的阻抗倍數（幾倍 ρc）是伺服器算好回來填的；載入時出錯會被接住、訊息又被「檢查通過」蓋掉，
        # 只有這幾格空著看得出來。
        page.wait_for_function("""() => {
          const cells = [...document.querySelectorAll("#walls span[id^='multiple-']")];
          return cells.length > 0 && cells.every((cell) => cell.textContent.trim() !== "");
        }""")
        page.locator("#messages").evaluate("(element) => { element.textContent = ''; }")
        page.locator("#check").click()
        page.wait_for_function("() => document.getElementById('messages').textContent !== ''")
        assert page.locator("#messages").inner_text() == "檢查通過"
        _assert_text_is_formatted(page)
        _assert_quiet(watched)


def test_zoom_labels_legend_and_marker_detail(tmp_path: Path, browser: Browser) -> None:
    with _serve(tmp_path) as base, _open(browser, f"{base}/") as watched:
        page = watched.page
        page.locator("#zoom-xy circle").first.wait_for()
        for svg in ("#zoom-xy", "#zoom-xz"):
            boxes = page.locator(f"{svg} text").evaluate_all(
                "nodes => nodes.map(node => { const b = node.getBBox(); return {x:b.x,y:b.y,w:b.width,h:b.height}; })")
            for index, first in enumerate(boxes):
                for second in boxes[index + 1:]:
                    overlap = max(0, min(first["x"] + first["w"], second["x"] + second["w"]) -
                                  max(first["x"], second["x"])) * max(
                                      0, min(first["y"] + first["h"], second["y"] + second["h"]) -
                                      max(first["y"], second["y"]))
                    assert overlap == 0
            collisions = page.locator(svg).evaluate("""svg => {
              const groups = [...svg.querySelectorAll('g[data-keys]')];
              return groups.flatMap(group => {
                const box = group.querySelector('text').getBBox();
                return groups.filter(other => other !== group).filter(other => {
                  const circle = other.querySelector('circle');
                  const x = Number(circle.getAttribute('cx'));
                  const y = Number(circle.getAttribute('cy'));
                  const r = Number(circle.getAttribute('r'));
                  return box.x < x + r && x - r < box.x + box.width &&
                    box.y < y + r && y - r < box.y + box.height;
                }).map(other => [group.dataset.keys, other.dataset.keys]);
              });
            }""")
            assert collisions == [], (svg, collisions)
        scheme = page.request.get(f"{base}/api/example").json()["scheme"]
        expected = {f"speaker:{name}" for name in scheme["speakers"]} | {
            f"receiver:{point['receiver_id']}" for point in scheme["receiver_set"]["points"]}
        shown = set(page.locator("#plan-legend li").evaluate_all("nodes => nodes.map(node => node.dataset.id)"))
        assert shown == expected
        plan = page.request.post(f"{base}/api/plan", data=scheme).json()
        for svg, name in (("#zoom-xy", "zoom_plan"), ("#zoom-xz", "zoom_side")):
            drawn = page.locator(f"{svg} g[data-keys]").evaluate_all(
                "nodes => nodes.map(node => [node.dataset.keys.split(' ').sort(), node.querySelector('text').textContent])")
            # 瀏覽器傳回來的是清單，伺服器這邊也組成清單再比（組合跟清單不相等）。
            server = [[sorted(item["keys"]), item["caption"]] for item in plan["views"][name]]
            assert sorted(drawn) == sorted(server)
            listening = {point["key"] for point in plan["receivers"]
                         if point["role"] in {"primary", "surrounding"}}
            assert {key for keys, _ in drawn for key in keys} == listening
        marker = page.locator("#zoom-xy g[data-keys]").first
        detail = marker.locator("title").text_content()
        marker.locator("circle").click()
        assert page.locator("#plan-detail").inner_text() == detail
        _assert_quiet(watched)


def test_plan_refresh_clears_stale_detail_and_legend(tmp_path: Path, browser: Browser) -> None:
    with _serve(tmp_path) as base, _open(browser, f"{base}/") as watched:
        page = watched.page
        page.locator("#zoom-xy g[data-keys]").first.locator("circle").click()
        assert page.locator("#plan-detail").inner_text()
        page.locator("#receiver-front-x").fill("3.0")
        page.locator("#check").click()
        page.wait_for_function("() => document.querySelector('#plan-detail').textContent === ''")
        assert "x 3.00" in page.locator("#plan-legend li[data-id='receiver:front']").inner_text()
        page.locator("#room-Lx").fill("")
        page.locator("#check").click()
        page.wait_for_function("() => document.querySelector('#messages').textContent.includes('必填')")
        assert not page.locator("#plan-legend li").all_inner_texts()
        assert page.locator("#plan-detail").inner_text() == ""
        for svg in ("#plan-xy", "#plan-xz", "#zoom-xy", "#zoom-xz"):
            assert not page.locator(f"{svg} circle").evaluate_all("nodes => nodes.map(node => node.outerHTML)")
        # 長度清空時伺服器照設計回 422，瀏覽器會記一筆「載入失敗」，那是預期的；JS 自己不准拋錯。
        assert all("422" in text for text in watched.console_errors), watched.console_errors
        assert watched.page_errors == []


def test_plan_collision_and_other_seat_draw_server_captions(tmp_path: Path,
                                                            browser: Browser) -> None:
    with _serve(tmp_path) as base, _open(browser, f"{base}/") as watched:
        page = watched.page
        page.evaluate("""() => {
          scheme.receiver_set.points.push({receiver_id: 'seat2', position_m: [1.5, 1.2, 1.2],
            role: 'other_seat', importance: 0, direction_relative_to_primary: null});
          renderForm();
        }""")
        page.locator("#speaker-right-x").fill("3.2")
        page.locator("#speaker-right-y").fill("1.8")
        page.locator("#check").click()
        page.wait_for_function("() => document.querySelectorAll('#plan-legend li').length > 0")
        scheme = page.evaluate("() => collect()")
        plan = page.request.post(f"{base}/api/plan", data=scheme).json()
        keys = {item["key"] for item in plan["speakers"] + plan["receivers"]}
        assert set(page.locator("#plan-legend li").evaluate_all(
            "nodes => nodes.map(node => node.dataset.id)")) == keys
        for svg, name in (("#plan-xy", "plan"), ("#plan-xz", "side")):
            drawn = page.locator(f"{svg} g[data-keys]").evaluate_all(
                "nodes => nodes.map(node => [node.dataset.keys.split(' ').sort(), node.querySelector('text').textContent])")
            server = [[sorted(item["keys"]), item["caption"]] for item in plan["views"][name]
                      if item["caption"]]
            assert sorted(drawn) == sorted(server)
            assert any("receiver:seat2" in group and "座" in caption for group, caption in drawn)
        collision = page.locator("#plan-xy g[data-keys*='speaker:right']")
        assert "receiver:left" in (collision.get_attribute("data-keys") or "")
        assert collision.locator("text").text_content() == "R"
        for svg in ("#zoom-xy", "#zoom-xz"):
            assert all("receiver:seat2" not in keys for keys in page.locator(
                f"{svg} g[data-keys]").evaluate_all("nodes => nodes.map(node => node.dataset.keys)"))
        _assert_quiet(watched)


def test_results_page_draws_lines_for_every_speaker(tmp_path: Path, browser: Browser,
                                                     result: SchemeResult) -> None:
    _save(tmp_path, result)
    with _serve(tmp_path) as base, _open(browser, f"{base}/results/{RUN_ID}") as watched:
        page = watched.page
        _wait_for_lines(page)
        speakers = page.locator("#speaker-buttons button")
        assert speakers.all_inner_texts(), "沒有可切換的喇叭"
        for index, name in enumerate(speakers.all_inner_texts()):
            speakers.nth(index).click()
            _wait_for_lines(page)
            # 點了有反應：圖例換成這一支喇叭的線。
            labels = _legend_labels(page)
            assert labels and all(label.startswith(f"{name}・") for label in labels), (name, labels)
        for section in ("#categories", "#alerts", "#reverb", "#reflections", "#summary"):
            assert page.locator(section).inner_text().strip(), section
        # 警戒標題照印伺服器拼好的字串，網頁不自己拼。
        alerts = page.request.get(f"{base}/api/results/{RUN_ID}").json()["alerts"]
        assert page.locator("#alerts h3").all_inner_texts() == [item["heading_text"] for item in alerts]
        page.evaluate("() => { view.reflections[0].reason_codes = []; drawReflections(); }")
        assert "原因：無" in page.locator("#reflections p").first.inner_text()
        assert page.locator("#loading").is_hidden()
        assert page.locator("#rejection").is_hidden(), page.locator("#reject-reason").inner_text()
        _assert_text_is_formatted(page)
        _assert_quiet(watched)


def test_choosing_a_pair_shows_that_pair_and_fills_its_detail(tmp_path: Path, browser: Browser,
                                                             result: SchemeResult) -> None:
    _save(tmp_path, result)
    with _serve(tmp_path) as base, _open(browser, f"{base}/results/{RUN_ID}") as watched:
        page = watched.page
        _wait_for_lines(page)
        pairs = page.locator("#pair-buttons button")
        assert pairs.all_inner_texts(), "沒有可點的位置對"
        # 還沒選之前明細是空的，點了才填——證明是點選觸發的，不是一打開就在。
        assert not page.locator("#pair-detail td").all_inner_texts()
        for index, caption in enumerate(pairs.all_inner_texts()):
            reference_id, receiver_id = caption.split("：", 1)[1].split(" ↔ ")
            # 每一對點之前先清空明細，上一對留下的表才不會讓「這一次沒反應」看起來也有字。
            page.locator("#pair-detail").evaluate("(element) => element.replaceChildren()")
            pairs.nth(index).click()
            page.locator("#pair-detail td").first.wait_for()
            _wait_for_lines(page)
            shown = {match for label in _legend_labels(page) for match in LEGEND_RECEIVER.findall(label)}
            assert shown == {reference_id, receiver_id}, caption
            cells = page.locator("#pair-detail td").all_inner_texts()
            assert all(cell.strip() for cell in cells), caption
            _assert_text_is_formatted(page)
        assert page.locator("#rejection").is_hidden(), page.locator("#reject-reason").inner_text()
        _assert_quiet(watched)


def test_rejected_result_shows_reason_and_rerun_button(tmp_path: Path, browser: Browser,
                                                        result: SchemeResult,
                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    _save(tmp_path, result)

    def reject(*args: object, **kwargs: object) -> SchemeResult:
        raise ValueError("評估器版本對不上")

    # 伺服器跟考卷同一個行程（另一條執行緒），所以換掉讀回函式就是換掉伺服器看到的那一支。
    monkeypatch.setattr("aosr.gui.app.load_result", reject)
    with _serve(tmp_path) as base, _open(browser, f"{base}/results/{RUN_ID}") as watched:
        page = watched.page
        assert page.locator("#loading").is_hidden()
        page.locator("#rejection").wait_for()
        assert "評估器版本對不上" in page.locator("#reject-reason").inner_text()
        assert page.locator("#rerun").is_visible()
        assert page.locator("#content").is_hidden()
        _assert_text_is_formatted(page)
        # 409 回應本身瀏覽器會記一筆「載入失敗」，那是預期的；JS 自己不准拋錯。
        assert all("409" in text for text in watched.console_errors), watched.console_errors
        assert watched.page_errors == []
