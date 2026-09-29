"""畫面考卷：用真的無頭瀏覽器開本機網頁，驗事先想得到的那幾種壞法。

程式驗得出來的只有事先想得到的：每一頁打得開、瀏覽器沒報錯、圖上真的有線、點了有反應、
畫面上沒有沒格式化的數字或 JS 的空值字樣。事先想不到的（字擠在一起、難看）靠人截圖親眼看，不在這裡。
瀏覽器（無頭 Chromium：Chrome 的核心、不開視窗）缺席就紅、不跳過：雲端 verify 在考卷前那一步先裝好，
本機第一次要先跑 `uv run playwright install --only-shell chromium`。
"""
from __future__ import annotations

import re
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import uvicorn
from playwright.sync_api import Browser, ConsoleMessage, Error, Page, Route, sync_playwright

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
# 結果頁圖例一條線的標籤是「喇叭顯示名 → 座位顯示名」，取出座位顯示名。
LEGEND_RECEIVER = re.compile(r" → (.+)$")
# JS 讀到不存在的欄位、或把物件直接印出來時會出現的字樣；Python 的字典樣子表示伺服器轉印了原物件。
# 頁面自己會接住載入時的例外、把訊息寫進畫面（結果頁寫進拒收區塊、輸入頁寫進訊息列），
# 那種錯瀏覽器不會記成 JS 例外，所以成功的頁面另外要查拒收區塊藏著、該有字的格子有字。
RAW_VALUES = re.compile(r"undefined|NaN|\[object Object\]|\bnull\b|\bNone\b|\{'")
# 一顆按鈕（或做成按鈕樣子的連結）看起來的樣子：底色、字色、圓角、內距、字級、底線、高度。
BUTTON_LOOK_JS = """(node) => {
  const own = getComputedStyle(node);
  return [own.backgroundColor, own.color, own.borderRadius, own.padding, own.fontSize,
          own.textDecorationLine, Math.round(node.getBoundingClientRect().height)];
}"""


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
def _serve(data_dir: Path, runner: tuple[str, ...] | None = None) -> Iterator[str]:
    """在迴圈位址挑一個空的埠起真的伺服器（跟 python -m aosr.gui 同一個 uvicorn），用完就關。"""
    app = create_app(GuiSettings(engine_commit="a" * 40, data_dir=data_dir, runner=runner))
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
def _open(browser: Browser, url: str, device_scale_factor: float = 1,
          viewport_width: int = 1400) -> Iterator[Watched]:
    page = browser.new_page(viewport={"width": viewport_width, "height": 1100},
                            device_scale_factor=device_scale_factor)
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
    # 結果頁圖例不跟游標（live: false）：uPlot 不畫 x 軸那一格，也不在線名後面掛「--」，每一格都是一條曲線。
    return [text.strip() for text in page.locator("#chart .u-legend .u-series").all_inner_texts()]


def _assert_text_is_formatted(page: Page) -> None:
    text = page.locator("body").inner_text()
    assert not LONG_DECIMAL.findall(text), LONG_DECIMAL.findall(text)
    assert not RAW_VALUES.findall(text), RAW_VALUES.findall(text)


def _assert_quiet(watched: Watched) -> None:
    assert watched.console_errors == []
    assert watched.page_errors == []


def _copy_runner(tmp_path: Path, result: SchemeResult, delay_s: float = 0.0) -> tuple[str, ...]:
    """假計算：等 delay_s 秒，把考卷結果抄到 --out。"""
    source = tmp_path / "fixture-result.json"
    save_result(result, source)
    script = tmp_path / "copy-result.py"
    script.write_text(f"import shutil,sys,time\ntime.sleep({delay_s})\n"
                      f"shutil.copyfile({str(source)!r}, sys.argv[sys.argv.index('--out') + 1])\n")
    return (sys.executable, str(script))


def test_editing_after_calculation_marks_result_stale(tmp_path: Path, browser: Browser,
                                                      result: SchemeResult) -> None:
    with _serve(tmp_path, _copy_runner(tmp_path, result)) as base, _open(browser, f"{base}/") as watched:
        page = watched.page
        page.locator("#calculate").click()
        page.locator("#result-link").wait_for(state="visible")
        first = page.locator("#result-link").get_attribute("href")
        assert first is not None
        # 算完的訊息講白話：不印結果檔的路徑、也不印 32 位計算代號，看結果交給旁邊的「查看結果頁」連結。
        finished = page.locator("#messages").inner_text()
        assert "算完了" in finished and "查看結果頁" in finished
        assert first.rsplit("/", 1)[1] not in finished and "/" not in finished and ".json" not in finished
        assert page.locator("#result-stale").is_hidden()
        page.locator("#speaker-left-x").fill("1.7")
        assert page.locator("#result-stale").is_visible()
        assert page.locator("#result-link").get_attribute("href") == first
        page.locator("#save-as-id").fill("next")
        page.locator("#save-as").click()
        page.wait_for_function("() => document.querySelector('#save-id').value === 'next'")
        page.locator("#calculate").click()
        page.wait_for_function("old => document.querySelector('#result-link').getAttribute('href') !== old", arg=first)
        assert page.locator("#result-stale").is_hidden()
        # 開舊方案時結果連結與舊結果標示一起藏：畫面上那份結果不一定是打開的這一份算的。
        page.locator("#speaker-left-x").fill("1.8")
        assert page.locator("#result-stale").is_visible()
        page.locator("#scheme-list").select_option("next")
        page.locator("#open-scheme").click()
        page.wait_for_function("() => document.querySelector('#result-link').hidden")
        assert page.locator("#result-stale").is_hidden()
        _assert_quiet(watched)


def test_finished_run_is_not_attached_to_a_different_form(tmp_path: Path, browser: Browser,
                                                         result: SchemeResult) -> None:
    # 算完時表單已經不是算的那一份（重新整理回到範本、或開算後改過），連結不准掛在表單旁邊。
    with _serve(tmp_path, _copy_runner(tmp_path, result, delay_s=4)) as base, \
            _open(browser, f"{base}/") as watched:
        page = watched.page
        page.locator("#calculate").click()
        page.wait_for_function("() => !document.querySelector('#stop').disabled")
        page.reload(wait_until="networkidle")
        page.wait_for_function("() => document.querySelector('#messages').textContent.includes('算完了')",
                               timeout=15_000)
        assert page.locator("#result-link").is_hidden()
        page.locator("#calculate").click()
        page.wait_for_function("() => !document.querySelector('#stop').disabled")
        page.locator("#speaker-left-x").fill("1.7")
        page.wait_for_function("() => document.querySelector('#stop').disabled", timeout=15_000)
        assert "算完了" in page.locator("#messages").inner_text()
        assert page.locator("#result-link").is_hidden()
        _assert_quiet(watched)


def test_blocked_save_keeps_the_edited_mark(tmp_path: Path, browser: Browser,
                                            result: SchemeResult) -> None:
    # 在算的時候改一格、再按計算：正在算的那份改不得，存檔被 409 擋。「開算後改過」的記號要留著，
    # 不然原本那筆算完會把連結掛在改過的表單旁邊。
    with _serve(tmp_path, _copy_runner(tmp_path, result, delay_s=4)) as base, \
            _open(browser, f"{base}/") as watched:
        page = watched.page
        page.locator("#calculate").click()
        page.wait_for_function("() => document.querySelector('#run-state').textContent.includes('計算中')")
        page.locator("#room-Lx").fill("5.5")
        page.locator("#calculate").click()
        page.wait_for_function("() => document.querySelector('#messages').textContent.includes('正在計算')")
        page.wait_for_function("() => document.querySelector('#messages').textContent.includes('算完了')",
                               timeout=15_000)
        assert page.locator("#result-link").is_hidden()
        assert all("409" in text for text in watched.console_errors), watched.console_errors
        assert watched.page_errors == []


def test_second_calculation_leaves_no_orphan_polling(tmp_path: Path, browser: Browser,
                                                     result: SchemeResult) -> None:
    # 第一筆還在算就再按一次計算：第一筆的計時器要停掉，不然算完後它每秒還在查、蓋掉訊息。
    with _serve(tmp_path, _copy_runner(tmp_path, result, delay_s=2)) as base, \
            _open(browser, f"{base}/") as watched:
        page = watched.page
        page.locator("#calculate").click()
        page.wait_for_function("() => document.querySelector('#run-state').textContent.includes('計算中')")
        page.locator("#calculate").click()
        page.locator("#result-link").wait_for(state="visible", timeout=15_000)
        polled: list[str] = []
        page.on("request", lambda request: polled.append(request.url) if "/api/runs/" in request.url else None)
        page.wait_for_timeout(2500)
        assert polled == []
        _assert_quiet(watched)


def test_buttons_work_even_if_a_list_fails_to_load(tmp_path: Path, browser: Browser) -> None:
    def broken(route: Route) -> None:
        route.fulfill(status=500, content_type="application/json", body='{"error": "清單壞了"}')

    with _serve(tmp_path) as base:
        page = browser.new_page(viewport={"width": 1400, "height": 1100})
        try:
            page.route("**/api/results", broken)
            page.goto(f"{base}/", wait_until="networkidle")
            page.evaluate("() => { document.querySelector('#messages').textContent = ''; }")
            page.locator("#check").click()
            page.wait_for_function("() => document.querySelector('#messages').textContent === '檢查通過'")
        finally:
            page.close()


def test_label_table_failure_stays_on_screen(tmp_path: Path, browser: Browser) -> None:
    # 名稱表載不到：列名退回代號，旁邊要有一行一直看得到的說明。以前寫在訊息列，
    # 同一次開頁就被「檢查通過」蓋掉，畫面上只剩英文代號配綠字的檢查通過。
    def broken(route: Route) -> None:
        route.fulfill(status=500, content_type="application/json", body='{"error": "名稱表壞了"}')

    with _serve(tmp_path) as base, _open(browser, "about:blank") as watched:
        page = watched.page
        page.route("**/api/labels", broken)
        page.goto(f"{base}/", wait_until="networkidle")
        page.wait_for_function("() => document.querySelector('#messages').textContent === '檢查通過'")
        notice = page.locator("#labels-notice")
        assert notice.is_visible() and "名稱表壞了" in notice.inner_text()
        title = page.locator("#speaker-left-x").locator("xpath=ancestor::div[@class='row']/strong")
        assert title.inner_text() == "left"
        # 名稱表的說明跟喇叭與座位的列放在同一區，再按一次檢查也還在。
        assert notice.evaluate("node => node.closest('section').querySelector('h2').textContent") == "喇叭與座位"
        page.locator("#messages").evaluate("node => { node.textContent = ''; }")
        page.locator("#check").click()
        page.wait_for_function("() => document.querySelector('#messages').textContent === '檢查通過'")
        assert notice.is_visible() and "名稱表壞了" in notice.inner_text()
        assert all("500" in text for text in watched.console_errors), watched.console_errors
        assert watched.page_errors == []


def test_problem_messages_name_the_form_field_in_chinese(tmp_path: Path, browser: Browser) -> None:
    # 伺服器回的問題帶的是方案裡的欄位路徑（英文，例如 receiver_set.points.5.position_m.2），
    # 訊息列改寫成表單上那一格的中文名；座位用順序號找到是哪一個座位、軸用 0／1／2 對 x／y／z。
    with _serve(tmp_path) as base, _open(browser, f"{base}/") as watched:
        page = watched.page
        page.locator("#plan-legend li").first.wait_for()
        page.locator("#use-scattering").check()
        for scatter in page.locator("#scattering input").all():
            scatter.fill("0.2")

        def shown_after_check() -> str:
            page.locator("#messages").evaluate("node => { node.textContent = ''; }")
            page.locator("#check").click()
            page.wait_for_function("() => document.querySelector('#messages').textContent !== ''")
            return page.locator("#messages").inner_text()

        for field, name in (("#room-Ly", "寬 Ly（公尺）"), ("#wall-floor", "地板阻抗（帕·秒／公尺）"),
                            ("#scatter-ceiling", "天花散射"), ("#speaker-right-y", "右聲道喇叭 y 座標"),
                            ("#receiver-up-z", "主位上方 z 座標"), ("#receiver-main-x", "主位 x 座標")):
            before = page.locator(field).input_value()
            page.locator(field).fill("")
            assert shown_after_check() == f"{name}：必填", field
            page.locator(field).fill(before)
            assert shown_after_check() == "檢查通過", field
        # 方案區的格子也寫中文名（方案代號是唯讀格，這裡直接清空它來考）。
        page.locator("#save-id").evaluate("node => { node.value = ''; }")
        assert shown_after_check().startswith("方案代號：")
        page.locator("#save-id").evaluate("(node, name) => { node.value = name; }",
                                          page.request.get(f"{base}/api/example").json()["scheme"]["scheme_id"])
        assert shown_after_check() == "檢查通過"
        # 整份方案的問題（喇叭跑出房間）沒有對應的一格：只印訊息，不掛英文的「scheme：」。
        page.locator("#speaker-left-x").fill("99")
        assert not shown_after_check().startswith("scheme")
        # 每一對的問題：「喇叭中文名 → 座位中文名」，每個座位都對得到它的中文名。
        primary = page.evaluate("() => scheme.receiver_set.points.find((point) => point.role === 'primary').position_m")
        for axis, value in zip("xyz", primary, strict=True):
            page.locator(f"#speaker-left-{axis}").fill(str(value))
        lines = shown_after_check().split("\n")
        labels = page.request.get(f"{base}/api/labels").json()
        example = page.request.get(f"{base}/api/example").json()["scheme"]
        role = next(channel["role"] for channel in example["channel_group"]["channels"]
                    if channel["speaker_id"] == "left")
        points = [point["receiver_id"] for point in example["receiver_set"]["points"]]
        assert {line.split("：", 1)[0] for line in lines} == {
            f"{labels['speakers'][role]} → {labels['listening_points'][code]}" for code in points}
        assert all("422" in text for text in watched.console_errors), watched.console_errors
        assert watched.page_errors == []


def test_open_saved_scheme_then_save_as_keeps_original(tmp_path: Path, browser: Browser) -> None:
    with _serve(tmp_path) as base, _open(browser, f"{base}/") as watched:
        page = watched.page
        page.locator("#room-Lx").fill("5.5")
        page.locator("#save-as-id").fill("A")
        page.locator("#save-as").click()
        page.wait_for_function("() => document.querySelector('#save-id').value === 'A'")
        before = (tmp_path / "schemes" / "A.json").read_bytes()
        page.locator("#room-Lx").fill("4.5")
        page.locator("#scheme-list").select_option("A")
        page.locator("#open-scheme").click()
        page.wait_for_function("() => document.querySelector('#room-Lx').value === '5.5'")
        # 打開的那一份還沒有結果：改了按「儲存」要真的存進去，不是當成另存撞名。
        page.locator("#room-Lx").fill("5.7")
        page.locator("#save").click()
        page.wait_for_function("() => document.querySelector('#messages').textContent === '方案已儲存'")
        assert (tmp_path / "schemes" / "A.json").read_bytes() != before
        before = (tmp_path / "schemes" / "A.json").read_bytes()
        page.locator("#save-as-id").fill("b")
        page.locator("#save-as").click()
        page.wait_for_function("() => document.querySelector('#save-id').value === 'b'")
        page.locator("#room-Lx").fill("5.6")
        assert (tmp_path / "schemes" / "A.json").read_bytes() == before
        page.locator("#save-as").click()
        page.wait_for_function("() => document.querySelector('#messages').textContent.includes('已經有叫')")
        # 撞名時伺服器照設計回 409，瀏覽器會記一筆「載入失敗」，那是預期的；JS 自己不准拋錯。
        assert all("409" in text for text in watched.console_errors), watched.console_errors
        assert watched.page_errors == []


def test_template_save_does_not_overwrite_same_name_scheme(tmp_path: Path, browser: Browser) -> None:
    with _serve(tmp_path) as base, _open(browser, f"{base}/") as watched:
        page = watched.page
        page.locator("#save").click()
        page.wait_for_function("() => document.querySelector('#messages').textContent === '方案已儲存'")
        name = page.locator("#save-id").input_value()
        before = (tmp_path / "schemes" / f"{name}.json").read_bytes()
        # 重新整理後表單是範本、不是從那份打開的：同名儲存要當成另存，不准蓋掉已經存的那份。
        page.reload(wait_until="networkidle")
        page.locator("#room-Lx").fill("5.5")
        page.locator("#save").click()
        page.wait_for_function("() => document.querySelector('#messages').textContent.includes('已經有叫')")
        assert (tmp_path / "schemes" / f"{name}.json").read_bytes() == before
        assert all("409" in text for text in watched.console_errors), watched.console_errors
        assert watched.page_errors == []


def test_results_list_links_to_result_page(tmp_path: Path, browser: Browser,
                                          result: SchemeResult) -> None:
    _save(tmp_path, result)
    with _serve(tmp_path) as base, _open(browser, f"{base}/") as watched:
        page = watched.page
        page.locator("#results-list tr").first.wait_for()
        assert result.scheme.scheme_id in page.locator("#results-list tr").first.inner_text()
        _assert_text_is_formatted(page)
        # 「查看」仍是連結（連到結果頁），但外觀跟同一列的「選為 A／B」按鈕一樣：底色、字色、圓角、
        # 內距、高度都相同，沒有底線（以前是藍色底線字，跟旁邊的按鈕不像同一組動作）。
        row = page.locator("#results-list tr").first
        looks = [row.locator("a").evaluate(BUTTON_LOOK_JS),
                 *row.locator("button").evaluate_all(f"nodes => nodes.map({BUTTON_LOOK_JS})")]
        assert looks[0][5] == "none" and looks[1:]
        assert all(item == looks[0] for item in looks[1:]), looks
        # 選好 A、B 才出現的「比較」也是同一種按鈕（以前是藍色底線小字）；沒選好之前藏著。
        assert page.locator("#compare-link").is_hidden()
        row.get_by_role("button", name="選為 A").click()
        row.get_by_role("button", name="選為 B").click()
        assert page.locator("#compare-link").is_visible()
        assert page.locator("#compare-link").evaluate(BUTTON_LOOK_JS) == looks[0]
        page.locator("#results-list tr a").first.click()
        _wait_for_lines(page)
        _assert_quiet(watched)


def test_reload_recovers_running_job_and_stop(tmp_path: Path, browser: Browser) -> None:
    script = tmp_path / "sleep-runner.py"
    script.write_text("import time\ntime.sleep(60)\n")
    with _serve(tmp_path, (sys.executable, str(script))) as base, _open(browser, f"{base}/") as watched:
        page = watched.page
        page.locator("#calculate").click()
        page.wait_for_function("() => !document.querySelector('#stop').disabled")
        run_id = page.evaluate("() => runId")
        try:
            page.reload(wait_until="networkidle")
            page.wait_for_function("() => !document.querySelector('#stop').disabled")
            assert "計算中" in page.locator("#run-state").inner_text()
            page.locator("#stop").click()
            page.wait_for_function("() => document.querySelector('#run-state').textContent.includes('已停止')")
            assert page.locator("#stop").is_disabled()
            # 已結束的不接回：表單開的是範本，貼上一筆已停止的狀態會讓人以為是這份的。
            page.reload(wait_until="networkidle")
            assert page.locator("#run-state").inner_text() == ""
            assert page.locator("#stop").is_disabled()
            _assert_quiet(watched)
        finally:
            if page.request.get(f"{base}/api/runs/{run_id}").json()["status"] == "running":
                page.request.post(f"{base}/api/runs/{run_id}/stop", data="{}",
                                  headers={"Content-Type": "application/json"})


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
                "nodes => nodes.map(node => [node.dataset.keys.split(' ').sort(), (node.querySelector('text') || {textContent: ''}).textContent])")
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
        # 失敗那一支也要清明細：先再點一次，讓明細有字，才考得到。
        page.locator("#zoom-xy g[data-keys]").first.locator("circle").click()
        assert page.locator("#plan-detail").inner_text()
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
                "nodes => nodes.map(node => [node.dataset.keys.split(' ').sort(), (node.querySelector('text') || {textContent: ''}).textContent])")
            server = [[sorted(item["keys"]), item["caption"]] for item in plan["views"][name]
                      if item["drawn"]]
            assert sorted(drawn) == sorted(server)
            assert any("receiver:seat2" in group and caption == "" for group, caption in drawn)
        collision = page.locator("#plan-xy g[data-keys*='speaker:right']")
        assert "receiver:left" in (collision.get_attribute("data-keys") or "")
        assert collision.locator("text").text_content() == "R"
        for svg in ("#zoom-xy", "#zoom-xz"):
            assert all("receiver:seat2" not in keys for keys in page.locator(
                f"{svg} g[data-keys]").evaluate_all("nodes => nodes.map(node => node.dataset.keys)"))
        _assert_quiet(watched)


def test_result_page_shows_fingerprint_rerun_only_when_different(
        tmp_path: Path, browser: Browser, result: SchemeResult,
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("aosr.gui.app.calculation_fingerprint", lambda **kwargs:
                        result.calculation_fingerprint)
    _save(tmp_path, result)
    with _serve(tmp_path) as base:
        with _open(browser, f"{base}/results/{RUN_ID}") as watched:
            page = watched.page
            page.locator("#content").wait_for(state="visible")
            assert "跟現在相同" in page.locator("#fingerprint-status").inner_text()
            assert page.locator("#fingerprint-rerun").is_hidden()
            assert page.locator("#fingerprint-notice").is_hidden()
        changed = result.model_copy(update={"calculation_fingerprint": "calc-v1:" + "1" * 64})
        save_result(changed, tmp_path / "results" / (RUN_ID + ".json"))
        with _open(browser, f"{base}/results/{RUN_ID}") as watched:
            page = watched.page
            page.locator("#fingerprint-rerun").wait_for(state="visible")
            # 按鈕字跟舊格式頁那一顆同一句白話（不寫「引擎」）。
            assert page.locator("#fingerprint-rerun").inner_text() == "用現在的程式重算"
            assert "計算指紋跟現在不同" in page.locator("#fingerprint-notice").inner_text()
            assert "跟現在不同" in page.locator("#fingerprint-status").inner_text()
            requested: list[str] = []
            def capture(route: Route) -> None:
                requested.append(route.request.url)
                route.fulfill(status=200, content_type="application/json", body='{"run_id":"started"}')
            page.route("**/api/results/*/rerun", capture)
            page.locator("#fingerprint-rerun").click()
            page.locator("#fingerprint-rerun-state").filter(has_text="已開始重算").wait_for()
            assert requested == [f"{base}/api/results/{RUN_ID}/rerun"]


def test_updated_server_notice_on_home_and_no_rerun_on_results_or_compare(
        tmp_path: Path, browser: Browser, result: SchemeResult,
        monkeypatch: pytest.MonkeyPatch) -> None:
    state = {"current": result.calculation_fingerprint}
    monkeypatch.setattr("aosr.gui.app.calculation_fingerprint", lambda **kwargs: state["current"])
    _save(tmp_path, result)
    other_id = "b" * 32
    other = result.model_copy(update={"scheme": result.scheme.model_copy(
        update={"scheme_id": "wall-2"})})
    save_result(other, (tmp_path / "results" / other_id).with_suffix(".json"))
    notice = "程式已更新，網頁伺服器要重開才看得了結果（請助理重開）"
    with _serve(tmp_path) as base:
        with _open(browser, base) as watched:
            watched.page.locator("#results-list tr").first.wait_for(state="visible")
            assert watched.page.locator("#server-notice").is_hidden()
        state["current"] = "calc-v1:" + "1" * 64
        with _open(browser, base) as watched:
            watched.page.locator("#server-notice").wait_for(state="visible")
            assert watched.page.locator("#server-notice").inner_text() == notice
        with _open(browser, f"{base}/results/{RUN_ID}") as watched:
            page = watched.page
            page.locator("#rejection").wait_for(state="visible")
            assert page.locator("#rejection-title").inner_text() == "網頁伺服器要重開"
            assert page.locator("#server-notice").inner_text() == notice
            assert page.locator("#reject-reason").inner_text() == ""
            assert page.locator("#rerun").is_hidden()
            assert page.locator("#fingerprint-rerun").is_hidden()
        with _open(browser, f"{base}/compare/{RUN_ID}/{other_id}") as watched:
            page = watched.page
            page.locator("#rejection").wait_for(state="visible")
            assert page.locator("#rejection-title").inner_text() == "網頁伺服器要重開"
            assert page.locator("#server-notice").inner_text() == notice
            assert page.locator("#reject-reason").inner_text() == ""
            assert not page.locator("#rerun-holder button").all()


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
            # 點了有反應：圖例換成這一支喇叭的線（「喇叭顯示名 → 座位顯示名」）。
            labels = _legend_labels(page)
            assert labels and all(label.startswith(f"{name} → ") for label in labels), (name, labels)
        for section in ("#categories", "#alerts", "#reverb", "#reflections", "#summary"):
            assert page.locator(section).inner_text().strip(), section
        # 警戒標題照印伺服器拼好的字串，網頁不自己拼：先逐筆的峰谷與聆聽區，再按牆對合併的顫動。
        data = page.request.get(f"{base}/api/results/{RUN_ID}").json()
        assert page.locator("#alerts h3").all_inner_texts() == [
            item["heading_text"] for item in [*data["alerts"], *data["flutter_groups"]]]
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
        data = page.request.get(f"{base}/api/results/{RUN_ID}").json()
        role = data["frequency_responses"][0]["role"]
        names = data["point_names"]
        # 主位對周圍點做成按鈕（按鈕字是伺服器給的顯示名）；周圍點彼此在下拉選單，另一張考卷考。
        choices = [item for item in data["listening_area"]["pair_choices"]
                   if item["role"] == role and item["group"] == "primary_to_surrounding"]
        assert choices, "沒有可點的位置對"
        # 還沒選之前明細是空的，點了才填——證明是點選觸發的，不是一打開就在。
        assert not page.locator("#pair-detail td").all_inner_texts()
        for choice in choices:
            # 每一對點之前先清空明細，上一對留下的表才不會讓「這一次沒反應」看起來也有字。
            page.locator("#pair-detail").evaluate("(element) => element.replaceChildren()")
            chosen = page.get_by_role("button", name=choice["text"], exact=True)
            chosen.click()
            page.locator("#pair-detail td").first.wait_for()
            _wait_for_lines(page)
            shown = {match for label in _legend_labels(page) for match in LEGEND_RECEIVER.findall(label)}
            assert shown == {names[choice["reference_id"]], names[choice["receiver_id"]]}, choice
            assert chosen.get_attribute("aria-pressed") == "true"
            cells = page.locator("#pair-detail td").all_inner_texts()
            assert all(cell.strip() for cell in cells), choice
            _assert_text_is_formatted(page)
        # 「全部位置」回到這支喇叭的全部曲線，明細清掉。
        page.get_by_role("button", name="全部位置", exact=True).click()
        _wait_for_lines(page)
        shown = {match for label in _legend_labels(page) for match in LEGEND_RECEIVER.findall(label)}
        assert shown == {names[item["receiver_id"]] for item in data["frequency_responses"]
                         if item["role"] == role and item["receiver_role"] in {"primary", "surrounding"}}
        assert not page.locator("#pair-detail td").all_inner_texts()
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
