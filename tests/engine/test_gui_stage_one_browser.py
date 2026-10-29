"""第一階段真網頁：伺服器字串逐格呈現、兩種寬度與第二階段相容。"""
from pathlib import Path
import json

import pytest
from playwright.sync_api import Browser, Page

from aosr.runtime import configure_calculation_jax
from tests.engine._gui_cache import gui_startup_identity_memo
from tests.engine.test_gui_browser import _assert_quiet, _assert_text_is_formatted, _open, _serve, browser
from tests.engine.test_stage_one_shortlist import _finished
from aosr.search.stage_one_shortlist import RANK_AND_QUALITY
from tests.engine._search_room_loop_cases import layout_for
from tests.engine.test_gui_stage_one_view import second_stage
from tests.engine._stage_one_post_check_cases import prepared
from tests.engine.test_search_stage_one_post_check import run
from aosr.search.store import SearchStore
from tests.engine.test_gui_stage_one_best import ABSENT, AXES


def chart_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SearchStore:
    store = _finished(tmp_path, monkeypatch)
    # 對稱房間替身的左右主位曲線會重疊；只改考卷自己的右聲道原始能量，讓兩色像素都可辨識。
    for target in (*store.candidate_path(0).parent.glob("*.json"), *store.refine_dir.glob("*.json")):
        raw = json.loads(target.read_bytes())
        if "curves" not in raw:
            continue
        roles = {channel["speaker_id"]: channel["role"] for channel in raw["scheme"]["channel_group"]["channels"]}
        for curve in raw["curves"]:
            if roles[curve["speaker_id"]] == "right":
                curve["primary"]["fem_energy"] = [energy * 1.25 for energy in curve["primary"]["fem_energy"]]
        target.write_text(json.dumps(raw))
    return store


TWO_CURVE_COLORS = """() => {
  const colors = [[27,111,138], [192,57,43]], found = colors.map(() => false);
  for (const canvas of document.querySelectorAll('#best-frequency-chart canvas')) {
    const pixels = canvas.getContext('2d').getImageData(0,0,canvas.width,canvas.height).data;
    for (let i=0;i<pixels.length;i+=4) {
      colors.forEach((color,index) => {
        if (pixels[i+3] && color.every((value,offset) => Math.abs(pixels[i+offset]-value)<8)) found[index]=true;
      });
    }
  }
  return found.every(Boolean);
}"""
# 頻率軸刻度畫在畫布上讀不到；考卷自己包一層圖表庫，記下每張圖，讀它算好的刻度字（空字是不標的那格）。
CHART_SPY = """(() => {
  let wrapped;
  Object.defineProperty(window, "uPlot", {configurable: true, get: () => wrapped, set: (real) => {
    wrapped = function (opts, data, target) { const plot = new real(opts, data, target); (window.testCharts ||= []).push(plot); return plot; };
    Object.assign(wrapped, real);
  }});
})()"""
FREQUENCY_CHART = "window.testCharts.filter(plot => plot.root.closest('#best-frequency-chart')).at(-1)"
FREQUENCY_TICKS = f"() => {FREQUENCY_CHART}.axes[0]._values.filter(Boolean)"


def watch_charts(page: Page) -> None:
    page.add_init_script(CHART_SPY)
    page.reload(wait_until="networkidle")


def standard_ticks(low: float, high: float) -> list[str]:
    return [str(lead * 10 ** power) for power in (1, 2, 3) for lead in (1, 2, 3, 5) if low <= lead * 10 ** power <= high]


def drag_zoom(page: Page, start: float, end: float) -> tuple[float, float]:
    """在頻響圖上按住滑鼠從寬度的 start 拖到 end（比例），放開後回圖表庫的可見範圍。"""
    box = page.locator("#best-frequency-chart .u-over").bounding_box()
    assert box is not None
    middle = box["y"] + box["height"] / 2
    before = page.evaluate(f"() => {FREQUENCY_CHART}.scales.x.min")
    page.mouse.move(box["x"] + box["width"] * start, middle)
    page.mouse.down()
    page.mouse.move(box["x"] + box["width"] * end, middle, steps=8)
    page.mouse.up()
    page.wait_for_function(f"before => {FREQUENCY_CHART}.scales.x.min !== before", arg=before)
    page.evaluate("() => new Promise(done => requestAnimationFrame(() => done(null)))")
    low, high = page.evaluate(f"() => [{FREQUENCY_CHART}.scales.x.min, {FREQUENCY_CHART}.scales.x.max]")
    return float(low), float(high)


def _check_best_charts_at_width(page: Page, refine: tuple[Path, int], search: tuple[Path, int], tmp_path: Path,
                                width: int) -> None:
    """一種寬度：細算、搜尋、細算三次切換，每次核標題、兩色曲線、說明句、刻度、版面與截圖；再拖曳縮放核刻度。"""
    for which in ("refine", "search", "refine"):
        if which == "search" or page.locator("#best-search").get_attribute("aria-pressed") == "true":
            page.locator(f"#best-{which}").click()
        target, number = refine if which == "refine" else search
        raw = json.loads(target.read_bytes())
        label = "細算第一名" if which == "refine" else "搜尋第一名"
        title = f"目前最佳：{label}／試算 {number}／{AXES[raw['low_frequency_axis']]}"
        page.wait_for_function("title => document.getElementById('best-title').textContent === title", arg=title)
        page.wait_for_function(TWO_CURVE_COLORS)
        assert page.locator("#live-best .chart-error").all_text_contents() == ["" for _ in ("frequency", "rfz", "plan")]
        assert page.locator("#best-frequency-note").inner_text() == raw["wall_assumption_text"]
        assert page.locator("#best-rfz-absent").inner_text() == ABSENT
        # 沒有反射圖就不留「怎麼讀反射圖」那一句；20～300 Hz 標首位 1、2、3、5 的刻度（主對話看真資料截圖抓到）。
        assert page.locator("#rfz-explain").is_hidden()
        frequencies = raw["curves"][0]["primary"]["frequencies_hz"]
        assert page.evaluate(FREQUENCY_TICKS) == standard_ticks(frequencies[0], frequencies[-1])
        for selector in ("#best-frequency-note", "#best-rfz-absent"):
            assert page.locator(selector).is_visible()
            assert page.locator(selector).evaluate("e => getComputedStyle(e).color === getComputedStyle(document.body).color")
        assert page.locator("#best-plan-xy circle").count() > 0
        assert page.locator("#best-plan-xz circle").count() > 0
        assert page.locator("body").evaluate("e => e.scrollWidth <= window.innerWidth")
        assert page.locator("#live-best article").evaluate_all("nodes => nodes.every(e => e.scrollWidth <= e.clientWidth)")
        page.screenshot(path=str(tmp_path / f"best-{which}-{width}.png"), full_page=True)
    # 拖曳縮放後刻度照樣落在標準頻率上、不會整排消失（審查抓到：圖表庫縮放後的分割點從可見下界起算，21、31…）。
    low, high = drag_zoom(page, 0.05, 0.9)
    assert frequencies[0] < low < high < frequencies[-1]
    assert page.evaluate(FREQUENCY_TICKS) == standard_ticks(low, high)


def test_stage_one_best_charts_switch_without_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                   browser: Browser) -> None:
    """1440 與 900 兩種寬度同一頁輪流（#972 由兩格合成：換視窗大小後重新整理，每種寬度三次切換、刻度、版面、
    拖曳縮放照舊各核一次）；最後刪掉細算第一名的結果檔、假時鐘快轉一個輪詢週期，三張圖都改寫讀不到。"""
    configure_calculation_jax()
    store = chart_store(tmp_path, monkeypatch)
    summary = json.loads((store.path / "stage_one_refine.json").read_bytes())
    winner = next(row for row in summary["results"] if row["state"] == "scored" and row["refine_rank"] == 1)
    search = json.loads(store.status_path.read_bytes())
    with _serve(tmp_path) as base, _open(browser, f"{base}/searches/{store.search_id}", viewport_width=1440,
                                         fake_clock=True) as watched:
        page = watched.page
        watch_charts(page)
        for width in (1440, 900):
            if width != 1440:
                page.set_viewport_size({"width": width, "height": 1100})
                page.reload(wait_until="networkidle")
            _check_best_charts_at_width(page, (store.path / winner["result_file"], winner["trial_number"]),
                                        (store.candidate_path(search["best_trial"]), search["best_trial"]), tmp_path, width)
        (store.path / winner["result_file"]).unlink()
        page.clock.run_for(5000)
        page.wait_for_function("() => [...document.querySelectorAll('#live-best .chart-error')].every(e => e.textContent.startsWith('讀不到：'))", timeout=12000)
        assert page.locator("#best-frequency-note").is_hidden()
        assert page.locator("#best-rfz-absent").is_hidden()
        # 讀不到時不知道是哪一階段的圖，說明句跟第二階段一樣留著。
        assert page.locator("#rfz-explain").is_visible()
        assert page.locator("#best-frequency-chart canvas").all() == []
        assert page.locator("#best-plan-content").is_hidden()
        _assert_text_is_formatted(page)
        _assert_quiet(watched)


def test_stage_one_table_scope_pause_and_screenshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                  browser: Browser) -> None:
    """1440 與 900 兩種寬度同一頁輪流（#972 由兩格合成：換視窗大小後重新整理，斷言照舊每種寬度各核一次）。"""
    configure_calculation_jax()
    layout = layout_for(room_range={"Lx": {"low": 7.9, "high": 8.1}, "Ly": {"low": 7.9, "high": 8.1},
        "Lz": {"low": 2.6, "high": 2.7}})
    store = _finished(tmp_path, monkeypatch, budget=8, top_n=5, layout=layout)
    raw = json.loads(store.status_path.read_bytes())
    # 本題只驗搜尋投影；保留真細算文件，但搜尋尚無最佳結果可供三張圖讀回。
    raw.update(state="paused_time_limit", best_trial=None, best_score=None, message="單題時間上限暫停")
    store.status_path.write_text(json.dumps(raw))
    with _serve(tmp_path) as base, _open(browser, f"{base}/searches/{store.search_id}", viewport_width=1440) as watched:
        page = watched.page
        for width in (1440, 900):
            if width != 1440:
                page.set_viewport_size({"width": width, "height": 1100})
                page.reload(wait_until="networkidle")
            page.locator("#shortlist table").wait_for()
            data = page.request.get(f"{base}/api/searches/{store.search_id}").json()
            block = next(block for block in data["blocks"] if block["key"] == "shortlist")
            assert page.locator("#shortlist th").all_text_contents() == block["table"]["headings"]
            assert page.locator("#shortlist tr").evaluate_all("nodes => nodes.slice(1).map(n => [...n.cells].map(c => c.textContent))") == block["table"]["rows"]
            assert page.locator("#search-scope").inner_text() == "第一階段：房間尺寸與喇叭、座位擺位一起搜尋，只算 20～300 Hz 有限元素"
            pause = page.locator("#stage p", has_text="搜尋：因單題時間上限暫停")
            assert "notice" in (pause.get_attribute("class") or "")
            # 清單的固定說明句帶「未確認」三個字，但不是出錯，不准標紅（主對話看截圖抓到）。
            explanation = page.locator("#shortlist p", has_text=RANK_AND_QUALITY)
            assert "未確認" in RANK_AND_QUALITY and "notice" not in (explanation.get_attribute("class") or "")
            assert page.locator("body").evaluate("e => e.scrollWidth <= window.innerWidth")
            assert page.locator("#search-content section").evaluate_all("nodes => nodes.every(e => e.scrollWidth <= e.clientWidth)")
            wrapper = page.locator("#shortlist .table-scroll")
            assert wrapper.evaluate("e => e.scrollWidth > e.clientWidth")
            # 清單是全寬一塊，跟目前階段同寬，不是擠進兩欄的半格（植錯存活補的）。
            widths = page.evaluate("() => ['#stage', '#shortlist'].map(id => document.querySelector(id).getBoundingClientRect().width)")
            assert widths[0] == widths[1]
            page.screenshot(path=str(tmp_path / f"stage-one-{width}.png"), full_page=True)
            wrapper.evaluate("e => { e.scrollLeft = e.scrollWidth; }")
            page.locator("#shortlist").screenshot(path=str(tmp_path / f"stage-one-ratios-{width}.png"))
            _assert_text_is_formatted(page)
            _assert_quiet(watched)


def test_old_python_without_optional_fields_keeps_second_stage_scope(tmp_path: Path, browser: Browser) -> None:
    store = second_stage(tmp_path, False)
    with _serve(tmp_path) as base, _open(browser, f"{base}/searches/{store.search_id}", viewport_width=1440) as watched:
        page = watched.page
        page.locator("#stage").wait_for()
        data = page.request.get(f"{base}/api/searches/{store.search_id}").json()
        assert "scope_text" not in data
        assert all("table" not in block for block in data["blocks"])
        assert page.locator("#search-scope").inner_text() == "第二階段子集：房間與材料固定，只搜尋對稱的喇叭與座位擺位。"
        page.screenshot(path=str(tmp_path / "stage-two-1440.png"), full_page=True)
        _assert_text_is_formatted(page)
        _assert_quiet(watched)


def test_post_check_reason_words_do_not_turn_red(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, browser: Browser) -> None:
    # 入選後檢查的原因是寫入端原文（例如跳過時寫「讀不到細算結果」）：伺服器沒標警示就不准因字面標紅（審查反例）。
    configure_calculation_jax()
    store = prepared(tmp_path, monkeypatch)
    run(store, tmp_path / "child")
    path = store.path / "stage-one-check" / "summary.json"
    raw = json.loads(path.read_bytes())
    raw["seats"][0]["reason_text"] = "考卷：原因原文寫讀不到也失敗過，但這一席照常診斷"
    path.write_text(json.dumps(raw))
    with _serve(tmp_path) as base, _open(browser, f"{base}/searches/{store.search_id}", fake_clock=True) as watched:
        page = watched.page
        line = page.locator("#modal p", has_text="考卷：原因原文寫讀不到也失敗過")
        line.wait_for()
        data = page.request.get(f"{base}/api/searches/{store.search_id}").json()
        modal = next(block for block in data["blocks"] if block["key"] == "modal")
        assert modal["red"] == "warning_only" and modal["warning"] is False
        assert "notice" not in (line.get_attribute("class") or "")
        # 超線共振表照伺服器給的格子逐格畫出，包在可橫捲的框裡。
        cells = page.locator("#modal .table-scroll table tr").evaluate_all(
            "rows => rows.map(row => [...row.children].map(cell => cell.textContent))")
        assert modal["table"]["rows"] and cells == [modal["table"]["headings"], *modal["table"]["rows"]]
        # 每 5 秒整塊重畫：框內捲動位置要留著（審查抓到捲到一半被拉回最上面）。框改矮才捲得動。
        page.add_style_tag(content="#modal .table-scroll{max-height:4rem}")
        box = page.locator("#modal .table-scroll")
        old = box.element_handle()
        top = box.evaluate("element => { element.scrollTop = element.scrollHeight; return element.scrollTop; }")
        assert top > 0
        # 假時鐘快轉一個輪詢週期（#972），不再真等 5 秒。
        page.clock.run_for(5000)
        page.wait_for_function("element => !element.isConnected", arg=old, timeout=20000)
        assert box.evaluate("element => element.scrollTop") == top
        _assert_quiet(watched)
