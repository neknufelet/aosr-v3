"""材料頁吸音曲線的真瀏覽器考卷（#1082 段 2-2b）：圖真的有畫、標語與提醒看得到、未存的值能重畫、晚到的舊回應不蓋新的、手機寬度放得下。

曲線要雙精度，伺服器用正式網頁入口起在獨立行程（``real_server``），不改 pytest 行程的算數設定。
截圖不寫在這裡（考卷不准寫真 repo 以外的地方）；給人看的截圖由施工者另外照同一套動作截。
"""
from __future__ import annotations

from pathlib import Path
from typing import cast

from playwright.sync_api import Browser, Page, Route, expect

from aosr.gui.material_editor import user_library_path
from tests.engine._construction_gui_server import real_server
from tests.engine.test_gui_browser import RAW_VALUES, _open, browser as browser

# 等伺服器算完的斷言上限：冷行程第一次算一種新構造要編譯，幾個工人同時跑或雲端機器上會超過預設的 5 秒。
COMPUTED_MS = 60_000
# 數畫布上的像素：曲線色（#15617b，不透明）、外推的橘底、補值的灰底（兩種底色都是半透明，只看色相與透明度）。
PIXELS_JS = """(canvas) => {
  const data = canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data;
  const near = (r, g, b, R, G, B) => Math.abs(r - R) < 25 && Math.abs(g - G) < 25 && Math.abs(b - B) < 25;
  const counts = {line: 0, extrapolation: 0, model_fill: 0};
  for (let i = 0; i < data.length; i += 4) {
    const [r, g, b, a] = [data[i], data[i + 1], data[i + 2], data[i + 3]];
    if (a > 200 && near(r, g, b, 0x15, 0x61, 0x7b)) counts.line += 1;
    else if (a > 30 && a < 90 && near(r, g, b, 230, 140, 40)) counts.extrapolation += 1;
    else if (a > 30 && a < 90 && near(r, g, b, 110, 125, 135)) counts.model_fill += 1;
  }
  return counts;
}"""


def _pixels(page: Page) -> dict[str, int]:
    return cast(dict[str, int], page.locator("#curve-chart canvas").evaluate(PIXELS_JS))


def _row(page: Page, material_id: str) -> None:
    page.locator(f'#builtin-list tr[data-id="{material_id}"] button.open').click()


def test_a_builtin_opens_with_its_curve_labels_and_shaded_extrapolation(tmp_path: Path, browser: Browser) -> None:
    with real_server(tmp_path) as base, _open(browser, f"{base}/materials") as watched:
        page = watched.page
        _row(page, "glass_wool_48k")
        expect(page.locator("#curve-chart canvas")).to_have_count(1, timeout=COMPUTED_MS)
        expect(page.locator("#curve-basis")).to_have_text("阻抗怎麼得到：模擬，之後要拿實測對照")
        expect(page.locator("#curve-basis")).to_have_class("badge basis-simulated")
        expect(page.locator("#curve-provenance")).to_have_text("數字從哪來：文獻")
        expect(page.locator("#curve-arrangement")).to_contain_text("這條是初值排法")
        note = page.locator('#curve-warnings li[data-code="komatsu_frequency_extrapolation"]')
        expect(note).to_be_visible()
        expect(note).to_have_class("note")
        expect(page.locator('#curve-key li[data-kind="extrapolation"]')).to_be_visible()
        expect(page.locator('#curve-key li[data-kind="model_fill"]')).to_be_hidden()
        expect(page.locator("#curve-scope")).to_contain_text("沒有提醒不代表在可信範圍內")
        pixels = _pixels(page)
        assert pixels["line"] > 200 and pixels["extrapolation"] > 500 and pixels["model_fill"] == 0, pixels
        assert not RAW_VALUES.search(page.locator("body").inner_text())
        assert (watched.console_errors, watched.page_errors) == ([], [])


def test_redraw_uses_unsaved_form_values_and_says_when_the_curve_is_stale(tmp_path: Path, browser: Browser) -> None:
    with real_server(tmp_path) as base, _open(browser, f"{base}/materials") as watched:
        page = watched.page
        state = page.locator("#curve-state")
        page.locator("#new-fixed").click()
        expect(state).to_have_text("表單還沒填齊，曲線畫不出來：數字從哪來：必填；出處與條件：不可空白；"
                                   "多層構造›第 1 層›多孔層›層厚度：必填；多層構造›第 1 層›多孔層›流阻：必填", timeout=COMPUTED_MS)
        expect(page.locator("#curve-chart canvas")).to_have_count(0)
        expect(page.locator("#curve-labels")).to_be_hidden()
        page.locator("#provenance").select_option("model")
        page.locator("#source").fill("自己估的")
        first = page.locator("#assembly-layers > li").nth(0)
        first.locator('input[name="thickness_m"]').fill("0.05")
        first.locator('input[name="flow_resistivity_pa_s_m2"]').fill("15000")
        page.locator("#curve-redraw").click()
        expect(page.locator("#curve-chart canvas")).to_have_count(1, timeout=COMPUTED_MS)
        expect(state).to_have_text("")
        page.locator("#panel-assembly button.add-layer").click()  # 加層是按鈕，不觸發 input，也算改過
        expect(state).to_contain_text("表單改過，曲線還是上次畫的")
        page.locator("#assembly-layers > li").nth(1).locator("button.remove").click()
        first.locator('input[name="flow_resistivity_pa_s_m2"]').fill("5000")
        page.locator("#curve-redraw").click()
        expect(page.locator('#curve-warnings li.strong[data-code="komatsu_flow_below_samples"]')).to_be_visible(timeout=COMPUTED_MS)
        expect(state).to_have_text("")
        first.locator('input[name="flow_resistivity_pa_s_m2"]').fill("3000")
        page.locator("#curve-redraw").click()
        expect(state).to_contain_text("這組參數算不出吸音曲線", timeout=COMPUTED_MS)
        expect(page.locator("#curve-chart canvas")).to_have_count(0)
        expect(page.locator("#curve-warnings li")).to_have_count(0)
        page.locator("#fixed-kind").select_option("measured_impedance")
        page.locator("#points").fill("100 950\n400 700")
        page.locator("#measured-source").fill("阻抗管、垂直入射")
        page.locator("#curve-redraw").click()
        expect(page.locator("#curve-basis")).to_have_text("阻抗怎麼得到：近似", timeout=COMPUTED_MS)
        expect(page.locator("#curve-basis")).to_have_class("badge basis-approximate")
        expect(page.locator('#curve-key li[data-kind="model_fill"]')).to_be_visible()
        assert _pixels(page)["model_fill"] > 500
        assert not user_library_path(tmp_path).exists()
        assert (watched.console_errors, watched.page_errors) == ([], [])


def test_a_late_reply_for_the_previous_material_never_replaces_the_one_on_screen(tmp_path: Path, browser: Browser) -> None:
    with real_server(tmp_path) as base, _open(browser, f"{base}/materials") as watched:
        page = watched.page
        held: list[Route] = []

        def hold_first(route: Route) -> None:
            if held:
                route.continue_()
            else:
                held.append(route)

        page.route("**/api/material-curve", hold_first)
        _row(page, "glass_wool_48k")
        expect(page.locator("#curve-state")).to_contain_text("計算吸音曲線中")
        _row(page, "fabric_pet_a_gap100")
        expect(page.locator("#curve-chart canvas")).to_have_count(1, timeout=COMPUTED_MS)
        expect(page.locator("#curve-arrangement")).to_be_hidden()
        shown = page.locator("#curve-source").inner_text()
        with page.expect_response("**/api/material-curve", timeout=COMPUTED_MS):
            held[0].continue_()
        page.wait_for_timeout(500)
        expect(page.locator("#curve-arrangement")).to_be_hidden()  # 48K 棉那份晚到的回應有初值排法；蓋上去就會現出來
        assert page.locator("#curve-source").inner_text() == shown
        expect(page.locator("#curve-redraw")).to_be_enabled()
        assert (watched.console_errors, watched.page_errors) == ([], [])


def test_the_curve_fits_a_phone_and_follows_a_resize(tmp_path: Path, browser: Browser) -> None:
    with real_server(tmp_path) as base, _open(browser, f"{base}/materials", viewport_width=390) as watched:
        page = watched.page
        _row(page, "glass_wool_48k")
        expect(page.locator("#curve-chart canvas")).to_have_count(1, timeout=COMPUTED_MS)
        assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")
        chart, canvas = page.locator("#curve-chart").bounding_box(), page.locator("#curve-chart canvas").bounding_box()
        assert chart is not None and canvas is not None
        assert canvas["width"] <= chart["width"] + 0.5
        page.set_viewport_size({"width": 1000, "height": 1100})
        page.wait_for_function("document.querySelector('#curve-chart canvas').getBoundingClientRect().width > 500")
        assert (watched.console_errors, watched.page_errors) == ([], [])
