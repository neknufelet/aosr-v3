"""目前最佳三圖：真伺服器、版本輪詢、切換、缺檔清圖與 1440 寬版面。"""
from pathlib import Path

from playwright.sync_api import Browser, Route

from aosr.reporting.result import SchemeResult
from tests.engine._gui_best_cases import best_result, best_store
from tests.engine._gui_cache import gui_startup_identity_memo
from tests.engine.test_gui_browser import _assert_quiet, _assert_text_is_formatted, _open, _serve, browser
from tests.engine.test_gui_stage_one_browser import FREQUENCY_TICKS, watch_charts


def _colored_canvas(selector: str) -> str:
    return """() => [...document.querySelectorAll('%s canvas')].some(canvas => {
      const pixels = canvas.getContext('2d').getImageData(0,0,canvas.width,canvas.height).data;
      for(let i=0;i<pixels.length;i+=4) {
        if(pixels[i+3] && Math.max(pixels[i],pixels[i+1],pixels[i+2])-Math.min(pixels[i],pixels[i+1],pixels[i+2])>60) return true;
      } return false;
    })""" % selector


def test_best_charts_switch_poll_only_versions_and_clear_missing(tmp_path: Path, browser: Browser,
                                                              best_result: SchemeResult) -> None:
    store = best_store(tmp_path, best_result, "baseline")
    # 假時鐘（#972）：輪詢 5 秒一次改用 run_for 快轉，不再真等；時間照常走，快轉只是不必等。
    with _serve(tmp_path) as base, _open(browser, f"{base}/searches/{store.search_id}", viewport_width=1440,
                                         fake_clock=True) as watched:
        page = watched.page
        watch_charts(page)
        page.wait_for_function("() => document.getElementById('best-title').textContent.includes('細算第一名／原方案')")
        page.wait_for_function(_colored_canvas("#best-frequency-chart"))
        # 第二階段跨三個十倍以上，頻率軸照圖表庫原樣只標十的次方、帶一位小數；反射圖說明句照留（第一階段改刻度與藏說明句時補的）。
        frequencies = page.request.get(f"{base}/api/searches/{store.search_id}/best?which=refine").json()["frequency"]["data"][0]
        assert frequencies[-1] / frequencies[0] >= 100
        assert page.evaluate(FREQUENCY_TICKS) == [f"{10 ** power:.1f}" for power in range(1, 6) if frequencies[0] <= 10 ** power <= frequencies[-1]]
        assert page.locator("#rfz-explain").is_visible()
        page.wait_for_function(_colored_canvas("#best-rfz-charts"))
        rfz_data = page.request.get(f"{base}/api/searches/{store.search_id}/best?which=refine").json()["rfz"]
        assert page.locator("#best-rfz-charts h4").all_text_contents() == [channel["label"] for channel in rfz_data["channels"]]
        for index, _ in enumerate(rfz_data["channels"], start=1):
            page.wait_for_function(_colored_canvas(f"#best-rfz-charts > div:nth-child({index})"))
        assert page.locator("#best-plan-xy circle").count() > 0
        assert page.locator("#best-plan-xz circle").count() > 0
        assert "窗內" in page.locator("#best-rfz").inner_text()
        assert "dB" in page.locator("#rfz-zones").inner_text()
        requests: list[str] = []
        page.on("request", lambda request: requests.append(request.url) if "/best?" in request.url else None)
        # 快轉一個輪詢週期，等到那一次輪詢的回覆真的回來（證明輪詢有發生），再留一點真時間給回覆後的頁面腳本：
        # 版本沒變就不准重抓三張圖。
        with page.expect_response(lambda response: response.url == f"{base}/api/searches/{store.search_id}"):
            page.clock.run_for(5000)
        page.wait_for_timeout(500)
        assert requests == []
        page.locator("#best-search").click()
        page.wait_for_function("() => document.getElementById('best-title').textContent.includes('搜尋第一名／試算 0')")
        page.wait_for_function(_colored_canvas("#best-frequency-chart"))
        assert requests == [f"{base}/api/searches/{store.search_id}/best?which=search"]
        boxes = page.locator("#live-best article").evaluate_all(
            "nodes => nodes.map(n => {const b=n.getBoundingClientRect();return {id:n.id,x:b.x,y:b.y,w:b.width,h:b.height,scroll:n.scrollWidth,client:n.clientWidth};})")
        for box in boxes:
            assert box["w"] > 0 and box["h"] > 0
            assert box["x"] >= 0 and box["x"] + box["w"] <= 1440
            assert box["scroll"] <= box["client"], box
            for other in boxes:
                if box["id"] != other["id"]:
                    assert box["y"] + box["h"] <= other["y"] or other["y"] + other["h"] <= box["y"]
        store.candidate_path(0).unlink()
        page.clock.run_for(5000)
        page.wait_for_function("() => [...document.querySelectorAll('#live-best .chart-error')].every(p=>p.textContent.includes('讀不到：檔案不存在'))", timeout=12000)
        assert page.locator("#live-best canvas").all() == []
        assert page.locator("#best-plan-content").is_hidden()
        _assert_text_is_formatted(page)
        _assert_quiet(watched)


def test_chart_failing_midway_clears_only_that_chart(tmp_path: Path, browser: Browser,
                                                    best_result: SchemeResult) -> None:
    store = best_store(tmp_path, best_result, "baseline")
    with _serve(tmp_path) as base, _open(browser, f"{base}/searches/{store.search_id}", viewport_width=1440) as watched:
        page = watched.page
        page.wait_for_function(_colored_canvas("#best-frequency-chart"))

        def broken(route: Route) -> None:
            # 反射圖的門檻與時間窗已寫上、聲道缺了點列：畫到一半出錯，只清那一張（連已寫的門檻文字）、寫原因，
            # 頻響與平面圖照常顯示。
            data = route.fetch().json()
            data["rfz"]["channels"] = [{"label": "缺點列的聲道"}]
            route.fulfill(json=data)
        page.route("**/best?**", broken)
        page.locator("#best-search").click()
        page.wait_for_function("() => document.querySelector('#best-rfz .chart-error').textContent.startsWith('讀不到：這張圖畫不出來')")
        page.wait_for_function(_colored_canvas("#best-frequency-chart"))
        assert page.locator("#best-rfz canvas").all() == []
        assert page.locator("#rfz-window").inner_text() == ""
        assert page.locator("#rfz-zones span").all() == []
        assert page.locator("#best-rfz-charts > *").all() == []
        assert page.locator("#best-plan-content").is_visible()
        assert page.locator("#best-frequency .chart-error").inner_text() == ""
        assert page.locator("#best-plan .chart-error").inner_text() == ""
        _assert_quiet(watched)
