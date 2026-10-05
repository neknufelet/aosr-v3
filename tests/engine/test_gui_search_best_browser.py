"""目前最佳三圖：真伺服器、版本輪詢、切換、缺檔清圖與 1440 寬版面。"""
from pathlib import Path

from playwright.sync_api import Browser

from aosr.reporting.result import SchemeResult
from tests.engine._gui_best_cases import best_result, best_store
from tests.engine._gui_cache import gui_startup_identity_memo
from tests.engine.test_gui_browser import _assert_quiet, _assert_text_is_formatted, _open, _serve, browser


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
    with _serve(tmp_path) as base, _open(browser, f"{base}/searches/{store.search_id}", viewport_width=1440) as watched:
        page = watched.page
        page.wait_for_function("() => document.getElementById('best-title').textContent.includes('細算第一名／原方案')")
        page.wait_for_function(_colored_canvas("#best-frequency-chart"))
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
        page.wait_for_timeout(5500)
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
        page.wait_for_function("() => [...document.querySelectorAll('#live-best .chart-error')].every(p=>p.textContent.includes('讀不到：檔案不存在'))", timeout=12000)
        assert page.locator("#live-best canvas").all() == []
        assert page.locator("#best-plan-content").is_hidden()
        _assert_text_is_formatted(page)
        _assert_quiet(watched)
