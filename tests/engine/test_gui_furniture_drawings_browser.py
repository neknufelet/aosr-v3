"""四頁家具圖面與實際 SVG：位置、比例、點的層序、天雲及搜尋重讀。"""
import copy
from math import sqrt
from pathlib import Path
from typing import cast

import pytest
from playwright.sync_api import Browser, Page

from aosr.reporting.result import SchemeResult, save_result
from aosr.reporting.scheme import Scheme
from tests.engine._gui_best_cases import best_store
from tests.engine._gui_cache import gui_startup_identity_memo as gui_startup_identity_memo
from tests.engine._gui_furniture_drawings import listening_document, working_document
from tests.engine._furniture_cases import relative_item
from tests.engine.test_scheme_furniture import _validation_document
from tests.engine.test_gui_browser import (
    _assert_quiet, _assert_text_is_formatted, _open, _serve, browser as browser,
)
from tests.engine.test_scheme_pipeline import _run_control


@pytest.fixture(scope="module")
def furnished_results() -> tuple[SchemeResult, SchemeResult]:
    first = listening_document()
    second = copy.deepcopy(first)
    second["scheme_id"] = "listen-moved"
    receivers = second["receiver_set"]
    assert isinstance(receivers, dict)
    for point in receivers["points"]:
        point["position_m"][0] += 0.15
    return _run_control(Scheme.model_validate(first)), _run_control(Scheme.model_validate(second))


def _assert_boxes(page: Page, xy: str, xz: str, furniture_ids: set[str]) -> None:
    for selector in (xy, xz):
        svg = page.locator(selector)
        assert set(svg.locator("[data-furniture]").evaluate_all("nodes => nodes.map(n => n.dataset.furniture)")) == furniture_ids
        assert set(svg.locator("[data-cabinet]").evaluate_all("nodes => nodes.map(n => n.dataset.cabinet)")) == {"left", "right"}
        assert svg.locator("[data-furniture] text, [data-cabinet] text").all() == []
        assert svg.locator("[data-furniture] rect, [data-cabinet] rect").all() == []
        assert svg.evaluate("svg => [...svg.querySelectorAll('[data-furniture], [data-cabinet]')].every(box => [...svg.querySelectorAll('g[data-keys]')].every(point => !!(box.compareDocumentPosition(point) & Node.DOCUMENT_POSITION_FOLLOWING)))")
        assert svg.evaluate("svg => [...svg.querySelectorAll('polygon')].every(p => [...p.points].every(point => point.x >= 0 && point.y >= 0 && point.x <= svg.viewBox.baseVal.width && point.y <= svg.viewBox.baseVal.height))")
        assert svg.locator("[data-furniture] title").all_text_contents()
    _assert_text_is_formatted(page)


def _polygon_meters(page: Page, selector: str) -> list[list[float]]:
    return cast(list[list[float]], page.locator(selector).evaluate("""p => {
      const room = p.ownerSVGElement.querySelector('rect');
      const scale = Number(room.getAttribute('width')) / 6;
      return [...p.points].map(q => [(q.x - Number(room.getAttribute('x'))) / scale,
        (Number(room.getAttribute('y')) + Number(room.getAttribute('height')) - q.y) / scale]);
    }"""))


@pytest.mark.parametrize("work", [False, True])
def test_input_draws_boxes_before_points_and_leaves_zoom_clear(browser: Browser, tmp_path: Path, work: bool) -> None:
    content = working_document() if work else listening_document()
    scheme = Scheme.model_validate(content)
    (tmp_path / "schemes").mkdir()
    (tmp_path / "schemes" / f"{scheme.scheme_id}.json").write_text(scheme.model_dump_json())
    with _serve(tmp_path) as url, _open(browser, url, viewport_width=1440) as watched:
        page = watched.page
        page.locator("#scheme-list").select_option(scheme.scheme_id)
        page.locator("#open-scheme").click()
        page.wait_for_selector("#plan-xy [data-cabinet]")
        names = {item.furniture_id for item in scheme.furniture or ()}
        _assert_boxes(page, "#plan-xy", "#plan-xz", names)
        assert page.locator("#zoom-xy polygon, #zoom-xz polygon").all() == []
        assert names <= set(page.locator("#plan-legend li").evaluate_all("nodes => nodes.map(n => n.dataset.id.replace('furniture:', ''))"))
        if not work:
            cloud = page.locator('#plan-xy [data-furniture="cloud"] polygon')
            assert cloud.get_attribute("fill") == "none"
            assert cloud.get_attribute("stroke-dasharray")
            assert page.locator('#plan-xy [data-furniture="sofa"] polygon').get_attribute("fill-opacity") == "0.22"
            assert page.locator('#plan-xy [data-furniture="table"] polygon').get_attribute("fill") == "none"
            # 用房間框的比例把畫出的沙發還原成公尺，答案手算，不取 /api/plan 當答案。
            points = _polygon_meters(page, '#plan-xy [data-furniture="sofa"] polygon')
            for actual, expected in zip(points, [(3.025, 1), (3.775, 1), (3.775, 2.8), (3.025, 2.8)], strict=True):
                assert actual == pytest.approx(expected)
            side = _polygon_meters(page, '#plan-xz [data-furniture="sofa"] polygon')
            for actual, expected in zip(side, [(3.025, 0), (3.775, 0), (3.775, 0.65), (3.025, 0.65)], strict=True):
                assert actual == pytest.approx(expected)
            # 左箱體朝主位 (11,3)/√130；側面只取 x 投影及腳架箱底 1.2−0.205。
            low, high = 1 - (0.28 * 11 + 0.105 * 3) / sqrt(130), 1 + 0.105 * 3 / sqrt(130)
            side = _polygon_meters(page, '#plan-xz [data-cabinet="left"] polygon')
            for actual, expected in zip(side, [(low, 0.995), (high, 0.995), (high, 1.35), (low, 1.35)], strict=True):
                assert actual == pytest.approx(expected)
        target = page.locator("#plan-xy [data-furniture]").last
        detail = target.locator("title").text_content()
        target.focus()
        target.press("Enter")
        assert page.locator("#plan-detail").inner_text() == detail
        page.locator("#plan-detail").evaluate("node => node.textContent = ''")
        target.locator("polygon").click(force=True)
        assert page.locator("#plan-detail").inner_text() == detail
        assert "箱體只做碰撞檢查，反射暫不計" in page.locator("#plan-legend").inner_text()
        target.evaluate("node => node.blur()")
        page.locator("#plan-drawings").locator("xpath=..").screenshot(path=str(tmp_path / f"input-{'work' if work else 'listen'}.png"))
        _assert_quiet(watched)


def test_result_draws_saved_snapshot_even_when_scheme_file_has_changed(
    browser: Browser, tmp_path: Path, furnished_results: tuple[SchemeResult, SchemeResult],
) -> None:
    result, _ = furnished_results
    (tmp_path / "results").mkdir()
    (tmp_path / "schemes").mkdir()
    (tmp_path / "schemes" / "listen.json").write_text(Scheme.model_validate(working_document()).model_dump_json())
    run_id = "a" * 32
    save_result(result, tmp_path / "results" / f"{run_id}.json")
    with _serve(tmp_path) as url, _open(browser, f"{url}/results/{run_id}", viewport_width=1440) as watched:
        page = watched.page
        page.wait_for_selector("#result-plan-xy [data-furniture]")
        _assert_boxes(page, "#result-plan-xy", "#result-plan-xz", {"sofa", "table", "cloud"})
        assert page.locator("#result-plan [data-reflection]").all() == []
        page.locator("#result-plan").screenshot(path=str(tmp_path / "result.png"))
        _assert_quiet(watched)


def test_compare_marks_absolute_furniture_move_without_changing_relative_fields(
    browser: Browser, tmp_path: Path, furnished_results: tuple[SchemeResult, SchemeResult],
) -> None:
    first, second = furnished_results
    assert first.scheme.furniture == second.scheme.furniture
    (tmp_path / "results").mkdir()
    for name, result in (("a", first), ("b", second)):
        save_result(result, tmp_path / "results" / f"{name * 32}.json")
    with _serve(tmp_path) as url, _open(browser, f"{url}/compare/{'a' * 32}/{'b' * 32}", viewport_width=1440) as watched:
        page = watched.page
        page.wait_for_selector("#plan-a-xy [data-furniture]")
        for side in ("a", "b"):
            _assert_boxes(page, f"#plan-{side}-xy", f"#plan-{side}-xz", {"sofa", "table", "cloud"})
            assert set(page.locator(f"#plan-{side}-xy .changed-furniture").evaluate_all("nodes => nodes.map(n => n.parentElement.dataset.furniture)")) == {"sofa", "table"}
        assert page.locator('#plan-key [data-mark="changed-furniture"]').is_visible()
        assert page.locator('#plan-key [data-mark="changed-furniture"]').inner_text() == (
            "紫紅加粗邊線是實際位置或大小有變的家具（座位一搬，跟著走的家具也算）")
        assert "外框" not in page.locator("#plan-key").inner_text()
        page.locator("#plan-key").locator("xpath=..").screenshot(path=str(tmp_path / "compare.png"))
        page.evaluate("() => {document.querySelectorAll('[data-furniture]').forEach(node => {if(node.dataset.furniture !== 'cloud') node.remove();}); drawPlanKey();}")
        assert page.locator('#plan-key [data-mark="furniture"]').is_hidden()
        assert page.locator('#plan-key [data-mark="cloud"]').is_visible()
        _assert_quiet(watched)


def test_input_undefined_cabinet_note_keeps_other_cabinet(browser: Browser, tmp_path: Path) -> None:
    content = listening_document()
    content.pop("furniture")
    content["speakers"] = {"left": {"x": 3.2, "y": 1.9, "z": 0.5},
                           "right": {"x": 1.0, "y": 2.5, "z": 1.2}}
    scheme = Scheme.model_validate(content)
    (tmp_path / "schemes").mkdir()
    (tmp_path / "schemes" / "listen.json").write_text(scheme.model_dump_json())
    with _serve(tmp_path) as url, _open(browser, url) as watched:
        page = watched.page
        page.locator("#scheme-list").select_option("listen")
        page.locator("#open-scheme").click()
        page.wait_for_selector('#plan-xy [data-cabinet="right"]')
        for selector in ("#plan-xy", "#plan-xz"):
            assert page.locator(f"{selector} [data-cabinet]").evaluate_all(
                "nodes => nodes.map(n => n.dataset.cabinet)") == ["right"]
        assert "左聲道喇叭箱體沒畫：喇叭跟主位在同一個水平位置，定不出朝向" in page.locator("#plan-legend").inner_text()
        page.locator("#plan-drawings").locator("xpath=..").screenshot(path=str(tmp_path / "undefined-cabinet.png"))
        _assert_quiet(watched)


def test_input_outside_and_blocked_furniture_remains_visible(browser: Browser, tmp_path: Path) -> None:
    content = _validation_document("both") | {"scheme_id": "mix"}
    # 一件在 x 負側完全出界，另一件擋直達。
    content["furniture"] = [*cast(list[dict[str, object]], content["furniture"]),
        relative_item(placement={"forward_m": 1, "left_m": 4, "bottom_height_m": 0, "yaw_deg": 0})]
    scheme = Scheme.model_validate(content)
    (tmp_path / "schemes").mkdir()
    (tmp_path / "schemes" / "mix.json").write_text(scheme.model_dump_json())
    with _serve(tmp_path) as url, _open(browser, url, viewport_width=1440) as watched:
        page = watched.page
        page.locator("#scheme-list").select_option("mix")
        page.locator("#open-scheme").click()
        page.wait_for_selector('#plan-xy [data-furniture="seat"]')
        page.locator("#check").click()
        page.wait_for_function("document.getElementById('messages').textContent.includes('超出房間')")
        messages = page.locator("#messages").inner_text()
        assert "家具 seat 超出房間接觸界線" in messages
        assert "直達路徑被家具 desk 擋住" in messages
        for selector in ("#plan-xy", "#plan-xz"):
            svg = page.locator(selector)
            assert svg.locator('[data-furniture="desk"] .blocked-furniture').is_visible()
            assert svg.locator('[data-furniture="seat"] .blocked-furniture').all() == []
            assert svg.evaluate("svg => [...svg.querySelectorAll('polygon')].every(p => [...p.points].every(q => q.x > 0 && q.y > 0 && q.x < svg.viewBox.baseVal.width && q.y < svg.viewBox.baseVal.height))")
            svg.locator('[data-furniture="seat"] polygon').click()
            assert "沙發（seat）" in page.locator("#plan-detail").inner_text()
        page.locator('#plan-xz [data-furniture="seat"]').evaluate("node => node.blur()")
        assert not watched.page_errors
        assert all("422" in error for error in watched.console_errors), watched.console_errors
        page.locator("#messages").screenshot(path=str(tmp_path / "outside-blocked-messages.png"))
        page.locator("#plan-drawings").locator("xpath=..").screenshot(path=str(tmp_path / "outside-blocked-plan.png"))


def test_search_best_draws_boxes_and_does_not_reload_unchanged_winner(
    browser: Browser, tmp_path: Path, furnished_results: tuple[SchemeResult, SchemeResult],
) -> None:
    result, _ = furnished_results
    store = best_store(tmp_path, result, "baseline")
    with _serve(tmp_path) as url, _open(browser, f"{url}/searches/{store.search_id}", viewport_width=1440) as watched:
        page = watched.page
        page.wait_for_selector("#best-plan-xy [data-furniture]")
        _assert_boxes(page, "#best-plan-xy", "#best-plan-xz", {"sofa", "table", "cloud"})
        requests: list[str] = []
        page.on("request", lambda request: requests.append(request.url) if "/best?" in request.url else None)
        page.wait_for_timeout(5500)
        assert requests == []
        page.locator("#best-plan").screenshot(path=str(tmp_path / "search.png"))
        _assert_quiet(watched)
