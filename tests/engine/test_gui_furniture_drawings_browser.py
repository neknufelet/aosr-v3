"""四頁家具圖面與實際 SVG：位置、比例、點的層序、天雲及搜尋重讀。"""
import copy
from pathlib import Path

import pytest
from playwright.sync_api import Browser, Page

from aosr.reporting.result import SchemeResult, save_result
from aosr.reporting.scheme import Scheme
from tests.engine._gui_best_cases import best_store
from tests.engine._gui_cache import gui_startup_identity_memo as gui_startup_identity_memo
from tests.engine._gui_furniture_drawings import listening_document, working_document
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
            points = page.locator('#plan-xy [data-furniture="sofa"] polygon').evaluate("p => {const room=p.ownerSVGElement.querySelector('rect'); const scale=Number(room.getAttribute('width'))/6; return [...p.points].map(q => [(q.x-Number(room.getAttribute('x')))/scale, (Number(room.getAttribute('y'))+Number(room.getAttribute('height'))-q.y)/scale]);}")
            for actual, expected in zip(points, [(3.025, 1), (3.775, 1), (3.775, 2.8), (3.025, 2.8)], strict=True):
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
        assert "外框" not in page.locator("#plan-key").inner_text()
        page.locator("#plan-key").locator("xpath=..").screenshot(path=str(tmp_path / "compare.png"))
        page.evaluate("() => {document.querySelectorAll('[data-furniture]').forEach(node => {if(node.dataset.furniture !== 'cloud') node.remove();}); drawPlanKey();}")
        assert page.locator('#plan-key [data-mark="furniture"]').is_hidden()
        assert page.locator('#plan-key [data-mark="cloud"]').is_visible()
        _assert_quiet(watched)


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
