"""構造註記、白話比較列與輸入拒收在真瀏覽器顯示。"""
from pathlib import Path

import pytest
from playwright.sync_api import Browser, Route

from aosr.gui.compare_view import scheme_differences
from aosr.reporting.result import SchemeResult
from aosr.reporting.scheme import Scheme
from tests.engine._gui_cache import gui_load_result_memo as gui_load_result_memo, gui_startup_identity_memo as gui_startup_identity_memo
from tests.engine._surface_zone_cases import mixed_scheme
from tests.engine._furniture_cases import reference_document
from tests.engine.test_construction_report_notes import note
from tests.engine.test_gui_browser import _open, _serve, browser as browser, result as result
from tests.engine.test_gui_compare_browser import A_ID, B_ID, pair as pair
from tests.engine.test_gui_compare_routes import _files


def test_result_shows_saved_wall_layer_and_extrapolation(tmp_path: Path, browser: Browser, result: SchemeResult) -> None:
    first = result.pairs[0]
    changed = first.model_copy(update={"report": first.report.model_copy(update={
        "construction_notes_by_wall": {"x0": (note((20., 40.)),)}})})
    saved = result.model_copy(update={"pairs": (changed, *result.pairs[1:])})
    with _serve(tmp_path) as base:
        _files(tmp_path, saved, A_ID)
        with _open(browser, f"{base}/results/{A_ID}") as watched:
            lines = watched.page.locator("#construction-note-lines")
            lines.wait_for(state="visible")
            assert lines.inner_text() == "x 起點牆第 1 層：20–40 Hz 超出材料模型原研究的量測範圍，仍以外推照算"
            assert watched.page_errors == []


@pytest.mark.parametrize("width", (1440, 390))
def test_compare_keeps_plain_construction_summary(tmp_path: Path, browser: Browser,
        pair: tuple[SchemeResult, SchemeResult], width: int) -> None:
    document = mixed_scheme().model_dump(mode="json")
    document["scene"]["room_m"]["Lx"] += 1.0  # 同一張表混一般列：手機上每一列都要自己標 A／B
    rows = scheme_differences(Scheme.model_validate(reference_document()), Scheme.model_validate(document))
    assert any(row.path.startswith("scene.surface_zones_by_wall.") for row in rows)
    assert any(not row.path.startswith("scene.surface_zones_by_wall.") for row in rows)
    with _serve(tmp_path) as base:
        _files(tmp_path, pair[0], A_ID)
        _files(tmp_path, pair[1], B_ID)
        with _open(browser, f"{base}/compare/{A_ID}/{B_ID}", viewport_width=width) as watched:
            def construction_rows(route: Route) -> None:
                response = route.fetch()
                data = response.json()
                data["changes"] = [row.model_dump(mode="json") for row in rows]
                route.fulfill(response=response, json=data)
            watched.page.route(f"**/api/compare/{A_ID}/{B_ID}", construction_rows)
            watched.page.reload()
            table = watched.page.locator("#changes")
            watched.page.wait_for_function("document.getElementById('changes').textContent.includes('多層構造')")
            text = table.inner_text()
            assert "固定阻抗" in text and "薄板層" in text and "空氣層" in text
            assert "0.15 kg/m²" in text and "0.43 m" in text
            for code in ("assembly", "constant_impedance", "board", "rigid"):
                assert code not in text
            sides = watched.page.evaluate("""() => [...document.querySelectorAll('#changes tr')].slice(1).map((row) =>
                [row.cells[1].dataset.side, row.cells[2].dataset.side, getComputedStyle(row.cells[1], '::before').content,
                 getComputedStyle(row.cells[2], '::before').content])""")
            assert sides and all(a == "A" and b == "B" for a, b, _a_mark, _b_mark in sides)
            header = watched.page.locator("#changes tr:has(th)")
            if width < 600:
                assert header.is_hidden()
                assert all("A" in a_mark and "B" in b_mark for _a, _b, a_mark, b_mark in sides), sides
            else:
                assert header.is_visible()
                assert all(a_mark == b_mark == "none" for _a, _b, a_mark, b_mark in sides), sides
            assert watched.page_errors == []


def test_input_explains_construction_refusal(tmp_path: Path, browser: Browser) -> None:
    (tmp_path / "schemes").mkdir()
    scheme = mixed_scheme().model_copy(update={"scheme_id": "construction"})
    (tmp_path / "schemes" / "construction.json").write_text(scheme.model_dump_json())
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        page.locator("#scheme-list").select_option("construction")
        with page.expect_response("**/api/input-preview") as response:
            page.locator("#open-scheme").click()
        assert response.value.status == 400
        page.wait_for_function("document.body.textContent.includes('這一版輸入頁還不能改構造牆')")
        assert "地板" in page.locator("#open-note").inner_text()
        assert watched.page_errors == []
