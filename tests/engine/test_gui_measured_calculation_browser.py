"""實測表由填表、存讀到一份真算：網頁填的值進了計算、結果頁顯示的是這一份（#972 施工二）。"""
import json
from pathlib import Path
import time
from typing import cast

from playwright.sync_api import Browser, Page, expect

from aosr.gui.measured_view import build_measured_view
from aosr.reporting.result import SchemeResult
from aosr.reporting.scheme import Scheme
from governance.exit_codes import note
from tests.engine._construction_gui_server import real_server
from tests.engine.test_gui_browser import LONG_DECIMAL, RAW_VALUES, _open, browser as browser
from tests.engine.test_gui_construction_calculation_browser import calculate, recompute_floor_thickness, small_document
from tests.engine.test_gui_input_setup_browser import _save_readback
from tests.engine.test_gui_measured_input_browser import _filled_draft_preset, apply_draft, start_draft, table_points
from tests.engine.test_measured_web_input import floor


def check_result_page(page: Page) -> None:
    expect(page.locator('#construction-note-lines')).to_contain_text('模型補值')
    expect(page.locator('#measured-walls')).to_contain_text('實測段假設與角度無關')
    expect(page.locator('#measured-walls')).to_contain_text('不代表現場安裝')
    frequencies = page.locator('.measured-seams tr td:first-child').all_text_contents()
    assert set(frequencies) == {'75 Hz', '1500 Hz'}
    text = page.locator('body').inner_text()
    assert not RAW_VALUES.search(text) and not LONG_DECIMAL.search(text)
    expect(page.locator('#rejection')).to_be_hidden()
    check_mobile_seam_scroll(page)


def check_mobile_seam_scroll(page: Page) -> None:
    viewport = page.viewport_size
    assert viewport is not None
    if viewport['width'] > 390:
        return
    heading = page.locator('#measured-walls h2')
    before = heading.bounding_box()
    assert before is not None
    wrapper = page.locator('#measured-walls .table-wrap')
    metrics = cast(dict[str, float], wrapper.evaluate('box => {box.scrollLeft=box.scrollWidth; return {left:box.scrollLeft};}'))
    assert metrics['left'] > 0
    expect(page.locator('.measured-seams th').last).to_be_in_viewport()
    assert heading.bounding_box() == before
    wrapper.evaluate('box => {box.scrollLeft=0;}')


# 網頁填的值（都不是預設）：表的 75 Hz 端點實部（預設 400）與底構造第 1 層厚度（玻璃棉預設 0.05 m）。
TYPED_REAL = 1100.
TYPED_BASE_THICKNESS = .08


def check_typed_values_reached_calculation(saved: dict[str, object], result: SchemeResult) -> None:
    """存檔與結果檔裡記的那幾格等於網頁填的值、不是預設；而且計算真的吃到了表那一格：結果檔保存的表端點接縫值
    （計算時從表端點取的，不在顯示時重算）等於填的那一格。網頁輸入在存檔或交給計算時被丟掉會紅；
    求解器拿到填的厚度卻偷用預設厚度，由 check_solver_used_typed_base_thickness 的對照小重算守。"""
    material = floor(saved)
    points = cast(list[dict[str, float]], material['points'])
    layers = cast(list[dict[str, object]], cast(dict[str, object], material['base_assembly'])['layers'])
    assert points[0]['real_pa_s_m'] == TYPED_REAL != table_points()[0]['real_pa_s_m']
    assert layers[0]['thickness_m'] == TYPED_BASE_THICKNESS
    assert result.scheme.model_dump(mode='json') == Scheme.model_validate(saved).model_dump(mode='json')
    for pair in result.pairs:
        # 子行程實際讀進求解的那一份輸入（結果模型另核它跟方案逐格相同）：表端點與底構造厚度就是填的值。
        zones = cast(dict[str, list[dict[str, dict[str, object]]]], pair.input_document['surface_zones_by_wall'])
        assert zones['floor'][0]['material'] == material
        seams = pair.report.measured_seams_by_wall
        assert seams is not None
        assert seams['floor'][0].frequency_hz == points[0]['frequency_hz']
        assert seams['floor'][0].measured.real_pa_s_m == TYPED_REAL


def check_solver_used_typed_base_thickness(tmp_path: Path, result: SchemeResult, default_thickness: float) -> None:
    """結果檔保存的接縫底構造值（表兩端點的阻抗實部、虛部與無規吸音率，計算時由底構造求的）等於「填的厚度」
    重算、不等於「預設厚度」重算：求解器偷用預設厚度就紅。"""
    assert default_thickness != TYPED_BASE_THICKNESS
    typed, default = recompute_floor_thickness(tmp_path, 'seams', result, (TYPED_BASE_THICKNESS, default_thickness))
    for pair in result.pairs:
        seams = pair.report.measured_seams_by_wall
        assert seams is not None
        saved = [[seam.base_assembly.real_pa_s_m, seam.base_assembly.imag_pa_s_m, seam.base_assembly.random_absorption]
                 for seam in seams['floor']]
        assert saved == typed
        assert saved != default


def check_page_shows_this_result(page: Page, identifier: str, result: SchemeResult) -> None:
    """結果頁顯示的是這一次算出來的那份：方案代號、日期，以及接縫表每一格都等於這份結果檔保存的數字。"""
    assert page.url.endswith(f'/results/{identifier}')
    expect(page.locator('#identity')).to_have_text(f'方案：{result.scheme.scheme_id}；日期：{result.run_date}')
    views = build_measured_view(result)
    assert views is not None
    shown = page.locator('.measured-seams').first.evaluate(
        "table => [...table.querySelectorAll('tr')].filter(row => row.querySelector('td'))"
        ".map(row => [...row.cells].map(cell => cell.textContent))")
    assert shown == [list(row) for row in views[0].rows]


def test_fill_save_reload_one_real_run_and_result_pages(tmp_path: Path, browser: Browser) -> None:
    """#972 施工二：只留一次真算。改表後三路變、表內外與底構造互不干擾交給命令列考卷
    test_measured_cli_end_to_end::test_measured_three_real_cli_runs_isolate_each_frequency；比較頁的表差異列
    交給 test_gui_compare_browser::test_measured_table_changes_render_on_both_sides。這一題守網頁填的值真的進了計算（存檔、求解輸入、
    表端點接縫，以及接縫底構造值等於填的厚度重算、不等於預設厚度重算）、結果頁顯示的是這一次的結果。"""
    (tmp_path / 'schemes').mkdir()
    (tmp_path / 'schemes' / 'small.json').write_text(Scheme.model_validate(small_document()).model_dump_json())
    start = time.perf_counter()
    with real_server(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        page.locator('#scheme-list').select_option('small')
        with page.expect_response('**/api/input-preview'):
            page.locator('#open-scheme').click()
        page.wait_for_function("!document.querySelector('main').inert")
        start_draft(page)
        _filled_draft_preset(page, 'glass_wool_96k_50mm')
        apply_draft(page)
        saved = _save_readback(page, 'measured-a')
        assert floor(saved)['points'] == table_points() and 'fill_rule' in floor(saved)
        base_layers = cast(list[dict[str, object]], cast(dict[str, object], floor(saved)['base_assembly'])['layers'])
        preset_thickness = cast(float, base_layers[0]['thickness_m'])
        # 改兩格後不另外按存檔：「計算」自己先存目前的表單再起算，改了沒進計算就會紅。
        page.locator('#point-floor-0-real_pa_s_m').fill(str(TYPED_REAL))
        page.locator('#layer-floor-0-thickness_m').fill(str(TYPED_BASE_THICKNESS))
        identifier, result = calculate(page, tmp_path, start + 600)
        assert page.locator('#result-link').get_attribute('href') == f'/results/{identifier}'
        on_disk = json.loads((tmp_path / 'schemes' / 'measured-a.json').read_text())
        check_typed_values_reached_calculation(on_disk, result)
        check_solver_used_typed_base_thickness(tmp_path, result, preset_thickness)
        for width in (1400, 390):
            with _open(browser, f'{base}/results/{identifier}', viewport_width=width) as shown:
                check_result_page(shown.page)
                check_page_shows_this_result(shown.page, identifier, result)
                shown.page.locator('#measured-walls').screenshot(path=tmp_path / f'result-{identifier}-{width}.png')
                assert shown.page_errors == []
        assert watched.page_errors == []
    note(f'實測填表、存讀、一份真算與結果頁整題牆鐘：{time.perf_counter() - start:.3f} 秒')
