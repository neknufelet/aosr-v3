"""實測表由填表、存讀到三份真算；隔離表內材料與表外底構造的影響。"""
from pathlib import Path
import time
from typing import cast

from playwright.sync_api import Browser, Page, expect

from aosr.config.frequency_axis import FEM_GEOMETRIC_CROSSOVER_CAP_HZ, geometric_band_frequencies
from aosr.config.three_lane_crossover import CROSSOVER_LOWER_FLOOR_HZ
from aosr.gui.measured_view import measured_materials
from aosr.reporting.result import SchemeResult
from aosr.reporting.scheme import Scheme
from governance.exit_codes import note
from tests.engine._construction_gui_server import real_server
from tests.engine.test_gui_browser import LONG_DECIMAL, RAW_VALUES, _open, browser as browser
from tests.engine.test_gui_construction_calculation_browser import calculate, small_document
from tests.engine.test_gui_input_setup_browser import _save_readback
from tests.engine.test_gui_measured_input_browser import _filled_draft_preset, apply_draft, start_draft, table_points
from tests.engine.test_measured_web_input import floor


def verify_crossover_and_base_isolation(a: SchemeResult, c: SchemeResult) -> None:
    assert a.scheme.scene.room_m == c.scheme.scene.room_m
    left, right = measured_materials(a.scheme)['floor'], measured_materials(c.scheme)['floor']
    assert left.points == right.points and left.source_and_conditions == right.source_and_conditions
    assert left.fill_rule == right.fill_rule and left.base_assembly != right.base_assembly
    for old, changed in zip(a.pairs, c.pairs, strict=True):
        x, y = old.report, changed.report
        material = measured_materials(a.scheme)['floor']
        low, high = material.points[0].frequency_hz, material.points[-1].frequency_hz
        assert all(low <= f <= high for f in (500., 1000.))
        assert x.top.f_s_hz == y.top.f_s_hz
        assert x.top.crossover_lower_hz == y.top.crossover_lower_hz == max(x.top.f_s_hz, CROSSOVER_LOWER_FLOOR_HZ)
        assert x.top.crossover_upper_hz == y.top.crossover_upper_hz == FEM_GEOMETRIC_CROSSOVER_CAP_HZ
        assert x.points is not None and y.points is not None
        inside, outside = [], []
        for p, q in zip(x.points, y.points, strict=True):
            assert p.frequency_hz == q.frequency_hz
            if low <= p.frequency_hz <= high:
                inside.append(p)
                for field in ('fem_energy', 'reflected_energy', 'interference_energy', 'late_energy', 'total_energy'):
                    assert getattr(p, field) == getattr(q, field), (p.frequency_hz, field)
            else:
                outside.append(p.total_energy != q.total_energy)
        assert inside and any(p.fem_energy is not None for p in inside) and any(outside)
        for band, changed_band in zip(x.bands, y.bands, strict=True):
            axis = geometric_band_frequencies(band.center_frequency_hz)
            if low <= min(axis) and max(axis) <= high:
                for field in ('fem_energy', 'reflected_energy', 'interference_energy', 'late_energy'):
                    assert getattr(band, field) == getattr(changed_band, field)
        assert x.measured_seams_by_wall is not None and y.measured_seams_by_wall is not None
        for seam, changed_seam in zip(x.measured_seams_by_wall['floor'], y.measured_seams_by_wall['floor'], strict=True):
            assert seam.measured == changed_seam.measured and seam.base_assembly != changed_seam.base_assembly
        assert x.construction_notes_by_wall is not None and y.construction_notes_by_wall is not None
        fills = [next(n for n in notes['floor'] if n.code == 'measured_table_model_fill')
                 for notes in (x.construction_notes_by_wall, y.construction_notes_by_wall)]
        assert set(fills[0].frequencies_hz) == set(fills[1].frequencies_hz)


def verify_table_lanes_changed(a: SchemeResult, b: SchemeResult) -> None:
    for old, changed in zip(a.pairs, b.pairs, strict=True):
        assert old.report.points is not None and changed.report.points is not None
        seen_fem, seen_bands = [], []
        for p, q in zip(old.report.points, changed.report.points, strict=True):
            if 75. < p.frequency_hz < 1500. and p.fem_energy is not None:
                seen_fem.append(p.frequency_hz)
                assert p.fem_energy != q.fem_energy
        for band, changed_band in zip(old.report.bands, changed.report.bands, strict=True):
            axis = geometric_band_frequencies(band.center_frequency_hz)
            if 75. <= min(axis) and max(axis) <= 1500.:
                seen_bands.append(band.center_frequency_hz)
                for field in ('reflected_energy', 'interference_energy', 'late_energy'):
                    assert getattr(band, field) != getattr(changed_band, field)
        assert seen_fem and seen_bands


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


def test_fill_save_reload_three_real_runs_and_result_pages(tmp_path: Path, browser: Browser) -> None:
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
        aid, a = calculate(page, tmp_path, start + 600)
        page.locator('#point-floor-1-real_pa_s_m').fill('1100')
        _save_readback(page, 'measured-b')
        bid, b = calculate(page, tmp_path, start + 600)
        verify_table_lanes_changed(a, b)
        page.locator('#point-floor-1-real_pa_s_m').fill(str(table_points()[1]['real_pa_s_m']))
        page.locator('#layer-floor-0-thickness_m').fill('0.08')
        _save_readback(page, 'measured-c')
        cid, c = calculate(page, tmp_path, start + 600)
        verify_crossover_and_base_isolation(a, c)
        for identifier in (aid, bid, cid):
            for width in (1400, 390):
                with _open(browser, f'{base}/results/{identifier}', viewport_width=width) as shown:
                    check_result_page(shown.page)
                    shown.page.locator('#measured-walls').screenshot(path=tmp_path / f'result-{identifier}-{width}.png')
                    assert shown.page_errors == []
        with _open(browser, f'{base}/compare/{aid}/{bid}') as compared:
            expect(compared.page.locator('.measured-changes')).to_contain_text(['300 Hz', '300 Hz'])
            assert compared.page_errors == []
        assert watched.page_errors == []
    note(f'實測填表、存讀、三份真算與結果頁整題牆鐘：{time.perf_counter() - start:.3f} 秒')
