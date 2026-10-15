"""實測草稿、貼表、逐列編輯與拒收；桌面與手機都走真網頁端點。"""
from pathlib import Path
from typing import cast

import pytest
from playwright.sync_api import Browser, Page, expect

from aosr.config.paths import config_path
from aosr.config.porous_model_ranges import load_porous_model_ranges
from tests.engine._gui_cache import gui_startup_identity_memo as gui_startup_identity_memo
from tests.engine.test_gui_browser import _open, _serve, browser as browser
from tests.engine.test_gui_input_setup_browser import _filled, _save_readback
from tests.engine.test_measured_cli_end_to_end import variants
from tests.engine.test_measured_web_input import floor
from tests.engine.test_measured_reporting import document


def table_points() -> list[dict[str, float]]:
    zones = cast(dict[str, list[dict[str, object]]], variants()[0]['surface_zones_by_wall'])
    material = cast(dict[str, object], zones['x0'][0]['material'])
    return cast(list[dict[str, float]], material['points'])


def pasted(separator: str = '\t', header: bool = False) -> str:
    text = '\n'.join(separator.join(str(value) for value in point.values()) for point in table_points())
    return ('頻率 實部 虛部\n' if header else '') + text


def start_draft(page: Page) -> None:
    original = page.evaluate('collect()')
    page.locator('#material-kind-floor').select_option('measured_impedance')
    expect(page.locator('#measured-draft-floor')).to_be_visible()
    assert page.evaluate('collect()') == original
    page.locator('#measured-paste-floor').fill(pasted())
    page.locator('#measured-source-floor').fill('阻抗管，垂直入射，樣品厚度 50 mm')


def apply_draft(page: Page, status: int = 200) -> None:
    with page.expect_response('**/api/input-preview') as reply:
        page.locator('#measured-apply-floor').click()
    assert reply.value.status == status, reply.value.text()
    page.wait_for_function("!document.querySelector('main').inert")


@pytest.mark.parametrize('width', (1400, 390))
def test_draft_edit_roundtrip_switch_and_paste_formats(tmp_path: Path, browser: Browser, width: int) -> None:
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=width) as watched:
        page = watched.page
        start_draft(page)
        expect(page.locator('#preset-floor')).to_have_value('')
        _filled_draft_preset(page, 'glass_wool_96k_50mm')
        for separator, header in [('\t', False), (',', False), (' ', True)]:
            page.locator('#measured-paste-floor').fill(pasted(separator, header))
            parsed = page.evaluate('text => parsePastedTable(text)', pasted(separator, header))
            assert parsed == table_points()
        apply_draft(page)
        expect(page.locator('#wall-editor-floor')).to_contain_text('e^{+jωt}；e^{−iωt} 的表須將虛部變號')
        assert floor(page.evaluate('collect()'))['points'] == table_points()
        page.locator('#point-floor-1-real_pa_s_m').fill('1100')
        page.locator('#measured-add-floor').click()
        added = page.locator('[data-measured-row]').last
        for name, value in [('frequency_hz', '2000'), ('real_pa_s_m', '900'), ('imag_pa_s_m', '10')]:
            added.locator(f'input[id$="-{name}"]').fill(value)
        added.get_by_role('button', name='刪除這列').click()
        _filled(page, '#preset-floor', 'glass_wool_48k_50mm_gap50')
        page.locator('#layer-floor-1-thickness_m').fill('0.075')
        expect(page.locator('#preset-state-floor')).to_contain_text('自訂構造')
        saved = _save_readback(page, 'measured-saved')
        material = floor(saved)
        points = cast(list[dict[str, object]], material['points'])
        assert points[1]['real_pa_s_m'] == 1100 and 'fill_rule' in material
        base_material = material['base_assembly']
        page.locator('#material-kind-floor').select_option('assembly')
        page.wait_for_function("!document.querySelector('main').inert")
        assert floor(page.evaluate('collect()')) == base_material
        start_draft(page)
        assert page.evaluate('measuredDrafts.floor.base_assembly') == base_material
        apply_draft(page)
        assert floor(page.evaluate('collect()'))['base_assembly'] == base_material
        for code in ('measured_impedance', 'base_assembly', 'fill_rule'):
            assert code not in page.locator('#walls').inner_text()
        page.screenshot(path=tmp_path / f'measured-input-{width}.png', full_page=True)
        assert watched.page_errors == []


def _filled_draft_preset(page: Page, preset: str) -> None:
    page.locator('#preset-floor').select_option(preset)
    expect(page.locator('#preset-state-floor')).to_contain_text('使用預設')


@pytest.mark.parametrize('width', (1400, 390))
@pytest.mark.parametrize('case', ['paste', 'ascending', 'real', 'source', 'base', 'range'])
def test_refusal_keeps_draft_and_original_scheme(tmp_path: Path, browser: Browser, case: str, width: int) -> None:
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=width) as watched:
        page = watched.page
        start_draft(page)
        original = page.evaluate('collect()')
        if case != 'base':
            _filled_draft_preset(page, 'glass_wool_96k_50mm')
        if case == 'paste':
            page.locator('#measured-paste-floor').fill('頻率 實部 虛部\n75 400 -200\n300 壞了 -50')
        elif case == 'ascending':
            page.locator('#measured-paste-floor').fill('75 400 -200\n74 600 -50\n1500 800 100')
        elif case == 'real':
            page.locator('#measured-paste-floor').fill('75 0 -200\n300 600 -50\n1500 800 100')
        elif case == 'source':
            page.locator('#measured-source-floor').fill(' ')
        elif case == 'range':
            ranges = load_porous_model_ranges(config_path('porous_model_ranges.toml')).komatsu
            page.locator('#layer-floor-0-flow_resistivity_pa_s_m2').fill(str(ranges.sample_flow_resistivity_max_pa_s_m2 * 2))
        if case in ('paste', 'base'):
            page.locator('#measured-apply-floor').click()
        else:
            apply_draft(page, 400)
        reason = {'paste': '第 3 行', 'ascending': '實測表第 2 列', 'real': '阻抗實部',
                  'source': '量測出處與條件', 'base': '請選底構造', 'range': '流阻'}[case]
        expect(page.locator('#measured-error-floor')).to_contain_text(reason)
        assert page.evaluate('collect()') == original
        expect(page.locator('#measured-draft-floor')).to_be_visible()
        assert watched.page_errors == []


@pytest.mark.parametrize('width', (1400, 390))
def test_existing_measured_base_range_save_is_plain_422(tmp_path: Path, browser: Browser, width: int) -> None:
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=width) as watched:
        page = watched.page
        start_draft(page)
        _filled_draft_preset(page, 'glass_wool_96k_50mm')
        apply_draft(page)
        ranges = load_porous_model_ranges(config_path('porous_model_ranges.toml')).komatsu
        with page.expect_response('**/api/input-preview') as preview:
            page.locator('#layer-floor-0-flow_resistivity_pa_s_m2').fill(str(ranges.sample_flow_resistivity_max_pa_s_m2 * 2))
        assert preview.value.status == 400
        page.locator('#save-as-id').fill('bad-base')
        with page.expect_response('**/api/schemes/bad-base') as saved:
            page.locator('#save-as').click()
        assert saved.value.status == 422
        expect(page.locator('#save-as-note')).to_contain_text('同一面牆的底構造 › 第 1 層 › 流阻')
        assert watched.page_errors == []


@pytest.mark.parametrize('width', (1400, 390))
def test_raw_point_and_base_values_survive_save_and_parser_errors_keep_line_numbers(
        tmp_path: Path, browser: Browser, width: int) -> None:
    from aosr.reporting.scheme import Scheme
    data = document() | {'scheme_id': 'precision'}
    points = cast(list[dict[str, float]], floor(data)['points'])
    points[0]['real_pa_s_m'] = 400.12345678901234
    assembly = cast(dict[str, object], floor(data)['base_assembly'])
    air = next(layer for layer in cast(list[dict[str, object]], assembly['layers']) if layer['kind'] == 'air')
    air['thickness_m'] = .05123456789012345
    (tmp_path / 'schemes').mkdir()
    (tmp_path / 'schemes' / 'precision.json').write_text(Scheme.model_validate(data).model_dump_json())
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=width) as watched:
        page = watched.page
        page.locator('#scheme-list').select_option('precision')
        with page.expect_response('**/api/input-preview'):
            page.locator('#open-scheme').click()
        page.wait_for_function("!document.querySelector('main').inert")
        page.locator('#measured-source-floor').fill('只改出處；原始數字必須原樣保存')
        saved = _save_readback(page, 'precision-saved')
        assert floor(saved)['points'] == floor(data)['points']
        assert floor(saved)['base_assembly'] == floor(data)['base_assembly']
        for bad in ('300 2', '300 2 3 4', '300 NaN 1', '300 Infinity 1', '', '300 0x10 1'):
            text = f'頻率 實部 虛部\n75 400 -200\n{bad}\n1500 800 100'
            error = page.evaluate('text => {try {parsePastedTable(text); return null;} catch (e) {return e.message;}}', text)
            assert error is not None and '第 3 行' in error
        assert watched.page_errors == []
