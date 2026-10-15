"""實測表網頁編輯、底構造範圍、原值與主線控制組。"""
from copy import deepcopy
import json
from pathlib import Path
from typing import cast

import pytest

from aosr.config.paths import config_path
from aosr.config.porous_model_ranges import load_porous_model_ranges
from aosr.gui.construction_input import material_field_values
from aosr.gui.input_setup import input_defaults, input_preview
from aosr.gui.problem_text import material_form_spec
from aosr.materials.construction_models import MeasuredImpedancePoint
from aosr.reporting.scheme import Scheme
from tests.engine._directivity import DIRECTIVITY
from tests.engine._furniture_cases import CAPABILITIES, reference_document
from tests.engine._gui_cache import gui_startup_identity_memo as gui_startup_identity_memo
from tests.engine.test_construction_presets import preset_document
from tests.engine.test_gui_app import _app
from tests.engine.test_measured_reporting import document


def floor(document: dict[str, object]) -> dict[str, object]:
    scene = cast(dict[str, object], document['scene'])
    zones = cast(dict[str, list[dict[str, object]]], scene['surface_zones_by_wall'])
    return cast(dict[str, object], zones['floor'][0]['material'])


@pytest.mark.parametrize('action,value', [('shortcut', 'listening'), ('setup', True),
    ('speaker_kind', 'bookshelf'), ('representative', True), ('mount', 'stand'),
    ('furniture', 'sofa'), ('remove_furniture', 'sofa')])
def test_measured_preview_and_edits_are_accepted(tmp_path: Path, action: str, value: str | bool) -> None:
    with _app(tmp_path) as client:
        data = document()
        if action in ('speaker_kind', 'representative', 'mount'):
            data = client.post('/api/input-edit', json={'scheme': data, 'action': 'setup', 'value': True}).json()['scheme']
        assert client.post('/api/input-preview', json=data).status_code == 200
        response = client.post('/api/input-edit', json={'scheme': data, 'action': action, 'value': value})
        assert response.status_code == 200, response.text
        assert floor(response.json()['scheme']) == floor(data)


def test_preset_changes_only_base_and_preserves_table_source_rule(tmp_path: Path) -> None:
    with _app(tmp_path) as client:
        data = Scheme.model_validate(document()).model_dump(mode='json')
        preset = client.get('/api/input-defaults').json()['construction_presets'][1]
        response = client.post('/api/input-edit', json={'scheme': data, 'action': 'wall_preset',
            'wall': 'floor', 'value': preset['id']})
    assert response.status_code == 200, response.text
    expected = deepcopy(data)
    floor(expected)['base_assembly'] = preset['material']
    assert response.json()['scheme'] == expected


@pytest.mark.parametrize('edge', ['low', 'high'])
def test_base_flow_range_is_refused_on_all_three_routes(tmp_path: Path, edge: str) -> None:
    data = document() | {'scheme_id': 'range'}
    ranges = load_porous_model_ranges(config_path('porous_model_ranges.toml')).komatsu
    sigma = ranges.sample_flow_resistivity_min_pa_s_m2 / 2 if edge == 'low' else ranges.sample_flow_resistivity_max_pa_s_m2 * 2
    base = input_defaults()['construction_presets']
    assert isinstance(base, list)
    floor(data)['base_assembly'] = deepcopy(base[0]['material'])
    assembly = cast(dict[str, object], floor(data)['base_assembly'])
    cast(list[dict[str, object]], assembly['layers'])[0]['flow_resistivity_pa_s_m2'] = sigma
    with _app(tmp_path) as client:
        preview = client.post('/api/input-preview', json=data)
        validation = client.post('/api/validate', json=data)
        saved = client.put('/api/schemes/range', json=data)
    assert preview.status_code == 400 and validation.status_code == 200 and saved.status_code == 422
    texts = [preview.json()['error'], *(r['text'] for reply in (validation, saved) for r in reply.json()['problems'])]
    assert texts and all('地板材料構造 › 同一面牆的底構造 › 第 1 層 › 流阻' in t for t in texts)
    assert all('小松纖維模型' in t and '可用範圍' in t for t in texts)
    assert not (tmp_path / 'schemes' / 'range.json').exists()


def test_row_specs_and_original_values_come_from_models() -> None:
    data = document()
    fields = material_field_values(data)
    for index, point in enumerate(cast(list[dict[str, object]], floor(data)['points'])):
        for name, value in point.items():
            assert fields[f'point-floor-{index}-{name}'] == value
    specs = material_form_spec()['measured_impedance']['point_fields']
    assert isinstance(specs, dict)
    for name, field in MeasuredImpedancePoint.model_fields.items():
        assert field.description is not None
        assert specs[name]['description'] == field.description.split('（', 1)[-1].removesuffix('）')
    assert 'e^{+jωt}；e^{−iωt}' in specs['imag_pa_s_m']['description']


def test_mainline_existing_forms_and_preview_values_are_unchanged() -> None:
    """答案是施工前 3cc9e0c1 實跑 JSON，逐值對照，不從當下實作造答案。"""
    control = json.loads((Path(__file__).resolve().parents[2] / 'blueprint' / 'measured_web_control.json').read_text())
    forms = material_form_spec()
    assert {kind: forms[kind] for kind in control['forms']} == control['forms']
    presets = input_defaults()['construction_presets']
    assert isinstance(presets, list)
    documents = {'fixed': reference_document(), 'assembly': preset_document(presets[0]['material'])}
    for name, data in documents.items():
        assert input_preview(data, CAPABILITIES, DIRECTIVITY) == control['previews'][name]


def test_measured_editor_asset_is_served(tmp_path: Path) -> None:
    with _app(tmp_path) as client:
        asset = client.get(f"/static/{'measured.js'}")
        assert asset.status_code == 200
        assert 'parsePastedTable' in asset.text and 'collectMeasured' in asset.text
        assert 'measured.js' in client.get('/').text


def test_fixed_impedance_error_points_at_both_supported_frequency_dependent_forms() -> None:
    from aosr.gui.problem_text import plain_problems
    from aosr.reporting.validation import SchemeProblem
    text = plain_problems((SchemeProblem('scene.impedance_pa_s_per_m_by_wall.floor',
        'impedance_pa_s_per_m_by_wall.floor：不收這一種格式'),), {})[0]['text']
    assert '固定阻抗欄只收正實數；隨頻率變的牆請填構造或實測阻抗表' in str(text)
