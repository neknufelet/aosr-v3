"""實測表的輸出界線、能力、存讀指紋與網頁伺服器；不開瀏覽器。"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Literal, cast

import pytest
from pydantic import ValidationError

from aosr.geometry.shoebox import Wall
from aosr.materials.construction_models import ConstructionNote
from aosr.physics import report_io, report_output
from aosr.physics.report_measured import MeasuredSeam, SeamValues
from aosr.physics.wall_surfaces import extrapolated_frequency_text
from aosr.reporting.modal_lookup import key_from_scheme, scheme_scope_reason
from aosr.reporting.scheme import Scheme
from tests.engine._directivity import DIRECTIVITY
from tests.engine._furniture_cases import CAPABILITIES, pair
from tests.engine._gui_cache import gui_startup_identity_memo as gui_startup_identity_memo
from tests.engine._surface_zone_cases import NEW, zone_document
from tests.engine.test_measured_impedance import measured
from tests.engine.test_report_output_contract import _output


def document() -> dict[str, object]:
    data = zone_document(mixed=True)
    scene = cast(dict[str, object], data['scene'])
    zones = cast(dict[str, list[dict[str, object]]], scene[NEW])
    zones['floor'][0]['material'] = measured()
    return data


def test_measured_seams_output_is_optional_closed_and_has_facts() -> None:
    assert 'measured_seams_by_wall' not in _output().model_dump(mode='json')
    values = SeamValues(real_pa_s_m=400., imag_pa_s_m=-200., normal_absorption=.5, random_absorption=.4)
    seams = (MeasuredSeam(frequency_hz=100., measured=values, base_assembly=values),
             MeasuredSeam(frequency_hz=1000., measured=values, base_assembly=values))
    output = _output(measured_seams_by_wall={'floor': seams})
    assert type(output).model_validate_json(output.model_dump_json()) == output
    facts = type(output).model_fields['measured_seams_by_wall'].json_schema_extra
    assert isinstance(facts, dict) and '沒有基準' in str(facts['reference'])
    assert {'quantity', 'unit', 'reference', 'validity'} <= facts.keys()
    names = cast(dict[str, list[str]], facts['propertyNames'])
    assert set(names['enum']) == set(Wall.wall_names())
    for bad in ({}, {'unknown': seams}, {'floor': tuple(reversed(seams))}, {'floor': seams[:1]}):
        with pytest.raises(ValidationError):
            _output(measured_seams_by_wall=bad)
    assert 'difference' not in output.model_dump_json()


@pytest.mark.parametrize('code', ('komatsu_frequency_extrapolation', 'komatsu_material_class_outside_fit'))
def test_komatsu_notes_still_require_layer_and_material_class(
    code: Literal["komatsu_frequency_extrapolation", "komatsu_material_class_outside_fit"],
) -> None:
    with pytest.raises(ValidationError):
        ConstructionNote(code=code, message='外推')
    valid = ConstructionNote(code=code, layer_index=1, material_class='fibrous', message='外推')
    assert valid.model_dump_json() == ('{"code":"' + code
        + '","layer_index":1,"frequencies_hz":[],"material_class":"fibrous","message":"外推"}')
    assert valid.model_dump(mode='json') == {'code': code, 'layer_index': 1,
        'frequencies_hz': [], 'material_class': 'fibrous', 'message': '外推'}


@pytest.mark.parametrize('extra', ({'layer_index': 1}, {'material_class': 'fibrous'}, {'layer_index': None}))
def test_fill_note_cannot_claim_a_layer(extra: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ConstructionNote.model_validate({'code': 'measured_table_model_fill', 'message': '模型補值', **extra})


def test_fill_spans_split_at_table_bounds_instead_of_komatsu_limit() -> None:
    assert extrapolated_frequency_text((200., 300., 3000., 4000.), table_bounds_hz=(400., 2000.)) == '200–300 Hz、3000–4000 Hz'


def test_measured_normalization_fingerprint_capability_and_modal_scope() -> None:
    original = Scheme.model_validate(document())
    for mode in ('python', 'json'):
        assert Scheme.model_validate(original.model_dump(mode=mode)) == original
    inputs = report_io.load_input_document(pair(original), CAPABILITIES, DIRECTIVITY)
    solved = report_io.solver_inputs(inputs)
    assert solved.impedance_by_wall is None
    assert report_output.report_capability(CAPABILITIES, solved.materials_by_wall).materials == 'measured_impedance_by_wall'
    assert scheme_scope_reason(original) == 'construction_walls'
    assert key_from_scheme(original) is None
    fingerprint = report_io.scene_fingerprint(inputs)
    for field in ('source_and_conditions', 'points', 'base_assembly'):
        changed = deepcopy(document())
        scene = cast(dict[str, object], changed['scene'])
        zones = cast(dict[str, list[dict[str, object]]], scene[NEW])
        material = cast(dict[str, object], zones['floor'][0]['material'])
        if field == 'source_and_conditions':
            material[field] = '另一支管'
        elif field == 'points':
            cast(list[dict[str, object]], material[field])[0]['real_pa_s_m'] = 500.
        else:
            base = cast(dict[str, object], material[field])
            cast(list[dict[str, object]], base['layers'])[1]['thickness_m'] = .04
        changed_input = report_io.load_input_document(pair(Scheme.model_validate(changed)), CAPABILITIES, DIRECTIVITY)
        assert report_io.scene_fingerprint(changed_input) != fingerprint
    assert 'fill_rule' in original.model_dump_json()


def test_measured_web_input_is_plain_400_and_validation_save_work(tmp_path: Path) -> None:
    import asyncio
    asyncio.run(_web_requests(tmp_path))


async def _web_requests(tmp_path: Path) -> None:
    from httpx import ASGITransport, AsyncClient
    from aosr.gui.app import GuiSettings, create_app
    app = create_app(GuiSettings(engine_commit='a' * 40, data_dir=tmp_path))
    async with AsyncClient(transport=ASGITransport(app=app), base_url='http://localhost') as client:
        data = document()
        validation = await client.post('/api/validate', json=data)
        assert validation.status_code == 200, validation.text
        assert not validation.json()['problems']
        assert 'floor' not in validation.json()['impedance_multiples']
        saved = await client.put('/api/schemes/measured', json=data | {'scheme_id': 'measured'})
        assert saved.status_code == 200, saved.text
        for route, body in (('/api/input-preview', data),
            ('/api/input-edit', {'scheme': data, 'action': 'shortcut', 'value': 'listening'})):
            response = await client.post(route, json=body)
            assert response.status_code == 400, response.text
            assert '這一版網頁還不能編輯實測阻抗表的牆' in response.json()['error']
            assert '地板' in response.json()['error']


def test_measured_scheme_cli_saved_result_displays_model_fill(tmp_path: Path) -> None:
    import asyncio
    import json
    import subprocess
    import sys
    import time
    from aosr.config.paths import config_path
    from aosr.reporting.result import SchemeResult
    from governance.exit_codes import note
    from tests.engine.test_construction_cli_end_to_end import environment
    data = document()
    scene = cast(dict[str, object], data['scene'])
    scene['room_m'] = {'Lx': 1.6, 'Ly': 1.2, 'Lz': 1.0}
    zones = cast(dict[str, list[dict[str, object]]], scene[NEW])
    material = cast(dict[str, object], zones['floor'][0]['material'])
    material['base_assembly'] = {'kind': 'assembly', 'backing': 'rigid', 'layers': [{'kind': 'porous',
        'thickness_m': .05, 'flow_resistivity_pa_s_m2': 60000., 'material_class': 'foam'}]}
    data['speakers'] = {'left': {'x': .3, 'y': .3, 'z': .4}, 'right': {'x': .3, 'y': .9, 'z': .4}}
    data['receiver_set'] = {'points': [
        {'receiver_id': 'main', 'role': 'primary', 'position_m': [1.2, .6, .6], 'importance': 1.},
        {'receiver_id': 'side', 'role': 'surrounding', 'position_m': [1.2, .7, .6],
         'importance': 1., 'direction_relative_to_primary': 'left'}]}
    data['source_model'] = 'omnidirectional'
    path = tmp_path / 'scheme.json'
    path.write_text(json.dumps(data))
    results = tmp_path / 'results'
    results.mkdir()
    identifier = 'a' * 32
    out = results / f'{identifier}.json'
    start = time.perf_counter()
    child = subprocess.run([sys.executable, '-B', '-m', 'aosr.reporting.scheme_cli', 'run', str(path),
        '--out', str(out), '--capabilities', str(config_path('capabilities.toml')), '--engine-commit', 'a' * 40],
        cwd=tmp_path, env=environment(tmp_path), capture_output=True, text=True, timeout=120)
    assert child.returncode == 0, child.stdout + child.stderr
    saved = SchemeResult.model_validate_json(out.read_bytes())
    assert all(p.report.measured_seams_by_wall for p in saved.pairs)
    assert all(p.report.construction_notes_by_wall for p in saved.pairs)
    assert type(saved).model_validate_json(saved.model_dump_json()) == saved
    note(f'實測方案命令列含四對真有限元素、保存與重讀牆鐘：{time.perf_counter() - start:.3f} 秒')
    probe = Path(__file__).with_name('_measured_gui_probe.py')
    response = subprocess.run([sys.executable, '-B', str(probe), str(tmp_path), identifier],
        cwd=tmp_path, env=environment(tmp_path), capture_output=True, text=True, timeout=60)
    assert response.returncode == 0, response.stdout + response.stderr


def test_seam_text_computes_base_minus_measured_only_for_display(monkeypatch: pytest.MonkeyPatch) -> None:
    from aosr.physics import report_measured_text
    inputs = report_io.load_input_document(pair(Scheme.model_validate(document())), CAPABILITIES, DIRECTIVITY)
    left = SeamValues(real_pa_s_m=400., imag_pa_s_m=-200., normal_absorption=.5, random_absorption=.4)
    right = SeamValues(real_pa_s_m=500., imag_pa_s_m=-250., normal_absorption=.6, random_absorption=.3)
    row = MeasuredSeam(frequency_hz=100., measured=left, base_assembly=right)
    monkeypatch.setattr(report_measured_text, 'measured_seams', lambda *args, **kwargs: {'floor': (row,)})
    text = '\n'.join(report_measured_text.measured_text(inputs))
    assert '100｜阻抗實部 Pa·s/m｜400｜500｜100' in text
    assert '100｜阻抗虛部 Pa·s/m｜-200｜-250｜-50' in text
    assert '100｜無規入射吸音率｜0.4｜0.3｜-0.1' in text


def test_measured_fields_do_not_rename_existing_seat_path_words() -> None:
    from aosr.gui.problem_text import field_name
    assert field_name('unrecognized.points', {}) == 'unrecognized › 座位'
    label = field_name(f'scene.{NEW}.floor.0.material.measured_impedance.points.0.imag_pa_s_m', {})
    assert label is not None and '阻抗虛部' in label and 'e^{+jωt}' in label
