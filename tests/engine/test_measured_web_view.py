"""保存的接縫才是網頁答案；不在顯示時重算，也不盲取第一對。"""
from copy import deepcopy
import json
import re
from typing import cast

import pytest

from aosr.gui.compare_view import scheme_differences
from aosr.gui.modal_view import build_modal_view
from aosr.materials.construction_models import ConstructionNote
from aosr.physics.report_measured import MeasuredSeam, SeamValues
from aosr.reporting.modal_diagnosis_model import ModalDiagnosis, ModalDiagnosisState
from aosr.reporting.result import PairResult, SchemeResult
from aosr.reporting.scheme import Scheme
from tests.engine.test_gui_problem_text import ENGLISH
from tests.engine.test_measured_reporting import document
from tests.engine.test_measured_web_input import floor
from tests.engine.test_report_output_contract import _output


def saved_result() -> SchemeResult:
    measured = SeamValues(real_pa_s_m=400., imag_pa_s_m=-200., normal_absorption=.5, random_absorption=.4)
    base = SeamValues(real_pa_s_m=450., imag_pa_s_m=-180., normal_absorption=.7, random_absorption=.3)
    seams = tuple(MeasuredSeam(frequency_hz=f, measured=measured, base_assembly=base) for f in (100., 1000.))
    notes = (ConstructionNote(code='measured_table_model_fill', message='模型補值', frequencies_hz=(20., 2000.)),
        ConstructionNote(code='komatsu_frequency_extrapolation', layer_index=1, material_class='fibrous',
                         message='外推', frequencies_hz=(20.,)))
    output = _output(measured_seams_by_wall={'floor': seams}, construction_notes_by_wall={'floor': notes})
    pairs = tuple(PairResult.model_construct(report=output) for _ in ('left', 'right'))
    return SchemeResult.model_construct(scheme=Scheme.model_validate(document()), pairs=pairs)


def test_saved_header_seams_differences_and_chinese() -> None:
    from aosr.gui.measured_view import build_measured_view
    views = build_measured_view(saved_result())
    assert views is not None
    view = views[0]
    text = '\n'.join(view.header_lines)
    for phrase in ('地板', '涵蓋 100–1000 Hz', '100 Hz 以下與 1000 Hz 以上是模型補值（底構造）',
                   '實測段假設與角度無關', '樣品在管子裡量到的結果，不代表現場安裝', '材料模型外推'):
        assert phrase in text
    assert {row[0] for row in view.rows} == {'100 Hz', '1000 Hz'}
    assert ('100 Hz', '阻抗實部（Pa·s/m）', '400', '450', '50') in view.rows
    assert ('100 Hz', '阻抗虛部（Pa·s/m）', '-200', '-180', '20') in view.rows
    assert ('100 Hz', '無規入射吸音率', '0.4', '0.3', '-0.1') in view.rows
    assert '差＝底構造減實測' in view.seam_heading
    assert '無規吸音率本來就會跳一下' in view.seam_note
    shown = json.dumps([v.model_dump(mode='json') for v in views], ensure_ascii=False)
    visible = '\n'.join([*view.header_lines, view.seam_heading, view.seam_note, *(s for row in view.rows for s in row)])
    assert not ENGLISH.search(visible) and not re.search(r'\d+\.\d{13,}', shown)


def test_measured_saved_class_note_is_chinese_without_changing_construction_text() -> None:
    from aosr.gui.result_view import _construction_notes
    result = saved_result()
    message = '底構造第 1 層：泡棉材料類別不在 Komatsu（小松）纖維模型擬合範圍，照算但不宣稱準確性'
    note = ConstructionNote(code='komatsu_material_class_outside_fit', layer_index=1,
                            material_class='foam', message=message)
    pairs = tuple(pair.model_copy(update={'report': pair.report.model_copy(update={
        'construction_notes_by_wall': {'floor': (note,)}})}) for pair in result.pairs)
    lines = _construction_notes(result.model_copy(update={'pairs': pairs}))
    assert lines and all('小松纖維模型' in line and not ENGLISH.search(line) for line in lines)
    assert all('照算但不宣稱準確性' in line for line in lines)
    data = document()
    base = deepcopy(floor(data)['base_assembly'])
    floor(data).clear()
    floor(data).update(cast(dict[str, object], base))
    construction = result.model_copy(update={'scheme': Scheme.model_validate(data), 'pairs': pairs})
    assert _construction_notes(construction) == (f'地板第 1 層：{message}',)


def test_header_unions_saved_extrapolation_frequencies_across_pairs() -> None:
    from aosr.gui.measured_view import build_measured_view
    result = saved_result()
    pair = result.pairs[-1]
    notes = pair.report.construction_notes_by_wall
    assert notes is not None
    changed = notes['floor'][-1].model_copy(update={'frequencies_hz': (40.,)})
    pair = pair.model_copy(update={'report': pair.report.model_copy(update={
        'construction_notes_by_wall': {'floor': (notes['floor'][0], changed)}})})
    views = build_measured_view(result.model_copy(update={'pairs': (*result.pairs[:-1], pair)}))
    assert views is not None
    lines = tuple(line for line in views[0].header_lines if '材料模型外推' in line)
    assert lines == ('底構造第 1 層另有材料模型外推（小松纖維模型）：20–40 Hz',)


@pytest.mark.parametrize(('case', 'message'), [('different', '各位置對保存的接縫不一致'), ('missing', '接縫牆面與方案'),
                                             ('wrong-wall', '接縫牆面與方案'), ('wrong-endpoint', '接縫端點與實測表不一致')],
                         ids=['different', 'missing', 'wrong-wall', 'wrong-endpoint'])
def test_all_pairs_and_walls_must_agree_before_showing_seams(case: str, message: str) -> None:
    from aosr.gui.measured_view import build_measured_view
    result = saved_result()
    report = deepcopy(result.pairs[-1].report)
    seams = report.measured_seams_by_wall
    assert seams is not None
    if case == 'missing':
        report = report.model_copy(update={'measured_seams_by_wall': None})
    elif case == 'wrong-wall':
        report = report.model_copy(update={'measured_seams_by_wall': {'x0': seams['floor']}})
    else:
        first = seams['floor'][0]
        changed = first.model_copy(update={'frequency_hz': 101.}) if case == 'wrong-endpoint' else first.model_copy(
            update={'base_assembly': first.base_assembly.model_copy(update={'real_pa_s_m': 999.})})
        report = report.model_copy(update={'measured_seams_by_wall': {'floor': (changed, seams['floor'][-1])}})
    pairs = (*result.pairs[:-1], result.pairs[-1].model_copy(update={'report': report}))
    if case == 'wrong-endpoint':  # 每一對都存錯，各位置對一致也擋得住
        pairs = tuple(pair.model_copy(update={'report': report}) for pair in result.pairs)
    altered = result.model_copy(update={'pairs': pairs})
    with pytest.raises(ValueError, match=message):
        build_measured_view(altered)


def test_non_measured_result_has_no_new_block() -> None:
    from aosr.gui.measured_view import build_measured_view
    from aosr.gui.result_view import ResultView
    result = saved_result()
    data = document()
    floor(data).clear()
    floor(data).update({'kind': 'constant_impedance', 'impedance_pa_s_m': 1600.})
    result = result.model_copy(update={'scheme': Scheme.model_validate(data)})
    assert build_measured_view(result) is None
    assert 'measured_walls' not in ResultView.model_construct().model_dump()


def test_modal_reason_names_measured_table_without_new_reason_code() -> None:
    diagnosis = ModalDiagnosis(state=ModalDiagnosisState.OUT_OF_SCOPE, reason_code='construction_walls')
    text = build_modal_view(diagnosis, Scheme.model_validate(document()))['reason_text']
    assert '實測阻抗表' in text and '六面固定阻抗' in text
    assert diagnosis.reason_code == 'construction_walls'


def test_compare_details_match_frequency_so_insertion_does_not_shift_later_rows() -> None:
    a, b = document(), document()
    points = cast(list[dict[str, object]], floor(b)['points'])
    points.insert(1, {'frequency_hz': 300., 'real_pa_s_m': 500., 'imag_pa_s_m': -30.})
    changes = scheme_differences(Scheme.model_validate(a), Scheme.model_validate(b))
    assert {c.path for c in changes} == {'scene.surface_zones_by_wall.floor'}
    assert changes[0].details == (('300 Hz', '未設定', '阻抗實部 500、阻抗虛部 -30 Pa·s/m'),)
    points[-1]['imag_pa_s_m'] = 70.
    changes = scheme_differences(Scheme.model_validate(a), Scheme.model_validate(b))
    assert {r[0] for r in changes[0].details} == {'300 Hz', '1000 Hz'}


def test_compare_details_detect_a_change_hidden_by_display_rounding() -> None:
    a, b = document(), document()
    points = cast(list[dict[str, float]], floor(b)['points'])
    points[0]['real_pa_s_m'] += .0000001
    changes = scheme_differences(Scheme.model_validate(a), Scheme.model_validate(b))
    assert {c.path for c in changes} == {'scene.surface_zones_by_wall.floor'}
    assert {row[0] for row in changes[0].details} == {'100 Hz'}
    assert all(row[1] != row[2] for row in changes[0].details)


def test_compare_details_name_a_base_assembly_change_on_the_same_wall() -> None:
    a, b = document(), document()
    layers = cast(list[dict[str, object]], cast(dict[str, object], floor(b)['base_assembly'])['layers'])
    layers[1]['thickness_m'] = .05
    changes = scheme_differences(Scheme.model_validate(a), Scheme.model_validate(b))
    assert {c.path for c in changes} == {'scene.surface_zones_by_wall.floor'}
    rows = {row[0]: row for row in changes[0].details}
    assert rows.keys() == {'同一面牆的底構造'}
    assert rows['同一面牆的底構造'][1] != rows['同一面牆的底構造'][2]


def test_construction_and_fixed_result_json_omit_measured_block(tmp_path_factory: pytest.TempPathFactory, worker_id: str) -> None:
    from aosr.config.paths import config_path
    from aosr.gui.result_view import build_result_view
    from aosr.gui.input_setup import input_defaults
    from tests.engine.test_scheme_pipeline import shared_control_result
    fixed = shared_control_result(tmp_path_factory, worker_id, 'wall-1')
    presets = input_defaults()['construction_presets']
    assert isinstance(presets, list)
    data = fixed.scheme.model_dump(mode='json')
    scene = cast(dict[str, object], data['scene'])
    walls = scene.pop('impedance_pa_s_per_m_by_wall')
    assert isinstance(walls, dict)
    scene['surface_zones_by_wall'] = {wall: [{'area': 'whole_wall', 'material': {
        'kind': 'constant_impedance', 'impedance_pa_s_m': value}}] for wall, value in walls.items()}
    floor(data).clear()
    floor(data).update(deepcopy(presets[0]['material']))
    construction_scheme = Scheme.model_validate(data)
    zones = construction_scheme.scene.surface_zones_by_wall
    assert zones is not None and zones['floor'][0].material.kind == 'assembly'
    construction = fixed.model_copy(update={'scheme': construction_scheme})
    for result in (fixed, construction):
        view = build_result_view(result, quality_targets_path=config_path('quality_targets.toml'))
        assert 'measured_walls' not in view.model_dump(mode='json')
