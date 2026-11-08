"""六類搜尋接線：版本拒收、逐類身分與不可估不落帳。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

from aosr.gradient import endpoint_run, material_gradient_ledger as ledger, run


def _scores() -> ledger.ScoreConstants:
    return ledger.ScoreConstants.model_validate(dict(component_weights=(.5, .5), worse_references=(1., 1.),
        proxy_weight=.25, proxy_worse_reference=1., proxy_temperature_db=.5,
        category_weights=(1., 1., 1., 1., 1., .25)))


def _row() -> ledger.MaterialGradientRow:
    # 低頻三項 2,4,8 → screening=3、低頻搜尋=5；五類和15、產品低頻7。
    return ledger.MaterialGradientRow.model_validate(dict(step=0, phase='snap', round=None, candidate=None,
        theta_hex=(), value=16.25, screening_total=3., components=(2., 4., 8., 1., 2., 3., 4., 5., 7., 16.25, 16.75),
        gradient_norm=None, step_size=None, seconds=0., stop_reason=None, final_trial_step=None,
        evaluations=None, snap=(), jump=None, decision=None))


def test_six_category_score_identities_reject_missing_category_or_proxy_in_product() -> None:
    constants, row = _scores(), _row()
    ledger.check_scores(constants, [row])
    for changes in ({'value': 15.25}, {'components': (*row.components[:10], 16.25)},
                    {'components': (*row.components[:3], 0., *row.components[4:])}):
        with pytest.raises(ValueError):
            ledger.check_scores(constants, [row.model_copy(update=changes)])
    with pytest.raises(ValueError):
        ledger.check_scores(constants.model_copy(update={'category_weights': (1., 1., 1., 1., 1., 1.)}), [row])


def test_nan_row_and_null_totals_cannot_be_written() -> None:
    row = _row().model_dump()
    for changes in ({'value': float('nan')}, {'components': (*row['components'][:9], None, 16.75)},
                    {'components': (None, *row['components'][1:])}):
        with pytest.raises((ValidationError, ValueError)):
            candidate = ledger.MaterialGradientRow.model_validate({**row, **changes})
            ledger.check_scores(_scores(), [candidate])


def test_gpu_search_explicitly_refuses_until_next_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run, 'check_inputs', lambda project: pytest.fail('GPU 靜默落入搜尋接線'))
    project = run.GradientProject.model_construct(device='gpu')
    with pytest.raises(ValueError, match='六類目標的顯卡單精度與精度尺在下一支'):
        run.load_problem(project)


@pytest.mark.parametrize('version', ('aosr.material_gradient_result.v6', 'aosr.material_gradient_result.v7'))
def test_new_result_endpoint_explicitly_refuses_until_later_branch(tmp_path: Path, version: str) -> None:
    (tmp_path / run.RESULT_FILE).write_text(json.dumps({'format_version': version}))
    with pytest.raises(ValueError, match='終點三組數在後續支'):
        endpoint_run.read_result(tmp_path)


@pytest.fixture(scope='module')
def measured_nan(tmp_path_factory: pytest.TempPathFactory, worker_id: str) -> tuple[Path, dict[str, object]]:
    import os
    import subprocess
    import sys
    from aosr.runtime import child_process_env, configure_jax
    from tests.engine._scheme_cache import shared_json

    def produce() -> str:
        directory = tmp_path_factory.mktemp('six-search-nan')
        output = directory / 'nan.json'
        env = child_process_env(threads=1).copy()
        configure_jax(platforms='cpu', enable_x64=True, env=env)
        env['PYTHONPATH'] = os.pathsep.join((env['PYTHONPATH'], str(Path(__file__).resolve().parents[2])))
        done = subprocess.run([sys.executable, '-m', 'tests.engine._six_search_nan_probe', str(output)],
            cwd=directory, env=env, capture_output=True, text=True, timeout=180)
        assert done.returncode == 0, done.stdout + done.stderr
        return json.dumps({'directory':str(directory), 'result':json.loads(output.read_text())})
    data = json.loads(shared_json(tmp_path_factory, worker_id, 'six-search-nan', produce))
    return Path(data['directory']), cast(dict[str, object], data['result'])


def test_nan_start_writes_no_step(measured_nan: tuple[Path, dict[str, object]]) -> None:
    row = measured_nan[1]
    assert row['start_steps'] == 0 and row['start_reason'] == '起點不可估'
    assert row['start_rows'] == 0 and row['start_ledger_reason'] == '起點不可估'
    assert row['per_category_start_stopped'] == [True] * 6


def test_nan_descent_trials_are_not_accepted(measured_nan: tuple[Path, dict[str, object]]) -> None:
    row = measured_nan[1]
    assert row['middle_steps'] == 1 and row['middle_reason'] == '步長到下限'
    assert cast(int, row['middle_rejected']) > 0


def test_nan_candidate_is_not_selected_or_written(measured_nan: tuple[Path, dict[str, object]]) -> None:
    row = measured_nan[1]
    assert row['candidate_accepted'] is False and row['candidate_rejected']
    assert set(cast(list[str], row['candidate_rejected'])) == {'起點不可估'}
    assert row['candidate_ledger_finite'] is True and row['candidate_ledger_reasons']


def test_partial_quick_batch_resumes_to_bitwise_same_result(measured_nan: tuple[Path, dict[str, object]]) -> None:
    scenarios = cast(list[dict[str, object]], measured_nan[1]['templates'])
    for scenario in scenarios:
        assert scenario['rows_finite'] is True
        assert scenario['rejected_quick'] == scenario['first_quick']
        assert scenario['rejected_retune'] == scenario['first_retune']
        assert scenario['rejected_quick'] or scenario['rejected_retune']
    assert scenarios[-1]['resumed_bitwise'] is True
    assert scenarios[-1]['resumed_result_bitwise'] is True


def test_weighted_product_does_not_include_proxy(measured_nan: tuple[Path, dict[str, object]]) -> None:
    assert measured_nan[1]['weighted_totals'] == [16.25,16.75]


def test_unestimable_template_cannot_replace_ranked_candidate_with_ghost(
        measured_nan: tuple[Path, dict[str, object]], tmp_path: Path) -> None:
    from aosr.gradient.template_ledger import read_template_ledger
    lines = [json.loads(line) for line in (measured_nan[0] / 'retune.jsonl').read_text().splitlines()]
    original = lines[1]['rejected_candidates'][0]
    lines[1]['rejected_candidates'][0] = 'ghost->none'
    for item in lines[-1]['decision']['outcomes']:
        if item['candidate'] == original:
            item['candidate'] = 'ghost->none'
    source = tmp_path / 'tampered.jsonl'
    source.write_text(''.join(json.dumps(line)+'\n' for line in lines))
    with pytest.raises(ValueError, match='前幾名'):
        read_template_ledger(source)


def _header() -> ledger.MaterialGradientHeader:
    return ledger.MaterialGradientHeader.model_validate(dict(ledger_version=ledger.LEDGER_VERSION,
        input_hash='input', mesh_hash='mesh', physics_identity='physics', surfaces=(('x0', None),),
        variable_names=(), settings_fingerprint='settings', objective_fingerprint='a'*64, kept_digests={'x0':'b'*64},
        score_constants=_scores().model_dump()))


def _snap() -> ledger.MaterialGradientRow:
    choice = ledger.SnapChoice(surface_id='x0', nearest_thickness_m=None, thickness_m=None,
        cavity_m=None, cavity_clipped=False, thickness_retreated=False)
    return _row().model_copy(update={'snap': (choice,)})


def test_v5_is_readable_but_cannot_resume_and_v6_requires_all_scores(tmp_path: Path) -> None:
    path = tmp_path / 'ledger.jsonl'
    ledger.write_ledger(path, _header(), [_snap()])
    header, row = [json.loads(line) for line in path.read_text().splitlines()]
    header['ledger_version'] = 'aosr.material_gradient_ledger.v5'
    del header['score_constants']['category_weights']
    row.update(value=5., components=row['components'][:3])
    path.write_text(json.dumps(header)+'\n'+json.dumps(row)+'\n')
    old_header, rows = ledger.read_ledger(path, legacy=True)
    assert rows[0].value == 5. and rows[0].components == (2., 4., 8.)
    with pytest.raises(ValueError, match='六類'):
        ledger.read_ledger(path)
    with pytest.raises(ValueError):
        ledger.append_rows(path, old_header, [])
    header['ledger_version'] = ledger.LEDGER_VERSION
    header['score_constants'] = _scores().model_dump()
    path.write_text(json.dumps(header)+'\n'+json.dumps(row)+'\n')
    with pytest.raises(ValueError):
        ledger.read_ledger(path)
    del header['score_constants']['category_weights']
    path.write_text(json.dumps(header)+'\n')
    with pytest.raises(ValueError, match='category_weights'):
        ledger.read_ledger(path)



@pytest.mark.parametrize('prefix', ('timbre_balance.', 'listening_area_stability.', 'reflections_and_echo.',
                                  'reverberation.', 'channel_matching.', 'ranking.category_weights'))
def test_settings_fingerprint_includes_six_category_summary(prefix: str) -> None:
    from aosr.config.paths import config_path
    from aosr.config.quality_targets import load_quality_targets, WeightTable, QualityEntry
    from aosr.scoring.gradient_objective_settings import gradient_objective_settings
    purpose = 'dedicated_two_channel_listening_room'
    registry = load_quality_targets(config_path('quality_targets.toml'))
    section = registry.purpose(purpose)
    chosen = next(entry for entry in section.entries if entry.key.startswith(prefix))
    changed: QualityEntry
    if isinstance(chosen, WeightTable):
        changed = chosen.model_copy(update={'item': (chosen.item[0].model_copy(update={'value':chosen.item[0].value+1.}), *chosen.item[1:])})
        field = 'weight'
    else:
        from aosr.config.quality_targets import SettingEntry, TargetEntry
        assert isinstance(chosen, (SettingEntry, TargetEntry))
        value = chosen.value
        changed_value = tuple(x+1 for x in value) if isinstance(value, tuple) else value+1
        changed = chosen.model_copy(update={'value':changed_value})
        field = 'setting' if chosen in section.setting else 'target'
    changed_section = section.model_copy(update={field:tuple(changed if x.key == chosen.key else x for x in getattr(section,field))})
    altered = registry.model_copy(update={'purposes':tuple(changed_section if x.name == purpose else x for x in registry.purposes)})
    assert gradient_objective_settings(registry,purpose).fingerprint != gradient_objective_settings(altered,purpose).fingerprint


def test_endpoint_reader_keeps_non_object_rejection_as_value_error(tmp_path: Path) -> None:
    (tmp_path / run.RESULT_FILE).write_text('[]')
    with pytest.raises(ValueError, match='版本'):
        endpoint_run.read_result(tmp_path)


@pytest.mark.parametrize('book', ('gradient', 'templates'))
def test_pre_proxy_versions_reject_proxy_row_shape(tmp_path: Path, book: str) -> None:
    from tests.engine.test_gradient_objective_jax import _material_ledger, _template_ledger
    from aosr.gradient import template_ledger
    if book == 'gradient':
        path, _ = _material_ledger(tmp_path)
        version = ledger.PRE_PROXY_LEDGER_VERSION
    else:
        path, _ = _template_ledger(tmp_path)
        version = template_ledger.PRE_PROXY_TEMPLATE_LEDGER_VERSION
    header, *rows = [json.loads(line) for line in path.read_text().splitlines()]
    header['ledger_version'] = version
    header.pop('score_constants')
    for row in rows:
        row['components'] = row['components'][:3]
    path.write_text(''.join(json.dumps(item)+'\n' for item in (header,*rows)))
    with pytest.raises(ValueError, match='形狀'):
        if book == "gradient":
            ledger.read_ledger(path, legacy=True)
        else:
            template_ledger.read_template_ledger(path, legacy=True)
    # 舊兩分項的 None 以前是型別錯誤；新 nullable 類別欄不能放寬舊格式。
    for row in rows:
        row['components'] = [None, row['components'][1]]
        row['screening_total'] = None
    path.write_text(''.join(json.dumps(item)+'\n' for item in (header,*rows)))
    with pytest.raises(ValueError, match='不可估'):
        if book == "gradient":
            ledger.read_ledger(path, legacy=True)
        else:
            template_ledger.read_template_ledger(path, legacy=True)
