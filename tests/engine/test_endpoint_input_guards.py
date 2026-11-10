"""六類入口先核完整頻率軸與方案原牆；拒收前不得求解或開始搜尋。"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import subprocess
import sys
from typing import NoReturn

import numpy as np
import pytest

from aosr.config.capabilities import load_capabilities
from aosr.config.directivity_defaults import load_directivity_defaults
from aosr.config.frequency_axis import FEM_LANE_FREQUENCIES_HZ
from aosr.config.material_gradient import load_material_gradient_settings
from aosr.config.paths import config_path
from aosr.gradient import cli, endpoint_run, endpoint_six, run
from aosr.gradient.boundary import scheme_patch_keys
from aosr.materials.construction_models import Assembly, MeasuredImpedance, MountedOnWall
from aosr.physics import fem_reduced
from aosr.physics.fem_reduced import BuildLedger, ReducedMatrices, ReducedModel
from aosr.reporting import fem_reduced_scheme
from aosr.reporting.scheme import Scheme
from tests.engine._six_category_endpoint_probe import _small_scheme
from tests.engine.test_fem_reduced import _settings
from tests.engine.test_material_gradient_cli import _env, _start_args
from tests.engine.test_material_gradient_mount import _project

PRESETS = 'glass_wool_48k_50mm_gap50,none,glass_wool_48k_50mm_gap50,none,none,none'
REJECTION = '方案原本的牆.*實測原牆的表內／表外吸音分流尚未接入六類終點'


def _model(scheme: Scheme) -> ReducedModel:
    """一階小矩陣，只供入口查身分與軸，不冒充驗過的替身、不跑完整解。"""
    sources = tuple(channel.speaker_id for channel in scheme.channel_group.channels)
    receivers = tuple(point.receiver_id for point in scheme.receiver_set.points)
    keys = scheme_patch_keys(scheme)
    matrices = ReducedMatrices(np.eye(1), np.eye(1), np.zeros((len(keys), 1, 1)),
        np.ones((len(sources), 1), dtype=np.complex128), np.ones((len(receivers), 1)))
    ledger = BuildLedger(0, 0, 1, (), 'synthetic', {}, _settings())
    return ReducedModel(matrices, sources, receivers, FEM_LANE_FREQUENCIES_HZ,
        scheme.scene.density_kg_m3, scheme.scene.sound_speed_m_s, '0' * 64, 'phys-v1:' + '0' * 64,
        '0' * 64, ledger, keys)


def _context(scheme: Scheme, model: ReducedModel) -> endpoint_six.ScoringContext:
    return endpoint_six.scoring_context(scheme, model, _project().purpose,
        capabilities=load_capabilities(config_path('capabilities.toml')),
        directivity=load_directivity_defaults(config_path('directivity_defaults.toml')),
        capabilities_path=config_path('capabilities.toml'), engine_commit='test')


def _unexpected_work(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError('拒收前不准完整求解或進搜尋迴圈')


def _axis_check(changed: bool) -> None:
    """雙精度冷行程的真常數與一階替身；只改計分範圍外的一點。"""
    scheme = _small_scheme()
    model = _model(scheme)
    if changed:
        axis = (*model.frequencies_hz[:-1], model.frequencies_hz[-1] + 1.)
        with pytest.raises(ValueError, match='頻率軸'):
            _context(scheme, replace(model, frequencies_hz=axis))
    else:
        context = _context(scheme, model)
        assert np.array_equal(context.constants.fem_frequencies_hz, model.frequencies_hz)


def _axis_probe(tmp_path: Path, changed: bool) -> None:
    source = f'from tests.engine.test_endpoint_input_guards import _axis_check; _axis_check({changed!r})'
    done = subprocess.run([sys.executable, '-c', source], cwd=tmp_path, env=_env(tmp_path),
        capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stdout + done.stderr


def test_scoring_context_rejects_same_length_axis_with_one_changed_frequency(tmp_path: Path) -> None:
    """抓舊搜尋軸在計分範圍外偏一點，卻被終點當成六類軸配對。"""
    _axis_probe(tmp_path, True)


def test_scoring_context_accepts_matching_frequency_axis(tmp_path: Path) -> None:
    _axis_probe(tmp_path, False)


def _start_inputs(tmp_path: Path, scheme: Scheme) -> list[str]:
    (tmp_path / 'scheme.json').write_text(scheme.model_dump_json())
    (tmp_path / 'settings.toml').write_bytes(config_path('material_gradient.toml').read_bytes())
    return _start_args(tmp_path, tmp_path / 'root', presets=PRESETS)[3:]


@pytest.mark.parametrize('mount', ('rigid', 'on_wall'))
def test_start_refuses_bare_original_wall_before_solving_or_creating_folder(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mount: str) -> None:
    """即使新樣板蓋住實測牆也拒收；檢查在建立根目錄及 project 之前。"""
    args = cli._parser().parse_args([*_start_inputs(tmp_path, _small_scheme(True)), '--mount', mount])
    monkeypatch.setattr(fem_reduced_scheme, 'export_operators', _unexpected_work)
    monkeypatch.setattr(fem_reduced.FullSolver, 'solve', _unexpected_work)
    monkeypatch.setattr(run, '_fresh', _unexpected_work)
    with pytest.raises(ValueError, match=REJECTION):
        cli._start(args)
    assert not (tmp_path / 'root').exists()


def _saved_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scheme = _small_scheme(True)
    model = _model(scheme)
    settings = load_material_gradient_settings(config_path('material_gradient.toml'))
    project = _project().model_copy(update={'capabilities_path': str(config_path('capabilities.toml')),
        'presets': tuple(None if p == 'none' else p for p in PRESETS.split(','))})
    run.write_json_atomic(tmp_path / run.PROJECT_FILE, project)
    # 替身准用與檔案摘要不屬本題；材料解析、入口重建與拒收仍跑真程式。
    monkeypatch.setattr(run, 'check_inputs', lambda *a, **kw: (scheme, model, settings))
    monkeypatch.setattr(endpoint_run, 'check_inputs', lambda *a, **kw: (scheme, model, settings))


def test_resume_refuses_bare_original_wall_before_search(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _saved_inputs(tmp_path, monkeypatch)
    monkeypatch.setattr(run, 'build_constants', _unexpected_work)
    monkeypatch.setattr(run, 'six_category_chain_objective', _unexpected_work)
    monkeypatch.setattr(run, '_resume', _unexpected_work)
    monkeypatch.setattr(fem_reduced.FullSolver, 'solve', _unexpected_work)
    with pytest.raises(ValueError, match=REJECTION):
        run.resume_search(tmp_path, should_stop=None)
    assert not (tmp_path / run.LEDGER_FILE).exists()


def test_endpoint_refuses_bare_original_wall_before_full_solves(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _saved_inputs(tmp_path, monkeypatch)
    # 結果身分與帳本由既有慢考卷核；此題在完整算子與兩套完整解前設哨兵。
    (tmp_path / run.RESULT_FILE).write_text('{}')
    monkeypatch.setattr(endpoint_run, 'read_result', lambda folder: run.GradientResult.model_construct(
        format_version=run.RESULT_VERSION))
    monkeypatch.setattr(endpoint_run, 'export_operators', _unexpected_work)
    monkeypatch.setattr(endpoint_run, 'endpoint_check', _unexpected_work)
    monkeypatch.setattr(endpoint_run, 'full_model_pressures', _unexpected_work)
    monkeypatch.setattr(fem_reduced.FullSolver, 'solve', _unexpected_work)
    before = {path.name for path in tmp_path.iterdir()}
    with pytest.raises(ValueError, match=REJECTION):
        endpoint_run.run_endpoint(tmp_path)
    assert {path.name for path in tmp_path.iterdir()} == before


class _ReachedOperators(Exception):
    """合法材料已通過入口檢查，到達本題不跑的完整算子邊界。"""


def test_start_accepts_original_mounted_on_measured_wall(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """抓把方案原本已貼好的構造，錯當成裸實測保留牆。"""
    scheme = _small_scheme(True)
    zones = scheme.scene.surface_zones_by_wall
    assert zones is not None
    measured = zones['floor'][0].material
    assembly = zones['x0'][0].material
    assert isinstance(measured, MeasuredImpedance) and isinstance(assembly, Assembly)
    mounted = MountedOnWall(layers=assembly.layers, wall=measured)
    changed = {**zones, 'floor': (zones['floor'][0].model_copy(update={'material': mounted}),)}
    scheme = Scheme.model_validate(scheme.model_copy(update={'scene': scheme.scene.model_copy(
        update={'surface_zones_by_wall': changed})}).model_dump())
    def reached(*args: object, **kwargs: object) -> NoReturn:
        raise _ReachedOperators
    monkeypatch.setattr(fem_reduced_scheme, 'export_operators', reached)
    args = cli._parser().parse_args(_start_inputs(tmp_path, scheme))
    with pytest.raises(_ReachedOperators):
        cli._start(args)
