"""六類終點三組數：獨立評分、兩次完整解、版本與舊文字控制。"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import cast


import pytest

from aosr.gradient.endpoint_run import EndpointRecord, read_endpoint, run_endpoint
from aosr.gradient.report import _endpoint_section
from tests.engine._endpoint_v4_snapshot import SNAPSHOT_JSON
from tests.engine._scheme_cache import shared_json
from tests.engine._gradient_probe_env import endpoint_probe_env


@pytest.fixture(scope='module')
def measured(tmp_path_factory: pytest.TempPathFactory, worker_id: str) -> dict[str, object]:
    """沒有錄製答案；逐位控制同進程算，環境須跟不帶 CPU 釘選的 CLI 工作行程一致。"""
    name = 'six-endpoint'
    def produce() -> str:
        directory = tmp_path_factory.mktemp(name)
        command = [sys.executable, '-m', 'tests.engine._six_category_endpoint_probe', str(directory)]
        done = subprocess.run(command, cwd=directory, env=endpoint_probe_env(directory), capture_output=True, text=True, timeout=1800)
        (directory / 'probe.stdout').write_text(done.stdout)
        (directory / 'probe.stderr').write_text(done.stderr)
        assert done.returncode == 0, done.stdout + done.stderr
        return (directory / 'measure.json').read_text()
    return cast(dict[str, object], json.loads(shared_json(tmp_path_factory, worker_id, name, produce)))



def test_search_endpoint_report_three_groups_and_independent_scores(measured: dict[str, object]) -> None:
    """抓完整聲壓偷換替身、產品混 J、20/30 軸交換，以及多解一次完整模型。"""
    assert measured['version'] == 'aosr.material_gradient_endpoint.v5'
    assert measured['solves'] == measured['expected_solves']
    assert measured['jax_bitwise'] == [True] * 4
    assert measured['numpy_bitwise'] == [True] * 2
    assert measured['axes_distinct'] is True
    assert measured['diagnostics'] is True
    assert measured['all_scores_finite'] is True
    assert measured['within_registered_sum_bound'] == [True] * 2
    assert measured['numpy_failure_recorded'] is True
    assert measured['numpy_missing_category_recorded'] is True
    assert measured['invalid_crossover_recorded'] is True
    assert measured['source_order_reversed'] is True
    assert measured['purpose_overridden'] is True
    assert measured['bad_energy_key_failed'] is True
    text = str(measured['report'])
    for label in ('替身誤差', '實作差', '逐類表', '原牆基準', '改善', 'f_s（', 'capped_by_upper_limit（'):
        assert label in text


def test_improvements_bitwise_baseline_minus_endpoint(measured: dict[str, object]) -> None:
    """抓改善漏掉一類、基準誤用終點、或把兩個不同總分交叉相減。"""
    assert measured['improvement_bitwise'] is True


def test_v4_report_snapshot_and_read_only_versions(tmp_path: Path) -> None:
    """4b8223f7 改前由既有完整解紀錄存的段落，逐字守舊版報告。"""
    snapshot = json.loads(SNAPSHOT_JSON)
    document = snapshot['record']
    (tmp_path / 'endpoint.json').write_text(json.dumps(document))
    assert _endpoint_section(read_endpoint(tmp_path, legacy=True)) == snapshot['lines']
    with pytest.raises(ValueError, match='版本'):
        read_endpoint(tmp_path)
    with pytest.raises(ValueError, match='不准有'):
        EndpointRecord.model_validate({**document, 'six_category': None})
    document['format_version'] = 'aosr.material_gradient_endpoint.v5'
    with pytest.raises(ValueError, match='三組數'):
        EndpointRecord.model_validate(document)


def test_v4_rejects_v5_fields_and_v5_requires_complete_groups(measured: dict[str, object]) -> None:
    document = cast(dict[str, object], measured['record']).copy()
    document['format_version'] = 'aosr.material_gradient_endpoint.v4'
    with pytest.raises(ValueError, match='不准有'):
        EndpointRecord.model_validate(document)
    document = json.loads(json.dumps(measured['record']))
    for name in ('endpoint', 'baseline', 'improvement'):
        bad = json.loads(json.dumps(document))
        del bad['six_category'][name]
        with pytest.raises(ValueError, match='Field required'):
            EndpointRecord.model_validate(bad)


def test_implementation_differences_and_single_material_crossover(measured: dict[str, object]) -> None:
    record = EndpointRecord.model_validate(measured['record'])
    six = record.six_category
    assert six is not None
    for item in (six.endpoint, six.baseline):
        assert all(d == j - n for j, n, d in zip(item.full.categories, item.numpy_categories, item.implementation_categories, strict=True) if j is not None and n is not None)
    text = str(measured['report'])
    assert 'Σ w·tol' not in text and '超界' not in text
    assert text.count('f_s（') == 2
    assert text.count('匯出方案：') == 1



@pytest.mark.parametrize('version', ('aosr.material_gradient_endpoint.v5', 'aosr.material_gradient_endpoint.v4',
                                  'aosr.material_gradient_endpoint.v3', 'aosr.material_gradient_endpoint.v2', 'broken'))
def test_old_endpoint_cannot_be_recomputed_or_overwritten(tmp_path: Path, version: str) -> None:
    """在讀專案或求解前就拒收舊 endpoint；舊檔位元組不變。"""
    path = tmp_path / 'endpoint.json'
    path.write_text('{' if version == 'broken' else json.dumps({'format_version': version}))
    before = path.read_bytes()
    with pytest.raises(ValueError, match='要重算先把舊檔移走'):
        run_endpoint(tmp_path)
    assert path.read_bytes() == before


def test_bare_measured_original_wall_refused_without_endpoint_or_temporary_files(tmp_path: Path) -> None:
    """裸實測原牆沿正式終點入口拒收；不得寫出結果或留下半份暫存檔。"""
    done = subprocess.run([sys.executable, '-m', 'tests.engine._six_category_endpoint_probe', str(tmp_path), 'measured-rejection'],
        cwd=tmp_path, env=endpoint_probe_env(tmp_path), capture_output=True, text=True, timeout=1800)
    (tmp_path / 'probe.stdout').write_text(done.stdout)
    (tmp_path / 'probe.stderr').write_text(done.stderr)
    assert done.returncode == 0, done.stdout + done.stderr
    result = json.loads((tmp_path / 'rejection.json').read_text())
    assert result['error_type'] == 'ValueError', result
    assert result['error_origin'] == 'baseline_impedances', result
    assert '實測原牆的表內／表外吸音分流尚未接入六類終點' in result['error_message']
    assert result['endpoint_exists'] is False, result
    assert result['new_files'] == [], result
    assert result['endpoint_temporary_files'] == [], result


def test_endpoint_probe_uses_worker_environment_without_cpu_pins(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """父考卷即使釘住 AVX2，探針也必須跟不繼承釘選的正式工作行程一致。"""
    pins = {'NPY_DISABLE_CPU_FEATURES': 'X86_V4 AVX512_ICL AVX512_SPR',
            'OPENBLAS_CORETYPE': 'Haswell', 'XLA_FLAGS': '--xla_cpu_max_isa=AVX2'}
    for name, value in pins.items():
        monkeypatch.setenv(name, value)
    env = endpoint_probe_env(tmp_path)
    assert not (pins.keys() & env.keys())
    assert env['JAX_PLATFORMS'] == 'cpu' and env['JAX_ENABLE_X64'] == 'true'
    assert all(env[name] == '1' for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'))
