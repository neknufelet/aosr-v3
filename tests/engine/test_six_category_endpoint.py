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
from tests.engine.test_material_gradient_cli import _env


@pytest.fixture(scope='module')
def measured(tmp_path_factory: pytest.TempPathFactory, worker_id: str) -> dict[str, object]:
    def produce() -> str:
        directory = tmp_path_factory.mktemp('six-endpoint')
        done = subprocess.run([sys.executable, '-m', 'tests.engine._six_category_endpoint_probe', str(directory)],
            cwd=directory, env=_env(directory), capture_output=True, text=True, timeout=1800)
        assert done.returncode == 0, done.stdout + done.stderr
        return (directory / 'measure.json').read_text()
    return cast(dict[str, object], json.loads(shared_json(tmp_path_factory, worker_id, 'six-endpoint', produce)))


def test_search_endpoint_report_three_groups_and_independent_scores(measured: dict[str, object]) -> None:
    """抓完整聲壓偷換替身、產品混 J、20/30 軸交換，以及多解一次完整模型。"""
    assert measured['version'] == 'aosr.material_gradient_endpoint.v5'
    assert measured['solves'] == measured['expected_solves']
    assert measured['jax_bitwise'] == [True] * 4
    assert measured['numpy_bitwise'] == [True] * 2
    assert measured['axes_distinct'] is True
    assert measured['diagnostics'] is True
    assert measured['all_scores_finite'] is True
    assert measured['bound_bitwise'] == [True] * 2
    assert measured['numpy_failure_recorded'] is True
    assert measured['numpy_missing_category_recorded'] is True
    assert measured['invalid_crossover_recorded'] is True
    assert measured['source_order_reversed'] is True
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


def test_over_bound_is_reported_without_blocking(measured: dict[str, object]) -> None:
    """抓把實作總分界誤作終點閘門；超界只在報告標出。"""
    assert measured['over_bound_entry_allowed'] is True


@pytest.mark.parametrize('version', ('aosr.material_gradient_endpoint.v4', 'aosr.material_gradient_endpoint.v3',
                                  'aosr.material_gradient_endpoint.v2'))
def test_old_endpoint_cannot_be_recomputed_or_overwritten(tmp_path: Path, version: str) -> None:
    """在讀專案或求解前就拒收舊 endpoint；舊檔位元組不變。"""
    path = tmp_path / 'endpoint.json'
    path.write_text(json.dumps({'format_version': version}))
    before = path.read_bytes()
    with pytest.raises(ValueError, match='舊版只准報告讀，不准接續或重算'):
        run_endpoint(tmp_path)
    assert path.read_bytes() == before
