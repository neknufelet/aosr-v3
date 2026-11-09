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
