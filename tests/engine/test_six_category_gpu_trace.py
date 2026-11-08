"""單精度低頻只建一次值／梯度圖；這題不等待數值編譯與完整小搜尋。"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

from aosr.runtime import child_process_env, configure_jax
from tests.engine._scheme_cache import shared_json


@pytest.fixture(scope='module')
def traced(tmp_path_factory: pytest.TempPathFactory, worker_id: str) -> dict[str,int]:
    def produce() -> str:
        directory = tmp_path_factory.mktemp('single-low-trace')
        output = directory/'trace.json'
        env = child_process_env(threads=1)
        configure_jax(platforms='cpu',enable_x64=True,env=env)
        env['PYTHONPATH'] = os.pathsep.join((env['PYTHONPATH'],str(Path(__file__).resolve().parents[2])))
        done = subprocess.run([sys.executable,'-m','tests.engine._six_category_gpu_review_probe','--trace',str(output)],
            cwd=directory,env=env,capture_output=True,text=True,timeout=120)
        assert done.returncode == 0,done.stdout+done.stderr
        return output.read_text()
    return cast(dict[str,int],json.loads(shared_json(tmp_path_factory,worker_id,'single-low-trace',produce)))


@pytest.mark.parametrize('case',('normal','kept'))
def test_single_builds_only_one_low_frequency_objective(traced: dict[str,int],case: str) -> None:
    assert traced[case] == 1


def test_precomputed_low_keeps_invalid_pressure_unestimable_and_valid_gradient_finite(tmp_path: Path) -> None:
    output = tmp_path/'invalid.json'
    env = child_process_env(threads=1)
    configure_jax(platforms='cpu',enable_x64=True,env=env)
    env['PYTHONPATH'] = os.pathsep.join((env['PYTHONPATH'],str(Path(__file__).resolve().parents[2])))
    done = subprocess.run([sys.executable,'-m','tests.engine._six_category_gpu_review_probe','--invalid',str(output)],
        cwd=tmp_path,env=env,capture_output=True,text=True,timeout=900)
    assert done.returncode == 0,done.stdout+done.stderr
    data = json.loads(output.read_text())
    assert data['valid']=={'estimable':True,'value_nan':False,'gradient_finite':True}
    for name in ('zero','nan'):
        assert data[name]['estimable'] is False and data[name]['value_nan'] is True
