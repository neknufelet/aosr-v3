"""B2a 審查第 1 輪：混合拒收、產品行程不可估與 CLI 換版邊界。"""
from __future__ import annotations

from collections.abc import Callable
import json
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

from tests.engine._scheme_cache import shared_json
from tests.engine._six_search_cli_case import model as model, operators as operators
from tests.engine.test_material_gradient_cli import _env
from tests.engine.test_material_gradient_endpoint_cli import _inputs, _surrogate


def _probe(directory: Path, mode: str) -> dict[str,object]:
    done = subprocess.run([sys.executable,'-m','tests.engine._six_search_review_probe',mode,str(directory)],
        cwd=directory,env=_env(directory),capture_output=True,text=True,timeout=180)
    assert done.returncode==0,done.stdout+done.stderr
    return cast(dict[str,object],json.loads((directory/(mode+'.json')).read_text()))


@pytest.fixture(scope='module')
def measured(request: pytest.FixtureRequest,tmp_path_factory: pytest.TempPathFactory,worker_id: str
             ) -> Callable[[str],dict[str,object]]:
    def get(mode: str) -> dict[str,object]:
        def produce() -> str:
            directory=tmp_path_factory.mktemp('six-review-'+mode)
            if mode=='cli':
                _inputs(directory)
                _surrogate(directory,request)
            return json.dumps(_probe(directory,mode))
        return cast(dict[str,object],json.loads(shared_json(tmp_path_factory,worker_id,'six-review-'+mode,produce)))
    return get


@pytest.mark.parametrize('book',('template','material'))
def test_first_rejected_second_valid_survives_candidate_batch_resume(measured: Callable[[str],dict[str,object]],book: str) -> None:
    row=cast(dict[str,object],measured('mixed')[book])
    assert row['first_rejected'] is True and row['second_valid'] is True
    assert row['selected']==row['second']
    assert row['prefix_last']=='candidate_retune' and row['candidate_batches']==1
    assert row['ledger_bitwise'] is True and row['result_bitwise'] is True


@pytest.mark.parametrize('phase',('start','gradient','gradient_inf','template'))
def test_worker_unestimable_is_deterministic_stopped_and_reported(measured: Callable[[str],dict[str,object]],phase: str) -> None:
    row=cast(dict[str,object],measured('worker')[phase])
    reason='換樣板基準重調起點不可估' if phase=='template' else '起點不可估'
    assert row['codes']==[3,3] and row['state']=='stopped'
    assert row['reason']==reason
    assert f'停止原因：{reason}' in str(row['report']).splitlines()
    assert '決定性的' in str(row['message']) and '接續會得到同一個結果' in str(row['stderr'])
    assert '要換輸入或設定' in str(row['message']) and '接續用 resume' not in str(row['stderr'])
    if phase!='template': assert row['rows']==0


def test_old_v5_resume_cli_explains_six_category_migration(measured: Callable[[str],dict[str,object]]) -> None:
    row=measured('cli')
    assert row['resume_code']==1
    for text in ('六類目標換版','梯度目標 v2','帳本 v6','舊資料夾只讀','要重跑'):
        assert text in str(row['resume_stderr'])


def test_gpu_cli_creates_single_precision_project(measured: Callable[[str],dict[str,object]]) -> None:
    row=measured('cli')
    assert row['gpu_code']==0 and row['gpu_wrappers']==[True]
    assert row['gpu_root_exists'] is True and len(cast(list[str],row['gpu_projects']))==1


def test_load_problem_validates_pair_inputs_only_once(measured: Callable[[str],dict[str,object]]) -> None:
    assert measured('cli')['checked_calls']==[1]


@pytest.mark.parametrize(('version','expected'),(
    ('aosr.material_gradient_ledger.v6','aosr.material_gradient_templates.v3'),
    ('aosr.material_gradient_ledger.v5','aosr.material_gradient_templates.v2'),
    ('aosr.material_gradient_ledger.v4','aosr.material_gradient_templates.v1')))
def test_template_header_uses_explicit_ledger_version_mapping(version: str,expected: str) -> None:
    from aosr.gradient import run
    from aosr.geometry.shoebox import Wall
    from tests.engine._material_gradient_loop_probe import synthetic_header
    from tests.engine._material_gradient_templates_probe import kept_walls
    from tests.engine.test_material_gradient import DESCENT_CASE,_designs,_settings
    from tests.engine.test_material_gradient_templates import KEPT
    header=synthetic_header(_designs(DESCENT_CASE),_settings())
    constants=header.score_constants
    assert constants is not None
    low=constants.model_copy(update={'category_weights':None})
    header=header.model_copy(update={'ledger_version':version,
        'score_constants':constants if version.endswith('v6') else low if version.endswith('v5') else None})
    materials=kept_walls(KEPT)
    kept={wall:materials[wall.wall_name()] for wall in Wall.all()}
    result=run.template_header_for(header,_settings(),kept,base_ledger_sha256='e'*64)
    assert result.ledger_version==expected
