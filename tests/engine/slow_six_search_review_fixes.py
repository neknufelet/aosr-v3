"""本機合併前驗收的搜尋與終點重考卷；考卷本體由原檔原樣搬移。"""
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

from tests.engine.test_six_search_review_fixes import (
    _probe as _probe,
    measured as measured,
)


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
