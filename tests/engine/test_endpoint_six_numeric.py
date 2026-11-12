"""六類終點的數值接線控制，供六種植錯獨立重跑。"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.engine.test_material_gradient_cli import _env


@pytest.mark.parametrize('measured_case', (False, True), ids=('constant_and_assembly', 'measured_floor'))
def test_endpoint_scores_and_improvements_use_matching_inputs(tmp_path: Path, measured_case: bool) -> None:
    """抓聲壓偷換、J 混產品、低頻軸互換、改善漏減、聲源錯位、原牆偷用終點阻抗；實測表那一場（#1082 段 2-3d）另抓
    終點六類吸音沒照產品入口表內／表外分流（JAX 分數對正式 numpy 排名總分超出登記容差加權和）。"""
    target = tmp_path / 'values.json'
    result = subprocess.run([sys.executable, '-m', 'tests.engine._endpoint_six_numeric_probe', str(target),
                             *(('measured',) if measured_case else ())],
        cwd=tmp_path, env=_env(tmp_path), text=True, capture_output=True, timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr
    measured = json.loads(target.read_text())
    assert measured['jax_bitwise'] == [True, True]
    assert measured['numpy_bitwise'] is True
    assert measured['improvement_bitwise'] is True
    assert measured['within_registered_sum_bound'] == [True, True]
    assert measured['measured_kept'] is measured_case
