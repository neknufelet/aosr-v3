"""六類終點的數值接線控制，供六種植錯獨立重跑。"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tests.engine.test_material_gradient_cli import _env


def test_endpoint_scores_and_improvements_use_matching_inputs(tmp_path: Path) -> None:
    """抓聲壓偷換、J 混產品、低頻軸互換、改善漏減、聲源錯位、原牆偷用終點阻抗。"""
    target = tmp_path / 'values.json'
    result = subprocess.run([sys.executable, '-m', 'tests.engine._endpoint_six_numeric_probe', str(target)],
        cwd=tmp_path, env=_env(tmp_path), text=True, capture_output=True, timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr
    measured = json.loads(target.read_text())
    assert measured['jax_bitwise'] == [True, True]
    assert measured['numpy_bitwise'] is True
    assert measured['improvement_bitwise'] is True
