"""守共用層不掩蓋重建漂移。"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from tests.engine._gradient_probe_env import endpoint_probe_env


def test_resume_reconstruction_matches_its_baseline(tmp_path: Path) -> None:
    result = subprocess.run([sys.executable, '-m', 'tests.engine._gradient_rebuild_guard_probe', str(tmp_path)],
        cwd=tmp_path, env=endpoint_probe_env(tmp_path), capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    records = json.loads((tmp_path / 'result.json').read_text())
    control, drift = records['control'], records['drift']
    assert control['reproduced'] is True, control
    assert control['error'] is None and control['error_type'] is None, control
    assert control['reconstructed_objectives'] >= 2, control
    assert drift['reproduced'] is False, drift
    assert drift['error_type'] == 'ValueError', drift
    assert drift['error'] == '重建目標在起點的值、分項或梯度逐位不同，不准共用', drift
    assert drift['reconstructed_objectives'] >= 2, drift
