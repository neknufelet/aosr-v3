"""接續必須使用逐位相同的新建目標；不讓共用層掩蓋重建後係數漂移。"""
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
    record = json.loads((tmp_path / 'result.json').read_text())
    assert record['reproduced'] is True, record
    assert record['reconstructed_objectives'] >= 2, record
