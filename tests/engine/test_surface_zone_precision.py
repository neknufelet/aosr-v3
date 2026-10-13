"""格式驗證的冷匯入邊界：JAX（數值函式庫）不可在開門驗輸入時被載入。"""
import json
import subprocess
import sys
from pathlib import Path

from aosr.runtime import child_process_env
from tests.engine._surface_zone_cases import zone_document


def test_cold_construction_validation_does_not_import_jax(tmp_path: Path) -> None:
    document = tmp_path / "scheme.json"
    document.write_text(json.dumps(zone_document(mixed=True)), encoding="utf-8")
    probe = """
import json, sys
import aosr.reporting.physics_stage
from aosr.config.capabilities import load_capabilities
from aosr.config.paths import config_path
from aosr.config.directivity_defaults import load_directivity_defaults
from aosr.reporting.validation import checked_inputs
value = json.load(open(sys.argv[1], encoding='utf-8'))
checked_inputs(value, capabilities=load_capabilities(config_path('capabilities.toml')),
               directivity=load_directivity_defaults(config_path('directivity_defaults.toml')))
assert 'jax' not in sys.modules
assert 'aosr.materials.construction' not in sys.modules
"""
    completed = subprocess.run([sys.executable, "-B", "-c", probe, str(document)], cwd=tmp_path,
                              env=child_process_env(threads=1), capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stdout + completed.stderr
