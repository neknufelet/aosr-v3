"""物理改動的成本偵測器：答案由目前入口靜態分析產生。"""
from aosr.config.capabilities import load_capabilities
from aosr.config.directivity_defaults import load_directivity_defaults
from aosr.config.paths import config_path
from aosr.reporting.physics_identity import physics_identity_parts
import sys

from tests.engine._physics_identity_answers import CLOSURE, CODE_DIGEST, DATA_FILES, PYTHON_MINOR


def test_code_part_of_physics_identity_is_pinned() -> None:
    parts = physics_identity_parts(
        capabilities=load_capabilities(config_path("capabilities.toml")),
        directivity=load_directivity_defaults(config_path("directivity_defaults.toml")),
    )
    assert sys.version_info[:2] == PYTHON_MINOR, (
        "直譯器小版本跟錄答案時不同：程式摘要（ast.dump）要重錄，這不是物理變更")
    assert parts.closure == CLOSURE
    assert parts.data_files == DATA_FILES
    assert parts.code_digest == CODE_DIGEST
