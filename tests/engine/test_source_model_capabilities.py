"""聲源模型能力表入口；未開放種類不假裝有驗證收據。"""

from importlib.util import find_spec

from aosr.config.capabilities import load_capabilities
from aosr.config.paths import config_path
from aosr.physics.report_source import SourceModelKind


def test_source_directivity_entry_declares_unsupported_models() -> None:
    entry = load_capabilities(config_path("capabilities.toml")).for_entry("source_directivity")
    assert entry.module == "aosr.physics.source_directivity"
    assert find_spec(entry.module) is not None
    assert {item.materials for item in entry.capability} == {
        SourceModelKind.ANALYTIC_AXISYMMETRIC_TWO_PARAMETER_V1.value,
        "measured_polar_data",
    }
    assert all(item.room == "none" and item.status == "unsupported" for item in entry.capability)
    assert all(not item.evidence for item in entry.capability)
    analytic = next(item for item in entry.capability if item.materials == SourceModelKind.ANALYTIC_AXISYMMETRIC_TWO_PARAMETER_V1.value)
    assert "第四刀" in analytic.note
    assert "上下方向" in analytic.note and "尚未獨立驗證" in analytic.note

