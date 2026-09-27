"""聲源模型能力表入口；解析近似只標 experimental、實測仍未開放，兩列都不帶驗證收據。"""

from importlib.util import find_spec

from aosr.config.capabilities import load_capabilities
from aosr.config.paths import config_path
from aosr.physics.report_io import PathRow, quantity_table
from aosr.physics.report_source import SourceModelKind


def test_source_directivity_entry_declares_analytic_experimental_and_measured_unsupported() -> None:
    entry = load_capabilities(config_path("capabilities.toml")).for_entry("source_directivity")
    assert entry.module == "aosr.physics.source_directivity"
    assert find_spec(entry.module) is not None
    assert {item.materials for item in entry.capability} == {
        SourceModelKind.ANALYTIC_AXISYMMETRIC_TWO_PARAMETER_V1.value,
        "measured_polar_data",
    }
    assert all(item.room == "none" for item in entry.capability)
    assert all(not item.evidence for item in entry.capability)
    analytic = next(item for item in entry.capability if item.materials == SourceModelKind.ANALYTIC_AXISYMMETRIC_TWO_PARAMETER_V1.value)
    assert analytic.status == "experimental"
    assert next(item for item in entry.capability if item.materials == "measured_polar_data").status == "unsupported"
    assert "上下方向" in analytic.note and "尚未獨立驗證" in analytic.note
    for phrase in ("俯仰", "有限元素", "g 一次", "T20／T30", "房間平均", "35 格", "完整聲場的保證", "125 Hz 以下"):
        assert phrase in analytic.note
    fields = quantity_table()
    for output in analytic.outputs:
        if output.startswith("reflection_window.rows."):
            assert output.removeprefix("reflection_window.rows.") in PathRow.model_fields
        else:
            assert output in fields
