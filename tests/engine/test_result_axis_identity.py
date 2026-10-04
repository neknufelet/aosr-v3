"""報表宣告的低頻軸身分要跟逐點表、路徑表的頻率逐點相同（#494）：新算的與從檔案讀回的方案結果都擋。"""

import json
import math
import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from aosr.config.capabilities import load_capabilities
from aosr.config.frequency_axis import LowFrequencyAxis, low_frequency_axis_frequencies
from aosr.config.paths import config_path
from aosr.reporting.evaluation import load_result
from aosr.reporting.result import SchemeResult, save_result
from tests.engine import _scoring_source_model_control as control
from tests.engine._directivity import DIRECTIVITY
from tests.engine.test_scheme_pipeline import shared_control_result

Document = dict[str, object]


@pytest.fixture(scope="module")
def wall_1(tmp_path_factory: pytest.TempPathFactory, worker_id: str) -> SchemeResult:
    return shared_control_result(tmp_path_factory, worker_id, "wall-1")


def _report(document: Document, index: int) -> dict[str, object]:
    pairs = document["pairs"]
    assert isinstance(pairs, list)
    report = pairs[index]["report"]
    assert isinstance(report, dict)
    return report


def _refusal(axis: str, table: str) -> str:
    return re.escape(f"報表宣告的低頻軸（{axis}）跟{table}的頻率對不上")


def test_control_pairs_declare_the_axis_their_rows_are_on(wall_1: SchemeResult) -> None:
    """前提：控制組每一組都有逐點表與路徑表、宣告搜尋軸、跟軸逐點相同——下面的竄改才有意義。"""
    expected = low_frequency_axis_frequencies(LowFrequencyAxis.SEARCH)[1]
    for pair in wall_1.pairs:
        assert pair.report.top.low_frequency_axis is LowFrequencyAxis.SEARCH
        assert pair.report.points is not None and pair.report.path_table is not None
        assert tuple(point.frequency_hz for point in pair.report.points) == expected
        assert pair.report.path_table.frequencies_hz == expected


def test_declaring_another_axis_is_refused(wall_1: SchemeResult) -> None:
    document = wall_1.model_dump(mode="json")
    top = _report(document, 0)["top"]
    assert isinstance(top, dict)
    top["low_frequency_axis"] = LowFrequencyAxis.VERIFICATION.value
    with pytest.raises(ValidationError, match=_refusal(LowFrequencyAxis.VERIFICATION.value, "逐點表")):
        SchemeResult.model_validate(document)


def test_points_off_the_declared_axis_are_refused(wall_1: SchemeResult) -> None:
    """宣告與路徑表都對、只有逐點表最後一點偏一點點：照樣擋（逐點比，不是只比點數或頭尾）。"""
    document = wall_1.model_dump(mode="json")
    points = _report(document, 0)["points"]
    assert isinstance(points, list)
    points[-1]["frequency_hz"] = math.nextafter(points[-1]["frequency_hz"], math.inf)
    with pytest.raises(ValidationError, match=_refusal(LowFrequencyAxis.SEARCH.value, "逐點表")):
        SchemeResult.model_validate(document)


def test_path_table_off_the_declared_axis_is_refused(wall_1: SchemeResult) -> None:
    document = wall_1.model_dump(mode="json")
    table = _report(document, -1)["path_table"]
    assert isinstance(table, dict)
    frequencies = table["frequencies_hz"]
    assert isinstance(frequencies, list)
    frequencies[-1] = math.nextafter(frequencies[-1], math.inf)
    with pytest.raises(ValidationError, match=_refusal(LowFrequencyAxis.SEARCH.value, "路徑表")):
        SchemeResult.model_validate(document)


def test_a_saved_result_with_a_mislabelled_axis_does_not_load(wall_1: SchemeResult, tmp_path: Path) -> None:
    """從檔案讀回是碰得到的那條路：存檔後把一組的宣告改成驗證軸，讀回就拒收，不重評、不進比較。"""
    path = tmp_path / "result.json"
    save_result(wall_1, path)
    document = json.loads(path.read_text(encoding="utf-8"))
    top = _report(document, 0)["top"]
    assert isinstance(top, dict)
    top["low_frequency_axis"] = LowFrequencyAxis.VERIFICATION.value
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValidationError, match=_refusal(LowFrequencyAxis.VERIFICATION.value, "逐點表")):
        load_result(path, capabilities=load_capabilities(config_path("capabilities.toml")), directivity=DIRECTIVITY,
                    quality_targets_path=control.TARGETS, physics_identity="phys-v1:" + "0" * 64)
