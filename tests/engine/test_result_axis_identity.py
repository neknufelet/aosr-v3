"""報表宣告的低頻軸身分要跟逐點表、路徑表的頻率逐點相同（#494）：物理身分跟現在相同的結果讀回就擋；
物理改過的舊結果（日後改軸）照讀回分級標「要重算」，不當壞檔擋（複查）。"""

import json
import math
import re
from pathlib import Path

import pytest
from aosr.config.capabilities import load_capabilities
from aosr.config.frequency_axis import LowFrequencyAxis, low_frequency_axis_frequencies
from aosr.config.paths import config_path
from aosr.reporting.evaluation import ResultStanding, load_result
from aosr.reporting.result import SchemeResult, check_declared_axes, save_result
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


def _checked(document: Document) -> None:
    check_declared_axes(SchemeResult.model_validate(document))


def test_the_control_result_passes(wall_1: SchemeResult) -> None:
    check_declared_axes(SchemeResult.model_validate(wall_1.model_dump(mode="json")))


def test_declaring_another_axis_is_refused(wall_1: SchemeResult) -> None:
    document = wall_1.model_dump(mode="json")
    top = _report(document, 0)["top"]
    assert isinstance(top, dict)
    top["low_frequency_axis"] = LowFrequencyAxis.VERIFICATION.value
    with pytest.raises(ValueError, match=_refusal(LowFrequencyAxis.VERIFICATION.value, "逐點表")):
        _checked(document)


def test_points_off_the_declared_axis_are_refused(wall_1: SchemeResult) -> None:
    """宣告與路徑表都對、只有逐點表最後一點偏一點點：照樣擋（逐點比，不是只比點數或頭尾）。"""
    document = wall_1.model_dump(mode="json")
    points = _report(document, 0)["points"]
    assert isinstance(points, list)
    points[-1]["frequency_hz"] = math.nextafter(points[-1]["frequency_hz"], math.inf)
    with pytest.raises(ValueError, match=_refusal(LowFrequencyAxis.SEARCH.value, "逐點表")):
        _checked(document)


def test_path_table_off_the_declared_axis_is_refused(wall_1: SchemeResult) -> None:
    document = wall_1.model_dump(mode="json")
    table = _report(document, -1)["path_table"]
    assert isinstance(table, dict)
    frequencies = table["frequencies_hz"]
    assert isinstance(frequencies, list)
    frequencies[-1] = math.nextafter(frequencies[-1], math.inf)
    with pytest.raises(ValueError, match=_refusal(LowFrequencyAxis.SEARCH.value, "路徑表")):
        _checked(document)


def _mislabelled(wall_1: SchemeResult, tmp_path: Path) -> Path:
    """存檔後把一組的宣告改成驗證軸（逐點表、路徑表還是搜尋軸）。"""
    path = tmp_path / "result.json"
    save_result(wall_1, path)
    document = json.loads(path.read_text(encoding="utf-8"))
    top = _report(document, 0)["top"]
    assert isinstance(top, dict)
    top["low_frequency_axis"] = LowFrequencyAxis.VERIFICATION.value
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _load(path: Path, physics_identity: str) -> ResultStanding:
    return load_result(path, capabilities=load_capabilities(config_path("capabilities.toml")), directivity=DIRECTIVITY,
                       quality_targets_path=control.TARGETS, physics_identity=physics_identity).standing


def test_a_current_result_with_a_mislabelled_axis_does_not_load(wall_1: SchemeResult, tmp_path: Path) -> None:
    """物理身分跟現在相同、宣告卻跟表對不上：讀回就拒收，不重評、不進比較。"""
    with pytest.raises(ValueError, match=_refusal(LowFrequencyAxis.VERIFICATION.value, "逐點表")):
        _load(_mislabelled(wall_1, tmp_path), wall_1.physics_identity)


def test_an_older_physics_result_is_marked_for_recalculation_not_refused(wall_1: SchemeResult, tmp_path: Path) -> None:
    """物理改過的舊結果（例如日後改了軸，舊報表的表就不等於現在的軸）：照讀回分級標要重算，不當壞檔擋。"""
    older = "phys-v1:" + "0" * 64
    assert wall_1.physics_identity != older
    assert _load(_mislabelled(wall_1, tmp_path), older) is ResultStanding.NEEDS_PHYSICS
