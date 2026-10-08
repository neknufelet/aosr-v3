"""存檔重建雙向核家具表頭，接觸尺只由入口讀一次並顯式轉傳。"""
from __future__ import annotations

from pathlib import Path

import pytest

from aosr.config.precision_contracts import default_precision_contracts_path
from aosr.physics.reflection_window import ReflectionWindow, build_reflection_window
from aosr.physics.report_io import ReportInput, ReportOutput
from aosr.reporting import evaluation
from tests.engine import _furniture_energy_cases as case, test_reflections
from tests.engine.test_furniture_reflection_window_wiring import inputs
from tests.engine.test_furniture_scoring_flags import furniture_report


@pytest.fixture(scope="module")
def plain_report() -> ReportOutput:
    return test_reflections._pair()[0].report


@pytest.mark.parametrize(("mismatch", "message"), [
    ("input_only", "輸入有家具，存下來的路徑表表頭沒有家具"),
    ("header_only", "輸入沒有家具，存下來的路徑表表頭卻有家具"),
    ("different_ids", "存下來的路徑表表頭與輸入的家具代號不一致"),
    ("different_materials", "存下來的路徑表表頭與輸入的家具材質不一致"),
])
def test_stored_header_must_match_input_in_both_directions(
    plain_report: ReportOutput, mismatch: str, message: str,
) -> None:
    data, report = inputs(), furniture_report(plain_report)
    if mismatch == "input_only":
        report = plain_report
    elif mismatch == "header_only":
        data = inputs(furnished=False)
    elif mismatch == "different_ids":
        report = furniture_report(plain_report, "coffee")
    else:
        assert data.furniture is not None
        data = data.model_copy(update={"furniture": (data.furniture[0].model_copy(update={"material": "glass"}),)})
    with pytest.raises(ValueError, match=message):
        evaluation.build_pair_window(data, report, 0.015)


def test_matching_stored_header_equals_direct_window(plain_report: ReportOutput) -> None:
    data, report = inputs(), furniture_report(plain_report)
    assert report.path_table is not None
    expected = build_reflection_window(data, frequencies_hz=report.path_table.frequencies_hz,
        scattering_coefficient=report.path_table.scattering_coefficient, window_s=0.015,
        contact_rel=case.CONTACT_REL)
    assert evaluation.build_pair_window(data, report, 0.015) == expected


@pytest.mark.parametrize("furnished", [False, True])
def test_pair_hands_the_contact_it_read_to_the_window(
    plain_report: ReportOutput, monkeypatch: pytest.MonkeyPatch, furnished: bool,
) -> None:
    data = inputs(furnished=furnished)
    report = furniture_report(plain_report) if furnished else plain_report
    marker = case.CONTACT_REL * 3.0
    reads: list[Path] = []
    original = build_reflection_window

    def read_contact(path: Path) -> float:
        if not furnished:
            pytest.fail("沒家具不准讀接觸登記簿")
        reads.append(path)
        return marker

    def supplied_window(inputs: ReportInput, *, frequencies_hz: tuple[float, ...],
                        scattering_coefficient: tuple[float, ...], window_s: float,
                        contact_rel: float | None = None) -> ReflectionWindow:
        assert contact_rel == (marker if furnished else None)
        return original(inputs, frequencies_hz=frequencies_hz,
            scattering_coefficient=scattering_coefficient, window_s=window_s, contact_rel=contact_rel)

    monkeypatch.setattr(evaluation, "furniture_contact_rel", read_contact, raising=False)
    monkeypatch.setattr(evaluation, "build_reflection_window", supplied_window)
    result = evaluation.build_pair_window(data, report, 0.015)
    assert result.furniture_ids == (("desk",) if furnished else None)
    assert reads == ([default_precision_contracts_path()] if furnished else [])


def test_raw_unfurnished_window_never_reads_furniture_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(path: Path) -> float:
        pytest.fail("物理核心不准讀接觸登記簿")

    monkeypatch.setattr("aosr.config.precision_contracts.furniture_contact_rel", forbidden)
    result = build_reflection_window(inputs(furnished=False), frequencies_hz=(125.0, 250.0),
        scattering_coefficient=(0.2, 0.3), window_s=0.015)
    assert result.coverage == "complete" and result.furniture_ids is None
