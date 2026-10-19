"""報告開頭材料研究提醒：依方案條件，不依外推註記；下限取暫存登記簿。"""
from copy import deepcopy
from pathlib import Path
from typing import cast

import pytest

from aosr.config.paths import config_path
from aosr.gui.result_view import build_result_view
from aosr.physics import report_io, three_lane_report
from aosr.physics.three_lane_report_cli import _text_sections
from aosr.reporting.result import SchemeResult
from aosr.reporting.scheme import Scheme
from tests.engine._directivity import DIRECTIVITY
from tests.engine._furniture_cases import CAPABILITIES, pair
from tests.engine._report_cache import shared_report
from tests.engine._surface_zone_cases import ASSEMBLY
from tests.engine.test_construction_presets import preset_document
from tests.engine.test_gui_construction_options import facing_document
from tests.engine.test_report_scene import _inputs, _solved_report
from tests.engine.test_scheme_pipeline import shared_control_result


@pytest.fixture(scope="module")
def result(tmp_path_factory: pytest.TempPathFactory, worker_id: str) -> SchemeResult:
    return shared_control_result(tmp_path_factory, worker_id, "wall-1")


@pytest.fixture(scope="module")
def cli_report(tmp_path_factory: pytest.TempPathFactory, worker_id: str) -> three_lane_report.ThreeLaneReport:
    def produce() -> three_lane_report.ThreeLaneReport:
        with pytest.MonkeyPatch.context() as patch:
            return _solved_report(patch, _inputs())
    return shared_report(tmp_path_factory, worker_id, "construction-notice-cli", produce)


def notice_scheme(case: str) -> Scheme:
    if case == "board":
        return Scheme.model_validate(preset_document(deepcopy(ASSEMBLY)))
    data = facing_document(measured=case == "measured-base")
    if case == "miki":
        scene = cast(dict[str, object], data["scene"])
        zones = cast(dict[str, list[dict[str, object]]], scene["surface_zones_by_wall"])
        material = cast(dict[str, object], zones["floor"][0]["material"])
        cast(list[dict[str, object]], material["layers"])[1]["model"] = "miki"
    return Scheme.model_validate(data)


@pytest.mark.parametrize("case,present", (("porous", True), ("measured-base", True), ("board", False), ("miki", False), ("fixed", False)))
def test_cli_notice_follows_materials_including_measured_base(
    cli_report: three_lane_report.ThreeLaneReport, case: str, present: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs() if case == "fixed" else report_io.load_input_document(pair(notice_scheme(case)), CAPABILITIES, DIRECTIVITY)
    # 本題只考報告條件與排序；接縫數值已有獨立雙精度子行程考卷。
    monkeypatch.setattr("aosr.physics.report_measured_text.measured_seams", lambda *args, **kwargs: None)
    sections = _text_sections(cli_report, inputs=inputs, with_points=False)
    text = "\n".join(sections)
    assert ("Komatsu（小松）纖維模型：原論文只量到 100 Hz 以上" in text) is present
    assert sections[0].startswith("capability ")
    assert sections[1].startswith("scene ")


@pytest.mark.parametrize("case,present", (("porous", True), ("measured-base", True), ("board", False), ("miki", False), ("fixed", False)))
def test_result_notice_follows_materials_without_saved_extrapolation_notes(result: SchemeResult, case: str, present: bool) -> None:
    changed = display_result(result, case)
    view = build_result_view(changed, quality_targets_path=config_path("quality_targets.toml"))
    expected = "小松纖維模型：原論文只量到 100 Hz 以上" if present else None
    assert view.material_research_notice == expected
    assert "Komatsu" not in view.model_dump_json()


def copy_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "ranges.toml"
    source = config_path("porous_model_ranges.toml").read_text()
    path.write_text(source.replace("measured_frequency_min_hz = 100", "measured_frequency_min_hz = 137"))
    assert "measured_frequency_min_hz = 137" in path.read_text()
    monkeypatch.setattr("aosr.physics.report_research_text.config_path", lambda name: path)


def test_cli_notice_reads_measured_lower_bound_from_temporary_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cli_report: three_lane_report.ThreeLaneReport,
) -> None:
    copy_registry(tmp_path, monkeypatch)
    inputs = report_io.load_input_document(pair(notice_scheme("porous")), CAPABILITIES, DIRECTIVITY)
    text = "\n".join(_text_sections(cli_report, inputs=inputs, with_points=False))
    assert "原論文只量到 137 Hz 以上" in text
    assert "原論文只量到 100 Hz 以上" not in text


def test_result_notice_reads_measured_lower_bound_from_temporary_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, result: SchemeResult,
) -> None:
    copy_registry(tmp_path, monkeypatch)
    changed = display_result(result, "measured-base")
    view = build_result_view(changed, quality_targets_path=config_path("quality_targets.toml"))
    assert view.material_research_notice == "小松纖維模型：原論文只量到 137 Hz 以上"


def display_result(result: SchemeResult, case: str) -> SchemeResult:
    """只替換已存結果的材料以考顯示；保留原候選代號、聲道與座位，接縫用手寫保存值。"""
    from aosr.physics.report_measured import MeasuredSeam, SeamValues
    if case == "fixed":
        return result
    scene = result.scheme.scene.model_copy(update={"surface_zones_by_wall": notice_scheme(case).scene.surface_zones_by_wall,
                                                   "impedance_pa_s_per_m_by_wall": None})
    changed = result.model_copy(update={"scheme": result.scheme.model_copy(update={"scene": scene})})
    if case == "measured-base":
        values = SeamValues(real_pa_s_m=400., imag_pa_s_m=-200., normal_absorption=.5, random_absorption=.4)
        seams = {"floor": tuple(MeasuredSeam(frequency_hz=f, measured=values, base_assembly=values) for f in (100., 1000.))}
        pairs = tuple(p.model_copy(update={"report": p.report.model_copy(update={"measured_seams_by_wall": seams})}) for p in result.pairs)
        changed = changed.model_copy(update={"pairs": pairs})
    return changed
