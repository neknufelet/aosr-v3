"""考卷手餵家具表頭與近似窗，完整走評分編排及伺服器比較畫面。"""
from __future__ import annotations

import pytest

from aosr.config.paths import config_path
from aosr.config.capabilities import load_capabilities
from aosr.config.directivity_defaults import load_directivity_defaults
from aosr.physics.report_io import ReportInput, ReportOutput
from aosr.physics.reflection_window import ReflectionWindow
from aosr.reporting import evaluation
from aosr.reporting.result import SchemeResult
from tests.engine.test_gui_compare_view import pair as pair, _view
from tests.engine.test_furniture_scoring_flags import furniture_report


def _furnished_result(result: SchemeResult, furniture_id: str,
                      monkeypatch: pytest.MonkeyPatch) -> SchemeResult:
    original = evaluation.build_pair_window

    def supplied_window(inputs: ReportInput, report: ReportOutput, window_s: float) -> ReflectionWindow:
        assert report.path_table is not None
        window = original(inputs, report, window_s)
        document = window.model_dump(mode="python")
        document.update(coverage="approximate", furniture_ids=report.path_table.furniture_ids)
        return ReflectionWindow.model_validate(document)

    changed = result.model_copy(update={"pairs": tuple(pair.model_copy(update={
        "report": furniture_report(pair.report, furniture_id)}) for pair in result.pairs)})
    targets = config_path("quality_targets.toml")
    with monkeypatch.context() as patch:
        patch.setattr(evaluation, "build_pair_window", supplied_window)
        candidate = evaluation.evaluate_parts(changed, targets,
            evaluation.read_registry_settings(targets, result.scheme.purpose),
            capabilities=load_capabilities(config_path("capabilities.toml")),
            directivity=load_directivity_defaults(config_path("directivity_defaults.toml")))
    return changed.model_copy(update={"candidate": candidate})


def test_mixed_furniture_comparison_marks_only_reflections_and_omits_total(
    pair: tuple[SchemeResult, SchemeResult], monkeypatch: pytest.MonkeyPatch,
) -> None:
    approximate = _furnished_result(pair[1], "desk", monkeypatch)
    view = _view((pair[0], approximate))
    assert not view.table.same_table
    assert view.table.a_text == view.table.b_text == "不列總代價"
    assert (view.table.verdict_text, view.table.better) == ("", "")
    assert {row.category for row in view.categories if row.comparison_text} == {"reflections_and_echo"}
    assert next(row.comparison_text for row in view.categories if row.category == "reflections_and_echo") == (
        "評分條件不同，這一類代價不能直接比")
    assert all(row.better in {"a", "b", "same"} for row in view.categories
               if row.category != "reflections_and_echo" and row.a.cost is not None and row.b.cost is not None)
    flags = {row.category: row.b.flags for row in view.categories}
    assert {category for category, values in flags.items() if "furniture_model_approximate" in values} == {
        "timbre_balance", "listening_area_stability", "channel_matching", "reflections_and_echo"}
    assert all("近似" in row.b.flags_text for row in view.categories
               if "furniture_model_approximate" in row.b.flags)


def test_same_model_different_furniture_stays_in_one_comparison_table(
    pair: tuple[SchemeResult, SchemeResult], monkeypatch: pytest.MonkeyPatch,
) -> None:
    furnished = (_furnished_result(pair[0], "desk", monkeypatch),
                 _furnished_result(pair[1], "coffee", monkeypatch))
    view = _view(furnished)
    assert view.table.same_table
    assert not any(row.comparison_text for row in view.categories)
    assert view.table.a_text != "不列總代價" and view.table.b_text != "不列總代價"
