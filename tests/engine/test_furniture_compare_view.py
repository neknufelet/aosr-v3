"""家具比較與匯出；B1／B3 答案來自決策紙第 13 條及已拍設計。"""
from __future__ import annotations

from datetime import date

import pytest

from aosr.config.paths import config_path
from aosr.config.quality_targets import load_quality_targets
from aosr.gui import compare_view
from aosr.gui.compare_view import build_compare_view, curves_csv, scheme_differences, summary_csv
from aosr.gui.jobs import ResultStatus
from aosr.gui.result_view import build_result_view
from aosr.reporting.result import SchemeResult
from aosr.scoring.contract import QualityCategory
from tests.engine._furniture_scheme_results import scheme_pair as scheme_pair
from tests.engine.test_gui_compare_view import _renamed, _view


def test_only_furniture_changes_are_listed_and_approximation_stays_on_its_side(
    scheme_pair: tuple[SchemeResult, ...],
) -> None:
    plain, furnished = scheme_pair
    for pair, side in (((plain, furnished), "b"), ((furnished, plain), "a")):
        view = _view(pair)
        assert view.changes
        assert "兩份方案設定相同" not in view.summary_text
        assert {change.label for change in view.changes} == {"沙發（seat）"}
        assert {change.path for change in view.changes} == {"furniture.seat"}
        for row in view.categories:
            for letter in ("a", "b"):
                cell = getattr(row, letter)
                assert ("近似" in getattr(row, f"{letter}_state_text")) == (
                    letter == side and "furniture_model_approximate" in cell.flags)
        reflection = next(row for row in view.categories if row.category == "reflections_and_echo")
        assert reflection.comparison_text == "評分條件不同，這一類代價不能直接比；兩者計算涵蓋範圍不同"
        assert {row.category for row in view.categories if row.comparison_text} == {"reflections_and_echo"}
        assert "未包含家具吸音" in next(row.note_text for row in view.categories if row.category == "reverberation")
        reason = ("已含家具一次反射、遮擋與有限尺寸鏡面修正；未含家具與牆之間的多次反射、"
                  "完整繞射，以及家具吸音對整房殘響的影響")
        assert reason in view.notes
        assert view.overlay_note == "家具模型：近似"
        assert "家具模型：近似" in curves_csv(view)
        assert "近似" in summary_csv(view)


def test_empty_change_list_has_raw_scheme_difference_guard(
    scheme_pair: tuple[SchemeResult, ...], monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(compare_view, "_collect_scheme", lambda *args: None)
    changes = scheme_differences(scheme_pair[0].scheme, scheme_pair[1].scheme)
    assert {row.label for row in changes} == {"其他設定（未逐項列出）"}
    assert not scheme_differences(scheme_pair[0].scheme, _renamed(scheme_pair[0], "renamed").scheme)


def test_furniture_id_with_dots_is_still_listed_as_furniture(scheme_pair: tuple[SchemeResult, ...]) -> None:
    plain, furnished = scheme_pair
    items = furnished.scheme.furniture
    assert items
    changed = furnished.scheme.model_copy(update={
        "furniture": tuple(item.model_copy(update={"furniture_id": "seat.part"}) for item in items)})
    changes = scheme_differences(plain.scheme, changed)
    assert {row.label for row in changes} == {"沙發（seat.part）"}
    assert {row.path for row in changes} == {"furniture.seat%2Epart"}


def test_two_furnished_results_have_separate_ranking_notice(scheme_pair: tuple[SchemeResult, ...]) -> None:
    furnished = scheme_pair[1]
    view = _view((furnished, _renamed(furnished, "second-furnished")))
    assert view.table.same_table and view.table.verdict_text
    assert "家具模型：近似" in view.ranking_approximation_text
    assert "家具模型" not in view.table.verdict_text
    assert "兩者計算涵蓋範圍不同" not in view.table.reason_text
    assert all("兩者計算涵蓋範圍不同" not in row.comparison_text for row in view.categories)


def test_other_reflection_identity_difference_does_not_add_furniture_reason(
    scheme_pair: tuple[SchemeResult, ...],
) -> None:
    furnished = scheme_pair[1]
    other = _renamed(furnished, "other-settings")
    evaluations = tuple(item.model_copy(update={"settings_fingerprint": "other-reflection-settings"})
                        if item.category is QualityCategory.REFLECTIONS_AND_ECHO else item
                        for item in other.candidate.evaluations)
    other = other.model_copy(update={"candidate": other.candidate.model_copy(update={"evaluations": evaluations})})
    view = _view((furnished, other))
    reflection = next(row for row in view.categories if row.category == "reflections_and_echo")
    assert reflection.comparison_text == "評分條件不同，這一類代價不能直接比"
    assert all("兩者計算涵蓋範圍不同" not in row.note_text for row in view.categories)
    assert not view.ranking_approximation_text


def test_failed_furnished_run_has_no_ranking_approximation_notice(scheme_pair: tuple[SchemeResult, ...]) -> None:
    furnished = scheme_pair[1]
    other = _renamed(furnished, "failed-furnished")
    targets = config_path("quality_targets.toml")
    view = build_compare_view(
        a_run_id="a" * 32, a=furnished, view_a=build_result_view(furnished, quality_targets_path=targets),
        b_run_id="b" * 32, b=other, view_b=build_result_view(other, quality_targets_path=targets),
        quality_targets=load_quality_targets(targets), run_date=date(2026, 10, 8),
        b_status=ResultStatus("failed", 1))
    assert "不下哪一份比較好的結論" in view.table.verdict_text
    assert not view.table.better
    assert not view.ranking_approximation_text
