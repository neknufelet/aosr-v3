"""使用真的家具方案驗證混比與同模型比較。"""
from __future__ import annotations

import pytest

from aosr.reporting.result import SchemeResult
from tests.engine._furniture_scheme_results import scheme_pair as scheme_pair, shared_scheme_result
from tests.engine.test_gui_compare_view import _view


def test_mixed_furniture_comparison_marks_only_reflections_and_omits_total(
    scheme_pair: tuple[SchemeResult, ...],
) -> None:
    plain, furnished = scheme_pair
    view = _view((plain, furnished))
    assert not view.table.same_table
    assert view.table.a_text == view.table.b_text == "不列總代價"
    assert (view.table.verdict_text, view.table.better) == ("", "")
    assert {row.category for row in view.categories if row.comparison_text} == {"reflections_and_echo"}
    assert next(row.comparison_text for row in view.categories if row.category == "reflections_and_echo") == (
        "評分條件不同，這一類代價不能直接比；兩者計算涵蓋範圍不同")
    assert all(row.better in {"a", "b", "same"} for row in view.categories
               if row.category != "reflections_and_echo" and row.a.cost is not None and row.b.cost is not None)
    flags = {row.category: row.b.flags for row in view.categories}
    assert {category for category, values in flags.items() if "furniture_model_approximate" in values} == {
        "timbre_balance", "listening_area_stability", "channel_matching", "reflections_and_echo"}
    assert all("近似" in row.b.flags_text for row in view.categories
               if "furniture_model_approximate" in row.b.flags)


def test_two_real_furnished_results_stay_in_one_comparison_table(
    scheme_pair: tuple[SchemeResult, ...], tmp_path_factory: pytest.TempPathFactory, worker_id: str,
) -> None:
    furnished = scheme_pair[1]
    other = shared_scheme_result(tmp_path_factory, worker_id, True, "other-seat")
    assert furnished.scheme.furniture != other.scheme.furniture
    view = _view((furnished, other))
    assert view.table.same_table
    assert not any(row.comparison_text for row in view.categories)
    assert view.table.a_text != "不列總代價" and view.table.b_text != "不列總代價"
