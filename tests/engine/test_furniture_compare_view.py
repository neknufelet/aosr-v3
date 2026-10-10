"""家具比較與匯出；B1／B3 答案來自決策紙第 13 條及已拍設計。"""
from __future__ import annotations

import json
from datetime import date

import pytest

from aosr.config.paths import config_path
from aosr.config.quality_targets import load_quality_targets
from aosr.gui import compare_view
from aosr.gui.compare_view import build_compare_view, curves_csv, scheme_differences, summary_csv
from aosr.gui.jobs import ResultStatus
from aosr.gui.result_view import build_result_view
from aosr.reporting.display import reflection_models_differ
from aosr.reporting.result import SchemeResult
from aosr.reporting.scheme import Scheme
from aosr.scoring.contract import EvaluationState, QualityCategory, ReasonCode
from aosr.scoring.reflections_cost import comparison_support
from tests.engine._furniture_scheme_results import scheme_pair as scheme_pair
from tests.engine import _furniture_cases as schemes
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
        expected_note = f"{side.upper()}：家具模型：近似"
        assert view.overlay_note == expected_note
        assert curves_csv(view).splitlines()[-1] == expected_note
        notes = summary_csv(view).splitlines()
        assert expected_note in notes[notes.index("說明") + 1:]


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
    assert view.overlay_note == "家具模型：近似"
    assert curves_csv(view).splitlines()[-1] == "家具模型：近似"
    notes = summary_csv(view).splitlines()
    assert "家具模型：近似" in notes[notes.index("說明") + 1:]
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


@pytest.mark.parametrize("reverse", [False, True])
def test_unavailable_reflection_does_not_claim_different_model_coverage(
    scheme_pair: tuple[SchemeResult, ...], reverse: bool,
) -> None:
    furnished = scheme_pair[1]
    other = _renamed(furnished, "unavailable-reflection")
    evaluations = tuple(item.model_copy(update={"payload": None, "flags": (),
                        "state": EvaluationState.UNAVAILABLE, "raw_quantities": (),
                        "category_cost": None, "reason_codes": (ReasonCode.SOLVER_UNAVAILABLE,)})
                        if item.category is QualityCategory.REFLECTIONS_AND_ECHO else item
                        for item in other.candidate.evaluations)
    other = other.model_copy(update={"candidate": other.candidate.model_copy(update={"evaluations": evaluations})})
    view = _view((other, furnished) if reverse else (furnished, other))
    row = next(row for row in view.categories if row.category == "reflections_and_echo")
    assert "兩者計算涵蓋範圍不同" not in row.comparison_text


@pytest.mark.parametrize("reverse", [False, True])
def test_same_id_changing_placement_shape_lists_all_relative_and_room_fields(reverse: bool) -> None:
    # 列名原文來自決策紙第 6、7 條；兩種角度雖同存 yaw_deg，基準各自不同。
    a = Scheme.model_validate(schemes.document(schemes.relative_item()))
    b = Scheme.model_validate(schemes.document(schemes.cloud_item(
        furniture_id="seat", placement={"bottom_center_m": [2.0, 3.0, 2.5], "yaw_deg": 0})))
    changes = scheme_differences(b, a) if reverse else scheme_differences(a, b)
    by_label = {row.label.rsplit("） ", 1)[-1]: row for row in changes}
    expected = {"前方": ("1 公尺", "未設定"), "左方": ("0.5 公尺", "未設定"),
                "底面離地": ("0 公尺", "未設定"), "相對角": ("0 度", "未設定"),
                "底面中心的房間座標": ("未設定", "2、3、2.5"), "房間角度": ("未設定", "0 度")}
    for label, texts in expected.items():
        assert label in by_label
        row = by_label[label]
        assert (row.a_text, row.b_text) == (tuple(reversed(texts)) if reverse else texts)


@pytest.mark.parametrize("reverse", [False, True])
def test_added_and_removed_furniture_has_exact_a_and_b_text(reverse: bool) -> None:
    plain = Scheme.model_validate(schemes.document())
    furnished = Scheme.model_validate(schemes.document(schemes.relative_item()))
    changes = scheme_differences(furnished, plain) if reverse else scheme_differences(plain, furnished)
    row = next(row for row in changes if row.path == "furniture.seat")
    assert row.label == "沙發（seat）"
    assert (row.a_text, row.b_text) == (("只有 A 有", "無") if reverse else ("無", "只有 B 有"))


@pytest.mark.parametrize(("field", "value", "label", "cloud"), [
    ("forward_m", 1.2, "前方", False), ("left_m", 0.6, "左方", False),
    ("bottom_height_m", 0.1, "底面離地", False), ("yaw_deg", 90, "相對角", False),
    ("bottom_center_m", [2.1, 2.0, 2.5], "底面中心的房間座標", True), ("yaw_deg", 0, "房間角度", True),
])
def test_placement_field_labels_match_decision_sections_six_and_seven(
    field: str, value: object, label: str, cloud: bool,
) -> None:
    item = schemes.cloud_item() if cloud else schemes.relative_item()
    a = Scheme.model_validate(schemes.document(item))
    placement = item["placement"]
    assert isinstance(placement, dict)
    b = Scheme.model_validate(schemes.document(item | {"placement": placement | {field: value}}))
    changes = scheme_differences(a, b)
    identifier, name = ("cloud", "天雲（cloud）") if cloud else ("seat", "沙發（seat）")
    row = next(row for row in changes if row.path == f"furniture.{identifier}.placement.{field}")
    assert row.label == f"{name} {label}"


def test_b1_stays_only_on_reflection_when_listening_identity_also_differs(
    scheme_pair: tuple[SchemeResult, ...],
) -> None:
    plain, furnished = scheme_pair
    evaluations = tuple(item.model_copy(update={"settings_fingerprint": "different-listening-settings"})
                        if item.category is QualityCategory.LISTENING_AREA_STABILITY else item
                        for item in plain.candidate.evaluations)
    changed = plain.model_copy(update={"candidate": plain.candidate.model_copy(update={"evaluations": evaluations})})
    view = _view((changed, furnished))
    rows = {row.category: row for row in view.categories}
    assert "評分條件不同" in rows["listening_area_stability"].comparison_text
    assert {row.category for row in view.categories if "兩者計算涵蓋範圍不同" in row.comparison_text} == {
        "reflections_and_echo"}


def test_b1_judges_only_the_furniture_model_not_other_support_fields(scheme_pair: tuple[SchemeResult, ...]) -> None:
    # 第 79 行：B1 理由只給有家具對沒家具；兩份都有家具而比較支撐因別的欄位不同（例如主位代號）時不給。
    plain, furnished = (comparison_support(next(item for item in result.candidate.evaluations
                                                 if item.category is QualityCategory.REFLECTIONS_AND_ECHO))
                        for result in scheme_pair)
    renamed = json.dumps(json.loads(furnished) | {"primary_receiver_id": "listener"}, sort_keys=True, separators=(",", ":"))
    assert renamed != furnished
    assert reflection_models_differ(plain, furnished) and reflection_models_differ(furnished, plain)
    assert not reflection_models_differ(furnished, renamed)
    assert not reflection_models_differ(furnished, "") and not reflection_models_differ("", plain)



@pytest.mark.parametrize("reverse", [False, True])
def test_same_id_changing_kind_names_both_kinds(reverse: bool) -> None:
    # #559 老闆試用：同一個代號 table，A 是茶几、B 是書桌；列名原本只照 B 寫「書桌（table）」。
    coffee = schemes.relative_item(furniture_id="table", kind="coffee_table", material="wood",
        width_m=1.2, depth_m=0.6, height_m=0.03,
        placement={"forward_m": 1.15, "left_m": 0.0, "bottom_height_m": 0.47, "yaw_deg": 0})
    desk = schemes.relative_item(furniture_id="table", kind="desk", material="wood",
        width_m=1.4, depth_m=0.75, height_m=0.03,
        placement={"forward_m": 0.825, "left_m": 0.0, "bottom_height_m": 0.72, "yaw_deg": 0})
    a = Scheme.model_validate(schemes.document(coffee))
    b = Scheme.model_validate(schemes.document(desk))
    changes = scheme_differences(b, a) if reverse else scheme_differences(a, b)
    name = "書桌／茶几（table）" if reverse else "茶几／書桌（table）"
    labels = {row.label for row in changes}
    assert f"{name} 種類" in labels and f"{name} 寬" in labels
    assert not [label for label in labels if label.startswith(("茶几（table）", "書桌（table）"))]
