"""比較資料的差異、曲線與同表契約。"""
from __future__ import annotations

import re
from datetime import date

import pytest

from aosr.config.paths import config_path
from aosr.config.quality_targets import load_quality_targets
from aosr.geometry.shoebox import Point
from aosr.gui.app import STATIC
from aosr.gui.compare_view import CompareView, build_compare_view, scheme_differences
from aosr.reporting.compare import compare_results, comparison_problems
from aosr.reporting.result import SchemeResult
from aosr.reporting.result_view import FrequencyPoint, FrequencyResponse, build_result_view
from aosr.scoring.contract import QualityCategory
from tests.engine.test_scheme_pipeline import shared_control_result


@pytest.fixture(scope="module")
def pair(tmp_path_factory: pytest.TempPathFactory, worker_id: str) -> tuple[SchemeResult, SchemeResult]:
    return (shared_control_result(tmp_path_factory, worker_id, "wall-1"),
            shared_control_result(tmp_path_factory, worker_id, "wall-2"))


def _view(pair: tuple[SchemeResult, SchemeResult]) -> CompareView:
    a, b = pair
    targets = config_path("quality_targets.toml")
    return build_compare_view(
        a_run_id="a" * 32, a=a, view_a=build_result_view(a, quality_targets_path=targets),
        b_run_id="b" * 32, b=b, view_b=build_result_view(b, quality_targets_path=targets),
        quality_targets=load_quality_targets(targets), run_date=date(2026, 9, 27))


def test_changes_name_each_changed_field(pair: tuple[SchemeResult, SchemeResult]) -> None:
    scheme = pair[0].scheme
    left = next(iter(scheme.speakers))
    original = scheme.speakers[left]
    speaker = Point(original.x + 0.1, original.y, original.z)
    speakers = {**scheme.speakers, left: speaker}
    wall = next(iter(scheme.scene.impedance_pa_s_per_m_by_wall))
    impedances = {**scheme.scene.impedance_pa_s_per_m_by_wall,
                  wall: scheme.scene.impedance_pa_s_per_m_by_wall[wall] + 1}
    scene = scheme.scene.model_copy(update={"impedance_pa_s_per_m_by_wall": impedances})
    receiver_set = scheme.receiver_set.model_copy(update={
        "points": scheme.receiver_set.points[:-1]})
    changed = scheme.model_copy(update={"speakers": speakers, "scene": scene,
                                        "receiver_set": receiver_set})
    rows = scheme_differences(scheme, changed)
    assert {row.path for row in rows} == {
        f"speakers.{left}.x", f"scene.impedance_pa_s_per_m_by_wall.{wall}",
        f"receiver_set.points.{scheme.receiver_set.points[-1].receiver_id}"}
    assert all(isinstance(row.a_text, str) and isinstance(row.b_text, str)
               and row.a_text != row.b_text for row in rows)


def test_wall_names_match_input_page(pair: tuple[SchemeResult, SchemeResult]) -> None:
    # 跟輸入頁同一套牆名；x、y 起點終點沒有前後左右的定義，翻成前牆後牆會讓人看錯是哪一面。
    scheme = pair[0].scheme
    walls = scheme.scene.impedance_pa_s_per_m_by_wall
    scene = scheme.scene.model_copy(update={"impedance_pa_s_per_m_by_wall": {
        wall: value + 1 for wall, value in walls.items()}})
    rows = scheme_differences(scheme, scheme.model_copy(update={"scene": scene}))
    script = (STATIC / "app.js").read_text(encoding="utf-8")
    names = dict(re.findall(r'(\w+): "([^"]+)"', script[script.index("wallNames"):
                                                        script.index("};", script.index("wallNames"))]))
    assert {row.path: row.label for row in rows} == {
        f"scene.impedance_pa_s_per_m_by_wall.{wall}": f"{names[wall]}阻抗" for wall in walls}
    assert all("帕·秒／公尺" in row.a_text for row in rows)


def test_change_text_adds_digits_until_sides_differ(pair: tuple[SchemeResult, SchemeResult]) -> None:
    a = pair[0].scheme
    b = a.model_copy(update={"scene": a.scene.model_copy(update={"density_kg_m3": 1.30001})})
    a = a.model_copy(update={"scene": a.scene.model_copy(update={"density_kg_m3": 1.3})})
    row = next(row for row in scheme_differences(a, b) if row.path == "scene.density_kg_m3")
    assert row.a_text != row.b_text
    assert "1.30001" in row.b_text


def test_default_pair_uses_channel_role_not_speaker_id(pair: tuple[SchemeResult, SchemeResult]) -> None:
    a, b = pair
    base = build_result_view(a, quality_targets_path=config_path("quality_targets.toml"))
    point = FrequencyPoint(frequency_hz=100, level_db=3, frequency_text="100 Hz", level_text="3 dB")
    left = FrequencyResponse(role="left", speaker_id="spk-a", receiver_id="seat-1",
                             receiver_role="primary", receiver_label="主位", points=(point,))
    trap = left.model_copy(update={"role": "right", "speaker_id": "left"})
    altered = base.model_copy(update={"frequency_responses": (trap, left)})
    other = altered.model_copy(update={"frequency_responses": (left, trap)})
    view = build_compare_view(a_run_id="a" * 32, a=a, view_a=altered,
                              b_run_id="b" * 32, b=b, view_b=other,
                              quality_targets=load_quality_targets(config_path("quality_targets.toml")),
                              run_date=date(2026, 9, 27))
    selected = [next(row for row in view.overlay.series if row.key == key)
                for key in view.overlay.default_keys]
    assert {(row.role, row.speaker_id, row.receiver_id) for row in selected} == {
        ("left", "spk-a", "seat-1")}


def test_overlay_keeps_levels_on_union_axis(pair: tuple[SchemeResult, SchemeResult]) -> None:
    a, b = pair
    base = build_result_view(a, quality_targets_path=config_path("quality_targets.toml"))
    def response(axis: tuple[tuple[float, float], ...]) -> FrequencyResponse:
        return FrequencyResponse(role="left", speaker_id="spk-a", receiver_id="seat-1",
                                 receiver_role="primary", receiver_label="主位",
                                 points=tuple(FrequencyPoint(frequency_hz=f, level_db=v,
                                                              frequency_text=str(f), level_text=str(v))
                                              for f, v in axis))
    view = build_compare_view(a_run_id="a" * 32, a=a,
                              view_a=base.model_copy(update={"frequency_responses": (response(((100, 1), (300, 3))),)}),
                              b_run_id="b" * 32, b=b,
                              view_b=base.model_copy(update={"frequency_responses": (response(((200, 8), (300, 9))),)}),
                              quality_targets=load_quality_targets(config_path("quality_targets.toml")),
                              run_date=date(2026, 9, 27))
    assert view.overlay.frequency_hz == (100, 200, 300)
    assert {row.side: row.levels_db for row in view.overlay.series} == {
        "a": (1, None, 3), "b": (None, 8, 9)}


def test_categories_pair_in_quality_category_order_and_match_joint_ranking(
        pair: tuple[SchemeResult, SchemeResult]) -> None:
    view = _view(pair)
    assert tuple(row.category for row in view.categories) == tuple(item.value for item in QualityCategory)
    assert all(row.a.category == row.b.category == row.category for row in view.categories)
    assert view.table.same_table
    ranking = compare_results(pair, quality_targets=load_quality_targets(
        config_path("quality_targets.toml")), run_date=date(2026, 9, 27))
    costs = {row.candidate_id: {line.identity.category.value: line.category_cost
                                for line in row.categories} for row in ranking.rankable}
    assert {row.category: row.a.cost for row in view.categories if row.a.cost is not None} == costs["wall-1"]
    assert {row.category: row.b.cost for row in view.categories if row.b.cost is not None} == costs["wall-2"]


def test_comparison_problems_lists_every_mismatch(pair: tuple[SchemeResult, SchemeResult]) -> None:
    first = pair[0]
    altered = first.model_copy(update={"engine_commit": "e53bfae" + "f" * 33})
    problems = comparison_problems((first, altered))
    assert any("候選代號重複" in row for row in problems)
    assert any("engine_commit" in row and first.engine_commit[:7] in row and "e53bfae" in row
               for row in problems)


def test_low_frequency_decay_is_not_evaluated_in_notes_and_summary(
        pair: tuple[SchemeResult, SchemeResult]) -> None:
    view = _view(pair)
    assert "低頻拖尾：尚未評估" in view.notes
    assert "低頻拖尾：尚未評估" in view.summary_text


def test_summary_never_says_better_or_worse(pair: tuple[SchemeResult, SchemeResult]) -> None:
    view = _view(pair)
    text = " ".join((view.summary_text, view.table.a_text, view.table.b_text,
                     view.table.reason_text, view.table.calibration_text))
    assert all(word not in text for word in ("較好", "較差", "更好"))
