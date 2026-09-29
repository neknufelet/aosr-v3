"""比較資料的差異、曲線與同表契約。"""
from __future__ import annotations

import re
from datetime import date

import pytest

from aosr.config.paths import config_path
from aosr.config.quality_targets import load_quality_targets
from aosr.geometry.shoebox import Point
from aosr.gui.app import STATIC
from aosr.gui.compare_view import CompareView, _table, build_compare_view, scheme_differences
from aosr.reporting.compare import compare_results, comparison_problems
from aosr.reporting.result import SchemeResult
from aosr.reporting.result_view import FrequencyPoint, FrequencyResponse, build_result_view
from aosr.reporting.scheme import Scheme
from aosr.scoring.contract import QualityCategory
from tests.engine.test_scheme_pipeline import (
    _scheme, shared_control_result, shared_control_scheme_result)


@pytest.fixture(scope="module")
def pair(tmp_path_factory: pytest.TempPathFactory, worker_id: str) -> tuple[SchemeResult, SchemeResult]:
    return (shared_control_result(tmp_path_factory, worker_id, "wall-1"),
            shared_control_result(tmp_path_factory, worker_id, "wall-2"))


@pytest.fixture(scope="module")
def moved(tmp_path_factory: pytest.TempPathFactory, worker_id: str) -> SchemeResult:
    # 只把主位抬高 0.1 公尺、周圍點沒跟著搬：座位相對佈局變了，跟 wall-1 不同表。老闆最常這樣改。
    scheme = _scheme("wall-2")
    points = tuple(point.model_copy(update={"position_m": (*point.position_m[:2], point.position_m[2] + 0.1)})
                   if point.role.value == "primary" else point for point in scheme.receiver_set.points)
    changed = scheme.model_copy(update={"receiver_set": scheme.receiver_set.model_copy(
        update={"points": points})})
    return shared_control_scheme_result(tmp_path_factory, worker_id, "wall-2-primary-up",
                                        Scheme.model_validate(changed.model_dump(mode="json")))


def _renamed(result: SchemeResult, scheme_id: str) -> SchemeResult:
    return result.model_copy(update={
        "scheme": result.scheme.model_copy(update={"scheme_id": scheme_id}),
        "candidate": result.candidate.model_copy(update={"candidate_id": scheme_id})})


def _view(pair: tuple[SchemeResult, SchemeResult]) -> CompareView:
    a, b = pair
    targets = config_path("quality_targets.toml")
    return build_compare_view(
        a_run_id="a" * 32, a=a, view_a=build_result_view(a, quality_targets_path=targets),
        b_run_id="b" * 32, b=b, view_b=build_result_view(b, quality_targets_path=targets),
        quality_targets=load_quality_targets(targets), run_date=date(2026, 9, 27))


def test_pairs_pair_primary_by_role_and_others_by_seat_id(
        pair: tuple[SchemeResult, SchemeResult]) -> None:
    a, b = pair
    base = build_result_view(a, quality_targets_path=config_path("quality_targets.toml"))
    point = FrequencyPoint(frequency_hz=100, level_db=3, frequency_text="100 Hz", level_text="3 dB")
    def response(role: str, seat: str, seat_role: str, seat_label: str) -> FrequencyResponse:
        return FrequencyResponse(role=role, speaker_id=role, receiver_id=seat,
                                 receiver_role=seat_role, receiver_label=seat_label,
                                 points=(point,))
    # 座位標籤照結果頁真的給法（角色：主位／周圍點／其他座位）；按鈕上的方向要從方案的座位方向來。
    front = next(point for point in a.scheme.receiver_set.points
                 if point.direction_relative_to_primary == "front").receiver_id
    # 右聲道主位排第一：預設那一對（左聲道主位）要被排到按鈕第一個，不是照原順序。
    a_rows = (response("right", "a-main", "primary", "主位"),
              response("left", "a-main", "primary", "主位"),
              response("left", front, "other_seat", "其他座位"),
              response("left", "extra", "surrounding", "周圍點"),
              response("right", "trap", "surrounding", "周圍點"))
    b_rows = (response("left", "b-main", "primary", "主位"),
              response("right", "b-main", "primary", "主位"),
              response("left", front, "surrounding", "周圍點"),
              response("left", "trap", "surrounding", "周圍點"))
    result = build_compare_view(
        a_run_id="a" * 32, a=a, view_a=base.model_copy(update={"frequency_responses": a_rows}),
        b_run_id="b" * 32, b=b, view_b=base.model_copy(update={"frequency_responses": b_rows}),
        quality_targets=load_quality_targets(config_path("quality_targets.toml")),
        run_date=date(2026, 9, 27))
    overlay = result.overlay
    assert {(item.a_key, item.b_key) for item in overlay.pairs} == {
        ("a:left:a-main", "b:left:b-main"), ("a:right:a-main", "b:right:b-main"),
        (f"a:left:{front}", f"b:left:{front}")}
    assert overlay.default_keys == ("a:left:a-main", "b:left:b-main")
    assert (overlay.pairs[0].a_key, overlay.pairs[0].b_key) == overlay.default_keys
    assert {item.label for item in overlay.pairs} == {
        "左聲道・主位", "右聲道・主位", f"左聲道・{front}（主位前方）"}


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
    seat = next(row for row in rows if row.path.startswith("receiver_set.points."))
    assert (seat.a_text, seat.b_text) == ("只有 A 有", "無")
    reverse = next(row for row in scheme_differences(changed, scheme)
                   if row.path.startswith("receiver_set.points."))
    assert (reverse.a_text, reverse.b_text) == ("無", "只有 B 有")


def test_codes_are_named_in_chinese_by_their_own_table(pair: tuple[SchemeResult, SchemeResult]) -> None:
    # 座位方向走方向表（主位左方），不走結果頁的聲道表（左聲道）；聲源模型照輸入頁選單的字。
    scheme = pair[0].scheme
    front = next(point for point in scheme.receiver_set.points
                 if point.direction_relative_to_primary == "front")
    points = tuple(point.model_copy(update={"direction_relative_to_primary": "left"})
                   if point is front else point for point in scheme.receiver_set.points)
    changed = scheme.model_copy(update={
        "receiver_set": scheme.receiver_set.model_copy(update={"points": points}),
        "source_model": "product_default"})
    rows = {row.path: row for row in scheme_differences(scheme, changed)}
    direction = rows[f"receiver_set.points.{front.receiver_id}.direction"]
    assert (direction.a_text, direction.b_text) == ("主位前方", "主位左方")
    page = (STATIC / "index.html").read_text(encoding="utf-8")
    options = dict(re.findall(r'<option value="(\w+)">([^<]+)</option>', page))
    source = rows["source_model"]
    assert (source.a_text, source.b_text) == (options[scheme.source_model], options["product_default"])


def test_wall_names_match_input_page(pair: tuple[SchemeResult, SchemeResult]) -> None:
    # 跟輸入頁同一套牆名；x、y 起點終點沒有前後左右的定義，翻成前牆後牆會讓人看錯是哪一面。
    scheme = pair[0].scheme
    walls = scheme.scene.impedance_pa_s_per_m_by_wall
    scene = scheme.scene.model_copy(update={"impedance_pa_s_per_m_by_wall": {
        wall: value * 10 for wall, value in walls.items()}})
    rows = scheme_differences(scheme, scheme.model_copy(update={"scene": scene}))
    script = (STATIC / "app.js").read_text(encoding="utf-8")
    names = dict(re.findall(r'(\w+): "([^"]+)"', script[script.index("wallNames"):
                                                        script.index("};", script.index("wallNames"))]))
    assert {row.path: row.label for row in rows} == {
        f"scene.impedance_pa_s_per_m_by_wall.{wall}": f"{names[wall]}阻抗" for wall in walls}
    assert all("帕·秒／公尺" in row.a_text for row in rows)
    # 一萬以上也照一般寫法：1.04e+04 會被讀成 1.04。
    assert not [row for row in rows if "e+" in row.a_text + row.b_text]
    assert any(float(row.b_text.split()[0]) >= 10000 for row in rows)


def test_change_text_adds_digits_until_sides_differ(pair: tuple[SchemeResult, SchemeResult]) -> None:
    a = pair[0].scheme
    b = a.model_copy(update={"scene": a.scene.model_copy(update={"density_kg_m3": 1.30001})})
    a = a.model_copy(update={"scene": a.scene.model_copy(update={"density_kg_m3": 1.3})})
    row = next(row for row in scheme_differences(a, b) if row.path == "scene.density_kg_m3")
    assert row.a_text != row.b_text
    assert "1.30001" in row.b_text
    # 只差浮點尾巴時註明微小差異，不印 0.30000000000000004。
    tiny = b.model_copy(update={"scene": b.scene.model_copy(update={"density_kg_m3": 0.1 + 0.2})})
    base = a.model_copy(update={"scene": a.scene.model_copy(update={"density_kg_m3": 0.3})})
    tail = next(row for row in scheme_differences(base, tiny) if row.path == "scene.density_kg_m3")
    assert (tail.a_text, tail.b_text) == ("0.3 公斤／立方公尺", "0.3 公斤／立方公尺（微小差異）")


def test_default_pair_uses_channel_role_not_speaker_id(pair: tuple[SchemeResult, SchemeResult]) -> None:
    a, b = pair
    base = build_result_view(a, quality_targets_path=config_path("quality_targets.toml"))
    point = FrequencyPoint(frequency_hz=100, level_db=3, frequency_text="100 Hz", level_text="3 dB")
    left = FrequencyResponse(role="left", speaker_id="spk-a", receiver_id="seat-1",
                             receiver_role="primary", receiver_label="主位", points=(point,))
    trap = left.model_copy(update={"role": "right", "speaker_id": "left"})
    # 左聲道但不是主位、而且排第一：只看聲道不看主位的挑法會挑到它。
    beside = left.model_copy(update={"receiver_id": "seat-2", "receiver_role": "surrounding"})
    altered = base.model_copy(update={"frequency_responses": (beside, trap, left)})
    other = altered.model_copy(update={"frequency_responses": (beside, left, trap)})
    view = build_compare_view(a_run_id="a" * 32, a=a, view_a=altered,
                              b_run_id="b" * 32, b=b, view_b=other,
                              quality_targets=load_quality_targets(config_path("quality_targets.toml")),
                              run_date=date(2026, 9, 27))
    selected = [next(row for row in view.overlay.series if row.key == key)
                for key in view.overlay.default_keys]
    assert {(row.role, row.speaker_id, row.receiver_id) for row in selected} == {
        ("left", "spk-a", "seat-1")}
    # 圖例是疊圖上唯一分得出 A、B 的字；線的代號也不准重複。
    assert all(row.legend_text.startswith(f"{row.side.upper()}・") for row in view.overlay.series)
    keys = [row.key for row in view.overlay.series]
    assert sorted(keys) == sorted(set(keys))


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


def test_split_tables_never_print_rank_and_name_the_side(
        pair: tuple[SchemeResult, SchemeResult], moved: SchemeResult) -> None:
    # 不同表時哪一張算主表只看比較身分排序；落在主表那份的「名次 1」只是一個人的名次。
    for a, b in ((pair[0], moved), (moved, pair[0])):
        view = _view((a, b))
        table = view.table
        assert not table.same_table
        assert "名次" not in table.a_text + table.b_text
        assert "總代價" not in table.a_text + table.b_text
        assert "比較身分不同" in table.reason_text
        assert "同一類但身分不同：" in table.reason_text
        assert table.reason_text.split(" 與 ")[0] in {"A", "B"}
        assert not re.search(r"[a-z]+_[a-z_]+", table.reason_text)
        assert "不可同表" in view.summary_text
        checks = {check.label: check.same for check in view.fingerprints}
        assert checks == {"座位組": False, "座位相對佈局": False, "聲道組": True}


def test_equal_totals_print_no_rank(pair: tuple[SchemeResult, SchemeResult]) -> None:
    # 同內容換名字（另存新名字常見）：總代價一樣時名次只照代號排，不代表好壞。
    view = _view((pair[0], _renamed(pair[0], "wall-1-copy")))
    assert view.table.same_table
    assert all("總代價相同" in text and "名次" not in text and "微小差異" not in text
               for text in (view.table.a_text, view.table.b_text))
    assert view.changes == ()


def _without(result: SchemeResult, category: QualityCategory) -> SchemeResult:
    return result.model_copy(update={"candidate": result.candidate.model_copy(update={
        "evaluations": tuple(item for item in result.candidate.evaluations
                             if item.category is not category)})})


def test_unranked_sides_say_which_or_neither(pair: tuple[SchemeResult, SchemeResult]) -> None:
    # 少一類評估就排不上（未評估）：寫清楚是只有一份排得上，還是兩份都排不上，也都不印名次。
    targets = config_path("quality_targets.toml")
    missing = QualityCategory.TIMBRE_BALANCE
    only_b = _table(_without(pair[0], missing), pair[1], load_quality_targets(targets),
                    date(2026, 9, 27))
    neither = _table(_without(pair[0], missing), _without(pair[1], missing),
                     load_quality_targets(targets), date(2026, 9, 27))
    assert "只有 B 排得上" in only_b.reason_text
    assert "兩份都排不上" in neither.reason_text
    assert all("名次" not in table.a_text + table.b_text for table in (only_b, neither))


def test_ranked_sides_print_rank_total_and_chinese_status(pair: tuple[SchemeResult, SchemeResult]) -> None:
    view = _view(pair)
    assert view.table.same_table
    assert all(text.startswith("可排名；名次 ") and "總代價 " in text
               for text in (view.table.a_text, view.table.b_text))
    assert view.table.a_text.split("總代價 ")[1] != view.table.b_text.split("總代價 ")[1]


def test_identity_and_summary_texts_follow_their_side(pair: tuple[SchemeResult, SchemeResult]) -> None:
    other = pair[1].model_copy(update={"engine_commit": "e53bfae" + "f" * 33})
    view = _view((pair[0], other))
    assert (view.a.engine_text, view.b.engine_text) == (pair[0].engine_commit[:7], "e53bfae")
    assert (view.a.scheme_id, view.b.scheme_id) == ("wall-1", "wall-2")
    assert not view.table.same_table
    assert "引擎版本（engine_commit）不同" in view.table.reason_text
    extra = f"另 {len(view.changes) - 5} 處"
    assert (extra in view.summary_text) == (len(view.changes) > 5)
    assert "等 " not in view.summary_text


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
