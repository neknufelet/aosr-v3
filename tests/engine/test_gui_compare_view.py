"""比較資料的差異、曲線與同表契約。"""
from __future__ import annotations

import csv
import io
import math
import re
from datetime import date

import pytest

from aosr.config.paths import config_path
from aosr.config.quality_targets import load_quality_targets
from aosr.geometry.shoebox import Point, Room
from aosr.gui.app import STATIC
from aosr.gui.compare_view import (
    CALIBRATION_TEXTS, COMPARABLE_TEXT, NO_TOTAL_TEXT, CompareView, _changed_keys, _table,
    build_compare_view, curves_csv, scheme_differences)
from aosr.gui.labels import DIRECTIONS, LISTENING_POINTS, SPEAKERS
from aosr.reporting.calculation_fingerprint import short_fingerprint
from aosr.reporting.compare import compare_results, comparison_problems
from aosr.reporting.result import SchemeResult
from aosr.gui.result_view import LABELS, FrequencyPoint, FrequencyResponse, build_result_view
from aosr.reporting.scheme import Scheme
from aosr.scoring.contract import QualityCategory
from tests.engine.test_scheme_pipeline import (
    _scheme, shared_control_result, shared_control_scheme_result)


@pytest.fixture(scope="module")
def pair(tmp_path_factory: pytest.TempPathFactory, worker_id: str) -> tuple[SchemeResult, SchemeResult]:
    return (shared_control_result(tmp_path_factory, worker_id, "wall-1"),
            shared_control_result(tmp_path_factory, worker_id, "wall-2"))


def moved_primary_result(tmp_path_factory: pytest.TempPathFactory, worker_id: str) -> SchemeResult:
    """只把主位抬高 0.1 公尺、周圍點沒跟著搬：座位相對佈局變了，跟 wall-1 不同表。老闆最常這樣改。"""
    scheme = _scheme("wall-2")
    points = tuple(point.model_copy(update={"position_m": (*point.position_m[:2], point.position_m[2] + 0.1)})
                   if point.role.value == "primary" else point for point in scheme.receiver_set.points)
    changed = scheme.model_copy(update={"receiver_set": scheme.receiver_set.model_copy(
        update={"points": points})})
    return shared_control_scheme_result(tmp_path_factory, worker_id, "wall-2-primary-up",
                                        Scheme.model_validate(changed.model_dump(mode="json")))


def shorter_room_result(tmp_path_factory: pytest.TempPathFactory, worker_id: str) -> SchemeResult:
    """wall-2 的房間三邊各短 10%：比較頁共用比例要取較大那間，兩間一樣大時考不出來。"""
    scheme = _scheme("wall-2")
    room = scheme.scene.room_m
    shorter = scheme.model_copy(update={"scene": scheme.scene.model_copy(update={
        "room_m": Room(room.Lx * 0.9, room.Ly * 0.9, room.Lz * 0.9)})})
    return shared_control_scheme_result(tmp_path_factory, worker_id, "wall-2-shorter",
                                        Scheme.model_validate(shorter.model_dump(mode="json")))


@pytest.fixture(scope="module")
def moved(tmp_path_factory: pytest.TempPathFactory, worker_id: str) -> SchemeResult:
    return moved_primary_result(tmp_path_factory, worker_id)


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
    # 按鈕上寫顯示名，不寫座位代號（front）。
    assert {item.label for item in overlay.pairs} == {
        "左聲道・主位", "右聲道・主位", "左聲道・主位前方"}
    # 聲道切換加位置按鈕：每一對都在兩張表的交叉上，表上每一格也都有用到；右聲道在前方沒有配對（那一格要不能按）。
    assert {(item.channel, item.position) for item in overlay.pairs} == {
        ("left", "primary"), ("right", "primary"), ("left", f"seat:{front}")}
    assert {(item.key, item.label) for item in overlay.channels} == {("left", "左聲道"), ("right", "右聲道")}
    assert [item.key for item in overlay.positions][0] == "primary"
    assert {(item.key, item.label) for item in overlay.positions} == {
        ("primary", "主位"), (f"seat:{front}", "主位前方")}
    assert ("right", f"seat:{front}") not in {(item.channel, item.position) for item in overlay.pairs}


def test_channel_switch_and_positions_reach_every_pair(pair: tuple[SchemeResult, SchemeResult]) -> None:
    # 聲道切換加位置按鈕要挑得到每一對，而且一個（聲道、位置）只對到一對，不然有一對永遠按不到。
    overlay = _view(pair).overlay
    combos = [(item.channel, item.position) for item in overlay.pairs]
    assert sorted(combos) == sorted(set(combos))
    assert {channel for channel, _ in combos} == {item.key for item in overlay.channels}
    assert {position for _, position in combos} == {item.key for item in overlay.positions}
    # 名字查顯示名稱表（跟 /api/labels 同一張）：主位與周圍點的方向全名、左右聲道。
    assert {item.label for item in overlay.positions} <= {LABELS["primary"], *LISTENING_POINTS.values()}
    assert {item.label for item in overlay.channels} == {LABELS[item.key] for item in overlay.channels}
    labels = {item.key: item.label for item in overlay.channels} | {
        item.key: item.label for item in overlay.positions}
    assert all(item.label == f"{labels[item.channel]}・{labels[item.position]}" for item in overlay.pairs)


def test_legends_name_channel_and_seat_without_codes(pair: tuple[SchemeResult, SchemeResult]) -> None:
    # 圖例寫「哪一份・哪個聲道・哪個位置」，位置用那一份自己的座位名，不再帶座位代號（main、front）。
    view = _view(pair)
    names = {"a": pair[0], "b": pair[1]}
    for series in view.overlay.series:
        point = next(item for item in names[series.side].scheme.receiver_set.points
                     if item.receiver_id == series.receiver_id)
        seat = ("主位" if point.role.value == "primary"
                else DIRECTIONS[str(point.direction_relative_to_primary)][1])
        assert series.legend_text == f"{series.side.upper()}・{LABELS[series.role]}・{seat}"
        assert series.receiver_id not in series.legend_text


def test_pair_label_shows_both_directions_when_sides_differ(
        pair: tuple[SchemeResult, SchemeResult]) -> None:
    # 按鈕上的座位方向兩邊都要看：B 同代號的座位方向不同時，只寫 A 的會讓人以為兩點同一處。
    b = pair[1]
    front = next(point for point in b.scheme.receiver_set.points
                 if point.direction_relative_to_primary == "front")
    points = tuple(point.model_copy(update={"direction_relative_to_primary": "left"})
                   if point is front else point for point in b.scheme.receiver_set.points)
    turned = b.model_copy(update={"scheme": b.scheme.model_copy(update={
        "receiver_set": b.scheme.receiver_set.model_copy(update={"points": points})})})
    labels = {item.label for item in _view((pair[0], turned)).overlay.pairs}
    # 兩邊方向不同時代號表上的名字（主位前方）會跟 B 矛盾，改寫座位代號並列出兩邊的方向。
    assert f"左聲道・座位 {front.receiver_id}（A：主位前方／B：主位左方）" in labels
    # 一邊沒設方向也算不同。
    points = tuple(point.model_copy(update={"direction_relative_to_primary": None})
                   if point is front else point for point in b.scheme.receiver_set.points)
    blank = b.model_copy(update={"scheme": b.scheme.model_copy(update={
        "receiver_set": b.scheme.receiver_set.model_copy(update={"points": points})})})
    assert f"左聲道・座位 {front.receiver_id}（A：主位前方／B：沒設方向）" in {
        item.label for item in _view((pair[0], blank)).overlay.pairs}
    assert "左聲道・主位前方" in {item.label for item in _view(pair).overlay.pairs}
    # 圖例各寫各的方向：B 那條線寫 B 自己的方向。
    legends = {item.legend_text for item in _view((pair[0], turned)).overlay.series}
    assert {"A・左聲道・主位前方", "B・左聲道・主位左方"} <= legends


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


def test_change_labels_use_display_names(pair: tuple[SchemeResult, SchemeResult]) -> None:
    # 「改了哪裡」寫喇叭與座位的顯示名（左聲道喇叭、主位前方），不寫代號；表上沒有的代號才照原樣寫。
    scheme = pair[0].scheme
    channel = scheme.channel_group.channels[0]
    point = next(item for item in scheme.receiver_set.points if item.role.value == "surrounding")
    original = scheme.speakers[channel.speaker_id]
    speakers = {**scheme.speakers, channel.speaker_id: Point(original.x + 0.1, original.y, original.z),
                "spare": Point(original.x, original.y + 0.2, original.z)}
    points = tuple(item.model_copy(update={"position_m": (item.position_m[0] + 0.1, *item.position_m[1:])})
                   if item is point else item for item in scheme.receiver_set.points)
    added = point.model_copy(update={"receiver_id": "sofa-2"})
    changed = scheme.model_copy(update={"speakers": speakers, "receiver_set": scheme.receiver_set.model_copy(
        update={"points": (*points, added)})})
    labels = {row.path: row.label for row in scheme_differences(scheme, changed)}
    assert labels[f"speakers.{channel.speaker_id}.x"] == f"{SPEAKERS[channel.role]} x 座標"
    assert labels[f"receiver_set.points.{point.receiver_id}.position.x"] == (
        f"{LISTENING_POINTS[point.receiver_id]} x 座標")
    assert labels["speakers.spare.y"] == "喇叭 spare y 座標"
    assert labels["receiver_set.points.sofa-2"] == "座位 sofa-2"
    assert not [label for label in labels.values()
                if re.search(rf"\b({channel.speaker_id}|{point.receiver_id})\b", label)]


def test_changed_keys_follow_position_changes(pair: tuple[SchemeResult, SchemeResult]) -> None:
    scheme = pair[0].scheme
    speaker_id = next(iter(scheme.speakers))
    original = scheme.speakers[speaker_id]
    speakers = {**scheme.speakers,
                speaker_id: Point(original.x + 0.1, original.y, original.z)}
    primary = scheme.receiver_set.primary
    moved = primary.model_copy(update={"position_m": (*primary.position_m[:2],
                                                    primary.position_m[2] + 0.1)})
    added = scheme.receiver_set.points[-1].model_copy(update={"receiver_id": "extra-seat"})
    points = tuple(moved if item.receiver_id == primary.receiver_id else item
                   for item in scheme.receiver_set.points) + (added,)
    wall = next(iter(scheme.scene.impedance_pa_s_per_m_by_wall))
    impedances = {**scheme.scene.impedance_pa_s_per_m_by_wall,
                  wall: scheme.scene.impedance_pa_s_per_m_by_wall[wall] + 1}
    changed = scheme.model_copy(update={
        "scheme_id": pair[1].scheme.scheme_id, "speakers": speakers,
        "scene": scheme.scene.model_copy(update={"impedance_pa_s_per_m_by_wall": impedances}),
        "receiver_set": scheme.receiver_set.model_copy(update={"points": points})})
    targets = config_path("quality_targets.toml")
    result = build_compare_view(
        a_run_id="a" * 32, a=pair[0], view_a=build_result_view(pair[0], quality_targets_path=targets),
        b_run_id="b" * 32, b=pair[1].model_copy(update={"scheme": changed}),
        view_b=build_result_view(pair[1], quality_targets_path=targets),
        quality_targets=load_quality_targets(targets), run_date=date(2026, 9, 27))
    assert set(result.changed_keys) == {
        f"speaker:{speaker_id}", f"receiver:{primary.receiver_id}", "receiver:extra-seat"}


def test_changed_keys_include_receiver_role_and_direction(
        pair: tuple[SchemeResult, SchemeResult]) -> None:
    scheme = pair[0].scheme
    point = next(point for point in scheme.receiver_set.points if point.role.value == "surrounding")
    for update in ({"role": point.role.__class__.OTHER_SEAT},
                   {"direction_relative_to_primary": (
                       "left" if point.direction_relative_to_primary != "left" else "right")}):
        changed_points = tuple(item.model_copy(update=update)
                               if item.receiver_id == point.receiver_id else item
                               for item in scheme.receiver_set.points)
        changed = scheme.model_copy(update={"receiver_set": scheme.receiver_set.model_copy(
            update={"points": changed_points})})
        assert set(_changed_keys(scheme_differences(scheme, changed))) == {
            f"receiver:{point.receiver_id}"}


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
    # 匯出的頻響資料：聯集軸上缺的格子留空白，不寫 0（0 dB 是一個真的音量）。
    rows = list(csv.reader(io.StringIO(curves_csv(view).removeprefix("\ufeff"))))
    assert [[float(cell) if cell else None for cell in row] for row in rows[1:]] == [
        [100, 1, None], [200, None, 8], [300, 3, 9]]


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
    altered = first.model_copy(update={"calculation_fingerprint": "calc-v1:" + "1" * 64})
    problems = comparison_problems((first, altered))
    assert any("候選代號重複" in row for row in problems)
    assert any("計算指紋" in row and short_fingerprint(first.calculation_fingerprint) in row
               and short_fingerprint(altered.calculation_fingerprint) in row
               for row in problems)


def test_split_tables_print_no_total_and_mark_categories(
        pair: tuple[SchemeResult, SchemeResult], moved: SchemeResult) -> None:
    # 不同表時哪一張算主表只看比較身分排序；落在主表那份的「名次 1」只是一個人的名次。
    for a, b in ((pair[0], moved), (moved, pair[0])):
        view = _view((a, b))
        table = view.table
        assert not table.same_table
        # 哪一份落在主表只看身分排序：兩邊寫一樣的字、不印任何數字，也不判哪一份比較好。
        assert table.a_text == table.b_text == NO_TOTAL_TEXT
        assert not re.search(r"\d", table.a_text + table.b_text)
        assert (table.verdict_text, table.better) == ("", "")
        marked = {row.category for row in view.categories if row.comparison_text}
        assert marked == {"channel_matching", "listening_area_stability"}
        assert all(row.comparison_text == "評分條件不同，這一類代價不能直接比"
                   and row.comparison_text in row.note_text and (row.better, row.better_text) == ("", "")
                   for row in view.categories if row.category in marked)
        # 條件相同的類照樣逐類判（兩份都有代價的才判）。
        assert all(row.better in {"a", "b", "same"} for row in view.categories
                   if row.category not in marked and row.a.cost is not None and row.b.cost is not None)
        assert table.reason_text.startswith("兩份不能直接比總代價：")
        assert "兩份都有但條件不同：" in table.reason_text
        assert "身分" not in table.reason_text
        assert not re.search(r"[a-z]+_[a-z_]+", table.reason_text)
        checks = {check.label: check.same for check in view.fingerprints}
        assert checks == {"座位組": False, "座位相對佈局": False, "聲道組": True}


def test_equal_totals_print_no_rank(pair: tuple[SchemeResult, SchemeResult]) -> None:
    # 同內容換名字（另存新名字常見）：總代價一樣時名次只照代號排，不代表好壞，要寫分不出來。
    view = _view((pair[0], _renamed(pair[0], "wall-1-copy")))
    assert view.table.same_table
    assert view.table.a_text == view.table.b_text
    assert all(text.startswith("總代價 ") and text.endswith("（越低越好）") and "名次" not in text
               and "微小差異" not in text for text in (view.table.a_text, view.table.b_text))
    assert (view.table.verdict_text, view.table.better) == ("兩份總代價相同，分不出哪一份比較好", "same")
    assert all(row.better in {"same", ""} for row in view.categories)
    assert view.changes == ()
    assert view.summary_text == "兩份方案設定相同"


def _without(result: SchemeResult, category: QualityCategory) -> SchemeResult:
    return result.model_copy(update={"candidate": result.candidate.model_copy(update={
        "evaluations": tuple(item for item in result.candidate.evaluations
                             if item.category is not category)})})


def test_unranked_sides_say_which_or_neither(pair: tuple[SchemeResult, SchemeResult]) -> None:
    # 少一類評估就排不上（未評估）：寫清楚是只有一份排得上，還是兩份都排不上，也都不印名次。
    targets = config_path("quality_targets.toml")
    missing = QualityCategory.TIMBRE_BALANCE
    only_b, _ = _table(_without(pair[0], missing), pair[1], load_quality_targets(targets),
                    date(2026, 9, 27))
    neither, _ = _table(_without(pair[0], missing), _without(pair[1], missing),
                     load_quality_targets(targets), date(2026, 9, 27))
    assert "只有 B 排得上" in only_b.reason_text
    assert "兩份都排不上" in neither.reason_text
    assert all("名次" not in table.a_text + table.b_text for table in (only_b, neither))


def test_ranked_sides_print_totals_lower_is_better_and_verdict(
        pair: tuple[SchemeResult, SchemeResult]) -> None:
    # 兩份都排得上：各印總代價並註明越低越好；伺服器判哪一份比較好、低多少（跟排名層的總代價對）。
    view = _view(pair)
    table = view.table
    assert table.same_table and table.reason_text == COMPARABLE_TEXT
    assert all(text.startswith("總代價 ") and text.endswith("（越低越好）") and "名次" not in text
               for text in (table.a_text, table.b_text))
    assert table.a_text != table.b_text
    ranking = compare_results(pair, quality_targets=load_quality_targets(
        config_path("quality_targets.toml")), run_date=date(2026, 9, 27))
    totals = {row.candidate_id: row.total_cost for row in ranking.rankable}
    total_a, total_b = totals[pair[0].scheme.scheme_id], totals[pair[1].scheme.scheme_id]
    winner, loser = ("A", "B") if total_a < total_b else ("B", "A")
    match = re.fullmatch(rf"{winner} 比較好：總代價比 {loser} 低 ([0-9.]+)", table.verdict_text)
    assert match is not None, table.verdict_text
    assert math.isclose(float(match[1]), abs(total_a - total_b), rel_tol=5e-3)
    assert table.better == winner.lower()
    # 校準那句照排名層的校準狀態選白話，不再是「未校準，不是品質判決」。
    assert table.calibration_text == CALIBRATION_TEXTS[ranking.header.calibration]
    assert "品質判決" not in table.calibration_text and "合格" in table.calibration_text


def test_category_rows_mark_lower_cost_and_ties_by_printed_text(
        pair: tuple[SchemeResult, SchemeResult]) -> None:
    # 每一類代價越低越好，伺服器判哪一份較好；印出來一樣（看不到的小數位不同）就寫相同，缺代價不判。
    targets = config_path("quality_targets.toml")
    view_a = build_result_view(pair[0], quality_targets_path=targets)
    costed = [(row, row.cost) for row in view_a.categories if row.cost is not None]
    (lower, lower_cost), (hidden, hidden_cost) = costed[0], costed[1]
    # B 的第一類低 0.5（印出來也不同）；第二類只多 1e-9，印出來跟 A 一樣。
    rows_b = tuple(
        row.model_copy(update={"cost": lower_cost - 0.5, "cost_text": f"{lower_cost - 0.5:.4f}"})
        if row is lower else row.model_copy(update={"cost": hidden_cost + 1e-9})
        if row is hidden else row for row in view_a.categories)
    view = build_compare_view(
        a_run_id="a" * 32, a=pair[0], view_a=view_a, b_run_id="b" * 32, b=pair[1],
        view_b=view_a.model_copy(update={"categories": rows_b}),
        quality_targets=load_quality_targets(targets), run_date=date(2026, 9, 27))
    rows = {row.category: row for row in view.categories}
    assert (rows[lower.category].better, rows[lower.category].better_text) == ("b", "B 較好")
    assert (rows[hidden.category].better, rows[hidden.category].better_text) == ("same", "相同")
    assert all((row.better, row.better_text) == ("", "") for row in view.categories
               if row.a.cost is None or row.b.cost is None)
    for row in _view(pair).categories:
        if row.a.cost is None or row.b.cost is None:
            continue
        side = ("same" if row.a.cost_text == row.b.cost_text
                else "a" if row.a.cost < row.b.cost else "b")
        assert row.better == side
        assert row.better_text == ("相同" if side == "same" else f"{side.upper()} 較好")


def test_identity_and_summary_texts_follow_their_side(pair: tuple[SchemeResult, SchemeResult]) -> None:
    other = pair[1].model_copy(update={
        "engine_commit": "e53bfae" + "f" * 33,
        "calculation_fingerprint": "calc-v1:" + "1" * 64})
    view = _view((pair[0], other))
    assert (view.a.engine_text, view.b.engine_text) == (pair[0].engine_commit[:7], "e53bfae")
    assert (view.a.scheme_id, view.b.scheme_id) == ("wall-1", "wall-2")
    assert not view.table.same_table
    assert (view.a.fingerprint_text, view.b.fingerprint_text) == (
        short_fingerprint(pair[0].calculation_fingerprint), short_fingerprint(other.calculation_fingerprint))
    assert "計算指紋（calculation_fingerprint）不同" in view.table.reason_text
    extra = f"另 {len(view.changes) - 5} 處"
    assert (extra in view.summary_text) == (len(view.changes) > 5)
    assert "等 " not in view.summary_text


def test_not_evaluated_categories_are_said_once_in_summary(
        pair: tuple[SchemeResult, SchemeResult]) -> None:
    # 低頻拖尾「尚未評估」摘要寫一次（決策紙：摘要要標出來），分項表說明欄與說明區不再重複。
    view = _view(pair)
    pending = [row for row in view.categories
               if row.a.state == row.b.state == "not_evaluated"]
    assert "low_frequency_decay" in {row.category for row in pending}
    assert view.pending_text == (
        "、".join(row.label for row in pending) + "：兩份都尚未評估，不算進總代價")
    texts = [view.summary_text, view.pending_text, *view.notes,
             *(row.note_text for row in view.categories)]
    assert [text for text in texts if "低頻拖尾" in text] == [view.pending_text]
    assert all(not row.note_text for row in pending)
    # 校準那句只在摘要（calibration_text），說明區不再放一句「尚未正式校準」。
    assert not [note for note in view.notes if "校準" in note]


def test_better_or_worse_only_when_totals_are_comparable(
        pair: tuple[SchemeResult, SchemeResult], moved: SchemeResult) -> None:
    # 總代價能直接比才說哪一份比較好；不同表、固定身分對不上時一個字都不判，分項也不偷判。
    ranked = _view(pair)
    assert "比較好" in ranked.table.verdict_text
    other = pair[1].model_copy(update={"calculation_fingerprint": "calc-v1:" + "1" * 64})
    for view in (_view((pair[0], moved)), _view((pair[0], other))):
        text = " ".join((view.summary_text, view.pending_text, view.table.a_text, view.table.b_text,
                         view.table.reason_text, view.table.verdict_text))
        assert all(word not in text for word in ("較好", "較差", "更好", "比較好"))
        assert view.table.better == ""
    # 固定身分對不上（計算指紋不同）：排名層不收，分項一類都不判，也不說不算進總代價。
    unmatched = _view((pair[0], other))
    assert all(row.better == "" for row in unmatched.categories)
    assert "不算進總代價" not in unmatched.pending_text
