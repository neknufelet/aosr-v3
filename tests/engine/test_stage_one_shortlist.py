"""入選清單只讀細算；答案由各席原始結果獨立計算。"""
from pathlib import Path
from collections.abc import Iterator, Sequence
from dataclasses import FrozenInstanceError, replace
import json

import pytest

from aosr.search.settings import SearchSettings
from aosr.reporting.stage_one_result import save_stage_one_result
from aosr.scoring.stage_one_score import score_stage_one
from aosr.search.run import CandidateJob
from aosr.search.stage_one_worker import StageOneComputed
from aosr.search.stage_one_report import build_stage_one_report, render_stage_one_text
from aosr.search.store import SearchStore
from tests.engine import _search_stage_one_cases as search
from tests.engine._search_room_loop_cases import layout_for
from tests.engine._stage_one_refine_cases import pool, snapshot
from tests.engine.test_search_stage_one_refine import refine


def _finished(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, variable: bool = True) -> SearchStore:
    source = search.make_store(tmp_path / "source", budget=4, layout=layout_for() if variable else None)
    settings = SearchSettings.model_validate(source.settings.model_dump(mode="json") | {
        "stage_one_refine": {"top_n": 3, "far_control_min_front_distance_m": .8}})
    store = SearchStore.create(tmp_path / "searches", project=source.project, settings=settings,
        identity=source.identity, versions={"python": "test", "numpy": "test", "optuna": "test"})
    search.run(store, search.FakeCompute(store, monkeypatch))
    refine(store, search.FakeCompute(store, monkeypatch))
    return store


def test_report_inserts_shortlist_between_refine_and_post_check(tmp_path: Path,
                                                              monkeypatch: pytest.MonkeyPatch) -> None:
    # 漏接清單，或插到入選後檢查之後，都必須失敗。
    store = _finished(tmp_path, monkeypatch)
    text = render_stage_one_text(build_stage_one_report(store))
    title = "入選清單（細算後；前 3 席照細算名次，對照列出來、不算入選）"
    assert title in text
    assert text.index("細算（每 1 Hz 一點，只算有限元素）") < text.index(title)
    assert text.index(title) < text.index("入選後檢查（低頻模態診斷）")


def _numbers(store: SearchStore, number: int) -> tuple[float, ...]:
    # 獨立讀 JSON，禁止拿清單自己的資料或面積函式當答案。
    raw = json.loads(store.refine_result_path(number).read_bytes())
    room, score = raw["scheme"]["scene"]["room_m"], raw["score"]
    x, y, z = (room[key] for key in ("Lx", "Ly", "Lz"))
    parts = {item["name"]: item["value_db"] for item in score["components"]}
    return (x, y, z, x * y, x * y * z, parts["own_line_ripple_rms_db"],
            parts["seat_robustness_rms_db"], score["screening_total"], score["principal_only_sort_key_db"])


def test_variable_rooms_values_deltas_order_and_no_writes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from aosr.search.stage_one_shortlist import stage_one_shortlist, stage_one_shortlist_lines
    store = _finished(tmp_path, monkeypatch)
    raw = json.loads((store.path / "stage_one_refine.json").read_bytes())
    top = {seat["trial_number"] for seat in raw["selection"]["candidates"] if "top_n" in seat["roles"]}
    ranks = {seat["trial_number"]: seat["refine_rank"] for seat in raw["results"]}
    expected = sorted(top, key=lambda n: ranks[n])
    expected.extend(seat["trial_number"] for seat in raw["selection"]["candidates"] if seat["trial_number"] not in top)
    before = snapshot(store)
    shortlist = stage_one_shortlist(store)
    assert shortlist is not None
    assert [row.trial_number for row in shortlist.rows] == expected
    assert len({_numbers(store, n)[:3] for n in top}) >= raw["selection"]["top_n"]
    base = _numbers(store, expected[0])
    lines = stage_one_shortlist_lines(store)
    for row in shortlist.rows:
        want = _numbers(store, row.trial_number)
        values, delta = row.values, row.delta
        assert values is not None and delta is not None
        assert (values.length_m, values.width_m, values.height_m, values.floor_area_m2, values.volume_m3,
                values.principal_db, values.robustness_db, values.screening_total,
                values.principal_only_sort_key_db) == want
        difference = tuple(want[index] - base[index] for index in (5, 6, 7, 3, 4))
        assert (delta.principal_db, delta.robustness_db, delta.screening_total,
                delta.floor_area_m2, delta.volume_m3) == difference
        line = next(line for line in lines if line.startswith(f"試算 {row.trial_number}｜"))
        assert "×".join(f"{value:.6f}" for value in want[:3]) in line
        assert "｜".join(f"{value:.6f}" for value in want[3:]) in line
        assert "／".join(f"{value:+.6f}" for value in difference) in line
    assert snapshot(store) == before
    with pytest.raises(FrozenInstanceError):
        setattr(shortlist, "top_n", 999)


def test_principal_rank_can_disagree_with_total_and_ties_use_trial_number(tmp_path: Path,
                                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    from aosr.search.stage_one_shortlist import stage_one_shortlist
    # 替身能量與房間無關；在細算保存前，明造兩分項此消彼長並重算合法合分。
    # 試算 1 主項最低卻穩健性最差；2、8 主項相同，用試算編號破同分；對照 5 更低也不能進榜。
    pairs = {8: (2., 1.), 2: (2., 0.), 1: (1., 8.), 5: (.5, 0.)}
    store = pool(tmp_path, monkeypatch, [(8, 1., 5., .5), (2, 2., 8., .6),
                                           (1, 3., 4., .7), (5, 4., 7., .8)], top_n=3)
    fake = search.FakeCompute(store, monkeypatch)
    def compute(jobs: Sequence[CandidateJob], workers: int) -> Iterator[StageOneComputed]:
        for item in fake(jobs, workers):
            number = item.job.trial_number
            assert number is not None
            principal, robustness = pairs[number]
            result = item.result
            speakers = []
            for speaker in result.evaluation.speakers:
                assert speaker.payload is not None
                payload = speaker.payload.model_copy(update={"own_line_ripple_rms_db": principal,
                    "seat_robustness_rms_db": robustness})
                speakers.append(speaker.model_copy(update={"payload": payload}))
            evaluation = result.evaluation.model_copy(update={"speakers": tuple(speakers)})
            result = result.model_copy(update={"evaluation": evaluation, "score": score_stage_one(evaluation, result.score_settings)})
            save_stage_one_result(result, item.job.result_path)
            yield replace(item, result=result)
    refine(store, compute)
    shortlist = stage_one_shortlist(store)
    assert shortlist is not None
    top = [row for row in shortlist.rows if row.refine_rank is not None]
    assert [row.trial_number for row in top] == [2, 8, 1]
    assert [row.trial_number for row in sorted(top, key=lambda row: row.principal_rank or float("inf"))] == [1, 2, 8]
    for row in top:
        assert row.values is not None
        assert (row.values.principal_db, row.values.robustness_db) == pairs[row.trial_number]
    assert all(row.principal_rank is None for row in shortlist.rows if row.trial_number == 5)


def test_fixed_room_keeps_seats_and_zero_space_deltas(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from aosr.search.stage_one_shortlist import stage_one_shortlist, stage_one_shortlist_lines
    store = _finished(tmp_path, monkeypatch, variable=False)
    shortlist = stage_one_shortlist(store)
    assert shortlist is not None and shortlist.rows
    assert "房間固定，下列是同一間房的不同擺法" in stage_one_shortlist_lines(store)
    for row in shortlist.rows:
        assert row.values is not None and row.delta is not None
        assert (row.values.length_m, row.values.width_m, row.values.height_m) == _numbers(store, row.trial_number)[:3]
        assert _numbers(store, row.trial_number)[:3] == _numbers(store, shortlist.rows[0].trial_number)[:3]
        assert (row.delta.floor_area_m2, row.delta.volume_m3) == (0., 0.)


def test_controls_have_roles_deltas_no_ranks_and_overlapping_top_stays_top(tmp_path: Path,
                                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    from aosr.search.stage_one_shortlist import stage_one_shortlist, stage_one_shortlist_lines
    store = pool(tmp_path, monkeypatch, [(8, 1., 5., .9), (2, 2., 8., .6),
                                           (1, 3., 3., .7), (5, 4., 7., .8)])
    status = refine(store, search.FakeCompute(store, monkeypatch))
    shortlist = stage_one_shortlist(store)
    assert shortlist is not None
    top = {seat.trial_number for seat in status.selection.candidates if "top_n" in seat.roles}
    assert {row.trial_number for row in shortlist.rows[:len(top)]} == top
    assert [row.trial_number for row in shortlist.rows].count(8) == sum(seat.trial_number == 8 for seat in status.selection.candidates)
    overlap = next(row for row in shortlist.rows if row.trial_number == 8)
    assert overlap.roles == ("前 N（搜尋第 1 名）", "較遠對照")
    assert overlap.refine_rank is not None and overlap.principal_rank is not None
    lines = stage_one_shortlist_lines(store)
    control = next(row for row in shortlist.rows if row.trial_number == 1)
    assert control.roles == ("大峰對照",)
    assert control.refine_rank is None and control.principal_rank is None and control.delta is not None
    assert any(line.startswith("試算 1｜大峰對照｜3｜無｜無｜已計分｜") for line in lines)


@pytest.mark.parametrize("state", ["selected", "interrupted"])
def test_unfinished_without_winner_shows_status_and_no_numbers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                              state: str) -> None:
    from aosr.search.stage_one_shortlist import stage_one_shortlist, stage_one_shortlist_lines
    store = _finished(tmp_path, monkeypatch)
    path = store.path / "stage_one_refine.json"
    raw = json.loads(path.read_bytes())
    raw.update(state=state, results=[])
    path.write_text(json.dumps(raw))
    shortlist = stage_one_shortlist(store)
    assert shortlist is not None
    assert [row.trial_number for row in shortlist.rows] == [seat["trial_number"] for seat in raw["selection"]["candidates"]]
    for row in shortlist.rows:
        assert row.state == "pending" and row.reason
        assert row.values is None and row.delta is None and row.refine_rank is None and row.principal_rank is None
    rows = [line for line in stage_one_shortlist_lines(store) if line.startswith("試算 ")]
    assert rows and all("尚未完成" in line and "沒有細算第一名，不算差值" in line for line in rows)
    assert all(".000000" not in line for line in rows)


@pytest.mark.parametrize("state,label", [("pending", "尚未完成"), ("unassessed", "不可估"),
                                        ("incomparable", "不能同表"), ("unreadable", "讀不回")])
def test_unscored_seat_moves_after_ranked_seats_and_hides_numbers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                               state: str, label: str) -> None:
    from aosr.search.stage_one_shortlist import stage_one_shortlist, stage_one_shortlist_lines
    store = _finished(tmp_path, monkeypatch)
    path = store.path / "stage_one_refine.json"
    raw = json.loads(path.read_bytes())
    first = next(seat for seat in raw["results"] if seat["refine_rank"] == 1)
    number = first["trial_number"]
    first.update(state=state, score=None, refine_rank=None, reason="合成原因")
    path.write_text(json.dumps(raw))
    shortlist = stage_one_shortlist(store)
    assert shortlist is not None
    row = next(row for row in shortlist.rows if row.trial_number == number)
    assert row.state == state and row.reason == "合成原因" and row.values is None
    top_order = [row for row in shortlist.rows if row.refine_rank is not None]
    assert shortlist.rows.index(row) == len(top_order)
    line = next(line for line in stage_one_shortlist_lines(store) if line.startswith(f"試算 {number}｜"))
    assert f"{label}；合成原因" in line and "沒有細算第一名，不算差值" in line
    assert ".000000" not in line


@pytest.mark.parametrize("damage", ["missing", "corrupt", "mismatch"])
def test_bad_result_only_breaks_one_seat_and_missing_winner_is_not_replaced(tmp_path: Path,
                                                                         monkeypatch: pytest.MonkeyPatch,
                                                                         damage: str) -> None:
    from aosr.search.stage_one_shortlist import stage_one_shortlist, stage_one_shortlist_lines
    store = _finished(tmp_path, monkeypatch)
    summary = json.loads((store.path / "stage_one_refine.json").read_bytes())
    number = next(seat["trial_number"] for seat in summary["results"] if seat["refine_rank"] == 1)
    path = store.refine_result_path(number)
    if damage == "missing":
        path.unlink()
    elif damage == "corrupt":
        path.write_text("{")
    else:
        target = next(seat for seat in summary["results"] if seat["trial_number"] == number)
        target["score"] += 1.
        (store.path / "stage_one_refine.json").write_text(json.dumps(summary))
    shortlist = stage_one_shortlist(store)
    assert shortlist is not None
    for row in shortlist.rows:
        assert (row.values is None) == (row.trial_number == number)
        assert row.delta is None
    line = next(line for line in stage_one_shortlist_lines(store) if line.startswith(f"試算 {number}｜"))
    assert "讀不回" in line and "沒有細算第一名，不算差值" in line


def test_missing_summary_leaves_report_byte_identical_to_original(tmp_path: Path,
                                                                monkeypatch: pytest.MonkeyPatch) -> None:
    from aosr.search.stage_one_shortlist import stage_one_shortlist_lines
    store = _finished(tmp_path, monkeypatch)
    (store.path / "stage_one_refine.json").unlink()
    report = build_stage_one_report(store)
    assert stage_one_shortlist_lines(store) == () and report.shortlist_lines == ()
    # 重建舊版組字順序：每一段的原文都由既有報告提供，這裡不呼叫新清單算答案。
    original = render_stage_one_text(replace(report, refine_lines=(), post_check_lines=()))
    marker = "300 Hz 交接敏感度（不改名次）\n"
    original = original.replace(marker, "\n".join((*report.refine_lines, *report.post_check_lines)) + "\n" + marker)
    assert render_stage_one_text(report).encode() == original.encode()


def test_exact_notices_and_no_ratio_claims_or_repeated_assumptions(tmp_path: Path,
                                                                monkeypatch: pytest.MonkeyPatch) -> None:
    from aosr.search import stage_one_shortlist as module
    store = _finished(tmp_path, monkeypatch)
    lines = module.stage_one_shortlist_lines(store)
    expected = (
        "入選清單（細算後；前 3 席照細算名次，對照列出來、不算入選）",
        "細算第一名不是最終推薦：名次只照現行低頻合分排",
        "名次是低頻合分的排序；品質可不可接受另看入選後檢查（超線共振；改善能力尚未確認），兩件事分開看",
        "差值都對細算第一名，用 1 Hz 驗證軸；分項是原始 dB",
        "房間隨候選而變，面積與容積照每一席自己的房間算")
    assert lines[:len(expected)] == expected
    assert module.TITLE.format(top_n=3) == expected[0]
    assert (module.NOT_FINAL, module.RANK_AND_QUALITY, module.DELTA_NOTICE, module.VARIABLE_ROOM) == expected[1:]
    assert module.FIXED_ROOM == "房間固定，下列是同一間房的不同擺法"
    assert module.NO_WINNER == "沒有細算第一名，不算差值"
    text = "\n".join(lines)
    assert all(word not in text for word in ("比例較好", "最佳比例", "黃金比例", "比例最好",
                                            "在這個牆面假設下", "未加入實際喇叭的低頻衰減", "低頻拖尾不進自動排名"))


@pytest.mark.parametrize("damage", ["json", "content", "unreadable"])
def test_bad_summary_is_one_error_line_and_other_sections_survive(tmp_path: Path,
                                                                 monkeypatch: pytest.MonkeyPatch,
                                                                 damage: str) -> None:
    from aosr.search.stage_one_shortlist import stage_one_shortlist_lines
    store = _finished(tmp_path, monkeypatch)
    before = build_stage_one_report(store)
    path = store.path / "stage_one_refine.json"
    if damage == "json":
        path.write_text("{")
    elif damage == "content":
        path.write_text("{}")
    else:
        path.unlink()
        path.mkdir()
    report = build_stage_one_report(store)
    assert report.ranked == before.ranked and report.post_check_lines == before.post_check_lines
    lines = stage_one_shortlist_lines(store)
    assert lines == report.shortlist_lines
    assert lines == (next(line for line in lines if line.startswith("入選清單讀不回：")),)
    text = render_stage_one_text(report)
    assert text.startswith(before.wall_text) and "入選後檢查（低頻模態診斷）" in text
    assert lines == tuple("\n".join(lines).splitlines())


def test_unscored_order_uses_selection_order_and_dimensions_ignore_selection_meters(tmp_path: Path,
                                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    from aosr.search.stage_one_shortlist import stage_one_shortlist
    store = _finished(tmp_path, monkeypatch)
    path = store.path / "stage_one_refine.json"
    raw = json.loads(path.read_bytes())
    top = [seat for seat in raw["selection"]["candidates"] if "top_n" in seat["roles"]]
    ranked, *pending = top
    # 未完成席故意反轉搜尋名次順序；另把帳列公尺量改成錯誤數字，不能拿它當房間答案。
    raw["selection"]["candidates"] = [*reversed(pending), ranked,
        *(seat for seat in raw["selection"]["candidates"] if "top_n" not in seat["roles"])]
    for seat in raw["selection"]["candidates"]:
        seat["params_m"].update(room_Lx=999., room_Ly=999., room_Lz=999.)
    for seat in raw["results"]:
        if seat["trial_number"] in {item["trial_number"] for item in pending}:
            seat.update(state="pending", refine_rank=None, score=None)
    raw["state"] = "interrupted"
    path.write_text(json.dumps(raw))
    shortlist = stage_one_shortlist(store)
    assert shortlist is not None
    assert [row.trial_number for row in shortlist.rows if row.is_top_n] == [
        ranked["trial_number"], *(seat["trial_number"] for seat in reversed(pending))]
    row = next(row for row in shortlist.rows if row.trial_number == ranked["trial_number"])
    assert row.values is not None
    assert (row.values.length_m, row.values.width_m, row.values.height_m) == _numbers(store, row.trial_number)[:3]
