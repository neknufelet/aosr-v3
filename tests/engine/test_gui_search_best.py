"""目前最佳只讀已存結果；三圖獨立失敗、快照門檻與有界快取。"""
import hashlib
import json
from pathlib import Path
from dataclasses import asdict
from typing import Literal

import pytest

from aosr.config.directivity_defaults import load_directivity_defaults
from aosr.config.paths import config_path
from aosr.gui.search_best import BestCache, build_best_view
from aosr.gui.search_view import build_search_view
from aosr.reporting.display import level_db
from aosr.reporting.result import SchemeResult
from aosr.scoring.contract import QualityCategory
from aosr.scoring.reflections_contract import ReflectionsAndEchoPayload
from tests.engine._gui_best_cases import best_result, best_store
from tests.engine.test_gui_search_view import _snapshot


def test_approximate_channel_summary_explains_model_without_claiming_missing_coverage() -> None:
    from aosr.gui.search_best import _channel
    from tests.engine.test_furniture_reflections import furniture_records
    from tests.engine import test_reflections as fixtures
    result = fixtures._evaluate(furniture_records())
    assert isinstance(result.payload, ReflectionsAndEchoPayload)
    channel = next(item for item in result.payload.channels if item.is_primary)
    shown = _channel(channel, dict.fromkeys(("front", "lateral", "rear", "vertical"), -10.0), 15.0)
    summary = shown["summary_text"]
    assert isinstance(summary, str)
    assert "近似" in summary
    assert "家具參與的多次反射未納入" in summary
    assert "覆蓋或量測未完成" not in summary


def _best(path: Path, which: str = "search", cache: BestCache | None = None) -> dict[str, object]:
    return build_best_view(path, which=which, cache=cache or BestCache(),
                           directivity=load_directivity_defaults(config_path("directivity_defaults.toml")))


def test_best_frequency_reflections_and_plan_match_saved_result(tmp_path: Path, best_result: SchemeResult) -> None:
    store = best_store(tmp_path, best_result)
    before = _snapshot(store.path)
    view = _best(store.path)
    frequency = view["frequency"]
    assert isinstance(frequency, dict) and not frequency["error"]
    expected = [pair for pair in best_result.pairs if pair.receiver_id == best_result.scheme.receiver_set.primary.receiver_id]
    assert frequency["curves"] == [{"role": pair.role, "label": f"{'左' if pair.role == 'left' else '右'}聲道 → 主位",
                                    "points": [[point.frequency_hz, level_db(point.total_energy)] for point in pair.report.points or ()]}
                                   for pair in expected]
    rfz = view["rfz"]
    assert isinstance(rfz, dict) and not rfz["error"]
    payload = next(item.payload for item in best_result.candidate.evaluations if item.category == QualityCategory.REFLECTIONS_AND_ECHO)
    assert isinstance(payload, ReflectionsAndEchoPayload)
    for shown, saved in zip(rfz["channels"], (item for item in payload.channels if item.is_primary), strict=True):
        assert [(p["delay_ms"], p["level_db"], p["zone"]) for p in shown["points"]] == [
            (p.relative_direct_delay_s * 1000, p.broadband_level_db, p.zone.value) for p in saved.reflections]
    plan = view["plan"]
    assert isinstance(plan, dict) and not plan["error"]
    assert plan["data"]["room"] == asdict(best_result.scheme.scene.room_m)
    assert {p["id"]: p["point"] for p in plan["data"]["speakers"]} == {
        key: asdict(point) for key, point in best_result.scheme.speakers.items()}
    assert {p["id"]: tuple(p["point"].values()) for p in plan["data"]["receivers"]} == {
        p.receiver_id: p.position_m for p in best_result.scheme.receiver_set.points}
    assert _snapshot(store.path) == before


def _rewrite_purpose(store_path: Path, ledger_path: Path, thresholds: dict[str, float], window: float | None) -> None:
    purpose = json.loads((store_path / "purpose.json").read_text())
    content = purpose["content"]
    for entry in [*content["target"], *content["setting"]]:
        if entry["key"].startswith("reflections_and_echo.zone_threshold_db."):
            entry["value"] = thresholds[entry["key"].rsplit(".", 1)[1]]
    for entry in content["setting"]:
        if window is not None and entry["key"] == "reflections_and_echo.window_upper_ms":
            entry["value"] = window
    purpose["fingerprint"] = hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    (store_path / "purpose.json").write_text(json.dumps(purpose))
    # 帳表頭也固定快照指紋；用途快照改過的樣本要同步表頭，才是同一場有效搜尋。
    lines = ledger_path.read_text().splitlines()
    header = json.loads(lines[0])
    header["purpose_fingerprint"] = purpose["fingerprint"]
    ledger_path.write_text("\n".join([json.dumps(header), *lines[1:]]) + "\n")


def test_thresholds_and_window_use_search_snapshot(tmp_path: Path, best_result: SchemeResult) -> None:
    store = best_store(tmp_path, best_result)
    payload = next(item.payload for item in best_result.candidate.evaluations if item.category == QualityCategory.REFLECTIONS_AND_ECHO)
    assert isinstance(payload, ReflectionsAndEchoPayload)
    # 門檻壓到最低，讓窗內有聲級的點全部超線、窗外的點不算。
    expected = {"front": -99.0, "lateral": -98.0, "rear": -97.0, "vertical": -96.0}
    _rewrite_purpose(store.path, store.ledger_path, expected, None)
    rfz = _best(store.path)["rfz"]
    assert isinstance(rfz, dict) and not rfz["error"]
    assert rfz["thresholds"] == expected and rfz["window_ms"] == payload.window_upper_ms
    saved = [item for item in payload.channels if item.is_primary]
    for channel, stored in zip(rfz["channels"], saved, strict=True):
        assert [p["within_window"] for p in channel["points"]] == [p.within_window for p in stored.reflections]
        assert [p["over_limit"] for p in channel["points"]] == [
            p.within_window and p.broadband_level_db is not None for p in stored.reflections]
        assert channel["inside_count"] == sum(p.within_window for p in stored.reflections)
        assert channel["over_count"] == sum(p.within_window and p.broadband_level_db is not None for p in stored.reflections)
    assert any(not p.within_window and p.broadband_level_db is not None for item in saved for p in item.reflections)
    assert "門檻相同" not in rfz["window_text"]
    _rewrite_purpose(store.path, store.ledger_path, dict.fromkeys(expected, -20.0), None)
    shared = _best(store.path)["rfz"]
    assert isinstance(shared, dict) and "門檻相同的分區合畫成一條灰色虛線" in shared["window_text"]
    # 快照的時間窗跟存下的反射資料不同就照實寫，不挑一邊畫。
    _rewrite_purpose(store.path, store.ledger_path, expected, payload.window_upper_ms + 1.0)
    view = _best(store.path)
    rfz, frequency, plan = view["rfz"], view["frequency"], view["plan"]
    assert isinstance(rfz, dict) and isinstance(frequency, dict) and isinstance(plan, dict)
    assert "反射時間窗" in rfz["error"] and not frequency["error"] and not plan["error"]


@pytest.mark.parametrize("part", ["pairs", "candidate"])
def test_broken_chart_part_keeps_other_charts(tmp_path: Path, best_result: SchemeResult, part: str) -> None:
    store = best_store(tmp_path, best_result)
    document = best_result.model_dump(mode="json")
    document[part] = [] if part == "pairs" else {}
    store.candidate_path(0).write_text(json.dumps(document))
    view = _best(store.path)
    broken = "frequency" if part == "pairs" else "rfz"
    for key in ("frequency", "rfz", "plan"):
        chart = view[key]
        assert isinstance(chart, dict)
        assert bool(chart["error"]) == (key == broken)


def test_plan_failure_does_not_hide_frequency_or_reflections(tmp_path: Path, best_result: SchemeResult,
                                                            monkeypatch: pytest.MonkeyPatch) -> None:
    store = best_store(tmp_path, best_result)
    def broken(*args: object, **kwargs: object) -> dict[str, object]:
        raise ValueError("平面圖欄位無效")
    monkeypatch.setattr("aosr.gui.search_best.plan_for", broken)
    view = _best(store.path)
    plan, frequency, rfz = view["plan"], view["frequency"], view["rfz"]
    assert isinstance(plan, dict) and isinstance(frequency, dict) and isinstance(rfz, dict)
    assert plan["error"] == "讀不到：平面圖欄位無效"
    assert not frequency["error"] and not rfz["error"]


def test_cache_reads_once_invalidates_and_does_not_write_or_reevaluate(tmp_path: Path, best_result: SchemeResult,
                                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    from aosr.gui.search_view import _document
    store = best_store(tmp_path, best_result)
    path = store.candidate_path(0)
    cache = BestCache()
    reads: list[Path] = []
    def tracked(target: Path) -> dict[str, object]:
        reads.append(target)
        return _document(target)
    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("顯示不能寫檔、拿搜尋鎖或重評")
    before = _snapshot(store.path)
    monkeypatch.setattr("aosr.gui.search_best._document", tracked)
    with monkeypatch.context() as patch:
        for symbol in ("aosr.reporting.evaluation.reevaluate", "aosr.gui.app.load_result",
                       "aosr.gui.app.build_result_view", "fcntl.flock", "pathlib.Path.write_text", "pathlib.Path.write_bytes"):
            patch.setattr(symbol, forbidden)
        first = _best(store.path, cache=cache)
        assert _best(store.path, cache=cache) == first
    assert reads == [path]
    assert _snapshot(store.path) == before
    path.write_text("{")
    second = _best(store.path, cache=cache)
    assert second["version"] != first["version"]
    frequency = second["frequency"]
    assert isinstance(frequency, dict) and frequency["error"].startswith("讀不到：")
    assert reads == [path, path]
    assert _best(store.path, cache=cache) == second
    assert reads == [path, path]


def test_cache_is_bounded(tmp_path: Path, best_result: SchemeResult, monkeypatch: pytest.MonkeyPatch) -> None:
    from aosr.gui.search_view import _document
    from aosr.reporting.result import save_result
    capacity = 2
    paths = [tmp_path / f"result-{i}.json" for i in range(capacity + 1)]
    for path in paths:
        save_result(best_result, path)
    reads: list[Path] = []
    def tracked(path: Path) -> dict[str, object]:
        reads.append(path)
        return _document(path)
    monkeypatch.setattr("aosr.gui.search_best._document", tracked)
    cache = BestCache(capacity)
    for path in paths:
        cache.read(path)
    cache.read(paths[0])
    assert reads == [*paths, paths[0]]


@pytest.mark.parametrize("winner", ["baseline", 7])
def test_default_prefers_refine_and_can_select_search(tmp_path: Path, best_result: SchemeResult, winner: Literal["baseline"] | int) -> None:
    store = best_store(tmp_path, best_result, winner)
    status = build_search_view(store.path, server_physics="測試", server_program="測試")
    assert status.best_default == "refine"
    assert status.best_versions["refine"].candidate == str(winner)
    assert status.best_versions["refine"].result_file == str(store.refine_result_path(None if winner == "baseline" else winner).relative_to(store.path))
    assert _best(store.path, "refine")["which"] == "refine"
    assert _best(store.path)["which"] == "search"
    store.candidate_path(0).write_text("{")
    refined = _best(store.path, "refine")
    for key in ("frequency", "rfz", "plan"):
        chart = refined[key]
        assert isinstance(chart, dict) and not chart["error"]


def test_search_default_tracks_new_winner_and_axis(tmp_path: Path, best_result: SchemeResult) -> None:
    from aosr.gui.labels import LOW_FREQUENCY_AXES
    from aosr.reporting.result import save_result
    from aosr.search import ledger
    from aosr.search.run import SearchStatus
    from aosr.search.store import candidate_name
    store = best_store(tmp_path, best_result)
    first = build_search_view(store.path, server_physics="測試", server_program="測試")
    assert first.best_default == "search"
    row = ledger.read_for(store).rows[0].model_copy(update={"trial_number": 3, "score": 0.1, "result_file": candidate_name(3)})
    ledger.Ledger.open(store.ledger_path).append(row)
    save_result(best_result, store.candidate_path(3))
    store.status_path.write_text(SearchStatus(state="budget_exhausted", best_trial=3, best_score=0.1).model_dump_json())
    second = build_search_view(store.path, server_physics="測試", server_program="測試")
    assert second.best_versions["search"].candidate == "3"
    assert second.best_versions["search"].version != first.best_versions["search"].version
    view = _best(store.path)
    assert "搜尋第一名／試算 3" in str(view["title"])
    assert LOW_FREQUENCY_AXES[best_result.pairs[0].report.top.low_frequency_axis.value] in str(view["title"])


@pytest.mark.parametrize(("trial", "score"), [(0, 0.4), (9, 0.5)])
def test_search_best_must_match_ledger_row(tmp_path: Path, best_result: SchemeResult, trial: int, score: float) -> None:
    from aosr.search.run import SearchStatus
    store = best_store(tmp_path, best_result)
    # 狀態檔說的第一名在搜尋帳裡找不到、或分數對不上（可能正在寫）：不畫任何一份結果冒充第一名。
    store.status_path.write_text(SearchStatus(state="budget_exhausted", best_trial=trial, best_score=score).model_dump_json())
    version = build_search_view(store.path, server_physics="測試", server_program="測試").best_versions["search"]
    assert "不一致" in version.error and not version.result_file
    view = _best(store.path)
    for key in ("frequency", "rfz", "plan"):
        chart = view[key]
        assert isinstance(chart, dict) and "不一致" in chart["error"]


def test_refine_not_started_says_no_winner_yet(tmp_path: Path, best_result: SchemeResult) -> None:
    store = best_store(tmp_path, best_result)
    assert not store.refine_ledger_path.exists()
    version = build_search_view(store.path, server_physics="測試", server_program="測試").best_versions["refine"]
    assert version.error == "讀不到：尚無細算第一名"


@pytest.mark.parametrize("broken", ["missing", "json", "schema"])
def test_unreadable_result_clears_every_chart(tmp_path: Path, best_result: SchemeResult, broken: str) -> None:
    store = best_store(tmp_path, best_result)
    path = store.candidate_path(0)
    if broken == "missing":
        path.unlink()
    elif broken == "json":
        path.write_text("{")
    else:
        path.write_text(best_result.model_dump_json().replace("aosr.scheme_result.v4", "future"))
    before = _snapshot(store.path)
    view = _best(store.path)
    for key in ("frequency", "rfz", "plan"):
        chart = view[key]
        assert isinstance(chart, dict) and chart["error"].startswith("讀不到：")
        assert set(chart) == {"error"}
    assert _snapshot(store.path) == before
