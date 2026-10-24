"""第一階段搜尋的時間預算（老闆 2026-10-23：搜尋階段三小時、到預算就停、標因預算停止）與設定檔的單題上限（#902）。"""
from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest

from aosr.search import timings
from aosr.search.run import CandidateJob, SearchStatus
from aosr.search.settings import SearchSettings
from aosr.search.stage_one_worker import StageOneComputed
from aosr.search.store import SearchStore
from aosr.config.paths import config_path
from tests.engine import _search_stage_one_cases as search
from tests.engine import _stage_one_cases as cases

HOUR = 3600.


class TimedCompute(search.FakeCompute):
    """每一批算完，替身時鐘走一小時：搜尋牆鐘只跟批數有關，答案可以從批數推。"""

    def __init__(self, store: SearchStore, monkeypatch: pytest.MonkeyPatch, clock: list[float]) -> None:
        super().__init__(store, monkeypatch)
        self.clock = clock
        self.batches: list[int] = []

    def __call__(self, jobs: Sequence[CandidateJob], workers: int) -> Iterator[StageOneComputed]:
        yield from super().__call__(jobs, workers)
        self.clock[0] += HOUR
        self.batches.append(len(jobs))


def _store(tmp_path: Path, extra: dict[str, object]) -> SearchStore:
    base = search.make_store(tmp_path / "base", budget=12)
    settings = SearchSettings.model_validate(base.settings.model_dump(mode="json") | extra)
    return SearchStore.create(tmp_path / "configured", project=base.project, settings=settings, identity=base.identity,
                              versions={"python": "test", "numpy": "test", "optuna": "test"})


def _time_message(budget_s: float, used_h: float) -> str:
    return (f"因預算停止：搜尋時間預算 {budget_s / HOUR:g} 小時用完（到這一批結束，搜尋已用 {used_h:.2f} 小時；"
            "批界才停，最後一批可能跨過）；第一名是本次預算內最佳，不代表已收斂，也不代表找到全域最佳")


def _run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extra: dict[str, object]) -> tuple[SearchStatus, TimedCompute]:
    clock = [0.]
    monkeypatch.setattr(timings, "_now", lambda: clock[0])
    store = _store(tmp_path, extra)
    compute = TimedCompute(store, monkeypatch, clock)
    return search.run(store, compute), compute


# 每批 3 題、題數上限 12（四批）、每批一小時：剛好 2 小時算用完（等號也停）、多半秒就再跑一批、沒設就跑到題數上限。
@pytest.mark.parametrize("budget_s,batches", [(2 * HOUR, 2), (2 * HOUR + .5, 3), (None, 4)])
def test_time_budget_stops_at_batch_boundary_as_budget_exhausted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                 budget_s: float | None, batches: int) -> None:
    status, compute = _run(tmp_path, monkeypatch, {} if budget_s is None else {"search_time_budget_s": budget_s})
    assert compute.batches == [3] * batches
    assert status.state == "budget_exhausted" and status.asked == 3 * batches
    assert sum(status.search_seconds.values()) == batches * HOUR
    if budget_s is None:
        assert status.message == "因預算停止（第一名是本次預算內最佳，不代表找到全域最佳）"
    else:
        assert status.message == _time_message(budget_s, batches)


@pytest.mark.parametrize("convergence_run", [1, 2])
def test_time_budget_wins_when_the_same_batch_also_reaches_the_no_improvement_stop(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, convergence_run: int) -> None:
    # 對照組：同設定不設時間預算，第 k 批結束時達到連續沒改善；設時間預算 k 小時，第 k 批兩個都成立，要標因預算停止。
    reference, before = _run(tmp_path / "reference", monkeypatch, {"convergence_run": convergence_run})
    assert reference.state == "converged" and len(before.batches) < 4
    hours = len(before.batches)
    status, compute = _run(tmp_path / "timed", monkeypatch,
                           {"convergence_run": convergence_run, "search_time_budget_s": hours * HOUR})
    assert compute.batches == before.batches and status.streak == reference.streak >= convergence_run
    assert status.state == "budget_exhausted" and status.message == _time_message(hours * HOUR, hours)


def _first_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extra: dict[str, object], *,
               kill_after: int | None) -> tuple[SearchStore, SearchStatus | None]:
    """第一段跑的時候鐘不走；kill_after 有值就在那一題被砍，沒值就跑到停並交回停下來的狀態。"""
    monkeypatch.setattr(timings, "_now", lambda: 0.)
    store = _store(tmp_path, {"search_time_budget_s": 2 * HOUR} | extra)
    first = search.FakeCompute(store, monkeypatch, kill_after=kill_after)
    if kill_after is None:
        return store, search.run(store, first)
    with pytest.raises(search.Killed):
        search.run(store, first)
    return store, None


def _resume(store: SearchStore, monkeypatch: pytest.MonkeyPatch) -> tuple[SearchStatus, TimedCompute]:
    from aosr.search.stage_one_run import resume_stage_one_search
    clock = [0.]
    monkeypatch.setattr(timings, "_now", lambda: clock[0])
    compute = TimedCompute(store, monkeypatch, clock)
    status = resume_stage_one_search(store, compute=compute, probe=lambda: store.identity,
        registry_path=config_path("quality_targets.toml"), run_date=cases.run_kwargs()["run_date"],
        engine_version=store.identity.program_fingerprint)
    return status, compute


def _saved_then_resumed(store: SearchStore, monkeypatch: pytest.MonkeyPatch, saved_s: float) -> tuple[SearchStatus, TimedCompute]:
    """狀態檔停在進行中、留著先前的搜尋秒數（被砍在某一次存檔之後），再接續；接續時從狀態檔讀回秒數一起算。"""
    saved = SearchStatus.model_validate_json(store.status_path.read_bytes())
    store.status_path.write_text(saved.model_copy(update={"state": "running", "search_seconds": {1: saved_s}}).model_dump_json())
    return _resume(store, monkeypatch)


def _killed_then_resumed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, kill_after: int,
                         saved_s: float) -> tuple[SearchStatus, TimedCompute]:
    store, _ = _first_run(tmp_path, monkeypatch, {}, kill_after=kill_after)
    return _saved_then_resumed(store, monkeypatch, saved_s)


def test_time_budget_counts_seconds_saved_before_a_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # 第一題就被砍：先前已用 1.5 小時、預算 2 小時，接續後只再跑一批就停。
    status, compute = _killed_then_resumed(tmp_path, monkeypatch, kill_after=0, saved_s=1.5 * HOUR)
    assert status.state == "budget_exhausted" and compute.batches == [3] and status.asked == 3
    assert sum(status.search_seconds.values()) == 2.5 * HOUR


def test_time_budget_finishes_a_partial_batch_before_stopping(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # 第一批算完、第二批只落帳一題就被砍；接續時預算早就用完，也要先把第二批剩的兩題補齊才停（批界才停）。
    status, compute = _killed_then_resumed(tmp_path, monkeypatch, kill_after=4, saved_s=2.5 * HOUR)
    assert status.state == "budget_exhausted" and compute.batches == [2] and status.asked == 6
    assert sum(status.search_seconds.values()) == 3.5 * HOUR


# 被砍在某一批存檔之後、停止那一次存檔之前：狀態檔還是進行中。接續時一題都不用再問，舊的停法也成立，時間用完照樣標因預算停止。
# 第一段問到 6 題就收斂：接續停在迴圈頂；問滿 12 題（題數用完，或最後一批才收斂）：接續不進迴圈、停在迴圈外（複查）。
@pytest.mark.parametrize("extra,stopped_as,asked", [({"convergence_run": 1}, "converged", 6), ({}, "budget_exhausted", 12),
                                                    ({"convergence_run": 3}, "converged", 12)])
def test_time_budget_wins_when_a_resume_starts_with_an_older_stop_already_reached(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extra: dict[str, object], stopped_as: str, asked: int) -> None:
    store, first = _first_run(tmp_path, monkeypatch, extra, kill_after=None)
    assert first is not None and first.state == stopped_as and first.asked == asked and sum(first.search_seconds.values()) == 0.
    status, compute = _saved_then_resumed(store, monkeypatch, 5 * HOUR)
    assert compute.batches == [] and status.asked == first.asked
    assert status.state == "budget_exhausted" and status.message == _time_message(2 * HOUR, 5)


# 狀態檔遺失、帳本已有候選：重建出來的秒數是空的。設了時間預算就拒絕接續（不把追不回來的成本當零）；沒設照舊接續（對照組）。
@pytest.mark.parametrize("budget_s", [2 * HOUR, None])
def test_resume_refuses_a_time_budget_when_the_status_was_lost(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                              budget_s: float | None) -> None:
    store, _ = _first_run(tmp_path, monkeypatch, {"search_time_budget_s": budget_s}, kill_after=4)
    store.status_path.unlink()
    before = {path.relative_to(store.path): path.read_bytes() for path in store.path.rglob("*") if path.is_file()}
    if budget_s is None:
        status, _ = _resume(store, monkeypatch)
        assert status.state == "budget_exhausted" and status.asked == 12 and not status.timed_from_start
        return
    with pytest.raises(ValueError, match="先前用掉的搜尋時間追不回來，拒絕接續"):
        _resume(store, monkeypatch)
    assert {path.relative_to(store.path): path.read_bytes() for path in store.path.rglob("*") if path.is_file()} == before


def test_settings_fields_are_optional_cap_stage_one_only_and_each_enters_the_fingerprint(tmp_path: Path) -> None:
    store = search.make_store(tmp_path)
    document = store.settings.model_dump(mode="json")
    assert "search_time_budget_s" not in document and "stage_one_candidate_cap_s" not in document
    assert store.settings.stage_one_search_cap_s == 600.
    configured = SearchSettings.model_validate(document | {"search_time_budget_s": 3 * HOUR, "stage_one_candidate_cap_s": 450.})
    assert configured.stage_one_search_cap_s == 450. and configured.fingerprint != store.settings.fingerprint
    for field, value in (("search_time_budget_s", 3 * HOUR), ("stage_one_candidate_cap_s", 450.)):
        assert SearchSettings.model_validate(document | {field: value}).fingerprint != store.settings.fingerprint
    assert SearchSettings.model_validate(configured.model_dump(mode="json")) == configured
    # 時間預算 2026-10-24 起第二階段也收（test_search_stage_two_time_budget.py）；單題上限仍只准第一階段。
    second_stage = {key: value for key, value in document.items() if key != "stage"}
    with pytest.raises(ValueError, match="單題搜尋成本上限只准第一階段"):
        SearchSettings.model_validate(second_stage | {"stage_one_candidate_cap_s": 60.})
    for field in ("search_time_budget_s", "stage_one_candidate_cap_s"):
        for bad in (0., -1.):
            with pytest.raises(ValueError):
                SearchSettings.model_validate(document | {field: bad})
