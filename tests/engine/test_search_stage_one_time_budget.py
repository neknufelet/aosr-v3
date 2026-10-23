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
        assert status.message == (f"因預算停止：搜尋時間預算 {budget_s / HOUR:g} 小時用完（到這一批結束，搜尋已用 {batches:.2f} 小時；"
                                  "批界才停，最後一批可能跨過）；第一名是本次預算內最佳，不代表已收斂，也不代表找到全域最佳")


def _killed_then_resumed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, kill_after: int,
                         saved_s: float) -> tuple[SearchStatus, TimedCompute]:
    """先被砍（狀態檔留著先前的搜尋秒數），再接續；接續時從狀態檔讀回秒數一起算。"""
    from aosr.search.stage_one_run import resume_stage_one_search
    clock = [0.]
    monkeypatch.setattr(timings, "_now", lambda: clock[0])
    store = _store(tmp_path, {"search_time_budget_s": 2 * HOUR})
    with pytest.raises(search.Killed):
        search.run(store, search.FakeCompute(store, monkeypatch, kill_after=kill_after))
    saved = SearchStatus.model_validate_json(store.status_path.read_bytes())
    store.status_path.write_text(saved.model_copy(update={"search_seconds": {1: saved_s}}).model_dump_json())
    compute = TimedCompute(store, monkeypatch, clock)
    status = resume_stage_one_search(store, compute=compute, probe=lambda: store.identity,
        registry_path=config_path("quality_targets.toml"), run_date=cases.run_kwargs()["run_date"],
        engine_version=store.identity.program_fingerprint)
    return status, compute


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


def test_settings_fields_are_optional_stage_one_only_and_keep_old_fingerprints(tmp_path: Path) -> None:
    store = search.make_store(tmp_path)
    document = store.settings.model_dump(mode="json")
    assert "search_time_budget_s" not in document and "stage_one_candidate_cap_s" not in document
    assert store.settings.stage_one_search_cap_s == 600.
    configured = SearchSettings.model_validate(document | {"search_time_budget_s": 3 * HOUR, "stage_one_candidate_cap_s": 450.})
    assert configured.stage_one_search_cap_s == 450. and configured.fingerprint != store.settings.fingerprint
    assert SearchSettings.model_validate(configured.model_dump(mode="json")) == configured
    second_stage = {key: value for key, value in document.items() if key != "stage"}
    for field in ("search_time_budget_s", "stage_one_candidate_cap_s"):
        with pytest.raises(ValueError, match="只准第一階段"):
            SearchSettings.model_validate(second_stage | {field: 60.})
        for bad in (0., -1.):
            with pytest.raises(ValueError):
                SearchSettings.model_validate(document | {field: bad})
