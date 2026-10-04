"""假牆鐘驗證各輪累計、被砍接續、舊檔與報告；只替換計算，不解物理。"""

import json
import time
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest
from pydantic import ValidationError

from aosr.config.quality_targets import QualityTargets, load_quality_targets
from aosr.search.feedback import feedback_search
from aosr.search import run as search_run
from aosr.search.ledger import read_for
from aosr.search.outer import conclude
from aosr.search.outer_status import conclusion_message
from aosr.search.report import build_report, render_text
from aosr.search.refine_run import refine_search
from aosr.search.run import CandidateJob, Compute, ComputedCandidate, SearchStatus, resume_search, start_search
from aosr.search.store import SearchStore
from tests.engine._search_refine_cases import RefineCompute, SearchCompute
from tests.engine._search_run_cases import ENGINE, Killed, RUN_DATE, make_store


def execute(store: SearchStore, registry: Path, compute: Compute, command: str) -> SearchStatus:
    operation = {"search": start_search, "resume": resume_search, "refine": refine_search}[command]
    return operation(store, compute=compute, probe=lambda: store.identity,
                     registry_path=registry, run_date=RUN_DATE, engine_version=ENGINE)


class Clock:
    def __init__(self) -> None:
        self.seconds = 1000.0

    def __call__(self) -> float:
        return self.seconds

    def advance(self, seconds: float) -> None:
        self.seconds += seconds


class TimedCompute:
    """每個合成結果推進時鐘；被砍前也經過一段未保存時間。"""

    def __init__(self, compute: Compute, clock: Clock, seconds: float, *, shared_seconds: float = 0.0) -> None:
        self.compute, self.clock, self.seconds = compute, clock, seconds
        self.shared_seconds = shared_seconds
        self.candidate_seconds: dict[int | None, float] = {}

    def __call__(self, jobs: Sequence[CandidateJob], workers: int) -> Iterator[ComputedCandidate]:
        self.clock.advance(self.shared_seconds)
        try:
            for result in self.compute(jobs, workers):
                self.candidate_seconds[result.job.trial_number] = result.seconds
                self.clock.advance(self.seconds)
                yield result
        except Killed:
            self.clock.advance(7.0)
            raise


def test_round_timings_feedback_and_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clock = Clock()
    monkeypatch.setattr(time, "monotonic", clock)
    store, registry = make_store(tmp_path, batch=2, budget=40, convergence=7,
                                refine={"budget": 4, "convergence_run": 50}, feedback={"offset": 0.125})
    first = execute(store, registry, TimedCompute(SearchCompute(store, flat=True, persist_baseline=True), clock, 60.0), "search")
    assert first.timings.search == {1: 540.0}
    first = execute(store, registry, TimedCompute(RefineCompute(store, {None: 4.0, 1: 0.1}), clock, 120.0), "refine")
    assert first.timings.refine == {1: 600.0}
    clock.advance(999.0)
    feedback = feedback_search(store)
    assert feedback.timings == first.timings
    clock.advance(888.0)
    second = execute(store, registry, TimedCompute(SearchCompute(
        store, flat=True, persist_baseline=True, values={first.asked: 0.5, first.asked + 1: 0.5}), clock, 30.0), "resume")
    assert second.timings.search == {1: 540.0, 2: 240.0}
    final = execute(store, registry, TimedCompute(RefineCompute(store, {}), clock, 90.0), "refine")
    assert final.timings.refine == {1: 600.0, 2: 360.0}
    assert final.timings.search == second.timings.search
    text = render_text(build_report(store, quality_targets_path=registry, run_date=RUN_DATE))
    expected = ["各輪搜尋花的時間：第 1 輪 9.0 分、第 2 輪 4.0 分；搜尋合計 13.0 分",
                "各輪細算花的時間：第 1 輪 10.0 分、第 2 輪 6.0 分；細算合計 16.0 分",
                "搜尋＋細算合計 29.0 分（牆鐘，含共用有限元素；被砍後未寫回狀態的那一小段不在內）"]
    assert [line for line in text.splitlines() if "花的時間" in line or line.startswith("搜尋＋細算合計")] == expected
    assert text.index(expected[0]) < text.index("細算做完沒") < text.index(expected[1])
    assert text.index(expected[1]) < text.index(expected[2]) < text.index("\n名次\n")


def test_killed_search_resume_loses_only_unsaved_tail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clock = Clock()
    monkeypatch.setattr(time, "monotonic", clock)
    store, registry = make_store(tmp_path, batch=2, budget=4)
    with pytest.raises(Killed):
        execute(store, registry, TimedCompute(SearchCompute(
            store, persist_baseline=True, fail_after=3, kill=True), clock, 10.0), "search")
    saved = SearchStatus.model_validate_json(store.status_path.read_bytes())
    assert saved.timings.search == {1: 30.0}
    assert clock.seconds == 1047.0
    clock.advance(500.0)
    final = execute(store, registry, TimedCompute(SearchCompute(store, persist_baseline=True), clock, 10.0), "resume")
    assert final.timings.search == {1: 40.0}
    assert final.timings.refine == {}


def test_killed_refine_resume_loses_only_unsaved_tail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clock = Clock()
    monkeypatch.setattr(time, "monotonic", clock)
    store, registry = make_store(tmp_path, batch=2, budget=4, refine={"budget": 4, "convergence_run": 50})
    before = execute(store, registry, SearchCompute(store), "search")
    with pytest.raises(Killed):
        execute(store, registry, TimedCompute(RefineCompute(store, {}, kill_after=4), clock, 10.0), "refine")
    saved = SearchStatus.model_validate_json(store.status_path.read_bytes())
    assert saved.timings.refine == {1: 30.0}
    assert clock.seconds == 1047.0
    clock.advance(500.0)
    final = execute(store, registry, TimedCompute(RefineCompute(store, {}), clock, 10.0), "refine")
    assert final.timings.refine == {1: 40.0}
    assert final.timings.search == before.timings.search


def test_legacy_status_and_report(tmp_path: Path) -> None:
    store, registry = make_store(tmp_path)
    legacy = SearchStatus().model_dump(mode="json", exclude={"timings"})
    store.status_path.write_text(json.dumps(legacy))
    status = SearchStatus.model_validate_json(store.status_path.read_bytes())
    assert status.timings.search == {} and status.timings.refine == {}
    text = render_text(build_report(store, quality_targets_path=registry, run_date=RUN_DATE))
    assert "這個搜尋資料夾沒有時間紀錄（第 585 支之前開的）" in text
    assert "搜尋＋細算合計 0.0 分" not in text


def test_timings_do_not_stale_conclusion_or_change_other_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clock = Clock()
    monkeypatch.setattr(time, "monotonic", clock)
    store, registry = make_store(tmp_path, budget=4, batch=2)
    first = execute(store, registry, TimedCompute(SearchCompute(store), clock, 60.0), "search")
    first = conclude(store, first, "search_failed")
    before = render_text(build_report(store, quality_targets_path=registry, run_date=RUN_DATE))
    changed = first.model_copy(update={"timings": first.timings.model_copy(update={
        "search": {1: 600.0}, "refine": {1: 300.0},
    })})
    store.status_path.write_text(changed.model_dump_json())
    assert conclusion_message(changed) == first.outer.message
    after = render_text(build_report(store, quality_targets_path=registry, run_date=RUN_DATE))
    def without_times(text: str) -> list[str]:
        return [line for line in text.splitlines() if "花的時間" not in line and not line.startswith("搜尋＋細算合計")]
    assert without_times(after) == without_times(before)


@pytest.mark.parametrize("change", [{"extra": {}}, {"search": {0: 2.0}}, {"refine": {1: -1.0}},
                                   {"search": {1: float("inf")}}, {"refine": {1: float("nan")}}])
def test_timings_reject_invalid_values(tmp_path: Path, change: dict[str, object]) -> None:
    store, _ = make_store(tmp_path)
    document = SearchStatus().model_dump() | {"timings": change}
    with pytest.raises(ValidationError):
        SearchStatus.model_validate(document)
    assert not store.status_path.exists()


def test_each_checkpoint_counts_setup_and_prior_write_time(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clock = Clock()
    monkeypatch.setattr(time, "monotonic", clock)
    store, registry = make_store(tmp_path, budget=2, batch=2)
    write = search_run._write_status
    checkpoints: list[SearchStatus] = []

    def timed_load(path: Path) -> QualityTargets:
        clock.advance(5.0)
        return load_quality_targets(path)

    def timed_write(store: SearchStore, status: SearchStatus) -> SearchStatus:
        saved = write(store, status)
        checkpoints.append(SearchStatus.model_validate_json(store.status_path.read_bytes()))
        clock.advance(2.0)
        return saved

    monkeypatch.setattr(search_run, "load_quality_targets", timed_load)
    monkeypatch.setattr(search_run, "_write_status", timed_write)
    final = execute(store, registry, TimedCompute(SearchCompute(store), clock, 10.0), "search")
    assert [status.timings.search for status in checkpoints] == [{1: 5.0}, {1: 17.0}, {1: 39.0}, {1: 41.0}]
    assert final.timings.search == {1: 41.0}
    assert clock.seconds == 1043.0


def test_unstarted_refine_with_search_timings_does_not_claim_legacy_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = Clock()
    monkeypatch.setattr(time, "monotonic", clock)
    store, registry = make_store(tmp_path, budget=2, batch=2)
    execute(store, registry, TimedCompute(SearchCompute(store), clock, 60.0), "search")
    text = render_text(build_report(store, quality_targets_path=registry, run_date=RUN_DATE))
    assert "各輪細算花的時間：尚無紀錄" in text
    assert "第 585 支之前開的" not in text
    assert "搜尋＋細算合計 3.0 分（牆鐘，含共用有限元素；被砍後未寫回狀態的那一小段不在內）" in text


def test_timings_partial_defaults_and_zero_is_a_record(tmp_path: Path) -> None:
    store, registry = make_store(tmp_path)
    document = SearchStatus().model_dump(mode="json") | {"timings": {"search": {1: 0.0}}}
    status = SearchStatus.model_validate(document)
    store.status_path.write_text(status.model_dump_json())
    assert status.timings.refine == {}
    text = render_text(build_report(store, quality_targets_path=registry, run_date=RUN_DATE))
    assert "各輪搜尋花的時間：第 1 輪 0.0 分；搜尋合計 0.0 分" in text
    assert "第 585 支之前開的" not in text


def test_shared_compute_time_is_included_without_changing_candidate_seconds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = Clock()
    monkeypatch.setattr(time, "monotonic", clock)
    store, registry = make_store(tmp_path, budget=2, batch=2, refine={"budget": 2, "convergence_run": 50})
    compute = TimedCompute(SearchCompute(store), clock, 10.0, shared_seconds=120.0)
    first = execute(store, registry, compute, "search")
    assert first.timings.search == {1: 270.0}
    assert {row.trial_number: row.seconds for row in read_for(store).rows} == {
        number: seconds for number, seconds in compute.candidate_seconds.items() if number is not None
    }
    final = execute(store, registry, TimedCompute(RefineCompute(store, {}), clock, 10.0, shared_seconds=120.0), "refine")
    assert final.timings.refine == {1: 270.0}
    assert final.timings.search == first.timings.search


def test_report_rounds_minutes_after_summing_seconds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clock = Clock()
    monkeypatch.setattr(time, "monotonic", clock)
    store, registry = make_store(tmp_path, budget=2, batch=2, refine={"budget": 2, "convergence_run": 50})
    execute(store, registry, TimedCompute(SearchCompute(store), clock, 1.01), "search")
    execute(store, registry, TimedCompute(RefineCompute(store, {}), clock, 1.01), "refine")
    text = render_text(build_report(store, quality_targets_path=registry, run_date=RUN_DATE))
    assert [line for line in text.splitlines() if "花的時間" in line or line.startswith("搜尋＋細算合計")] == [
        "各輪搜尋花的時間：第 1 輪 0.1 分；搜尋合計 0.1 分",
        "各輪細算花的時間：第 1 輪 0.1 分；細算合計 0.1 分",
        "搜尋＋細算合計 0.1 分（牆鐘，含共用有限元素；被砍後未寫回狀態的那一小段不在內）",
    ]
