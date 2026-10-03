"""搜尋迴圈考卷：完成順序不能決定回報、最佳或收斂。"""

import importlib.util
from pathlib import Path

from aosr.search.sampler import SamplerAdapter, SamplerSettings


def test_search_loop_contract_is_available(tmp_path: Path) -> None:
    assert tmp_path.is_dir()
    assert importlib.util.find_spec("aosr.search.run") is not None
    assert importlib.util.find_spec("aosr.search.scoring") is not None


def test_enqueue_validates_and_precedes_asking(tmp_path: Path) -> None:
    import pytest

    assert tmp_path.is_dir()
    adapter = SamplerAdapter({"x": (0.0, 1.0)}, SamplerSettings(1, 2))
    with pytest.raises(ValueError):
        adapter.enqueue({"wrong": 0.5})
    with pytest.raises(ValueError):
        adapter.enqueue({"x": 2.0})
    adapter.enqueue({"x": 0})
    proposal, = adapter.ask_batch(1)
    assert proposal.params == {"x": 0.0}
    assert type(proposal.params["x"]) is float
    with pytest.raises(RuntimeError):
        adapter.enqueue({"x": 0.5})

from dataclasses import replace

import pytest

from aosr.search import layout
from aosr.search.layout_settings import Box, Span
from aosr.search.ledger import Ledger, read_for
from aosr.search.store import SearchIdentity, candidate_name
from tests.engine._search_run_cases import (
    ENGINE, RUN_DATE, FakeCompute, Killed, make_store, next_params, rows, run,
)


def test_single_and_multi_worker_runs_are_bit_identical(tmp_path: Path) -> None:
    single, registry = make_store(tmp_path / "single", workers=1, budget=41)
    multi, other_registry = make_store(tmp_path / "multi", workers=4, budget=41)
    one, four = FakeCompute(single), FakeCompute(multi)
    first, second = run(single, registry, one), run(multi, other_registry, four)
    assert rows(single) == rows(multi)
    assert first == second
    assert next_params(single) == next_params(multi)
    assert one.calls != four.calls
    assert first.best_trial == min((row.score, row.trial_number) for row in rows(single) if row.score is not None)[1]


def test_illegal_placements_never_reach_compute(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    box = Box(x=Span(low=1.5, high=2.5), y=Span(low=0.01, high=4.0), z=Span(low=0.01, high=2.0))
    store, registry = make_store(tmp_path, layout_changes={"keep_out": (box,)})
    from aosr.search import constraints
    from aosr.reporting.scheme import Scheme

    original = layout.to_scheme

    def checked(project: Scheme, placement: layout.Placement, scheme_id: str) -> Scheme:
        assert not constraints.check(project, store.settings.layout, placement)
        return original(project, placement, scheme_id)

    monkeypatch.setattr(layout, "to_scheme", checked)
    compute = FakeCompute(store)
    status = run(store, registry, compute)
    illegal = [row for row in rows(store) if row.outcome == "illegal"]
    assert illegal
    assert any(row.outcome == "scored" for row in rows(store))
    assert status.illegal == len(illegal)
    assert status.computed == sum(row.outcome != "illegal" for row in rows(store))
    assert status.illegal_reasons
    for row in illegal:
        assert row.reason and row.violation_m and row.violation_m > 0.0
        assert row.trial_number not in compute.calls
        assert row.result_file is None and row.seconds == 0.0


def test_missing_category_candidate_is_not_favoured(tmp_path: Path) -> None:
    store, registry = make_store(tmp_path)
    status = run(store, registry, FakeCompute(store, missing=frozenset({0})))
    first = rows(store)[0]
    assert first.outcome == "excluded" and first.reason == "unassessed"
    assert first.score is None
    assert status.best_trial != first.trial_number
    assert status.excluded["unassessed"] == sum(row.reason == "unassessed" for row in rows(store))


def test_incomparable_candidate_is_excluded_by_pinned_identity(tmp_path: Path) -> None:
    store, registry = make_store(tmp_path)
    status = run(store, registry, FakeCompute(store, different=frozenset({0})))
    first = rows(store)[0]
    assert first.outcome == "excluded" and first.reason == "incomparable"
    assert first.score is None and status.best_trial != first.trial_number
    assert "incomparable" in status.excluded


def test_baseline_that_cannot_be_ranked_fails_the_search(tmp_path: Path) -> None:
    store, registry = make_store(tmp_path)
    compute = FakeCompute(store, missing=frozenset({None}))
    status = run(store, registry, compute)
    assert status.state == "failed"
    assert status.message == "原方案缺類：timbre_balance（mandatory_category_missing），所以定不出比較身分"
    assert status.baseline_outcome == "unassessed"
    assert not rows(store) and not any(number is not None for number in compute.calls)
    assert store.baseline_path.exists()


def test_program_change_midway_marks_interrupted(tmp_path: Path) -> None:
    from aosr.search.run import start_search

    store, registry = make_store(tmp_path)
    compute = FakeCompute(store)

    def probe() -> SearchIdentity:
        return store.identity if not rows(store) else replace(store.identity, program_fingerprint="changed")

    status = start_search(store, compute=compute, probe=probe, registry_path=registry,
                          run_date=RUN_DATE, engine_version=ENGINE)
    assert status.state == "interrupted" and "身分" in status.message
    assert {row.batch_index for row in rows(store)} == {0}
    assert status.asked == store.settings.batch_size


def test_user_stop_marker_stops_before_next_batch(tmp_path: Path) -> None:
    from aosr.search.run import start_search

    store, registry = make_store(tmp_path)

    def probe() -> SearchIdentity:
        if rows(store):
            store.stop_path.touch()
        return store.identity

    status = start_search(store, compute=FakeCompute(store), probe=probe, registry_path=registry,
                          run_date=RUN_DATE, engine_version=ENGINE)
    assert status.state == "user_stopped"
    assert {row.batch_index for row in rows(store)} == {0}


def test_budget_counts_every_asked_trial_and_last_batch_is_short(tmp_path: Path) -> None:
    box = Box(x=Span(low=1.5, high=2.5), y=Span(low=0.01, high=4.0), z=Span(low=0.01, high=2.0))
    store, registry = make_store(tmp_path, budget=11, batch=3, layout_changes={"keep_out": (box,)})
    status = run(store, registry, FakeCompute(store))
    recorded = rows(store)
    assert status.state == "budget_exhausted"
    assert status.illegal > 0 and status.computed > 0
    assert status.asked == store.settings.budget == status.computed + status.illegal
    assert [row.trial_number for row in recorded] == list(range(store.settings.budget))
    last = [row for row in recorded if row.batch_index == max(item.batch_index for item in recorded)]
    assert len(last) == store.settings.budget % store.settings.batch_size


def test_convergence_stops_after_streak(tmp_path: Path) -> None:
    store, registry = make_store(tmp_path, convergence=4, budget=30)
    status = run(store, registry, FakeCompute(store, flat=True))
    assert status.state == "converged"
    assert status.best_trial == 0
    assert status.streak == status.computed - 1
    assert status.streak >= store.settings.convergence_run
    assert status.asked < store.settings.budget


def test_stop_messages_do_not_claim_convergence(tmp_path: Path) -> None:
    """2026-10-03（#585）：停止條件是工程設定，訊息寫連續幾個沒改善、不寫已收斂；預算用完只稱本次預算內最佳。"""
    store, registry = make_store(tmp_path, convergence=4, budget=30)
    stopped = run(store, registry, FakeCompute(store, flat=True))
    assert stopped.state == "converged"
    assert f"連續 {store.settings.convergence_run} 個候選" in stopped.message and "暫行" in stopped.message
    assert "不代表找到全域最佳" in stopped.message and "已收斂" not in stopped.message
    other, other_registry = make_store(tmp_path / "budget", budget=6, batch=3)
    exhausted = run(other, other_registry, FakeCompute(other))
    assert exhausted.state == "budget_exhausted"
    assert "本次預算內最佳" in exhausted.message and "已收斂" not in exhausted.message


def test_compute_failure_fails_the_search(tmp_path: Path) -> None:
    from aosr.search.run import SearchStatus

    store, registry = make_store(tmp_path)
    compute = FakeCompute(store, fail_after=1)
    with pytest.raises(RuntimeError, match="計算失敗"):
        run(store, registry, compute)
    status = SearchStatus.model_validate_json(store.status_path.read_bytes())
    assert status.state == "failed" and "計算失敗" in status.message
    assert status.computed == sum(row.outcome != "illegal" for row in rows(store))
    assert len(set(compute.calls)) == len(compute.calls)


@pytest.mark.parametrize("keep_baseline", (True, False))
def test_resume_after_crash_continues_bit_identically(tmp_path: Path, keep_baseline: bool) -> None:
    from aosr.search.run import resume_search

    whole, registry = make_store(tmp_path / "whole", budget=41)
    split, split_registry = make_store(tmp_path / "split", budget=41)
    expected = run(whole, registry, FakeCompute(whole))
    crashed = FakeCompute(split, fail_after=5, kill=True, persist_baseline=True)
    with pytest.raises(Killed):
        run(split, split_registry, crashed)
    saved = rows(split)
    if not keep_baseline:
        split.baseline_path.unlink()
    compute = FakeCompute(split)
    status = resume_search(split, compute=compute, probe=lambda: split.identity,
                           registry_path=split_registry, run_date=RUN_DATE, engine_version=ENGINE)
    assert rows(whole) == rows(split)
    assert status == expected
    assert (None not in compute.calls) == keep_baseline
    assert not {row.trial_number for row in saved} & set(compute.calls)
    assert next_params(whole) == next_params(split)


def test_resume_refuses_a_ledger_from_another_search(tmp_path: Path) -> None:
    from aosr.search.run import resume_search

    first, registry = make_store(tmp_path / "first")
    second, other_registry = make_store(tmp_path / "second")
    run(first, registry, FakeCompute(first))
    second.ledger_path.write_bytes(first.ledger_path.read_bytes())
    compute = FakeCompute(second)
    status = resume_search(second, compute=compute, probe=lambda: second.identity,
                           registry_path=other_registry, run_date=RUN_DATE, engine_version=ENGINE)
    assert status.state == "interrupted" and "表頭" in status.message
    assert not compute.calls


def test_finished_search_cannot_resume(tmp_path: Path) -> None:
    from aosr.search.run import resume_search

    store, registry = make_store(tmp_path)
    run(store, registry, FakeCompute(store))
    with pytest.raises(ValueError, match="停止"):
        resume_search(store, compute=FakeCompute(store), probe=lambda: store.identity,
                      registry_path=registry, run_date=RUN_DATE, engine_version=ENGINE)


def test_enqueued_start_is_the_first_trial(tmp_path: Path) -> None:
    store, registry = make_store(tmp_path / "inside")
    status = run(store, registry, FakeCompute(store))
    start = layout.standard_start(store.project, store.settings.layout)
    assert start is not None and status.start_enqueued
    expected = layout.unit_from_params(start, store.settings.layout)
    assert rows(store)[0].unit_params_hex == {key: value.hex() for key, value in expected.items()}
    outside, other_registry = make_store(tmp_path / "outside", layout_changes={"spacing_m": Span(low=1.5, high=2.0)})
    other = run(outside, other_registry, FakeCompute(outside))
    assert not other.start_enqueued
    assert "起點不在搜尋範圍內，沒有排入" in other.message


def test_resume_checks_partial_batch_parameters_bitwise(tmp_path: Path) -> None:
    import json
    from aosr.search.run import resume_search

    store, registry = make_store(tmp_path)
    with pytest.raises(Killed):
        run(store, registry, FakeCompute(store, fail_after=1, kill=True, persist_baseline=True))
    lines = store.ledger_path.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[-1])
    import math

    original = float.fromhex(row["unit_params_hex"]["front_distance"])
    row["unit_params_hex"]["front_distance"] = math.nextafter(original, 1.0).hex()
    lines[-1] = json.dumps(row)
    store.ledger_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    compute = FakeCompute(store)
    status = resume_search(store, compute=compute, probe=lambda: store.identity,
                           registry_path=registry, run_date=RUN_DATE, engine_version=ENGINE)
    assert status.state == "interrupted" and "參數" in status.message
    assert not compute.calls


def test_resume_snapshot_read_error_is_interrupted(tmp_path: Path) -> None:
    from aosr.search.run import resume_search
    from aosr.search.store import PROJECT_FILE

    store, registry = make_store(tmp_path)
    with pytest.raises(Killed):
        run(store, registry, FakeCompute(store, fail_after=1, kill=True, persist_baseline=True))
    (store.path / PROJECT_FILE).unlink()
    status = resume_search(store, compute=FakeCompute(store), probe=lambda: store.identity,
                           registry_path=registry, run_date=RUN_DATE, engine_version=ENGINE)
    assert status.state == "interrupted" and "快照" in status.message


def test_fake_baseline_is_rankable(tmp_path: Path) -> None:
    from aosr.config.quality_targets import load_quality_targets
    from aosr.scoring.ranking import RankingContext, rank_candidates
    from aosr.search.run import CandidateJob

    store, registry = make_store(tmp_path)
    job = CandidateJob(None, store.project, store.baseline_path)
    result, = FakeCompute(store)((job,), 1)
    context = RankingContext(purpose=job.scheme.purpose,
                             receiver_set_fingerprint=job.scheme.receiver_set.fingerprint,
                             channel_group_fingerprint=job.scheme.channel_group.fingerprint,
                             run_date=RUN_DATE, engine_version=ENGINE)
    ranking = rank_candidates([result.candidate], load_quality_targets(registry), context)
    assert ranking.rankable, ([(row.candidate_id, row.missing) for row in ranking.not_evaluated],
                              [(row.candidate_id, row.reasons) for row in ranking.eliminated])


def test_unscored_trials_do_not_claim_convergence(tmp_path: Path) -> None:
    store, registry = make_store(tmp_path, convergence=2)
    status = run(store, registry, FakeCompute(store, missing=frozenset(range(store.settings.budget))))
    assert status.state == "budget_exhausted"
    assert status.best_trial is None and status.best_score is None
    assert status.asked == store.settings.budget
    # 沒有第一名就不准說「本次預算內最佳」（2026-10-03 停止訊息照實寫的審查）。
    assert "沒有任何候選拿到分數" in status.message and "第一名" not in status.message


def test_resume_incomplete_nonfinal_batch_is_interrupted(tmp_path: Path) -> None:
    from aosr.search.run import resume_search

    store, registry = make_store(tmp_path)
    with pytest.raises(Killed):
        run(store, registry, FakeCompute(store, fail_after=5, kill=True, persist_baseline=True))
    lines = store.ledger_path.read_bytes().splitlines(keepends=True)
    store.ledger_path.write_bytes(b"".join((lines[0], *lines[2:])))
    compute = FakeCompute(store)
    status = resume_search(store, compute=compute, probe=lambda: store.identity,
                           registry_path=registry, run_date=RUN_DATE, engine_version=ENGINE)
    assert status.state == "interrupted" and "重播" in status.message
    assert not compute.calls


def test_resume_checks_complete_batch_parameters_bitwise(tmp_path: Path) -> None:
    import json
    import math
    from aosr.search.run import resume_search

    store, registry = make_store(tmp_path)
    with pytest.raises(Killed):
        run(store, registry, FakeCompute(store, fail_after=5, kill=True, persist_baseline=True))
    lines = store.ledger_path.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[1])
    original = float.fromhex(row["unit_params_hex"]["front_distance"])
    row["unit_params_hex"]["front_distance"] = math.nextafter(original, 1.0).hex()
    lines[1] = json.dumps(row)
    store.ledger_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    compute = FakeCompute(store)
    status = resume_search(store, compute=compute, probe=lambda: store.identity,
                           registry_path=registry, run_date=RUN_DATE, engine_version=ENGINE)
    assert status.state == "interrupted" and "重播" in status.message
    assert not compute.calls


def test_baseline_is_computed_before_sampler_enqueue(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from collections.abc import Iterator, Mapping, Sequence
    from aosr.search.run import CandidateJob, ComputedCandidate, start_search

    store, registry = make_store(tmp_path)
    events: list[str] = []
    original = SamplerAdapter.enqueue

    def enqueue(adapter: SamplerAdapter, params: Mapping[str, float]) -> None:
        events.append("enqueue")
        original(adapter, params)

    fake = FakeCompute(store)

    def compute(jobs: Sequence[CandidateJob], workers: int) -> Iterator[ComputedCandidate]:
        assert read_for(store).header.search_id == store.search_id
        events.append("baseline" if jobs[0].trial_number is None else "trial")
        yield from fake(jobs, workers)

    monkeypatch.setattr(SamplerAdapter, "enqueue", enqueue)
    status = start_search(store, compute=compute, probe=lambda: store.identity,
                          registry_path=registry, run_date=RUN_DATE, engine_version=ENGINE)
    assert status.baseline_outcome == "scored"
    assert events.index("baseline") < events.index("enqueue") < events.index("trial")


def test_resume_compute_replay_exception_is_still_a_failure(tmp_path: Path) -> None:
    from collections.abc import Iterator, Sequence
    from aosr.search.run import CandidateJob, ComputedCandidate, SearchStatus, resume_search
    from aosr.search.sampler import ReplayMismatch

    store, registry = make_store(tmp_path)
    with pytest.raises(Killed):
        run(store, registry, FakeCompute(store, fail_after=1, kill=True, persist_baseline=True))

    def compute(jobs: Sequence[CandidateJob], workers: int) -> Iterator[ComputedCandidate]:
        raise ReplayMismatch("計算端例外")

    with pytest.raises(ReplayMismatch, match="計算端例外"):
        resume_search(store, compute=compute, probe=lambda: store.identity,
                      registry_path=registry, run_date=RUN_DATE, engine_version=ENGINE)
    status = SearchStatus.model_validate_json(store.status_path.read_bytes())
    assert status.state == "failed"


def test_screening_returns_identity_only_for_scored_candidates(tmp_path: Path) -> None:
    from aosr.config.quality_targets import load_quality_targets
    from aosr.search.run import CandidateJob
    from aosr.search.sampler import Excluded, RankingZone, Scored
    from aosr.search.scoring import screening_outcome

    store, registry_path = make_store(tmp_path)
    job = CandidateJob(None, store.project, store.baseline_path)
    result, = FakeCompute(store)((job,), store.settings.max_workers)
    registry = load_quality_targets(registry_path)
    outcome, identity = screening_outcome(result.candidate, job.scheme, registry=registry,
                                          run_date=RUN_DATE, engine_version=ENGINE, pinned=None)
    assert isinstance(outcome, Scored) and identity is not None
    assert [part.category.value for part in identity] == sorted(part.category.value for part in identity)
    different = (identity[0].model_copy(update={"settings_fingerprint": "other"}), *identity[1:])
    excluded, excluded_identity = screening_outcome(result.candidate, job.scheme, registry=registry,
                                                   run_date=RUN_DATE, engine_version=ENGINE, pinned=different)
    assert excluded == Excluded(RankingZone.INCOMPARABLE)
    assert excluded_identity is None


def test_streak_restarts_when_a_later_trial_improves(tmp_path: Path) -> None:
    """中途有沒變好的、之後又變好：連續數要從最後一次變好重算。分數一路變好再持平的序列抓不到這件事。"""
    from aosr.search.ledger import row_from_outcome
    from aosr.search.run import SearchStatus, _progress
    from aosr.search.sampler import Proposal, Scored

    scores = (5.0, 6.0, 7.0, 4.0, 8.0)
    meters = {"front_distance": 1.0, "spacing": 1.2, "listening_distance": 2.0}
    built = tuple(
        row_from_outcome(batch_index=0, proposal=Proposal(number, dict.fromkeys(meters, 0.5)), params_m=meters,
                         outcome=Scored(score), seconds=0.0, result_file=candidate_name(number))
        for number, score in enumerate(scores))
    status = _progress(SearchStatus(), built)
    assert (status.best_trial, status.best_score, status.streak) == (3, 4.0, 1)
    assert tmp_path.is_dir()


@pytest.mark.parametrize("resume", [False, True])
def test_identity_changed_during_compute_marks_interrupted(tmp_path: Path, resume: bool) -> None:
    from collections.abc import Iterator, Sequence
    from aosr.search.run import CandidateJob, ComputedCandidate, SearchStatus, resume_search, start_search
    from aosr.search.run import IdentityChanged

    store, registry = make_store(tmp_path)
    if resume:
        with pytest.raises(Killed):
            run(store, registry, FakeCompute(store, fail_after=0, kill=True, persist_baseline=True))

    def changed(jobs: Sequence[CandidateJob], workers: int) -> Iterator[ComputedCandidate]:
        raise IdentityChanged("identity changed in child")
        yield

    entry = resume_search if resume else start_search
    status = entry(store, compute=changed, probe=lambda: store.identity, registry_path=registry,
                   run_date=RUN_DATE, engine_version=ENGINE)
    assert status.state == "interrupted" and "identity changed in child" in status.message
    assert SearchStatus.model_validate_json(store.status_path.read_bytes()) == status
