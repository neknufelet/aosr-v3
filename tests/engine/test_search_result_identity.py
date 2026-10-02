"""每個計算交回結果在落帳與回報取樣器前核對搜尋快照。"""

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import pytest

from aosr.search.run import CandidateJob, ComputedCandidate, start_search
from aosr.search.sampler import Outcome, SamplerAdapter
from aosr.search.store import SearchIdentity
from tests.engine._search_run_cases import ENGINE, RUN_DATE, FakeCompute, make_store, rows
from tests.engine._search_store_cases import purpose_settings


def changed_identity(identity: SearchIdentity, field: str) -> SearchIdentity:
    if field == "physics_identity":
        return replace(identity, physics_identity="phys-v1:" + "f" * 64)
    if field == "program_fingerprint":
        return replace(identity, program_fingerprint="calc-v1:" + "f" * 64)
    return replace(identity, purpose_settings=purpose_settings(identity.purpose_settings.purpose))


@pytest.mark.parametrize("field", ("physics_identity", "program_fingerprint", "purpose_settings", "all"))
@pytest.mark.parametrize("target", (None, 0, 1))
def test_changed_result_interrupts_before_ledger_and_feedback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, target: int | None,
) -> None:
    store, registry = make_store(tmp_path, budget=2, batch=1)
    fake = FakeCompute(store)
    feedback: list[int] = []
    original = SamplerAdapter.tell_batch

    def tell(adapter: SamplerAdapter, outcomes: Mapping[int, Outcome]) -> None:
        feedback.extend(outcomes)
        original(adapter, outcomes)

    def compute(jobs: Sequence[CandidateJob], workers: int) -> Iterator[ComputedCandidate]:
        for result in fake(jobs, workers):
            if result.job.trial_number == target:
                identity = store.identity
                for name in fields:
                    identity = changed_identity(identity, name)
                result = replace(result, identity=identity)
            yield result

    fields = ("physics_identity", "program_fingerprint", "purpose_settings") if field == "all" else (field,)
    monkeypatch.setattr(SamplerAdapter, "tell_batch", tell)
    status = start_search(store, compute=compute, probe=lambda: store.identity,
                          registry_path=registry, run_date=RUN_DATE, engine_version=ENGINE)
    assert status.state == "interrupted"
    assert ("原方案" if target is None else f"試算 {target}") in status.message
    assert all(name in status.message for name in fields)
    assert target not in {row.trial_number for row in rows(store)}
    assert target not in feedback
    if target == 1:
        assert status.asked == store.settings.budget
        assert any(row.trial_number == 0 for row in rows(store))


def test_changed_result_in_converging_batch_interrupts(tmp_path: Path) -> None:
    store, registry = make_store(tmp_path, budget=3, batch=1, convergence=1)
    fake = FakeCompute(store, flat=True)

    def compute(jobs: Sequence[CandidateJob], workers: int) -> Iterator[ComputedCandidate]:
        for result in fake(jobs, workers):
            if result.job.trial_number == 1:
                result = replace(result, identity=changed_identity(store.identity, "program_fingerprint"))
            yield result

    status = start_search(store, compute=compute, probe=lambda: store.identity,
                          registry_path=registry, run_date=RUN_DATE, engine_version=ENGINE)
    assert status.state == "interrupted"
    assert 1 not in {row.trial_number for row in rows(store)}


def test_identity_change_mid_batch_keeps_only_checked_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, registry = make_store(tmp_path, budget=2, batch=2)
    fake = FakeCompute(store)
    feedback: list[int] = []
    original = SamplerAdapter.tell_batch

    def tell(adapter: SamplerAdapter, outcomes: Mapping[int, Outcome]) -> None:
        feedback.extend(outcomes)
        original(adapter, outcomes)

    def compute(jobs: Sequence[CandidateJob], workers: int) -> Iterator[ComputedCandidate]:
        for job in jobs:
            for result in fake((job,), workers):
                if job.trial_number == 1:
                    result = replace(result, identity=changed_identity(store.identity, "program_fingerprint"))
                yield result

    monkeypatch.setattr(SamplerAdapter, "tell_batch", tell)
    status = start_search(store, compute=compute, probe=lambda: store.identity,
                          registry_path=registry, run_date=RUN_DATE, engine_version=ENGINE)
    assert status.state == "interrupted"
    recorded = rows(store)
    assert recorded
    assert all(row.trial_number == 0 for row in recorded)
    assert not feedback
