"""细算接續核快取與表頭，丟棄半列；完成順序不能改變重排。"""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from aosr.search.ledger import read_for
from aosr.search.refine import RefineLedger, refine_order
from aosr.search.run import CandidateJob, ComputedCandidate, SearchStatus
from aosr.search.store import SearchIdentity, SearchStore
from tests.engine._search_refine_cases import RefineCompute, refine, stopped_store
from tests.engine._search_run_cases import ENGINE, RUN_DATE, Killed


def clone(store: SearchStore, tmp_path: Path) -> SearchStore:
    target = tmp_path / store.search_id
    shutil.copytree(store.path, target)
    return SearchStore.open(target)


@pytest.mark.parametrize("partial", [False, True])
def test_killed_resume_matches_uninterrupted_and_repairs_tail(tmp_path: Path, partial: bool) -> None:
    store, registry = stopped_store(tmp_path / "input", budget=7)
    whole = clone(store, tmp_path / "whole")
    expected = refine(whole, registry, RefineCompute(whole, {None: 4.0}))
    killed = RefineCompute(store, {None: 4.0}, kill_after=3)
    with pytest.raises(Killed):
        refine(store, registry, killed)
    recorded = RefineLedger.read(store.refine_ledger_path)[1]
    assert SearchStatus.model_validate_json(store.status_path.read_bytes()).refine.state == "running"
    if partial:
        with store.refine_ledger_path.open("ab") as handle:
            handle.write(b'{"round":')
    continued = RefineCompute(store, {None: 4.0})
    status = refine(store, registry, continued)
    assert {job.trial_number for job in continued.jobs}.isdisjoint(row.trial_number for row in recorded)
    order = refine_order(read_for(store).rows)
    finished = sum(row.trial_number is not None for row in recorded)
    boundary = (finished // store.settings.batch_size + 1) * store.settings.batch_size
    assert continued.batches[0] == order[finished:boundary]
    assert store.refine_ledger_path.read_bytes() == whole.refine_ledger_path.read_bytes()
    assert status == expected


def test_resume_mid_batch_finishes_batch_before_judging_stop(tmp_path: Path) -> None:
    """半批時已滿足停止條件（連續 1 個）也要先補完那一批再判；接續後的帳跟一次跑完的逐位相同。"""
    store, registry = stopped_store(tmp_path / "input", batch=3, convergence=1)
    order = refine_order(read_for(store).rows)
    values = {None: 4.0, order[0]: 1.0, order[1]: 1.0, order[2]: 1.0}
    whole = clone(store, tmp_path / "whole")
    expected = refine(whole, registry, RefineCompute(whole, values))
    assert expected.refine.refined == store.settings.batch_size
    with pytest.raises(Killed):
        refine(store, registry, RefineCompute(store, values, kill_after=3))
    # 被砍在半批：原方案與這一批的前兩個已落帳。
    assert [row.trial_number for row in RefineLedger.read(store.refine_ledger_path)[1]] == [None, *order[:2]]
    status = refine(store, registry, RefineCompute(store, values))
    assert store.refine_ledger_path.read_bytes() == whole.refine_ledger_path.read_bytes()
    assert status == expected


def test_single_and_multi_completion_order_is_identical(tmp_path: Path) -> None:
    single, registry = stopped_store(tmp_path / "single", workers=1)
    multi, other_registry = stopped_store(tmp_path / "multi", workers=4)
    one, four = RefineCompute(single, {None: 4.0}), RefineCompute(multi, {None: 4.0})
    first, second = refine(single, registry, one), refine(multi, other_registry, four)
    assert RefineLedger.read(single.refine_ledger_path)[1] == RefineLedger.read(multi.refine_ledger_path)[1]
    assert single.refine_ledger_path.read_bytes().split(b"\n", 1)[1] == multi.refine_ledger_path.read_bytes().split(b"\n", 1)[1]
    assert first.refine == second.refine and one.batches == four.batches


@pytest.mark.parametrize("damage", ["missing", "unreadable", "identity", "header"])
def test_resume_validates_saved_results_and_header(tmp_path: Path, damage: str) -> None:
    store, registry = stopped_store(tmp_path)
    with pytest.raises(Killed):
        refine(store, registry, RefineCompute(store, {}, kill_after=2))
    order = refine_order(read_for(store).rows)
    result = store.refine_result_path(order[0])
    before = store.refine_ledger_path.read_bytes()
    if damage == "missing":
        result.unlink()
    elif damage == "unreadable":
        result.write_text("{")
    elif damage == "identity":
        result.write_text(json.dumps(json.loads(result.read_bytes()) | {"physics_identity": "other"}))
    else:
        lines = before.splitlines(keepends=True)
        lines[0] = (json.dumps(json.loads(lines[0]) | {"project_fingerprint": "other"}) + "\n").encode()
        store.refine_ledger_path.write_bytes(b"".join(lines))
    damaged = store.refine_ledger_path.read_bytes()
    compute = RefineCompute(store, {})
    status = refine(store, registry, compute)
    if damage in ("identity", "header"):
        assert status.refine.state == "interrupted" and not compute.jobs
        assert store.refine_ledger_path.read_bytes() == damaged
    else:
        assert status.refine.state == "stopped"
        assert order[0] in [job.trial_number for job in compute.jobs]
        assert None not in [job.trial_number for job in compute.jobs]


def test_batch_boundary_identity_interrupts_without_new_rows(tmp_path: Path) -> None:
    from aosr.search.refine_run import refine_search

    store, registry = stopped_store(tmp_path)
    def probe() -> SearchIdentity:
        if store.refine_ledger_path.exists() and RefineLedger.read(store.refine_ledger_path)[1]:
            return replace(store.identity, physics_identity="changed")
        return store.identity
    compute = RefineCompute(store, {})
    status = refine_search(store, compute=compute, probe=probe, registry_path=registry,
                           run_date=RUN_DATE, engine_version=ENGINE)
    assert status.refine.state == "interrupted"
    assert [job.trial_number for job in compute.jobs] == [None]
    assert [row.trial_number for row in RefineLedger.read(store.refine_ledger_path)[1]] == [None]


@pytest.mark.parametrize("field", [
    "search_id", "settings_fingerprint", "project_fingerprint", "purpose_fingerprint",
    "physics_identity", "program_fingerprint", "ledger_version",
])
def test_resume_header_every_cell_is_checked(tmp_path: Path, field: str) -> None:
    store, registry = stopped_store(tmp_path)
    with pytest.raises(Killed):
        refine(store, registry, RefineCompute(store, {}, kill_after=2))
    lines = store.refine_ledger_path.read_bytes().splitlines(keepends=True)
    lines[0] = (json.dumps(json.loads(lines[0]) | {field: "other"}) + "\n").encode()
    store.refine_ledger_path.write_bytes(b"".join(lines))
    before = store.refine_ledger_path.read_bytes()
    compute = RefineCompute(store, {})
    status = refine(store, registry, compute)
    assert status.refine.state == "interrupted" and not compute.jobs
    assert store.refine_ledger_path.read_bytes() == before


@pytest.mark.parametrize("number", [None, "candidate"])
def test_resume_saved_baseline_and_candidate_identity_mismatch(
    tmp_path: Path, number: str | None,
) -> None:
    store, registry = stopped_store(tmp_path)
    with pytest.raises(Killed):
        refine(store, registry, RefineCompute(store, {}, kill_after=2))
    trial = None if number is None else refine_order(read_for(store).rows)[0]
    path = store.refine_result_path(trial)
    document = json.loads(path.read_bytes()) | {"program_fingerprint": "other"}
    path.write_text(json.dumps(document))
    before = store.refine_ledger_path.read_bytes()
    compute = RefineCompute(store, {})
    status = refine(store, registry, compute)
    assert status.refine.state == "interrupted" and not compute.jobs
    assert store.refine_ledger_path.read_bytes() == before
    assert ("原方案" if trial is None else f"{trial} 號") in status.refine.message


def test_identity_failure_at_resume_preserves_saved_progress(tmp_path: Path) -> None:
    from aosr.search.refine_run import refine_search

    store, registry = stopped_store(tmp_path, batch=3)
    with pytest.raises(Killed):
        refine(store, registry, RefineCompute(store, {}, kill_after=4))
    before = SearchStatus.model_validate_json(store.status_path.read_bytes()).refine
    status = refine_search(store, compute=RefineCompute(store, {}),
                           probe=lambda: replace(store.identity, physics_identity="changed"),
                           registry_path=registry, run_date=RUN_DATE, engine_version=ENGINE)
    assert status.refine.state == "interrupted"
    assert status.refine.model_dump(exclude={"state", "message"}) == before.model_dump(exclude={"state", "message"})


def test_identity_is_remeasured_before_later_candidate_batch(tmp_path: Path) -> None:
    from aosr.search.refine_run import refine_search

    store, registry = stopped_store(tmp_path)
    order = refine_order(read_for(store).rows)
    def probe() -> SearchIdentity:
        if store.refine_ledger_path.exists():
            numbers = [row.trial_number for row in RefineLedger.read(store.refine_ledger_path)[1]]
            if numbers == [None, *order[:store.settings.batch_size]]:
                return replace(store.identity, program_fingerprint="changed")
        return store.identity
    compute = RefineCompute(store, {})
    status = refine_search(store, compute=compute, probe=probe, registry_path=registry,
                           run_date=RUN_DATE, engine_version=ENGINE)
    assert status.refine.state == "interrupted" and status.refine.refined == store.settings.batch_size
    assert [job.trial_number for job in compute.jobs] == [None, *order[:store.settings.batch_size]]


def test_baseline_compute_has_its_own_identity_boundary(tmp_path: Path) -> None:
    from aosr.search.refine_run import refine_search

    store, registry = stopped_store(tmp_path)
    identities = iter((store.identity, replace(store.identity, physics_identity="changed")))
    compute = RefineCompute(store, {})
    status = refine_search(store, compute=compute, probe=lambda: next(identities), registry_path=registry,
                           run_date=RUN_DATE, engine_version=ENGINE)
    assert status.refine.state == "interrupted" and not compute.jobs
    assert not RefineLedger.read(store.refine_ledger_path)[1]
