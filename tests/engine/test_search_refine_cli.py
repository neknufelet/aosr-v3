"""細算命令的拒跑、退出碼與獨立停止記號；不啟動物理子行程。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from aosr.config.paths import config_path
from aosr.search import cli
from aosr.search.run import Compute, RefineStatus, SearchStatus, _write_status
from aosr.search.store import SearchStore
from tests.engine._search_refine_cases import RefineCompute, stopped_store
from tests.engine._search_run_cases import FakeCompute, make_store, run


def invoke(store: SearchStore, registry: Path, compute: Compute, monkeypatch: pytest.MonkeyPatch) -> int:
    monkeypatch.setattr(cli, "config_path", lambda name: registry if name.startswith("quality_targets") else config_path(name))
    monkeypatch.setattr(cli, "_identity", lambda purpose, capabilities: store.identity)
    return cli.main(["refine", str(store.path), "--engine-commit", "requested", "--capabilities", str(registry)],
                    compute_factory=lambda opened, capabilities, commit: compute)


@pytest.mark.parametrize("state", ["running", "failed", "interrupted", "user_stopped"])
def test_refuse_unfinished_search_without_mutation(
    tmp_path: Path, state: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    store, registry = stopped_store(tmp_path)
    status = SearchStatus.model_validate_json(store.status_path.read_bytes())
    _write_status(store, SearchStatus.model_validate(status.model_dump() | {"state": state}))
    before = store.status_path.read_bytes()
    compute = RefineCompute(store, {})
    code = invoke(store, registry, compute, monkeypatch)
    assert code == 1
    assert store.status_path.read_bytes() == before
    assert not store.refine_dir.exists() and not store.refine_ledger_path.exists()
    assert not compute.jobs and state in capsys.readouterr().err


@pytest.mark.parametrize("state", ["stopped", "failed", "interrupted"])
def test_refuse_terminal_refinement_without_mutation(
    tmp_path: Path, state: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    store, registry = stopped_store(tmp_path)
    status = SearchStatus.model_validate_json(store.status_path.read_bytes())
    refine = RefineStatus.model_validate({"state": state, "stop_reason": "stable" if state == "stopped" else None})
    _write_status(store, status.model_copy(update={"refine": refine}))
    before = store.status_path.read_bytes()
    code = invoke(store, registry, RefineCompute(store, {}), monkeypatch)
    assert code == 1
    assert store.status_path.read_bytes() == before
    assert not store.refine_dir.exists() and state in capsys.readouterr().err


def test_refuse_missing_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store, registry = make_store(tmp_path, budget=2)
    run(store, registry, FakeCompute(store))
    before = store.status_path.read_bytes()
    code = invoke(store, registry, RefineCompute(store, {}), monkeypatch)
    assert code == 1
    assert store.status_path.read_bytes() == before and not store.refine_dir.exists()


def test_markers_are_separate_and_stale_refine_marker_is_cleared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, registry = stopped_store(tmp_path, budget=3)
    code = cli.main(["stop", str(store.path)])
    assert code == 0
    code = cli.main(["stop", str(store.path), "--refine"])
    assert code == 0
    assert store.stop_path.is_file() and store.refine_stop_path.is_file()
    code = invoke(store, registry, RefineCompute(store, {}), monkeypatch)
    assert code == 0
    status = SearchStatus.model_validate_json(store.status_path.read_bytes())
    assert status.refine.stop_reason == "refine_budget"
    assert not store.refine_stop_path.exists() and store.stop_path.exists()
    assert "殘留" in status.refine.message and "刪除" in status.refine.message


def test_identity_before_start_interrupts_without_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, registry = stopped_store(tmp_path)
    monkeypatch.setattr(cli, "config_path", lambda name: registry if name.startswith("quality_targets") else config_path(name))
    monkeypatch.setattr(cli, "_identity", lambda purpose, capabilities: replace(store.identity, program_fingerprint="changed"))
    compute = RefineCompute(store, {})
    before = SearchStatus.model_validate_json(store.status_path.read_bytes())
    code = cli.main(["refine", str(store.path), "--engine-commit", "test"],
                    compute_factory=lambda store, capabilities, commit: compute)
    assert code == 3
    after = SearchStatus.model_validate_json(store.status_path.read_bytes())
    assert after.refine.state == "interrupted" and not compute.jobs
    assert not store.refine_ledger_path.exists() and not store.refine_dir.exists()
    assert after.model_dump(exclude={"refine"}) == before.model_dump(exclude={"refine"})


def test_compute_failure_preserves_original_error_and_exit_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    store, registry = stopped_store(tmp_path)
    def broken(jobs: object, workers: int) -> object:
        raise RuntimeError("child original error")
    from typing import cast
    code = invoke(store, registry, cast(Compute, broken), monkeypatch)
    assert code == 1
    status = SearchStatus.model_validate_json(store.status_path.read_bytes())
    assert status.refine.state == "failed" and "child original error" in status.refine.message
    assert "child original error" in capsys.readouterr().err


def test_marker_requested_during_batch_stops_before_next_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from collections.abc import Iterator, Sequence
    from aosr.search.run import CandidateJob, ComputedCandidate
    from aosr.search.refine import RefineLedger

    store, registry = stopped_store(tmp_path)
    fake = RefineCompute(store, {})
    def compute(jobs: Sequence[CandidateJob], workers: int) -> Iterator[ComputedCandidate]:
        for result in fake(jobs, workers):
            if result.job.trial_number is not None:
                code = cli.main(["stop", str(store.path), "--refine"])
                assert code == 0
            yield result
    code = invoke(store, registry, compute, monkeypatch)
    assert code == 0
    status = SearchStatus.model_validate_json(store.status_path.read_bytes())
    assert status.refine.stop_reason == "user_stopped"
    assert status.refine.refined == store.settings.batch_size
    assert [row.trial_number for row in RefineLedger.read(store.refine_ledger_path)[1]] == [job.trial_number for job in fake.jobs]


@pytest.mark.parametrize("target", ["baseline", "candidate"])
def test_returned_identity_interrupts_without_recording_changed_result(
    tmp_path: Path, target: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from collections.abc import Iterator, Sequence
    from aosr.search.run import CandidateJob, ComputedCandidate
    from aosr.search.refine import RefineLedger

    store, registry = stopped_store(tmp_path)
    fake = RefineCompute(store, {})
    def compute(jobs: Sequence[CandidateJob], workers: int) -> Iterator[ComputedCandidate]:
        for result in fake(jobs, workers):
            if (result.job.trial_number is None) == (target == "baseline"):
                result = replace(result, identity=replace(store.identity, physics_identity="changed"))
            yield result
    code = invoke(store, registry, compute, monkeypatch)
    assert code == 3
    status = SearchStatus.model_validate_json(store.status_path.read_bytes())
    assert status.refine.state == "interrupted"
    rows = RefineLedger.read(store.refine_ledger_path)[1]
    assert [row.trial_number for row in rows] == ([] if target == "baseline" else [None])


def test_converged_search_can_be_refined(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store, registry = stopped_store(tmp_path, budget=3)
    status = SearchStatus.model_validate_json(store.status_path.read_bytes())
    _write_status(store, status.model_copy(update={"state": "converged"}))
    code = invoke(store, registry, RefineCompute(store, {}), monkeypatch)
    assert code == 0
    assert SearchStatus.model_validate_json(store.status_path.read_bytes()).state == "converged"
