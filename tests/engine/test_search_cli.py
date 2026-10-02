"""搜尋命令列以注入計算驗狀態與離開碼，不跑完整物理。"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest

from aosr.search.run import CandidateJob, Compute, ComputedCandidate, SearchStatus
from aosr.search.store import SearchIdentity, SearchStore
from tests.engine._search_run_cases import FakeCompute, make_store


def start_args(tmp_path: Path) -> list[str]:
    store, _ = make_store(tmp_path / "input", budget=2, batch=1)
    project, settings = tmp_path / "project", tmp_path / "settings"
    project.write_text(store.project.model_dump_json())
    settings.write_text(store.settings.model_dump_json())
    return ["start", "--project", str(project), "--settings", str(settings),
            "--root", str(tmp_path / "output"), "--engine-commit", "test"]


def opened(tmp_path: Path) -> SearchStore:
    folder, = (tmp_path / "output").iterdir()
    return SearchStore.open(folder)


def fake_factory(store: SearchStore, capabilities: Path, commit: str) -> Compute:
    return FakeCompute(store)


def test_stop_creates_marker_and_is_silent(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from aosr.search.cli import main

    store, _ = make_store(tmp_path)
    code = main(["stop", str(store.path)])
    assert code == 0
    assert store.stop_path.is_file()
    assert capsys.readouterr() == ("", "")


def test_start_writes_snapshots_ledger_status_and_rejects_stopped_resume(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    from aosr.search.cli import main
    from aosr.search.ledger import read_for

    code = main(start_args(tmp_path), compute_factory=fake_factory)
    assert code == 0
    store = opened(tmp_path)
    status = SearchStatus.model_validate_json(store.status_path.read_bytes())
    assert status.state == "budget_exhausted" and status.asked == store.settings.budget
    assert read_for(store).rows and store.baseline_path.is_file()
    assert store.identity.purpose_settings.purpose == store.project.purpose
    assert all(store.versions[name] for name in ("python", "optuna", "numpy"))
    assert capsys.readouterr() == ("", "")
    code = main(["resume", str(store.path), "--engine-commit", "test"], compute_factory=fake_factory)
    assert code == 1
    assert SearchStatus.model_validate_json(store.status_path.read_bytes()) == status
    out, err = capsys.readouterr()
    assert out == "" and err.startswith("搜尋失敗：")


def test_start_with_broken_settings_reports_error_and_creates_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """搜尋資料夾還沒建好就出錯：錯誤原文寫到標準錯誤，不留半個資料夾。"""
    from aosr.search.cli import main

    args = start_args(tmp_path)
    settings = Path(args[args.index("--settings") + 1])
    settings.write_text("{", encoding="utf-8")
    code = main(args, compute_factory=fake_factory)
    assert code == 1
    out, err = capsys.readouterr()
    assert out == "" and err.startswith("搜尋失敗：")
    assert not (tmp_path / "output").exists() or not any((tmp_path / "output").iterdir())


@pytest.mark.parametrize("changed", [False, True])
def test_compute_errors_write_state_and_exit_code(tmp_path: Path, changed: bool) -> None:
    from aosr.search.cli import main
    from aosr.search.worker import IdentityChanged

    def factory(store: SearchStore, capabilities: Path, commit: str) -> Compute:
        def compute(jobs: Sequence[CandidateJob], workers: int) -> Iterator[ComputedCandidate]:
            raise IdentityChanged("child changed") if changed else RuntimeError("child failed")
            yield
        return compute

    assert main(start_args(tmp_path), compute_factory=factory) == (2 if changed else 1)
    status = SearchStatus.model_validate_json(opened(tmp_path).status_path.read_bytes())
    assert status.state == ("interrupted" if changed else "failed")
    assert ("child changed" if changed else "child failed") in status.message


@pytest.mark.parametrize("state", ["converged", "user_stopped"])
def test_normal_stop_reasons_exit_zero(tmp_path: Path, state: str) -> None:
    from aosr.search.cli import main
    from aosr.search.settings import SearchSettings

    args = start_args(tmp_path)
    settings_path = Path(args[args.index("--settings") + 1])
    settings = SearchSettings.model_validate_json(settings_path.read_bytes())
    settings_path.write_text(settings.model_copy(update={"convergence_run": 1}).model_dump_json())

    def factory(store: SearchStore, capabilities: Path, commit: str) -> Compute:
        if state == "user_stopped":
            store.stop_path.touch()
        return FakeCompute(store, flat=True)

    code = main(args, compute_factory=factory)
    assert code == 0
    assert SearchStatus.model_validate_json(opened(tmp_path).status_path.read_bytes()).state == state


def test_resume_running_search_finishes_missing_work(tmp_path: Path) -> None:
    from aosr.search.cli import main
    from tests.engine._search_run_cases import Killed

    def killed_factory(store: SearchStore, capabilities: Path, commit: str) -> Compute:
        return FakeCompute(store, fail_after=0, kill=True, persist_baseline=True)

    with pytest.raises(Killed):
        main(start_args(tmp_path), compute_factory=killed_factory)
    store = opened(tmp_path)
    assert SearchStatus.model_validate_json(store.status_path.read_bytes()).state == "running"
    code = main(["resume", str(store.path), "--engine-commit", "test"], compute_factory=fake_factory)
    assert code == 0
    status = SearchStatus.model_validate_json(store.status_path.read_bytes())
    assert status.state == "budget_exhausted" and status.asked == store.settings.budget


def test_probe_remeasures_identity_between_batches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from dataclasses import replace
    from aosr.search import cli
    from tests.engine._search_run_cases import rows

    original = cli._identity

    def factory(store: SearchStore, capabilities: Path, commit: str) -> Compute:
        def probe(purpose: str, capabilities_path: Path) -> SearchIdentity:
            identity = original(purpose, capabilities_path)
            return replace(identity, program_fingerprint="changed") if rows(store) else identity
        monkeypatch.setattr(cli, "_identity", probe)
        return FakeCompute(store)

    code = cli.main(start_args(tmp_path), compute_factory=factory)
    assert code == 2
    store = opened(tmp_path)
    status = SearchStatus.model_validate_json(store.status_path.read_bytes())
    assert status.state == "interrupted" and status.asked == store.settings.batch_size
