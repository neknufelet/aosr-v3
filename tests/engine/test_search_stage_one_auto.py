"""第一階段自動外圈的接續、停止窄縫、誠實出口與入選後檢查。"""
from collections.abc import Iterator, Sequence
from datetime import date
from pathlib import Path
import json

import pytest

from aosr.search import cli
from aosr.search.outer_status import OUTER_MESSAGES, OuterStatus, snapshot_of
from aosr.search.run import CandidateJob, ComputeFailed, IdentityChanged, IdentityProbe, SearchStatus
from aosr.search.stage_one_refine_record import StageOneRefineStatus
from aosr.search.stage_one_worker import StageOneComputed, StageOneComputeFunction
from aosr.search.store import SearchStore
from tests.engine import _search_stage_one_cases as search
from tests.engine._stage_one_auto_cases import invoke, no_compute, read, running
from tests.engine._stage_one_post_check_cases import calls, diagnosis, prepared, runner
from tests.engine._stage_one_refine_cases import pool, snapshot
from tests.engine.test_search_stage_one_refine import ROWS


def test_running_search_finishes_refine_and_verification_diagnosis(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = running(tmp_path, monkeypatch)
    root = tmp_path / "child"
    command = runner(root, store)
    fake = search.FakeCompute(store, monkeypatch)
    def compute(jobs: Sequence[CandidateJob], workers: int) -> Iterator[StageOneComputed]:
        for item in fake(jobs, workers):
            if item.job.result_path.parent == store.refine_dir:
                number = item.job.trial_number
                assert number is not None
                (root / f"{number}.json").write_text(diagnosis(store, number).model_dump_json())
            yield item
    code = invoke(store, tmp_path, monkeypatch, compute=compute, runner=command)
    assert not code
    status = read(store)
    assert status.state == "budget_exhausted" and status.outer.conclusion == "complete"
    assert status.round == SearchStatus().round  # 不追加回饋或新一輪搜尋
    refined = json.loads((store.path / "stage_one_refine.json").read_bytes())
    checked = json.loads((store.path / "stage-one-check" / "summary.json").read_bytes())
    assert refined["state"] == checked["state"] == "done"
    assert {entry["trial_number"] for entry in refined["results"]} == {entry["trial_number"] for entry in checked["seats"]}
    assert calls(root)
    for argv in calls(root):
        number = int(Path(argv[0]).name.split("-")[1])
        assert Path(argv[0]) == store.path / "verification" / f"trial-{number:06d}-scheme.json"
    assert all(not (store.path / name).exists() for name in (
        "modal-diagnosis", "crossover-sensitivity", "placement-stability"))


@pytest.mark.parametrize("state", ["converged", "budget_exhausted"])
def test_completed_auto_reuses_all_computation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state: str) -> None:
    store = prepared(tmp_path, monkeypatch)
    store.status_path.write_text(SearchStatus.model_validate({"state": state}).model_dump_json())
    code = invoke(store, tmp_path, monkeypatch, runner=runner(tmp_path / "child", store))
    assert not code
    before = calls(tmp_path / "child")
    code = cli.main(["auto", str(store.path), "--engine-commit", "test", "--modal-cache-dir", str(tmp_path / "cache")],
                    compute_factory=no_compute, modal_runner=("must-not-run",))
    assert not code
    assert read(store).outer.conclusion == "complete"
    assert calls(tmp_path / "child") == before
    checked = json.loads((store.path / "stage-one-check" / "summary.json").read_bytes())
    assert checked["state"] == "done" and all(s["state"] == "diagnosed_not_scored" for s in checked["seats"])


def test_selected_batch_resumes_without_recomputing_saved_seat(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = pool(tmp_path, monkeypatch, ROWS)
    fake = search.FakeCompute(store, monkeypatch)
    def killed(jobs: Sequence[CandidateJob], workers: int) -> Iterator[StageOneComputed]:
        yield next(fake(jobs, workers))
        raise search.Killed()
    with pytest.raises(search.Killed):
        invoke(store, tmp_path, monkeypatch, compute=killed)
    summary = store.path / "stage_one_refine.json"
    assert json.loads(summary.read_bytes())["state"] == "selected"
    resumed = search.FakeCompute(store, monkeypatch)
    root = tmp_path / "child"
    command = runner(root, store)
    def compute(jobs: Sequence[CandidateJob], workers: int) -> Iterator[StageOneComputed]:
        for item in resumed(jobs, workers):
            number = item.job.trial_number
            assert number is not None
            (root / f"{number}.json").write_text(diagnosis(store, number).model_dump_json())
            yield item
    code = invoke(store, tmp_path, monkeypatch, compute=compute, runner=command)
    assert not code
    assert read(store).outer.conclusion == "complete"
    assert not set(fake.calls) & set(resumed.calls)
    assert set(fake.calls + resumed.calls) == {n for n, _, _, _ in ROWS}
    assert json.loads(summary.read_bytes())["state"] == "done"
    assert json.loads((store.path / "stage-one-check" / "summary.json").read_bytes())["state"] == "done"


@pytest.mark.parametrize("marker", ["search", "refine"])
@pytest.mark.parametrize("saved", [False, True])
def test_existing_marker_preserves_refine_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                               marker: str, saved: bool) -> None:
    store = prepared(tmp_path, monkeypatch) if saved else pool(tmp_path, monkeypatch, ROWS)
    path = store.path / "stage_one_refine.json"
    before = path.read_bytes() if path.exists() else None
    stop = store.stop_path if marker == "search" else store.refine_stop_path
    stop.touch()
    code = invoke(store, tmp_path, monkeypatch, compute=no_compute())
    assert not code
    assert read(store).outer.conclusion == "user_stopped" and stop.exists()
    assert (path.read_bytes() if path.exists() else None) == before
    assert not (store.path / "stage-one-check").exists()


def test_refine_marker_during_search_prevents_any_refine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = running(tmp_path, monkeypatch)
    fake = search.FakeCompute(store, monkeypatch)
    def compute(jobs: Sequence[CandidateJob], workers: int) -> Iterator[StageOneComputed]:
        assert all(job.result_path.parent != store.refine_dir for job in jobs)
        store.refine_stop_path.touch()
        yield from fake(jobs, workers)
    code = invoke(store, tmp_path, monkeypatch, compute=compute)
    assert not code
    assert read(store).outer.conclusion == "user_stopped" and store.refine_stop_path.exists()
    assert not (store.path / "stage_one_refine.json").exists()
    assert not store.refine_dir.exists() and not (store.path / "stage-one-check").exists()


@pytest.mark.parametrize("marker", ["search", "refine"])
@pytest.mark.parametrize("saved", [False, True])
def test_marker_after_auto_check_preserves_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, saved: bool,
                                                 marker: str) -> None:
    from aosr.search import stage_one_refine
    from aosr.search.stage_one_refine import refine_stage_one_search as real
    store = prepared(tmp_path, monkeypatch) if saved else pool(tmp_path, monkeypatch, ROWS)
    path = store.path / "stage_one_refine.json"
    if saved:
        doc = json.loads(path.read_bytes()); doc["state"] = "selected"; path.write_text(json.dumps(doc))
    before = path.read_bytes() if path.exists() else None
    stop = store.stop_path if marker == "search" else store.refine_stop_path
    # 真細算入口前的最後窄縫，外圈已讀完記號與摘要；兩種記號都要照停（複查）。
    def entry(opened: SearchStore, *, compute: StageOneComputeFunction, probe: IdentityProbe,
              registry_path: Path, run_date: date, engine_version: str,
              keep_stop_marker: bool = False) -> StageOneRefineStatus:
        stop.touch()
        return real(opened, compute=compute, probe=probe, registry_path=registry_path, run_date=run_date,
                    engine_version=engine_version, keep_stop_marker=keep_stop_marker)
    monkeypatch.setattr(stage_one_refine, "refine_stage_one_search", entry)
    code = invoke(store, tmp_path, monkeypatch, compute=no_compute())
    assert not code
    assert read(store).outer.conclusion == "user_stopped" and stop.exists()
    assert (path.read_bytes() if path.exists() else None) == before
    assert not (store.path / "stage-one-check").exists()


@pytest.mark.parametrize("phase,state,conclusion,expected", [
    ("search", "failed", "search_failed", 1), ("search", "interrupted", "search_interrupted", 3),
    ("refine", "failed", "refine_failed", 1), ("refine", "interrupted", "refine_interrupted", 3)])
def test_failed_or_interrupted_phase_skips_post_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], phase: str, state: str, conclusion: str, expected: int) -> None:
    store = running(tmp_path, monkeypatch) if phase == "search" else pool(tmp_path, monkeypatch, ROWS)
    def fail(jobs: Sequence[CandidateJob], workers: int) -> Iterator[StageOneComputed]:
        if state == "interrupted":
            raise IdentityChanged("替身中斷") if phase == "search" else KeyboardInterrupt()
        raise ComputeFailed("替身失敗")
        yield
    code = invoke(store, tmp_path, monkeypatch, compute=fail)
    assert code == expected
    assert read(store).outer.conclusion == conclusion
    assert not (store.path / "stage-one-check").exists()
    assert "入選後檢查這次不跑" in capsys.readouterr().err


@pytest.mark.parametrize("state,conclusion,expected", [
    ("user_stopped", "user_stopped", 0), ("failed", "search_failed", 1), ("interrupted", "search_interrupted", 3)])
def test_saved_search_terminal_never_computes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                             state: str, conclusion: str, expected: int) -> None:
    store = pool(tmp_path, monkeypatch, ROWS)
    store.status_path.write_text(SearchStatus.model_validate({"state": state}).model_dump_json())
    code = invoke(store, tmp_path, monkeypatch, compute=no_compute())
    assert code == expected and read(store).outer.conclusion == conclusion
    assert not (store.path / "stage-one-check").exists()


@pytest.mark.parametrize("state,expected", [("failed", 1), ("interrupted", 3)])
def test_saved_refine_terminal_never_computes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                             state: str, expected: int) -> None:
    store = prepared(tmp_path, monkeypatch)
    path = store.path / "stage_one_refine.json"
    doc = json.loads(path.read_bytes()); doc["state"] = state; path.write_text(json.dumps(doc))
    code = invoke(store, tmp_path, monkeypatch, compute=no_compute())
    assert code == expected and read(store).outer.conclusion == f"refine_{state}"
    assert not (store.path / "stage-one-check").exists()


@pytest.mark.parametrize("state", ["failed", "interrupted"])
def test_stop_has_priority_over_saved_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state: str) -> None:
    store = pool(tmp_path, monkeypatch, ROWS)
    store.status_path.write_text(SearchStatus.model_validate({"state": state}).model_dump_json())
    store.stop_path.touch()
    code = invoke(store, tmp_path, monkeypatch, compute=no_compute())
    assert not code and read(store).outer.conclusion == "user_stopped"


@pytest.mark.parametrize("outcome", ["failed", "stopped", "exception", "interrupt"])
def test_post_check_problems_preserve_complete_and_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                       capsys: pytest.CaptureFixture[str], outcome: str) -> None:
    from aosr.search import stage_one_post_check
    store = prepared(tmp_path, monkeypatch)
    command: tuple[str, ...]
    if outcome in ("exception", "interrupt"):
        def fail(*args: object, **kwargs: object) -> object:
            raise KeyboardInterrupt if outcome == "interrupt" else OSError("替身入口出錯")
        monkeypatch.setattr(stage_one_post_check, "run_stage_one_post_check", fail)
        command = ("must-not-run",)
    else:
        command = runner(tmp_path / "child", store, state="failed" if outcome == "failed" else "diagnosed_not_scored",
                         stop_after=outcome == "stopped")
    code = invoke(store, tmp_path, monkeypatch, compute=no_compute(), runner=command)
    assert not code and read(store).outer.conclusion == "complete"
    error = capsys.readouterr().err
    assert "入選後檢查" in error and "細算結果不受影響" in error
    assert ("已停止" if outcome in ("stopped", "interrupt") else "失敗") in error


def test_selection_integrity_rejection_does_not_fail_search(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                         capsys: pytest.CaptureFixture[str]) -> None:
    store = prepared(tmp_path, monkeypatch)
    summary = store.path / "stage_one_refine.json"
    doc = json.loads(summary.read_bytes()); doc["state"] = "selected"; summary.write_text(json.dumps(doc))
    lines = store.ledger_path.read_text().splitlines()
    row = json.loads(lines[1]); row["score"] += .1; lines[1] = json.dumps(row)
    store.ledger_path.write_text("\n".join(lines) + "\n")
    # 上一次外圈留下的結論；這次的不跑原因不准拿它來講。
    old = read(store)
    store.status_path.write_text(old.model_copy(update={"outer": OuterStatus(conclusion="user_stopped",
        message=OUTER_MESSAGES["user_stopped"], snapshot=snapshot_of(old))}).model_dump_json())
    before = store.status_path.read_bytes()
    code = invoke(store, tmp_path, monkeypatch, compute=no_compute())
    assert code == 1
    error = capsys.readouterr().err
    assert "自動外圈失敗：挑席跟上次不同" in error
    assert "入選後檢查這次不跑：自動外圈沒有正常收尾" in error and OUTER_MESSAGES["user_stopped"] not in error
    assert store.status_path.read_bytes() == before and read(store).state != "failed"
    assert not (store.path / "stage-one-check").exists()


def test_missing_auto_cache_argparse_preserves_every_byte(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = pool(tmp_path, monkeypatch, ROWS)
    before = snapshot(store)
    with pytest.raises(SystemExit) as error:
        cli.main(["auto", str(store.path), "--engine-commit", "test"], compute_factory=no_compute)
    assert error.value.code == 2
    assert snapshot(store) == before


def test_exception_with_new_stop_marker_concludes_user_stop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = running(tmp_path, monkeypatch)
    def fail(jobs: Sequence[CandidateJob], workers: int) -> Iterator[StageOneComputed]:
        store.refine_stop_path.touch()
        raise ComputeFailed("停止同時交回失敗")
        yield
    code = invoke(store, tmp_path, monkeypatch, compute=fail)
    assert not code and read(store).outer.conclusion == "user_stopped"
    assert store.refine_stop_path.exists() and not (store.path / "stage_one_refine.json").exists()
    assert not (store.path / "stage-one-check").exists()
