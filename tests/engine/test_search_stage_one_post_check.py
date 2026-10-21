"""入選後檢查獨立保存、串行、判舊與停止；不解模態。"""
from pathlib import Path
import json
import os

import pytest

from aosr.reporting.modal_diagnosis import modal_identity
from aosr.search.store import SearchStore
from aosr.search.stage_one_post_check_record import StageOnePostCheckSummary
from tests.engine._stage_one_post_check_cases import prepared, runner, calls
from tests.engine._stage_one_refine_cases import snapshot


def run(store: SearchStore, root: Path, *, state: str = "diagnosed_not_scored", code: int = 0,
        stop_after: bool = False) -> StageOnePostCheckSummary:
    from aosr.search.stage_one_post_check import run_stage_one_post_check
    return run_stage_one_post_check(store, cache_dir=root / "cache", lock_fd=None,
                                   runner=runner(root, store, state=state, code=code, stop_after=stop_after))


def test_sequential_all_seats_correct_schemes_cache_and_no_search_writes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from aosr.search.stage_one_post_check import run_stage_one_post_check
    store = prepared(tmp_path, monkeypatch)
    before = snapshot(store)
    root = tmp_path / "child"
    descriptor = os.open(store.path, os.O_RDONLY)
    try:
        result = run_stage_one_post_check(store, cache_dir=root / "cache", lock_fd=descriptor, runner=runner(root, store))
    finally:
        os.close(descriptor)
    assert result.state == "done"
    expected = [8, 2, 1, 5]
    assert [seat.trial_number for seat in result.seats] == expected
    assert [Path(args[0]) for args in calls(root)] == [store.scheme_path_for(store.refine_result_path(n)) for n in expected]
    assert all(args[args.index("--cache-dir") + 1] == str(root / "cache") for args in calls(root))
    assert all(s.state == "diagnosed_not_scored" and s.seconds is not None for s in result.seats)
    assert result.modal_identity == modal_identity()
    after = snapshot(store)
    assert {p: after[p] for p in before} == before
    assert all(p.startswith("stage-one-check/") for p in after.keys() - before.keys())
    assert not (store.path / "modal-diagnosis").exists()


def test_overlapping_roles_preserved_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.engine._stage_one_refine_cases import pool
    from tests.engine import _search_stage_one_cases as search
    from tests.engine.test_search_stage_one_refine import refine
    store = pool(tmp_path, monkeypatch, [(8, 1., 5., .9), (2, 2., 8., .6)])
    refine(store, search.FakeCompute(store, monkeypatch))
    result = run(store, tmp_path / "child")
    assert result.seats[0].roles == ("top_n", "far_control")
    assert [Path(args[0]).name for args in calls(tmp_path / "child")] == ["trial-000008-scheme.json", "trial-000002-scheme.json"]


@pytest.mark.parametrize("state", ["selected", "interrupted", "failed", "broken", "missing"])
def test_not_done_no_directory_or_subprocess(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state: str) -> None:
    from aosr.search.stage_one_post_check import run_stage_one_post_check
    store = prepared(tmp_path, monkeypatch)
    path = store.path / "stage_one_refine.json"
    if state == "missing":
        path.unlink()
    elif state == "broken":
        path.write_text("{")
    else:
        doc = json.loads(path.read_bytes()); doc["state"] = state; path.write_text(json.dumps(doc))
    before = snapshot(store)
    with pytest.raises((OSError, ValueError)):
        run_stage_one_post_check(store, cache_dir=tmp_path / "cache", lock_fd=None, runner=("must-not-run",))
    assert snapshot(store) == before
    assert not (store.path / "stage-one-check").exists()


@pytest.mark.parametrize("damage", ["missing_result", "broken_result", "pending", "unreadable"])
def test_unreadable_seat_skipped_others_continue(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str) -> None:
    store = prepared(tmp_path, monkeypatch)
    if damage == "missing_result":
        store.refine_result_path(8).unlink()
    elif damage == "broken_result":
        store.refine_result_path(8).write_text("{")
    else:
        path = store.path / "stage_one_refine.json"; doc = json.loads(path.read_bytes())
        doc["results"][0]["state"] = damage; path.write_text(json.dumps(doc))
    result = run(store, tmp_path / "child")
    assert result.seats[0].state == "skipped" and result.seats[0].reason_text
    skipped = result.seats[0]
    assert skipped.diagnosis_file is not None
    document = store.path / "stage-one-check" / skipped.diagnosis_file
    assert json.loads(document.read_bytes())["state"] == "not_computed"
    assert document.with_suffix(".stderr").is_file()
    assert all(s.state == "diagnosed_not_scored" for s in result.seats[1:])
    assert [Path(args[0]).name for args in calls(tmp_path / "child")] == [f"trial-{n:06d}-scheme.json" for n in (2, 1, 5)]


def test_unassessed_incomparable_still_diagnosed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.engine._stage_one_refine_cases import pool
    from tests.engine.test_search_stage_one_refine import ROWS, refine
    from tests.engine import _search_stage_one_cases as search
    store = pool(tmp_path, monkeypatch, ROWS)
    status = refine(store, search.FakeCompute(store, monkeypatch, unavailable=frozenset({8}), different=frozenset({2})))
    assert {s.state for s in status.results} >= {"unassessed", "incomparable"}
    assert all(s.state == "diagnosed_not_scored" for s in run(store, tmp_path / "child").seats)


@pytest.mark.parametrize("damage", ["none", "selection", "identity", "one_file", "placement", "outside"])
def test_reuse_requires_selection_identity_and_valid_placement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str) -> None:
    from aosr.search.stage_one_post_check import run_stage_one_post_check
    store = prepared(tmp_path, monkeypatch)
    first = run(store, tmp_path / "first")
    path = store.path / "stage-one-check" / "summary.json"
    doc = json.loads(path.read_bytes())
    if damage == "selection":
        doc["selection"]["top_n"] += 1
    elif damage == "identity":
        doc["modal_identity"] = "modal-v1:" + "b" * 64
    elif damage in ("one_file", "placement"):
        target = path.parent / doc["seats"][0]["diagnosis_file"]
        if damage == "one_file":
            target.write_text("{")
        else:
            content = json.loads(target.read_bytes())
            for pair in content["placement_layer"]["pairs"]:
                pair["speaker_position_m"][0] += .1
            target.write_text(json.dumps(content))
    elif damage == "outside":
        doc["seats"][0]["diagnosis_file"] = str(Path("..") / "elsewhere.json")
    path.write_text(json.dumps(doc))
    root = tmp_path / "second"
    command = ("must-not-run",) if damage == "none" else runner(root, store)
    result = run_stage_one_post_check(store, cache_dir=root / "cache", lock_fd=None, runner=command)
    assert result.state == "done"
    expected = [] if damage == "none" else [8, 2, 1, 5] if damage in ("selection", "identity") else [8]
    assert [int(Path(args[0]).name.split('-')[1]) for args in calls(root)] == expected
    if damage == "none":
        assert result.seats == first.seats


@pytest.mark.parametrize("state", ["diagnosed_not_scored", "not_computed", "failed", "out_of_scope"])
def test_four_document_states(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state: str) -> None:
    result = run(prepared(tmp_path, monkeypatch), tmp_path / "child", state=state)
    assert all(s.state == state for s in result.seats)
    assert all(s.reason_text for s in result.seats if state != "diagnosed_not_scored")


@pytest.mark.parametrize("code,expected", [(7, "failed"), (-15, "stopped"), (143, "stopped"), (-1, "stopped"), (129, "stopped"), (137, "failed"), (-9, "failed")])
def test_nonzero_exit_and_external_signals(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, code: int, expected: str) -> None:
    # Python sys.exit 負數會變成 256 加代碼；訊號替身改成真的向自己送訊號。
    store = prepared(tmp_path, monkeypatch); root = tmp_path / "child"
    command = runner(root, store, code=code)
    if code < 0:
        script = Path(command[1]); script.write_text(script.read_text().replace(f"sys.exit({code})", f"os.kill(os.getpid(), {-code})"))
    from aosr.search.stage_one_post_check import run_stage_one_post_check
    result = run_stage_one_post_check(store, cache_dir=root / "cache", lock_fd=None, runner=command)
    assert result.state == expected
    assert result.seats[0].state == expected and result.seats[0].exit_code == code
    assert "替身錯誤原文" in result.seats[0].reason_text
    if code in (-9, 137):
        assert "可能是記憶體不足" in result.seats[0].reason_text
    if expected == "stopped":
        assert all(s.state == "stopped" for s in result.seats)
        assert [Path(args[0]).name for args in calls(root)] == ["trial-000008-scheme.json"]


def test_marker_before_second_preserves_first_and_marker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = prepared(tmp_path, monkeypatch)
    result = run(store, tmp_path / "child", stop_after=True)
    assert result.state == "stopped" and result.seats[0].state == "diagnosed_not_scored"
    assert all(s.state == "stopped" for s in result.seats[1:])
    assert store.refine_stop_path.exists()
    for seat in result.seats:
        assert seat.diagnosis_file is not None
        document = store.path / "stage-one-check" / seat.diagnosis_file
        assert document.is_file() and document.with_suffix(".stderr").is_file()


def test_keyboard_interrupt_preserves_finished_seats(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from aosr.search import stage_one_post_check as module
    from aosr.search.modal_attach import _compute
    from aosr.search.modal_record import RoleInput
    store = prepared(tmp_path, monkeypatch); root = tmp_path / "child"
    count: list[bool] = []
    def compute(item: RoleInput, output: Path, *, cache_dir: Path, lock_fd: int | None, runner: tuple[str, ...]) -> int:
        if count:
            raise KeyboardInterrupt
        count.append(True)
        return _compute(item, output, cache_dir=cache_dir, lock_fd=lock_fd, runner=runner)
    monkeypatch.setattr(module, "_compute", compute)
    result = module.run_stage_one_post_check(store, cache_dir=root / "cache", lock_fd=None, runner=runner(root, store))
    assert result.state == "stopped" and result.seats[0].state == "diagnosed_not_scored"
    assert all(s.state == "stopped" for s in result.seats[1:])


def test_atomic_write_failure_keeps_original(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from aosr.search.stage_one_post_check_record import write_summary
    store = prepared(tmp_path, monkeypatch); result = run(store, tmp_path / "child")
    path = store.path / "stage-one-check" / "summary.json"; before = path.read_bytes()
    original = os.fdopen
    class Broken:
        def __init__(self, descriptor: int) -> None:
            self.handle = original(descriptor, "w", encoding="utf-8")
        def __enter__(self) -> "Broken":
            return self
        def __exit__(self, *args: object) -> None:
            self.handle.close()
        def write(self, text: str) -> None:
            self.handle.write(text[:len(text) // 2]); raise OSError("partial write")
    monkeypatch.setattr("aosr.search.stage_one_post_check_record.os.fdopen", lambda fd, *a, **k: Broken(fd))
    with pytest.raises(OSError, match="partial write"):
        write_summary(store.path, result)
    assert path.read_bytes() == before
    assert not list(path.parent.glob("*.tmp"))


@pytest.mark.parametrize("damage", ["missing", "broken", "wrong_placement"])
def test_exit_zero_without_valid_document_is_failed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str) -> None:
    from aosr.search.stage_one_post_check import run_stage_one_post_check
    store = prepared(tmp_path, monkeypatch); root = tmp_path / "child"
    command = runner(root, store)
    fixture = root / "8.json"
    if damage == "missing":
        script = Path(command[1]); script.write_text(script.read_text().replace("shutil.copyfile(root/f'{number}.json',out)", "pass"))
    elif damage == "broken":
        fixture.write_text("{")
    else:
        doc = json.loads(fixture.read_bytes())
        for pair in doc["placement_layer"]["pairs"]:
            pair["speaker_position_m"][0] += .1
        fixture.write_text(json.dumps(doc))
    result = run_stage_one_post_check(store, cache_dir=root / "cache", lock_fd=None, runner=command)
    assert result.state == "failed" and result.seats[0].state == "failed"
    assert result.seats[0].reason_text


def test_preexisting_stop_opens_no_child_and_remains(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = prepared(tmp_path, monkeypatch)
    store.refine_stop_path.touch()
    result = run(store, tmp_path / "child")
    assert result.state == "stopped" and all(s.state == "stopped" for s in result.seats)
    assert not calls(tmp_path / "child") and store.refine_stop_path.exists()
