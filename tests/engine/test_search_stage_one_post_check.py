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
        result = run_stage_one_post_check(store, cache_dir=root / "cache", lock_fd=descriptor,
                                          runner=runner(root, store, lock_fd=descriptor))
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
    if damage in ("pending", "unreadable"):
        assert {"pending": "細算狀態 還沒算", "unreadable": "細算狀態 讀不回"}[damage] in result.seats[0].reason_text
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
        # 資料夾外放一份合法的診斷檔：只靠讀不到檔會放過「不限附件資料夾」的錯。
        source = path.parent / doc["seats"][0]["diagnosis_file"]
        (store.path / "elsewhere.json").write_bytes(source.read_bytes())
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


def test_finish_keeps_two_generations_and_foreign_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # 每一輪弄壞試算 8 的診斷逼它重算；收尾只留這一代與上一代指到的文件，不碰別人的檔。
    store = prepared(tmp_path, monkeypatch)
    folder = store.path / "stage-one-check"
    names = [run(store, tmp_path / "round-0").seats[0].diagnosis_file]
    foreign = folder / "notes.json"; foreign.write_text("{}")
    for round_ in (1, 2):
        (folder / str(names[-1])).write_text("{")
        result = run(store, tmp_path / f"round-{round_}")
        assert result.state == "done"
        names.append(result.seats[0].diagnosis_file)
    first, second, third = (folder / str(name) for name in names)
    assert len({first, second, third}) == len(names)
    assert not first.exists() and not first.with_suffix(".stderr").exists()
    assert second.exists() and third.exists() and third.with_suffix(".stderr").exists()
    assert foreign.exists()
    assert all((folder / str(seat.diagnosis_file)).exists() for seat in result.seats)


def test_scheme_file_must_match_refined_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # 細算結果照舊讀得回，但 verification 裡試算 8 的方案檔被換成試算 2 的：不准拿去診斷。
    store = prepared(tmp_path, monkeypatch)
    target = store.scheme_path_for(store.refine_result_path(8))
    target.write_bytes(store.scheme_path_for(store.refine_result_path(2)).read_bytes())
    result = run(store, tmp_path / "child")
    assert result.seats[0].state == "skipped" and "細算方案檔跟細算結果裡的方案不同" in result.seats[0].reason_text
    assert [Path(args[0]).name for args in calls(tmp_path / "child")] == [f"trial-{n:06d}-scheme.json" for n in (2, 1, 5)]


def test_rerun_killed_after_first_write_keeps_reuse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # 重跑剛寫完第一份摘要就被砍：沿用的診斷要已經在那一份裡，下次不必再開子行程，也不會被當孤檔清掉。
    from aosr.search import stage_one_post_check as module
    store = prepared(tmp_path, monkeypatch); first = run(store, tmp_path / "first")
    from aosr.search.stage_one_post_check_record import write_summary as real
    def killed(folder: Path, summary: StageOnePostCheckSummary) -> None:
        real(folder, summary)
        raise KeyboardInterrupt
    monkeypatch.setattr(module, "write_summary", killed)
    with pytest.raises(KeyboardInterrupt):
        module.run_stage_one_post_check(store, cache_dir=tmp_path / "cache", lock_fd=None, runner=("must-not-run",))
    monkeypatch.setattr(module, "write_summary", real)
    result = module.run_stage_one_post_check(store, cache_dir=tmp_path / "cache", lock_fd=None, runner=("must-not-run",))
    assert result.state == "done" and result.seats == first.seats
    assert all((store.path / "stage-one-check" / str(seat.diagnosis_file)).is_file() for seat in result.seats)


def test_interrupt_while_saving_keeps_finished_seat(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # 第一席診斷完、存摘要那一刻按 Ctrl-C：第一席照舊是已診斷、指著它自己的文件，其餘記停止。
    from aosr.search import stage_one_post_check as module
    store = prepared(tmp_path, monkeypatch)
    from aosr.search.stage_one_post_check_record import write_summary as real
    hit: list[bool] = []
    def interrupted(folder: Path, summary: StageOnePostCheckSummary) -> None:
        if not hit and summary.seats[0].state == "diagnosed_not_scored":
            hit.append(True)
            raise KeyboardInterrupt
        real(folder, summary)
    monkeypatch.setattr(module, "write_summary", interrupted)
    result = module.run_stage_one_post_check(store, cache_dir=tmp_path / "cache", lock_fd=None, runner=runner(tmp_path / "child", store))
    assert hit and result.state == "stopped"
    assert result.seats[0].state == "diagnosed_not_scored"
    assert json.loads((store.path / "stage-one-check" / str(result.seats[0].diagnosis_file)).read_bytes())["state"] == "diagnosed_not_scored"
    assert all(s.state == "stopped" for s in result.seats[1:])


def test_unexpected_error_closes_summary_as_failed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from aosr.search import stage_one_post_check as module
    store = prepared(tmp_path, monkeypatch)
    def broken(*args: object, **kwargs: object) -> None:
        raise OSError("disk failure")
    monkeypatch.setattr(module, "_diagnose", broken)
    with pytest.raises(OSError, match="disk failure"):
        module.run_stage_one_post_check(store, cache_dir=tmp_path / "cache", lock_fd=None, runner=("must-not-run",))
    saved = StageOnePostCheckSummary.model_validate_json((store.path / "stage-one-check" / "summary.json").read_bytes())
    assert saved.state == "failed" and "disk failure" in saved.message


def test_interrupt_on_first_write_closes_as_stopped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # 第一份摘要寫入時按一次 Ctrl-C：當次摘要要收成停止，沿用的席照舊、要重算的那席記停止。
    from aosr.search import stage_one_post_check as module
    store = prepared(tmp_path, monkeypatch); first = run(store, tmp_path / "first")
    folder = store.path / "stage-one-check"
    (folder / str(first.seats[0].diagnosis_file)).write_text("{")
    from aosr.search.stage_one_post_check_record import write_summary as real
    hit: list[bool] = []
    def interrupted(path: Path, summary: StageOnePostCheckSummary) -> None:
        if not hit:
            hit.append(True)
            real(path, summary)
            raise KeyboardInterrupt
        real(path, summary)
    monkeypatch.setattr(module, "write_summary", interrupted)
    try:
        result = module.run_stage_one_post_check(store, cache_dir=tmp_path / "cache", lock_fd=None, runner=("must-not-run",))
    except KeyboardInterrupt:
        pytest.fail("第一份摘要寫入時的 Ctrl-C 漏到外面，沒有收成停止")
    saved = StageOnePostCheckSummary.model_validate_json((folder / "summary.json").read_bytes())
    assert hit and result.state == "stopped" and saved == result
    assert result.seats[0].state == "stopped" and result.seats[1:] == first.seats[1:]
