"""搜尋資料夾鎖：真命令列與真子行程，計算只用合成資料。"""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
from collections.abc import Generator, Iterator, Sequence
from contextlib import ExitStack, contextmanager
from pathlib import Path

import pytest

from aosr.config.paths import config_path
from aosr.runtime import child_process_env
from aosr.search import cli, ledger
from aosr.search.run import CandidateJob, ComputedCandidate, Compute, SearchStatus, _write_status
from aosr.search.store import SearchStore
from aosr.search.worker import SubprocessCompute
from tests.engine._search_outer_cases import OuterCompute
from tests.engine._search_refine_cases import RefineCompute, SearchCompute
from tests.engine._search_run_cases import FakeCompute, make_store
from tests.engine._search_select_cases import refined_store
from tests.engine.test_search_cli import opened, start_args


BUSY_MESSAGE = "這個搜尋資料夾還有計算在跑（可能是上一次被強制結束後留下的子行程），等它結束再試"

COMMAND_SCRIPT = '''
import sys
from pathlib import Path
from aosr.config.paths import config_path
from aosr.search import cli
from aosr.search.store import SearchStore
from tests.engine._search_run_cases import FakeCompute

store = SearchStore.open(Path(sys.argv[1]))
registry = Path(sys.argv[2])
# start 的資料夾剛建好這個接縫，仍走真狀態寫入與命令分派。
cli._create = lambda args: store
cli._identity = lambda purpose, capabilities: store.identity
cli.config_path = lambda name: registry if name.startswith("quality_targets") else config_path(name)
raise SystemExit(cli.main(sys.argv[3:], compute_factory=lambda opened, capabilities, commit: FakeCompute(opened)))
'''


def invoke(store: SearchStore, registry: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    env = child_process_env(threads=1) | {"PYTHONDONTWRITEBYTECODE": "1"}
    return subprocess.run([sys.executable, "-c", COMMAND_SCRIPT, str(store.path), str(registry), *args],
                          capture_output=True, text=True, timeout=15, env=env)


@contextmanager
def _folder(store: SearchStore) -> Iterator[int]:
    """模擬另一支命令列：開搜尋資料夾本身的描述子，離開時關掉。"""
    descriptor = store.open_folder_lock()
    try:
        yield descriptor
    finally:
        os.close(descriptor)


def snapshot(store: SearchStore) -> dict[Path, bytes | None]:
    return {path.relative_to(store.path): path.read_bytes() if path.is_file() else None
            for path in store.path.rglob("*")}


@pytest.mark.parametrize("command", ["start", "resume", "refine", "feedback", "auto"])
def test_busy_mutating_commands_refuse_without_writes(tmp_path: Path, command: str) -> None:
    store, registry = make_store(tmp_path)
    if command != "start":
        _write_status(store, SearchStatus())
    args = [command, str(store.path)]
    if command == "start":
        args = [command, "--project", str(tmp_path / "project"), "--settings", str(tmp_path / "settings"),
                "--root", str(tmp_path)]
    if command != "feedback":
        args += ["--engine-commit", "test"]
    if command == "auto":
        args += ["--modal-cache-dir", str(tmp_path / "modal-cache")]
    with _folder(store) as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        before = snapshot(store)
        result = invoke(store, registry, args)
        code = result.returncode
        assert code == 1
        assert result.stdout == "" and result.stderr == BUSY_MESSAGE + "\n"
        assert snapshot(store) == before


@pytest.mark.parametrize("command", ["stop", "stop-refine", "report", "select"])
def test_unlocked_commands_still_work_while_busy(tmp_path: Path, command: str) -> None:
    store = refined_store(tmp_path)
    registry = tmp_path / "registry"
    args = ["stop", str(store.path), "--refine"] if command == "stop-refine" else [command, str(store.path)]
    if command == "report":
        args = [command, "--search", str(store.path)]
    if command == "select":
        args += ["--baseline", "--data-dir", str(tmp_path / "data")]
    with _folder(store) as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = invoke(store, registry, args)
        code = result.returncode
        assert code == 0 and result.stderr == ""
    if command.startswith("stop"):
        assert (store.refine_stop_path if command == "stop-refine" else store.stop_path).is_file()
    elif command == "report":
        assert result.stdout
    else:
        assert "已放進結果清單" in result.stdout


def test_child_keeps_lock_after_parent_closes_until_child_exits(tmp_path: Path) -> None:
    store, _ = make_store(tmp_path)
    script = "import os, sys, time; os.fstat(int(sys.argv[1])); sys.stdout.write('ready\\n'); sys.stdout.flush(); time.sleep(120)"
    with ExitStack() as resources:
        with cli._search_lock(store) as fd:
            child = resources.enter_context(subprocess.Popen(
                [sys.executable, "-c", script, str(fd)], pass_fds=(fd,), stdout=subprocess.PIPE))
        try:
            assert child.stdout is not None and child.stdout.readline() == b"ready\n"
            with _folder(store) as contender:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
                child.kill()
                child.wait(timeout=5)
                fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5)


@pytest.mark.parametrize("kind", ["candidate", "slice"])
def test_worker_children_can_see_lock_descriptor(tmp_path: Path, kind: str) -> None:
    store, _ = make_store(tmp_path)
    job = CandidateJob(None, store.project, store.baseline_path)
    script = '''
import json, os, sys
from pathlib import Path
fd = int(sys.argv[1])
out = Path(sys.argv[sys.argv.index("--out") + 1])
out.write_text(json.dumps({"inode": os.fstat(fd).st_ino}))
'''
    with cli._search_lock(store) as fd:
        runner = (sys.executable, "-c", script, str(fd))
        worker = SubprocessCompute(capabilities_path=tmp_path, engine_commit="test", search_id=store.search_id,
                                   fem_root=store.fem_path, runner=runner, slice_runner=runner, lock_fd=fd)
        outputs: tuple[Path, ...]
        if kind == "candidate":
            item = worker._start(job)
            code = item.process.wait(timeout=5)
            assert code == 0
            outputs = (job.result_path,)
        else:
            outputs = worker._slices((job,), 1)
        assert outputs
        assert all(json.loads(path.read_bytes())["inode"] == store.path.stat().st_ino for path in outputs)


def test_normal_start_holds_lock_then_releases_for_next_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def factory(store: SearchStore, capabilities: Path, commit: str) -> Compute:
        with _folder(store) as contender:
            with pytest.raises(BlockingIOError):
                fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert store.status_path.is_file()
        return SearchCompute(store, persist_baseline=True)

    args = start_args(tmp_path)
    settings = Path(args[args.index("--settings") + 1])
    settings.write_text(json.dumps(json.loads(settings.read_bytes()) | {
        "refine": {"budget": 1, "convergence_run": 50},
    }))
    code = cli.main(args, compute_factory=factory)
    assert code == 0
    store = opened(tmp_path)
    with _folder(store) as contender:
        fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
    monkeypatch.setattr(cli, "_identity", lambda purpose, capabilities: store.identity)
    code = cli.main(["refine", str(store.path), "--engine-commit", "test"],
                    compute_factory=lambda opened, capabilities, commit: RefineCompute(opened, {}))
    assert code == 0
    with _folder(store) as contender:
        fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_auto_holds_one_lock_through_search_and_refine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store, registry = make_store(tmp_path, budget=40, batch=2, convergence=7,
                                 refine={"budget": 20, "convergence_run": 2}, feedback={"offset": 0.125})
    ledger.create_for(store)
    _write_status(store, SearchStatus())
    monkeypatch.setattr(cli, "config_path", lambda name: registry if name.startswith("quality_targets") else config_path(name))
    monkeypatch.setattr(cli, "_identity", lambda purpose, capabilities: store.identity)
    outer = OuterCompute(store)

    def compute(jobs: Sequence[CandidateJob], workers: int) -> Iterator[ComputedCandidate]:
        with _folder(store) as contender:
            with pytest.raises(BlockingIOError):
                fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield from outer(jobs, workers)

    code = cli.main(["auto", str(store.path), "--engine-commit", "test",
                     "--modal-cache-dir", str(tmp_path / "modal-cache")],
                    compute_factory=lambda opened, capabilities, commit: compute)
    assert code == 0
    assert outer.search.calls and outer.refine.jobs and store.feedback_path.is_file()
    with _folder(store) as contender:
        fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_default_cli_factory_passes_held_lock_to_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class ObservedCompute(SubprocessCompute):
        def __call__(self, jobs: Sequence[CandidateJob], max_workers: int) -> Generator[ComputedCandidate, None, None]:
            script = "import os, sys; os.fstat(int(sys.argv[1]))"
            with self._spawn((sys.executable, "-c", script, str(self.lock_fd)), tmp_path / "stderr") as child:
                code = child.wait(timeout=5)
                assert code == 0
            yield from FakeCompute(SearchStore.open(self.fem_root.parent))(jobs, max_workers)

    monkeypatch.setattr(cli, "SubprocessCompute", ObservedCompute)
    code = cli.main(start_args(tmp_path))
    assert code == 0


@pytest.mark.parametrize("command", ["resume", "refine", "auto"])
def test_default_factory_hands_held_lock_to_every_computing_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: str,
) -> None:
    """預設工廠對每一支會開子行程的命令都把持有中的資料夾鎖交出去（複查：原本只考 start）。"""
    from tests.engine._search_refine_cases import stopped_store
    from tests.engine.test_search_outer_auto import seed

    store, registry = stopped_store(tmp_path) if command == "refine" else seed(tmp_path)
    held: list[bool] = []

    class Recorder(SubprocessCompute):
        def __init__(self, **kwargs: object) -> None:
            descriptor = kwargs["lock_fd"]
            assert isinstance(descriptor, int)
            assert os.fstat(descriptor).st_ino == store.path.stat().st_ino
            with _folder(store) as contender:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
            held.append(True)
            raise RuntimeError("記錄完就停")

    monkeypatch.setattr(cli, "SubprocessCompute", Recorder)
    monkeypatch.setattr(cli, "config_path",
                        lambda name: registry if name.startswith("quality_targets") else config_path(name))
    args = [command, str(store.path), "--engine-commit", "test"]
    if command == "auto":
        args += ["--modal-cache-dir", str(tmp_path / "modal-cache")]
    exit_code = cli.main(args)
    assert exit_code == 1
    assert held == [True]


def test_error_exit_that_fails_is_not_handled_twice(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """拿到鎖之後的錯誤出口本身再出錯時照舊往外丟、只走一次（複查：外層曾再接一次、重複寫失敗）。"""
    from tests.engine.test_search_outer_auto import seed

    store, registry = seed(tmp_path)
    calls: list[str] = []

    def failing_exit(opened: SearchStore | None, error: Exception) -> int:
        calls.append(str(error))
        raise OSError("狀態寫不進去")

    def broken(opened: SearchStore, capabilities: Path, commit: str) -> Compute:
        raise RuntimeError("工廠壞了")

    monkeypatch.setattr(cli, "_failed", failing_exit)
    monkeypatch.setattr(cli, "config_path",
                        lambda name: registry if name.startswith("quality_targets") else config_path(name))
    with pytest.raises(OSError, match="狀態寫不進去"):
        cli.main(["resume", str(store.path), "--engine-commit", "test"], compute_factory=broken)
    assert calls == ["工廠壞了"]


@pytest.mark.parametrize("where", ["open", "flock"])
def test_lock_errors_other_than_busy_refuse_without_touching_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], where: str,
) -> None:
    """拿鎖本身出錯（例如資料夾沒有讀權限）也照拿不到收：不寫失敗、進行中的搜尋之後還接得回來（複查）。"""
    from tests.engine.test_search_outer_auto import seed

    store, registry = seed(tmp_path)

    def refused(*args: object, **kwargs: object) -> int:
        raise PermissionError(13, "權限不足")

    monkeypatch.setattr(cli, "config_path",
                        lambda name: registry if name.startswith("quality_targets") else config_path(name))
    before = snapshot(store)
    with monkeypatch.context() as patched:
        if where == "open":
            patched.setattr(SearchStore, "open_folder_lock", refused)
        else:
            patched.setattr(fcntl, "flock", refused)
        exit_code = cli.main(["resume", str(store.path), "--engine-commit", "test"],
                             compute_factory=lambda opened, capabilities, commit: FakeCompute(opened))
    assert exit_code == 1
    assert capsys.readouterr().err.startswith("拿不到搜尋資料夾的鎖，沒有動任何檔：")
    assert snapshot(store) == before
    assert SearchStatus.model_validate_json(store.status_path.read_bytes()).state == "running"
