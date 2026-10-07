"""真父子行程驗鎖繼承、單緒、訊號收尾及兩秒後強停；全部資料在考卷暫存。"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from aosr.runtime import child_process_env
from aosr.search.modal_attach import attach_modal
from aosr.search.modal_record import read_summary
from aosr.search.outer_status import OuterConclusion
from aosr.search.store import SearchStore
from tests.engine._modal_cases import runner
from aosr.reporting.modal_diagnosis_model import ModalDiagnosis, ModalDiagnosisState
from tests.engine._search_modal_cases import prepared, protected


def _wait(path: Path, process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 20
    while not path.exists():
        assert process.poll() is None, "子行程提前退出"
        assert time.monotonic() < deadline, "子行程未進入診斷"
        time.sleep(0.02)


def test_attachment_child_inherits_folder_lock_session_and_single_threads(tmp_path: Path) -> None:
    from aosr.search.cli import _search_lock
    store, _, status = prepared(tmp_path, refined=False)
    fake = runner(tmp_path / "runner", ModalDiagnosis(state=ModalDiagnosisState.NOT_COMPUTED))
    script = Path(fake[-1])
    proof = tmp_path / "proof.json"
    script.write_text("import os,json,fcntl\nfrom pathlib import Path\n"
        f"folder=Path({str(store.path)!r}); inode=folder.stat().st_ino\n"
        "inherited=[]\nfor entry in Path('/proc/self/fd').iterdir():\n"
        "    try:\n        fd=int(entry.name)\n        if os.fstat(fd).st_ino == inode: inherited.append(fd)\n"
        "    except OSError: pass\n"
        "contender=os.open(folder,os.O_RDONLY|os.O_DIRECTORY)\n"
        "blocked=False\ntry: fcntl.flock(contender,fcntl.LOCK_EX|fcntl.LOCK_NB)\n"
        "except BlockingIOError: blocked=True\n"
        f"Path({str(proof)!r}).write_text(json.dumps(dict(inherited=inherited,blocked=blocked,"
        "session=os.getsid(0),pid=os.getpid(),threads=os.environ['OMP_NUM_THREADS'])))\n" + script.read_text())
    with _search_lock(store) as descriptor:
        attach_modal(store, status=status, cache_dir=tmp_path / "cache", runner=fake, lock_fd=descriptor)
        data = json.loads(proof.read_bytes())
        assert data["inherited"] == [descriptor] and data["blocked"]
        assert data["session"] == data["pid"] and data["threads"] == "1"


def _launch_signal_parent(tmp_path: Path, store: SearchStore, registry: Path, stubborn: bool, *,
                          setup: str = "") -> tuple[subprocess.Popen[str], Path, Path, list[str]]:
    ready, term_seen = tmp_path / "ready.json", tmp_path / "term-seen"
    child_script = tmp_path / "diagnosis-child.py"
    if stubborn:
        child_script.write_text("import os,sys,time,signal,json\nfrom pathlib import Path\n"
            f"signal.signal(signal.SIGTERM,lambda *args: Path({str(term_seen)!r}).touch())\n"
            f"Path({str(ready)!r}).write_text(json.dumps(dict(pid=os.getpid())))\n"
            "while True: time.sleep(0.02)\n")
    else:
        child_script.write_text("import runpy,sys,time,os,json\nfrom pathlib import Path\nfrom tempfile import TemporaryDirectory\n"
            "from aosr.reporting import modal_lookup\n"
            "def calculate(scheme, *, cache_dir):\n"
            "    cache_dir.mkdir(parents=True,exist_ok=True)\n"
            "    with TemporaryDirectory(dir=cache_dir,prefix='modal-write-') as name:\n"
            f"        Path({str(ready)!r}).write_text(json.dumps(dict(pid=os.getpid())))\n"
            "        while True: time.sleep(0.02)\n"
            "modal_lookup._calculate=calculate\n"
            "sys.argv=['scheme_cli','modal',*sys.argv[1:]]\n"
            "runpy.run_module('aosr.reporting.scheme_cli',run_name='__main__')\n")
    parent_script = tmp_path / "search-parent.py"
    args = ["auto", str(store.path), "--engine-commit", "test", "--modal-cache-dir", str(tmp_path / "cache")]
    parent_script.write_text("from pathlib import Path\nfrom aosr.search import cli\nfrom aosr.config.paths import config_path\n"
        "from aosr.search.store import SearchStore\n"
        f"store=SearchStore.open(Path({str(store.path)!r}))\ncli._identity=lambda *args: store.identity\n"
        f"cli.config_path=lambda name: Path({str(registry)!r}) if name.startswith('quality_targets') else config_path(name)\n"
        + setup +
        "cli._interrupt_on_termination()\n"
        f"raise SystemExit(cli.main({args!r},compute_factory=lambda *args: object(),"
        f"modal_runner=({sys.executable!r},{str(child_script)!r})))\n")
    parent = subprocess.Popen([sys.executable, str(parent_script)], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, env=child_process_env(threads=1), start_new_session=True)
    return parent, ready, term_seen, args


@pytest.mark.parametrize("conclusion", ["complete", "refine_budget"])
def test_one_sigterm_during_modal_child_never_starts_crossover(tmp_path: Path, conclusion: OuterConclusion) -> None:
    from aosr.search.crossover_record import STOPPED_REASON, read_summary as read_crossover
    from aosr.search.modal_attach import STOPPED_NOTE
    from aosr.search.outer import conclude
    from tests.engine._crossover_cases import prepared as crossover_prepared, protected as crossover_protected
    store, registry, status = crossover_prepared(tmp_path)
    status = conclude(store, status, conclusion)
    before = crossover_protected(store)
    calls = tmp_path / "reevaluations"
    calls.write_text("")
    setup = ("from aosr.search import modal_attach, crossover_sensitivity\n"
        "from aosr.search.run import SearchStatus\n"
        "cli.auto_search=lambda *args, **kwargs: SearchStatus.model_validate_json(store.status_path.read_bytes())\n"
        "modal_attach.key_from_scheme=lambda scheme: 'synthetic-modal-key'\n"
        "def counted(*args, **kwargs):\n"
        f"    with Path({str(calls)!r}).open('a') as output: output.write('reevaluated\\n')\n"
        "    raise AssertionError('低頻被停後不准交接重評')\n"
        "crossover_sensitivity.reevaluate=counted\n")
    parent, ready, _, _ = _launch_signal_parent(tmp_path, store, registry, False, setup=setup)
    child_pid: int | None = None
    try:
        _wait(ready, parent)
        child_pid = json.loads(ready.read_bytes())["pid"]
        parent.send_signal(signal.SIGTERM)
        _, stderr = parent.communicate(timeout=15)
        assert parent.returncode == 0  # 兩種已保存的外圈結論都正常離開。
        assert stderr == STOPPED_NOTE + "\n" and "Traceback" not in stderr
        assert not calls.read_text()
        crossover = read_crossover(store.path)
        assert crossover is not None and crossover.state == "stopped" and not crossover.completed
        assert crossover.reason_text == STOPPED_REASON
        modal = read_summary(store.path)
        assert modal is not None and modal.completed and all(r.state == "stopped" for r in modal.roles)
        assert crossover_protected(store) == before
        assert not tuple((tmp_path / "cache").glob("modal-write-*"))
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
    finally:
        if parent.poll() is None:
            os.killpg(parent.pid, signal.SIGKILL)
            parent.communicate(timeout=10)
        if child_pid is not None:
            try:
                os.killpg(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.parametrize("termination", [signal.SIGTERM, signal.SIGHUP])
@pytest.mark.parametrize("stubborn", [False, True])
@pytest.mark.parametrize("twice", [False, True])
def test_signal_stops_only_attachment_and_preserves_exit_and_bytes(tmp_path: Path, termination: signal.Signals,
                                                                  stubborn: bool, twice: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    store, registry, _ = prepared(tmp_path)
    before = protected(store)
    parent, ready, term_seen, args = _launch_signal_parent(tmp_path, store, registry, stubborn)
    child_pid: int | None = None
    try:
        _wait(ready, parent)
        child_pid = json.loads(ready.read_bytes())["pid"]
        started = time.monotonic()
        parent.send_signal(termination)
        if twice and stubborn:
            time.sleep(0.5)
            parent.send_signal(termination)
        _stdout, stderr = parent.communicate(timeout=15)
        assert parent.returncode == 0 and "低頻診斷被停止，搜尋結果不受影響" in stderr
        assert "Traceback" not in stderr
        assert protected(store) == before
        assert not store.stop_path.exists() and not store.refine_stop_path.exists()
        summary = read_summary(store.path)
        assert summary is not None and summary.completed and all(r.state == "stopped" for r in summary.roles)
        assert not tuple((tmp_path / "cache").glob("modal-write-*"))
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
        if stubborn:
            assert term_seen.exists()
            if not twice:
                assert time.monotonic() - started >= 2
        # 人手再跑只換成不會求解的替身；停止角色會再試，搜尋檔案仍不變。
        from aosr.search import cli
        fake = runner(tmp_path / "retry", ModalDiagnosis(state=ModalDiagnosisState.NOT_COMPUTED))
        from aosr.config.paths import config_path
        from tests.engine._search_outer_cases import OuterCompute
        with monkeypatch.context() as patch:
            patch.setattr(cli, "config_path", lambda name: registry if name.startswith("quality_targets") else config_path(name))
            retry_code = cli.main(args, compute_factory=lambda opened, *_: OuterCompute(opened), modal_runner=fake)
            assert retry_code == 0
        retried = read_summary(store.path)
        assert retried is not None and all(r.state == "not_computed" for r in retried.roles)
        assert protected(store) == before
    finally:
        if parent.poll() is None:
            os.killpg(parent.pid, signal.SIGKILL)
            parent.communicate(timeout=10)
        if child_pid is not None:
            try:
                os.killpg(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_relative_search_path_keeps_output_in_search_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from aosr.search import cli, modal_attach
    from tests.engine._search_outer_cases import OuterCompute
    store, _, status = prepared(tmp_path, refined=False)
    before = protected(store)
    fake = runner(tmp_path / "runner", ModalDiagnosis(state=ModalDiagnosisState.NOT_COMPUTED))
    isolated_repo = tmp_path / "child-cwd"
    isolated_repo.mkdir()
    # 真子行程照產品入口換工作目錄；用隔離目錄代替真 repo，壞實作也不寫真 repo。
    monkeypatch.setattr(modal_attach, "__file__", str(isolated_repo / "src/aosr/search/modal_attach.py"))
    monkeypatch.setattr(cli, "auto_search", lambda *args, **kwargs: status)
    monkeypatch.chdir(store.path.parent)
    code = cli.main(["auto", store.path.name, "--engine-commit", "test", "--modal-cache-dir", str(tmp_path / "cache")],
                    compute_factory=lambda opened, *_: OuterCompute(opened), modal_runner=fake)
    summary = read_summary(store.path)
    assert code == 0 and summary is not None and summary.completed
    assert all(r.state == "not_computed" for r in summary.roles)
    assert all((store.path / "modal-diagnosis" / str(r.diagnosis_file)).is_file() for r in summary.roles)
    assert not tuple(isolated_repo.iterdir()) and protected(store) == before
    args = json.loads((tmp_path / "runner" / "modal-args.json").read_bytes())
    assert Path(args[args.index("--out") + 1]).is_absolute()
