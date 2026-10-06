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
from aosr.search.run import SearchStatus
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


@pytest.mark.parametrize("termination", [signal.SIGTERM, signal.SIGHUP])
@pytest.mark.parametrize("stubborn", [False, True])
def test_signal_stops_only_attachment_and_preserves_exit_and_bytes(tmp_path: Path, termination: signal.Signals,
                                                                  stubborn: bool) -> None:
    store, registry, _ = prepared(tmp_path)
    before = protected(store)
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
        "cli._interrupt_on_termination()\n"
        f"raise SystemExit(cli.main({args!r},compute_factory=lambda *args: object(),"
        f"modal_runner=({sys.executable!r},{str(child_script)!r})))\n")
    parent = subprocess.Popen([sys.executable, str(parent_script)], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, env=child_process_env(threads=1), start_new_session=True)
    child_pid: int | None = None
    try:
        _wait(ready, parent)
        child_pid = json.loads(ready.read_bytes())["pid"]
        started = time.monotonic()
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
            assert term_seen.exists() and time.monotonic() - started >= 2
        # 人手再跑只換成不會求解的替身；停止角色會再試，搜尋檔案仍不變。
        attach_modal(store, status=SearchStatus.model_validate_json(store.status_path.read_bytes()),
            cache_dir=tmp_path / "cache", runner=runner(tmp_path / "retry", ModalDiagnosis(state=ModalDiagnosisState.NOT_COMPUTED)))
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
