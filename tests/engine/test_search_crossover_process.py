"""真行程驗證附件第二個訊號與警告斷管：量 Python 結束後的離開碼。"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from aosr.runtime import child_process_env
from aosr.search.crossover_record import STOPPED_NOTE, read_summary
from aosr.search.outer_status import OuterStatus
from aosr.search.store import SearchStore
from tests.engine._crossover_cases import prepared, protected


def _parent(tmp_path: Path, conclusion: str, *, second: signal.Signals = signal.SIGTERM) -> tuple[Path, SearchStore]:
    store, registry, status = prepared(tmp_path)
    status = status.model_copy(update={"outer": OuterStatus.model_validate({"conclusion": conclusion})})
    ready, go = tmp_path / "ready", tmp_path / "go"
    args = ["auto", str(store.path), "--engine-commit", "test", "--modal-cache-dir", str(tmp_path / "cache")]
    parent = tmp_path / "parent.py"
    parent.write_text(
        "import os, signal, sys, time, warnings\nfrom pathlib import Path\n"
        f"sys.path.insert(0, {str(Path(__file__).resolve().parents[2])!r})\n"
        "from aosr.search import cli, crossover_sensitivity as module\n"
        "from aosr.config.paths import config_path\nfrom aosr.search.store import SearchStore\n"
        "from aosr.search.run import SearchStatus\nfrom tests.engine._crossover_cases import evaluate\n"
        f"store = SearchStore.open(Path({str(store.path)!r}))\n"
        f"status = SearchStatus.model_validate_json({status.model_dump_json()!r})\n"
        "cli.auto_search = lambda *args, **kwargs: status\ncli.attach_modal = lambda *args, **kwargs: None\n"
        # 強制進重評，只測附件收尾的離開碼；正常資格規則另有全結論考卷。
        "module.attachment_skip_reason = lambda conclusion: ''\n"
        f"cli.config_path = lambda name: Path({str(registry)!r}) if name.startswith('quality_targets') else config_path(name)\n"
        "def reevaluate(result, **kwargs):\n"
        f"    Path({str(ready)!r}).touch()\n"
        f"    while not Path({str(go)!r}).exists():\n        time.sleep(0.01)\n"
        "    warnings.warn('交接重評警告', RuntimeWarning)\n    return evaluate(result)\n"
        "module.reevaluate = reevaluate\noriginal = cli._stderr\n"
        "def stderr(message):\n    original(message)\n"
        f"    if {STOPPED_NOTE!r} in message:\n        os.kill(os.getpid(), {int(second)!r})\n"
        "cli._stderr = stderr\ncli._interrupt_on_termination()\n"
        f"raise SystemExit(cli.main({args!r}, compute_factory=lambda *args: object()))\n")
    return parent, store


def _ready(process: subprocess.Popen[bytes], tmp_path: Path) -> None:
    deadline = time.monotonic() + 60
    while not (tmp_path / "ready").exists():
        assert process.poll() is None, process.stderr.read().decode() if process.stderr is not None else "行程提前結束"
        assert time.monotonic() < deadline
        time.sleep(0.02)


@pytest.mark.parametrize("conclusion,code", [("complete", 0), ("refine_failed", 1), ("refine_interrupted", 3)])
@pytest.mark.parametrize("second", [signal.SIGTERM, signal.SIGHUP, signal.SIGINT])
def test_real_second_signal_on_stop_line_keeps_outer_code(tmp_path: Path, conclusion: str, code: int,
                                                        second: signal.Signals) -> None:
    parent, store = _parent(tmp_path, conclusion, second=second)
    before = protected(store)
    process = subprocess.Popen([sys.executable, str(parent)], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                               env=child_process_env(threads=1), start_new_session=True)
    try:
        _ready(process, tmp_path)
        process.send_signal(signal.SIGTERM)
        _, stderr = process.communicate(timeout=60)
        assert process.returncode == code and STOPPED_NOTE in stderr.decode() and b"Traceback" not in stderr
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    summary = read_summary(store.path)
    assert summary is not None and summary.state == "stopped" and not summary.completed
    assert protected(store) == before


@pytest.mark.parametrize("conclusion,code", [("complete", 0), ("refine_failed", 1), ("refine_interrupted", 3)])
def test_real_warning_with_broken_stderr_keeps_outer_code(tmp_path: Path, conclusion: str, code: int) -> None:
    parent, store = _parent(tmp_path, conclusion)
    process = subprocess.Popen([sys.executable, str(parent)], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                               env=child_process_env(threads=1), start_new_session=True)
    try:
        _ready(process, tmp_path)
        assert process.stderr is not None
        process.stderr.close()
        (tmp_path / "go").touch()
        assert process.wait(timeout=60) == code
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    summary = read_summary(store.path)
    assert summary is not None and summary.state == "done" and summary.completed
