"""模態命令四态都交文件，只有程式錯誤回非零；只查快取永不求解。"""
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from aosr.reporting import modal_lookup as api
from aosr.reporting.modal_diagnosis_model import ModalDiagnosis, ModalDiagnosisState
from aosr.reporting.scheme_cli import main
from tests.engine._modal_cases import sample, scheme


@pytest.mark.parametrize("state", list(ModalDiagnosisState))
def test_modal_cli_writes_each_state_and_returns_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                      state: ModalDiagnosisState) -> None:
    diagnosis = sample()[0] if state is ModalDiagnosisState.DIAGNOSED_NOT_SCORED else ModalDiagnosis(
        state=state, reason_text="原始錯誤" if state is ModalDiagnosisState.FAILED else None,
        reason_code="unsupported_impedance" if state is ModalDiagnosisState.OUT_OF_SCOPE else None)
    monkeypatch.setattr(api, "diagnose_scheme", lambda *args, **kwargs: diagnosis)
    source, out = tmp_path / "scheme.json", tmp_path / "diagnosis.json"
    source.write_text(scheme().model_dump_json())
    previous = signal.getsignal(signal.SIGTERM)
    exit_code = main(["modal", str(source), "--out", str(out), "--cache-dir", str(tmp_path / "cache")])
    assert exit_code == 0
    assert signal.getsignal(signal.SIGTERM) == previous
    assert ModalDiagnosis.model_validate_json(out.read_text()) == diagnosis


def test_cache_only_cli_never_solves(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("命令列只查快取不可求解")
    monkeypatch.setattr(api, "diagnose_modes", forbidden)
    source, out = tmp_path / "scheme.json", tmp_path / "diagnosis.json"
    source.write_text(scheme().model_dump_json())
    exit_code = main(["modal", str(source), "--out", str(out), "--cache-dir", str(tmp_path / "cache"), "--cache-only"])
    assert exit_code == 0
    assert ModalDiagnosis.model_validate_json(out.read_text()).state is ModalDiagnosisState.NOT_COMPUTED


def test_modal_cli_program_error_leaves_no_document(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(*args: object, **kwargs: object) -> None:
        raise RuntimeError("入口壞了")
    monkeypatch.setattr(api, "diagnose_scheme", broken)
    source, out = tmp_path / "scheme.json", tmp_path / "diagnosis.json"
    source.write_text(scheme().model_dump_json())
    exit_code = main(["modal", str(source), "--out", str(out), "--cache-dir", str(tmp_path)])
    assert exit_code != 0
    assert not out.exists()


def test_modal_cli_requires_explicit_cache_directory(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["modal", "scheme.json", "--out", str(tmp_path / "out.json")])
    assert caught.value.code != 0


def test_sigterm_cleans_modal_write_directory_without_traceback(tmp_path: Path) -> None:
    source, out, cache = tmp_path / "scheme.json", tmp_path / "diagnosis.json", tmp_path / "cache"
    source.write_text(scheme().model_dump_json())
    cache.mkdir()
    script = tmp_path / "termination-runner.py"
    script.write_text("import runpy,sys,time\nfrom pathlib import Path\nfrom tempfile import TemporaryDirectory\n"
        "from aosr.reporting import modal_lookup\n"
        "def calculate(scheme, *, cache_dir):\n"
        "    with TemporaryDirectory(dir=cache_dir, prefix='modal-write-') as name:\n"
        "        Path(name, 'ready').touch()\n"
        "        while True: time.sleep(0.02)\n"
        "modal_lookup._calculate = calculate\n"
        f"sys.argv = ['scheme_cli', 'modal', {str(source)!r}, '--out', {str(out)!r}, '--cache-dir', {str(cache)!r}]\n"
        "runpy.run_module('aosr.reporting.scheme_cli', run_name='__main__')\n")
    process = subprocess.Popen([sys.executable, str(script)], cwd=tmp_path, start_new_session=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 20
        while not tuple(cache.glob("modal-write-*/ready")):
            assert process.poll() is None, "模態替身提前結束"
            assert time.monotonic() < deadline, "模態替身沒有進入暫存寫入"
            time.sleep(0.02)
        os.killpg(process.pid, signal.SIGTERM)
        _stdout, stderr = process.communicate(timeout=20)
        assert process.returncode == 128 + signal.SIGTERM
        assert stderr == ""
        assert not tuple(cache.glob("modal-write-*"))
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate(timeout=20)
