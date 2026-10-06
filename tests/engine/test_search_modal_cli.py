"""搜尋 CLI 在結果已保存後附診斷；附件異常與停止都不改外圈離開碼。"""
from __future__ import annotations

from pathlib import Path

import pytest

from aosr.config.paths import config_path
from aosr.reporting.modal_diagnosis_model import ModalDiagnosis, ModalDiagnosisState
from aosr.search import cli
from aosr.search.modal_record import read_summary
from tests.engine._modal_cases import runner, sample
from tests.engine._search_modal_cases import prepared, protected
from tests.engine._search_outer_cases import OuterCompute


@pytest.mark.parametrize("outcome", ["diagnosed", "failed", "stopped", "exception"])
def test_auto_attachment_preserves_every_saved_byte_and_exit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                            outcome: str, capsys: pytest.CaptureFixture[str]) -> None:
    store, registry, status = prepared(tmp_path)
    # 全部同擺位；成功替身只有這組明列資料。取法與不同擺位另有考卷。
    for result in (store.candidate_path(0), store.refine_result_path(1)):
        store.scheme_path_for(result).write_text(store.project.model_dump_json())
    before = protected(store)
    monkeypatch.setattr(cli, "auto_search", lambda *args, **kwargs: status)
    diagnosis = sample(store.project)[0] if outcome == "diagnosed" else ModalDiagnosis(
        state=ModalDiagnosisState.FAILED, reason_text="模態原始錯誤")
    if outcome in ("stopped", "exception"):
        def broken(*args: object, **kwargs: object) -> None:
            raise KeyboardInterrupt if outcome == "stopped" else RuntimeError("附件入口錯誤")
        monkeypatch.setattr(cli, "attach_modal", broken)
    code = cli.main(["auto", str(store.path), "--engine-commit", "test", "--modal-cache-dir", str(tmp_path / "cache")],
                    compute_factory=lambda opened, capabilities, commit: OuterCompute(opened),
                    modal_runner=runner(tmp_path / "runner", diagnosis))
    assert code == 0 and protected(store) == before
    assert not store.stop_path.exists() and not store.refine_stop_path.exists()
    summary = read_summary(store.path)
    assert summary is not None
    if outcome == "diagnosed":
        assert all(role.state == "diagnosed_not_scored" for role in summary.roles)
    elif outcome == "failed":
        assert all(role.state == "failed" for role in summary.roles)
    else:
        assert "搜尋結果不受影響" in capsys.readouterr().err


@pytest.mark.parametrize("code", [0, 1, 143, -15, 129, -1, -9, 137])
def test_missing_document_nonzero_and_external_stop(tmp_path: Path, code: int) -> None:
    import sys
    from aosr.search.modal_attach import attach_modal
    store, _, status = prepared(tmp_path, refined=False)
    before = protected(store)
    script = tmp_path / "no-document.py"
    # 先把訊號處理還原成預設：從 nohup 之類忽略掛斷訊號的環境跑，替身會繼承「忽略」，送給自己也死不了。
    ending = ("signal.signal(signal.SIGHUP, signal.SIG_DFL); signal.signal(signal.SIGTERM, signal.SIG_DFL); "
              f"os.kill(os.getpid(), {-code})" if code < 0 else f"sys.exit({code})")
    script.write_text("import sys,os,signal\nsys.stderr.write('原文第一行\\n原文第二行\\n'); sys.stderr.flush()\n" + ending + "\n")
    attach_modal(store, status=status, cache_dir=tmp_path / "cache", runner=(sys.executable, str(script)))
    summary = read_summary(store.path)
    assert summary is not None and summary.completed
    role = summary.roles[0]
    assert role.state == ("stopped" if code in (143, -15, 129, -1) else "failed")
    if code == 0:
        assert "離開碼 0" in role.reason_text and "診斷文件" in role.reason_text
    else:
        assert "原文第一行\n原文第二行\n" in role.reason_text
        if code in (-9, 137):
            assert f"模態工作非正常結束（離開碼 {code}）" in role.reason_text
            assert "可能是記憶體不足被系統強制結束" in role.reason_text
        elif role.state == "stopped":
            assert f"已停止（子行程被外部訊號 {abs(code) if code < 0 else code - 128} 停止）" in role.reason_text
    assert protected(store) == before


def test_auto_requires_explicit_modal_cache_directory(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as caught:
        cli.main(["auto", str(tmp_path), "--engine-commit", "test"])
    assert caught.value.code != 0


def test_real_auto_manual_rerun_reuses_success_without_rewriting_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store, registry, _ = prepared(tmp_path)
    for result in (store.candidate_path(0), store.refine_result_path(1)):
        store.scheme_path_for(result).write_text(store.project.model_dump_json())
    monkeypatch.setattr(cli, "config_path", lambda name: registry if name.startswith("quality_targets") else config_path(name))
    monkeypatch.setattr(cli, "_identity", lambda *args: store.identity)
    args = ["auto", str(store.path), "--engine-commit", "test", "--modal-cache-dir", str(tmp_path / "cache")]
    before = protected(store)
    first = cli.main(args, compute_factory=lambda opened, capabilities, commit: OuterCompute(opened),
                     modal_runner=runner(tmp_path / "runner", sample(store.project)[0]))
    assert first == 0
    summary = read_summary(store.path)
    assert summary is not None and summary.roles[0].state == "diagnosed_not_scored"
    second = cli.main(args, compute_factory=lambda opened, capabilities, commit: OuterCompute(opened), modal_runner=("must-not-run",))
    assert second == 0 and protected(store) == before
    again = read_summary(store.path)
    assert again is not None and again.roles[0].diagnosis_file == summary.roles[0].diagnosis_file


@pytest.mark.parametrize("branch,conclusion,code", [("roles", "refine_budget", 0), ("entry", "refine_budget", 0),
                                                   ("entry", "refine_failed", 1), ("entry", "refine_interrupted", 3)])
def test_broken_stderr_preserves_completed_stop_summary_and_outer_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, branch: str, conclusion: str, code: int,
) -> None:
    import sys
    from aosr.search import modal_attach
    from aosr.search.outer_status import OuterStatus
    store, _, status = prepared(tmp_path)
    status = status.model_copy(update={"outer": OuterStatus.model_validate({"conclusion": conclusion})})
    before = protected(store)
    monkeypatch.setattr(cli, "auto_search", lambda *args, **kwargs: status)

    def interrupted(*args: object, **kwargs: object) -> None:
        raise KeyboardInterrupt

    class BrokenStderr:
        def write(self, text: str) -> int:
            summary = read_summary(store.path)
            assert summary is not None and summary.completed
            assert all(r.state == "stopped" for r in summary.roles)
            raise BrokenPipeError("標準錯誤管線已關")

    monkeypatch.setattr(cli if branch == "entry" else modal_attach, "attach_modal" if branch == "entry" else "_compute", interrupted)
    args = ["auto", str(store.path), "--engine-commit", "test", "--modal-cache-dir", str(tmp_path / "cache")]
    with monkeypatch.context() as patch:
        patch.setattr(sys, "stderr", BrokenStderr())
        result = cli.main(args, compute_factory=lambda opened, *_: OuterCompute(opened), modal_runner=("must-not-run",))
    assert result == code and protected(store) == before
    summary = read_summary(store.path)
    assert summary is not None and summary.completed and all(r.state == "stopped" for r in summary.roles)


def test_stop_after_completion_warns_and_next_auto_preserves_diagnosed_roles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    from aosr.search.modal_attach import attach_modal
    from aosr.search.modal_record import read_diagnosis
    from aosr.search.run import SearchStatus
    from tests.engine._search_modal_cases import scheme_for
    store, registry, status = prepared(tmp_path)
    store.scheme_path_for(store.candidate_path(0)).write_text(store.project.model_dump_json())
    previous = attach_modal(store, status=status, cache_dir=tmp_path / "cache",
                            runner=runner(tmp_path / "runner", sample(store.project)[0]))
    saved = {r.role: r for r in previous.roles if r.state == "diagnosed_not_scored"}
    documents = {r.diagnosis_file: (store.path / "modal-diagnosis" / str(r.diagnosis_file)).read_bytes() for r in saved.values()}
    monkeypatch.setattr(cli, "config_path", lambda name: registry if name.startswith("quality_targets") else config_path(name))
    monkeypatch.setattr(cli, "_identity", lambda *args: store.identity)
    stop_code = cli.main(["stop", str(store.path)])
    assert stop_code == 0
    stopped_bytes = protected(store)
    args = ["auto", str(store.path), "--engine-commit", "test", "--modal-cache-dir", str(tmp_path / "cache")]
    auto_code = cli.main(args, compute_factory=lambda opened, *_: OuterCompute(opened), modal_runner=("must-not-run",))
    assert auto_code == 0
    current = SearchStatus.model_validate_json(store.status_path.read_bytes())
    assert current.outer.conclusion == "user_stopped" and store.stop_path.exists()
    assert {k: v for k, v in protected(store).items() if k != "status.json"} == {k: v for k, v in stopped_bytes.items() if k != "status.json"}
    summary = read_summary(store.path)
    assert summary is not None and summary.completed and "搜尋沒有正常收尾" in summary.reason_text
    for role in summary.roles:
        if role.role in saved:
            assert role == saved[role.role]
            assert read_diagnosis(store.path, role, scheme_for(store, role.role)).state is ModalDiagnosisState.DIAGNOSED_NOT_SCORED
            assert (store.path / "modal-diagnosis" / str(role.diagnosis_file)).read_bytes() == documents[role.diagnosis_file]
        else:
            assert role.state == "skipped"
    warning = capsys.readouterr().err
    assert "這個搜尋已有外圈結論（細算用完上限，未完成）" in warning
    assert "下一次 auto 會照停止記號判成使用者停止" in warning
    assert "SIGTERM（終止訊號）" in warning


def test_real_broken_stderr_pipe_keeps_outer_exit_code(tmp_path: Path) -> None:
    """真的開行程：標準錯誤接在沒人讀的管子上時被中斷，結束前清緩衝也不准把離開碼改成 120。

    替身換掉 sys.stderr 的考卷量的是 cli.main 的回傳值，量不到 Python 結束那一步；這題看行程真正的離開碼。
    """
    import json
    import os
    import signal
    import subprocess
    import sys
    import time
    from aosr.runtime import child_process_env
    store, registry, _ = prepared(tmp_path)
    ready = tmp_path / "ready.json"
    child = tmp_path / "child.py"
    child.write_text("import json, os, time\nfrom pathlib import Path\n"
                     f"Path({str(ready)!r}).write_text(json.dumps({{'pid': os.getpid()}}))\n"
                     "while True:\n    time.sleep(0.02)\n")
    args = ["auto", str(store.path), "--engine-commit", "test", "--modal-cache-dir", str(tmp_path / "cache")]
    parent = tmp_path / "parent.py"
    parent.write_text("from pathlib import Path\nfrom aosr.search import cli\nfrom aosr.config.paths import config_path\n"
                      "from aosr.search.store import SearchStore\n"
                      f"store = SearchStore.open(Path({str(store.path)!r}))\ncli._identity = lambda *args: store.identity\n"
                      f"cli.config_path = lambda name: Path({str(registry)!r}) if name.startswith('quality_targets') else config_path(name)\n"
                      f"raise SystemExit(cli.main({args!r}, compute_factory=lambda *args: object(), "
                      f"modal_runner=({sys.executable!r}, {str(child)!r})))\n")
    process = subprocess.Popen([sys.executable, str(parent)], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                               env=child_process_env(threads=1), start_new_session=True)
    try:
        deadline = time.monotonic() + 60
        while not ready.exists():
            assert process.poll() is None and time.monotonic() < deadline
            time.sleep(0.02)
        assert process.stderr is not None
        process.stderr.close()
        process.send_signal(signal.SIGINT)
        assert process.wait(timeout=30) == 0
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        if ready.exists():
            try:
                os.kill(json.loads(ready.read_text())["pid"], signal.SIGKILL)
            except ProcessLookupError:
                pass
    summary = read_summary(store.path)
    assert summary is not None and summary.completed and all(r.state == "stopped" for r in summary.roles)
