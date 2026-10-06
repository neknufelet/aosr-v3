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


@pytest.mark.parametrize("code", [0, 1, 143])
def test_missing_document_nonzero_and_external_stop(tmp_path: Path, code: int) -> None:
    import sys
    from aosr.search.modal_attach import attach_modal
    store, _, status = prepared(tmp_path, refined=False)
    before = protected(store)
    script = tmp_path / "no-document.py"
    script.write_text(f"import sys\nsys.stderr.write('原文第一行\\n原文第二行\\n')\nsys.exit({code})\n")
    attach_modal(store, status=status, cache_dir=tmp_path / "cache", runner=(sys.executable, str(script)))
    summary = read_summary(store.path)
    assert summary is not None and summary.completed
    role = summary.roles[0]
    assert role.state == ("stopped" if code == 143 else "failed")
    if code == 0:
        assert "離開碼 0" in role.reason_text and "診斷文件" in role.reason_text
    elif code == 1:
        assert "原文第一行\n原文第二行\n" in role.reason_text
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
