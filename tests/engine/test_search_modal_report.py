"""附件報告純讀、段內不留空行、共振與分組不冒充排名。"""
from __future__ import annotations

from pathlib import Path

import pytest

from aosr.config.quality_targets import load_quality_targets
from aosr.reporting.modal_diagnosis_model import ModalDiagnosis, ModalDiagnosisState, RoomGroup
from aosr.reporting.modal_lookup import save_diagnosis
from aosr.search.modal_record import ModalRole, ModalSummary, write_summary
from aosr.search.report_modal import modal_report, modal_text
from tests.engine._modal_cases import sample
from tests.engine._search_modal_cases import prepared, protected


@pytest.mark.parametrize("state,text", [
    (ModalDiagnosisState.DIAGNOSED_NOT_SCORED, "已診斷不計分"),
    (ModalDiagnosisState.NOT_COMPUTED, "未計算"),
    (ModalDiagnosisState.FAILED, "失敗"),
    (ModalDiagnosisState.OUT_OF_SCOPE, "範圍外"),
])
def test_modal_report_four_states_and_readonly(tmp_path: Path, state: ModalDiagnosisState, text: str) -> None:
    store, registry, status = prepared(tmp_path)
    diagnosis = sample(store.project)[0] if state is ModalDiagnosisState.DIAGNOSED_NOT_SCORED else ModalDiagnosis(
        state=state, reason_text="保留原文\n第二行" if state is ModalDiagnosisState.FAILED else "",
        reason_code="unsupported_impedance" if state is ModalDiagnosisState.OUT_OF_SCOPE else None)
    save_diagnosis(diagnosis, store.path / "modal-diagnosis" / "original.json")
    write_summary(store.path, ModalSummary(cache_dir=str(tmp_path / "cache"), completed=True,
        roles=(ModalRole(role="baseline", state=state.value, diagnosis_file="original.json"),)))
    before = {p: p.read_bytes() for p in store.path.rglob("*") if p.is_file()}
    report = modal_report(store, status, load_quality_targets(registry))
    rendered = modal_text(report)
    assert text in rendered and "不計分、不改名次" in rendered
    assert "\n\n" not in rendered
    assert before == {p: p.read_bytes() for p in store.path.rglob("*") if p.is_file()}
    if state is ModalDiagnosisState.DIAGNOSED_NOT_SCORED:
        assert "沒有保證" in rendered
        assert "成員 4；25.00–270.00 Hz" in rendered
        assert "T60" in rendered and "超過目標線" in rendered and "參考線範圍外" in rendered
        assert "主位" in rendered and "主位前方" not in rendered
        assert "100.00 Hz" in rendered and "0.00 dB" in rendered
        assert "相對同一喇叭到同一座位最強的共振" in rendered
        assert "前幾名只是版面長度，不是門檻" in rendered
    elif state is ModalDiagnosisState.FAILED:
        assert "保留原文；第二行" in rendered


def test_singleton_group_is_not_called_overlap(tmp_path: Path) -> None:
    store, registry, status = prepared(tmp_path)
    diagnosis = sample(store.project)[0]
    assert diagnosis.room_layer is not None
    modes = tuple(mode.model_copy(update={"group_id": mode.mode_index}) for mode in diagnosis.room_layer.modes)
    groups = tuple(RoomGroup(group_id=m.mode_index, member_indices=(m.mode_index,),
        lower_hz=m.frequency_hz, upper_hz=m.frequency_hz) for m in modes)
    diagnosis = diagnosis.model_copy(update={"room_layer": diagnosis.room_layer.model_copy(update={"modes": modes, "groups": groups})})
    save_diagnosis(diagnosis, store.path / "modal-diagnosis" / "original.json")
    write_summary(store.path, ModalSummary(cache_dir="cache", completed=True,
        roles=(ModalRole(role="baseline", state="diagnosed_not_scored", diagnosis_file="original.json"),)))
    rendered = modal_text(modal_report(store, status, load_quality_targets(registry)))
    assert "單獨共振組" in rendered
    assert "重疊組 0 組" in rendered
    assert all("重疊組" not in line for line in rendered.splitlines() if "成員 1；" in line)


def test_duplicate_stop_and_incomplete_summary_are_honest(tmp_path: Path) -> None:
    store, registry, status = prepared(tmp_path)
    write_summary(store.path, ModalSummary(cache_dir="cache", roles=(
        ModalRole(role="baseline", state="stopped", reason_text="已停止"),
        ModalRole(role="search_best", trial_number=0, state="stopped", duplicate_of="baseline", reason_text="已停止"))))
    before = protected(store)
    rendered = modal_text(modal_report(store, status, load_quality_targets(registry)))
    assert "上次沒做完（可能進行中或被中斷）" in rendered
    assert "未計算：已停止" in rendered and "與 原方案 同擺位" in rendered
    assert protected(store) == before


@pytest.mark.parametrize("broken", ["summary", "diagnosis"])
def test_broken_attachment_is_local_to_modal_report(tmp_path: Path, broken: str) -> None:
    store, registry, status = prepared(tmp_path)
    write_summary(store.path, ModalSummary(cache_dir="cache", completed=True,
        roles=(ModalRole(role="baseline", state="diagnosed_not_scored", diagnosis_file="original.json"),)))
    path = store.path / "modal-diagnosis" / ("summary.json" if broken == "summary" else "original.json")
    path.write_text("{")
    rendered = modal_text(modal_report(store, status, load_quality_targets(registry)))
    assert "讀不回" in rendered
    assert "不計分、不改名次" in rendered
