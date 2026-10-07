"""附件報告純讀、段內不留空行、共振與分組不冒充排名。"""
from __future__ import annotations

from pathlib import Path

import pytest

from aosr.config.quality_targets import load_quality_targets
from aosr.reporting.modal_diagnosis_model import ModalDiagnosis, ModalDiagnosisState, RoomGroup
from aosr.reporting.modal_lookup import save_diagnosis
from aosr.reporting.scheme import Scheme
from aosr.search.modal_record import ModalRole, ModalSummary, write_summary
from aosr.search.report_modal import modal_report, modal_text
from tests.engine._modal_cases import sample
from tests.engine._search_modal_cases import change_first, prepared, protected


def _assert_each_resonance(rendered: str, diagnosis: ModalDiagnosis, registry: Path, scheme: Scheme) -> None:
    """由样本絕對大小及登記錨點獨立算判讀，不呼叫被測 readout 或顯示函式。"""
    import math
    import numpy as np
    from aosr.config.quality_targets import SettingEntry
    settings = load_quality_targets(registry).purpose(scheme.purpose)
    values = {}
    for suffix in ("reference_anchor_frequencies_hz", "reference_anchor_t60_s", "reference_extension_range_hz", "reference_extension_t60_s"):
        entry = settings.entry("low_frequency_decay." + suffix)
        assert isinstance(entry, SettingEntry)
        values[suffix] = entry.value
    frequencies = np.asarray(values["reference_anchor_frequencies_hz"], dtype=float)
    seconds = np.asarray(values["reference_anchor_t60_s"], dtype=float)
    extension = np.asarray(values["reference_extension_range_hz"], dtype=float)
    extension_seconds = values["reference_extension_t60_s"]
    assert isinstance(extension_seconds, (int, float))
    room, placement = diagnosis.room_layer, diagnosis.placement_layer
    assert room is not None and placement is not None
    modes = {m.mode_index: m for m in room.modes}
    verdicts = {}
    for mode in room.modes:
        assert mode.t60_s is not None
        f = mode.frequency_hz
        target = (float(extension_seconds) if extension[0] <= f < extension[1]
                  else float(np.interp(math.log(f), np.log(frequencies), seconds)) if frequencies[0] <= f <= frequencies[-1] else None)
        verdicts[mode.mode_index] = "參考線範圍外" if target is None else "超過" if mode.t60_s > target else "未超過"
    exceeding = sum(v == "超過" for v in verdicts.values())
    outside = sum(v == "參考線範圍外" for v in verdicts.values())
    assert f"超過目標線 {exceeding} 個；參考線範圍外 {outside} 個" in rendered
    channels = {c.speaker_id: c.role for c in scheme.channel_group.channels}
    for pair in placement.pairs:
        if pair.receiver_id != scheme.receiver_set.primary.receiver_id:
            continue
        strongest = max(pair.resonance_magnitude)
        selected = sorted(zip(pair.resonance_indices, pair.resonance_magnitude), key=lambda item: (-item[1], item[0]))[:3]
        chunks = []
        for index, magnitude in selected:
            mode = modes[index]
            assert mode.t60_s is not None
            f = mode.frequency_hz
            relative = "無有限相對 dB" if magnitude == 0 else f"{20 * math.log10(magnitude / strongest):.2f} dB"
            chunks.append(f"{f:.2f} Hz（T60 {mode.t60_s:.4g} 秒，{verdicts[index]}，{relative}）")
        label = {"left": "左聲道喇叭", "right": "右聲道喇叭"}[channels[pair.speaker_id]]
        assert f"原方案／{label} → 主位：" + "；".join(chunks) in rendered.splitlines()


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
    if diagnosis.room_layer is not None:
        # 100 Hz 的明列樣本延長衰減；同一張考卷同時出現超過、未超過及範圍外。
        import math
        modes = tuple(m.model_copy(update={"t60_s": 3.0, "omega_imag_rad_s": math.log(10),
                                          "q": math.pi * m.frequency_hz / math.log(10)}) if m.frequency_hz == 100 else m
                      for m in diagnosis.room_layer.modes)
        diagnosis = diagnosis.model_copy(update={"room_layer": diagnosis.room_layer.model_copy(update={"modes": modes})})
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
        assert "前幾名只是版面長度，不是品質門檻" in rendered
        _assert_each_resonance(rendered, diagnosis, registry, store.project)
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


@pytest.mark.parametrize("role,old", [("search_best", 0), ("refine_best", 1)])
def test_changed_first_validates_historical_scheme_and_keeps_diagnosis(tmp_path: Path, role: str, old: int) -> None:
    from aosr.search.modal_record import role_inputs
    store, registry, status = prepared(tmp_path)
    inputs = role_inputs(store, status)
    item = next(item for item in inputs if item.record.role == role)
    assert item.scheme is not None
    save_diagnosis(sample(item.scheme)[0], store.path / "modal-diagnosis" / "historical.json")
    record = item.record.model_copy(update={"state": "diagnosed_not_scored", "diagnosis_file": "historical.json"})
    write_summary(store.path, ModalSummary(cache_dir="cache", completed=True, roles=(record,)))
    change_first(store, role)
    before = {p: p.read_bytes() for p in store.path.rglob("*") if p.is_file()}
    rendered = modal_text(modal_report(store, status, load_quality_targets(registry)))
    assert "已診斷不計分" in rendered and "失敗" not in rendered
    assert f"舊的：當時是試算 {old}，現在是試算 2" in rendered
    assert "這是上次收尾時的診斷，之後搜尋又動過" in rendered
    assert "→ 主位：" in rendered
    assert before == {p: p.read_bytes() for p in store.path.rglob("*") if p.is_file()}
    item.scheme_path.unlink()
    unreadable = modal_text(modal_report(store, status, load_quality_targets(registry)))
    assert "失敗" in unreadable and "方案檔讀不回" in unreadable


@pytest.mark.parametrize("path", ["normal", "stop", "error"])
def test_report_reading_previous_summary_survives_concurrent_cleanup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                     path: str) -> None:
    """報告不拿資料夾鎖：剛讀完上一份摘要時，下一次 auto 的附件收尾清理，報告照樣讀得到那一份指到的文件。

    三條收尾都要守：正常收尾、算到一半被停（`_run_roles` 自己收）、附件出錯（`attach_modal` 照上一份摘要收）。
    """
    import aosr.search.report_modal as report_modal
    from aosr.search import modal_attach
    from aosr.search.modal_attach import AttachmentRecorded, attach_modal
    from aosr.search.modal_record import RoleInput, read_summary, role_inputs
    from aosr.search.run import SearchStatus
    from aosr.search.store import SearchStore
    from tests.engine._modal_cases import runner
    store, registry, status = prepared(tmp_path, refined=False)
    fake = runner(tmp_path / "runner", ModalDiagnosis(state=ModalDiagnosisState.NOT_COMPUTED))
    attach_modal(store, status=status, cache_dir=tmp_path / "cache", runner=fake)
    first = read_summary(store.path)
    assert first is not None
    quiet = modal_text(modal_report(store, status, load_quality_targets(registry)))
    assert "未計算" in quiet and "讀不回" not in quiet
    def interrupted(*args: object, **kwargs: object) -> int:
        raise KeyboardInterrupt

    def broken(*args: object, **kwargs: object) -> str:
        raise OSError("模擬磁碟滿")

    if path == "stop":
        monkeypatch.setattr(modal_attach, "_compute", interrupted)
    elif path == "error":
        monkeypatch.setattr(modal_attach, "_eligibility", broken)

    def interleaved(opened: SearchStore, current: SearchStatus, *, read_scope: bool = True) -> tuple[RoleInput, ...]:
        try:
            attach_modal(store, status=status, cache_dir=tmp_path / "cache", runner=fake)
        except AttachmentRecorded:
            assert path == "error"
        return role_inputs(opened, current, read_scope=read_scope)

    monkeypatch.setattr(report_modal, "role_inputs", interleaved)
    raced = modal_text(modal_report(store, status, load_quality_targets(registry)))
    second = read_summary(store.path)
    assert second is not None and {r.diagnosis_file for r in second.roles}.isdisjoint({r.diagnosis_file for r in first.roles})
    assert "未計算" in raced and "讀不回" not in raced
