"""四態、逐模態參考線、重疊分組與擺位排序的畫面資料。"""
from aosr.gui.modal_view import build_modal_view
from aosr.reporting.modal_diagnosis_model import ModalDiagnosis, ModalDiagnosisState
from tests.engine._modal_cases import sample, scheme


def test_groups_keep_bounds_members_and_each_resonance() -> None:
    diagnosis, _ = sample()
    view = build_modal_view(diagnosis, scheme())
    assert "不是一個共振" in view["group_note"]
    assert "相鄰共振的峰寬互相重疊" in view["group_note"]
    room = diagnosis.room_layer
    assert room is not None
    for shown, group in zip(view["groups"], room.groups, strict=True):
        assert shown["lower_text"] == f"{group.lower_hz:.2f} Hz"
        assert shown["upper_text"] == f"{group.upper_hz:.2f} Hz"
        assert shown["member_text"] == str(len(group.member_indices))
    assert [row["index"] for row in view["modes"]] == [m.mode_index for m in room.modes]
    outside = [row for row in view["modes"] if row["index"] in {0, 3}]
    assert outside and all(row["target_text"] == "參考線範圍外，未評估" for row in outside)
    assert all(row["excess_text"] == "未評估" for row in outside)
    assert "沒有保證" in view["guarantee_text"] and "無限" not in view["guarantee_text"]
    assert all("t60_text" not in row for row in view["groups"])


def test_placement_sorted_by_absolute_size_without_quality_cutoff() -> None:
    view = build_modal_view(sample()[0], scheme())
    assert "相對同一喇叭到同一座位最強的共振" in view["placement_note"]
    assert "不是品質門檻" in view["placement_note"]
    for pair in view["placements"]:
        assert [row["index"] for row in pair["rows"]] == [1, 2, 0, 3]
        assert pair["rows"][0]["relative_text"] == "0.00 dB"
        assert pair["rows"][-1]["relative_text"] == "無有限相對 dB"
        assert pair["profiles"]


def test_four_state_texts_and_original_failure() -> None:
    cases = (
        (sample()[0], "已診斷不計分"),
        (ModalDiagnosis(state=ModalDiagnosisState.NOT_COMPUTED), "未計算"),
        (ModalDiagnosis(state=ModalDiagnosisState.FAILED, reason_text="ValueError('原文')"), "失敗"),
        (ModalDiagnosis(state=ModalDiagnosisState.OUT_OF_SCOPE, reason_code="unsupported_impedance"), "範圍外"),
    )
    for diagnosis, text in cases:
        shown = build_modal_view(diagnosis, scheme())
        assert shown["state_text"] == text
        assert shown["can_calculate"] == (diagnosis.state is ModalDiagnosisState.NOT_COMPUTED)
    assert build_modal_view(cases[2][0], scheme())["reason_text"] == "ValueError('原文')"
