"""四態、逐模態參考線、重疊分組與擺位排序的畫面資料。"""
import math
from dataclasses import replace

from aosr.gui.modal_view import build_modal_view
from aosr.physics.fem_modal_check import FemModalCheck
from aosr.physics.modal_convention import modal_quantities
from aosr.reporting.modal_diagnosis import room_layer_from_spectrum
from aosr.reporting.modal_diagnosis_model import ModalDiagnosis, ModalDiagnosisState
from aosr.reporting.scheme import Scheme
from tests.engine._modal_cases import sample, scheme


def test_groups_keep_bounds_members_and_each_resonance() -> None:
    diagnosis, _ = sample()
    view = build_modal_view(diagnosis, scheme())
    assert "不是只有一個共振" in view["group_note"]
    assert "自成一組" in view["group_note"]
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


def narrow_sample() -> ModalDiagnosis:
    """獨立窄峰資料，不改其他考卷共用的頻率和擺位四元組。"""
    diagnosis, cached = sample()
    modes = tuple(replace(mode, omega=complex(2 * math.pi * f, 17),
        raw_omega=complex(2 * math.pi * f, 17), frequency_hz=f,
        t60_s=3 * math.log(10) / 17, q=math.pi * f / 17)
        for mode, f in zip(cached.spectrum.solutions, (28.6, 42.9, 51.6, 57.2), strict=True))
    assert diagnosis.key is not None and diagnosis.modal_identity is not None
    room = room_layer_from_spectrum(replace(cached.spectrum, solutions=modes), key=diagnosis.key,
        identity=diagnosis.modal_identity, mesh_sha256=cached.room_layer.mesh_sha256, solve_seconds=0)
    return diagnosis.model_copy(update={"room_layer": room})


def test_narrow_peaks_explain_standalone_resonance() -> None:
    view = build_modal_view(narrow_sample(), scheme())
    assert any(row["lower_text"] == row["upper_text"] for row in view["groups"])
    assert "單獨共振" in view["group_note"] and "自成一組" in view["group_note"]


def checked_sample(check: FemModalCheck, value: Scheme | None = None) -> ModalDiagnosis:
    diagnosis, cached = sample(value)
    assert diagnosis.key is not None and diagnosis.modal_identity is not None
    room = room_layer_from_spectrum(replace(cached.spectrum, check=check), key=diagnosis.key,
        identity=diagnosis.modal_identity, mesh_sha256=cached.room_layer.mesh_sha256, solve_seconds=0)
    return diagnosis.model_copy(update={"room_layer": room})


def test_self_check_explains_formula_and_t60_threshold() -> None:
    t60 = modal_quantities(complex(0, 287.4), zero_rad_s=0.0).t60_s
    assert t60 is not None
    check = FemModalCheck(287.4, t60, (), 1, 0, 0, 1, 61)
    view = build_modal_view(checked_sample(check), scheme())
    assert "Neumann:" not in view["check_note"] and "不含稜邊項" in view["check_note"]
    assert view["weyl_raw"] == check.weyl_terms
    assert "T60 不短於" in view["guarantee_text"]
    assert f"{check.guaranteed_min_t60_s:.6g}" in view["guarantee_text"]
    assert "這是門檻，不是找到的共振裡最短的 T60" in view["guarantee_text"]
    custom = replace(check, weyl_terms="求解器另報的原式")
    assert build_modal_view(checked_sample(custom), scheme())["weyl_raw"] == custom.weyl_terms
    assert "沒有 T60 門檻" in build_modal_view(sample()[0], scheme())["guarantee_text"]


def test_first_calculation_memory_is_measured_only_in_reference_room() -> None:
    view = build_modal_view(ModalDiagnosis(state=ModalDiagnosisState.NOT_COMPUTED), scheme())
    assert "記憶體約 5 GB（也只在上述房間量過" in view["compute_note"]


def test_placement_sorted_by_absolute_size_without_quality_cutoff() -> None:
    view = build_modal_view(sample()[0], scheme())
    assert "相對同一喇叭到同一座位最強的共振" in view["placement_note"]
    assert view["placement_note"] == "相對 dB：相對同一喇叭到同一座位最強的共振。前幾名只是版面長度，不是品質門檻；完整排序與整群大小剖面可展開。"
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
