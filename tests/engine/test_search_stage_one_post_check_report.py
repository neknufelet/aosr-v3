"""診斷報告全列超線、依超出量排序並隔離壞附件。"""
from pathlib import Path
import math

import pytest

from aosr.config.paths import config_path
from aosr.config.quality_targets import load_quality_targets
from aosr.scoring.low_frequency_decay_reference import load_reference
from aosr.search.stage_one_report import build_stage_one_report, render_stage_one_text
from tests.engine._stage_one_post_check_cases import prepared, diagnosis
from tests.engine.test_search_stage_one_post_check import run


def test_report_required_text_order_values_participation_and_view(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from aosr.search.stage_one_post_check_report import stage_one_post_check_lines
    from aosr.gui.search_view import build_search_view
    store = prepared(tmp_path, monkeypatch); result = run(store, tmp_path / "child")
    text = "\n".join(stage_one_post_check_lines(store))
    for required in ("入選後檢查（低頻模態診斷）", "不計分、不改名次；搜尋結果與外圈結論不受診斷影響",
        "超過聽感參考線的共振（照超出量排序）", "參考線範圍外（32～250 Hz 以外），未評估",
        "主位參與門檻（『主位參與大』的量法與數值）還沒定，超線的先全列；每一席的主位參與附在下面，只排序、不設 dB 切線"):
        assert required in text
    for forbidden in ("好聽", "通過", "合格", "救得了", "救不了", "建議處理"):
        assert forbidden not in text
    sample = diagnosis(store, 8); assert sample.room_layer is not None
    reference = load_reference(load_quality_targets(config_path("quality_targets.toml")).purpose(store.settings.purpose))
    # 答案從原始 T60 與參考線獨立計算，不借 readout（讀回）的排序或超出量。
    expected = []
    outside = []
    for mode in sample.room_layer.modes:
        target = reference.target_at(mode.frequency_hz).t60_s
        if target is None:
            outside.append(mode)
        elif mode.t60_s is not None and mode.t60_s > target:
            expected.append((mode.t60_s - target, mode.frequency_hz, mode.mode_index, mode.t60_s, target))
    expected.sort(key=lambda row: (-row[0], row[1], row[2]))
    rows = [line for line in text.splitlines() if line.endswith("改善能力尚未確認")]
    assert rows == [f"{f:.2f} Hz｜T60 {t60:.6f} 秒｜參考線 T60 {target:.6f} 秒｜超出量 {excess:.6f} 秒；改善能力尚未確認"
                    for excess, f, _, t60, target in expected]
    assert f"房間層：共振 {len(sample.room_layer.modes)} 個；超過參考線 {len(expected)} 個；參考線範圍外 {len(outside)} 個" in text
    assert f"{len(outside)} 根參考線範圍外" in text
    placements = [line for line in text.splitlines() if "→ 主位：" in line]
    assert sample.placement_layer is not None
    for seat in result.seats:
        for speaker in store.project.speakers:
            line = next(line for line in placements if f"試算 {seat.trial_number}／" in line and speaker in line)
            assert [line.index(f"{row[1]:.2f} Hz") for row in expected] == sorted(line.index(f"{row[1]:.2f} Hz") for row in expected)
            assert "無有限相對 dB" in line and f"{20 * math.log10(4 / 4):.2f} dB" in line
    timed = [s.seconds for s in result.seats if s.seconds is not None]
    assert f"總秒數：{sum(timed):.6f} 秒" in text and "沒有計時紀錄 0 份" in text
    full = render_stage_one_text(build_stage_one_report(store))
    assert full.index("入選後檢查見下一段") < full.index("入選後檢查（低頻模態診斷）")
    view = build_search_view(store.path, server_physics=store.identity.physics_identity, server_program=store.identity.program_fingerprint)
    assert view.search_id == store.search_id
    assert "modal" in {block.key for block in view.blocks}


@pytest.mark.parametrize("damage", ["missing", "summary", "one_file", "incomplete", "missing_file_field"])
def test_bad_attachment_does_not_break_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str) -> None:
    import json
    store = prepared(tmp_path, monkeypatch)
    if damage != "missing":
        result = run(store, tmp_path / "child")
        path = store.path / "stage-one-check" / "summary.json"
        if damage == "summary":
            path.write_text("{")
        elif damage == "missing_file_field":
            doc = json.loads(path.read_bytes()); doc["seats"][0]["diagnosis_file"] = None; path.write_text(json.dumps(doc))
        else:
            target = path.parent / str(result.seats[0].diagnosis_file)
            target.write_text("{" if damage == "one_file" else '{"state":"diagnosed_not_scored"}')
    text = render_stage_one_text(build_stage_one_report(store))
    assert "名次｜試算編號" in text and "細算名次｜試算" in text
    if damage == "missing":
        assert "入選後檢查還沒做" in text
    elif damage == "summary":
        assert "入選後檢查摘要讀不回" in text
    else:
        assert "試算 8" in text and "診斷文件讀不回" in text
        assert "試算 2" in text and "已診斷不計分" in text


def test_report_untimed_is_not_zero_seconds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json
    store = prepared(tmp_path, monkeypatch); result = run(store, tmp_path / "child")
    path = store.path / "stage-one-check" / "summary.json"
    doc = json.loads(path.read_bytes()); doc["seats"][0]["seconds"] = None; path.write_text(json.dumps(doc))
    timed = [s.seconds for s in result.seats[1:] if s.seconds is not None]
    text = render_stage_one_text(build_stage_one_report(store))
    assert f"總秒數：{sum(timed):.6f} 秒（有計時的席加總）；沒有計時紀錄 1 份" in text
