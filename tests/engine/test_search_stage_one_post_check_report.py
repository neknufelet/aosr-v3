"""診斷報告全列超線、依超出量排序並隔離壞附件；主位參與夠大的先列成優先組。"""
from pathlib import Path
import json
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
        "優先組＝超過聽感參考線、而且主位參與（兩支喇叭各自算、取較大）不低於 -10 dB 的共振；這是優先檢視門檻、工程暫定，"
        "不是聽感門檻，也不是品質可不可接受的分界；其餘超線一根不刪、列在優先組後面；優先組為零，不表示沒有問題"):
        assert required in text
    for forbidden in ("好聽", "通過", "合格", "救得了", "救不了", "建議處理"):
        assert forbidden not in text
    assert "入選後檢查狀態：已收尾；" in text and "舊版模態程式" not in text
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
    # 只列主位：每一席每支喇叭恰好一行（替身另有六個非主位座位，不准出現）。
    assert sorted((line.split("／")[0], line.split(" → ")[0].rsplit("／", 1)[1].split("（")[0]) for line in placements) == \
        sorted((f"試算 {seat.trial_number}", speaker) for seat in result.seats for speaker in store.project.speakers)
    for seat in result.seats:
        for speaker in store.project.speakers:
            line = next(line for line in placements if f"試算 {seat.trial_number}／" in line and speaker in line)
            assert [line.index(f"{row[1]:.2f} Hz") for row in expected] == sorted(line.index(f"{row[1]:.2f} Hz") for row in expected)
            # 100 Hz 那根大小 4、最強那根 n+5：各席相對值不同，錯拿別席的值會對不上。
            assert "無有限相對 dB" in line and f"100.00 Hz（{20 * math.log10(4 / (seat.trial_number + 5)):.2f} dB）" in line
    assert "相對同一支喇叭到同一個座位最強的那根共振" in text
    timed = [s.seconds for s in result.seats if s.seconds is not None]
    assert f"總秒數：{sum(timed):.6f} 秒" in text and "沒有計時紀錄 0 份" in text
    full = render_stage_one_text(build_stage_one_report(store))
    assert full.index("入選後檢查見下一段") < full.index("入選後檢查（低頻模態診斷）")
    view = build_search_view(store.path, server_physics=store.identity.physics_identity, server_program=store.identity.program_fingerprint)
    assert view.search_id == store.search_id
    assert "modal" in {block.key for block in view.blocks}


@pytest.mark.parametrize("damage", ["missing", "summary", "one_file", "incomplete", "missing_file_field", "scheme_file",
                                    "state_mismatch", "dropped_seat", "room_differs"])
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
        elif damage == "scheme_file":
            doc = json.loads(path.read_bytes()); doc["seats"][0]["scheme_file"] = doc["seats"][1]["scheme_file"]; path.write_text(json.dumps(doc))
        elif damage == "state_mismatch":
            target = path.parent / str(result.seats[0].diagnosis_file)
            target.write_text(json.dumps({"state": "not_computed", "reason_text": "換過的文件"}))
        elif damage == "dropped_seat":
            doc = json.loads(path.read_bytes()); del doc["seats"][1]; path.write_text(json.dumps(doc))
        elif damage == "room_differs":
            # 試算 8 是第一份、照舊；試算 2 的房間層被改過但其他欄位合法：不准跟上面的超線表混在一起。
            target = path.parent / str(result.seats[1].diagnosis_file)
            doc = json.loads(target.read_bytes()); doc["room_layer"]["modes"][1]["t60_s"] = 20.; target.write_text(json.dumps(doc))
        else:
            target = path.parent / str(result.seats[0].diagnosis_file)
            target.write_text("{" if damage == "one_file" else '{"state":"diagnosed_not_scored"}')
    text = render_stage_one_text(build_stage_one_report(store))
    assert "名次｜試算編號" in text and "細算名次｜試算" in text
    if damage == "missing":
        assert "入選後檢查還沒做" in text
    elif damage in ("summary", "dropped_seat"):
        assert "入選後檢查摘要讀不回或內容不完整" in text
    elif damage == "room_differs":
        assert "試算 2／前 N（搜尋第 2 名）：失敗；診斷文件讀不回或內容不完整：這一席的房間層跟上面那份不同" in text
        assert "試算 8／" in text and not [line for line in text.splitlines() if line.startswith("試算 2／") and "→ 主位：" in line]
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


def test_old_modal_identity_is_flagged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # 模態程式改版：摘要與診斷彼此一致、都是舊身分；報告照樣讀得回，但要說是舊版算的。
    from aosr.search import stage_one_post_check_report as module
    store = prepared(tmp_path, monkeypatch); run(store, tmp_path / "child")
    monkeypatch.setattr(module, "modal_identity", lambda: "modal-v1:" + "c" * 64)
    text = "\n".join(module.stage_one_post_check_lines(store))
    assert module.OLD_IDENTITY in text and "舊版模態程式" in text
    assert "試算 2／前 N（搜尋第 2 名）：已診斷不計分" in text


def test_reference_span_and_below_line_follow_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # 範圍字照登記簿讀；拖尾低於參考線（超出量為零）的那一根不算超線。
    import dataclasses
    import json
    from aosr.config.quality_targets import QualityPurpose
    from aosr.scoring.low_frequency_decay_reference import LowFrequencyDecayReference
    from aosr.search import stage_one_post_check_report as module
    from aosr.search.stage_one_post_check import run_stage_one_post_check
    from tests.engine._stage_one_post_check_cases import runner
    store = prepared(tmp_path, monkeypatch); root = tmp_path / "child"
    command = runner(root, store)
    for fixture in root.glob("*.json"):
        doc = json.loads(fixture.read_bytes())
        doc["room_layer"]["modes"][1]["t60_s"] = .05
        fixture.write_text(json.dumps(doc))
    assert run_stage_one_post_check(store, cache_dir=root / "cache", lock_fd=None, runner=command).state == "done"
    original = load_reference
    def shifted(purpose: QualityPurpose) -> LowFrequencyDecayReference:
        reference = original(purpose)
        assert reference.target.extension is not None
        low, high, t60 = reference.target.extension
        return dataclasses.replace(reference, target=dataclasses.replace(reference.target, extension=(low + 8, high, t60)))
    monkeypatch.setattr(module, "load_reference", shifted)
    text = "\n".join(module.stage_one_post_check_lines(store))
    reference = shifted(load_quality_targets(config_path("quality_targets.toml")).purpose(store.settings.purpose))
    assert reference.target.extension is not None
    span = f"{reference.target.extension[0]:g}～{reference.target.frequencies_hz[-1]:g} Hz"
    assert span != "32～250 Hz" and f"參考線範圍外（{span} 以外），未評估" in text
    rows = [line for line in text.splitlines() if line.endswith("改善能力尚未確認")]
    assert [row.split("｜")[0] for row in rows] == ["108.00 Hz"]
    assert "超過參考線 1 個" in text


PRIORITY_KEY = "stage_one_post_check.priority_participation_db"


def test_priority_threshold_is_the_boss_value_in_the_stage_one_registry() -> None:
    # 老闆 2026-10-23：−10 dB，暫定工程的優先檢視門檻；住第一階段那一段、不進第二階段整份指紋。
    from aosr.config.quality_targets import SettingEntry
    entry = load_quality_targets(config_path("quality_targets.toml")).stage_one("dedicated_two_channel_listening_room").entry(PRIORITY_KEY)
    assert isinstance(entry, SettingEntry)
    assert (entry.value, entry.unit, entry.status, entry.source_kind) == (-10., "dB", "baseline", "engineering_recommendation")


def _registry_with(tmp_path: Path, threshold: float) -> Path:
    text = config_path("quality_targets.toml").read_text(encoding="utf-8")
    old = f'key = "{PRIORITY_KEY}"\nvalue = -10.0\n'
    assert old in text
    path = tmp_path / "quality_targets.toml"
    path.write_text(text.replace(old, f'key = "{PRIORITY_KEY}"\nvalue = {threshold!r}\n'), encoding="utf-8")
    return path


def _largest_db(pairs: list[dict[str, list[float]]], index: int) -> float | None:
    """從診斷文件原始大小獨立算：每支喇叭相對自己最強那根的 20·log10，大小為零的不算，取較大。"""
    values = [20 * math.log10(own / max(pair["resonance_magnitude"])) for pair in pairs
              if (own := dict(zip(pair["resonance_indices"], pair["resonance_magnitude"], strict=True))[index]) > 0]
    return max(values, default=None)


# 門檻：登記簿原值、剛好等於試算 8 那根 4／13 的相對 dB（等於也算優先組）、比它高一個最小浮點間隔、0（只有最強那根，它在範圍外）。
EDGE = 20 * math.log10(4 / 13)


@pytest.mark.parametrize("threshold", [-10., EDGE, math.nextafter(EDGE, 0.), 0.])
def test_priority_group_splits_by_larger_speaker_and_registry_threshold(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, threshold: float) -> None:
    from aosr.reporting.scheme import load_scheme
    from aosr.search import stage_one_post_check_report as module
    from aosr.search.stage_one_post_check import run_stage_one_post_check
    from tests.engine._stage_one_post_check_cases import runner
    store = prepared(tmp_path, monkeypatch); root = tmp_path / "child"
    command = runner(root, store)
    primary = load_scheme(store.scheme_path_for(store.refine_result_path(2))).receiver_set.primary.receiver_id
    # 超線的是 100 Hz（大小 4）與 108 Hz（大小 0，兩支都沒有有限相對值）。試算 2：第一支喇叭 100 Hz 改成 0，只剩另一支；
    # 試算 1：第一支喇叭 100 Hz 改成 0.5（-21.6 dB），另一支 -3.5 dB，取較大才進優先組；試算 8 不動，留給剛好等於門檻那一題。
    for number, index, size in ((2, 1, 0.), (1, 1, .5)):
        path = root / f"{number}.json"
        doc = json.loads(path.read_bytes())
        pair = next(pair for pair in doc["placement_layer"]["pairs"] if pair["receiver_id"] == primary)
        pair["resonance_magnitude"][pair["resonance_indices"].index(doc["room_layer"]["modes"][index]["mode_index"])] = size
        path.write_text(json.dumps(doc))
    result = run_stage_one_post_check(store, cache_dir=root / "cache", lock_fd=None, runner=command)
    assert result.state == "done"
    monkeypatch.setattr(module, "config_path", lambda name: _registry_with(tmp_path, threshold))
    text = "\n".join(module.stage_one_post_check_lines(store))
    assert f"不低於 {threshold:g} dB 的共振" in text
    reference = load_reference(load_quality_targets(config_path("quality_targets.toml")).purpose(store.settings.purpose))
    seen = []
    for seat in result.seats:
        doc = json.loads((store.path / "stage-one-check" / str(seat.diagnosis_file)).read_bytes())
        pairs = [pair for pair in doc["placement_layer"]["pairs"] if pair["receiver_id"] == primary]
        over = sorted(((mode["t60_s"] - target, mode["frequency_hz"], mode["mode_index"]) for mode in doc["room_layer"]["modes"]
                       if (target := reference.target_at(mode["frequency_hz"]).t60_s) is not None and mode["t60_s"] > target),
                      key=lambda row: (-row[0], row[1], row[2]))
        first = [f"{f:.2f} Hz" for _, f, i in over if (db := _largest_db(pairs, i)) is not None and db >= threshold]
        rest = [f"{f:.2f} Hz" for _, f, i in over if f"{f:.2f} Hz" not in first]
        line = next(line for line in text.splitlines() if line.startswith(f"試算 {seat.trial_number}／") and "／主位：" in line)
        head, groups = line.split("／主位：")[1].split("；優先組：")
        shown_first, shown_rest = groups.split("；其餘超線：")
        assert head == f"超過參考線 {len(over)} 個、優先組 {len(first)} 個" + ("" if first else "（優先組為零，不表示沒有問題）")
        assert [part.split("（")[0] for part in shown_first.split("、")] == (first or ["沒有"])
        assert [part.split("（")[0] for part in shown_rest.split("、")] == (rest or ["沒有"])
        seen.append((seat.trial_number, first))
    # 每個門檻各自的分組（從上面的原始算式來，這裡只核這組門檻真的走到了想測的分支）。
    usual = [(8, []), (2, ["100.00 Hz"]), (1, ["100.00 Hz"]), (5, ["100.00 Hz"])]
    expected = {-10.: usual, EDGE: [(8, ["100.00 Hz"]), *usual[1:]], math.nextafter(EDGE, 0.): usual,
                0.: [(8, []), (2, []), (1, []), (5, [])]}
    assert seen == expected[threshold]


def test_positive_priority_threshold_breaks_only_this_section(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # 相對最強那根的 dB 不會大於 0：正的門檻等於永遠沒有優先組，報讀不回、不悄悄全列成其餘。
    from aosr.search import stage_one_post_check_report as module
    store = prepared(tmp_path, monkeypatch); run(store, tmp_path / "child")
    monkeypatch.setattr(module, "config_path", lambda name: _registry_with(tmp_path, .5))
    text = "\n".join(module.stage_one_post_check_lines(store))
    assert f"入選後檢查摘要讀不回或內容不完整：{PRIORITY_KEY} 必須是不大於 0 的 dB 門檻" in text
    assert module.stage_one_post_check_summary(store).lines[-1] == f"入選後檢查摘要讀不回或內容不完整：{PRIORITY_KEY} 必須是不大於 0 的 dB 門檻"
