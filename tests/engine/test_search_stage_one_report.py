"""第一階段只讀報告、完整峰谷寬度與現有網頁的相容性。"""
from __future__ import annotations

import json
import math
from pathlib import Path
from collections.abc import Callable
from typing import NoReturn

import pytest

from aosr.reporting.stage_one_result import load_stage_one_result, save_stage_one_result
from aosr.scoring.contract import Feature
from aosr.scoring.contract_base import Flag
from aosr.scoring.stage_one_low_frequency_contract import IDEAL_SOURCE_ASSUMPTION_TEXT
from tests.engine import _search_stage_one_cases as cases


def test_report_assumptions_order_scores_and_cli_dispatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                        capsys: pytest.CaptureFixture[str]) -> None:
    from aosr.search import cli
    from aosr.search.stage_one_report import build_stage_one_report, render_stage_one_text
    store = cases.make_store(tmp_path)
    cases.run(store, cases.FakeCompute(store, monkeypatch))
    report = build_stage_one_report(store)
    text = render_stage_one_text(report)
    assert text.startswith("在這個牆面假設下")
    assert IDEAL_SOURCE_ASSUMPTION_TEXT in text
    assert "拖尾不進自動排名" in text
    assert all(word not in text for word in ("原方案", "與喇叭無關", "第二階段子集"))
    expected = sorted((load_stage_one_result(p).score.screening_total, int(p.stem.split('-')[-1]))
                      for p in (store.path / "candidates").glob("trial-*.json") if "-scheme" not in p.stem)
    assert len({total for total, _ in expected}) > 1  # 替身合分要分得開，升降序才考得出來
    assert [row.trial_number for row in report.ranked] == [number for _, number in expected]
    def forbidden(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError("第一階段不可進第二階段報告")
    monkeypatch.setattr(cli, "build_report", forbidden)
    assert not cli.main(["report", "--search", str(store.path)])
    assert capsys.readouterr().out == text


@pytest.mark.parametrize("width,flags,label", [(.125, (), "0.12 八度"),
    (None, (Flag.FEATURE_BOUNDARY_INCOMPLETE,), "量不到（邊界不完整）"),
    (.0625, (Flag.FEATURE_NARROWER_THAN_AXIS,), "0.06 八度（窄於頻率軸）")])
def test_peak_and_dip_features_are_selected_by_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                    width: float | None, flags: tuple[Flag, ...], label: str) -> None:
    from aosr.search.stage_one_report import build_stage_one_report, render_stage_one_text
    store = cases.make_store(tmp_path, budget=1)
    cases.run(store, cases.FakeCompute(store, monkeypatch))
    result = load_stage_one_result(store.candidate_path(0))
    features = (Feature(kind="dip", center_frequency_hz=71., depth_db=-8.25, width_octave=width, flags=flags),
                Feature(kind="peak", center_frequency_hz=43., depth_db=6.75, width_octave=width, flags=flags))
    speakers = list(result.evaluation.speakers)
    for index, speaker in enumerate(speakers):
        assert speaker.payload is not None
        payload = speaker.payload.model_copy(update={"features": features, "strongest_peak_index": 1 if not index else None,
                                                    "deepest_dip_index": 0 if not index else None})
        speakers[index] = speaker.model_copy(update={"payload": payload})
    result = result.model_copy(update={"evaluation": result.evaluation.model_copy(update={"speakers": tuple(speakers)})})
    save_stage_one_result(result, store.candidate_path(0))
    text = render_stage_one_text(build_stage_one_report(store))
    assert "自己那條直線" in text
    assert f"43.00 Hz，6.75 dB，{label}" in text
    assert f"71.00 Hz，-8.25 dB，{label}" in text
    assert "最強峰 無；最深谷 無" in text


def test_unassessed_and_unreadable_results_keep_their_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from aosr.search.stage_one_report import build_stage_one_report, render_stage_one_text
    store = cases.make_store(tmp_path)
    cases.run(store, cases.FakeCompute(store, monkeypatch, unavailable=frozenset({1})))
    store.candidate_path(0).write_text("broken")
    text = render_stage_one_text(build_stage_one_report(store))
    assert "試算 0：結果讀回失敗" in text
    assert "試算 1：不可估" in text
    assert "non_positive_energy" in text
    assert "試算 2" in text


@pytest.mark.parametrize("damage", ["swapped", "identity"])
def test_result_file_must_belong_to_its_row(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str) -> None:
    # 放錯檔（別的試算的合法結果）或身分不同，這一列只寫讀回失敗，不印別人的主項與峰谷。
    from aosr.search.stage_one_report import build_stage_one_report, render_stage_one_text
    store = cases.make_store(tmp_path)
    cases.run(store, cases.FakeCompute(store, monkeypatch))
    target = store.candidate_path(0)
    assert target.is_file() and store.candidate_path(1).is_file()
    if damage == "swapped":
        target.write_bytes(store.candidate_path(1).read_bytes())
    else:
        result = load_stage_one_result(target)
        target.write_text(result.model_copy(update={"program_fingerprint": "calc-v1:" + "a" * 64}).model_dump_json())
    report = build_stage_one_report(store)
    row = next(item for item in (*report.ranked, *report.excluded) if item.trial_number == 0)
    assert row.result is None and row.error is not None and row.error.startswith("結果讀回失敗：結果檔的")
    assert f"試算 0：{row.error}" in render_stage_one_text(report)


def test_existing_search_page_and_best_endpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio
    from starlette.requests import Request
    from aosr.gui.app import GuiHandlers
    from aosr.gui.app import GuiSettings
    from aosr.gui.search_view import build_search_view
    store = cases.make_store(tmp_path, budget=1)
    cases.run(store, cases.FakeCompute(store, monkeypatch))
    view = build_search_view(store.path, server_physics=store.identity.physics_identity,
                             server_program=store.identity.program_fingerprint)
    best = next(block for block in view.blocks if block.key == "search-best")
    assert any("試算 0" in line for line in best.lines) and not best.warning
    # 沙箱不開網路與背景事件迴圈；只把執行緒排程改成同緒，端點與結果讀取照真流程。
    async def inline(function: Callable[..., object], *args: object, **kwargs: object) -> object:
        return function(*args, **kwargs)
    monkeypatch.setattr("aosr.gui.app.run_in_threadpool", inline)
    handlers = GuiHandlers(GuiSettings(data_dir=tmp_path, engine_commit="a" * 40))
    request = Request({"type": "http", "path_params": {"search_id": store.search_id}, "query_string": b""})
    response = asyncio.run(handlers.search_best(request))
    # 原本釘的是改動前三圖讀不到的行為；現在由搜尋第一名原始能量獨立算頻響答案。
    assert response.status_code == 200
    body = json.loads(bytes(response.body))
    status = json.loads(store.status_path.read_bytes())
    raw = json.loads(store.candidate_path(status["best_trial"]).read_bytes())
    assert body["frequency"]["error"] == ""
    assert [curve["points"] for curve in body["frequency"]["curves"]] == [
        [[f, 10 * math.log10(e) if e is not None and e > 0 else None]
            for f, e in zip(curve["primary"]["frequencies_hz"], curve["primary"]["fem_energy"], strict=True)]
        for curve in raw["curves"]]
    assert body["rfz"] == {"error": "", "absent": "第一階段只算 20～300 Hz 有限元素，沒有早期反射，這張圖不適用"}
    assert body["plan"]["error"] == "" and body["plan"]["data"]["room"] == raw["scheme"]["scene"]["room_m"]


def test_refine_report_required_labels_and_existing_view(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from aosr.search.stage_one_report import build_stage_one_report, render_stage_one_text
    from aosr.gui.search_view import build_search_view
    from tests.engine._stage_one_refine_cases import pool
    from tests.engine.test_search_stage_one_refine import ROWS, refine
    store = pool(tmp_path, monkeypatch, ROWS)
    assert "還沒細算" in render_stage_one_text(build_stage_one_report(store))
    status = refine(store, cases.FakeCompute(store, monkeypatch))
    text = render_stage_one_text(build_stage_one_report(store))
    assert status.state == "done"
    for phrase in ("細算（每 1 Hz 一點，只算有限元素）", "大峰＝最強峰高出這支喇叭自己那條直線多少，左右兩支取較差",
        "細算名次", "搜尋名次", "角色", "搜尋軸合分", "1 Hz 合分", "主項／穩健性", "大峰（搜尋軸／1 Hz）",
        "前 N 裡最低的大峰減差距", "4.000000", "總秒數", "1.000000",
        "較遠對照＝試算 5：主搜尋已採到的較遠候選中最佳（離前牆至少 0.8 m；0.8 是工程暫定的選席分界）",
        "大峰對照＝試算 1：", "房間固定，面積對照不適用", "入選後檢查見下一段"):
        assert phrase in text
    assert "本次未取得" not in text
    assert all(word not in text for word in ("原方案", "與喇叭無關", "第二階段子集", "較遠區最佳", "較遠區的最佳解"))
    view = build_search_view(store.path, server_physics=store.identity.physics_identity,
        server_program=store.identity.program_fingerprint)
    assert view.blocks


def test_refine_report_missing_controls_and_reasons(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from aosr.search.stage_one_report import build_stage_one_report, render_stage_one_text
    from tests.engine._stage_one_refine_cases import pool
    from tests.engine.test_search_stage_one_refine import refine
    store = pool(tmp_path, monkeypatch, [(0, 1., None, .5), (1, 2., None, .6)])
    store.candidate_path(0).write_text("broken")
    refine(store, cases.FakeCompute(store, monkeypatch, unavailable=frozenset({1})))
    text = render_stage_one_text(build_stage_one_report(store))
    assert "本次未取得大峰對照：前 N 名都量不到大峰" in text
    assert "本次未取得較遠對照，不因此判定較遠區不好" in text
    # 沒取到的席不留標籤：較遠的那句標籤與「大峰對照＝試算」都不准出現。
    assert "主搜尋已採到的較遠候選中最佳" not in text and "大峰對照＝試算" not in text
    assert "大峰對照門檻：無" in text
    assert "不可估" in text and "non_positive_energy" in text
    assert "讀不回" in text and "結果讀回失敗" in text


def test_refine_report_incomparable_and_bad_verification_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from aosr.search.stage_one_report import build_stage_one_report, render_stage_one_text
    from tests.engine._stage_one_refine_cases import pool
    from tests.engine.test_search_stage_one_refine import ROWS, refine
    store = pool(tmp_path, monkeypatch, ROWS)
    refine(store, cases.FakeCompute(store, monkeypatch, different=frozenset({2})))
    text = render_stage_one_text(build_stage_one_report(store))
    assert "不能同表：合分設定指紋" in text
    store.refine_result_path(8).write_bytes(store.candidate_path(8).read_bytes())
    text = render_stage_one_text(build_stage_one_report(store))
    assert "細算結果讀不回" in text


def test_refine_report_flags_result_swapped_after_refine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from aosr.search.stage_one_refine_report import stage_one_refine_lines
    from tests.engine._stage_one_refine_cases import pool
    from tests.engine.test_search_stage_one_refine import ROWS, refine
    store = pool(tmp_path, monkeypatch, ROWS)
    refine(store, cases.FakeCompute(store, monkeypatch))
    summary = store.path / "stage_one_refine.json"
    kept = summary.read_bytes()
    # 換成同方案、同出處、同物理與程式指紋、只有合分設定指紋不同的自洽結果，再放回舊摘要。
    store.refine_result_path(8).unlink()
    summary.unlink()
    refine(store, cases.FakeCompute(store, monkeypatch, different=frozenset({8})))
    summary.write_bytes(kept)
    row = next(line for line in stage_one_refine_lines(store) if "｜試算 8｜" in line)
    assert row.startswith("—｜") and "細算結果跟摘要記的比較身分不同" in row


def test_malformed_summary_spoils_only_refine_section(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from aosr.config.paths import config_path
    from aosr.search.stage_one_report import build_stage_one_report, render_stage_one_text
    from aosr.search.stage_one_selection import select_stage_one_candidates
    from tests.engine._stage_one_refine_cases import pool
    from tests.engine.test_search_stage_one_refine import ROWS, refine
    store = pool(tmp_path, monkeypatch, ROWS)
    refine(store, cases.FakeCompute(store, monkeypatch))
    summary = store.path / "stage_one_refine.json"
    document = json.loads(summary.read_bytes())
    del document["selection"]["candidates"][0]["params_m"]["spacing"]
    summary.write_text(json.dumps(document))
    assert "細算摘要內容不完整" in render_stage_one_text(build_stage_one_report(store))
    document["selection"]["missing"]["far_control"] = "本次未取得較遠對照"  # 有取到較遠對照又寫沒取到：模型拒收
    summary.write_text(json.dumps(document))
    assert "細算摘要讀不回" in render_stage_one_text(build_stage_one_report(store))
    # 壞的細算摘要不准擋住重新挑席。
    assert select_stage_one_candidates(store, registry_path=config_path("quality_targets.toml")).candidates
