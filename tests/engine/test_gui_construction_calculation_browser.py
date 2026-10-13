"""網頁正式方案管線真算兩份小房間，只有一層厚度不同，三路與保存註記都要改變。"""
from __future__ import annotations

import time
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import cast

import httpx
import pytest
from playwright.sync_api import Browser, Page, expect

from governance.exit_codes import note
from aosr.reporting.result import SchemeResult
from aosr.reporting.scheme import Scheme
from tests.engine._furniture_cases import reference_document
from tests.engine._construction_gui_server import real_server
from tests.engine.test_gui_browser import _open, browser as browser
from tests.engine.test_gui_input_setup_browser import _filled, _save_readback


def test_real_server_cleans_calculation_group_when_exam_raises(tmp_path: Path) -> None:
    pid: int | None = None
    # 未被這個資料夾登記的行程必須存活；它與計算一樣另成行程組。
    with subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True) as unrelated:
        try:
            with pytest.raises(RuntimeError, match="考卷故意中斷"), real_server(tmp_path) as base:
                with httpx.Client(trust_env=False) as client:
                    response = client.put(f"{base}/api/schemes/small", json=small_document())
                    assert response.status_code == 200, response.text
                    response = client.post(f"{base}/api/runs", json={"scheme_id": "small"})
                    assert response.status_code == 200, response.text
                    state = client.get(f"{base}/api/runs/{response.json()['run_id']}").json()
                    pid = state["pid"]
                    assert state["status"] == "running"
                    os.killpg(pid, 0)
                    raise RuntimeError("考卷故意中斷")
            assert pid is not None
            with pytest.raises(ProcessLookupError):
                os.killpg(pid, 0)
            assert unrelated.poll() is None
        finally:
            if pid is not None:
                try:
                    os.killpg(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            unrelated.terminate()
            unrelated.wait(timeout=15)


def small_document() -> dict[str, object]:
    document = reference_document()
    document["scheme_id"] = "small"
    document["source_model"] = "omnidirectional"
    scene = cast(dict[str, object], document["scene"])
    scene["room_m"] = {"Lx": 1.6, "Ly": 1.2, "Lz": 1.0}
    scene["scattering_by_wall"] = {wall: .2 for wall in cast(dict[str, object], scene["impedance_pa_s_per_m_by_wall"])}
    document["speakers"] = {"left": {"x": .3, "y": .3, "z": .4}, "right": {"x": .3, "y": .9, "z": .4}}
    document["receiver_set"] = {"points": [
        {"receiver_id": "main", "role": "primary", "position_m": [1.2, .6, .6], "importance": 1.0},
        {"receiver_id": "side", "role": "surrounding", "position_m": [1.2, .7, .6],
         "importance": 1.0, "direction_relative_to_primary": "left"}]}
    return document


def calculate(page: Page, folder: Path, deadline: float) -> tuple[str, SchemeResult]:
    start = time.perf_counter()
    with page.expect_response("**/api/runs") as response:
        page.locator("#calculate").click()
    assert response.value.ok, response.value.text()
    identifier = str(response.value.json()["run_id"])
    page.wait_for_function("id => !document.getElementById('result-link').hidden && document.getElementById('result-link').href.includes(id)",
                           arg=identifier, timeout=max(1., (deadline - time.perf_counter()) * 1000))
    result = SchemeResult.model_validate_json((folder / "results" / f"{identifier}.json").read_text())
    note(f"網頁正式計算 {result.scheme.scheme_id} 牆鐘：{time.perf_counter() - start:.3f} 秒")
    return identifier, result


def assert_lanes_changed(a: SchemeResult, b: SchemeResult) -> None:
    assert a.scheme.scene.room_m == b.scheme.scene.room_m
    assert a.scheme.scene != b.scheme.scene
    for old, new in zip(a.pairs, b.pairs, strict=True):
        assert (old.speaker_id, old.receiver_id) == (new.speaker_id, new.receiver_id)
        assert old.report.scene.scene_fingerprint != new.report.scene.scene_fingerprint
        assert old.report.points is not None and new.report.points is not None
        left = {p.frequency_hz: p.fem_energy for p in old.report.points if p.fem_energy is not None}
        right = {p.frequency_hz: p.fem_energy for p in new.report.points if p.fem_energy is not None}
        common = left.keys() & right.keys()
        assert common and all(left[f] != right[f] for f in common)
        for x, y in zip(old.report.bands, new.report.bands, strict=True):
            assert x.center_frequency_hz == y.center_frequency_hz
            for field in ("reflected_energy", "interference_energy", "late_energy"):
                assert getattr(x, field) != getattr(y, field), field
        assert all(value != new.report.top.eyring_t60_by_band_s[f]
                   for f, value in old.report.top.eyring_t60_by_band_s.items())
        for pair in (old, new):
            notes = pair.report.construction_notes_by_wall
            assert notes is not None and "floor" in notes
            assert any(n.code == "komatsu_frequency_extrapolation" and n.layer_index == 1 and n.frequencies_hz
                       for n in notes["floor"])


def test_web_save_reload_real_calculation_changes_all_lanes_and_shows_notes(tmp_path: Path, browser: Browser) -> None:
    (tmp_path / "schemes").mkdir()
    (tmp_path / "schemes" / "small.json").write_text(Scheme.model_validate(small_document()).model_dump_json())
    before = time.perf_counter()
    # 等待上限只防卡死、不是時間預算：本機整題約 58 秒，雲端機器較慢，牆鐘照實印在 note。
    with real_server(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        page.locator("#scheme-list").select_option("small")
        with page.expect_response("**/api/input-preview"):
            page.locator("#open-scheme").click()
        page.wait_for_function("!document.querySelector('main').inert")
        _filled(page, "#material-kind-floor", "assembly")
        _filled(page, "#preset-floor", "glass_wool_96k_50mm")
        _save_readback(page, "construction-a")
        first_id, a = calculate(page, tmp_path, before + 600)
        page.locator("#layer-floor-0-thickness_m").fill("0.08")
        _save_readback(page, "construction-b")
        second_id, b = calculate(page, tmp_path, before + 600)
        assert first_id != second_id
        assert_lanes_changed(a, b)
        for identifier in (first_id, second_id):
            with _open(browser, f"{base}/results/{identifier}") as result_page:
                expect(result_page.page.locator("#construction-note-lines")).to_contain_text("地板第 1 層")
                expect(result_page.page.locator("#construction-note-lines")).to_contain_text("外推照算")
                assert result_page.page_errors == []
        assert watched.page_errors == []
    note(f"填表、存讀、兩份真計算與結果顯示整題牆鐘：{time.perf_counter() - before:.3f} 秒")
