"""#1092 分區結果頁走正式網頁行程（貼棉那一區要雙精度讀回，pytest 本身不開雙精度，見 tests/engine/conftest.py）。

一題真瀏覽器走完整條路：打開設計的結果頁先看到逐區表、原牆基準還沒算、而且沒有任何計算被開；按「算原牆基準」，
正式網頁行程自己開正式計算（真算，不替身）；算完這一節自己換成改善多少，數字等於拿兩份結果檔獨立算的那一份。
算好之後再改檔走過其餘狀態：較舊的紀錄不蓋掉最新的、基準方案對不上、基準或設計在正式讀回被拒收、舊版物理、讀不出、沒算完。
另一題只打入口：整面一區的結果沒有這一節、舊版物理的設計不給開、原方案切線劃過設計的一區照實拒收、原方案不在了、
設計設定讀壞了都照講。
"""
from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from playwright.sync_api import Browser, expect

from aosr.config.paths import config_path
from aosr.config.quality_targets import load_quality_targets
from aosr.gui.zone_design import DesignSettings
from aosr.gui.zone_result import derive_baseline, improvement
from aosr.physics.report_surfaces import RectangleZone
from aosr.reporting.result import SchemeResult
from aosr.reporting.scheme import Scheme
from tests.engine._construction_gui_server import real_server
from tests.engine._zone_result_cases import original_x0, real_design
from tests.engine._zoned_report_cases import small_scheme, with_x0
from tests.engine.test_gui_browser import _open, browser as browser
from tests.engine.test_scheme_pipeline import shared_control_result
from aosr.materials.construction_models import ConstantImpedance

DESIGN_RUN, OLD_RUN, WHOLE_RUN = "d" * 32, "e" * 32, "f" * 32


@pytest.fixture(scope="module")
def design_text(tmp_path_factory: pytest.TempPathFactory, worker_id: str) -> str:
    return real_design(tmp_path_factory, worker_id)


def _put(folder: Path, run_id: str, text: str) -> None:
    (folder / "results").mkdir(exist_ok=True)
    (folder / "results" / f"{run_id}.json").write_text(text, encoding="utf-8")


def _runs(folder: Path) -> list[Path]:
    return sorted((folder / "runs").glob("*.json"))


def test_zone_page_lists_zones_then_computes_the_baseline_and_shows_the_improvement(
        tmp_path: Path, browser: Browser, design_text: str) -> None:
    design = SchemeResult.model_validate_json(design_text)
    _put(tmp_path, DESIGN_RUN, design_text)
    with real_server(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        # 冷的網頁行程第一次讀回這一份要編譯重評（本機 16.7 秒、熱的 7.4 秒；雲端搶核時超過導覽等網路靜下來的 30 秒，
        # 11-11 shard-c 紅過）：導覽只等頁面本身，讀回交給下面有寬限的等待。
        page.goto(f"{base}/results/{DESIGN_RUN}", wait_until="domcontentloaded")
        expect(page.locator("#zone-result")).to_be_visible(timeout=180_000)
        expect(page.locator("#zone-rows tr")).to_have_count(1 + 7)
        expect(page.locator('#zone-rows tr[data-changed="true"]')).to_have_count(1)
        expect(page.locator('#zone-rows tr[data-changed="true"]')).to_contain_text("拆掉這一區加貼的層")
        expect(page.locator("#zone-state")).to_contain_text("原牆基準還沒算")
        assert _runs(tmp_path) == []  # 打開結果頁、查附表都不開算
        page.locator("#zone-baseline-start").click()
        expect(page.locator("#zone-start-state")).to_contain_text("開始算了", timeout=60_000)
        expect(page.locator("#zone-baseline-start")).to_be_hidden(timeout=60_000)
        with httpx.Client(base_url=base, trust_env=False, timeout=120) as client:
            running = client.get(f"/api/results/{DESIGN_RUN}/zones").json()
            assert (running["state"], running["running"], running["can_start"]) == ("running", True, False)
            assert client.post(f"/api/results/{DESIGN_RUN}/zones/baseline", json={}).status_code == 409
        expect(page.locator("#zone-improvement tr.zone-total")).to_be_visible(timeout=600_000)
        (run,) = _runs(tmp_path)
        baseline = SchemeResult.model_validate_json((tmp_path / "results" / f"{run.stem}.json").read_bytes())
        assert baseline.scheme == derive_baseline(design.scheme).scheme
        expected = improvement(design, baseline, load_quality_targets(config_path("quality_targets.toml")))
        assert expected.total is not None
        shown = page.locator("#zone-improvement tr").all_inner_texts()[1:]
        assert [line.split("\t") for line in shown] == [
            [row.category, row.baseline, row.design, row.change] for row in (*expected.rows, expected.total)]
        verdicts = page.locator("#zone-improvement td[data-verdict]").evaluate_all(
            "cells => cells.map(cell => cell.dataset.verdict)")
        assert verdicts == [row.verdict for row in (*expected.rows, expected.total)]
        expect(page.locator("#zone-baseline-link a")).to_have_attribute("href", f"/results/{run.stem}")
        with httpx.Client(base_url=base, trust_env=False, timeout=120) as client:
            assert client.get(f"/api/results/{DESIGN_RUN}/zones").json()["improvement"] == expected.model_dump(mode="json")
            assert client.post(f"/api/results/{DESIGN_RUN}/zones/baseline", json={}).status_code == 409
            assert len(_runs(tmp_path)) == 1
            assert watched.page_errors == [] and watched.console_errors == []
            _later_states(tmp_path, client, run, expected.model_dump(mode="json"))


def _tampered(document: dict[str, object]) -> str:
    """格式照樣讀得進來、正式讀回才拒收的一份：逐點表第一點的頻率跟宣告的低頻軸差 0.001 Hz。"""
    changed = json.loads(json.dumps(document))
    changed["pairs"][0]["report"]["points"][0]["frequency_hz"] += 0.001
    return json.dumps(changed)


def _later_states(folder: Path, client: httpx.Client, run: Path, expected: object) -> None:
    """算好之後的幾格（改檔模擬，不再真算）。

    多一筆較舊的同方案紀錄不換掉最新那一筆；基準方案跟這一次推出來的不一樣、基準或設計格式讀得進來卻在正式讀回
    被拒收，都照講、不出數字；舊版物理、讀不出來、上一次沒算完，都要重算。
    """
    zones = f"/api/results/{DESIGN_RUN}/zones"
    record = json.loads(run.read_text(encoding="utf-8"))
    older = record | {"run_id": "0" * 32, "status": "failed", "exit_code": 1,
                      "started_at": float(record["started_at"]) - 60}
    (folder / "runs" / f"{'0' * 32}.json").write_text(json.dumps(older), encoding="utf-8")
    kept = client.get(zones).json()
    assert (kept["state"], kept["baseline_run_id"], kept["improvement"]) == ("done", run.stem, expected)
    result_path = folder / "results" / f"{run.stem}.json"
    stored = json.loads(result_path.read_text(encoding="utf-8"))
    design_path = folder / "results" / f"{DESIGN_RUN}.json"
    design_text = design_path.read_text(encoding="utf-8")
    # 基準位置放一份完整合格、同版物理、但方案不是這一次推出來的那一份的結果（設計自己那一份）。
    for text, words in ((design_text, "跟這一次推出來的不一樣"), (_tampered(stored), "讀回時被拒收")):
        result_path.write_text(text, encoding="utf-8")
        view = client.get(zones).json()
        assert (view["state"], view["can_start"], view["improvement"]) == ("baseline_rejected", True, None), view
        assert words in view["state_text"], view["state_text"]
    result_path.write_text(json.dumps(stored), encoding="utf-8")
    design_path.write_text(_tampered(json.loads(design_text)), encoding="utf-8")
    view = client.get(zones).json()
    assert (view["state"], view["can_start"], view["improvement"]) == ("cannot_compare", False, None), view
    assert "設計結果讀回時被拒收" in view["state_text"]
    design_path.write_text(design_text, encoding="utf-8")
    for text, state, words in ((json.dumps(stored | {"physics_identity": "phys-v1:" + "0" * 64}), "baseline_stale", "舊版物理"),
                               ("{}", "baseline_rejected", "讀不出來")):
        result_path.write_text(text, encoding="utf-8")
        view = client.get(f"/api/results/{DESIGN_RUN}/zones").json()
        assert (view["state"], view["can_start"], view["improvement"]) == (state, True, None) and words in view["state_text"]
    run.write_text(json.dumps(record | {"status": "failed", "exit_code": 1}), encoding="utf-8")
    failed = client.get(f"/api/results/{DESIGN_RUN}/zones").json()
    assert (failed["state"], failed["can_start"]) == ("failed", True) and "沒算完" in failed["state_text"]


def test_zone_routes_refuse_whole_walls_old_physics_cut_originals_and_missing_originals(
        tmp_path: Path, design_text: str, tmp_path_factory: pytest.TempPathFactory, worker_id: str) -> None:
    design = SchemeResult.model_validate_json(design_text)
    old = json.loads(design_text) | {"physics_identity": "phys-v1:" + "0" * 64}
    _put(tmp_path, OLD_RUN, json.dumps(old))
    _put(tmp_path, WHOLE_RUN, shared_control_result(tmp_path_factory, worker_id, "wall-1").model_dump_json())
    (tmp_path / "schemes").mkdir()
    (tmp_path / "schemes" / f"{design.scheme.scheme_id}.json").write_text(design.scheme.model_dump_json())
    other = ConstantImpedance(impedance_pa_s_m=2000.0)
    cut = with_x0((RectangleZone(area="rectangle", id="l", u_m=(0.0, 1.0), v_m=(0.0, 1.5), material=other),
                   RectangleZone(area="rectangle", id="r", u_m=(1.0, 2.0), v_m=(0.0, 1.5), material=original_x0())))
    (tmp_path / "schemes" / "orig.json").write_text(
        Scheme.model_validate(cut.model_dump(mode="json") | {"scheme_id": "orig"}).model_dump_json())
    zones, start = (f"/api/results/{OLD_RUN}/zones", f"/api/results/{OLD_RUN}/zones/baseline")
    with real_server(tmp_path) as base, httpx.Client(base_url=base, trust_env=False, timeout=120) as client:
        whole = client.get(f"/api/results/{WHOLE_RUN}/zones").json()
        assert whole["zoned"] is False and whole["rows"] == [] and whole["state"] is None
        assert client.post(f"/api/results/{WHOLE_RUN}/zones/baseline", json={}).status_code == 409
        assert client.get(f"/api/results/{'a' * 32}/zones").status_code == 404
        assert client.get("/api/results/not-a-run/zones").status_code == 400
        stale = client.get(zones).json()
        assert stale["state"] == "design_stale" and not stale["can_start"] and "舊版物理" in stale["state_text"]
        assert client.post(start, json={}).status_code == 409
        six = Scheme.model_validate(small_scheme().model_dump(mode="json") | {"scheme_id": "orig6"})
        (tmp_path / "schemes" / "orig6.json").write_text(six.model_dump_json())
        pointed = DesignSettings(scheme_id=design.scheme.scheme_id, original_scheme_id="orig6").model_dump(mode="json")
        assert client.put(f"/api/designs/{design.scheme.scheme_id}", json=pointed).status_code == 200
        laid = client.get(zones).json()  # 六個數字的原方案照它的牆鋪（不用表單外重建材料），逐區表照出
        assert laid["state"] == "design_stale" and any("照原方案「orig6」" in row["baseline_source"] for row in laid["rows"])
        settings = DesignSettings(scheme_id=design.scheme.scheme_id, original_scheme_id="orig")
        saved = client.put(f"/api/designs/{design.scheme.scheme_id}", json=settings.model_dump(mode="json"))
        assert saved.status_code == 200, saved.text
        refused = client.get(zones).json()
        assert refused["state"] == "refused" and "劃過設計的「b」" in refused["state_text"] and not refused["can_start"]
        assert {(row["baseline"], row["baseline_source"]) for row in refused["rows"]} == {("—", "—")}
        assert client.post(start, json={}).status_code == 409
        (tmp_path / "schemes" / "orig.json").unlink()
        gone = client.get(zones).json()
        assert gone["state"] == "refused" and "讀不到或不在了" in gone["state_text"]
        saved_path = tmp_path / "designs" / f"{design.scheme.scheme_id}.json"
        saved_path.write_text(json.dumps(settings.model_dump(mode="json") | {"zones": {"x0": {"a": {"mode": "bad"}}}}))
        broken = client.get(zones).json()  # 審查第 1 條：讀壞的設定拿不準有沒有指原方案，不准改用拆加貼層
        assert (broken["state"], broken["can_start"]) == ("refused", False) and "設計設定讀不出來" in broken["state_text"]
        assert client.post(start, json={}).status_code == 409
        for target, code, words in (("orig", 404, "沒有方案"), (design.scheme.scheme_id, 400, "不能是這一份自己")):
            body = DesignSettings(scheme_id=design.scheme.scheme_id, original_scheme_id=target).model_dump(mode="json")
            response = client.put(f"/api/designs/{design.scheme.scheme_id}", json=body)
            assert response.status_code == code and words in response.text, response.text
    assert _runs(tmp_path) == []
