"""#1092 分區結果頁走正式網頁行程（貼棉那一區要雙精度讀回，pytest 本身不開雙精度，見 tests/engine/conftest.py）。

一題真瀏覽器走完整條路：打開設計的結果頁先看到逐區表、原牆基準還沒算、而且沒有任何計算被開；按「算原牆基準」，
正式網頁行程自己開正式計算（真算，不替身）；算完這一節自己換成改善多少，數字等於拿兩份結果檔獨立算的那一份。
另一題只打入口：整面一區的結果沒有這一節、舊版物理的設計不給開、原方案切線劃過設計的一區照實拒收、原方案不在了也照講。
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
from tests.engine._zoned_report_cases import with_x0
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
    with real_server(tmp_path) as base, _open(browser, f"{base}/results/{DESIGN_RUN}") as watched:
        page = watched.page
        expect(page.locator("#zone-result")).to_be_visible(timeout=60_000)
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
            _later_states(tmp_path, client, run)


def _later_states(folder: Path, client: httpx.Client, run: Path) -> None:
    """算好之後基準又出事的三格（改檔模擬，不再真算）：舊版物理要重算、讀不出來要重算、上一次沒算完要重算。"""
    result_path = folder / "results" / f"{run.stem}.json"
    stored = json.loads(result_path.read_text(encoding="utf-8"))
    for text, state, words in ((json.dumps(stored | {"physics_identity": "phys-v1:" + "0" * 64}), "baseline_stale", "舊版物理"),
                               ("{}", "baseline_rejected", "讀不出來")):
        result_path.write_text(text, encoding="utf-8")
        view = client.get(f"/api/results/{DESIGN_RUN}/zones").json()
        assert (view["state"], view["can_start"], view["improvement"]) == (state, True, None) and words in view["state_text"]
    record = json.loads(run.read_text(encoding="utf-8")) | {"status": "failed", "exit_code": 1}
    run.write_text(json.dumps(record), encoding="utf-8")
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
        for target, code, words in (("orig", 404, "沒有方案"), (design.scheme.scheme_id, 400, "不能是這一份自己")):
            body = DesignSettings(scheme_id=design.scheme.scheme_id, original_scheme_id=target).model_dump(mode="json")
            response = client.put(f"/api/designs/{design.scheme.scheme_id}", json=body)
            assert response.status_code == code and words in response.text, response.text
    assert _runs(tmp_path) == []
