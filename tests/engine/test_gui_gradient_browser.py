"""真瀏覽器考第三階段搜尋頁（#1106，#1082 段 1-6b）：選方案、開跑選項、預覽、沒替身先建、開跑、進度、停止、接續、
失敗原文、終點、存成方案接結果頁；替身被准用閘門拒收時叫人重建。

開服務那一層換成假的（搜尋與建替身都不真跑，考卷自己寫離開碼、帳本、狀態檔假裝它們跑到哪），一般計算的執行程式是
「什麼都不做」的 Python。輪詢用瀏覽器假時鐘快轉，不真等 5 秒。截圖存 ``AOSR_SHOTS``（有給才存）。
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import cast

import uvicorn
from playwright.sync_api import Browser, Page, Route, expect

from aosr.config.frequency_axis import LowFrequencyAxis
from aosr.gradient import folder_files
from aosr.gradient.cli import EXIT_STOPPED
from aosr.gradient.surrogate_files import PROGRESS_FILE, RoundRecord, SurrogateProgress
from aosr.gui.surrogate_builds import BUILD_UNIT_PREFIX
from aosr.gui.surrogate_slots import SLOTS_DIR
from aosr.gui.app import GuiSettings, create_app
from aosr.gui.gradient_jobs import GradientJobs
from aosr.gui.gradient_plan import GradientRequest
from aosr.gui.zone_design import write_design
from aosr.reporting.scheme import Scheme
from tests.engine.test_gui_browser import RAW_VALUES, _open, browser as browser
from tests.engine.test_gui_gradient_jobs import (
    NAME, SEARCH, FakeLauncher, Record, _design, _finish, _jobs, _place_surrogate, _scheme,
)

POLL_MS = 5000
NOTE = "注意：材料 my_wool 的空腔不調（固定 0.05 m）"
FAILURE = "梯度搜尋接續失敗：MaterialsRefused：使用者材料庫快照被改過（SHA-256 不同），拒收"
REFUSED = "梯度搜尋開始失敗：ReducedModelNotValidated：替身讀不進來或對不上，拒用：物理身分不同"


def _axis_scheme() -> Scheme:
    """建替身要低頻軸（沒有的方案網頁照實拒建）；格子代號吃低頻軸，替身也照這一份放。"""
    base = _scheme()
    return base.model_copy(update={"scene": base.scene.model_copy(update={"low_frequency_axis": LowFrequencyAxis.SEARCH})})


def _data(tmp_path: Path, *, surrogate: bool = True) -> Path:
    data_dir = tmp_path / "data"
    (data_dir / "schemes").mkdir(parents=True)
    (data_dir / "designs").mkdir()
    scheme = _axis_scheme()
    (data_dir / "schemes" / f"{NAME}.json").write_text(scheme.model_dump_json(), encoding="utf-8")
    write_design(data_dir, _design(SEARCH), scheme)
    if surrogate:
        _place_surrogate(data_dir, scheme)
    return data_dir


@contextmanager
def _serve(data_dir: Path, launcher: FakeLauncher) -> Iterator[str]:
    app = create_app(GuiSettings(engine_commit="a" * 40, data_dir=data_dir, gradient_launcher=launcher,
                                 runner=(sys.executable, "-c", "pass")))
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 30
    while not server.started:
        if not thread.is_alive() or time.monotonic() > deadline:
            raise RuntimeError("本機網頁沒有起來")
        time.sleep(0.05)
    port = int(server.servers[0].sockets[0].getsockname()[1])
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=30)


def _shot(page: Page, name: str) -> None:
    folder = os.environ.get("AOSR_SHOTS")
    if folder:
        page.screenshot(path=str(Path(folder) / f"{name}.png"), full_page=True)


def _tick(page: Page) -> None:
    page.clock.run_for(POLL_MS)


def _view(jobs: GradientJobs, job_id: str) -> Record:
    return jobs.get(job_id)


def _clean(page: Page, errors: list[str]) -> None:
    assert errors == []
    text = page.locator("main").inner_text()
    assert not RAW_VALUES.search(text), RAW_VALUES.search(text)


def test_no_surrogate_builds_one_with_the_chosen_budget_and_can_stop(browser: Browser, tmp_path: Path) -> None:
    """抓沒替身還讓人按開始、建替身的預算沒送出去、建模進度與停止沒接上。"""
    data_dir = _data(tmp_path, surrogate=False)
    launcher = FakeLauncher()
    with _serve(data_dir, launcher) as url, _open(browser, f"{url}/gradient?scheme={NAME}", fake_clock=True) as watched:
        page = watched.page
        expect(page.locator("#form-block")).to_be_visible()
        expect(page.locator("#depth-table tbody tr")).to_have_count(2)
        expect(page.locator("#surrogate-state")).to_contain_text("還沒有替身，要先建")
        expect(page.locator("#start")).to_be_disabled()
        expect(page.locator("#max-full-solves")).to_have_value("600")
        expect(page.locator("#surrogate-history")).to_contain_text("還沒建過替身")
        page.locator("#max-full-solves").fill("1000")
        page.locator("#build-surrogate").click()
        expect(page.locator("#build-state")).to_contain_text("建替身：匯出完整模型")
        unit, argv, _, _ = launcher.launched[-1]
        assert "surrogate" in unit and argv[argv.index("--max-full-solves") + 1] == "1000"
        assert "--stop-residual" not in argv
        build_id = unit.removeprefix(BUILD_UNIT_PREFIX)
        work = data_dir / SLOTS_DIR / "builds" / build_id / "work"
        work.mkdir(parents=True)
        (work / PROGRESS_FILE).write_text(_progress(time.time()).model_dump_json(), encoding="utf-8")
        _tick(page)
        expect(page.locator("#build-state")).to_contain_text("完整解快照、補點、投影")
        expect(page.locator("#build-estimate")).to_contain_text("推估")
        expect(page.locator("#build-estimate")).to_contain_text("還要約")
        expect(page.locator("#build-surrogate")).to_be_disabled()
        page.locator("#load-plan").click()
        expect(page.locator("#busy-line")).to_contain_text("方案「zoned」正在建替身")
        expect(page.locator("#busy-open")).to_have_text("看建替身進度")
        _shot(page, "1-6b-1-surrogate-building")
        page.locator("#stop-build").click()
        expect(page.locator("#build-state")).to_contain_text("停止中")
        assert launcher.stopped == [unit]
        _clean(page, watched.page_errors + watched.console_errors)


def _progress(now: float) -> SurrogateProgress:
    """建模第二輪做完的進度（同建替身考卷那一份的數字）：推估量得到速度。"""
    return SurrogateProgress.model_validate({
        "phase": "building", "started_at": now - 120.0, "phase_started_at": now - 110.0, "updated_at": now - 5.0,
        "workers": 4, "patches": 7, "dofs": 3389, "axis_points": 94, "training_sets": 41, "holdout_sets": 9,
        "max_full_solves": 1000, "stop_residual": 0.001, "full_solves": 96, "rank": 267, "snapshot_seconds": 48.0,
        "rounds": [RoundRecord(round=0, full_solves=64, rank=200, max_residual=0.5, elapsed_s=40.0),
                   RoundRecord(round=1, full_solves=96, rank=267, max_residual=0.02, elapsed_s=70.0)]})


def _preview_and_start(page: Page, launcher: FakeLauncher) -> str:
    expect(page.locator("#surrogate-state")).to_contain_text("已經有替身")
    expect(page.locator("#max-steps")).to_have_value("60")
    page.locator("#max-steps").fill("45")
    page.locator('#depth-table input[data-wall="x0"][data-zone="r0_c0"]').fill("0.1")
    page.locator("#preview").click()
    expect(page.locator("#plan-summary")).to_contain_text("往下走步數上限 45 步")
    expect(page.locator('#plan-table tr[data-surface="x0@r0_c0"]')).to_contain_text("0.1 公尺")
    expect(page.locator('#plan-table tr[data-surface="ceiling@r0_c1"]')).to_contain_text("保留原牆")
    expect(page.locator('#plan-table tr[data-surface="x0@r1_c1"]')).to_contain_text("沒有限制（家族都試、也試不貼）")
    _shot(page, "1-6b-2-preview")
    page.locator("#start").click()
    expect(page.locator("#job-state")).to_contain_text("開跑前核對中")
    unit, argv, _, _ = launcher.launched[-1]
    assert argv[argv.index("--max-steps") + 1] == "45"
    assert "x0@r0_c0=0.1" in argv
    page.locator("#preview").click()
    expect(page.locator("#busy-line")).to_contain_text("方案「zoned」的第三階段搜尋")
    expect(page.locator("#start")).to_be_disabled()
    return unit.split("-")[2]


def _searching(page: Page, data_dir: Path, job_id: str) -> Path:
    """假裝搜尋開好資料夾、往下走了 10 步、印了一行注意。"""
    folder = data_dir / "gradient" / job_id / "root" / ("f" * 32)
    folder.mkdir(parents=True)
    (folder / folder_files.PROJECT_FILE).write_text("{}", encoding="utf-8")
    header = json.dumps({"ledger_version": "v6"})
    rows = [json.dumps({"step": index, "phase": "descent"}) for index in range(11)]
    (folder / folder_files.LEDGER_FILE).write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    (folder / folder_files.STATUS_FILE).write_text(json.dumps({
        "state": "running", "started_at": "2026-11-11T00:00:00+00:00", "updated_at": "2026-11-11T00:01:00+00:00",
        "evaluations": 12, "rounds_done": 0, "last_value": 3.25, "stop_reason": None, "pid": 4242}), encoding="utf-8")
    (data_dir / "gradient" / job_id / "steps" / "0.log").write_text(NOTE + "\n", encoding="utf-8")
    _tick(page)
    expect(page.locator("#job-state")).to_have_text("搜尋中")
    expect(page.locator("#job-progress")).to_contain_text("往下走：10／45 步")
    expect(page.locator("#job-notes")).to_contain_text(NOTE)
    _shot(page, "1-6b-3-searching")
    return folder


def _stop_resume_fail(page: Page, data_dir: Path, launcher: FakeLauncher, jobs: GradientJobs, job_id: str) -> None:
    page.locator("#job-stop").click()
    expect(page.locator("#job-state")).to_contain_text("停止中")
    _finish(data_dir, launcher, _view(jobs, job_id), EXIT_STOPPED)
    _tick(page)
    expect(page.locator("#job-resume")).to_be_visible()
    page.locator("#job-resume").click()
    expect(page.locator("#job-state")).to_have_text("搜尋中")
    lines = [FAILURE, *(f"收尾第 {index} 行" for index in range(20))]
    (data_dir / "gradient" / job_id / "steps" / "1.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _finish(data_dir, launcher, _view(jobs, job_id), 1)
    _tick(page)
    expect(page.locator("#job-state")).to_have_text("搜尋失敗")
    expect(page.locator("#job-failures")).to_contain_text(FAILURE)
    _shot(page, "1-6b-4-failed-with-reason")


def _finish_endpoint_and_adopt(page: Page, data_dir: Path, launcher: FakeLauncher, jobs: GradientJobs, job_id: str,
                               folder: Path) -> None:
    page.locator("#job-resume").click()
    expect(page.locator("#job-state")).to_have_text("搜尋中")
    _finish(data_dir, launcher, _view(jobs, job_id), 0)
    _tick(page)
    page.locator("#job-endpoint").click()
    expect(page.locator("#job-state")).to_contain_text("終點計算中")
    _finish(data_dir, launcher, _view(jobs, job_id), 0)
    exported = cast(Record, json.loads(_axis_scheme().model_dump_json())) | {"scheme_id": f"{NAME}-gradient-0123abcd"}
    (folder / folder_files.ENDPOINT_FILE).write_text("{}", encoding="utf-8")
    (folder / folder_files.EXPORTED_SCHEME_FILE).write_text(json.dumps(exported), encoding="utf-8")
    _tick(page)
    expect(page.locator("#job-state")).to_contain_text("做完了")
    expect(page.locator("#job-endpoint-note")).to_contain_text("改善多少以結果頁算的為準")
    page.locator("#job-adopt").click()
    link = page.locator("#result-link")
    expect(link).to_be_visible()
    record = json.loads((data_dir / "gradient" / job_id / "adopted.json").read_text(encoding="utf-8"))
    assert link.get_attribute("href") == f"/results/{record['run_id']}"
    expect(page.locator("#job-adopted")).to_contain_text(f"{NAME}-search-{job_id[:8]}")
    expect(page.locator("#job-adopt")).to_be_hidden()
    _shot(page, "1-6b-5-done-linked-to-result")


def test_search_start_progress_stop_resume_failure_endpoint_and_result_page(browser: Browser, tmp_path: Path) -> None:
    """整條：改步數與深度上限、預覽照改的值、開跑帶對參數、進度印「往下走 n／上限」與注意行、停止、接續、
    失敗原文（不只最後幾行）、再接續做完、跑終點、存成方案、連到結果頁。"""
    data_dir = _data(tmp_path)
    launcher = FakeLauncher()
    jobs = _jobs(data_dir, launcher)
    with _serve(data_dir, launcher) as url, _open(browser, f"{url}/gradient?scheme={NAME}", fake_clock=True) as watched:
        page = watched.page
        job_id = _preview_and_start(page, launcher)
        folder = _searching(page, data_dir, job_id)
        _stop_resume_fail(page, data_dir, launcher, jobs, job_id)
        _finish_endpoint_and_adopt(page, data_dir, launcher, jobs, job_id, folder)
        _clean(page, watched.page_errors + watched.console_errors)


def test_a_refused_surrogate_offers_a_rebuild_that_replaces_it(browser: Browser, tmp_path: Path) -> None:
    """抓替身被准用閘門拒收時畫面只說「失敗」、沒有重建的路，或重建沒帶「換掉舊的」。"""
    data_dir = _data(tmp_path)
    launcher = FakeLauncher()
    jobs = _jobs(data_dir, launcher)
    view = jobs.start(NAME, GradientRequest())
    job_id = str(view["job_id"])
    (data_dir / "gradient" / job_id / "steps" / "0.log").write_text(REFUSED + "\n", encoding="utf-8")
    _finish(data_dir, launcher, view, 1)
    with _serve(data_dir, launcher) as url, \
            _open(browser, f"{url}/gradient?scheme={NAME}&job={job_id}", fake_clock=True) as watched:
        page = watched.page
        expect(page.locator("#job-state")).to_contain_text("替身核不過")
        expect(page.locator("#job-failures")).to_contain_text("ReducedModelNotValidated")
        page.locator("#job-rebuild").click()
        expect(page.locator("#surrogate-message")).to_contain_text("被准用閘門拒收")
        _shot(page, "1-6b-6-surrogate-refused-rebuild")
        page.locator("#build-surrogate").click()
        expect(page.locator("#build-state")).to_contain_text("建替身")
        _, argv, _, _ = launcher.launched[-1]
        assert "--replace" in argv
        _clean(page, watched.page_errors + watched.console_errors)


def test_server_errors_and_a_long_wait_are_said_plainly(browser: Browser, tmp_path: Path) -> None:
    """抓伺服器回 500（不是 JSON）時頁面丟例外或什麼都不說、等開跑鎖等很久時畫面像當掉（取鎖沒有逾時）。"""
    data_dir = _data(tmp_path)
    launcher = FakeLauncher()
    held: list[Route] = []
    with _serve(data_dir, launcher) as url, _open(browser, f"{url}/gradient?scheme={NAME}", fake_clock=True) as watched:
        page = watched.page
        expect(page.locator("#start")).to_be_enabled()
        start = "**/api/gradient/schemes/*/start"
        page.route(start, lambda route: route.fulfill(status=500, body="Internal Server Error", content_type="text/plain"))
        page.locator("#start").click()
        expect(page.locator("#start-message")).to_contain_text("伺服器出錯（狀態 500）")
        page.unroute(start)
        page.route(start, lambda route: held.append(route))
        page.locator("#start").click()
        page.wait_for_function("() => document.getElementById('start-message').textContent === '正在開跑…'")
        page.clock.run_for(15_000)
        expect(page.locator("#start-message")).to_contain_text("還在等伺服器回應（可能在等開跑鎖")
        _shot(page, "1-6b-7-waiting-for-lock")
        assert len(held) == 1
        held[0].fulfill(status=409, json={"error": "已經有一跑在算（abc）", "busy": "aosr-gradient-elsewhere-0"})
        expect(page.locator("#start-message")).to_contain_text("別的資料夾或命令列開的第三階段計算（服務 aosr-gradient-elsewhere-0）")
        assert launcher.launched == []
        _clean(page, watched.page_errors + [error for error in watched.console_errors
                                            if "status of 500" not in error and "status of 409" not in error])
