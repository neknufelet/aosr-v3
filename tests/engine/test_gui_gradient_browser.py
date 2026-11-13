"""真瀏覽器考第三階段搜尋頁（#1106，#1082 段 1-6b）：選方案、開跑選項、預覽、沒替身先建、開跑、進度、停止、接續、
失敗原文、終點、存成方案接結果頁；替身被准用閘門拒收時叫人重建；材料庫的材料替身沒訓練過時說要重建（#1082 段 2-3c）。

開服務那一層換成假的（搜尋與建替身都不真跑，考卷自己寫離開碼、帳本、狀態檔假裝它們跑到哪），一般計算的執行程式是
「什麼都不做」的 Python。輪詢用瀏覽器假時鐘快轉，不真等 5 秒。截圖存 ``AOSR_SHOTS``（有給才存）。
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import cast
from zoneinfo import ZoneInfo

import pytest
import uvicorn
from playwright.sync_api import Browser, Page, Route, expect

from aosr.config.frequency_axis import LowFrequencyAxis
from aosr.gradient import folder_files
from aosr.gradient.cli import EXIT_STOPPED
from aosr.gradient.surrogate_files import FIXED_ROUNDS_FILE, PROGRESS_FILE, FixedCheckRound, RoundRecord, SurrogateProgress
from aosr.gui.surrogate_builds import BUILD_UNIT_PREFIX
from aosr.gui.surrogate_slots import SLOTS_DIR
from aosr.gui.app import GuiSettings, create_app
from aosr.gui.gradient_jobs import GradientJobs, LaunchFailed
from aosr.gui.gradient_plan import GradientRequest
from aosr.gui.material_editor import user_library_path
from aosr.gui.zone_design import write_design
from aosr.materials import library
from aosr.reporting.scheme import Scheme
from tests.engine.test_gui_browser import RAW_VALUES, _open, browser as browser
from tests.engine.test_gui_gradient_jobs import (
    NAME, SEARCH, FakeLauncher, Record, _design, _finish, _jobs, _place_surrogate, _scheme,
)
from tests.engine.test_material_gradient_library import _optimizable

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
        expect(page.locator("#max-full-solves")).to_have_value("1000")
        expect(page.locator("#surrogate-history")).to_contain_text("還沒建過替身")
        page.locator("#max-full-solves").fill("1500")
        page.locator("#build-surrogate").click()
        expect(page.locator("#build-state")).to_contain_text("建替身：匯出完整模型")
        unit, argv, _, _ = launcher.launched[-1]
        assert "surrogate" in unit and argv[argv.index("--max-full-solves") + 1] == "1500"
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
        # 建模中重開頁面（#1119）：印這一份實際的完整解上限（另填的 1500，不是預設 1000），兩格換成這一份的值並鎖住。
        page.reload()
        expect(page.locator("#build-state")).to_contain_text("完整解上限 1500 次（這一份另填）、停止殘差 0.001（預設）")
        expect(page.locator("#max-full-solves")).to_have_value("1500")
        expect(page.locator("#max-full-solves")).to_be_disabled()
        expect(page.locator("#stop-residual")).to_be_disabled()
        _shot(page, "1119-surrogate-reopened-budget")
        page.locator("#load-plan").click()
        expect(page.locator("#busy-line")).to_contain_text("方案「zoned」正在建替身")
        expect(page.locator("#busy-open")).to_have_text("看建替身進度")
        _shot(page, "1-6b-1-surrogate-building")
        page.locator("#stop-build").click()
        expect(page.locator("#build-state")).to_contain_text("停止中")
        assert launcher.stopped == [unit]
        # 建模真的停了（服務不在了）：兩格放回預設、解鎖，下一次開建不會悄悄帶這一份另填的 1500（#1119）。
        launcher.active.discard(unit)
        _tick(page)
        expect(page.locator("#max-full-solves")).to_be_enabled()
        expect(page.locator("#max-full-solves")).to_have_value("1000")
        expect(page.locator("#stop-residual")).to_be_enabled()
        _clean(page, watched.page_errors + watched.console_errors)


def _progress(now: float) -> SurrogateProgress:
    """建模第二輪做完的進度（數字同建替身考卷那一份，只有完整解上限換成這一份另填的 1500）：推估量得到速度。"""
    return SurrogateProgress.model_validate({
        "phase": "building", "started_at": now - 120.0, "phase_started_at": now - 110.0, "updated_at": now - 5.0,
        "workers": 4, "patches": 7, "dofs": 3389, "axis_points": 94, "training_sets": 41, "holdout_sets": 9,
        "max_full_solves": 1500, "stop_residual": 0.001, "full_solves": 96, "rank": 267, "snapshot_seconds": 48.0,
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
    # 每一組跟自己的上限取較嚴（#1082 段 2-4）：這一跑另填的 0.1 比玻璃棉自己的 0.3 嚴。
    expect(page.locator('#plan-table tr[data-surface="x0@r0_c0"] td.depth-cell p.note').first).to_have_text(
        "48K 玻璃棉（可調厚度與空腔）：生效 0.1 公尺（這一跑另填的上限擋住；材料組自己 0.3 公尺比較寬）")
    expect(page.locator('#plan-table tr[data-surface="ceiling@r0_c1"]')).to_contain_text("保留原牆")
    # 代號改印材料庫的名字（#1119）：起跑樣板、家族選單；代號只留在命令列參數。
    expect(page.locator('#plan-table tr[data-surface="x0@r0_c0"] td').nth(3)).to_have_text("48K 玻璃棉 50 mm＋50 mm 空腔")
    expect(page.locator("#plan-summary")).to_contain_text("家族 48K 玻璃棉（可調厚度與空腔）")
    expect(page.locator('#family option[value="glass_wool_48k"]')).to_have_text("48K 玻璃棉（可調厚度與空腔）")
    expect(page.locator("#plan-arguments")).to_contain_text("glass_wool_48k_50mm_gap50")
    expect(page.locator('#plan-table tr[data-surface="x0@r1_c1"]')).to_contain_text("沒有限制（家族都試、也試不貼）")
    _shot(page, "1-6b-2-preview")
    # 預覽之後改了選項：畫面上的預覽不再是會跑的那一份，開始要停用；改回來又能按（抓「開跑重讀表單、跑的跟預覽不同」）。
    page.locator("#max-steps").fill("50")
    expect(page.locator("#start")).to_be_disabled()
    expect(page.locator("#start-message")).to_contain_text("選項改過了")
    page.locator("#max-steps").fill("45")
    expect(page.locator("#start")).to_be_enabled()
    page.locator("#start").click()
    expect(page.locator("#job-state")).to_contain_text("開跑前核對中")
    # 工作頁留著開跑時的深度上限判決（個案哪一層、每一組生效多少、哪一條擋住）。
    expect(page.locator('#job-depths [data-surface="x0@r0_c0"]')).to_contain_text("個案上限 0.1 公尺")
    expect(page.locator('#job-depths [data-surface="x0@r0_c0"]')).to_contain_text("生效 0.1 公尺（這一跑另填的上限擋住")
    unit, argv, _, _ = launcher.launched[-1]
    assert argv[argv.index("--max-steps") + 1] == "45"
    assert "x0@r0_c0=0.1" in argv
    # 開跑之後不用再按預覽（#1119）：輪詢的清單就帶「整台在算」，開始鈕立刻停用、忙碌行出來。
    expect(page.locator("#start")).to_be_disabled()
    expect(page.locator("#busy-line")).to_contain_text("方案「zoned」的第三階段搜尋")
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
    (folder / folder_files.LEDGER_FILE).write_text(header + "\n", encoding="utf-8")
    # 搜尋本體寫世界時（+00:00）；時間取這一刻，截圖上的「狀態檔最後更新」才跟「開始」對得上。
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (folder / folder_files.STATUS_FILE).write_text(json.dumps({
        "state": "running", "started_at": now, "updated_at": now,
        "evaluations": 12, "rounds_done": 0, "last_value": 3.25, "stop_reason": None, "pid": 4242}), encoding="utf-8")
    (data_dir / "gradient" / job_id / "steps" / "0.log").write_text(NOTE + "\n", encoding="utf-8")
    _tick(page)
    expect(page.locator("#job-state")).to_have_text("搜尋中")
    # 往下走那一段整段算完才一次寫帳：帳本只有標頭時不印「0／45」裝成沒在動。
    expect(page.locator("#job-progress")).to_contain_text("整段算完才一次記帳，正在算")
    rows = [json.dumps({"step": index, "phase": "descent"}) for index in range(11)]
    (folder / folder_files.LEDGER_FILE).write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    _tick(page)
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
    expect(page.locator("#start")).to_be_enabled()
    expect(page.locator("#busy-line")).to_be_hidden()
    # 查不到整台機器忙不忙（systemctl 叫不動）：不准開、照講原因；查得到了又能開（#1119）。
    def blind() -> set[str]:
        raise LaunchFailed("systemctl 離開碼 1")
    setattr(launcher, "active_units", blind)
    _tick(page)
    expect(page.locator("#start")).to_be_disabled()
    expect(page.locator("#busy-line")).to_contain_text("查不到整台機器現在忙不忙，先不開新的一跑：systemctl 離開碼 1")
    delattr(launcher, "active_units")
    _tick(page)
    expect(page.locator("#start")).to_be_enabled()
    expect(page.locator("#busy-line")).to_be_hidden()
    page.locator("#job-endpoint").click()
    expect(page.locator("#job-state")).to_contain_text("終點計算中")
    # 終點在算時（#1119，109 示範看到開始鈕還亮著）：開始鈕停用、忙碌行點名是這一跑的終點。
    _tick(page)
    expect(page.locator("#start")).to_be_disabled()
    expect(page.locator("#busy-line")).to_contain_text("方案「zoned」的第三階段搜尋（終點計算中")
    _shot(page, "1119-endpoint-running-start-disabled")
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


def _shown(epoch: float, zone: str) -> str:
    """瀏覽器 ``toLocaleString("zh-TW", {hour12: false})`` 的樣子：年月日不補零、時分秒補零。"""
    moment = datetime.fromtimestamp(epoch, ZoneInfo(zone))
    return f"{moment.year}/{moment.month}/{moment.day} {moment:%H:%M:%S}"


@pytest.mark.parametrize(("zone", "updated"), [("Asia/Taipei", "2026/11/11 20:01:00"),
                                               ("America/New_York", "2026/11/11 07:01:00")])
def test_every_time_on_the_page_is_shown_in_the_browser_time_zone(browser: Browser, tmp_path: Path, zone: str,
                                                                  updated: str) -> None:
    """抓世界時沒換成本機時間（截圖上「狀態檔最後更新」比「開始」少 8 小時那一種）：搜尋本體寫的帶時區字串
    ``2026-11-11T12:01:00+00:00`` 要照瀏覽器時區印；兩個時區各考一次，寫死加 8 小時也過不了。秒數（epoch）的
    「開始」「第 1 步」同樣照瀏覽器時區；沒寫時區的字串照原文印、標明沒寫時區，不替它猜。"""
    data_dir = _data(tmp_path)
    launcher = FakeLauncher()
    view = _jobs(data_dir, launcher).start(NAME, GradientRequest())
    job_id = str(view["job_id"])
    folder = data_dir / "gradient" / job_id / "root" / ("f" * 32)
    folder.mkdir(parents=True)
    (folder / folder_files.PROJECT_FILE).write_text("{}", encoding="utf-8")
    status = {"state": "running", "started_at": "2026-11-11T12:00:00+00:00", "updated_at": "2026-11-11T12:01:00+00:00",
              "evaluations": 12, "rounds_done": 0, "last_value": 3.25, "stop_reason": None, "pid": 4242}
    (folder / folder_files.STATUS_FILE).write_text(json.dumps(status), encoding="utf-8")
    steps = cast(list[Record], view["steps"])
    with _serve(data_dir, launcher) as url, _open(browser, f"{url}/gradient?scheme={NAME}&job={job_id}",
                                                  fake_clock=True, timezone_id=zone) as watched:
        page = watched.page
        expect(page.locator("#job-progress")).to_contain_text(f"狀態檔最後更新：{updated}")
        expect(page.locator("#job-title")).to_contain_text(f"開始 {_shown(float(str(view['created_at'])), zone)}")
        expect(page.locator("#job-steps")).to_contain_text(f"（{_shown(float(str(steps[0]['launched_at'])), zone)}）")
        expect(page.locator(f'#job-list tr[data-job="{job_id}"]')).to_contain_text(
            _shown(float(str(view["created_at"])), zone))
        (folder / folder_files.STATUS_FILE).write_text(json.dumps({**status, "updated_at": "2026-11-11T12:02:00"}),
                                                       encoding="utf-8")
        _tick(page)
        expect(page.locator("#job-progress")).to_contain_text("狀態檔最後更新：2026-11-11T12:02:00（沒寫時區）")
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


# ---- 替身沒用這些材料訓練過（#1082 段 2-3c）----

NOT_TRAINED = "梯度搜尋開始失敗：SurrogateNotTrainedOn：替身沒用這些材料訓練過，要重建：家族 rock_wool_60k 沒訓練過"


def test_library_families_the_surrogate_never_saw_say_rebuild_and_rebuild_with_them(browser: Browser,
                                                                                    tmp_path: Path) -> None:
    """抓材料庫加了可最佳化材料、格子裡的替身沒訓練過它時，畫面照樣說「已經有替身」讓人開跑，
    或重建沒帶「換掉舊的」與材料庫（重建了還是沒訓練到它）。"""
    data_dir = _data(tmp_path)
    user_library_path(data_dir).write_text(library.dump_user_library((_optimizable(),)), encoding="utf-8")
    launcher = FakeLauncher()
    with _serve(data_dir, launcher) as url, _open(browser, f"{url}/gradient?scheme={NAME}", fake_clock=True) as watched:
        page = watched.page
        expect(page.locator("#surrogate-state")).to_contain_text("替身沒用這些材料訓練過，要重建")
        expect(page.locator("#surrogate-state")).to_contain_text("rock_wool_60k 沒訓練過")
        expect(page.locator("#surrogate-state")).to_have_class("notice")
        expect(page.locator("#start")).to_be_disabled()
        expect(page.locator("#surrogate-build-form")).to_be_visible()
        _shot(page, "2-3c-1-not-trained-rebuild")
        page.locator("#build-surrogate").click()
        expect(page.locator("#build-state")).to_contain_text("建替身")
        _, argv, _, _ = launcher.launched[-1]
        assert "--replace" in argv and "--materials" in argv
        _clean(page, watched.page_errors + watched.console_errors)


def test_a_start_refused_for_untrained_materials_says_so_and_offers_a_rebuild(browser: Browser, tmp_path: Path) -> None:
    """抓背景開跑那一步才發現材料沒訓練過（網頁預覽之後材料庫才改）時，狀態只說「替身核不過」講不出原因、或沒有重建的路。"""
    data_dir = _data(tmp_path)
    launcher = FakeLauncher()
    jobs = _jobs(data_dir, launcher)
    view = jobs.start(NAME, GradientRequest())
    job_id = str(view["job_id"])
    (data_dir / "gradient" / job_id / "steps" / "0.log").write_text(NOT_TRAINED + "\n", encoding="utf-8")
    _finish(data_dir, launcher, view, 1)
    assert jobs.get(job_id)["state"] == "surrogate_refused"
    with _serve(data_dir, launcher) as url, \
            _open(browser, f"{url}/gradient?scheme={NAME}&job={job_id}", fake_clock=True) as watched:
        page = watched.page
        expect(page.locator("#job-state")).to_contain_text("替身沒用這些材料訓練過，要重建")
        expect(page.locator("#job-failures")).to_contain_text("rock_wool_60k 沒訓練過")
        expect(page.locator("#job-rebuild")).to_be_visible()
        _shot(page, "2-3c-2-start-refused-not-trained")
        _clean(page, watched.page_errors + watched.console_errors)


# ---- 109 示範找到的顯示毛病（#1082 段 2-9）----

def _installed(data_dir: Path, launcher: FakeLauncher, unit: str) -> None:
    """假裝這一頁開的建模做完、放進格子（命令列寫 installed 結果、離開碼 0、服務不在了）。"""
    from aosr.gui import gradient_exec
    from aosr.gradient.surrogate_files import RESULT_FILE, SurrogateResult

    build_id = unit.removeprefix(BUILD_UNIT_PREFIX)
    work = data_dir / SLOTS_DIR / "builds" / build_id / "work"
    work.mkdir(parents=True, exist_ok=True)
    (work / PROGRESS_FILE).write_text(_progress(time.time()).model_copy(update={"phase": "installing"}).model_dump_json(),
                                      encoding="utf-8")
    result = SurrogateResult(verdict="installed", message="好了", stop_reason="抽到的候選殘差都在停止值內", full_solves=96,
                             rank=267, build_seconds=70.0, validation_seconds=30.0, within_contract=True,
                             finished_at=time.time())
    (work / RESULT_FILE).write_text(result.model_dump_json(), encoding="utf-8")
    launcher.active.discard(unit)
    gradient_exec.write_exit(data_dir / SLOTS_DIR / "builds" / build_id / "steps" / "0.exit.json",
                             gradient_exec.StepExit(exit_code=0, error=None, finished_at=time.time()))


def _plan_posts(page: Page) -> list[str]:
    """這一頁送出的開跑預覽請求（``/schemes/<方案>/plan``，不含建替身那一支 ``/surrogate/plan``）。"""
    seen: list[str] = []
    page.on("request", lambda request: seen.append(request.url) if request.method == "POST" and
            request.url.endswith("/plan") and not request.url.endswith("/surrogate/plan") else None)
    return seen


def _settles(page: Page, seen: list[str], most: int) -> None:
    """真時間等 2 秒（預覽迴圈是請求一回來就再送、不靠計時器，假時鐘擋不住它），送出的預覽不准超過 ``most`` 次。"""
    page.wait_for_timeout(2000)
    assert len(seen) <= most, f"2 秒內送了 {len(seen)} 次預覽（迴圈）"


def test_an_installed_build_the_materials_outgrow_does_not_loop_and_keeps_the_typed_budget(browser: Browser,
                                                                                         tmp_path: Path) -> None:
    """抓 109 示範那一個迴圈：格子裡最新一份替身已裝好、但這一場的材料它沒訓練過時，showBuild 叫預覽、預覽又叫回
    showBuild（幾分鐘六千多次預覽），每一圈把「完整解預算」改回預設，使用者另填的數字留不住（當時預設 600、打的 1000；預設改 1000 後考卷另填 1500）。
    也抓修過頭：建好、這一次涵蓋得到時，畫面要自己換成「已經有替身」、開始鈕亮起來。"""
    data_dir = _data(tmp_path)
    library_file = user_library_path(data_dir)
    library_file.write_text(library.dump_user_library((_optimizable(),)), encoding="utf-8")
    launcher = FakeLauncher()
    with _serve(data_dir, launcher) as url, _open(browser, f"{url}/gradient?scheme={NAME}", fake_clock=True) as watched:
        page = watched.page
        seen = _plan_posts(page)
        expect(page.locator("#surrogate-state")).to_contain_text("替身沒用這些材料訓練過，要重建")
        page.locator("#build-surrogate").click()
        expect(page.locator("#build-state")).to_contain_text("建替身")
        unit = launcher.launched[-1][0]
        _installed(data_dir, launcher, unit)
        seen.clear()
        _tick(page)
        expect(page.locator("#build-state")).to_contain_text("已放進這份切法的格子")
        # 建好那一刻重新預覽一次（材料還是沒涵蓋：照講要重建），之後不再自己送。
        _settles(page, seen, 2)
        expect(page.locator("#surrogate-state")).to_contain_text("替身沒用這些材料訓練過，要重建")
        expect(page.locator("#max-full-solves")).to_be_enabled()
        page.locator("#max-full-solves").fill("1500")
        seen.clear()
        _tick(page)
        page.locator("#preview").click()
        expect(page.locator("#form-message")).to_contain_text("預覽好了")
        _settles(page, seen, 1)
        expect(page.locator("#max-full-solves")).to_have_value("1500")
        # 109 示範的重現路：格子裡已經是裝好的那一份，重開頁面不准自己轉起來。
        seen.clear()
        page.reload()
        expect(page.locator("#surrogate-state")).to_contain_text("替身沒用這些材料訓練過，要重建")
        _settles(page, seen, 1)
        _shot(page, "2-9-3-installed-not-covered-no-loop")
        page.locator("#max-full-solves").fill("1500")
        page.locator("#build-surrogate").click()
        expect(page.locator("#build-state")).to_contain_text("完整解上限 1500 次（這一份另填）")
        second = launcher.launched[-1]
        assert second[1][second[1].index("--max-full-solves") + 1] == "1500"
        # 這一次涵蓋得到（材料庫拿掉沒訓練過的那一組）：建好就自己換成「已經有替身」、可以開跑。
        library_file.unlink()
        _installed(data_dir, launcher, second[0])
        _tick(page)
        expect(page.locator("#surrogate-state")).to_have_text("這份切法已經有替身，可以直接開搜尋。")
        expect(page.locator("#start")).to_be_enabled()
        _clean(page, watched.page_errors + watched.console_errors)


def test_a_build_this_page_just_started_is_named_as_its_own(browser: Browser, tmp_path: Path) -> None:
    """抓 109 示範那一句：這一頁剛按下建替身、還沒輪詢就按預覽，忙碌行與替身區說那件是「別的資料夾或命令列開的」。"""
    data_dir = _data(tmp_path, surrogate=False)
    launcher = FakeLauncher()
    with _serve(data_dir, launcher) as url, _open(browser, f"{url}/gradient?scheme={NAME}", fake_clock=True) as watched:
        page = watched.page
        expect(page.locator("#surrogate-state")).to_contain_text("還沒有替身，要先建")
        page.locator("#build-surrogate").click()
        expect(page.locator("#build-state")).to_contain_text("建替身")
        page.locator("#preview").click()
        expect(page.locator("#form-message")).to_contain_text("預覽好了")
        expect(page.locator("#busy-line")).to_contain_text("方案「zoned」正在建替身")
        expect(page.locator("#busy-line")).not_to_contain_text("別的資料夾或命令列")
        expect(page.locator("#surrogate-message")).not_to_contain_text("別的資料夾或命令列")
        _shot(page, "2-9-4-own-build-named")
        _clean(page, watched.page_errors + watched.console_errors)


def test_the_preview_judges_a_fixed_library_table_like_the_zone_editor(browser: Browser, tmp_path: Path) -> None:
    """抓 109 示範那一格：固定區選了材料庫的實測表，分區編輯寫「總深 0.049 公尺，放得進……」，第三階段預覽卻寫「不加層」。
    預覽那一格要跟立面圖限制檢查（``/api/zone-limits``）逐字同一句；保留原牆的區照舊「不加層」。"""
    import httpx

    from tests.engine.test_gui_zone_limits import _design as _limits_design, _picked
    from tests.engine.test_material_gradient_allowed import _measured

    table = _measured(imaginary=False)
    data_dir = tmp_path / "data"
    (data_dir / "schemes").mkdir(parents=True)
    (data_dir / "designs").mkdir()
    user_library_path(data_dir).write_text(library.dump_user_library((table,)), encoding="utf-8")
    document = _picked(table)
    design = _limits_design(zones={"x0": {"z1": {"mode": "fixed", "max_depth_m": 0.05}, "auto1": {"mode": "search"},
                                          "auto2": {"mode": "keep_original"}}})
    # 帶低頻軸：建替身那一支預覽才不會照實回 400（這一題不考它，主控台一個錯都不准有）。
    plain = Scheme.model_validate(document)
    scheme = plain.model_copy(update={"scene": plain.scene.model_copy(update={"low_frequency_axis": LowFrequencyAxis.SEARCH})})
    scheme = Scheme.model_validate(scheme.model_dump(mode="json"))
    (data_dir / "schemes" / "room109.json").write_text(scheme.model_dump_json(), encoding="utf-8")
    write_design(data_dir, design, scheme)
    launcher = FakeLauncher()
    with _serve(data_dir, launcher) as url, _open(browser, f"{url}/gradient?scheme=room109", fake_clock=True) as watched:
        page = watched.page
        with httpx.Client(base_url=url, trust_env=False, timeout=60) as client:
            limits = client.post("/api/zone-limits", json={"scheme": scheme.model_dump(mode="json"),
                                                           "design": design.model_dump(mode="json")}).json()
        (line,) = limits["zones"]["x0"]["z1"]
        assert line["code"] == "depth_fits" and "總深 0.05 公尺，放得進深度上限 0.05 公尺" in line["text"]
        cell = page.locator('#plan-table tr[data-surface="x0@z1"] td.depth-cell')
        expect(cell).to_contain_text(line["text"])
        expect(cell).not_to_contain_text("不加層")
        expect(page.locator('#plan-table tr[data-surface="x0@auto2"] td.depth-cell')).to_have_text("不加層")
        _shot(page, "2-9-2-preview-fixed-table-depth")
        _clean(page, watched.page_errors + watched.console_errors)


def test_template_tries_table_keeps_quick_and_retuned_apart_and_says_why_a_round_never_opened(
        browser: Browser, tmp_path: Path) -> None:
    """#1128（審查）：換樣板試過什麼那一塊，快篩與重調後分兩欄、沒進重調的「比現狀」標「（快篩）」（不跟重調後的混成
    同一把尺）、固定候選寫在外面、沒有候選的那一輪寫它停的理由（不說成「沒換」）。"""
    data_dir = _data(tmp_path)
    launcher = FakeLauncher()
    view = _jobs(data_dir, launcher).start(NAME, GradientRequest())
    job_id = str(view["job_id"])
    folder = data_dir / "gradient" / job_id / "root" / ("f" * 32)
    folder.mkdir(parents=True)
    (folder / folder_files.PROJECT_FILE).write_text("{}", encoding="utf-8")
    rows = [{"ledger_version": "v3"},
            {"phase": "quick_score", "round": 0, "candidate": "x0@r0_c0->fixed:glass_wool_48k_50mm_gap50", "value": 3.7415},
            {"phase": "quick_score", "round": 0, "candidate": "x0@r0_c0->glass_wool_48k", "value": 3.1},
            {"phase": "template_decision", "round": 0, "value": 2.9358, "stop_reason": "沒有超過改善門檻",
             "decision": {"baseline_value": 2.9358, "accepted": None,
                          "outcomes": [{"candidate": "x0@r0_c0->glass_wool_48k", "value": 2.99}]}},
            {"phase": "template_decision", "round": 1, "value": 2.9358, "stop_reason": "換樣板求值到上限",
             "decision": {"baseline_value": None, "accepted": None}}]
    (folder / folder_files.TEMPLATES_FILE).write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    with _serve(data_dir, launcher) as url, _open(browser, f"{url}/gradient?scheme={NAME}&job={job_id}",
                                                  fake_clock=True) as watched:
        page = watched.page
        box = page.locator("#job-template-tries")
        expect(box.locator("p.tried-fixed")).to_contain_text("整塊換成「48K 玻璃棉 50 mm＋50 mm 空腔（固定，整塊選或不選）」快篩 3.7415（沒選）")
        box.locator("summary").click()
        expect(box.locator("thead th")).to_have_text(["牆／區", "換成", "快篩合分", "重調後", "比現狀", "結果"])
        fixed, family = box.locator("tbody tr").nth(0).locator("td"), box.locator("tbody tr").nth(1).locator("td")
        expect(fixed.nth(3)).to_have_text("沒進重調")
        expect(fixed.nth(4)).to_have_text("+0.8057（快篩）")
        expect(family.nth(2)).to_have_text("3.1000")
        expect(family.nth(3)).to_have_text("2.9900")
        expect(family.nth(4)).to_have_text("+0.0542")
        expect(box).to_contain_text("第 1 輪：現狀 2.9358；都沒換：沒有超過改善門檻")
        expect(box).to_contain_text("第 2 輪沒開：換樣板求值到上限")


def test_build_sends_the_options_on_the_form_and_lists_every_fixed_check(browser: Browser, tmp_path: Path) -> None:
    """#1128（#1127 審查第 8 條）：預覽之後改了開跑選項、沒再按預覽就按建替身，送出去的要是表單上現在那一份
    （搜尋真的要用的那一組起跑配置），不是舊預覽、也不是預設；建模進度列出開跑那一道固定材料核對每一次的數字。"""
    data_dir = _data(tmp_path, surrogate=False)
    launcher = FakeLauncher()
    with _serve(data_dir, launcher) as url, _open(browser, f"{url}/gradient?scheme={NAME}", fake_clock=True) as watched:
        page = watched.page
        expect(page.locator("#surrogate-state")).to_contain_text("還沒有替身，要先建")
        page.locator("#mount").select_option("rigid")
        page.locator("#build-surrogate").click()
        expect(page.locator("#build-state")).to_contain_text("建替身")
        unit, argv, _, _ = launcher.launched[-1]
        assert argv[argv.index("--mount") + 1] == "rigid"
        work = data_dir / SLOTS_DIR / "builds" / unit.removeprefix(BUILD_UNIT_PREFIX) / "work"
        work.mkdir(parents=True)
        (work / PROGRESS_FILE).write_text(_progress(time.time()).model_dump_json(), encoding="utf-8")
        row = FixedCheckRound(check=1, full_solves=96, rank=267, target="fixed:glass_wool_48k_50mm_gap50", frequency_db=0.388, band_db=0.0047,
                              worst_hz=293.441, passed=False, seconds=840.0)
        (work / FIXED_ROUNDS_FILE).write_text(row.model_dump_json() + "\n", encoding="utf-8")
        _tick(page)
        expect(page.locator("#build-checks")).to_be_visible()
        expect(page.locator("#build-checks li")).to_have_text(
            "固定材料核對第 1 次（完整解合計 96 次、秩 267）：「48K 玻璃棉 50 mm＋50 mm 空腔」逐頻 0.388 dB、帶能量 0.0047 dB，沒過（最差在 293.441 Hz）")
        expect(page.locator("#build-estimate")).to_contain_text("過了就停")
        _clean(page, watched.page_errors + watched.console_errors)
