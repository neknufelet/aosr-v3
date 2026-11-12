"""真瀏覽器考開始擺位搜尋頁（#700）：選方案、帶入之前一場、填表、預覽、開始、進度、停止、做完放進結果清單；
填錯照講哪一格、整台忙時不准開、第一階段與鎖座位各自換欄位。

開服務那一層換成假的（搜尋命令列不真跑，考卷自己寫代號檔、狀態檔、離開碼檔假裝它跑到哪）；輪詢用瀏覽器假時鐘快轉。
截圖存 ``AOSR_SHOTS``（有給才存）。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
from playwright.sync_api import Browser, Page, expect

from aosr.gui import search_jobs as search_jobs_module
from aosr.gui.search_jobs import SearchJobs
from aosr.search.select import SelectOutcome
from aosr.search.store import SearchStore
from tests.engine._search_speaker_setup_cases import flow_project
from tests.engine.test_gui_browser import _open, browser as browser
from tests.engine.test_gui_gradient_browser import _clean, _serve, _shot, _tick
from tests.engine.test_gui_gradient_jobs import FakeLauncher
from tests.engine.test_gui_search_jobs import NAME, Record, _body, _done_status, _finish, _folder, _jobs

def _data(tmp_path: Path, *names: str) -> Path:
    data_dir = tmp_path / "data"
    (data_dir / "schemes").mkdir(parents=True)
    for name in (NAME, *names):
        (data_dir / "schemes" / f"{name}.json").write_text(flow_project("stand").model_dump_json(), encoding="utf-8")
    return data_dir


def _fill(page: Page, body: Record) -> None:
    layout = cast(Record, body["layout"])
    for prefix, key in (("front", "front_distance_m"), ("spacing", "spacing_m"), ("listen", "listening_distance_m")):
        value = cast(dict[str, float], layout[key])
        page.fill(f"#{prefix}-low", str(value["low"]))
        page.fill(f"#{prefix}-high", str(value["high"]))
    page.fill("#ear-height", str(layout["ear_height_m"]))
    page.locator("#advanced").evaluate("details => { details.open = true; }")
    for selector, key in (("#budget", "budget"), ("#convergence", "convergence_run"), ("#startup", "n_startup_trials"),
                          ("#batch", "batch_size"), ("#workers", "max_workers"), ("#seed", "seed")):
        page.fill(selector, str(body[key]))


def _without_none(value: object) -> object:
    """畫面上的設定 JSON 省略空值（跟不寫那一格同義）；比之前先把記下的那一份也拿掉空值。"""
    if isinstance(value, dict):
        return {key: _without_none(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [_without_none(item) for item in value]
    return value


def _jobs_for(data_dir: Path, launcher: FakeLauncher) -> SearchJobs:
    return _jobs(data_dir, launcher)


def _shown(page: Page) -> Record:
    return cast(Record, json.loads(cast(str, page.locator("#preview-json").text_content())))


def _rich_body() -> Record:
    """表單每一種格都用到的第二階段設定（帶入來回要每一格都對）；時間上限故意取小時換不整的秒數。"""
    body = _body() | {"budget": 40, "refine": {"budget": 4, "convergence_run": 2}, "search_time_budget_s": 3607.0,
                      "feedback": {"offset": 0.25}}
    cast(Record, body["layout"]).update(
        axis_offset_m=0.1, wall_gap_m=0.05, base_angle_deg={"low": 20.0, "high": 80.0},
        listening_range_m={"low": 0.3, "high": 5.0},
        keep_out=[{"x": {"low": 5.0, "high": 6.0}, "y": {"low": 0.0, "high": 1.0}, "z": {"low": 0.0, "high": 2.0}}],
        speaker_areas=[{"x": {"low": 0.0, "high": 2.5}, "y": {"low": 0.0, "high": 4.0}, "z": {"low": 0.5, "high": 2.0}}])
    return body


def _pretend_done(data_dir: Path, launcher: FakeLauncher, jobs: SearchJobs, job_id: str, folder: Path) -> None:
    """換成一場做完的：考卷直接改工作檔裡的停止旗標、寫做完的狀態與離開碼，只為了接著考「做完」那一段。"""
    (folder / "stop").unlink()
    (folder / "refine-stop").unlink()
    job_path = data_dir / "search-jobs" / job_id / "job.json"
    document = json.loads(job_path.read_text(encoding="utf-8"))
    document["steps"][0]["stop_requested"] = False
    job_path.write_text(json.dumps(document), encoding="utf-8")
    (folder / "status.json").write_text(_done_status().model_dump_json(), encoding="utf-8")
    launcher.active.clear()
    _finish(jobs, job_id, 0, start=0, auto=0)


def test_scheme_to_preview_start_progress_stop_and_result(browser: Browser, tmp_path: Path,
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    """抓表單沒照填的送、預覽沒擋改過的表單、開了沒進清單、進度與進度頁連結沒接上、停止沒放記號、做完沒法放進結果清單。"""
    data_dir = _data(tmp_path)
    launcher = FakeLauncher()
    calls: list[int | None] = []

    def fake(store: SearchStore, trial_number: int | None, data: Path) -> SelectOutcome:
        calls.append(trial_number)
        return SelectOutcome(run_id="r" * 32, result_path=data / "results" / "x.json", is_new=True)

    monkeypatch.setattr(search_jobs_module, "select_refined", fake)
    with _serve(data_dir, launcher) as url, _open(browser, f"{url}/search-start?scheme={NAME}", fake_clock=True) as watched:
        page = watched.page
        expect(page.locator("#facts-list")).to_contain_text("書架喇叭")
        expect(page.locator("#previous-message")).to_contain_text("還沒有任何一場搜尋可以帶入")
        expect(page.locator("#speaker-fields")).to_be_hidden()
        body = _body()
        _fill(page, body)
        page.click("#preview")
        expect(page.locator("#preview-list")).to_contain_text("搜尋 11 題為上限、連續 5 個沒改善就停")
        expect(page.locator("#preview-list")).to_contain_text("喇叭聲學中心高 1.2 公尺")
        _shot(page, "search-start-preview")
        page.fill("#budget", "12")
        expect(page.locator("#preview-block")).to_be_hidden()
        page.click("#preview")
        expect(page.locator("#start")).to_be_enabled()
        page.click("#start")
        expect(page.locator("#job-state")).to_contain_text("開跑中")
        assert len(launcher.launched) == 1
        jobs = _jobs_for(data_dir, launcher)
        job_id = cast(str, cast(list[Record], jobs.listing()["jobs"])[0]["job_id"])
        settings = json.loads((data_dir / "search-jobs" / job_id / "settings.json").read_text(encoding="utf-8"))
        # 整份比：表單每一格都照填的送（間距與聆聽距離對調、漏一格都會紅）。
        assert _without_none(settings) == _without_none(jobs.preview(NAME, body | {"budget": 12})["settings"])
        folder = _folder(jobs, job_id)
        _tick(page)
        expect(page.locator("#job-state")).to_contain_text("搜尋中")
        expect(page.locator("#job-progress")).to_have_attribute("href", f"/searches/{folder.name}")
        expect(page.locator("#busy-line")).to_contain_text(f"方案「{NAME}」的擺位搜尋")
        _shot(page, "search-start-running")
        page.click("#job-stop")
        expect(page.locator("#job-state")).to_contain_text("停止中")
        assert (folder / "stop").is_file() and (folder / "refine-stop").is_file()
        _pretend_done(data_dir, launcher, jobs, job_id, folder)
        _tick(page)
        expect(page.locator("#job-state")).to_contain_text("做完了")
        page.click("#job-select")
        expect(page.locator("#job-result a")).to_have_attribute("href", f"/results/{'r' * 32}")
        assert calls == [7]
        _shot(page, "search-start-done")
        page.click("#preview")
        expect(page.locator("#start")).to_be_visible()
        expect(page.locator("#start-message")).to_have_text("")
        _clean(page, watched.page_errors + watched.console_errors)


def test_previous_search_of_the_same_scheme_fills_the_form(browser: Browser, tmp_path: Path) -> None:
    """抓同一份方案搜過卻沒自動帶入、帶入後的預覽跟那一場的設定不同（每一種格都有填）、時間上限來回換算走樣。"""
    data_dir = _data(tmp_path)
    launcher = FakeLauncher()
    jobs = _jobs(data_dir, launcher)
    body = _rich_body()
    job_id = cast(str, jobs.start(NAME, body)["job_id"])
    folder = _folder(jobs, job_id)
    launcher.active.clear()
    _finish(jobs, job_id, 0, start=1, auto=None)
    with _serve(data_dir, launcher) as url, _open(browser, f"{url}/search-start?scheme={NAME}", fake_clock=True) as watched:
        page = watched.page
        expect(page.locator("#previous-message")).to_contain_text("同一份方案")
        expect(page.locator("#previous-message")).to_contain_text("喇叭高與箱體")
        # 秒換小時：3607 秒顯示成 1.0019… 小時（不是 3607，也不是分鐘）。
        assert abs(float(page.locator("#time-budget").input_value()) * 3600 - 3607) < 1e-6
        expect(page.locator("#budget")).to_have_value("40")
        expect(page.locator("#keep-out .box-row")).to_have_count(1)
        expect(page.locator("#speaker-areas .box-row")).to_have_count(1)
        page.click("#preview")
        expect(page.locator("#preview-list")).to_contain_text("禁區 1 塊；喇叭可用區 1 塊")
        assert _shown(page) == _without_none(SearchStore.open(folder).settings.model_dump(mode="json"))
        # 手填的小時照乘 3600 送秒數。
        page.fill("#time-budget", "1.5")
        page.click("#preview")
        expect(page.locator("#preview-list")).to_contain_text("搜尋最多 1.5 小時")
        assert _shown(page)["search_time_budget_s"] == 5400
        # 重讀同一份方案不清掉剛填的表單。
        page.click("#load-scheme")
        expect(page.locator("#scheme-message")).to_contain_text("表單照你填的留著")
        expect(page.locator("#time-budget")).to_have_value("1.5")
        _clean(page, watched.page_errors + watched.console_errors)


def test_form_problems_name_the_field_and_busy_blocks_start(browser: Browser, tmp_path: Path) -> None:
    """抓空白必填沒擋、伺服器拒收沒照原文、整台忙時還能按開始。"""
    data_dir = _data(tmp_path)
    launcher = FakeLauncher()
    launcher.active = {"aosr-gradient-elsewhere-0"}
    with _serve(data_dir, launcher) as url, _open(browser, f"{url}/search-start?scheme={NAME}", fake_clock=True) as watched:
        page = watched.page
        expect(page.locator("#facts-list")).to_be_visible()
        _fill(page, _body())
        page.fill("#budget", "")
        page.click("#preview")
        expect(page.locator("#form-message")).to_have_text("「搜尋題數上限（題）」要填")
        page.fill("#budget", "0")
        page.click("#preview")
        expect(page.locator("#form-message")).to_contain_text("budget")
        page.fill("#budget", "11")
        # 瀏覽器解析不了的字在數字格裡讀出來是空字串：不准當成「不限制」省略。
        page.locator("#wall-gap").press_sequentially("0.3e")
        page.click("#preview")
        expect(page.locator("#form-message")).to_have_text("「箱體離四面牆至少（公尺）」要是數字")
        page.fill("#wall-gap", "")
        page.click("#add-keep-out")
        row = page.locator("#keep-out .box-row input")
        for index, value in enumerate(("5", "6", "0", "1", "0")):
            row.nth(index).fill(value)
        row.nth(5).press_sequentially("2e")
        page.click("#preview")
        expect(page.locator("#form-message")).to_have_text("第 1 塊禁區的六格都要填數字")
        page.click("#keep-out .box-row button")
        page.click("#preview")
        expect(page.locator("#busy-line")).to_contain_text("第三階段搜尋或建替身")
        expect(page.locator("#start")).to_be_disabled()
        _shot(page, "search-start-busy")
        _clean(page, watched.page_errors + [error for error in watched.console_errors if "400" not in error])


def test_stage_one_and_seat_lock_switch_their_fields(browser: Browser, tmp_path: Path) -> None:
    """抓第一階段沒換出房間範圍與細算保留名額、鎖座位還要聆聽距離、細算保留名額沒配較遠對照，或送出去不是第一階段。"""
    data_dir = _data(tmp_path)
    with _serve(data_dir, FakeLauncher()) as url, _open(browser, f"{url}/search-start?scheme={NAME}", fake_clock=True) as watched:
        page = watched.page
        expect(page.locator("#facts-list")).to_be_visible()
        _fill(page, _body())
        page.check("#seat-locked")
        expect(page.locator("#free-seat")).to_be_hidden()
        page.click("#preview")
        expect(page.locator("#preview-block")).to_be_visible()
        locked = cast(Record, _shown(page)["layout"])
        assert locked["seat_locked"] is True and "listening_distance_m" not in locked
        page.uncheck("#seat-locked")
        page.check("input[name=stage][value=one]")
        expect(page.locator("#room-fields")).to_be_visible()
        expect(page.locator("#stage-two-budget")).to_be_hidden()
        page.fill("#top-n", "3")
        page.click("#preview")
        expect(page.locator("#form-message")).to_contain_text("較遠對照")
        page.fill("#far-control", "0.8")
        page.click("#preview")
        # 這份方案的喇叭有指向性，第一階段的檢查照實拒收：拒收原因本身就證明送出去的是第一階段（建資料夾前那一道）。
        expect(page.locator("#form-message")).to_contain_text("第一階段只收全向聲源")
        _shot(page, "search-start-stage-one")
        _clean(page, watched.page_errors + [error for error in watched.console_errors if "400" not in error])


def test_switching_scheme_voids_the_preview_and_clears_the_form(browser: Browser, tmp_path: Path) -> None:
    """抓換方案讀失敗時還能拿上一份方案預覽過的設定開在新方案上，或換到沒搜過的方案時表單留著上一份的值。"""
    data_dir = _data(tmp_path, "other")
    launcher = FakeLauncher()
    with _serve(data_dir, launcher) as url, _open(browser, f"{url}/search-start?scheme={NAME}", fake_clock=True) as watched:
        page = watched.page
        expect(page.locator("#facts-list")).to_be_visible()
        _fill(page, _body())
        page.click("#preview")
        expect(page.locator("#preview-list")).to_contain_text(f"方案：「{NAME}」")
        page.route("**/api/search-jobs/schemes/other", lambda route: route.abort())
        page.select_option("#scheme-pick", "other")
        page.click("#load-scheme")
        expect(page.locator("#scheme-message")).to_contain_text("連不到網頁伺服器")
        expect(page.locator("#preview-block")).to_be_hidden()
        expect(page.locator("#form-block")).to_be_hidden()
        page.evaluate("start()")
        expect(page.locator("#start-message")).to_contain_text("要先重新預覽")
        assert launcher.launched == []
        page.unroute("**/api/search-jobs/schemes/other")
        page.click("#load-scheme")
        expect(page.locator("#scheme-message")).to_have_text("已讀方案「other」")
        for selector in ("#front-low", "#spacing-high", "#listen-low", "#budget"):
            expect(page.locator(selector)).to_have_value("")
        expect(page.locator("#ear-height")).to_have_value("1.2")
        _clean(page, watched.page_errors + [error for error in watched.console_errors if "ERR_FAILED" not in error])


def test_job_buttons_follow_the_actions_and_resume_launches(browser: Browser, tmp_path: Path,
                                                            monkeypatch: pytest.MonkeyPatch) -> None:
    """抓停止／接續／放進結果清單沒照後端 actions 顯示、接續沒開服務、換看另一跑時還掛著上一跑的結果連結。"""
    data_dir = _data(tmp_path)
    launcher = FakeLauncher()
    jobs = _jobs(data_dir, launcher)
    monkeypatch.setattr(search_jobs_module, "select_refined", lambda store, trial, data: SelectOutcome(
        run_id="r" * 32, result_path=data / "results" / "x.json", is_new=True))
    done = cast(str, jobs.start(NAME, _body())["job_id"])
    _folder(jobs, done, _done_status())
    launcher.active.clear()
    _finish(jobs, done, 0, start=0, auto=0)
    cut = cast(str, jobs.start(NAME, _body())["job_id"])
    _folder(jobs, cut)
    launcher.active.clear()
    with _serve(data_dir, launcher) as url, _open(browser, f"{url}/search-start?job={done}", fake_clock=True) as watched:
        page = watched.page
        expect(page.locator("#job-state")).to_contain_text("做完了")
        expect(page.locator("#job-select")).to_be_visible()
        expect(page.locator("#job-resume")).to_be_hidden()
        expect(page.locator("#job-stop")).to_be_hidden()
        page.click("#job-select")
        expect(page.locator("#job-result")).to_be_visible()
        page.locator("#job-list tr", has_text="中斷").get_by_role("button").click()
        expect(page.locator("#job-result")).to_be_hidden()
        expect(page.locator("#job-resume")).to_be_visible()
        expect(page.locator("#job-select")).to_be_hidden()
        expect(page.locator("#job-stop")).to_be_hidden()
        page.click("#job-resume")
        expect(page.locator("#job-message")).to_have_text("已接續")
        assert len(launcher.launched) == 3
        _clean(page, watched.page_errors + watched.console_errors)
