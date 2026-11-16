"""真瀏覽器考第三階段搜尋頁換方案時回覆晚到（#1188 第 1 條）：新那份的開跑選項還沒回來之前，上一份的預覽、表單、
「開始搜尋」都不能用；晚到的回覆（連換兩次的中間那一份、換方案前按的預覽）整個丟掉；讀不到就留在不能開跑。

不靠真時間等：用路由攔住回覆、等請求與回覆事件、頁內跟伺服器來回，再看畫面。開服務那一層是假的（同 test_gui_gradient_browser）。
"""
from __future__ import annotations

from pathlib import Path
from urllib.parse import urlsplit

from playwright.sync_api import Browser, Page, Route, expect

from aosr.gui.gradient_plan import GradientRequest
from aosr.gui.surrogate_builds import BUILD_UNIT_PREFIX
from aosr.gui.zone_design import write_design
from tests.engine.test_gui_browser import _open, browser as browser
from tests.engine.test_gui_gradient_browser import (
    REFUSED, _axis_scheme, _clean, _data, _installed, _plan_posts, _serve, _tick,
)
from tests.engine.test_gui_gradient_jobs import NAME, FakeLauncher, _design, _finish, _jobs

SECOND, THIRD, BARE = "second", "third", "bare"
ONE_ZONE = {"x0": {"r0_c0": "search"}}
THREE_ZONES = {"x0": {"r0_c0": "search", "r1_c1": "search"}, "xL": {"r0_c0": "search"}}
BLOCKS = ("form-block", "plan-block", "surrogate-block", "start-block")


def _add_scheme(data_dir: Path, name: str, zones: dict[str, dict[str, str]] | None) -> None:
    """同一份切法換個名字再存一份（替身照切法放，共用 ``_data`` 放的那一份）；``zones`` 給 None 就不寫設計設定。"""
    scheme = _axis_scheme().model_copy(update={"scheme_id": name})
    (data_dir / "schemes" / f"{name}.json").write_text(scheme.model_dump_json(), encoding="utf-8")
    if zones is not None:
        write_design(data_dir, _design(zones).model_copy(update={"scheme_id": name}), scheme)


def _plan_path(name: str) -> str:
    return f"/api/gradient/schemes/{name}/plan"


def _count_starts(page: Page) -> list[str]:
    """瀏覽器真的送出去的開跑請求（記下網址再照常放行給伺服器）。"""
    starts: list[str] = []

    def handle(route: Route) -> None:
        starts.append(urlsplit(route.request.url).path)
        route.continue_()

    page.route("**/api/gradient/schemes/*/start", handle)
    return starts


def _hold_plan(page: Page, name: str) -> list[Route]:
    held: list[Route] = []
    page.route(f"**{_plan_path(name)}", lambda route: held.append(route))
    return held


def _switch(page: Page, name: str) -> None:
    """選另一份方案、按「讀這份方案的開跑選項」，等到那一份的請求真的送出去。"""
    page.locator("#scheme-pick").select_option(name)
    with page.expect_request(lambda request: urlsplit(request.url).path == _plan_path(name)):
        page.locator("#load-plan").click()


def _flush(page: Page) -> None:
    """頁內跟伺服器來回兩趟：在這之前已經回到瀏覽器的回覆，頁面都處理完了（不靠真時間等）。"""
    for _ in range(2):
        page.evaluate("() => fetch('/api/schemes').then((reply) => reply.json())")


def _release(page: Page, held: list[Route], name: str) -> None:
    """放行攔住的那一份回覆，等它整份回到瀏覽器、頁面處理完。"""
    assert len(held) == 1
    with page.expect_response(lambda response: urlsplit(response.url).path == _plan_path(name)) as event:
        held[0].continue_()
    event.value.finished()
    _flush(page)


def _force_start(page: Page) -> None:
    """繞過畫面直接按「開始搜尋」（藏著、停用都照按）：考的是開跑那一支自己認不認這份預覽。"""
    page.evaluate("() => { const button = document.getElementById('start'); button.disabled = false; button.click(); }")
    _flush(page)


def _poll_once(page: Page) -> None:
    """推一格輪詢、等清單回來處理完：輪詢會重算「開始搜尋」亮不亮。"""
    with page.expect_response(lambda response: urlsplit(response.url).path == "/api/gradient/jobs") as event:
        _tick(page)
    event.value.finished()
    _flush(page)


def test_switching_scheme_voids_the_old_preview_until_the_new_plan_is_back(browser: Browser, tmp_path: Path) -> None:
    """抓換方案後、新那份的開跑選項還沒回來時，舊那份的預覽、表單、「開始搜尋」還開著，按下去拿新方案代號配舊表單開跑。"""
    data_dir = _data(tmp_path)
    _add_scheme(data_dir, SECOND, ONE_ZONE)
    launcher = FakeLauncher()
    with _serve(data_dir, launcher) as url, _open(browser, f"{url}/gradient?scheme={NAME}", fake_clock=True) as watched:
        page = watched.page
        page.locator("#max-steps").fill("45")
        page.locator("#preview").click()
        expect(page.locator("#plan-summary")).to_contain_text("交給搜尋 2 區")
        expect(page.locator("#plan-summary")).to_contain_text("往下走步數上限 45 步")
        expect(page.locator("#start")).to_be_enabled()
        starts = _count_starts(page)
        held = _hold_plan(page, SECOND)
        _switch(page, SECOND)
        _poll_once(page)
        shown = [block for block in BLOCKS if page.locator(f"#{block}").is_visible()]
        usable = page.locator("#start").is_visible() and page.locator("#start").is_enabled()
        _force_start(page)
        _release(page, held, SECOND)
        expect(page.locator("#plan-summary")).to_contain_text("交給搜尋 1 區")
        expect(page.locator("#plan-summary")).to_contain_text("往下走步數上限 60 步")
        expect(page.locator("#max-steps")).to_have_value("60")
        assert starts == [] and launcher.launched == [], (
            f"等新方案回覆時送出了開跑請求 {starts}；伺服器照它開了服務，步數上限 "
            f"{[argv[argv.index('--max-steps') + 1] for _, argv, _, _ in launcher.launched]}（45 是舊那份表單填的）")
        assert shown == [] and not usable, f"等新方案回覆時舊那份還開著：{shown}，開始搜尋按得下去：{usable}"
        # 新那份回來、照它自己的選項預覽過，才開得了；開的是新那份、帶它自己的步數。
        expect(page.locator("#start")).to_be_enabled()
        page.locator("#start").click()
        expect(page.locator("#job-title")).to_contain_text(f"這一跑：{SECOND}")
        assert starts == [f"/api/gradient/schemes/{SECOND}/start"]
        _, argv, _, _ = launcher.launched[-1]
        assert argv[argv.index("--max-steps") + 1] == "60"
        _clean(page, watched.page_errors + watched.console_errors)


def test_a_plan_reply_that_lands_after_a_later_switch_is_dropped(browser: Browser, tmp_path: Path) -> None:
    """抓連換兩次（第一份 → 第二份 → 第三份）、第二份的回覆比第三份晚到時，畫面被晚到的第二份蓋掉、開跑拿第三份的代號配第二份的表單。"""
    data_dir = _data(tmp_path)
    _add_scheme(data_dir, SECOND, ONE_ZONE)
    _add_scheme(data_dir, THIRD, THREE_ZONES)
    launcher = FakeLauncher()
    with _serve(data_dir, launcher) as url, _open(browser, f"{url}/gradient?scheme={NAME}", fake_clock=True) as watched:
        page = watched.page
        expect(page.locator("#plan-summary")).to_contain_text("交給搜尋 2 區")
        starts = _count_starts(page)
        held = _hold_plan(page, SECOND)
        _switch(page, SECOND)
        _switch(page, THIRD)
        expect(page.locator("#plan-summary")).to_contain_text("交給搜尋 3 區")
        # 第二份等超過 15 秒：它已經不算數，不准在第三份底下寫「還在等伺服器回應」（回覆被丟掉，那一句沒人清）。
        page.clock.run_for(15_000)
        expect(page.locator("#scheme-message")).to_have_text("")
        _release(page, held, SECOND)
        expect(page.locator("#scheme-message")).to_have_text("")
        expect(page.locator("#plan-summary")).to_contain_text("交給搜尋 3 區")
        expect(page.locator("#depth-table tbody tr")).to_have_count(3)
        expect(page.locator("#scheme-pick")).to_have_value(THIRD)
        expect(page.locator("#start")).to_be_enabled()
        page.locator("#start").click()
        expect(page.locator("#job-title")).to_contain_text(f"這一跑：{THIRD}")
        assert starts == [f"/api/gradient/schemes/{THIRD}/start"]
        _, argv, _, _ = launcher.launched[-1]
        # 開的是第三份的三區：第二份裡固定不調的那兩區，這一跑沒有被列成固定面。
        assert not [word for word in argv if word.startswith(("xL@r0_c0=", "x0@r1_c1="))]
        assert [word for word in argv if word.startswith("xL@r0_c1=")]
        _clean(page, watched.page_errors + watched.console_errors)


def test_a_scheme_that_cannot_run_does_not_fall_back_to_the_old_preview(browser: Browser, tmp_path: Path) -> None:
    """抓換到開不了第三階段搜尋的方案後，舊那份的預覽還算數：按「開始搜尋」拿開不了的那份代號配舊表單送出去。"""
    data_dir = _data(tmp_path)
    _add_scheme(data_dir, BARE, None)
    launcher = FakeLauncher()
    with _serve(data_dir, launcher) as url, _open(browser, f"{url}/gradient?scheme={NAME}", fake_clock=True) as watched:
        page = watched.page
        expect(page.locator("#start")).to_be_enabled()
        starts = _count_starts(page)
        _switch(page, BARE)
        expect(page.locator("#scheme-message")).to_contain_text("這份方案開不了第三階段搜尋")
        # 換方案那一刻「開始搜尋」就停用了：當下看一次、不重試（重試會等到下一輪輪詢替它停用）。
        assert page.locator("#start").is_disabled()
        _poll_once(page)
        for block in BLOCKS:
            expect(page.locator(f"#{block}")).to_be_hidden()
        _force_start(page)
        assert starts == [] and launcher.launched == [], f"開不了的方案送出了開跑請求 {starts}"
        _clean(page, watched.page_errors + [error for error in watched.console_errors if "status of 400" not in error])


def test_a_preview_reply_that_lands_after_a_switch_is_dropped(browser: Browser, tmp_path: Path) -> None:
    """抓按了「預覽」還沒回來就換方案、舊那份的預覽晚到時，畫面被舊那份蓋掉、「開始搜尋」照舊那份亮起來。"""
    data_dir = _data(tmp_path)
    _add_scheme(data_dir, SECOND, ONE_ZONE)
    launcher = FakeLauncher()
    with _serve(data_dir, launcher) as url, _open(browser, f"{url}/gradient?scheme={NAME}", fake_clock=True) as watched:
        page = watched.page
        expect(page.locator("#start")).to_be_enabled()
        starts = _count_starts(page)
        held = _hold_plan(page, NAME)
        with page.expect_request(lambda request: urlsplit(request.url).path == _plan_path(NAME)):
            page.locator("#preview").click()
        _switch(page, SECOND)
        expect(page.locator("#plan-summary")).to_contain_text("交給搜尋 1 區")
        # 換方案那一刻「正在預覽…」就清掉；舊那份預覽等超過 15 秒，也不准在新那份底下寫「還在等伺服器回應」。
        expect(page.locator("#form-message")).to_have_text("")
        page.clock.run_for(15_000)
        expect(page.locator("#form-message")).to_have_text("")
        _release(page, held, NAME)
        expect(page.locator("#plan-summary")).to_contain_text("交給搜尋 1 區")
        expect(page.locator("#form-message")).to_have_text("")
        expect(page.locator("#start")).to_be_enabled()
        page.locator("#start").click()
        expect(page.locator("#job-title")).to_contain_text(f"這一跑：{SECOND}")
        assert starts == [f"/api/gradient/schemes/{SECOND}/start"]
        _clean(page, watched.page_errors + watched.console_errors)


def test_a_rebuild_overtaken_by_a_switch_stops_there(browser: Browser, tmp_path: Path) -> None:
    """抓按了「重建替身」、那一份的開跑選項還沒回來人就換到別份時，重建照樣往下走：拿別份方案去問「換掉舊替身」、
    把「被准用閘門拒收」那一句寫到別份方案底下。"""
    data_dir = _data(tmp_path)
    _add_scheme(data_dir, SECOND, ONE_ZONE)
    _add_scheme(data_dir, THIRD, THREE_ZONES)
    launcher = FakeLauncher()
    jobs = _jobs(data_dir, launcher)
    view = jobs.start(SECOND, GradientRequest())
    job_id = str(view["job_id"])
    (data_dir / "gradient" / job_id / "steps" / "0.log").write_text(REFUSED + "\n", encoding="utf-8")
    _finish(data_dir, launcher, view, 1)
    with _serve(data_dir, launcher) as url, \
            _open(browser, f"{url}/gradient?scheme={NAME}&job={job_id}", fake_clock=True) as watched:
        page = watched.page
        expect(page.locator("#job-state")).to_contain_text("替身核不過")
        replaced: list[str] = []
        page.on("request", lambda request: replaced.append(urlsplit(request.url).path)
                if request.url.endswith("/surrogate/plan") and (request.post_data_json or {}).get("replace") else None)
        held = _hold_plan(page, SECOND)
        with page.expect_request(lambda request: urlsplit(request.url).path == _plan_path(SECOND)):
            page.locator("#job-rebuild").click()
        _switch(page, THIRD)
        expect(page.locator("#plan-summary")).to_contain_text("交給搜尋 3 區")
        _release(page, held, SECOND)
        _flush(page)
        assert replaced == [], f"重建被換方案追過之後還去問了換掉舊替身：{replaced}"
        expect(page.locator("#surrogate-message")).not_to_contain_text("被准用閘門拒收")
        expect(page.locator("#plan-summary")).to_contain_text("交給搜尋 3 區")
        expect(page.locator("#scheme-pick")).to_have_value(THIRD)
        _clean(page, watched.page_errors + watched.console_errors)


def test_a_build_finishing_after_a_switch_does_not_preview_for_the_other_scheme(browser: Browser, tmp_path: Path) -> None:
    """抓建替身途中換到開不了的方案、替身建好那一刻，頁面拿那份開不了的方案代號配舊表單自己送一次預覽。"""
    data_dir = _data(tmp_path, surrogate=False)
    _add_scheme(data_dir, BARE, None)
    launcher = FakeLauncher()
    with _serve(data_dir, launcher) as url, _open(browser, f"{url}/gradient?scheme={NAME}", fake_clock=True) as watched:
        page = watched.page
        expect(page.locator("#surrogate-state")).to_contain_text("還沒有替身")
        page.locator("#build-surrogate").click()
        expect(page.locator("#build-state")).to_contain_text("建替身")
        unit = launcher.launched[-1][0]
        _switch(page, BARE)
        expect(page.locator("#scheme-message")).to_contain_text("這份方案開不了第三階段搜尋")
        seen = _plan_posts(page)
        _installed(data_dir, launcher, unit)
        build = f"/api/gradient/surrogates/{unit.removeprefix(BUILD_UNIT_PREFIX)}"
        with page.expect_response(lambda response: urlsplit(response.url).path == build) as event:
            _tick(page)
        event.value.finished()
        _flush(page)
        assert seen == [], f"替身建好時替開不了的那份方案送了預覽：{seen}"
        for block in BLOCKS:
            expect(page.locator(f"#{block}")).to_be_hidden()
        _clean(page, watched.page_errors + [error for error in watched.console_errors if "status of 400" not in error])


def test_reading_the_same_scheme_again_voids_its_old_preview_until_the_reply_is_back(browser: Browser, tmp_path: Path) -> None:
    """抓同一份方案再按一次「讀這份方案的開跑選項」、回覆還沒回來（或回來是開不了）時，上一次讀出來的預覽還能開跑。"""
    data_dir = _data(tmp_path)
    launcher = FakeLauncher()
    with _serve(data_dir, launcher) as url, _open(browser, f"{url}/gradient?scheme={NAME}", fake_clock=True) as watched:
        page = watched.page
        expect(page.locator("#start")).to_be_enabled()
        starts = _count_starts(page)
        held = _hold_plan(page, NAME)
        _switch(page, NAME)
        _poll_once(page)
        _force_start(page)
        assert starts == [] and launcher.launched == [], f"重讀同一份方案、回覆還沒回來時送出了開跑請求 {starts}"
        assert len(held) == 1
        with page.expect_response(lambda response: urlsplit(response.url).path == _plan_path(NAME)) as event:
            held[0].fulfill(status=400, json={"error": "方案在別處被改過，沒有交給搜尋的區了"})
        event.value.finished()
        expect(page.locator("#scheme-message")).to_contain_text("這份方案開不了第三階段搜尋：方案在別處被改過")
        _force_start(page)
        assert starts == [] and launcher.launched == [], f"重讀同一份方案、這一次開不了時送出了開跑請求 {starts}"
        for block in BLOCKS:
            expect(page.locator(f"#{block}")).to_be_hidden()
        _clean(page, watched.page_errors + [error for error in watched.console_errors if "status of 400" not in error])
