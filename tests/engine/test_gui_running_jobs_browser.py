"""同時在算好幾筆時首頁怎麼跟：重新整理接回全部、改看別筆、等回覆時又換過也只留一條輪詢（#796）。"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest
from playwright.sync_api import Browser, Page

from aosr.reporting.result import SchemeResult, save_result
from aosr.reporting.scheme import Scheme
from tests.engine.test_gui_browser import _assert_quiet, _open, _serve, browser, result


def test_reload_lists_every_running_job_and_follows_the_chosen_one(tmp_path: Path, browser: Browser) -> None:
    # #796：同時在算兩筆時重新整理，主畫面跟最新開的那一筆，另一筆列在底下；改看那一筆之後停得掉，
    # 底下換成列原本那一筆；那筆在別處停掉，幾秒內就從底下拿掉。
    script = tmp_path / "sleep-runner.py"
    script.write_text("import time\ntime.sleep(60)\n")
    headers = {"Content-Type": "application/json"}
    started: list[str] = []
    with _serve(tmp_path, (sys.executable, str(script))) as base, _open(browser, f"{base}/") as watched:
        page = watched.page
        try:
            document = page.request.get(f"{base}/api/example").json()["scheme"]
            for name in ("older", "newer"):
                document["scheme_id"] = name
                assert page.request.put(f"{base}/api/schemes/{name}", data=json.dumps(document), headers=headers).ok
                reply = page.request.post(f"{base}/api/runs", data=json.dumps({"scheme_id": name}), headers=headers)
                assert reply.ok
                started.append(reply.json()["run_id"])
                time.sleep(0.05)
            older, newer = started
            page.reload(wait_until="networkidle")
            page.wait_for_function("() => !document.querySelector('#other-runs').hidden")
            assert page.evaluate("() => runId") == newer
            assert "計算中" in page.locator("#run-state").inner_text()
            assert [text.startswith("另一筆：「older」計算中") and text.endswith("改看這一筆")
                    for text in page.locator("#other-runs li").all_inner_texts()] == [True]
            page.locator("#other-runs button").click()
            page.wait_for_function("() => document.querySelector('#other-runs').textContent.includes('「newer」')")
            assert page.evaluate("() => runId") == older
            assert [text.startswith("另一筆：「newer」計算中")
                    for text in page.locator("#other-runs li").all_inner_texts()] == [True]
            page.locator("#stop").click()
            page.wait_for_function("() => document.querySelector('#run-state').textContent.includes('已停止')")
            assert page.request.get(f"{base}/api/runs/{older}").json()["status"] != "running"
            assert page.request.get(f"{base}/api/runs/{newer}").json()["status"] == "running"
            assert page.request.post(f"{base}/api/runs/{newer}/stop", data="{}", headers=headers).ok
            page.wait_for_function("() => document.querySelector('#other-runs').hidden", timeout=10_000)
            assert page.locator("#other-runs li").all_inner_texts() == []
            _assert_quiet(watched)
        finally:
            for run_id in started:
                if page.request.get(f"{base}/api/runs/{run_id}").json()["status"] == "running":
                    page.request.post(f"{base}/api/runs/{run_id}/stop", data="{}", headers=headers)


_HEADERS = {"Content-Type": "application/json"}
# 頁面上的 fetch 換成有關卡的版本：網址路徑命中 held 的請求卡住，考卷呼叫 openGate() 才放行。
# 另外記：計算清單（/api/runs 本身的 GET）同時在路上幾個、總共問幾次；每筆計算被 poll 完幾次；follow 做完幾次。
# poll、follow 是頁面腳本最外層的函式宣告，換掉 window 上那一個，按鈕與計時器照名字找就會用到換過的。
_GATED_PAGE = """(held) => {
  const original = window.fetch;
  let open;
  window.gate = new Promise((done) => { open = done; });
  window.openGate = () => open();
  window.inFlight = 0; window.mostInFlight = 0; window.listCalls = 0;
  window.fetch = async (input, init) => {
    const path = String(input);
    const list = path === "/api/runs" && (init === undefined || init.method === "GET");
    if (list) { window.listCalls += 1; window.inFlight += 1; window.mostInFlight = Math.max(window.mostInFlight, window.inFlight); }
    try {
      if (new RegExp(held).test(path)) await window.gate;
      return await original(input, init);
    } finally {
      if (list) window.inFlight -= 1;
    }
  };
  window.pollsDone = {}; window.followsDone = 0;
  const realPoll = window.poll;
  window.poll = async () => {
    const polled = runId;
    try { return await realPoll(); } finally { window.pollsDone[polled] = (window.pollsDone[polled] || 0) + 1; }
  };
  const realFollow = window.follow;
  window.follow = async (id) => {
    try { return await realFollow(id); } finally { window.followsDone += 1; }
  };
}"""


def _start_runs(page: Page, base: str, names: tuple[str, ...]) -> list[str]:
    """用範本存幾份方案、依序開算（開始時間錯開），回傳計算代號。"""
    document = page.request.get(f"{base}/api/example").json()["scheme"]
    started: list[str] = []
    for name in names:
        document["scheme_id"] = name
        assert page.request.put(f"{base}/api/schemes/{name}", data=json.dumps(document), headers=_HEADERS).ok
        reply = page.request.post(f"{base}/api/runs", data=json.dumps({"scheme_id": name}), headers=_HEADERS)
        assert reply.ok
        started.append(reply.json()["run_id"])
        time.sleep(0.05)
    return started


def _stop_runs(page: Page, base: str, started: list[str]) -> None:
    for run_id in started:
        if page.request.get(f"{base}/api/runs/{run_id}").json()["status"] == "running":
            page.request.post(f"{base}/api/runs/{run_id}/stop", data="{}", headers=_HEADERS)


def _stop_and_expect_no_polling(page: Page) -> None:
    """按停止、等到「已停止」，之後 2.5 秒內不准再有任何一筆計算被輪詢（留下來的計時器每秒查一次）。"""
    page.locator("#stop").click()
    page.wait_for_function("() => document.querySelector('#run-state').textContent.includes('已停止')")
    polled: list[str] = []
    page.on("request", lambda request: polled.append(request.url) if "/api/runs/" in request.url else None)
    page.wait_for_timeout(2500)
    assert polled == []


def _sleep_runner(tmp_path: Path) -> tuple[str, ...]:
    script = tmp_path / "sleep-runner.py"
    script.write_text("import time\ntime.sleep(60)\n")
    return (sys.executable, str(script))


@pytest.mark.parametrize("clicks", [("second", "first"), ("second", "second")])
def test_switching_twice_before_the_first_answer_leaves_one_poller(tmp_path: Path, browser: Browser,
                                                                   clicks: tuple[str, str]) -> None:
    # #796 審查：「改看這一筆」第一下的回覆還沒到就再按一下（換別列，或同一列再按）：計時器只能留最後那一次排的，
    # 停掉之後不再輪詢。第一下的回覆由考卷卡住，兩下都做完才放行、才按停止。
    started: list[str] = []
    with _serve(tmp_path, _sleep_runner(tmp_path)) as base, _open(browser, f"{base}/") as watched:
        page = watched.page
        try:
            started = _start_runs(page, base, ("first", "second", "third"))
            ids = dict(zip(("first", "second", "third"), started))
            page.reload(wait_until="networkidle")
            page.wait_for_function("() => document.querySelectorAll('#other-runs li').length >= 2")
            page.evaluate(_GATED_PAGE, f"^/api/runs/{ids['second']}$")
            # 兩下在同一段頁面腳本裡按：中間若輪到每兩秒的清單重問，回來會把正在看的那一列拿掉，同一列的第二下就按不到。
            page.evaluate("""(names) => {
              const rows = [...document.querySelectorAll('#other-runs li')];
              for (const name of names) rows.find((row) => row.textContent.includes(`「${name}」`)).querySelector('button').click();
            }""", list(clicks))
            page.evaluate("() => openGate()")
            page.wait_for_function("() => window.followsDone >= 2")
            assert page.evaluate("() => runId") == ids[clicks[-1]]
            _stop_and_expect_no_polling(page)
            _assert_quiet(watched)
        finally:
            _stop_runs(page, base, started)


def test_switching_while_a_new_calculation_answers_leaves_one_poller(tmp_path: Path, browser: Browser) -> None:
    # #796 審查：按「計算這個方案」後、新那筆的第一次回覆還沒到就改看另一筆：計時器歸改看的那一次，停掉之後不再輪詢。
    started: list[str] = []
    with _serve(tmp_path, _sleep_runner(tmp_path)) as base, _open(browser, f"{base}/") as watched:
        page = watched.page
        try:
            started = _start_runs(page, base, ("other", "followed"))
            other, followed = started
            page.reload(wait_until="networkidle")
            page.wait_for_function("() => !document.querySelector('#other-runs').hidden")
            # 只卡新開那一筆的狀態查詢（不是這兩筆的任何一筆）。
            page.evaluate(_GATED_PAGE, f"^/api/runs/(?!{other}|{followed})[0-9a-f]{{32}}$")
            with page.expect_response(lambda response: response.url == f"{base}/api/runs"
                                      and response.request.method == "POST") as created:
                page.locator("#calculate").click()
            new_run = created.value.json()["run_id"]
            started.append(new_run)
            page.locator("#other-runs li", has_text="「other」").locator("button").click()
            page.wait_for_function("() => window.followsDone >= 1")
            page.evaluate("() => openGate()")
            page.wait_for_function(f"() => (window.pollsDone['{new_run}'] || 0) >= 1")
            assert page.evaluate("() => runId") == other
            _stop_and_expect_no_polling(page)
            _assert_quiet(watched)
        finally:
            _stop_runs(page, base, started)


def test_other_runs_list_keeps_one_request_in_flight(tmp_path: Path, browser: Browser) -> None:
    # #796 審查：「另一筆」那份清單的回覆卡住時又被叫好幾次：一次只准一個請求在路上；放行後重問照樣接著跑。
    started: list[str] = []
    with _serve(tmp_path, _sleep_runner(tmp_path)) as base, _open(browser, f"{base}/") as watched:
        page = watched.page
        try:
            started = _start_runs(page, base, ("older", "newer"))
            page.reload(wait_until="networkidle")
            page.wait_for_function("() => !document.querySelector('#other-runs').hidden")
            page.evaluate(_GATED_PAGE, "^/api/runs$")
            page.evaluate("() => { showOtherRuns(); showOtherRuns(); showOtherRuns(); }")
            page.wait_for_function("() => window.inFlight >= 1")
            assert page.evaluate("() => window.mostInFlight === 1")
            page.evaluate("() => openGate()")
            page.wait_for_function("() => window.listCalls >= 2", timeout=15_000)
            assert page.evaluate("() => window.mostInFlight === 1")
            _assert_quiet(watched)
        finally:
            _stop_runs(page, base, started)


def test_each_calculation_runs_the_form_on_screen(tmp_path: Path, browser: Browser, result: SchemeResult) -> None:
    """#972 施工二：網頁自己的狀態接線，不交給命令列考卷。假計算把收到的方案檔原文記下來再交出考卷結果：
    改了表單直接按計算（不另外存）、另存成別的名字再算，每一筆子行程收到的都是那一刻畫面上的方案，不是上一筆的參數；
    算完的連結指向這一筆自己的計算代號，不是上一筆的。"""
    source = tmp_path / "fixture-result.json"
    save_result(result, source)
    log = tmp_path / "received.jsonl"
    script = tmp_path / "record-and-copy.py"
    script.write_text("import json,shutil,sys\nfrom pathlib import Path\n"
                      f"with open({str(log)!r}, 'a') as handle:\n"
                      "    handle.write(json.dumps({'path': sys.argv[1], 'scheme': json.loads(Path(sys.argv[1]).read_text())}) + '\\n')\n"
                      f"shutil.copyfile({str(source)!r}, sys.argv[sys.argv.index('--out') + 1])\n")
    with _serve(tmp_path, (sys.executable, str(script))) as base, _open(browser, f"{base}/") as watched:
        page = watched.page
        links = []
        for step, (lx, save_as) in enumerate(((None, None), ("5.5", None), ("5.6", "next"))):
            if lx is not None:
                page.locator("#room-Lx").fill(lx)
            if save_as is not None:
                page.locator("#save-as-id").fill(save_as)
                page.locator("#save-as").click()
                page.wait_for_function("name => document.querySelector('#save-id').value === name", arg=save_as)
            shown = page.evaluate("collect()")
            with page.expect_response(lambda response: response.url == f"{base}/api/runs"
                                      and response.request.method == "POST") as started:
                page.locator("#calculate").click()
            run_id = started.value.json()["run_id"]
            page.wait_for_function("id => !document.querySelector('#result-link').hidden && "
                                   "document.querySelector('#result-link').getAttribute('href') === `/results/${id}`",
                                   arg=run_id, timeout=15_000)
            links.append(page.locator("#result-link").get_attribute("href"))
            received = [json.loads(line) for line in log.read_text().splitlines()]
            assert len(received) == step + 1
            assert Path(received[-1]["path"]).name == f"{shown['scheme_id']}.json"
            assert received[-1]["scheme"] == Scheme.model_validate(shown).model_dump(mode="json")
            if lx is not None:
                assert received[-1]["scheme"]["scene"]["room_m"]["Lx"] == float(lx)
        assert len(set(links)) == len(links)
        _assert_quiet(watched)
