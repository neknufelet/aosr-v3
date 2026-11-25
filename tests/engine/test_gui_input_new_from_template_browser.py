"""#1265 單元 5b：輸入頁「從範本開一份新的」（取代「載入範本」）與完全沒有方案時的空畫面。

新的一份先取名、先存（PUT 帶 If-None-Match: *），存成了才打開：畫面上從頭到尾不會有一份沒名字、沒存的範本。
名字照「另存新名字」同一套規則擋，擋的那一句寫在那一塊裡；開著那一份有沒存的改動先問；鎖頁期間按了照講、不送出。
空畫面：方案一份都沒有（或清單讀不到）時整頁只剩一張「從一份範本開始」卡，`inputReady`（頁面載完的旗子）等開好一份才設。
方案名字與內容都是考卷自己編的或範本本身；只攔網路，伺服器是考卷自己起的。"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import cast
from urllib.parse import urlsplit

import pytest
from playwright.sync_api import Browser, Page, Request, Route, expect

from aosr.gui.app import repo_root
from tests.engine._input_page_steps import (fill_new_scheme, new_from_template, open_example_scheme,
                                            open_new_scheme_box, open_save_as, open_scheme, reveal)
from tests.engine.test_gui_browser import _assert_quiet, _open, _serve, browser as browser

TEMPLATE_FILES = {"reference": "scheme_reference_room_centred.json", "stage_one": "scheme_stage_one_template.json"}
# 伺服器收不下的名字（方案代號的規則：英文字母、數字、_、-，第一個字是英文字母或數字）。
REFUSED_SHAPES = ("客廳一號", "my room", "a/b", "-x")
UNLOCKED = "!document.querySelector('main').inert && !inputBusy"
COLLECT = "structuredClone(collect())"


def _raw(template: str) -> dict[str, object]:
    return cast(dict[str, object], json.loads((repo_root() / "blueprint" / TEMPLATE_FILES[template]).read_text(encoding="utf-8")))


def _served_template(base: str, template: str) -> dict[str, object]:
    reply = json.loads(urllib.request.urlopen(f"{base}/api/example?template={template}").read())
    return cast(dict[str, object], reply["scheme"])


def _folder(data_dir: Path) -> dict[str, bytes]:
    """資料夾裡每一份方案檔的位元組（逐位比：沒多、沒少、沒改）。"""
    return {path.name: path.read_bytes() for path in sorted((data_dir / "schemes").glob("*.json"))}


def _sent(page: Page) -> list[str]:
    seen: list[str] = []

    def note(request: Request) -> None:
        parts = urlsplit(request.url)
        seen.append(f"{request.method} {parts.path}{'?' + parts.query if parts.query else ''}")
    page.on("request", note)
    return seen


def _press_new(page: Page) -> None:
    page.locator("#new-scheme-go").click()


def _note(page: Page) -> str:
    return page.locator("#new-scheme-note").inner_text()


@pytest.mark.parametrize("template", ["reference", "stage_one"])
def test_a_new_scheme_from_the_menu_is_saved_before_it_opens(tmp_path: Path, browser: Browser, template: str) -> None:
    """頂條選單「從範本開一份新的…」：存成的檔就是那份範本只換名字；存檔還在路上時表單仍是原本那一份（不先開再存）；
    存好才打開，方案名字那格換成新名字。"""
    name = f"fresh-{template.replace('_', '-')}"
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        open_example_scheme(page, base, tmp_path)
        before = page.evaluate(COLLECT)
        seen = _sent(page)
        held: list[Route] = []
        page.route(f"**/api/schemes/{name}", lambda route: held.append(route) if route.request.method == "PUT" else route.continue_())
        fill_new_scheme(page, name, template)
        with page.expect_request(lambda request: request.method == "PUT"):
            _press_new(page)
        assert page.evaluate("[openedId, document.getElementById('save-id').value]") == ["example", "example"]
        assert page.evaluate(COLLECT) == before
        assert not (tmp_path / "schemes" / f"{name}.json").exists()
        held[0].continue_()
        page.wait_for_function("name => openedId === name && document.getElementById('save-id').value === name", arg=name)
        page.wait_for_function(UNLOCKED)
        stored = json.loads((tmp_path / "schemes" / f"{name}.json").read_text(encoding="utf-8"))
        assert stored == _served_template(base, template) | {"scheme_id": name}
        assert page.evaluate(COLLECT) == stored
        expect(page.locator("#save-id")).to_have_value(name)
        expect(page.locator("#template-box")).to_be_hidden()
        expect(page.locator("#open-note")).to_contain_text(f"開一份新的「{name}」，存好了")
        expect(page.locator("#save-state")).to_have_text("都存好了（還沒算過）")
        order = [line for line in seen if line.startswith(("GET /api/example", f"PUT /api/schemes/{name}", f"GET /api/schemes/{name}"))]
        assert order == [f"GET /api/example?template={template}", f"PUT /api/schemes/{name}", f"GET /api/schemes/{name}"]
        _assert_quiet(watched)


def test_names_the_box_refuses_are_said_in_it_and_nothing_is_sent(tmp_path: Path, browser: Browser) -> None:
    """空的、形狀不對、跟已存的撞名：那一句寫在這一塊裡，一個 PUT（存檔請求）都不送，表單與資料夾逐位沒變。"""
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        open_example_scheme(page, base, tmp_path, "taken")
        open_example_scheme(page, base, tmp_path)
        before, folder = page.evaluate(COLLECT), _folder(tmp_path)
        seen = _sent(page)
        open_new_scheme_box(page)
        note = page.locator("#new-scheme-note")
        page.locator("#new-scheme-id").fill("")
        _press_new(page)
        expect(note).to_have_text("沒有開新的：「新方案的名字」這一格還空著：先填名字，再按「開一份新的」")
        for name in REFUSED_SHAPES:
            page.locator("#new-scheme-id").fill(name)
            live = note.inner_text()
            assert "「新方案的名字」" in live and "中文" in live, live
            _press_new(page)
            expect(note).to_have_text(f"沒有開新的：{live}")
            assert page.evaluate("document.activeElement.id") == "new-scheme-id"
        page.locator("#new-scheme-id").fill("taken")
        expect(note).to_have_text("")
        _press_new(page)
        expect(note).to_have_text("沒有開新的：已經有叫「taken」的方案；換一個名字（要改那一份，從「換一個方案」打開它）")
        expect(note).to_have_class("notice")
        assert page.locator("#template-box").evaluate("(box, note) => box.contains(note)", note.element_handle())
        expect(note).to_be_visible()
        page.wait_for_load_state("networkidle")
        assert not [line for line in seen if line.startswith(("PUT", "GET /api/example"))], seen
        assert _folder(tmp_path) == folder and page.evaluate(COLLECT) == before
        assert page.evaluate("openedId") == "example"
        _assert_quiet(watched)


def test_a_save_the_server_turns_down_or_a_template_it_cannot_give_leaves_form_and_folder_alone(
        tmp_path: Path, browser: Browser) -> None:
    """伺服器拒收存檔（攔 PUT 回 422）、範本讀不到（攔範本回 400）：原因寫在這一塊，表單、開著的那一份、資料夾都沒變。"""
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        open_example_scheme(page, base, tmp_path)
        before, folder = page.evaluate(COLLECT), _folder(tmp_path)
        page.route("**/api/schemes/fresh", lambda route: route.fulfill(status=422, json={"error": "伺服器不收這一份"})
                   if route.request.method == "PUT" else route.continue_())
        fill_new_scheme(page, "fresh", "stage_one")
        _press_new(page)
        expect(page.locator("#new-scheme-note")).to_have_text("沒有開新的：伺服器不收這一份")
        page.wait_for_function(UNLOCKED)
        assert page.evaluate("[openedId, document.getElementById('save-id').value]") == ["example", "example"]
        assert page.evaluate(COLLECT) == before and _folder(tmp_path) == folder
        page.unroute("**/api/schemes/fresh")
        page.route(lambda url: "template=" in url, lambda route: route.fulfill(status=400, json={"error": "範本壞了"}))
        _press_new(page)
        expect(page.locator("#new-scheme-note")).to_have_text("沒有開新的：範本壞了")
        page.wait_for_function(UNLOCKED)
        assert page.evaluate(COLLECT) == before and _folder(tmp_path) == folder
        expect(page.locator("#template-box")).to_be_visible()
        statuses = sorted(text.split("status of ")[1][:3] for text in watched.console_errors)
        assert statuses == ["400", "422"], watched.console_errors
        assert watched.page_errors == []


def test_cancel_changes_nothing(tmp_path: Path, browser: Browser) -> None:
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        open_example_scheme(page, base, tmp_path)
        before, folder = page.evaluate(COLLECT), _folder(tmp_path)
        seen = _sent(page)
        fill_new_scheme(page, "never-made", "stage_one")
        page.locator("#new-scheme-cancel").click()
        expect(page.locator("#template-box")).to_be_hidden()
        expect(page.locator("#scheme-menu-button")).to_be_focused()
        page.wait_for_load_state("networkidle")
        assert not [line for line in seen if line.startswith(("PUT", "GET /api/example"))], seen
        assert page.evaluate(COLLECT) == before and _folder(tmp_path) == folder
        assert page.evaluate("openedId") == "example"
        _assert_quiet(watched)


def test_unsaved_edits_are_asked_about_before_anything_is_saved(tmp_path: Path, browser: Browser) -> None:
    """開著那一份有改動沒存：跟「換一個方案」同一道詢問（#leave-edits）；按「取消」什麼都沒存、改動留著，
    按「不存走」才存新的那一份、打開它。"""
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        open_example_scheme(page, base, tmp_path)
        folder = _folder(tmp_path)
        page.locator("#room-Lx").fill("5.5")
        fill_new_scheme(page, "asked", "stage_one")
        _press_new(page)
        expect(page.locator("#leave-edits")).to_be_visible()
        expect(page.locator("#leave-note")).to_have_text("要先儲存目前的改動嗎？")
        page.locator("#leave-edits").get_by_role("button", name="取消", exact=True).click()
        expect(page.locator("#leave-edits")).to_be_hidden()
        page.wait_for_load_state("networkidle")
        assert _folder(tmp_path) == folder
        assert page.locator("#room-Lx").input_value() == "5.5" and page.evaluate("openedId") == "example"
        new_from_template(page, "asked", "stage_one")
        assert page.locator("#room-Lx").input_value() == "6"
        assert set(_folder(tmp_path)) == set(folder) | {"asked.json"}
        assert _folder(tmp_path)["example.json"] == folder["example.json"]
        _assert_quiet(watched)


def test_pressing_while_the_page_is_locked_is_refused_and_sends_nothing(tmp_path: Path, browser: Browser) -> None:
    """存檔卡在圖面檢查時（整頁鎖著）由頁面程式按「開一份新的」：照鎖頁的說法講在這一塊裡，不要範本、不存新檔；
    解鎖之後那一句還在。"""
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        open_example_scheme(page, base, tmp_path)
        page.locator("#room-Lx").fill("5.5")
        fill_new_scheme(page, "locked-out")
        held: list[Route] = []
        page.route("**/api/plan", lambda route: held.append(route) if not held else route.continue_())
        seen = _sent(page)
        with page.expect_request("**/api/plan"):
            page.locator("#save").evaluate("node => node.click()")
        page.locator("#new-scheme-go").evaluate("node => node.click()")
        refused = "上一個動作還沒做完，沒有開新的；等它做完再按一次「開一份新的」"
        expect(page.locator("#new-scheme-note")).to_have_text(refused)
        assert page.evaluate("document.querySelector('main').inert") is True
        held[0].continue_()
        page.wait_for_function(UNLOCKED)
        page.wait_for_load_state("networkidle")
        expect(page.locator("#messages")).to_have_text("方案已儲存")
        expect(page.locator("#new-scheme-note")).to_have_text(refused)
        assert not [line for line in seen if line.startswith("GET /api/example")], seen
        assert not (tmp_path / "schemes" / "locked-out.json").exists() and page.evaluate("openedId") == "example"
        _assert_quiet(watched)


# 從空畫面開新的時記下：inputReady 設好那一刻，開著的是哪一份、空畫面收掉了沒。
_WATCH_READY = """() => {
  window.readyAt = [];
  new MutationObserver(() => {
    if (document.body.dataset.inputReady === 'true' && !window.readyAt.length)
      window.readyAt.push([openedId, document.getElementById('empty-start').hidden, document.body.dataset.inputStart ?? null]);
  }).observe(document.body, {attributes: true, attributeFilter: ['data-input-ready']});
}"""


def _empty_start(page: Page) -> None:
    page.wait_for_function("document.body.dataset.inputStart === 'empty'")


def test_an_empty_folder_shows_only_the_start_card_and_is_ready_once_the_new_scheme_opens(
        tmp_path: Path, browser: Browser) -> None:
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _empty_start(page)
        page.wait_for_load_state("networkidle")
        expect(page.locator("#empty-start")).to_be_visible()
        expect(page.locator("#empty-start h2")).to_have_text("從一份範本開始")
        expect(page.locator("#empty-title")).to_have_text("還沒有開任何方案")
        expect(page.locator("#empty-title")).to_be_visible()
        for gone in ("#save", "#save-as-toggle", "#show-results", "#scheme-section", "#wall-section", "#input-footer",
                     "#home", "#save-state", "#save-id", "#scheme-menu-button", "#messages", "#empty-archived", "#new-scheme-cancel",
                     "#nav-zones"):
            expect(page.locator(gone)).to_be_hidden()
        expect(page.locator("#template-list")).to_have_count(0)
        expect(page.locator("#load-template")).to_have_count(0)
        assert page.locator("#empty-start").evaluate("(card, form) => card.contains(form)", page.locator("#new-scheme").element_handle())
        assert page.evaluate("document.body.dataset.inputReady") is None
        expect(page.locator("#new-template-reference")).to_be_checked()
        page.evaluate(_WATCH_READY)
        new_from_template(page, "first-room")
        assert page.evaluate("window.readyAt") == [["first-room", True, None]]
        for back in ("#save", "#save-as-toggle", "#show-results", "#scheme-section", "#wall-section", "#input-footer", "#save-id"):
            expect(page.locator(back)).to_be_visible()
        expect(page.locator("#empty-start")).to_be_hidden()
        assert page.locator("#template-box").evaluate("(box, form) => box.contains(form)", page.locator("#new-scheme").element_handle())
        assert set(_folder(tmp_path)) == {"first-room.json"}
        _assert_quiet(watched)


def test_the_empty_screen_shows_archived_schemes_and_a_restored_one_opens_straight_away(
        tmp_path: Path, browser: Browser) -> None:
    with _serve(tmp_path) as base:
        for name, lx in (("kept", 5.2), ("kept-too", 5.4)):
            scheme = _served_template(base, "reference") | {"scheme_id": name}
            cast(dict[str, dict[str, float]], scheme["scene"])["room_m"]["Lx"] = lx
            (tmp_path / "schemes").mkdir(exist_ok=True)
            (tmp_path / "schemes" / f"{name}.json").write_text(json.dumps(scheme), encoding="utf-8")
            request = urllib.request.Request(f"{base}/api/schemes/{name}/archive", data=b"{}", method="POST",
                                             headers={"Content-Type": "application/json"})
            assert json.loads(urllib.request.urlopen(request).read())["message"]
    assert _folder(tmp_path) == {}
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _empty_start(page)
        archived = page.locator("#empty-archived")
        expect(archived).to_have_text("看已封存的方案（2 份）")
        archived.click()
        expect(page.locator("#results-panel")).to_be_visible()
        expect(page.locator("#archived")).to_have_attribute("open", "")
        row = page.locator('#archived-packages-list tr[data-scheme-id="kept"]')
        expect(row).to_be_visible()
        page.evaluate(_WATCH_READY)
        row.get_by_role("button", name="搬回").click()
        page.wait_for_function("openedId === 'kept' && document.body.dataset.inputReady === 'true'")
        page.wait_for_function(UNLOCKED)
        assert page.evaluate("window.readyAt") == [["kept", True, None]]
        expect(page.locator("#results-panel")).to_be_hidden()
        expect(page.locator("#save-id")).to_have_value("kept")
        assert page.locator("#room-Lx").input_value() == "5.2"
        assert set(_folder(tmp_path)) == {"kept.json"}
        _assert_quiet(watched)


def test_a_link_to_a_scheme_that_is_not_there_on_an_empty_folder(tmp_path: Path, browser: Browser) -> None:
    with _serve(tmp_path) as base, _open(browser, f"{base}/?scheme=ghost", viewport_width=1440) as watched:
        page = watched.page
        _empty_start(page)
        expect(page.locator("#empty-start")).to_be_visible()
        expect(page.locator("#open-note")).to_have_text(
            "沒有打開「ghost」：已存的方案裡沒有這一份（可能封存了或改了名字）；從「換一個方案」另選一份")
        expect(page.locator("#scheme-section")).to_be_hidden()
        assert page.evaluate("document.body.dataset.inputReady") is None
        _assert_quiet(watched)


def test_an_unreadable_scheme_list_is_said_and_not_taken_for_no_schemes(tmp_path: Path, browser: Browser) -> None:
    """方案清單讀不到（攔 /api/schemes 回 500）：卡片上寫清楚讀不到與原因，不靜靜當成沒有方案；卡片照樣開得了新的，
    跟已存的撞名由伺服器擋（頁面不知道有哪些名字），撞到的那一份逐位沒變。"""
    with _serve(tmp_path) as base:
        taken = _served_template(base, "reference") | {"scheme_id": "taken"}
    cast(dict[str, dict[str, float]], taken["scene"])["room_m"]["Lx"] = 5.0
    (tmp_path / "schemes").mkdir(exist_ok=True)
    (tmp_path / "schemes" / "taken.json").write_text(json.dumps(taken), encoding="utf-8")
    folder = _folder(tmp_path)
    with _serve(tmp_path) as base, _open(browser, "about:blank", viewport_width=1440) as watched:
        page = watched.page
        page.route(f"{base}/api/schemes", lambda route: route.fulfill(status=500, json={"error": "清單壞了"}))
        page.goto(base, wait_until="networkidle")
        _empty_start(page)
        note = page.locator("#empty-list-note")
        expect(note).to_be_visible()
        expect(note).to_contain_text("方案清單讀不到（清單壞了）")
        expect(note).to_contain_text("伺服器會擋下來")
        assert page.evaluate("document.body.dataset.inputReady") is None
        fill_new_scheme(page, "taken")
        _press_new(page)
        expect(page.locator("#new-scheme-note")).to_contain_text("沒有開新的：已經有叫「taken」的方案")
        page.wait_for_function(UNLOCKED)
        assert _folder(tmp_path) == folder
        assert page.evaluate("document.body.dataset.inputStart") == "empty"
        new_from_template(page, "fresh")
        expect(page.locator("#empty-start")).to_be_hidden()
        assert _folder(tmp_path)["taken.json"] == folder["taken.json"]
        statuses = sorted(text.split("status of ")[1][:3] for text in watched.console_errors)
        assert statuses == ["409", "500", "500"], watched.console_errors
        assert watched.page_errors == []


def test_a_new_stage_one_scheme_fills_its_own_wall_value_when_a_wall_goes_back_to_fixed(
        tmp_path: Path, browser: Browser) -> None:
    """從第一階段範本開的新方案：牆換成構造再換回固定阻抗，填回去的是那份範本的 30ρc，不是開頁時參考房的值。"""
    walls = cast(dict[str, dict[str, float]], _raw("stage_one")["scene"])["impedance_pa_s_per_m_by_wall"]
    reference = cast(dict[str, dict[str, float]], _raw("reference")["scene"])["impedance_pa_s_per_m_by_wall"]
    assert walls["floor"] != reference["floor"]
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _empty_start(page)
        new_from_template(page, "stage-a", "stage_one")
        reveal(page, "#material-kind-floor").select_option("assembly")
        page.wait_for_selector("#wall-editor-floor [data-layer-field]")
        page.locator("#material-kind-floor").select_option("constant_impedance")
        page.wait_for_selector("#wall-floor")
        assert float(page.locator("#wall-floor").input_value()) == walls["floor"] == 12348
        # 換開一份地板是構造的方案（先開過第一階段那份）：那一面改回固定阻抗填開頁範本的值，不看上一份開過什麼。
        page.locator("#material-kind-floor").select_option("assembly")
        page.wait_for_selector("#wall-editor-floor [data-layer-field]")
        open_save_as(page)
        page.locator("#save-as-id").fill("built-floor")
        page.locator("#save-as").click()
        page.wait_for_function("openedId === 'built-floor' && !inputBusy")
        for name in ("stage-a", "built-floor"):
            open_scheme(page, name)
            page.wait_for_function(f"openedId === '{name}' && {UNLOCKED}")
        reveal(page, "#material-kind-floor").select_option("constant_impedance")
        page.wait_for_selector("#wall-floor")
        assert float(page.locator("#wall-floor").input_value()) == reference["floor"]
        _assert_quiet(watched)


def test_a_new_stage_one_scheme_replaces_the_whole_form_and_never_overwrites_a_saved_one(
        tmp_path: Path, browser: Browser) -> None:
    """原 test_gui_browser::test_template_save_load_failure_and_stage_one_template 第三段，照新流程：開第一階段範本的新方案
    跟「打開」一樣把整張表單換掉（上一份的量測草稿、結果連結、封存提示都收掉）；同名再開一次被擋、已存那份逐位沒變；
    再從參考房開一份，整張表單跟參考房一樣。"""
    raw = _raw("stage_one")
    walls = cast(dict[str, dict[str, float]], raw["scene"])["impedance_pa_s_per_m_by_wall"]
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        open_example_scheme(page, base, tmp_path)
        page.locator("#room-Lx").fill("5.5")
        reveal(page, "#material-kind-ceiling").select_option("measured_impedance")
        page.wait_for_selector("#measured-cancel-ceiling")
        page.evaluate("() => { for (const id of ['result-link', 'result-stale', 'scheme-archive-notice'])"
                      " document.getElementById(id).hidden = false; }")
        new_from_template(page, "stage-copy", "stage_one")
        assert page.locator("#source-model").input_value() == raw["source_model"]
        assert page.locator("#room-Lx").input_value() == f"{cast(dict[str, dict[str, float]], raw['scene'])['room_m']['Lx']:g}"
        for wall, value in walls.items():
            assert float(reveal(page, f"#wall-{wall}").input_value()) == value
        assert page.evaluate("() => document.getElementById('measured-cancel-ceiling')") is None
        for hidden in ("result-link", "result-stale", "scheme-archive-notice"):
            assert page.evaluate("id => document.getElementById(id).hidden", hidden) is True, hidden
        saved = tmp_path / "schemes" / "stage-copy.json"
        before = saved.read_bytes()
        assert json.loads(before)["scene"]["impedance_pa_s_per_m_by_wall"] == walls
        fill_new_scheme(page, "stage-copy", "stage_one")
        _press_new(page)
        expect(page.locator("#new-scheme-note")).to_contain_text("已經有叫「stage-copy」")
        assert saved.read_bytes() == before
        new_from_template(page, "reference-again", "reference")
        assert page.locator("#source-model").input_value() == "product_default"
        assert page.evaluate(COLLECT) == _served_template(base, "reference") | {"scheme_id": "reference-again"}
        _assert_quiet(watched)
