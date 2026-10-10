"""方案與預覽交接的瀏覽器考卷；只攔網路，逐欄答案手寫。"""
import time
from pathlib import Path
from typing import cast

import pytest
from playwright.sync_api import Browser, Request, Route
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from aosr.reporting.scheme import Scheme
from tests.engine import _furniture_cases as cases
from tests.engine._gui_cache import gui_startup_identity_memo as gui_startup_identity_memo
from tests.engine.test_gui_browser import _open, _serve, browser as browser
from tests.engine.test_gui_furniture_problem_text import assert_chinese_lines


def _posted_scheme(request: Request) -> dict[str, object]:
    document: object = request.post_data_json
    assert isinstance(document, dict)
    return cast(dict[str, object], document)


def test_delayed_open_preserves_readonly_same_id_table(browser: Browser, tmp_path: Path) -> None:
    editable = {"furniture_id": "table", "kind": "desk", "material": "wood",
                "width_m": 1.4, "depth_m": 0.75, "height_m": 0.03,
                "placement": {"forward_m": 0.825, "left_m": 0, "bottom_height_m": 0.72, "yaw_deg": 0}}
    readonly = {"furniture_id": "table", "kind": "desk", "material": "glass",
                "width_m": 1.65, "depth_m": 0.85, "height_m": 0.04,
                "placement": {"forward_m": 1.6, "left_m": -0.4, "bottom_height_m": 0.68, "yaw_deg": 90}}
    (tmp_path / "schemes").mkdir()
    for name, table in (("editable", editable), ("readonly", readonly)):
        document = cases.document(table) | {"scheme_id": name}
        (tmp_path / "schemes" / f"{name}.json").write_text(Scheme.model_validate(document).model_dump_json())
    with _serve(tmp_path) as url, _open(browser, url) as watched:
        page = watched.page
        page.locator("#scheme-list").select_option("editable")
        page.locator("#open-scheme").click()
        page.wait_for_function("document.getElementById('save-id').value === 'editable' && !document.querySelector('main').inert")
        page.wait_for_load_state("networkidle")
        held: list[Route] = []
        delayed = False

        def delay_preview(route: Route) -> None:
            nonlocal delayed
            if _posted_scheme(route.request)["scheme_id"] == "readonly" and not delayed:
                delayed = True
                held.append(route)
            else:
                route.continue_()

        page.route("**/api/input-preview", delay_preview)
        page.locator("#scheme-list").select_option("readonly")
        with page.expect_request(lambda request: request.url.endswith("/api/input-preview")
                                 and _posted_scheme(request)["scheme_id"] == "readonly"):
            page.locator("#open-scheme").click()
        # 真鍵盤嘗試修改任一數字格：新版 inert（暫停互動）擋住，舊版會收舊桌子格進新方案。
        page.locator("#room-Lx").evaluate("node => node.focus()")
        page.keyboard.press("ControlOrMeta+A")
        page.keyboard.insert_text("9")
        during = page.evaluate("({locked: document.querySelector('main').inert, id: scheme.scheme_id, shown: document.getElementById('save-id').value, furniture: scheme.furniture})")
        assert held
        held[0].continue_()
        page.wait_for_function("document.getElementById('save-id').value === 'readonly' && !document.querySelector('main').inert")
        page.wait_for_load_state("networkidle")
        page.locator("#save-as-id").fill("kept-readonly")
        with page.expect_response(lambda response: response.url.endswith("/api/schemes/kept-readonly")
                                  and response.request.method == "PUT") as saved:
            page.locator("#save-as").click()
        assert saved.value.ok, saved.value.text()
        readback = page.request.get(f"{url}/api/schemes/kept-readonly").json()["scheme"]
        assert readback["furniture"] == [readonly]
        assert during == {"locked": True, "id": "editable", "shown": "editable", "furniture": [editable]}
        assert page.locator("#furniture-table-kind").is_disabled()
        assert watched.page_errors == []


def test_failed_preview_after_cloud_edit_keeps_old_form_and_saves(browser: Browser, tmp_path: Path) -> None:
    with _serve(tmp_path) as url, _open(browser, url) as watched:
        page = watched.page
        failed = False

        def fail_first_cloud_preview(route: Route) -> None:
            nonlocal failed
            items = cast(list[dict[str, object]], _posted_scheme(route.request).get("furniture") or [])
            if any(item["furniture_id"] == "cloud" for item in items) and not failed:
                failed = True
                route.fulfill(status=503, json={"error": "預覽暫時讀不到，請重試"})
            else:
                route.continue_()

        page.route("**/api/input-preview", fail_first_cloud_preview)
        with page.expect_response(lambda response: response.url.endswith("/api/input-edit")) as edited:
            page.locator("#furniture-cloud").click()
        assert edited.value.ok
        page.wait_for_function("!document.getElementById('input-setup').hasAttribute('aria-busy') && document.getElementById('messages').textContent !== '檢查通過'")
        assert failed
        failure_text = page.locator("#messages").inner_text()
        assert_chinese_lines(failure_text.splitlines())
        # 失敗後立即存檔；舊版補問預覽後仍缺天雲格子，這裡會印出英文空元素錯誤。
        page.locator("#save-as-id").fill("kept-without-cloud")
        page.locator("#save-as").click()
        page.wait_for_function("document.getElementById('save-as-note').textContent !== ''")
        assert_chinese_lines(page.locator("#messages").inner_text().splitlines())
        saved = page.request.get(f"{url}/api/schemes/kept-without-cloud")
        assert saved.ok, saved.text()
        readback = saved.json()["scheme"]
        assert readback["furniture"] is None
        assert not page.locator("#furniture-cloud").is_checked()
        assert page.locator("#furniture-cloud-fields input").all() == []
        assert "換不過去" in failure_text
        # 再改數字也要能收值，不留下缺格子的方案／預覽組合。
        page.locator("#room-Lx").fill(page.locator("#room-Lx").input_value())
        page.locator("#check").click()
        page.wait_for_function("document.getElementById('messages').textContent === '檢查通過'")
        assert watched.page_errors == []


@pytest.mark.parametrize("selector", ["#shortcut-working", "#shortcut-listening", "#use-speaker-setup"])
def test_failed_preview_keeps_shortcut_and_speaker_setup_old(browser: Browser, tmp_path: Path,
                                                           selector: str) -> None:
    with _serve(tmp_path) as url, _open(browser, url) as watched:
        page = watched.page
        page.route("**/api/input-preview", lambda route: route.fulfill(status=503, json={"error": "預覽暫時讀不到"}), times=1)
        with page.expect_response(lambda response: response.url.endswith("/api/input-edit")) as edited:
            page.locator(selector).click()
        assert edited.value.ok
        page.wait_for_function("!document.querySelector('main').inert && document.getElementById('messages').textContent.includes('換不過去')")
        assert_chinese_lines(page.locator("#messages").inner_text().splitlines())
        assert page.locator("#receiver-main-z").input_value() == "1.2"
        assert not page.locator("#use-speaker-setup").is_checked()
        assert not page.locator("#furniture-sofa").is_checked()
        assert page.locator("#furniture-table-kind").input_value() == "none"
        page.locator("#save-as-id").fill("unchanged")
        with page.expect_response(lambda response: response.url.endswith("/api/schemes/unchanged")
                                  and response.request.method == "PUT") as saved:
            page.locator("#save-as").click()
        assert saved.value.ok, saved.value.text()
        document = page.request.get(f"{url}/api/schemes/unchanged").json()["scheme"]
        assert document["furniture"] is None
        assert "speaker_setup" not in document
        assert {point["receiver_id"]: point["position_m"][2] for point in document["receiver_set"]["points"]} == {
            "main": 1.2, "front": 1.2, "back": 1.2, "left": 1.2, "right": 1.2, "up": 1.3, "down": 1.1}
        assert watched.page_errors == []


def test_delayed_old_preview_cannot_replace_new_scheme_preview(browser: Browser, tmp_path: Path) -> None:
    table = {"furniture_id": "table", "kind": "desk", "material": "glass",
             "width_m": 1.65, "depth_m": 0.85, "height_m": 0.04,
             "placement": {"forward_m": 1.6, "left_m": -0.4, "bottom_height_m": 0.68, "yaw_deg": 90}}
    (tmp_path / "schemes").mkdir()
    document = cases.document(table) | {"scheme_id": "readonly"}
    (tmp_path / "schemes" / "readonly.json").write_text(Scheme.model_validate(document).model_dump_json())
    with _serve(tmp_path) as url, _open(browser, url) as watched:
        page = watched.page
        held: list[Route] = []
        page.route("**/api/input-preview", lambda route: held.append(route), times=1)
        with page.expect_request(lambda request: request.url.endswith("/api/input-preview")):
            page.locator("#room-Lx").fill("6")
        page.locator("#scheme-list").select_option("readonly")
        page.locator("#open-scheme").click()
        page.wait_for_function("document.getElementById('save-id').value === 'readonly' && !document.querySelector('main').inert")
        assert held
        with page.expect_response(lambda response: response.url.endswith("/api/input-preview")
                                  and _posted_scheme(response.request)["scheme_id"] != "readonly"):
            held[0].continue_()
        page.wait_for_load_state("networkidle")
        assert page.locator("#furniture-table-kind").is_disabled()
        assert page.locator("#furniture-table-fields input").all() == []
        assert "玻璃" in page.locator("#readonly-furniture").inner_text()
        with page.expect_response(lambda response: response.url.endswith("/api/schemes/readonly")
                                  and response.request.method == "PUT") as saved:
            page.locator("#save").click()
        assert saved.value.ok, saved.value.text()
        assert page.request.get(f"{url}/api/schemes/readonly").json()["scheme"]["furniture"] == [table]
        assert watched.page_errors == []


def test_delayed_cloud_edit_keeps_controls_and_fields_old_until_preview(browser: Browser, tmp_path: Path) -> None:
    with _serve(tmp_path) as url, _open(browser, url) as watched:
        page = watched.page
        held: list[Route] = []
        page.route("**/api/input-preview", lambda route: held.append(route), times=1)
        with page.expect_request(lambda request: request.url.endswith("/api/input-preview")):
            with page.expect_response(lambda response: response.url.endswith("/api/input-edit")) as edited:
                page.locator("#furniture-cloud").click()
        assert edited.value.ok
        assert page.locator("main").evaluate("node => node.inert")
        assert not page.locator("#furniture-cloud").is_checked()
        assert page.locator("#furniture-cloud-fields input").all() == []
        assert page.evaluate("scheme.furniture") is None
        assert held
        held[0].continue_()
        page.wait_for_function("document.getElementById('furniture-cloud').checked && !document.querySelector('main').inert")
        page.locator("#save-as-id").fill("confirmed-cloud")
        with page.expect_response(lambda response: response.url.endswith("/api/schemes/confirmed-cloud")
                                  and response.request.method == "PUT") as saved:
            page.locator("#save-as").click()
        assert saved.value.ok, saved.value.text()
        assert page.request.get(f"{url}/api/schemes/confirmed-cloud").json()["scheme"]["furniture"] == [{
            "furniture_id": "cloud", "kind": "ceiling_cloud", "material": "absorptive_cloud",
            "width_m": 1.35, "depth_m": 1.8, "height_m": 0.05,
            "placement": {"bottom_center_m": [2.1, 1.9, 2.85], "yaw_deg": 90}}]
        assert watched.page_errors == []


def test_failed_handover_refreshes_preview_for_edited_values(browser: Browser, tmp_path: Path) -> None:
    # 複查二：改了主位 y、那次預覽還在路上就勾天雲，天雲預覽又失敗。解鎖後要替目前這份補問一次，
    # 書桌換算座標從 y 3.825（主位 y 3）變成 y 4.325（主位 y 3.5），不能停在舊值等下一次輸入。
    desk = {"furniture_id": "table", "kind": "desk", "material": "wood",
            "width_m": 1.4, "depth_m": 0.75, "height_m": 0.03,
            "placement": {"forward_m": 0.825, "left_m": 0, "bottom_height_m": 0.72, "yaw_deg": 0}}
    (tmp_path / "schemes").mkdir()
    document = cases.document(desk) | {"scheme_id": "desk-scheme"}
    (tmp_path / "schemes" / "desk-scheme.json").write_text(Scheme.model_validate(document).model_dump_json())
    with _serve(tmp_path) as url, _open(browser, url) as watched:
        page = watched.page
        page.locator("#scheme-list").select_option("desk-scheme")
        page.locator("#open-scheme").click()
        page.wait_for_function("document.getElementById('save-id').value === 'desk-scheme' && !document.querySelector('main').inert")
        page.wait_for_load_state("networkidle")
        assert "y 3.825" in page.locator("#furniture-table-coordinate").inner_text()
        held: list[Route] = []

        def hold_then_fail_cloud(route: Route) -> None:
            items = cast(list[dict[str, object]], _posted_scheme(route.request).get("furniture") or [])
            if any(item["furniture_id"] == "cloud" for item in items):
                route.fulfill(status=503, json={"error": "預覽暫時讀不到，請重試"})
            elif not held:
                held.append(route)
            else:
                route.continue_()

        page.route("**/api/input-preview", hold_then_fail_cloud)
        with page.expect_request(lambda request: request.url.endswith("/api/input-preview")):
            page.locator("#receiver-main-y").fill("3.5")
        page.locator("#furniture-cloud").click()
        page.wait_for_function("!document.querySelector('main').inert && document.getElementById('messages').textContent.includes('換不過去')")
        assert held
        held[0].continue_()
        page.wait_for_function("document.getElementById('furniture-table-coordinate').textContent.includes('y 4.325')")
        assert page.locator("#receiver-main-y").input_value() == "3.5"
        assert not page.locator("#furniture-cloud").is_checked()
        assert watched.page_errors == []


def test_stale_refresh_failure_does_not_overwrite_newer_message(browser: Browser, tmp_path: Path) -> None:
    # 複查三：天雲交接失敗後補問的那一次預覽被卡住；接著勾沙發交接成功、檢查通過；
    # 這時舊補問才失敗，不准把「檢查通過」蓋成舊錯誤。
    with _serve(tmp_path) as url, _open(browser, url) as watched:
        page = watched.page
        page.wait_for_function("document.getElementById('messages').textContent === '檢查通過'")
        held: list[Route] = []
        phase = "fail-cloud"

        def route_preview(route: Route) -> None:
            nonlocal phase
            items = cast(list[dict[str, object]], _posted_scheme(route.request).get("furniture") or [])
            has_cloud = any(item["furniture_id"] == "cloud" for item in items)
            if phase == "fail-cloud" and has_cloud:
                phase = "hold-refresh"
                route.fulfill(status=503, json={"error": "預覽暫時讀不到，請重試"})
            elif phase == "hold-refresh" and not has_cloud:
                phase = "pass"
                held.append(route)
            else:
                route.continue_()

        page.route("**/api/input-preview", route_preview)
        page.locator("#furniture-cloud").click()
        page.wait_for_function("!document.querySelector('main').inert")
        # 補問那一次是失敗解鎖後才發的；等它真的被卡住（輪詢條件，不靠睡多久排先後）。
        deadline = time.monotonic() + 20
        while not held and time.monotonic() < deadline:
            page.wait_for_timeout(20)
        assert held
        with page.expect_response(lambda response: response.url.endswith("/api/input-edit")):
            page.locator("#furniture-sofa").click()
        page.wait_for_function("document.getElementById('furniture-sofa').checked"
                               " && document.getElementById('messages').textContent === '檢查通過'")
        held[0].fulfill(status=503, json={"error": "舊補問失敗"})
        # 修好的版本訊息不會變，只能等一段時間看它有沒有變；沒修好的版本很快就變成舊錯誤。
        with pytest.raises(PlaywrightTimeoutError):
            page.wait_for_function("document.getElementById('messages').textContent.includes('舊補問失敗')",
                                   timeout=2000)
        assert page.locator("#messages").inner_text() == "檢查通過"
        assert watched.page_errors == []
