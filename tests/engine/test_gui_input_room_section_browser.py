"""輸入頁第 1 段「房間和擺位」（#1265）：圖邊的長寬高、四種圖、右卡「這間房現在」逐行、兩個摺疊、高度與「同一格只在一個地方」。

方案資料都是這裡自己編的（_furniture_cases 的手定房間改尺寸與代號）；算過的結果用考卷的假計算或控制組結果。"""
from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import cast

import pytest
from playwright.sync_api import Browser, Page, Route, expect

from aosr.reporting.result import SchemeResult
from aosr.reporting.scheme import Scheme
from tests.engine import _furniture_cases as cases, _speaker_setup_cases as speaker_cases
from tests.engine._input_page_steps import (archive_scheme, open_example_scheme, open_fact_editor, open_fold, open_scheme,
                                            show_view)
from tests.engine.test_gui_archive import _bundle, _saved
from tests.engine.test_gui_archive_browser import _confirm
from tests.engine.test_gui_browser import _assert_quiet, _copy_runner, _open, _serve, browser as browser, result as result
from tests.engine.test_gui_home_page_visual import LABEL_PX_JS
from tests.engine.test_gui_input_setup_browser import _filled
from tests.engine.test_gui_speaker_setup_browser import MOUNTS
from tests.engine.test_scheme_pipeline import _run_control

ROOM = "room-facts"
BEFORE = "room-before"
RUN = "7" * 32
VIEWS = ("plan-xy", "plan-xz", "zoom-xy", "zoom-xz")
NEVER = "按頁底「照現在牆上貼的算一次」；算好後，結果就出現在這一行。"
Doc = dict[str, object]


def _room(name: str) -> Doc:
    """喇叭靠 x 起點牆（x=1、y=2 與 4），主位 (3, 3, 1.2)、周圍一個點；房間長 7、寬 6、高 3.2。"""
    document = cases.document(facing="west")
    document["scheme_id"] = name
    cast(Doc, document["scene"])["room_m"] = {"Lx": 7.0, "Ly": 6.0, "Lz": 3.2}
    return document


def _write(root: Path, document: Doc) -> Path:
    path = root / "schemes" / f"{document['scheme_id']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(Scheme.model_validate(document).model_dump_json(), encoding="utf-8")
    return path


def _ready(page: Page) -> None:
    page.wait_for_function("document.body.dataset.inputReady === 'true'")


def _opened(page: Page, name: str) -> None:
    open_scheme(page, name)
    page.wait_for_function("name => document.getElementById('save-id').value === name", arg=name)
    page.wait_for_function("!document.querySelector('main').inert && !inputBusy")


def _reopen(page: Page, name: str) -> None:
    page.reload()
    _ready(page)
    _opened(page, name)


def _save(page: Page, *, design: bool = False) -> None:
    """按「儲存」，等方案（要的話連設計設定）的 PUT 回來。"""
    path = f"/api/designs/{ROOM}" if design else f"/api/schemes/{ROOM}"
    with page.expect_response(lambda r: r.url.endswith(path) and r.request.method == "PUT") as saved:
        page.locator("#save").click()
    assert saved.value.ok, saved.value.text()
    page.wait_for_function("!document.querySelector('main').inert && !inputBusy")


def _box(page: Page, selector: str) -> dict[str, float]:
    box = page.locator(selector).bounding_box()
    assert box is not None, selector
    return cast(dict[str, float], box)


def _stored(root: Path, folder: str) -> Doc:
    return cast(Doc, json.loads((root / folder / f"{ROOM}.json").read_text(encoding="utf-8")))


def test_size_boxes_sit_on_the_plan_edges_and_a_new_size_saves(tmp_path: Path, browser: Browser) -> None:
    _write(tmp_path, _room(ROOM))
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _opened(page, ROOM)
        drawing = _box(page, "#plan-drawings")
        across, along, height = (_box(page, f"#room-{key}") for key in ("Ly", "Lx", "Lz"))
        bottom = drawing["y"] + drawing["height"]
        # 寬 Ly 在圖的左邊（平面圖直的那一軸）、長 Lx 在圖下（橫的那一軸）、高 Lz 在圖下右邊。
        assert across["x"] + across["width"] <= drawing["x"] and drawing["y"] < across["y"] < bottom
        assert along["y"] >= bottom and drawing["x"] < along["x"] < drawing["x"] + drawing["width"]
        assert height["y"] >= bottom and height["x"] > along["x"] + along["width"]
        expect(page.locator("#fact-room-text")).to_have_text("寬 6（Ly）× 深 7（Lx）× 高 3.2 公尺")
        page.locator("#room-Lx").fill("7.5")
        page.locator("#room-Lx").blur()
        expect(page.locator("#fact-room-text")).to_have_text("寬 6（Ly）× 深 7.5（Lx）× 高 3.2 公尺")
        _save(page)
        assert cast(Doc, cast(Doc, _stored(tmp_path, "schemes")["scene"])["room_m"])["Lx"] == 7.5
        _reopen(page, ROOM)
        expect(page.locator("#room-Lx")).to_have_value("7.5")
        expect(page.locator("#fact-room-text")).to_have_text("寬 6（Ly）× 深 7.5（Lx）× 高 3.2 公尺")
        _assert_quiet(watched)


@pytest.mark.parametrize("view", VIEWS)
def test_each_view_button_shows_only_its_drawing(tmp_path: Path, browser: Browser, view: str) -> None:
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        open_example_scheme(page, base, tmp_path)
        page.locator("#zoom-xz circle").first.wait_for(state="attached")
        expect(page.locator("#view-plan-xy")).to_have_attribute("aria-pressed", "true")
        page.locator(f"#view-{view}").click()
        for other in VIEWS:
            pressed = "true" if other == view else "false"
            expect(page.locator(f"#view-{other}")).to_have_attribute("aria-pressed", pressed)
            if other == view:
                expect(page.locator(f"#{other}")).to_be_visible()
            else:
                expect(page.locator(f"#{other}")).to_be_hidden()
        # 露出來的那一張字照樣量得到、不小於 12 像素；圖上的點拿得到焦點，按 Enter 寫出那一點的明細。
        sizes = cast(list[float], page.locator(f"#{view}").evaluate(LABEL_PX_JS))
        assert sizes and all(size >= 12 for size in sizes), sizes
        marker = page.locator(f"#{view} g[data-keys]").first
        marker.focus()
        expect(marker).to_be_focused()
        page.keyboard.press("Enter")
        expect(page.locator("#plan-detail")).not_to_be_empty()
        _assert_quiet(watched)


SIZE_LABELS = {"Lx": "長 Lx（公尺）", "Ly": "寬 Ly（公尺）", "Lz": "高 Lz（公尺）"}


def _on_plan_edges(page: Page) -> None:
    drawing = _box(page, "#plan-xy")
    across, along, height = (_box(page, f"label:has(#room-{key})") for key in ("Ly", "Lx", "Lz"))
    bottom = drawing["y"] + drawing["height"]
    assert across["x"] + across["width"] <= drawing["x"] and drawing["y"] < across["y"] < bottom
    assert along["y"] >= bottom and drawing["x"] < along["x"] < drawing["x"] + drawing["width"]
    assert height["y"] >= bottom and height["x"] > along["x"] + along["width"]


def test_size_boxes_move_under_the_other_drawings_and_back_to_the_plan_edges(tmp_path: Path, browser: Browser) -> None:
    """側面、聆聽區平面放大、聆聽區側面放大：同一組長寬高三格（同一批元素，不另做一份）排成圖底下一列，
    看得到、不壓在圖上、各只有一個；改了照樣連動。切回平面圖，三格回到圖邊。"""
    _write(tmp_path, _room(ROOM))
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _opened(page, ROOM)
        _on_plan_edges(page)
        boxes = {key: page.locator(f"#room-{key}").element_handle() for key in SIZE_LABELS}
        for view in VIEWS[1:]:
            show_view(page, view)
            drawing = _box(page, f"#{view}")
            bottom = drawing["y"] + drawing["height"]
            rows = []
            for key, word in SIZE_LABELS.items():
                expect(page.locator(f"#room-{key}")).to_have_count(1)
                expect(page.get_by_label(word)).to_have_count(1)
                expect(page.locator(f"#room-{key}")).to_be_visible()
                assert boxes[key].evaluate(f"node => node === document.getElementById('room-{key}')"), (view, key)
                label = _box(page, f"label:has(#room-{key})")
                # 在圖底下、橫向不超出圖的左右兩邊：不壓在圖上。
                assert label["y"] >= bottom, (view, key, label, drawing)
                assert drawing["x"] - 1 <= label["x"] and label["x"] + label["width"] <= drawing["x"] + drawing["width"] + 1, (view, key)
                rows.append(label["y"])
            assert max(rows) - min(rows) < 2, (view, rows)
        page.locator("#room-Lx").fill("7.5")
        page.locator("#room-Lx").blur()
        expect(page.locator("#fact-room-text")).to_have_text("寬 6（Ly）× 深 7.5（Lx）× 高 3.2 公尺")
        show_view(page, "plan-xy")
        _on_plan_edges(page)
        _assert_quiet(watched)


def test_card_lines_follow_the_coordinates_and_the_posture(tmp_path: Path, browser: Browser) -> None:
    _write(tmp_path, _room(ROOM))
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _opened(page, ROOM)
        expect(page.locator("#fact-speakers-text")).to_have_text("兩支在前牆前 1 公尺，相距 2 公尺，高 1.2 公尺")
        expect(page.locator("#fact-speakers-kind")).to_have_text("沒設喇叭類型與擺法")
        expect(page.locator("#fact-primary-text")).to_have_text("離前牆 3 公尺，在兩支喇叭的中線上，耳朵高 1.2 公尺")
        expect(page.locator("#fact-furniture-text")).to_have_text("沒有家具")
        # 周圍點幾個照資料數（這份只有一個）。
        expect(page.locator("#coordinates-summary")).to_have_text("座標數字（喇叭、主位，和主位周圍一起算聲音的 1 個點；x、y、z）")
        open_fold(page, "coordinates-fold")
        for side in ("left", "right"):
            page.locator(f"#speaker-{side}-x").fill("0.75")
        expect(page.locator("#fact-speakers-text")).to_have_text("兩支在前牆前 0.75 公尺，相距 2 公尺，高 1.2 公尺")
        # 面向 x 起點牆時左手邊是 y 變小那一邊：y 往大挪 0.25 是偏右。
        page.locator("#receiver-main-y").fill("3.25")
        expect(page.locator("#fact-primary-text")).to_have_text("離前牆 3 公尺，在兩支喇叭的中線右邊 0.25 公尺，耳朵高 1.2 公尺")
        for name in ("working", "listening"):
            expect(page.locator(f"#shortcut-{name}")).to_have_attribute("aria-pressed", "false")
        page.locator("#shortcut-listening").click()
        expect(page.locator("#shortcut-listening")).to_have_attribute("aria-pressed", "true")
        expect(page.locator("#fact-furniture-text")).to_have_text("沙發、茶几")
        expect(page.locator("#fact-primary-text")).to_contain_text("耳朵高 1.05 公尺")
        expect(page.locator("#furniture-editor #furniture-notice")).to_contain_text("這份方案有 2 件家具：按「改家具」修改")
        page.locator("#shortcut-working").click()
        expect(page.locator("#shortcut-working")).to_have_attribute("aria-pressed", "true")
        expect(page.locator("#shortcut-listening")).to_have_attribute("aria-pressed", "false")
        expect(page.locator("#fact-furniture-text")).to_have_text("書桌")
        _assert_quiet(watched)


@pytest.mark.parametrize("button,area,first", [
    ("edit-depth", "depth-editor", "depth-room"), ("edit-speakers", "speaker-editor", "use-speaker-setup"),
    ("edit-furniture", "furniture-editor", "furniture-sofa"), ("edit-original", "original-editor", "original-scheme"),
])
def test_each_change_button_opens_its_one_editor_with_focus_inside(tmp_path: Path, browser: Browser,
                                                                    button: str, area: str, first: str) -> None:
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        open_example_scheme(page, base, tmp_path)
        expect(page.locator(f"#{area}")).to_be_hidden()
        page.locator(f"#{button}").click()
        expect(page.locator(f"#{button}")).to_have_attribute("aria-expanded", "true")
        expect(page.locator(f"#{area}")).to_be_visible()
        expect(page.locator(f"#{first}")).to_be_focused()
        page.locator(f"#{button}").click()
        expect(page.locator(f"#{button}")).to_have_attribute("aria-expanded", "false")
        expect(page.locator(f"#{area}")).to_be_hidden()
        _assert_quiet(watched)


def test_speaker_setup_line_moves_from_none_to_the_set_kind(tmp_path: Path, browser: Browser) -> None:
    _write(tmp_path, _room(ROOM))
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _opened(page, ROOM)
        expect(page.locator("#fact-speakers-kind")).to_have_text("沒設喇叭類型與擺法")
        # 長提示住在「改喇叭與擺法」打開的那一塊裡：先打開再看它藏著（收著時看不到不算數）。
        open_fact_editor(page, "edit-speakers")
        expect(page.locator("#speaker-setup-notice")).to_be_hidden()
        page.locator("#use-speaker-setup").click()
        page.wait_for_function("!document.querySelector('main').inert && !inputBusy && scheme.speaker_setup")
        expect(page.locator("#fact-speakers-kind")).to_be_visible()
        expect(page.locator("#fact-speakers-kind")).to_have_text(MOUNTS[0][2])
        expect(page.locator("#speaker-editor #speaker-setup-notice")).to_be_visible()
        expect(page.locator("#speaker-setup-notice")).to_have_text(f"這份方案設了喇叭：{MOUNTS[0][2]}；按「改喇叭與擺法」修改，平面圖與側面圖照方案檔畫箱體。")
        expect(page.locator("#fact-speakers .v #speaker-setup-notice")).to_have_count(0)
        _assert_quiet(watched)


def test_speaker_line_is_the_short_name_for_each_mount_and_the_long_notice_waits_in_the_editor(
        tmp_path: Path, browser: Browser) -> None:
    """右卡喇叭那一行只寫短句（名字跟長提示同一份名稱表）；長提示只在「改喇叭與擺法」那一塊、收著就看不到。"""
    for mount, representative, title in MOUNTS:
        document = speaker_cases.document(mount)
        cast(Doc, document["speaker_setup"])["representative"] = representative
        _write(tmp_path, document | {"scheme_id": f"mount-{mount}"})
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _ready(page)
        for mount, _, title in MOUNTS:
            _opened(page, f"mount-{mount}")
            expect(page.locator("#fact-speakers-kind")).to_have_text(title)
            expect(page.locator("#speaker-setup-notice")).to_be_hidden()
            open_fact_editor(page, "edit-speakers")
            expect(page.locator("#speaker-setup-notice")).to_have_text(f"這份方案設了喇叭：{title}；按「改喇叭與擺法」修改，平面圖與側面圖照方案檔畫箱體。")
            page.locator("#edit-speakers").click()
            expect(page.locator("#speaker-editor")).to_be_hidden()
        _assert_quiet(watched)


def test_speaker_line_falls_back_to_one_phrase_when_the_names_did_not_load(tmp_path: Path, browser: Browser) -> None:
    """名稱表載不到：短句整句換成備用句，不逐格代入英文代號（同長提示那一句的做法）。"""
    _write(tmp_path, speaker_cases.document() | {"scheme_id": ROOM})
    with _serve(tmp_path) as base, _open(browser, "about:blank", viewport_width=1440) as watched:
        page = watched.page
        page.route("**/api/labels", lambda route: route.fulfill(
            status=500, content_type="application/json", body='{"error": "名稱表壞了"}'))
        page.goto(base, wait_until="networkidle")
        _opened(page, ROOM)
        expect(page.locator("#fact-speakers-kind")).to_have_text("設了喇叭類型與擺法（中文名沒載到）")
        # 故意讓名稱表回 500，主控台只准出現那一筆；頁面不准有錯。
        assert all("500" in text for text in watched.console_errors), watched.console_errors
        assert watched.page_errors == []


def test_total_depth_limit_saves_reopens_and_is_what_the_zone_page_reads(tmp_path: Path, browser: Browser) -> None:
    _write(tmp_path, _room(ROOM))
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _opened(page, ROOM)
        expect(page.locator("#fact-depth-text")).to_contain_text("房間預設沒設上限")
        open_fact_editor(page, "edit-depth")
        depth = page.locator("#depth-room")
        # 不合理的值不收：卡上旁邊照講，格子畫回原值。
        depth.fill("-1")
        depth.blur()
        expect(page.locator("#depth-room-note")).to_contain_text("大於 0")
        expect(depth).to_have_value("")
        depth.fill("15")  # #1265 單元 3 起填公分（存的照舊是公尺）
        depth.blur()
        expect(page.locator("#fact-depth-text")).to_have_text(
            "貼的東西總厚度最多 15 公分（從原牆表面往房間裡量，程式挑材料也不會超過）；這是房間預設，某一面要不同在下面立面圖那一面另設")
        _save(page, design=True)
        assert _stored(tmp_path, "designs")["room_max_depth_m"] == 0.15
        _reopen(page, ROOM)
        expect(page.locator("#fact-depth-text")).to_contain_text("最多 15 公分")
        open_fact_editor(page, "edit-depth")
        expect(page.locator("#depth-room")).to_have_value("15")
        # 「這間房貼什麼」那一頁不另寫房間上限：它跟立面圖的限制判決用的是開檔時從伺服器讀回的那一份設計設定。
        # 打開那一頁、送去判限制的那一份帶的就是 0.15；伺服器存的也是。
        page.locator("#zone-page-open").click()  # #1265 單元 3 起只剩第 2 段右欄那一顆
        expect(page.locator("#zone-page")).to_be_visible()
        with page.expect_request("**/api/zone-limits") as sent:
            page.evaluate("refreshZoneLimits()")
        assert cast(Doc, cast(Doc, sent.value.post_data_json)["design"])["room_max_depth_m"] == 0.15
        served = page.request.get(f"{base}/api/designs/{ROOM}").json()
        assert served["design"]["room_max_depth_m"] == 0.15
        _assert_quiet(watched)


def test_enter_in_the_total_depth_box_keeps_focus_and_the_verdict_draws(tmp_path: Path, browser: Browser) -> None:
    """總厚度上限打完按 Enter：值確定、限制判決照樣畫上去，焦點留在那一格（那一格是乾淨的，不算打到一半）。"""
    _write(tmp_path, _room(ROOM))
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _opened(page, ROOM)
        open_fact_editor(page, "edit-depth")
        depth = page.locator("#depth-room")
        before = cast(int, page.evaluate("zoneLimitsGeneration"))
        depth.fill("30")  # #1265 單元 3 起填公分
        depth.press("Enter")
        page.wait_for_function("before => zoneLimitsGeneration > before && zoneLimitsShown === zoneLimitsGeneration",
                               arg=before)
        expect(depth).to_be_focused()
        assert page.evaluate("zoneDesign.room_max_depth_m") == 0.3
        expect(page.locator("#fact-depth-text")).to_contain_text("最多 30 公分")
        _assert_quiet(watched)


def test_a_height_warning_from_the_mount_opens_the_coordinates_fold(tmp_path: Path, browser: Browser) -> None:
    """擺法在「改喇叭與擺法」改；推出的高度跟 z 不同時，提醒寫在「座標數字」裡那一支的 z 旁邊：摺疊收著就自己打開。"""
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        open_example_scheme(page, base, tmp_path)
        fold = page.locator("#coordinates-fold")
        _filled(page, "#furniture-table-kind", "desk")
        _filled(page, "#use-speaker-setup", True)
        # 放腳架時高度照 z，沒有不同：摺疊照樣收著。
        assert fold.evaluate("fold => fold.open") is False
        _filled(page, "#speaker-mount", "desk")
        expect(fold).to_have_attribute("open", "")
        expect(page.locator("#height-left")).to_be_visible()
        expect(page.locator("#height-left")).to_contain_text("跟擺法推出值不同")
        # 高度跟擺法不同時伺服器照設計回 422（同 test_gui_input_setup_browser 那一題），JS 自己不准拋錯。
        assert watched.page_errors == []
        assert all("422" in line for line in watched.console_errors), watched.console_errors


def test_original_scheme_can_be_chosen_saved_and_cleared(tmp_path: Path, browser: Browser) -> None:
    _write(tmp_path, _room(ROOM))
    _write(tmp_path, _room(BEFORE))
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _opened(page, ROOM)
        text = page.locator("#fact-original-text")
        expect(text).to_have_text("每一面都只有原本的牆（沒指定原方案）")
        open_fact_editor(page, "edit-original")
        expect(page.locator("#original-clear")).to_be_disabled()
        page.locator("#original-scheme").select_option(BEFORE)
        expect(text).to_have_text(f"照「{BEFORE}」那一份方案")
        _save(page, design=True)
        assert _stored(tmp_path, "designs")["original_scheme_id"] == BEFORE
        _reopen(page, ROOM)
        expect(text).to_have_text(f"照「{BEFORE}」那一份方案")
        open_fact_editor(page, "edit-original")
        expect(page.locator("#original-scheme")).to_have_value(BEFORE)
        page.locator("#original-clear").click()
        expect(text).to_have_text("每一面都只有原本的牆（沒指定原方案）")
        expect(page.locator("#original-scheme")).to_have_value("")
        _save(page, design=True)
        assert _stored(tmp_path, "designs")["original_scheme_id"] is None
        _assert_quiet(watched)


def test_both_folds_start_closed_and_what_is_changed_inside_saves(tmp_path: Path, browser: Browser) -> None:
    _write(tmp_path, _room(ROOM))
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _opened(page, ROOM)
        for fold in ("coordinates-fold", "advanced-fold"):
            assert page.locator(f"#{fold}").evaluate("fold => fold.open") is False, fold
        expect(page.locator("#speaker-right-z")).to_be_hidden()
        expect(page.locator("#source-model")).to_be_hidden()
        open_fold(page, "coordinates-fold")
        page.locator("#speaker-right-z").fill("1.1")
        expect(page.locator("#fact-speakers-text")).to_have_text(
            "兩支在前牆前 1 公尺，相距 2 公尺，高 左聲道喇叭 1.2、右聲道喇叭 1.1 公尺")
        open_fold(page, "advanced-fold")
        page.locator("#source-model").select_option("omnidirectional")
        for line in ("#feature-note", "#rho-c"):
            expect(page.locator(line)).not_to_be_empty()
        _save(page)
        stored = _stored(tmp_path, "schemes")
        assert stored["source_model"] == "omnidirectional"
        assert cast(dict[str, Doc], stored["speakers"])["right"]["z"] == 1.1
        _reopen(page, ROOM)
        for fold in ("coordinates-fold", "advanced-fold"):
            assert page.locator(f"#{fold}").evaluate("fold => fold.open") is False, fold
        expect(page.locator("#speaker-right-z")).to_have_value("1.1")
        expect(page.locator("#source-model")).to_have_value("omnidirectional")
        _assert_quiet(watched)


def test_section_one_fits_in_700_pixels_at_1440_by_900(tmp_path: Path, browser: Browser) -> None:
    """開頁預設（摺疊收著）不超過 700 像素；示意圖那個狀態（沙發、茶几、設了喇叭類型）也一樣。
    那個狀態右卡只有短句，兩段長提示收在「改喇叭與擺法」「改家具」各自那一塊裡（收著）。"""
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        page.set_viewport_size({"width": 1440, "height": 900})
        open_example_scheme(page, base, tmp_path)
        assert _box(page, "#scheme-section")["height"] <= 700
        page.locator("#shortcut-listening").click()
        page.wait_for_function("!document.querySelector('main').inert && !inputBusy && scheme.furniture?.length === 2")
        open_fact_editor(page, "edit-speakers")
        page.locator("#use-speaker-setup").click()
        page.wait_for_function("!document.querySelector('main').inert && !inputBusy && scheme.speaker_setup")
        page.locator("#edit-speakers").click()
        expect(page.locator("#fact-speakers-kind")).to_have_text(MOUNTS[0][2])
        expect(page.locator("#fact-furniture-text")).to_have_text("沙發、茶几")
        expect(page.locator("#speaker-setup-notice")).to_be_hidden()
        expect(page.locator("#furniture-notice")).to_be_hidden()
        assert _box(page, "#scheme-section")["height"] <= 700
        _assert_quiet(watched)


# 搬進第 1 段的格子：每一格只住一個地方（以前散在四段的同一批格子搬走就刪）。
MOVED = ("room-Lx", "room-Ly", "room-Lz", "source-model", "use-scattering", "scatter-floor", "speaker-left-x",
         "receiver-main-z", "shortcut-working", "furniture-sofa", "furniture-table-kind", "use-speaker-setup",
         "depth-room", "original-scheme")
LABEL_WORDS = ("長 Lx（公尺）", "寬 Ly（公尺）", "高 Lz（公尺）", "聲源模型", "填入六面散射係數", "沙發", "前方桌面", "天雲",
               "設定喇叭箱體與擺法", "貼的東西總厚度最多", "改造前用哪一份方案")


def test_every_moved_box_lives_in_one_place(tmp_path: Path, browser: Browser) -> None:
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        open_example_scheme(page, base, tmp_path)
        for fold in ("coordinates-fold", "advanced-fold"):
            open_fold(page, fold)
        for button in ("edit-depth", "edit-speakers", "edit-furniture", "edit-original"):
            open_fact_editor(page, button)
        ids = cast(list[str], page.evaluate("() => [...document.querySelectorAll('[id]')].map(node => node.id)"))
        assert sorted({name for name in ids if ids.count(name) > 1}) == []
        for name in MOVED:
            assert page.locator(f"#scheme-section #{name}").count() == 1, name
        labels = cast(list[str], page.evaluate(
            "() => [...document.querySelectorAll('label')].filter(label => label.control).map(label => label.textContent.trim())"))
        for word in LABEL_WORDS:
            assert sum(text.startswith(word) for text in labels) == 1, (word, [text for text in labels if word in text])
        # 舊的叫法與舊位置不留：立面圖那一區只剩每一面那一格深度上限。
        assert page.get_by_text("深度上限：房間預設").count() == 0
        assert page.locator("#elevation-panel #depth-room, #elevation-panel #original-scheme").count() == 0
        _assert_quiet(watched)


def _computed(result: SchemeResult) -> SchemeResult:
    return _run_control(result.scheme.model_copy(update={"scheme_id": ROOM}))


def test_result_line_names_when_this_scheme_was_computed(tmp_path: Path, browser: Browser, result: SchemeResult) -> None:
    room = _computed(result)
    _saved(tmp_path, room)
    _bundle(tmp_path, room, RUN)
    stamp = datetime(2026, 11, 20, 14, 5).timestamp()
    os.utime(tmp_path / "results" / f"{RUN}.json", (stamp, stamp))
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        open_example_scheme(page, base, tmp_path)
        # 開著的範例沒算過。
        expect(page.locator("#room-result-title")).to_have_text("還沒算過")
        expect(page.locator("#room-result-text")).to_have_text(NEVER)
        expect(page.locator("#room-result-link")).to_be_hidden()
        _opened(page, ROOM)
        expect(page.locator("#room-result-title")).to_have_text("算好的結果")
        expect(page.locator("#room-result-text")).to_have_text("2026-11-20 14:05")
        link = page.locator("#room-result-link")
        expect(link).to_have_text("看結果 ▸")
        expect(link).to_have_attribute("href", f"/results/{RUN}")
        link.click()
        page.wait_for_url(f"{base}/results/{RUN}")
        expect(page.locator("#page-title")).to_contain_text(ROOM)
        _assert_quiet(watched)


def test_result_line_says_the_result_was_computed_before_an_edit(tmp_path: Path, browser: Browser,
                                                                 result: SchemeResult) -> None:
    """算過的那一份改了一格：頂條寫「有改動還沒存」的同一份狀態，結果那一行的字後面註明是修改前的設定算的。"""
    room = _computed(result)
    _saved(tmp_path, room)
    _bundle(tmp_path, room, RUN)
    stamp = datetime(2026, 11, 20, 14, 5).timestamp()
    os.utime(tmp_path / "results" / f"{RUN}.json", (stamp, stamp))
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _ready(page)
        _opened(page, ROOM)
        expect(page.locator("#room-result-text")).to_have_text("2026-11-20 14:05")
        height = page.locator("#room-Lz")
        assert height.input_value() != "3.3"
        height.fill("3.3")
        expect(page.locator("#save-state")).to_contain_text("有改動還沒存")
        expect(page.locator("#room-result-title")).to_have_text("算好的結果")
        expect(page.locator("#room-result-text")).to_have_text("2026-11-20 14:05（修改前的設定算的）")
        expect(page.locator("#room-result-link")).to_have_attribute("href", f"/results/{RUN}")
        _assert_quiet(watched)


def test_result_line_follows_a_running_then_finished_calculation(tmp_path: Path, browser: Browser,
                                                                 result: SchemeResult) -> None:
    room = _computed(result)
    _saved(tmp_path, room)
    with _serve(tmp_path, _copy_runner(tmp_path, room, wait=True)) as base, \
            _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _opened(page, ROOM)
        page.locator("#calculate").click()
        expect(page.locator("#room-result-title")).to_have_text("正在算")
        expect(page.locator("#room-result-text")).to_have_text(re.compile(r"^已跑 \d+ 秒；算完這裡會出現「看結果 ▸」$"))
        expect(page.locator("#next-title")).to_have_text(re.compile(r"^正在算這一份：已跑 \d+ 秒$"))
        (tmp_path / "release").write_text("", encoding="utf-8")
        expect(page.locator("#room-result-title")).to_have_text("算好的結果", timeout=20_000)
        expect(page.locator("#room-result-link")).to_be_visible()
        expect(page.locator("#room-result-text")).not_to_contain_text("修改前的設定算的")
        # 算完再改一格：頁底那句「設定已修改，這份結果是修改前算的」亮起來，結果那一行也註明（同一份狀態）。
        page.locator("#room-Lz").fill("3.3")
        expect(page.locator("#result-stale")).to_be_visible()
        expect(page.locator("#room-result-text")).to_contain_text("（修改前的設定算的）")
        _assert_quiet(watched)


def test_result_line_says_why_a_calculation_did_not_finish(tmp_path: Path, browser: Browser,
                                                         result: SchemeResult) -> None:
    _saved(tmp_path, _computed(result))
    runner = (sys.executable, "-c", "import sys; sys.exit('考卷房的計算停在一半')")
    with _serve(tmp_path, runner) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _opened(page, ROOM)
        page.locator("#calculate").click()
        expect(page.locator("#next-title")).to_have_text("沒算完：考卷房的計算停在一半")
        expect(page.locator("#room-result-title")).to_have_text("沒算完")
        expect(page.locator("#room-result-text")).to_have_text("考卷房的計算停在一半；看頁底的出錯紀錄")
        expect(page.locator("#room-result-link")).to_be_hidden()
        _assert_quiet(watched)


# ---- #1265 單元 2 審查修補（build-brief-4b-u2fix.md）：每一題先在修之前跑紅，再修。 ----


def test_a_refused_total_depth_goes_back_to_the_last_confirmed_value(tmp_path: Path, browser: Browser) -> None:
    """審查發現 1：確認過 15 公分（按 Enter）之後同一格再填 -1 按 Enter，格子退回 15（最後一次確認的值），不是建格當時的
    空白；焦點還在那一格，但不算打到一半（不擋後面的限制判決重畫）；離開那一格也還是 15。"""
    _write(tmp_path, _room(ROOM))
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _opened(page, ROOM)
        open_fact_editor(page, "edit-depth")
        depth = page.locator("#depth-room")
        before = cast(int, page.evaluate("zoneLimitsGeneration"))
        depth.fill("15")
        depth.press("Enter")
        page.wait_for_function("before => zoneLimitsGeneration > before && zoneLimitsShown === zoneLimitsGeneration",
                               arg=before)
        depth.fill("-1")
        depth.press("Enter")
        expect(depth).to_be_focused()
        expect(depth).to_have_value("15")
        assert page.evaluate("zoneDesign.room_max_depth_m") == 0.15
        assert page.evaluate("Boolean(typingInElevation())") is False
        depth.blur()
        expect(depth).to_have_value("15")
        _assert_quiet(watched)


def test_a_refused_total_depth_note_stays_through_redraws_until_a_good_value(tmp_path: Path, browser: Browser) -> None:
    """審查發現 2：填錯的那一句寫在那一格底下，之後任何重畫（計算中每秒輪詢走的 renderInputBars）都不洗掉；
    改成合法值才消失。收不收只在 depthInput 判一次（room_facts.js 不另寫一份判斷）。"""
    _write(tmp_path, _room(ROOM))
    refused = "總厚度上限要填大於 0 的數字（公分，最少 0.1）；空白表示房間不設上限"
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _opened(page, ROOM)
        open_fact_editor(page, "edit-depth")
        depth, note = page.locator("#depth-room"), page.locator("#depth-room-note")
        depth.fill("-1")
        depth.blur()
        expect(note).to_have_text(refused)
        page.evaluate("renderInputBars()")
        expect(note).to_have_text(refused)
        page.evaluate("renderElevation()")  # 第 2 段整個重畫（換牆、判決回來都走這裡）
        expect(note).to_have_text(refused)
        depth = page.locator("#depth-room")
        depth.fill("12")
        depth.blur()
        expect(note).to_have_text("")
        assert page.evaluate("zoneDesign.room_max_depth_m") == 0.12
        _assert_quiet(watched)


def test_result_line_does_not_say_twice_that_the_result_is_older_than_the_scheme(tmp_path: Path, browser: Browser,
                                                                                  result: SchemeResult) -> None:
    """審查設計洞 1 後半：存著的方案檔在算完之後改過，結果列自己就寫「改之前的方案算的」；表單再改一格，
    結果那一行不再加「（修改前的設定算的）」——同一件事不講兩次。"""
    room = _computed(result)
    _saved(tmp_path, room)
    _bundle(tmp_path, room, RUN)
    stamp = datetime(2026, 11, 20, 14, 5).timestamp()
    os.utime(tmp_path / "results" / f"{RUN}.json", (stamp, stamp))
    path = tmp_path / "schemes" / f"{ROOM}.json"
    document = cast(Doc, json.loads(path.read_text(encoding="utf-8")))
    size = cast(dict[str, float], cast(Doc, document["scene"])["room_m"])
    size["Lz"] = round(size["Lz"] + 0.25, 2)
    path.write_text(Scheme.model_validate(document).model_dump_json(), encoding="utf-8")
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _ready(page)
        _opened(page, ROOM)
        text = page.locator("#room-result-text")
        expect(text).to_have_text("2026-11-20 14:05，改之前的方案算的")
        height = page.locator("#room-Lz")
        height.fill(str(round(float(height.input_value()) + 0.1, 2)))
        expect(page.locator("#save-state")).to_contain_text("有改動還沒存")
        expect(page.locator("#room-result-title")).to_have_text("算好的結果")
        expect(text).to_have_text("2026-11-20 14:05，改之前的方案算的")
        _assert_quiet(watched)


def test_a_late_scheme_list_reaches_the_original_choice_once_focus_leaves(tmp_path: Path, browser: Browser) -> None:
    """審查設計洞 3：方案清單晚到時焦點正在「改造前」那個選單上：那時不換（換掉會搶走正在用的選單），焦點離開那一塊
    就補畫一次，選單列的是新清單。"""
    _write(tmp_path, _room(ROOM))
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _opened(page, ROOM)
        open_fact_editor(page, "edit-original")
        select = page.locator("#original-scheme")
        expect(select.locator(f'option[value="{BEFORE}"]')).to_have_count(0)
        held: list[Route] = []
        page.route("**/api/schemes", lambda route: held.append(route))
        _write(tmp_path, _room(BEFORE))  # 別處另存了一份：下一次讀清單才有
        select.focus()
        page.evaluate("void loadSchemeList()")
        while not held:
            page.wait_for_timeout(20)
        held[0].continue_()
        page.unroute("**/api/schemes")
        page.wait_for_function("name => schemeNames.includes(name)", arg=BEFORE)
        expect(select).to_be_focused()
        expect(select.locator(f'option[value="{BEFORE}"]')).to_have_count(0)  # 焦點還在：先不換
        page.locator("#fact-room-text").click()
        expect(page.locator(f'#original-scheme option[value="{BEFORE}"]')).to_have_count(1)
        _assert_quiet(watched)


def test_original_pointing_at_an_archived_scheme_says_it_is_not_in_the_list(tmp_path: Path, browser: Browser) -> None:
    """審查第 6 條：改造前指到一份已經封存的方案，那一行照講是哪一份，再註明「不在方案清單裡」。"""
    _write(tmp_path, _room(ROOM))
    _write(tmp_path, _room(BEFORE))
    design = tmp_path / "designs" / f"{ROOM}.json"
    design.parent.mkdir(exist_ok=True)
    design.write_text(json.dumps({"schema_version": "aosr.zone_design.v1", "scheme_id": ROOM,
                                  "original_scheme_id": BEFORE}), encoding="utf-8")
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _ready(page)
        _confirm(page)  # 封存前那一句「確定封存嗎？」按確定
        archive_scheme(page, BEFORE)
        page.wait_for_function("name => !schemeNames.includes(name)", arg=BEFORE)
        _opened(page, ROOM)
        expect(page.locator("#fact-original-text")).to_have_text(f"照「{BEFORE}」那一份方案（不在方案清單裡）")
        _assert_quiet(watched)


def test_a_redraw_before_any_scheme_has_loaded_puts_no_editing_box_on_the_card(tmp_path: Path, browser: Browser) -> None:
    """審查第 6 條留下的那一行（room_facts.js :: renderRoomEditors 的「還沒有方案就不擺編輯格」）：開頁讀不到範例時整頁
    一直沒有方案，之後的重畫（renderInputBars：鎖頁、輪詢都叫它）走得到這一行；那時總厚度上限與改造前的編輯格不擺
    （沒有方案可改），頁面不出錯。開頁讀得到範例時三種先後都走不到（回報第 6 條附量法）。"""
    with _serve(tmp_path) as base:
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        errors: list[str] = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        try:
            page.route("**/api/example", lambda route: route.abort())
            page.goto(f"{base}/", wait_until="networkidle")
            assert page.evaluate("scheme === undefined")
            page.evaluate("renderInputBars()")
            expect(page.locator("#depth-room")).to_have_count(0)
            expect(page.locator("#original-scheme")).to_have_count(0)
            assert errors == []
        finally:
            page.close()
