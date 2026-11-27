"""真瀏覽器考超過限制的區塊標紅（#1111，#1082 第 1-4 段）：立面圖下填深度上限（房間預設、這面牆、每一塊三層），
固定區貼得比上限深就在立面圖與表上標紅、寫出哪一層擋的，「拿去搜尋」停用；改好就解除；交給搜尋的區寫出生效的
上限；存檔再打開三層都在；再切同一面，區的上限跟著做法走。房間一律用 109 號房的長寬高。"""
import json
import os
import re
from pathlib import Path
from typing import cast

from playwright.sync_api import Browser, Page, Route, expect

from tests.engine._input_page_steps import limits_all_clear, reveal
from tests.engine._zone_panel_steps import choose_fixed, choose_mode, edit_candidates, edit_depth
from tests.engine._construction_gui_server import real_server
from tests.engine._gui_cache import gui_startup_identity_memo as gui_startup_identity_memo
from tests.engine._surface_zone_cases import ROOM_109, room109_document
from tests.engine.test_gui_browser import _open, _serve, browser as browser
from tests.engine.test_gui_input_clear_undo_browser import _release
from tests.engine.test_gui_zone_elevation_browser import _drag, _open_scheme, _write, _zones
from tests.engine.test_gui_zone_modes_browser import _modes, _write_room109_design, _region, _save

LY, LZ = ROOM_109["Ly"], ROOM_109["Lz"]
PRESET = "glass_wool_48k_50mm"          # 總深 0.05 m
DEEP = "glass_wool_48k_50mm_gap100"     # 總深 0.15 m
RED = re.compile(r"\bzone-red\b")


def _settled(page: Page) -> None:
    """最新一次限制判決已經畫上畫面（不是「某一個回覆到了」：前一個改動的回覆可能還在路上）。"""
    page.wait_for_function("zoneLimitsGeneration > 0 && zoneLimitsShown === zoneLimitsGeneration")


def _depth(page: Page, selector: str, value: str) -> None:
    """填一格深度上限（離開那一格才算確定），等這一次的判決畫上去。#1265 單元 3 起兩格都填公分（存的照舊是公尺）。"""
    reveal(page, selector)  # #1265 起房間那一格收在第 1 段「改總厚度上限」裡，單元 3 起這一面那一格收在「這一面另設」裡，先打開
    before = cast(int, page.evaluate("zoneLimitsGeneration"))
    page.locator(selector).fill(value)
    page.locator(selector).blur()
    assert cast(int, page.evaluate("zoneLimitsGeneration")) > before
    _settled(page)


def _red(page: Page, zone: str) -> None:
    expect(page.locator(f"#elevation [data-zone-id={zone}]")).to_have_class(RED)
    expect(page.locator("#zone-to-search")).to_be_disabled()


def _clear(page: Page, zone: str) -> None:
    expect(page.locator(f"#elevation [data-zone-id={zone}]")).not_to_have_class(RED)
    limits_all_clear(page)
    expect(page.locator("#zone-to-search")).to_be_enabled()


def _no_limit_lines(page: Page, *zones: str) -> None:
    """不紅的區底下不寫限制（#1265 單元 3 之二收尾 G）。"""
    for zone in zones:
        expect(page.locator(f"[id^=limit-x0-{zone}-], #wall-zone-x0-{zone} .limit-lines")).to_have_count(0)


def _red_zone_lines(page: Page, zone: str, *words: str) -> None:
    """紅的那一區底下寫出它的每一條限制（#limit-x0-z1-0：舊做法表那一區底下那幾行的 id，#1265 單元 3 之二收尾 G 搬到第 2 段
    右欄那一區底下），字照判決原文、整頁只一份。"""
    first = page.locator(f"#wall-zone-x0-{zone} #limit-x0-{zone}-0")
    for word in words:
        expect(first).to_contain_text(word)
    expect(first).to_have_class("limit-red")
    expect(page.locator(f"#limit-x0-{zone}-0")).to_have_count(1)
    served = cast(list[str], page.evaluate(f"zoneLimits.zones.x0.{zone}.map(line => line.text)"))
    expect(page.locator(f"#wall-zone-x0-{zone} .limit-lines p")).to_have_text(served)


def test_too_deep_fixed_zone_goes_red_names_the_level_and_blocks_the_search(tmp_path: Path, browser: Browser) -> None:
    """用正式網頁入口起（雙精度開著、可信範圍真的判）：截圖就是產品真正的樣子。"""
    _write_room109_design(tmp_path, {"left_right_symmetry": "off"})  # 這一題不考左右對稱，明寫關著
    with real_server(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-x0").click()
        _drag(page, LY, LZ, (0.02, 0.02), (1.0, 1.2))
        choose_fixed(page, "x0", "z1", PRESET)
        choose_mode(page, "x0", "auto1", "search")
        _clear(page, "z1")

        # 這面牆上限 0.1：貼 0.05 的還在限內；換成 0.15 的那一刻（換材料也重判）就紅。
        _depth(page, "#depth-wall-x0", "10")
        _clear(page, "z1")
        # 不紅的區底下不寫限制（#1265 單元 3 之二收尾 G）：auto2（沒設做法、照預設算指定貼一種，材料是原本那一面）這時判決裡有一條說明（直接取代牆面、上限比不了，
        # 不擋），右欄那一區底下照樣沒有；z1 不紅也沒有。
        assert cast(list[str], page.evaluate("zoneLimits.zones.x0.auto2.map(line => line.level)")) == ["note"]
        _no_limit_lines(page, "auto2", "z1")
        expect(page.locator("#depth-cap-x0-auto1")).to_have_text("個案上限 0.1 公尺（這面牆）；每一組跟自己的上限取較嚴：")
        expect(page.locator("#depth-target-x0-auto1-0")).to_have_text("48K 玻璃棉（可調厚度與空腔）：生效 0.1 公尺（這面牆的上限擋住；材料組自己 0.3 公尺比較寬）")
        choose_fixed(page, "x0", "z1", DEEP)
        _settled(page)
        _red(page, "z1")
        # 紅的那一區底下寫出它的每一條限制；不紅的那兩塊（交給搜尋的 auto1、判決有一條說明的 auto2）底下沒有。
        _red_zone_lines(page, "z1", "0.15 公尺", "這面牆的上限")
        _no_limit_lines(page, "auto1", "auto2")
        # 立面圖那一塊滑過去的 title 照樣寫同一份（一區全部幾條接在一起，所以比的是「含」）。
        hover = page.locator("#elevation [data-zone-id=z1] title")
        expect(hover).to_contain_text("0.15 公尺")
        expect(hover).to_contain_text("這面牆的上限")
        expect(page.locator("#zone-limits-summary")).to_contain_text("有 1 條超過限制")
        expect(page.locator("#zone-to-search-note")).to_contain_text("超過限制")
        expect(page.locator("#elevation [data-zone-id=z1] title")).to_contain_text("這面牆的上限")
        page.locator("#elevation-panel").screenshot(path=str(tmp_path / "limits-wall-red.png"))

        # 牆不給、改給這一塊 0.12：還是紅，寫的是「這一塊」。
        _depth(page, "#depth-wall-x0", "")
        _clear(page, "z1")
        edit_depth(page, "x0", "z1", "12")
        _red(page, "z1")
        expect(page.locator("#limit-x0-z1-0")).to_contain_text("這一塊的上限")
        expect(page.locator("#elevation [data-zone-id=z1] title")).to_contain_text("這一塊的上限")
        # 舊表那一格「生效：0.12 公尺（這一塊擋的）」跟著表走；指定貼一種那一區的上限改讀右欄那一句（改字：公尺→公分、「擋的」拿掉）。
        expect(page.locator("#wall-zone-what-x0-z1")).to_contain_text("總厚度最多 12 公分（這一塊）")

        # 這一塊不給、房間預設 0.1：紅，寫「房間預設」；房間放寬到 0.2 就解除，交給搜尋那塊照房間預設。
        edit_depth(page, "x0", "z1", "")
        _depth(page, "#depth-room", "10")
        _red(page, "z1")
        expect(page.locator("#limit-x0-z1-0")).to_contain_text("房間預設的上限")
        expect(page.locator("#elevation [data-zone-id=z1] title")).to_contain_text("房間預設的上限")
        _depth(page, "#depth-room", "20")
        _clear(page, "z1")
        expect(page.locator("#depth-cap-x0-auto1")).to_have_text("個案上限 0.2 公尺（房間預設）；每一組跟自己的上限取較嚴：")

        # 不合理的值不收：照講、格子畫回原值、不打伺服器。
        page.locator("#depth-room").fill("-1")
        page.locator("#depth-room").blur()
        # #1265 單元 2 審查發現 2 起收不收只判一次、提示句寫在那一格底下（#depth-room-note），不再寫進做法表底下那一行。
        expect(page.locator("#depth-room-note")).to_contain_text("大於 0")
        expect(page.locator("#depth-room")).to_have_value("20")
        assert watched.page_errors == [] and watched.console_errors == []


def test_depth_limits_survive_a_reopen_and_a_zone_cap_follows_a_recut(tmp_path: Path, browser: Browser) -> None:
    _write_room109_design(tmp_path, {"left_right_symmetry": "off"})  # 這一題不考左右對稱，明寫關著
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-x0").click()
        _drag(page, LY, LZ, (0.02, 0.02), (1.0, 1.2))
        choose_fixed(page, "x0", "z1", DEEP)
        choose_mode(page, "x0", "auto1", "search")
        _depth(page, "#depth-room", "30")
        _depth(page, "#depth-wall-x0", "20")
        edit_depth(page, "x0", "auto1", "8")
        expect(page.locator("#depth-cap-x0-auto1")).to_have_text("個案上限 0.08 公尺（這一塊）；每一組跟自己的上限取較嚴：")
        # 在右邊那一大塊（交給搜尋、上限 0.08）裡再畫一框：自動區重編，上限跟著做法走。
        _drag(page, LY, LZ, (2.02, 0.02), (3.0, 1.0))
        after = {str(zone["id"]): zone for zone in _zones(page, "x0")}
        modes = _modes(page, "x0")
        right = [key for key, zone in after.items() if _region(zone) == "right" and key != "z2"]
        # #1176 起改成由程式最佳化就帶好一份明確清單（三種內建玻璃棉＋不貼）。
        allowed = ["glass_wool_48k", "glass_wool_24k", "glass_wool_96k", "none"]
        assert right and all(modes[key] == {"mode": "search", "allowed": allowed, "max_depth_m": 0.08} for key in right)
        _save(page, "room109")
        stored = json.loads((tmp_path / "designs" / "room109.json").read_text(encoding="utf-8"))
        assert stored["room_max_depth_m"] == 0.3 and stored["wall_max_depth_m"] == {"x0": 0.2}
        assert stored["zones"]["x0"] == modes

        page.reload()
        page.wait_for_function("typeof inputDefaults !== 'undefined' && inputDefaults !== undefined")
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-x0").click()
        expect(page.locator("#depth-room")).to_have_value("30")
        expect(page.locator("#depth-wall-x0")).to_have_value("20")
        # 舊表那一格「這一區給 0.08」跟著表走（丙）；同一件事由下一行右欄那一區的個案上限（這一塊 0.08）考。
        expect(page.locator(f"#depth-cap-x0-{right[0]}")).to_have_text("個案上限 0.08 公尺（這一塊）；每一組跟自己的上限取較嚴：")
        _clear(page, "z1")
        # 別面沒填牆的上限：照房間預設。
        page.locator("#elevation-wall-y0").click()
        expect(page.locator("#depth-wall-y0")).to_have_value("")
        page.locator("#elevation-panel").screenshot(path=str(tmp_path / "limits-reopened.png"))
        assert watched.page_errors == [] and watched.console_errors == []


def test_send_to_search_stays_put_while_red(tmp_path: Path, browser: Browser) -> None:
    """紅著的時候「拿去搜尋」按不下去；就算繞過按鈕直接叫，也只照講、不存不跳。"""
    _write(tmp_path, room109_document())
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-x0").click()
        _drag(page, LY, LZ, (0.02, 0.02), (1.0, 1.2))
        choose_fixed(page, "x0", "z1", DEEP, one_side=True)  # 舊做法表只改 z1 一區；小框左右對稱開著會連 z2 一起選，這一題不考左右對稱
        choose_mode(page, "x0", "auto1", "search")
        _depth(page, "#depth-wall-x0", "10")
        _red(page, "z1")
        page.evaluate("sendToSearch()")
        expect(page.locator("#zone-to-search-note")).to_contain_text("改好之前不能拿去搜尋")  # 送出那一路的回話（searchNote）
        assert page.url == f"{base}/"
        assert not (tmp_path / "designs" / "room109.json").exists()
        assert cast(int, page.evaluate("limitsRedCount()")) == 1
        assert watched.page_errors == [] and watched.console_errors == []


def test_a_late_verdict_never_eats_a_half_typed_limit(tmp_path: Path, browser: Browser) -> None:
    """判決回來的時候正在立面圖那一區打字：不重畫（重畫會換掉那一格），打到一半的字留著；離開那一格才算確定、才畫。"""
    _write(tmp_path, room109_document())
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-x0").click()
        _drag(page, LY, LZ, (0.02, 0.02), (1.0, 1.2))
        choose_fixed(page, "x0", "z1", DEEP, one_side=True)  # 舊做法表只改 z1 一區；小框左右對稱開著會連 z2 一起選，這一題不考左右對稱
        choose_mode(page, "x0", "auto1", "search")
        _settled(page)
        reveal(page, "#depth-wall-x0")  # #1265 單元 3 起這一面那一格收在「這一面另設」裡：扣住判決之前先打開
        held: list[Route] = []
        page.route("**/api/zone-limits", lambda route: held.append(route))
        with page.expect_request("**/api/zone-limits"):
            page.evaluate("() => { window.before = zoneLimits; refreshZoneLimits(); }")   # 例如換材料之後那一次，回覆還在路上
        page.locator("#depth-wall-x0").fill("10")    # 這時開始打牆的上限（公分），還沒離開那一格
        with page.expect_response("**/api/zone-limits"):
            held[0].continue_()
        page.unroute("**/api/zone-limits")
        page.wait_for_function("zoneLimits !== window.before")   # 判決收下了，但正在打字：不畫
        assert page.evaluate("zoneLimitsShown < zoneLimitsGeneration")
        expect(page.locator("#depth-wall-x0")).to_have_value("10")
        expect(page.locator("#depth-wall-x0")).to_be_focused()
        page.locator("#depth-wall-x0").blur()           # 離開才算確定：記進設計設定、重判、畫紅
        _settled(page)
        assert page.evaluate("zoneDesign.wall_max_depth_m") == {"x0": 0.1}
        _red(page, "z1")

        # 空白的房間格只打了「.」：值讀起來是空的、跟確定過的一樣，但那是打到一半的字，不准被重畫吃掉。
        reveal(page, "#depth-room")
        page.route("**/api/zone-limits", lambda route: held.append(route))
        with page.expect_request("**/api/zone-limits"):
            page.evaluate("() => { window.before = zoneLimits; refreshZoneLimits(); }")
        page.locator("#depth-room").focus()
        page.keyboard.type(".")
        assert page.locator("#depth-room").evaluate("e => e.validity.badInput")
        with page.expect_response("**/api/zone-limits"):
            held[-1].continue_()
        page.unroute("**/api/zone-limits")
        page.wait_for_function("zoneLimits !== window.before")
        assert page.evaluate("zoneLimitsShown < zoneLimitsGeneration")
        assert page.locator("#depth-room").evaluate("e => e.validity.badInput")   # 還是同一格、字還在
        page.keyboard.type("2")
        page.locator("#depth-room").blur()
        _settled(page)
        assert page.evaluate("zoneDesign.room_max_depth_m") == 0.002  # #1265 單元 3 起填公分：「.2」公分＝0.002 公尺
        assert watched.page_errors == [] and watched.console_errors == []


def test_a_verdict_arriving_mid_drag_waits_for_the_mouse_and_replies_out_of_order_are_dropped(tmp_path: Path,
                                                                                              browser: Browser) -> None:
    """拖框拖到一半判決回來：先收著不重畫（重畫會換掉正在拖的那張圖），放開滑鼠才畫，框照樣切成。
    兩次判決倒序回來：晚到的舊判決丟掉，畫面上是最新那次。"""
    _write_room109_design(tmp_path, {"left_right_symmetry": "off"})  # 這一題不考左右對稱，明寫關著
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-x0").click()
        _settled(page)
        held: list[Route] = []
        page.locator("#elevation-frame").scroll_into_view_if_needed()   # 滑鼠座標以視窗為準
        page.route("**/api/zone-limits", lambda route: held.append(route))
        with page.expect_request("**/api/zone-limits"):
            page.evaluate("() => { refreshZoneLimits(); }")
        frame = page.locator("#elevation-frame").bounding_box()
        assert frame is not None
        page.mouse.move(frame["x"] + 2, frame["y"] + frame["height"] - 2)
        page.mouse.down()
        page.mouse.move(frame["x"] + frame["width"] * 0.3, frame["y"] + frame["height"] * 0.6, steps=4)
        with page.expect_response("**/api/zone-limits"):
            held[0].continue_()
        page.wait_for_function("zoneLimitsReceived === zoneLimitsGeneration")
        assert page.evaluate("[elevationDrawing(), zoneLimitsShown < zoneLimitsGeneration]") == [True, True]   # 收下了，但還在拖：不畫
        page.unroute("**/api/zone-limits")
        with page.expect_response("**/api/zone-cut") as cut:
            page.mouse.up()
        assert cut.value.ok, cut.value.text()
        page.wait_for_function("!document.querySelector('main').inert")
        assert sorted(str(zone["id"]) for zone in _zones(page, "x0")) == ["auto1", "auto2", "z1"]

        # 倒序：先發的那次最後才回來，丟掉；畫面上是後發的那次。
        page.wait_for_function("zoneLimitsReceived === zoneLimitsGeneration")
        held.clear()
        page.route("**/api/zone-limits", lambda route: held.append(route))
        page.evaluate("() => { refreshZoneLimits(); refreshZoneLimits(); }")   # 兩次都在同一拍送出
        for _ in range(200):                                                   # 等攔截收齊兩個請求（最多 10 秒）
            if len(held) == 2:
                break
            page.wait_for_timeout(50)
        assert len(held) == 2
        newest = cast(int, page.evaluate("zoneLimitsGeneration"))
        with page.expect_response("**/api/zone-limits"):
            held[1].continue_()
        page.wait_for_function("zoneLimitsShown === zoneLimitsGeneration")
        with page.expect_response("**/api/zone-limits"):
            held[0].continue_()
        page.unroute("**/api/zone-limits")
        assert page.evaluate("[zoneLimitsShown, zoneLimitsReceived]") == [newest, newest]
        assert watched.page_errors == [] and watched.console_errors == []


def test_send_to_search_waits_for_the_verdict_and_refuses_when_the_check_failed(tmp_path: Path, browser: Browser) -> None:
    """「拿去搜尋」等最新那次判決；檢查沒做成（斷線）就照講、不存不跳——伺服器開跑那邊雖然還會擋，這頁不先存一份過去。"""
    _write(tmp_path, room109_document())
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-x0").click()
        choose_mode(page, "x0", "whole_wall", "search")
        _settled(page)
        page.route("**/api/zone-limits", lambda route: route.abort())
        page.evaluate("() => { refreshZoneLimits(); }")
        page.locator("#zone-to-search").click()
        expect(page.locator("#zone-to-search-note")).to_contain_text("限制沒檢查成")  # 送出那一路的回話（searchNote）
        assert page.url == f"{base}/"
        assert not (tmp_path / "designs" / "room109.json").exists()
        assert watched.page_errors == []


def test_waiting_for_the_verdict_on_a_request_that_already_came_back_lets_the_page_breathe(tmp_path: Path,
                                                                                         browser: Browser) -> None:
    """等最新判決時，手上那一個請求已經回來、判決卻還不是最新（世代號又往前了）：不准在同一個已經好了的請求上空轉。
    空轉只讓出微任務、不讓出事件迴圈，計時器、網路回覆都進不來，整頁卡死（雲端考卷關分頁卡 30 分鐘，#1265）。
    這裡把手上那一個換成會數次數的「已經好了」的請求（數到 1000 次就放它走，考卷不會跟著卡死），再排一個計時器 100 毫秒後重問：
    讓出事件迴圈的話，計時器會跑、新的一次回來、等待結束，那個舊請求只被等了幾次。"""
    _write(tmp_path, room109_document())
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        _settled(page)
        spins = page.evaluate("""async () => {
          let calls = 0;
          zoneLimitsRequest = {then(resolve) {
            calls += 1;
            if (calls >= 1000) zoneLimitsReceived = zoneLimitsGeneration;
            resolve();
          }};
          zoneLimitsGeneration += 1;
          setTimeout(() => { refreshZoneLimits(); }, 100);
          await settledZoneLimits();
          return calls;
        }""")
        assert spins < 10, f"在同一個已經回來的請求上空轉了 {spins} 次"
        _settled(page)
        assert watched.page_errors == []


def test_a_plain_click_and_tab_or_enter_after_a_limit_still_show_the_verdict(tmp_path: Path, browser: Browser) -> None:
    """只點一下沒拖（收尾那條路會先重畫）不讓「拖框中」卡住；打完上限按 Tab 換格或按 Enter，判決照樣畫上去、
    焦點留在原來那一格（只有那一格打了還沒確定的字才延後）。
    #1265 起房間預設那一格在第 1 段右卡，Tab 換到的是右卡下一顆鈕；「換到立面圖同一區另一個、判決重畫整區後焦點放回那一個」
    那條路改用這一面那一格考（Tab 換到同一區下一個帶 id 的，單元 3 起是第 2 段右欄清單第一區的「改這一區 ▸」）。"""
    _write(tmp_path, room109_document())
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-x0").click()
        _drag(page, LY, LZ, (0.02, 0.02), (1.0, 1.2))
        choose_fixed(page, "x0", "z1", DEEP, one_side=True)  # 舊做法表只改 z1 一區；小框左右對稱開著會連 z2 一起選，這一題不考左右對稱
        choose_mode(page, "x0", "auto1", "search")
        _settled(page)
        page.locator("#elevation-frame").scroll_into_view_if_needed()
        box = page.locator("#elevation [data-zone-id=auto1]").bounding_box()
        assert box is not None
        page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        assert page.evaluate("elevationDrawing()") is False
        _depth(page, "#depth-wall-x0", "10")
        _red(page, "z1")
        _depth(page, "#depth-wall-x0", "")
        _clear(page, "z1")

        # 房間預設那一格：Tab 落到右卡下一顆鈕（不在限制區），判決照畫、焦點不被拉走。
        reveal(page, "#depth-room").fill("10")
        page.locator("#depth-room").press("Tab")
        _settled(page)
        _red(page, "z1")
        expect(page.locator("#edit-speakers")).to_be_focused()
        page.locator("#depth-room").fill("30")
        page.locator("#depth-room").press("Enter")
        _settled(page)
        _clear(page, "z1")
        expect(page.locator("#depth-room")).to_be_focused()
        # 這一面那一格：Tab 換到立面圖同一區下一個帶 id 的；判決回來整區重畫，焦點放回那一個（不掉到頁面本身）。
        # #1265 單元 3 起下一個帶 id 的是第 2 段右欄清單第一區的「改這一區 ▸」：清單照做法分組、由程式最佳化排最前，
        # 是 auto1（舊做法表照區號排，第一列是 z1）。清單跟判決一起重畫，考的還是「重畫後焦點放回同一個」。
        page.locator("#depth-wall-x0").fill("10")
        page.locator("#depth-wall-x0").press("Tab")
        _settled(page)
        _red(page, "z1")
        expect(page.locator("#elevation-panel #zone-goto-x0-auto1")).to_be_focused()
        assert watched.page_errors == [] and watched.console_errors == []


def test_which_limit_blocks_each_group_and_a_zone_nothing_fits_goes_red(tmp_path: Path, browser: Browser) -> None:
    """#1082 段 2-4：交給搜尋的區逐組寫出生效上限與哪一條擋住（個案哪一層／材料組自己）；比某一組最淺排法還淺，
    那一組標「不可用」但不擋；准用的組全放不下才紅、擋「拿去搜尋」。用正式網頁入口起、109 號房，截圖存證。"""
    _write_room109_design(tmp_path, {"left_right_symmetry": "off"})  # 這一題不考左右對稱，明寫關著
    (tmp_path / "materials.json").write_text(_deep_user_library(), encoding="utf-8")
    with real_server(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-x0").click()
        _drag(page, LY, LZ, (0.02, 0.02), (1.0, 1.2))
        choose_fixed(page, "x0", "z1", PRESET)
        choose_mode(page, "x0", "auto1", "search")
        # 一開始只帶三種內建玻璃棉＋不貼（#1176）：使用者材料要在小框自己加進來，勾的排在最後（第 4 組、編號 3）。
        edit_candidates(page, "x0", "auto1", add=("deep_rock_wool",))
        assert _modes(page, "x0")["auto1"]["allowed"] == ["glass_wool_48k", "glass_wool_24k", "glass_wool_96k", "none",
                                                          "deep_rock_wool"]
        # 房間預設 0.2：玻璃棉照個案（房間預設）擋，使用者材料自己 0.15 比較嚴、寫「材料組自己的上限擋住」。
        _depth(page, "#depth-room", "20")
        _clear(page, "z1")
        expect(page.locator("#depth-target-x0-auto1-0")).to_have_text(
            "48K 玻璃棉（可調厚度與空腔）：生效 0.2 公尺（房間預設的上限擋住；材料組自己 0.3 公尺比較寬）")
        expect(page.locator("#depth-target-x0-auto1-3")).to_have_text(
            "深背岩棉（自填）：生效 0.15 公尺（材料組自己的上限擋住；0.2 公尺（房間預設的上限）比較寬）")
        # 這一塊 0.06：比使用者材料最淺排法 0.08 還淺，那一組不可用（橘字、不擋），玻璃棉照樣可用。
        edit_depth(page, "x0", "auto1", "6")
        _clear(page, "auto1")
        unusable = page.locator("#depth-target-x0-auto1-3")
        expect(unusable).to_have_text("深背岩棉（自填）：不可用——深度上限 0.06 公尺（這一塊的上限）比這一組最淺排法 0.08 公尺還淺（個案上限擋住）")
        expect(unusable).to_have_class("limit-warn")
        shots = Path(os.environ.get("AOSR_SHOTS") or tmp_path)  # 給了資料夾就把截圖存去那裡（證據），沒給存暫存
        page.locator("#elevation-panel").screenshot(path=str(shots / "depth-which-limit.png"))
        # 這一塊 0.03：每一組都放不下（「不貼」不算救）→ 紅、擋拿去搜尋，原文講哪一條擋。
        edit_depth(page, "x0", "auto1", "3")
        _red(page, "auto1")
        # 右欄正列著前牆：那一句寫在 auto1 底下（紅字），總述只指過去（#1265 單元 3 之二收尾第二趟 A；原本在總述裡考同樣兩段字）。
        nothing_fits = page.locator("#wall-zone-x0-auto1 .limit-lines")
        expect(nothing_fits).to_contain_text("程式能選的材料沒有一種放得進深度上限，這一區什麼都貼不上")
        expect(nothing_fits).to_contain_text("0.03 公尺（這一塊的上限）比這一組最淺排法 0.05 公尺還淺")
        expect(page.locator("#zone-limits-summary")).to_contain_text("的 1 條寫在那一區底下")
        page.locator("#elevation-panel").screenshot(path=str(shots / "depth-nothing-fits-red.png"))
        assert watched.page_errors == [] and watched.console_errors == []


def test_changing_the_list_rechecks_the_limits(tmp_path: Path, browser: Browser) -> None:
    """審查 11-15：在小框套用候選清單跟改做法一樣重判限制；不然深度上限那一欄每一組的生效上限、
    紅燈、「拿去搜尋」停不停用都是改之前的判決（改之前就這樣，這一支把清單變成到處都能改，一起補）。"""
    _write_room109_design(tmp_path, {"left_right_symmetry": "off"})  # 這一題不考左右對稱，明寫關著
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-x0").click()
        _drag(page, LY, LZ, (0.02, 0.02), (1.0, 1.2))
        choose_mode(page, "x0", "z1", "search")
        edit_depth(page, "x0", "z1", "8")
        expect(page.locator("#depth-target-x0-z1-0")).to_contain_text("48K 玻璃棉")
        expect(page.locator("#depth-target-x0-z1-2")).to_contain_text("96K 玻璃棉")
        edit_candidates(page, "x0", "z1", remove=("glass_wool_48k",))
        _settled(page)
        expect(page.locator("#depth-target-x0-z1-0")).to_contain_text("24K 玻璃棉")
        expect(page.locator("#depth-target-x0-z1-2")).to_have_count(0)
        # 0.03：剩下兩種玻璃棉都放不下→紅、擋拿去搜尋；清單只留「不貼」就不判這一項，紅燈跟著熄、按鈕跟著恢復。
        edit_depth(page, "x0", "z1", "3")
        _red(page, "z1")
        edit_candidates(page, "x0", "z1", remove=("glass_wool_24k",))
        edit_candidates(page, "x0", "z1", remove=("glass_wool_96k",))
        _settled(page)
        _clear(page, "z1")
        assert watched.page_errors == [] and watched.console_errors == []



GROUPS_EVERYTHING = ["glass_wool_48k", "glass_wool_24k", "glass_wool_96k", "none", "deep_rock_wool"]


def _nothing_fits_room(tmp_path: Path) -> None:
    """前牆整面由程式最佳化（三種玻璃棉、不貼、深背岩棉），房間預設上限 3 公分：每一組都放不下，整面紅。"""
    _write_room109_design(tmp_path, {"left_right_symmetry": "off", "room_max_depth_m": 0.03,
                                     "zones": {"x0": {"whole_wall": {"mode": "search", "allowed": GROUPS_EVERYTHING}}}})
    (tmp_path / "materials.json").write_text(_deep_user_library(), encoding="utf-8")


def _times_in_right_column(page: Page, sentence: str) -> int:
    return cast(int, page.evaluate("text => document.getElementById('elevation-list').innerText.split(text).length - 1", sentence))


def test_a_zone_where_nothing_fits_says_each_group_once_in_the_whole_right_column(tmp_path: Path, browser: Browser) -> None:
    """#1265 單元 3 之二收尾第二趟 A：前牆整面由程式最佳化、房間預設上限 3 公分，每一組都放不下（整面紅）。
    整個右欄（含上方總述）每一組那一句只出現 1 次：寫在那一區底下的紅字；總述不抄、改指到那一區；逐組那幾行不再列一遍。
    上限從哪裡來照樣看得到。換到左牆，前牆那幾句畫面上看不到，總述照列。房間預設放寬到 6 公分：不紅、只有深背岩棉放不下，
    逐組那幾行照舊在（告訴人搜尋會跳過哪一組）。資料是考卷自己編的（109 號房、深背岩棉那一筆使用者材料）。"""
    _nothing_fits_room(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-x0").click()
        _settled(page)
        sentences = cast(list[str], page.evaluate("zoneLimits.caps.x0.whole_wall.targets.map(target => target.text)"))
        refusal = cast(list[str], page.evaluate("zoneLimits.refusals"))
        assert len(sentences) == 4 and len(refusal) == 1, (sentences, refusal)
        summary = page.locator("#zone-limits-summary")
        head = "有 1 條超過限制（標紅的區），改好之前不能拿去搜尋："
        expect(summary).to_have_text(f"{head}前牆「整面」的 1 條寫在那一區底下")
        expect(page.locator("#wall-zone-x0-whole_wall #limit-x0-whole_wall-0")).to_contain_text(sentences[0])
        for sentence in sentences:
            assert _times_in_right_column(page, sentence) == 1, sentence
        expect(page.locator("[id^=depth-target-x0-whole_wall-]")).to_have_count(0)
        expect(page.locator("#wall-zone-what-x0-whole_wall")).to_contain_text("總厚度最多 3 公分（房間預設）")
        page.locator("#elevation-wall-y0").click()
        expect(summary).to_have_text(head + refusal[0])
        page.locator("#elevation-wall-x0").click()
        _depth(page, "#depth-room", "6")
        limits_all_clear(page)
        expect(page.locator("[id^=depth-target-x0-whole_wall-]")).to_have_count(4)
        expect(page.locator("#depth-target-x0-whole_wall-3")).to_have_class("limit-warn")
        assert watched.page_errors == [] and watched.console_errors == []


def test_while_a_recut_is_checked_the_summary_keeps_the_zone_lines_no_zone_shows(tmp_path: Path, browser: Browser) -> None:
    """同上那一間：在前牆畫一個框，新的判決還在路上。上一版判決紅的是「整面」，那一區已經不在右欄（換成新切的幾區），
    底下沒有人寫那一條——總述照抄，不准說寫在那一區底下而把它藏掉。新的判決回來照新的寫。"""
    _nothing_fits_room(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-x0").click()
        _settled(page)
        refusal = cast(list[str], page.evaluate("zoneLimits.refusals"))
        held: list[Route] = []
        page.route("**/api/zone-limits", lambda route: held.append(route))
        _drag(page, LY, LZ, (0.02, 0.02), (1.0, 1.2))
        page.wait_for_function("!document.querySelector('main').inert && !inputBusy && zoneLimitsReceived !== zoneLimitsGeneration")
        assert held, "限制檢查應該還在路上"
        expect(page.locator("#wall-zone-x0-whole_wall")).to_have_count(0)
        expect(page.locator("#zone-limits-summary")).to_have_text("有 1 條超過限制（標紅的區），改好之前不能拿去搜尋：" + refusal[0])
        _release(held)
        page.unroute("**/api/zone-limits")
        _settled(page)
        expect(page.locator("#zone-limits-summary")).not_to_contain_text(refusal[0])
        assert watched.page_errors == []

def _numbered_zones_room(tmp_path: Path) -> None:
    """前牆切成四塊，代號有兩個是純數字（"10"、"2"，瀏覽器會把純數字的鍵排到前面），那兩塊由程式最佳化；左牆整面也由程式
    最佳化。房間預設上限 3 公分：那三區每一組都放不下，都紅（前牆同一面兩區紅）。資料是考卷自己編的。"""
    document = room109_document()
    zones = cast(dict[str, list[dict[str, object]]], cast(dict[str, object], document["scene"])["surface_zones_by_wall"])
    width, height = LY, LZ
    zones["x0"] = [{"area": "rectangle", "id": zone_id, "u_m": [col * width / 2, width if col else width / 2],
                    "v_m": [row * height / 2, height if row else height / 2],
                    "material": {"kind": "constant_impedance", "impedance_pa_s_m": 4000.0}}
                   for zone_id, (col, row) in zip(["10", "z1", "2", "z2"], [(0, 0), (0, 1), (1, 0), (1, 1)], strict=True)]
    _write(tmp_path, document)
    search = {"mode": "search", "allowed": GROUPS_EVERYTHING}
    design = {"schema_version": "aosr.zone_design.v1", "scheme_id": "room109", "left_right_symmetry": "off", "room_max_depth_m": 0.03,
              "zones": {"x0": {"10": search, "2": search}, "y0": {"whole_wall": search}}}
    (tmp_path / "designs").mkdir(exist_ok=True)
    (tmp_path / "designs" / "room109.json").write_text(json.dumps(design), encoding="utf-8")
    (tmp_path / "materials.json").write_text(_deep_user_library(), encoding="utf-8")


def test_two_red_zones_with_number_codes_on_one_wall_point_below_and_other_walls_stay_listed(tmp_path: Path,
                                                                                              browser: Browser) -> None:
    """#1265 單元 3 之二收尾第三趟順手（審查建議 3）：同一面兩區紅、代號是純數字跟英數混排（瀏覽器把 "2"、"10" 排到前面，
    判決 refusals 照伺服器的順序）。正在看前牆：總述指到「各區底下」、兩區照號碼列名，前牆那兩條不抄；別面（左牆）那一條照抄。
    換到左牆：前牆那兩條照抄、左牆那一條改指到那一區底下。每一區底下寫的是它自己那一條。"""
    _numbered_zones_room(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-x0").click()
        _settled(page)
        assert cast(list[str], page.evaluate("Object.keys(zoneLimits.zones.x0)"))[:2] == ["2", "10"], "瀏覽器應該把純數字代號排到前面"
        refusal = cast(list[str], page.evaluate("zoneLimits.refusals"))
        mine = {key: cast(list[str], page.evaluate(f"zoneLimits.zones.x0['{key}'].filter(l => l.level === 'red').map(l => l.text)"))
                for key in ("2", "10")}
        left = cast(list[str], page.evaluate("zoneLimits.zones.y0.whole_wall.filter(l => l.level === 'red').map(l => l.text)"))
        assert [len(mine["2"]), len(mine["10"]), len(left)] == [1, 1, 1] and len(refusal) == 3, (mine, left, refusal)
        order = cast(list[str], page.evaluate("window.zoneFaceZones('x0').filter(i => ['2', '10'].includes(i.key)).map(i => i.key)"))
        names = [cast(str, page.evaluate(f"zoneCalled('x0', '{key}')")) for key in order]
        head = "有 3 條超過限制（標紅的區），改好之前不能拿去搜尋："
        # 左牆那一條是 refusals 裡最後一條（伺服器照 x0、y0 排），照抄在前；前牆兩區改指到下面。
        summary = page.locator("#zone-limits-summary")
        expect(summary).to_have_text(f"{head}{refusal[2]}；{'、'.join(names)}的 2 條寫在各區底下")
        # 三區的那一條原文一字不差（都是房間預設擋住、同一張清單），所以不數整欄出現幾次：總述上面已逐字比過，這裡比每一區底下
        # 各自一條、就是它自己那一條；前牆兩條帶區名的 refusals 原文不在總述裡（上面逐字比已經排除，這裡明寫一次）。
        for key in ("2", "10"):
            expect(page.locator(f"#wall-zone-x0-{key} .limit-lines p")).to_have_text([mine[key][0]])
        assert refusal[0] not in summary.inner_text() and refusal[1] not in summary.inner_text()
        page.locator("#elevation-wall-y0").click()
        _settled(page)
        expect(summary).to_have_text(f"{head}{refusal[0]}；{refusal[1]}；左牆「整面」的 1 條寫在那一區底下")
        expect(page.locator("#wall-zone-y0-whole_wall #limit-y0-whole_wall-0")).to_have_text(left[0])
        assert watched.page_errors == [] and watched.console_errors == []


def _deep_user_library() -> str:
    """最淺排法 0.08 m（最薄 0.05 m＋空腔下界 0.03 m）、自己的上限 0.15 m 的使用者可最佳化材料。"""
    from pydantic import TypeAdapter

    from aosr.materials import library

    adapter: TypeAdapter[library.LibraryEntry] = TypeAdapter(library.LibraryEntry)
    entry = adapter.validate_python({
        "kind": "optimizable", "id": "deep_rock_wool", "label": "深背岩棉（自填）", "provenance": "model", "source": "自填",
        "porous": {"kind": "porous", "thickness_m": 0.05, "flow_resistivity_pa_s_m2": 20000.0, "material_class": "fibrous"},
        "porous_thickness": {"adjustable": True, "range_m": [0.05, 0.1], "choices_m": [0.05, 0.1], "initial_m": 0.05},
        "cavity": {"adjustable": True, "range_m": [0.03, 0.1], "initial_m": 0.04}, "max_total_depth_m": 0.15})
    return library.dump_user_library((entry,))
