"""牆名跟著方案走（#1134）：前牆照 /search-start 前牆預設那一條推（喇叭中點最近的牆），左右是面向前牆時的左右。

109 part2d 的擺位當考卷（左喇叭 y 0.94、主位面向 x0）：x0＝前牆、xL＝後牆、y0＝左牆、yL＝右牆。
同一間房轉 90、180、270 度再考一次——只考 109 分不出「推出來的」與「寫死 x0＝前牆」。
"""
from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from playwright.sync_api import Browser, Page, Route, expect
from starlette.testclient import TestClient

from aosr.gui.app import GuiSettings, create_app
from aosr.gui.gradient_plan import GradientRequest
from aosr.gui.labels import WALLS
from aosr.gui.wall_words import draft_wall_words, front_wall, wall_words
from aosr.reporting.scheme import Scheme
from tests.engine._furniture_cases import reference_document
from tests.engine._surface_zone_cases import ROOM_109, room109_document
from tests.engine.test_gui_browser import _assert_quiet, _open, _serve, browser
from tests.engine.test_gui_gradient_jobs import NAME, FakeLauncher, _data, _jobs

# 109 part2d 存檔裡的喇叭與主位（公尺）。
LEFT_109 = (0.36810808858895405, 0.937648486381631)
RIGHT_109 = (0.36810808858895405, 3.4822329683114788)
PRIMARY_109 = (2.3865101734352736, 2.209940727346555)
Doc = dict[str, object]


def _part2d_layout() -> Doc:
    """109 號房、part2d 的喇叭與主位；周圍點跟著主位平移。"""
    document = room109_document("part2d-layout")
    speakers = cast(dict[str, dict[str, float]], document["speakers"])
    for name, (x, y) in (("left", LEFT_109), ("right", RIGHT_109)):
        speakers[name]["x"], speakers[name]["y"] = x, y
    points = cast(list[dict[str, object]], cast(Doc, document["receiver_set"])["points"])
    primary = next(cast(list[float], point["position_m"]) for point in points if point["role"] == "primary")
    dx, dy = PRIMARY_109[0] - primary[0], PRIMARY_109[1] - primary[1]
    for point in points:
        position = cast(list[float], point["position_m"])
        point["position_m"] = [position[0] + dx, position[1] + dy, position[2]]
    return document


def _turned(document: Doc, quarter_turns: int) -> Scheme:
    """整間房連喇叭、座位逆時針轉 90 度 × 次數（左右手不變，不是鏡像）：(x, y) → (Ly − y, x)，房間長寬互換。"""
    for _ in range(quarter_turns % 4):
        room = cast(dict[str, float], cast(Doc, document["scene"])["room_m"])
        ly = room["Ly"]

        def turn(x: float, y: float, ly: float = ly) -> tuple[float, float]:
            return ly - y, x

        for point in cast(dict[str, dict[str, float]], document["speakers"]).values():
            point["x"], point["y"] = turn(point["x"], point["y"])
        for receiver in cast(list[dict[str, object]], cast(Doc, document["receiver_set"])["points"]):
            position = cast(list[float], receiver["position_m"])
            receiver["position_m"] = [*turn(position[0], position[1]), position[2]]
        room["Lx"], room["Ly"] = room["Ly"], room["Lx"]
    return Scheme.model_validate(document)


def test_part2d_layout_names_x0_front_y0_left() -> None:
    names = wall_words(_turned(_part2d_layout(), 0))
    assert names == {"floor": "地板", "ceiling": "天花", "x0": "前牆", "xL": "後牆", "y0": "左牆", "yL": "右牆"}


@pytest.mark.parametrize(("turns", "front", "back", "left", "right"), [
    (1, "y0", "yL", "xL", "x0"), (2, "xL", "x0", "yL", "y0"), (3, "yL", "y0", "x0", "xL")])
def test_turned_room_moves_every_name_with_the_layout(turns: int, front: str, back: str, left: str, right: str) -> None:
    scheme = _turned(_part2d_layout(), turns)
    names = wall_words(scheme)
    assert (names[front], names[back], names[left], names[right]) == ("前牆", "後牆", "左牆", "右牆")
    # 獨立核一次「左」：左聲道喇叭離左牆比離右牆近（不經過 LEFT_OF 那張表）。
    room = scheme.scene.room_m
    speaker = scheme.speakers["left"]
    distance = {"x0": speaker.x, "xL": room.Lx - speaker.x, "y0": speaker.y, "yL": room.Ly - speaker.y}
    assert distance[left] < distance[right]


def test_names_fall_back_to_coordinates_when_the_two_rules_disagree_or_tie() -> None:
    # 主位坐在喇叭和最近那面牆之間：喇叭靠 x0，人卻面向 +x，兩套說法不同 → 不挑一套。
    document = _part2d_layout()
    for receiver in cast(list[dict[str, object]], cast(Doc, document["receiver_set"])["points"]):
        position = cast(list[float], receiver["position_m"])
        receiver["position_m"] = [position[0] - 2.2, position[1], position[2]]
    behind = Scheme.model_validate(document)
    assert front_wall(behind) is None and wall_words(behind) == WALLS
    # 喇叭中點離 x0 與 y0 一樣近：平手不猜。
    tied = _part2d_layout()
    speakers = cast(dict[str, dict[str, float]], tied["speakers"])
    middle_y = (speakers["left"]["y"] + speakers["right"]["y"]) / 2
    for point in speakers.values():
        point["x"] = middle_y
    room = cast(dict[str, float], cast(Doc, tied["scene"])["room_m"])
    assert room == ROOM_109
    assert wall_words(Scheme.model_validate(tied)) == WALLS


def test_input_preview_hands_the_page_the_names(tmp_path: Path) -> None:
    """首頁表單每改一格就問的那一支帶牆名（圖面資料 /api/plan 是凍結的答案，不加欄位）。"""
    client = TestClient(create_app(GuiSettings(engine_commit="a" * 40, data_dir=tmp_path)), base_url="http://localhost")
    document = _turned(_part2d_layout(), 2).model_dump(mode="json")
    preview = client.post("/api/input-preview", json=document)
    assert preview.status_code == 200, preview.text
    names = preview.json()["wall_names"]
    assert (names["xL"], names["x0"], names["yL"], names["y0"]) == ("前牆", "後牆", "左牆", "右牆")


def test_home_form_renames_walls_when_the_speakers_move_to_the_other_end(tmp_path: Path, browser: Browser) -> None:
    with _serve(tmp_path) as base, _open(browser, f"{base}/") as watched:
        page = watched.page
        # 範本（參考房）喇叭靠 x0、主位面向 x0。
        expect(page.locator("#wall-editor-x0")).to_contain_text("前牆材料")
        expect(page.locator("#wall-editor-y0")).to_contain_text("左牆材料")
        expect(page.locator("#elevation-wall-xL")).to_have_text("後牆")
        # 兩支喇叭搬到主位另一邊、靠 xL：前後左右整組換過來，表單與立面圖按鈕都跟著改。
        for name in ("left", "right"):
            page.locator(f"#speaker-{name}-x").fill("5")
        expect(page.locator("#wall-editor-xL")).to_contain_text("前牆材料")
        expect(page.locator("#wall-editor-x0")).to_contain_text("後牆材料")
        expect(page.locator("#wall-editor-yL")).to_contain_text("左牆材料")
        expect(page.locator("#elevation-wall-y0")).to_have_text("右牆")
        expect(page.locator("#scatter-xL").locator("xpath=..")).to_contain_text("前牆散射")
        # 重畫牆面表單會清空「約 ρc 的幾倍」那幾格：換名之後要重問、填回來（審查實測三次都被洗掉）。
        expect(page.locator("#multiple-x0")).to_contain_text("ρc")
        _assert_quiet(watched)


def _late_rename(page: Page) -> list[Route]:
    """清空寬 Ly，把伺服器回的牆名扣在路上（房間不完整推不出前牆，回的是座標名）；回傳扣住的那一份。"""
    held: list[Route] = []
    page.route("**/api/input-preview", lambda route: held.append(route))
    with page.expect_request("**/api/input-preview"):
        page.locator("#room-Ly").fill("")
    return held


def _arrive(page: Page, held: list[Route]) -> None:
    """放行扣住的那一份，等頁面收下（牆名表換成座標名）。"""
    with page.expect_response("**/api/input-preview"):
        held[0].continue_()
    page.unroute("**/api/input-preview")
    page.wait_for_function("() => wallNames.x0 === 'x 起點牆'")


def test_a_late_rename_never_swaps_the_wall_box_being_typed_in(tmp_path: Path, browser: Browser) -> None:
    # #1145：換名那份回覆晚到時，人已經點進地板阻抗那一格。以前整排牆面表單當場重畫、那一格被換掉，
    # 下一個按鍵落到頁面本身：清空沒清到、打的字沒進去（雲端首頁另存考卷兩種偶發紅都是這樣來的）。
    # 現在游標在牆面表單裡就先不換名，在表單裡換格也不換；離開表單（包括直接去按按鈕）才換，打的字照留。
    with _serve(tmp_path) as base, _open(browser, f"{base}/") as watched:
        page = watched.page
        expect(page.locator("#wall-editor-x0")).to_contain_text("前牆材料")
        held = _late_rename(page)
        floor = page.locator("#wall-floor")
        floor.focus()
        floor.select_text()
        box = floor.element_handle()
        _arrive(page, held)
        assert box.evaluate("node => node.isConnected && node === document.activeElement")
        page.keyboard.press("Delete")
        assert floor.input_value() == ""
        page.keyboard.insert_text("2000")
        assert floor.input_value() == "2000"
        # 用滑鼠點表單裡的另一格（瀏覽器自己換焦點，不是程式呼叫 focus()）：點到的那一格也不准被換掉。
        ceiling = page.locator("#wall-ceiling").element_handle()
        page.locator("#wall-ceiling").click()
        assert ceiling.evaluate("node => node.isConnected && node === document.activeElement")
        assert box.evaluate("node => node.isConnected")
        # 用 Tab 換到表單裡的下一格（沒有按著滑鼠）：一樣不算離開。
        kind = page.locator("#material-kind-x0").element_handle()
        page.keyboard.press("Tab")
        assert kind.evaluate("node => node.isConnected && node === document.activeElement")
        expect(page.locator("#wall-editor-x0")).to_contain_text("前牆材料")
        # 從牆面表單直接按「檢查」：那一下照樣按得到，按的時候換名，表單的值是剛打的那個。
        page.locator("#messages").evaluate("node => { node.textContent = ''; }")
        page.locator("#check").click()
        expect(page.locator("#messages")).to_have_text("寬 Ly（公尺）：空著沒填")
        expect(page.locator("#wall-editor-x0")).to_contain_text("x 起點牆材料")
        assert floor.input_value() == "2000"
        # 換過名就沒有擱著的了：再點進來、再離開，不准又整排重畫一次。
        again = floor.element_handle()
        floor.click()
        page.locator("#check").click()
        # 放開滑鼠時排的那一次（setTimeout 0）先跑完：同樣延遲的計時器照排進去的先後跑。
        page.evaluate("() => new Promise(resolve => setTimeout(resolve, 0))")
        assert again.evaluate("node => node.isConnected")
        assert all("422" in text for text in watched.console_errors), watched.console_errors
        assert watched.page_errors == []


def test_a_click_on_a_wall_label_while_a_rename_waits_lands_in_its_box(tmp_path: Path, browser: Browser) -> None:
    # 複查實測：換名擱著時點表單裡另一格的標籤字，以前在按下那一刻就重畫，標籤被換掉、那一下沒點到，
    # 焦點落到頁面本身、接著打的字落空。現在滑鼠按著不重畫，放開、點完才看游標在哪。
    with _serve(tmp_path) as base, _open(browser, f"{base}/") as watched:
        page = watched.page
        expect(page.locator("#wall-editor-x0")).to_contain_text("前牆材料")
        held = _late_rename(page)
        page.locator("#wall-floor").focus()
        _arrive(page, held)
        page.locator("#wall-ceiling").locator("xpath=..").click(position={"x": 4, "y": 4})
        assert page.evaluate("() => document.activeElement.id") == "wall-ceiling"
        expect(page.locator("#wall-editor-x0")).to_contain_text("前牆材料")
        # 點表單外不能聚焦的地方（區塊標題）離開：放開之後換名，頁面不出錯。
        page.locator("h2", has_text="房間與材料").click()
        expect(page.locator("#wall-editor-x0")).to_contain_text("x 起點牆材料")
        _assert_quiet(watched)


def test_a_button_pressed_from_the_wall_form_on_a_narrow_window_is_pressed(tmp_path: Path, browser: Browser) -> None:
    # 複查實測：600 寬時換名讓牆名變長、折行，下面的格子往下移；在按下與放開之間重畫，那一下就按到別處。
    with _serve(tmp_path) as base, _open(browser, f"{base}/", viewport_width=600) as watched:
        page = watched.page
        expect(page.locator("#wall-editor-x0")).to_contain_text("前牆材料")
        held = _late_rename(page)
        page.locator("#wall-floor").focus()
        _arrive(page, held)
        page.locator("#use-scattering").click()
        expect(page.locator("#use-scattering")).to_be_checked()
        expect(page.locator("#wall-editor-x0")).to_contain_text("x 起點牆材料")
        _assert_quiet(watched)


def test_a_rename_waits_while_the_pointer_is_down_and_a_cancelled_press_releases_it(tmp_path: Path,
                                                                                      browser: Browser) -> None:
    # 按著（拖框、按住捲動）的時候離開也先不換；觸控被瀏覽器收回（pointercancel，不會有放開）一樣算放開，換名不准卡住。
    with _serve(tmp_path) as base, _open(browser, f"{base}/") as watched:
        page = watched.page
        expect(page.locator("#wall-editor-x0")).to_contain_text("前牆材料")
        held = _late_rename(page)
        page.locator("#wall-floor").focus()
        _arrive(page, held)
        page.evaluate("() => document.dispatchEvent(new PointerEvent('pointerdown'))")
        page.locator("#room-Lx").focus()
        expect(page.locator("#wall-editor-x0")).to_contain_text("前牆材料")
        page.evaluate("() => document.dispatchEvent(new PointerEvent('pointercancel'))")
        expect(page.locator("#wall-editor-x0")).to_contain_text("x 起點牆材料")
        _assert_quiet(watched)


def test_a_late_rename_never_swaps_the_elevation_box_being_typed_in(tmp_path: Path, browser: Browser) -> None:
    # 換名也重畫立面圖（按鈕與深度上限那幾格都帶牆名）：游標在立面圖裡一樣先不換，離開才換。
    with _serve(tmp_path) as base, _open(browser, f"{base}/") as watched:
        page = watched.page
        page.locator("#elevation-wall-x0").click()
        held = _late_rename(page)
        # 寬 Ly 一離開就發超限檢查（房間尺寸改了，#1111），回來會重畫立面圖、照 id 把焦點放回去：那是另一條路，
        # 不是這裡考的，先等它畫完再點進去。
        page.locator("#room-Lx").focus()
        page.wait_for_function("() => zoneLimitsShown === zoneLimitsGeneration")
        depth = page.locator("#depth-room")
        depth.focus()
        box = depth.element_handle()
        _arrive(page, held)
        assert box.evaluate("node => node.isConnected && node === document.activeElement")
        expect(page.locator("#elevation-wall-x0")).to_have_text("前牆")
        page.locator("#room-Lx").focus()
        expect(page.locator("#elevation-wall-x0")).to_have_text("x 起點牆")
        expect(page.locator("#wall-editor-x0")).to_contain_text("x 起點牆材料")
        _assert_quiet(watched)


def test_a_draft_with_a_bad_wall_value_still_names_walls_and_its_problem_says_front_wall(tmp_path: Path) -> None:
    """別格填錯（x0 阻抗填負的）不影響前後左右；同一頁的檢查訊息跟表單同一套叫法。"""
    document = reference_document()
    walls = cast(dict[str, float], cast(Doc, document["scene"])["impedance_pa_s_per_m_by_wall"])
    walls["x0"] = -5.0
    assert draft_wall_words(document)["x0"] == "前牆"
    # 存檔名稱清空（準備改名）、用途清空、密度填錯：一樣叫前牆（審查實測這幾樣都會整組退回座標名）。
    renaming = reference_document()
    renaming["scheme_id"], renaming["purpose"] = "", ""
    cast(Doc, renaming["scene"])["density_kg_m3"] = -1.0
    del cast(Doc, renaming["scene"])["sound_speed_m_s"]
    del renaming["source_model"]
    assert draft_wall_words(renaming)["x0"] == "前牆"
    client = TestClient(create_app(GuiSettings(engine_commit="a" * 40, data_dir=tmp_path)), base_url="http://localhost")
    problems = cast(list[dict[str, str]], client.post("/api/validate", json=document).json()["problems"])
    texts = [problem["text"] for problem in problems]
    assert any(text.startswith("前牆阻抗") for text in texts), texts
    assert not any("x 起點牆" in text for text in texts), texts
    # 房間、喇叭、座位本身壞了才退回座標名。
    del document["speakers"]
    assert draft_wall_words(document) == WALLS


def test_gradient_preview_and_each_job_use_their_own_scheme_names(tmp_path: Path) -> None:
    """第三階段頁：預覽帶選的那份方案的牆名；工作清單跨方案，每一跑用自己那份快照的牆名。"""
    data_dir = _data(tmp_path)
    jobs = _jobs(data_dir, FakeLauncher())
    assert cast(dict[str, str], jobs.preview(NAME, GradientRequest())["wall_names"])["x0"] == "前牆"
    view = jobs.start(NAME, GradientRequest())
    assert cast(dict[str, str], view["wall_names"])["x0"] == "前牆"
    # 這一跑的快照換成喇叭靠 xL 的擺法（存著的方案不動）：看這一跑時 xL 是前牆。
    snapshot = data_dir / "gradient" / str(view["job_id"]) / "scheme.json"
    turned = _turned(Scheme.model_validate_json(snapshot.read_text(encoding="utf-8")).model_dump(mode="json"), 2)
    snapshot.write_text(turned.model_dump_json(), encoding="utf-8")
    names = cast(dict[str, str], jobs.get(str(view["job_id"]))["wall_names"])
    assert (names["xL"], names["x0"]) == ("前牆", "後牆")
