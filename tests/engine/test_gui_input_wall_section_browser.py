"""輸入頁第 2 段「牆上切區」（#1265 單元 3）：牆名鈕寫幾區、兩邊一起畫、牆那一行（換原本的牆、這一面另設、清掉與復原）、
右欄唯讀清單照做法分組與「改這一區 ▸」、「每一區的位置與大小」摺疊、第 2 段高度、每一格只在一個地方。

方案資料都是考卷自己編的：區名房間（_zone_naming_room，六面各種切法、門窗地毯）與 109 號房的長寬高（_surface_zone_cases）。
期待的字從資料算（幾區、哪幾種、名字），不寫死成某一間房的樣子。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
from playwright.sync_api import Browser, Page, Route, expect

from aosr.materials.library import builtin_entries
from aosr.reporting.scheme import Scheme
from tests.engine._input_page_steps import open_fact_editor, open_fold, open_scheme, reveal
from tests.engine._gui_cache import gui_startup_identity_memo as gui_startup_identity_memo
from tests.engine._surface_zone_cases import ROOM_109, room109_document
from tests.engine._zone_naming_room import ZONE_NAMING_SCHEME, write_zone_naming_library
from tests.engine.test_gui_browser import _assert_quiet, _open, _serve, browser as browser
from tests.engine.test_gui_zone_elevation_browser import _cut_by, _drag, _open_scheme, _write, _zones
from tests.engine.test_gui_zone_modes_browser import PRESET, _first_cut, _modes, _save, _write_room109_design

LY, LZ = ROOM_109["Ly"], ROOM_109["Lz"]
SAMPLE = "zone-naming-sample"
WALLS = ("x0", "xL", "y0", "yL", "floor", "ceiling")
Doc = dict[str, object]


def _sample(tmp_path: Path) -> None:
    (tmp_path / "schemes").mkdir(exist_ok=True)
    (tmp_path / "schemes" / f"{SAMPLE}.json").write_text(Scheme.model_validate(ZONE_NAMING_SCHEME).model_dump_json(),
                                                         encoding="utf-8")
    write_zone_naming_library(tmp_path)


def _sample_zones(wall: str) -> list[Doc]:
    return cast(list[Doc], cast(Doc, cast(Doc, ZONE_NAMING_SCHEME["scene"])["surface_zones_by_wall"])[wall])


def _settled(page: Page) -> None:
    page.wait_for_function("!document.querySelector('main').inert && !inputBusy")


def _wall(page: Page, wall: str) -> None:
    page.locator(f"#elevation-wall-{wall}").click()
    expect(page.locator(f"#elevation-wall-{wall}")).to_have_attribute("aria-pressed", "true")
    _settled(page)


def _stored_zones(tmp_path: Path, wall: str, name: str = "room109") -> list[Doc]:
    stored = json.loads((tmp_path / "schemes" / f"{name}.json").read_text(encoding="utf-8"))
    return cast(list[Doc], stored["scene"]["surface_zones_by_wall"][wall])


def _height(page: Page, selector: str) -> float:
    box = page.locator(selector).bounding_box()
    assert box is not None, selector
    return box["height"]


def test_each_wall_button_says_how_many_zones_that_wall_has(tmp_path: Path, browser: Browser) -> None:
    """牆名鈕：名字＋幾區（一區寫「整面」；門、窗、地毯照區名那一支認出來的寫名字，其餘寫「＋N 區」）。幾區照資料數。"""
    _sample(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        page.wait_for_function("document.body.dataset.inputReady === 'true'")
        open_scheme(page, SAMPLE)
        page.wait_for_function("name => document.getElementById('save-id').value === name", arg=SAMPLE)
        _settled(page)
        counts = {wall: len(_sample_zones(wall)) for wall in WALLS}
        # 區名房間：x0 整面一區；xL 有門與窗；地板有一塊地毯（材料庫那兩筆門窗、一筆地毯，區名那一支認得出來）。
        assert counts["x0"] == 1
        expected = {wall: f"{count} 區" for wall, count in counts.items()}
        expected |= {"x0": "整面", "xL": f"門、窗＋{counts['xL'] - 2} 區", "floor": f"地毯＋{counts['floor'] - 1} 區"}
        names = cast(dict[str, str], page.evaluate("structuredClone(wallNames)"))
        for wall in WALLS:
            button = page.locator(f"#elevation-wall-{wall}")
            expect(button.locator(".wall-name")).to_have_text(names[wall])
            expect(page.locator(f"#elevation-wall-count-{wall}")).to_have_text(expected[wall])
        _assert_quiet(watched)


def test_drawing_both_sides_switches_on_and_off_and_the_mirror_line_is_dash_dot(tmp_path: Path, browser: Browser) -> None:
    """「兩邊一起畫」是左右對稱改名：開著時立面圖上畫鏡子線（點劃線，跟補滿格子的灰虛線分得開），底下那一句寫鏡子在哪；
    關了鏡子線不見、那一句改講同一個開關在「這間房貼什麼」也有；開關存在設計設定（只存「關」）。"""
    _write(tmp_path, room109_document())
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        _wall(page, "x0")
        expect(page.locator("#elevation-symmetry-title")).to_have_text("兩邊一起畫")
        on, off = page.locator("#elevation-symmetry-on"), page.locator("#elevation-symmetry-off")
        expect(on).to_have_attribute("aria-pressed", "true")
        axis = page.locator("#elevation .mirror-axis line")
        expect(axis).to_have_count(1)
        assert axis.evaluate("line => getComputedStyle(line).strokeDasharray") == "10px, 4px, 2px, 4px"
        expect(page.locator("#elevation-axis-words")).to_contain_text("離左牆")
        off.click()
        expect(off).to_have_attribute("aria-pressed", "true")
        expect(page.locator("#elevation .mirror-axis")).to_have_count(0)
        expect(page.locator("#elevation-axis-words")).to_have_text("跟「這間房貼什麼」那一頁的「左右對稱」是同一個開關，哪一頁切換都一樣。")
        assert page.evaluate("zoneDesign.left_right_symmetry") == "off"
        on.click()
        expect(on).to_have_attribute("aria-pressed", "true")
        expect(page.locator("#elevation .mirror-axis line")).to_have_count(1)
        assert page.evaluate("zoneDesign.left_right_symmetry ?? null") is None
        _assert_quiet(watched)


def test_change_the_original_wall_opens_only_that_walls_form_and_saves(tmp_path: Path, browser: Browser) -> None:
    """牆那一行「原本的牆：…」＋「換原本的牆」：打開的是選中那一面的牆面表單（別面收著）；改成多層構造、存檔、重開，
    那一行寫預設構造的名字（牆面表單選單上那一個），表單裡也是它。"""
    _write(tmp_path, room109_document())
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        _wall(page, "x0")
        kinds = cast(dict[str, dict[str, str]], page.evaluate("structuredClone(inputDefaults.material_form)"))
        expect(page.locator("#elevation-original")).to_have_text(f"原本的牆：{kinds['constant_impedance']['label']}")
        expect(page.locator("#wall-original-editor")).to_be_hidden()
        open_fact_editor(page, "edit-wall-original")
        expect(page.locator("#wall-editor-x0")).to_be_visible()
        for wall in WALLS[1:]:
            expect(page.locator(f"#wall-editor-{wall}")).to_be_hidden()
        expect(page.locator("#material-kind-x0")).to_be_focused()
        # 切到別一面：表單跟著換成那一面（打開的那一塊照開著）。
        _wall(page, "y0")
        expect(page.locator("#wall-editor-y0")).to_be_visible()
        expect(page.locator("#wall-editor-x0")).to_be_hidden()
        _wall(page, "x0")
        preset = cast(str, page.evaluate("inputDefaults.construction_presets[0].label"))
        with page.expect_response("**/api/input-edit") as edited:
            page.locator("#material-kind-x0").select_option("assembly")
        assert edited.value.ok, edited.value.text()
        _settled(page)
        expect(page.locator("#elevation-original")).to_have_text(f"原本的牆：{preset}")
        with page.expect_response(lambda r: r.url.endswith("/api/schemes/room109") and r.request.method == "PUT") as saved:
            page.locator("#save").click()
        assert saved.value.ok, saved.value.text()
        assert cast(Doc, _stored_zones(tmp_path, "x0")[0]["material"])["kind"] == "assembly"
        page.reload()
        page.wait_for_function("document.body.dataset.inputReady === 'true'")
        _open_scheme(page, "room109")
        _wall(page, "x0")
        expect(page.locator("#elevation-original")).to_have_text(f"原本的牆：{preset}")
        open_fact_editor(page, "edit-wall-original")
        expect(page.locator("#material-kind-x0")).to_have_value("assembly")
        _assert_quiet(watched)


def test_this_wall_depth_is_typed_in_centimetres_saved_in_metres_and_is_what_the_zone_page_reads(
        tmp_path: Path, browser: Browser) -> None:
    """牆那一行「貼的東西總厚度…（照房間預設）」＋「這一面另設」：打開的那一格用公分打，存的是公尺；
    那一行改寫「（這一面另設）」；存檔、重開都在；「這間房貼什麼」那一頁判限制送的也是同一份。"""
    _write_room109_design(tmp_path, {"left_right_symmetry": "off", "room_max_depth_m": 0.2})
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        _wall(page, "x0")
        line = page.locator("#elevation-depth")
        expect(line).to_have_text("貼的東西總厚度最多 20 公分（照房間預設）")
        open_fact_editor(page, "edit-wall-depth")
        box = page.locator("#depth-wall-x0")
        expect(box).to_be_focused()
        expect(box).to_have_value("")
        before = cast(int, page.evaluate("zoneLimitsGeneration"))
        box.fill("12")
        box.press("Enter")
        page.wait_for_function("before => zoneLimitsGeneration > before && zoneLimitsShown === zoneLimitsGeneration",
                               arg=before)
        assert page.evaluate("zoneDesign.wall_max_depth_m") == {"x0": 0.12}
        expect(line).to_have_text("貼的東西總厚度最多 12 公分（這一面另設）")
        # 別一面沒另設：照房間預設。
        _wall(page, "y0")
        expect(line).to_have_text("貼的東西總厚度最多 20 公分（照房間預設）")
        _wall(page, "x0")
        _save(page, "room109")
        stored = json.loads((tmp_path / "designs" / "room109.json").read_text(encoding="utf-8"))
        assert stored["wall_max_depth_m"] == {"x0": 0.12} and stored["room_max_depth_m"] == 0.2
        page.reload()
        page.wait_for_function("document.body.dataset.inputReady === 'true'")
        _open_scheme(page, "room109")
        _wall(page, "x0")
        expect(line).to_have_text("貼的東西總厚度最多 12 公分（這一面另設）")
        open_fact_editor(page, "edit-wall-depth")
        expect(page.locator("#depth-wall-x0")).to_have_value("12")
        page.locator("#zone-page-open").click()
        expect(page.locator("#zone-page")).to_be_visible()
        with page.expect_request("**/api/zone-limits") as sent:
            page.evaluate("refreshZoneLimits()")
        assert cast(Doc, cast(Doc, sent.value.post_data_json)["design"])["wall_max_depth_m"] == {"x0": 0.12}
        assert page.request.get(f"{base}/api/designs/room109").json()["design"]["wall_max_depth_m"] == {"x0": 0.12}
        _assert_quiet(watched)


def test_a_refused_wall_depth_goes_back_to_the_last_confirmed_value(tmp_path: Path, browser: Browser) -> None:
    """單元 2 審查發現 1 的同一支 depthInput（這一面那一格）：確認過 15 公分、判決還在路上時再填 -1 按 Enter，
    格子退回 15（最後一次確認的值），不是建格當時的空白；不算打到一半；提示句寫在那一格底下。"""
    _write_room109_design(tmp_path, {"left_right_symmetry": "off"})
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        _wall(page, "x0")
        open_fact_editor(page, "edit-wall-depth")
        held: list[Route] = []
        page.route("**/api/zone-limits", lambda route: held.append(route))
        box = page.locator("#depth-wall-x0")
        box.fill("15")
        box.press("Enter")
        assert page.evaluate("zoneDesign.wall_max_depth_m") == {"x0": 0.15}
        box.fill("-1")
        box.press("Enter")
        expect(box).to_have_value("15")
        assert page.evaluate("zoneDesign.wall_max_depth_m") == {"x0": 0.15}
        assert page.evaluate("Boolean(typingInElevation())") is False
        expect(page.locator("#depth-wall-note")).to_have_text("總厚度上限要填大於 0 的數字（公分，最少 0.1）；空白表示照房間預設")
        for route in held:
            route.continue_()
        page.unroute("**/api/zone-limits")
        page.wait_for_function("zoneLimitsShown === zoneLimitsGeneration")
        # 判決回來重畫了那一格：提示句照留（還沒處理）；改回合法值才消失。
        expect(page.locator("#depth-wall-note")).to_contain_text("大於 0")
        box = page.locator("#depth-wall-x0")
        box.fill("10")
        box.press("Enter")
        expect(page.locator("#depth-wall-note")).to_have_text("")
        assert page.evaluate("zoneDesign.wall_max_depth_m") == {"x0": 0.1}
        _assert_quiet(watched)


def _snapshot(page: Page) -> Doc:
    return cast(Doc, page.evaluate("structuredClone({zones: scheme.scene.surface_zones_by_wall, design: zoneDesign})"))


def test_clearing_a_wall_comes_back_exactly_with_undo_until_the_next_change(tmp_path: Path, browser: Browser) -> None:
    """清掉這一面的切法：紅字連結，清掉後同一處寫「已清掉前牆 N 區｜復原」；按復原，六面的切法與每一區的做法逐項回到清掉前；
    有沒有改動照存檔規則（清掉、復原都算改動）。復原只到下一個改動之前：清掉之後改了別的，復原就不見。"""
    _write_room109_design(tmp_path, {"left_right_symmetry": "off"})
    _write(tmp_path, room109_document("room109-original"))
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        _wall(page, "x0")
        expect(page.locator("#elevation-clear")).to_be_disabled()  # 整面一區：沒有可清的
        _first_cut(page, tmp_path)
        _save(page, "room109")
        expect(page.locator("#save-state")).to_have_attribute("data-kind", "saved")
        before = _snapshot(page)
        count = len(_zones(page, "x0"))
        name = cast(str, page.evaluate("wallNames.x0"))
        clear = page.locator("#elevation-clear")
        expect(clear).to_have_text("清掉這一面的切法")
        _cut_by(page, "#elevation-clear")
        assert [zone["area"] for zone in _zones(page, "x0")] == ["whole_wall"]
        assert _modes(page, "x0") != cast(dict[str, Doc], cast(Doc, before["design"])["zones"])["x0"]
        expect(page.locator("#elevation-cleared")).to_have_text(f"已清掉{name} {count} 區｜復原")
        expect(page.locator("#elevation-clear")).to_have_count(0)
        expect(page.locator("#save-state")).to_have_attribute("data-kind", "dirty")
        page.locator("#elevation-undo-clear").click()
        expect(page.locator("#elevation-note")).to_have_text(f"復原了：{name}回到清掉前的 {count} 區")
        _settled(page)
        after = _snapshot(page)
        # 逐項比：六面每一區（位置、材料）、設計設定每一格（每一區的做法、清單、原方案、對稱開關）。
        for wall in WALLS:
            assert cast(Doc, after["zones"])[wall] == cast(Doc, before["zones"])[wall], wall
        for key in set(cast(Doc, before["design"])) | set(cast(Doc, after["design"])):
            assert cast(Doc, after["design"]).get(key) == cast(Doc, before["design"]).get(key), key
        expect(page.locator("#elevation-undo-clear")).to_have_count(0)
        expect(page.locator("#elevation-clear")).to_be_enabled()
        expect(page.locator("#save-state")).to_have_attribute("data-kind", "dirty")
        # 存下去的也跟清掉前存的那一份一樣。
        saved_design = (tmp_path / "designs" / "room109.json").read_text(encoding="utf-8")
        _save(page, "room109")
        assert _stored_zones(tmp_path, "x0") == cast(Doc, before["zones"])["x0"]
        assert json.loads((tmp_path / "designs" / "room109.json").read_text(encoding="utf-8")) == json.loads(saved_design)

        # 清掉之後先改了別的（主位那一格）：復原就沒了，回到「清掉這一面的切法」。
        _cut_by(page, "#elevation-clear")
        expect(page.locator("#elevation-undo-clear")).to_be_visible()
        primary = reveal(page, "#receiver-main-z")
        primary.fill(str(float(primary.input_value()) + 0.05))
        expect(page.locator("#elevation-undo-clear")).to_have_count(0)
        expect(page.locator("#elevation-clear")).to_be_visible()
        _assert_quiet(watched)


def test_the_list_counts_the_zones_and_each_goto_opens_that_zone(tmp_path: Path, browser: Browser) -> None:
    """右欄「前牆這 N 區」只列出來看：切之前「前牆整面一區」、牆名鈕寫「整面」；切成三區後兩處都跟著變。每一區一顆
    「改這一區 ▸」（整頁只有這一處，住在立面圖那一塊裡、舊做法表裡沒有），按了打開「這間房貼什麼」那一區。"""
    _write_room109_design(tmp_path, {"left_right_symmetry": "off"})
    _write(tmp_path, room109_document("room109-original"))
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        _wall(page, "x0")
        name = cast(str, page.evaluate("wallNames.x0"))
        expect(page.locator("#elevation-list-title")).to_have_text(f"{name}整面一區")
        expect(page.locator("#elevation-wall-count-x0")).to_have_text("整面")
        _first_cut(page, tmp_path)
        keys = [str(zone["id"]) for zone in _zones(page, "x0")]
        expect(page.locator("#elevation-list-title")).to_have_text(f"{name}這 {len(keys)} 區")
        expect(page.locator("#elevation-wall-count-x0")).to_have_text(f"{len(keys)} 區")  # 切了，牆名鈕上的區數跟著變
        expect(page.locator("#elevation-list-zones .zone-item")).to_have_count(len(keys))
        for key in keys:
            expect(page.locator(f"#zone-goto-x0-{key}")).to_have_count(1)
            expect(page.locator(f"#elevation-panel #elevation-list #wall-zone-x0-{key} #zone-goto-x0-{key}")).to_have_count(1)
            expect(page.locator(f"#zone-goto-x0-{key}")).to_have_text("改這一區 ▸")
        expect(page.locator("#zone-table-fold [id^=zone-goto-]")).to_have_count(0)
        page.locator("#zone-goto-x0-auto1").click()
        expect(page.locator("#zone-panel")).to_have_attribute("data-zones", "x0/auto1")
        _assert_quiet(watched)


# 三組各一題：_first_cut 切出的三區各給一種做法（z1 指定貼一種、auto1 由程式最佳化、auto2 保留原樣）。
GROUPS = {"search": "auto1", "fixed": "z1", "still": "auto2"}


@pytest.mark.parametrize("group", list(GROUPS))
def test_the_list_puts_each_zone_in_its_group_with_words_from_the_data(tmp_path: Path, browser: Browser,
                                                                        group: str) -> None:
    """右欄清單照做法分組，組的順序照示意圖（由程式最佳化、指定貼一種、保留原樣）；每一區那一句的字從資料來：
    由程式最佳化寫從幾種裡挑、是哪幾種（那一區存的清單）、現在存的直接算就照它算；指定貼一種寫貼的那一筆（材料庫的名字）；
    保留原樣寫不貼。號碼與名字跟立面圖上那一塊同一個。"""
    _write_room109_design(tmp_path, {"left_right_symmetry": "off"})
    _write(tmp_path, room109_document("room109-original"))
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        _wall(page, "x0")
        _first_cut(page, tmp_path)
        order = [cast(str, block.get_attribute("id")) for block in page.locator("#elevation-list-zones > .zone-group").all()]
        assert order == ["wall-zones-search", "wall-zones-fixed", "wall-zones-still"]
        key = GROUPS[group]
        block = page.locator(f"#wall-zones-{group}")
        expect(block.locator(".zone-item")).to_have_count(1)
        row = block.locator(f"#wall-zone-x0-{key}")
        words = cast(dict[str, str], page.evaluate("structuredClone(ZONE_MODE_WORDS)"))
        expect(row.locator(".tag")).to_have_text({"search": words["search"], "fixed": words["fixed"], "still": "保留原樣"}[group])
        badge = page.locator(f"#elevation .zone-number[data-zone-key={key}]")
        expect(row.locator(".zone-num")).to_have_text(cast(str, badge.get_attribute("data-number")))
        expect(row.locator(f"#wall-zone-name-x0-{key}")).to_have_text(
            cast(str, page.evaluate("key => zoneLabels('x0').get(key).name", key)))
        what = row.locator(f"#wall-zone-what-x0-{key}")
        setting = _modes(page, "x0")[key]
        if group == "search":
            allowed = cast(list[str], setting["allowed"])
            names = cast(str, page.evaluate("allowed => allowed.map(choiceName).join('、')", allowed))
            expect(what).to_contain_text(f"已交給程式挑：從 {len(allowed)} 種裡挑（{names}）")
            expect(what).to_contain_text("直接算就照它算")
        elif group == "fixed":
            assert setting["mode"] == "fixed"
            label = next(entry.label for entry in builtin_entries() if entry.id == PRESET)
            expect(what).to_contain_text(f"已貼 {label}")
        else:
            assert setting["mode"] == "keep_original"
            expect(what).to_contain_text("不貼")
        _assert_quiet(watched)


def test_the_numbers_fold_matches_the_drawing(tmp_path: Path, browser: Browser) -> None:
    """「每一區的位置與大小（公尺）」預設收著；打開後每一區一列：號碼名字跟右欄清單同一個，離左、離地、寬 × 高跟立面圖上
    那一塊量出來的一樣（從圖上的方塊與牆框換算，不重抄程式的換算）；現在貼的寫牆面表單那種寫法（固定阻抗帶數字）。"""
    _sample(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        page.wait_for_function("document.body.dataset.inputReady === 'true'")
        open_scheme(page, SAMPLE)
        page.wait_for_function("name => document.getElementById('save-id').value === name", arg=SAMPLE)
        _settled(page)
        for wall in ("yL", "xL"):
            _wall(page, wall)
            fold = page.locator("#zone-numbers-fold")
            assert fold.evaluate("fold => fold.open") is False
            table = page.locator(f"#zoned-list-{wall}")
            expect(table).to_be_hidden()
            open_fold(page, "zone-numbers-fold")
            zones = _sample_zones(wall)
            expect(table.locator("tr")).to_have_count(len(zones) + 1)
            frame = cast(dict[str, float], page.locator("#elevation-frame").evaluate(
                "rect => ({x: rect.x.baseVal.value, y: rect.y.baseVal.value, w: rect.width.baseVal.value, h: rect.height.baseVal.value})"))
            width = cast(float, page.evaluate("wall => elevationGeometry.wallSize(wall)[0]", wall))
            per = width / frame["w"]
            for zone in zones:
                key = cast(str, zone["id"])
                cells = page.locator(f"#zone-size-{wall}-{key} td").all_inner_texts()
                number = page.locator(f"#wall-zone-{wall}-{key} .zone-num").inner_text()
                label = page.locator(f"#wall-zone-name-{wall}-{key}").inner_text()
                assert cells[0] == f"{chr(0x2460 + int(number) - 1)} {label}", cells
                box = cast(dict[str, float], page.locator(f"#elevation [data-zone-id={key}] > rect").evaluate(
                    "rect => ({x: rect.x.baseVal.value, y: rect.y.baseVal.value, w: rect.width.baseVal.value, h: rect.height.baseVal.value})"))
                left, bottom = (box["x"] - frame["x"]) * per, (frame["y"] + frame["h"] - box["y"] - box["h"]) * per
                size = [float(part) for part in cells[3].split(" × ")]
                for shown, measured in ((cells[1], left), (cells[2], bottom), (size[0], box["w"] * per), (size[1], box["h"] * per)):
                    assert abs(float(shown) - measured) < 0.002, (key, cells, measured)
            # 現在貼的：牆面表單的寫法（固定阻抗帶數字與單位；數字照只給人看的收位寫法帶千分位，#1119）。
            plain = [zone for zone in zones if cast(Doc, zone["material"])["kind"] == "constant_impedance"]
            for zone in plain:
                impedance = cast(Doc, zone["material"])["impedance_pa_s_m"]
                expect(page.locator(f"#zone-size-{wall}-{zone['id']} td").nth(4)).to_contain_text(f"{impedance:,g}")
            fold.locator("> summary").click()
        _assert_quiet(watched)


def test_dragging_a_box_shows_the_snap_and_smallest_side_beside_it_until_release(tmp_path: Path, browser: Browser) -> None:
    """homes.md「框邊吸附 5 公分格、最小邊 20 公分」那一列的「拖的時候框旁的小字」：按著滑鼠拖框時，框的左上角旁寫吸附格與
    最小邊（數字從伺服器給的切牆規格讀，不寫死）；放開就收掉。"""
    _write(tmp_path, room109_document())
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        _wall(page, "x0")
        spec = cast(Doc, page.evaluate("inputDefaults.zone_cut"))
        said = (f"框邊吸附 {round(cast(float, spec['snap_m']) * 100)} 公分格，"
                f"最小邊 {round(cast(float, spec['min_side_m']) * 100)} 公分")
        hint, preview = page.locator("#elevation .zone-preview-hint"), page.locator("#elevation .zone-preview")

        def during() -> None:
            expect(hint).to_have_text(said)
            assert hint.get_attribute("x") == preview.get_attribute("x")  # 跟著框的左緣走

        _drag(page, LY, LZ, (0.5, 0.5), (1.5, 1.6), during=during)
        expect(hint).to_have_count(0)
        _assert_quiet(watched)


def test_section_two_fits_in_800_pixels_at_1440_by_900(tmp_path: Path, browser: Browser) -> None:
    """1440×900、開頁預設（範本、摺疊收著）：第 2 段不超過 800 像素——開頁沒選牆、選了前牆都量；
    區名房間切得最多的那一面（門、窗＋6 區）也量。整頁高度另在回報裡量。"""
    _sample(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        page.set_viewport_size({"width": 1440, "height": 900})
        page.wait_for_function("document.body.dataset.inputReady === 'true'")
        _settled(page)
        heights = {"開頁": _height(page, "#wall-section")}
        _wall(page, "x0")
        heights["範本前牆"] = _height(page, "#wall-section")
        open_scheme(page, SAMPLE)
        page.wait_for_function("name => document.getElementById('save-id').value === name", arg=SAMPLE)
        _settled(page)
        for wall in WALLS:
            _wall(page, wall)
            heights[f"區名房間 {wall}"] = _height(page, "#wall-section")
        assert all(height <= 800 for height in heights.values()), heights
        _assert_quiet(watched)


def test_every_box_in_section_two_lives_in_one_place(tmp_path: Path, browser: Browser) -> None:
    """搬過的每一格整頁只有一個：牆面表單只在「換原本的牆」那一塊、這一面的總厚度上限只在「這一面另設」那一塊、
    房間的只在第 1 段右卡、進「這間房貼什麼」只剩第 2 段右欄那一顆、每一區的位置大小只在摺疊裡；整頁沒有重複的 id。"""
    _write(tmp_path, room109_document())
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        _wall(page, "x0")
        open_fact_editor(page, "edit-wall-original")
        open_fact_editor(page, "edit-wall-depth")
        open_fact_editor(page, "edit-depth")
        open_fold(page, "zone-numbers-fold")
        places = {"#material-kind-x0": "#wall-original-editor", "#depth-wall-x0": "#wall-depth-editor",
                  "#depth-room": "#depth-editor", "#zone-page-open": "#elevation-list",
                  "#zoned-list-x0": "#zone-numbers-fold", "#zone-goto-x0-whole_wall": "#elevation-list",
                  "#elevation-clear": "#elevation-wall-line", "#elevation-symmetry-on": "#elevation-panel"}
        for box, holder in places.items():
            expect(page.locator(box)).to_have_count(1)
            expect(page.locator(f"{holder} {box}")).to_have_count(1)
        # 用標籤字找也只有一個（同一句標籤不准在別處再掛一格，例如舊做法表底下還留一格這一面的總厚度上限）。
        name = cast(str, page.evaluate("wallNames.x0"))
        labels = {f"{name}材料": "#material-kind-x0", f"{name}總厚度最多（公分；空白表示照房間預設）": "#depth-wall-x0",
                  cast(str, page.locator("#depth-room").evaluate("box => box.labels[0].firstChild.textContent")): "#depth-room"}
        for label, box in labels.items():
            expect(page.get_by_label(label)).to_have_count(1)
            assert page.get_by_label(label).evaluate("node => '#' + node.id") == box, label
        expect(page.get_by_role("button", name="設定每一區貼什麼 ▸")).to_have_count(1)
        expect(page.locator("#scheme-zone-open")).to_have_count(0)
        expect(page.locator("#zone-table-fold #depth-wall-x0, #zone-table-fold [id^=zone-goto-]")).to_have_count(0)
        duplicated = page.evaluate("""() => { const ids = [...document.querySelectorAll('[id]')].map(node => node.id);
          return ids.filter((id, index) => ids.indexOf(id) !== index); }""")
        assert duplicated == []
        _assert_quiet(watched)
