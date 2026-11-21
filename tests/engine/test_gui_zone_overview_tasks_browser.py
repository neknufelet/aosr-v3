"""真瀏覽器照「這間房貼什麼」那一頁的畫面按，比存檔的內容（#1176 第 C 支）。

資料全是合成的：區名考卷那間假房間（``_zone_naming_room`` 的 zone-naming-sample：前牆 yL、後牆 y0、左牆 x0 整面一區、
右牆 xL 有門有窗）配它的 5 筆假材料，加上合成材料庫 12 筆（``_sample_material_library``，裡面有名字以 VPR 開頭的三筆）。
假房間的左牆只有一區，所以樣稿那句「只有左牆③由程式最佳化」在這裡改用右牆③（xL 的 auto4）。

兩句任務各一題（按完比設計檔每一區的做法與清單、方案裡那幾區的材料）、打開不改就存逐位不變、改清單寫進每一個由程式最佳化的區、
伺服器回應晚到重畫時選取與剛點的選項還在、VPR 帶中文、選到天花時材料只列天花能貼的。"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import cast

from playwright.sync_api import Browser, Page, Route, expect
from pydantic import TypeAdapter

from aosr.materials.library import LibraryEntry, dump_user_library
from tests.engine._construction_cases import construction
from tests.engine._gui_cache import gui_startup_identity_memo as gui_startup_identity_memo
from tests.engine._sample_material_library import SAMPLE_USER_ENTRIES
from tests.engine._zone_naming_room import ZONE_NAMING_LIBRARY, ZONE_NAMING_SCHEME
from tests.engine.test_gui_browser import _open, _serve, browser as browser
from tests.engine.test_gui_zone_elevation_browser import _open_scheme, _write
from tests.engine.test_gui_zone_overview_browser import SAMPLE, SAMPLE_DESIGN, SHORT, WOOLS

VPR = "pet60_vpr_steel1p2mm_foam90_sample"
PRODUCT = "slotted_wood_typ9_absorber40_gap98"  # 內建的一筆一般固定成品（牆面能貼、不被可調玻璃棉包含）
KEEP = {"mode": "keep_original"}
SEARCH = {"mode": "search", "allowed": WOOLS}
_SELECTED = """() => [...document.querySelectorAll('#zone-map [data-selected=true]')].map(zone => `${zone.dataset.wall}/${zone.dataset.zone}`)"""
_GROUPS = """() => [...document.querySelectorAll('#zone-special .zv-group')].map(group =>
    ({zones: group.dataset.zones.split(' '), what: group.querySelector('.zv-what').textContent}))"""
_ALLOWED = """() => Object.fromEntries(Object.entries(zoneDesign.zones).flatMap(([wall, zones]) => Object.entries(zones)
    .filter(([, setting]) => setting.mode === 'search').map(([key, setting]) => [`${wall}/${key}`, setting.allowed])))"""
_SHOWN_OPTION = "document.getElementById('zone-panel-material').selectedOptions[0].textContent"
# 下拉收起來那一格：顯示的那一筆的字寬（照那一格自己的字型量），與格子的寬（右邊還有箭頭與留白）。
_SHOWN_WIDTHS = """() => {
    const select = document.getElementById('zone-panel-material'), style = getComputedStyle(select);
    const pen = document.createElement('canvas').getContext('2d');
    pen.font = `${style.fontWeight} ${style.fontSize} ${style.fontFamily}`;
    return [pen.measureText(select.selectedOptions[0].textContent).width, select.clientWidth];
}"""


def _library(tmp_path: Path) -> None:
    adapter: TypeAdapter[tuple[LibraryEntry, ...]] = TypeAdapter(tuple[LibraryEntry, ...])
    entries = adapter.validate_python([*ZONE_NAMING_LIBRARY, *SAMPLE_USER_ENTRIES])
    (tmp_path / "materials.json").write_text(dump_user_library(entries), encoding="utf-8")


def _prepare(tmp_path: Path, design: dict[str, object] | None = None) -> None:
    _write(tmp_path, ZONE_NAMING_SCHEME)
    _library(tmp_path)
    if design is not None:
        (tmp_path / "designs").mkdir(exist_ok=True)
        (tmp_path / "designs" / f"{SAMPLE}.json").write_text(json.dumps(design), encoding="utf-8")


def _open_overview(page: Page) -> None:
    _open_scheme(page, SAMPLE)
    page.locator("#zone-page-open").click()
    expect(page.locator("#zone-map")).to_be_visible()


def _click_zone(page: Page, zone: str) -> None:
    wall, key = zone.split("/")
    page.locator(f"#zone-map [data-wall='{wall}'][data-zone='{key}'] .zv-fill").click()


def _settled(page: Page) -> None:
    """一批改動送完、畫面放開、平面圖檢查回來了（成不成都算）、限制檢查也回來畫好了。"""
    page.wait_for_function("""() => { const page = document.getElementById('zone-page');
        return !page.inert && !inputBusy && !(Number(page.dataset.changing) > 0) && zoneLimitsShown === zoneLimitsGeneration; }""")


def _apply(page: Page) -> None:
    page.locator("#zone-panel-apply").click()
    expect(page.locator("#zone-panel")).to_have_count(0)
    _settled(page)


def _send_to_search(page: Page) -> None:
    with page.expect_response(lambda r: r.url.endswith(f"/api/designs/{SAMPLE}") and r.request.method == "PUT") as design:
        page.locator("#zone-page-search").click()
    assert design.value.ok
    page.wait_for_url(f"**/gradient?scheme={SAMPLE}")


def _stored(tmp_path: Path) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, dict[str, object]]]]:
    """存檔的設計檔每一區，與方案檔每一區的材料（牆 → 區代號 → 材料）。"""
    design = json.loads((tmp_path / "designs" / f"{SAMPLE}.json").read_text(encoding="utf-8"))
    scheme = json.loads((tmp_path / "schemes" / f"{SAMPLE}.json").read_text(encoding="utf-8"))
    materials = {wall: {zone.get("id", "whole_wall"): zone["material"] for zone in zones}
                 for wall, zones in scheme["scene"]["surface_zones_by_wall"].items()}
    return design["zones"], materials


def _materials_before() -> dict[str, dict[str, dict[str, object]]]:
    scene = cast(dict[str, dict[str, list[dict[str, object]]]], ZONE_NAMING_SCHEME["scene"])
    return {wall: {cast(str, zone.get("id", "whole_wall")): cast(dict[str, object], zone["material"]) for zone in zones}
            for wall, zones in scene["surface_zones_by_wall"].items()}


def test_batch_material_pick_shows_the_recorded_construction_note(tmp_path: Path, browser: Browser) -> None:
    _prepare(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        entry = construction("rigid", id="batch_panel", label="自編硬背板")
        assert page.request.put(f"{base}/api/materials/batch_panel", data=entry).ok
        _open_overview(page)
        _click_zone(page, "yL/z1")
        _click_zone(page, "yL/z2")
        page.locator("#zone-panel-material").select_option("batch_panel")
        _apply(page)
        note = page.locator("#zone-page-note")
        expect(note).to_be_visible()
        expect(note).to_contain_text("2 區改成")
        expect(note).to_contain_text("記成選上去的「自編硬背板」，整塊構造直接取代這一區的牆面。")


def test_task_one_front_panels_fixed_and_the_other_walls_handed_to_the_computer(tmp_path: Path, browser: Browser) -> None:
    """「前牆兩塊板指定貼一種（合成庫裡名字以 VPR 開頭的那一筆），其他牆面由程式最佳化（三種內建玻璃棉、也可以不貼）；
    門、窗、天花、地板不動」：點前牆兩塊 → 指定貼一種 → 套用；「其他牆面」下拉改成由程式最佳化；剩下兩塊原本貼著別的，
    點它們 → 由程式最佳化（標著「＝其他牆面」的那一項）→ 套用；拿去搜尋（先存檔）。"""
    _prepare(tmp_path)
    before = _materials_before()
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        _click_zone(page, "yL/z1")
        _click_zone(page, "yL/z2")
        expect(page.locator("#zone-panel h2")).to_have_text("已選 2 區")
        assert page.evaluate(_SELECTED) == ["yL/z1", "yL/z2"]
        # 面板不蓋圖：整塊在圖的右邊。
        panel, chart = page.locator("#zone-panel").bounding_box(), page.locator("#zone-map").bounding_box()
        assert panel is not None and chart is not None and panel["x"] >= chart["x"] + chart["width"]
        page.locator("#zone-panel-material").select_option(VPR)
        expect(page.locator("#zone-panel-mode-fixed")).to_be_checked()
        expect(page.locator("#zone-panel-apply")).to_have_text("套用到選中的 2 區")
        _apply(page)
        groups = cast(list[dict[str, object]], page.evaluate(_GROUPS))
        assert groups[0]["zones"] == ["yL/z1", "yL/z2"]
        # VPR 第一次出現帶中文；圖上那一塊「VPR」底下一行小字。
        assert cast(str, groups[0]["what"]).startswith("指定貼一種：VPR（鋼板共振吸音）＋前面聚酯棉 60 mm")
        words = page.locator("#zone-map [data-wall='yL'][data-zone='z1'] .zv-fixed-words text").all_text_contents()
        assert words[:2] == ["VPR", "鋼板共振吸音"], words
        # 改規則：一次寫到所有沒另外指定的區。
        page.locator("#zone-rule-mode").select_option("search")
        _settled(page)
        expect(page.locator("#zone-rule")).to_have_attribute("data-mode", "search")
        assert [group["zones"] for group in cast(list[dict[str, object]], page.evaluate(_GROUPS))] == [
            ["yL/z1", "yL/z2"], ["yL/z3"], ["y0/z1"]]
        # 原本貼著別的那兩塊：選「＝其他牆面」那一項，就不再另外指定。
        _click_zone(page, "yL/z3")
        _click_zone(page, "y0/z1")
        expect(page.locator("#zone-panel-same")).to_have_count(1)
        expect(page.locator("#zone-panel-option-search #zone-panel-same")).to_have_text("＝其他牆面")
        page.locator("#zone-panel-mode-search").check()
        page.screenshot(path=str(tmp_path / "task-one-same-as-rule.png"))
        _apply(page)
        assert [group["zones"] for group in cast(list[dict[str, object]], page.evaluate(_GROUPS))] == [["yL/z1", "yL/z2"]]
        expect(page.locator("#zone-list-who")).to_have_text("現在管 15 區：沒另外指定的牆區")
        assert page.locator("#zone-list-checks label").evaluate_all("rows => rows.map(row => row.dataset.choice)") == WOOLS
        expect(page.locator("#zone-result-fixed")).to_contain_text("指定貼 4 區")
        page.screenshot(path=str(tmp_path / "task-one-done.png"))
        _send_to_search(page)
        assert watched.page_errors == [] and watched.console_errors == []
    design, materials = _stored(tmp_path)
    assert design == {
        "yL": {"z1": {"mode": "fixed"}, "z2": {"mode": "fixed"}, "z3": SEARCH, "auto1": SEARCH, "auto2": SEARCH},
        "y0": {key: SEARCH for key in ("z1", "auto1", "auto2", "auto3", "auto4")},
        "xL": {f"auto{n}": SEARCH for n in range(1, 7)},
        "x0": {"whole_wall": SEARCH},
    }
    vpr_layers = next(cast(dict[str, dict[str, object]], entry)["material"]["layers"] for entry in SAMPLE_USER_ENTRIES if entry["id"] == VPR)
    for key in ("z1", "z2"):
        assert materials["yL"][key]["kind"] == "mounted_on_wall"
        stored_layers = cast(list[dict[str, object]], materials["yL"][key]["layers"])
        assert len(stored_layers) == len(cast(list[object], vpr_layers))
        # 每一層連數值一起比：材料庫那一筆寫的每一格（種類、厚度、流阻、面密度…），存下來的那一層都是同一個值。
        for stored_layer, layer in zip(stored_layers, cast(list[dict[str, object]], vpr_layers)):
            assert {name: stored_layer[name] for name in layer} == layer
    # 由程式最佳化的區放回原樣（原本貼著東西的那兩塊也是）；門、窗、天花、地板一個字都沒動。
    assert materials["yL"]["z3"]["kind"] == "constant_impedance" and materials["y0"]["z1"]["kind"] == "constant_impedance"
    assert materials["xL"]["door"] == before["xL"]["door"] and materials["xL"]["window"] == before["xL"]["window"]
    assert materials["ceiling"] == before["ceiling"] and materials["floor"] == before["floor"]


def test_task_two_other_walls_bare_and_only_one_zone_handed_to_the_computer(tmp_path: Path, browser: Browser) -> None:
    """「其他牆面都不貼，只有右牆③由程式最佳化；前牆兩塊照舊」：從其他牆面由程式最佳化的那份設計開始。「其他牆面」下拉改成
    不貼（前牆兩塊是另外指定的，不動）；後牆清單不一樣的那一區還是另外指定，點它 → 不貼（＝其他牆面）；點右牆③ →
    由程式最佳化 → 套用（清單照畫面上那一份，也就是天花那一區用的）；拿去搜尋。"""
    _prepare(tmp_path, SAMPLE_DESIGN)
    before = _materials_before()
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        page.locator("#zone-rule-mode").select_option("keep_original")
        _settled(page)
        expect(page.locator("#zone-rule")).to_have_attribute("data-mode", "keep_original")
        assert [group["zones"] for group in cast(list[dict[str, object]], page.evaluate(_GROUPS))] == [["yL/z1", "yL/z2"], ["y0/auto1"]]
        page.locator("#zone-special-edit-1").click()
        assert page.evaluate(_SELECTED) == ["y0/auto1"]
        expect(page.locator("#zone-panel-mode-search")).to_be_checked()
        expect(page.locator("#zone-panel-option-keep_original #zone-panel-same")).to_have_text("＝其他牆面")
        expect(page.locator("#zone-panel-same")).to_have_count(1)
        page.locator("#zone-panel-mode-keep_original").check()
        _apply(page)
        _click_zone(page, "xL/auto4")
        expect(page.locator("#zone-panel .zv-chip")).to_have_text("右牆 3 靠前牆那條不選這區")
        expect(page.locator("#zone-panel-mode-keep_original")).to_be_checked()
        page.locator("#zone-panel-mode-search").check()
        _apply(page)
        assert [group["zones"] for group in cast(list[dict[str, object]], page.evaluate(_GROUPS))] == [["yL/z1", "yL/z2"], ["xL/auto4"]]
        expect(page.locator("#zone-list-who")).to_have_text("現在管 2 區：右牆 3 靠前牆那條、天花 1 主位上面那塊")
        expect(page.locator("#zone-result-still")).to_contain_text("牆面 14 區")
        page.screenshot(path=str(tmp_path / "task-two-done.png"))
        _send_to_search(page)
        assert watched.page_errors == [] and watched.console_errors == []
    design, materials = _stored(tmp_path)
    assert design == {
        "yL": {"z3": KEEP, "auto1": KEEP, "auto2": KEEP},
        "y0": {key: KEEP for key in ("z1", "auto1", "auto2", "auto3", "auto4")},
        "xL": {f"auto{n}": SEARCH if n == 4 else KEEP for n in range(1, 7)},
        "x0": {"whole_wall": KEEP},
        "ceiling": {"z1": SEARCH},
    }
    # 前牆兩塊照舊（設計檔沒列、材料沒動）；改成不貼的那幾區放回原樣。
    assert materials["yL"]["z1"] == before["yL"]["z1"] and materials["yL"]["z2"] == before["yL"]["z2"]
    assert materials["yL"]["z3"]["kind"] == "constant_impedance" and materials["y0"]["z1"]["kind"] == "constant_impedance"
    assert materials["ceiling"] == before["ceiling"] and materials["floor"] == before["floor"]


def _save(page: Page, *, design: bool) -> None:
    with page.expect_response(lambda r: r.url.endswith(f"/api/schemes/{SAMPLE}") and r.request.method == "PUT") as scheme:
        if design:
            with page.expect_response(lambda r: r.url.endswith(f"/api/designs/{SAMPLE}") and r.request.method == "PUT") as stored:
                page.locator("#save").click()
            assert stored.value.ok
        else:
            page.locator("#save").click()
    assert scheme.value.ok
    expect(page.locator("#messages")).to_have_text(re.compile("已儲存|方案沒有變動"))


def test_opening_the_page_and_saving_without_changes_leaves_both_files_byte_identical(tmp_path: Path, browser: Browser) -> None:
    """打開、看、選一區再取消、關掉，然後存檔：設計檔與方案檔跟沒打開那一頁就存的一模一樣（逐位）。沒貼東西的指定貼區畫面上叫
    不貼，但不會被寫成不貼；沒有設計檔的方案也不會多出一份。"""
    for design in (SAMPLE_DESIGN, None):
        folder = tmp_path / ("with-design" if design else "no-design")
        folder.mkdir()
        _prepare(folder, design)
        scheme_file, design_file = folder / "schemes" / f"{SAMPLE}.json", folder / "designs" / f"{SAMPLE}.json"
        with _serve(folder) as base, _open(browser, base) as watched:
            page = watched.page
            _open_scheme(page, SAMPLE)
            _save(page, design=design is not None)
            saved = (scheme_file.read_bytes(), design_file.read_bytes() if design else None)
            page.locator("#zone-page-open").click()
            expect(page.locator("#zone-map")).to_be_visible()
            _click_zone(page, "y0/auto2")
            _click_zone(page, "ceiling/z2")
            expect(page.locator("#zone-panel h2")).to_have_text("已選 2 區")
            page.locator("#zone-panel-mode-search").check()
            page.locator("#zone-panel-cancel").click()
            expect(page.locator("#zone-panel")).to_have_count(0)
            page.locator("#zone-details summary").click()
            page.locator("#zone-page-close").click()
            _save(page, design=design is not None)
            assert watched.page_errors == [] and watched.console_errors == []
        assert scheme_file.read_bytes() == saved[0]
        assert (design_file.read_bytes() if design else None) == saved[1]
        assert design_file.exists() == (design is not None)
        if design:
            assert json.loads(design_file.read_text(encoding="utf-8"))["zones"] == design["zones"]


def test_editing_a_list_card_writes_the_list_into_every_zone_that_shares_it_and_no_other(tmp_path: Path, browser: Browser) -> None:
    _prepare(tmp_path, SAMPLE_DESIGN)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        searching = set(cast(dict[str, list[str]], page.evaluate(_ALLOWED)))
        assert {"y0/auto1", "ceiling/z1", "xL/auto4"} <= searching and len(searching) == 15
        assert cast(dict[str, list[str]], page.evaluate(_ALLOWED))["y0/auto1"] == SHORT
        # 拿掉一種：用這一份清單的 14 區都少一種；清單不一樣的那一區（後牆 auto1，自己一張卡）不動（#1199 修補 1）。
        page.locator("#zone-list-check-glass_wool_96k").click()
        shorter = ["glass_wool_48k", "glass_wool_24k", "none"]
        assert cast(dict[str, list[str]], page.evaluate(_ALLOWED)) == {**dict.fromkeys(searching, shorter), "y0/auto1": SHORT}
        # 在那一區自己的卡上加 24K：它的清單跟另外 14 區變成同樣三種，兩張卡併成一張。
        page.locator("#zone-list-2-add").click()
        page.locator("#zone-list-2-add-glass_wool_24k").click()
        expect(page.locator("#zone-page .zv-list")).to_have_count(1)
        assert set(cast(dict[str, list[str]], page.evaluate(_ALLOWED))["y0/auto1"]) == set(shorter)
        shorter_everywhere = {**dict.fromkeys(searching, shorter), "y0/auto1": [*SHORT, "glass_wool_24k"]}
        # 加別種材料：勾了就加進每一區的清單。
        page.locator("#zone-list-add").click()
        # （#1208 起合成庫那幾筆 VPR 列著但勾不了，另有考卷；這裡加的是一筆一般的固定成品。）
        expect(page.locator(f"label[data-choice={VPR}]")).to_contain_text("VPR（鋼板共振吸音）＋前面聚酯棉 60 mm")
        page.locator(f"#zone-list-add-{PRODUCT}").click()
        assert cast(dict[str, list[str]], page.evaluate(_ALLOWED)) == {zone: [*own, PRODUCT] for zone, own in shorter_everywhere.items()}
        # 只剩固定成品：程式沒得選，卡上講明，「拿去搜尋」按不下去；最後一種拿不掉。
        for choice in shorter:
            page.locator(f"#zone-list-check-{choice}").click()
        expect(page.locator("#zone-list-problem")).to_contain_text("只放了固定成品，程式沒得選")
        # 三條路都講到：加可調的、加「也可以不貼」、要的就是這一筆就把那一區改成指定貼一種。
        problem = cast(str, page.locator("#zone-list-problem").text_content())
        assert "加一種可調的材料" in problem and "加上「也可以不貼」" in problem
        assert "要的就是這一筆" in problem and "點圖上那一區，改成「指定貼一種」" in problem
        assert not any("a" <= char.lower() <= "z" for char in problem)
        expect(page.locator("#zone-list-problem button")).to_have_count(0)
        expect(page.locator("#zone-page-search")).to_be_disabled()
        expect(page.locator("#zone-page-blocked")).to_contain_text("的紅框（只放了固定成品，程式沒得選），處理好才能拿去搜尋")
        page.locator(f"#zone-list-check-{PRODUCT}").click()
        expect(page.locator("#zone-page-note")).to_contain_text("清單至少要留一種")
        expect(page.locator(f"#zone-list-check-{PRODUCT}")).to_be_checked()
        assert cast(dict[str, list[str]], page.evaluate(_ALLOWED)) == dict.fromkeys(searching, [PRODUCT])
        # 「加別種材料」那一格還開著（重畫後照樣開著）：把「也可以不貼」加回去就好了。
        page.locator("#zone-list-add-none").click()
        _settled(page)
        expect(page.locator("#zone-page-search")).to_be_enabled()
        expect(page.locator("#zone-list-problem")).to_have_count(0)
        assert watched.page_errors == [] and watched.console_errors == []


def test_a_late_server_reply_redraws_the_page_but_keeps_the_selection_and_the_option_just_picked(tmp_path: Path, browser: Browser) -> None:
    """限制檢查的回覆晚到會整塊重畫（做法表與這一頁一起）。多選到一半、面板上剛點了選項時它才回來：選的兩區、剛點的做法、
    剛選的材料都還在。"""
    _prepare(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        _settled(page)
        held: list[Route] = []
        page.route("**/api/zone-limits", lambda route: held.append(route))
        _click_zone(page, "yL/z1")
        page.locator("#zone-panel-mode-keep_original").check()
        page.evaluate("void refreshZoneLimits()")
        page.wait_for_function("zoneLimitsShown !== zoneLimitsGeneration")
        # 回覆還在路上：再加選一區、改點指定貼一種並選好材料。
        _click_zone(page, "yL/z2")
        page.locator("#zone-panel-material").select_option(VPR)
        expect(page.locator("#zone-panel-mode-fixed")).to_be_checked()
        page.evaluate("document.getElementById('zone-panel').dataset.before = 'yes'")
        while not held:
            page.wait_for_timeout(20)
        held[0].continue_()
        page.wait_for_function("zoneLimitsShown === zoneLimitsGeneration")
        # 真的重畫過（面板是新的一塊），選取與剛點的都在。
        expect(page.locator("#zone-panel")).not_to_have_attribute("data-before", "yes")
        assert page.evaluate(_SELECTED) == ["yL/z1", "yL/z2"]
        expect(page.locator("#zone-map .zv-ring")).to_have_count(2)
        expect(page.locator("#zone-panel h2")).to_have_text("已選 2 區")
        expect(page.locator("#zone-panel-mode-fixed")).to_be_checked()
        expect(page.locator("#zone-panel-material")).to_have_value(VPR)
        # 再點一次取消那一區；剛點的選項照留。
        _click_zone(page, "yL/z1")
        assert page.evaluate(_SELECTED) == ["yL/z2"]
        expect(page.locator("#zone-panel-material")).to_have_value(VPR)
        assert watched.page_errors == [] and watched.console_errors == []


def test_vpr_gets_its_plain_name_once_per_sentence(tmp_path: Path, browser: Browser) -> None:
    _write(tmp_path, ZONE_NAMING_SCHEME)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        words = "text => zoneVprWords(text)"
        assert page.evaluate(words, "指定貼一種：VPR＋前面吸音棉") == "指定貼一種：VPR（鋼板共振吸音）＋前面吸音棉"
        assert page.evaluate(words, "VPR 與另一種 VPR") == "VPR（鋼板共振吸音） 與另一種 VPR"
        # 名字自己已經接著括號說明：不重複。
        assert page.evaluate(words, "VPR（複合板共振）＋布") == "VPR（複合板共振）＋布"
        assert page.evaluate(words, "48K 玻璃棉") == "48K 玻璃棉"
        assert watched.page_errors == [] and watched.console_errors == []


def test_picking_the_ceiling_or_the_floor_lists_only_what_can_go_there_and_has_no_same_as_rule_mark(tmp_path: Path, browser: Browser) -> None:
    """天花、地板也點得到，同一個面板；材料只列那一面能貼的（伺服器每一筆給的「能用在哪幾面」），地板能貼的跟牆不一樣。
    「＝其他牆面」只在選到的都是算進規則的牆區時才標。"""
    _prepare(tmp_path)
    usable = """walls => (zoneLibrary || []).filter(entry => entry.kind === 'fixed' && walls.every(wall => entry.placement.surfaces.includes(wall)))
        .map(entry => entry.id)"""
    options = "options => options.map(option => option.value).filter(Boolean)"
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        on_ceiling, on_floor, on_wall = (cast(list[str], page.evaluate(usable, [wall])) for wall in ("ceiling", "floor", "yL"))
        assert on_ceiling and on_wall and set(on_floor) != set(on_wall)
        _click_zone(page, "ceiling/auto1")
        expect(page.locator("#zone-panel .zv-chip")).to_have_text("天花 3 靠後牆那塊不選這區")
        assert page.locator("#zone-panel-material option").evaluate_all(options) == on_ceiling
        expect(page.locator("#zone-panel-same")).to_have_count(0)
        page.screenshot(path=str(tmp_path / "ceiling-picked.png"))
        page.locator("#zone-panel-cancel").click()
        _click_zone(page, "floor/auto1")
        expect(page.locator("#zone-panel .zv-chip")).to_have_text("地板 4 主位底下那塊不選這區")
        assert page.locator("#zone-panel-material option").evaluate_all(options) == on_floor
        # 地板與牆一起選：只列兩面都能貼的。
        _click_zone(page, "yL/auto1")
        assert page.locator("#zone-panel-material option").evaluate_all(options) == cast(list[str], page.evaluate(usable, ["floor", "yL"]))
        expect(page.locator("#zone-panel-same")).to_have_count(0)
        assert watched.page_errors == [] and watched.console_errors == []


def test_same_as_rule_needs_the_same_material_and_a_long_name_wraps_on_the_map(tmp_path: Path, browser: Browser) -> None:
    """規則是「指定貼一種」某一筆時：選到的區貼的是別筆，「指定貼一種」那一列不准標「＝其他牆面」（改標在下拉裡規則那一筆後面），
    下拉換成規則那一筆才標。另外指定的區名字太長一行放不下：照空白折行寫在圖上，不是整個不寫。"""
    _prepare(tmp_path)
    option_words = "options => options.map(option => [option.value, option.textContent])"
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        wool = cast(str, page.evaluate("zoneLibrary.find(entry => entry.label === '48K 玻璃棉 50 mm 貼牆').id"))
        page.locator("#zone-rule-mode").select_option("fixed")
        page.locator("#zone-rule-material").select_option(VPR)
        _settled(page)
        expect(page.locator("#zone-rule")).to_have_attribute("data-mode", "fixed")
        # 原本貼著別的那一塊（另外指定）：現在貼的不是規則那一筆。
        _click_zone(page, "yL/z3")
        expect(page.locator("#zone-panel-mode-fixed")).to_be_checked()
        expect(page.locator("#zone-panel-same")).to_have_count(0)
        marked = [value for value, words in cast(list[list[str]], page.locator("#zone-panel-material option").evaluate_all(option_words))
                  if words.endswith("（＝其他牆面）")]
        assert marked == [VPR]
        page.locator("#zone-panel-material").select_option(wool)
        expect(page.locator("#zone-panel-same")).to_have_count(0)
        page.locator("#zone-panel-material").select_option(VPR)
        expect(page.locator("#zone-panel-option-fixed #zone-panel-same")).to_have_text("＝其他牆面")
        # 列上已經標了：下拉裡那一筆不再帶「（＝其他牆面）」（收起來那一格顯示的就是它）；整個名字另外寫一次、會折行。
        assert "＝其他牆面" not in cast(str, page.evaluate(_SHOWN_OPTION))
        expect(page.locator("#zone-panel-name")).to_have_text(cast(str, page.evaluate(_SHOWN_OPTION)))
        # 改貼玻璃棉那一筆（跟規則不同筆 → 仍是另外指定）。
        page.locator("#zone-panel-material").select_option(wool)
        _apply(page)
        _click_zone(page, "yL/z1")
        page.locator("#zone-panel-material").select_option(wool)
        _apply(page)
        assert ["yL/z1", "yL/z3"] in [group["zones"] for group in cast(list[dict[str, object]], page.evaluate(_GROUPS))]
        # 寬的那一塊一行放不下整句：照空白折成幾行、整句都在；窄的那一塊折了也寫不完，就整個不寫（不寫半句）。
        words = "#zone-map [data-wall='yL'][data-zone='{}'] .zv-fixed-words text"
        wide = page.locator(words.format("z1")).all_text_contents()
        assert len(wide) > 1 and " ".join(wide) == "48K 玻璃棉 50 mm 貼牆", wide
        assert page.locator(words.format("z3")).all_text_contents() == []
        page.screenshot(path=str(tmp_path / "wrapped-name.png"))
        # 規則換成名字短的那一筆，點一塊跟著規則的區：收起來那一格就是材料名本身，沒被截掉（量字寬比格子寬）。
        page.locator("#zone-rule-material").select_option(wool)
        _settled(page)
        _click_zone(page, "xL/auto2")
        expect(page.locator("#zone-panel-option-fixed #zone-panel-same")).to_have_text("＝其他牆面")
        assert page.evaluate(_SHOWN_OPTION) == "48K 玻璃棉 50 mm 貼牆"
        text_width, box_width = cast(list[float], page.evaluate(_SHOWN_WIDTHS))
        assert text_width + 40 <= box_width, (text_width, box_width)
        assert watched.page_errors == [] and watched.console_errors == []


def test_a_rule_pasted_with_something_outside_the_library_still_marks_its_followers_as_same(tmp_path: Path, browser: Browser) -> None:
    """規則貼的不是材料庫裡的成品（下拉裡沒有那一筆）：點一塊跟著規則的區，「指定貼一種」那一列照樣標「＝其他牆面」
    （下拉停在「照現在貼的，不換」）；下拉換成別筆就不標，換回來又標。
    假房間改一份：右牆六塊空牆都貼上前牆兩塊板那一種（合成的、不在材料庫裡），它就成了牆面裡最多區的做法。"""
    scheme = json.loads(json.dumps(ZONE_NAMING_SCHEME))
    walls = scheme["scene"]["surface_zones_by_wall"]
    pasted = next(zone["material"] for zone in walls["yL"] if zone["id"] == "z1")
    for zone in walls["xL"]:
        if zone["id"].startswith("auto"):
            zone["material"] = pasted
    _write(tmp_path, scheme)
    _library(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        expect(page.locator("#zone-rule")).to_have_attribute("data-mode", "fixed")
        expect(page.locator("#zone-rule-material")).to_have_value("")
        _click_zone(page, "xL/auto2")
        expect(page.locator("#zone-panel-material")).to_have_value("")
        expect(page.locator("#zone-panel-option-fixed #zone-panel-same")).to_have_text("＝其他牆面")
        page.locator("#zone-panel-material").select_option(VPR)
        expect(page.locator("#zone-panel-same")).to_have_count(0)
        page.locator("#zone-panel-material").select_option("")
        expect(page.locator("#zone-panel-same")).to_have_count(1)
        assert watched.page_errors == [] and watched.console_errors == []


def test_changes_made_on_the_page_are_stored_by_the_ordinary_save_after_going_back(tmp_path: Path, browser: Browser) -> None:
    """新頁沒有自己的存檔鈕：在新頁改完 → 按「◂ 回房間方案輸入」→ 按頁面原本的存檔，改的東西都存進設計檔與方案檔。
    兩種起點各一次：有設計檔的（改規則、把一區改成跟規則一樣、指定一區由程式最佳化）、沒有設計檔的（指定貼一種、一區不貼）。"""
    with_design, no_design = tmp_path / "with-design", tmp_path / "no-design"
    with_design.mkdir()
    no_design.mkdir()
    _prepare(with_design, SAMPLE_DESIGN)
    before = _materials_before()
    with _serve(with_design) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        page.locator("#zone-rule-mode").select_option("keep_original")
        _settled(page)
        page.locator("#zone-special-edit-1").click()
        page.locator("#zone-panel-mode-keep_original").check()
        _apply(page)
        _click_zone(page, "xL/auto4")
        page.locator("#zone-panel-mode-search").check()
        _apply(page)
        page.locator("#zone-page-close").click()
        expect(page.locator("#zone-page")).to_be_hidden()
        _save(page, design=True)
        assert watched.page_errors == [] and watched.console_errors == []
    design, materials = _stored(with_design)
    assert design == {
        "yL": {"z3": KEEP, "auto1": KEEP, "auto2": KEEP},
        "y0": {key: KEEP for key in ("z1", "auto1", "auto2", "auto3", "auto4")},
        "xL": {f"auto{n}": SEARCH if n == 4 else KEEP for n in range(1, 7)},
        "x0": {"whole_wall": KEEP},
        "ceiling": {"z1": SEARCH},
    }
    assert materials["yL"]["z1"] == before["yL"]["z1"] and materials["yL"]["z3"]["kind"] == "constant_impedance"
    assert materials["y0"]["z1"]["kind"] == "constant_impedance" and materials["floor"] == before["floor"]

    _prepare(no_design)
    with _serve(no_design) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        _click_zone(page, "y0/auto2")
        page.locator("#zone-panel-material").select_option(VPR)
        _apply(page)
        _click_zone(page, "yL/z3")
        page.locator("#zone-panel-mode-keep_original").check()
        _apply(page)
        page.locator("#zone-page-close").click()
        expect(page.locator("#zone-page")).to_be_hidden()
        _save(page, design=True)
        assert watched.page_errors == [] and watched.console_errors == []
    design, materials = _stored(no_design)
    assert cast(dict[str, object], design["y0"]["auto2"])["mode"] == "fixed" and design["yL"]["z3"] == KEEP
    # 指定貼的那一區：方案檔裡真的貼上了（貼在原牆上），不貼的那一區放回原樣，沒碰的照舊。
    assert materials["y0"]["auto2"]["kind"] == "mounted_on_wall" and materials["y0"]["auto2"] != before["y0"]["auto2"]
    assert materials["yL"]["z3"]["kind"] == "constant_impedance"
    assert materials["yL"]["z1"] == before["yL"]["z1"] and materials["ceiling"] == before["ceiling"]


def test_other_walls_never_changes_by_itself_while_zones_are_changed_one_group_at_a_time(tmp_path: Path, browser: Browser) -> None:
    """「其他牆面」是這次打開後定下來的那一種，使用者沒碰下拉它就不變：一組一組把牆區改成由程式最佳化，改到跟「不貼」剛好一樣多、
    再改到比它多，下拉都還是「不貼，保留原樣」，改掉的那幾區列在另外指定；存檔後重新打開方案才重推（這時由程式最佳化最多）。"""
    _prepare(tmp_path)
    count = "key => [...document.querySelectorAll('#zone-map .zv-zone')].filter(zone => zone.dataset.shown === key && !['floor', 'ceiling'].includes(zone.dataset.wall)).length"
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        expect(page.locator("#zone-rule-mode")).to_have_value("keep_original")
        # 先讓一塊空牆指定貼一種：剩 12 區不貼。
        _click_zone(page, "yL/auto1")
        page.locator("#zone-panel-material").select_option(VPR)
        _apply(page)
        for zones, still, search in ((["xL/auto1", "xL/auto2", "xL/auto3"], 9, 3), (["xL/auto4", "xL/auto5", "xL/auto6"], 6, 6),
                                     (["y0/auto1", "y0/auto2"], 4, 8)):
            for zone in zones:
                _click_zone(page, zone)
            page.locator("#zone-panel-mode-search").check()
            _apply(page)
            assert (page.evaluate(count, "still"), page.evaluate(count, "search")) == (still, search)
            expect(page.locator("#zone-rule-mode")).to_have_value("keep_original")
            expect(page.locator("#zone-rule")).to_have_attribute("data-mode", "keep_original")
        groups = [group["zones"] for group in cast(list[dict[str, object]], page.evaluate(_GROUPS))]
        assert [f"xL/auto{n}" for n in range(1, 7)] + ["y0/auto1", "y0/auto2"] in [sorted(cast(list[str], zones)) for zones in groups]
        # 使用者自己在下拉改規則：規則就是剛設的那一種（照著它的只有 4＋1 區，比由程式最佳化的 8 區少，也不會被搶走）。
        page.locator("#zone-rule-mode").select_option("fixed")
        page.locator("#zone-rule-material").select_option(VPR)
        _settled(page)
        expect(page.locator("#zone-rule")).to_have_attribute("data-mode", "fixed")
        expect(page.locator("#zone-rule-material")).to_have_value(VPR)
        assert (page.evaluate(count, "fixed"), page.evaluate(count, "search")) == (9, 8)
        # 重新打開方案才重推：這時由程式最佳化的區最多。
        page.locator("#zone-page-close").click()
        _save(page, design=True)
        _open_scheme(page, SAMPLE)
        page.locator("#zone-page-open").click()
        expect(page.locator("#zone-rule")).to_have_attribute("data-mode", "search")
        assert watched.page_errors == [] and watched.console_errors == []


def test_the_first_view_shows_the_way_into_the_page_and_keeps_the_old_table_folded(tmp_path: Path, browser: Browser) -> None:
    """開方案後立面圖那一段：第一眼是進「這間房貼什麼」的那顆按鈕（在那一段最上面），舊做法表收在「深度上限與舊做法表」裡、
    預設關著；點開看得到整張表；表裡有要人處理的提醒時它自己打開。"""
    _prepare(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        page.locator("#scheme-list").select_option(SAMPLE)
        page.locator("#open-scheme").click()
        page.wait_for_function("name => document.getElementById('save-id').value === name", arg=SAMPLE)
        page.wait_for_function("!document.querySelector('main').inert")
        fold, table, enter = page.locator("#zone-table-fold"), page.locator("#elevation-zones"), page.locator("#zone-page-open")
        assert fold.evaluate("fold => fold.open") is False
        expect(table).to_be_hidden()
        expect(fold.locator("> summary")).to_have_text("深度上限與舊做法表")
        expect(enter).to_be_visible()
        # 按鈕講的是動作（去設定），不是去看。
        expect(enter).to_have_text("設定每一區貼什麼 ▸")
        expect(page.locator("#zone-methods-row")).to_contain_text("都在「這間房貼什麼」那一頁設定")
        # 按鈕在立面圖那一段的最上面：在牆的圖、舊表的那一行之上。
        top = "selector => document.querySelector(selector).getBoundingClientRect().top"
        assert cast(float, page.evaluate(top, "#zone-page-open")) < cast(float, page.evaluate(top, "#elevation")) < cast(float, page.evaluate(top, "#zone-table-fold"))
        page.locator("#elevation-panel").scroll_into_view_if_needed()
        page.screenshot(path=str(tmp_path / "first-view.png"))
        page.locator("#elevation-wall-xL").click()
        expect(table).to_be_hidden()
        fold.locator("> summary").click()
        expect(table).to_be_visible()
        assert cast(int, table.evaluate("area => area.children.length")) > 0
        fold.locator("> summary").click()
        expect(table).to_be_hidden()
        # 有要人處理的提醒：自己打開。
        page.evaluate("zoneNote('考卷放的一句提醒', 'notice'); window.renderElevation()")
        expect(table).to_be_visible()
        enter.click()
        expect(page.locator("#zone-map")).to_be_visible()
        assert watched.page_errors == [] and watched.console_errors == []


def test_what_a_first_time_reader_needs_is_written_on_the_page(tmp_path: Path, browser: Browser) -> None:
    """沒看過說明的人只看畫面要讀得出來的四件：另外指定的區圖上有虛線框（不靠顏色與名字）、圖例有講；「拿去搜尋」灰掉時
    旁邊講為什麼；右欄摘要直接點名正在改的是哪一區（號碼塗黑、另起一句），不用跟在清單尾巴的括號；選到由程式最佳化時，
    選項底下講程式從哪幾種裡選，套用後寫進去的就是那幾種。"""
    _prepare(tmp_path)
    framed = "() => [...document.querySelectorAll('#zone-map .zv-zone')].filter(zone => zone.querySelector('.zv-apart')).map(zone => `${zone.dataset.wall}/${zone.dataset.zone}`).sort()"
    marked = "() => [...document.querySelectorAll('#zone-rules-brief .zv-now')].map(badge => badge.dataset.zone)"
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        # 另外指定的四區（前牆兩塊板、原本貼著別的兩塊）都有虛線框，跟著規則的區、天花、地板都沒有。
        apart = sorted(zone for group in cast(list[dict[str, list[str]]], page.evaluate(_GROUPS)) for zone in group["zones"])
        assert apart == ["y0/z1", "yL/z1", "yL/z2", "yL/z3"] and page.evaluate(framed) == apart
        expect(page.locator("#zone-legend-apart")).to_have_text("虛線框：另外指定（跟其他牆面不同）")
        # 沒有由程式最佳化的區：按鈕灰掉，旁邊講為什麼。
        expect(page.locator("#zone-page-search")).to_be_disabled()
        expect(page.locator("#zone-page-search-why")).to_have_text("還沒有由程式最佳化的區")
        # 選一區：摘要點名是哪一區。
        _click_zone(page, "yL/z3")
        assert page.evaluate(marked) == ["yL/z3", "yL/z3"]
        number = page.locator("#zone-map [data-wall='yL'][data-zone='z3'] .zv-badge text").text_content()
        expect(page.locator("#zone-brief-editing")).to_have_text(f"正在改：前牆 {number}")
        assert "（正在改）" not in page.locator("#zone-rules-brief").inner_text()
        # 選到由程式最佳化：底下講從哪幾種裡挑；套用後那一區的清單就是那幾種。
        expect(page.locator("#zone-panel-search-from")).to_have_count(0)
        page.locator("#zone-panel-mode-search").check()
        # （#1199：是哪幾種改在底下的材料框一種一列勾著，這一句只講幾種。）
        expect(page.locator("#zone-panel-search-from")).to_have_text("程式從下面勾的 4 種裡選，每一區各自最佳化。")
        assert page.locator("#zone-panel input[id^='zone-panel-allow-']:checked").evaluate_all("boxes => boxes.map(box => box.id.slice(17))") == [
            "glass_wool_48k", "glass_wool_24k", "glass_wool_96k", "none"]
        page.screenshot(path=str(tmp_path / "first-time-reader.png"))
        _apply(page)
        assert cast(dict[str, list[str]], page.evaluate(_ALLOWED))["yL/z3"] == WOOLS
        expect(page.locator("#zone-page-search-why")).to_have_count(0)
        expect(page.locator("#zone-page-search")).to_be_enabled()
        assert page.evaluate(marked) == []
        # 規則改成指定貼一種：跟著規則的區也是指定貼的顏色，但沒有虛線框；有框的還是只有另外指定的那幾區。
        page.locator("#zone-rule-mode").select_option("fixed")
        page.locator("#zone-rule-material").select_option(VPR)
        _settled(page)
        assert page.locator("#zone-map [data-wall='xL'][data-zone='auto2']").get_attribute("data-shown") == "fixed"
        apart = sorted(zone for group in cast(list[dict[str, list[str]]], page.evaluate(_GROUPS)) for zone in group["zones"])
        assert apart == ["y0/z1", "yL/z1", "yL/z2", "yL/z3"] and page.evaluate(framed) == apart
        assert watched.page_errors == [] and watched.console_errors == []


def test_no_item_on_the_bottom_bar_breaks_in_the_middle_whether_the_button_is_grey_or_lit(tmp_path: Path, browser: Browser) -> None:
    """1400×1000 下，「拿去搜尋」灰掉與亮著兩種狀態：底列三項（由程式最佳化…、指定貼…、保留原樣…）每一項自己都在同一行
    （括號裡的面積不准掉到下一行；整項換行可以）；灰掉時原因那一句看得到，在按鈕正下方。"""
    _prepare(tmp_path)
    # 每一項佔幾行：那一項的高除以它自己一行的高（放一個同字型、只有一個字的探針量一行多高）。
    lines = """() => ['search', 'fixed', 'still'].map(key => {
        const item = document.getElementById(`zone-result-${key}`), probe = item.cloneNode(true);
        probe.removeAttribute('id'); probe.style.cssText = 'position:absolute;visibility:hidden;white-space:nowrap';
        item.parentNode.append(probe);
        const one = probe.getBoundingClientRect().height;
        probe.remove();
        return Math.round(item.getBoundingClientRect().height / one);
    })"""
    context = browser.new_context(viewport={"width": 1400, "height": 1000})
    with _serve(tmp_path) as base:
        page = context.new_page()
        errors: list[str] = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(base)
        page.wait_for_function("document.body.dataset.inputReady === 'true'")
        _open_overview(page)
        expect(page.locator("#zone-page-search")).to_be_disabled()
        assert page.evaluate(lines) == [1, 1, 1]
        why, go = page.locator("#zone-page-search-why").bounding_box(), page.locator("#zone-page-search").bounding_box()
        assert why is not None and go is not None
        expect(page.locator("#zone-page-search-why")).to_be_visible()
        assert why["y"] >= go["y"] + go["height"] and abs((why["x"] + why["width"] / 2) - (go["x"] + go["width"] / 2)) < 2
        _click_zone(page, "yL/z3")
        page.locator("#zone-panel-mode-search").check()
        _apply(page)
        expect(page.locator("#zone-page-search")).to_be_enabled()
        assert page.evaluate(lines) == [1, 1, 1]
        # 這一頁有最小寬度，縮視窗擠不到底列；直接把底列壓到比最長那一項還窄：
        # 三項各自換到不同行，每一項自己還是一行（不是靠 1400 剛好夠寬）。
        page.evaluate("() => { document.querySelector('#zone-page .zv-foot').style.width = '400px'; }")
        tops = "() => ['search', 'fixed', 'still'].map(key => document.getElementById(`zone-result-${key}`).getBoundingClientRect().top)"
        first, second, third = cast(list[float], page.evaluate(tops))
        assert second - first > 15 and third - second > 15
        assert page.evaluate(lines) == [1, 1, 1]
        assert errors == []
    context.close()
