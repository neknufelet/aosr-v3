"""分區頁分組與存檔的瀏覽器考卷。"""

from __future__ import annotations
import json
from pathlib import Path
from typing import cast
import pytest
from playwright.sync_api import Browser, Locator, expect
from tests.engine._construction_cases import construction
from tests.engine._construction_gui_server import real_server
from tests.engine._gui_cache import gui_startup_identity_memo as gui_startup_identity_memo
from tests.engine.test_gui_browser import _open, _serve, browser as browser
from tests.engine.test_gui_zone_elevation_browser import _open_scheme
from tests.engine.test_gui_zone_overview_browser import SAMPLE, SAMPLE_DESIGN, SHORT, WOOLS
from tests.engine.test_gui_zone_overview_tasks_browser import (
    _ALLOWED, _SELECTED, _apply, _click_zone, _open_overview, _prepare, _save, _settled, _stored)
from tests.engine.test_gui_zone_material_pick_browser import (
    FLOOR_ONLY,
    FRONT,
    ONLY_48K,
    PLAIN_PRODUCT,
    _add_candidate,
    _candidate_state,
    _design,
    _pick_search,
)


def test_a_tick_is_added_only_to_the_zones_whose_face_can_take_it(tmp_path: Path, browser: Browser) -> None:
    """加一種材料只加到貼得了的那一面，而且勾得上也拿得掉（複查 r2 第二件）：前牆一區（48K 加不貼）與地板一區（只有不貼）一起選。
    48K 地板貼不了，不算進「幾區有」：貼得了的那一區有，就是勾著，旁邊講有一區貼不了；點一下拿掉、再點一下勾回來，
    每一下都照使用者點的畫，不會卡在第三種狀態。拿掉後套用：前牆那一區沒有 48K，地板不變。
    從卡選這兩區，再加選已有 24K 的牆區。小框按「加到 1 區」只補前牆，地板與原有 24K 的區不動。"""
    _prepare(tmp_path, _design({"yL": {"auto1": {"mode": "search", "allowed": ONLY_48K}}, "floor": {"auto1": {"mode": "search", "allowed": ["none"]}}}))
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        _click_zone(page, "yL/auto1")
        _click_zone(page, "floor/auto1")
        choice = WOOLS[0]
        assert _candidate_state(page, choice) == "all"
        expect(page.locator(f"#zone-panel-row-{choice}")).to_contain_text("選中的 2 區裡 1 區那一面貼不了這一種，不會加")
        page.locator(f"#zone-panel-remove-{choice}").click()
        assert _candidate_state(page, choice) == "none"
        page.locator("#zone-panel-undo").click()
        assert _candidate_state(page, choice) == "all"
        page.locator(f"#zone-panel-remove-{choice}").click()
        assert _candidate_state(page, choice) == "none"
        _apply(page)
        assert cast(dict[str, list[str]], page.evaluate(_ALLOWED)) == {"yL/auto1": ["none"], "floor/auto1": ["none"]}
        # 選這兩區，再加選一個已有 24K 的牆區，才能在同一列按「加到 1 區」。
        expect(page.locator("#zone-page .zv-list")).to_have_count(1)
        _click_zone(page, "yL/auto2")
        _pick_search(page)
        _add_candidate(page, "glass_wool_24k")
        _apply(page)
        page.locator("#zone-list-select").click()
        _click_zone(page, "yL/auto2")
        row = page.locator("#zone-panel-row-glass_wool_24k")
        expect(row).to_contain_text("1 區那一面貼不了這一種，不會加")
        add = row.get_by_role("button", name="加到 1 區")
        add.click()
        _apply(page)
        assert cast(dict[str, list[str]], page.evaluate(_ALLOWED)) == {
            # yL/auto2 也釘住新開的區照上一份套用的清單起頭（zoneLastAllowed）。
            "yL/auto1": ["none", "glass_wool_24k"], "floor/auto1": ["none"], "yL/auto2": ["none", "glass_wool_24k"]}
        assert watched.page_errors == [] and watched.console_errors == []


def test_a_half_ticked_material_follows_each_click_and_can_be_taken_off(tmp_path: Path, browser: Browser) -> None:
    """第三種狀態只給使用者還沒動過的勾：兩區一區有 24K、一區沒有，打開是第三種狀態；點一下勾上、再點一下拿掉，
    拿掉後套用，兩區都沒有 24K。"""
    _prepare(tmp_path, SAMPLE_DESIGN)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        before = cast(dict[str, list[str]], page.evaluate(_ALLOWED))
        _click_zone(page, "y0/auto1")
        _click_zone(page, "y0/auto2")
        choice = WOOLS[1]
        assert _candidate_state(page, choice) == "some"
        page.locator(f"#zone-panel-addto-{choice}").click()
        assert _candidate_state(page, choice) == "all"
        page.locator(f"#zone-panel-remove-{choice}").click()
        assert _candidate_state(page, choice) == "none"
        expect(page.locator(f"#zone-panel-row-{choice}")).to_have_count(0)
        _apply(page)
        assert cast(dict[str, list[str]], page.evaluate(_ALLOWED)) == {**before, "y0/auto2": ["glass_wool_48k", "glass_wool_96k", "none"]}
        assert watched.page_errors == [] and watched.console_errors == []


CARD_HINT = "只看；要改就點圖上的區，或按卡上的「選這幾區」"


def test_cards_show_their_choices_depth_and_only_a_group_selection_button(tmp_path: Path, browser: Browser) -> None:
    front: dict[str, object] = {key: {"mode": "search", "allowed": ["none", WOOLS[0]], "max_depth_m": 0.125}
                                for key in ("z1", "z2", "z3", "auto1", "auto2")}
    ceiling: dict[str, object] = {"z1": {"mode": "search", "allowed": WOOLS}, "auto1": {"mode": "search", "allowed": WOOLS, "max_depth_m": 0.1}}
    floor: dict[str, object] = {"auto1": {"mode": "search", "allowed": ["none"]}}
    _prepare(tmp_path, _design({"yL": front, "ceiling": ceiling, "floor": floor}))
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        hint = page.get_by_text(CARD_HINT, exact=True)
        cards = page.locator("#zone-page .zv-list")
        expect(cards).to_have_count(3)
        expected = [(set(FRONT), ONLY_48K, "最深 12.5 cm"), ({"ceiling/z1", "ceiling/auto1"}, WOOLS, "最深各區不同"),
                    ({"floor/auto1"}, ["none"], "最深不限")]
        for card, (zones, choices, depth) in zip(cards.all(), expected, strict=True):
            assert set((card.get_attribute("data-zones") or "").split()) == zones
            expect(card.locator(".zv-who").first).to_contain_text(f"{len(zones)} 區")
            assert card.locator("[data-choice]").evaluate_all("rows => rows.map(r => r.dataset.choice)") == choices
            expect(card.locator("input,select")).to_have_count(0)
            expect(card.get_by_text(depth, exact=True)).to_be_visible()
            expect(card.locator("button")).to_have_id(f"{card.get_attribute('id')}-select")
            expect(card.locator("button")).to_have_text("選這幾區")
        expect(hint).to_have_count(1)
        for index, panel_depth in enumerate(("12.5", "")):
            card = cards.nth(index)
            card.get_by_role("button", name="選這幾區").click()
            assert set(page.evaluate(_SELECTED)) == expected[index][0]
            expect(page.locator("#zone-panel-mode-search")).to_be_checked()
            expect(page.locator("#zone-panel-depth")).to_have_value(panel_depth)
            expect(page.locator("#zone-panel-notice")).to_have_count(0)
            assert page.locator("#zone-panel-list [data-choice]").evaluate_all("rows => rows.map(r => r.dataset.choice)") == expected[index][1]
            expect(card.locator(".zv-card-summary")).to_have_text(expected[index][2])
            expect(card.locator("button")).to_have_count(0)
            expect(card.locator(f"span#{card.get_attribute('id')}-select")).to_have_text("正在改")
            expect(cards.locator("input,select")).to_have_count(0)
            expect(cards.locator("button")).to_have_count(2)
            expect(page.get_by_text(CARD_HINT + "；寫「正在改」的那一組，就在上面的小框改", exact=True)).to_have_count(1)
        page.locator("#zone-panel-remove-glass_wool_96k").click()
        _apply(page)
        page.locator("#zone-page-close").click()
        _save(page, design=True)
        assert watched.page_errors == [] and watched.console_errors == []
    stored, _ = _stored(tmp_path)
    remaining = [WOOLS[0], WOOLS[1], "none"]
    assert stored == {"yL": front, "floor": floor, "ceiling": {
        "z1": {"mode": "search", "allowed": remaining},
        "auto1": {"mode": "search", "allowed": remaining, "max_depth_m": 0.1}}}


def test_no_search_zone_has_no_card_instructions(tmp_path: Path, browser: Browser) -> None:
    _prepare(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        expect(page.locator("#zone-list")).to_have_text("現在沒有哪一區由程式最佳化")
        expect(page.get_by_text(CARD_HINT, exact=True)).to_have_count(0)
        page.locator("#zone-rule-mode").select_option("search")
        _settled(page)
        expect(page.get_by_text(CARD_HINT, exact=True)).to_have_count(1)


def test_changing_only_the_deepest_keeps_a_code_this_face_cannot_take_in_the_list(tmp_path: Path, browser: Browser) -> None:
    """守門題（複查 r2 植錯 M2）：前牆一區的清單裡有一筆這一面貼不了的（只有地板能用的那一筆）。只改最深就套用：清單原封不動。"""
    on_wall = ["glass_wool_48k", FLOOR_ONLY, "none"]
    _prepare(tmp_path, _design({"yL": {"auto1": {"mode": "search", "allowed": on_wall}}}))
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        _click_zone(page, "yL/auto1")
        page.locator("#zone-panel-depth").fill("9")
        _apply(page)
        assert page.evaluate("() => zoneDesign.zones.yL.auto1") == {"mode": "search", "allowed": on_wall, "max_depth_m": 0.09}
        assert watched.page_errors == [] and watched.console_errors == []


def test_two_zones_pasted_with_different_materials_keep_their_own_when_picked_together_and_applied(tmp_path: Path, browser: Browser) -> None:
    """守門題（複查 r2 植錯 M3）：兩區指定貼的是材料庫裡不同的兩筆。一起選：下拉停在「照現在貼的，不換」，不帶出哪一筆；
    什麼都不動就套用，兩區貼的都不變（第一區的材料不准被貼到第二區）；再一起選、按取消，也不變。"""
    _prepare(tmp_path, SAMPLE_DESIGN)
    pasted = """() => ['yL/z1', 'ceiling/z2'].map(id => { const [wall, key] = id.split('/');
        return JSON.stringify(scheme.scene.surface_zones_by_wall[wall].find(zone => zone.id === key).material); })"""
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        options = cast(list[str], page.evaluate("() => libraryEntries('fixed').filter(entry => usableHere(entry, 'yL') && usableHere(entry, 'ceiling')).map(entry => entry.id)"))
        assert len(options) >= 2
        for zone, material in (("yL/z1", options[0]), ("ceiling/z2", options[1])):
            _click_zone(page, zone)
            page.locator("#zone-panel-mode-fixed").check()
            page.locator("#zone-panel-material").select_option(material)
            _apply(page)
        before = cast(list[str], page.evaluate(pasted))
        assert before[0] != before[1]
        _click_zone(page, "yL/z1")
        _click_zone(page, "ceiling/z2")
        expect(page.locator("#zone-panel-mode-fixed")).to_be_checked()
        _apply(page)
        assert cast(list[str], page.evaluate(pasted)) == before
        _click_zone(page, "yL/z1")
        _click_zone(page, "ceiling/z2")
        expect(page.locator("#zone-panel-material")).to_have_value("")
        page.locator("#zone-panel-cancel").click()
        assert cast(list[str], page.evaluate(pasted)) == before
        assert watched.page_errors == [] and watched.console_errors == []


def test_apply_is_blocked_when_one_of_the_picked_zones_would_be_left_with_nothing(tmp_path: Path, browser: Browser) -> None:
    """選了兩區，拿掉一種之後其中一區的清單會變成空的（另一區還有「不貼」）：不能套用，那一句點名是哪一區。"""
    _prepare(tmp_path, _design({"yL": {"auto1": {"mode": "search", "allowed": ["glass_wool_48k"]}, "auto2": {"mode": "search", "allowed": ONLY_48K}}}))
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        _click_zone(page, "yL/auto1")
        _click_zone(page, "yL/auto2")
        expect(page.locator("#zone-panel-apply")).to_be_enabled()
        page.locator("#zone-panel-remove-glass_wool_48k").click()
        expect(page.locator("#zone-panel-apply")).to_be_disabled()
        number = int(cast(str, page.locator("#zone-panel .zv-chip[data-zone='yL/auto1'] .zv-num").text_content()))
        expect(page.locator("#zone-panel-problem")).to_contain_text(f"前牆 {'①②③④⑤'[number - 1]} ")
        expect(page.locator("#zone-panel-problem")).to_contain_text("至少勾一種材料")
        page.locator("#zone-panel-addto-none").click()
        expect(page.locator("#zone-panel-apply")).to_be_enabled()
        _apply(page)
        assert cast(dict[str, list[str]], page.evaluate(_ALLOWED)) == {"yL/auto1": ["none"], "yL/auto2": ["none"]}
        assert watched.page_errors == [] and watched.console_errors == []


def test_the_front_wall_is_set_to_48k_only_and_no_zone_on_another_face_changes(tmp_path: Path, browser: Browser) -> None:
    """老闆那句話在已經有 15 區由程式最佳化的房間做：前牆五區變成 48K 加不貼，後牆、右牆、天花的清單一個字都不變。"""
    _prepare(tmp_path, SAMPLE_DESIGN)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        before = cast(dict[str, list[str]], page.evaluate(_ALLOWED))
        rule = page.locator("#zone-rule").get_attribute("data-key")
        expect(page.locator("#zone-special")).to_have_attribute("data-count", "4")
        page.locator("#zone-whole-yL").click()
        _pick_search(page)
        page.locator("#zone-panel-remove-glass_wool_24k").click()
        page.locator("#zone-panel-remove-glass_wool_96k").click()
        _apply(page)
        assert cast(dict[str, list[str]], page.evaluate(_ALLOWED)) == {**before, **dict.fromkeys(FRONT, ONLY_48K)}
        # 守門（複查 r2 植錯 M1）：「其他牆面」那條規則還是同一份；另外指定只多了前牆原本跟著規則的那三區，別面沒有被算成另外指定。
        assert rule is not None and page.locator("#zone-rule").get_attribute("data-key") == rule
        expect(page.locator("#zone-special")).to_have_attribute("data-count", "7")
        assert watched.page_errors == [] and watched.console_errors == []


def test_the_panel_opens_on_the_picked_zones_own_lists_and_a_half_ticked_material_is_left_alone(tmp_path: Path, browser: Browser) -> None:
    """入口二（打開）：小框的勾從選中的區自己存的清單開始，不是別區那一份。選了兩區、清單不同：兩區都有的勾著、
    只有一區有的是第三種狀態（indeterminate），旁邊寫「選中的 2 區裡 1 區有」；沒動它，套用後各區照舊——只改最深，清單一個字都不動。
    動了它（勾下去）才兩區都有。"""
    _prepare(tmp_path, SAMPLE_DESIGN)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        before = cast(dict[str, list[str]], page.evaluate(_ALLOWED))
        assert before["y0/auto1"] == SHORT and before["y0/auto2"] == WOOLS
        # 只選清單比較短的那一區：它沒有的兩種是空的。
        _click_zone(page, "y0/auto1")
        expect(page.locator("#zone-panel-mode-search")).to_be_checked()
        assert [_candidate_state(page, choice) for choice in WOOLS] == ["all", "none", "none", "all"]
        # 這一區的清單跟「其他牆面」那一份不一樣：不標「＝其他牆面」；把缺的兩種勾上、跟它一樣了才標。
        expect(page.locator("#zone-panel-same")).to_have_count(0)
        _add_candidate(page, "glass_wool_24k")
        _add_candidate(page, "glass_wool_96k")
        expect(page.locator("#zone-panel-option-search #zone-panel-same")).to_have_text("＝其他牆面")
        page.locator("#zone-panel-cancel").click()
        _click_zone(page, "y0/auto1")
        # 再加選清單比較長的那一區。
        _click_zone(page, "y0/auto2")
        expect(page.locator("#zone-panel-mode-search")).to_be_checked()
        assert [_candidate_state(page, choice) for choice in WOOLS] == ["all", "some", "some", "all"]
        expect(page.locator("#zone-panel-list [data-choice=glass_wool_24k]")).to_contain_text("選中的 2 區裡 1 區有")
        expect(page.locator("#zone-panel-list [data-choice=glass_wool_48k]")).not_to_contain_text("選中的")
        expect(page.locator("#zone-panel-search-from")).to_have_text("選中的 2 區清單不一樣：沒動的勾，各區照目前的。每一區各自最佳化。")
        page.locator("#zone-panel-depth").fill("10")
        _apply(page)
        zones = cast(dict[str, dict[str, object]], page.evaluate("() => zoneDesign.zones.y0"))
        assert zones["auto1"] == {"mode": "search", "allowed": SHORT, "max_depth_m": 0.1}
        assert zones["auto2"] == {"mode": "search", "allowed": WOOLS, "max_depth_m": 0.1}
        assert cast(dict[str, list[str]], page.evaluate(_ALLOWED)) == before
        # 把只有一區有的那一種勾下去：兩區都有；另一種沒動，照舊只有一區有。
        _click_zone(page, "y0/auto1")
        _click_zone(page, "y0/auto2")
        page.locator("#zone-panel-addto-glass_wool_24k").click()
        assert [_candidate_state(page, choice) for choice in WOOLS] == ["all", "all", "some", "all"]
        _apply(page)
        assert cast(dict[str, list[str]], page.evaluate(_ALLOWED)) == {**before, "y0/auto1": [*SHORT, "glass_wool_24k"]}
        assert watched.page_errors == [] and watched.console_errors == []


def test_zones_with_different_lists_get_one_list_card_each_and_a_tick_on_a_card_changes_only_its_zones(tmp_path: Path, browser: Browser) -> None:
    """入口三（右欄清單卡）：前牆五區只准 48K 加不貼、天花一區三種棉加不貼——兩張卡，各寫是哪幾區、准哪些，
    都不寫「都從這幾種裡選」。在前牆那張加 24K，天花不動；在天花那張拿掉 96K，前牆不動；兩邊清單變成一樣，就併回一張卡。"""
    front: dict[str, object] = dict.fromkeys(("z1", "z2", "z3", "auto1", "auto2"), {"mode": "search", "allowed": ONLY_48K})
    _prepare(tmp_path, _design({"yL": front, "ceiling": {"z1": {"mode": "search", "allowed": WOOLS}}}))
    choices = "rows => rows.map(row => row.dataset.choice)"
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        expect(page.locator("#zone-page .zv-list")).to_have_count(2)
        expect(page.locator("#zone-list h2")).to_have_text("由程式最佳化的 6 區裡的 5 區每一區各自最佳化，這 5 區從這幾種裡選")
        expect(page.locator("#zone-list-2 h2")).to_have_text("由程式最佳化的 6 區裡的 1 區這一區從這幾種裡選")
        assert set(cast(str, page.locator("#zone-list").get_attribute("data-zones")).split()) == set(FRONT)
        assert page.locator("#zone-list-2").get_attribute("data-zones") == "ceiling/z1"
        expect(page.locator("#zone-list-who")).to_have_text("這一張管 5 區：前牆整面 12345")
        expect(page.locator("#zone-list-2-who")).to_have_text("這一張管 1 區：天花 1 主位上面那塊")
        assert page.locator("#zone-list-choices > [data-choice]").evaluate_all(choices) == ONLY_48K
        assert page.locator("#zone-list-2-choices > [data-choice]").evaluate_all(choices) == WOOLS
        # 前牆那張拿掉「也可以不貼」：只有前牆五區少一種，天花那一區的「不貼」還在；再加回去。
        page.locator("#zone-list-select").click()
        page.locator("#zone-panel-remove-none").click()
        _apply(page)
        assert cast(dict[str, list[str]], page.evaluate(_ALLOWED)) == {**dict.fromkeys(FRONT, ["glass_wool_48k"]), "ceiling/z1": WOOLS}
        page.locator("#zone-list-select").click()
        page.locator("#zone-panel-addto-none").click()
        _apply(page)
        assert cast(dict[str, list[str]], page.evaluate(_ALLOWED)) == {**dict.fromkeys(FRONT, ONLY_48K), "ceiling/z1": WOOLS}
        # 前牆那張加 24K：只有前牆五區多一種。
        page.locator("#zone-list-select").click()
        _add_candidate(page, "glass_wool_24k")
        _apply(page)
        longer = [*ONLY_48K, "glass_wool_24k"]
        assert cast(dict[str, list[str]], page.evaluate(_ALLOWED)) == {**dict.fromkeys(FRONT, longer), "ceiling/z1": WOOLS}
        # 天花那張拿掉 96K：只有天花那一區少一種；兩邊變成同樣三種，併回一張卡，這時才寫「都從這幾種裡選」。
        page.locator("#zone-list-2-select").click()
        page.locator("#zone-panel-remove-glass_wool_96k").click()
        _apply(page)
        assert cast(dict[str, list[str]], page.evaluate(_ALLOWED)) == {**dict.fromkeys(FRONT, longer), "ceiling/z1": ["glass_wool_48k", "glass_wool_24k", "none"]}
        expect(page.locator("#zone-page .zv-list")).to_have_count(1)
        expect(page.locator("#zone-list h2")).to_have_text("由程式最佳化的 6 區每一區各自最佳化，都從這幾種裡選")
        # 併成一張之後，在那一張拿掉 24K：六區一起少一種。
        page.locator("#zone-list-select").click()
        page.locator("#zone-panel-remove-glass_wool_24k").click()
        _apply(page)
        assert cast(dict[str, list[str]], page.evaluate(_ALLOWED)) == dict.fromkeys([*FRONT, "ceiling/z1"], ONLY_48K)
        assert watched.page_errors == [] and watched.console_errors == []


def test_the_deepest_for_the_picked_zones_is_written_in_metres_and_only_when_it_was_filled_in(tmp_path: Path, browser: Browser) -> None:
    """「這一區最深」填公分、存公尺，只寫進選中的區；沒填就不寫那一格；原本有的，打開看得到（公分），清空再套用就拿掉。"""
    design: dict[str, object] = {"schema_version": "aosr.zone_design.v1", "scheme_id": SAMPLE,
                                 "zones": {"yL": {"auto2": {"mode": "search", "allowed": ONLY_48K, "max_depth_m": 0.125}}}}
    _prepare(tmp_path, design)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        _click_zone(page, "yL/auto1")
        _pick_search(page)
        expect(page.locator("#zone-panel-depth")).to_have_value("")
        expect(page.locator("#zone-panel-deepest")).to_contain_text("空白＝不限")
        expect(page.locator("#zone-panel-deepest")).to_contain_text("這一區最深")
        page.locator("#zone-panel-depth").fill("20")
        _apply(page)
        _click_zone(page, "yL/z3")
        _pick_search(page)
        _apply(page)
        zones = cast(dict[str, dict[str, object]], page.evaluate("() => zoneDesign.zones.yL"))
        assert zones["auto1"] == {"mode": "search", "allowed": ONLY_48K, "max_depth_m": 0.2}
        assert zones["z3"] == {"mode": "search", "allowed": ONLY_48K}
        assert zones["auto2"] == {"mode": "search", "allowed": ONLY_48K, "max_depth_m": 0.125}
        # 原本有上限的那一區：打開看得到 12.5；不碰就套用，那一格一個位元都不變；清空再套用就拿掉。
        _click_zone(page, "yL/auto2")
        expect(page.locator("#zone-panel-depth")).to_have_value("12.5")
        _apply(page)
        assert page.evaluate("() => zoneDesign.zones.yL.auto2.max_depth_m") == 0.125
        _click_zone(page, "yL/auto2")
        page.locator("#zone-panel-depth").fill("")
        _apply(page)
        assert page.evaluate("() => zoneDesign.zones.yL.auto2") == {"mode": "search", "allowed": ONLY_48K}
        # 不是正數就不收：照講，不能套用。
        _click_zone(page, "yL/auto2")
        page.locator("#zone-panel-depth").fill("-3")
        page.locator("#zone-panel-depth").press("Tab")
        expect(page.locator("#zone-panel-apply")).to_be_disabled()
        expect(page.locator("#zone-panel-problem")).to_contain_text("要填大於 0 的數字")
        assert page.evaluate("() => zoneDesign.zones.yL.auto2") == {"mode": "search", "allowed": ONLY_48K}
        # 不到 1 公釐（存公尺、記到公釐會變成 0，整份做法存不進去）、打到一半不成數字的：一樣不收。
        for typed in ("0.04", "1e"):
            page.locator("#zone-panel-depth").fill("")
            page.locator("#zone-panel-depth").press_sequentially(typed)
            expect(page.locator("#zone-panel-apply")).to_be_disabled()
            expect(page.locator("#zone-panel-problem")).to_contain_text("要填大於 0 的數字")
        # 存公尺、記到公釐：33.3 cm 是 0.333，不是除出來帶尾巴的那個數。
        page.locator("#zone-panel-depth").fill("33.3")
        _apply(page)
        assert page.evaluate("() => JSON.stringify(zoneDesign.zones.yL.auto2.max_depth_m)") == "0.333"
        assert watched.page_errors == [] and watched.console_errors == []


def test_constructions_can_be_assigned_but_only_standalone_ones_can_be_checked(tmp_path: Path, browser: Browser) -> None:
    _prepare(tmp_path)
    with real_server(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        for backing in ("rigid", "air", "zone_wall"):
            doc = construction(backing, id=f"panel_{backing}", label=f"自編面板 {backing}")
            assert page.request.put(f"{base}/api/materials/{doc['id']}", data=doc).ok
        page.reload()
        _open_overview(page)
        _click_zone(page, "yL/z3")
        page.locator("#zone-panel-mode-fixed").check()
        select = page.locator("#zone-panel-material")
        assert not any(identifier.startswith("panel_") for identifier in page.evaluate("serverDefaultAllowed()"))
        for backing in ("zone_wall", "rigid", "air"):
            expect(select.locator(f"option[value='panel_{backing}']")).to_have_count(1)
            with page.expect_response(lambda response: response.url.endswith("/api/material-curve")) as curve:
                select.select_option(f"panel_{backing}")
            assert curve.value.json()["status"] == "drawn"
        _apply(page)
        _click_zone(page, "yL/z3")
        expect(page.locator("#zone-panel")).to_contain_text("自編面板 air")
        expect(page.locator("#zone-panel-now-fixed")).to_have_count(1)
        expect(select).to_have_value("panel_air")
        _pick_search(page)
        for backing in ("rigid", "air"):
            page.locator("#zone-panel-add").click()
            choice = page.locator(f"#zone-panel-pick-panel_{backing}")
            expect(choice).to_be_enabled()
            expect(page.locator(f"#zone-panel-row-panel_{backing}")).to_have_count(0)
            choice.click()
        page.locator("#zone-panel-add").click()
        expect(page.locator("#zone-panel-pick-panel_zone_wall")).to_have_count(0)
        assert watched.page_errors == [] and watched.console_errors == []


@pytest.mark.parametrize("with_original", [False, True])
def test_zone_map_uses_the_saved_original_to_identify_a_bare_copy(tmp_path: Path, browser: Browser, with_original: bool) -> None:
    _prepare(tmp_path, {"scheme_id": SAMPLE, "original_scheme_id": "before"} if with_original else None)
    if with_original:
        folder = tmp_path / "schemes"
        original = json.loads((folder / f"{SAMPLE}.json").read_text()) | {"scheme_id": "before"}
        (folder / "before.json").write_text(json.dumps(original))
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        saved = page.request.get(f"{base}/api/schemes/{SAMPLE}").json()["scheme"]
        ceiling = next(zone for zone in saved["scene"]["surface_zones_by_wall"]["ceiling"] if zone["id"] == "auto1")
        doc = construction("rigid", id="ceiling_copy", label="布置用面板", layers=ceiling["material"]["layers"])
        assert page.request.put(f"{base}/api/materials/ceiling_copy", data=doc).ok
        _open_overview(page)
        def shown(wall: str, key: str) -> Locator:
            return page.locator(f"#zone-map [data-wall='{wall}'][data-zone='{key}']")
        expect(shown("ceiling", "auto1")).to_have_attribute("data-shown", "still" if with_original else "unknown")
        _click_zone(page, "yL/z3")
        page.locator("#zone-panel-mode-fixed").check()
        page.locator("#zone-panel-material").select_option("ceiling_copy")
        _apply(page)
        for reopened in (False, True):
            if reopened:
                page.locator("#zone-page-close").click()
                _save(page, design=True)
                page.reload()
                _open_overview(page)
            zone = shown("yL", "z3")
            expect(zone).to_have_attribute("data-shown", "fixed")
            expect(shown("ceiling", "auto1")).to_have_attribute("data-shown", "still" if with_original else "unknown")
            if not with_original:
                expect(zone.locator("title")).to_contain_text("布置用面板")
                assert page.evaluate('zoneSetting("yL", "z3").picked_entry') == "ceiling_copy"
                expect(page.locator(".zv-legend")).to_contain_text("看不出")
        if not with_original:
            _click_zone(page, "yL/z3")
            page.locator("#zone-panel-mode-keep_original").check()
            _apply(page)
            expect(zone).to_have_attribute("data-shown", "still")
            assert page.evaluate("""() => {
                const mat = zoneList("yL").find(z => z.id === "z3").material;
                return [plainNow(mat, "yL", "z3"), originalWords(mat, "yL", "z3")];
            }""") == ["不貼，保留原樣"] * 2
            expect(zone.locator("title")).to_contain_text("不貼，保留原樣")
            _click_zone(page, "yL/z3")
            expect(page.locator("#zone-panel-now-keep_original")).to_have_attribute("data-zones", "yL/z3")
            expect(page.locator(".zv-chip[data-zone='yL/z3']")).not_to_contain_text("看不出")
        assert watched.page_errors == [] and watched.console_errors == []


def test_table_pick_saves_the_entry_and_clears_it_on_replacement_or_mode_change(tmp_path: Path, browser: Browser) -> None:
    from tests.engine.test_gui_zone_modes_browser import _choose

    _prepare(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        doc = construction("rigid", id="table_sample")
        assert page.request.put(f"{base}/api/materials/table_sample", data=doc).ok
        _open_scheme(page, SAMPLE)
        page.locator("#elevation-wall-yL").click()
        selector = "#zone-fixed-yL-z3"
        _choose(page, selector, "table_sample")
        _save(page, design=True)
        path = tmp_path / "designs" / f"{SAMPLE}.json"
        assert json.loads(path.read_bytes())["zones"]["yL"]["z3"] == {"mode": "fixed", "picked_entry": "table_sample"}
        _choose(page, selector, PLAIN_PRODUCT)
        _save(page, design=True)
        assert json.loads(path.read_bytes())["zones"]["yL"]["z3"] == {"mode": "fixed"}
        _choose(page, selector, "table_sample")
        _choose(page, "#zone-mode-yL-z3", "keep_original")
        _save(page, design=True)
        assert json.loads(path.read_bytes())["zones"]["yL"]["z3"] == {"mode": "keep_original"}


@pytest.mark.parametrize(("batch", "recorded"), [(False, None), (True, "former_sample")])
def test_invalid_draft_depth_does_not_block_a_material_change(tmp_path: Path, browser: Browser, batch: bool, recorded: str | None) -> None:
    _prepare(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        if batch:
            _open_overview(page)
            _click_zone(page, "yL/z3")
            page.locator("#zone-panel-mode-keep_original").check()
        else:
            _open_scheme(page, SAMPLE)
            page.locator("#elevation-wall-yL").click()
        # 深度輸入框本身會擋零。直接放入尚未存檔的草稿，再走兩個真正的換材料入口。
        page.evaluate("""id => setZoneSetting("yL", "z3", {mode: "fixed", max_depth_m: 0, picked_entry: id})""", recorded)
        path = "/api/zone-material/batch" if batch else "/api/zone-material"
        with page.expect_response(lambda response: response.url.endswith(path)) as changed:
            if batch:
                page.locator("#zone-panel-apply").click()
            else:
                page.locator("#zone-mode-yL-z3").select_option("keep_original")
        assert changed.value.status == 200, changed.value.text()
        body = changed.value.request.post_data_json
        assert isinstance(body, dict)
        choice = body["changes"][0] if batch else body
        assert choice["picked_entry"] == recorded and "setting" not in choice


@pytest.mark.parametrize("with_original", [False, True])
def test_keep_original_table_words_follow_the_ceiling_verdict(tmp_path: Path, browser: Browser, with_original: bool) -> None:
    design: dict[str, object] = {"scheme_id": SAMPLE, "zones": {"ceiling": {"auto1": {"mode": "keep_original"}}}}
    _prepare(tmp_path, design | ({"original_scheme_id": "before"} if with_original else {}))
    doc = construction("air", id="ceiling_sample", label="自編天花板")
    path = tmp_path / "schemes" / f"{SAMPLE}.json"
    scheme = json.loads(path.read_bytes())
    ceiling = next(z for z in scheme["scene"]["surface_zones_by_wall"]["ceiling"] if z["id"] == "auto1")
    ceiling["material"] = {"kind": "assembly", "layers": doc["layers"], "backing": doc["backing"]}
    path.write_text(json.dumps(scheme))
    if with_original:
        (path.parent / "before.json").write_text(json.dumps(scheme | {"scheme_id": "before"}))
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        assert page.request.put(f"{base}/api/materials/ceiling_sample", data=doc).ok
        _open_scheme(page, SAMPLE)
        page.locator("#elevation-wall-ceiling").click()
        page.wait_for_function('state => constructionAt("ceiling", "auto1")?.state === state', arg="original" if with_original else "unknown")
        words = "原本的天花「自編天花板」原樣"
        expect(page.locator("#zone-keep-ceiling-auto1")).to_have_text(f"{words}，不貼東西" if with_original else "不貼，保留原樣")
        now = page.evaluate('plainNow(zoneList("ceiling").find(z => z.id === "auto1").material, "ceiling", "auto1")')
        assert now == (f"沒貼東西（{words}）" if with_original else "不貼，保留原樣")


def test_fixed_only_card_explains_how_to_fix_it_in_both_views(tmp_path: Path, browser: Browser) -> None:
    _prepare(tmp_path, _design({"yL": {"auto1": {"mode": "search", "allowed": [PLAIN_PRODUCT]}}}))
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        for compact in (False, True):
            if compact:
                page.locator("#zone-list-select").click()
            expect(page.locator("#zone-list-problem")).to_have_text(
                "只放了固定成品，程式沒得選。程式要從一個能調的東西開始：加一種可調的材料（例如玻璃棉），或加上「也可以不貼」。"
                "要的就是這一筆、不必讓程式選：" + ("在上面的小框" if compact else "按這張卡的「選這幾區」，") + "改成「指定貼一種」。")
