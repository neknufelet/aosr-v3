"""真瀏覽器考「這間房貼什麼」那一頁（#1176 第 C 支）的推法與看到什麼：「其他牆面」那條規則怎麼從各區推，以及打開那一頁
看到什麼（照畫面按、比存檔的那幾題在 ``test_gui_zone_overview_tasks_browser``）。

推法（``zone_rules.js`` 的純函式，在頁面裡餵手造的區）：出現最多次的做法是規則、平手取最先出現的（跨牆照交進來的順序）、
指定貼不同的材料算不同的做法、沒貼東西的指定貼跟不貼算同一個做法、由程式最佳化的清單當集合比；清單卡列哪一份、哪幾區「另外選」。

畫面用區名考卷那間假房間（``_zone_naming_room`` 的 zone-naming-sample，數字是考卷自己編的；前牆是 yL、後牆 y0、左牆 x0
整面一區、右牆 xL 有門有窗）：一題照存檔原樣打開（沒有設計檔，每一區都是指定貼），一題配一份考卷自己寫的設計檔
（其他牆面由程式最佳化、前牆兩塊照舊、後牆一區清單不一樣、左牆不貼、天花一區由程式最佳化）。比的是具名的區，不是數量。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from playwright.sync_api import Browser, Page, expect

from tests.engine._gui_cache import gui_startup_identity_memo as gui_startup_identity_memo
from tests.engine._zone_naming_room import ZONE_NAMING_SCHEME, write_zone_naming_library
from tests.engine.test_gui_browser import _open, _serve, browser as browser
from tests.engine.test_gui_zone_elevation_browser import _open_scheme, _write

SAMPLE = "zone-naming-sample"
WOOLS = ["glass_wool_48k", "glass_wool_24k", "glass_wool_96k", "none"]
SHORT = ["glass_wool_48k", "none"]
SEARCH = {"mode": "search", "allowed": WOOLS}
# 考卷自己寫的設計檔：沒列到的區照設計檔的規矩當指定貼（前牆 z1、z2，右牆的門與窗，天花其餘、地板）。
SAMPLE_DESIGN: dict[str, object] = {
    "schema_version": "aosr.zone_design.v1", "scheme_id": SAMPLE,
    "zones": {
        "yL": {"z3": SEARCH, "auto1": SEARCH, "auto2": SEARCH},
        "y0": {"z1": SEARCH, "auto2": SEARCH, "auto3": SEARCH, "auto4": SEARCH, "auto1": {"mode": "search", "allowed": SHORT}},
        "xL": {f"auto{n}": SEARCH for n in range(1, 7)},
        "x0": {"whole_wall": {"mode": "keep_original"}},
        "ceiling": {"z1": SEARCH},
    },
}
_RULES = """zones => { const found = inferOtherWalls(zones);
    return {rule: found.rule && {mode: found.rule.mode, zones: found.rule.zones.map(zone => zone.name)},
            special: found.special.map(group => ({mode: group.mode, zones: group.zones.map(zone => zone.name)}))}; }"""
_LIST = """([zones, ruleKey]) => { const list = zoneSearchList(zones, ruleKey);
    return {zones: list.zones.map(zone => zone.name), common: list.common && list.common.zones.map(zone => zone.name),
            different: list.different.map(zone => zone.name)}; }"""
# 圖上每一區畫成哪一種：「牆/區代號」→ search／fixed／still／opening。
_SHOWN = """() => Object.fromEntries([...document.querySelectorAll('#zone-map [data-zone]')].map(zone =>
    [`${zone.dataset.wall}/${zone.dataset.zone}`, zone.dataset.shown]))"""
_GROUPS = """() => [...document.querySelectorAll('#zone-special .zv-group')].map(group =>
    ({zones: group.dataset.zones.split(' '), what: group.querySelector('.zv-what').textContent}))"""
_STATE = "() => JSON.stringify([zoneDesign, scheme])"


def _zone(name: str, mode: str, *, allowed: list[str] | None = None, fixed: str = "") -> dict[str, object]:
    return {"name": name, "mode": mode, "allowed": allowed, "fixed": fixed}


def _rules(page: Page, zones: list[dict[str, object]]) -> dict[str, object]:
    return cast(dict[str, object], page.evaluate(_RULES, zones))


def _open_overview(page: Page) -> None:
    _open_scheme(page, SAMPLE)
    page.locator("#zone-page-open").click()
    expect(page.locator("#zone-map")).to_be_visible()


def _shown(page: Page, kind: str) -> set[str]:
    return {name for name, shown in cast(dict[str, str], page.evaluate(_SHOWN)).items() if shown == kind}


def _dataset_zones(page: Page, selector: str) -> set[str]:
    return set(cast(str, page.locator(selector).get_attribute("data-zones")).split())


def test_the_method_most_wall_zones_share_is_the_rule_and_the_rest_are_listed_apart(tmp_path: Path, browser: Browser) -> None:
    _write(tmp_path, ZONE_NAMING_SCHEME)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        # 由程式最佳化的清單當集合比：順序不同、寫了兩次都算同一種。
        found = _rules(page, [
            _zone("前1", "fixed", fixed="library:a"), _zone("前2", "search", allowed=["x", "y"]), _zone("前3", "keep_original"),
            _zone("後1", "search", allowed=["y", "x"]), _zone("後2", "keep_original"), _zone("左1", "search", allowed=["x", "y", "x"])])
        assert found == {"rule": {"mode": "search", "zones": ["前2", "後1", "左1"]},
                         "special": [{"mode": "fixed", "zones": ["前1"]}, {"mode": "keep_original", "zones": ["前3", "後2"]}]}
        # 清單不一樣就是不一樣的做法（各區清單不同時，少數那幾區算另外指定）。
        found = _rules(page, [_zone("前1", "search", allowed=["x", "y"]), _zone("前2", "search", allowed=["x"]),
                              _zone("後1", "search", allowed=["x", "y"])])
        assert found == {"rule": {"mode": "search", "zones": ["前1", "後1"]}, "special": [{"mode": "search", "zones": ["前2"]}]}
        assert _rules(page, []) == {"rule": None, "special": []}
        assert watched.page_errors == [] and watched.console_errors == []


def test_a_tie_goes_to_the_method_that_appears_first_across_walls(tmp_path: Path, browser: Browser) -> None:
    _write(tmp_path, ZONE_NAMING_SCHEME)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        keep_first = [_zone("前1", "keep_original"), _zone("前2", "search", allowed=["x"]),
                      _zone("後1", "search", allowed=["x"]), _zone("後2", "keep_original")]
        assert _rules(page, keep_first) == {"rule": {"mode": "keep_original", "zones": ["前1", "後2"]},
                                            "special": [{"mode": "search", "zones": ["前2", "後1"]}]}
        # 同樣四區、前牆兩區對調：先出現的換成由程式最佳化，規則跟著換。
        search_first = [keep_first[1], keep_first[0], keep_first[2], keep_first[3]]
        assert _rules(page, search_first) == {"rule": {"mode": "search", "zones": ["前2", "後1"]},
                                              "special": [{"mode": "keep_original", "zones": ["前1", "後2"]}]}
        # 三種各兩區：還是最先出現的那一種。
        three = [_zone("前1", "fixed", fixed="library:a"), _zone("前2", "keep_original"), _zone("後1", "search", allowed=["x"]),
                 _zone("後2", "search", allowed=["x"]), _zone("左1", "keep_original"), _zone("右1", "fixed", fixed="library:a")]
        assert cast(dict[str, dict[str, object]], _rules(page, three))["rule"] == {"mode": "fixed", "zones": ["前1", "右1"]}
        assert watched.page_errors == [] and watched.console_errors == []


def test_fixed_zones_with_different_materials_are_different_methods(tmp_path: Path, browser: Browser) -> None:
    _write(tmp_path, ZONE_NAMING_SCHEME)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        # 指定貼 a 兩區、指定貼 b 兩區、不貼三區：不分材料的話指定貼有四區會變成規則。
        found = _rules(page, [
            _zone("前1", "fixed", fixed="library:a"), _zone("前2", "fixed", fixed="library:b"), _zone("後1", "keep_original"),
            _zone("後2", "fixed", fixed="library:a"), _zone("左1", "keep_original"), _zone("右1", "fixed", fixed="library:b"),
            _zone("右2", "keep_original")])
        assert found == {"rule": {"mode": "keep_original", "zones": ["後1", "左1", "右2"]},
                         "special": [{"mode": "fixed", "zones": ["前1", "後2"]}, {"mode": "fixed", "zones": ["前2", "右1"]}]}
        # 是哪一種：材料庫認得就是那一筆；認不到，貼上去的比那幾層（鍵的順序、值是 null 的鍵不算），沒貼的比整份。
        wool = {"kind": "porous", "thickness_m": 0.03, "flow_resistivity_pa_s_m2": 9100.0}
        same = {"flow_resistivity_pa_s_m2": 9100.0, "thickness_m": 0.03, "kind": "porous", "note": None}
        thicker = wool | {"thickness_m": 0.06}
        bare = {"kind": "constant_impedance", "impedance_pa_s_m": 51000.0}
        on = {"kind": "mounted_on_wall", "wall": bare}
        identity = "([material, id]) => zoneFixedIdentity(material, id)"
        assert page.evaluate(identity, [on | {"layers": [wool]}, "sample_entry"]) == "library:sample_entry"
        layered = page.evaluate(identity, [on | {"layers": [wool]}, None])
        assert layered == page.evaluate(identity, [on | {"layers": [same]}, None])
        assert layered != page.evaluate(identity, [on | {"layers": [thicker]}, None])
        # 同樣幾層貼在不一樣的原牆上還是同一種；沒貼東西的跟貼了的不同。
        assert layered == page.evaluate(identity, [on | {"layers": [wool], "wall": bare | {"impedance_pa_s_m": 62000.0}}, None])
        assert page.evaluate(identity, [bare, None]) != layered
        assert watched.page_errors == [] and watched.console_errors == []


def test_a_fixed_zone_with_nothing_pasted_counts_as_keep_original(tmp_path: Path, browser: Browser) -> None:
    """設計檔沒列到的區照規矩是指定貼，可是上面什麼都沒貼（材料就是原牆原樣）：跟「不貼，保留原樣」同一個做法。
    兩區沒貼的指定貼＋一區不貼＝三區，贏過兩區由程式最佳化；分開算的話是 2:2:1，規則會變成最先出現的那一種。"""
    _write(tmp_path, ZONE_NAMING_SCHEME)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        identity = "([material, id]) => zoneFixedIdentity(material, id)"
        wall = page.evaluate(identity, [{"kind": "constant_impedance", "impedance_pa_s_m": 51000.0}, None])
        other = page.evaluate(identity, [{"kind": "constant_impedance", "impedance_pa_s_m": 62000.0}, None])
        zones = [_zone("前1", "search", allowed=["x"]), _zone("前2", "fixed", fixed=wall), _zone("後1", "search", allowed=["x"]),
                 _zone("後2", "keep_original"), _zone("左1", "fixed", fixed=other), _zone("右1", "fixed", fixed="library:a")]
        assert _rules(page, zones) == {"rule": {"mode": "keep_original", "zones": ["前2", "後2", "左1"]},
                                       "special": [{"mode": "search", "zones": ["前1", "後1"]}, {"mode": "fixed", "zones": ["右1"]}]}
        assert watched.page_errors == [] and watched.console_errors == []


def test_the_list_card_shows_one_list_and_names_the_zones_that_chose_differently(tmp_path: Path, browser: Browser) -> None:
    _write(tmp_path, ZONE_NAMING_SCHEME)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        zones = [_zone("前1", "search", allowed=["x"]), _zone("前2", "keep_original"), _zone("後1", "search", allowed=["x", "y"]),
                 _zone("後2", "search", allowed=["y", "x"]), _zone("天花1", "search", allowed=["x", "y"]), _zone("右1", "fixed", fixed="library:a")]
        # 沒有規則可靠：列最多區用的那一份，其餘「另外選」。
        assert page.evaluate(_LIST, [zones, "keep_original"]) == {
            "zones": ["前1", "後1", "後2", "天花1"], "common": ["後1", "後2", "天花1"], "different": ["前1"]}
        # 規則本身是由程式最佳化：列規則那一份，就算別份的區比較多。
        assert page.evaluate(_LIST, [zones, "search:x"]) == {
            "zones": ["前1", "後1", "後2", "天花1"], "common": ["前1"], "different": ["後1", "後2", "天花1"]}
        # 兩份一樣多：最先出現的那一份。
        tie = [_zone("前1", "search", allowed=["x"]), _zone("後1", "search", allowed=["y"])]
        assert page.evaluate(_LIST, [tie, "keep_original"]) == {"zones": ["前1", "後1"], "common": ["前1"], "different": ["後1"]}
        assert page.evaluate(_LIST, [[zones[1], zones[5]], "keep_original"]) == {"zones": [], "common": None, "different": []}
        assert watched.page_errors == [] and watched.console_errors == []


def test_the_sample_room_opens_as_saved_with_a_rule_its_exceptions_and_nothing_written(tmp_path: Path, browser: Browser) -> None:
    """沒有設計檔：每一區照規矩都是指定貼。四面牆扣門窗 17 區裡 13 區沒貼東西，算「不貼，保留原樣」，那就是規則；
    前牆兩塊同一種、前牆一塊、後牆一塊各是另外指定。打開那一頁只看不改：頁面上的方案與做法一個字都不變，也不多出設計檔
    （沒貼的那 13 區不會因為畫面叫它不貼就被寫成不貼）。"""
    _write(tmp_path, ZONE_NAMING_SCHEME)
    write_zone_naming_library(tmp_path)
    scheme_bytes = (tmp_path / "schemes" / f"{SAMPLE}.json").read_bytes()
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, SAMPLE)
        before = page.evaluate(_STATE)
        page.locator("#zone-page-open").click()
        expect(page.locator("#zone-map")).to_be_visible()
        expect(page.locator("main")).to_be_hidden()
        expect(page.locator("#zone-page h1")).to_have_text("這間房貼什麼")
        # 規則：沒貼東西的那 13 區，畫面上叫不貼（下拉顯示「不貼，保留原樣」，沒有材料那一格）。
        expect(page.locator("#zone-rule")).to_have_attribute("data-mode", "keep_original")
        expect(page.locator("#zone-rule-mode")).to_have_value("keep_original")
        expect(page.locator("#zone-rule-mode option:checked")).to_have_text("不貼，保留原樣")
        expect(page.locator("#zone-rule-material")).to_have_count(0)
        groups = cast(list[dict[str, object]], page.evaluate(_GROUPS))
        assert [group["zones"] for group in groups] == [["yL/z1", "yL/z2"], ["yL/z3"], ["y0/z1"]]
        assert all(cast(str, group["what"]).startswith("指定貼一種：") for group in groups)
        expect(page.locator("#zone-special h2")).to_have_text("另外指定的 4 區")
        expect(page.locator("#zone-special-reset")).to_have_text("這 4 區改回跟其他牆面一樣")
        expect(page.locator("#zone-special .zv-group").first.locator(".zv-where-line").first).to_have_text("前牆 1 左聲道喇叭後面那塊")
        # 圖：貼了東西的六塊是指定貼的顏色，門窗畫斜線，沒有哪一區由程式最佳化；地毯算保留原樣。
        assert _shown(page, "fixed") == {"yL/z1", "yL/z2", "yL/z3", "y0/z1", "ceiling/z1", "ceiling/z2"}
        assert _shown(page, "opening") == {"xL/door", "xL/window"}
        assert _shown(page, "search") == set()
        assert "floor/rug" in _shown(page, "still") and "x0/whole_wall" in _shown(page, "still")
        expect(page.locator("#zone-others")).to_have_text("天花：指定貼 2 區、保留原樣 4 區地板、門、窗：保留原樣")
        expect(page.locator("#zone-list")).to_have_text("現在沒有哪一區由程式最佳化")
        assert _dataset_zones(page, "#zone-result-fixed") == _shown(page, "fixed")
        expect(page.locator("#zone-result-search")).to_contain_text("由程式最佳化 0 區")
        expect(page.locator("#zone-result-still")).to_have_text("保留原樣：牆面 13 區、天花 4 區、地板、門 1 扇、窗 1 扇")
        expect(page.locator("#zone-page-search")).to_be_disabled()
        # 代號與尺寸平常收著，點開才有。
        expect(page.locator("#zone-details table")).to_be_hidden()
        page.locator("#zone-details summary").click()
        expect(page.locator("#zone-details tr[data-zone='yL/z3']")).to_contain_text("代號 z3")
        page.screenshot(path=str(tmp_path / "overview-as-saved.png"))
        # 只看不改；回到原本那一頁照樣在。
        assert page.evaluate(_STATE) == before
        page.locator("#zone-page-close").click()
        expect(page.locator("main")).to_be_visible()
        expect(page.locator("#zone-page")).to_be_hidden()
        assert page.evaluate(_STATE) == before
        assert watched.page_errors == [] and watched.console_errors == []
    assert not (tmp_path / "designs" / f"{SAMPLE}.json").exists()
    assert (tmp_path / "schemes" / f"{SAMPLE}.json").read_bytes() == scheme_bytes


def test_a_design_with_a_search_rule_shows_the_list_card_and_the_zones_that_chose_differently(tmp_path: Path, browser: Browser) -> None:
    _write(tmp_path, ZONE_NAMING_SCHEME)
    write_zone_naming_library(tmp_path)
    (tmp_path / "designs").mkdir()
    design_file = tmp_path / "designs" / f"{SAMPLE}.json"
    design_file.write_text(json.dumps(SAMPLE_DESIGN), encoding="utf-8")
    design_bytes = design_file.read_bytes()
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        # 規則：由程式最佳化（13 區同一份清單）。另外指定照最先出現的順序：前牆兩塊指定貼、後牆清單不一樣的那一區、左牆不貼。
        expect(page.locator("#zone-rule")).to_have_attribute("data-mode", "search")
        expect(page.locator("#zone-rule-mode")).to_have_value("search")
        expect(page.locator("#zone-rule-material")).to_have_count(0)
        groups = cast(list[dict[str, object]], page.evaluate(_GROUPS))
        assert [group["zones"] for group in groups] == [["yL/z1", "yL/z2"], ["y0/auto1"], ["x0/whole_wall"]]
        assert [cast(str, group["what"]).split("：")[0] for group in groups] == ["指定貼一種", "由程式最佳化", "不貼，保留原樣"]
        expect(page.locator("#zone-special h2")).to_have_text("另外指定的 4 區")
        searching = ({f"yL/{key}" for key in ("z3", "auto1", "auto2")} | {f"y0/{key}" for key in ("z1", "auto1", "auto2", "auto3", "auto4")}
                     | {f"xL/auto{n}" for n in range(1, 7)} | {"ceiling/z1"})
        assert _shown(page, "search") == searching
        assert _shown(page, "fixed") == {"yL/z1", "yL/z2", "ceiling/z2"}
        assert _shown(page, "opening") == {"xL/door", "xL/window"}
        assert _dataset_zones(page, "#zone-result-search") == searching
        # 清單卡：管全部由程式最佳化的區，列規則那一份；清單不一樣的那一區另外數。
        expect(page.locator("#zone-list h2")).to_have_text("由程式最佳化的 15 區每一區各自最佳化，都從這幾種裡選")
        expect(page.locator("#zone-list-who")).to_have_text("現在管 15 區：沒另外指定的牆區、後牆 5 靠地板那條、天花 1 主位上面那塊")
        assert page.locator("#zone-list-checks label").evaluate_all("rows => rows.map(row => row.dataset.choice)") == WOOLS
        expect(page.locator("#zone-list-checks label").last).to_have_text("也可以不貼（保留原樣）")
        expect(page.locator("#zone-list-different summary")).to_have_text("有 1 區另外選")
        assert _dataset_zones(page, "#zone-list-different") == {"y0/auto1"}
        # 可調的各自能調多少直接寫在那一列底下（#1199：不再收在「玻璃棉各自能調多厚」後面）。
        expect(page.locator("#zone-list-ranges")).to_have_count(0)
        expect(page.locator("#zone-list-checks label[data-choice=glass_wool_48k] .zv-range")).to_have_text("可調：厚 5～20 cm、空腔 0～25 cm，加起來最深 30 cm")
        expect(page.locator("#zone-list-checks label[data-choice=glass_wool_24k] .zv-range")).to_have_text("可調：厚 5～20 cm、空腔：沒有（直接貼牆）")
        expect(page.locator("#zone-others")).to_have_text("天花：由程式最佳化 1 區、指定貼 1 區、保留原樣 4 區地板、門、窗：保留原樣")
        expect(page.locator("#zone-page-search")).to_be_enabled()
        page.screenshot(path=str(tmp_path / "overview-search-rule.png"))
        assert watched.page_errors == [] and watched.console_errors == []
    assert design_file.read_bytes() == design_bytes
