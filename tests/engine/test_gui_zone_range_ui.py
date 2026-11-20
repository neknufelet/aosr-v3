"""每一區自己的可調範圍（#1199 範圍第 2 支）在頁面裡的純函式：清單卡怎麼照範圍分張（``zone_rules.js::zoneSearchList``）、
做法換掉或清單拿掉一種時範圍跟著拿掉（``zone_modes.js::settledSetting``）、第三階段頁預覽怎麼唸範圍（``gradient.js::rangeLines``）。
小框裡套用時每一區寫什麼、公分與公尺怎麼換住在 ``zone_overview.js`` 的閉包裡（頁面外叫不到），由 ``test_gui_zone_range_browser``
照畫面按、比寫死的公尺。函式在頁面裡餵手造的區與材料（數字是考卷自己編的），比的是具名的區與寫死的數字。"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from playwright.sync_api import Browser

from tests.engine._gui_cache import gui_startup_identity_memo as gui_startup_identity_memo
from tests.engine._zone_naming_room import ZONE_NAMING_SCHEME
from tests.engine.test_gui_browser import _open, _serve, browser as browser
from tests.engine.test_gui_zone_elevation_browser import _write

_LIST = """([zones, ruleKey]) => { const list = zoneSearchList(zones, ruleKey);
    return {common: list.common && list.common.zones.map(zone => zone.name),
            groups: list.groups.map(group => group.zones.map(zone => zone.name)), different: list.different.map(zone => zone.name)}; }"""
Doc = dict[str, object]


def _zone(name: str, allowed: list[str], ranges: Mapping[str, object] | None = None) -> Doc:
    return {"name": name, "mode": "search", "allowed": allowed, "fixed": "", **({"ranges": ranges} if ranges else {})}


def test_list_cards_split_one_list_by_range_and_keep_the_rule_card_first(tmp_path: Path, browser: Browser) -> None:
    """同一份清單：沒寫範圍的一張、寫了同一份範圍的一張（材料順序不同算同一份）、寫了另一份的又一張；規則那一份裡最多區的那一張排第一。
    沒寫範圍的區照第 1 支一張（控制）。"""
    _write(tmp_path, ZONE_NAMING_SCHEME)
    narrow = {"deep": {"porous_m": [0.04, 0.08]}, "flat": {"porous_m": [0.08, 0.12]}}
    reordered = {"flat": {"porous_m": [0.08, 0.12]}, "deep": {"porous_m": [0.04, 0.08]}}
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        zones = [_zone("前1", ["deep", "flat"], narrow), _zone("前2", ["deep", "flat"]), _zone("後1", ["deep", "flat"], reordered),
                 _zone("後2", ["deep", "flat"], {"deep": {"porous_m": [0.08, 0.12]}}), _zone("天花1", ["deep"])]
        assert page.evaluate(_LIST, [zones, "search:deep,flat"]) == {
            "common": ["前1", "後1"], "groups": [["前1", "後1"], ["前2"], ["後2"], ["天花1"]], "different": ["前2", "後2", "天花1"]}
        plain = [_zone("前1", ["deep", "flat"]), _zone("前2", ["deep", "flat"]), _zone("天花1", ["deep"])]
        assert page.evaluate(_LIST, [plain, "keep_original"]) == {
            "common": ["前1", "前2"], "groups": [["前1", "前2"], ["天花1"]], "different": ["天花1"]}
        assert watched.page_errors == [] and watched.console_errors == []


def test_a_range_goes_away_with_its_material_or_with_the_program_choosing(tmp_path: Path, browser: Browser) -> None:
    """``settledSetting``：清單拿掉 deep，deep 那一份範圍跟著拿掉、flat 的留著；一種都不剩整格不寫；改成不貼也不寫；
    沒改清單、範圍都在清單裡的原封不動。"""
    _write(tmp_path, ZONE_NAMING_SCHEME)
    ranges = {"deep": {"porous_m": [0.04, 0.08], "cavity_m": [0.0, 0.1]}, "flat": {"porous_m": [0.08, 0.12]}}
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        settle = "([setting, kept]) => settledSetting(setting, kept)"
        whole = {"mode": "search", "allowed": ["deep", "flat", "none"], "ranges": ranges}
        assert page.evaluate(settle, [whole, None]) == whole
        assert page.evaluate(settle, [{**whole, "allowed": ["flat", "none"]}, None]) == {
            "mode": "search", "allowed": ["flat", "none"], "ranges": {"flat": {"porous_m": [0.08, 0.12]}}}
        assert page.evaluate(settle, [{**whole, "allowed": ["none"]}, None]) == {"mode": "search", "allowed": ["none"]}
        assert page.evaluate(settle, [{"mode": "keep_original", "ranges": ranges}, None]) == {"mode": "keep_original"}
        assert watched.page_errors == [] and watched.console_errors == []


def test_preview_lines_read_out_centimetres_and_leave_out_the_cavity_when_it_is_null(tmp_path: Path, browser: Browser) -> None:
    """第三階段頁 ``gradient.js::rangeLines``：預覽回的 ``ranges``（公尺）一種一行唸成公分；不調空腔的那一種 ``cavity_m`` 是 null，
    只寫棉；沒寫範圍的一片沒有這一行。"""
    _write(tmp_path, ZONE_NAMING_SCHEME)
    preview = {"labels": {"deep": "深的那種", "flat": "平的那種"},
               "ranges": {"yL@z3": {"deep": {"porous_m": [0.04, 0.08], "thicknesses_m": [0.04, 0.08], "cavity_m": [0.0, 0.15]},
                                    "flat": {"porous_m": [0.12, 0.12], "thicknesses_m": [0.12], "cavity_m": None}}}}
    lines = "surface => rangeLines(surface).map(line => [line.dataset.family, line.textContent])"
    with _serve(tmp_path) as base, _open(browser, f"{base}/gradient") as watched:
        page = watched.page
        page.evaluate("preview => { lastPreview = preview; }", preview)
        assert page.evaluate(lines, "yL@z3") == [["deep", "深的那種：棉 4～8 cm、空腔 0～15 cm"], ["flat", "平的那種：棉 12 cm"]]
        assert page.evaluate(lines, "yL@z1") == []
        assert watched.page_errors == [] and watched.console_errors == []
