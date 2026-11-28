"""分區頁的材料指定、選整面、候選清單與存檔。資料使用區名考卷的合成房間與材料。
前牆整面只留 48K 與不貼的情境，用同一支工具計數五個動作，再比存檔。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from playwright.sync_api import Browser, Locator, Page, expect
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from tests.engine._zone_material_steps import open_groups
from tests.engine._zone_material_steps import row_options
from tests.engine._gui_cache import gui_startup_identity_memo as gui_startup_identity_memo
from tests.engine.test_gui_browser import _open, _serve, browser as browser
from tests.engine.test_gui_zone_elevation_browser import _open_scheme
from tests.engine.test_gui_zone_overview_browser import SAMPLE, SAMPLE_DESIGN, WOOLS
from tests.engine.test_gui_zone_overview_tasks_browser import (
    _ALLOWED, _apply, _click_zone, _expect_selected, _materials_before, _open_overview, _prepare, _save, _send_to_search,
    _settled, _stored)

FRONT = ["yL/z1", "yL/z2", "yL/z3", "yL/auto1", "yL/auto2"]
RIGHT_WITHOUT_OPENINGS = [f"xL/auto{n}" for n in range(1, 7)]
ONLY_48K = ["glass_wool_48k", "none"]
FLOOR_ONLY = "sample_carpet_11mm"  # 假材料庫裡只有地板能用的那一筆（考卷裡再跟網頁讀到的核一次）
# 一筆一般的固定成品（內建）：沒被哪一種可調玻璃棉包含、也不是「鋼板直接貼在棉上」的構造（#1208 起那一類勾不進清單，另有考卷）。
PLAIN_PRODUCT = "slotted_wood_typ9_absorber40_gap98"
OLD_WORDS = "交給" + "電腦挑"  # 舊的那一句（拆開寫：改字時整個 repo 搜這一句要是 0 筆）
# 小框與清單卡裡每一段看得到的字（清單卡按鈕除外）：字級（px）、字色對它底下那一層底色的對比。
# 字色照實際畫出來的算：字色自己的透明度、從這一層往上每一層的 opacity 都乘進去，跟底色混過再比（灰掉的列不准靠調淡）。
_TEXT_STYLES = """root => {
    const channel = value => { const share = value / 255; return share <= 0.03928 ? share / 12.92 : ((share + 0.055) / 1.055) ** 2.4; };
    const parse = text => text.match(/[\\d.]+/g).map(Number);
    const light = ([r, g, b]) => 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b);
    const behind = element => {
        for (let at = element; at; at = at.parentElement) {
            const [r, g, b, alpha = 1] = parse(getComputedStyle(at).backgroundColor);
            if (alpha > 0) return [r, g, b];
        }
        return [255, 255, 255];
    };
    const ratio = (a, b) => (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05);
    const rows = [];
    for (const element of root.querySelectorAll('*')) {
        if ((!root.matches('#zone-panel') && element.closest('button')) || !element.getClientRects().length) continue;
        const words = [...element.childNodes].filter(node => node.nodeType === 3 && node.textContent.trim()).map(node => node.textContent.trim()).join('');
        if (!words) continue;
        const style = getComputedStyle(element);
        const [r, g, b, alpha = 1] = parse(style.color), under = behind(element);
        let seen = alpha;
        for (let at = element; at && at !== document.documentElement; at = at.parentElement) seen *= parseFloat(getComputedStyle(at).opacity);
        const ink = light([r, g, b].map((value, index) => value * seen + under[index] * (1 - seen))), paper = light(under);
        const row = {words, size: parseFloat(style.fontSize), contrast: ratio(ink, paper)};
        if (element.matches('.zv-dots .zv-num')) {
            row.border = ratio(light(parse(style.borderTopColor)), light(behind(element.parentElement)));
            row.stroke = style.borderTopStyle;
        }
        rows.push(row);
    }
    return rows;
}"""


class _Actions:
    """數使用者按了幾下：考卷裡每一下都從這裡按。"""

    def __init__(self) -> None:
        self.done: list[str] = []

    def click(self, target: Locator, what: str) -> None:
        target.click()
        self.done.append(what)

    def fill(self, target: Locator, value: str) -> None:
        target.fill(value)
        self.done.append(value)


def _boss_room(tmp_path: Path) -> None:
    _prepare(tmp_path, _design({"yL": {
        "z3": {"mode": "search", "allowed": [WOOLS[0], WOOLS[1], "none"]},
        "auto2": {"mode": "search", "allowed": [WOOLS[0], WOOLS[2], "none"]},
        "auto1": {"mode": "search", "allowed": [WOOLS[1], "none"]},
    }}))
    path = tmp_path / "materials.json"
    library = json.loads(path.read_bytes())
    layers = _materials_before()["yL"]["z1"]["layers"]
    library["materials"].append({"id": "sample_front_panel", "kind": "fixed", "label": "自編前牆吸音板",
                                 "source": "考卷自編", "provenance": "model",
                                 "material": {"kind": "assembly", "layers": layers, "backing": "rigid"}})
    path.write_text(json.dumps(library))


def _add_candidate(page: Page, choice: str) -> None:
    page.locator("#zone-panel-picker-search").focus()
    page.locator(f"#zone-panel-pick-{choice}").click()


_STATES_HELD = """wanted => Object.entries(wanted).every(([choice, state]) => {
  const row = document.querySelector(`#zone-panel-list [data-choice="${choice}"]`);
  return (row ? row.getAttribute("data-state") || "none" : "none") === state;
})"""
_STATES_NOW = """choices => Object.fromEntries(choices.map(choice => {
  const row = document.querySelector(`#zone-panel-list [data-choice="${choice}"]`);
  return [choice, row ? row.getAttribute("data-state") || "none" : "none"];
}))"""


def _expect_candidate_states(page: Page, wanted: dict[str, str]) -> None:
    """等小框清單每一列畫成要的狀態（沒有那一列算 none）。開著左右對稱時，點區要先等限制檢查回來才換選取
    （``zone_overview.js`` 的 ``pick``，#1231：本機連跑 5 次紅 5 次）：讀一次就判會讀到上一個畫面。按鈕那幾條路是當下重畫，
    等一下也無妨。等 10 秒還不對才紅，並印出當下每一列的狀態。"""
    assert wanted, "至少要指定一列，空的一定綠"
    try:
        page.wait_for_function(_STATES_HELD, arg=wanted, timeout=10_000)
    except PlaywrightTimeoutError as exc:
        raise AssertionError(f"等了 10 秒，清單是 {page.evaluate(_STATES_NOW, list(wanted))}，要 {wanted}") from exc


def _pick_search(page: Page) -> None:
    page.locator("#zone-panel-mode-search").check()
    expect(page.locator("#zone-panel-form")).to_be_visible()


def test_the_front_wall_is_handed_to_the_program_with_only_48k_or_nothing_in_five_actions(tmp_path: Path, browser: Browser) -> None:
    """選整面、選做法、只留 48K、輸入 15、套用。五個動作後存檔核對每區。"""
    _boss_room(tmp_path)
    before = _materials_before()
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        actions = _Actions()
        actions.click(page.locator("#zone-whole-yL"), "前牆「選整面」")
        _expect_selected(page, FRONT)
        actions.click(page.locator("#zone-panel-mode-search"), "由程式最佳化")
        page.locator("#zone-panel-row-glass_wool_48k").hover()
        page.locator("#zone-panel-options-glass_wool_48k summary").hover()
        actions.click(page.locator("#zone-panel-only-glass_wool_48k"), "清掉其他候選")
        actions.fill(page.locator("#zone-panel-depth"), "15")
        expect(page.locator("#zone-panel-apply")).to_have_text("套用到 5 區")
        actions.click(page.locator("#zone-panel-apply"), "套用")
        expect(page.locator("#zone-panel")).to_have_count(0)
        _settled(page)
        assert len(actions.done) == 5, actions.done
        # 畫面上：前牆五區都是由程式最佳化的顏色，底列數得到五區。
        for zone in FRONT:
            wall, key = zone.split("/")
            assert page.locator(f"#zone-map [data-wall='{wall}'][data-zone='{key}']").get_attribute("data-shown") == "search"
        expect(page.locator("#zone-result-search")).to_have_attribute("data-count", "5")
        expect(page.locator("#zone-result-search")).to_contain_text("由程式最佳化 5 區")
        # 右欄把這五區寫成一行「前牆整面」，清單卡只剩 48K 玻璃棉與也可以不貼。
        open_groups(page)
        expect(page.locator("#zone-special [data-whole='yL']")).to_have_text("前牆整面 12345")
        expect(page.locator("#zone-group-who")).to_have_text("這組有 5 區：前牆整面 12345")
        assert page.locator("#zone-group-choices > [data-choice]").evaluate_all("rows => rows.map(row => row.dataset.choice)") == ONLY_48K
        _send_to_search(page)
        assert watched.page_errors == [] and watched.console_errors == []
    zones, materials = _stored(tmp_path)
    assert set(zones) == {"yL"} and set(zones["yL"]) == {zone.split("/")[1] for zone in FRONT}
    for setting in cast(dict[str, dict[str, object]], zones["yL"]).values():
        assert setting == {"mode": "search", "allowed": setting["allowed"], "max_depth_m": 0.15}
        assert sorted(cast(list[str], setting["allowed"])) == sorted(ONLY_48K)
    # 前牆貼著東西的三區放回原牆，其餘每一面都沒動。
    for key in ("z1", "z2", "z3"):
        assert materials["yL"][key] == before["yL"][key]["wall"]
    assert {wall: zones_ for wall, zones_ in materials.items() if wall != "yL"} == {wall: zones_ for wall, zones_ in before.items() if wall != "yL"}


def test_select_whole_face_takes_every_zone_of_that_wall_except_doors_and_windows(tmp_path: Path, browser: Browser) -> None:
    """四面牆的名字旁各一顆「選整面」：一次選那一面全部的區，門、窗不算。選了整面，小框只寫一行「右牆整面」加號碼，
    要拿掉幾區就點開「看每一區、拿掉幾區」；只選了一區時，小框有一條「改選右牆整面（6 區）」。"""
    _prepare(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        assert page.locator("#zone-page [id^='zone-whole-']").evaluate_all("buttons => buttons.map(button => button.id).sort()") == [
            "zone-whole-x0", "zone-whole-xL", "zone-whole-y0", "zone-whole-yL"]
        expect(page.locator("#zone-whole-xL")).to_have_text("選整面")
        page.locator("#zone-whole-xL").click()
        _expect_selected(page, RIGHT_WITHOUT_OPENINGS)
        expect(page.locator("#zone-panel-apply")).to_have_text("套用到 6 區")
        expect(page.locator("#zone-panel h2")).to_have_text("右牆 345678 整面")
        assert page.locator("#zone-panel h2 .zv-num").count() == 6
        # 一區一行的那幾列收在摺疊裡；點開拿掉一區，就不再是整面。
        expect(page.locator("#zone-panel-zones summary")).to_have_text("整面選區")
        expect(page.locator("#zone-panel-drop-xL-auto3")).to_be_hidden()
        page.locator("#zone-panel-zones summary").click()
        page.locator("#zone-panel-drop-xL-auto3").click()
        expect(page.locator("#zone-panel-apply")).to_have_text("套用到 5 區")
        expect(page.locator("#zone-panel h2")).not_to_contain_text("整面")
        expect(page.locator("#zone-panel-whole")).to_have_text("改選右牆整面（6 區）")
        page.locator("#zone-panel-zones summary").click()
        page.locator("#zone-panel-whole").click()
        _expect_selected(page, RIGHT_WITHOUT_OPENINGS)
        # 只點圖上一區：也有那一條；選到的跨兩面牆就沒有。
        page.locator("#zone-panel-cancel").click()
        _click_zone(page, "yL/z3")
        expect(page.locator("#zone-panel-whole")).to_have_text("改選前牆整面（5 區）")
        _click_zone(page, "xL/auto2")
        expect(page.locator("#zone-panel-whole")).to_have_count(0)
        assert watched.page_errors == [] and watched.console_errors == []


def test_the_panel_marks_what_each_picked_zone_is_now_and_cancel_changes_nothing(tmp_path: Path, browser: Browser) -> None:
    """小框打開時，每一區現在的做法旁邊掛「目前」：選到的都同一種就只有那一項掛（指定貼的寫現在貼的是什麼）；
    做法不同就每一項各掛一個、寫是哪幾區。取消鈕旁寫「取消＝不改」，按了什麼都沒變。"""
    _prepare(tmp_path, SAMPLE_DESIGN)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        state = page.evaluate("() => JSON.stringify([zoneDesign, scheme])")
        _click_zone(page, "yL/z1")
        expect(page.locator("#zone-panel [id^='zone-panel-now-']")).to_have_count(1)
        expect(page.locator("#zone-panel-now-fixed")).to_have_text("目前指定貼一種")
        expect(page.locator("#zone-panel-now-fixed")).to_have_attribute("data-zones", "yL/z1")
        expect(page.locator("#zone-panel-now-fixed + .zv-now-what")).to_contain_text("160 mm")
        assert "原本" not in page.locator("#zone-panel").inner_text()
        # 再加選一區由程式最佳化的、一區不貼的：三項各掛一個「目前」，各寫是哪一區的號碼。
        _click_zone(page, "yL/z3")
        _click_zone(page, "x0/whole_wall")
        expect(page.locator("#zone-panel [id^='zone-panel-now-']")).to_have_count(3)
        for mode, zone in (("fixed", "yL/z1"), ("search", "yL/z3"), ("keep_original", "x0/whole_wall")):
            expect(page.locator(f"#zone-panel-now-{mode}")).to_have_attribute("data-zones", zone)
        number = page.locator("#zone-map [data-wall='yL'][data-zone='z3'] .zv-badge text").text_content()
        expect(page.locator("#zone-panel-now-search + .zv-now-what")).to_have_text(f"前牆 {number}")
        expect(page.locator("#zone-panel-now-keep_original + .zv-now-what")).to_have_text("左牆整面")
        # 點別的選項，「目前」不跟著跑（它講的是打開時的樣子）。
        page.locator("#zone-panel-mode-keep_original").check()
        expect(page.locator("#zone-panel-now-fixed")).to_have_attribute("data-zones", "yL/z1")
        expect(page.locator("#zone-panel-cancel")).to_have_text("取消")
        page.locator("#zone-panel-cancel").click()
        expect(page.locator("#zone-panel")).to_have_count(0)
        assert page.evaluate("() => JSON.stringify([zoneDesign, scheme])") == state
        assert watched.page_errors == [] and watched.console_errors == []


def test_allowed_materials_are_laid_out_one_per_row_with_their_ranges_as_plain_text(tmp_path: Path, browser: Browser) -> None:
    """選「由程式最佳化」：材料框裡可調的一種一列，各寫調什麼、從多少到多少；框裡除了勾選格，只有每一種底下改這一區範圍的那幾格
    （#1199 範圍第 2 支，怎麼改見 ``test_gui_zone_range_browser``）；框上一句說範圍從材料頁帶來、只改這一區，旁邊一條去材料頁的連結。
    「也可以不貼」與「這一區最深」在材料框外面。"""
    _prepare(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        _click_zone(page, "yL/auto1")
        _expect_selected(page, ["yL/auto1"])  # 小框換成這一區之後再判，不然還沒畫出來就先過
        expect(page.locator("#zone-panel-form")).to_have_count(0)
        _pick_search(page)
        rows = page.locator("#zone-panel-list .zv-candidate[data-kind='optimizable']")
        assert rows.evaluate_all("rows => rows.map(row => row.dataset.choice)") == ["glass_wool_48k", "glass_wool_24k", "glass_wool_96k"]
        expect(rows.nth(0)).to_contain_text("48K 玻璃棉")
        expect(rows.nth(0).locator(".zv-range")).to_have_text("棉厚 5–20 cm背後空腔 0–25 cm兩個加起來最多 30 cm（材料本身）")
        expect(rows.nth(1).locator(".zv-range")).to_have_text("棉厚 5–20 cm背後空腔 沒有（直接貼牆）")
        expect(rows.nth(2).locator(".zv-range")).to_have_text("棉厚 5–20 cm背後空腔 沒有（直接貼牆）")
        cells = page.locator("#zone-panel-list input, #zone-panel-list select, #zone-panel-list textarea")
        expect(cells).to_have_count(0)
        for choice in WOOLS[:-1]:
            row_options(page, choice)
            page.locator(f"#zone-panel-range-toggle-{choice}").click()
        assert cells.evaluate_all("cells => cells.map(cell => cell.id)") == [
            f"zone-panel-range-{choice}-{part}-{end}" for choice, parts in (
                ("glass_wool_48k", ("porous", "cavity")), ("glass_wool_24k", ("porous",)), ("glass_wool_96k", ("porous",)))
            for part in parts for end in ("low", "high")]
        expect(page.locator("#zone-panel-range-note")).to_contain_text("範圍只改這一區，不改材料頁")
        expect(page.locator("#zone-panel-materials")).to_have_attribute("href", "/materials")
        expect(page.locator("#zone-panel-search-from")).to_have_text("程式從下面勾的 4 種裡選，每一區各自最佳化。")
        expect(page.locator("#zone-panel-list #zone-panel-none-row")).to_contain_text("也可以不貼")
        expect(page.locator("#zone-panel-pick-none")).to_have_text("也可以不貼")
        expect(page.locator("#zone-panel-depth")).to_be_visible()
        expect(page.locator("#zone-panel-deepest label")).to_have_text("最深（含空腔）")
        _click_zone(page, "yL/auto2")
        expect(page.locator("#zone-panel-apply")).to_have_text("套用到 2 區")
        # 「材料的勾全拿掉」不動「也可以不貼」；一種都沒勾（連不貼也拿掉）就不能套用。
        row_options(page, "glass_wool_48k")
        page.locator("#zone-panel-only-glass_wool_48k").click()
        row_options(page, "glass_wool_48k")
        page.locator("#zone-panel-remove-glass_wool_48k").click()
        assert page.locator("#zone-panel-list .zv-candidate[data-choice]").evaluate_all("rows => rows.map(row => row.dataset.choice)") == ["none"]
        expect(page.locator("#zone-panel-none-row")).to_have_attribute("data-state", "all")
        expect(page.locator("#zone-panel-apply")).to_be_enabled()
        row_options(page, "none")
        page.locator("#zone-panel-remove-none").click()
        expect(page.locator("#zone-panel-apply")).to_be_disabled()
        expect(page.locator("#zone-panel-problem")).to_contain_text("至少勾一種")
        assert watched.page_errors == [] and watched.console_errors == []


def test_products_covered_by_a_checked_family_are_read_only_members(tmp_path: Path, browser: Browser) -> None:
    """三種玻璃棉各自展開唯讀規格；取消家族後可單選成品，套用後保留選擇。"""
    _prepare(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        open_groups(page)
        covered = cast(dict[str, list[str]], page.evaluate(
            "() => Object.fromEntries(zoneLibrary.filter(entry => entry.kind === 'optimizable').map(entry => [entry.id, entry.preset_members]))"))
        assert len(covered["glass_wool_48k"]) == 4 and len(covered["glass_wool_24k"]) == 1 and len(covered["glass_wool_96k"]) == 1
        _click_zone(page, "yL/auto1")
        _pick_search(page)
        page.locator("#zone-panel-picker-search").focus()
        for family in WOOLS[:-1]:
            detail = page.locator(f"#zone-panel-members-{family}")
            expect(detail.locator("summary")).to_have_text(f"含 {len(covered[family])} 款")
            detail.locator("summary").click()
            assert detail.locator("[data-member]").evaluate_all("els => els.map(e => e.dataset.member)") == covered[family]
            for product in covered[family]:
                expect(page.locator(f"#zone-panel-pick-{product}")).to_have_count(0)
                expect(detail.locator(f"[data-member={product}]")).to_be_visible()
            expect(detail.locator("button,input")).to_have_count(0)
        expect(page.locator(f"#zone-panel-pick-{PLAIN_PRODUCT}")).to_be_enabled()
        # 取消 48K：它那四筆勾得到了，24K、96K 的照舊灰著。
        row_options(page, "glass_wool_48k")
        page.locator("#zone-panel-remove-glass_wool_48k").click()
        page.locator("#zone-panel-picker-search").focus()
        for product in covered["glass_wool_48k"]:
            expect(page.locator(f"#zone-panel-pick-{product}")).to_be_enabled()
            expect(page.locator(f"#zone-panel-pick-{product} .zv-covered-why")).to_have_count(0)
        expect(page.locator(f"#zone-panel-pick-{covered['glass_wool_24k'][0]}")).to_have_count(0)
        page.locator(f"#zone-panel-pick-{covered['glass_wool_48k'][0]}").click()
        _apply(page)
        assert cast(dict[str, list[str]], page.evaluate(_ALLOWED))["yL/auto1"] == [
            "glass_wool_24k", "glass_wool_96k", "none", covered["glass_wool_48k"][0]]
        # 從清單卡重新打開小框，也遵守同一條規則：清單裡的 24K 已經包含的那一筆灰著、勾不了、寫為什麼；48K 不在清單裡，它那幾筆勾得到。
        page.locator("#zone-group-select").click()
        page.locator("#zone-panel-picker-search").focus()
        inside = covered["glass_wool_24k"][0]
        expect(page.locator(f"#zone-panel-pick-{inside}")).to_have_count(0)
        detail = page.locator("#zone-panel-members-glass_wool_24k")
        expect(detail.locator("summary")).to_have_text("含 1 款")
        if detail.get_attribute("open") is None:
            detail.locator("summary").click()
        expect(detail.locator(f"[data-member={inside}]")).to_be_visible()
        expect(page.locator(f"#zone-panel-pick-{covered['glass_wool_48k'][1]}")).to_be_enabled()
        assert watched.page_errors == [] and watched.console_errors == []


def test_fixed_products_stay_folded_and_the_line_says_how_many_there_are_and_how_many_are_ticked(tmp_path: Path, browser: Browser) -> None:
    """固定成品平常收著，那一行寫這一面用得到的有幾種、勾了幾種（照材料庫算）；點開勾一種，數字跟著變，重畫後照樣開著。"""
    _prepare(tmp_path)
    vpr = PLAIN_PRODUCT
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        count = cast(int, page.evaluate("() => zoneLibrary.filter(entry => entry.kind === 'fixed' && entry.placement.surfaces.includes('yL') && !zoneLibrary.some(f => f.preset_members?.includes(entry.id))).length"))
        assert count > 4
        _click_zone(page, "yL/auto1")
        _pick_search(page)
        expect(page.locator("#zone-panel-picker")).to_be_visible()
        expect(page.locator(f"#zone-panel-row-{vpr}")).to_have_count(0)
        page.locator("#zone-panel-picker-search").focus()
        assert page.locator("#zone-panel-picker [data-entry-kind='fixed'][aria-checked=false]").count() == count
        page.locator(f"#zone-panel-pick-{vpr}").click()
        expect(page.locator("#zone-panel-picker")).to_be_visible()
        expect(page.locator(f"#zone-panel-row-{vpr}")).to_be_visible()
        page.locator("#zone-panel-picker-search").focus()
        assert page.locator("#zone-panel-picker [data-entry-kind='fixed'][aria-checked=false]").count() == count - 1
        expect(page.locator("#zone-panel-search-from")).to_have_text("程式從下面勾的 5 種裡選，每一區各自最佳化。")
        assert watched.page_errors == [] and watched.console_errors == []


def test_the_page_says_optimized_by_the_program_everywhere(tmp_path: Path, browser: Browser) -> None:
    """全頁用「由程式最佳化」：圖例、小框的選項、其他牆面的下拉、清單卡、底列、輸入頁進這一頁的那一行，都沒有舊的那一句，
    這一頁講准用清單時也不說「電腦」。"""
    _prepare(tmp_path, SAMPLE_DESIGN)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, SAMPLE)
        # #1265 單元 3 起進這一頁那一行照示意圖改成「到「這間房貼什麼」，六面一起看…」，不再講做法；輸入頁講做法的地方
        # 是第 2 段的圖例與右欄清單的組名。
        # 圖例選了一面牆才露出來：先點前牆、再看得見的圖例（審查：以前先量，那時圖例還藏著）。
        page.locator("#elevation-wall-yL").click()
        expect(page.locator("#elevation-legend")).to_be_visible()
        expect(page.locator("#elevation-legend")).to_contain_text("由程式最佳化")
        expect(page.locator("#wall-zones-search .tag").first).to_have_text("由程式最佳化")
        assert OLD_WORDS not in page.locator("body").inner_text()
        page.locator("#zone-page-open").click()
        open_groups(page)
        expect(page.locator("#zone-map")).to_be_visible()
        expect(page.locator("#zone-page .zv-legend")).to_contain_text("由程式最佳化")
        expect(page.locator("#zone-rule-mode option[value=search]")).to_have_text("由程式最佳化")
        expect(page.locator("#zone-result-search")).to_have_attribute("data-count", "15")
        expect(page.locator("#zone-result-search")).to_contain_text("由程式最佳化 15 區")
        _click_zone(page, "yL/z1")
        expect(page.locator("#zone-panel-option-search .zv-opt-words")).to_contain_text("由程式最佳化")
        _pick_search(page)
        for part in ("#zone-page", "#elevation-panel"):  # 舊做法表（#zone-table-fold）拆了，改看它原本在的第 2 段整段
            words = cast(str, page.locator(part).evaluate("element => element.textContent"))
            assert OLD_WORDS not in words and "電腦" not in words, part
        assert watched.page_errors == [] and watched.console_errors == []


def test_nothing_new_on_the_panel_is_smaller_than_thirteen_pixels_or_too_pale_to_read(tmp_path: Path, browser: Browser) -> None:
    """小框（選了由程式最佳化、固定成品點開、有灰掉的列）與清單卡裡每一段字：不小於 13 px，字色對底色的對比至少 4.5:1。"""
    _prepare(tmp_path, SAMPLE_DESIGN)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        open_groups(page)
        # 地圖看得到就開量，雲端慢的時候清單卡還沒排好版，量到零段字（PR #1252 第一跑）；等量得到字再量。
        expect(page.locator("#zone-result-search")).to_have_attribute("data-count", "15")
        page.wait_for_function(f"() => ({_TEXT_STYLES})(document.getElementById('zone-group')).length > 5")
        card = cast(list[dict[str, object]], page.locator("#zone-group").evaluate(_TEXT_STYLES))
        page.locator("#zone-whole-yL").click()
        _pick_search(page)
        page.locator("#zone-panel-picker-search").focus()
        detail = page.locator("#zone-panel-members-glass_wool_48k")
        detail.locator("summary").click()
        expect(detail.locator("[data-member]").first).to_be_visible()
        panel = cast(list[dict[str, object]], page.locator("#zone-panel").evaluate(_TEXT_STYLES))
        assert len(panel) > 30 and len(card) > 5
        assert any("含 4 款" in cast(str, row["words"]) for row in panel)
        depth = page.locator("#zone-panel-depth")
        expect(depth).to_have_attribute("placeholder", "空白＝不限")
        assert page.locator("#zone-panel input[type=number][placeholder]:not(#zone-panel-depth)").count() == 0
        assert depth.evaluate("e => getComputedStyle(e, '::placeholder').color") == "rgb(160, 167, 173)"  # 色票 --soft（#1360 深底）
        expect(page.get_by_role("searchbox", name="找材料")).to_be_visible()
        small = [row for row in [*panel, *card] if cast(float, row["size"]) < 13]
        pale = [row for row in [*panel, *card] if cast(float, row["contrast"]) < 4.5]
        assert small == [] and pale == []
        assert watched.page_errors == [] and watched.console_errors == []


def test_the_white_frame_of_a_picked_zone_is_drawn_under_its_number(tmp_path: Path, browser: Browser) -> None:
    """選到的區加一圈白框：號碼畫在白框上面（窄條區的號碼圈比區還寬，白框後畫就把號碼蓋掉）。"""
    _prepare(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        page.locator("#zone-whole-xL").click()
        _expect_selected(page, RIGHT_WITHOUT_OPENINGS)
        order = cast(list[list[object]], page.evaluate("""() => [...document.querySelectorAll('#zone-map .zv-zone.selected')].map(zone => {
            const ring = zone.querySelector('.zv-ring'), badge = zone.querySelector('.zv-badge');
            return [zone.dataset.zone, Boolean(ring && badge && (ring.compareDocumentPosition(badge) & Node.DOCUMENT_POSITION_FOLLOWING))]; })"""))
        assert sorted(cast(str, zone) for zone, _ in order) == sorted(zone.split("/")[1] for zone in RIGHT_WITHOUT_OPENINGS)
        assert all(under for _, under in order), order
        assert watched.page_errors == [] and watched.console_errors == []


def test_picking_a_whole_face_changing_the_draft_then_cancelling_stores_both_files_byte_identical(tmp_path: Path, browser: Browser) -> None:
    """同一份方案，打開、選整面、在小框裡改了勾與最深、取消、存回：設計檔與方案檔逐位相同（小框裡改的只是草稿，套用才寫）。"""
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
            page.locator("#zone-whole-yL").click()
            _pick_search(page)
            row_options(page, "glass_wool_24k")
            page.locator("#zone-panel-remove-glass_wool_24k").click()
            row_options(page, "none")
            page.locator("#zone-panel-remove-none").click()
            page.locator("#zone-panel-depth").fill("12")
            page.locator("#zone-panel-depth").press("Tab")
            page.locator("#zone-panel-cancel").click()
            expect(page.locator("#zone-panel")).to_have_count(0)
            page.locator("#zone-whole-xL").click()
            page.locator("#zone-panel-cancel").click()
            page.locator("#zone-page-close").click()
            _save(page, design=design is not None)
            assert watched.page_errors == [] and watched.console_errors == []
        assert scheme_file.read_bytes() == saved[0]
        assert (design_file.read_bytes() if design else None) == saved[1]
        assert design_file.exists() == (design is not None)


def _design(zones: dict[str, dict[str, object]]) -> dict[str, object]:
    return {"schema_version": "aosr.zone_design.v1", "scheme_id": SAMPLE, "zones": zones}


def test_ticking_a_product_on_the_floor_leaves_every_other_zone_with_its_own_list(tmp_path: Path, browser: Browser) -> None:
    """小框只改選中的區（審查 r1 第一件）：地板一區改由程式最佳化、勾地毯、套用——小框只列地板能用的，
    另外 15 區（四面牆、天花）的清單一個字都不變，地板那一區是「不貼」加地毯。"""
    _prepare(tmp_path, SAMPLE_DESIGN)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        before = cast(dict[str, list[str]], page.evaluate(_ALLOWED))
        assert len(before) == 15
        _click_zone(page, "floor/auto1")
        _pick_search(page)
        _add_candidate(page, "sample_carpet_11mm")
        told = page.locator("#zone-panel-shared").count()
        _apply(page)
        after = cast(dict[str, list[str]], page.evaluate(_ALLOWED))
        assert after == {**before, "floor/auto1": ["none", "sample_carpet_11mm"]}
        assert told == 0  # 小框不再有「另外 N 區也跟著改」那一句
        assert watched.page_errors == [] and watched.console_errors == []


def test_unticking_and_ticking_back_on_the_floor_changes_no_other_zone(tmp_path: Path, browser: Browser) -> None:
    """審查 r1 第一件的另一條路：地板一區 → 由程式最佳化 →「也可以不貼」取消再勾回（畫面前後一樣）→ 套用：別區都不變。"""
    _prepare(tmp_path, SAMPLE_DESIGN)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        before = cast(dict[str, list[str]], page.evaluate(_ALLOWED))
        _click_zone(page, "floor/auto1")
        _pick_search(page)
        row_options(page, "none")
        page.locator("#zone-panel-remove-none").click()
        expect(page.locator("#zone-panel-apply")).to_be_disabled()
        row_options(page, "none")
        page.locator("#zone-panel-addto-none").click()
        _apply(page)
        assert cast(dict[str, list[str]], page.evaluate(_ALLOWED)) == {**before, "floor/auto1": ["none"]}
        assert watched.page_errors == [] and watched.console_errors == []


def test_two_zones_with_different_deepest_keep_their_own_when_the_box_is_typed_in_and_cleared(tmp_path: Path, browser: Browser) -> None:
    """審查 r1 第二件：兩區上限不同（一區 12.5 cm、一區沒有），格子是空的、底下寫「空白＝各區照目前的」（不用「原本」）。
    打個 5 再清掉＝沒改：套用後兩區的上限都照舊。真的填了數字才兩區都寫。"""
    _prepare(tmp_path, _design({"yL": {"auto1": {"mode": "search", "allowed": ONLY_48K},
                                       "auto2": {"mode": "search", "allowed": ONLY_48K, "max_depth_m": 0.125}}}))
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        _click_zone(page, "yL/auto1")
        _click_zone(page, "yL/auto2")
        expect(page.locator("#zone-panel-mode-search")).to_be_checked()
        expect(page.locator("#zone-panel-depth")).to_have_value("")
        page.locator("#zone-panel-depth").fill("5")
        page.locator("#zone-panel-depth").fill("")
        _apply(page)
        zones = cast(dict[str, dict[str, object]], page.evaluate("() => zoneDesign.zones.yL"))
        assert zones == {"auto1": {"mode": "search", "allowed": ONLY_48K}, "auto2": {"mode": "search", "allowed": ONLY_48K, "max_depth_m": 0.125}}
        _click_zone(page, "yL/auto1")
        _click_zone(page, "yL/auto2")
        expect(page.locator("#zone-panel-depth")).to_have_attribute("placeholder", "空白＝各區照目前的")
        assert "原本" not in cast(str, page.locator("#zone-panel-deepest").text_content())
        # 不收的時候那一句講的空白也要是同一個意思（各區不同時空白不是「不限」）。
        page.locator("#zone-panel-depth").fill("-3")
        expect(page.locator("#zone-panel-problem")).to_contain_text("空白＝各區照目前的")
        assert "不限" not in cast(str, page.locator("#zone-panel-problem").text_content())
        page.locator("#zone-panel-depth").fill("5")
        _apply(page)
        zones = cast(dict[str, dict[str, object]], page.evaluate("() => zoneDesign.zones.yL"))
        assert zones == {key: {"mode": "search", "allowed": ONLY_48K, "max_depth_m": 0.05} for key in ("auto1", "auto2")}
        assert watched.page_errors == [] and watched.console_errors == []


def test_applying_the_panel_only_adds_and_removes_what_was_ticked_and_keeps_the_rest_of_each_zones_list(tmp_path: Path, browser: Browser) -> None:
    """入口一（套用）：使用者多勾了哪幾種、拿掉了哪幾種，加減到每一個選中區自己的清單上，不拿「照這一面濾過」的那份蓋回去。
    前牆一區的清單裡有一筆這一面用不了的材料（現行分組沒有「只有天花能用」的，拿只有地板能用的那一筆代）：
    在前牆的小框勾 24K、拿掉 48K，套用後那一筆原封留著；地板那一區的清單（也有那一筆）一個字都不變。"""
    on_wall = ["glass_wool_48k", FLOOR_ONLY, "none"]
    on_floor = [FLOOR_ONLY, "none"]
    _prepare(tmp_path, _design({"yL": {"auto1": {"mode": "search", "allowed": on_wall}}, "floor": {"auto1": {"mode": "search", "allowed": on_floor}}}))
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        assert page.evaluate("id => zoneLibrary.find(entry => entry.id === id).placement.surfaces", FLOOR_ONLY) == ["floor"]
        assert cast(dict[str, list[str]], page.evaluate(_ALLOWED)) == {"yL/auto1": on_wall, "floor/auto1": on_floor}
        _click_zone(page, "yL/auto1")
        expect(page.locator("#zone-panel-mode-search")).to_be_checked()
        _expect_candidate_states(page, {WOOLS[0]: "all", WOOLS[1]: "none"})
        _add_candidate(page, "glass_wool_24k")
        row_options(page, "glass_wool_48k")
        page.locator("#zone-panel-remove-glass_wool_48k").click()
        _apply(page)
        assert cast(dict[str, list[str]], page.evaluate(_ALLOWED)) == {"yL/auto1": [FLOOR_ONLY, "none", "glass_wool_24k"], "floor/auto1": on_floor}
        assert watched.page_errors == [] and watched.console_errors == []
