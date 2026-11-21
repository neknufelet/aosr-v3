"""真瀏覽器考「這間房貼什麼」那一頁的材料指定（#1199 第一支，純改畫面）：選整面、小框標出「目前」、准用材料一列一列攤開
（可調的寫範圍，只是文字）、被可調材料包含的成品灰掉、固定成品收起來寫幾種勾了幾種、「也可以不貼」與「這一區最深」在材料框外、
全頁用「由程式最佳化」。

資料全是合成的：區名考卷那間假房間（``_zone_naming_room`` 的 zone-naming-sample：前牆 yL 五區沒有門窗、右牆 xL 有門有窗）
配它的假材料與合成材料庫；可調的三種玻璃棉與它們包含的成品是產品內建的那幾筆（包含關係從網頁讀到的材料庫取，不在考卷寫死）。

老闆那句話「前牆整面由程式最佳化，只准 48K 玻璃棉，也可以不貼」一題：每按一下都走同一支計數的小工具，數出來要在 5 下以內，
再比存檔。其餘每一項新行為各一題。"""
from __future__ import annotations

from pathlib import Path
from typing import cast

from playwright.sync_api import Browser, Locator, Page, expect

from tests.engine.test_material_construction import construction
from tests.engine._gui_cache import gui_startup_identity_memo as gui_startup_identity_memo
from tests.engine.test_gui_browser import _open, _serve, browser as browser
from tests.engine.test_gui_zone_elevation_browser import _open_scheme
from tests.engine.test_gui_zone_overview_browser import SAMPLE, SAMPLE_DESIGN, SHORT, WOOLS
from tests.engine.test_gui_zone_overview_tasks_browser import (
    _ALLOWED, _SELECTED, _apply, _click_zone, _materials_before, _open_overview, _prepare, _save, _send_to_search, _settled, _stored)

FRONT = ["yL/z1", "yL/z2", "yL/z3", "yL/auto1", "yL/auto2"]
RIGHT_WITHOUT_OPENINGS = [f"xL/auto{n}" for n in range(1, 7)]
ONLY_48K = ["glass_wool_48k", "none"]
FLOOR_ONLY = "sample_carpet_11mm"  # 假材料庫裡只有地板能用的那一筆（考卷裡再跟網頁讀到的核一次）
# 一筆一般的固定成品（內建）：沒被哪一種可調玻璃棉包含、也不是「鋼板直接貼在棉上」的構造（#1208 起那一類勾不進清單，另有考卷）。
PLAIN_PRODUCT = "slotted_wood_typ9_absorber40_gap98"
OLD_WORDS = "交給" + "電腦挑"  # 舊的那一句（拆開寫：改字時整個 repo 搜這一句要是 0 筆）
# 小框與清單卡裡每一段看得到的字（按鈕除外）：字級（px）、字色對它底下那一層底色的對比。
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
    const rows = [];
    for (const element of root.querySelectorAll('*')) {
        if (element.closest('button') || !element.getClientRects().length) continue;
        const words = [...element.childNodes].filter(node => node.nodeType === 3 && node.textContent.trim()).map(node => node.textContent.trim()).join('');
        if (!words) continue;
        const style = getComputedStyle(element);
        const [r, g, b, alpha = 1] = parse(style.color), under = behind(element);
        let seen = alpha;
        for (let at = element; at && at !== document.documentElement; at = at.parentElement) seen *= parseFloat(getComputedStyle(at).opacity);
        const ink = light([r, g, b].map((value, index) => value * seen + under[index] * (1 - seen))), paper = light(under);
        rows.push({words, size: parseFloat(style.fontSize), contrast: (Math.max(ink, paper) + 0.05) / (Math.min(ink, paper) + 0.05)});
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


def _pick_search(page: Page) -> None:
    page.locator("#zone-panel-mode-search").check()
    expect(page.locator("#zone-panel-form")).to_be_visible()


def test_the_front_wall_is_handed_to_the_program_with_only_48k_or_nothing_in_five_actions(tmp_path: Path, browser: Browser) -> None:
    """老闆那句話：「前牆整面由程式最佳化，只准 48K 玻璃棉，也可以不貼」。從打開這一頁算起按五下：
    前牆「選整面」→「由程式最佳化」→ 取消 24K → 取消 96K →「套用」。做完再存檔（不算在那句話裡）比設計檔與方案檔。"""
    _prepare(tmp_path)
    before = _materials_before()
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        actions = _Actions()
        actions.click(page.locator("#zone-whole-yL"), "前牆「選整面」")
        assert sorted(cast(list[str], page.evaluate(_SELECTED))) == sorted(FRONT)
        actions.click(page.locator("#zone-panel-mode-search"), "由程式最佳化")
        # 一開始勾的是三種玻璃棉，「也可以不貼」在材料框外面、也勾著。
        for wool in ("glass_wool_48k", "glass_wool_24k", "glass_wool_96k"):
            expect(page.locator(f"#zone-panel-allow-{wool}")).to_be_checked()
        expect(page.locator("#zone-panel-allow-none")).to_be_checked()
        actions.click(page.locator("#zone-panel-allow-glass_wool_24k"), "取消 24K 玻璃棉")
        actions.click(page.locator("#zone-panel-allow-glass_wool_96k"), "取消 96K 玻璃棉")
        expect(page.locator("#zone-panel-apply")).to_have_text("套用到選中的 5 區")
        actions.click(page.locator("#zone-panel-apply"), "套用")
        expect(page.locator("#zone-panel")).to_have_count(0)
        _settled(page)
        assert len(actions.done) <= 5, actions.done
        # 畫面上：前牆五區都是由程式最佳化的顏色，底列數得到五區。
        for zone in FRONT:
            wall, key = zone.split("/")
            assert page.locator(f"#zone-map [data-wall='{wall}'][data-zone='{key}']").get_attribute("data-shown") == "search"
        expect(page.locator("#zone-result-search")).to_have_attribute("data-count", "5")
        expect(page.locator("#zone-result-search")).to_contain_text("由程式最佳化 5 區")
        # 右欄把這五區寫成一行「前牆整面」，清單卡只剩 48K 玻璃棉與也可以不貼。
        expect(page.locator("#zone-special [data-whole='yL']")).to_have_text("前牆整面 12345")
        expect(page.locator("#zone-list-who")).to_have_text("現在管 5 區：前牆整面 12345")
        assert page.locator("#zone-list-checks label").evaluate_all("rows => rows.map(row => row.dataset.choice)") == ONLY_48K
        _send_to_search(page)
        assert watched.page_errors == [] and watched.console_errors == []
    zones, materials = _stored(tmp_path)
    assert zones == {"yL": {key: {"mode": "search", "allowed": ONLY_48K} for key in ("z1", "z2", "z3", "auto1", "auto2")}}
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
        assert sorted(cast(list[str], page.evaluate(_SELECTED))) == RIGHT_WITHOUT_OPENINGS
        expect(page.locator("#zone-panel h2")).to_have_text("已選 6 區")
        expect(page.locator("#zone-panel-face")).to_contain_text("右牆整面")
        assert page.locator("#zone-panel-face .zv-num").count() == 6
        # 一區一行的那幾列收在摺疊裡；點開拿掉一區，就不再是整面。
        expect(page.locator("#zone-panel-zones summary")).to_have_text("看每一區、拿掉幾區")
        expect(page.locator("#zone-panel-drop-xL-auto3")).to_be_hidden()
        page.locator("#zone-panel-zones summary").click()
        page.locator("#zone-panel-drop-xL-auto3").click()
        expect(page.locator("#zone-panel h2")).to_have_text("已選 5 區")
        expect(page.locator("#zone-panel-face")).to_have_count(0)
        expect(page.locator("#zone-panel-whole")).to_have_text("改選右牆整面（6 區）")
        page.locator("#zone-panel-whole").click()
        assert sorted(cast(list[str], page.evaluate(_SELECTED))) == RIGHT_WITHOUT_OPENINGS
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
        expect(page.locator("#zone-panel-now-fixed")).to_have_text("目前")
        expect(page.locator("#zone-panel-now-fixed")).to_have_attribute("data-zones", "yL/z1")
        expect(page.locator("#zone-panel-option-fixed .zv-now-what")).to_contain_text("160 mm")
        assert "原本" not in page.locator("#zone-panel").inner_text()
        # 再加選一區由程式最佳化的、一區不貼的：三項各掛一個「目前」，各寫是哪一區的號碼。
        _click_zone(page, "yL/z3")
        _click_zone(page, "x0/whole_wall")
        expect(page.locator("#zone-panel [id^='zone-panel-now-']")).to_have_count(3)
        for mode, zone in (("fixed", "yL/z1"), ("search", "yL/z3"), ("keep_original", "x0/whole_wall")):
            expect(page.locator(f"#zone-panel-now-{mode}")).to_have_attribute("data-zones", zone)
        number = page.locator("#zone-map [data-wall='yL'][data-zone='z3'] .zv-badge text").text_content()
        expect(page.locator("#zone-panel-option-search .zv-now-what")).to_have_text(f"前牆 {number}")
        expect(page.locator("#zone-panel-option-keep_original .zv-now-what")).to_have_text("左牆整面")
        # 點別的選項，「目前」不跟著跑（它講的是打開時的樣子）。
        page.locator("#zone-panel-mode-keep_original").check()
        expect(page.locator("#zone-panel-now-fixed")).to_have_attribute("data-zones", "yL/z1")
        expect(page.locator("#zone-panel-cancel-note")).to_have_text("取消＝不改")
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
        expect(page.locator("#zone-panel-form")).to_have_count(0)
        _pick_search(page)
        rows = page.locator("#zone-panel-form label[data-kind='optimizable']")
        assert rows.evaluate_all("rows => rows.map(row => row.dataset.choice)") == ["glass_wool_48k", "glass_wool_24k", "glass_wool_96k"]
        expect(rows.nth(0)).to_contain_text("48K 玻璃棉")
        expect(rows.nth(0).locator(".zv-range")).to_have_text("可調：厚 5～20 cm、空腔 0～25 cm，加起來最深 30 cm")
        expect(rows.nth(1).locator(".zv-range")).to_have_text("可調：厚 5～20 cm、空腔：沒有（直接貼牆）")
        expect(rows.nth(2).locator(".zv-range")).to_have_text("可調：厚 5～20 cm、空腔：沒有（直接貼牆）")
        # 框裡除了勾選格，能打字的只有範圍那幾格（每一種的棉厚、48K 的空腔）。
        cells = page.locator("#zone-panel-form input:not([type=checkbox]), #zone-panel-form select, #zone-panel-form textarea")
        assert cells.evaluate_all("cells => cells.map(cell => cell.id)") == [
            f"zone-panel-range-{choice}-{part}-{end}" for choice, parts in (
                ("glass_wool_48k", ("porous", "cavity")), ("glass_wool_24k", ("porous",)), ("glass_wool_96k", ("porous",)))
            for part in parts for end in ("low", "high")]
        expect(page.locator("#zone-panel-range-note")).to_contain_text("範圍從材料頁帶來，可以改（只改這一區，不改材料頁）")
        expect(page.locator("#zone-panel-materials")).to_have_attribute("href", "/materials")
        expect(page.locator("#zone-panel-search-from")).to_have_text("程式從下面勾的 4 種裡選，每一區各自最佳化。")
        # 材料框外：也可以不貼、這一區最深。
        for outside in ("#zone-panel-allow-none", "#zone-panel-depth"):
            expect(page.locator(outside)).to_be_visible()
            assert page.locator(f"#zone-panel-form {outside}").count() == 0
        expect(page.locator("#zone-panel-none")).to_contain_text("也可以不貼")
        expect(page.locator("#zone-panel-none")).to_contain_text("程式算出來不貼比較好的區，就不貼，保留原樣")
        expect(page.locator("#zone-panel-deepest")).to_contain_text("這一區最深")
        _click_zone(page, "yL/auto2")
        expect(page.locator("#zone-panel-deepest")).to_contain_text("這幾區最深")
        # 「材料的勾全拿掉」不動「也可以不貼」；一種都沒勾（連不貼也拿掉）就不能套用。
        page.locator("#zone-panel-clear").click()
        assert page.locator("#zone-panel-form input:checked").count() == 0
        expect(page.locator("#zone-panel-allow-none")).to_be_checked()
        expect(page.locator("#zone-panel-apply")).to_be_enabled()
        page.locator("#zone-panel-allow-none").uncheck()
        expect(page.locator("#zone-panel-apply")).to_be_disabled()
        expect(page.locator("#zone-panel-problem")).to_contain_text("至少勾一種")
        assert watched.page_errors == [] and watched.console_errors == []


def test_products_a_ticked_tunable_material_already_covers_are_greyed_out_with_the_reason(tmp_path: Path, browser: Browser) -> None:
    """可調的 48K 玻璃棉勾著時，它已經包含的那幾筆成品（材料庫那一筆的 preset_members）在固定成品裡灰掉、勾不了，
    旁邊寫為什麼、要怎麼才勾得到；取消 48K 就勾得到。24K、96K 各包含一筆，一樣。"""
    _prepare(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        covered = cast(dict[str, list[str]], page.evaluate(
            "() => Object.fromEntries(zoneLibrary.filter(entry => entry.kind === 'optimizable').map(entry => [entry.id, entry.preset_members]))"))
        assert len(covered["glass_wool_48k"]) == 4 and len(covered["glass_wool_24k"]) == 1 and len(covered["glass_wool_96k"]) == 1
        _click_zone(page, "yL/auto1")
        _pick_search(page)
        page.locator("#zone-panel-fixed summary").click()
        for family, name in (("glass_wool_48k", "48K 玻璃棉"), ("glass_wool_24k", "24K 玻璃棉"), ("glass_wool_96k", "96K 玻璃棉")):
            for product in covered[family]:
                row = page.locator(f"#zone-panel-fixed label[data-choice='{product}']")
                expect(row).to_have_attribute("data-covered", family)
                expect(page.locator(f"#zone-panel-allow-{product}")).to_be_disabled()
                expect(row.locator(".zv-covered-why")).to_have_text(f"上面可調的 {name} 已經包含它；只想用成品，先取消上面的 {name}")
        # 沒被包含的成品照常勾得到。
        free = PLAIN_PRODUCT
        expect(page.locator(f"#zone-panel-allow-{free}")).to_be_enabled()
        assert page.locator(f"#zone-panel-fixed label[data-choice='{free}']").get_attribute("data-covered") is None
        # 取消 48K：它那四筆勾得到了，24K、96K 的照舊灰著。
        page.locator("#zone-panel-allow-glass_wool_48k").uncheck()
        for product in covered["glass_wool_48k"]:
            expect(page.locator(f"#zone-panel-allow-{product}")).to_be_enabled()
            expect(page.locator(f"#zone-panel-fixed label[data-choice='{product}'] .zv-covered-why")).to_have_count(0)
        expect(page.locator(f"#zone-panel-allow-{covered['glass_wool_24k'][0]}")).to_be_disabled()
        page.locator(f"#zone-panel-allow-{covered['glass_wool_48k'][0]}").check()
        _apply(page)
        assert cast(dict[str, list[str]], page.evaluate(_ALLOWED))["yL/auto1"] == [
            "glass_wool_24k", "glass_wool_96k", "none", covered["glass_wool_48k"][0]]
        # 清單卡的「加別種材料」同一條規則：清單裡的 24K 已經包含的那一筆灰著、勾不了、寫為什麼；48K 不在清單裡，它那幾筆勾得到。
        page.locator("#zone-list-add").click()
        inside = covered["glass_wool_24k"][0]
        expect(page.locator(f"#zone-list-add-{inside}")).to_be_disabled()
        expect(page.locator(f"#zone-list-add-box label[data-choice='{inside}']")).to_have_attribute("data-covered", "glass_wool_24k")
        expect(page.locator(f"#zone-list-add-box label[data-choice='{inside}'] .zv-covered-why")).to_have_text(
            "上面可調的 24K 玻璃棉 已經包含它；只想用成品，先取消上面的 24K 玻璃棉")
        expect(page.locator(f"#zone-list-add-{covered['glass_wool_48k'][1]}")).to_be_enabled()
        assert watched.page_errors == [] and watched.console_errors == []


def test_fixed_products_stay_folded_and_the_line_says_how_many_there_are_and_how_many_are_ticked(tmp_path: Path, browser: Browser) -> None:
    """固定成品平常收著，那一行寫這一面用得到的有幾種、勾了幾種（照材料庫算）；點開勾一種，數字跟著變，重畫後照樣開著。"""
    _prepare(tmp_path)
    vpr = PLAIN_PRODUCT
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        count = cast(int, page.evaluate("() => zoneLibrary.filter(entry => entry.kind === 'fixed' && entry.placement.surfaces.includes('yL')).length"))
        assert count > 4
        _click_zone(page, "yL/auto1")
        _pick_search(page)
        summary = page.locator("#zone-panel-fixed summary")
        expect(summary).to_have_text(f"固定成品（整塊貼或不貼，不能調） {count} 種，勾了 0 種")
        assert page.locator("#zone-panel-fixed").evaluate("fold => fold.open") is False
        expect(page.locator(f"#zone-panel-allow-{vpr}")).to_be_hidden()
        summary.click()
        assert page.locator("#zone-panel-fixed label[data-kind='fixed']").count() == count
        page.locator(f"#zone-panel-allow-{vpr}").check()
        expect(summary).to_have_text(f"固定成品（整塊貼或不貼，不能調） {count} 種，勾了 1 種")
        assert page.locator("#zone-panel-fixed").evaluate("fold => fold.open") is True
        expect(page.locator("#zone-panel-search-from")).to_have_text("程式從下面勾的 5 種裡選，每一區各自最佳化。")
        assert watched.page_errors == [] and watched.console_errors == []


def test_the_page_says_optimized_by_the_program_everywhere(tmp_path: Path, browser: Browser) -> None:
    """全頁用「由程式最佳化」：圖例、小框的選項、其他牆面的下拉、清單卡、底列、輸入頁進這一頁的那一行，都沒有舊的那一句，
    這一頁講准用清單時也不說「電腦」。"""
    _prepare(tmp_path, SAMPLE_DESIGN)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, SAMPLE)
        expect(page.locator("#zone-methods-row")).to_contain_text("由程式最佳化")
        assert OLD_WORDS not in page.locator("body").inner_text()
        page.locator("#zone-page-open").click()
        expect(page.locator("#zone-map")).to_be_visible()
        expect(page.locator("#zone-page .zv-legend")).to_contain_text("由程式最佳化")
        expect(page.locator("#zone-rule-mode option[value=search]")).to_have_text("由程式最佳化")
        expect(page.locator("#zone-list h2")).to_contain_text("由程式最佳化的 15 區")
        expect(page.locator("#zone-result-search")).to_contain_text("由程式最佳化 15 區")
        _click_zone(page, "yL/z1")
        expect(page.locator("#zone-panel-option-search label")).to_contain_text("由程式最佳化")
        _pick_search(page)
        for part in ("#zone-page", "#zone-table-fold"):
            words = cast(str, page.locator(part).evaluate("element => element.textContent"))
            assert OLD_WORDS not in words and "電腦" not in words, part
        assert watched.page_errors == [] and watched.console_errors == []


def test_nothing_new_on_the_panel_is_smaller_than_twelve_pixels_or_too_pale_to_read(tmp_path: Path, browser: Browser) -> None:
    """小框（選了由程式最佳化、固定成品點開、有灰掉的列）與清單卡裡每一段字：不小於 12 px，字色對底色的對比至少 4.5:1。"""
    _prepare(tmp_path, SAMPLE_DESIGN)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        card = cast(list[dict[str, object]], page.locator("#zone-list").evaluate(_TEXT_STYLES))
        page.locator("#zone-whole-yL").click()
        _pick_search(page)
        page.locator("#zone-panel-fixed summary").click()
        expect(page.locator("#zone-panel-fixed .zv-covered-why").first).to_be_visible()
        panel = cast(list[dict[str, object]], page.locator("#zone-panel").evaluate(_TEXT_STYLES))
        assert len(panel) > 30 and len(card) > 5
        assert any("已經包含它" in cast(str, row["words"]) for row in panel)
        # 空格子裡的提示字是瀏覽器另外畫的灰字，上面那一支量不到：小框裡不准有，空白的意思寫成看得到的字。
        assert page.locator("#zone-panel input[placeholder]").count() == 0
        assert any("空白＝不限" in cast(str, row["words"]) for row in panel)
        small = [row for row in [*panel, *card] if cast(float, row["size"]) < 12]
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
            page.locator("#zone-panel-allow-glass_wool_24k").uncheck()
            page.locator("#zone-panel-allow-none").uncheck()
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
        page.locator("#zone-panel-fixed summary").click()
        page.locator("#zone-panel-allow-sample_carpet_11mm").check()
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
        page.locator("#zone-panel-allow-none").uncheck()
        expect(page.locator("#zone-panel-apply")).to_be_disabled()
        page.locator("#zone-panel-allow-none").check()
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
        expect(page.locator("#zone-panel-deepest small")).to_contain_text("空白＝各區照目前的")
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
        expect(page.locator("#zone-panel-allow-glass_wool_48k")).to_be_checked()
        expect(page.locator("#zone-panel-allow-glass_wool_24k")).not_to_be_checked()
        page.locator("#zone-panel-allow-glass_wool_24k").check()
        page.locator("#zone-panel-allow-glass_wool_48k").uncheck()
        _apply(page)
        assert cast(dict[str, list[str]], page.evaluate(_ALLOWED)) == {"yL/auto1": [FLOOR_ONLY, "none", "glass_wool_24k"], "floor/auto1": on_floor}
        assert watched.page_errors == [] and watched.console_errors == []


def test_a_tick_is_added_only_to_the_zones_whose_face_can_take_it(tmp_path: Path, browser: Browser) -> None:
    """加一種材料只加到貼得了的那一面，而且勾得上也拿得掉（複查 r2 第二件）：前牆一區（48K 加不貼）與地板一區（只有不貼）一起選。
    48K 地板貼不了，不算進「幾區有」：貼得了的那一區有，就是勾著，旁邊講有一區貼不了；點一下拿掉、再點一下勾回來，
    每一下都照使用者點的畫，不會卡在第三種狀態。拿掉後套用：前牆那一區沒有 48K，地板不變。
    右欄清單卡也一樣：兩區清單相同時是一張卡，「加別種材料」列這兩面各自貼得了的，加 24K 只加到前牆那一區，底下那一句照實講幾區。"""
    _prepare(tmp_path, _design({"yL": {"auto1": {"mode": "search", "allowed": ONLY_48K}}, "floor": {"auto1": {"mode": "search", "allowed": ["none"]}}}))
    state = "box => box.indeterminate ? 'some' : box.checked ? 'all' : 'none'"
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        _click_zone(page, "yL/auto1")
        _click_zone(page, "floor/auto1")
        wool = page.locator("#zone-panel-allow-glass_wool_48k")
        assert wool.evaluate(state) == "all"
        expect(page.locator("#zone-panel-form label[data-choice=glass_wool_48k]")).to_contain_text("選中的 2 區裡 1 區那一面貼不了這一種，不會加")
        seen = []
        for _ in range(3):
            wool.click()
            seen.append(wool.evaluate(state))
        assert seen == ["none", "all", "none"]
        _apply(page)
        assert cast(dict[str, list[str]], page.evaluate(_ALLOWED)) == {"yL/auto1": ["none"], "floor/auto1": ["none"]}
        # 清單卡：兩區清單相同（只有不貼），是一張卡。
        expect(page.locator("#zone-page .zv-list")).to_have_count(1)
        page.locator("#zone-list-add").click()
        offered = cast(list[str], page.locator("#zone-list-add-box label").evaluate_all("rows => rows.map(row => row.dataset.choice)"))
        assert "glass_wool_24k" in offered and FLOOR_ONLY in offered
        page.locator("#zone-list-add-glass_wool_24k").click()
        _settled(page)
        assert cast(dict[str, list[str]], page.evaluate(_ALLOWED)) == {"yL/auto1": ["none", "glass_wool_24k"], "floor/auto1": ["none"]}
        expect(page.locator("#zone-page .zv-list")).to_have_count(2)
        expect(page.locator("#zone-page-note")).to_have_text("1 區的清單改成 不貼、24K 玻璃棉；另外 1 區那一面貼不了這一種，沒加")
        # 那一組的區拆成兩張卡了：開著的「加別種材料」關掉，兩張都不開。
        expect(page.locator("#zone-page .zv-add")).to_have_count(0)
        assert watched.page_errors == [] and watched.console_errors == []


def test_a_half_ticked_material_follows_each_click_and_can_be_taken_off(tmp_path: Path, browser: Browser) -> None:
    """第三種狀態只給使用者還沒動過的勾：兩區一區有 24K、一區沒有，打開是第三種狀態；點一下勾上、再點一下拿掉，
    拿掉後套用，兩區都沒有 24K。"""
    _prepare(tmp_path, SAMPLE_DESIGN)
    state = "box => box.indeterminate ? 'some' : box.checked ? 'all' : 'none'"
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        before = cast(dict[str, list[str]], page.evaluate(_ALLOWED))
        _click_zone(page, "y0/auto1")
        _click_zone(page, "y0/auto2")
        wool = page.locator("#zone-panel-allow-glass_wool_24k")
        assert wool.evaluate(state) == "some"
        seen = []
        for _ in range(2):
            wool.click()
            seen.append(wool.evaluate(state))
        assert seen == ["all", "none"]
        expect(page.locator("#zone-panel-form label[data-choice=glass_wool_24k]")).not_to_contain_text("選中的")
        _apply(page)
        assert cast(dict[str, list[str]], page.evaluate(_ALLOWED)) == {**before, "y0/auto2": ["glass_wool_48k", "glass_wool_96k", "none"]}
        assert watched.page_errors == [] and watched.console_errors == []


def test_an_open_add_box_stays_with_the_zones_it_was_opened_for_and_closes_when_they_join_another_card(tmp_path: Path, browser: Browser) -> None:
    """複查 r2 第一件：「加別種材料」的框記的是哪幾區，不是第幾張卡。前牆五區 48K 加不貼、天花 z1 只有 48K、天花 auto1 三種棉加不貼：
    在天花 z1 那張（第二張）開框、加「不貼」，它跟前牆一樣了、併進第一張——框關掉，不准跑到天花 auto1 那張，每一區的清單除了天花 z1 都不變。
    同一組的區沒變時（在同一張卡上加了還是那幾區），框照樣開著。"""
    front: dict[str, object] = dict.fromkeys(("z1", "z2", "z3", "auto1", "auto2"), {"mode": "search", "allowed": ONLY_48K})
    _prepare(tmp_path, _design({"yL": front, "ceiling": {"z1": {"mode": "search", "allowed": ["glass_wool_48k"]}, "auto1": {"mode": "search", "allowed": WOOLS}}}))
    cards = """() => [...document.querySelectorAll('#zone-page .zv-list')].map(card => ({id: card.id, zones: card.dataset.zones.split(' ').sort().join(' '),
        open: Boolean(card.querySelector('.zv-add'))}))"""
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        before = cast(dict[str, list[str]], page.evaluate(_ALLOWED))
        assert page.locator("#zone-list-2").get_attribute("data-zones") == "ceiling/z1"
        page.locator("#zone-list-2-add").click()
        expect(page.locator("#zone-list-2-add-box")).to_be_visible()
        page.locator("#zone-list-2-add-none").click()
        _settled(page)
        assert cast(dict[str, list[str]], page.evaluate(_ALLOWED)) == {**before, "ceiling/z1": ONLY_48K}
        shown = cast(list[dict[str, object]], page.evaluate(cards))
        assert [card["zones"] for card in shown] == [" ".join(sorted([*FRONT, "ceiling/z1"])), "ceiling/auto1"]
        assert [card["open"] for card in shown] == [False, False]
        # 同一張卡上加一種、那幾區沒變：框還開著，再加一種也寫進同樣那幾區。
        page.locator("#zone-list-2-add").click()
        page.locator("#zone-list-2-check-glass_wool_96k").click()
        _settled(page)
        expect(page.locator("#zone-list-2-add-box")).to_be_visible()
        page.locator("#zone-list-2-add-glass_wool_96k").click()
        _settled(page)
        expect(page.locator("#zone-list-2-add-box")).to_be_visible()
        assert cast(dict[str, list[str]], page.evaluate(_ALLOWED)) == {**before, "ceiling/z1": ONLY_48K, "ceiling/auto1": ["glass_wool_48k", "glass_wool_24k", "none", "glass_wool_96k"]}
        assert watched.page_errors == [] and watched.console_errors == []


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
        page.locator("#zone-panel-allow-glass_wool_48k").uncheck()
        expect(page.locator("#zone-panel-apply")).to_be_disabled()
        number = int(cast(str, page.locator("#zone-panel .zv-chip[data-zone='yL/auto1'] .zv-num").text_content()))
        expect(page.locator("#zone-panel-problem")).to_contain_text(f"前牆 {'①②③④⑤'[number - 1]} ")
        expect(page.locator("#zone-panel-problem")).to_contain_text("至少勾一種材料")
        page.locator("#zone-panel-allow-none").check()
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
        page.locator("#zone-panel-allow-glass_wool_24k").uncheck()
        page.locator("#zone-panel-allow-glass_wool_96k").uncheck()
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
    state = "box => box.indeterminate ? 'some' : box.checked ? 'all' : 'none'"
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        before = cast(dict[str, list[str]], page.evaluate(_ALLOWED))
        assert before["y0/auto1"] == SHORT and before["y0/auto2"] == WOOLS
        # 只選清單比較短的那一區：它沒有的兩種是空的。
        _click_zone(page, "y0/auto1")
        expect(page.locator("#zone-panel-mode-search")).to_be_checked()
        assert [page.locator(f"#zone-panel-allow-{choice}").evaluate(state) for choice in WOOLS] == ["all", "none", "none", "all"]
        # 這一區的清單跟「其他牆面」那一份不一樣：不標「＝其他牆面」；把缺的兩種勾上、跟它一樣了才標。
        expect(page.locator("#zone-panel-same")).to_have_count(0)
        page.locator("#zone-panel-allow-glass_wool_24k").check()
        page.locator("#zone-panel-allow-glass_wool_96k").check()
        expect(page.locator("#zone-panel-option-search #zone-panel-same")).to_have_text("＝其他牆面")
        page.locator("#zone-panel-cancel").click()
        _click_zone(page, "y0/auto1")
        # 再加選清單比較長的那一區。
        _click_zone(page, "y0/auto2")
        expect(page.locator("#zone-panel-mode-search")).to_be_checked()
        assert [page.locator(f"#zone-panel-allow-{choice}").evaluate(state) for choice in WOOLS] == ["all", "some", "some", "all"]
        expect(page.locator("#zone-panel-form label[data-choice=glass_wool_24k]")).to_contain_text("選中的 2 區裡 1 區有")
        expect(page.locator("#zone-panel-form label[data-choice=glass_wool_48k]")).not_to_contain_text("選中的")
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
        page.locator("#zone-panel-allow-glass_wool_24k").click()
        assert [page.locator(f"#zone-panel-allow-{choice}").evaluate(state) for choice in WOOLS] == ["all", "all", "some", "all"]
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
        assert page.locator("#zone-list-checks label").evaluate_all(choices) == ONLY_48K
        assert page.locator("#zone-list-2-checks label").evaluate_all(choices) == WOOLS
        # 前牆那張拿掉「也可以不貼」：只有前牆五區少一種，天花那一區的「不貼」還在；再加回去。
        page.locator("#zone-list-check-none").click()
        _settled(page)
        assert cast(dict[str, list[str]], page.evaluate(_ALLOWED)) == {**dict.fromkeys(FRONT, ["glass_wool_48k"]), "ceiling/z1": WOOLS}
        page.locator("#zone-list-add").click()
        page.locator("#zone-list-add-none").click()
        _settled(page)
        assert cast(dict[str, list[str]], page.evaluate(_ALLOWED)) == {**dict.fromkeys(FRONT, ONLY_48K), "ceiling/z1": WOOLS}
        # 前牆那張加 24K：只有前牆五區多一種（「加別種材料」那一格還開著）。
        page.locator("#zone-list-add-glass_wool_24k").click()
        _settled(page)
        longer = [*ONLY_48K, "glass_wool_24k"]
        assert cast(dict[str, list[str]], page.evaluate(_ALLOWED)) == {**dict.fromkeys(FRONT, longer), "ceiling/z1": WOOLS}
        # 天花那張拿掉 96K：只有天花那一區少一種；兩邊變成同樣三種，併回一張卡，這時才寫「都從這幾種裡選」。
        page.locator("#zone-list-2-check-glass_wool_96k").click()
        _settled(page)
        assert cast(dict[str, list[str]], page.evaluate(_ALLOWED)) == {**dict.fromkeys(FRONT, longer), "ceiling/z1": ["glass_wool_48k", "glass_wool_24k", "none"]}
        expect(page.locator("#zone-page .zv-list")).to_have_count(1)
        expect(page.locator("#zone-list h2")).to_have_text("由程式最佳化的 6 區每一區各自最佳化，都從這幾種裡選")
        # 併成一張之後，在那一張拿掉 24K：六區一起少一種。
        page.locator("#zone-list-check-glass_wool_24k").click()
        _settled(page)
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
    with _serve(tmp_path) as base, _open(browser, base) as watched:
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
            assert curve.value.json()["status"] != "incomplete"
        page.locator("#zone-panel-apply").click()
        expect(page.locator("#zone-panel")).to_have_count(0)
        _click_zone(page, "yL/z3")
        expect(page.locator("#zone-panel-now-fixed")).to_be_visible()
        expect(select).to_have_value("panel_air")
        _pick_search(page)
        page.locator("#zone-panel-fixed summary").click()
        for backing in ("rigid", "air"):
            choice = page.locator(f"#zone-panel-allow-panel_{backing}")
            expect(choice).to_be_enabled()
            expect(choice).not_to_be_checked()
            choice.check()
        expect(page.locator("#zone-panel-allow-panel_zone_wall")).to_have_count(0)
        assert watched.page_errors == []
