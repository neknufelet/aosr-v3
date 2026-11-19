"""真瀏覽器考「鋼板直接貼在棉上」的構造（#1208 第 1 支）在材料庫頁與分區頁看到什麼。

資料全是自編的：材料是 ``test_gui_vpr_reference`` 那兩筆（同樣的鋼板與棉，一筆背襯黏住、一筆沒有），房間是左右對稱考卷
那間假房間（5.0 × 4.0 × 2.7 m）。畫面上的字逐字跟伺服器那一份（``reference_only.words``）比；那一份的原文另由函式層考卷釘住。
曲線要雙精度，材料庫頁與面板曲線那幾題的伺服器用正式網頁入口起在獨立行程（``real_server``）。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
from playwright.sync_api import Browser, Page, Route, expect

from aosr.gui import reference_only
from tests.engine._construction_gui_server import real_server
from tests.engine._gui_cache import gui_startup_identity_memo as gui_startup_identity_memo
from tests.engine._zone_symmetry_room import FRONT_PAIR, NAME, SIDE, symmetry_document
from tests.engine.test_gui_browser import _open, _serve, browser as browser
from tests.engine._bonded_backing_cases import BONDED_VPR, MELAMINE_BONDED, STEEL_2_5MM
from tests.engine.test_gui_vpr_reference import ENTRIES, HIT, HIT_LABEL, PLAIN_ID, _ENTRY, _fixed, _write_user_library
from tests.engine.test_gui_zone_elevation_browser import _open_scheme, _write

COMPUTED_MS = 60_000  # 等伺服器算完曲線的上限（冷行程第一次算一種新構造要編譯）
WORDS = reference_only.words()
# 藍底該到哪裡：橫軸是對數軸，界線頻率落在圖寬的幾分之幾（從頁面那張圖自己的橫軸頭尾算，不用它的換算函式）。
_BAND = """() => {
  const over = document.querySelector('#curve-chart .u-over'), band = document.getElementById('curve-ref-band');
  const word = document.getElementById('curve-ref-band-word'), {min, max} = plot.scales.x;
  const box = element => { const rect = element.getBoundingClientRect(); return [rect.left, rect.top, rect.width, rect.height]; };
  return {over: box(over), band: box(band), word: box(word), min, max, fill: getComputedStyle(band).backgroundColor};
}"""


def _open_material(page: Page, material_id: str) -> None:
    page.locator(f'#user-list tr[data-id="{material_id}"] button.open').click()
    expect(page.locator("#curve-chart canvas")).to_have_count(1, timeout=COMPUTED_MS)


def test_the_material_page_shades_the_band_and_says_it_three_ways(tmp_path: Path, browser: Browser) -> None:
    """材料庫頁打開命中的那一筆：曲線圖界線以下淡藍底、頂端內側一行小字；圖例一行「藍底：」加那一句，「為什麼」點開才有原因；
    「種類」那一行後面接一句。原本的橘底外推照留。打開沒黏住的那一筆：這三樣一樣都沒有。"""
    import math

    _write_user_library(tmp_path)
    with real_server(tmp_path) as base, _open(browser, f"{base}/materials") as watched:
        page = watched.page
        _open_material(page, HIT)
        expect(page.locator("#curve-ref-band")).to_be_visible()
        found = cast(dict[str, object], page.evaluate(_BAND))
        over, band, word = (cast(list[float], found[key]) for key in ("over", "band", "word"))
        share = math.log(cast(float, WORDS["band_hz"]) / cast(float, found["min"])) / math.log(cast(float, found["max"]) / cast(float, found["min"]))
        assert abs(band[0] - over[0]) < 1 and abs(band[1] - over[1]) < 1 and abs(band[3] - over[3]) < 1, found
        assert abs(band[2] - over[2] * share) < 2 and 0.2 < share < 0.6, (found, share)
        assert found["fill"] == "rgba(60, 120, 210, 0.16)"
        expect(page.locator("#curve-ref-band-word")).to_have_text(cast(str, WORDS["band_words"]))
        assert abs(word[0] - band[0]) < 1 and abs(word[1] - band[1]) < 1 and word[2] <= band[2], found  # 小字在藍底頂端內側、不比藍底寬
        note = page.locator("#curve-key li#curve-ref-note")
        expect(note).to_be_visible()
        expect(note.locator(".ref-say")).to_have_text(cast(str, WORDS["sentence"]))
        assert note.inner_text().startswith("藍底：" + cast(str, WORDS["sentence"]))
        expect(note.locator("summary")).to_have_text("為什麼")
        assert note.locator("summary").evaluate("node => getComputedStyle(node, '::after').content") == '" ▸"'
        expect(note.locator("details p")).to_be_hidden()
        note.locator("summary").click()
        expect(note.locator("details p")).to_have_text(cast(str, WORDS["why_only"]))
        expect(note.locator("details p")).to_be_visible()
        expect(page.locator("#editor-kind")).to_have_text("種類：固定材料（整塊選或不選）" + cast(str, WORDS["kind_words"]))
        expect(page.locator('#curve-key li[data-kind="extrapolation"]')).to_be_visible()  # 泡棉的外推橘底照留
        # 按重畫：藍底、圖例、那一句都只有一份（不疊第二份）。
        page.locator("#curve-redraw").click()
        expect(page.locator("#curve-redraw")).to_be_enabled(timeout=COMPUTED_MS)
        for selector in ("#curve-chart .ref-band", "#curve-chart .ref-band-word", "#curve-key .ref-note", "#editor-kind .ref-kind"):
            expect(page.locator(selector)).to_have_count(1)
        # 換開同樣的鋼板與棉、背襯沒黏住的那一筆：沒有藍底、沒有那一行圖例、「種類」那一行照舊。
        _open_material(page, PLAIN_ID)
        expect(page.locator("#curve-basis")).to_be_visible(timeout=COMPUTED_MS)
        for selector in (".ref-band", ".ref-band-word", ".ref-note", ".ref-kind"):
            expect(page.locator(selector)).to_have_count(0)
        expect(page.locator("#editor-kind")).to_have_text("種類：固定材料（整塊選或不選）")
        assert (watched.console_errors, watched.page_errors) == ([], [])


# ---- 分區頁（「這間房貼什麼」那一頁與舊做法表） ----

WOOLS = ["glass_wool_48k", "glass_wool_24k", "glass_wool_96k", "none"]
LISTED = ["glass_wool_48k", HIT, "none"]  # 存檔裡本來就有命中那一筆的清單
CUTS = {"x0": FRONT_PAIR, "y0": SIDE, "yL": SIDE}
_ALLOWED = """() => Object.fromEntries(Object.entries(zoneDesign.zones).flatMap(([wall, zones]) => Object.entries(zones)
    .filter(([, setting]) => setting.mode === 'search').map(([key, setting]) => [`${wall}/${key}`, setting.allowed])))"""
_STATE = "box => box.indeterminate ? 'some' : box.checked ? 'all' : 'none'"


def _prepare(tmp_path: Path, lists: dict[str, list[str]] | None = None, document: dict[str, object] | None = None) -> None:
    """自編的假房間（5.0 × 4.0 × 2.7 m，前牆左右各一塊、左右牆各一塊）、自編的兩筆材料；``lists``：哪幾區由程式最佳化、清單各是什麼。"""
    _write(tmp_path, document or symmetry_document(CUTS))
    _write_user_library(tmp_path)
    if lists is not None:
        zones: dict[str, dict[str, object]] = {}
        for zone, allowed in lists.items():
            wall, key = zone.split("/")
            zones.setdefault(wall, {})[key] = {"mode": "search", "allowed": allowed}
        (tmp_path / "designs").mkdir(exist_ok=True)
        (tmp_path / "designs" / f"{NAME}.json").write_text(json.dumps(
            {"schema_version": "aosr.zone_design.v1", "scheme_id": NAME, "zones": zones}), encoding="utf-8")


def _settled(page: Page) -> None:
    page.wait_for_function("""() => { const page = document.getElementById('zone-page');
        return !page.inert && !inputBusy && !(Number(page.dataset.changing) > 0) && zoneLimitsShown === zoneLimitsGeneration; }""")


def _open_page(page: Page) -> None:
    _open_scheme(page, NAME)
    page.locator("#zone-page-open").click()
    expect(page.locator("#zone-map")).to_be_visible()
    _settled(page)


def _click_zone(page: Page, zone: str) -> None:
    wall, key = zone.split("/")
    page.locator(f"#zone-map [data-wall='{wall}'][data-zone='{key}'] .zv-fill").click()


def _apply(page: Page) -> None:
    page.locator("#zone-panel-apply").click()
    expect(page.locator("#zone-panel")).to_have_count(0)
    _settled(page)


def _allowed(page: Page) -> dict[str, list[str]]:
    return cast(dict[str, list[str]], page.evaluate(_ALLOWED))


def _refused(page: Page, zone: str) -> str:
    """開跑會拒收的那一句（伺服器的原文）：區的叫法照頁面自己的牆名。"""
    wall, key = zone.split("/")
    name = cast(str, page.evaluate("wall => wallNames[wall]", wall))
    return reference_only.plan_refused(f"{name}「{key}」", HIT_LABEL)


def _map_words(page: Page, zone: str) -> list[str]:
    wall, key = zone.split("/")
    return page.locator(f"#zone-map [data-wall='{wall}'][data-zone='{key}'] .zv-ref-map").all_text_contents()


def test_pasting_such_a_construction_shades_the_panel_curve_and_marks_the_zone_on_the_map(tmp_path: Path, browser: Browser) -> None:
    """分區頁指定貼命中的那一筆：面板曲線界線以下淡藍底（寬度照對數橫軸算）、圖下「藍底：」那一句、「為什麼」點開才有原因
    （含最後「指定貼一種照舊能選」那一句）；套用後面板關了，圖上那一區材料字下面兩行小字照樣在、滑過去的說明也有那一句。
    另一區貼同樣鋼板與棉、背襯沒黏住的那一筆：曲線沒有藍底與那一行，圖上沒有那兩行。照舊貼得上去、存得進方案。"""
    import math

    _prepare(tmp_path)
    with real_server(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_page(page)
        _click_zone(page, "x0/a")
        page.locator("#zone-panel-material").select_option(HIT)
        band = page.locator("#zone-panel .zv-curve-chart rect.zv-ref-band")
        expect(band).to_have_count(1, timeout=COMPUTED_MS)
        edge = cast(float, WORDS["band_hz"])
        assert float(cast(str, band.get_attribute("data-hz"))) == edge
        # 藍底從圖的左緣畫到界線：寬度跟橫軸刻度字的位置對（刻度 100 與 1000 之間，300 落在 log10(3) 那個比例）。
        seen = cast(dict[str, object], page.evaluate("""() => {
          const chart = document.querySelector('#zone-panel .zv-curve-chart'), band = chart.querySelector('rect.zv-ref-band');
          const tick = text => Number([...chart.querySelectorAll('text')].find(node => node.textContent === text).getAttribute('x'));
          return {right: Number(band.getAttribute('x')) + Number(band.getAttribute('width')), hundred: tick('100'), thousand: tick('1000'),
                  fill: getComputedStyle(band).fill}; }"""))
        share = math.log10(edge / 100.0)
        right, hundred, thousand = (cast(float, seen[key]) for key in ("right", "hundred", "thousand"))
        assert abs(right - (hundred + share * (thousand - hundred))) < 1.5, seen
        assert seen["fill"] == "rgba(60, 120, 210, 0.16)", seen
        note = page.locator("#zone-panel-ref-note")
        expect(note.locator(".ref-say")).to_have_text(cast(str, WORDS["sentence"]))
        assert note.inner_text().startswith("藍底：" + cast(str, WORDS["sentence"]))
        expect(note.locator("summary")).to_have_text("為什麼")
        assert note.locator("summary").evaluate("node => getComputedStyle(node, '::after').content") == '" ▸"'
        expect(note.locator("details p")).to_be_hidden()
        note.locator("summary").click()
        expect(note.locator("details p")).to_be_visible()
        expect(note.locator("details p")).to_have_text(cast(str, WORDS["why"]))
        assert cast(str, WORDS["why"]).endswith("「指定貼一種」照舊能選、能貼，「由程式最佳化」不會挑它。")
        _apply(page)
        # 面板關著：圖上那一區材料字下面兩行，滑過去的說明接那一句。
        assert _map_words(page, "x0/a") == WORDS["map_words"]
        for line in page.locator("#zone-map [data-wall='x0'][data-zone='a'] .zv-ref-map").all():
            expect(line).to_be_visible()
        title = page.locator("#zone-map [data-wall='x0'][data-zone='a'] title").first.text_content()
        assert title is not None and title.endswith("；" + cast(str, WORDS["sentence"])), title
        # 對照：同樣的鋼板與棉、背襯沒黏住。
        _click_zone(page, "x0/b")
        page.locator("#zone-panel-material").select_option(PLAIN_ID)
        expect(page.locator("#zone-panel .zv-curve-chart")).to_have_count(1, timeout=COMPUTED_MS)
        expect(page.locator("#zone-panel .zv-ref-band")).to_have_count(0)
        expect(page.locator("#zone-panel-ref-note")).to_have_count(0)
        _apply(page)
        assert _map_words(page, "x0/b") == [] and _map_words(page, "x0/a") == WORDS["map_words"]
        title = page.locator("#zone-map [data-wall='x0'][data-zone='b'] title").first.text_content()
        assert title is not None and cast(str, WORDS["sentence"]) not in title
        assert cast(dict[str, object], page.evaluate("zoneDesign.zones.x0")) == {"a": {"mode": "fixed"}, "b": {"mode": "fixed"}}
        assert (watched.console_errors, watched.page_errors) == ([], [])


def test_such_a_construction_is_greyed_out_of_the_panel_list_and_the_add_box_with_the_reason(tmp_path: Path, browser: Browser) -> None:
    """「由程式最佳化」的准用清單：命中的那一筆灰掉、勾不了、小字寫為什麼（面板的固定成品那一列、清單卡「加別種材料」那一格）；
    沒黏住的那一筆照常勾得到、加得進去。"""
    _prepare(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_page(page)
        _click_zone(page, "y0/s")
        page.locator("#zone-panel-mode-search").check()
        page.locator("#zone-panel-fixed summary").click()
        row = page.locator(f"#zone-panel-fixed label[data-choice='{HIT}']")
        expect(row).to_have_class("zv-mrow zv-covered")
        expect(row).to_have_attribute("data-reference", "off")
        expect(page.locator(f"#zone-panel-allow-{HIT}")).to_be_disabled()
        expect(page.locator(f"#zone-panel-allow-{HIT}")).not_to_be_checked()
        expect(row.locator(".zv-covered-why")).to_have_text(cast(str, WORDS["search_why"]))
        plain = page.locator(f"#zone-panel-fixed label[data-choice='{PLAIN_ID}']")
        expect(plain).to_have_class("zv-mrow")
        expect(plain.locator(".zv-covered-why")).to_have_count(0)
        page.locator(f"#zone-panel-allow-{PLAIN_ID}").check()
        expect(page.locator("#zone-panel-search-from")).to_have_text("程式從下面勾的 5 種裡選，每一區各自最佳化。")
        _apply(page)
        assert _allowed(page) == {"y0/s": [*WOOLS, PLAIN_ID]}
        # 清單卡「加別種材料」同一條規則。
        page.locator("#zone-list-add").click()
        add = page.locator(f"#zone-list-add-box label[data-choice='{HIT}']")
        expect(add).to_have_attribute("data-reference", "off")
        expect(page.locator(f"#zone-list-add-{HIT}")).to_be_disabled()
        expect(add.locator(".zv-covered-why")).to_have_text(cast(str, WORDS["search_why"]))
        assert _allowed(page) == {"y0/s": [*WOOLS, PLAIN_ID]}
        assert (watched.console_errors, watched.page_errors) == ([], [])


def test_a_stored_list_with_such_a_construction_shows_red_and_the_tick_can_only_be_taken_off(tmp_path: Path, browser: Browser) -> None:
    """存檔的清單裡本來就有命中的那一筆：清單卡那一列照樣勾著、灰字、寫「清單裡本來有它，程式不會挑」；卡上紅框與頁底紅字
    寫開跑會拒收的那一句（伺服器原文）、「拿去搜尋」旁邊寫清單有紅框要先處理、直接叫送出也不存不跳頁；面板那一列勾著、可取消，「從下面勾的 N 種」不算它；
    取消後那一格馬上勾不回來，套用後清單沒有它、紅字消失。"""
    _prepare(tmp_path, {"x0/a": LISTED})
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_page(page)
        refused = _refused(page, "x0/a")
        card = page.locator(f"#zone-page .zv-list label[data-choice='{HIT}']")
        expect(card).to_have_attribute("data-reference", "listed")
        expect(page.locator(f"#zone-list-check-{HIT}")).to_be_checked()
        expect(page.locator(f"#zone-list-check-{HIT}")).to_be_enabled()
        expect(card.locator(".zv-covered-why")).to_have_text(cast(str, WORDS["listed_why"]))
        expect(page.locator("#zone-list-reference")).to_have_text(refused)
        expect(page.locator("#zone-list-reference")).to_have_class("zv-problem")
        expect(page.locator("#zone-page-blocked")).to_have_text(refused)
        expect(page.locator("#zone-page-blocked")).to_have_class("limit-red")
        expect(page.locator("#zone-page-search-why")).to_have_text("清單有紅框要先處理")
        # 直接叫送出也照樣擋：不存、不跳頁，訊息就是那一句。
        page.evaluate("sendToSearch()")
        expect(page.locator("#zone-page-note")).to_have_text(refused)
        assert not page.url.endswith("/gradient") and "gradient?" not in page.url
        # 面板：勾著、灰字、可取消；N 不算它（48K 與不貼＝2 種）。
        _click_zone(page, "x0/a")
        expect(page.locator("#zone-panel-fixed summary")).to_contain_text("勾了 1 種")
        page.locator("#zone-panel-fixed summary").click()
        row = page.locator(f"#zone-panel-form label[data-choice='{HIT}']")
        box = page.locator(f"#zone-panel-allow-{HIT}")
        expect(row).to_have_attribute("data-reference", "listed")
        expect(row).to_have_class("zv-mrow zv-covered")
        expect(row.locator(".zv-covered-why")).to_have_text(cast(str, WORDS["listed_why"]))
        expect(box).to_be_checked()
        expect(box).to_be_enabled()
        expect(page.locator("#zone-panel-search-from")).to_have_text("程式從下面勾的 2 種裡選，每一區各自最佳化。")
        box.uncheck()
        box = page.locator(f"#zone-panel-allow-{HIT}")
        expect(box).not_to_be_checked()
        expect(box).to_be_disabled()
        expect(page.locator(f"#zone-panel-form label[data-choice='{HIT}']")).to_have_attribute("data-reference", "off")
        expect(page.locator(f"#zone-panel-form label[data-choice='{HIT}'] .zv-covered-why")).to_have_text(cast(str, WORDS["search_why"]))
        _apply(page)
        assert _allowed(page) == {"x0/a": ["glass_wool_48k", "none"]}
        expect(page.locator("#zone-page-blocked")).to_have_count(0)
        expect(page.locator("#zone-list-reference")).to_have_count(0)
        assert (watched.console_errors, watched.page_errors) == ([], [])


def test_taking_it_off_on_the_list_card_is_final_and_the_add_box_will_not_put_it_back(tmp_path: Path, browser: Browser) -> None:
    """清單卡上取消命中的那一筆：清單少了它；「加別種材料」那一格裡它灰著、勾不了；直接叫寫清單的那一支要它加回來也不動。"""
    _prepare(tmp_path, {"x0/a": LISTED})
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_page(page)
        page.locator(f"#zone-list-check-{HIT}").click()
        _settled(page)
        assert _allowed(page) == {"x0/a": ["glass_wool_48k", "none"]}
        expect(page.locator(f"#zone-list-check-{HIT}")).to_have_count(0)
        expect(page.locator("#zone-page-blocked")).to_have_count(0)
        page.locator("#zone-list-add").click()
        expect(page.locator(f"#zone-list-add-{HIT}")).to_be_disabled()
        # 那一格被人改成勾得到再點：寫清單的那一支自己也不收。
        page.locator(f"#zone-list-add-{HIT}").evaluate("box => { box.disabled = false; box.click(); }")
        _settled(page)
        assert _allowed(page) == {"x0/a": ["glass_wool_48k", "none"]}
        assert (watched.console_errors, watched.page_errors) == ([], [])


def test_a_half_ticked_one_can_only_be_taken_off_and_never_spreads_to_the_other_zone(tmp_path: Path, browser: Browser) -> None:
    """兩區一區的清單有命中那一筆、一區沒有：一起選，那一格是第三種狀態；不碰它就套用，各區照舊（沒有的那一區不會多出它）；
    點一下是拿掉（不是勾上），套用後兩區都沒有。"""
    _prepare(tmp_path, {"x0/a": LISTED, "x0/b": WOOLS})
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_page(page)
        _click_zone(page, "x0/a")
        _click_zone(page, "x0/b")
        box = page.locator(f"#zone-panel-allow-{HIT}")
        assert box.evaluate(_STATE) == "some"
        expect(box).to_be_enabled()
        page.locator("#zone-panel-allow-glass_wool_48k").uncheck()  # 動別的勾（兩區都有的 48K），命中那一格不碰
        _apply(page)
        assert _allowed(page) == {"x0/a": [HIT, "none"], "x0/b": ["glass_wool_24k", "glass_wool_96k", "none"]}
        _click_zone(page, "x0/a")
        _click_zone(page, "x0/b")
        box = page.locator(f"#zone-panel-allow-{HIT}")
        assert box.evaluate(_STATE) == "some"
        page.locator("#zone-panel-fixed summary").click()
        box.click()
        box = page.locator(f"#zone-panel-allow-{HIT}")
        assert box.evaluate(_STATE) == "none"
        expect(box).to_be_disabled()
        _apply(page)
        assert _allowed(page) == {"x0/a": ["none"], "x0/b": ["glass_wool_24k", "glass_wool_96k", "none"]}
        assert (watched.console_errors, watched.page_errors) == ([], [])


def test_picking_the_whole_face_never_copies_it_into_the_zones_that_did_not_have_it(tmp_path: Path, browser: Browser) -> None:
    """整面套用：前牆一區的清單有命中那一筆，其餘幾區還沒交給程式。「選整面」→ 由程式最佳化 → 套用：本來有的那一區照舊有，
    其餘每一區的清單都沒有它。"""
    _prepare(tmp_path, {"x0/a": LISTED})
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_page(page)
        page.locator("#zone-whole-x0").click()
        page.locator("#zone-panel-mode-search").check()
        expect(page.locator("#zone-panel-form")).to_be_visible()
        _apply(page)
        lists = _allowed(page)
        assert len(lists) > 2 and lists["x0/a"] == LISTED
        assert [zone for zone, allowed in lists.items() if HIT in allowed] == ["x0/a"], lists
        assert (watched.console_errors, watched.page_errors) == ([], [])


def test_handing_zones_to_the_program_never_carries_it_over_from_the_last_list_or_the_pasted_one(tmp_path: Path, browser: Browser) -> None:
    """切成由程式最佳化時預帶的清單不准帶它，三條來路：①「其他牆面」規則照畫面上那一份清單套到別區；②剛改過的那一份清單
    （上一次的組合）帶到下一區；③這一區現在就貼著它，改成由程式最佳化（舊做法表不給「只讓程式決定：這一筆貼或不貼」）。"""
    _prepare(tmp_path, {"x0/a": LISTED})
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_page(page)
        # ② 動一下有它的那一份清單（這一頁「上一次的組合」就是它），再把別區交給程式。
        page.locator("#zone-list-check-glass_wool_48k").click()
        _settled(page)
        assert _allowed(page) == {"x0/a": [HIT, "none"]}
        _click_zone(page, "y0/s")
        page.locator("#zone-panel-mode-search").check()
        expect(page.locator(f"#zone-panel-allow-{HIT}")).not_to_be_checked()
        _apply(page)
        assert HIT not in _allowed(page)["y0/s"] and _allowed(page)["x0/a"] == [HIT, "none"]
        # ① 規則改成由程式最佳化：沒另外指定的區全部交給程式，一區都不帶它。
        page.locator("#zone-rule-mode").select_option("search")
        _settled(page)
        lists = _allowed(page)
        assert len(lists) > 3 and [zone for zone, allowed in lists.items() if HIT in allowed] == ["x0/a"], lists
        # 直接叫最底下那一支也一樣：新清單裡本來沒有的那一筆被拿掉，拿完空了只放不貼。
        page.evaluate(f"setZoneSetting('yL', 's', {{mode: 'search', allowed: ['{HIT}']}})")
        assert cast(list[str], page.evaluate("zoneDesign.zones.yL.s.allowed")) == ["none"]
        assert (watched.console_errors, watched.page_errors) == ([], [])

    # ③ 舊做法表：貼著它的區改成由程式最佳化。
    _prepare(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, NAME)
        page.locator("#elevation-wall-x0").click()
        page.locator("#zone-mode-x0-a").select_option("fixed")
        page.wait_for_function("!document.querySelector('main').inert")
        with page.expect_response("**/api/zone-material"):
            page.locator("#zone-fixed-x0-a").select_option(HIT)
        page.wait_for_function("!document.querySelector('main').inert")
        with page.expect_response("**/api/zone-material"):
            page.locator("#zone-mode-x0-a").select_option("search")
        page.wait_for_function("!document.querySelector('main').inert && !inputBusy && zoneLimitsShown === zoneLimitsGeneration")
        expect(page.locator("#zone-mode-x0-a")).to_have_value("search")
        assert cast(list[str], page.evaluate("zoneDesign.zones.x0.a.allowed")) == WOOLS
        expect(page.locator("#zone-preset-x0-a-this_or_none")).to_have_count(0)
        # 勾選表：它勾不了、滑過去寫為什麼；沒黏住的那一筆勾得到。
        page.locator("#zone-allowed-summary-x0-a").click()
        expect(page.locator("#zone-allowed-x0-a")).to_be_visible()
        expect(page.locator(f"#zone-allow-x0-a-{HIT}")).to_be_disabled()
        expect(page.locator("label", has=page.locator(f"#zone-allow-x0-a-{HIT}"))).to_have_attribute("title", cast(str, WORDS["search_why"]))
        page.locator(f"#zone-allow-x0-a-{PLAIN_ID}").check()
        assert cast(list[str], page.evaluate("zoneDesign.zones.x0.a.allowed")) == [*WOOLS, PLAIN_ID]
        assert (watched.console_errors, watched.page_errors) == ([], [])


def test_the_old_table_lets_a_stored_one_be_unticked_and_not_ticked_back(tmp_path: Path, browser: Browser) -> None:
    """舊做法表的勾選表：存檔清單裡本來有它的那一格勾著、可以取消（滑過去寫「清單裡本來有它」那一句）；取消後勾不回來。"""
    _prepare(tmp_path, {"x0/a": LISTED})
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, NAME)
        page.locator("#elevation-wall-x0").click()
        page.locator("#zone-allowed-summary-x0-a").click()
        box = page.locator(f"#zone-allow-x0-a-{HIT}")
        expect(box).to_be_checked()
        expect(box).to_be_enabled()
        expect(page.locator("label", has=box)).to_have_attribute("title", cast(str, WORDS["listed_why"]))
        box.uncheck()
        box = page.locator(f"#zone-allow-x0-a-{HIT}")
        expect(box).not_to_be_checked()
        expect(box).to_be_disabled()
        assert cast(list[str], page.evaluate("zoneDesign.zones.x0.a.allowed")) == ["glass_wool_48k", "none"]
        assert (watched.console_errors, watched.page_errors) == ([], [])


# 把最底下那一道（``setZoneSetting`` 裡的那一段）拿掉，只留原本的寫入：考每一條路自己的那一道。
_NO_LAST_GUARD = """() => { setZoneSetting = (wall, key, setting) => {
    const design = designForEdit(); design.zones[wall] ??= {}; design.zones[wall][key] = setting; }; }"""


def test_each_way_of_writing_a_list_keeps_it_out_by_itself_without_the_last_guard(tmp_path: Path, browser: Browser) -> None:
    """最底下那一道之外，每一條路自己也擋（把最底下那一道換掉再試）：①舊做法表剛改過的清單（上一次的組合）帶到下一區；
    ②舊做法表勾選表那一格被人改成勾得到再點；③清單卡「加別種材料」那一格被人改成勾得到再點；④選整面套用時，
    還沒交給程式的區照畫面上那一份清單起頭、但不帶它。"""
    _prepare(tmp_path, {"x0/a": LISTED})
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, NAME)
        page.evaluate(_NO_LAST_GUARD)
        page.locator("#elevation-wall-x0").click()
        # ① 動有它的那一份清單，再把另一區改成由程式最佳化：預帶的清單與底下那一句都沒有它。
        page.locator("#zone-chip-remove-x0-a-glass_wool_48k").click()
        assert cast(list[str], page.evaluate("zoneDesign.zones.x0.a.allowed")) == [HIT, "none"]
        with page.expect_response("**/api/zone-material"):
            page.locator("#zone-mode-x0-b").select_option("search")
        page.wait_for_function("!document.querySelector('main').inert && !inputBusy && zoneLimitsShown === zoneLimitsGeneration")
        assert cast(list[str], page.evaluate("zoneDesign.zones.x0.b.allowed")) == WOOLS
        # ② 勾選表那一格硬改成勾得到再點：清單不變。
        page.locator("#zone-allowed-summary-x0-b").click()
        expect(page.locator("#zone-allowed-x0-b")).to_be_visible()
        page.locator(f"#zone-allow-x0-b-{HIT}").evaluate("box => { box.disabled = false; box.click(); }")
        expect(page.locator(f"#zone-allow-x0-b-{HIT}")).not_to_be_checked()
        assert cast(list[str], page.evaluate("zoneDesign.zones.x0.b.allowed")) == WOOLS
        # ③ 清單卡「加別種材料」那一格硬改成勾得到再點：清單不變。
        page.locator("#zone-page-open").click()
        expect(page.locator("#zone-map")).to_be_visible()
        _settled(page)
        before = _allowed(page)
        page.locator("#zone-list-add").click()
        expect(page.locator(f"#zone-list-add-{HIT}")).to_be_disabled()
        page.locator(f"#zone-list-add-{HIT}").evaluate("box => { box.disabled = false; box.click(); }")
        _settled(page)
        assert _allowed(page) == before and HIT not in before["x0/b"]
        # ④ 選整面套用：先把沒有它的那一區放回不貼，剩下有它的那一區（畫面上那一份清單就是它的）；整面交給程式套用，
        # 還沒交給程式的區拿到的起始清單照那一份、但不帶它。
        page.locator("#zone-list-add").click()
        _click_zone(page, "x0/b")
        page.locator("#zone-panel-mode-keep_original").check()
        _apply(page)
        page.locator("#zone-whole-x0").click()
        page.locator("#zone-panel-mode-search").check()
        expect(page.locator("#zone-panel-form")).to_be_visible()
        _apply(page)
        lists = _allowed(page)
        assert len(lists) > 2 and [zone for zone, allowed in lists.items() if HIT in allowed] == ["x0/a"], lists
        assert (watched.console_errors, watched.page_errors) == ([], [])


# ---- 修補第 1 輪 ----

KIND = "這類「鋼板直接貼在棉上」的構造"
_OLD_TABLE_SETTLED = "!document.querySelector('main').inert && !inputBusy && zoneLimitsShown === zoneLimitsGeneration"


def _wall_zones() -> list[str]:
    """四面牆每一區的代號（方案裡的順序）。"""
    scene = cast(dict[str, object], symmetry_document(CUTS)["scene"])
    zones = cast(dict[str, list[dict[str, object]]], scene["surface_zones_by_wall"])
    return [f"{wall}/{zone.get('id', 'whole_wall')}" for wall in ("x0", "xL", "y0", "yL") for zone in zones[wall]]


def _served_refusals(page: Page) -> list[dict[str, object]]:
    return cast(list[dict[str, object]], page.evaluate("zoneLimits.reference_only ?? []"))


def test_the_old_table_search_row_is_stopped_and_says_the_refusal_in_red(tmp_path: Path, browser: Browser) -> None:
    """舊做法表「拿去搜尋」那一列：清單裡有命中那一筆，按鈕停用、旁邊紅字就是開跑會拒收的那一句（伺服器原文）；
    在勾選表取消它，按鈕恢復、旁邊回到平常那一句。"""
    _prepare(tmp_path, {"x0/a": LISTED, "x0/b": WOOLS})
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, NAME)
        page.locator("#elevation-wall-x0").click()
        page.wait_for_function(_OLD_TABLE_SETTLED)
        expect(page.locator("#zone-to-search")).to_be_disabled()
        expect(page.locator("#zone-to-search-note")).to_have_class("limit-red")
        expect(page.locator("#zone-to-search-note")).to_have_text(_refused(page, "x0/a"))
        page.locator("#zone-allowed-summary-x0-a").click()
        page.locator(f"#zone-allow-x0-a-{HIT}").uncheck()
        page.wait_for_function(_OLD_TABLE_SETTLED)
        expect(page.locator("#zone-to-search")).to_be_enabled()
        expect(page.locator("#zone-to-search-note")).to_have_class("note")
        expect(page.locator("#zone-to-search-note")).to_have_text("會先照畫面儲存，再打開第三階段搜尋頁")
        assert (watched.console_errors, watched.page_errors) == ([], [])


def test_zones_sharing_a_list_are_named_in_one_sentence_on_the_card_and_at_the_bottom(tmp_path: Path, browser: Browser) -> None:
    """18 區共用一份有命中那一筆的清單：清單卡紅框與頁底紅字各只有一句（伺服器那一句，區名用「、」接了 18 個），不是照區數重複。
    兩張清單卡各有命中：每張卡寫自己那一句，頁底兩句照伺服器的順序。"""
    shared = [zone for zone in _wall_zones() if zone != "y0/s"]
    assert len(shared) == 18
    _prepare(tmp_path, dict.fromkeys(shared, LISTED))
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_page(page)
        served = _served_refusals(page)
        assert len(served) == 1 and len(cast(list[object], served[0]["zones"])) == 18
        text = cast(str, served[0]["text"])
        assert text.count(KIND) == 1 and text.count("、") == 17 and text.endswith(_refused(page, "x0/a").split("」", 1)[1])
        expect(page.locator("#zone-list-reference")).to_have_text(text)
        expect(page.locator("#zone-page-blocked")).to_have_text(text)
        assert (watched.console_errors, watched.page_errors) == ([], [])

    _prepare(tmp_path, {"x0/a": LISTED, "x0/b": [HIT, "none"]})
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_page(page)
        first, second = _refused(page, "x0/b"), _refused(page, "x0/a")  # 方案裡 b 排在 a 前面
        assert [row["text"] for row in _served_refusals(page)] == [first, second]
        cards = {cast(str, card.get_attribute("id")): card.text_content() for card in page.locator("#zone-page .zv-list .zv-problem[id$='-reference']").all()}
        assert sorted(cast(str, text) for text in cards.values()) == sorted([first, second]) and len(cards) == 2, cards
        expect(page.locator("#zone-page-blocked")).to_have_text(f"{first}；{second}")
        assert (watched.console_errors, watched.page_errors) == ([], [])


def test_the_other_walls_card_carries_the_reminder_when_the_rule_pastes_such_a_construction(tmp_path: Path, browser: Browser) -> None:
    """「其他牆面」指定貼命中那一筆：跟著規則的區圖上不寫名字，提醒寫在右欄那張卡的材料下面（伺服器給的一行），面板關著就看得到；
    換成沒黏住的那一筆，那一行就沒有。"""
    _prepare(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_page(page)
        page.locator("#zone-rule-mode").select_option("fixed")
        expect(page.locator("#zone-rule-reference")).to_have_count(0)  # 還沒選貼哪一種
        page.locator("#zone-rule-material").select_option(HIT)
        _settled(page)
        expect(page.locator("#zone-rule-material")).to_have_value(HIT)
        expect(page.locator("#zone-panel")).to_have_count(0)
        mark = page.locator("#zone-rule #zone-rule-reference")
        expect(mark).to_be_visible()
        expect(mark).to_have_text(cast(str, WORDS["map_line"]))
        assert WORDS["map_line"] == "".join(cast(list[str], WORDS["map_words"]))
        assert cast(float, mark.evaluate("node => parseFloat(getComputedStyle(node).fontSize)")) >= 12
        box, select = mark.bounding_box(), page.locator("#zone-rule-material").bounding_box()
        assert box is not None and select is not None and box["y"] >= select["y"] + select["height"]  # 在材料下拉的下面
        page.locator("#zone-rule-material").select_option(PLAIN_ID)
        _settled(page)
        expect(page.locator("#zone-rule-material")).to_have_value(PLAIN_ID)
        expect(page.locator("#zone-rule-reference")).to_have_count(0)
        assert (watched.console_errors, watched.page_errors) == ([], [])


def test_a_plain_product_pasted_on_an_original_wall_that_is_such_a_construction_keeps_the_two_lines(tmp_path: Path, browser: Browser) -> None:
    """原牆本身就是命中的構造（等於材料庫那一筆），上面貼沒黏住的一般成品：整份（貼的＋原牆）伺服器判命中，
    圖上那一區照樣有那兩行。原牆是一般牆面、貼同一筆一般成品的那一區沒有。"""
    document = symmetry_document(CUTS)
    zones = cast(dict[str, list[dict[str, object]]], cast(dict[str, object], document["scene"])["surface_zones_by_wall"])
    next(zone for zone in zones["x0"] if zone.get("id") == "a")["material"] = BONDED_VPR
    _prepare(tmp_path, document=document)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_page(page)
        _click_zone(page, "x0/a")
        _click_zone(page, "x0/b")
        page.locator("#zone-panel-material").select_option(PLAIN_ID)
        _apply(page)
        pasted = cast(dict[str, dict[str, object]], page.evaluate(
            "Object.fromEntries(scheme.scene.surface_zones_by_wall.x0.filter(zone => zone.id).map(zone => [zone.id, zone.material]))"))
        assert pasted["a"]["kind"] == pasted["b"]["kind"] == "mounted_on_wall" and pasted["a"]["wall"] == BONDED_VPR
        # 伺服器的判斷（同一支）：a 那一份命中、b 那一份不命中。
        from pydantic import TypeAdapter

        from aosr.materials.construction_models import SurfaceMaterial
        adapter: TypeAdapter[SurfaceMaterial] = TypeAdapter(SurfaceMaterial)
        assert [reference_only.reference_only(adapter.validate_python(pasted[zone])) for zone in ("a", "b")] == [True, False]
        assert page.locator("#zone-map [data-wall='x0'][data-zone='a'] .zv-fixed-label").text_content() == "自編對照板"
        assert _map_words(page, "x0/a") == WORDS["map_words"]
        assert page.locator("#zone-map [data-wall='x0'][data-zone='b'] .zv-fixed-label").text_content() == "自編對照板"
        assert _map_words(page, "x0/b") == []
        assert (watched.console_errors, watched.page_errors) == ([], [])


def test_back_to_the_other_walls_says_why_it_cannot_and_changes_nothing(tmp_path: Path, browser: Browser) -> None:
    """其他牆面是由程式最佳化、清單裡有命中那一筆；另外指定的一區按「改回跟其他牆面一樣」：那一筆帶不過來、併不回去，
    所以不改，訊息是伺服器那一句（不寫「改好了」）。其他牆面的清單沒有它時照常改得回去。"""
    shared = [zone for zone in _wall_zones() if zone != "y0/s"]
    _prepare(tmp_path, dict.fromkeys(shared, LISTED))
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_page(page)
        expect(page.locator("#zone-rule")).to_have_attribute("data-mode", "search")
        before = page.evaluate("JSON.stringify([zoneDesign.zones, scheme.scene.surface_zones_by_wall])")
        page.locator("#zone-special-reset").click()
        expect(page.locator("#zone-page-note")).to_have_text(
            f"其他牆面的清單裡有{HIT_LABEL}（不進程式最佳化）：先在「其他牆面」取消它，才能改回跟其他牆面一樣。")
        _settled(page)
        assert page.evaluate("JSON.stringify([zoneDesign.zones, scheme.scene.surface_zones_by_wall])") == before
        expect(page.locator("#zone-special-reset")).to_have_count(1)
        assert (watched.console_errors, watched.page_errors) == ([], [])

    _prepare(tmp_path, dict.fromkeys(shared, WOOLS))
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_page(page)
        page.locator("#zone-special-reset").click()
        _settled(page)
        assert _allowed(page)["y0/s"] == WOOLS
        expect(page.locator("#zone-special-reset")).to_have_count(0)
        assert (watched.console_errors, watched.page_errors) == ([], [])


def test_a_carried_list_left_with_only_nothing_falls_back_to_the_three_wools_on_the_page(tmp_path: Path, browser: Browser) -> None:
    """分區頁（來源：畫面上那一份清單）：那一份是「命中那一筆或不貼」，其他牆面改成由程式最佳化時，新的區拿到的不是
    「只有不貼」而是預設的三種玻璃棉＋不貼。來源本來就只有「不貼」的，照舊帶「只有不貼」。"""
    _prepare(tmp_path, {"x0/a": [HIT, "none"]})
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_page(page)
        page.locator("#zone-rule-mode").select_option("search")
        _settled(page)
        lists = _allowed(page)
        assert len(lists) > 3 and lists["x0/a"] == [HIT, "none"]
        assert {zone: allowed for zone, allowed in lists.items() if zone != "x0/a"} == dict.fromkeys([z for z in lists if z != "x0/a"], WOOLS)
        assert (watched.console_errors, watched.page_errors) == ([], [])

    _prepare(tmp_path, {"x0/a": ["none"]})
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_page(page)
        page.locator("#zone-rule-mode").select_option("search")
        _settled(page)
        lists = _allowed(page)
        assert len(lists) > 3 and set(map(tuple, lists.values())) == {("none",)}, lists
        assert (watched.console_errors, watched.page_errors) == ([], [])


def test_a_carried_list_left_with_only_nothing_falls_back_to_the_three_wools_in_the_old_table(tmp_path: Path, browser: Browser) -> None:
    """舊做法表（來源：這一頁上一次改過的組合）：剛改過的那一份剩「命中那一筆或不貼」，再把另一區改成由程式最佳化，
    那一區拿到三種玻璃棉＋不貼。剛改過的那一份沒有命中、只剩「不貼」的，照舊帶「只有不貼」。"""
    for start, left in ((LISTED, [HIT, "none"]), (["glass_wool_48k", "none"], ["none"])):
        _prepare(tmp_path, {"x0/a": start})
        with _serve(tmp_path) as base, _open(browser, base) as watched:
            page = watched.page
            _open_scheme(page, NAME)
            page.locator("#elevation-wall-x0").click()
            page.locator("#zone-chip-remove-x0-a-glass_wool_48k").click()
            assert cast(list[str], page.evaluate("zoneDesign.zones.x0.a.allowed")) == left
            with page.expect_response("**/api/zone-material"):
                page.locator("#zone-mode-x0-b").select_option("search")
            page.wait_for_function(_OLD_TABLE_SETTLED)
            assert cast(list[str], page.evaluate("zoneDesign.zones.x0.b.allowed")) == (WOOLS if HIT in left else ["none"]), left
            assert (watched.console_errors, watched.page_errors) == ([], [])


class _HeldLimits:
    """把限制檢查（``/api/zone-limits``）的回覆扣住，到 :meth:`release` 才放行。:meth:`wait` 等到攔截函式真的收到請求才回來
    （頁面那邊世代號先加、請求後到，只看世代號會搶先）；等不到就紅並講原因。"""

    def __init__(self, page: Page) -> None:
        self.page, self.routes, self.open = page, cast(list[Route], []), False
        page.route("**/api/zone-limits", self._take)

    def _take(self, route: Route) -> None:
        if self.open:
            route.continue_()
        else:
            self.routes.append(route)

    def wait(self, limit_ms: int = 15_000) -> None:
        waited = 0
        while not self.routes:
            assert waited < limit_ms, f"等了 {limit_ms} 毫秒，限制檢查的請求還沒送到攔截函式"
            self.page.wait_for_timeout(25)
            waited += 25

    def release(self) -> None:
        self.open = True
        for route in self.routes:
            route.continue_()


_CARD_SENTENCES = "#zone-page .zv-list .zv-problem[id$='-reference']"


def _no_old_sentence_anywhere(page: Page) -> None:
    """卡、頁底、「拿去搜尋」旁邊三處都沒有舊句。"""
    expect(page.locator(_CARD_SENTENCES)).to_have_count(0)
    expect(page.locator("#zone-page-blocked")).to_have_count(0)
    expect(page.locator("#zone-page-search-why")).to_have_count(0)
    expect(page.locator("#zone-page-search")).to_be_enabled()


def test_splitting_a_shared_list_hides_the_old_sentence_everywhere_until_the_new_check_comes_back(tmp_path: Path, browser: Browser) -> None:
    """兩區共用一份有命中的清單（一句點名兩區）。扣住限制檢查，給其中一區多勾一種玻璃棉並套用（兩區都還有那一筆，但不再是
    同一份清單）：新結果回來之前，卡、頁底、按鈕旁三處都沒有舊的那一句；回來之後兩張卡各寫自己那一句。"""
    _prepare(tmp_path, {"x0/a": LISTED, "x0/b": LISTED})
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_page(page)
        both = cast(str, _served_refusals(page)[0]["text"])
        expect(page.locator(_CARD_SENTENCES)).to_have_text([both])
        expect(page.locator("#zone-page-search-why")).to_have_text("清單有紅框要先處理")
        limits = _HeldLimits(page)
        _click_zone(page, "x0/a")
        page.locator("#zone-panel-allow-glass_wool_24k").check()
        page.locator("#zone-panel-apply").click()
        expect(page.locator("#zone-panel")).to_have_count(0)
        page.wait_for_function("zoneDesign.zones.x0.a.allowed.includes('glass_wool_24k')")
        limits.wait()
        assert [row["text"] for row in _served_refusals(page)] == [both] and HIT in _allowed(page)["x0/a"]  # 手上還是舊的那一份
        _no_old_sentence_anywhere(page)
        limits.release()
        _settled(page)
        first, second = _refused(page, "x0/b"), _refused(page, "x0/a")
        assert sorted(page.locator(_CARD_SENTENCES).all_text_contents()) == sorted([first, second])
        expect(page.locator("#zone-page-blocked")).to_have_text(f"{first}；{second}")
        expect(page.locator("#zone-page-search-why")).to_have_text("清單有紅框要先處理")
        assert (watched.console_errors, watched.page_errors) == ([], [])


def test_a_sentence_for_a_list_that_no_longer_has_it_is_hidden_until_the_new_check_comes_back(tmp_path: Path, browser: Browser) -> None:
    """只有一區的清單有命中那一筆。扣住限制檢查，取消它並套用（區還是同一區，清單沒有那一筆了）：新結果回來之前三處都沒有
    舊的那一句；回來之後也沒有。"""
    _prepare(tmp_path, {"x0/a": LISTED})
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_page(page)
        expect(page.locator(_CARD_SENTENCES)).to_have_text([_refused(page, "x0/a")])
        limits = _HeldLimits(page)
        _click_zone(page, "x0/a")
        page.locator("#zone-panel-fixed summary").click()
        page.locator(f"#zone-panel-allow-{HIT}").uncheck()
        page.locator("#zone-panel-apply").click()
        expect(page.locator("#zone-panel")).to_have_count(0)
        page.wait_for_function("zoneDesign.zones.x0.a.allowed.length === 2")
        limits.wait()
        assert len(_served_refusals(page)) == 1  # 手上還是舊的那一份
        _no_old_sentence_anywhere(page)
        limits.release()
        _settled(page)
        assert _served_refusals(page) == []
        _no_old_sentence_anywhere(page)
        assert (watched.console_errors, watched.page_errors) == ([], [])


def test_merging_into_a_shared_list_hides_the_old_sentences_everywhere_until_the_new_check_comes_back(tmp_path: Path, browser: Browser) -> None:
    """前牆兩區共用一份有命中的清單，左牆那一區本來就存著那一筆、清單少一種玻璃棉（兩句）。扣住限制檢查，給左牆那一區
    補勾那一種（三區變成同一份）：新結果回來之前三處都沒有舊的兩句；回來之後一張卡、一句點名三區。"""
    _prepare(tmp_path, {"x0/a": LISTED, "x0/b": LISTED, "y0/s": [HIT, "none"]})
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_page(page)
        old = [cast(str, row["text"]) for row in _served_refusals(page)]
        assert len(old) == 2
        assert sorted(page.locator(_CARD_SENTENCES).all_text_contents()) == sorted(old)
        limits = _HeldLimits(page)
        _click_zone(page, "y0/s")
        page.locator("#zone-panel-allow-glass_wool_48k").check()
        page.locator("#zone-panel-apply").click()
        expect(page.locator("#zone-panel")).to_have_count(0)
        page.wait_for_function("zoneDesign.zones.y0.s.allowed.includes('glass_wool_48k')")
        limits.wait()
        assert [row["text"] for row in _served_refusals(page)] == old and HIT in _allowed(page)["y0/s"]
        _no_old_sentence_anywhere(page)
        limits.release()
        _settled(page)
        served = _served_refusals(page)
        assert len(served) == 1 and sorted(f"{zone['wall']}/{zone['zone']}" for zone in cast(list[dict[str, str]], served[0]["zones"])) == ["x0/a", "x0/b", "y0/s"]
        expect(page.locator(_CARD_SENTENCES)).to_have_text([cast(str, served[0]["text"])])
        expect(page.locator("#zone-page-blocked")).to_have_text(cast(str, served[0]["text"]))
        assert (watched.console_errors, watched.page_errors) == ([], [])


def test_a_design_zone_the_scheme_does_not_have_keeps_the_red_box_on_the_card(tmp_path: Path, browser: Browser) -> None:
    """設計檔裡多一區方案沒有的（x0/ghost），跟前牆兩區共用一份有命中的清單：清單卡照樣有紅框，句子只點名方案裡那兩區，
    頁底同一句。"""
    _prepare(tmp_path, {"x0/a": LISTED, "x0/b": LISTED, "x0/ghost": LISTED})
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_page(page)
        name = cast(str, page.evaluate("wallNames.x0"))
        text = reference_only.plan_refused(f"{name}「b」、{name}「a」", HIT_LABEL)
        expect(page.locator(_CARD_SENTENCES)).to_have_text([text])
        expect(page.locator("#zone-page-blocked")).to_have_text(text)
        expect(page.locator("#zone-page-search-why")).to_have_text("清單有紅框要先處理")
        assert "ghost" not in text
        assert (watched.console_errors, watched.page_errors) == ([], [])


def _scheme_with_original(zones: list[str]) -> dict[str, object]:
    """假房間，指名的幾區原牆換成命中的構造（等於材料庫那一筆）。"""
    document = symmetry_document(CUTS)
    by_wall = cast(dict[str, list[dict[str, object]]], cast(dict[str, object], document["scene"])["surface_zones_by_wall"])
    for wall, entries in by_wall.items():
        for entry in entries:
            if f"{wall}/{entry.get('id', 'whole_wall')}" in zones:
                entry["material"] = BONDED_VPR
    return document


@pytest.mark.parametrize("zone", ["x0/a", "x0/b"])
def test_the_other_walls_card_carries_the_reminder_whichever_following_zone_is_such_a_construction(tmp_path: Path, browser: Browser, zone: str) -> None:
    """其他牆面指定貼一般成品，跟著它的區裡有一區原牆是命中的構造（圖上不寫名字、那兩行沒地方放）：不管那一區是規則裡的
    第一區還是不是（兩區各試一次；只看第一區的寫法會漏掉其中一個），卡上都有那一行。"""
    _prepare(tmp_path, document=_scheme_with_original([zone]))
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_page(page)
        page.locator("#zone-rule-mode").select_option("fixed")
        page.locator("#zone-rule-material").select_option(PLAIN_ID)
        _settled(page)
        expect(page.locator("#zone-rule #zone-rule-reference")).to_have_text(cast(str, WORDS["map_line"]))
        assert _map_words(page, zone) == []
        assert (watched.console_errors, watched.page_errors) == ([], [])


def test_the_other_walls_card_has_no_reminder_while_the_material_is_still_being_picked(tmp_path: Path, browser: Browser) -> None:
    """每一面牆每一區的原牆都是命中的構造；其他牆面下拉改成「指定貼一種」、還沒選貼哪一種：卡上沒有那一行。"""
    _prepare(tmp_path, document=_scheme_with_original(_wall_zones()))
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_page(page)
        page.locator("#zone-rule-mode").select_option("fixed")
        expect(page.locator("#zone-rule-material")).to_have_value("")
        expect(page.locator("#zone-rule-material option").first).to_have_text("選要貼哪一種")
        expect(page.locator("#zone-rule-reference")).to_have_count(0)
        assert (watched.console_errors, watched.page_errors) == ([], [])


def _no_library(page: Page) -> None:
    page.route("**/api/materials", lambda route: route.fulfill(status=500, json={"error": "暫時讀不到材料庫（考卷植的）"}))


def test_with_the_library_unread_no_list_is_written_for_other_zones_from_the_panel_or_the_rule(tmp_path: Path, browser: Browser) -> None:
    """材料庫清單那一次沒讀到（認不出哪一筆是這一類）：別區改成由程式最佳化（面板套用、其他牆面下拉），都不寫清單
    （照材料庫沒讀全的慣例），存著命中那一筆的還是只有原來那一區。"""
    _prepare(tmp_path, {"x0/a": LISTED})
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _no_library(page)
        _open_page(page)
        assert page.evaluate("zoneReference") is None
        _click_zone(page, "x0/b")
        page.locator("#zone-panel-mode-search").check()
        _apply(page)
        lists = cast(dict[str, list[str] | None], page.evaluate(_ALLOWED))
        assert lists == {"x0/a": LISTED, "x0/b": None}
        page.locator("#zone-rule-mode").select_option("search")
        _settled(page)
        lists = cast(dict[str, list[str] | None], page.evaluate(_ALLOWED))
        assert len(lists) == 19 and lists.pop("x0/a") == LISTED and set(map(str, lists.values())) == {"None"}


def test_with_the_library_unread_back_to_the_other_walls_writes_no_list(tmp_path: Path, browser: Browser) -> None:
    """同樣材料庫沒讀到：其他牆面的清單裡有命中那一筆，另外指定的那一區按「改回跟其他牆面一樣」，不寫清單、那一筆沒有被帶過去。"""
    shared = [zone for zone in _wall_zones() if zone != "y0/s"]
    _prepare(tmp_path, dict.fromkeys(shared, LISTED))
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _no_library(page)
        _open_page(page)
        page.locator("#zone-special-reset").click()
        _settled(page)
        lists = cast(dict[str, list[str] | None], page.evaluate(_ALLOWED))
        assert lists.pop("y0/s") is None and list(lists.values()) == [LISTED] * 18


def test_a_sentence_naming_a_zone_that_no_longer_lists_it_is_hidden_until_the_new_check_comes_back(tmp_path: Path, browser: Browser) -> None:
    """兩區共用一份有命中那一筆的清單（一句點名兩區）。把限制檢查的回覆扣住，在面板取消其中一區的那一筆並套用：
    新結果回來之前，舊的那一句（還寫著兩區）不留在畫面上——卡上、頁底都沒有；放行之後換成只點名剩下那一區的新句子。"""
    _prepare(tmp_path, {"x0/a": LISTED, "x0/b": LISTED})
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_page(page)
        both = cast(str, _served_refusals(page)[0]["text"])
        assert both.count("、") == 1
        expect(page.locator("#zone-page-blocked")).to_have_text(both)
        limits = _HeldLimits(page)
        _click_zone(page, "x0/a")
        page.locator("#zone-panel-fixed summary").click()
        page.locator(f"#zone-panel-allow-{HIT}").uncheck()
        page.locator("#zone-panel-apply").click()
        expect(page.locator("#zone-panel")).to_have_count(0)
        page.wait_for_function("zoneDesign.zones.x0.a.allowed.length === 2 && zoneLimitsShown !== zoneLimitsGeneration")
        limits.wait()
        assert _served_refusals(page)[0]["text"] == both  # 手上還是舊的那一份
        expect(page.locator("#zone-page-blocked")).to_have_count(0)
        expect(page.locator("#zone-page .zv-problem[id$='-reference']")).to_have_count(0)
        limits.release()
        _settled(page)
        expect(page.locator("#zone-page-blocked")).to_have_text(_refused(page, "x0/b"))
        assert (watched.console_errors, watched.page_errors) == ([], [])


# 命中、但曲線沒有橘底外推的一筆：同樣的鋼板，背襯換成沒登記可信範圍的三木經驗式。
NO_ORANGE = "own_miki_panel"
NO_ORANGE_ENTRIES = (*ENTRIES, _ENTRY.validate_python(_fixed(NO_ORANGE, "自編三木板", {
    "kind": "assembly", "backing": "rigid", "layers": [STEEL_2_5MM, {**MELAMINE_BONDED, "model": "miki"}]})))


def _band_at(page: Page, hz: float) -> None:
    """把曲線介面回覆裡的界線換掉（伺服器起在另一個行程，改不到它的設定檔）：界線是從設定檔讀的，會變。"""
    def changed(route: Route) -> None:
        response = route.fetch()
        body = response.json()
        if "reference_only" in body:
            body["reference_only"]["band_hz"] = hz
        route.fulfill(response=response, json=body)
    page.unroute("**/api/material-curve")
    page.route("**/api/material-curve", changed)


def test_the_material_page_keeps_the_legend_without_an_orange_span_and_the_band_inside_the_chart(tmp_path: Path, browser: Browser) -> None:
    """材料庫頁：①命中、但沒有橘底外推的一筆，圖例那一塊照樣出現（裡面只有藍底那一行）。②界線落在橫軸左邊外面（10 Hz）：
    藍底與小字都不畫；落在右邊外面（20000 Hz）：藍底只畫到圖的右緣、不超出去；圖例那一行都照樣在，頁面沒有錯。"""
    _write_user_library(tmp_path, NO_ORANGE_ENTRIES)
    with real_server(tmp_path) as base, _open(browser, f"{base}/materials") as watched:
        page = watched.page
        _open_material(page, NO_ORANGE)
        expect(page.locator("#curve-ref-band")).to_be_visible(timeout=COMPUTED_MS)
        expect(page.locator('#curve-key li[data-kind="extrapolation"]')).to_be_hidden()
        assert cast(int, page.evaluate("plot.series.length")) >= 2 and cast(list[object], page.evaluate(
            "[...document.querySelectorAll('#curve-key li')].filter(item => !item.hidden).map(item => item.id)")) == ["curve-ref-note"]
        expect(page.locator("#curve-key")).to_be_visible()
        expect(page.locator("#curve-key li#curve-ref-note .ref-say")).to_be_visible()
        expect(page.locator("#curve-key li#curve-ref-note .ref-say")).to_have_text(cast(str, WORDS["sentence"]))
        _band_at(page, 10.0)
        _open_material(page, HIT)
        expect(page.locator("#curve-ref-note")).to_be_visible(timeout=COMPUTED_MS)
        page.wait_for_function("document.getElementById('curve-ref-band')?.dataset.hz === '10'")
        expect(page.locator("#curve-ref-band")).to_be_hidden()
        expect(page.locator("#curve-ref-band-word")).to_be_hidden()
        _band_at(page, 20000.0)
        _open_material(page, NO_ORANGE)
        page.wait_for_function("document.getElementById('curve-ref-band')?.dataset.hz === '20000'", timeout=COMPUTED_MS)
        found = cast(dict[str, object], page.evaluate(_BAND))
        over, band = cast(list[float], found["over"]), cast(list[float], found["band"])
        assert cast(float, found["max"]) < 20000 and abs(band[0] - over[0]) < 1 and abs(band[2] - over[2]) < 1.5, found
        expect(page.locator("#curve-ref-note")).to_be_visible()
        assert (watched.console_errors, watched.page_errors) == ([], [])


def test_the_panel_curve_keeps_the_band_inside_the_chart_and_folds_the_reason_again_for_the_next_pick(tmp_path: Path, browser: Browser) -> None:
    """分區頁面板曲線：①「為什麼」點開之後取消、再選一區，摺疊是收著的（上一次開著不帶到下一次選的區）。
    ②界線在橫軸左邊外面（10 Hz）：不畫藍底，圖例那一行照樣在；在右邊外面（20000 Hz）：藍底只畫到圖的右緣。"""
    _prepare(tmp_path)
    with real_server(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_page(page)
        _click_zone(page, "x0/a")
        page.locator("#zone-panel-material").select_option(HIT)
        expect(page.locator("#zone-panel .zv-ref-band")).to_have_count(1, timeout=COMPUTED_MS)
        page.locator("#zone-panel-ref-note summary").click()
        expect(page.locator("#zone-panel-ref-note details p")).to_be_visible()
        page.locator("#zone-panel-cancel").click()
        expect(page.locator("#zone-panel")).to_have_count(0)
        _click_zone(page, "x0/b")
        page.locator("#zone-panel-material").select_option(HIT)
        expect(page.locator("#zone-panel-ref-note")).to_have_count(1, timeout=COMPUTED_MS)
        assert page.locator("#zone-panel-ref-why").evaluate("fold => fold.open") is False
        expect(page.locator("#zone-panel-ref-note details p")).to_be_hidden()
        assert (watched.console_errors, watched.page_errors) == ([], [])

    edge = """() => { const chart = document.querySelector('#zone-panel .zv-curve-chart'), band = chart.querySelector('rect.zv-ref-band');
        const grid = [...chart.querySelectorAll('line.zv-grid')].map(line => Number(line.getAttribute('x2')));
        return {band: band ? Number(band.getAttribute('x')) + Number(band.getAttribute('width')) : null, right: Math.max(...grid)}; }"""
    for hz, drawn in ((10.0, False), (20000.0, True)):
        _prepare(tmp_path)
        with real_server(tmp_path) as base, _open(browser, base) as watched:
            page = watched.page
            _band_at(page, hz)
            _open_page(page)
            _click_zone(page, "x0/a")
            page.locator("#zone-panel-material").select_option(HIT)
            expect(page.locator("#zone-panel-ref-note .ref-say")).to_have_text(cast(str, WORDS["sentence"]), timeout=COMPUTED_MS)
            seen = cast(dict[str, float | None], page.evaluate(edge))
            if drawn:
                assert seen["band"] is not None and abs(seen["band"] - cast(float, seen["right"])) < 0.01, seen
            else:
                assert seen["band"] is None, seen
            assert (watched.console_errors, watched.page_errors) == ([], [])
