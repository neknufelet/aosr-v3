"""真瀏覽器考「鋼板直接貼在棉上」的構造（#1208 第 1 支）在材料庫頁與分區頁看到什麼。
資料使用自編對稱房與鋼板構造，文案逐字與伺服器比對，曲線由獨立行程以雙精度計算。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from playwright.sync_api import Browser, Page, Route, expect

from aosr.gui import reference_only
from tests.engine._zone_panel_steps import apply_and_return, choose_fixed, choose_mode, edit_candidates, expect_mode, open_panel
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
    page.locator(f'#panel-fixed [data-id="{material_id}"] button.open').click()
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


def _prepare(tmp_path: Path, lists: dict[str, list[str]] | None = None, document: dict[str, object] | None = None,
             *, left_right_symmetry: str | None = None) -> None:
    """自編的假房間（5.0 × 4.0 × 2.7 m，前牆左右各一塊、左右牆各一塊）、自編的兩筆材料；``lists``：哪幾區由程式最佳化、清單各是什麼。"""
    _write(tmp_path, document or symmetry_document(CUTS))
    _write_user_library(tmp_path)
    if lists is not None or left_right_symmetry is not None:
        zones: dict[str, dict[str, object]] = {}
        for zone, allowed in (lists or {}).items():
            wall, key = zone.split("/")
            zones.setdefault(wall, {})[key] = {"mode": "search", "allowed": allowed}
        (tmp_path / "designs").mkdir(exist_ok=True)
        (tmp_path / "designs" / f"{NAME}.json").write_text(json.dumps(
            {"schema_version": "aosr.zone_design.v1", "scheme_id": NAME, "zones": zones,
             **({"left_right_symmetry": left_right_symmetry} if left_right_symmetry else {})}), encoding="utf-8")


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

    _prepare(tmp_path, left_right_symmetry="off")  # 這一題不考左右對稱，明寫關著
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
    """「由程式最佳化」的准用清單：命中的那一筆灰掉、勾不了、小字寫為什麼（從地圖或清單卡選區後的小框）；
    沒黏住的那一筆照常勾得到、加得進去。"""
    _prepare(tmp_path, left_right_symmetry="off")  # 這一題不考左右對稱，明寫關著
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_page(page)
        _click_zone(page, "y0/s")
        page.locator("#zone-panel-mode-search").check()
        page.locator("#zone-panel-add").click()
        row = page.locator(f"#zone-panel-pick-{HIT}")
        expect(row.locator(".zv-covered-why")).to_be_visible()
        expect(row).to_have_attribute("data-reference", "off")
        expect(row).to_be_disabled()
        expect(page.locator(f"#zone-panel-row-{HIT}")).to_have_count(0)
        expect(row.locator(".zv-covered-why")).to_have_text(cast(str, WORDS["search_why"]))
        plain = page.locator(f"#zone-panel-pick-{PLAIN_ID}")
        expect(plain).to_be_enabled()
        expect(plain.locator(".zv-covered-why")).to_have_count(0)
        plain.click()
        expect(page.locator("#zone-panel-search-from")).to_have_text("程式從下面勾的 5 種裡選，每一區各自最佳化。")
        _apply(page)
        assert _allowed(page) == {"y0/s": [*WOOLS, PLAIN_ID]}
        # 從清單卡選區後，小框也照同一條規則。
        page.locator("#zone-list-select").click()
        page.locator("#zone-panel-add").click()
        add = page.locator(f"#zone-panel-pick-{HIT}")
        expect(add).to_have_attribute("data-reference", "off")
        expect(page.locator(f"#zone-panel-pick-{HIT}")).to_be_disabled()
        expect(add.locator(".zv-covered-why")).to_have_text(cast(str, WORDS["search_why"]))
        assert _allowed(page) == {"y0/s": [*WOOLS, PLAIN_ID]}
        assert (watched.console_errors, watched.page_errors) == ([], [])


def test_a_stored_list_with_such_a_construction_shows_red_and_the_tick_can_only_be_taken_off(tmp_path: Path, browser: Browser) -> None:
    """存檔的清單裡本來就有命中的那一筆：清單卡那一列照樣寫名字、灰字、寫「清單裡本來有它，程式不會挑」；卡上紅框與頁底紅字
    寫開跑會拒收的那一句（伺服器原文）、「拿去搜尋」旁邊寫清單有紅框要先處理、直接叫送出也不存不跳頁；面板那一列勾著、可取消，「從下面勾的 N 種」不算它；
    取消後那一格馬上勾不回來，套用後清單沒有它、紅字消失。"""
    _prepare(tmp_path, {"x0/a": LISTED}, left_right_symmetry="off")  # 這一題不考左右對稱，明寫關著
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_page(page)
        refused = _refused(page, "x0/a")
        card = page.locator(f"#zone-page .zv-list [data-choice='{HIT}']")
        expect(card).to_have_attribute("data-reference", "listed")
        expect(card.locator("input,button")).to_have_count(0)
        expect(card).to_contain_text(HIT_LABEL)
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
        page.locator("#zone-list-select").click()
        expect(page.locator("#zone-list-reference")).to_have_text(refused)
        expect(page.locator("#zone-list")).to_contain_text(cast(str, WORDS["listed_why"]))
        expect(page.locator("#zone-panel-list [data-kind=fixed]")).to_have_count(1)
        row = page.locator(f"#zone-panel-row-{HIT}")
        box = page.locator(f"#zone-panel-remove-{HIT}")
        expect(row).to_have_attribute("data-reference", "listed")
        expect(row.locator(".zv-covered-why")).to_be_visible()
        expect(row.locator(".zv-covered-why")).to_have_text(cast(str, WORDS["listed_why"]))
        expect(row).to_have_attribute("data-state", "all")
        expect(box).to_be_enabled()
        expect(page.locator("#zone-panel-search-from")).to_have_text("程式從下面勾的 2 種裡選，每一區各自最佳化。")
        box.click()
        expect(row).to_have_count(0)
        page.locator("#zone-panel-add").click()
        box = page.locator(f"#zone-panel-pick-{HIT}")
        expect(box).to_be_disabled()
        expect(page.locator(f"#zone-panel-pick-{HIT}")).to_have_attribute("data-reference", "off")
        expect(page.locator(f"#zone-panel-pick-{HIT} .zv-covered-why")).to_have_text(cast(str, WORDS["search_why"]))
        _apply(page)
        assert _allowed(page) == {"x0/a": ["glass_wool_48k", "none"]}
        expect(page.locator("#zone-page-blocked")).to_have_count(0)
        expect(page.locator("#zone-list-reference")).to_have_count(0)
        assert (watched.console_errors, watched.page_errors) == ([], [])


def test_taking_it_off_on_the_list_card_is_final_and_the_add_box_will_not_put_it_back(tmp_path: Path, browser: Browser) -> None:
    """從清單卡選區，拿掉命中那一筆並套用，再開小框也加不回來。"""
    _prepare(tmp_path, {"x0/a": LISTED}, left_right_symmetry="off")  # 這一題不考左右對稱，明寫關著
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_page(page)
        page.locator("#zone-list-select").click()
        page.locator(f"#zone-panel-remove-{HIT}").click()
        _apply(page)
        assert _allowed(page) == {"x0/a": ["glass_wool_48k", "none"]}
        expect(page.locator(f"#zone-list [data-choice={HIT}]")).to_have_count(0)
        expect(page.locator("#zone-page-blocked")).to_have_count(0)
        page.locator("#zone-list-select").click()
        page.locator("#zone-panel-add").click()
        expect(page.locator(f"#zone-panel-pick-{HIT}")).to_be_disabled()
        # 那一格被人改成勾得到再點：寫清單的那一支自己也不收。
        page.locator(f"#zone-panel-pick-{HIT}").evaluate("box => { box.disabled = false; box.click(); }")
        _apply(page)
        assert _allowed(page) == {"x0/a": ["glass_wool_48k", "none"]}
        assert (watched.console_errors, watched.page_errors) == ([], [])


def test_a_half_ticked_one_can_only_be_taken_off_and_never_spreads_to_the_other_zone(tmp_path: Path, browser: Browser) -> None:
    """兩區一區的清單有命中那一筆、一區沒有：一起選，那一格是第三種狀態；不碰它就套用，各區照舊（沒有的那一區不會多出它）；
    點一下是拿掉（不是勾上），套用後兩區都沒有。"""
    _prepare(tmp_path, {"x0/a": LISTED, "x0/b": WOOLS}, left_right_symmetry="off")  # 這一題不考左右對稱，明寫關著
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_page(page)
        _click_zone(page, "x0/a")
        _click_zone(page, "x0/b")
        box = page.locator(f"#zone-panel-remove-{HIT}")
        expect(page.locator(f"#zone-panel-row-{HIT}")).to_have_attribute("data-state", "some")
        expect(page.locator(f"#zone-panel-addto-{HIT}")).to_have_count(0)
        expect(box).to_be_enabled()
        page.locator("#zone-panel-remove-glass_wool_48k").click()  # 動別的勾（兩區都有的 48K），命中那一格不碰
        _apply(page)
        assert _allowed(page) == {"x0/a": [HIT, "none"], "x0/b": ["glass_wool_24k", "glass_wool_96k", "none"]}
        _click_zone(page, "x0/a")
        _click_zone(page, "x0/b")
        box = page.locator(f"#zone-panel-remove-{HIT}")
        expect(page.locator(f"#zone-panel-row-{HIT}")).to_have_attribute("data-state", "some")
        expect(page.locator(f"#zone-panel-addto-{HIT}")).to_have_count(0)
        box.click()
        expect(page.locator(f"#zone-panel-row-{HIT}")).to_have_count(0)
        page.locator("#zone-panel-add").click()
        expect(page.locator(f"#zone-panel-pick-{HIT}")).to_be_disabled()
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
    _prepare(tmp_path, {"x0/a": LISTED}, left_right_symmetry="off")  # 這一題不考左右對稱，明寫關著
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_page(page)
        # ② 動一下有它的那一份清單（這一頁「上一次的組合」就是它），再把別區交給程式。
        page.locator("#zone-list-select").click()
        page.locator("#zone-panel-remove-glass_wool_48k").click()
        _apply(page)
        assert _allowed(page) == {"x0/a": [HIT, "none"]}
        _click_zone(page, "y0/s")
        page.locator("#zone-panel-mode-search").check()
        expect(page.locator(f"#zone-panel-row-{HIT}")).to_have_count(0)
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

    # ③ 貼著它的區改成由程式最佳化：舊做法表那一路（/api/zone-material）跟著表拆了，改從輸入頁「改這一區 ▸」開小框走
    # （/api/zone-material/batch；跟 ① 從那一頁點卡選區不是同一個入口）。
    _prepare(tmp_path, left_right_symmetry="off")  # 這一題不考左右對稱，明寫關著
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, NAME)
        page.locator("#elevation-wall-x0").click()
        choose_fixed(page, "x0", "a", HIT)
        choose_mode(page, "x0", "a", "search")
        page.wait_for_function("!document.querySelector('main').inert && !inputBusy && zoneLimitsShown === zoneLimitsGeneration")
        expect_mode(page, "x0", "a", "search")
        assert cast(list[str], page.evaluate("zoneDesign.zones.x0.a.allowed")) == WOOLS
        expect(page.locator("#zone-preset-x0-a-this_or_none")).to_have_count(0)
        open_panel(page, "x0", "a")
        page.locator("#zone-panel-add").click()
        pick = page.locator(f"#zone-panel-pick-{HIT}")
        expect(pick).to_be_disabled()
        expect(pick.locator(".zv-covered-why")).to_have_text(cast(str, WORDS["search_why"]))
        page.locator(f"#zone-panel-pick-{PLAIN_ID}").click()
        apply_and_return(page)
        assert cast(list[str], page.evaluate("zoneDesign.zones.x0.a.allowed")) == [*WOOLS, PLAIN_ID]
        assert (watched.console_errors, watched.page_errors) == ([], [])


def test_the_panel_says_a_stored_one_is_already_listed_and_lets_it_be_unticked_but_not_ticked_back(tmp_path: Path, browser: Browser) -> None:
    """小框那一列寫明清單裡本來有它；小框可以拿掉、不能加回（#1265 單元 3 之二收尾改名，原名
    test_the_old_table_lets_a_stored_one_be_unticked_and_not_ticked_back：舊做法表拆了，那一句只剩小框那一列）。"""
    _prepare(tmp_path, {"x0/a": LISTED}, left_right_symmetry="off")  # 這一題不考左右對稱，明寫關著
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, NAME)
        page.locator("#elevation-wall-x0").click()
        # 舊表方塊上那一句 → 小框那一列（同一份 reference_only 的字）。
        open_panel(page, "x0", "a")
        expect(page.locator(f"#zone-panel-row-{HIT} .zv-covered-why")).to_have_text(cast(str, WORDS["listed_why"]))
        page.locator("#zone-page-close").click()
        edit_candidates(page, "x0", "a", remove=(HIT,))
        open_panel(page, "x0", "a")
        expect(page.locator(f"#zone-panel-row-{HIT}")).to_have_count(0)
        page.locator("#zone-panel-add").click()
        expect(page.locator(f"#zone-panel-pick-{HIT}")).to_be_disabled()
        assert cast(list[str], page.evaluate("zoneDesign.zones.x0.a.allowed")) == ["glass_wool_48k", "none"]
        assert (watched.console_errors, watched.page_errors) == ([], [])


# 把最底下那一道（``setZoneSetting`` 裡的那一段）拿掉，只留原本的寫入：考每一條路自己的那一道。
_NO_LAST_GUARD = """() => { setZoneSetting = (wall, key, setting) => {
    const design = designForEdit(); design.zones[wall] ??= {}; design.zones[wall][key] = setting; }; }"""


def test_each_way_of_writing_a_list_keeps_it_out_by_itself_without_the_last_guard(tmp_path: Path, browser: Browser) -> None:
    """最底下那一道之外，每一條路自己也擋（把最底下那一道換掉再試）：①剛改過的清單（上一次的組合）帶到下一區
    （#1265 單元 3 之二第二趟起舊做法表拆了，改從輸入頁「改這一區 ▸」開小框）；②輸入頁沒有逐區表；③從卡選區後小框的候選按鈕被人改成勾得到再點；④選整面套用時，
    還沒交給程式的區照畫面上那一份清單起頭、但不帶它。"""
    _prepare(tmp_path, {"x0/a": LISTED}, left_right_symmetry="off")  # 這一題不考左右對稱，明寫關著
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, NAME)
        page.evaluate(_NO_LAST_GUARD)
        page.locator("#elevation-wall-x0").click()
        # ① 動有它的那一份清單，再把另一區改成由程式最佳化：預帶的清單與底下那一句都沒有它。
        edit_candidates(page, "x0", "a", remove=("glass_wool_48k",))
        assert cast(list[str], page.evaluate("zoneDesign.zones.x0.a.allowed")) == [HIT, "none"]
        # ①（舊做法表那一路拆了）改從輸入頁「改這一區 ▸」開小框把另一區改成由程式最佳化：小框自己那一道也要擋。
        choose_mode(page, "x0", "b", "search")
        page.wait_for_function("!document.querySelector('main').inert && !inputBusy && zoneLimitsShown === zoneLimitsGeneration")
        assert cast(list[str], page.evaluate("zoneDesign.zones.x0.b.allowed")) == WOOLS
        # ② 輸入頁沒有這個寫入入口（逐區表整張拆了）。
        expect(page.locator("#zone-table-fold")).to_have_count(0)
        assert cast(list[str], page.evaluate("zoneDesign.zones.x0.b.allowed")) == WOOLS
        # ③ 從卡選區後小框的候選按鈕硬改成勾得到再點：清單不變。
        page.locator("#zone-page-open").click()
        expect(page.locator("#zone-map")).to_be_visible()
        _settled(page)
        before = _allowed(page)
        page.locator("#zone-list-select").click()
        page.locator("#zone-panel-add").click()
        expect(page.locator(f"#zone-panel-pick-{HIT}")).to_be_disabled()
        page.locator(f"#zone-panel-pick-{HIT}").evaluate("box => { box.disabled = false; box.click(); }")
        _apply(page)
        assert _allowed(page) == before and HIT not in before["x0/b"]
        # ④ 選整面套用：先把沒有它的那一區放回不貼，剩下有它的那一區（畫面上那一份清單就是它的）；整面交給程式套用，
        # 還沒交給程式的區拿到的起始清單照那一份、但不帶它。
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
_UNLOCKED_AND_VERDICT_SHOWN = "!document.querySelector('main').inert && !inputBusy && zoneLimitsShown === zoneLimitsGeneration"


def _wall_zones() -> list[str]:
    """四面牆每一區的代號（方案裡的順序）。"""
    scene = cast(dict[str, object], symmetry_document(CUTS)["scene"])
    zones = cast(dict[str, list[dict[str, object]]], scene["surface_zones_by_wall"])
    return [f"{wall}/{zone.get('id', 'whole_wall')}" for wall in ("x0", "xL", "y0", "yL") for zone in zones[wall]]


def _served_refusals(page: Page) -> list[dict[str, object]]:
    return cast(list[dict[str, object]], page.evaluate("zoneLimits.reference_only ?? []"))


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


def _scheme_with_original(zones: list[str]) -> dict[str, object]:
    """假房間，指名的幾區原牆換成命中的構造（等於材料庫那一筆）。"""
    document = symmetry_document(CUTS)
    by_wall = cast(dict[str, list[dict[str, object]]], cast(dict[str, object], document["scene"])["surface_zones_by_wall"])
    for wall, entries in by_wall.items():
        for entry in entries:
            if f"{wall}/{entry.get('id', 'whole_wall')}" in zones:
                entry["material"] = BONDED_VPR
    return document


def _no_library(page: Page) -> None:
    page.route("**/api/materials", lambda route: route.fulfill(status=500, json={"error": "暫時讀不到材料庫（考卷植的）"}))


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
