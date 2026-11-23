"""分區頁分組與存檔的瀏覽器考卷。"""

from __future__ import annotations
from pathlib import Path
from typing import cast
import pytest
from playwright.sync_api import Browser, expect
from aosr.gui import reference_only
from tests.engine._zone_panel_steps import edit_candidates
from tests.engine._construction_gui_server import real_server
from tests.engine._gui_cache import gui_startup_identity_memo as gui_startup_identity_memo
from tests.engine._zone_symmetry_room import NAME, symmetry_document
from tests.engine.test_gui_browser import _open, _serve, browser as browser
from tests.engine._bonded_backing_cases import BONDED_VPR
from tests.engine.test_gui_vpr_reference import HIT, HIT_LABEL, PLAIN_ID, _write_user_library
from tests.engine.test_gui_zone_elevation_browser import _open_scheme
from tests.engine.test_gui_zone_material_pick_browser import _add_candidate
from tests.engine.test_gui_vpr_reference_browser import (
    COMPUTED_MS,
    CUTS,
    KIND,
    LISTED,
    NO_ORANGE,
    NO_ORANGE_ENTRIES,
    WOOLS,
    WORDS,
    _ALLOWED,
    _BAND,
    _CARD_SENTENCES,
    _HeldLimits,
    _OLD_TABLE_SETTLED,
    _allowed,
    _apply,
    _band_at,
    _click_zone,
    _map_words,
    _no_library,
    _no_old_sentence_anywhere,
    _open_material,
    _open_page,
    _prepare,
    _refused,
    _scheme_with_original,
    _served_refusals,
    _settled,
    _wall_zones,
)


def test_the_old_table_search_row_is_stopped_and_says_the_refusal_in_red(tmp_path: Path, browser: Browser) -> None:
    """舊做法表「拿去搜尋」那一列：清單裡有命中那一筆，按鈕停用、旁邊紅字就是開跑會拒收的那一句（伺服器原文）；
    從這列進小框拿掉它，按鈕恢復、旁邊回到平常那一句。"""
    _prepare(tmp_path, {"x0/a": LISTED, "x0/b": WOOLS}, left_right_symmetry="off")  # 這一題不考左右對稱，明寫關著
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, NAME)
        page.locator("#elevation-wall-x0").click()
        page.wait_for_function(_OLD_TABLE_SETTLED)
        expect(page.locator("#zone-to-search")).to_be_disabled()
        expect(page.locator("#zone-to-search-note")).to_have_class("limit-red")
        expect(page.locator("#zone-to-search-note")).to_have_text(_refused(page, "x0/a"))
        edit_candidates(page, "x0", "a", remove=(HIT,))
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
    _prepare(tmp_path, document=document, left_right_symmetry="off")  # 這一題不考左右對稱，明寫關著
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
        _prepare(tmp_path, {"x0/a": start}, left_right_symmetry="off")  # 這一題不考左右對稱，明寫關著
        with _serve(tmp_path) as base, _open(browser, base) as watched:
            page = watched.page
            _open_scheme(page, NAME)
            page.locator("#elevation-wall-x0").click()
            edit_candidates(page, "x0", "a", remove=("glass_wool_48k",))
            assert cast(list[str], page.evaluate("zoneDesign.zones.x0.a.allowed")) == left
            with page.expect_response("**/api/zone-material"):
                page.locator("#zone-mode-x0-b").select_option("search")
            page.wait_for_function(_OLD_TABLE_SETTLED)
            assert cast(list[str], page.evaluate("zoneDesign.zones.x0.b.allowed")) == (WOOLS if HIT in left else ["none"]), left
            assert (watched.console_errors, watched.page_errors) == ([], [])


def test_splitting_a_shared_list_hides_the_old_sentence_everywhere_until_the_new_check_comes_back(tmp_path: Path, browser: Browser) -> None:
    """兩區共用一份有命中的清單（一句點名兩區）。扣住限制檢查，給其中一區多勾一種玻璃棉並套用（兩區都還有那一筆，但不再是
    同一份清單）：新結果回來之前，卡、頁底、按鈕旁三處都沒有舊的那一句；回來之後兩張卡各寫自己那一句。"""
    _prepare(tmp_path, {"x0/a": LISTED, "x0/b": LISTED}, left_right_symmetry="off")  # 這一題不考左右對稱，明寫關著
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_page(page)
        both = cast(str, _served_refusals(page)[0]["text"])
        expect(page.locator(_CARD_SENTENCES)).to_have_text([both])
        expect(page.locator("#zone-page-search-why")).to_have_text("清單有紅框要先處理")
        limits = _HeldLimits(page)
        _click_zone(page, "x0/a")
        _add_candidate(page, "glass_wool_24k")
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
    _prepare(tmp_path, {"x0/a": LISTED}, left_right_symmetry="off")  # 這一題不考左右對稱，明寫關著
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_page(page)
        expect(page.locator(_CARD_SENTENCES)).to_have_text([_refused(page, "x0/a")])
        limits = _HeldLimits(page)
        _click_zone(page, "x0/a")
        page.locator(f"#zone-panel-remove-{HIT}").click()
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
    _prepare(tmp_path, {"x0/a": LISTED, "x0/b": LISTED, "y0/s": [HIT, "none"]}, left_right_symmetry="off")  # 這一題不考左右對稱，明寫關著
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_page(page)
        old = [cast(str, row["text"]) for row in _served_refusals(page)]
        assert len(old) == 2
        assert sorted(page.locator(_CARD_SENTENCES).all_text_contents()) == sorted(old)
        limits = _HeldLimits(page)
        _click_zone(page, "y0/s")
        _add_candidate(page, "glass_wool_48k")
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
        unknown = page.locator(".zv-list[data-zones='x0/b']")
        expect(unknown).to_contain_text("程式從全部可調材料與「不貼」裡選（材料庫沒讀全，列不出是哪幾種）")
        unknown.get_by_role("button", name="選這幾區").click()
        expect(page.locator("#zone-panel-mode-search")).to_be_checked()
        page.locator("#zone-panel-cancel").click()
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
    _prepare(tmp_path, {"x0/a": LISTED, "x0/b": LISTED}, left_right_symmetry="off")  # 這一題不考左右對稱，明寫關著
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_page(page)
        both = cast(str, _served_refusals(page)[0]["text"])
        assert both.count("、") == 1
        expect(page.locator("#zone-page-blocked")).to_have_text(both)
        limits = _HeldLimits(page)
        _click_zone(page, "x0/a")
        page.locator(f"#zone-panel-remove-{HIT}").click()
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
