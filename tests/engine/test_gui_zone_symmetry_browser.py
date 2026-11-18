"""真瀏覽器考「這間房貼什麼」那一頁的「左右對稱」（#1206 第 2 支）：開著、關掉、沒有交給程式的區、配不上，四種各一題。

資料全是考卷自己編的（``_zone_symmetry_room``，5.0 × 4.0 × 2.7 m，前牆 x0）。配對是伺服器算的（限制檢查那一支順便回），
這裡只看畫面照著畫：成對兩區掛同一個圖形與「和○牆○一對」、配不上的兩區掛「沒配成一對」、原因與兩種切法點開才看得到、
底下那一條照單位數寫、開關整頁只有一個、關掉才寫進設計檔。"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import cast

from playwright.sync_api import Browser, Page, Route, expect

from aosr.gui.zone_design import DesignSettings, write_design
from aosr.reporting.scheme import Scheme
from tests.engine._gui_cache import gui_startup_identity_memo as gui_startup_identity_memo
from tests.engine._zone_symmetry_room import CENTRED, FRONT_PAIR, NAME, SIDE, Box, Doc, symmetry_document
from tests.engine.test_gui_browser import _open, _serve, browser as browser
from tests.engine.test_gui_zone_elevation_browser import _open_scheme, _write
from tests.engine.test_gui_zone_material_pick_browser import _TEXT_STYLES

WOOLS = ["glass_wool_48k", "glass_wool_24k", "glass_wool_96k", "none"]
ALL_CUTS: dict[str, list[Box]] = {"x0": FRONT_PAIR, "y0": SIDE, "yL": SIDE, "ceiling": CENTRED}
UNEVEN: dict[str, list[Box]] = {"x0": [FRONT_PAIR[0], ("a", (2.5, 3.4), (0.3, 2.4))], "y0": SIDE, "yL": SIDE}
_MARKS = """() => Object.fromEntries([...document.querySelectorAll('#zone-map .zv-zone')].filter(zone => zone.dataset.pair || zone.dataset.missed)
    .map(zone => [`${zone.dataset.wall}/${zone.dataset.zone}`, {pair: zone.dataset.pair ?? null, shape: zone.dataset.shape ?? null,
        missed: zone.dataset.missed ?? null, words: zone.querySelector('.zv-pair-words')?.textContent ?? null}]))"""


def _design(zones: dict[str, list[str]], **fields: object) -> Doc:
    return {"schema_version": "aosr.zone_design.v1", "scheme_id": NAME,
            "zones": {wall: {zone: {"mode": "search", "allowed": WOOLS} for zone in names} for wall, names in zones.items()}, **fields}


def _prepare(tmp_path: Path, cuts: dict[str, list[Box]], design: Doc | None) -> Path:
    _write(tmp_path, symmetry_document(cuts))
    path = tmp_path / "designs" / f"{NAME}.json"
    if design is not None:
        path.parent.mkdir(exist_ok=True)
        # 用產品自己的寫法存（跟網頁存出來的同一種排法），「打開不改就存」才比得出逐位相同。
        write_design(tmp_path, DesignSettings.model_validate(design), Scheme.model_validate(symmetry_document(cuts)))
    return path


def _settled(page: Page) -> None:
    page.wait_for_function("""() => { const page = document.getElementById('zone-page');
        return !page.inert && !inputBusy && !(Number(page.dataset.changing) > 0) && zoneLimitsShown === zoneLimitsGeneration; }""")


def _open_page(page: Page) -> None:
    _open_scheme(page, NAME)
    page.locator("#zone-page-open").click()
    expect(page.locator("#zone-map")).to_be_visible()
    _settled(page)
    expect(page.locator("#zone-symmetry")).not_to_have_attribute("data-state", "pending")


def _save(page: Page) -> None:
    page.locator("#zone-page-close").click()
    with page.expect_response(lambda r: r.url.endswith(f"/api/designs/{NAME}") and r.request.method == "PUT") as stored:
        page.locator("#save").click()
    assert stored.value.ok
    expect(page.locator("#messages")).to_have_text(re.compile("已儲存|方案沒有變動"))


def _marks(page: Page) -> dict[str, dict[str, str | None]]:
    return cast(dict[str, dict[str, str | None]], page.evaluate(_MARKS))


def test_switched_on_paired_zones_carry_the_same_shape_and_name_each_other(tmp_path: Path, browser: Browser) -> None:
    """開著（設計檔沒有那一格）：前牆兩區、左右牆兩區各一對；每一對兩區掛同一個圖形、互相寫「和○牆○一對」，兩對圖形不同；
    天花置中那一區不掛記號；底下那一條寫「2 對一起調、1 區各自調」；開關整頁只有一個（選了區、右欄換成小框時也是）；
    那一塊的字不小於 12 px、對比至少 4.5；打開不改就存，設計檔不多出那一格。"""
    path = _prepare(tmp_path, ALL_CUTS, _design({"x0": ["b", "a"], "y0": ["s"], "yL": ["s"], "ceiling": ["c"]}))
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_page(page)
        box = page.locator("#zone-symmetry")
        expect(box).to_have_attribute("data-state", "paired")
        expect(page.locator("#zone-symmetry-on")).to_have_attribute("aria-pressed", "true")
        expect(page.locator("#zone-symmetry-off")).to_have_attribute("aria-pressed", "false")
        expect(box).to_contain_text("只管由程式最佳化的區")
        expect(box).to_contain_text("兩區一起調，沒有主次")
        expect(box).to_contain_text("對稱軸")
        expect(page.locator("#zone-symmetry-pairs .zv-sym-pair")).to_have_count(2)
        marks = _marks(page)
        assert set(marks) == {"x0/b", "x0/a", "y0/s", "yL/s"}
        assert marks["x0/b"]["pair"] == "x0/a" and marks["x0/a"]["pair"] == "x0/b"
        assert marks["y0/s"]["pair"] == "yL/s" and marks["yL/s"]["pair"] == "y0/s"
        assert marks["x0/b"]["shape"] == marks["x0/a"]["shape"] and marks["y0/s"]["shape"] == marks["yL/s"]["shape"]
        assert marks["x0/b"]["shape"] != marks["y0/s"]["shape"]
        assert all(mark["missed"] is None for mark in marks.values())
        assert re.fullmatch(r"和前牆.+一對", cast(str, marks["x0/b"]["words"]))
        assert re.fullmatch(r"和右牆.+一對", cast(str, marks["y0/s"]["words"]))
        assert re.fullmatch(r"和左牆.+一對", cast(str, marks["yL/s"]["words"]))
        # 記號要真的看得見（審查 B 第 1 條：只讀屬性與字，藏起來也讀得到）：每一區的圖形與「和○牆○一對」那一句。
        for wall, zone in (("x0", "b"), ("x0", "a"), ("y0", "s"), ("yL", "s")):
            shape = page.locator(f"#zone-map [data-wall='{wall}'][data-zone='{zone}']")
            expect(shape.locator(".zv-pair-mark")).to_be_visible()
            expect(shape.locator(".zv-pair-words")).to_be_visible()
        expect(page.locator("#zone-map .zv-sym-axis").first).to_be_visible()
        expect(page.locator("#zone-result-search")).to_contain_text("由程式最佳化 5 區：2 對一起調、1 區各自調")
        rows = cast(list[dict[str, object]], box.evaluate(_TEXT_STYLES))
        assert rows and all(cast(float, row["size"]) >= 12 and cast(float, row["contrast"]) >= 4.5 for row in rows), rows
        expect(page.locator("text=左右對稱")).to_have_count(1)
        page.locator("#zone-map [data-wall='x0'][data-zone='b'] .zv-fill").click()
        expect(page.locator("#zone-panel")).to_be_visible()
        expect(page.locator("text=左右對稱")).to_have_count(1)
        page.locator("#zone-panel-cancel").click()
        before = path.read_bytes()
        _save(page)
        assert watched.page_errors == [] and watched.console_errors == []
    assert path.read_bytes() == before and "left_right_symmetry" not in json.loads(before)


def test_switching_off_clears_the_marks_and_only_then_the_design_file_says_so(tmp_path: Path, browser: Browser) -> None:
    """關掉：圖上的記號全拿掉、底下那一條不再寫幾對、那一塊寫每一區各自調；存檔後設計檔多一格「關」，其他不變；
    重新打開還是關著；再開回來存檔，那一格拿掉。"""
    design = _design({"x0": ["b", "a"], "y0": ["s"], "yL": ["s"]})
    path = _prepare(tmp_path, ALL_CUTS, design)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_page(page)
        assert len(_marks(page)) == 4
        page.locator("#zone-symmetry-off").click()
        _settled(page)
        expect(page.locator("#zone-symmetry")).to_have_attribute("data-state", "off")
        expect(page.locator("#zone-symmetry-off")).to_have_attribute("aria-pressed", "true")
        expect(page.locator("#zone-symmetry")).to_contain_text("每一區各自調")
        assert _marks(page) == {}
        expect(page.locator("#zone-map .zv-sym-axis")).to_have_count(0)
        expect(page.locator("#zone-result-search")).to_contain_text("由程式最佳化 4 區")
        expect(page.locator("#zone-result-search")).not_to_contain_text("一起調")
        expect(page.locator("#zone-page-unsaved")).to_be_visible()
        _save(page)
        stored = json.loads(path.read_text(encoding="utf-8"))
        assert stored == design | {"left_right_symmetry": "off", "original_scheme_id": None}
        page.reload(wait_until="networkidle")
        _open_page(page)
        expect(page.locator("#zone-symmetry")).to_have_attribute("data-state", "off")
        assert _marks(page) == {}
        page.locator("#zone-symmetry-on").click()
        _settled(page)
        expect(page.locator("#zone-symmetry")).to_have_attribute("data-state", "paired")
        assert len(_marks(page)) == 4
        _save(page)
        assert watched.page_errors == [] and watched.console_errors == []
    assert "left_right_symmetry" not in json.loads(path.read_text(encoding="utf-8"))


def test_with_no_zone_handed_to_the_program_it_says_there_is_nothing_to_pair(tmp_path: Path, browser: Browser) -> None:
    """沒有一區交給程式：開著，但只寫一句現在沒有要配對的；沒有記號、沒有配對細節、沒有紅字。"""
    _prepare(tmp_path, ALL_CUTS, None)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_page(page)
        box = page.locator("#zone-symmetry")
        expect(box).to_have_attribute("data-state", "none")
        expect(page.locator("#zone-symmetry-on")).to_have_attribute("aria-pressed", "true")
        expect(box).to_contain_text("現在沒有由程式最佳化的區，所以沒有要配對的")
        assert _marks(page) == {}
        expect(page.locator("#zone-symmetry-details")).to_have_count(0)
        expect(page.locator("#zone-symmetry .limit-red")).to_have_count(0)
        expect(page.locator("#zone-result-search")).to_contain_text("由程式最佳化 0 區")
        assert watched.page_errors == [] and watched.console_errors == []


def test_zones_that_do_not_mirror_say_so_and_the_details_spell_out_both_cuts(tmp_path: Path, browser: Browser) -> None:
    """前牆兩區寬度不同：兩區都掛「沒配成一對」、不掛成對的圖形；左右牆那一對照配；原因與兩種切法點開「配對細節」才看得到，
    數字是伺服器算的那兩句；底下那一條寫「1 對一起調、2 區各自調」。"""
    _prepare(tmp_path, UNEVEN, _design({"x0": ["b", "a"], "y0": ["s"], "yL": ["s"]}))
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_page(page)
        expect(page.locator("#zone-symmetry")).to_have_attribute("data-state", "paired")
        marks = _marks(page)
        assert marks["x0/b"]["missed"] == "true" and marks["x0/a"]["missed"] == "true"
        assert marks["x0/b"]["pair"] is None and marks["x0/b"]["words"] == "沒配成一對" and marks["x0/a"]["words"] == "沒配成一對"
        assert marks["y0/s"]["pair"] == "yL/s"
        # 「沒配成一對」要真的看得見（審查 B 第 2 條）：兩區的黑條與那一句。
        for zone in ("b", "a"):
            shape = page.locator(f"#zone-map [data-wall='x0'][data-zone='{zone}']")
            expect(shape.locator(".zv-missed-bg")).to_be_visible()
            expect(shape.locator(".zv-missed-words")).to_be_visible()
        expect(page.locator("#zone-symmetry-missed .zv-sym-missed")).to_have_count(1)
        # 那一列直接寫原因與怎麼辦（老闆點頭的樣稿）；差幾公分與切法才在配對細節。
        expect(page.locator("#zone-symmetry-missed .zv-sym-missed")).to_contain_text(
            "大小不同，各自調；要配對，回「房間方案輸入」切到左右對稱的位置")
        expect(page.locator("#zone-symmetry-pairs")).to_contain_text("配成 1 對")
        expect(page.locator("#zone-result-search")).to_contain_text("由程式最佳化 4 區：1 對一起調、2 區各自調")
        expect(page.locator("#zone-symmetry-detail")).to_have_count(0)
        expect(page.locator("#zone-page")).not_to_contain_text("2.50～3.60")
        page.locator("#zone-symmetry-details").click()
        detail = page.locator("#zone-symmetry-detail")
        expect(detail).to_be_visible()
        expect(detail).to_contain_text("切到離左牆 2.50～3.60 公尺")
        expect(detail).to_contain_text("切到離左牆 0.60～1.50 公尺")
        expect(detail).to_contain_text("1 公分")
        expect(detail).to_contain_text("靠右牆那一邊差 20 公分")
        expect(page.locator("#zone-symmetry-axis-words")).to_contain_text("現在離左牆 2.00 公尺、在房間正中")
        expect(detail.locator("svg")).to_be_visible()
        rows = cast(list[dict[str, object]], detail.evaluate(_TEXT_STYLES))
        assert rows and all(cast(float, row["size"]) >= 12 and cast(float, row["contrast"]) >= 4.5 for row in rows), rows
        # 晚到的限制檢查重畫之後，細節還開著。
        page.evaluate("refreshZoneLimits()")
        _settled(page)
        expect(page.locator("#zone-symmetry-detail")).to_be_visible()
        page.locator("#zone-symmetry-detail-close").click()
        expect(page.locator("#zone-symmetry-detail")).to_have_count(0)
        assert watched.page_errors == [] and watched.console_errors == []


def test_half_a_pair_is_marked_missed_and_the_reason_only_shows_in_the_details(tmp_path: Path, browser: Browser) -> None:
    """半對兩種（主對話補的規格）：前牆只切了左邊一塊、鏡像位置沒有一樣的區；左牆那一區由程式最佳化、右牆同位置那一區沒交給程式。
    兩區都掛跟「位置對不上」同一種「沒配成一對」；沒交給程式的那一區自己不掛；那一列寫一句原因，位置數字在「配對細節」。"""
    _prepare(tmp_path, {"x0": [FRONT_PAIR[0]], "y0": SIDE, "yL": SIDE}, _design({"x0": ["b"], "y0": ["s"]}))
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_page(page)
        expect(page.locator("#zone-symmetry")).to_have_attribute("data-state", "paired")
        marks = _marks(page)
        assert set(marks) == {"x0/b", "y0/s"}
        assert all(mark["missed"] == "true" and mark["pair"] is None and mark["words"] == "沒配成一對" for mark in marks.values())
        expect(page.locator("#zone-symmetry-missed .zv-sym-missed")).to_have_count(2)
        expect(page.locator("#zone-symmetry-pairs .zv-sym-pair")).to_have_count(0)
        expect(page.locator("#zone-result-search")).not_to_contain_text("一起調")
        # 原因直接寫在那一列（主對話 11-18 更正：以老闆點頭的樣稿為準，不是點開才看得到）。
        expect(page.locator("#zone-symmetry-missed")).to_contain_text("對面沒有一樣大小的區，各自調")
        expect(page.locator("#zone-symmetry-missed")).to_contain_text("對面那一區不是由程式最佳化，各自調")
        expect(page.locator("#zone-page")).not_to_contain_text("2.50～3.60")
        page.locator("#zone-symmetry-details").click()
        detail = page.locator("#zone-symmetry-detail")
        expect(detail.locator(".zv-sym-why[data-kind='no_zone']")).to_contain_text("鏡像位置沒有一樣大小的區")
        expect(detail.locator(".zv-sym-why[data-kind='no_zone']")).to_contain_text("離左牆 2.50～3.60 公尺")
        expect(detail.locator(".zv-sym-why[data-kind='not_search']")).to_contain_text("鏡像位置那一區不是由程式最佳化")
        expect(detail.locator(".zv-sym-why[data-kind='not_search']")).to_contain_text(re.compile("把.*右牆.*也改成由程式最佳化"))
        expect(detail).not_to_contain_text("{")
        rows = cast(list[dict[str, object]], detail.evaluate(_TEXT_STYLES))
        assert rows and all(cast(float, row["size"]) >= 12 and cast(float, row["contrast"]) >= 4.5 for row in rows), rows
        assert watched.page_errors == [] and watched.console_errors == []


_HOLD = "**/api/zone-limits"


def test_a_reply_still_on_its_way_never_leaves_the_old_pairing_on_screen(tmp_path: Path, browser: Browser) -> None:
    """不准畫舊回覆（審查 B 的 G 組）：畫著兩對的時候改了做法（前牆②改成保留原樣），新的限制檢查還沒回來。
    這段時間不准還畫著上一次的配對（那已經不是現在的做法）：寫計算中、圖上沒有記號；回來了才照新的畫。"""
    _prepare(tmp_path, ALL_CUTS, _design({"x0": ["b", "a"], "y0": ["s"], "yL": ["s"]}))
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_page(page)
        assert len(_marks(page)) == 4
        held: list[Route] = []
        page.route(_HOLD, lambda route: held.append(route))
        page.evaluate("""() => { designForEdit().zones.x0.a = {mode: 'keep_original'}; refreshZoneLimits(); renderZoneOverview(); }""")
        expect(page.locator("#zone-symmetry")).to_have_attribute("data-state", "pending")
        expect(page.locator("#zone-symmetry")).to_contain_text("配對計算中")
        assert _marks(page) == {} and len(held) == 1
        expect(page.locator("#zone-result-search")).not_to_contain_text("一起調")
        held[0].continue_()
        page.unroute(_HOLD)
        _settled(page)
        expect(page.locator("#zone-symmetry")).to_have_attribute("data-state", "paired")
        marks = _marks(page)
        assert marks["x0/b"]["missed"] == "true" and "x0/a" not in marks and marks["y0/s"]["pair"] == "yL/s"
        assert watched.page_errors == [] and watched.console_errors == []


def test_when_the_pairing_cannot_be_computed_the_box_says_so(tmp_path: Path, browser: Browser) -> None:
    """配對算不出來的兩種（審查 B 的 G4、G5）：限制檢查整支失敗、或伺服器說這份草稿算不了配對。那一塊照講一句，
    圖上沒有記號，不留著上一次的。"""
    _prepare(tmp_path, ALL_CUTS, _design({"x0": ["b", "a"], "y0": ["s"], "yL": ["s"]}))
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_page(page)
        assert len(_marks(page)) == 4
        page.route(_HOLD, lambda route: route.fulfill(status=500, json={"error": "考卷讓它失敗"}))
        page.evaluate("refreshZoneLimits()")
        _settled(page)
        box = page.locator("#zone-symmetry")
        expect(box).to_have_attribute("data-state", "unavailable")
        expect(box).to_contain_text("配對沒算成：考卷讓它失敗")
        assert _marks(page) == {}
        page.unroute(_HOLD)

        def unavailable(route: Route) -> None:
            body = route.fetch().json()
            body["symmetry"] = {"available": False, "message": "考卷說這份草稿算不了"}
            route.fulfill(json=body)

        page.route(_HOLD, unavailable)
        page.evaluate("refreshZoneLimits()")
        _settled(page)
        expect(box).to_have_attribute("data-state", "unavailable")
        expect(box).to_contain_text("考卷說這份草稿算不了")
        expect(box).not_to_contain_text("配對沒算成")
        assert _marks(page) == {}
        page.unroute(_HOLD)


def test_a_narrow_missed_zone_still_shows_a_visible_mark(tmp_path: Path, browser: Browser) -> None:
    """窄到放不下那一句的區（審查 B 的 G7）：沒配成時至少有一個看得見的小記號；夠寬的那一區照掛整條。"""
    _prepare(tmp_path, {"x0": [FRONT_PAIR[0], ("a", (2.5, 3.1), (0.3, 2.4))]}, _design({"x0": ["b", "a"]}))
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_page(page)
        narrow = page.locator("#zone-map [data-wall='x0'][data-zone='a']")
        expect(narrow).to_have_attribute("data-missed", "true")
        expect(narrow.locator(".zv-missed-dot")).to_be_visible()
        expect(narrow.locator(".zv-missed-words")).to_be_hidden()
        wide = page.locator("#zone-map [data-wall='x0'][data-zone='b']")
        expect(wide.locator(".zv-missed-bg")).to_be_visible()
        expect(wide.locator(".zv-missed-dot")).to_have_count(0)
        assert watched.page_errors == [] and watched.console_errors == []


def test_the_details_close_by_themselves_once_nothing_is_missed(tmp_path: Path, browser: Browser) -> None:
    """配對細節開著的時候，沒配成的區都配上了（審查 B 的 G8）：細節框自己收起來，不留一張空的；之後又有沒配成的，也不自己跳出來。"""
    _prepare(tmp_path, {"y0": SIDE, "yL": SIDE}, _design({"y0": ["s"]}))
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_page(page)
        page.locator("#zone-symmetry-details").click()
        expect(page.locator("#zone-symmetry-detail")).to_be_visible()
        page.evaluate("""() => { const zones = designForEdit().zones; zones.yL = {s: structuredClone(zones.y0.s)};
            refreshZoneLimits(); renderZoneOverview(); }""")
        _settled(page)
        expect(page.locator("#zone-symmetry-pairs .zv-sym-pair")).to_have_count(1)
        expect(page.locator("#zone-symmetry-detail")).to_have_count(0)
        expect(page.locator("#zone-symmetry-details")).to_have_count(0)
        page.evaluate("() => { delete designForEdit().zones.yL; refreshZoneLimits(); renderZoneOverview(); }")
        _settled(page)
        expect(page.locator("#zone-symmetry-details")).to_have_count(1)
        expect(page.locator("#zone-symmetry-detail")).to_have_count(0)
        assert watched.page_errors == [] and watched.console_errors == []


def test_off_then_on_then_save_leaves_no_unsaved_banner(tmp_path: Path, browser: Browser) -> None:
    """審查 B 第 3 條：沒有設計檔的方案，「關→開→存」之後回到那一頁，「改的還沒存進檔案」那一句要消；也不多出設計檔。"""
    _prepare(tmp_path, ALL_CUTS, None)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_page(page)
        page.locator("#zone-symmetry-off").click()
        _settled(page)
        expect(page.locator("#zone-page-unsaved")).to_be_visible()
        page.locator("#zone-symmetry-on").click()
        _settled(page)
        page.locator("#zone-page-close").click()
        page.locator("#save").click()
        expect(page.locator("#messages")).to_have_text(re.compile("已儲存|方案沒有變動"))
        page.locator("#zone-page-open").click()
        _settled(page)
        expect(page.locator("#zone-page-unsaved")).to_have_count(0)
        assert watched.page_errors == [] and watched.console_errors == []
    assert not (tmp_path / "designs").exists() or not any((tmp_path / "designs").iterdir())


_HIT = """() => { const link = document.getElementById('zone-symmetry-details'); const box = link.getBoundingClientRect();
    const hit = document.elementFromPoint(box.x + box.width / 2, box.y + box.height / 2);
    return {onLink: Boolean(hit && link.contains(hit)), scrollY: window.scrollY,
            barHeight: document.getElementById('zone-result').getBoundingClientRect().height}; }"""


def test_the_details_link_is_reachable_without_scrolling_at_1440_by_900(tmp_path: Path, browser: Browser) -> None:
    """審查 B：樣稿的視窗大小（1440×900）、不捲動時，「配對細節」不准被底下那一條蓋住；那一條是一行（面積寫小字）。"""
    _prepare(tmp_path, {**UNEVEN, "ceiling": CENTRED}, _design({"x0": ["b", "a"], "y0": ["s"], "yL": ["s"], "ceiling": ["c"]}))
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        page.set_viewport_size({"width": 1440, "height": 900})
        _open_page(page)
        page.evaluate("window.scrollTo(0, 0)")
        found = cast(dict[str, object], page.evaluate(_HIT))
        assert found["onLink"] is True and found["scrollY"] == 0, found
        assert cast(float, found["barHeight"]) <= 80, found
        expect(page.locator("#zone-result-search")).to_contain_text("由程式最佳化 5 區：1 對一起調、3 區各自調")
        assert watched.page_errors == [] and watched.console_errors == []
