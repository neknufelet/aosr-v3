"""真瀏覽器考每一區的做法（#1105，#1082 第 1-3 段）：立面圖下的逐區表選固定材料／不貼／交給搜尋並限定准用；
再切同一面別處，沒動到的區做法照留（跟著自動補滿重編的代號走）；存檔再打開，切法、做法、准用、原方案都在；
沒設過做法的舊方案存檔不多寫一份設計設定。房間一律用 109 號房的長寬高。"""
import json
from pathlib import Path
from typing import cast

from playwright.sync_api import Browser, Page, Route, expect

from aosr.materials.construction_models import Assembly
from aosr.materials.library import FixedEntry, builtin_entries
from tests.engine._gui_cache import gui_startup_identity_memo as gui_startup_identity_memo
from tests.engine._surface_zone_cases import ROOM_109, room109_document
from tests.engine.test_gui_browser import _open, _serve, browser as browser
from tests.engine.test_gui_zone_elevation_browser import _drag, _open_scheme, _write, _zones

LY, LZ = ROOM_109["Ly"], ROOM_109["Lz"]
PRESET = "glass_wool_48k_50mm"


def _layers(material_id: str) -> object:
    entry = next(item for item in builtin_entries() if item.id == material_id)
    assert isinstance(entry, FixedEntry) and isinstance(entry.material, Assembly)
    return entry.material.model_dump(mode="json")["layers"]


def _choose(page: Page, selector: str, value: str) -> None:
    """選一個會換方案材料的選單：等 /api/zone-material 回來、整頁解鎖。"""
    with page.expect_response("**/api/zone-material") as response:
        page.locator(selector).select_option(value)
    assert response.value.ok, response.value.text()
    page.wait_for_function("!document.querySelector('main').inert")


def _design(page: Page) -> dict[str, object]:
    return cast(dict[str, object], page.evaluate("structuredClone(zoneDesign)"))


def _modes(page: Page, wall: str) -> dict[str, dict[str, object]]:
    return cast(dict[str, dict[str, dict[str, object]]], _design(page)["zones"])[wall]


def _save(page: Page, name: str) -> None:
    """按「儲存」，等方案與設計設定兩個 PUT 都回來。"""
    with page.expect_response(lambda r: r.url.endswith(f"/api/designs/{name}") and r.request.method == "PUT") as design:
        page.locator("#save").click()
    assert design.value.ok, design.value.text()
    expect(page.locator("#messages")).to_have_class("ok")


def _region(zone: dict[str, object]) -> str:
    """區中心落在第一次切完的哪一塊：z1（左下）、左上那塊、右邊那一大塊。"""
    u = sum(cast(list[float], zone["u_m"])) / 2
    v = sum(cast(list[float], zone["v_m"])) / 2
    return "z1" if u < 1.0 and v < 1.2 else "top-left" if u < 1.0 else "right"


def _first_cut(page: Page, tmp_path: Path) -> tuple[dict[str, dict[str, object]], list[str]]:
    """x 起點牆不鏡像：左下畫一框 → z1、右邊一大塊 auto1、z1 上面 auto2（三塊都還是原牆）；三塊各選一種做法、搜尋那塊限定准用。"""
    _drag(page, LY, LZ, (0.02, 0.02), (1.0, 1.2))
    assert sorted(str(zone["id"]) for zone in _zones(page, "x0")) == ["auto1", "auto2", "z1"]
    wall = _zones(page, "x0")[0]["material"]
    assert all(zone["material"] == wall for zone in _zones(page, "x0"))
    _choose(page, "#zone-fixed-x0-z1", PRESET)
    _choose(page, "#zone-mode-x0-auto1", "search")
    _choose(page, "#zone-mode-x0-auto2", "keep_original")
    page.locator("#zone-limit-x0-auto1").check()
    page.locator("#zone-allow-x0-auto1-glass_wool_96k").uncheck()
    page.locator("#zone-allow-x0-auto1-carpet_on_concrete").check()
    page.locator("#original-scheme").select_option("room109-original")
    expect(page.locator("#elevation [data-zone-id=auto1]")).to_have_class("zone zone-auto zone-mode-search")
    expect(page.locator("#zone-note")).to_contain_text("auto2")

    zones = {str(zone["id"]): zone for zone in _zones(page, "x0")}
    assert zones["z1"]["material"] == {"kind": "mounted_on_wall", "layers": _layers(PRESET), "wall": wall}
    assert zones["auto1"]["material"] == wall and zones["auto2"]["material"] == wall
    allowed = ["glass_wool_48k", "glass_wool_24k", "carpet_on_concrete", "none"]
    assert _modes(page, "x0") == {"z1": {"mode": "fixed"}, "auto1": {"mode": "search", "allowed": allowed},
                                  "auto2": {"mode": "keep_original"}}
    page.locator("#elevation-panel").screenshot(path=str(tmp_path / "modes-first-cut.png"))
    return zones, allowed


def test_each_zone_picks_a_mode_survives_a_recut_and_a_reopen(tmp_path: Path, browser: Browser) -> None:
    _write(tmp_path, room109_document())
    _write(tmp_path, room109_document("room109-original"))
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-x0").click()
        expect(page.locator("#zone-mode-x0-whole_wall")).to_have_value("fixed")

        zones, allowed = _first_cut(page, tmp_path)
        # 在右邊那一大塊（交給搜尋）裡再畫一框：自動區重編，每一塊照中心落在哪一塊沿用那一塊的做法與准用。
        _drag(page, LY, LZ, (2.02, 0.02), (3.0, 1.0))
        after = {str(zone["id"]): zone for zone in _zones(page, "x0")}
        assert len(after) > 3
        expected = {"z1": {"mode": "fixed"}, "top-left": {"mode": "keep_original"},
                    "right": {"mode": "search", "allowed": allowed}}
        assert _modes(page, "x0") == {key: expected[_region(zone)] for key, zone in after.items()}
        assert after["z1"]["material"] == zones["z1"]["material"]
        _save(page, "room109")
        stored = json.loads((tmp_path / "designs" / "room109.json").read_text(encoding="utf-8"))
        assert stored["original_scheme_id"] == "room109-original"
        assert stored["zones"]["x0"] == _modes(page, "x0")

        # 重新整理再打開：切法、每區做法、准用、原方案都在，畫面照存的長回來。
        page.reload()
        page.wait_for_function("typeof inputDefaults !== 'undefined' && inputDefaults !== undefined")
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-x0").click()
        assert {str(zone["id"]): zone for zone in _zones(page, "x0")} == after
        assert _modes(page, "x0") == stored["zones"]["x0"]
        expect(page.locator("#original-scheme")).to_have_value("room109-original")
        right = next(key for key, zone in after.items() if _region(zone) == "right" and key != "z2")
        expect(page.locator(f"#zone-mode-x0-{right}")).to_have_value("search")
        # 准用清單收在一行摘要後面：摘要寫出勾了哪幾種，點開才看得到勾選格。
        expect(page.locator(f"#zone-allowed-summary-x0-{right}")).to_contain_text("准用 4 種")
        expect(page.locator(f"#zone-allowed-summary-x0-{right}")).to_contain_text("地毯直接鋪混凝土")
        page.locator(f"#zone-allowed-summary-x0-{right}").click()
        expect(page.locator(f"#zone-allow-x0-{right}-glass_wool_96k")).not_to_be_checked()
        expect(page.locator(f"#zone-allow-x0-{right}-carpet_on_concrete")).to_be_checked()
        expect(page.locator("#zone-fixed-x0-z1")).to_have_value(PRESET)
        page.locator("#elevation-panel").screenshot(path=str(tmp_path / "modes-reopened.png"))

        # 准用清單拿到只剩固定材料：照實提醒（搜尋要一個起點），開跑時伺服器再擋一次。
        for choice in ("glass_wool_48k", "glass_wool_24k", "none"):
            page.locator(f"#zone-allow-x0-{right}-{choice}").uncheck()
        expect(page.locator(f"#zone-allowed-problem-x0-{right}")).to_contain_text("起點")
        assert watched.page_errors == [] and watched.console_errors == []


def test_old_scheme_opens_with_every_zone_fixed_and_saving_writes_no_design(tmp_path: Path, browser: Browser) -> None:
    _write(tmp_path, room109_document("plain"))
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, "plain")
        for wall in ("floor", "x0", "yL"):
            page.locator(f"#elevation-wall-{wall}").click()
            expect(page.locator(f"#zone-mode-{wall}-whole_wall")).to_have_value("fixed")
        with page.expect_response(lambda r: r.url.endswith("/api/schemes/plain") and r.request.method == "PUT") as saved:
            page.locator("#save").click()
        assert saved.value.ok
        expect(page.locator("#messages")).to_have_class("ok")
        assert not (tmp_path / "designs" / "plain.json").exists()
        assert watched.page_errors == [] and watched.console_errors == []


def test_auto_label_is_left_out_of_a_strip_too_narrow_for_it(tmp_path: Path, browser: Browser) -> None:
    """窄條裡「自動」兩字放不下就不寫（滑過去看 title），不擠出界壓到隔壁區。"""
    _write(tmp_path, room109_document())
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-x0").click()
        # 框離右牆 25 公分：右邊補出一條 25 公分寬的窄條（過最小邊、但寫不下兩個字）。
        _drag(page, LY, LZ, (0.02, 0.02), (LY - 0.25, 1.2))
        strip = next(str(zone["id"]) for zone in _zones(page, "x0")
                     if cast(list[float], zone["u_m"])[0] > LY - 0.3)
        expect(page.locator(f"#elevation [data-zone-id={strip}] text")).to_have_count(0)
        expect(page.locator(f"#elevation [data-zone-id={strip}] title")).to_contain_text("自動")
        assert watched.page_errors == [] and watched.console_errors == []


def test_unreadable_design_is_not_overwritten_and_a_dropped_request_puts_the_menu_back(tmp_path: Path,
                                                                                      browser: Browser) -> None:
    """存著的設計設定讀不到：照講、存方案時不拿空的一份蓋掉原檔；換材料半路斷線：照講、選單畫回原樣、方案不動。"""
    _write(tmp_path, room109_document())
    (tmp_path / "designs").mkdir(exist_ok=True)
    broken = tmp_path / "designs" / "room109.json"
    broken.write_text("{not json", encoding="utf-8")
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-x0").click()
        expect(page.locator("#elevation-zones")).to_contain_text("不會存回去")
        page.route("**/api/zone-material", lambda route: route.abort())
        page.locator("#zone-mode-x0-whole_wall").select_option("search")
        expect(page.locator("#zone-note")).to_contain_text("沒有換")
        expect(page.locator("#zone-mode-x0-whole_wall")).to_have_value("fixed")
        assert page.evaluate("zoneDesign") is None
        page.unroute("**/api/zone-material")
        _choose(page, "#zone-mode-x0-whole_wall", "search")
        with page.expect_response(lambda r: r.url.endswith("/api/schemes/room109") and r.request.method == "PUT") as saved:
            page.locator("#save").click()
        assert saved.value.ok
        expect(page.locator("#messages")).to_contain_text("每一區的做法沒存成")
        assert broken.read_text(encoding="utf-8") == "{not json"
        assert watched.page_errors == []


def test_opening_a_scheme_swaps_the_form_only_after_its_zone_design_arrives(tmp_path: Path, browser: Browser) -> None:
    """打開方案：每一區的做法還沒讀回來時表單不換、整頁鎖著（不准在表單已換、做法還是上一份的空檔按「儲存」，
    雲端 #1109 第一版就是這樣讓別的考卷按到儲存撲空）；讀回來才跟方案一起換上。"""
    _write(tmp_path, room109_document())
    _write(tmp_path, room109_document("room109-original"))
    (tmp_path / "designs").mkdir(exist_ok=True)
    (tmp_path / "designs" / "room109.json").write_text(json.dumps(
        {"schema_version": "aosr.zone_design.v1", "scheme_id": "room109", "original_scheme_id": "room109-original",
         "zones": {"x0": {"whole_wall": {"mode": "search"}}}}), encoding="utf-8")
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        held: list[Route] = []
        page.route("**/api/designs/room109", lambda route: held.append(route))
        page.locator("#scheme-list").select_option("room109")
        with page.expect_request("**/api/designs/room109"):
            page.locator("#open-scheme").click()
        page.wait_for_function("document.querySelector('main').inert")
        assert page.locator("#save-id").input_value() != "room109"
        assert page.evaluate("zoneDesign") is None
        held[0].continue_()
        page.wait_for_function("document.getElementById('save-id').value === 'room109'")
        assert page.evaluate("[zoneDesign.scheme_id, zoneDesign.original_scheme_id]") == ["room109", "room109-original"]
        page.wait_for_function("!document.querySelector('main').inert")
        page.locator("#elevation-wall-x0").click()
        expect(page.locator("#zone-mode-x0-whole_wall")).to_have_value("search")
        assert watched.page_errors == [] and watched.console_errors == []


def test_send_to_search_appears_only_with_a_search_zone_saves_then_opens_the_search_page(tmp_path: Path,
                                                                                         browser: Browser) -> None:
    """「拿去搜尋」：有一區交給搜尋才出現、改回去就收掉；按下去先照畫面存方案與做法，兩樣都存成才跳到
    /gradient?scheme=<方案>；做法存不成就留在這頁照講（搜尋頁讀的是存好的那一份，不准帶著舊做法過去）。"""
    _write(tmp_path, room109_document())
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-x0").click()
        expect(page.locator("#zone-to-search")).to_have_count(0)
        _choose(page, "#zone-mode-x0-whole_wall", "search")
        expect(page.locator("#zone-to-search")).to_contain_text("1 區交給搜尋")
        page.locator("#elevation-panel").screenshot(path=str(tmp_path / "to-search.png"))
        page.locator("#zone-mode-x0-whole_wall").select_option("fixed")  # 改回固定不換材料、不打伺服器
        expect(page.locator("#zone-to-search")).to_have_count(0)
        _choose(page, "#zone-mode-x0-whole_wall", "search")

        page.route("**/api/designs/room109", lambda route: route.abort() if route.request.method == "PUT"
                   else route.continue_())
        page.locator("#zone-to-search").click()
        expect(page.locator("#zone-note")).to_contain_text("做法沒存成")
        assert page.url == f"{base}/"
        assert not (tmp_path / "designs" / "room109.json").exists()
        page.unroute("**/api/designs/room109")

        page.locator("#zone-to-search").click()
        page.wait_for_url(f"{base}/gradient?scheme=room109")
        stored = json.loads((tmp_path / "designs" / "room109.json").read_text(encoding="utf-8"))
        assert stored["zones"]["x0"] == {"whole_wall": {"mode": "search"}}
        assert watched.page_errors == []
