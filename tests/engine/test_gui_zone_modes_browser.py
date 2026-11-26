"""真瀏覽器考每一區的做法（#1105，#1082 第 1-3 段）：「這間房貼什麼」的小框選指定貼一種／不貼／由程式最佳化並改程式可以挑的清單（#1265 單元 3 之二第二趟起舊逐區表拆了）
（#1176 起改成由程式最佳化就帶好一份明確清單，存檔也寫明確清單）；再切同一面別處，沒動到的區做法照留（跟著自動補滿重編的代號走）；
存檔再打開，切法、做法、清單、原方案都在；沒設過做法的舊方案存檔不多寫一份設計設定。房間一律用 109 號房的長寬高。"""
import json
from pathlib import Path
from typing import cast

from playwright.sync_api import Browser, Page, Route, expect

from aosr.materials.construction_models import Assembly
from aosr.materials.library import FixedEntry, builtin_entries
from tests.engine._input_page_steps import open_example_scheme, reveal, scheme_open_button, scripted_open_scheme
from tests.engine._zone_panel_steps import (apply_and_return, choose_fixed, choose_mode, edit_candidates, expect_mode, fixed_value,
                                            listed, open_alone, open_panel)
from tests.engine._gui_cache import gui_startup_identity_memo as gui_startup_identity_memo
from tests.engine._surface_zone_cases import ROOM_109, room109_document
from tests.engine._zone_naming_room import ZONE_NAMING_SCHEME
from tests.engine.test_gui_browser import _open, _serve, browser as browser
from tests.engine.test_gui_input_clear_undo_browser import _release
from tests.engine.test_gui_zone_elevation_browser import _drag, _open_scheme, _write, _zones

LY, LZ = ROOM_109["Ly"], ROOM_109["Lz"]
PRESET = "glass_wool_48k_50mm"
# 加進清單尾巴的那一筆固定成品（#1176 起清單只列這一面能貼的：地毯只列在地板，牆上改用一筆牆面的固定成品）。
WALL_FIXED = "glass_wool_48k_50mm_gap100"
# 改成由程式最佳化時一開始放的清單：一律是三種內建玻璃棉＋不貼（材料庫多了別的可調材料也不放進來）。
DEFAULT_ALLOWED = ["glass_wool_48k", "glass_wool_24k", "glass_wool_96k", "none"]


def _layers(material_id: str) -> object:
    entry = next(item for item in builtin_entries() if item.id == material_id)
    assert isinstance(entry, FixedEntry) and isinstance(entry.material, Assembly)
    return entry.material.model_dump(mode="json")["layers"]


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


def _write_room109_design(tmp_path: Path, settings: dict[str, object]) -> None:
    _write(tmp_path, room109_document())
    path = tmp_path / "designs" / "room109.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps({"schema_version": "aosr.zone_design.v1", "scheme_id": "room109", **settings}), encoding="utf-8")


def _first_cut(page: Page, tmp_path: Path) -> tuple[dict[str, dict[str, object]], list[str]]:
    """x 起點牆不鏡像：左下畫一框 → z1、右邊一大塊 auto1、z1 上面 auto2（三塊都還是原牆）；三塊各選一種做法、由程式最佳化那塊改清單。"""
    _drag(page, LY, LZ, (0.02, 0.02), (1.0, 1.2))
    assert sorted(str(zone["id"]) for zone in _zones(page, "x0")) == ["auto1", "auto2", "z1"]
    wall = _zones(page, "x0")[0]["material"]
    assert all(zone["material"] == wall for zone in _zones(page, "x0"))
    choose_fixed(page, "x0", "z1", PRESET)
    choose_mode(page, "x0", "auto1", "search")
    choose_mode(page, "x0", "auto2", "keep_original")
    # 改成由程式最佳化就帶好一份清單（不用先全勾再取消）；進小框拿掉一種、加一種。
    assert _modes(page, "x0")["auto1"] == {"mode": "search", "allowed": DEFAULT_ALLOWED}
    open_panel(page, "x0", "auto1")
    # 48K 固定成品在可調 48K 已列入時不能新增。先拿掉，再把可調那筆加回。
    for wool in ("glass_wool_96k", "glass_wool_48k"):
        page.locator(f"#zone-panel-remove-{wool}").click()
    page.locator("#zone-panel-add").click()
    expect(page.locator("#zone-panel-pick-carpet_on_concrete")).to_have_count(0)
    page.locator(f"#zone-panel-pick-{WALL_FIXED}").click()
    page.locator("#zone-panel-add").click()
    page.locator("#zone-panel-pick-glass_wool_48k").click()
    # 舊做法表底下那一行（#zone-note）講的那一句，表拆了改看那一頁結果列上面那一行（#zone-page-note），回輸入頁前讀。
    apply_and_return(page, said="1 區改成「由程式最佳化」")
    # 用號碼與位置名叫那一區（#1176）：109 號房的兩支喇叭都投影在右邊那一大塊。
    expect(page.locator("#zone-goto-x0-auto1")).to_have_attribute("aria-label", "到「這間房貼什麼」改前牆 ③「兩支喇叭後面那塊」")
    reveal(page, "#original-scheme").select_option("room109-original")  # #1265 起在第 1 段「指定一份方案當改造前」裡
    expect(page.locator("#elevation [data-zone-id=auto1]")).to_have_class("zone zone-auto zone-mode-search")

    zones = {str(zone["id"]): zone for zone in _zones(page, "x0")}
    assert zones["z1"]["material"] == {"kind": "mounted_on_wall", "layers": _layers(PRESET), "wall": wall}
    assert zones["auto1"]["material"] == wall and zones["auto2"]["material"] == wall
    # 預設可調材料在前、不貼在後；之後勾的一種接在尾巴（存著的順序原樣帶回，不照選單重排）。
    allowed = ["glass_wool_48k", "glass_wool_24k", "none", WALL_FIXED]
    assert _modes(page, "x0") == {"z1": {"mode": "fixed"}, "auto1": {"mode": "search", "allowed": allowed},
                                  "auto2": {"mode": "keep_original"}}
    page.locator("#elevation-panel").screenshot(path=str(tmp_path / "modes-first-cut.png"))
    return zones, allowed


def test_each_zone_picks_a_mode_survives_a_recut_and_a_reopen(tmp_path: Path, browser: Browser) -> None:
    _write_room109_design(tmp_path, {"left_right_symmetry": "off"})  # 這一題不考左右對稱，明寫關著
    _write(tmp_path, room109_document("room109-original"))
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-x0").click()
        expect_mode(page, "x0", "whole_wall", "fixed")

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
        expect_mode(page, "x0", right, "search")
        # 清單在小框裡一種一列（舊做法表那一格的方塊跟著表拆了）；在小框加減候選。
        candidates = listed(page, "x0", right)
        assert len(candidates) == 4 and "glass_wool_96k" not in candidates and WALL_FIXED in candidates, candidates
        open_panel(page, "x0", right)
        expect(page.locator(f"#zone-panel-row-{WALL_FIXED}")).to_contain_text("48K 玻璃棉 50 mm＋100 mm 空腔")
        page.locator("#zone-page-close").click()
        assert fixed_value(page, "x0", "z1") == PRESET
        page.locator("#elevation-panel").screenshot(path=str(tmp_path / "modes-reopened.png"))

        # 清單拿到只剩固定材料：照實提醒（程式沒得選），開跑時伺服器再擋一次。
        open_panel(page, "x0", right)
        for choice in ("glass_wool_48k", "glass_wool_24k", "none"):
            page.locator(f"#zone-panel-remove-{choice}").click()
        expect(page.locator("#zone-panel-problem")).to_contain_text("只放了固定成品，程式沒得選")
        expect(page.locator("#zone-panel-apply")).to_be_disabled()
        assert watched.page_errors == [] and watched.console_errors == []


def test_old_scheme_opens_with_every_zone_fixed_and_saving_writes_no_design(tmp_path: Path, browser: Browser) -> None:
    _write(tmp_path, room109_document("plain"))
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, "plain")
        for wall in ("floor", "x0", "yL"):
            page.locator(f"#elevation-wall-{wall}").click()
            expect_mode(page, wall, "whole_wall", "fixed")
        with page.expect_response(lambda r: r.url.endswith("/api/schemes/plain") and r.request.method == "PUT") as saved:
            page.locator("#save").click()
        assert saved.value.ok
        expect(page.locator("#messages")).to_have_class("ok")
        assert not (tmp_path / "designs" / "plain.json").exists()
        assert watched.page_errors == [] and watched.console_errors == []


def test_zone_number_stays_inside_a_narrow_strip(tmp_path: Path, browser: Browser) -> None:
    """窄條裡的號碼圓（#1176 起不寫「自動」，畫號碼）放得下才畫、畫了就整個在那一條裡面，不擠出界壓到隔壁區；
    滑過去看 title 照樣有號碼、位置名與代號。"""
    _write(tmp_path, room109_document())
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-x0").click()
        # 框離右牆 25 公分：右邊補出一條 25 公分寬的窄條（過最小邊、但寫不下兩個字）。
        _drag(page, LY, LZ, (0.02, 0.02), (LY - 0.25, 1.2))
        strip = next(str(zone["id"]) for zone in _zones(page, "x0")
                     if cast(list[float], zone["u_m"])[0] > LY - 0.3)
        # 先等立面圖畫好這一條（title 有位置名），再在頁面裡一次量完每一區的框與號碼圓：分開量的話中間重畫一次，
        # 量到的是被換掉的舊節點（本機紅過一次：框量出 None）。
        expect(page.locator(f"#elevation [data-zone-id={strip}] title")).to_contain_text(f"靠右牆那條（代號 {strip}）")
        measured = cast(dict[str, dict[str, list[list[float]]]], page.evaluate(
            """() => {
              const edges = node => { const box = node.getBoundingClientRect(); return [box.left, box.right, box.top, box.bottom]; };
              return Object.fromEntries([...document.querySelectorAll("#elevation [data-zone-id]")].map(zone => [
                zone.dataset.zoneId,
                {rect: [edges(zone.querySelector("rect"))], circles: [...zone.querySelectorAll(".zone-number circle")].map(edges)}]));
            }"""))
        assert strip in measured, measured
        # 每一區最多一個號碼圓、畫了就整個在那一區的框裡（窄條這一格這裡被喇叭的字擋住、沒畫，照樣算）；
        # 整面上至少量到一個圓，免得這一題一個都沒量到也綠。
        assert sum(len(zone["circles"]) for zone in measured.values()) >= 1, measured
        for key, zone in measured.items():
            [[left, right, top, bottom]] = zone["rect"]
            assert len(zone["circles"]) <= 1, (key, zone)
            for c_left, c_right, c_top, c_bottom in zone["circles"]:
                assert left <= c_left and c_right <= right and top <= c_top and c_bottom <= bottom, (key, zone)
        assert watched.page_errors == [] and watched.console_errors == []


def test_zone_number_is_left_out_when_the_strip_is_narrower_than_the_circle(tmp_path: Path, browser: Browser) -> None:
    """窄條比號碼圓（直徑加 2 像素）窄就不畫圓，title 照樣有號碼與名字。窄條放在假房間左牆離喇叭與主位最遠的那一端
    （喇叭、主位離它 2 公尺以上），不讓「被字擋住」頂替「太窄」（上一題的窄條就是被喇叭的字擋住才沒畫）；
    先量窄條真的比圓窄，量不到就紅，不靜靜綠。"""
    document = json.loads(json.dumps(ZONE_NAMING_SCHEME))
    whole = document["scene"]["surface_zones_by_wall"]["x0"][0]["material"]
    height, length = document["scene"]["room_m"]["Lz"], document["scene"]["room_m"]["Ly"]
    document["scene"]["surface_zones_by_wall"]["x0"] = [
        {"area": "rectangle", "id": "z1", "u_m": [0.2, length], "v_m": [0.0, height], "material": whole},
        {"area": "rectangle", "id": "auto1", "u_m": [0.0, 0.2], "v_m": [0.0, height], "material": whole}]
    _write(tmp_path, document)
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1000) as watched:
        page = watched.page
        _open_scheme(page, str(document["scheme_id"]))
        page.locator("#elevation-wall-x0").click()
        expect(page.locator("#elevation [data-zone-id=auto1] title")).to_contain_text("代號 auto1")
        measured = cast(dict[str, float], page.evaluate("""() => {
          const strip = document.querySelector("#elevation [data-zone-id=auto1]");
          const big = document.querySelector("#elevation [data-zone-id=z1] .zone-number circle");
          return {width: strip.querySelector("rect").getBoundingClientRect().width, circles: strip.querySelectorAll(".zone-number").length,
                  diameter: big ? big.getBoundingClientRect().width : -1};
        }"""))
        assert measured["diameter"] > 0 and measured["width"] < measured["diameter"] + 2, measured
        assert measured["circles"] == 0, measured
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
        # 讀不到那一句：舊做法表（#elevation-zones）拆了，同一支 zoneProblemLines 寫在第 2 段右欄清單上方。
        expect(page.locator("#elevation-list-notes")).to_contain_text("不會存回去")
        # 換做法改從小框（舊做法表那一路打 /api/zone-material，表拆了；小框打 /api/zone-material/batch）。
        page.route("**/api/zone-material/batch", lambda route: route.abort())
        open_alone(page, "x0", "whole_wall")
        page.locator("#zone-panel-mode-search").check()
        page.locator("#zone-panel-apply").click()
        page.wait_for_function("() => !document.getElementById('zone-page').inert && !inputBusy")
        # 改字：舊做法表那一行寫「沒有換」，小框那一路的回話是「沒有改：…」（zone_overview.js 的 applyMethod）。
        expect(page.locator("#zone-page-note")).to_contain_text("沒有改")
        page.locator("#zone-page-close").click()
        expect_mode(page, "x0", "whole_wall", "fixed")
        assert page.evaluate("zoneDesign") is None
        page.unroute("**/api/zone-material/batch")
        choose_mode(page, "x0", "whole_wall", "search")
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
        # 先開著另一份存好的（範例，沒有做法檔）：等待中表單、方案名字與做法都還是那一份的。
        open_example_scheme(page, base, tmp_path)
        held: list[Route] = []
        page.route("**/api/designs/room109", lambda route: held.append(route))
        open_button = scheme_open_button(page, "room109")
        with page.expect_request("**/api/designs/room109"):
            open_button.click()
        page.wait_for_function("document.querySelector('main').inert")
        assert page.locator("#save-id").input_value() == "example"
        assert page.evaluate("[zoneDesign.scheme_id, scheme.scheme_id, openedId]") == ["example", "example", "example"]
        held[0].continue_()
        page.wait_for_function("document.getElementById('save-id').value === 'room109'")
        assert page.evaluate("[zoneDesign.scheme_id, zoneDesign.original_scheme_id]") == ["room109", "room109-original"]
        page.wait_for_function("!document.querySelector('main').inert")
        page.locator("#elevation-wall-x0").click()
        expect_mode(page, "x0", "whole_wall", "search")
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
        expect(page.locator("#zone-to-search")).to_be_visible()
        expect(page.locator("#zone-to-search")).to_be_disabled()
        expect(page.locator("#zone-to-search-note")).to_contain_text("這間房貼什麼")
        choose_mode(page, "x0", "whole_wall", "search")
        expect(page.locator("#zone-to-search-note")).to_contain_text("交出黃色的 1 區")
        page.locator("#elevation-panel").screenshot(path=str(tmp_path / "to-search.png"))
        # 改回不交給程式（舊做法表選「指定貼一種」不換材料；小框指定貼一種要選一筆，改選「不貼，保留原樣」，一樣沒有交給程式的區）。
        choose_mode(page, "x0", "whole_wall", "keep_original")
        expect(page.locator("#zone-to-search")).to_be_visible()
        expect(page.locator("#zone-to-search")).to_be_disabled()
        expect(page.locator("#zone-to-search-note")).to_contain_text("這間房貼什麼")
        choose_mode(page, "x0", "whole_wall", "search")

        page.route("**/api/designs/room109", lambda route: route.abort() if route.request.method == "PUT"
                   else route.continue_())
        page.locator("#zone-to-search").click()
        expect(page.locator("#zone-to-search-note")).to_contain_text("做法沒存成")  # 頁底那顆鈕的回話（searchNote）
        assert page.url == f"{base}/"
        assert not (tmp_path / "designs" / "room109.json").exists()
        page.unroute("**/api/designs/room109")

        # 只考跳到哪一個網址：搜尋頁一打開就向後端要開跑計畫（重計算），它自己有考卷（#1108），這裡用一頁空殼接住。
        page.route(f"{base}/gradient?scheme=room109",
                   lambda route: route.fulfill(status=200, content_type="text/plain", body="gradient"))
        page.locator("#zone-to-search").click()
        page.wait_for_url(f"{base}/gradient?scheme=room109")
        stored = json.loads((tmp_path / "designs" / "room109.json").read_text(encoding="utf-8"))
        assert stored["zones"]["x0"] == {"whole_wall": {"mode": "search", "allowed": DEFAULT_ALLOWED}}
        assert watched.page_errors == []


def test_opening_another_scheme_while_send_to_search_saves_is_refused_and_the_jump_goes_to_the_saved_one(
        tmp_path: Path, browser: Browser) -> None:
    """按「拿去搜尋」存檔途中（設計設定的回覆還沒到）去按「打開」另一份：整頁鎖著、打開被擋下照講（#1187，
    之前是放它換、晚到的回覆再自己作廢）；放行後存的與跳去的都是按下那一刻那一份。"""
    _write(tmp_path, room109_document())
    _write(tmp_path, room109_document("other"))
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-x0").click()
        choose_mode(page, "x0", "whole_wall", "search")
        open_other = scheme_open_button(page, "other")
        held: list[Route] = []
        page.route("**/api/designs/room109",
                   lambda route: held.append(route) if route.request.method == "PUT" else route.continue_())
        # 只考跳到哪一個網址（同上一題）：搜尋頁用一頁空殼接住。
        page.route(f"{base}/gradient?scheme=room109",
                   lambda route: route.fulfill(status=200, content_type="text/plain", body="gradient"))
        with page.expect_request(lambda r: r.url.endswith("/api/designs/room109") and r.method == "PUT"):
            page.locator("#zone-to-search").click()
        assert page.evaluate("document.querySelector('main').inert") is True
        # 鎖頁擋得住滑鼠，擋不住頁面程式自己發的點擊：程式裡那道鎖也要擋。
        open_other.evaluate("node => node.click()")
        expect(page.locator("#open-note")).to_have_text("上一個動作還沒做完，沒有打開「other」；等它做完再按一次「打開」")
        assert page.evaluate("[scheme.scheme_id, zoneDesign.zones.x0.whole_wall.mode]") == ["room109", "search"]
        assert page.locator("#save-id").input_value() == "room109"
        assert watched.page_errors == []
        held[0].continue_()
        page.wait_for_url(f"{base}/gradient?scheme=room109")
        stored = json.loads((tmp_path / "designs" / "room109.json").read_text(encoding="utf-8"))
        assert stored["zones"]["x0"] == {"whole_wall": {"mode": "search", "allowed": DEFAULT_ALLOWED}}
        assert not (tmp_path / "designs" / "other.json").exists()


def test_another_scheme_cannot_be_opened_between_the_save_of_send_to_search_and_the_jump(
        tmp_path: Path, browser: Browser) -> None:
    """「拿去搜尋」（舊做法表那一顆）存好、結果清單還在重讀的那一段，頁面還鎖著（#1188 第 5 條；#1187 那時鎖已放開、
    真人點得到「打開」）：這時連頁面程式自己發的「打開」也被擋下、旁邊照講，表單還是原本那一份。
    清單回來後照常跳到原本那一份的搜尋頁，它的做法存好了，另一份沒被寫到。"""
    _write(tmp_path, room109_document())
    _write(tmp_path, room109_document("other"))
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-x0").click()
        choose_mode(page, "x0", "whole_wall", "search")
        held: list[Route] = []
        page.route("**/api/results",
                   lambda route: held.append(route) if route.request.method == "GET" and not held else route.continue_())
        page.route("**/gradient?scheme=*",
                   lambda route: route.fulfill(status=200, content_type="text/plain", body="gradient"))
        opened: list[str] = []
        page.on("request", lambda request: opened.append(request.url) if "/api/schemes/other" in request.url else None)
        with page.expect_request("**/api/results"):
            page.locator("#zone-to-search").click()
        assert page.evaluate("[document.querySelector('main').inert, inputBusy]") == [True, True]
        scripted_open_scheme(page, "other")
        expect(page.locator("#open-note")).to_have_text("上一個動作還沒做完，沒有打開「other」；等它做完再按一次「打開」")
        assert page.evaluate("[scheme.scheme_id, document.querySelector('main').inert]") == ["room109", True]
        held[0].continue_()
        page.wait_for_url("**/gradient?scheme=room109")
        assert opened == [] and watched.page_errors == []
    stored = json.loads((tmp_path / "designs" / "room109.json").read_text(encoding="utf-8"))
    assert stored["zones"]["x0"] == {"whole_wall": {"mode": "search", "allowed": DEFAULT_ALLOWED}}
    assert not (tmp_path / "designs" / "other.json").exists()


def test_stored_allowed_list_comes_back_as_is_through_open_edit_and_save(tmp_path: Path, browser: Browser) -> None:
    """2-3b 交接：存著的清單（ZoneDesign.allowed）打開再存要原樣帶回——順序不動、材料庫找不到的代號不偷丟
    （照講、留著，開跑時伺服器點名，進小框才拿掉）；勾一種只加在尾巴、拿掉一種只拿那一種；最後一種拿不掉。"""
    _write(tmp_path, room109_document())
    stored = ["none", "ghost_material", "glass_wool_48k"]
    design = tmp_path / "designs" / "room109.json"
    design.parent.mkdir(exist_ok=True)
    design.write_text(json.dumps({"schema_version": "aosr.zone_design.v1", "scheme_id": "room109",
                                  "zones": {"x0": {"whole_wall": {"mode": "search", "allowed": stored}}}}),
                      encoding="utf-8")
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-x0").click()
        # 舊表三個方塊 → 小框三列（小框把「不貼」排最後；存的順序由底下存檔那幾行考）。
        candidates = listed(page, "x0", "whole_wall")
        assert len(candidates) == 3 and set(candidates) == set(stored), candidates
        open_panel(page, "x0", "whole_wall")
        expect(page.locator("#zone-panel-row-ghost_material")).to_contain_text("材料庫找不到")
        page.locator("#zone-page-close").click()
        # 舊表底下「清單裡有材料庫找不到的：ghost_material…」那一行 → 第 2 段右欄那一區那一句點名它。
        expect(page.locator("#wall-zone-what-x0-whole_wall")).to_contain_text("ghost_material")
        _save(page, "room109")
        assert json.loads(design.read_text(encoding="utf-8"))["zones"]["x0"]["whole_wall"]["allowed"] == stored

        edit_candidates(page, "x0", "whole_wall", remove=("glass_wool_48k",), add=("glass_wool_24k",))
        assert _modes(page, "x0")["whole_wall"]["allowed"] == ["none", "ghost_material", "glass_wool_24k"]
        _save(page, "room109")
        saved = json.loads(design.read_text(encoding="utf-8"))["zones"]["x0"]["whole_wall"]
        assert saved == {"mode": "search", "allowed": ["none", "ghost_material", "glass_wool_24k"]}

        # 找不到的那一種按 × 拿掉、只拿它；拿到只剩一種，最後一種拿不掉（空清單伺服器整份拒收）：勾還在、清單不動、照講。
        edit_candidates(page, "x0", "whole_wall", remove=("ghost_material",))
        assert _modes(page, "x0")["whole_wall"]["allowed"] == ["none", "glass_wool_24k"]
        expect(page.locator("#wall-zone-what-x0-whole_wall")).not_to_contain_text("ghost_material")
        edit_candidates(page, "x0", "whole_wall", remove=("glass_wool_24k",))
        assert _modes(page, "x0")["whole_wall"]["allowed"] == ["none"]
        open_panel(page, "x0", "whole_wall")
        page.locator("#zone-panel-remove-none").click()
        expect(page.locator("#zone-panel-problem")).to_contain_text("至少勾一種材料")
        expect(page.locator("#zone-panel-apply")).to_be_disabled()
        page.locator("#zone-panel-cancel").click()
        page.locator("#zone-page-close").click()
        assert listed(page, "x0", "whole_wall") == ["none"]
        assert _modes(page, "x0")["whole_wall"]["allowed"] == ["none"]
        _save(page, "room109")
        assert json.loads(design.read_text(encoding="utf-8"))["zones"]["x0"]["whole_wall"]["allowed"] == ["none"]
        assert watched.page_errors == [] and watched.console_errors == []


def test_search_zone_says_what_the_scheme_really_holds(tmp_path: Path, browser: Browser) -> None:
    """#1128：交給搜尋的區寫方案裡現在真的放什麼。網頁改成交給搜尋時放原牆（寫原牆、會貼在它上面）；搜尋存成的方案
    （或它的另存）那一區放的是搜尋選的貼牆構造，不准還寫「先放原牆」。這裡把貼了一層的 z1 在設計設定改成交給搜尋，
    就是存成方案那一份的形狀。"""
    _write_room109_design(tmp_path, {"left_right_symmetry": "off"})  # 這一題不考左右對稱，明寫關著
    _write(tmp_path, room109_document("room109-original"))
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-x0").click()
        _first_cut(page, tmp_path)
        _save(page, "room109")
        # #1176：原牆原樣寫白話，阻抗數字收在點開的「數字」裡（第 2 段「每一區的位置與大小」那一格）。
        # 舊做法表那一格（#zone-search-now-）拆了，改讀右欄那一句（改字：「現在：沒貼東西（原牆原樣），…貼在它上面」→
        # 「現在存的是「沒貼東西（原牆原樣）」，直接算就照它算」）。
        what = page.locator("#wall-zone-what-x0-auto1")
        expect(what).to_contain_text("現在存的是「沒貼東西（原牆原樣）」")
        expect(what).not_to_contain_text("Pa·s/m")
        expect(page.locator("#zone-numbers-x0-auto1")).to_contain_text("原牆：固定阻抗")
        path = tmp_path / "designs" / "room109.json"
        stored = json.loads(path.read_text(encoding="utf-8"))
        stored["zones"]["x0"]["z1"] = {"mode": "search"}
        path.write_text(json.dumps(stored), encoding="utf-8")
        page.reload()
        page.wait_for_function("typeof inputDefaults !== 'undefined' && inputDefaults !== undefined")
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-x0").click()
        expect_mode(page, "x0", "z1", "search")
        line = page.locator("#wall-zone-what-x0-z1")  # 改字：「現在：已貼…」→「現在存的是「已貼…」」
        expect(line).to_contain_text("現在存的是「已貼 48K 玻璃棉 50 mm 貼牆（貼在原牆上）」")
        expect(line).not_to_contain_text("先放原牆")
        expect(line).not_to_contain_text("沒貼東西")
        page.locator("#elevation-panel").screenshot(path=str(tmp_path / "search-zone-holds.png"))
        assert watched.page_errors == [] and watched.console_errors == []


def test_with_no_wall_picked_the_footer_button_follows_the_limits_summary(tmp_path: Path, browser: Browser) -> None:
    """頁底「交給程式最佳化 ▸」講整份方案（#1265 單元 3 之二收尾追加第 1 條）：還沒點任何一面牆時，它跟右欄上方的限制總結
    一起重畫——開檔就有紅，鈕要鎖、底下寫超過限制；把房間預設放寬、紅解掉，鈕要開、那一句換成交出幾區。
    先在一頁把後牆整面改成指定貼一種、存起來（貼上去 5 公分，房間預設 2 公分 → 紅），再開一頁重開那一份、不點牆。"""
    _write_room109_design(tmp_path, {"left_right_symmetry": "off", "room_max_depth_m": 0.02,
                                     "zones": {"x0": {"whole_wall": {"mode": "search", "allowed": ["glass_wool_48k", "none"]}},
                                               "xL": {"whole_wall": {"mode": "fixed"}}}})
    with _serve(tmp_path) as base:
        with _open(browser, base, viewport_width=1440) as watched:
            page = watched.page
            _open_scheme(page, "room109")
            page.locator("#elevation-wall-xL").click()
            choose_fixed(page, "xL", "whole_wall", PRESET)
            page.locator("#save").click()
            page.wait_for_function("!document.querySelector('main').inert && !inputBusy")
            expect(page.locator("#messages")).to_have_class("ok")
        with _open(browser, base, viewport_width=1440) as watched:
            page = watched.page
            _open_scheme(page, "room109")
            page.wait_for_function("zoneLimitsGeneration > 0 && zoneLimitsShown === zoneLimitsGeneration")
            # 一面牆都沒點：#1265 單元 5c 起打開就選好前牆（x 起點牆），沒有「一面都沒選」的時候；頁底那顆照樣講整份方案。
            expect(page.locator("#elevation-walls button[aria-pressed=true]")).to_have_count(1)
            expect(page.locator("#elevation-wall-x0")).to_have_attribute("aria-pressed", "true")
            expect(page.locator("#zone-limits-summary")).to_contain_text("超過限制")
            expect(page.locator("#zone-to-search")).to_be_disabled()
            expect(page.locator("#zone-to-search-note")).to_contain_text("超過限制")
            page.locator("#edit-depth").click()
            page.locator("#depth-room").fill("20")
            page.locator("#depth-room").press("Tab")
            page.wait_for_function("zoneLimitsShown === zoneLimitsGeneration")
            expect(page.locator("#zone-limits-summary")).to_have_text("限制都沒超過")
            expect(page.locator("#zone-to-search")).to_be_enabled()
            expect(page.locator("#zone-to-search-note")).to_have_text("交出黃色的 1 區；下一頁按「開始搜尋」才開始算")
            assert watched.page_errors == [] and watched.console_errors == []


def test_a_footer_reply_is_dropped_once_a_later_verdict_comes_back(tmp_path: Path, browser: Browser) -> None:
    """頁底那一句（searchNote 寫的，#1265 單元 3 之二收尾追加第 2 條）：限制檢查斷了時按「交給程式最佳化 ▸」，底下寫沒檢查成；
    恢復後按「檢查」、新判決回來，那一句要收掉，照現在的樣子寫交出幾區。"""
    _write_room109_design(tmp_path, {"left_right_symmetry": "off",
                                     "zones": {"x0": {"whole_wall": {"mode": "search", "allowed": ["glass_wool_48k", "none"]}}}})
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-x0").click()
        page.wait_for_function("zoneLimitsShown === zoneLimitsGeneration")
        page.route("**/api/zone-limits", lambda route: route.abort())
        page.locator("#check").click()
        page.wait_for_function("zoneLimitsShown === zoneLimitsGeneration && zoneLimits.status === 'error'")
        page.locator("#zone-to-search").click()
        note = page.locator("#zone-to-search-note")
        expect(note).to_contain_text("限制沒檢查成")
        page.unroute("**/api/zone-limits")
        page.locator("#check").click()
        page.wait_for_function("zoneLimitsShown === zoneLimitsGeneration && zoneLimits.status === 'checked'")
        expect(page.locator("#zone-limits-summary")).to_have_text("限制都沒超過")
        expect(note).to_have_text("交出黃色的 1 區；下一頁按「開始搜尋」才開始算")
        assert watched.page_errors == []


def test_a_footer_reply_written_after_the_save_survives_the_verdict_that_save_asked_for(tmp_path: Path, browser: Browser) -> None:
    """同一條的另一半：「拿去搜尋」自己存檔時就問了一次限制判決，做法沒存成那一句寫在那一次問了之後；那一次判決晚一點回來，
    那一句不准被收掉（收的是之後又問的判決，不是按下之後的第一份回覆）。"""
    _write_room109_design(tmp_path, {"left_right_symmetry": "off",
                                     "zones": {"x0": {"whole_wall": {"mode": "search", "allowed": ["glass_wool_48k", "none"]}}}})
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-x0").click()
        page.wait_for_function("zoneLimitsShown === zoneLimitsGeneration")
        held: list[Route] = []
        page.route("**/api/zone-limits", lambda route: held.append(route))
        page.route("**/api/designs/**", lambda route: route.fulfill(status=500, content_type="application/json",
                                                                    body=json.dumps({"error": "考卷故意讓它壞"}))
                   if route.request.method == "PUT" else route.continue_())
        page.locator("#zone-to-search").click()
        note = page.locator("#zone-to-search-note")
        expect(note).to_have_text("每一區的做法沒存成，搜尋頁會讀到舊的做法：先處理好訊息列寫的問題再按一次")
        assert held, "存檔時問的那一次限制判決應該還在路上"
        page.unroute("**/api/zone-limits")
        for route in held:
            route.continue_()
        page.wait_for_function("zoneLimitsShown === zoneLimitsGeneration")
        expect(note).to_have_text("每一區的做法沒存成，搜尋頁會讀到舊的做法：先處理好訊息列寫的問題再按一次")
        assert watched.page_errors == []


SAVE_FAILED = "每一區的做法沒存成，搜尋頁會讀到舊的做法：先處理好訊息列寫的問題再按一次"
HANDED = "交出黃色的 1 區；下一頁按「開始搜尋」才開始算"


def test_a_save_failure_reply_outlives_the_relabel_after_unlock_and_goes_once_saved(tmp_path: Path, browser: Browser) -> None:
    """#1265 單元 3 之二收尾第二趟 B1：改了喇叭座標（牆名會跟著變）就立刻按「交給程式最佳化 ▸」、做法存檔失敗。
    解鎖後自動補問的預覽回來、換牆名、又問了一次限制判決——那一次不是存檔的結果，頁底「做法沒存成」那一句不准被它收掉。
    那一句講的是存檔：下一次做法存成了（按「儲存」）才收。"""
    _write_room109_design(tmp_path, {"left_right_symmetry": "off",
                                     "zones": {"x0": {"whole_wall": {"mode": "search", "allowed": ["glass_wool_48k", "none"]}}}})
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-x0").click()
        page.wait_for_function("zoneLimitsShown === zoneLimitsGeneration")
        # 打字那兩下的預覽扣住（改完座標就按，預覽還沒回來）；第三次是解鎖後自動補問的，放它回來換牆名、重問限制。
        held: list[Route] = []
        page.route("**/api/input-preview", lambda route: held.append(route) if len(held) < 2 else route.continue_())
        page.route("**/api/designs/**", lambda route: route.fulfill(status=500, content_type="application/json",
                                                                    body=json.dumps({"error": "考卷故意讓它壞"}))
                   if route.request.method == "PUT" else route.continue_())
        page.locator("#coordinates-fold summary").click()
        page.locator("#speaker-left-x").fill("4.2")
        page.locator("#speaker-right-x").fill("4.2")
        before = cast(int, page.evaluate("zoneLimitsGeneration"))
        page.locator("#zone-to-search").click()
        page.wait_for_function(f"zoneLimitsGeneration >= {before + 2} && zoneLimitsShown === zoneLimitsGeneration")
        expect(page.locator("#messages")).to_contain_text("每一區的做法沒存成")
        note = page.locator("#zone-to-search-note")
        expect(note).to_have_text(SAVE_FAILED)
        assert page.evaluate("zoneSearchSaid") is not None
        _release(held)
        page.unroute("**/api/input-preview")
        page.unroute("**/api/designs/**")
        with page.expect_response(lambda r: r.url.endswith("/api/designs/room109") and r.request.method == "PUT") as design:
            page.locator("#save").click()
        assert design.value.ok, design.value.text()
        page.wait_for_function("!document.querySelector('main').inert && !inputBusy && zoneLimitsShown === zoneLimitsGeneration")
        expect(note).to_have_text(HANDED)
        assert watched.page_errors == []


def test_the_zone_page_reply_goes_with_the_footer_reply_once_a_later_verdict_comes_back(tmp_path: Path, browser: Browser) -> None:
    """#1265 單元 3 之二收尾第二趟 D2：在「這間房貼什麼」按「拿去搜尋」、限制沒檢查成，那一頁寫原因；回輸入頁恢復連線按「檢查」，
    新判決回來，頁底那一句收掉——那一頁的回話（#zone-page-note）是同一句，要一起收，不准再打開那一頁時還掛著。"""
    _write_room109_design(tmp_path, {"left_right_symmetry": "off",
                                     "zones": {"x0": {"whole_wall": {"mode": "search", "allowed": ["glass_wool_48k", "none"]}}}})
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-x0").click()
        page.wait_for_function("zoneLimitsShown === zoneLimitsGeneration")
        page.route("**/api/zone-limits", lambda route: route.abort())
        page.locator("#check").click()
        page.wait_for_function("zoneLimitsShown === zoneLimitsGeneration && zoneLimits.status === 'error'")
        page.locator("#zone-page-open").click()
        page.locator("#zone-page-search").click()
        expect(page.locator("#zone-page-note")).to_contain_text("限制沒檢查成")
        page.locator("#zone-page-close").click()
        page.unroute("**/api/zone-limits")
        page.locator("#check").click()
        page.wait_for_function("zoneLimitsShown === zoneLimitsGeneration && zoneLimits.status === 'checked'")
        expect(page.locator("#zone-to-search-note")).to_have_text(HANDED)
        page.locator("#zone-page-open").click()
        expect(page.locator("#zone-page-note")).to_have_text("")
        assert watched.page_errors == []


NO_SEARCH_ZONE = "到「這間房貼什麼」設定要交給程式的區"


def test_a_save_failure_reply_goes_also_when_the_save_writes_no_design_file(tmp_path: Path, browser: Browser) -> None:
    """#1265 單元 3 之二收尾第三趟 E1：沒存過設計檔的方案，按「交給程式最佳化 ▸」時做法存檔失敗；再清掉那一面的切法，
    設計設定空了。這時頁底鈕沒有要交的區、已經停用，底下不准再寫「…再按一次」（審查設計洞 1）。
    接著按「儲存」成功——沒內容也沒存過就不寫檔，這一次也算做法存好了：頁底與「這間房貼什麼」那一頁的那一句都要收。"""
    _write(tmp_path, room109_document())
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-x0").click()
        page.wait_for_function("zoneLimitsShown === zoneLimitsGeneration")
        _drag(page, LY, LZ, (0.02, 0.02), (0.8, 1.0))
        page.wait_for_function("zoneLimitsShown === zoneLimitsGeneration")
        page.locator("#zone-goto-x0-z1").click()
        page.locator("#zone-panel").get_by_role("radio", name="由程式最佳化").check()
        page.locator("#zone-panel-apply").click()
        page.locator("#zone-page-close").click()
        page.wait_for_function("zoneLimitsShown === zoneLimitsGeneration")
        page.route("**/api/designs/**", lambda route: route.fulfill(status=500, content_type="application/json",
                                                                    body=json.dumps({"error": "考卷故意讓它壞"}))
                   if route.request.method == "PUT" else route.continue_())
        page.locator("#zone-to-search").click()
        page.wait_for_function("!document.querySelector('main').inert && !inputBusy && zoneLimitsShown === zoneLimitsGeneration")
        note = page.locator("#zone-to-search-note")
        expect(note).to_have_text(SAVE_FAILED)
        page.unroute("**/api/designs/**")
        page.locator("#elevation-clear").click()
        page.wait_for_function("!document.querySelector('main').inert && !inputBusy && zoneLimitsShown === zoneLimitsGeneration")
        assert page.evaluate("searchZoneCount()") == 0
        expect(page.locator("#zone-to-search")).to_be_disabled()
        expect(note).to_have_text(NO_SEARCH_ZONE)
        with page.expect_response(lambda r: r.url.endswith("/api/schemes/room109") and r.request.method == "PUT") as stored:
            page.locator("#save").click()
        assert stored.value.ok, stored.value.text()
        page.wait_for_function("!document.querySelector('main').inert && !inputBusy && zoneLimitsShown === zoneLimitsGeneration")
        expect(page.locator("#messages")).to_have_text("方案已儲存")
        assert not (tmp_path / "designs" / "room109.json").exists()
        assert page.evaluate("zoneSearchSaid") is None
        expect(note).to_have_text(NO_SEARCH_ZONE)
        page.locator("#zone-page-open").click()
        expect(page.locator("#zone-page-note")).to_have_text("")
        assert watched.page_errors == []
