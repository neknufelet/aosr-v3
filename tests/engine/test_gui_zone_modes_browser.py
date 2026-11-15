"""真瀏覽器考每一區的做法（#1105，#1082 第 1-3 段）：立面圖下的逐區表選指定貼一種／不貼／交給電腦挑並改電腦可以挑的清單
（#1176 起改成交給電腦挑就帶好一份明確清單，存檔也寫明確清單）；再切同一面別處，沒動到的區做法照留（跟著自動補滿重編的代號走）；
存檔再打開，切法、做法、清單、原方案都在；沒設過做法的舊方案存檔不多寫一份設計設定。房間一律用 109 號房的長寬高。"""
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
# 加進清單尾巴的那一筆固定成品（#1176 起清單只列這一面能貼的：地毯只列在地板，牆上改用一筆牆面的固定成品）。
WALL_FIXED = "glass_wool_48k_50mm_gap100"
# 改成交給電腦挑時一開始放的清單：一律是三種內建玻璃棉＋不貼（材料庫多了別的可調材料也不放進來）。
DEFAULT_ALLOWED = ["glass_wool_48k", "glass_wool_24k", "glass_wool_96k", "none"]


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
    """x 起點牆不鏡像：左下畫一框 → z1、右邊一大塊 auto1、z1 上面 auto2（三塊都還是原牆）；三塊各選一種做法、交給電腦挑那塊改清單。"""
    _drag(page, LY, LZ, (0.02, 0.02), (1.0, 1.2))
    assert sorted(str(zone["id"]) for zone in _zones(page, "x0")) == ["auto1", "auto2", "z1"]
    wall = _zones(page, "x0")[0]["material"]
    assert all(zone["material"] == wall for zone in _zones(page, "x0"))
    _choose(page, "#zone-fixed-x0-z1", PRESET)
    _choose(page, "#zone-mode-x0-auto1", "search")
    _choose(page, "#zone-mode-x0-auto2", "keep_original")
    # 改成交給電腦挑就帶好一份清單（不用先全勾再取消）；點「改清單」拿掉一種、加一種。
    assert _modes(page, "x0")["auto1"] == {"mode": "search", "allowed": DEFAULT_ALLOWED}
    page.locator("#zone-allowed-summary-x0-auto1").click()
    page.locator("#zone-allow-x0-auto1-glass_wool_96k").uncheck()
    expect(page.locator("#zone-allow-x0-auto1-carpet_on_concrete")).to_have_count(0)
    page.locator(f"#zone-allow-x0-auto1-{WALL_FIXED}").check()
    page.locator("#original-scheme").select_option("room109-original")
    expect(page.locator("#elevation [data-zone-id=auto1]")).to_have_class("zone zone-auto zone-mode-search")
    # 底部那一行寫最後一次改的：auto1 的清單（#1176 起改清單也更新那一行，不留 auto2 那一句）。
    # 訊息用號碼與位置名叫那一區（#1176）：109 號房的兩支喇叭都投影在右邊那一大塊。
    expect(page.locator("#zone-note")).to_contain_text("前牆 ③「兩支喇叭後面那塊」：清單改成 48K 玻璃棉、24K 玻璃棉、不貼、")

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
        # 清單一種一個方塊攤在外面；勾選表收在「改清單」後面，點開才看得到勾選格。
        expect(page.locator(f"#zone-allowed-chips-x0-{right} .zone-chip")).to_have_count(4)
        expect(page.locator(f"#zone-allowed-chips-x0-{right}")).to_contain_text("48K 玻璃棉 50 mm＋100 mm 空腔")
        expect(page.locator(f"#zone-allowed-summary-x0-{right}")).to_contain_text("現在 4 種")
        page.locator(f"#zone-allowed-summary-x0-{right}").click()
        expect(page.locator(f"#zone-allow-x0-{right}-glass_wool_96k")).not_to_be_checked()
        expect(page.locator(f"#zone-allow-x0-{right}-{WALL_FIXED}")).to_be_checked()
        expect(page.locator("#zone-fixed-x0-z1")).to_have_value(PRESET)
        page.locator("#elevation-panel").screenshot(path=str(tmp_path / "modes-reopened.png"))

        # 清單拿到只剩固定材料：照實提醒（電腦沒得挑），開跑時伺服器再擋一次。
        for choice in ("glass_wool_48k", "glass_wool_24k", "none"):
            page.locator(f"#zone-allow-x0-{right}-{choice}").uncheck()
        expect(page.locator(f"#zone-allowed-problem-x0-{right}")).to_contain_text("只放了固定成品，電腦沒得挑")
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
        expect(page.locator("#zone-to-search")).to_contain_text("1 區交給電腦挑")
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

        # 只考跳到哪一個網址：搜尋頁一打開就向後端要開跑計畫（重計算），它自己有考卷（#1108），這裡用一頁空殼接住。
        page.route(f"{base}/gradient?scheme=room109",
                   lambda route: route.fulfill(status=200, content_type="text/plain", body="gradient"))
        page.locator("#zone-to-search").click()
        page.wait_for_url(f"{base}/gradient?scheme=room109")
        stored = json.loads((tmp_path / "designs" / "room109.json").read_text(encoding="utf-8"))
        assert stored["zones"]["x0"] == {"whole_wall": {"mode": "search", "allowed": DEFAULT_ALLOWED}}
        assert watched.page_errors == []


def test_switching_scheme_while_send_to_search_saves_keeps_the_new_scheme_and_does_not_jump(tmp_path: Path,
                                                                                           browser: Browser) -> None:
    """按「拿去搜尋」存檔途中（設計設定的回覆還沒到）打開另一份：晚到的回覆不把舊方案的做法寫回頁面，也不跳頁；
    舊方案的做法照樣存好。"""
    _write(tmp_path, room109_document())
    _write(tmp_path, room109_document("other"))
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-x0").click()
        _choose(page, "#zone-mode-x0-whole_wall", "search")
        held: list[Route] = []
        page.route("**/api/designs/room109",
                   lambda route: held.append(route) if route.request.method == "PUT" else route.continue_())
        with page.expect_request(lambda r: r.url.endswith("/api/designs/room109") and r.method == "PUT"):
            page.locator("#zone-to-search").click()
        _open_scheme(page, "other")
        assert page.evaluate("[zoneDesign.scheme_id, zoneDesign.zones]") == ["other", {}]
        held[0].continue_()
        expect(page.locator("#zone-note")).to_contain_text("存檔途中換了方案")
        assert page.evaluate("[zoneDesign.scheme_id, zoneDesign.zones]") == ["other", {}]
        assert page.url == f"{base}/"
        assert page.locator("#save-id").input_value() == "other"
        stored = json.loads((tmp_path / "designs" / "room109.json").read_text(encoding="utf-8"))
        assert stored["zones"]["x0"] == {"whole_wall": {"mode": "search", "allowed": DEFAULT_ALLOWED}}
        assert watched.page_errors == []


def test_stored_allowed_list_comes_back_as_is_through_open_edit_and_save(tmp_path: Path, browser: Browser) -> None:
    """2-3b 交接：存著的清單（ZoneDesign.allowed）打開再存要原樣帶回——順序不動、材料庫找不到的代號不偷丟
    （照講、留著，開跑時伺服器點名，按它旁邊的 × 才拿掉）；勾一種只加在尾巴、拿掉一種只拿那一種；最後一種拿不掉。"""
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
        expect(page.locator("#zone-allowed-chips-x0-whole_wall .zone-chip")).to_have_count(3)
        expect(page.locator("#zone-chip-x0-whole_wall-ghost_material")).to_contain_text("材料庫找不到")
        expect(page.locator("#zone-allowed-missing-x0-whole_wall")).to_contain_text("ghost_material")
        _save(page, "room109")
        assert json.loads(design.read_text(encoding="utf-8"))["zones"]["x0"]["whole_wall"]["allowed"] == stored

        page.locator("#zone-allowed-summary-x0-whole_wall").click()
        page.locator("#zone-allow-x0-whole_wall-glass_wool_48k").uncheck()
        page.locator("#zone-allow-x0-whole_wall-glass_wool_24k").check()
        assert _modes(page, "x0")["whole_wall"]["allowed"] == ["none", "ghost_material", "glass_wool_24k"]
        _save(page, "room109")
        saved = json.loads(design.read_text(encoding="utf-8"))["zones"]["x0"]["whole_wall"]
        assert saved == {"mode": "search", "allowed": ["none", "ghost_material", "glass_wool_24k"]}

        # 找不到的那一種按 × 拿掉、只拿它；拿到只剩一種，最後一種拿不掉（空清單伺服器整份拒收）：勾還在、清單不動、照講。
        page.locator("#zone-chip-remove-x0-whole_wall-ghost_material").click()
        assert _modes(page, "x0")["whole_wall"]["allowed"] == ["none", "glass_wool_24k"]
        expect(page.locator("#zone-allowed-missing-x0-whole_wall")).to_have_count(0)
        page.locator("#zone-allow-x0-whole_wall-glass_wool_24k").uncheck()  # 勾選表重畫後照樣開著
        assert _modes(page, "x0")["whole_wall"]["allowed"] == ["none"]
        page.locator("#zone-allow-x0-whole_wall-none").click()  # 想拿掉最後一種
        expect(page.locator("#zone-note")).to_contain_text("至少要留一種")
        expect(page.locator("#zone-allow-x0-whole_wall-none")).to_be_checked()
        page.locator("#zone-chip-remove-x0-whole_wall-none").click()  # 從方塊的 × 拿也一樣
        expect(page.locator("#zone-chip-x0-whole_wall-none")).to_have_count(1)
        assert _modes(page, "x0")["whole_wall"]["allowed"] == ["none"]
        _save(page, "room109")
        assert json.loads(design.read_text(encoding="utf-8"))["zones"]["x0"]["whole_wall"]["allowed"] == ["none"]
        assert watched.page_errors == [] and watched.console_errors == []


def test_search_zone_says_what_the_scheme_really_holds(tmp_path: Path, browser: Browser) -> None:
    """#1128：交給搜尋的區寫方案裡現在真的放什麼。網頁改成交給搜尋時放原牆（寫原牆、會貼在它上面）；搜尋存成的方案
    （或它的另存）那一區放的是搜尋選的貼牆構造，不准還寫「先放原牆」。這裡把貼了一層的 z1 在設計設定改成交給搜尋，
    就是存成方案那一份的形狀。"""
    _write(tmp_path, room109_document())
    _write(tmp_path, room109_document("room109-original"))
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-x0").click()
        _first_cut(page, tmp_path)
        _save(page, "room109")
        # #1176：原牆原樣寫白話，阻抗數字收在點開的「數字」裡。
        expect(page.locator("#zone-search-now-x0-auto1")).to_contain_text("現在：沒貼東西（原牆原樣）")
        expect(page.locator("#zone-search-now-x0-auto1")).to_contain_text("貼在它上面")
        expect(page.locator("#zone-search-now-x0-auto1")).not_to_contain_text("Pa·s/m")
        expect(page.locator("#zone-numbers-x0-auto1")).to_contain_text("原牆：固定阻抗")
        path = tmp_path / "designs" / "room109.json"
        stored = json.loads(path.read_text(encoding="utf-8"))
        stored["zones"]["x0"]["z1"] = {"mode": "search"}
        path.write_text(json.dumps(stored), encoding="utf-8")
        page.reload()
        page.wait_for_function("typeof inputDefaults !== 'undefined' && inputDefaults !== undefined")
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-x0").click()
        expect(page.locator("#zone-mode-x0-z1")).to_have_value("search")
        line = page.locator("#zone-search-now-x0-z1")
        expect(line).to_contain_text("現在：已貼 48K 玻璃棉 50 mm 貼牆（貼在原牆上）")
        expect(line).not_to_contain_text("先放原牆")
        expect(line).not_to_contain_text("沒貼東西")
        page.locator("#elevation-panel").screenshot(path=str(tmp_path / "search-zone-holds.png"))
        assert watched.page_errors == [] and watched.console_errors == []
