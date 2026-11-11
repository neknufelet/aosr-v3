"""真瀏覽器考立面圖（#1091）：站在房間裡看那一面（鏡像對不對用喇叭左右考，不只考自己的翻轉表）、
拖曳畫框貼到不在格點上的牆邊、補滿、存檔再開、改數字、刪框回整面；補出窄條時標紅不換方案；
房間改矮後兩面都對不上，一面一面清還是做得到。房間一律用 109 號房的長寬高。"""
from pathlib import Path
from typing import cast

from playwright.sync_api import Browser, Page, expect

from aosr.physics.zone_fill import Box, fill_wall
from aosr.reporting.scheme import Scheme
from tests.engine._gui_cache import gui_startup_identity_memo as gui_startup_identity_memo
from tests.engine._surface_zone_cases import ASSEMBLY, NEW, ROOM_109, room109_document, room109_quarters
from tests.engine.test_gui_browser import _open, _serve, browser as browser
from tests.engine.test_gui_input_setup_browser import _save_readback

LX, LY, LZ = ROOM_109["Lx"], ROOM_109["Ly"], ROOM_109["Lz"]


def _write(tmp_path: Path, document: dict[str, object]) -> None:
    (tmp_path / "schemes").mkdir(exist_ok=True)
    name = cast(str, document["scheme_id"])
    (tmp_path / "schemes" / f"{name}.json").write_text(Scheme.model_validate(document).model_dump_json())


def _open_scheme(page: Page, name: str) -> None:
    page.locator("#scheme-list").select_option(name)
    page.locator("#open-scheme").click()
    page.wait_for_function("name => document.getElementById('save-id').value === name", arg=name)
    page.wait_for_function("!document.querySelector('main').inert")


def _zones(page: Page, wall: str) -> list[dict[str, object]]:
    return cast(list[dict[str, object]], page.evaluate(f"structuredClone(scheme.scene.surface_zones_by_wall.{wall})"))


def _centre_x(page: Page, selector: str) -> float:
    box = page.locator(selector).bounding_box()
    assert box is not None
    return box["x"] + box["width"] / 2


def _centre_y(page: Page, selector: str) -> float:
    box = page.locator(selector).bounding_box()
    assert box is not None
    return box["y"] + box["height"] / 2


def _drag(page: Page, width: float, height: float, start: tuple[float, float], end: tuple[float, float]) -> None:
    """畫面座標（公尺，從左緣往右、從下緣往上）拖一個框；換算照牆框在畫面上的位置。"""
    # 滑鼠座標以視窗為準：牆框要先捲進視窗，不然按下去的點落在視窗外，瀏覽器不送事件。
    page.locator("#elevation-frame").scroll_into_view_if_needed()
    frame = page.locator("#elevation-frame").bounding_box()
    assert frame is not None

    def pixel(point: tuple[float, float]) -> tuple[float, float]:
        return frame["x"] + point[0] / width * frame["width"], frame["y"] + (1 - point[1] / height) * frame["height"]

    page.mouse.move(*pixel(start))
    page.mouse.down()
    page.mouse.move(*pixel(end), steps=6)
    with page.expect_response("**/api/zone-cut") as response:
        page.mouse.up()
    assert response.value.ok, response.value.text()
    page.wait_for_function("!document.querySelector('main').inert")


def _cut_by(page: Page, action: str, *, ok: bool = True) -> None:
    """按一個鈕（或 ``press:`` 開頭的按鍵）送一次切牆面，等回覆與整頁解鎖。"""
    with page.expect_response("**/api/zone-cut") as response:
        if action.startswith("press:"):
            page.keyboard.press(action.removeprefix("press:"))
        else:
            page.locator(action).click()
    assert response.value.ok is ok, response.value.text()
    page.wait_for_function("!document.querySelector('main').inert")


def _check_orientation_and_markers(page: Page) -> None:
    """站在房裡看那一面：用喇叭左右考鏡像，不只考自己的翻轉表；表單上改座標，記號跟著搬。"""
    # 參考擺位面向 x 起點牆聽：站在房裡看 x 起點牆，左喇叭在左；轉身看 x 終點牆，左喇叭到右邊。
    page.locator("#elevation-wall-x0").click()
    assert _centre_x(page, "#elevation-speaker-left") < _centre_x(page, "#elevation-speaker-right")
    expect(page.locator("#elevation-edge-left")).to_have_text("y 起點牆")
    expect(page.locator("#elevation-edge-right")).to_have_text("y 終點牆")
    page.locator("#elevation-wall-xL").click()
    assert _centre_x(page, "#elevation-speaker-left") > _centre_x(page, "#elevation-speaker-right")
    expect(page.locator("#elevation-edge-left")).to_have_text("y 終點牆")
    # 地板跟平面圖同向（y 往上）：左喇叭 y 小、畫在右喇叭下面；天花抬頭看，上下對調。喇叭（x＝1）在主位（x＝3.2）左邊。
    page.locator("#elevation-wall-floor").click()
    assert _centre_y(page, "#elevation-speaker-left") > _centre_y(page, "#elevation-speaker-right")
    assert _centre_x(page, "#elevation-speaker-left") < _centre_x(page, "#elevation-receiver-main")
    page.locator("#elevation-wall-ceiling").click()
    assert _centre_y(page, "#elevation-speaker-left") < _centre_y(page, "#elevation-speaker-right")
    expect(page.locator("#elevation-edge-top")).to_have_text("y 起點牆")

    # 表單上改了左喇叭的 y（還沒存），立面圖上的記號跟著搬：x 起點牆上往右移。
    page.locator("#elevation-wall-x0").click()
    before = _centre_x(page, "#elevation-speaker-left")
    page.locator("#speaker-left-y").fill("2.0")
    page.wait_for_function("x => { const r = document.querySelector('#elevation-speaker-left rect').getBoundingClientRect();"
                           " return r.x + r.width / 2 > x + 1; }", arg=before)
    page.locator("#speaker-left-y").fill("1.3")


def test_elevation_faces_the_wall_draws_fills_saves_edits_and_deletes(tmp_path: Path, browser: Browser) -> None:
    document = room109_document()
    cast(dict[str, list[dict[str, object]]], cast(dict[str, object], document["scene"])[NEW])["ceiling"][0]["material"] = ASSEMBLY
    original = cast(dict[str, list[dict[str, object]]], cast(dict[str, object], document["scene"])[NEW])["xL"][0]["material"]
    _write(tmp_path, document)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        _check_orientation_and_markers(page)

        # x 終點牆是鏡像：畫面左下角拖到離左緣 1.0、離下緣 1.2 公尺（都在畫面格點上，留半格餘裕給像素誤差），
        # 存下的 u 在高的那一頭，貼牆那一邊正好等於牆寬（4.41988… 不在 5 公分格上），另一邊離牆寬正好 1 公尺。
        page.locator("#elevation-wall-xL").click()
        _drag(page, LY, LZ, (0.02, 0.02), (1.0, 1.2))
        zones = {str(zone["id"]): zone for zone in _zones(page, "xL")}
        assert zones["z1"]["u_m"] == [LY - 1.0, LY] and zones["z1"]["v_m"] == [0.0, 1.2]
        assert sorted(zones) == ["auto1", "auto2", "z1"]
        assert all(zone["material"] == original for zone in zones.values())
        expect(page.locator("#elevation .zone-drawn")).to_have_count(1)
        expect(page.locator("#elevation .zone-auto")).to_have_count(2)
        expect(page.locator("#elevation-note")).to_contain_text("切成 3 區")
        page.locator("#elevation").screenshot(path=str(tmp_path / "xl-drawn.png"))

        saved = _save_readback(page, "room109-cut")
        assert cast(dict[str, list[object]], cast(dict[str, object], saved["scene"])[NEW])["xL"] == _zones(page, "xL")
        expect(page.locator("#elevation [data-zone-id=z1]")).to_have_class("zone zone-drawn zone-mode-fixed selected")

        # 改寬：只動「寬」那一格，貼牆那一邊照舊正好等於牆寬。
        page.locator("#elevation [data-zone-id=z1] rect").click()
        page.locator("#elevation-width").fill("1.5")
        _cut_by(page, "press:Enter")
        z1 = next(zone for zone in _zones(page, "xL") if zone["id"] == "z1")
        assert z1["u_m"] == [LY - 1.5, LY] and z1["v_m"] == [0.0, 1.2]

        _cut_by(page, "#elevation-delete")
        assert _zones(page, "xL") == [{"area": "whole_wall", "material": original}]
        expect(page.locator("#zoned-xL")).to_have_count(0)

        # 多層構造的牆切了再清回整面：構造表單重新長出來、欄位是方案裡的值，存回去材料一字不差。
        page.locator("#elevation-wall-ceiling").click()
        _drag(page, LX, LY, (0.5, 0.5), (2.0, 2.0))
        expect(page.locator("#construction-ceiling")).to_have_count(0)
        _cut_by(page, "#elevation-clear")
        expect(page.locator("#layer-ceiling-1-thickness_m")).to_have_value("0.43")
        saved = _save_readback(page, "room109-ceiling")
        ceiling = cast(dict[str, list[dict[str, object]]], cast(dict[str, object], saved["scene"])[NEW])["ceiling"]
        assert ceiling == [{"area": "whole_wall", "material": ASSEMBLY}]
        assert watched.page_errors == [] and watched.console_errors == []


def _thin_wall_document() -> dict[str, object]:
    """y 終點牆（多層構造）有一框離左牆邊只有 10 公分（檔案裡的方案可以這樣存）；x 兩面是 109 的 2×2。"""
    document = room109_document("shrink")
    room109_quarters(document, "x0", (100.0, 200.0, 300.0, 400.0))
    room109_quarters(document, "xL", (500.0, 600.0, 700.0, 800.0))
    zones = cast(dict[str, list[dict[str, object]]], cast(dict[str, object], document["scene"])[NEW])
    material = ASSEMBLY
    drawn = [Box("z1", (0.1, 2.0), (0.0, LZ)), Box("z2", (3.0, 4.0), (0.0, 1.0))]
    filled = fill_wall("yL", LX, LZ, drawn, min_side_m=0.0)
    zones["yL"] = [{"area": "rectangle", "id": box.id, "u_m": list(box.u_m), "v_m": list(box.v_m), "material": material}
                   for box in filled.zones]
    return document


def test_thin_strip_is_marked_red_and_walls_can_be_cleared_one_by_one_after_the_room_shrinks(
        tmp_path: Path, browser: Browser) -> None:
    document = _thin_wall_document()
    _write(tmp_path, document)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, "shrink")
        before = _zones(page, "yL")
        page.locator("#elevation-wall-yL").click()
        page.locator("#elevation [data-zone-id=z2] rect").click()
        page.locator("#elevation-height").fill("1.5")
        _cut_by(page, "press:Enter", ok=False)
        expect(page.locator("#elevation .zone-thin")).to_have_count(1)
        expect(page.locator("#elevation-note")).to_contain_text("窄條")
        assert _zones(page, "yL") == before

        # 房間改矮到 2.5 公尺：x 兩面與 y 終點牆的分區都超出牆面；一面一面清，每一面都清得掉。
        # 先清 y 終點牆：別兩面還壞著、預覽一定被拒，多層構造表單照樣長回來、欄位是方案裡的值。
        page.locator("#room-Lz").fill("2.5")
        page.locator("#elevation-wall-x0").click()
        expect(page.locator("#elevation-size")).to_contain_text("超出目前的牆面")
        for wall in ("yL", "x0", "xL"):
            page.locator(f"#elevation-wall-{wall}").click()
            _cut_by(page, "#elevation-clear")
            assert [zone["area"] for zone in _zones(page, wall)] == ["whole_wall"]
            if wall == "yL":
                expect(page.locator("#layer-yL-1-thickness_m")).to_have_value("0.43")
                assert _zones(page, "yL")[0]["material"] == ASSEMBLY
        expect(page.locator("#messages")).to_have_class("ok")
        assert cast(dict[str, object], _zones(page, "x0")[0]["material"])["impedance_pa_s_m"] == 100.0
        saved = _save_readback(page, "shrunk")
        assert cast(dict[str, dict[str, float]], saved["scene"])["room_m"]["Lz"] == 2.5
        assert cast(dict[str, list[dict[str, object]]], cast(dict[str, object], saved["scene"])[NEW])["yL"] == [
            {"area": "whole_wall", "material": ASSEMBLY}]
        page.locator("#elevation").screenshot(path=str(tmp_path / "cleared.png"))
        assert watched.page_errors == []
