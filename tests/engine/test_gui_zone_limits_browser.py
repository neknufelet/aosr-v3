"""真瀏覽器考超過限制的區塊標紅（#1111，#1082 第 1-4 段）：立面圖下填深度上限（房間預設、這面牆、每一塊三層），
固定區貼得比上限深就在立面圖與表上標紅、寫出哪一層擋的，「拿去搜尋」停用；改好就解除；交給搜尋的區寫出生效的
上限；存檔再打開三層都在；再切同一面，區的上限跟著做法走。房間一律用 109 號房的長寬高。"""
import json
import re
from pathlib import Path
from typing import cast

from playwright.sync_api import Browser, Page, expect

from tests.engine._gui_cache import gui_startup_identity_memo as gui_startup_identity_memo
from tests.engine._surface_zone_cases import ROOM_109, room109_document
from tests.engine.test_gui_browser import _open, _serve, browser as browser
from tests.engine.test_gui_zone_elevation_browser import _drag, _open_scheme, _write, _zones
from tests.engine.test_gui_zone_modes_browser import _choose, _modes, _region, _save

LY, LZ = ROOM_109["Ly"], ROOM_109["Lz"]
PRESET = "glass_wool_48k_50mm"          # 總深 0.05 m
DEEP = "glass_wool_48k_50mm_gap100"     # 總深 0.15 m
RED = re.compile(r"\bzone-red\b")


def _depth(page: Page, selector: str, value: str) -> None:
    """填一格深度上限（離開那一格才算確定），等伺服器判完那一次。"""
    with page.expect_response("**/api/zone-limits") as response:
        page.locator(selector).fill(value)
        page.locator(selector).blur()
    assert response.value.ok, response.value.text()


def _red(page: Page, zone: str) -> None:
    expect(page.locator(f"#elevation [data-zone-id={zone}]")).to_have_class(RED)
    expect(page.locator("#zone-to-search")).to_be_disabled()



def _clear(page: Page, zone: str) -> None:
    expect(page.locator(f"#elevation [data-zone-id={zone}]")).not_to_have_class(RED)
    expect(page.locator("#zone-limits-summary")).to_have_text("限制都沒超過")
    expect(page.locator("#zone-to-search")).to_be_enabled()


def test_too_deep_fixed_zone_goes_red_names_the_level_and_blocks_the_search(tmp_path: Path, browser: Browser) -> None:
    _write(tmp_path, room109_document())
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-x0").click()
        _drag(page, LY, LZ, (0.02, 0.02), (1.0, 1.2))
        _choose(page, "#zone-fixed-x0-z1", PRESET)
        _choose(page, "#zone-mode-x0-auto1", "search")
        _clear(page, "z1")

        # 這面牆上限 0.1：貼 0.05 的還在限內；換成 0.15 的那一刻（換材料也重判）就紅。
        _depth(page, "#depth-wall-x0", "0.1")
        _clear(page, "z1")
        expect(page.locator("#depth-cap-x0-auto1")).to_have_text("生效：0.1 公尺（這面牆擋的）")
        with page.expect_response("**/api/zone-limits"):
            _choose(page, "#zone-fixed-x0-z1", DEEP)
        _red(page, "z1")
        expect(page.locator("#limit-x0-z1-0")).to_contain_text("0.15 公尺")
        expect(page.locator("#limit-x0-z1-0")).to_contain_text("這面牆的上限")
        expect(page.locator("#zone-limits-summary")).to_contain_text("有 1 條超過限制")
        expect(page.locator("#zone-to-search-note")).to_contain_text("超過限制")
        expect(page.locator("#elevation [data-zone-id=z1] title")).to_contain_text("這面牆的上限")
        page.locator("#elevation-panel").screenshot(path=str(tmp_path / "limits-wall-red.png"))

        # 牆不給、改給這一塊 0.12：還是紅，寫的是「這一塊」。
        _depth(page, "#depth-wall-x0", "")
        _clear(page, "z1")
        _depth(page, "#depth-zone-x0-z1", "0.12")
        _red(page, "z1")
        expect(page.locator("#limit-x0-z1-0")).to_contain_text("這一塊的上限")
        expect(page.locator("#depth-cap-x0-z1")).to_have_text("生效：0.12 公尺（這一塊擋的）")

        # 這一塊不給、房間預設 0.1：紅，寫「房間預設」；房間放寬到 0.2 就解除，交給搜尋那塊照房間預設。
        _depth(page, "#depth-zone-x0-z1", "")
        _depth(page, "#depth-room", "0.1")
        _red(page, "z1")
        expect(page.locator("#limit-x0-z1-0")).to_contain_text("房間預設的上限")
        _depth(page, "#depth-room", "0.2")
        _clear(page, "z1")
        expect(page.locator("#depth-cap-x0-auto1")).to_have_text("生效：0.2 公尺（房間預設擋的）")

        # 不合理的值不收：照講、格子畫回原值、不打伺服器。
        page.locator("#depth-room").fill("-1")
        page.locator("#depth-room").blur()
        expect(page.locator("#zone-note")).to_contain_text("大於 0")
        expect(page.locator("#depth-room")).to_have_value("0.2")
        assert watched.page_errors == [] and watched.console_errors == []


def test_depth_limits_survive_a_reopen_and_a_zone_cap_follows_a_recut(tmp_path: Path, browser: Browser) -> None:
    _write(tmp_path, room109_document())
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-x0").click()
        _drag(page, LY, LZ, (0.02, 0.02), (1.0, 1.2))
        _choose(page, "#zone-fixed-x0-z1", DEEP)
        _choose(page, "#zone-mode-x0-auto1", "search")
        _depth(page, "#depth-room", "0.3")
        _depth(page, "#depth-wall-x0", "0.2")
        _depth(page, "#depth-zone-x0-auto1", "0.08")
        expect(page.locator("#depth-cap-x0-auto1")).to_have_text("生效：0.08 公尺（這一塊擋的）")
        # 在右邊那一大塊（交給搜尋、上限 0.08）裡再畫一框：自動區重編，上限跟著做法走。
        _drag(page, LY, LZ, (2.02, 0.02), (3.0, 1.0))
        after = {str(zone["id"]): zone for zone in _zones(page, "x0")}
        modes = _modes(page, "x0")
        right = [key for key, zone in after.items() if _region(zone) == "right" and key != "z2"]
        assert right and all(modes[key] == {"mode": "search", "max_depth_m": 0.08} for key in right)
        _save(page, "room109")
        stored = json.loads((tmp_path / "designs" / "room109.json").read_text(encoding="utf-8"))
        assert stored["room_max_depth_m"] == 0.3 and stored["wall_max_depth_m"] == {"x0": 0.2}
        assert stored["zones"]["x0"] == modes

        page.reload()
        page.wait_for_function("typeof inputDefaults !== 'undefined' && inputDefaults !== undefined")
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-x0").click()
        expect(page.locator("#depth-room")).to_have_value("0.3")
        expect(page.locator("#depth-wall-x0")).to_have_value("0.2")
        expect(page.locator(f"#depth-zone-x0-{right[0]}")).to_have_value("0.08")
        expect(page.locator(f"#depth-cap-x0-{right[0]}")).to_have_text("生效：0.08 公尺（這一塊擋的）")
        _clear(page, "z1")
        # 別面沒填牆的上限：照房間預設。
        page.locator("#elevation-wall-y0").click()
        expect(page.locator("#depth-wall-y0")).to_have_value("")
        page.locator("#elevation-panel").screenshot(path=str(tmp_path / "limits-reopened.png"))
        assert watched.page_errors == [] and watched.console_errors == []


def test_send_to_search_stays_put_while_red(tmp_path: Path, browser: Browser) -> None:
    """紅著的時候「拿去搜尋」按不下去；就算繞過按鈕直接叫，也只照講、不存不跳。"""
    _write(tmp_path, room109_document())
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-x0").click()
        _drag(page, LY, LZ, (0.02, 0.02), (1.0, 1.2))
        _choose(page, "#zone-fixed-x0-z1", DEEP)
        _choose(page, "#zone-mode-x0-auto1", "search")
        _depth(page, "#depth-wall-x0", "0.1")
        _red(page, "z1")
        page.evaluate("sendToSearch()")
        expect(page.locator("#zone-note")).to_contain_text("改好之前不能拿去搜尋")
        assert page.url == f"{base}/"
        assert not (tmp_path / "designs" / "room109.json").exists()
        assert cast(int, page.evaluate("limitsRedCount()")) == 1
        assert watched.page_errors == [] and watched.console_errors == []
