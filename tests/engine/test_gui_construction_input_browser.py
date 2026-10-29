"""真瀏覽器填構造、逐值反查、修改、存讀與白話拒收；不使用假計算。"""
from pathlib import Path
from typing import cast

import pytest
from playwright.sync_api import Browser, expect

from tests.engine._gui_cache import gui_startup_identity_memo as gui_startup_identity_memo
from tests.engine.test_gui_browser import _open, _serve, browser as browser
from tests.engine.test_gui_input_setup_browser import _filled, _save_readback


@pytest.mark.parametrize("width", (1400, 390))
def test_select_edit_save_reload_and_reverse_lookup(tmp_path: Path, browser: Browser, width: int) -> None:
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=width) as watched:
        page = watched.page
        _filled(page, "#material-kind-floor", "assembly")
        _filled(page, "#preset-floor", "glass_wool_48k_50mm_gap50")
        expect(page.locator("#preset-state-floor")).to_have_text("使用預設：48K 玻璃棉 50 mm＋50 mm 空腔")
        _save_readback(page, "unchanged")
        expect(page.locator("#preset-state-floor")).to_contain_text("使用預設：48K")
        page.locator("#layer-floor-1-thickness_m").fill("0.075")
        expect(page.locator("#preset-state-floor")).to_contain_text("自訂構造")
        saved = _save_readback(page, "changed")
        scene = cast(dict[str, object], saved["scene"])
        zones = cast(dict[str, list[dict[str, object]]], scene["surface_zones_by_wall"])
        material = cast(dict[str, object], zones["floor"][0]["material"])
        assert cast(list[dict[str, object]], material["layers"])[1]["thickness_m"] == .075
        expect(page.locator("#layer-floor-1-thickness_m")).to_have_value("0.075")
        expect(page.locator("#preset-state-floor")).to_contain_text("自訂構造")
        with page.expect_response("**/api/input-preview"):
            page.locator("#material-kind-floor").select_option("constant_impedance")
        page.wait_for_function("!document.querySelector('main').inert")
        fixed = _save_readback(page, "fixed-again")
        fixed_scene = cast(dict[str, object], fixed["scene"])
        assert "surface_zones_by_wall" not in fixed_scene
        assert cast(dict[str, object], fixed_scene["impedance_pa_s_per_m_by_wall"])["floor"] == 1646.4
        for code in ("assembly", "constant_impedance", "komatsu", "rigid", "thickness_m"):
            assert code not in page.locator("#walls").inner_text()
        assert watched.page_errors == []


def test_layer_error_is_visible_plain_400(tmp_path: Path, browser: Browser) -> None:
    """層厚度清空時網頁照樣把伺服器的白話 400 印在訊息列（#972：原 [range] 那格的字由伺服器組，搬去
    test_gui_construction_repairs::test_preview_layer_range_error_is_plain_400；網頁印字這一段由這一格守）。"""
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _filled(page, "#material-kind-floor", "assembly")
        field = "thickness_m"
        with page.expect_response("**/api/input-preview") as response:
            page.locator(f"#layer-floor-0-{field}").fill("")
        assert response.value.status == 400, response.value.text()
        expect(page.locator("#messages")).to_contain_text("地板材料構造 › 第 1 層")
        text = page.locator("#messages").inner_text()
        assert "層厚度" in text
        assert "komatsu" not in text and field not in text
        assert watched.page_errors == []
