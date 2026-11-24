"""真瀏覽器打開分區方案（#1083）：分區的牆只列出每一區、不收值；改整面一區的牆再存，分區的牆一字不動。"""
from copy import deepcopy
from pathlib import Path
from typing import cast

from playwright.sync_api import Browser, expect

from aosr.reporting.scheme import Scheme
from tests.engine._input_page_steps import open_scheme, reveal
from tests.engine._gui_cache import gui_startup_identity_memo as gui_startup_identity_memo
from tests.engine._surface_zone_cases import ASSEMBLY, NEW, rectangle_document
from tests.engine.test_gui_browser import _open, _serve, browser as browser
from tests.engine.test_gui_input_setup_browser import _save_readback

ZONED_WALLS = ("ceiling", "x0", "xL", "y0", "yL")


def _zoned_document() -> dict[str, object]:
    """天花最後一區（r1_c1）換一個阻抗，抓「只畫第一區、存檔時把第一區抄給整面」；
    地板（整面一區）與 x 起點牆第一區是多層構造：改地板一層時，表單逐牆刷新的迴圈不准碰到分區的牆。"""
    document = rectangle_document() | {"scheme_id": "zoned"}
    zones = cast(dict[str, list[dict[str, object]]], cast(dict[str, object], document["scene"])[NEW])
    cast(dict[str, object], zones["ceiling"][-1]["material"])["impedance_pa_s_m"] = 1234.5
    zones["floor"][0]["material"] = deepcopy(ASSEMBLY)
    zones["x0"][0]["material"] = deepcopy(ASSEMBLY)
    return document


def test_zoned_walls_are_listed_and_left_alone_when_another_wall_is_edited(tmp_path: Path, browser: Browser) -> None:
    document = _zoned_document()
    (tmp_path / "schemes").mkdir()
    (tmp_path / "schemes" / "zoned.json").write_text(Scheme.model_validate(document).model_dump_json())
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        open_scheme(page, "zoned")
        page.wait_for_function("name => document.getElementById('save-id').value === name", arg="zoned")
        page.wait_for_function("!document.querySelector('main').inert")
        # #1265 單元 3 起分區的牆每一區列在第 2 段右欄「每一區的位置與大小」摺疊（選中那一面才列；一區一列，用區號不寫代號，
        # 列的 id 帶代號）：先點天花。
        page.locator("#elevation-wall-ceiling").click()
        expect(page.locator("#zoned-ceiling")).to_contain_text("分成 4 區")
        last = page.locator("#zoned-list-ceiling #zone-size-ceiling-r1_c1")
        expect(last).to_contain_text("1,234.5")  # 只給人看的收位寫法（#1119，千分位）
        for wall in ZONED_WALLS:
            assert page.locator(f"#wall-{wall}").count() == 0 and page.locator(f"#material-kind-{wall}").count() == 0
        reveal(page, "#layer-floor-1-thickness_m")  # 地板整面一區：表單在「換原本的牆」裡
        with page.expect_response("**/api/validate"):
            page.locator("#layer-floor-1-thickness_m").fill("0.5")
        saved = _save_readback(page, "zoned-edited")
        zones = cast(dict[str, list[dict[str, object]]], cast(dict[str, object], saved["scene"])[NEW])
        floor = cast(dict[str, object], zones["floor"][0]["material"])
        assert cast(list[dict[str, object]], floor["layers"])[1]["thickness_m"] == 0.5
        original = cast(dict[str, list[dict[str, object]]],
                        cast(dict[str, object], Scheme.model_validate(document).model_dump(mode="json")["scene"])[NEW])
        assert {wall: zones[wall] for wall in ZONED_WALLS} == {wall: original[wall] for wall in ZONED_WALLS}
        assert watched.page_errors == [] and watched.console_errors == []
