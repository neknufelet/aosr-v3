"""真瀏覽器打開分區方案（#1083）：分區的牆只列出每一區、不收值；改整面一區的牆再存，分區的牆一字不動。"""
from copy import deepcopy
from pathlib import Path
from typing import cast

from playwright.sync_api import Browser, expect

from aosr.reporting.scheme import Scheme
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
        page.locator("#scheme-list").select_option("zoned")
        page.locator("#open-scheme").click()
        page.wait_for_function("name => document.getElementById('save-id').value === name", arg="zoned")
        page.wait_for_function("!document.querySelector('main').inert")
        expect(page.locator("#zoned-ceiling")).to_contain_text("分成 4 區")
        last = page.locator("#zoned-list-ceiling li").nth(3)
        expect(last).to_contain_text("r1_c1")
        expect(last).to_contain_text("1234.5")
        for wall in ZONED_WALLS:
            assert page.locator(f"#wall-{wall}").count() == 0 and page.locator(f"#material-kind-{wall}").count() == 0
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
