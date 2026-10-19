"""真瀏覽器選新預設、選項送值、選填清空、背布兩格與實測草稿；截圖只寫暫存目錄。"""
from pathlib import Path
from typing import cast

import pytest
from playwright.sync_api import Browser, Page, expect

from tests.engine._gui_cache import gui_startup_identity_memo as gui_startup_identity_memo
from tests.engine.test_gui_browser import _open, _serve, browser as browser
from tests.engine.test_gui_construction_options import P1, first_layer
from tests.engine.test_gui_input_setup_browser import _filled
from tests.engine.test_gui_measured_input_browser import start_draft, apply_draft


def save_reload(page: Page, name: str) -> dict[str, object]:
    page.locator("#save-as-id").fill(name)
    with page.expect_response(lambda reply: reply.url.endswith(f"/api/schemes/{name}") and reply.request.method == "PUT") as reply:
        page.locator("#save-as").click()
    assert reply.value.ok, reply.value.text()
    saved = page.request.get(f"{page.url.rstrip('/')}/api/schemes/{name}").json()["scheme"]
    page.locator("#scheme-list").select_option(name)
    with page.expect_response("**/api/input-preview") as preview:
        page.locator("#open-scheme").click()
    assert preview.value.ok, preview.value.text()
    page.wait_for_function("!document.querySelector('main').inert")
    return cast(dict[str, object], saved)


def select_layer(page: Page, selector: str, value: str) -> None:
    with page.expect_response("**/api/input-preview") as reply:
        page.locator(selector).select_option(value)
    assert reply.value.ok, reply.value.text()


def take_shots(page: Page, folder: Path, name: str) -> None:
    page.screenshot(path=str(folder / f"{name}-page.png"), full_page=True)
    page.locator("#construction-floor").screenshot(path=str(folder / f"{name}-layers.png"))


@pytest.mark.parametrize("width", (1400, 390))
def test_p1_selects_clear_optional_save_reload_preview(tmp_path: Path, browser: Browser, width: int) -> None:
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=width) as watched:
        page = watched.page
        _filled(page, "#material-kind-floor", "assembly")
        _filled(page, "#preset-floor", P1)
        expect(page.locator("#open-area-floor-0")).to_have_text("算出的開孔率約 15.51%")
        expect(page.locator("#preset-state-floor")).to_contain_text("使用預設")
        select_layer(page, "#layer-floor-1-flow_resistivity_source", "catalog")
        select_layer(page, "#layer-floor-0-hole_pattern", "staggered_60")
        expect(page.locator("#open-area-floor-0")).to_have_text("算出的開孔率約 17.91%")
        page.locator("#layer-floor-0-surface_mass_kg_m2").fill("0.8")
        with page.expect_response("**/api/input-preview") as preview:
            page.locator("#layer-floor-0-surface_mass_kg_m2").fill("")
        assert preview.value.ok, preview.value.text()
        saved = save_reload(page, "p1-edited")
        layer = first_layer(saved)
        assert layer["hole_pattern"] == "staggered_60"
        assert "surface_mass_kg_m2" not in layer
        scene = cast(dict[str, object], saved["scene"])
        zones = cast(dict[str, list[dict[str, object]]], scene["surface_zones_by_wall"])
        layers = cast(list[dict[str, object]], cast(dict[str, object], zones["floor"][0]["material"])["layers"])
        assert layers[1]["flow_resistivity_source"] == "catalog"
        expect(page.locator("#layer-floor-1-flow_resistivity_source")).to_have_value("catalog")
        expect(page.locator("#layer-floor-0-hole_pattern")).to_have_value("staggered_60")
        expect(page.locator("#open-area-floor-0")).to_have_text("算出的開孔率約 17.91%")
        for code in ("perforated", "density_table", "staggered_60", "komatsu", "rigid"):
            assert code not in page.locator("#walls").inner_text()
        take_shots(page, tmp_path, f"p1-{width}")
        assert watched.page_errors == [] and watched.console_errors == []


def test_add_backing_fabric_in_two_empty_fields(tmp_path: Path, browser: Browser) -> None:
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _filled(page, "#material-kind-floor", "assembly")
        _filled(page, "#preset-floor", "perforated_0p8mm_d10_gw24k_50mm")
        for field in ("backing_fabric_surface_mass_kg_m2", "backing_fabric_flow_resistance_pa_s_m"):
            expect(page.locator(f"#layer-floor-0-{field}")).to_have_value("")
        # 同一次表單操作填好成對欄再發事件，不在兩格半填時要求驗證通過。
        with page.expect_response("**/api/input-preview") as preview:
            page.evaluate("""() => {
              const mass = document.getElementById('layer-floor-0-backing_fabric_surface_mass_kg_m2');
              const resistance = document.getElementById('layer-floor-0-backing_fabric_flow_resistance_pa_s_m');
              mass.value = '0.063'; resistance.value = '236';
              delete mass.dataset.originalValue; delete resistance.dataset.originalValue;
              mass.dispatchEvent(new Event('input', {bubbles: true}));
            }""")
        assert preview.value.ok, preview.value.text()
        saved = save_reload(page, "added-backing")
        layer = first_layer(saved)
        assert layer["backing_fabric_surface_mass_kg_m2"] == .063
        assert layer["backing_fabric_flow_resistance_pa_s_m"] == 236.
        expect(page.locator("#layer-floor-0-backing_fabric_surface_mass_kg_m2")).to_have_value("0.063")
        expect(page.locator("#layer-floor-0-backing_fabric_flow_resistance_pa_s_m")).to_have_value("236")
        take_shots(page, tmp_path, "added-backing")
        assert watched.page_errors == [] and watched.console_errors == []


def test_measured_draft_preserves_both_select_values(tmp_path: Path, browser: Browser) -> None:
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        start_draft(page)
        page.locator("#preset-floor").select_option(P1)
        page.locator("#layer-floor-1-flow_resistivity_source").select_option("measured")
        page.locator("#layer-floor-0-hole_pattern").select_option("staggered_45")
        draft = page.evaluate("measuredDrafts.floor.base_assembly.layers")
        assert draft[0]["hole_pattern"] == "staggered_45"
        assert draft[1]["flow_resistivity_source"] == "measured"
        apply_draft(page)
        expect(page.locator("#open-area-floor-0")).to_have_text("算出的開孔率約 31.03%")
        saved = save_reload(page, "draft-selects")
        assert first_layer(saved)["hole_pattern"] == "staggered_45"
        take_shots(page, tmp_path, "measured-draft")
        assert watched.page_errors == [] and watched.console_errors == []
