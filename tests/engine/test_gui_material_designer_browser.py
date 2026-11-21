"""從材料庫實際進設計器、存入暫存庫、重新開啟並到分區選材。"""
import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TypedDict, cast

import pytest

from playwright.sync_api import Browser, Dialog, Locator, Page, Response, Route, expect

from aosr.materials.library import SCHEMA_VERSION
from tests.engine._construction_cases import construction
from tests.engine._construction_gui_server import real_server
from tests.engine.test_gui_browser import _open, browser as browser
from tests.engine.test_gui_zone_overview_tasks_browser import _click_zone, _open_overview, _prepare


@contextmanager
def _designer(folder: Path, browser: Browser, width: int = 1440) -> Iterator[tuple[Page, str]]:
    with real_server(folder) as base, _open(browser, f"{base}/materials", viewport_width=width) as watched:
        page = watched.page
        page.locator("#tab-optimizable").click()
        new = page.get_by_role("button", name="新增構造", exact=True)
        expect(new).to_be_visible()
        new.click()
        expect(page.get_by_label("名稱", exact=True)).to_be_editable()
        yield page, base
        assert (watched.page_errors, watched.console_errors) == ([], [])


def _add(page: Page, *kinds: str) -> None:
    for kind in kinds:
        page.locator(f'#palette button[data-kind="{kind}"]').click()


def _save(page: Page, tab: str) -> None:
    page.get_by_role("button", name="儲存", exact=True).click()
    expect(page.locator("#save-message")).to_contain_text(f"已存到「{tab}」頁")


def _list_row(page: Page, base: str, tab: str, name: str) -> Locator:
    page.goto(f"{base}/materials")
    page.locator(f"#tab-{tab}").click()
    return page.get_by_role("tabpanel").get_by_role("article").filter(has=page.get_by_text(name, exact=True))


def _fields(page: Page, index: int) -> dict[str, str]:
    row = page.locator("#stack article").nth(index)
    head = row.locator("button.layer-select")
    if head.get_attribute("aria-expanded") == "false":
        head.click()
    return cast(dict[str, str], row.locator("[name]").evaluate_all("nodes => Object.fromEntries(nodes.map(n => [n.name, n.value]))"))


def test_build_move_remove_save_and_reopen(tmp_path: Path, browser: Browser) -> None:
    with _designer(tmp_path, browser) as (page, base):
        name = "自編疊層甲"
        page.get_by_label("名稱", exact=True).fill(name)
        _add(page, "fabric", "porous", "air")
        _fields(page, 2)
        page.get_by_role("button", name="往房間側移").click()
        _fields(page, 0)
        page.get_by_role("button", name="拿掉").click()
        page.locator('input[name="backing"][value="air"]').check()
        page.get_by_role("radio", name="牆", exact=True).check()
        values = [_fields(page, i) for i in range(2)]
        page.locator('input[name="thickness_m"]').fill("63.7")
        values[1] = _fields(page, 1)
        _save(page, "參數構造")
        row = _list_row(page, base, "optimizable", name)
        row.get_by_role("link", name=name, exact=True).click()
        expect(page.get_by_text("改這一筆，已經貼在分區上的不會跟著變")).to_be_visible()
        expect(page.get_by_label("名稱", exact=True)).to_have_value(name)
        expect(page.locator('input[name="backing"]:checked')).to_have_value("air")
        expect(page.get_by_role("radio", name="牆", exact=True)).to_be_checked()
        assert page.locator("#stack article").evaluate_all("rows => rows.map(r => r.dataset.kind)") == ["air", "porous"]
        assert [_fields(page, i) for i in range(2)] == values
        entries = page.request.get(f"{base}/api/materials").json()["materials"]
        stored, = [r["entry"] for r in entries if r["entry"]["label"] == name]
        assert stored["layers"][1]["thickness_m"] == .0637
        assert stored["provenance"] == "model"
        assert stored["kind"] == "construction" and stored["placement"] == "wall"


def _curve_for_thickness(response: Response, meters: float) -> bool:
    if not response.url.endswith("/api/material-curve"):
        return False
    document = response.request.post_data_json
    return isinstance(document, dict) and bool(document.get("layers")) and document["layers"][0].get("thickness_m") == meters


def test_change_thickness_updates_cutaway_curve_and_total(tmp_path: Path, browser: Browser) -> None:
    with _designer(tmp_path, browser) as (page, _):
        _add(page, "porous", "air")
        _fields(page, 0)
        thickness = page.locator('input[name="thickness_m"]')
        with page.expect_response(lambda r: _curve_for_thickness(r, .0371)) as before:
            thickness.fill("37.1")
        assert before.value.json()["status"] == "drawn"
        curve = page.locator("#curve canvas")
        expect(curve).to_be_visible()
        before_image = curve.evaluate("canvas => canvas.toDataURL()")
        slab = page.locator('#figure g[data-layer="0"] rect')
        width = float(slab.get_attribute("width") or "0")
        total = page.locator("#total").inner_text()
        with page.expect_response(lambda r: _curve_for_thickness(r, .0837)) as after:
            thickness.fill("83.7")
        assert after.value.json()["status"] == "drawn"
        expect(curve).to_be_visible()
        assert before_image != curve.evaluate("canvas => canvas.toDataURL()")
        assert float(slab.get_attribute("width") or "0") > width
        expect(page.locator("#total")).not_to_have_text(total)
        page.locator('#figure g[data-layer="1"] rect').click()
        expect(page.locator("#stack article").nth(1).locator(".layer-select")).to_have_attribute("aria-expanded", "true")
        previous = curve.evaluate("canvas => canvas.toDataURL()")
        held: list[Route] = []
        page.route("**/api/material-curve", lambda route: held.append(route), times=1)
        with page.expect_request("**/api/material-curve"):
            thickness.fill("71.2")
        with page.expect_response(lambda r: r.url.endswith("/api/material-curve")) as invalid:
            thickness.fill("-13")
        expect(page.locator("#curve")).to_have_text(invalid.value.json()["message"])
        with page.expect_event("requestfinished", predicate=lambda r: r == held[0].request):
            held[0].fulfill(json=after.value.json())
        page.evaluate("new Promise(done => requestAnimationFrame(() => requestAnimationFrame(done)))")
        expect(page.locator("#curve")).to_have_text(invalid.value.json()["message"])
        expect(curve).to_be_visible()
        assert curve.evaluate("canvas => canvas.toDataURL()") == previous


def test_product_copy_is_a_separate_editable_construction(tmp_path: Path, browser: Browser) -> None:
    with _designer(tmp_path, browser) as (page, base):
        name = "自編完成面"
        page.get_by_label("名稱", exact=True).fill(name)
        _add(page, "fabric")
        page.get_by_label("這是買來的或照抄的，不拆").check()
        _save(page, "成品")
        row = _list_row(page, base, "fixed", name)
        row.get_by_role("button", name="打開").click()
        expect(page.get_by_label("名稱", exact=True)).to_be_disabled()
        expect(page.get_by_role("button", name="儲存")).to_be_hidden()
        entries = f"{base}/api/materials"
        original, = [r["entry"] for r in page.request.get(entries).json()["materials"] if r["entry"]["label"] == name]
        page.get_by_role("button", name="複製到設計器").click()
        expect(page.get_by_label("名稱", exact=True)).to_have_value(name + "（複本）")
        expect(page.get_by_label("這是買來的或照抄的，不拆")).not_to_be_checked()
        _add(page, "air")
        _save(page, "參數構造")
        saved = {r["entry"]["label"]: r["entry"] for r in page.request.get(entries).json()["materials"]}
        assert original["product"] and saved[name] == original
        copy = saved[name + "（複本）"]
        assert copy["id"] != original["id"] and not copy["product"] and len(copy["layers"]) == 2
        expect(_list_row(page, base, "optimizable", copy["label"])).to_be_visible()


def test_saved_rigid_construction_is_selectable_in_zone(tmp_path: Path, browser: Browser) -> None:
    _prepare(tmp_path)
    with _designer(tmp_path, browser) as (page, base):
        name = "自編剛背牆材"
        page.get_by_label("名稱", exact=True).fill(name)
        _add(page, "porous")
        _save(page, "參數構造")
        page.goto(base)
        _open_overview(page)
        _click_zone(page, "yL/z3")
        page.locator("#zone-panel-mode-fixed").check()
        select = page.locator("#zone-panel-material")
        option = select.locator("option").filter(has_text=name)
        value = option.get_attribute("value")
        assert value
        select.select_option(value)
        expect(select).to_have_value(value)


def test_phone_stack_fields_figure_and_save_fit(tmp_path: Path, browser: Browser) -> None:
    with _designer(tmp_path, browser, 390) as (page, _):
        _add(page, "fabric", "porous", "air")
        _fields(page, 1)
        switches = page.get_by_role("switch")
        expect(switches).to_have_count(3)
        for switch in switches.all():
            expect(switch).to_be_disabled()
            expect(switch).not_to_be_checked()
        expect(page.get_by_text("第二批才能給程式調", exact=True)).to_have_count(3)
        expect(page.locator("#curve canvas")).to_be_visible(timeout=30000)
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        assert page.locator("section").evaluate_all("nodes => nodes.every(n => n.scrollWidth <= n.clientWidth)")
        _save(page, "參數構造")


class SavedMaterial(TypedDict):
    kind: str
    cavity: dict[str, object]
    id: str
    label: str
    layers: list[dict[str, object]]
    material: dict[str, object]
    backing: str
    source: str
    links: list[str]
    notes: list[str]
    provenance: str


def _stored(page: Page, base: str) -> dict[str, SavedMaterial]:
    rows = page.request.get(f"{base}/api/materials").json()["materials"]
    return {row["entry"]["id"]: row["entry"] for row in rows if not row["builtin"]}


def _legacy_copy(page: Page, base: str) -> SavedMaterial:
    old = {"kind": "fixed", "id": "loom_panel", "label": "織房試板", "source": "試片簿第七頁", "provenance": "measured",
           "links": ["https://example.org/loom"], "notes": ["樣片乙"], "material": {"kind": "assembly", "backing": "air",
           "layers": [{"kind": "porous", "thickness_m": .034, "flow_resistivity_pa_s_m2": 13217, "material_class": "fibrous"},
                      {"kind": "air", "thickness_m": .023}]}}
    assert page.request.put(f"{base}/api/materials/loom_panel", data=old).ok
    row = _list_row(page, base, "fixed", "織房試板")
    row.get_by_role("button", name="打開").click()
    page.get_by_role("button", name="複製到設計器").click()
    expect(page.locator("#workspace")).to_be_visible()
    return _stored(page, base)["loom_panel"]


def _advanced(page: Page) -> None:
    summary = page.get_by_text("進階，一般不用填", exact=True)
    expect(summary).to_be_visible()
    summary.click()


def test_converted_units_are_exact_in_saved_material(tmp_path: Path, browser: Browser) -> None:
    with _designer(tmp_path, browser) as (page, base):
        _add(page, "micro_perforated")
        _fields(page, 0)
        page.locator('[name="thickness_m"]').fill("5.1")
        area = page.locator('[name="open_area"]')
        area.fill("1.1")
        _save(page, "參數構造")
        entry, = _stored(page, base).values()
        layer, = entry["layers"]
        assert (layer["thickness_m"], layer["open_area"]) == (.0051, .011)


def test_measured_copy_is_model_with_editable_citation(tmp_path: Path, browser: Browser) -> None:
    with _designer(tmp_path, browser) as (page, base):
        original = _legacy_copy(page, base)
        _fields(page, 0)
        page.locator('[name="thickness_m"]').fill("41.3")
        _save(page, "參數構造")
        copy = _stored(page, base)["loom_panel_copy"]
        assert copy["provenance"] == "model"
        assert copy["source"] == "照「織房試板」改的"
        assert all(word in copy["notes"][0] for word in ("實測", original["source"], *original["links"]))
        assert copy["notes"][1:] == original["notes"]
        expect(page.get_by_label("出處文字")).to_be_hidden()
        page.get_by_text("出處與註記", exact=True).click()
        page.get_by_label("出處文字").fill("自編改版說明")
        page.get_by_label("網址").fill("https://example.org/revised")
        page.get_by_label("註記", exact=True).fill(copy["notes"][0] + "\n另裁一片")
        _save(page, "參數構造")
        copy = _stored(page, base)["loom_panel_copy"]
        assert (copy["source"], copy["links"], copy["notes"][1:]) == ("自編改版說明", ["https://example.org/revised"], ["另裁一片"])
        assert _stored(page, base)["loom_panel"] == original


def test_porous_fields_follow_model(tmp_path: Path, browser: Browser) -> None:
    with _designer(tmp_path, browser) as (page, base):
        _add(page, "porous")
        _fields(page, 0)
        fields = page.locator(".layer-body [name]:visible")
        assert fields.evaluate_all("ns => ns.map(n => n.name).sort()") == ["flow_resistivity_pa_s_m2", "material_class", "thickness_m"]
        _advanced(page)
        model = page.locator('[name="model"]')
        porosity = page.locator('[name="porosity"]')
        tortuosity = page.locator('[name="tortuosity"]')
        expect(porosity).to_be_hidden()
        model.select_option("wilson_2015")
        porosity.fill("0.91")
        tortuosity.fill("1.47")
        model.select_option("mechel_1976")
        expect(porosity).to_have_value("0.91")
        expect(tortuosity).to_be_hidden()
        model.select_option("miki")
        expect(porosity).to_be_hidden()
        _save(page, "參數構造")
        entry, = _stored(page, base).values()
        assert {"porosity", "tortuosity"}.isdisjoint(entry["layers"][0])


@pytest.mark.parametrize("kind", ["micro_perforated", "perforated", "slotted"])
def test_plate_saves_only_selected_geometry(tmp_path: Path, browser: Browser, kind: str) -> None:
    with _designer(tmp_path, browser) as (page, base):
        _add(page, kind)
        _fields(page, 0)
        visible = page.locator(".layer-body [name]:visible").evaluate_all("ns => ns.map(n => n.name)")
        assert visible.count("geometry") == 2
        assert {"thickness_m", "slot_width_m" if kind == "slotted" else "hole_diameter_m"} <= set(visible)
        assert {"open_area", "slot_spacing_m", "hole_spacing_m"}.intersection(visible)
        assert page.locator(".advanced [name]").evaluate_all("ns => ns.map(n => n.name)") == ["surface_mass_kg_m2"]
        slotted = kind == "slotted"
        pitch = "slot_spacing_m" if slotted else "hole_spacing_m"
        page.get_by_label("用縫距" if slotted else "用孔距＋排法", exact=True).check()
        expect(page.locator('[name="open_area"]')).to_be_hidden()
        page.locator(f'[name="{pitch}"]').fill("34.2")
        if not slotted:
            pattern = page.locator('[name="hole_pattern"]')
            pattern.select_option("rectangular")
            page.locator('[name="hole_spacing_2_m"]').fill("28.4")
            pattern.select_option("square")
            expect(page.locator('[name="hole_spacing_2_m"]')).to_be_hidden()
        if kind != "micro_perforated":
            cloth = page.get_by_role("checkbox", name="背面貼布", exact=True)
            cloth.check()
            page.locator('[name="backing_fabric_surface_mass_kg_m2"]').fill("0.13")
            page.locator('[name="backing_fabric_flow_resistance_pa_s_m"]').fill("347")
        _save(page, "參數構造")
        entry, = _stored(page, base).values()
        layer = entry["layers"][0]
        assert layer[pitch] == .0342 and {"open_area", "hole_spacing_2_m"}.isdisjoint(layer)
        if kind != "micro_perforated":
            assert (layer["backing_fabric_surface_mass_kg_m2"], layer["backing_fabric_flow_resistance_pa_s_m"]) == (.13, 347)
        page.get_by_label("用開孔率", exact=True).check()
        page.locator('[name="open_area"]').fill("8.2")
        if kind != "micro_perforated":
            page.get_by_role("checkbox", name="背面貼布", exact=True).uncheck()
        _save(page, "參數構造")
        layer = _stored(page, base)[entry["id"]]["layers"][0]
        assert layer["open_area"] == .082 and {pitch, "hole_pattern", "hole_spacing_2_m"}.isdisjoint(layer)
        assert not any(key.startswith("backing_fabric_") for key in layer)


def test_chart_ticks_legend_basis_and_debounce(tmp_path: Path, browser: Browser) -> None:
    with _designer(tmp_path, browser) as (page, _):
        page.evaluate("""() => {
          window.ticks = [];
          const original = CanvasRenderingContext2D.prototype.fillText;
          CanvasRenderingContext2D.prototype.fillText = function(text, ...args) {
            window.ticks.push(String(text)); return original.call(this, text, ...args);
          };
        }""")
        with page.expect_response(lambda r: r.url.endswith("/api/material-curve") and bool((r.request.post_data_json or {}).get("layers"))) as response:
            _add(page, "porous")
        expect(page.locator("#curve canvas")).to_be_visible()
        expect(page.locator("#curve-basis")).to_have_text(response.value.json()["basis"])
        expect(page.locator('#curve-key li[data-kind="extrapolation"]')).to_be_visible()
        ticks = page.evaluate("window.ticks")
        assert {"頻率（Hz）", "無規入射吸音係數"} <= set(ticks)
        numeric = {tick.replace(",", "") for tick in ticks}
        assert len([s for s in numeric if s.isdigit() and int(s) > 1]) >= 2
        assert any(s.startswith("0.") for s in ticks)
        requests: list[str] = []
        page.on("request", lambda r: requests.append(r.url) if r.url.endswith("/api/material-curve") else None)
        page.clock.install()
        page.get_by_label("名稱", exact=True).fill("只改標題")
        page.locator("#product").check()
        page.clock.run_for(1000)
        assert not requests
        _fields(page, 0)
        thickness = page.locator('[name="thickness_m"]')
        thickness.fill("38")
        page.clock.run_for(300)
        thickness.fill("39")
        page.clock.run_for(300)
        assert not requests
        with page.expect_request("**/api/material-curve"):
            page.clock.run_for(300)
        assert len(requests) == 1


@pytest.mark.parametrize(("provenance", "word"), [("measured", "實測"), ("literature", "文獻"), ("model", "模型")])
def test_readonly_header(tmp_path: Path, browser: Browser, provenance: str, word: str) -> None:
    with _designer(tmp_path, browser) as (page, base):
        product = construction(id="linen", product=True, provenance=provenance, source="自編織物紀錄")
        assert page.request.put(f"{base}/api/materials/linen", data=product).ok
        with page.expect_request("**/api/material-curve") as preview:
            page.goto(f"{base}/materials/designer?id=linen")
        body = preview.value.post_data_json
        assert isinstance(body, dict) and body["provenance"] == provenance
        expect(page.locator("#entry-status")).to_have_text("成品唯讀，可複製到設計器")
        page.get_by_text("出處與註記", exact=True).click()
        expect(page.locator("#source")).to_have_text(word)
        expect(page.locator("#citation-provenance")).to_have_text(word)
        expect(page.get_by_label("出處文字")).to_be_disabled()
        page.get_by_role("button", name="複製到設計器").click()
        _save(page, "參數構造")
        stored = _stored(page, base)
        assert stored["linen"]["provenance"] == provenance
        assert stored["linen_copy"]["provenance"] == "model"
        notes = stored["linen_copy"]["notes"]
        if provenance == "model":
            assert notes == product["notes"]
        else:
            assert notes[0].startswith(f"原來源：{word}，出處：自編織物紀錄")


def test_dirty_navigation_can_be_cancelled(tmp_path: Path, browser: Browser) -> None:
    with _designer(tmp_path, browser) as (page, _):
        _add(page, "air")
        _save(page, "參數構造")
        name = page.get_by_label("名稱", exact=True)
        saved = name.input_value()
        name.fill("暫時改名")
        dialogs: list[str] = []
        def cancel(dialog: Dialog) -> None:
            dialogs.append(dialog.type)
            dialog.dismiss()
        page.on("dialog", cancel)
        with page.expect_event("dialog"):
            page.get_by_role("link", name="參數構造").click(no_wait_after=True)
        expect(name).to_have_value("暫時改名")
        with page.expect_event("dialog"):
            page.evaluate("history.back()")
        # 直接按 Chromium 關頁，讓使用者能取消；Playwright 的 page.close 會連獨立的 context 一起銷毀。
        with page.expect_event("dialog"):
            page.context.new_cdp_session(page).send("Page.close")
        expect(name).to_have_value("暫時改名")
        assert dialogs == ["beforeunload"] * 3
        name.fill(saved)
        page.get_by_role("link", name="參數構造").click()
        expect(page.locator("#tab-optimizable")).to_be_visible()
        assert len(dialogs) == 3


def test_original_wall_is_pale_in_figure(tmp_path: Path, browser: Browser) -> None:
    with _designer(tmp_path, browser) as (page, _):
        _add(page, "air")
        page.locator('[name="backing"][value="zone_wall"]').check()
        wall = page.locator('#figure g[fill="url(#mat-wall)"]')
        expect(wall).to_have_count(1)
        assert 0 < float(wall.get_attribute("opacity") or "1") < 1
        expect(wall).not_to_have_attribute("data-layer", "1")
        page.locator('[name="backing"][value="air"]').check()
        expect(wall).to_have_count(0)


def test_layer_title_tracks_element_properties(tmp_path: Path, browser: Browser) -> None:
    with _designer(tmp_path, browser) as (page, base):
        elements = page.request.get(f"{base}/api/material-elements").json()["elements"]
        row = next(el for el in elements if el["kind"] == "porous")
        _add(page, "porous")
        _fields(page, 0)
        head = page.locator(".layer-select")
        expect(head).to_contain_text(row["name"])
        page.locator('[name="thickness_m"]').fill("73")
        expect(head).to_contain_text(row["name"])
        expect(head).to_contain_text("73 mm")
        page.locator('[name="flow_resistivity_pa_s_m2"]').fill("32417")
        expect(head).not_to_contain_text(row["name"])
        expect(head).to_contain_text("多孔層")


def test_own_construction_can_be_duplicated(tmp_path: Path, browser: Browser) -> None:
    with _designer(tmp_path, browser) as (page, base):
        page.get_by_label("名稱", exact=True).fill("我的襯層")
        _add(page, "air")
        _save(page, "參數構造")
        copy = page.get_by_role("button", name="複製一份")
        expect(copy).to_be_visible()
        _fields(page, 0)
        page.locator('[name="thickness_m"]').fill("70.9")
        original, = _stored(page, base).values()
        copy.click()
        expect(page.get_by_label("名稱", exact=True)).to_have_value("我的襯層（複本）")
        _save(page, "參數構造")
        entries = _stored(page, base)
        assert entries[original["id"]] == original
        assert entries[original["id"] + "_copy"]["notes"] == original["notes"]
        assert entries[original["id"] + "_copy"]["layers"][0]["thickness_m"] == .0709


def test_old_fixed_multilayer_copies_layers_backing_and_model(tmp_path: Path, browser: Browser) -> None:
    with _designer(tmp_path, browser) as (page, base):
        old = _legacy_copy(page, base)
        _save(page, "參數構造")
        copy = _stored(page, base)["loom_panel_copy"]
        assert copy["layers"] == old["material"]["layers"]
        assert copy["backing"] == old["material"]["backing"]
        assert copy["provenance"] == "model"


@pytest.mark.parametrize("kind", ["perforated", "slotted"])
def test_plate_cloth_matches_first_template_and_title(tmp_path: Path, browser: Browser, kind: str) -> None:
    with _designer(tmp_path, browser) as (page, base):
        elements = page.request.get(f"{base}/api/material-elements").json()["elements"]
        template = next(row["first_layer"] for row in elements if row["kind"] == kind)
        _add(page, kind)
        _fields(page, 0)
        cloth = page.get_by_label("背面貼布", exact=True)
        assert cloth.is_visible() and cloth.is_checked()
        title = page.locator(".layer-select")
        mass = page.locator('[name="backing_fabric_surface_mass_kg_m2"]')
        flow = page.locator('[name="backing_fabric_flow_resistance_pa_s_m"]')
        assert float(mass.input_value()) == template["backing_fabric_surface_mass_kg_m2"]
        assert float(flow.input_value()) == template["backing_fabric_flow_resistance_pa_s_m"]
        mass.fill("")
        flow.fill("")
        expect(title).to_contain_text("（背面貼布）")
        mass.fill("0.218")
        flow.fill("381")
        expect(title).to_contain_text("（背面貼布）")
        cloth.uncheck()
        expect(title).not_to_contain_text("（背面貼布）")
        expect(mass).to_be_hidden()
        expect(flow).to_be_hidden()
        cloth.check()
        expect(title).to_contain_text("（背面貼布）")
        expect(mass).to_have_value("0.218")
        expect(flow).to_have_value("381")


@pytest.mark.parametrize(("provenance", "word"), [("literature", "文獻"), ("measured", "實測")])
def test_inplace_source_recorded_once(tmp_path: Path, browser: Browser, provenance: str, word: str) -> None:
    with _designer(tmp_path, browser) as (page, base):
        entry = construction(id="woven", provenance=provenance, source="自編織物筆記", notes=["保留這行"])
        assert page.request.put(f"{base}/api/materials/woven", data=entry).ok
        page.goto(f"{base}/materials/designer?id=woven")
        page.get_by_text("出處與註記", exact=True).click()
        page.get_by_label("出處文字").fill("自編修改說明")
        page.get_by_label("網址").fill("https://example.org/revision")
        for _ in range(2):
            _save(page, "參數構造")
            saved = _stored(page, base)["woven"]
            assert saved["provenance"] == "model" and saved["source"] == "自編修改說明"
            assert saved["notes"] == [f"原來源：{word}，出處：自編織物筆記，網址：https://example.org/test", "保留這行"]
        page.get_by_role("button", name="複製一份").click()
        _save(page, "參數構造")
        assert _stored(page, base)["woven_copy"]["notes"] == saved["notes"]


@pytest.mark.parametrize("second_collision", [False, True])
def test_id_collision_retries_once(tmp_path: Path, browser: Browser, second_collision: bool) -> None:
    original = construction(id="weave", label="自編布樣")
    with real_server(tmp_path) as base, _open(browser, "about:blank") as watched:
        page = watched.page
        assert page.request.put(f"{base}/api/materials/weave", data=original).ok
        page.goto(f"{base}/materials/designer?copy=weave")
        page.get_by_label("名稱", exact=True).fill("另一片布")
        attempts: list[str] = []

        def intercept(route: Route) -> None:
            body = route.request.post_data_json
            assert isinstance(body, dict)
            material_id = body["id"]
            assert route.request.url.endswith("/" + material_id)
            assert route.request.headers["if-none-match"] == "*"
            attempts.append(material_id)
            if len(attempts) == 1 or (second_collision and len(attempts) == 2):
                rival = construction(id=material_id, label=f"同時存入的布{len(attempts)}")
                assert page.request.put(route.request.url, data=rival).ok
            route.continue_()

        page.route("**/api/materials/*", intercept)
        page.locator("#save").click()
        message = page.locator("#save-message")
        expect(message).not_to_be_empty()
        assert len(attempts) == len(set(attempts)) == 2
        entries = _stored(page, base)
        assert entries[attempts[0]]["label"] == "同時存入的布1"
        if second_collision:
            expect(message).to_have_text("自動編號又重複了，請再按一次儲存。")
            assert entries[attempts[1]]["label"] == "同時存入的布2"
            expect(page.get_by_label("名稱", exact=True)).to_have_value("另一片布")
            _save(page, "參數構造")
            assert len(attempts) == len(set(attempts)) == 3
        else:
            expect(message).to_contain_text("已存到")
            assert entries[attempts[1]]["label"] == "另一片布"
        assert not watched.page_errors
        assert all("409" in text for text in watched.console_errors)


def test_missing_geometry_names_field_before_sending(tmp_path: Path, browser: Browser) -> None:
    with _designer(tmp_path, browser) as (page, _):
        _add(page, "micro_perforated")
        _fields(page, 0)
        pitch = page.get_by_label("用孔距＋排法", exact=True)
        pitch.check()
        puts: list[str] = []
        page.on("request", lambda req: puts.append(req.url) if req.method == "PUT" else None)
        page.locator("#save").click()
        expect(page.locator("#save-message")).to_contain_text("孔距還沒填")
        assert not puts
        page.locator('[name="hole_spacing_m"]').fill("9.1")
        page.locator('[name="hole_pattern"]').select_option("rectangular")
        page.locator("#save").click()
        expect(page.locator("#save-message")).to_contain_text("排距還沒填")
        expect(page.locator("#curve-state")).to_contain_text("排距還沒填")
        assert not puts


def test_old_plot_stays_while_waiting(tmp_path: Path, browser: Browser) -> None:
    with _designer(tmp_path, browser) as (page, _):
        _add(page, "porous")
        _fields(page, 0)
        canvas = page.locator("#curve canvas")
        expect(canvas).to_be_visible()
        previous = canvas.evaluate("c => c.toDataURL()")
        panel = page.locator(".performance")
        height = panel.evaluate("n => n.offsetHeight")
        held: list[Route] = []
        page.route("**/api/material-curve", lambda route: held.append(route))
        with page.expect_request("**/api/material-curve"):
            page.locator('[name="thickness_m"]').fill("57.8")
        expect(page.locator("#curve-state")).to_have_text("改過了，停手後重算")
        assert canvas.evaluate("c => c.toDataURL()") == previous
        assert panel.evaluate("n => n.offsetHeight") == height
        held[0].continue_()
        expect(page.locator("#curve-state")).to_be_empty()
        assert canvas.evaluate("c => c.toDataURL()") != previous


def test_save_keeps_selected_advanced_open(tmp_path: Path, browser: Browser) -> None:
    with _designer(tmp_path, browser) as (page, _):
        _add(page, "air", "porous")
        _fields(page, 1)
        _advanced(page)
        _save(page, "參數構造")
        expect(page.locator("#stack article").nth(1).locator(".layer-select")).to_have_attribute("aria-expanded", "true")
        assert page.locator(".advanced").evaluate("n => n.open")


def test_inactive_geometry_and_model_values_return(tmp_path: Path, browser: Browser) -> None:
    with _designer(tmp_path, browser) as (page, base):
        _add(page, "porous", "perforated")
        _fields(page, 1)
        area = page.get_by_label("用開孔率", exact=True)
        area.check()
        page.locator('[name="open_area"]').fill("7.3")
        spacing = page.get_by_label("用孔距＋排法", exact=True)
        spacing.check()
        page.locator('[name="hole_spacing_m"]').fill("31.2")
        pattern = page.locator('[name="hole_pattern"]')
        pattern.select_option("rectangular")
        row_pitch = page.locator('[name="hole_spacing_2_m"]')
        row_pitch.fill("19.4")
        pattern.select_option("square")
        pattern.select_option("rectangular")
        expect(row_pitch).to_have_value("19.4")
        area.check()
        expect(page.locator('[name="open_area"]')).to_have_value("7.3")
        _fields(page, 0)
        _advanced(page)
        model = page.locator('[name="model"]')
        model.select_option("wilson_2015")
        page.locator('[name="porosity"]').fill("0.89")
        page.locator('[name="tortuosity"]').fill("1.32")
        model.select_option("miki")
        _save(page, "參數構造")
        saved, = _stored(page, base).values()
        porous, plate = saved["layers"]
        assert {"porosity", "tortuosity"}.isdisjoint(porous)
        assert {"hole_pattern", "hole_spacing_m", "hole_spacing_2_m"}.isdisjoint(plate)
        model.select_option("wilson_2015")
        expect(page.locator('[name="porosity"]')).to_have_value("0.89")
        expect(page.locator('[name="tortuosity"]')).to_have_value("1.32")
        _fields(page, 1)
        spacing.check()
        expect(pattern).to_have_value("rectangular")
        expect(row_pitch).to_have_value("19.4")
        expect(page.locator('[name="hole_spacing_m"]')).to_have_value("31.2")


def test_bonded_warning_only_in_legend(tmp_path: Path, browser: Browser) -> None:
    with _designer(tmp_path, browser) as (page, base):
        entry = construction(id="bonded_weave", layers=[
            {"kind": "board", "surface_mass_kg_m2": 2.7, "resistance_pa_s_m": 17},
            {"kind": "porous", "thickness_m": .043, "flow_resistivity_pa_s_m2": 18311, "material_class": "fibrous",
             "frame_compression_modulus_pa": 27100, "frame_loss_factor": .21}])
        assert page.request.put(f"{base}/api/materials/bonded_weave", data=entry).ok
        with page.expect_response("**/api/material-curve") as response:
            page.goto(f"{base}/materials/designer?id=bonded_weave")
        reply = response.value.json()
        for key in ("sentence", "why_only"):
            words = reply["reference_only"][key]
            expect(page.locator("#curve-key")).to_contain_text(words)
            expect(page.locator("#curve-notes")).not_to_contain_text(words)
        assert page.locator("#curve-notes li").all_text_contents() == [warn["text"] for warn in reply["warnings"]]


def _tuning_sample(full: bool = True) -> dict[str, object]:
    porous: dict[str, object] = {"kind": "porous", "material_class": "fibrous", "thickness_m": .037,
                                 "flow_resistivity_pa_s_m2": 16437}
    entry: dict[str, object] = {"label": "試織棉樣", "id": "woven_sample", "kind": "optimizable",
                                "source": "試片紀錄", "provenance": "model", "porous": porous,
                                "porous_thickness": {"range_m": [.037, .073], "initial_m": .037,
                                                     "adjustable": True, "choices_m": [.037, .073]}}
    if full:
        porous.update(flow_resistivity_source="catalog", model="wilson_2015", porosity=.94, tortuosity=1.17)
        entry.update(cavity={"adjustable": True, "range_m": [0, .09], "initial_m": .027, "choices_m": None},
                     max_total_depth_m=.15, suggested_total_depth_m=.085, preset_members=["woven_trial"],
                     template_preset="woven_trial", links=["https://example.org/woven"], notes=["自編試片\n第二面"])
    return entry


def _open_sample(page: Page, base: str) -> None:
    _list_row(page, base, "optimizable", "試織棉樣").get_by_role("button", name="打開").click()
    expect(page.locator("#name")).to_have_value("試織棉樣")


@pytest.mark.parametrize("full", [False, True])
def test_tuning_untouched_preserves_file_and_request_order(tmp_path: Path, browser: Browser, full: bool) -> None:
    library = tmp_path / "materials.json"
    entry = _tuning_sample(full) | {"provenance": "literature"}
    library.write_text(json.dumps({"schema_version": SCHEMA_VERSION, "materials": [entry]}, ensure_ascii=False))
    before = library.read_bytes()
    with _designer(tmp_path, browser) as (page, base):
        original = _stored(page, base)["woven_sample"]
        _open_sample(page, base)
        expect(page.get_by_role("switch")).to_have_count(2 if full else 1)
        with page.expect_request(lambda r: r.method == "PUT") as sent:
            _save(page, "參數構造")
        assert sent.value.post_data == page.evaluate("entry => JSON.stringify(entry)", original)
        assert library.read_bytes() == before


def test_tuning_range_edit_preserves_other_fields(tmp_path: Path, browser: Browser) -> None:
    with _designer(tmp_path, browser) as (page, base):
        assert page.request.put(f"{base}/api/materials/woven_sample", data=_tuning_sample()).ok
        before = _stored(page, base)["woven_sample"]
        _open_sample(page, base)
        _fields(page, 0)
        page.get_by_label("範圍上限（mm）").fill("81")
        page.get_by_label("買得到的厚度（mm）").fill("37, 81")
        _save(page, "參數構造")
        spec = {"range_m": [.037, .081], "initial_m": .037, "adjustable": True, "choices_m": [.037, .081]}
        assert _stored(page, base)["woven_sample"] == before | {"porous_thickness": spec}


def test_tuning_new_air_range_and_copy_on_kind_change(tmp_path: Path, browser: Browser) -> None:
    with _designer(tmp_path, browser) as (page, base):
        page.locator("#name").fill("試作空腔")
        _add(page, "porous", "air")
        switches = page.get_by_role("switch")
        expect(switches.nth(1)).to_be_enabled()
        switches.nth(1).click()
        _fields(page, 1)
        page.get_by_label("範圍下限（mm）").fill("0")
        page.get_by_label("範圍上限（mm）").fill("190")
        expect(page.locator(".where")).to_be_hidden()
        expect(page.get_by_text("背後：照第三階段搜尋頁選的裝法（貼在原牆上，或貼在硬底上）；這裡的曲線先當硬底畫")).to_be_visible()
        _save(page, "參數構造")
        before, = _stored(page, base).values()
        assert before["kind"] == "optimizable" and before["cavity"]["range_m"] == [0, .19]
        _list_row(page, base, "optimizable", "試作空腔").get_by_role("button", name="打開").click()
        expect(switches.nth(0)).not_to_be_checked()
        expect(switches.nth(1)).to_be_checked()
        _fields(page, 1)
        expect(page.get_by_label("範圍上限（mm）")).to_have_value("190")
        switches.nth(1).click()
        dialogs: list[str] = []

        def confirm(dialog: Dialog) -> None:
            dialogs.append(dialog.message)
            dialog.accept()

        page.once("dialog", confirm)
        _save(page, "參數構造")
        assert dialogs == ["這一筆已經存成可最佳化材料，改成構造只能另存一份新的；原本那筆不動。"]
        entries = _stored(page, base)
        assert json.dumps(entries[before["id"]]) == json.dumps(before)
        copy = entries[before["id"] + "_copy"]
        assert copy["label"] == "試作空腔（複本）" and copy["kind"] == "construction"
        switches.nth(1).click()
        page.once("dialog", confirm)
        _save(page, "參數構造")
        assert dialogs[-1] == "這一筆已經存成構造，改成可最佳化材料只能另存一份新的；原本那筆不動。"
        assert _stored(page, base)[copy["id"]] == copy


def test_tuning_product_and_third_layer_switch_off(tmp_path: Path, browser: Browser) -> None:
    with _designer(tmp_path, browser) as (page, _):
        _add(page, "porous", "air")
        switch = page.get_by_role("switch").first
        expect(switch).to_be_enabled()
        switch.click()
        page.locator("#product").check()
        for control in page.get_by_role("switch").all():
            expect(control).to_be_disabled()
            expect(control).not_to_be_checked()
        expect(page.locator(".tune").first).to_contain_text("成品不拆，不給程式調")
        page.locator("#product").uncheck()
        _add(page, "fabric")
        for control in page.get_by_role("switch").all():
            expect(control).to_be_disabled()
            expect(control).not_to_be_checked()
        expect(page.locator(".tune").first).to_contain_text("第二批才能給程式調")


@pytest.mark.parametrize("density", [24, 48, 96])
def test_tuning_builtin_readonly_then_copy(tmp_path: Path, browser: Browser, density: int) -> None:
    with _designer(tmp_path, browser) as (page, base):
        key = f"glass_wool_{density}k"
        page.goto(f"{base}/materials#param")
        page.locator(f'[data-id="{key}"] button.open').click()
        expect(page.locator("#name")).to_be_disabled()
        expect(page.get_by_role("switch").first).to_be_disabled()
        name = page.locator("#name").input_value()
        page.get_by_role("button", name="複製到設計器").click()
        expect(page.locator("#name")).to_have_value(name + "（複本）")
        _save(page, "參數構造")
        assert _stored(page, base)[key + "_copy"]["kind"] == "optimizable"


def test_move_preserves_advanced(tmp_path: Path, browser: Browser) -> None:
    with _designer(tmp_path, browser) as (page, _):
        _add(page, "board", "porous")
        _fields(page, 1)
        _advanced(page)
        for direction in ("往房間側移", "往牆側移"):
            page.get_by_role("button", name=direction).click()
            assert page.locator(".advanced").evaluate("n => n.open")


def test_tuning_missing_frame_partner(tmp_path: Path, browser: Browser) -> None:
    with _designer(tmp_path, browser) as (page, _):
        _add(page, "porous")
        _fields(page, 0)
        _advanced(page)
        requests: list[str] = []
        page.on("request", lambda r: requests.append(r.url) if r.method == "PUT" else None)
        page.locator('[name="frame_compression_modulus_pa"]').fill("16230")
        page.locator("#save").click()
        expect(page.locator("#save-message")).to_contain_text("第 1 層的骨架損耗因數還沒填")
        assert not requests


def test_tuning_missing_field_keeps_last_curve(tmp_path: Path, browser: Browser) -> None:
    with _designer(tmp_path, browser) as (page, _):
        _add(page, "porous")
        _fields(page, 0)
        canvas = page.locator("#curve canvas")
        expect(canvas).to_be_visible()
        image = canvas.evaluate("n => n.toDataURL()")
        page.locator('[name="thickness_m"]').fill("")
        expect(page.locator("#curve-state")).to_contain_text("還沒填")
        expect(canvas).to_be_visible()
        assert canvas.evaluate("n => n.toDataURL()") == image


def test_lost_create_response_updates_same_entry_after_rename(tmp_path: Path, browser: Browser) -> None:
    with real_server(tmp_path) as base, _open(browser, f"{base}/materials/designer") as watched:
        page = watched.page
        _add(page, "porous")
        def lose_reply(route: Route) -> None:
            assert route.fetch().ok
            route.abort()
        page.route("**/api/materials/*", lose_reply, times=1)
        page.locator("#save").click()
        expect(page.locator("#save-message")).not_to_be_empty()
        first, = _stored(page, base)
        page.locator("#name").fill("斷線後改名")
        with page.expect_request(lambda r: r.method == "GET" and r.url.endswith("/api/materials")):
            _save(page, "參數構造")
        entries = _stored(page, base)
        assert list(entries) == [first] and entries[first]["label"] == "斷線後改名"


def test_tuning_invalid_choices_initial_and_total_depths_stop_put(tmp_path: Path, browser: Browser) -> None:
    with _designer(tmp_path, browser) as (page, base):
        assert page.request.put(f"{base}/api/materials/woven_sample", data=_tuning_sample()).ok
        _open_sample(page, base)
        _fields(page, 0)
        puts: list[str] = []
        page.on("request", lambda r: puts.append(r.url) if r.method == "PUT" else None)
        choices = page.get_by_label("買得到的厚度（mm）")
        for field, bad, message, good in [
            (choices, "", "買得到的厚度", "37, 73"),
            (page.locator('[name="thickness_m"]'), "38", "初值", "37"),
            (page.get_by_label("總深度上限"), "60", "總深度上限", "150"),
            (page.get_by_label("建議總深"), "170", "建議總深", "85"),
        ]:
            field.fill(bad)
            page.locator("#save").click()
            expect(page.locator("#save-message")).to_contain_text(message)
            field.fill(good)
        assert not puts


def test_switch_kind_keeps_sparse_original_bytes(tmp_path: Path, browser: Browser) -> None:
    library = tmp_path / "materials.json"
    record = json.dumps(_tuning_sample(False), ensure_ascii=False, separators=(",", ": ")).encode()
    library.write_bytes(b'{"materials":[' + record + b'],"schema_version":"' + SCHEMA_VERSION.encode() + b'"}')
    with _designer(tmp_path, browser) as (page, base):
        _open_sample(page, base)
        page.get_by_role("switch").click()
        page.once("dialog", lambda dialog: dialog.dismiss())
        page.locator("#save").click()
        assert list(_stored(page, base)) == ["woven_sample"]
        page.once("dialog", lambda dialog: dialog.accept())
        _save(page, "參數構造")
        assert _stored(page, base)["woven_sample_copy"]["kind"] == "construction"
        assert record in library.read_bytes()


def test_tuning_save_reports_server_cavity_floor_verbatim(tmp_path: Path, browser: Browser) -> None:
    with real_server(tmp_path) as base, _open(browser, "about:blank") as watched:
        page = watched.page
        entry = _tuning_sample()
        assert page.request.put(f"{base}/api/materials/woven_sample", data=entry).ok
        with page.expect_response("**/api/material-curve") as preview:
            page.goto(f"{base}/materials/designer?id=woven_sample")
        body = preview.value.request.post_data_json
        assert isinstance(body, dict) and body["kind"] == "optimizable"
        assert "初值排法" in preview.value.json()["arrangement"]
        _fields(page, 1)
        page.get_by_label("範圍下限（mm）").fill("0.5")
        before = (tmp_path / "materials.json").read_bytes()
        with page.expect_response(lambda r: r.request.method == "PUT") as result:
            page.locator("#save").click()
        assert result.value.status == 400
        expect(page.locator("#save-message")).to_have_text(result.value.json()["error"])
        assert (tmp_path / "materials.json").read_bytes() == before
        assert not watched.page_errors and all("400" in line for line in watched.console_errors)
