"""材料目錄用同一把尺比較成品與參數構造。用自編材料考兩個收合群組與初值剖面。"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from playwright.sync_api import Browser, Route, expect

from aosr.gui.material_editor import user_library_path
from aosr.materials.library import SCHEMA_VERSION
from tests.engine._construction_gui_server import real_server
from tests.engine.test_gui_browser import _open, _serve, browser as browser
from tests.engine._construction_cases import construction
from tests.engine.test_material_library_vpr import _panel, _table, _vpr, _wool


def _sample_library(folder: Path) -> None:
    materials = [_panel(), _table(), _wool(), _vpr(),
                 _panel() | {"id": "demo109_sample", "label": "示範房吸音板"},
                 _table() | {"id": "trial_table", "label": "牆板草稿"}]
    user_library_path(folder).write_text(json.dumps({"schema_version": SCHEMA_VERSION, "materials": materials}), encoding="utf-8")


def test_material_tabs_count_folded_rows_and_show_sections_and_curves(tmp_path: Path, browser: Browser) -> None:
    _sample_library(tmp_path)
    with real_server(tmp_path) as base, _open(browser, f"{base}/materials") as watched:
        page = watched.page
        expect(page.get_by_role("tab")).to_have_count(3)
        product = page.get_by_role("tab", name=re.compile("成品"))
        parameters = page.get_by_role("tab", name=re.compile("參數構造"))
        expect(product).to_have_text(re.compile(r"成品\s*19 筆"))
        expect(parameters).to_have_text(re.compile(r"參數構造\s*5 筆"))
        expect(product).to_have_attribute("aria-selected", "true")
        legend = page.get_by_role("group", name="數據來源圖例")
        expect(legend).to_have_count(1)
        for source in ("實測", "文獻", "模型"):
            expect(legend).to_contain_text(source)
        panel = page.get_by_role("tabpanel")
        for group, material in (("示範房", "demo109_sample"), ("草稿", "trial_table")):
            row = panel.locator(f'[data-id="{material}"]')
            expect(row).to_be_hidden()
            panel.locator("summary").filter(has_text=group).click()
            expect(row).to_be_visible()
        for tab, count in ((product, 19), (parameters, 5)):
            tab.click()
            expect(legend).to_be_visible()
            rows = panel.locator("[data-id]")
            expect(rows).to_have_count(count)
            for row in rows.all():
                expect(row.get_by_role("img", name=re.compile("剖面"))).to_be_visible()
                expect(row.locator("svg.curve polyline")).to_be_visible(timeout=60_000)
            if tab == product:
                sample = panel.locator('[data-id="user_panel"]')
                expect(sample).to_contain_text("30 mm")
                expect(sample).to_contain_text("牆面")
                expect(panel.locator('[data-id="user_table"]')).to_contain_text("未知")
                expect(sample.get_by_text("user_panel", exact=True)).to_be_hidden()
                sample.get_by_text("細節", exact=True).click()
                expect(sample.get_by_text("user_panel", exact=True)).to_be_visible()
            else:
                expect(panel.locator('[data-id="user_wool_32k"]')).to_contain_text("70 mm")
                expect(panel.locator('[data-id="steel_vpr"]')).to_contain_text("81.5 mm")
        assert (watched.console_errors, watched.page_errors) == ([], [])


def test_preview_shows_flagged_bands_and_verbatim_warnings_after_reload(tmp_path: Path, browser: Browser) -> None:
    _sample_library(tmp_path)
    warnings = ["低頻使用考卷補值。", "高頻是考卷外推。"]
    reference = {"sentence": "界線以下僅供參考。", "why_only": "考卷給的原因。"}
    expected_warnings = [*warnings, reference["sentence"], reference["why_only"]]
    reply = {"status": "drawn", "frequencies_hz": [31.5, 100, 250, 500, 1000, 2000], "absorption": [.2] * 6,
             "spans": [{"kind": "model_fill", "low_hz": 31.5, "high_hz": 100},
                       {"kind": "extrapolation", "low_hz": 1000, "high_hz": 2000}],
             "warnings": [{"text": text} for text in warnings], "reference_only": {"band_hz": 250, **reference}}
    with _serve(tmp_path) as base, _open(browser, "about:blank") as watched:
        page = watched.page
        page.route("**/api/material-curve?preview", lambda route: route.fulfill(json=reply))
        page.goto(f"{base}/materials")
        row = page.locator('[data-id="user_table"]')
        for reload_list in (False, True):
            if reload_list:
                page.evaluate("loadListing()")
            badge = row.get_by_text("⚠ 有提醒", exact=True)
            expect(badge).to_be_visible()
            assert badge.get_attribute("title") == "\n".join(expected_warnings)
            curve = row.locator("svg.curve")
            band = curve.locator("rect.reference-only")
            expect(band).to_be_visible()
            assert float(band.get_attribute("width") or "0") == pytest.approx(84, abs=1)
            line = curve.locator("polyline")
            mask = line.get_attribute("mask")
            assert mask and mask.startswith("url(#") and mask.endswith(")")
            fades = curve.locator(f'{mask[4:-1]} rect[data-kind]')
            assert fades.evaluate_all("rs => rs.map(r => [r.dataset.kind, r.x.baseVal.value, r.width.baseVal.value])") == [
                ["model_fill", pytest.approx(2), pytest.approx(46, abs=1)],
                ["extrapolation", pytest.approx(139, abs=1), pytest.approx(27, abs=1)]]
            assert fades.evaluate_all("rs => rs.every(r => getComputedStyle(r).fill !== 'rgb(255, 255, 255)')")
            row.get_by_text("細節", exact=True).click()
            expect(row.get_by_role("listitem")).to_have_text(expected_warnings)


@pytest.mark.parametrize(("fields", "show_badge"), [
    pytest.param({"spans": [{"kind": "extrapolation", "low_hz": 20, "high_hz": 26},
                            {"kind": "extrapolation", "low_hz": 5900, "high_hz": 11000}],
                  "warnings": [{"level": "note", "text": "低頻段採外推。"}, {"level": "note", "text": "高頻段採外推。"}]},
                 False, id="offscreen-notes"),
    pytest.param({"spans": [{"kind": "extrapolation", "low_hz": 20, "high_hz": 71}]}, True, id="overlapping-span"),
    pytest.param({"warnings": [{"level": "strong", "text": "材料條件超出限制。"}]}, True, id="strong-warning"),
    pytest.param({"reference_only": {"band_hz": 25, "sentence": "低頻僅供參考。", "why_only": "超出適用範圍。"}},
                 True, id="reference-only"),
])
def test_preview_warning_badge(tmp_path: Path, browser: Browser, fields: dict[str, object], show_badge: bool) -> None:
    _sample_library(tmp_path)
    reply = {"status": "drawn", "frequencies_hz": [100, 500], "absorption": [.2, .5], "spans": [], "warnings": [], **fields}
    with _serve(tmp_path) as base, _open(browser, "about:blank") as watched:
        page = watched.page
        page.route("**/api/material-curve?preview", lambda route: route.fulfill(json=reply))
        page.goto(f"{base}/materials")
        row = page.locator('[data-id="user_panel"]')
        expect(row.locator("svg.curve")).to_be_visible()
        badge = row.get_by_text("⚠ 有提醒", exact=True)
        if show_badge:
            expect(badge).to_be_visible()
        else:
            expect(badge).to_have_count(0)
            row.get_by_text("細節", exact=True).click()
            expect(row.get_by_text("曲線提醒", exact=True)).to_be_visible()
            expect(row.get_by_role("listitem")).to_have_text(["低頻段採外推。", "高頻段採外推。"])


def test_timed_out_preview_allows_next_row_and_retries_after_reload(tmp_path: Path, browser: Browser) -> None:
    _sample_library(tmp_path)
    calls: list[str] = []

    def handle(route: Route) -> None:
        entry = route.request.post_data_json
        assert isinstance(entry, dict)
        material_id = entry["id"]
        calls.append(material_id)
        if material_id != "user_panel" or calls.count(material_id) > 1:
            route.fulfill(json={"status": "drawn", "frequencies_hz": [100, 500], "absorption": [.2, .5],
                                "spans": [], "warnings": []})

    with _serve(tmp_path) as base, _open(browser, "about:blank", fake_clock=True) as watched:
        page = watched.page
        page.route("**/api/material-curve?preview", handle)
        with page.expect_request(lambda req: req.url.endswith("?preview") and (req.post_data_json or {}).get("id") == "user_panel"):
            page.goto(f"{base}/materials")
        row = page.locator('[data-id="user_panel"]')
        page.clock.fast_forward(60_001)
        expect(row.locator(".preview")).to_have_text("曲線沒算完（等太久）")
        row.get_by_text("細節", exact=True).click()
        expect(row.locator("dd").filter(has_text="逾時")).to_be_visible()
        expect(page.locator('[data-id="glass_wool_48k_50mm"] svg.curve')).to_be_visible()
        page.evaluate("loadListing()")
        expect(row.locator("svg.curve")).to_be_visible()
        assert calls.count("user_panel") == 2


@pytest.mark.parametrize(("kind", "source", "saved_id", "label", "group"), [
    ("fixed", "user_panel", "demo109_copy", "自編示範吸音板", "示範房"),
    ("optimizable", "user_wool_32k", "draft_copy", "自編草稿", "草稿"),
])
def test_saved_copy_reveals_its_tab_group_and_row(tmp_path: Path, browser: Browser, kind: str, source: str,
                                                   saved_id: str, label: str, group: str) -> None:
    _sample_library(tmp_path)
    with _serve(tmp_path) as base, _open(browser, f"{base}/materials") as watched:
        page = watched.page
        page.locator(f"#tab-{kind}").click()
        page.locator(f'[data-id="{source}"] button.copy').click()
        if kind == "optimizable":
            page.locator("#name").fill(label)
            page.locator("#save").click()
            expect(page.locator("#save-message")).to_contain_text("已存到「參數構造」頁")
            page.get_by_role("link", name="參數構造").click()
            folded = page.locator(f'#panel-optimizable details[data-name="{group}"]')
            folded.locator("summary").first.click()
            expect(folded.locator(f'[data-id="{source}_copy"]')).to_contain_text(label)
            return
        page.locator("#material-id").fill(saved_id)
        page.locator("#material-label").fill(label)
        page.locator("#tab-" + ("fixed" if kind == "optimizable" else "optimizable")).click()
        page.locator("#save").click()
        message = page.locator("#editor-message")
        tab_name = "成品" if kind == "fixed" else "參數構造"
        expect(message).to_have_class("ok")
        expect(message).to_be_in_viewport(ratio=1)
        expect(message).to_have_text(f"已儲存「{label}」：在「{tab_name}」頁「{group}」")
        expect(page.locator(f"#tab-{kind}")).to_have_attribute("aria-selected", "true")
        row = page.get_by_role("tabpanel").locator(f'details[open][data-name="{group}"] [data-id="{saved_id}"]')
        expect(row.get_by_role("button", name="刪除")).to_be_visible()


def test_materials_page_lists_old_product_and_three_construction_backings(tmp_path: Path, browser: Browser) -> None:
    with _serve(tmp_path) as base, _open(browser, "about:blank") as watched:
        page = watched.page
        docs = [_panel(), *(construction(back, id=f"panel_{back}", label=label, product=back == "rigid")
                            for back, label in (("rigid", "剛背布面成品"), ("air", "空氣背布面"), ("zone_wall", "原牆布面")))]
        for doc in docs:
            response = page.request.put(f"{base}/api/materials/{doc['id']}", data=doc)
            assert response.ok, response.text()
        page.goto(f"{base}/materials")
        for tab, ids in (("fixed", ["user_panel", "panel_rigid"]), ("optimizable", ["panel_air", "panel_zone_wall"])):
            page.locator(f"#tab-{tab}").click()
            panel = page.get_by_role("tabpanel")
            for identifier in ids:
                row = panel.locator(f'[data-id="{identifier}"]')
                expect(row).to_be_visible()
                if identifier == "user_panel":
                    continue
                expect(row.locator("svg.cutaway g")).to_have_count(3 if identifier == "panel_rigid" else 2)
                if identifier == "panel_zone_wall":
                    expect(row.get_by_text("背後接原牆", exact=True)).to_be_visible()
                row.get_by_role("button", name="打開").click()
                name = page.get_by_label("名稱", exact=True)
                if tab == "fixed":
                    expect(name).to_be_disabled()
                    expect(page.locator("#save")).to_be_hidden()
                else:
                    expect(name).to_be_editable()
                    expect(page.locator("#save")).to_be_visible()
                page.goto(f"{base}/materials")
                page.locator(f"#tab-{tab}").click()
        row.get_by_role("button", name="複製").click()
        page.locator("#save").click()
        expect(page.locator("#save-message")).to_contain_text("已存到「參數構造」頁")
        page.goto(f"{base}/materials#param")
        copy = panel.locator('[data-id="panel_zone_wall_copy"]')
        expect(copy).to_be_visible()
        stored = page.request.get(f"{base}/api/materials").json()["materials"]
        entries = {item["entry"]["id"]: item["entry"] for item in stored}
        assert entries["panel_zone_wall_copy"] == {**entries["panel_zone_wall"], "id": "panel_zone_wall_copy", "label": "原牆布面（複本）",
            "source": "照「原牆布面」改的", "provenance": "model", "links": [],
            "notes": ["原來源：文獻，出處：考卷自編，網址：https://example.org/test", "自編條件"]}
        page.once("dialog", lambda dialog: dialog.accept())
        copy.get_by_role("button", name="刪除").click()
        expect(copy).to_have_count(0)
        assert watched.page_errors == []
