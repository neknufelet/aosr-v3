"""材料目錄用同一把尺比較成品與參數構造。用自編材料考兩個收合群組與初值剖面。"""
from __future__ import annotations

import json
import re
from pathlib import Path

from playwright.sync_api import Browser, expect

from aosr.gui.material_editor import user_library_path
from aosr.materials.library import SCHEMA_VERSION
from tests.engine._construction_gui_server import real_server
from tests.engine.test_gui_browser import _open, browser as browser
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
        expect(page.get_by_role("tab")).to_have_count(2)
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
