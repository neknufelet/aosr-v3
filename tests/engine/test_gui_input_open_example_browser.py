"""#1265 單元 5a：「開頁先開一份存好的範例」這一步本身（tests/engine/_input_page_steps.py::open_example_scheme）。

單元 5 起輸入頁開頁不再把沒存的範本塞進表單；只需要「開著一份方案」的考題改成先存一份跟範例一模一樣的方案、
用 ?scheme= 打開。這裡守兩種開法起點一樣：同一台伺服器上，舊開法（空資料夾開頁，表單是沒存的範本）與新步驟
打開後，頁面自己的收集函式 collect()（表單送出去的那一份）除了方案代號以外逐欄相等。資料都是範本本身，不另編。"""
from __future__ import annotations

from pathlib import Path

import pytest
from playwright.sync_api import Browser

from tests.engine._input_page_steps import open_example_scheme, open_templates
from tests.engine.test_gui_browser import _assert_quiet, _open, _serve, browser as browser

NAME = "example-copy"


@pytest.mark.parametrize("template", [None, "stage_one"])
def test_the_saved_example_opens_to_the_same_form_as_the_empty_folder_start(
        tmp_path: Path, browser: Browser, template: str | None) -> None:
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        page.wait_for_function("document.body.dataset.inputReady === 'true'")
        page.wait_for_function("!document.querySelector('main').inert && !inputBusy")
        if template is not None:
            open_templates(page)
            page.locator("#template-list").select_option(template)
            page.locator("#load-template").click()
            page.wait_for_function("document.getElementById('template-note').textContent.startsWith('已載入')")
            page.wait_for_function("!document.querySelector('main').inert && !inputBusy")
        old = page.evaluate("structuredClone(collect())")
        assert not (tmp_path / "schemes").exists() or not any((tmp_path / "schemes").iterdir())
        stored = open_example_scheme(page, base, tmp_path, NAME, template=template)
        new = page.evaluate("structuredClone(collect())")
        assert new["scheme_id"] == stored["scheme_id"] == NAME
        assert {key: value for key, value in new.items() if key != "scheme_id"} == \
            {key: value for key, value in old.items() if key != "scheme_id"}
        _assert_quiet(watched)
