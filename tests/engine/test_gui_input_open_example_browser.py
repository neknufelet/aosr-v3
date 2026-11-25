"""#1265 單元 5a：「開頁先開一份存好的範例」這一步本身（tests/engine/_input_page_steps.py::open_example_scheme）。

單元 5 起輸入頁開頁不再把沒存的範本塞進表單；只需要「開著一份方案」的考題改成先存一份跟範例一模一樣的方案、
用 ?scheme= 打開。這裡守兩種開法起點一樣：同一台伺服器上，產品自己的開法（單元 5b 起空資料夾開頁是空畫面，
從那張卡「從範本開一份新的」）與這個步驟打開後，頁面自己的收集函式 collect()（表單送出去的那一份）除了方案代號
以外逐欄相等，牆改回固定阻抗時填的預設值（fixedWallDefaults）也一樣。資料都是範本本身，不另編。"""
from __future__ import annotations

from pathlib import Path

import pytest
from playwright.sync_api import Browser

from tests.engine._input_page_steps import new_from_template, open_example_scheme
from tests.engine.test_gui_browser import _assert_quiet, _open, _serve, browser as browser

NAME = "example-copy"
STATE = "[structuredClone(collect()), structuredClone(fixedWallDefaults)]"


@pytest.mark.parametrize("template", [None, "stage_one"])
def test_the_saved_example_opens_to_the_same_form_as_a_new_scheme_from_the_empty_screen(
        tmp_path: Path, browser: Browser, template: str | None) -> None:
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        page.wait_for_function("document.body.dataset.inputStart === 'empty'")
        assert not (tmp_path / "schemes").exists() or not any((tmp_path / "schemes").iterdir())
        new_from_template(page, "made-here", template or "reference")
        old, old_defaults = page.evaluate(STATE)
        assert old["scheme_id"] == "made-here"
        stored = open_example_scheme(page, base, tmp_path, NAME, template=template)
        new, new_defaults = page.evaluate(STATE)
        assert new["scheme_id"] == stored["scheme_id"] == NAME
        assert {key: value for key, value in new.items() if key != "scheme_id"} == \
            {key: value for key, value in old.items() if key != "scheme_id"}
        assert new_defaults == old_defaults
        _assert_quiet(watched)
