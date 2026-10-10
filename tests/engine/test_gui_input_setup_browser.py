"""家具和喇叭編輯器的真瀏覽器考卷；答案手寫、資料只存臨時目錄。"""
from pathlib import Path
from typing import cast

import pytest
from playwright.sync_api import Browser, Page

from aosr.reporting.scheme import Scheme
from tests.engine import _furniture_cases as cases
from tests.engine._gui_input_answers import CABINETS, FURNITURE
from tests.engine._gui_cache import gui_startup_identity_memo as gui_startup_identity_memo
from tests.engine.test_gui_browser import LONG_DECIMAL, _assert_text_is_formatted, _open, _serve, browser as browser
from tests.engine.test_gui_furniture_problem_text import assert_chinese_lines


def _filled(page: Page, selector: str, value: str | bool) -> None:
    with page.expect_response(lambda response: response.url.endswith("/api/input-edit")) as response:
        if isinstance(value, bool):
            page.locator(selector).set_checked(value)
        else:
            page.locator(selector).select_option(value)
    assert response.value.ok
    page.wait_for_function("!document.getElementById('input-setup').hasAttribute('aria-busy')")


def _save_readback(page: Page, name: str) -> dict[str, object]:
    expected = page.evaluate("collect()") | {"scheme_id": name}
    page.locator("#save-as-id").fill(name)
    with page.expect_response(lambda response: response.url.endswith(f"/api/schemes/{name}")
                              and response.request.method == "PUT") as response:
        page.locator("#save-as").click()
    assert response.value.ok, response.value.text()
    saved = page.request.get(f"{page.url.rstrip('/')}/api/schemes/{name}").json()["scheme"]
    assert saved == Scheme.model_validate(expected).model_dump(mode="json")
    page.locator("#scheme-list").select_option(name)
    page.locator("#open-scheme").click()
    page.wait_for_function("name => document.getElementById('save-id').value === name", arg=name)
    return cast(dict[str, object], saved)


def test_edit_furniture_mount_cloud_coordinates_and_roundtrip(browser: Browser, tmp_path: Path) -> None:
    with _serve(tmp_path) as url, _open(browser, url) as watched:
        page = watched.page
        _filled(page, "#furniture-sofa", True)
        _filled(page, "#furniture-table-kind", "desk")
        _filled(page, "#use-speaker-setup", True)
        _filled(page, "#speaker-mount", "desk")
        _filled(page, "#furniture-cloud", True)
        page.wait_for_function("document.getElementById('height-left').textContent.includes('0.955 m')")
        assert page.locator("#height-left").inner_text() == (
            "左聲道喇叭 z 座標：高度 1.2 m 跟擺法推出值不同：桌面頂 0.75 m＋聲學中心離箱底 0.205 m＝0.955 m")
        assert_chinese_lines(page.locator("#height-left").inner_text().splitlines())
        assert page.locator("#speaker-left-z").is_enabled()
        assert page.locator("#speaker-left-z").input_value() == "1.2"
        page.locator("#speaker-left-z").fill("0.955")
        page.locator("#speaker-right-z").fill("0.955")
        page.wait_for_function("document.getElementById('height-left').textContent === ''")
        assert page.locator("#furniture-table-coordinate").inner_text() == "房間底面中心：x 2.375、y 1.9、z 0.72 公尺"
        saved = _save_readback(page, "furnished")
        expected_cloud = FURNITURE["ceiling_cloud"] | {"placement": {"bottom_center_m": [2.1, 1.9, 2.85], "yaw_deg": 90}}
        assert saved["furniture"] == [expected_cloud, FURNITURE["sofa"], FURNITURE["desk"]]
        # 每格都能改；材質選單只准用表那一組，平面角不露出。
        assert page.locator("#furniture-sofa-material option").evaluate_all("nodes => nodes.map(n => n.textContent)") == ["布面", "皮面"]
        assert page.locator("#furniture-table-material option").evaluate_all("nodes => nodes.map(n => n.textContent)") == ["木質", "玻璃"]
        assert page.locator("#furniture-cloud-material option").evaluate_all("nodes => nodes.map(n => n.textContent)") == ["木質", "吸音天雲"]
        assert page.locator('#input-setup input[id*="yaw"], #input-setup select[id*="yaw"]').all() == []
        # 選完以後尺寸、材質、相對位置及天雲房間座標都可覆寫，存回逐欄相等。
        for role, values in {
            "sofa": {"width_m": "1.7", "depth_m": "0.7", "height_m": "0.6", "forward_m": "0.1", "left_m": "0.1", "bottom_height_m": "0"},
            "table": {"width_m": "1.5", "depth_m": "0.8", "height_m": "0.04", "forward_m": "0.9", "left_m": "0.1", "bottom_height_m": "0.71"},
            "cloud": {"width_m": "1.4", "depth_m": "1.7", "height_m": "0.06", "x": "2.2", "y": "2", "z": "2.8"},
        }.items():
            for key, value in values.items():
                page.locator(f"#furniture-{role}-{key}").fill(value)
        page.locator("#furniture-sofa-material").select_option("leather")
        page.locator("#furniture-table-material").select_option("glass")
        page.locator("#furniture-cloud-material").select_option("wood")
        edited = _save_readback(page, "custom-furniture")
        items = {item["furniture_id"]: item for item in cast(list[dict[str, object]], edited["furniture"])}
        assert items["table"] == {"furniture_id": "table", "kind": "desk", "material": "glass",
                                  "width_m": 1.5, "depth_m": 0.8, "height_m": 0.04,
                                  "placement": {"forward_m": 0.9, "left_m": 0.1, "bottom_height_m": 0.71, "yaw_deg": 0}}
        assert items["cloud"]["placement"] == {"bottom_center_m": [2.2, 2, 2.8], "yaw_deg": 90}
        _assert_text_is_formatted(page)
        assert watched.page_errors == []
        assert all("422" in line for line in watched.console_errors)


def test_representative_lock_flag_and_kind_switch_height_preserved(browser: Browser, tmp_path: Path) -> None:
    with _serve(tmp_path) as url, _open(browser, url) as watched:
        page = watched.page
        _filled(page, "#use-speaker-setup", True)
        for kind, fields in CABINETS.items():
            if kind != "bookshelf":
                _filled(page, "#speaker-kind", kind)
            for key, value in fields.items():
                field = page.locator(f"#cabinet-{key}")
                assert field.is_disabled()
                assert float(field.input_value()) == value
            assert page.locator("#speaker-left-z").input_value() == "1.2"
        _filled(page, "#representative", False)
        assert page.locator("#representative-state").inner_text() == "實際型號"
        for key in CABINETS["floorstanding"]:
            assert page.locator(f"#cabinet-{key}").is_enabled()
        page.locator("#cabinet-acoustic_center_above_bottom_m").fill("1.2")
        page.locator("#cabinet-height_m").fill("1.3")
        saved = _save_readback(page, "actual")
        assert cast(dict[str, object], saved["speaker_setup"])["representative"] is False
        assert watched.page_errors == []


def test_without_table_cannot_choose_desk_and_stand_hint_does_not_block_save(browser: Browser, tmp_path: Path) -> None:
    with _serve(tmp_path) as url, _open(browser, url) as watched:
        page = watched.page
        _filled(page, "#use-speaker-setup", True)
        assert page.locator('#speaker-mount option[value="desk"]').all() == []
        page.locator("#speaker-left-z").fill("1.21")
        page.wait_for_function("document.getElementById('stand-height-hint').textContent !== ''")
        assert page.locator("#stand-height-hint").inner_text() == "搜尋只用一個高度，兩支要一樣"
        _save_readback(page, "different-height")
        _filled(page, "#furniture-table-kind", "desk")
        _filled(page, "#speaker-mount", "desk")
        _filled(page, "#furniture-table-kind", "none")
        assert page.locator("#speaker-mount").input_value() == "stand"
        assert page.locator("#speaker-left-z").input_value() == "1.21"
        assert watched.page_errors == []


def test_readonly_extra_item_survives_browser_collect_and_save(browser: Browser, tmp_path: Path) -> None:
    extra = cases.relative_item(furniture_id="舊座椅", kind="chair")
    document = cases.document(extra, cases.cloud_item(furniture_id="另一片天雲")) | {"scheme_id": "old"}
    (tmp_path / "schemes").mkdir()
    (tmp_path / "schemes" / "old.json").write_text(Scheme.model_validate(document).model_dump_json())
    with _serve(tmp_path) as url, _open(browser, url) as watched:
        page = watched.page
        page.locator("#scheme-list").select_option("old")
        page.locator("#open-scheme").click()
        page.wait_for_function("document.getElementById('save-id').value === 'old'")
        page.wait_for_function("document.getElementById('readonly-furniture').textContent.includes('舊座椅')")
        assert "這一頁不能改" in page.locator("#readonly-furniture").inner_text()
        assert_chinese_lines(page.locator("#readonly-furniture").inner_text().splitlines())
        _filled(page, "#furniture-table-kind", "desk")
        saved = _save_readback(page, "with-extra")
        items = {item["furniture_id"]: item for item in cast(list[dict[str, object]], saved["furniture"])}
        assert items["舊座椅"] == extra and items["另一片天雲"] == cast(list[dict[str, object]], document["furniture"])[1]
        assert watched.page_errors == []


@pytest.mark.parametrize("shortcut,kinds,ear,up,down", [
    ("working", ["desk"], 1.2, 1.3, 1.1), ("listening", ["sofa", "coffee_table"], 1.05, 1.15, 0.95),
])
def test_shortcut_browser_and_saved_fields(browser: Browser, tmp_path: Path, shortcut: str,
                                         kinds: list[str], ear: float, up: float, down: float) -> None:
    with _serve(tmp_path) as url, _open(browser, url) as watched:
        page = watched.page
        with page.expect_response(lambda response: response.url.endswith("/api/input-edit")) as response:
            page.locator(f"#shortcut-{shortcut}").click()
        assert response.value.ok
        page.wait_for_function("!document.getElementById('input-setup').hasAttribute('aria-busy')")
        saved = _save_readback(page, shortcut)
        assert saved["furniture"] == [FURNITURE[kind] for kind in kinds]
        points = cast(dict[str, list[dict[str, object]]], saved["receiver_set"])["points"]
        assert {point["receiver_id"]: cast(list[float], point["position_m"])[2] for point in points} == pytest.approx(
            {"main": ear, "front": ear, "back": ear, "left": ear, "right": ear, "up": up, "down": down})
        assert "shortcut" not in saved and "ear_height_m" not in saved and "mode" not in saved
        # inner_text 不包含輸入框的值；截圖發現快捷平移的浮點尾差會漏過原文字檢查。
        values = page.locator('input[type="number"]').evaluate_all("nodes => nodes.map(n => n.value).join('\\n')")
        assert not LONG_DECIMAL.findall(values), values
        _assert_text_is_formatted(page)
        assert watched.page_errors == []
