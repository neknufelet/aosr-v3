"""材料編輯頁的真瀏覽器考卷（#1082 段 2-2a）：複製內建、改可最佳化材料、多層構造加刪換排序、實測表、唯讀、刪除、壞檔。

截圖不寫在這裡（考卷不准寫真 repo 以外的地方）；給人看的截圖由施工者另外照同一套動作截。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from playwright.sync_api import Browser, Page, expect

from aosr.gui.material_editor import user_library_path
from aosr.materials.construction_models import AirLayer, Assembly, PorousLayer
from aosr.materials.library import builtin_entries
from tests.engine._gui_cache import gui_startup_identity_memo as gui_startup_identity_memo
from tests.engine.test_gui_browser import RAW_VALUES, _open, _serve, browser as browser

REFUSED = re.compile(r"Failed to load resource: the server responded with a status of 400")


def _saved(page: Page) -> None:
    page.locator("#save").click()
    expect(page.locator("#editor-message")).to_have_class("ok")


def _materials(data_dir: Path) -> dict[str, dict[str, object]]:
    document = json.loads(user_library_path(data_dir).read_text(encoding="utf-8"))
    return {entry["id"]: entry for entry in document["materials"]}


def _row(page: Page, where: str, material_id: str, button: str) -> None:
    page.locator(f'#{where} tr[data-id="{material_id}"] button.{button}').click()


def test_copy_a_builtin_family_change_its_thicknesses_and_save(tmp_path: Path, browser: Browser) -> None:
    with _serve(tmp_path) as base, _open(browser, f"{base}/materials") as watched:
        page = watched.page
        expect(page.locator("#list-state")).to_have_text("內建 18 筆、我的材料 0 筆")
        _row(page, "builtin-list", "glass_wool_48k", "copy")
        expect(page.locator("#material-id")).to_have_value("glass_wool_48k_copy")
        expect(page.locator("#porous-thickness")).to_be_hidden()
        page.locator("#material-label").fill("我的 48K 棉")
        page.locator("#porous-choices").fill("0.05, 0.075, 0.1")
        page.locator("#porous-initial").fill("0.075")
        page.locator("#cavity-adjustable").uncheck()
        expect(page.locator("#cavity-low")).to_be_hidden()
        page.locator("#cavity-depth").fill("0.05")
        _saved(page)
        assert not RAW_VALUES.search(page.locator("body").inner_text())
        expect(page.locator('#user-list tr[data-id="glass_wool_48k_copy"]')).to_contain_text("我的 48K 棉")
        assert (watched.console_errors, watched.page_errors) == ([], [])
    entry = _materials(tmp_path)["glass_wool_48k_copy"]
    assert entry["porous_thickness"] == {"adjustable": True, "choices_m": [0.05, 0.075, 0.1], "initial_m": 0.075, "range_m": [0.05, 0.1]}
    assert entry["cavity"] == {"adjustable": False, "choices_m": None, "initial_m": 0.05, "range_m": [0.05, 0.05]}
    assert _porous(entry)["thickness_m"] == 0.075
    assert (entry["preset_members"], entry["template_preset"]) == ([], None)


def _porous(entry: dict[str, object]) -> dict[str, object]:
    porous = entry["porous"]
    assert isinstance(porous, dict)
    return porous


def test_layers_are_added_retyped_reordered_and_removed_and_a_refusal_leaves_the_file_alone(
    tmp_path: Path, browser: Browser,
) -> None:
    with _serve(tmp_path) as base, _open(browser, f"{base}/materials") as watched:
        page = watched.page
        page.locator("#new-fixed").click()
        page.locator("#material-id").fill("my_panel")
        page.locator("#material-label").fill("我的吸音板")
        page.locator("#provenance").select_option("model")
        page.locator("#source").fill("自己估的")
        layers = page.locator("#assembly-layers > li")
        layers.nth(0).locator('input[name="thickness_m"]').fill("0.05")
        layers.nth(0).locator('input[name="flow_resistivity_pa_s_m2"]').fill("15000")
        page.locator("#panel-assembly button.add-layer").click()
        layers.nth(1).locator('select[name="layer-kind"]').select_option("air")
        layers.nth(1).locator('input[name="thickness_m"]').fill("0.1")
        page.locator("#panel-assembly button.add-layer").click()
        layers.nth(2).locator('select[name="layer-kind"]').select_option("fabric")
        layers.nth(2).locator("button.up").click()
        layers.nth(1).locator("button.up").click()
        expect(layers.locator("strong")).to_have_text(["第 1 層", "第 2 層", "第 3 層"])
        page.locator("#save").click()
        expect(page.locator("#editor-message")).to_have_class("notice")
        expect(page.locator("#editor-message")).to_contain_text("多層構造›第 1 層›布層›面密度：必填")
        assert not user_library_path(tmp_path).exists()
        layers.nth(0).locator("button.remove").click()
        layers.nth(0).locator("button.down").click()
        layers.nth(1).locator("button.up").click()
        page.locator("#assembly-backing").select_option("air")
        _saved(page)
        assert all(REFUSED.search(text) for text in watched.console_errors) and len(watched.console_errors) == 1
        assert watched.page_errors == []
    material = _materials(tmp_path)["my_panel"]["material"]
    porous = PorousLayer(thickness_m=0.05, flow_resistivity_pa_s_m2=15000.0, flow_resistivity_source="unspecified",
                         model="komatsu", material_class="fibrous")
    expected = Assembly(layers=(porous, AirLayer(thickness_m=0.1)), backing="air")
    assert material == expected.model_dump(mode="json")


def test_a_measured_table_without_imaginary_parts_saves_as_approximate(tmp_path: Path, browser: Browser) -> None:
    with _serve(tmp_path) as base, _open(browser, f"{base}/materials") as watched:
        page = watched.page
        page.locator("#new-fixed").click()
        page.locator("#material-id").fill("tube_panel")
        page.locator("#material-label").fill("阻抗管量的板")
        page.locator("#provenance").select_option("measured")
        page.locator("#source").fill("實驗室阻抗管")
        page.locator("#fixed-kind").select_option("measured_impedance")
        expect(page.locator("#panel-assembly")).to_be_hidden()
        page.locator("#points").fill("100 950\n200, 820\n400 610")
        page.locator("#measured-source").fill("阻抗管、垂直入射、50 mm 樣品")
        _saved(page)
        expect(page.locator('#user-list tr[data-id="tube_panel"]')).to_contain_text("近似")
        expect(page.locator("#points")).to_have_value("100 950\n200 820\n400 610")
        assert (watched.console_errors, watched.page_errors) == ([], [])
    material = _materials(tmp_path)["tube_panel"]["material"]
    assert isinstance(material, dict)
    assert [row["imag_pa_s_m"] for row in material["points"]] == [None, None, None]
    assert material["base_assembly"] == {"kind": "assembly", "backing": "rigid", "layers": [{"kind": "air", "thickness_m": 0.1}]}


def test_builtins_open_read_only_and_user_materials_can_be_deleted(tmp_path: Path, browser: Browser) -> None:
    with _serve(tmp_path) as base, _open(browser, f"{base}/materials") as watched:
        page = watched.page
        _row(page, "builtin-list", "glass_wool_48k_50mm", "open")
        expect(page.locator("#editor-readonly")).to_be_visible()
        expect(page.locator("#save")).to_be_hidden()
        assert page.locator("#editor-body input:enabled, #editor-body select:enabled, #editor-body textarea:enabled").count() == 0
        assert page.locator("#editor-body button").count() == 0
        page.locator("#copy").click()
        page.locator("#material-label").fill("板 A")
        _saved(page)
        page.once("dialog", lambda dialog: dialog.accept())
        _row(page, "user-list", "glass_wool_48k_50mm_copy", "delete")
        expect(page.locator("#list-state")).to_have_text("內建 18 筆、我的材料 0 筆；已刪除「板 A」")
        expect(page.locator("#editor")).to_be_hidden()
        assert (watched.console_errors, watched.page_errors) == ([], [])
    assert _materials(tmp_path) == {}


@pytest.mark.parametrize("width", [390, 1400])
def test_an_unreadable_user_file_is_shown_and_writes_are_disabled(tmp_path: Path, browser: Browser, width: int) -> None:
    library = user_library_path(tmp_path)
    library.write_text("{ broken", encoding="utf-8")
    with _serve(tmp_path) as base, _open(browser, f"{base}/materials", viewport_width=width) as watched:
        page = watched.page
        expect(page.locator("#file-problem")).to_contain_text("讀不進來")
        expect(page.locator("#new-optimizable")).to_be_disabled()
        _row(page, "builtin-list", "glass_wool_96k", "open")
        expect(page.locator("#copy")).to_be_disabled()
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        assert (watched.console_errors, watched.page_errors) == ([], [])
    assert library.read_text(encoding="utf-8") == "{ broken"


def test_every_builtin_copied_and_saved_untouched_keeps_every_field(tmp_path: Path, browser: Browser) -> None:
    # 表單只從畫出來的格子重排材料：規格表少一格、或空白被補成預設，複本就會悄悄跟內建不一樣（2-3a 會吃到）。
    builtins = builtin_entries()
    with _serve(tmp_path) as base, _open(browser, f"{base}/materials") as watched:
        page = watched.page
        for entry in builtins:
            _row(page, "builtin-list", entry.id, "copy")
            _saved(page)
        assert (watched.console_errors, watched.page_errors) == ([], [])
    saved = _materials(tmp_path)
    for entry in builtins:
        want = entry.model_dump(mode="json")
        got = saved[f"{entry.id}_copy"]
        for name in set(want) - {"id", "label", "preset_members", "template_preset"}:
            assert got[name] == want[name], (entry.id, name)


def test_constant_impedance_then_reopen_edit_and_save_in_place(tmp_path: Path, browser: Browser) -> None:
    with _serve(tmp_path) as base, _open(browser, f"{base}/materials") as watched:
        page = watched.page
        page.locator("#new-fixed").click()
        page.locator("#save").click()
        expect(page.locator("#editor-message")).to_contain_text("數字從哪來：必填")
        page.locator("#material-id").fill("hard_wall")
        page.locator("#material-label").fill("硬牆")
        page.locator("#provenance").select_option("literature")
        page.locator("#source").fill("混凝土牆，文獻值")
        page.locator("#fixed-kind").select_option("constant_impedance")
        page.locator("#impedance").fill("41000")
        _saved(page)
        page.locator("#close").click()
        _row(page, "user-list", "hard_wall", "open")
        expect(page.locator("#material-id")).to_be_disabled()
        expect(page.locator("#impedance")).to_have_value("41000")
        page.locator("#impedance").fill("82000")
        page.locator("#material-label").fill("很硬的牆")
        _saved(page)
        expect(page.locator('#user-list tr[data-id="hard_wall"]')).to_contain_text("很硬的牆")
        assert all(REFUSED.search(text) for text in watched.console_errors) and len(watched.console_errors) == 1
        assert watched.page_errors == []
    entry = _materials(tmp_path)["hard_wall"]
    assert (entry["label"], entry["material"]) == ("很硬的牆", {"kind": "constant_impedance", "impedance_pa_s_m": 82000.0})


def test_fixed_thickness_without_cavity_and_a_three_column_table(tmp_path: Path, browser: Browser) -> None:
    with _serve(tmp_path) as base, _open(browser, f"{base}/materials") as watched:
        page = watched.page
        page.locator("#new-optimizable").click()
        page.locator("#material-id").fill("slab")
        page.locator("#provenance").select_option("literature")
        page.locator("#source").fill("型錄")
        page.locator('#porous-fields input[name="flow_resistivity_pa_s_m2"]').fill("9000")
        page.locator("#porous-adjustable").uncheck()
        expect(page.locator("#porous-choices")).to_be_hidden()
        page.locator("#porous-thickness").fill("0.1")
        page.locator("#cavity-present").uncheck()
        expect(page.locator("#cavity-adjustable")).to_be_hidden()
        page.locator("#save").click()
        expect(page.locator("#editor-message")).to_contain_text("可最佳化材料至少要有一格能調")
        page.locator("#cavity-present").check()
        _saved(page)
        page.locator("#new-fixed").click()
        page.locator("#material-id").fill("tube_full")
        page.locator("#provenance").select_option("measured")
        page.locator("#source").fill("阻抗管")
        page.locator("#fixed-kind").select_option("measured_impedance")
        page.locator("#points").fill("100 950 -320\n200 820 -210")
        page.locator("#measured-source").fill("阻抗管、垂直入射")
        _saved(page)
        expect(page.locator('#user-list tr[data-id="tube_full"]')).to_contain_text("實測")
        assert all(REFUSED.search(text) for text in watched.console_errors) and len(watched.console_errors) == 1
        assert watched.page_errors == []
    saved = _materials(tmp_path)
    assert saved["slab"]["porous_thickness"] == {"adjustable": False, "choices_m": None, "initial_m": 0.1, "range_m": [0.1, 0.1]}
    assert _porous(saved["slab"])["thickness_m"] == 0.1
    assert saved["slab"]["cavity"] == {"adjustable": True, "choices_m": None, "initial_m": None, "range_m": [0.0, 0.2]}
    material = saved["tube_full"]["material"]
    assert isinstance(material, dict)
    assert [row["imag_pa_s_m"] for row in material["points"]] == [-320.0, -210.0]


def test_renaming_only_keeps_a_single_product_thickness_and_a_multi_line_note(tmp_path: Path, browser: Browser) -> None:
    # 不能調的厚度可以帶一個產品規格、一條註記可以有換行：表單畫不出來的這兩種，只改顯示名時要原樣存回。
    document = {"kind": "optimizable", "id": "my_wool", "label": "我的棉", "provenance": "model", "source": "型錄",
                "notes": ["第一行\n第二行"],
                "porous": {"kind": "porous", "thickness_m": 0.05, "flow_resistivity_pa_s_m2": 12000.0, "material_class": "fibrous"},
                "porous_thickness": {"adjustable": False, "range_m": [0.05, 0.05], "choices_m": [0.05], "initial_m": 0.05},
                "cavity": {"adjustable": True, "range_m": [0.0, 0.2], "initial_m": None}}
    with _serve(tmp_path) as base:
        context = browser.new_context()
        try:
            created = context.request.put(f"{base}/api/materials/my_wool", data=json.dumps(document, ensure_ascii=False),
                                          headers={"Content-Type": "application/json", "If-None-Match": "*"})
            assert created.ok, created.text()
        finally:
            context.close()
        before = _materials(tmp_path)["my_wool"]
        with _open(browser, f"{base}/materials") as watched:
            page = watched.page
            _row(page, "user-list", "my_wool", "open")
            page.locator("#material-label").fill("我的棉改名")
            _saved(page)
            assert (watched.console_errors, watched.page_errors) == ([], [])
    after = _materials(tmp_path)["my_wool"]
    assert after == before | {"label": "我的棉改名"}
