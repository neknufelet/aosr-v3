"""分區頁「由程式最佳化」那一格的「背後」（#1330）：背後換成材料庫一筆（VPR 鋼板共振吸音固定當背牆），程式只調前面的層。

用自編房間（``_zone_naming_room``）在真瀏覽器操作：選背後、填最深、套用、存檔、重開，核方案那一區的材料逐欄等於選的那一筆、
設計設定記得換之前的原牆、開跑預覽的前層上限＝最深－背後總深；換回原牆逐欄放回原本那一面；最深比背後還淺就擋。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import cast

from playwright.sync_api import Browser, Page, expect

from aosr.materials.library import builtin_entries, fixed_surface
from tests.engine._construction_preset_answers import FITTED_VPR_IDS
from tests.engine._gui_cache import gui_startup_identity_memo as gui_startup_identity_memo
from tests.engine.test_gui_browser import _open, _serve, browser as browser
from tests.engine.test_gui_zone_overview_browser import SAMPLE
from tests.engine.test_gui_zone_overview_tasks_browser import (
    _apply, _click_zone, _materials_before, _open_overview, _prepare, _save, _stored,
)

VPR, = FITTED_VPR_IDS
ZONE = ("yL", "z1")  # 自編房間裡貼在原牆上的一區：換回原牆要拆得回它記的原牆
SHOTS = Path(os.environ.get("AOSR_BACKING_SHOTS", "")) if os.environ.get("AOSR_BACKING_SHOTS") else None


def _vpr_material() -> dict[str, object]:
    entry, = [item for item in builtin_entries() if item.id == VPR]
    material = fixed_surface(entry)
    assert material is not None
    return cast(dict[str, object], material.model_dump(mode="json"))


def _original_wall() -> dict[str, object]:
    material = _materials_before()[ZONE[0]][ZONE[1]]
    assert material["kind"] == "mounted_on_wall"
    return cast(dict[str, object], material["wall"])


def _open_search_panel(page: Page) -> None:
    page.set_viewport_size({"width": 1440, "height": 900})
    _open_overview(page)
    _click_zone(page, "/".join(ZONE))
    page.locator("#zone-panel-mode-search").check()
    expect(page.locator("#zone-panel-backing-pick")).to_be_visible()


def _leave_and_save(page: Page) -> None:
    """回輸入頁按儲存（方案與設計設定一起存）。"""
    page.locator("#zone-details summary").click()
    page.locator("#zone-page-close").click()
    _save(page, design=True)


def _shot(page: Page, name: str) -> None:
    if SHOTS is not None:
        page.screenshot(path=str(SHOTS / name))


def test_backing_swaps_the_zone_wall_keeps_the_original_and_counts_in_the_depth(tmp_path: Path, browser: Browser) -> None:
    _prepare(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_search_panel(page)
        expect(page.locator("#zone-panel-backing-pick")).to_have_value("")
        _shot(page, "backing-plain.png")
        page.locator("#zone-panel-backing-pick").select_option(VPR)
        expect(page.locator("#zone-panel-backing-strip")).to_be_visible()
        page.locator("#zone-panel-depth").fill("25")
        expect(page.locator("#zone-panel-deepest .zv-extra-name")).to_contain_text("最深（含 VPR")
        expect(page.locator("#zone-panel-front-room")).to_have_text("前面最多 15 cm")
        expect(page.locator("#zone-panel-form")).to_have_attribute("data-candidates", "checked")
        expect(page.locator("#zone-panel-list")).to_contain_text("也可以不貼（只有 VPR")
        _shot(page, "backing-vpr.png")
        _apply(page)
        _leave_and_save(page)

        settings, materials = _stored(tmp_path)
        saved = cast(dict[str, dict[str, object]], settings[ZONE[0]])[ZONE[1]]
        assert saved["mode"] == "search"
        assert saved["backing_entry"] == VPR
        assert saved["backing_wall"] == _original_wall()
        assert saved["max_depth_m"] == 0.25
        assert materials[ZONE[0]][ZONE[1]] == _vpr_material()
        plan = page.request.post(f"{base}/api/gradient/schemes/{SAMPLE}/plan", data={}).json()["plan"]
        assert abs(plan["max_depths"][f"{ZONE[0]}@{ZONE[1]}"] - 0.15) < 1e-12

        page.reload()
        _open_search_panel(page)
        expect(page.locator("#zone-panel-backing-pick")).to_have_value(VPR)
        expect(page.locator("#zone-panel-front-room")).to_have_text("前面最多 15 cm")
        page.locator("#zone-panel-backing-pick").select_option("")
        expect(page.locator("#zone-panel-backing-strip")).to_have_count(0)
        _apply(page)
        _leave_and_save(page)

        settings, materials = _stored(tmp_path)
        saved = cast(dict[str, dict[str, object]], settings[ZONE[0]])[ZONE[1]]
        assert "backing_entry" not in saved and "backing_wall" not in saved
        assert materials[ZONE[0]][ZONE[1]] == _original_wall()


def test_backing_deeper_than_the_depth_cannot_be_applied(tmp_path: Path, browser: Browser) -> None:
    _prepare(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_search_panel(page)
        page.locator("#zone-panel-backing-pick").select_option(VPR)
        page.locator("#zone-panel-depth").fill("8")
        expect(page.locator("#zone-panel-apply")).to_be_disabled()
        expect(page.locator("#zone-panel")).to_contain_text("最深比背後的VPR")
        assert base


def test_backing_menu_lists_the_vpr_under_resonant_and_no_porous_product(tmp_path: Path, browser: Browser) -> None:
    _prepare(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_search_panel(page)
        groups = page.locator("#zone-panel-backing-pick optgroup")
        labels = [groups.nth(index).get_attribute("label") for index in range(groups.count())]
        assert "共振吸音" in labels
        resonant = page.locator('#zone-panel-backing-pick optgroup[label="共振吸音"] option')
        assert VPR in [resonant.nth(index).get_attribute("value") for index in range(resonant.count())]
        assert page.locator('#zone-panel-backing-pick option[value="glass_wool_48k_50mm"]').count() == 0
        assert base
