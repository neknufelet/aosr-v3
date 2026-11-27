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
import json

from tests.engine._construction_cases import construction
from tests.engine._construction_preset_answers import FITTED_VPR_IDS
from tests.engine.test_gui_zone_range_browser import _cell
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
        expect(page.locator("#zone-panel")).to_contain_text("最深沒有比背後的VPR深")
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


def test_measured_backing_counts_its_base_construction_like_the_server(tmp_path: Path, browser: Browser) -> None:
    """實測表當背後（實心木門）：面板照它的底構造扣，跟伺服器同一個數。"""
    _prepare(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_search_panel(page)
        page.locator("#zone-panel-backing-pick").select_option("door_wood_solid")
        page.locator("#zone-panel-depth").fill("25")
        expect(page.locator("#zone-panel-front-room")).to_have_text("前面最多 24.9 cm")
        assert base


def test_changing_only_the_backing_rechecks_the_candidates(tmp_path: Path, browser: Browser) -> None:
    """已存最深 18 公分的由程式最佳化區，只把背後換成 VPR：候選照剩下的 8 公分重判，不沿用已存那一份。"""
    _prepare(tmp_path, {"schema_version": "aosr.zone_design.v1", "scheme_id": SAMPLE,
                        "zones": {ZONE[0]: {ZONE[1]: {"mode": "search", "allowed": ["glass_wool_48k", "none"], "max_depth_m": 0.18}}}})
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_search_panel(page)
        expect(page.locator("#zone-panel-form")).to_have_attribute("data-candidates", "checked")
        expect(page.locator("#zone-panel-list")).to_contain_text("含 4 款")
        page.locator("#zone-panel-backing-pick").select_option(VPR)
        expect(page.locator("#zone-panel-form")).to_have_attribute("data-candidates", "checked")
        expect(page.locator("#zone-panel-list")).to_contain_text("含 1 款")
        assert base


def _vpr_applied(page: Page, depth: str = "25") -> None:
    _open_search_panel(page)
    page.locator("#zone-panel-backing-pick").select_option(VPR)
    page.locator("#zone-panel-depth").fill(depth)
    expect(page.locator("#zone-panel-form")).to_have_attribute("data-candidates", "checked")
    _apply(page)


def test_depth_only_apply_keeps_the_saved_backing_material(tmp_path: Path, browser: Browser) -> None:
    """背後沒重選、只改最深：送出的那一區帶 keep_backing（伺服器照方案裡那一份留著，不從材料庫換新版）。"""
    _prepare(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _vpr_applied(page)
        _click_zone(page, "/".join(ZONE))
        page.locator("#zone-panel-depth").fill("30")
        with page.expect_request(lambda r: r.url.endswith("/api/zone-material/batch")) as sent:
            _apply(page)
        change, = json.loads(cast(str, sent.value.post_data))["changes"]
        assert change["choice"] == "backing" and change["material_id"] == VPR and change["keep_backing"] is True
        assert change["backing_wall"] == _original_wall()
        assert base


def test_switching_back_and_narrowing_in_one_apply_checks_ranges_against_the_original_wall(tmp_path: Path, browser: Browser) -> None:
    """已存背後 VPR、最深 25（前面 15）。同一次把背後換回原牆、48K 棉厚收成 20 cm：套用前範圍檢查照換回後的 25 公分判，放得下就套用。"""
    _prepare(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _vpr_applied(page)
        _click_zone(page, "/".join(ZONE))
        page.locator("#zone-panel-backing-pick").select_option("")
        page.locator(_cell(page, "glass_wool_48k", "porous", "low")).select_option("0.2")
        page.locator(_cell(page, "glass_wool_48k", "porous", "high")).select_option("0.2")
        _apply(page)
        _leave_and_save(page)
        settings, materials = _stored(tmp_path)
        saved = cast(dict[str, dict[str, object]], settings[ZONE[0]])[ZONE[1]]
        assert "backing_entry" not in saved
        assert cast(dict[str, dict[str, object]], saved["ranges"])["glass_wool_48k"]["porous_m"] == [0.2, 0.2]
        assert materials[ZONE[0]][ZONE[1]] == _original_wall()
        assert base


def test_own_rigid_construction_as_backing_reads_as_a_normal_search_zone(tmp_path: Path, browser: Browser) -> None:
    """自己組的剛背構造當背後：選得到（我組的）、套用後那一區照由程式最佳化畫，不說看不出；截圖給人看。"""
    _prepare(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        own = construction(id="lab_rigid", label="自編剛背共振板")
        assert page.request.put(f"{base}/api/materials/lab_rigid", data=own).ok
        _open_search_panel(page)
        expect(page.locator('#zone-panel-backing-pick optgroup[label="我組的"] option[value="lab_rigid"]')).to_have_count(1)
        page.locator("#zone-panel-backing-pick").select_option("lab_rigid")
        page.locator("#zone-panel-depth").fill("25")
        expect(page.locator("#zone-panel-front-room")).to_have_text("前面最多 18.7 cm")
        expect(page.locator("#zone-panel-form")).to_have_attribute("data-candidates", "checked")
        _apply(page)
        zone = page.locator(f"#zone-map [data-wall='{ZONE[0]}'][data-zone='{ZONE[1]}']")
        expect(zone).not_to_contain_text("看不出")
        expect(page.locator("#zone-page")).not_to_contain_text("看不出")
        _shot(page, "backing-own-applied.png")
        _click_zone(page, "/".join(ZONE))
        expect(page.locator("#zone-panel-backing-pick")).to_have_value("lab_rigid")
        _shot(page, "backing-own-panel.png")
