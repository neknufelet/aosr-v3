"""分區的單一候選清單，七題操作真瀏覽器。兩款家族與未來的窗構造只放進考卷的材料庫。"""
from collections.abc import Iterator
from pathlib import Path

import pytest
from playwright.sync_api import Browser, Page, expect
from pydantic import TypeAdapter

from aosr.gui import material_editor
from aosr.gui.gradient_plan import GradientRequest, plan_search, start_arguments
from aosr.gui.zone_design import DesignSettings
from aosr.config.material_gradient import load_material_gradient_settings
from aosr.config.paths import config_path
from aosr.reporting.scheme import Scheme
from aosr.materials.library import LibraryEntry, builtin_entries
from tests.engine._sample_material_library import SAMPLE_USER_ENTRIES
from tests.engine._construction_cases import construction
from tests.engine._gui_cache import gui_startup_identity_memo as gui_startup_identity_memo
from tests.engine._construction_gui_server import real_server
from tests.engine.test_gui_browser import _open, browser as browser
from tests.engine.test_gui_zone_material_pick_browser import _design
from tests.engine.test_gui_zone_overview_tasks_browser import _apply, _click_zone, _open_overview, _prepare
from tests.engine.test_gui_zone_picker_browser import _append_material

WOOL = "glass_wool_48k"
MEMBERS = ["glass_wool_48k_50mm", "glass_wool_48k_50mm_gap50"]
GLASS = ["glass_single_3mm", "glass_double_gap30", "side_window_glass_12mm_single", "side_window_glass_table_upper"]


@pytest.fixture
def panel(tmp_path: Path, browser: Browser, monkeypatch: pytest.MonkeyPatch) -> Iterator[Page]:
    zones: dict[str, dict[str, object]] = {"yL": {"z1": {"mode": "search", "allowed": ["none"]},
                                              "z3": {"mode": "search", "allowed": [WOOL, MEMBERS[0], "none"]}}}
    _prepare(tmp_path, _design(zones))
    _append_material(tmp_path, construction(id="exam_floor", label="架高試片", placement="floor", product=True))
    floor = next(entry for entry in SAMPLE_USER_ENTRIES if entry["id"] == "joist_floor_plank_fixed")
    _append_material(tmp_path, dict(floor, id="exam_glass_floor", label="強化玻璃地板"))
    adapter: TypeAdapter[tuple[LibraryEntry, ...]] = TypeAdapter(tuple[LibraryEntry, ...])
    windows = adapter.validate_python([
        construction(id=key, label=label, placement="door_window", backing="air", product=True, layers=layers)
        for key, label, layers in (
            (GLASS[0], "單片玻璃", [{"kind": "board", "surface_mass_kg_m2": 7.4, "resistance_pa_s_m": 0.0}]),
            (GLASS[1], "雙片玻璃", [{"kind": "board", "surface_mass_kg_m2": 7.8, "resistance_pa_s_m": 0.0},
                                  {"kind": "air", "thickness_m": 0.03},
                                  {"kind": "board", "surface_mass_kg_m2": 7.4, "resistance_pa_s_m": 0.0}]))])
    original = builtin_entries()
    old_members = next(entry.preset_members for entry in original if entry.kind == "optimizable" and entry.id == WOOL)
    replaced = set(GLASS[:2]) | (set(old_members) - set(MEMBERS))
    entries = tuple(entry.model_copy(update={"preset_members": tuple(MEMBERS)}) if entry.id == WOOL else entry
                    for entry in original if entry.id not in replaced)
    monkeypatch.setattr(material_editor, "builtin_entries", lambda: (*entries, *windows))
    with real_server(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        page.set_viewport_size({"width": 1440, "height": 900})
        listing = material_editor.MaterialStore(tmp_path).listing()
        page.route("**/api/materials", lambda route: route.fulfill(json=listing))
        _open_overview(page)
        _click_zone(page, "yL/z3")
        expect(page.locator("#zone-panel-mode-search")).to_be_checked()
        yield page
        assert watched.page_errors == [] and watched.console_errors == []


def test_row_shows_depth_curve_and_selected_edge(panel: Page) -> None:
    checked = panel.locator(f"#zone-panel-pick-{WOOL}")
    other = panel.locator("#zone-panel-pick-glass_wool_24k")
    expect(checked).to_be_checked()
    expect(other).not_to_be_checked()
    expect(panel.locator(f"#zone-panel-row-{WOOL} > .zv-description")).to_have_text("5–45 cm")
    expect(panel.locator(f"#zone-panel-row-{WOOL} .zv-spark polyline")).to_be_visible()
    style = """el => [getComputedStyle(el).backgroundColor, getComputedStyle(el).borderLeftColor,
                       getComputedStyle(el.querySelector('strong')).fontWeight]"""
    before, after = checked.evaluate(style), other.evaluate(style)
    assert before[0] == after[0] and before[1] != after[1] and int(before[2]) > int(after[2])
    expect(panel.locator("#zone-panel-pick-fabric_pet_a_gap100 .zv-description")).to_have_text("10 cm")


def test_members_return_to_group_after_family_unchecked(panel: Page, tmp_path: Path) -> None:
    detail = panel.locator(f"#zone-panel-members-{WOOL}")
    expect(detail.locator("summary")).to_have_text("含 2 款")
    detail.locator("summary").click()
    for key, depth in zip(MEMBERS, ("5 cm", "10 cm"), strict=True):
        member = detail.locator(f"[data-member={key}]")
        expect(member).to_contain_text("48K 玻璃棉")
        expect(member.locator(".zv-member-depth")).to_have_text(depth)
        expect(panel.locator(f"#zone-panel-pick-{key}")).to_have_count(0)
    expect(detail.locator("input,button,[role=checkbox]")).to_have_count(0)
    fixed = panel.locator("#zone-panel-pick-fabric_pet_a_gap100")
    fixed.scroll_into_view_if_needed()
    expect(fixed.locator(".zv-spark polyline")).to_be_visible()
    detail.scroll_into_view_if_needed()
    expect(panel.locator(f"#zone-panel-row-{WOOL} .zv-spark polyline")).to_be_visible()
    panel.screenshot(path=tmp_path / "members.png")
    panel.locator(f"#zone-panel-pick-{WOOL}").click()
    for key in MEMBERS:
        expect(panel.locator(f"[data-material-group=porous] #zone-panel-pick-{key}")).to_be_enabled()
    expect(panel.locator(f"#zone-panel-pick-{MEMBERS[0]}")).to_be_checked()
    panel.locator(f"#zone-panel-pick-{MEMBERS[1]}").click()
    expect(panel.locator(f"#zone-panel-pick-{MEMBERS[1]}")).to_be_checked()


def test_none_is_an_independent_first_checkbox(panel: Page) -> None:
    none = panel.locator("#zone-panel-pick-none")
    expect(none).to_be_checked()
    assert none.evaluate("""el => !el.closest('[data-material-group]') &&
        !!(el.compareDocumentPosition(document.querySelector('[data-material-group]')) & Node.DOCUMENT_POSITION_FOLLOWING)""")
    none.click()
    expect(none).not_to_be_checked()
    expect(panel.locator(f"#zone-panel-pick-{WOOL}")).to_be_checked()


def test_candidate_filter_counts_independent_choices(panel: Page, tmp_path: Path) -> None:
    toggle = panel.get_by_role("button", name="只看候選（2）")
    expect(toggle).to_have_text("只看候選（2）")
    toggle.click()
    expect(toggle).to_have_attribute("aria-pressed", "true")
    rows = panel.locator("#zone-panel [role=checkbox]:visible")
    assert rows.evaluate_all("els => els.map(el => el.id)") == ["zone-panel-pick-none", f"zone-panel-pick-{WOOL}"]
    expect(panel.locator(f"#zone-panel-row-{WOOL} .zv-spark polyline")).to_be_visible()
    panel.screenshot(path=tmp_path / "candidates.png")
    _assert_start_arguments(panel, [WOOL, "none"])
    panel.locator("#zone-panel-pick-none").click()
    toggle = panel.get_by_role("button", name="只看候選（1）")
    expect(rows).to_have_count(1)
    toggle.click()
    expect(panel.locator("#zone-panel-pick-none")).to_be_visible()
    panel.locator("#zone-panel-cancel").click()
    panel.locator("#zone-page-close").click()
    panel.locator("#elevation-wall-yL").click()
    expect(panel.locator("#wall-zone-what-yL-z3")).to_contain_text("從 2 種裡挑")


def test_parameters_and_axes_are_in_the_expanded_detail(panel: Page) -> None:
    options = panel.locator(f"#zone-panel-options-{WOOL}")
    expect(options.locator(".zv-range")).to_be_hidden()
    options.locator("summary").press("Enter")
    parts = options.locator(".zv-parameter")
    expect(parts).to_have_text(["棉厚 5–20 cm", "背後空腔 0–25 cm"])
    first, last = parts.first.bounding_box(), parts.last.bounding_box()
    assert first and last and first["y"] < last["y"]
    axis = options.locator(".zv-depth-axis")
    band = axis.locator("i")
    expect(axis).to_have_attribute("aria-label", "總厚 5–30 cm")
    before_width = band.evaluate("e => e.getBoundingClientRect().width")
    panel.locator("#zone-panel-depth").fill("12")
    expect(axis).to_have_attribute("aria-label", "總厚 5–12 cm")
    assert 0 < band.evaluate("e => e.getBoundingClientRect().width") < before_width
    expect(options.locator(".zv-curve-chart polyline")).to_be_visible()


def test_groups_open_by_availability_and_remember_user_choice(panel: Page) -> None:
    panel.locator("#zone-panel-cancel").click()
    _click_zone(panel, "floor/auto1")
    for mode, container in (("fixed", "material"), ("search", "picker")):
        panel.locator(f"#zone-panel-mode-{mode}").check()
        groups = panel.locator(f"#zone-panel-{container} [data-material-group]")
        expect(groups).to_have_count(4)
        for group in groups.all():
            count = int(group.locator(".zv-family-count").inner_text())
            assert group.evaluate("e => e.open") == bool(count)
        mine = groups.locator("xpath=self::*[@data-material-group='custom']")
        expect(mine.locator(".zv-family-count")).to_have_text("1")
        mine.locator(":scope > summary").click()
        panel.locator("#zone-panel-mode-keep_original").check()
        panel.locator(f"#zone-panel-mode-{mode}").check()
        assert not mine.evaluate("e => e.open")


def test_glass_windows_and_group_search_in_both_modes(panel: Page, tmp_path: Path) -> None:
    for mode in ("search", "fixed"):
        panel.locator(f"#zone-panel-mode-{mode}").check()
        find = panel.get_by_role("searchbox", name="找材料")
        surface = panel.get_by_role("group", name="門、窗、一般牆面")
        all_rows = int(surface.locator(".zv-family-count").inner_text())
        assert all_rows >= len(GLASS)
        for name in ("窗", "門"):
            find.fill(name)
            expect(surface.locator(".zv-material:visible")).to_have_count(all_rows)
            expect(panel.locator("[data-material-group=porous]")).to_be_hidden()
        find.fill("玻璃")
        for key in GLASS:
            expect(surface.locator(f"[value={key}]")).to_be_visible()
            expected = panel.evaluate("id => libraryEntry(id).label", key)
            expect(surface.locator(f"[value={key}] strong")).to_have_text(expected)
        expect(panel.locator("[data-material-group=porous] .zv-material:visible").first).to_contain_text("玻璃棉")
        find.fill("玻璃窗")
        for key in GLASS:
            expect(surface.locator(f"[value={key}]")).to_be_visible()
        find.fill("玻璃")
    for family in ("porous", "resonant"):
        panel.locator(f"[data-material-group={family}] > summary").click()
    expect(panel.get_by_role("searchbox")).to_have_value("玻璃")
    for key in GLASS:
        expect(surface.locator(f"[value={key}]")).to_be_in_viewport()
        expect(surface.locator(f"[value={key}] .zv-spark polyline")).to_be_visible()
    panel.screenshot(path=tmp_path / "glass.png")
    panel.locator("#zone-panel-cancel").click()
    _click_zone(panel, "floor/auto1")
    for mode in ("fixed", "search"):
        panel.locator(f"#zone-panel-mode-{mode}").check()
        floor = panel.get_by_role("group", name="地板", exact=True)
        total = int(floor.locator(".zv-family-count").inner_text())
        assert total > 0
        panel.get_by_role("searchbox").fill("地板")
        expect(floor.locator(".zv-material:visible")).to_have_count(total)


def _assert_start_arguments(page: Page, choices: list[str]) -> None:
    scheme = Scheme.model_validate(page.evaluate("wallDocument()"))
    design = DesignSettings.model_validate(page.evaluate("zoneDesign"))
    settings = load_material_gradient_settings(config_path("material_gradient.toml"))
    plan = plan_search(scheme, design, GradientRequest(), settings, frozenset(MEMBERS), pair=False)
    args = start_arguments(plan)
    actual = [value for flag, value in zip(args, args[1:]) if flag == "--allowed" and value.startswith("yL@z3=")]
    assert actual == ["yL@z3=" + ",".join(choices)]


def test_cross_zone_member_remains_available_only_where_needed(panel: Page) -> None:
    _click_zone(panel, "yL/z1")
    choice = panel.locator(f"#zone-panel-pick-{MEMBERS[0]}")
    expect(choice).not_to_be_checked()
    choice.click()
    panel.locator(f"#zone-panel-pick-{MEMBERS[1]}").click()
    _apply(panel)
    _click_zone(panel, "yL/z1")
    expect(choice).to_be_checked()
    panel.locator("#zone-panel-cancel").click()
    _click_zone(panel, "yL/z3")
    expect(choice).to_have_count(0)
    panel.locator(f"#zone-panel-pick-{WOOL}").click()
    expect(choice).to_be_checked()
    expect(panel.locator(f"#zone-panel-pick-{MEMBERS[1]}")).not_to_be_checked()


def test_narrowed_family_keeps_five_centimetre_member_in_search(panel: Page) -> None:
    options = panel.locator(f"#zone-panel-options-{WOOL}")
    options.locator("summary").press("Enter")
    panel.locator(f"#zone-panel-range-toggle-{WOOL}").click()
    panel.locator(f"#zone-panel-range-{WOOL}-porous-low").select_option("0.1")
    expect(panel.locator(f"#zone-panel-pick-{MEMBERS[0]}")).to_be_checked()
    expect(panel.locator("#zone-panel-candidates-only")).to_have_text("只看候選（3）")
    _apply(panel)
    _assert_start_arguments(panel, [WOOL, "fixed:" + MEMBERS[0], "none"])


def test_training_notice_uses_effective_candidates(panel: Page) -> None:
    expect(panel.locator(f"#zone-panel-pick-{WOOL}")).to_be_checked()
    hint = panel.get_by_text("這次搜尋需要重新訓練", exact=True)
    expect(hint).to_have_count(0)
    expect(panel.locator(".zone-candidate-cost")).to_have_count(0)
    panel.locator(f"#zone-panel-pick-{WOOL}").click()
    expect(hint).to_be_visible()
    detail = panel.locator("#zone-panel-candidate-info")
    expect(detail.locator(".zone-candidate-cost")).to_be_hidden()
    detail.locator("summary").click()
    expect(detail.locator(".zone-candidate-cost")).to_contain_text("放進訓練")


def test_glass_floor_has_no_window_alias(panel: Page) -> None:
    panel.locator("#zone-panel-cancel").click()
    _click_zone(panel, "floor/auto1")
    panel.locator("#zone-panel-mode-fixed").check()
    search = panel.get_by_role("searchbox")
    row = panel.locator("#zone-panel-material-exam_glass_floor")
    search.fill("玻璃")
    expect(row.locator("strong")).to_have_text("強化玻璃地板")
    search.fill("玻璃窗")
    expect(row).to_be_hidden()
