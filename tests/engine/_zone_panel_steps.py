"""從房間方案逐區表開啟單區小框，套用後回原表。"""
from playwright.sync_api import Page, expect

from tests.engine.test_gui_zone_material_pick_browser import _add_candidate
from tests.engine.test_gui_zone_overview_tasks_browser import _apply


def open_panel(page: Page, wall: str, key: str) -> None:
    page.locator(f"#zone-goto-{wall}-{key}").click()
    expect(page.locator("#zone-panel")).to_have_attribute("data-zones", f"{wall}/{key}")


def apply_and_return(page: Page) -> None:
    _apply(page)
    page.locator("#zone-page-close").click()


def edit_candidates(page: Page, wall: str, key: str, *, remove: tuple[str, ...] = (),
                    add: tuple[str, ...] = (), only: str | None = None) -> None:
    open_panel(page, wall, key)
    for choice in remove:
        page.locator(f"#zone-panel-remove-{choice}").click()
    for choice in add:
        if choice == "none":
            page.locator("#zone-panel-addto-none").click()
        else:
            _add_candidate(page, choice)
    if only:
        page.locator(f"#zone-panel-only-{only}").click()
    apply_and_return(page)


def edit_depth(page: Page, wall: str, key: str, cm: str) -> None:
    open_panel(page, wall, key)
    page.locator("#zone-panel-depth").fill(cm)
    apply_and_return(page)
