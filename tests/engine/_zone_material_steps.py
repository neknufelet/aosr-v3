"""分區頁材料列的操作，先打開對應列的小選單。"""
from playwright.sync_api import Page


def row_options(page: Page, choice: str) -> None:
    menu = page.locator(f"#zone-panel-options-{choice}")
    if menu.get_attribute("open") is None:
        menu.locator("summary").click()


def pick_fixed_row(page: Page, choice: str) -> None:
    page.locator("#zone-panel-mode-fixed").check()
    page.locator(f"#zone-panel-material-{choice}").click()
