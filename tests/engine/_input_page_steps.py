"""輸入頁的選單、另存框與範本框。鎖頁考題另由頁面程式發點擊。"""
from playwright.sync_api import Locator, Page


def scheme_open_button(page: Page, name: str) -> Locator:
    if page.locator("#scheme-menu").is_hidden():
        page.locator("#scheme-menu-button").click()
    row = page.locator(f'[data-scheme="{name}"] .home-open')
    if row.is_hidden() and page.locator("#all-schemes").is_visible():
        page.locator("#all-schemes").click()
    return row


def open_scheme(page: Page, name: str) -> None:
    scheme_open_button(page, name).click()
    if page.locator("#leave-edits").is_visible():
        page.get_by_role("button", name="不存走", exact=True).click()


def archive_scheme(page: Page, name: str) -> None:
    scheme_open_button(page, name)
    page.get_by_role("menuitem", name=f"封存 {name}", exact=True).click()


def scripted_open_scheme(page: Page, name: str) -> None:
    """分區頁或鎖頁中，模擬頁面程式換方案，不等滑鼠可以點。"""
    page.locator(f'[data-scheme="{name}"] .home-open').evaluate("""node => {
        node.click();
        const discard = document.querySelector('#leave-edits[open] [value="discard"]');
        if (discard) discard.click();
    }""")


def open_save_as(page: Page) -> None:
    if page.locator("#save-as-box").is_hidden():
        page.locator("#save-as-toggle").click()


def open_templates(page: Page) -> None:
    if page.locator("#template-box").is_hidden():
        page.locator("#scheme-menu-button").click()
        page.locator("#choose-template").click()
