"""輸入頁共用的開方案步驟。鎖頁考題另保留頁面程式自己發點擊的路徑。"""
from playwright.sync_api import Locator, Page


def select_scheme(page: Page, name: str) -> None:
    page.locator("#scheme-list").select_option(name)


def scheme_open_button(page: Page, name: str) -> Locator:
    select_scheme(page, name)
    return page.locator("#open-scheme")


def open_scheme(page: Page, name: str) -> None:
    scheme_open_button(page, name).click()


def scripted_open_scheme(page: Page, name: str) -> None:
    """分區頁或鎖頁中，模擬頁面程式換方案，不等滑鼠可以點。"""
    page.locator("#scheme-list").evaluate("(node, name) => { node.value = name; }", name)
    page.locator("#open-scheme").evaluate("node => node.click()")
