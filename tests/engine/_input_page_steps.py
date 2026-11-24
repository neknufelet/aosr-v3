"""輸入頁的選單、另存框與範本框，第 1 段的摺疊、「改…」與看哪一張圖，第 2 段某一面的牆面表單。鎖頁考題另由頁面程式發點擊。"""
from typing import cast

from playwright.sync_api import Locator, Page, expect


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



def _unlocked(page: Page) -> None:
    # 開檔、存檔時整頁鎖著（main.inert）：這時點標題或按鈕不會有反應，等解鎖再點。
    page.wait_for_function("!document.querySelector('main').inert")


def open_fold(page: Page, fold: str) -> None:
    """第 1 段右卡的摺疊（#1265：座標數字 coordinates-fold、進階 advanced-fold）：收著就點標題展開，開著不動。"""
    details = page.locator(f"#{fold}")
    if not details.evaluate("node => node.open"):
        _unlocked(page)
        details.locator("> summary").click()
        expect(details).to_have_attribute("open", "")


def open_fact_editor(page: Page, button: str) -> None:
    """第 1 段右卡「改…」打開那件事唯一的編輯處（#1265：edit-depth、edit-speakers、edit-furniture、edit-original）。
    再按一次會收起，所以先看開了沒。"""
    toggle = page.locator(f"#{button}")
    if toggle.get_attribute("aria-expanded") != "true":
        _unlocked(page)
        toggle.click()
        expect(toggle).to_have_attribute("aria-expanded", "true")


def show_view(page: Page, view: str) -> None:
    """平面圖上方「看哪一張圖」切到那一張（#1265：plan-xy、plan-xz、zoom-xy、zoom-xz）；四張疊在同一格，一次露一張。"""
    _unlocked(page)
    page.locator(f"#view-{view}").click()
    expect(page.locator(f"#view-{view}")).to_have_attribute("aria-pressed", "true")


def show_wall_form(page: Page, wall: str) -> None:
    """第 2 段某一面原本的牆（#1265 單元 3：牆面表單只露選中那一面，住在牆那一行「換原本的牆」打開的那一塊）：
    先點那一面的牆名鈕，再打開那一塊；已經選中、已經打開就不動。"""
    tab = page.locator(f"#elevation-wall-{wall}")
    if tab.get_attribute("aria-pressed") != "true":
        _unlocked(page)
        tab.click()
        expect(tab).to_have_attribute("aria-pressed", "true")
    open_fact_editor(page, "edit-wall-original")


# 這一格住在哪一塊收起來的地方：第 2 段某一面的牆面表單、摺疊、「改…」打開的那一塊、或沒露出來的那一張圖；都不是回 null。
_HOLDER_JS = """node => {
  const wall = node.closest('#walls .wall-editor');
  if (wall) return ['wall', wall.id.slice('wall-editor-'.length)];
  const fold = node.closest('details.fold');
  if (fold) return ['fold', fold.id];
  const area = node.closest('.fact-editor');
  if (area) return ['editor', document.querySelector(`[aria-controls="${area.id}"]`).id];
  const drawing = node.closest('#plan-drawings .drawing');
  return drawing ? ['view', drawing.dataset.view] : null;
}"""


def reveal(page: Page, selector: str) -> Locator:
    """收起來的格子（#1265 起第 1 段的座標、進階、家具、喇叭擺法、總厚度上限、原方案與另外三張圖，單元 3 起第 2 段
    每一面的牆面表單，都要先打開才點得到）：先打開它住的那一塊，回傳那一格；本來就露著的不動。"""
    target = page.locator(selector)
    holder = cast(list[str] | None, target.first.evaluate(_HOLDER_JS))
    if holder is not None:
        kind, name = holder
        {"wall": show_wall_form, "fold": open_fold, "editor": open_fact_editor, "view": show_view}[kind](page, name)
    return target
