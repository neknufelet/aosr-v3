"""分區頁材料列的操作，先打開對應列的小選單。"""
from playwright.sync_api import Page

# 「由程式最佳化」的清單先照勾選畫一版暫定列（`#zone-panel-form` 標 data-candidates="pending"），伺服器逐區判決
# （/api/zone-candidates）回來再整塊重畫一次（標 "checked"）：重畫換掉面板裡每一列，展開的小選單收起、量到的舊節點拆掉
# 變成全 0 的方框。慢機器上判決晚到，考卷在暫定那一版上點開或量尺寸，判決一到就落空（#1231：#1365 雲端紅的
# 名稱格方框全 0、小選單裡的鈕等 30 秒）。沒有那張表單（不是這種清單、或面板關著）就不用等。
CANDIDATES_SHOWN = "document.querySelector('#zone-panel-form')?.dataset.candidates !== 'pending'"


def wait_candidates_shown(page: Page) -> None:
    """等清單畫的是伺服器判決後那一版，再點列上的東西或量尺寸。"""
    page.wait_for_function(f"() => {CANDIDATES_SHOWN}")


def row_options(page: Page, choice: str) -> None:
    wait_candidates_shown(page)
    menu = page.locator(f"#zone-panel-options-{choice}")
    if menu.get_attribute("open") is None:
        menu.locator("xpath=..").hover()
        menu.locator(":scope > summary").click()


def pick_fixed_row(page: Page, choice: str) -> None:
    page.locator("#zone-panel-mode-fixed").check()
    page.locator(f"#zone-panel-material-{choice}").click()


def open_groups(page: Page) -> None:
    groups = page.locator("#zone-groups")
    if groups.get_attribute("open") is None:
        groups.locator(":scope > summary").click()
