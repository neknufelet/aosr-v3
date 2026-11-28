"""分區頁材料列的操作，先打開對應列的小選單。"""
from playwright.sync_api import Page

# 「由程式最佳化」的清單先照勾選畫一版暫定列，伺服器逐區判決（/api/zone-candidates）回來再整塊重畫一次：重畫換掉面板裡
# 每一列，展開的小選單收起、量到的舊節點拆掉變成全 0 的方框。慢機器上判決晚到，考卷在暫定那一版上點開或量尺寸，判決一到
# 就落空（#1231：#1365 雲端紅的名稱格方框全 0、小選單裡的鈕等 30 秒）。
# 看的是頁面手上那一次判決（zone_overview.js 的 candidatePreview）回來了沒，不看表單上的標記：面板正顯示「正在核對候選」時
# 還沒有那張表單；判決回來時正在切區或分區頁收著，表單不重畫、標記留在 pending；改「最深」就地送出的新判決不改標記
# （#1231 審查 1–3）。判決回來那一刻（同一輪微任務）就重畫，等到它就等到了重畫。沒有在問（不是這種清單）就不用等。
# 沒載入分區頁程式的頁面（typeof 是 undefined）也不用等。
CANDIDATES_SHOWN = ("(typeof candidatePreview === 'undefined' || candidatePreview === null"
                    " || candidatePreview.rows !== undefined || candidatePreview.problem !== undefined)")
# 限制檢查（/api/zone-limits）晚到也會整頁重畫分區頁（zone_limits.js 的 showZoneLimits），換掉同一個面板（#1231 審查 4）。
LIMITS_SHOWN = "(typeof zoneLimitsShown === 'undefined' || zoneLimitsShown === zoneLimitsGeneration)"


def wait_candidates_shown(page: Page) -> None:
    """等清單畫的是伺服器判決後那一版、最新的限制檢查也畫上去了，再點列上的東西或量尺寸。"""
    page.wait_for_function(f"() => {CANDIDATES_SHOWN} && {LIMITS_SHOWN}")


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
