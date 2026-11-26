"""輸入頁的選單、另存框與「從範本開一份新的」那一塊，結果抽屜，第 1 段的摺疊、「改…」與看哪一張圖，第 2 段某一面的牆面表單。鎖頁考題另由頁面程式發點擊。"""
import json
from pathlib import Path
from typing import cast
from urllib.parse import quote, urlsplit

from playwright.sync_api import Locator, Page, expect


def scheme_open_button(page: Page, name: str) -> Locator:
    close_results(page)
    if page.locator("#scheme-menu").is_hidden():
        page.locator("#scheme-menu-button").click()
    row = page.locator(f'[data-scheme="{name}"] .home-open')
    if row.is_hidden() and page.locator("#all-schemes").is_visible():
        page.locator("#all-schemes").click()
    return row


def open_scheme(page: Page, name: str) -> None:
    # 先等方案清單讀到這一份：存檔、另存之後清單是解鎖後才重讀的，選單若在清單回來前就打開，
    # 已經展開的選單不會補畫新的那一列（#1265 單元 5b 第二趟審查 6A，整套 -n 8 時逾時）。
    page.wait_for_function("name => schemeNames.includes(name)", arg=name)
    scheme_open_button(page, name).click()
    if page.locator("#leave-edits").is_visible():
        page.get_by_role("button", name="不存走", exact=True).click()


# 開頁先開的那一份存好的範例預設叫這個名字（open_example_scheme）。
EXAMPLE = "example"


def open_example_scheme(page: Page, base_url: str, data_dir: Path, name: str = EXAMPLE, *,
                        template: str | None = None) -> dict[str, object]:
    """開頁先開一份存好的範例（#1265 單元 5a）：向這一台伺服器要 /api/example（範本，``template`` 指定哪一份，
    沒給是開頁預設那份）的方案，只把方案代號換成 ``name``、原樣存成 ``data_dir/schemes/<name>.json``，
    用 ``<base_url>/?scheme=<name>`` 打開；等到表單是那一份、``main`` 沒鎖、``inputBusy``（頁面忙碌旗）為假才回來。
    單元 5 起開頁不再把沒存的範本放進表單，只需要「開著一份方案」的考題都走這一步，起點跟舊開法一樣
    （test_gui_input_open_example_browser 守）。回傳存下去的那一份。"""
    query = "" if template is None else f"?template={quote(template)}"
    reply = page.request.get(f"{base_url}/api/example{query}")
    assert reply.ok, reply.text()
    scheme = cast(dict[str, object], reply.json()["scheme"]) | {"scheme_id": name}
    path = data_dir / "schemes" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scheme, ensure_ascii=False), encoding="utf-8")
    page.goto(f"{base_url}/?scheme={quote(name)}", wait_until="networkidle")
    page.wait_for_function("document.body.dataset.inputReady === 'true'")
    # 等 openedId（真正打開的方案代號）不等 save-id：名字撞上開頁範本時開檔失敗，save-id 也會剛好相等（5a 審查第 5 點）。
    page.wait_for_function("name => openedId === name && document.getElementById('save-id').value === name", arg=name)
    page.wait_for_function("!document.querySelector('main').inert && !inputBusy")
    return scheme


def page_origin(page: Page) -> str:
    """這一頁的伺服器位址（``http://127.0.0.1:<埠>``）：頁面網址可能帶 ``?scheme=`` 之類的查詢，不能直接接 /api/…。"""
    parts = urlsplit(page.url)
    return f"{parts.scheme}://{parts.netloc}"


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
    close_results(page)
    if page.locator("#save-as-box").is_hidden():
        page.locator("#save-as-toggle").click()


def open_new_scheme_box(page: Page) -> None:
    """「從範本開一份新的」那一塊（#1265 單元 5b）：空畫面上它就在卡片裡，不用開；平常從頂條「換一個方案 ▾」打開。"""
    if page.evaluate("document.body.dataset.inputStart === 'empty'"):
        return
    close_results(page)
    if page.locator("#template-box").is_hidden():
        page.locator("#scheme-menu-button").click()
        page.locator("#choose-template").click()
    expect(page.locator("#new-scheme")).to_be_visible()


def fill_new_scheme(page: Page, name: str, template: str = "reference") -> None:
    """打開那一塊、選範本、填名字，還沒按「開一份新的」。"""
    open_new_scheme_box(page)
    page.locator(f"#new-template-{template}").check()
    page.locator("#new-scheme-id").fill(name)


def new_from_template(page: Page, name: str, template: str = "reference") -> None:
    """從範本開一份新的（#1265 單元 5b，取代「載入範本」）：選範本、取名字、按「開一份新的」；等到那一份存好、打開、
    頁面解鎖、``inputReady``（頁面載完的旗子）設好才回來。開著那一份有沒存的改動時會先問，呼叫的人自己答。"""
    fill_new_scheme(page, name, template)
    page.locator("#new-scheme-go").click()
    if page.locator("#leave-edits").is_visible():
        page.get_by_role("button", name="不存走", exact=True).click()
    # 等最後一步：打開之後還有一趟圖面檢查，檢查完才收那一塊、清名字格、在「打開」旁邊寫這一句。
    page.wait_for_function("name => openedId === name && document.getElementById('save-id').value === name"
                           " && document.body.dataset.inputReady === 'true'"
                           " && document.getElementById('open-note').textContent.includes(`「${name}」，存好了`)", arg=name)
    page.wait_for_function("!document.querySelector('main').inert && !inputBusy")


def archive_then_recreate(page: Page, data_dir: Path, name: str) -> bytes:
    """把開著的 ``name`` 從選單封存（表單留著、寫「這份方案已封存，存檔會用這個名字重新建一份」），再在磁碟上放回一份
    同名的方案檔，頁面不知道：之後從這張表單存檔會當成新建（If-None-Match: *），伺服器照規矩不蓋掉同名的那一份、回 409。
    回傳放回去那一份的位元組。"""
    path = data_dir / "schemes" / f"{name}.json"
    content = path.read_bytes()
    page.once("dialog", lambda dialog: dialog.accept())
    archive_scheme(page, name)
    expect(page.locator("#scheme-archive-notice")).to_be_visible()
    page.wait_for_function("!document.querySelector('main').inert && !inputBusy")
    page.wait_for_load_state("networkidle")
    assert not path.exists()
    path.write_bytes(content)
    return content



def open_results(page: Page) -> None:
    """頂條「算過的結果與比較 ▸」打開結果抽屜（#1265 單元 4：結果清單、比較、已封存都搬進去）；
    已經開著就不動（再叫一次不會關掉），叫完一定看得見，抽屜裡的否定斷言才不會因為整塊藏著而假綠。"""
    if page.locator("#results-panel").is_hidden():
        page.wait_for_function("document.body.dataset.inputReady === 'true'")
        page.locator("#show-results").click()
    expect(page.locator("#results-panel")).to_be_visible()


def close_results(page: Page) -> None:
    """按抽屜的「關閉 ✕」；本來就關著不動。抽屜蓋著整頁，要點頁面上別的東西前先關。"""
    if page.locator("#results-panel").is_visible():
        page.locator("#results-close").click()
        expect(page.locator("#results-panel")).to_be_hidden()


def result_row(page: Page, run_id: str) -> Locator:
    """結果抽屜裡那一筆結果（一列＝一次計算，照計算代號找）；先打開抽屜。"""
    open_results(page)
    return page.locator(f'#results-list tr[data-run-id="{run_id}"]')


def check_result(page: Page, run_id: str) -> None:
    """勾那一筆來比較（舊的「選為 A／B」：先勾的是 A、第二筆是 B）。"""
    box = result_row(page, run_id).get_by_role("checkbox")
    box.check()
    expect(box).to_be_checked()


def open_archived(page: Page) -> None:
    """打開抽屜並展開「看已封存的結果」；同一頁第二次以後它可能還開著，再點會收起來，所以只在收著時點。"""
    open_results(page)
    expect(page.locator("#archived")).to_be_visible()
    if not page.locator("#archived").evaluate("node => node.open"):
        page.locator("#archived summary").click()


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


def picked_wall(page: Page) -> str | None:
    """第 2 段牆名那一排現在選中哪一面（``x0``、``floor``…）；一面都沒選回 None。"""
    return cast(str | None, page.evaluate(
        "document.querySelector('#elevation-walls button[aria-pressed=true]')?.id.replace('elevation-wall-', '') ?? null"))


def back_to_wall(page: Page, wall: str | None) -> None:
    """打開一份方案時第 2 段會換到那一份的前牆（#1265 單元 5c）：存檔重開之後要接著改原本那一面的考題，
    照使用者會做的點回那一面（``wall`` 是重開前 ``picked_wall`` 記下的；None 不動）。"""
    if wall is None:
        return
    tab = page.locator(f"#elevation-wall-{wall}")
    if tab.get_attribute("aria-pressed") != "true":
        _unlocked(page)
        tab.click()
        expect(tab).to_have_attribute("aria-pressed", "true")


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
