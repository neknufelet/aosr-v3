"""從輸入頁第 2 段右欄「改這一區 ▸」開啟單區小框，套用後回輸入頁（#1265 單元 3 之二第二趟起舊做法表拆了，做法、材料都在這裡改）。"""
import re
from typing import cast

from playwright.sync_api import Page, expect

from tests.engine._zone_material_steps import pick_fixed_row, row_options
from tests.engine.test_gui_zone_material_pick_browser import _add_candidate
from tests.engine.test_gui_zone_overview_tasks_browser import _apply


def open_panel(page: Page, wall: str, key: str) -> None:
    page.locator(f"#zone-goto-{wall}-{key}").click()
    expect(page.locator("#zone-panel")).to_have_attribute("data-zones", f"{wall}/{key}")


def apply_and_return(page: Page, *, said: str | None = None, said_part: str | None = None) -> None:
    """套用、回輸入頁。said／said_part：舊做法表底下那一行（#zone-note）講的那一句，現在寫在那一頁結果列上面那一行
    （#zone-page-note），回輸入頁之前先讀（逐字比／包含）。"""
    _apply(page)
    if said is not None:
        expect(page.locator("#zone-page-note")).to_have_text(said)
    if said_part is not None:
        expect(page.locator("#zone-page-note")).to_contain_text(said_part)
    page.locator("#zone-page-close").click()


def edit_candidates(page: Page, wall: str, key: str, *, remove: tuple[str, ...] = (),
                    add: tuple[str, ...] = (), only: str | None = None, said: str | None = None) -> None:
    open_panel(page, wall, key)
    for choice in remove:
        row_options(page, choice)
        page.locator(f"#zone-panel-remove-{choice}").click()
    for choice in add:
        if choice == "none":
            row_options(page, "none")
            page.locator("#zone-panel-addto-none").click()
        else:
            _add_candidate(page, choice)
    if only:
        row_options(page, only)
        page.locator(f"#zone-panel-only-{only}").click()
    apply_and_return(page, said=said)


def edit_depth(page: Page, wall: str, key: str, cm: str) -> None:
    open_panel(page, wall, key)
    page.locator("#zone-panel-depth").fill(cm)
    apply_and_return(page)


def open_alone(page: Page, wall: str, key: str, *, one_side: bool = False) -> None:
    """只改這一區：點「改這一區 ▸」，小框只選這一區。
    one_side：左右對稱開著時小框會連另一邊一起選（左右一對），「不選這區」也是一對一起放掉；舊做法表一列只改一區、不管左右對稱，
    搬過來的考卷要只改一邊時，先在那一頁把「左右對稱」按成「關」（只改在記憶裡，存檔才寫進設計設定），再放掉另一邊。"""
    page.locator(f"#zone-goto-{wall}-{key}").click()
    panel = page.locator("#zone-panel")
    expect(panel).to_have_count(1)
    others = [zone for zone in (panel.get_attribute("data-zones") or "").split() if zone != f"{wall}/{key}"]
    if others and one_side:
        page.locator("#zone-symmetry-off").click()
        expect(page.locator("#zone-symmetry-off")).to_have_attribute("aria-pressed", "true")
        page.locator("#zone-panel-zones summary").click()
        for other in others:
            page.locator(f"#zone-panel-drop-{other.replace('/', '-')}").click()
    expect(panel).to_have_attribute("data-zones", f"{wall}/{key}")


def choose_mode(page: Page, wall: str, key: str, mode: str, *, one_side: bool = False,
                said: str | None = None, said_part: str | None = None) -> None:
    """小框把那一區改成某一種做法（由程式最佳化 search、不貼 keep_original、指定貼一種 fixed），套用、回輸入頁。
    取代舊做法表的做法下拉 `#zone-mode-<牆>-<區>`。"""
    open_alone(page, wall, key, one_side=one_side)
    page.locator(f"#zone-panel-mode-{mode}").check()
    apply_and_return(page, said=said, said_part=said_part)


def choose_fixed(page: Page, wall: str, key: str, material: str, *, one_side: bool = False,
                 said: str | None = None, said_part: str | None = None) -> None:
    """小框把那一區改成指定貼一種、選那一筆，套用、回輸入頁。
    取代舊做法表的「指定貼哪一種」下拉 `#zone-fixed-<牆>-<區>`。"""
    open_alone(page, wall, key, one_side=one_side)
    page.locator("#zone-panel-mode-fixed").check()
    pick_fixed_row(page, material)
    apply_and_return(page, said=said, said_part=said_part)


def expect_mode(page: Page, wall: str, key: str, mode: str) -> None:
    """那一區現在的做法：舊做法表做法下拉的值（`#zone-mode-<牆>-<區>` 的 value）→ 第 2 段右欄那一列的 class `zone-mode-<做法>`
    （wall_section.js 的 zoneRow 照 zoneSetting 寫）。"""
    expect(page.locator(f"#wall-zone-{wall}-{key}")).to_have_class(re.compile(rf"(^|\s)zone-mode-{mode}(\s|$)"))


def _peek(page: Page, wall: str, key: str, script: str, *, fixed: bool = False) -> object:
    """點「改這一區 ▸」打開小框、讀一樣東西，不按套用就回輸入頁（只看，不寫設計）。左右對稱開著時小框可能連另一邊一起選，
    只要選的裡面有這一區就讀。"""
    page.locator(f"#zone-goto-{wall}-{key}").click()
    panel = page.locator("#zone-panel")
    expect(panel).to_have_attribute("data-zones", re.compile(rf"(^|\s){re.escape(wall)}/{re.escape(key)}(\s|$)"))
    if fixed:
        page.locator("#zone-panel-mode-fixed").check()
    found = panel.evaluate(script)
    page.locator("#zone-page-close").click()
    expect(page.locator("#zone-page")).to_be_hidden()
    return found


def listed(page: Page, wall: str, key: str) -> list[str]:
    """由程式最佳化那一區的清單，照畫出來的順序：舊做法表那一格的方塊（`#zone-allowed-chips-<牆>-<區> .zone-chip`）
    → 小框的候選列（`.zv-candidate` 的 data-choice）。"""
    return cast(list[str], _peek(page, wall, key, "panel => [...panel.querySelectorAll('#zone-panel-list .zv-candidate')].map(row => row.dataset.choice)"))


def mode_words(page: Page, wall: str, key: str) -> list[str]:
    """做法三種的字，照畫出來的順序：舊做法表做法下拉的選項（`#zone-mode-<牆>-<區> option`）→ 小框三列的字（`.zv-opt-words`
    第一段字；後面可能接「＝其他牆面」那個標記，不算做法的名字）。"""
    return cast(list[str], _peek(page, wall, key,
                                 "panel => [...panel.querySelectorAll('.zv-opt-words')].map(node => node.firstChild.textContent)"))


def material_values(page: Page, wall: str, key: str) -> list[str]:
    """指定貼一種能選的那幾筆（值）：舊做法表的下拉（`#zone-fixed-<牆>-<區> option`）→ 小框的材料下拉。"""
    return cast(list[str], _peek(page, wall, key,
                                 "panel => [...panel.querySelectorAll('#zone-panel-material [role=radio]')].map(node => node.value)", fixed=True))


def fixed_value(page: Page, wall: str, key: str) -> str:
    """指定貼一種那一區選的是哪一筆：舊做法表的下拉（`#zone-fixed-<牆>-<區>` 的 value）→ 小框的材料下拉。"""
    return cast(str, _peek(page, wall, key, "panel => panel.querySelector('#zone-panel-material [aria-checked=true]').value"))
