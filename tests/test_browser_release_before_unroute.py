"""#1307：瀏覽器考卷扣住請求之後，要先放行、再解除攔截。

反過來寫（``page.unroute(...)`` 之後才 ``_release(...)``／``route.continue_()``）時，同一個請求會被處理兩次：
第二次報「already handled」被吞掉，Playwright（瀏覽器自動化工具）裡等它的工作卻沒收尾，之後被回收時替已關掉的分頁
送訊息，考卷工人的連線分派器當掉，收尾關分頁就永遠等——雲端整片卡到 30 分鐘被砍。本機同一題排 48 次：修前 6 份卡 5 份、修後 0 份。
這一題掃 ``tests/`` 底下每一支 .py：``page.unroute(`` 那一行之後三行內不准出現放行。"""
from __future__ import annotations

import re
from pathlib import Path

TESTS = Path(__file__).resolve().parent
RELEASE = re.compile(r"\b_release\(|\.continue_\(")  # _let_go 自己就是先放行再解除
WINDOW = 3


def _late_releases(text: str) -> list[int]:
    lines = text.splitlines()
    return [number + 1 for number, line in enumerate(lines)
            if "page.unroute(" in line and any(RELEASE.search(after) for after in lines[number + 1:number + 1 + WINDOW])]


def test_no_browser_test_releases_held_requests_after_unrouting() -> None:
    found = {str(path.relative_to(TESTS)): hits for path in sorted(TESTS.rglob("*.py"))
             if path.resolve() != Path(__file__).resolve() and (hits := _late_releases(path.read_text(encoding="utf-8")))}
    assert found == {}


def test_the_scan_catches_the_old_order_and_lets_the_new_one_pass() -> None:
    old = 'page.unroute("**/api/zone-limits")\n_release(held)\n'
    new = '_release(held)\npage.unroute("**/api/zone-limits")\n'
    loop = 'page.unroute("**/x")\nfor route in held:\n    route.continue_()\n'
    assert _late_releases(old) == [1] and _late_releases(loop) == [1] and _late_releases(new) == []
