#!/usr/bin/env python3
"""入口檔的規矩節必須等於重生的結果，AGENTS.md 是指路牌，生成段不准留佔位，不准超過行數上限。

決策紙 `docs/decisions/rules-section-generated-rest-handwritten.md` 的機器版，
五件事（判準與訊息全部住在產生器 `governance/render_entry.py`，這裡不重寫一份）：

1. **標記之間必須等於重生結果**——手改那一段即紅，加了卡沒重生也紅。
2. **標記要剛好一對**——缺一個、多一個、前後顛倒都紅（界線量不出來，就不知道哪一段是產物）。
3. **卡宣告的每一份入口檔都要在**，而且第二份起是指路牌——內容逐字等於卡上登記的
   ``pointer_text``，多一字少一字都紅（2026-09-10 老闆把「逐字副本」改成這一條，理由與
   代價寫在卡的 [settings] 上面）。
4. **生成段裡不准出現卡上登記的佔位字樣**（`{{`、TODO 那幾種），標記外面的手寫段不管。
   這一條刻意不被第 1 條蓋住：佔位寫在某張卡的 `human` 裡，重生出來的那一段本來就帶著它，
   逐字比對照樣回綠，所以來源（重生結果）與產物（檔裡那一段）兩邊都咬。
   2026-09-10 老闆拍板把暫緩卡 `derived-content-rendered-not-handwritten` 併進來的就是這顆牙。
5. **每一份都不准超過卡上登記的行數上限。**（每一行多寬是另一張卡的事，見下面的分工。）

**為什麼判決寫在產生器裡、這裡只有三行。** 「重生比對」這件事只要有兩份實作就會漂：
一份說「一樣」、另一份說「不一樣」的時候沒人知道哪個對。所以 `--check` 那條路
（`uv run python -m governance.render_entry --check`）與這支檢查跑的是同一個
:func:`governance.render_entry.problems`、同一個 :func:`governance.render_entry.targets`，
只是入口不同。

血債 `governance-file-has-no-guard`（v2-audit/lessons.json，legacy_id L06）：那筆事故的
對策幾乎逐字寫了這道閘（「入口檔由生成器從單一來源產出，CI 做 regenerate-diff、手改生成檔
即紅；生成檔行數硬上限在生成器內 fail 而非警告」），而事故現場的入口檔一路長到四百多行、
四節摘要是手寫冒充同步——行數那一條在當下就會回紅。

已知的縫（留在卡面與這裡，不要當成漏掉）：這支檢查只驗「標記之間等於重生結果、裡面沒有
佔位」，管不到標記**外面**手寫段寫了什麼——寫進模型名由 no-model-names-in-entry-files 咬、
寫成一份進度帳本由 status-page-computed-not-typed 咬、某一行長到讀不完由
doc-frontmatter-and-dates 的單行字元上限咬（那顆牙把這兩份入口檔一起量），三張都已經立了。
另一個縫在渲染那一步：一張卡只渲染人話的第一句，所以卡的人話第二句以後不會漂進入口檔
——no-model-names-in-entry-files 也因此看不到那一段（理由與代價寫在卡的 [settings] 註解裡）。
"""
from __future__ import annotations

import sys

from governance.exit_codes import run
from governance.render_entry import problems, targets

if __name__ == "__main__":
    sys.exit(
        run(
            problems,
            description="入口檔的規矩節必須等於重生結果、指路牌等於登記的那一段、不超過行數上限",
            targets=targets,
        )
    )
