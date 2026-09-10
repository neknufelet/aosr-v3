#!/usr/bin/env python3
"""入口檔的規矩節必須等於重生的結果，兩份入口檔逐字相同，全檔不准超過行數上限。

決策紙 `docs/decisions/rules-section-generated-rest-handwritten.md` 的機器版，
四件事（判準與訊息全部住在產生器 `governance/render_entry.py`，這裡不重寫一份）：

1. **標記之間必須等於重生結果**——手改那一段即紅，加了卡沒重生也紅。
2. **標記要剛好一對**——缺一個、多一個、前後顛倒都紅（界線量不出來，就不知道哪一段是產物）。
3. **卡宣告的每一份入口檔都要在**，而且兩份逐字相同（第一份是原稿，其餘是它的副本）。
4. **每一份都不准超過卡上登記的行數上限。**

**為什麼判決寫在產生器裡、這裡只有三行。** 「重生比對」這件事只要有兩份實作就會漂：
一份說「一樣」、另一份說「不一樣」的時候沒人知道哪個對。所以 `--check` 那條路
（`uv run python -m governance.render_entry --check`）與這支檢查跑的是同一個
:func:`governance.render_entry.problems`、同一個 :func:`governance.render_entry.targets`，
只是入口不同。

血債 `governance-file-has-no-guard`（v2-audit/lessons.json，legacy_id L06）：那筆事故的
對策幾乎逐字寫了這道閘（「入口檔由生成器從單一來源產出，CI 做 regenerate-diff、手改生成檔
即紅；生成檔行數硬上限在生成器內 fail 而非警告」），而事故現場的入口檔一路長到四百多行、
四節摘要是手寫冒充同步——行數那一條在當下就會回紅。

已知的縫（留在卡面與這裡，不要當成漏掉）：這支檢查只驗「標記之間等於重生結果」，
管不到標記**外面**手寫段寫了什麼——寫進模型名、寫成一份進度帳本、或塞進第三類內容，
各由 no-model-names-in-entry-files、status-page-computed-not-typed、doc-size-cap 承接
（前兩張裡只有一張已經立）。
"""
from __future__ import annotations

import sys

from governance.exit_codes import run
from governance.render_entry import problems, targets

if __name__ == "__main__":
    sys.exit(
        run(
            problems,
            description="入口檔的規矩節必須等於重生結果、兩份逐字相同、不超過行數上限",
            targets=targets,
        )
    )
