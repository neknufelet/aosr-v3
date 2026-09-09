"""狀態頁：從版控與 GitHub 現算出「做到哪」，產一份 HTML。

決策紙：`docs/decisions/status-page-not-in-main.md`（狀態由機器現算、算出來的頁不進主線）、
`docs/decisions/status-page-on-github-pages.md`（那一頁掛 GitHub Pages 給固定網址）。

這個套件**不是**規矩卡的檢查程式（那些住在 `governance/checks/`），所以它沒有卡、
也不在 verify 那個 job 的判決裡。它借用的只有離開碼約定 `governance/exit_codes.py`：
0 算出來了、2 算不出來（外部工具不在、GitHub 回不了話、資料形狀看不懂）。
**沒有 1**——這支程式不下判決，只轉述。

四支各管一件事：`model.py` 是頁面要的那些格子（凍結的資料類別）、`collect.py` 去問
版控與 GitHub、`render.py` 只把資料變成 HTML（不碰網路、不碰子程序，所以測得動）、
`build_status.py` 是命令列入口。
"""
