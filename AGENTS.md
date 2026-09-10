# 這個 repo 的入口檔是 CLAUDE.md

開工先讀 `CLAUDE.md`：規矩、座標、驗證指令、七條習慣都在那一份。這一份只是給其他家工具的指路牌，不放內容——它由 `uv run python -m governance.render_entry` 寫出來，手改會被重生蓋掉，也會被雲端 `verify` 擋下來。

本機一行跑完 `uv run pytest && uv run ruff check`；本機只是宣稱，雲端綠了才算數。
