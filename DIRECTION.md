# 方向

2026-09-09 清空重來。規矩從 `v2-audit/` 的事故資料重新長出來，不從舊 repo 遷就。決定與理由在 `docs/decisions/`。

## 手上有什麼

| 路徑 | 是什麼 |
|---|---|
| `v2-audit/lessons.json` | v2 的 66 筆事故，每筆附證據（檔案＋行號）、事故自己記的對策（`v3_countermeasure`）。原始 364 筆去重而來 |
| `v2-audit/lessons.md` | 同一份的人話摘要 |
| `blueprint/rules-436.json` | 從四份設計報告抽出、去重後的 436 條候選規則，每條對回 lessons |
| `blueprint/batch1-127.json` | 上面那 436 條裡，不需要等程式碼就做得到的 127 條 |
| `blueprint/collapse-by-*.json` | 三個工人用三種視角各自把 127 條收斂的結果（41／41／39 張） |
| `blueprint/convergence-result.json` | 合成 38 張後、對著清空前舊 repo 做的可行性＋找碴（歷史紀錄，判斷已被 `cards-38.json` 取代） |
| `blueprint/cards-38.json` | **現行版本。** 38 張卡：規格、對到的 v2 事故（含出處）、對空 repo 重判的可行性、找碴結果。`blueprint/remap_cards.py` 產生，可重跑、自驗 |
| `blueprint/first-batch-review.json` | 第一批 12 張的內容確認原始紀錄 |
| `blueprint/first-batch-order.json` | 第一批立卡順序與第一個 PR 的組成 |
| `governance/` | 規矩卡、檢查程式、必紅樣本、共用零件 |
| `src/` | 產品程式（引擎）的家：v2 根層的鏡像。**今天是空的**——排除與白名單先立起來，樹要下一個 PR 才搬。搬進來之後七張卡先不看它，再一條一條放進去咬（決策紙 `docs/decisions/engine-tree-lands-unwatched-then-rules-bite-one-by-one.md`） |
| `docs/decisions/` | 決策紙，一題一檔 |

## 驗過的數字（機器算，可重跑）

- 來源資料：v2 事故 66 筆（`v2-audit/lessons.json` 的 incidents）；候選規則去重後 436 條（`blueprint/rules-436.json`），其中不必等程式碼就做得到的 127 條（`blueprint/batch1-127.json`），收斂成 38 張卡（`blueprint/cards-38.json`）。這一行是原始資料的大小，機器算不出來、也不會再變。
- 其餘的數字全部看狀態頁（機器現算）：https://neknufelet.github.io/aosr-v3/ ——主線有幾張卡、藍圖 38 張各自的去向、暫緩的在等什麼、血債對到幾件、收據分支上有幾份收據、主線最近那幾跑有沒有留下收據。這裡不再抄一份：抄進版控的數字當天就開始漂，那正是規矩卡 status-page-computed-not-typed 在擋的事。
- 一次性的稽核事實（不是狀態、不會再算一次）：合併那步自稱引用 59 個教訓 id，回原始檔對過之後真的只有 24 個、幽靈 35 個，已沿 covers 機器重對；上游四層幽靈 0。

## 鐵律

- 一張卡一個 PR，卡自帶檢查與必紅樣本，雲端 `verify` 綠了才算。
- 只會回綠的檢查等同沒有檢查。
- 待辦不寫在 repo 的 md 裡，走 issue。
- 檔名英文，內文中文。
