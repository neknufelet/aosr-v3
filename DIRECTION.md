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
| `docs/decisions/` | 決策紙，一題一檔 |

## 驗過的數字（機器算，可重跑）

- v2 事故 66 筆；五份報告抽出 605 條，去重 436 條；127 條今天做得到；收斂成 38 張。
- 38 張的事故對應：合併那步寫的 59 個 id 有 35 個是假的，上游四層 0 個假的，已沿 `covers` 機器重對（`cards-38.json` 的 `meta`）。
- 對空 repo 重判：18 張能立（3 張併進第一張，第一批 16 張）、20 張暫緩（分八組進 issue）。**第一批 16 張已全部立在主線（2026-09-10）**，另立 `uv-single-entrypoint`；`text-format-consistent` 砍掉。
- 內容層：改寫後 6 張能在自己對到的 v2 事故當下回紅；其餘掛的事故多半是機器管不到或該歸別張卡，待拿掉。

## 鐵律

- 一張卡一個 PR，卡自帶檢查與必紅樣本，雲端 `verify` 綠了才算。
- 只會回綠的檢查等同沒有檢查。
- 待辦不寫在 repo 的 md 裡，走 issue。
- 檔名英文，內文中文。
