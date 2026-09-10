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

- 來源資料：v2 事故 66 筆（`v2-audit/lessons.json` 的 incidents）；候選規則去重後 436 條（`blueprint/rules-436.json`），其中不必等程式碼就做得到的 127 條（`blueprint/batch1-127.json`），收斂成 38 張卡（`blueprint/cards-38.json`）。
- 藍圖 38 張今天全部有去向（跑 `uv run python blueprint/remap_cards.py` 現算，寫進 cards-38.json 的 meta.establishment，不手抄）：立了 29 張、砍掉 5 張、暫緩 4 張。29 張裡有 4 張是併進別張卡的牙——doc-size-cap 併進 doc-frontmatter-and-dates、derived-content-rendered-not-handwritten 併進 entry-files-rendered-from-registry、enforcer-must-be-machine-in-vcs 與 prove-the-bite 併進 rule-card-required-fields——不各自成卡。
- 主線 26 張卡（`ls governance/rules/*.toml | wc -l`）：立起來的 29 張扣掉併掉的那 4 張剩 25 張，加上藍圖 38 張以外另立的 uv-single-entrypoint。
- 暫緩那 4 張各在等一個對象出現：design-report-self-consistent 等 v3 自己產出的設計報告落檔；shell-scripts-are-bash 等版控樹裡出現 shell 腳本；signature-keyed-on-semantics 與 tests-land-with-code 等產品程式那棵樹搬進來。
- 同一跑算出來的對應：38 張覆蓋 v2 事故 43／66；重對後有血債的 34／38；合併那步自稱引用 59 個教訓 id，真的 24、幽靈 35，已沿 covers 機器重對；上游四層幽靈 0。
- 收據線整條在主線：生產者 `governance/status/record_step.py` 每一步原封不動記離開碼，鏡像 `governance/status/mirror_receipts.py` 把雲端收據拉回來給檢查讀，三張收據卡 receipt-schema-complete、receipt-authority-is-the-cloud-run、four-roles-different-actors 都已立。機器分支上的收據今天有 33 份（`git ls-tree -r --name-only origin/status -- receipts/ | wc -l`）。

## 鐵律

- 一張卡一個 PR，卡自帶檢查與必紅樣本，雲端 `verify` 綠了才算。
- 只會回綠的檢查等同沒有檢查。
- 待辦不寫在 repo 的 md 裡，走 issue。
- 檔名英文，內文中文。
