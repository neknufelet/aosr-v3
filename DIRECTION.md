# 方向

2026-09-09 清空重來。這個 repo 現在**沒有任何規矩、沒有任何檢查、沒有決策紙**，是故意的。
上一輪累積了一堆半成品，每一份都要問一次「這份怎麼辦」，那本身就在燒時間。

規矩要從 `v2-audit/` 的資料重新長出來，不是從舊 repo 遷就出來。

## 手上有什麼

| 路徑 | 是什麼 |
|---|---|
| `v2-audit/lessons.json` | v2 的 66 筆事故，每筆附證據（檔案＋行號）。原始 364 筆去重而來 |
| `v2-audit/lessons.md` | 同一份的人話摘要 |
| `blueprint/rules-436.json` | 從四份設計報告抽出、去重後的 436 條候選規則，已對回 lessons |
| `blueprint/batch1-127.json` | 上面那 436 條裡，不需要等程式碼就做得到的 127 條 |
| `blueprint/collapse-by-*.json` | 三個 agent 用三種視角各自把 127 條收斂的結果（41／41／39 張） |
| `blueprint/convergence-result.json` | 綜合成 38 張後，逐張驗可行性＋找碴的完整結果 |

四份設計報告本身不在這裡，備份在 repo 外。

## 驗過的數字

以下由機器算出，可重跑：

- v2 事故 66 筆，桶別 rule 35／structure 24／knowledge 5／obsolete 2
- 五份報告抽出 605 條，去重 436 條
- 436 條裡 110 條不卡任何未搬入的程式碼；加 GitHub 設定 6 條、派工工具 9 條、外部帶回 2 條 = 127 條
- 127 條收斂成 38 張，其中 12 張通過可行性驗證

## 未驗的、有問題的

- **35 個幽靈教訓 id 只在合併那一步產生，上游是乾淨的，已機器重對。** 38 張卡自稱引用 59 個 id，只有 24 個存在於 `lessons.json`，其餘 35 個是合併工人自己編的（多數是真教訓的改名，例如卡片寫 `skipped-tests-counted-as-passing`，真名是 `skips-disguise-red-as-green`）。
  但 `rules-436.json` 與 `batch1-127.json` 引用的教訓 id **一個幽靈都沒有**。所以修法不是逐條猜改名，而是丟掉合併那步手寫的那組，沿每張卡的 `covers` 回查候選規則的 `lesson_ids` 重建——`blueprint/remap_cards.py` 做這件事，產出 `blueprint/cards-38.json`，跑完自己驗每個 id 都對回 `lessons.json`。
- 重對後（數字見 `cards-38.json` 的 `meta`）：**34／38 張有血債**，12 張通過可行性的裡面 **10／12** 有，合計覆蓋 **43／66** 筆事故。「下限 25／38」那個推測作廢。
- **id 層有血債 ≠ 內容層擋得住。** 第一批 12 張逐條做過內容確認（`blueprint/first-batch-review.json`，每張卡一個獨立找碴席，預設立場「對不上」）：14 條判定裡 **0 條 prevents、13 條 related、1 條 unrelated**，所以 `net_blood_debt`（至少一條 prevents）是 **0／12**。
  意思是「這 12 張以目前 `check_idea` 的寫法，都擋不住它掛的那件 v2 事故本身」，不是「這 12 張沒用」。
  **後續**：12 張的「怎麼查／必紅樣本」已照那些 `critic_note` 重寫成 `check_idea_v2`／`fixture_idea_v2`（`blueprint/cards-38.json`），每張又開了一個獨立找碴席、改過一輪、再開一席。結果：**4 張被找碴確認會在事故當下回紅**（`green-must-be-real-green` 的 `skipped == 0` 對上那一跑的 3,311 collected／12 skipped；`commit-author-allowlisted` 對上那 11 筆 `test@example.invalid`；`rule-card-required-fields` 的第二關對上 hash guard「掃描根刪掉仍 PASS」；`doc-size-cap` 改成量份數之後對上 1046 份那件事——最後這張因為入口檔與 `docs/` 今天都不在 repo，仍是暫緩），但**兩輪之後 12 張全部仍標 `still_leaky: true`**——漏的多半不是寫法問題，是那件事故的違規物件根本不在任何檢查的掃描面裡（例如「決策只活在對話」、「本機 hook 特有的五種失效」，兩者的 `enforcer` 欄原本就寫「不適用」）。逐張的漏法寫在 `still_leaky_reason` 與 `critic_v2`。
- 一條真實在流血的問題**還沒有可行的卡**：門檻數字散落三處（入口檔行數、測試地板、ruff 規則），其中 ruff 規則有兩份且其中一份用 `--isolated` 完全不讀另一份。按數值比對的檢查形狀已實測行不通（會誤判 pytest 的離開碼 5）。需要改成查形狀的設計。

## 接下來的順序

1. ~~把 35 個幽靈 id 逐條按內容對回 `lessons.json` 的 66 筆，重算血債分布。~~ **做完**（`blueprint/cards-38.json`）。
2. ~~依重對後的結果排出第一批要立的卡，並為每張寫出必紅樣本。~~ **做完**：38 張對著清空後的空 repo 重判了一次（`feasibility_v2`），18 張能立、20 張暫緩，順序在 `blueprint/first-batch-order.json`。12 張原 pass 卡的「怎麼查」與「必紅樣本」照找碴結論重寫成 `check_idea_v2`／`fixture_idea_v2`。暫緩的 20 張照「在等什麼」分八組進了 GitHub issue（標籤 `deferred-cards`）。
3. 立卡。一張卡一個 PR，雲端綠了才算。第一張是 `rule-card-required-fields`（併 `prove-the-bite`、`enforcer-must-be-machine-in-vcs`），它的 PR 要一併帶進四個共用零件與 CI——追蹤在 issue [#17](https://github.com/neknufelet/aosr-v3/issues/17)。

**待辦不寫在這裡。** 老闆拍板：待辦全走 GitHub issue，repo 不放手寫待辦檔；狀態頁由機器算、不進主線。

## 規則

現在沒有規則。第一張卡立起來之前，這裡是空的。
