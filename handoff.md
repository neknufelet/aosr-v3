# Handoff — 2026-09-09 清空重來

給下一個對話。先讀這份，再讀 `DIRECTION.md`。

## 一句話

v3 的治理層在 2026-09-09 被整個清空，重新從 v2 的事故資料長規矩。
現在 repo 裡沒有任何規矩、檢查、決策紙、CI。**這是故意的，不是壞掉。**

## 現在的狀態（重要）

清空、重建、推送都做完了。**接手時工作區是乾淨的，沒有待處理的變更。**

| | |
|---|---|
| 分支 | `main`，追蹤 `origin/main`，兩邊同步 |
| 歷史 | **1 個 commit**（`ef80c68`），沒有父節點 |
| 檔案 | 10 個（含這份），內容見 `DIRECTION.md` |
| 遠端 | GitHub public `neknufelet/aosr-v3`，**整個 repo 刪掉重建過** |
| PR / issue | 各 0 張（舊的 3 張 PR 隨 repo 一起消失） |
| ruleset | `主線鎖死`（id `22615925`）active，三條：`deletion`／`non_fast_forward`／`pull_request` |
| CI | **沒有。** `.github/` 是空的 |

**ruleset 為什麼只有三條**：原本還有 `required_status_checks` 指名要一個叫 `verify` 的檢查。檢查程式清掉之後那個 job 不存在，required check 會永遠等不到、任何 PR 都合不進去，所以先移除。
**要加回去的時機**：第一張規矩卡連同它的檢查、必紅樣本、跑它的 CI 一起進來之後。

```bash
gh api -X PUT repos/neknufelet/aosr-v3/rulesets/22615925 --input <規格檔>
```

**不要先做一個空的 CI。** 一個永遠 exit 0 的 `verify` 就是 v2 的頭號死因——「只會回綠的檢查等同沒有檢查」。CI 要跟第一張有牙齒的卡一起長出來。

## 已經拍板、但還沒寫成決策紙的

決策紙全部被清掉了，所以以下只活在這份 handoff 裡。**要重新落檔。**

1. **清空重來**——不遷就舊的半成品，規矩從 `v2-audit/` 重新長。老闆的原話：「一點點就東西，然後 agent 為了要保留一直遷就被影響。」
2. **命名一律英文**——檔名、路徑、CLI 參數全部 ASCII。內文（人話、決策紙、說明）維持中文。
   理由：清空前 86 個版控檔案有 42 個是中文檔名，`git ls-files` 預設印成 `"\344\270\273\347\267\232..."`，要加 `-c core.quotePath=false` 才看得懂；macOS 的 NFC/NFD 正規化是已知地雷（未實測，因為目前全在 Linux）。
3. **git 歷史重來**——一個乾淨的 initial commit，不保留舊的 5 個 commit 與 3 張已合併 PR。
   **已執行完畢。** 做法是 `git checkout --orphan` 建無父節點的 commit，然後刪掉整個 GitHub repo 再用同名重建（GitHub 不提供刪除 PR 的功能，只有刪 repo 能清掉那 3 張）。repo 設定與 ruleset 已照原樣重設，唯一刻意的差異是移除 `required_status_checks`（理由見上）。

## 還沒拍板、掛著的

**全部進 GitHub issue 了**（老闆拍板：待辦全走 issue，repo 不放手寫待辦檔）。標籤 `decision`：

| issue | 題目 |
|---|---|
| [#3](https://github.com/neknufelet/aosr-v3/issues/3) | `final-acceptor-need-not-rerun`（終審不必自己重跑）與 `receipt-authority-is-the-cloud-run`（只認雲端那一跑的收據）直接衝突，要挑一邊 |
| [#4](https://github.com/neknufelet/aosr-v3/issues/4) | fork 出去討論的結論怎麼帶回來、放哪 |
| [#5](https://github.com/neknufelet/aosr-v3/issues/5) | `docs/` 目錄結構要分幾類、每類上限多少 |
| [#6](https://github.com/neknufelet/aosr-v3/issues/6) | `text-format-consistent` 砍到只剩一條，還算不算一張卡 |
| [#7](https://github.com/neknufelet/aosr-v3/issues/7) | 「環境交給 uv 管、單一入口」要不要現在另立一張卡 |
| [#8](https://github.com/neknufelet/aosr-v3/issues/8) | 補寫六張被清掉的決策紙（含上面「已經拍板但沒寫成決策紙」那三件，加上待辦走 issue、狀態頁不進主線、狀態頁掛 Pages） |

暫緩的 20 張規矩卡照「在等什麼」分成八組，一組一張 issue，標籤 `deferred-cards`：[#9](https://github.com/neknufelet/aosr-v3/issues/9)（等老闆拍板）、[#10](https://github.com/neknufelet/aosr-v3/issues/10)（等設計報告）、[#11](https://github.com/neknufelet/aosr-v3/issues/11)（等派工工具進 repo）、[#12](https://github.com/neknufelet/aosr-v3/issues/12)（等 `docs/` 結構）、[#13](https://github.com/neknufelet/aosr-v3/issues/13)（等入口檔）、[#14](https://github.com/neknufelet/aosr-v3/issues/14)（等收據）、[#15](https://github.com/neknufelet/aosr-v3/issues/15)（等 shell 腳本）、[#16](https://github.com/neknufelet/aosr-v3/issues/16)（等 `src/`）。

## 下一步（照這個順序）

1. **修資料**：~~38 張卡引用了 59 個教訓 id，只有 24 個存在，35 個是編的，要逐條按內容重對~~ **已做完（PR #2）。** 幽靈 id 只在「127 條收成 38 張」那一步產生，上游 `rules-436`／`batch1-127`／`collapse-by-*` 全部乾淨，所以沿每張卡的 `covers` 回上游機器重對即可，結果在 `blueprint/cards-38.json`（含來源出處），腳本 `blueprint/remap_cards.py` 可重跑並自驗。id 層：34／38 有血債、12 張裡 10／12。**內容層**（`blueprint/first-batch-review.json`）：12 張照現在的「怎麼查」寫法，沒有一張能在它對到的 v2 事故當下回紅——每張的洞與補法在 `critic_note`。另外 26 張的暫緩理由是對著清空前的舊 repo 判的（提到登記簿、舊卡、`cmd_prove` 的有 17 張），要對空 repo 重驗一次。
2. **排第一批**：~~重對後排出要立的卡~~ **已做完。** 38 張對著清空後的空 repo 重判了一次可行性（`feasibility_v2`），**18 張能立、20 張暫緩**（暫緩的照「在等什麼」分八組，見上表）。順序與每張一句理由在 `blueprint/first-batch-order.json`，追蹤 issue [#17](https://github.com/neknufelet/aosr-v3/issues/17)（milestone `batch-1`）。
   清空前那個「12 ＋ 6 = 18」的組成作廢——那 6 張是舊 repo 的現有卡，已經跟著清空消失了。現在的 18 張是重判出來的，其中 12 張來自原本通過可行性的那批（4 張因為對象不存在被降為暫緩）、10 張是從 26 張暫緩卡裡重判上來的（舊理由多半引用清空前的舊零件，已失效）、2 張建議併進第一張卡。
3. **立卡**：一張卡一個 PR。**第一張是 `rule-card-required-fields`**（併 `prove-the-bite` 與 `enforcer-must-be-machine-in-vcs`），它的 PR 要一併帶進四個共用零件（卡的載入器、0／1／2 離開碼約定、跑必紅樣本的後設測試、`.github/workflows/verify.yml`），job 名建議 `verify`；**雲端第一次跑綠之後**才把 `required_status_checks` 加回 ruleset 並指名 `verify`。從第二張起，雲端綠了才算。
   **立卡前一定要看**：卡的「怎麼查」已經照找碴結論重寫成 `check_idea_v2`／`fixture_idea_v2`（`blueprint/cards-38.json`，`meta.revision: 2`）。**20 張開過獨立找碴席（12 張各兩輪、8 張各一輪），20 張全部標 `still_leaky: true`**，漏在哪逐張寫在 `still_leaky_reason` 與 `critic_v2`。7 對「卡×事故」被確認會在事故當下回紅——**7 張卡各一對**（其中兩張咬的是同一筆事故的兩半，所以只覆蓋 6 筆不同的事故）。
   漏的多半不是寫法沒調好，是那件事故的違規物件根本不在任何檢查的掃描面裡。有幾張的 `still_leaky_reason` 裡寫了立卡前必須先做完的前置條件（例如 `secrets-never-committed` 要先把 CI checkout 設 `fetch-depth: 0`、`refs-and-links-resolve` 的乾淨樹今天不是 0、`style-guard` 要先拆 `main()`），照著做。

## 備份在哪（全部可還原）

`~/aosr-v3-blueprint-2026-09-09/`（5.4 MB）

| 檔 | 是什麼 |
|---|---|
| `aosr-v3-full-history.bundle` | 清空前的完整 git 歷史，`git bundle verify` 通過 |
| `design-reports/` | 四份設計報告＋交叉比對（opus 與 astra 各寫治理與架構） |
| `decisions-8/` | 被清掉的 8 張決策紙 |
| `merged.json` 等 | 這輪 workflow 的所有中間產物 |

還原方式：`git clone aosr-v3-full-history.bundle <目錄>`。

## 這輪踩到的坑（別再犯）

1. **不要把大包資料塞進 agent 的 prompt。** 第一個工作流把 605 條 `JSON.stringify` 進 prompt，那個 agent 被迫縮寫欄位、把真貨寫到檔案。正確做法：資料寫成檔案，prompt 只給路徑，agent 自己讀。第二個工作流照這樣做，順利。
2. **數字要帶證據。** 這輪講過「35 筆裡 17 筆沒人抓」，17 是人工比對的推測，卻被當事實連用好幾輪，被老闆抓到才承認。沒實跑過的數字一律標 `(未驗，推測)`。
3. **agent 會編出很像真的 id。** 35 個幽靈教訓 id 就是這樣來的。任何交叉引用都要對回原始檔驗一次。
4. **找碴那層要獨立而且要凶。** 這輪 38 張卡被刷掉 26 張，理由都很紮實（對象不存在、會永遠回綠、跟現有卡互相遮蔽）。沒有這層的話會立一堆不會咬的卡。

## 跟老闆講話的方式

- **一次一題，四格白話**：這是什麼／上一代出過什麼事／我建議／你回什麼。一次丟一堆會被退貨。
- **講人話。** 不要造新名詞（這輪造了「名簿」被罵）。要用新詞先解釋。
- **要決定的事最多給兩個選項**，附上快速判斷需要的資訊，並說你選哪個。
- 老闆會質疑數字。**不要用沒驗過的數字。**
