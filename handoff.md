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

- `final-acceptor-need-not-rerun`（終審不必自己重跑）與 `receipt-authority-is-the-cloud-run`（只認雲端那一跑的收據）**直接衝突**。收斂工作流判定這要老闆決定，需要一張決策紙。
- 「fork 出去討論的結論怎麼帶回來、放哪」目前沒有規則。這輪是老闆手動貼路徑。
- `docs/` 的目錄結構（清空前討論到：只准 contract／design／cairn 三類，一輪一個凍結資料夾）沒有定案。

## 下一步（照這個順序）

1. **修資料**：38 張卡引用了 59 個教訓 id，只有 24 個存在於 `v2-audit/lessons.json`，**35 個是編的**。多數看起來是真教訓的改名（卡片寫 `skipped-tests-counted-as-passing`，真名是 `skips-disguise-red-as-green`），但沒有一個是照抄的，所以分不出「改名」與「捏造」。要逐條按**內容**重對，不能按名字。對完才知道真正的血債分布（目前只知道下限 25／38）。
2. **排第一批**：重對後排出要立的卡。清空前算出的組成是 12 張通過可行性 ＋ 6 張現有卡沒被新清單收到的 = 18 張，但那 6 張的實作已經跟著清空了，只剩規格描述留在 `blueprint/convergence-result.json` 與備份裡。
3. **立卡**：一張卡一個 PR。**第一張卡的 PR 要一併帶進 CI**（`.github/workflows/`）與跑它的 job，之後才把 `required_status_checks` 加回 ruleset。從第二張起，雲端綠了才算。

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
