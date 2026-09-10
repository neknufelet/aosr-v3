---
title: 收據分兩層——雲端那一跑產機器收據推 status 分支，本機派工收據算宣稱
date_created: 2026-09-10
date_modified: 2026-09-10
status: accepted
kind: governance
supersedes: ""
superseded_by: ""
summary: "算數的收據由雲端 verify 那一跑機器打包（每支檢查的離開碼、報告行、junit、run id），跟狀態頁一樣推到機器分支 status，不進主線；本機派工收據留在 repo 外，PR 描述只貼摘要與雜湊，算宣稱。三張等收據的卡改掃 status 分支。"
---

# 收據分兩層——雲端那一跑產機器收據推 status 分支，本機派工收據算宣稱

## 問題

三張卡（`four-roles-different-actors` 誰跑誰驗、`receipt-schema-complete` 欄位齊不齊、`receipt-authority-is-the-cloud-run` 收據來自哪一跑）都要查「派工收據」，但今天沒有任何機器會把收據寫進 repo：派工工具在本機跑、寫的是 Markdown、落在暫存目錄。三張卡對真樹永遠回綠，等於沒立（issue #14）。票務那三張卡（issue #11）也排在同一件事後面。決策紙「雲端那一跑是唯一權威，本機收據只是宣稱」已經訂了 `authority` 只有 `cloud-run` 與 `claim` 兩值，但沒說收據由誰產、住哪裡。

## 選項

1. 收據分兩層。①雲端那一跑（`verify`，GitHub Actions 上跑檢查的工作）每次把每支檢查的離開碼、報告行、junit、run id 打包成機器產的收據，跟狀態頁一樣推到機器分支 `status`，不進主線；三張卡掃這裡。②本機派工收據留在 repo 外（照 delivery skill 那套流程機自己的帳本），PR 描述貼摘要與雜湊，標 `claim`。
2. 只做②，三張卡砍掉：承認派工收據永遠在 repo 外，機器守不到。
3. 收據寫進主線的一個目錄：每次 CI 跑完自動提交回 main。跟「主線只能走 PR」的 ruleset 衝突，也會讓主線長成帳本。

## 決定

選項 1。老闆原話（2026-09-10，issue #63）：「照建議做」。

具體：
- 收據的生產者是雲端 `verify` 那一跑，不是任何席位的自報。一跑一份，欄位至少有：run id、每支檢查的離開碼與報告行、pytest 的 junit、掃的是哪個 commit。
- 收據住在機器分支 `status`，跟狀態頁同一條分支、同一支推送流程（`.github/workflows/status.yml`），主線不放。
- 本機派工收據（工人、找碴席在本機跑出來的）留在 repo 外，PR 描述只貼摘要與雜湊，`authority = "claim"`。
- 三張卡從 `receipts-exist` 那一組解套的條件：生產者上線、`status` 分支上真的有第一份機器收據。在那之前照舊暫緩。

## 為什麼

v2 事故 `verifier-trusts-self-reported-fields`（驗證器信自報欄位，九起收據假綠）與 `builder-grades-own-work-edits-ruler`（施工者自報數字、自己改尺）的共同病根是收據由被查的那一方自己寫。收據由雲端那一跑機器打包，寫收據的人跟被查的人分開，三張卡才有真對象可咬。推到 `status` 分支而不是主線，是沿用決策紙「狀態頁不進主線」的同一個理由：機器產物不走 PR、不佔主線歷史。

## 代價

- 生產者今天還不存在，要一個自己的 PR 去改 `status.yml`：打包收據、推分支。那個 PR 沒合之前，三張卡照舊暫緩，這張紙只是把「等什麼」講清楚。
- 兩個實作題留給生產者那個 PR，這裡不定：①`verify` 檢出的是 PR 的樹，三張卡要掃 `status` 分支，得多一步抓那條分支或讀 fetched ref；②一跑的收據在那一跑的檢查跑完之後才寫得出來，所以卡驗的是之前幾跑的收據，不是當下這一跑。
- 本機派工收據永遠是 `claim`，四席是不是真的不同人、找碴席有沒有換一家模型，機器只能從雲端收據反推，本機那段守不到。
- issue #11 那三張票務卡在這之後還要另判：改寫成「看收據推回去」，還是承認由 delivery skill 自己的測試守。

## 拍板

老闆，2026-09-10 對話，記在 issue #63。
