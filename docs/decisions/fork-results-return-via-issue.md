---
title: fork 出去的結論先開 issue 帶回，討論後不做就關掉
date_created: 2026-09-09
date_modified: 2026-09-09
status: accepted
kind: governance
supersedes: ""
superseded_by: ""
summary: "在外面（另一個對話、別的模型、外部報告）產生的結論，帶回時先開一張 issue 放一句意圖與原文連結；討論後不做就關掉、什麼都不進 repo；做了才變成決策紙、cairn 或規矩卡。"
---

# fork 出去的結論先開 issue 帶回，討論後不做就關掉

## 問題

討論有時會 fork 出去——另開對話、找別的模型、在外面寫報告——結論產生在 repo 外。沒有規則說它怎麼帶回來、放哪、算不算數；這輪是老闆手動貼路徑。

## 選項

1. 帶回來的結論直接寫成決策紙或 cairn 進 repo，外部原文只留連結。
2. 另開 `docs/inbox/` 放外部原文，再由人挑出要升格的。
3. 先開 GitHub issue：一句「我想做這個」加外部原文連結，不複製內容。在 issue 裡討論；不做就關掉，repo 一個字都不留；做了才落成決策紙（拍板）、cairn（經驗）或規矩卡（規矩），PR 寫 `closes #N`。

## 決定

選項 3。老闆 2026-09-09 採納另一位 agent 的建議：「當我貼回來說我想要做這個，應該是先進去 issue，如果討論後不做就丟掉，這樣可以讓 issue 控制進度。」

- issue 是唯一的入口，也是進度的所在。標籤照現有用法（`decision`、`deferred-cards`，或新的 `idea`）。
- 外部原文一律只留連結或路徑指向備份，不複製進 repo。
- 誰帶回來誰開 issue（通常是指揮）。

## 為什麼

v2 事故 `decisions-live-only-in-conversation`：口述修正沒落檔，對話清掉就丟，重述三四次。`moved-files-leave-dead-references`：帶回的原文放錯地方、搬走後引用全死。issue 有開關狀態、能討論、機器查得到；repo 只收有結果的東西，不會長出第二個 `docs/inbox/` 帳本。跟「待辦走 issue」與「docs 三類」兩張決策紙對齊。

## 代價

一個想法要多開一張票才能討論；沒網路時開不了。可接受。

## 拍板

老闆，2026-09-09 對話。
