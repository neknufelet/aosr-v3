---
title: 狀態頁不進主線
date_created: 2026-09-09
date_modified: 2026-09-10
status: accepted
kind: governance
supersedes: ""
superseded_by: ""
summary: "「做到哪」由指令從 git 與 GitHub 現算，開工與併入主線各算一次；歷史推到專門分支；唯一手寫的是 20 行以內的下一步意圖。"
---

# 狀態頁不進主線

## 問題

專案進度、剩項、里程碑位置要怎麼記，才不會過期和自相矛盾。

## 選項

1. 手寫 docs/next.md，照 BASELINE 2.1.1 的規定。
2. 由指令現算，算出的頁不進主線。
3. 現算但 commit 回主線，CI 比對。

## 決定

選項 2。一支進版控的程式從 git 與 GitHub 算出：里程碑進度線、開放 PR 與 issue、每支檢查最近結果、主線領先落後。開工時 SessionStart hook 算一次注入對話；每次併入主線由 CI 算一次，推到一條只有機器寫的專門分支，歷史全留；不排 cron。
地圖由人定：里程碑與其任務單存在 GitHub Milestones 與 issues；位置由機器算。
唯一手寫的是「下一步意圖」，上限 20 行，只寫意圖不寫狀態。

## 為什麼

v2 的 next.md 同一份檔同時寫「已併回」與「未併回」而守衛照過，劃掉的完成項留 37 條，凍結當天的快照比實際進度舊。算得出來的東西存成檔就會過期。commit 回主線會灌滿機器 commit、且放在眼前就會有人手改。

## 代價

沒有 hook 的那幾家 CLI 開工要手打一次指令。BASELINE 的 docs/next.md 那節在 v3 不適用。

## 拍板

老闆，2026-09-09 對話。清空時被清掉，2026-09-09 老闆口頭重新確認後照原文補回（檔名改英文）。示範頁：https://claude.ai/code/artifact/07a6f39a-21a6-4993-8844-f1a95b78750d

2026-09-10：「代價」那一段原文點名了一家工具，被規矩卡 no-model-names-in-entry-files 咬掉
（每次開工會載入的檔不准出現那些名字），改成不指名的說法。上面那個示範頁的網址留著：
網址是一個位置不是一個路由，那張卡的 `[[settings.allow]]` 具名放行它並附了到期日。
