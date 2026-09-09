---
title: git 歷史重來，GitHub repo 刪掉重建
date_created: 2026-09-09
date_modified: 2026-09-09
status: accepted
kind: governance
supersedes: ""
superseded_by: ""
summary: "用無父節點的 initial commit 重開歷史；為了清掉舊 PR 把 GitHub repo 整個刪掉同名重建；ruleset 照原樣重設，required check 等第一張卡的 CI 綠了才加回。"
---

# git 歷史重來，GitHub repo 刪掉重建

## 問題

清空重來如果保留舊歷史，舊的 5 個 commit 與 3 張已合併 PR 會一直被翻出來「參考」，跟清空的目的相反。

## 選項

1. 保留歷史，只在新 commit 裡刪檔。
2. `git checkout --orphan` 建一個沒有父節點的 initial commit，並刪掉整個 GitHub repo 再同名重建。

## 決定

選項 2。GitHub 不提供刪除 PR 的功能，只有刪 repo 能清掉那 3 張。repo 設定與主線的 ruleset（GitHub 上鎖主線的設定，id `22615925`）照原樣重設：不准刪分支、不准改寫歷史、只能走 PR 進主線。

唯一刻意的差異：暫時移除「required status check `verify`」——檢查程式清掉後那個 job 不存在，留著會讓所有 PR 永遠合不進去。**第一張有牙齒的卡（PR #19）合進去、主線上的 `verify` 跑綠之後，2026-09-09 已加回。**

## 為什麼

一個永遠回綠的 `verify` 是 v2 的頭號死因（「只會回綠的檢查等同沒有檢查」）。CI 要跟第一張會咬的卡一起長出來，鎖才有意義。

## 代價

舊歷史只剩備份 bundle（`~/aosr-v3-blueprint-2026-09-09/aosr-v3-full-history.bundle`），GitHub 上找不到。

## 拍板

老闆，2026-09-09 對話。
