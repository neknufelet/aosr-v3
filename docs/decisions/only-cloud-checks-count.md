---
title: 只有雲端檢查算數
date_created: 2026-09-09
date_modified: 2026-09-10
status: accepted
kind: governance
supersedes: ""
superseded_by: ""
summary: "規矩的執行者只能是 GitHub Actions；本機 hook 只是加速，不得申報。測試分合併必過層與 GPU 長測試層。"
---

# 只有雲端檢查算數

## 問題

規矩被違反時，誰負責擋。本機 hook 算不算執行者。

## 選項

1. 本機 pre-commit 為主，雲端為輔。
2. 雲端 CI 為唯一執行者，本機 hook 只是便利。

## 決定

選項 2。文件裡「執行者」欄只能填 CI job 名。本機 hook 可以裝，但不得申報。
測試分兩層：合併必過層是無 GPU、無網路、15 分鐘內跑完的 hermetic 套件；GPU 與長測試在 florian-coder 自架 runner 上跑，不擋合併，紅了開 issue 並在狀態頁印紅字。第二層只由 push 到 main 與手動觸發啟動，外部 PR 需人工核准才能在自架 runner 上跑。

## 為什麼

v2 規範寫「執行牙：pre-commit」，實際 `.git/hooks/` 是空的，三個月沒人發現。後來加的「守衛的守衛」自己有五種靜默失效。四家 CLI 裡只有一家裝得起 hook，另外三家根本沒有這種東西。守衛交給 GitHub，誰來改都一樣被擋，且不需要「守衛的守衛」那一層。

## 代價

自架 runner 掛在公開 repo 有外人跑程式的風險，靠核准機制擋。GPU 層不擋合併，紅了要人看狀態頁。

## 拍板

老闆，2026-09-08 對話。

清空時被清掉，2026-09-10 照原文補回（檔名改英文）。

2026-09-10：「為什麼」那一段原文點名了兩家工具，被規矩卡 no-model-names-in-entry-files 咬掉
（每次開工會載入的檔不准出現那些名字），改成不指名的說法——哪一家裝得起 hook 是會變的事實，
決定本身不靠它。意思不變：執行者只能是雲端那一台。
