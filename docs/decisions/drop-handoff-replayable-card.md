---
title: 砍掉「交接單可重播」卡，牙齒已被兩張卡蓋住
date_created: 2026-09-10
date_modified: 2026-09-10
status: accepted
kind: governance
supersedes: ""
superseded_by: ""
summary: "handoff-must-be-replayable 不立卡：路徑要解析得到、不准指 repo 外歸 refs-and-links-resolve；不准長成帳本、限 60 行歸 status-page-computed-not-typed；剩下的 run id 解析要上網。"
---

# 砍掉「交接單可重播」卡，牙齒已被兩張卡蓋住

## 問題

候選卡 `handoff-must-be-replayable`（交接單要進版控且引用得到真的產物，丟在暫存目錄或重播不了就紅）等 `docs/` 結構拍板。結構拍了之後檢查它還剩什麼牙。

## 選項

1. 砍掉，資料檔記理由。
2. 併進 `refs-and-links-resolve` 當一條窄規則：交接單裡出現暫存目錄或家目錄路徑即紅。

## 決定

選項 1。它的三顆牙：「引用得到真的產物」已由 `refs-and-links-resolve` 在守（路徑要解析得到、絕對路徑與家目錄一律紅、備份路徑具名放行）；「不准長成帳本」已由 `status-page-computed-not-typed` 在守（`handoff.md` 放行但限 60 行、不准像待辦檔）；「run id 要解析得到」要上網，違反檢查不上網。選項 2 那條今天 `refs-and-links-resolve` 已經在咬。

## 為什麼

兩支檢查掃同一個對象會互相遮蔽（v2 事故 `guard-teeth-shadow-each-other`），沒有新牙的卡不立。

## 代價

「交接單引用的雲端 run 還在不在」沒人守；那要看狀態頁（機器現算）而不是交接單。

## 拍板

老闆，2026-09-10 對話：「照你的意見」（issue #54）。
