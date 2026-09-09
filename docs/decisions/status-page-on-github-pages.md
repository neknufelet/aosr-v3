---
title: 機器算出來的狀態頁掛 GitHub Pages
date_created: 2026-09-09
date_modified: 2026-09-09
status: accepted
kind: governance
supersedes: ""
superseded_by: ""
summary: "「狀態頁不進主線」算出來的那一頁，用 GitHub Pages 從機器專用分支掛成固定網址，每次併主線自動更新。"
---

# 機器算出來的狀態頁掛 GitHub Pages

## 問題

「狀態頁不進主線」決定了狀態由機器現算、推到只有機器寫的分支。老闆要一個能隨時打開的圖像化介面，看各條路線走到哪，含時間軸與任務分類。

## 選項

1. 只留在分支裡，要看就 clone 下來開。
2. 用 GitHub Pages（GitHub 免費把某條分支的內容當網頁掛出來）給那條分支一個固定網址。

## 決定

選項 2。狀態頁程式產出 HTML，推到機器專用分支，GitHub Pages 對著那條分支掛。網址固定，每次併入主線由 CI 重算重推，頁面自動更新。頁面內容照「狀態頁不進主線」：里程碑進度線、開著的 PR 與 issue、每支檢查最近結果、主線領先落後。

## 為什麼

算出來的頁不進主線，但人要看得到；Pages 剛好只讀那條分支，不會誘使人手改。

## 代價

repo 是 public，狀態頁也是 public。Pages 的啟用是 GitHub 設定，不在版控裡，要記在這張紙。**這張紙補寫時 Pages 尚未啟用、狀態頁程式尚未寫**，兩者等第一批卡立完再做。

## 拍板

老闆，2026-09-09 對話。
