---
title: 待辦全走 GitHub issue，repo 不放手寫待辦檔
date_created: 2026-09-09
date_modified: 2026-09-09
status: accepted
kind: governance
supersedes: ""
superseded_by: ""
summary: "還沒做的事一件一張 issue；暫緩的規矩卡照「在等什麼」分組開票；要拍板的事標 decision；repo 內不再有 next-intent 這類手寫待辦檔。"
---

# 待辦全走 GitHub issue，repo 不放手寫待辦檔

## 問題

還沒馬上做的事（暫緩的規矩卡、要補的決策紙、要拍板的題）放哪，才不會像 v2 的 `next-intent.md` 那樣新舊交雜、越寫越長。

## 選項

1. repo 裡一份手寫待辦檔。
2. 全走 GitHub issue（GitHub 上的待辦票，可開可關、可加標籤、機器查得到），repo 樹裡不放手寫待辦檔。

## 決定

選項 2。
- 一件事一張 issue，不是一張卡一張。暫緩的規矩卡照「在等什麼」分組，一組一張（標籤 `deferred-cards`），內文只列卡 id 與等待條件，規格不抄，留在 `blueprint/cards-38.json`。
- 要拍板的事各開一張，標籤 `decision`。決策紙進 repo 後關票。
- 立卡的 PR 描述寫 `closes #N`，合了自動關。
- `handoff.md` 只寫「現在在哪、去看哪幾張 issue」，不列待辦。

## 為什麼

v2 事故 `status-file-grows-into-history-ledger`：進度檔同時寫「已併回」與「未併回」，劃掉的完成項留 37 條，守衛照過。`index-docs-recopy-facts-that-drift`：人工抄現況進文件，原始資料改了文件沒改。手寫待辦檔一定會變成過期帳本，issue 有開關狀態、機器查得到，不會。

## 代價

離線看不到待辦。由「狀態頁不進主線」那張決策紙的機器現算補上。

## 拍板

老闆，2026-09-09 對話。
